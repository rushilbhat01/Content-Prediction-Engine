"""
predict.py — Given a video file + metadata, run the full pipeline and return recommendations.

Usage:
    python3 predict.py --video path/to/video.mp4 \
                       --title "Your video title" \
                       --tags "tag1,tag2,tag3" \
                       --desc "Your description" \
                       --hour 18 \
                       --weekday 2 \
                       --niche fitness
"""

import argparse
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import joblib
import torch
import clip_compat as clip
from PIL import Image
from pathlib import Path
from sentence_transformers import SentenceTransformer
import librosa
from runtime_device import move_model_to_device, select_torch_device

# ── Args ──────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--video',   required=True,  help='Path to video file')
parser.add_argument('--title',   default='',     help='Video title')
parser.add_argument('--tags',    default='',     help='Comma-separated tags')
parser.add_argument('--desc',    default='',     help='Description')
parser.add_argument('--hour',    type=int, default=18, help='Hour you plan to post (0-23)')
parser.add_argument('--weekday', type=int, default=1,  help='Weekday you plan to post (0=Mon, 6=Sun)')
parser.add_argument('--niche',   default='fitness')
parser.add_argument('--channel', default='', help='Channel URL or handle (e.g. @FitWithAlex)')
args = parser.parse_args()

video_path = Path(args.video)
if not video_path.exists():
    print(f"ERROR: Video file not found: {video_path}")
    exit()

NICHE      = args.niche
MODELS_DIR = Path("models")

# ── Device ────────────────────────────────────────────────
device = select_torch_device(torch, prefer=os.getenv("YT_SHORTS_DEVICE", "cuda"))

# ── Channel stats (fetched before heavy model loading) ────
from fetch_channel_stats import fetch_channel_stats as _fetch_channel_stats

channel_subscriber_count  = 0
channel_avg_views         = 0.0
channel_avg_like_rate     = 0.0
channel_avg_comment_rate  = 0.0

if args.channel:
    print(f"Fetching channel stats for: {args.channel}")
    try:
        _ch = _fetch_channel_stats(args.channel)
        channel_subscriber_count  = _ch["subscriber_count"]
        channel_avg_views         = _ch["channel_avg_views"]
        channel_avg_like_rate     = _ch.get("channel_avg_like_rate",    0.0)
        channel_avg_comment_rate  = _ch.get("channel_avg_comment_rate", 0.0)
        print(f"  Subscribers: {channel_subscriber_count:,} | Avg views: {channel_avg_views:,.0f} | Like rate: {channel_avg_like_rate:.4f}")
    except Exception as e:
        print(f"  Warning: could not fetch channel stats -- {e}")

# ── Load pretrained models ────────────────────────────────
print("Loading models...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
title_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

try:
    import torchvggish
    from torchvggish import vggish_input
    _vggish_model = move_model_to_device(torchvggish.vggish(), device)
    has_vggish = True
except Exception:
    has_vggish = False

try:
    import whisper as _whisper
    _whisper_model = _whisper.load_model("tiny", device=device)
    has_whisper = True
except Exception:
    has_whisper = False

try:
    from match_trending_audio import match_score as _trending_match, db_size as _trending_db_size
    has_trending_db = _trending_db_size() > 0
except Exception:
    has_trending_db = False

try:
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    _PROTOTXT = str(Path(__file__).parent / "models" / "deploy.prototxt")
    _WEIGHTS  = str(Path(__file__).parent / "models" / "res10_300x300_ssd_iter_140000.caffemodel")
    _face_net = cv2.dnn.readNetFromCaffe(_PROTOTXT, _WEIGHTS)
    _emo_recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
    has_fer = True
except Exception:
    has_fer = False

_YUNET_PATH   = str(Path(__file__).parent / "models" / "face_detection_yunet.onnx")
_PROTOTXT_VIS = str(Path(__file__).parent / "models" / "deploy.prototxt")
_WEIGHTS_VIS  = str(Path(__file__).parent / "models" / "res10_300x300_ssd_iter_140000.caffemodel")
try:
    _face_net_vis = cv2.FaceDetectorYN_create(_YUNET_PATH, "", (320, 320), score_threshold=0.4)
    _USE_YUNET = True
    _USE_DNN   = False
except Exception:
    _USE_YUNET = False
    try:
        _face_net_vis = cv2.dnn.readNetFromCaffe(_PROTOTXT_VIS, _WEIGHTS_VIS)
        _USE_DNN = True
    except Exception:
        _USE_DNN = False
        _FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

_EMO_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
EMOTION_MAP = {'Happiness': 1.0, 'Surprise': 0.8, 'Neutral': 0.5,
               'Sadness': 0.2, 'Anger': 0.3, 'Fear': 0.1, 'Disgust': 0.1, 'Contempt': 0.2}

print(f"  CLIP: ON | MiniLM: ON | VGGish: {'ON' if has_vggish else 'OFF'} | FER: {'ON' if has_fer else 'OFF'} | TrendingAudio: {'ON' if has_trending_db else 'OFF (run build_trending_audio_db.py)'}")

# ─────────────────────────────────────────────────────────
# FEATURE EXTRACTORS (same logic as extract_visual.py,
# extract_audio.py, extract_embeddings.py)
# ─────────────────────────────────────────────────────────

def extract_frame_features(frame, h, w):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    area = h * w

    brightness = float(gray.mean()) / 255
    contrast   = float(gray.std())  / 128
    saturation = float(hsv[:, :, 1].mean()) / 255
    sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    has_face  = 0
    face_size = 0.0
    if _USE_YUNET:
        _face_net_vis.setInputSize((w, h))
        _, dets = _face_net_vis.detect(frame)
        if dets is not None and len(dets) > 0:
            has_face = 1
            best = max(dets, key=lambda d: d[2] * d[3])
            face_size = float(best[2] * best[3]) / area
    elif _USE_DNN:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        _face_net_vis.setInput(blob)
        dets = _face_net_vis.forward()
        best_area = 0.0
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < 0.35:
                continue
            has_face = 1
            bw = float(dets[0, 0, i, 5] - dets[0, 0, i, 3])
            bh = float(dets[0, 0, i, 6] - dets[0, 0, i, 4])
            a  = bw * bh
            if a > best_area:
                best_area = a
        face_size = best_area
    else:
        faces = _FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        has_face = int(len(faces) > 0)
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            face_size = (fw * fh) / area

    edges        = cv2.Canny(gray, 100, 200)
    edge_density = float(edges.mean()) / 255

    hue_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180])
    hue_hist = hue_hist / (hue_hist.sum() + 1e-9)
    hue_entropy = float(-np.sum(hue_hist * np.log2(hue_hist + 1e-9)))

    cx, cy = w // 2, h // 2
    r      = min(cx, cy) // 2
    centre = gray[cy-r:cy+r, cx-r:cx+r]
    centre_brightness = float(centre.mean()) / 255 if centre.size > 0 else brightness

    top_third    = gray[:h//3, :]
    bottom_third = gray[2*h//3:, :]
    top_edges    = float(cv2.Canny(top_third,    100, 200).mean()) / 255
    bottom_edges = float(cv2.Canny(bottom_third, 100, 200).mean()) / 255
    text_proxy   = max(top_edges, bottom_edges)

    return {
        "brightness": brightness, "contrast": contrast,
        "saturation": saturation, "sharpness": sharpness,
        "has_face": has_face, "face_size": face_size,
        "edge_density": edge_density, "hue_entropy": hue_entropy,
        "centre_brightness": centre_brightness, "text_proxy": text_proxy,
    }


def extract_visual_features(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return {}
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if h == 0 or w == 0:
        cap.release()
        return {}

    all_frames = []
    hook_frames = []
    motion_scores = []
    prev_gray = None
    prev_bright = None
    cut_count = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % max(1, int(fps)) == 0:
            timestamp = frame_idx / fps
            try:
                feats = extract_frame_features(frame, h, w)
            except Exception:
                frame_idx += 1
                continue
            all_frames.append(feats)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                try:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    motion_scores.append(float(mag.mean()))
                except Exception:
                    pass
            if prev_bright is not None and abs(feats["brightness"] - prev_bright) > 0.15:
                cut_count += 1
            prev_gray   = curr_gray
            prev_bright = feats["brightness"]
            if timestamp <= 3.0:
                hook_frames.append(feats)
        frame_idx += 1
    cap.release()

    if not all_frames:
        return {}

    result = {}
    keys = list(all_frames[0].keys())
    for key in keys:
        vals = [f[key] for f in all_frames]
        result[f"mean_{key}"] = float(np.mean(vals))
        result[f"std_{key}"]  = float(np.std(vals))
        result[f"max_{key}"]  = float(np.max(vals))
    src = hook_frames if hook_frames else all_frames
    for key in keys:
        vals = [f[key] for f in src]
        result[f"hook_{key}"] = float(np.mean(vals))
    result["mean_motion"]    = float(np.mean(motion_scores)) if motion_scores else 0.0
    result["max_motion"]     = float(np.max(motion_scores))  if motion_scores else 0.0
    total_dur = frame_idx / max(fps, 1)
    result["cuts_per_second"] = cut_count / max(total_dur, 1)
    if len(all_frames) >= 2:
        mid = len(all_frames) // 2
        result["brightness_trend"] = float(
            np.mean([f["brightness"] for f in all_frames[mid:]]) -
            np.mean([f["brightness"] for f in all_frames[:mid]])
        )
    else:
        result["brightness_trend"] = 0.0
    return result


def extract_audio_features(video_path):
    try:
        y, sr = librosa.load(str(video_path), sr=16000, mono=True, duration=60)
    except Exception:
        return {}
    if len(y) == 0:
        return {}

    feats = {}
    rms = librosa.feature.rms(y=y)[0]
    feats["mean_loudness"]     = float(rms.mean())
    feats["max_loudness"]      = float(rms.max())
    feats["loudness_variance"] = float(rms.std())

    hook_samples = int(3.0 * sr)
    y_hook = y[:hook_samples] if len(y) > hook_samples else y
    feats["hook_loudness"] = float(librosa.feature.rms(y=y_hook)[0].mean())
    feats["silence_ratio"] = float((rms < 0.01).sum() / max(len(rms), 1))

    flatness = librosa.feature.spectral_flatness(y=y)[0]
    feats["music_score"] = float(1.0 - flatness.mean())

    flatness_hook = librosa.feature.spectral_flatness(y=y_hook)[0]
    feats["hook_music_score"] = float(1.0 - flatness_hook.mean())

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats["tempo_bpm"] = float(np.asarray(tempo).flat[0])
    except Exception:
        feats["tempo_bpm"] = 0.0

    feats["mean_zcr"] = float(librosa.feature.zero_crossing_rate(y)[0].mean())

    mid = len(rms) // 2
    feats["audio_energy_trend"] = float(rms[mid:].mean() - rms[:mid].mean()) if mid > 0 else 0.0

    try:
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_e = float(np.mean(y_harm ** 2))
        perc_e = float(np.mean(y_perc ** 2))
        total  = harm_e + perc_e + 1e-9
        feats["harmonic_ratio"]   = harm_e / total
        feats["percussive_ratio"] = perc_e / total
    except Exception:
        feats["harmonic_ratio"]   = 0.0
        feats["percussive_ratio"] = 0.0

    feats["dynamic_range"] = float(rms.max() - rms.min())
    return feats


def extract_clip_embeddings(video_path):
    cap   = cv2.VideoCapture(str(video_path))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total == 0:
        cap.release()
        return None
    duration = total / fps
    sample_times = [0.5, 2.0, duration * 0.25, duration * 0.50, duration * 0.75, max(0.1, duration - 1.0)]
    sample_times = [max(0.1, min(t, duration - 0.1)) for t in sample_times]

    frame_embeddings = []
    hook_embeddings  = []
    for i, t in enumerate(sample_times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inp = clip_preprocess(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(inp)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            emb = emb.cpu().numpy().flatten()
        frame_embeddings.append(emb)
        if i < 2:
            hook_embeddings.append(emb)
    cap.release()
    if not frame_embeddings:
        return None
    mean_emb = np.mean(frame_embeddings, axis=0)
    hook_emb = np.mean(hook_embeddings, axis=0) if hook_embeddings else mean_emb
    return mean_emb, hook_emb


def extract_vggish_embeddings(video_path):
    if not has_vggish:
        return {}
    try:
        y, sr = librosa.load(str(video_path), sr=16000, mono=True, duration=60)
        if len(y) == 0:
            return {}
        y        = y.astype(np.float32)
        examples = vggish_input.waveform_to_examples(y, sr)
        if len(examples) == 0:
            return {}
        with torch.no_grad():
            inp        = torch.tensor(examples).float().to(device)
            embeddings = _vggish_model(inp).detach().cpu().numpy()
        hook_emb = embeddings[:min(3, len(embeddings))].mean(axis=0)
        mean_emb = embeddings.mean(axis=0)
        result = {}
        for i, v in enumerate(mean_emb):
            result[f"vggish_mean_{i}"] = float(v)
        for i, v in enumerate(hook_emb):
            result[f"vggish_hook_{i}"] = float(v)
        return result
    except Exception:
        return {}


def extract_emotion_features(video_path):
    if not has_fer:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return {}
    try:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        _face_net.setInput(blob)
        dets = _face_net.forward()
        best_face, best_area = None, 0
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < 0.5:
                continue
            x1 = int(dets[0, 0, i, 3] * w)
            y1 = int(dets[0, 0, i, 4] * h)
            x2 = int(dets[0, 0, i, 5] * w)
            y2 = int(dets[0, 0, i, 6] * h)
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_face = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))
        if best_face is None:
            return {"df_emotion_score": 0.0, "df_happy_score": 0.0, "df_surprise_score": 0.0, "df_neutral_score": 0.0}
        x1, y1, x2, y2 = best_face
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return {"df_emotion_score": 0.0, "df_happy_score": 0.0, "df_surprise_score": 0.0, "df_neutral_score": 0.0}
        emotion, scores = _emo_recognizer.predict_emotions(face_crop, logits=False)
        scores_dict = dict(zip(_EMO_LABELS, scores))
        return {
            "df_emotion_score":  EMOTION_MAP.get(emotion, 0.5),
            "df_happy_score":    float(scores_dict.get('Happiness', 0.0)),
            "df_surprise_score": float(scores_dict.get('Surprise',  0.0)),
            "df_neutral_score":  float(scores_dict.get('Neutral',   0.0)),
        }
    except Exception:
        return {}


def extract_temporal_motion(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return {}
    motion_over_time = []
    prev_gray = None
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % max(1, int(fps)) == 0:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                try:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    motion_over_time.append(float(mag.mean()))
                except Exception:
                    pass
            prev_gray = curr_gray
        frame_idx += 1
    cap.release()
    if len(motion_over_time) < 2:
        return {}
    m = np.array(motion_over_time)
    return {
        "temporal_motion_mean":       float(m.mean()),
        "temporal_motion_max":        float(m.max()),
        "temporal_motion_std":        float(m.std()),
        "temporal_hook_motion":       float(m[:3].mean()) if len(m) >= 3 else float(m.mean()),
        "temporal_end_motion":        float(m[-3:].mean()) if len(m) >= 3 else float(m.mean()),
        "temporal_buildup":           float(m[-3:].mean() - m[:3].mean()) if len(m) >= 3 else 0.0,
        "temporal_peak_time":         float(np.argmax(m) / len(m)),
        "temporal_variance":          float(np.var(np.diff(m))),
        "temporal_acceleration":      float(np.mean(np.diff(m))),
        "temporal_high_motion_ratio": float((m > m.mean()).sum() / len(m)),
    }


# ─────────────────────────────────────────────────────────
# METADATA FEATURES from args
# ─────────────────────────────────────────────────────────
def build_metadata_features(title, tags_str, desc, hour, weekday):
    tags      = [t.strip() for t in tags_str.split(',') if t.strip()]
    tag_count = len(tags)

    return {
        "posted_hour":       hour,
        "posted_weekday":    weekday,
        "posted_on_weekend": int(weekday >= 5),
        "posted_prime_time": int(17 <= hour <= 22),
        "duration_seconds":  0,                   # filled from video below
        "title_length":      len(title),
        "title_has_number":  int(any(c.isdigit() for c in title)),
        "title_has_emoji":   int(any(ord(c) > 127 for c in title)),
        "title_caps_ratio":  sum(1 for c in title if c.isupper()) / max(len(title), 1),
        "tag_count":         tag_count,
        "desc_length":       len(desc),
    }


# ─────────────────────────────────────────────────────────
# MAIN — extract everything and predict
# ─────────────────────────────────────────────────────────
print(f"\nExtracting features from: {video_path.name}")

# Get video duration for metadata
cap = cv2.VideoCapture(str(video_path))
fps_v   = cap.get(cv2.CAP_PROP_FPS)
total_v = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
duration_seconds = (total_v / fps_v) if fps_v > 0 else 0

print("  Visual features...")
visual = extract_visual_features(video_path)

print("  Audio features...")
audio = extract_audio_features(video_path)

print("  CLIP embeddings...")
clip_result = extract_clip_embeddings(video_path)

print("  VGGish embeddings...")
vggish = extract_vggish_embeddings(video_path)

print("  Emotion features...")
emotion = extract_emotion_features(video_path)

print("  Temporal motion...")
temporal = extract_temporal_motion(video_path)

print("  Hook features (text overlay + audio type + speech)...")
try:
    from extract_hook_features import extract_all_hook_features
    _wm = _whisper_model if has_whisper else None
    hook_feats = extract_all_hook_features(
        str(video_path),
        whisper_model=_wm,
        run_ocr=True,
        run_whisper=has_whisper,
    )
except Exception as _e:
    print(f"    Hook features skipped: {_e}")
    hook_feats = {}

print("  Title embedding...")
title_emb = title_model.encode(args.title, normalize_embeddings=True) if args.title else np.zeros(384)

print("  Transcript (Whisper)...")
transcript_emb = None
if has_whisper:
    try:
        result_w = _whisper_model.transcribe(str(video_path), fp16=False, language="en")
        text = result_w.get("text", "").strip()
        if text:
            transcript_emb = title_model.encode(text, normalize_embeddings=True)
    except Exception:
        pass

# ── Build feature dict ────────────────────────────────────
feature_dict = {}
feature_dict.update(build_metadata_features(args.title, args.tags, args.desc, args.hour, args.weekday))
feature_dict["duration_seconds"] = duration_seconds
feature_dict.update(visual)
feature_dict.update(audio)
feature_dict.update(emotion)
feature_dict.update(temporal)
feature_dict.update(hook_feats)   # text overlay + audio hook + whisper hook

for i, v in enumerate(title_emb):
    feature_dict[f"title_emb_{i}"] = float(v)

if transcript_emb is not None:
    for i, v in enumerate(transcript_emb):
        feature_dict[f"transcript_emb_{i}"] = float(v)

if clip_result is not None:
    mean_emb, hook_emb = clip_result
    for i, v in enumerate(mean_emb):
        feature_dict[f"clip_mean_{i}"] = float(v)
    for i, v in enumerate(hook_emb):
        feature_dict[f"clip_hook_{i}"] = float(v)

feature_dict.update(vggish)

# Only add channel features if a channel was actually provided.
# If missing, _predict_infer.py falls back to training-set medians
# so the score stays calibrated rather than defaulting to 99th percentile.
if args.channel and channel_subscriber_count > 0:
    feature_dict["log1p_subscriber_count"]  = float(np.log1p(channel_subscriber_count))
    feature_dict["log1p_channel_avg_views"] = float(np.log1p(channel_avg_views))
    feature_dict["log1p_chan_like_rate"]     = float(np.log1p(channel_avg_like_rate))
    feature_dict["log1p_chan_comment_rate"]  = float(np.log1p(channel_avg_comment_rate))

print("  Trending audio match...")
if has_trending_db:
    trending_score = _trending_match(video_path)
    print(f"    trending_audio_score = {trending_score:.3f}")
else:
    trending_score = 0.0
feature_dict["trending_audio_score"] = trending_score

# ── Extract hook frame and save as temp image ─────────────
import json, subprocess, sys, tempfile, os

hook_frame_path = None
try:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_MSEC, 500)  # 0.5s into video
    ret, hook_frame = cap.read()
    cap.release()
    if ret:
        tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_img.close()
        cv2.imwrite(tmp_img.name, hook_frame)
        hook_frame_path = tmp_img.name
except Exception:
    pass

# ── Save features to temp file and run inference in subprocess ────
# GPU models (CLIP, VGGish) hold MPS memory even after del.
# Running inference in a fresh subprocess avoids the segfault.
tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode='w')
json.dump(feature_dict, tmp)
tmp.close()

print("\nRunning inference...")
cmd = [sys.executable, str(Path(__file__).parent / "_predict_infer.py"),
       "--features", tmp.name, "--niche", NICHE,
       "--title", args.title]
if hook_frame_path:
    cmd += ["--hook-frame", hook_frame_path]

result = subprocess.run(cmd, capture_output=False)
os.unlink(tmp.name)
if hook_frame_path:
    # Print path so app.py can read it
    print(f"HOOK_FRAME:{hook_frame_path}")
exit(result.returncode)
