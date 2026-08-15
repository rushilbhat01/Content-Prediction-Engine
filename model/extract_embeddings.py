import cv2
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import librosa
import warnings
warnings.filterwarnings('ignore')

# ── Device setup ──────────────────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Device: {device}")

# ── Load pretrained models ────────────────────────────────

# 1. CLIP — best for visual semantic understanding
#    Pretrained on 400M image-text pairs
#    Understands WHAT is in the frame semantically
print("Loading CLIP...")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# 2. MiniLM — best for title/text understanding
#    Pretrained on 1B sentence pairs
#    Understands what the title MEANS, not just its length
print("Loading MiniLM...")
title_model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. torchvggish — PyTorch port of VGGish, no TensorFlow needed
#    Pretrained on AudioSet (2M YouTube videos)
#    Understands sound categories: music, speech, noise etc
print("Loading Whisper (tiny)...")
try:
    import whisper as _whisper
    _whisper_model = _whisper.load_model("tiny", device="cpu")  # cpu — avoids MPS memory conflict
    has_whisper = True
    print("  Whisper loaded")
except Exception as e:
    print(f"  Whisper failed: {e}")
    has_whisper = False

print("Loading VGGish...")
try:
    import torchvggish
    from torchvggish import vggish_input
    _vggish_model = torchvggish.vggish()
    _vggish_model.eval()
    has_vggish = True
    print("  VGGish (torch) loaded successfully")
except Exception as e:
    print(f"  VGGish failed to load: {e}")
    has_vggish = False

# 4. hsemotion-onnx — ONNX facial emotion recognition, no TensorFlow needed
#    Uses DNN face detector (already downloaded) + ONNX emotion model
print("Loading FER (hsemotion-onnx)...")
try:
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    _PROTOTXT_EMO = str(Path(__file__).parent / "models" / "deploy.prototxt")
    _WEIGHTS_EMO  = str(Path(__file__).parent / "models" / "res10_300x300_ssd_iter_140000.caffemodel")
    _face_net_emo = cv2.dnn.readNetFromCaffe(_PROTOTXT_EMO, _WEIGHTS_EMO)
    _emo_recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew')
    has_fer = True
    print("  FER (hsemotion) ready")
except Exception as e:
    print(f"  FER failed: {e}")
    has_fer = False

# ── Paths ─────────────────────────────────────────────────
RAW_DIR   = Path("data/raw_videos")
META_PATH = Path("data/metadata_clean.csv")
OUT_PATH  = Path("data/features_embeddings.csv")

# Load titles for MiniLM
meta         = pd.read_csv(META_PATH)[["video_id", "title"]]
title_lookup = dict(zip(meta.video_id, meta.title))

# ─────────────────────────────────────────────────────────
# EXTRACTOR 1: CLIP visual embeddings
# What it gives: semantic understanding of video content
# ─────────────────────────────────────────────────────────
def extract_clip_embeddings(video_path):
    cap      = cv2.VideoCapture(str(video_path))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total == 0:
        cap.release()
        return None

    duration = total / fps

    # 6 strategic frames — hook weighted twice
    sample_times = [
        0.5,              # start of hook
        2.0,              # end of hook
        duration * 0.25,
        duration * 0.50,
        duration * 0.75,
        max(0.1, duration - 1.0),
    ]
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
        if i < 2:  # first 2 = hook frames
            hook_embeddings.append(emb)

    cap.release()

    if not frame_embeddings:
        return None

    mean_emb = np.mean(frame_embeddings, axis=0)  # 512-dim
    hook_emb = np.mean(hook_embeddings, axis=0) if hook_embeddings else mean_emb

    return mean_emb, hook_emb


# ─────────────────────────────────────────────────────────
# EXTRACTOR 2: FER — emotion detection from hook frame
# What it gives: is the person happy? surprised? confident?
# These are real virality signals
# ─────────────────────────────────────────────────────────
EMOTION_MAP = {
    'Happiness': 1.0,
    'Surprise':  0.8,
    'Neutral':   0.5,
    'Sadness':   0.2,
    'Anger':     0.3,
    'Fear':      0.1,
    'Disgust':   0.1,
    'Contempt':  0.2,
}
# Fixed label order for enet_b0_8_best_afew
_EMO_LABELS = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

_EMPTY_EMOTION = {
    "df_emotion_score":  0.0,
    "df_happy_score":    0.0,
    "df_surprise_score": 0.0,
    "df_neutral_score":  0.0,
    "df_face_detected":  0,
}

def extract_emotion_features(video_path):
    if not has_fer:
        return None

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    try:
        h, w = frame.shape[:2]

        # Detect face with DNN detector
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        _face_net_emo.setInput(blob)
        dets = _face_net_emo.forward()

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
            return _EMPTY_EMOTION.copy()

        x1, y1, x2, y2 = best_face
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return _EMPTY_EMOTION.copy()

        emotion, scores = _emo_recognizer.predict_emotions(face_crop, logits=False)
        scores_dict = dict(zip(_EMO_LABELS, scores))

        return {
            "df_emotion_score":  EMOTION_MAP.get(emotion, 0.5),
            "df_happy_score":    float(scores_dict.get('Happiness', 0.0)),
            "df_surprise_score": float(scores_dict.get('Surprise',  0.0)),
            "df_neutral_score":  float(scores_dict.get('Neutral',   0.0)),
            "df_face_detected":  1,
        }

    except Exception:
        return _EMPTY_EMOTION.copy()


# ─────────────────────────────────────────────────────────
# EXTRACTOR 3: torchvggish audio embeddings
# What it gives: semantic audio understanding
# Music type, energy category, sound texture
# ─────────────────────────────────────────────────────────
def extract_vggish_embeddings(video_path):
    if not has_vggish:
        return None

    try:
        y, sr = librosa.load(str(video_path), sr=16000, mono=True, duration=60)
        if len(y) == 0:
            return None

        y        = y.astype(np.float32)
        examples = vggish_input.waveform_to_examples(y, sr)
        if len(examples) == 0:
            return None

        with torch.no_grad():
            inp        = torch.tensor(examples).float()  # (n_frames, 96, 64)
            embeddings = _vggish_model(inp).numpy()      # (n_frames, 128)

        if len(embeddings) == 0:
            return None

        hook_emb = embeddings[:min(3, len(embeddings))].mean(axis=0)
        mean_emb = embeddings.mean(axis=0)

        result = {}
        for i, v in enumerate(mean_emb):
            result[f"vggish_mean_{i}"] = float(v)
        for i, v in enumerate(hook_emb):
            result[f"vggish_hook_{i}"] = float(v)

        return result

    except Exception:
        return None


# ─────────────────────────────────────────────────────────
# EXTRACTOR 4: Optical flow temporal motion
# What it gives: HOW the video changes over time
# Not a pretrained model but captures temporal patterns
# that CLIP (single frames) completely misses
# ─────────────────────────────────────────────────────────
def extract_temporal_motion(video_path):
    cap  = cv2.VideoCapture(str(video_path))
    fps  = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None

    motion_over_time = []
    prev_gray        = None
    frame_idx        = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % max(1, int(fps)) == 0:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                try:
                    flow      = cv2.calcOpticalFlowFarneback(
                        prev_gray, curr_gray,
                        None, 0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    mag, _    = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    motion_over_time.append(float(mag.mean()))
                except Exception:
                    pass
            prev_gray = curr_gray

        frame_idx += 1

    cap.release()

    if len(motion_over_time) < 2:
        return None

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
# EXTRACTOR 5: Whisper transcript embeddings
# What it gives: what is actually being SAID in the video
# Captures "challenge", "transformation", "watch till end"
# type signals that visual/audio features completely miss
# ─────────────────────────────────────────────────────────
def extract_transcript_embedding(video_path):
    if not has_whisper:
        return None
    try:
        result = _whisper_model.transcribe(str(video_path), fp16=False, language="en")
        text   = result.get("text", "").strip()
        if not text:
            return None
        emb = title_model.encode(text, normalize_embeddings=True)
        return emb  # 384-dim
    except Exception:
        return None


# ─────────────────────────────────────────────────────────
# EXTRACTOR 6: MiniLM title embeddings
# What it gives: semantic meaning of the title
# Understands that "30 day transformation" and
# "monthly fitness challenge" are similar
# ─────────────────────────────────────────────────────────
def get_title_embedding(title):
    if not isinstance(title, str) or len(title.strip()) == 0:
        return np.zeros(384)
    return title_model.encode(title, normalize_embeddings=True)


# ─────────────────────────────────────────────────────────
# Load already processed
# ─────────────────────────────────────────────────────────
def load_existing():
    if not OUT_PATH.exists():
        return set()
    try:
        return set(pd.read_csv(OUT_PATH)["video_id"])
    except Exception:
        return set()


# ─────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────
video_files = list(RAW_DIR.glob("*.mp4"))
existing    = load_existing()
to_process  = [v for v in video_files if v.stem not in existing]

print(f"\nTotal videos:  {len(video_files)}")
print(f"Already done:  {len(existing)}")
print(f"To process:    {len(to_process)}")
print(f"\nModels active:")
print(f"  CLIP ViT-B/32:     always on  (visual semantics)")
print(f"  MiniLM-L6-v2:      always on  (title meaning)")
print(f"  Optical flow:      always on  (temporal motion)")
print(f"  FER:               {'ON' if has_fer     else 'OFF'} (emotion detection)")
print(f"  VGGish:            {'ON' if has_vggish  else 'OFF'} (audio semantics)")

if not to_process:
    print("\nAll done already.")
    exit()

rows   = []
failed = 0

for video_path in tqdm(to_process, desc="Extracting embeddings"):
    video_id = video_path.stem
    row      = {"video_id": video_id}

    # ── 1. CLIP visual embeddings ─────────────────────────
    try:
        result = extract_clip_embeddings(video_path)
        if result is None:
            failed += 1
            continue
        mean_emb, hook_emb = result
        for i, v in enumerate(mean_emb):
            row[f"clip_mean_{i}"] = float(v)
        for i, v in enumerate(hook_emb):
            row[f"clip_hook_{i}"] = float(v)
    except Exception as e:
        print(f"\n  CLIP failed {video_id}: {e}")
        failed += 1
        continue

    # ── 2. FER emotion analysis ───────────────────────────
    if has_fer:
        try:
            fer_feats = extract_emotion_features(video_path)
            if fer_feats:
                row.update(fer_feats)
        except Exception as e:
            print(f"\n  FER failed {video_id}: {e}")

    # ── 3. VGGish audio embeddings ────────────────────────
    if has_vggish:
        try:
            vggish_feats = extract_vggish_embeddings(video_path)
            if vggish_feats:
                row.update(vggish_feats)
        except Exception as e:
            print(f"\n  VGGish failed {video_id}: {e}")

    # ── 4. Temporal motion ────────────────────────────────
    try:
        motion = extract_temporal_motion(video_path)
        if motion:
            row.update(motion)
    except Exception as e:
        print(f"\n  Motion failed {video_id}: {e}")

    # ── 5. Whisper transcript embedding ──────────────────
    if has_whisper:
        try:
            t_emb = extract_transcript_embedding(video_path)
            if t_emb is not None:
                for i, v in enumerate(t_emb):
                    row[f"transcript_emb_{i}"] = float(v)
        except Exception as e:
            print(f"\n  Whisper failed {video_id}: {e}")

    # ── 6. MiniLM title embedding ─────────────────────────
    try:
        title = title_lookup.get(video_id, "")
        t_emb = get_title_embedding(title)
        for i, v in enumerate(t_emb):
            row[f"title_emb_{i}"] = float(v)
    except Exception as e:
        print(f"\n  Title failed {video_id}: {e}")

    rows.append(row)

    # Checkpoint every 20 videos
    if len(rows) % 20 == 0:
        df           = pd.DataFrame(rows)
        write_header = not OUT_PATH.exists()
        df.to_csv(OUT_PATH, mode='a', header=write_header, index=False)
        rows = []

if rows:
    df           = pd.DataFrame(rows)
    write_header = not OUT_PATH.exists()
    df.to_csv(OUT_PATH, mode='a', header=write_header, index=False)

print(f"\nDone!")
print(f"  Extracted: {len(to_process) - failed}")
print(f"  Failed:    {failed}")
print(f"\nFeatures per video:")
print(f"  CLIP mean:      512 dims → PCA 50  (whole video semantics)")
print(f"  CLIP hook:      512 dims → PCA 30  (hook semantics)")
print(f"  VGGish mean:    128 dims → PCA 20  (audio semantics)")
print(f"  VGGish hook:    128 dims → PCA 10  (hook audio)")
print(f"  DeepFace:       8 values            (emotion, age, gender)")
print(f"  Temporal:       10 values           (motion over time)")
print(f"  Title MiniLM:   384 dims → PCA 20  (title meaning)")
print(f"  Saved → {OUT_PATH}")