import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import cv2
import numpy as np
import pandas as pd
import torch
import clip_compat as clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import librosa
from runtime_device import move_model_to_device, select_torch_device
import warnings
warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--device", default=os.getenv("YT_SHORTS_DEVICE", "cuda"),
                    choices=("cuda", "cpu", "auto"),
                    help="Torch device preference")
parser.add_argument("--fast", action="store_true", default=True,
                    help="Use balanced fast defaults: skip Whisper and sample less audio/motion")
parser.add_argument("--full", action="store_false", dest="fast",
                    help="Run the slower full feature extraction")
parser.add_argument("--whisper", action="store_true",
                    help="Enable transcript embeddings. Slow.")
parser.add_argument("--motion-seconds", type=float, default=2.0,
                    help="Seconds between motion samples in fast mode")
parser.add_argument("--audio-duration", type=float, default=20.0,
                    help="Seconds of audio for VGGish in fast mode")
parser.add_argument("--turbo", action="store_true",
                    help="Fastest mode: CLIP + title only, safest for large backfills")
parser.add_argument("--no-vggish", action="store_true",
                    help="Skip VGGish audio embeddings")
parser.add_argument("--no-fer", action="store_true",
                    help="Skip face emotion features")
parser.add_argument("--no-motion", action="store_true",
                    help="Skip temporal motion features")
parser.add_argument("--clip-frames", type=int, default=4,
                    help="Number of CLIP frames per video in fast mode")
parser.add_argument("--checkpoint", type=int, default=10,
                    help="Write progress every N videos")
parser.add_argument("--limit", type=int, default=0,
                    help="Process only the first N remaining videos")
parser.add_argument("--output", default="data/features_embeddings.csv",
                    help="Output CSV path")
parser.add_argument("--resume-mode", default="complete",
                    choices=("complete", "any"),
                    help="complete redoes rows missing active feature groups; any skips every existing video_id")
parser.add_argument("--workers", type=int, default=8,
                    help="CPU frame decode workers for turbo mode")
parser.add_argument("--batch-size", type=int, default=128,
                    help="CLIP frame batch size for turbo mode")
parser.add_argument("--video-batch", type=int, default=256,
                    help="Videos to stage at once in turbo mode")
parser.add_argument("--video-list", default=None,
                    help="Path to a text file containing video paths to process")
parser.add_argument("--append", action="store_true", default=False,
                    help="Append features to the output file")
args = parser.parse_args()
if args.turbo:
    args.no_vggish = True
    args.no_fer = True
    args.no_motion = True
    args.clip_frames = max(1, min(args.clip_frames, 2))

# ── Device setup ──────────────────────────────────────────
device = select_torch_device(torch, prefer=args.device)

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
title_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# 3. torchvggish — PyTorch port of VGGish, no TensorFlow needed
#    Pretrained on AudioSet (2M YouTube videos)
#    Understands sound categories: music, speech, noise etc
has_whisper = False
_whisper_model = None
if args.whisper or not args.fast:
    print("Loading Whisper (tiny)...")
    try:
        import whisper as _whisper
        _whisper_model = _whisper.load_model("tiny", device=device)
        has_whisper = True
        print("  Whisper loaded")
    except Exception as e:
        print(f"  Whisper failed: {e}")

has_vggish = False
if not args.no_vggish:
    print("Loading VGGish...")
    try:
        import torchvggish
        from torchvggish import vggish_input
        _vggish_model = move_model_to_device(torchvggish.vggish(), device)
        if hasattr(_vggish_model, "pproc"):
            _vggish_model.pproc._pca_matrix = _vggish_model.pproc._pca_matrix.to(device)
            _vggish_model.pproc._pca_means = _vggish_model.pproc._pca_means.to(device)
        has_vggish = True
        print("  VGGish (torch) loaded successfully")
    except Exception as e:
        print(f"  VGGish failed to load: {e}")

# 4. hsemotion-onnx — ONNX facial emotion recognition, no TensorFlow needed
#    Uses DNN face detector (already downloaded) + ONNX emotion model
has_fer = False
if not args.no_fer:
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

# ── Paths ─────────────────────────────────────────────────
RAW_DIR   = Path("data/raw_videos")
META_PATH = Path("data/metadata_clean.csv")
OUT_PATH  = Path(args.output)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

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
    if args.fast:
        sample_times = [0.5, 2.0, duration * 0.50, max(0.1, duration - 1.0)]
        sample_times = sample_times[:max(1, args.clip_frames)]
    else:
        sample_times = [
            0.5,
            2.0,
            duration * 0.25,
            duration * 0.50,
            duration * 0.75,
            max(0.1, duration - 1.0),
        ]
    sample_times = [max(0.1, min(t, duration - 0.1)) for t in sample_times]

    frames = []

    for i, t in enumerate(sample_times):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        frames.append(clip_preprocess(pil))

    cap.release()

    if not frames:
        return None

    with torch.no_grad():
        inp = torch.stack(frames).to(device)
        emb = clip_model.encode_image(inp)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        frame_embeddings = emb.detach().cpu().numpy()

    mean_emb = np.mean(frame_embeddings, axis=0)  # 512-dim
    hook_emb = np.mean(frame_embeddings[:min(2, len(frame_embeddings))], axis=0)

    return mean_emb, hook_emb


def read_clip_frame_images(video_path):
    cap      = cv2.VideoCapture(str(video_path))
    fps      = cap.get(cv2.CAP_PROP_FPS)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total == 0:
        cap.release()
        return video_path, []

    duration = total / fps
    sample_times = [0.5]
    if args.clip_frames > 1:
        sample_times.append(2.0)
    if args.clip_frames > 2:
        sample_times.extend([duration * 0.50, max(0.1, duration - 1.0)])
    sample_times = sample_times[:max(1, args.clip_frames)]
    sample_times = [max(0.1, min(t, duration - 0.1)) for t in sample_times]

    frames = []
    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))

    cap.release()
    return video_path, frames


def encode_clip_frame_batches(frame_tensors):
    chunks = []
    with torch.no_grad():
        for i in range(0, len(frame_tensors), args.batch_size):
            inp = torch.stack(frame_tensors[i:i + args.batch_size]).to(device)
            emb = clip_model.encode_image(inp)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            chunks.append(emb.detach().cpu().numpy())
    return np.vstack(chunks) if chunks else np.empty((0, 512), dtype=np.float32)


def process_turbo_batch(video_paths, title_embeddings):
    decoded = []
    failed = 0
    max_workers = max(1, args.workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(read_clip_frame_images, v) for v in video_paths]
        for fut in as_completed(futures):
            video_path, frames = fut.result()
            if frames:
                decoded.append((video_path, frames))
            else:
                failed += 1

    if not decoded:
        return [], failed

    frame_tensors = []
    spans = []
    pos = 0
    for video_path, frames in decoded:
        tensors = [clip_preprocess(frame) for frame in frames]
        frame_tensors.extend(tensors)
        spans.append((video_path, pos, pos + len(tensors)))
        pos += len(tensors)

    embeddings = encode_clip_frame_batches(frame_tensors)
    rows = []
    for video_path, start, end in spans:
        video_id = video_path.stem
        frame_embeddings = embeddings[start:end]
        if len(frame_embeddings) == 0:
            failed += 1
            continue

        row = {"video_id": video_id}
        mean_emb = np.mean(frame_embeddings, axis=0)
        hook_emb = np.mean(frame_embeddings[:min(2, len(frame_embeddings))], axis=0)

        for i, v in enumerate(mean_emb):
            row[f"clip_mean_{i}"] = float(v)
        for i, v in enumerate(hook_emb):
            row[f"clip_hook_{i}"] = float(v)

        title_emb = title_embeddings.get(video_id)
        if title_emb is None:
            title_emb = get_title_embedding(title_lookup.get(video_id, ""))
        for i, v in enumerate(title_emb):
            row[f"title_emb_{i}"] = float(v)

        rows.append(row)

    return rows, failed


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
        audio_duration = args.audio_duration if args.fast else 60
        y, sr = librosa.load(str(video_path), sr=16000, mono=True,
                             duration=audio_duration)
        if len(y) == 0:
            return None

        y        = y.astype(np.float32)
        examples = vggish_input.waveform_to_examples(y, sr)
        if len(examples) == 0:
            return None

        with torch.no_grad():
            inp        = torch.tensor(examples).float().to(device)  # (n_frames, 96, 64)
            embeddings = _vggish_model(inp).detach().cpu().numpy()  # (n_frames, 128)

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
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        cap.release()
        return None

    motion_over_time = []
    prev_gray        = None

    duration = total / fps if total > 0 else 0
    if args.fast:
        sample_times = np.arange(0.0, min(duration, 20.0), args.motion_seconds)
    else:
        sample_times = np.arange(0.0, duration, 1.0)

    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000)
        ret, frame = cap.read()
        if not ret:
            continue

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
        result = _whisper_model.transcribe(str(video_path), fp16=(device == "cuda"), language="en")
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


def batch_title_embeddings(video_paths):
    titles = [title_lookup.get(v.stem, "") for v in video_paths]
    clean_titles = [t if isinstance(t, str) and t.strip() else "" for t in titles]
    if not clean_titles:
        return {}
    embs = title_model.encode(
        clean_titles,
        batch_size=128 if device == "cuda" else 32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return {v.stem: embs[i] for i, v in enumerate(video_paths)}


def append_rows(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    write_header = not OUT_PATH.exists()
    if not write_header:
        try:
            existing_df = pd.read_csv(OUT_PATH)
            existing_cols = list(existing_df.columns)
            df = df.reindex(columns=existing_cols)
            replacing = set(df["video_id"].astype(str))
            existing_df = existing_df[
                ~existing_df["video_id"].astype(str).isin(replacing)
            ]
            df = pd.concat([existing_df, df], ignore_index=True)
            df.to_csv(OUT_PATH, index=False)
            return
        except Exception:
            write_header = True
    df.to_csv(OUT_PATH, mode='a', header=write_header, index=False)


# ─────────────────────────────────────────────────────────
# Load already processed
# ─────────────────────────────────────────────────────────
def load_existing():
    if not OUT_PATH.exists():
        return set()
    try:
        if args.resume_mode == "any":
            done = set()
            with open(OUT_PATH, newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if row and row[0]:
                        done.add(str(row[0]).strip())
            return done

        df = pd.read_csv(OUT_PATH)
        required = (
            [c for c in df.columns if c.startswith("clip_mean_")]
            + [c for c in df.columns if c.startswith("clip_hook_")]
            + [c for c in df.columns if c.startswith("title_emb_")]
        )
        if has_vggish and not args.no_vggish:
            required += [c for c in df.columns if c.startswith("vggish_mean_")]
            required += [c for c in df.columns if c.startswith("vggish_hook_")]
        if not args.no_motion:
            required += [c for c in df.columns if c.startswith("temporal_")]
        if has_fer and not args.no_fer:
            required += [
                "df_emotion_score",
                "df_happy_score",
                "df_surprise_score",
                "df_neutral_score",
                "df_face_detected",
            ]

        required = [c for c in required if c in df.columns]
        if not required:
            return set()
        complete = df[required].notna().all(axis=1)
        return set(df.loc[complete, "video_id"].astype(str))
    except Exception:
        return set()


# ─────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────
if args.video_list:
    with open(args.video_list, "r", encoding="utf-8") as f:
        video_files = [Path(line.strip()) for line in f if line.strip()]
else:
    video_files = list(RAW_DIR.glob("*.mp4"))
existing    = load_existing()
to_process  = [v for v in video_files if v.stem not in existing]
if args.limit > 0:
    to_process = to_process[:args.limit]

print(f"\nTotal videos:  {len(video_files)}")
print(f"Already done:  {len(existing)}")
print(f"To process:    {len(to_process)}")
print(f"\nModels active:")
print(f"  CLIP ViT-B/32:     always on  (visual semantics)")
print(f"  MiniLM-L6-v2:      always on  (title meaning)")
print(f"  Optical flow:      {'OFF' if args.no_motion else 'ON'} (temporal motion)")
print(f"  FER:               {'ON' if has_fer     else 'OFF'} (emotion detection)")
print(f"  VGGish:            {'ON' if has_vggish  else 'OFF'} (audio semantics)")
print(f"  Whisper:           {'ON' if has_whisper else 'OFF'} (transcripts)")
print(f"  Mode:              {'TURBO' if args.turbo else ('BALANCED' if args.fast else 'FULL')}")

if not to_process:
    print("\nAll done already.")
    exit()

rows   = []
failed = 0

if args.turbo:
    for i in tqdm(range(0, len(to_process), args.video_batch),
                  desc="Turbo batches"):
        batch = to_process[i:i + args.video_batch]
        title_embeddings = batch_title_embeddings(batch)
        batch_rows, batch_failed = process_turbo_batch(batch, title_embeddings)
        failed += batch_failed
        rows.extend(batch_rows)
        if len(rows) >= args.checkpoint:
            append_rows(rows)
            rows = []

    if rows:
        append_rows(rows)

    print(f"\nDone!")
    print(f"  Extracted: {len(to_process) - failed}")
    print(f"  Failed:    {failed}")
    print(f"  Saved → {OUT_PATH}")
    exit()

existing_data = {}
if OUT_PATH.exists():
    try:
        df_existing = pd.read_csv(OUT_PATH)
        existing_data = {str(r["video_id"]).strip(): r.to_dict() for _, r in df_existing.iterrows()}
        print(f"Loaded {len(existing_data)} existing video feature rows for update optimization.")
    except Exception as e:
        print(f"Warning: could not load existing features_embeddings.csv: {e}")

print("\nPrecomputing title embeddings...")
title_embeddings = batch_title_embeddings(to_process)

for video_path in tqdm(to_process, desc="Extracting embeddings"):
    video_id = video_path.stem
    
    # Load existing row features if available, otherwise start fresh
    existing_row = existing_data.get(video_id, {})
    row = {"video_id": video_id}
    # Clean up NaNs from loaded existing row
    for k, v in existing_row.items():
        if pd.notna(v) and k != "video_id":
            row[k] = v

    # ── 1. CLIP visual embeddings ─────────────────────────
    if "clip_mean_0" not in row:
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
    if has_fer and "df_emotion_score" not in row:
        try:
            fer_feats = extract_emotion_features(video_path)
            if fer_feats:
                row.update(fer_feats)
        except Exception as e:
            print(f"\n  FER failed {video_id}: {e}")

    # ── 3. VGGish audio embeddings ────────────────────────
    if has_vggish and "vggish_mean_0" not in row:
        try:
            vggish_feats = extract_vggish_embeddings(video_path)
            if vggish_feats:
                row.update(vggish_feats)
        except Exception as e:
            print(f"\n  VGGish failed {video_id}: {e}")

    # ── 4. Temporal motion ────────────────────────────────
    if not args.no_motion and "temporal_motion_mean" not in row:
        try:
            motion = extract_temporal_motion(video_path)
            if motion:
                row.update(motion)
        except Exception as e:
            print(f"\n  Motion failed {video_id}: {e}")

    # ── 5. Whisper transcript embedding ──────────────────
    if has_whisper and "transcript_emb_0" not in row:
        try:
            t_emb = extract_transcript_embedding(video_path)
            if t_emb is not None:
                for i, v in enumerate(t_emb):
                    row[f"transcript_emb_{i}"] = float(v)
        except Exception as e:
            print(f"\n  Whisper failed {video_id}: {e}")

    # ── 6. MiniLM title embedding ─────────────────────────
    if "title_emb_0" not in row:
        try:
            t_emb = title_embeddings.get(video_id)
            if t_emb is None:
                t_emb = get_title_embedding(title_lookup.get(video_id, ""))
            for i, v in enumerate(t_emb):
                row[f"title_emb_{i}"] = float(v)
        except Exception as e:
            print(f"\n  Title failed {video_id}: {e}")

    rows.append(row)

    if len(rows) >= args.checkpoint:
        append_rows(rows)
        rows = []


if rows:
    append_rows(rows)

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
