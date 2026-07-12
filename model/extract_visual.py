import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# YuNet face detector — handles angled, profile, and partially occluded faces
# much better than the old ResNet-SSD DNN detector
_YUNET_PATH = str(Path(__file__).parent / "models" / "face_detection_yunet.onnx")
_PROTOTXT   = str(Path(__file__).parent / "models" / "deploy.prototxt")
_WEIGHTS    = str(Path(__file__).parent / "models" / "res10_300x300_ssd_iter_140000.caffemodel")

_yunet = None
_USE_DNN = False
_face_net = None

try:
    _yunet   = cv2.FaceDetectorYN_create(_YUNET_PATH, "", (320, 320), score_threshold=0.4)
    _USE_YUNET = True
    print("Face detector: YuNet (handles angled/profile faces)")
except Exception:
    _USE_YUNET = False
    try:
        _face_net = cv2.dnn.readNetFromCaffe(_PROTOTXT, _WEIGHTS)
        _USE_DNN  = True
        print("Face detector: ResNet-SSD DNN (fallback)")
    except Exception:
        _FACE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("Face detector: Haar cascade (fallback)")

RAW_DIR  = Path("data/raw_videos")
OUT_PATH = Path("data/features_visual.csv")

def extract_frame_features(frame, h, w):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    area = h * w

    # ── Basic pixel stats ─────────────────────────────────
    brightness = float(gray.mean()) / 255
    contrast   = float(gray.std())  / 128
    saturation = float(hsv[:, :, 1].mean()) / 255
    sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # ── Face detection ────────────────────────────────────
    has_face  = 0
    face_size = 0.0
    if _USE_YUNET:
        _yunet.setInputSize((w, h))
        _, dets = _yunet.detect(frame)
        if dets is not None and len(dets) > 0:
            has_face = 1
            # dets columns: x, y, w, h, landmarks..., score
            best = max(dets, key=lambda d: d[2] * d[3])
            face_size = float(best[2] * best[3]) / area
    elif _USE_DNN:
        blob    = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        _face_net.setInput(blob)
        dets = _face_net.forward()
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
        faces = _FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        has_face = int(len(faces) > 0)
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            face_size = (fw * fh) / area

    # ── Edge density ──────────────────────────────────────
    edges        = cv2.Canny(gray, 100, 200)
    edge_density = float(edges.mean()) / 255

    # ── Colour diversity (hue entropy) ────────────────────
    hue_hist = cv2.calcHist([hsv], [0], None, [18], [0, 180])
    hue_hist = hue_hist / (hue_hist.sum() + 1e-9)
    hue_entropy = float(-np.sum(hue_hist * np.log2(hue_hist + 1e-9)))

    # ── Centre brightness ─────────────────────────────────
    cx, cy = w // 2, h // 2
    r      = min(cx, cy) // 2
    centre = gray[cy-r:cy+r, cx-r:cx+r]
    centre_brightness = float(centre.mean()) / 255 if centre.size > 0 else brightness

    # ── Text proxy ────────────────────────────────────────
    top_third    = gray[:h//3, :]
    bottom_third = gray[2*h//3:, :]
    top_edges    = float(cv2.Canny(top_third,    100, 200).mean()) / 255
    bottom_edges = float(cv2.Canny(bottom_third, 100, 200).mean()) / 255
    text_proxy   = max(top_edges, bottom_edges)

    return {
        "brightness":        brightness,
        "contrast":          contrast,
        "saturation":        saturation,
        "sharpness":         sharpness,
        "has_face":          has_face,
        "face_size":         face_size,
        "edge_density":      edge_density,
        "hue_entropy":       hue_entropy,
        "centre_brightness": centre_brightness,
        "text_proxy":        text_proxy,
    }


def compute_motion(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(magnitude.mean())


def extract_features(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None

    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if h == 0 or w == 0:
        cap.release()
        return None

    all_frames    = []
    hook_frames   = []
    motion_scores = []
    prev_gray     = None
    prev_bright   = None
    cut_count     = 0
    frame_idx     = 0

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

            # Motion
            if prev_gray is not None:
                try:
                    motion = compute_motion(prev_gray, curr_gray)
                    motion_scores.append(motion)
                except Exception:
                    pass

            # Cut detection
            if prev_bright is not None:
                if abs(feats["brightness"] - prev_bright) > 0.15:
                    cut_count += 1

            prev_gray   = curr_gray
            prev_bright = feats["brightness"]

            if timestamp <= 3.0:
                hook_frames.append(feats)

        frame_idx += 1

    cap.release()

    if not all_frames:
        return None

    result = {}
    keys   = list(all_frames[0].keys())

    # ── Whole video aggregates ────────────────────────────
    for key in keys:
        vals = [f[key] for f in all_frames]
        result[f"mean_{key}"] = float(np.mean(vals))
        result[f"std_{key}"]  = float(np.std(vals))
        result[f"max_{key}"]  = float(np.max(vals))

    # ── Hook aggregates (first 3 seconds) ─────────────────
    src = hook_frames if hook_frames else all_frames
    for key in keys:
        vals = [f[key] for f in src]
        result[f"hook_{key}"] = float(np.mean(vals))

    # ── Motion ────────────────────────────────────────────
    result["mean_motion"] = float(np.mean(motion_scores)) if motion_scores else 0.0
    result["max_motion"]  = float(np.max(motion_scores))  if motion_scores else 0.0

    # ── Cuts per second ───────────────────────────────────
    result["cuts_per_second"] = cut_count / max(duration, 1)

    # ── Brightness trajectory ─────────────────────────────
    if len(all_frames) >= 2:
        mid         = len(all_frames) // 2
        first_half  = [f["brightness"] for f in all_frames[:mid]]
        second_half = [f["brightness"] for f in all_frames[mid:]]
        result["brightness_trend"] = float(np.mean(second_half) - np.mean(first_half))
    else:
        result["brightness_trend"] = 0.0

    result["total_frames_sampled"] = len(all_frames)

    return result


def load_existing():
    if not OUT_PATH.exists():
        return set()
    try:
        return set(pd.read_csv(OUT_PATH)["video_id"])
    except Exception:
        return set()


# ── Main ──────────────────────────────────────────────────
video_files = list(RAW_DIR.glob("*.mp4"))
existing    = load_existing()
to_process  = [v for v in video_files if v.stem not in existing]

print(f"Total videos:   {len(video_files)}")
print(f"Already done:   {len(existing)}")
print(f"To process:     {len(to_process)}")

if not to_process:
    print("All videos already extracted.")
    exit()

rows   = []
failed = 0

for video_path in tqdm(to_process, desc="Extracting visual features"):
    video_id = video_path.stem

    try:
        feats = extract_features(video_path)
    except Exception as e:
        print(f"\n  Failed {video_id}: {e}")
        failed += 1
        continue

    if feats is None:
        failed += 1
        continue

    feats["video_id"] = video_id
    rows.append(feats)

    # Save every 20 videos so progress isn't lost if it crashes
    if len(rows) % 20 == 0:
        df           = pd.DataFrame(rows)
        write_header = not OUT_PATH.exists()
        df.to_csv(OUT_PATH, mode='a', header=write_header, index=False)
        rows = []
        print(f"\n  Saved checkpoint — {len(existing) + failed + 20} processed so far")

# Save remaining
if rows:
    df           = pd.DataFrame(rows)
    write_header = not OUT_PATH.exists()
    df.to_csv(OUT_PATH, mode='a', header=write_header, index=False)

print(f"\nDone!")
print(f"  Extracted: {len(to_process) - failed}")
print(f"  Failed:    {failed}")
print(f"  Saved to:  {OUT_PATH}")