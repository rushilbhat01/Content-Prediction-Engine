"""
Fast parallel visual feature extractor.

Speedups vs original:
  1. Parallel workers (default: 8) — biggest win
  2. Resize to 360p before all processing — ~4x faster per frame
  3. Sample every 2 seconds instead of 1
  4. Optical flow on 120p thumbnail — even faster motion estimation
  5. Each worker writes to its own tmp file — no lock contention

Usage:
    python extract_visual_fast.py              # 8 workers
    python extract_visual_fast.py --workers 12
    python extract_visual_fast.py --workers 16
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import multiprocessing as mp
import os
from pathlib import Path
from tqdm import tqdm

RAW_DIR  = Path("data/raw_videos")
OUT_PATH = Path("data/features_visual.csv")
TMP_DIR  = Path("data/_visual_tmp")

SAMPLE_EVERY_N_SEC = 2      # sample 1 frame per N seconds (was 1)
MAX_H              = 360    # resize frames to this height max
FLOW_H             = 120    # optical flow on even smaller frames
FACE_SCORE_THRESH  = 0.4

YUNET_PATH = str(Path(__file__).parent / "models" / "face_detection_yunet.onnx")
PROTOTXT   = str(Path(__file__).parent / "models" / "deploy.prototxt")
WEIGHTS    = str(Path(__file__).parent / "models" / "res10_300x300_ssd_iter_140000.caffemodel")


# ─── per-frame feature extraction ────────────────────────────────────────────

def make_detector():
    """Create face detector — called once per worker process."""
    try:
        det = cv2.FaceDetectorYN_create(YUNET_PATH, "", (320, 320), score_threshold=FACE_SCORE_THRESH)
        return ("yunet", det)
    except Exception:
        pass
    try:
        net = cv2.dnn.readNetFromCaffe(PROTOTXT, WEIGHTS)
        return ("dnn", net)
    except Exception:
        pass
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return ("haar", casc)


def extract_frame_features(frame, detector_tuple):
    h, w = frame.shape[:2]
    area  = h * w

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    brightness = float(gray.mean()) / 255
    contrast   = float(gray.std())  / 128
    saturation = float(hsv[:, :, 1].mean()) / 255
    sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Face detection
    det_type, det = detector_tuple
    has_face  = 0
    face_size = 0.0
    if det_type == "yunet":
        det.setInputSize((w, h))
        _, dets = det.detect(frame)
        if dets is not None and len(dets) > 0:
            has_face = 1
            best = max(dets, key=lambda d: d[2] * d[3])
            face_size = float(best[2] * best[3]) / area
    elif det_type == "dnn":
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        det.setInput(blob)
        detections = det.forward()
        best_area = 0.0
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < 0.35:
                continue
            has_face = 1
            bw = float(detections[0, 0, i, 5] - detections[0, 0, i, 3])
            bh = float(detections[0, 0, i, 6] - detections[0, 0, i, 4])
            a  = bw * bh
            if a > best_area:
                best_area = a
        face_size = best_area
    else:
        faces = det.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        has_face = int(len(faces) > 0)
        if len(faces) > 0:
            fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            face_size = (fw * fh) / area

    edges        = cv2.Canny(gray, 100, 200)
    edge_density = float(edges.mean()) / 255

    hue_hist    = cv2.calcHist([hsv], [0], None, [18], [0, 180])
    hue_hist    = hue_hist / (hue_hist.sum() + 1e-9)
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


def compute_motion(prev_flow_gray, curr_flow_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_flow_gray, curr_flow_gray,
        None, 0.5, 2, 10, 2, 5, 1.1, 0   # fewer pyramid levels + smaller window = faster
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(magnitude.mean())


def extract_features(video_path, detector_tuple):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None

    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if orig_h == 0 or orig_w == 0:
        cap.release()
        return None

    # Compute resize scale to cap at MAX_H
    scale = min(1.0, MAX_H / orig_h)
    proc_h = int(orig_h * scale)
    proc_w = int(orig_w * scale)
    flow_h = FLOW_H
    flow_w = max(1, int(orig_w * flow_h / max(orig_h, 1)))

    sample_interval = max(1, int(fps * SAMPLE_EVERY_N_SEC))

    all_frames    = []
    hook_frames   = []
    motion_scores = []
    prev_flow_gray = None
    prev_bright    = None
    cut_count      = 0
    frame_idx      = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            timestamp = frame_idx / fps

            # Resize for processing
            if scale < 1.0:
                frame_proc = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
            else:
                frame_proc = frame

            try:
                feats = extract_frame_features(frame_proc, detector_tuple)
            except Exception:
                frame_idx += 1
                continue

            all_frames.append(feats)

            # Motion: use tiny downscale
            curr_flow_gray = cv2.cvtColor(
                cv2.resize(frame_proc, (flow_w, flow_h), interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_BGR2GRAY
            )
            if prev_flow_gray is not None:
                try:
                    motion = compute_motion(prev_flow_gray, curr_flow_gray)
                    motion_scores.append(motion)
                except Exception:
                    pass

            # Cut detection
            if prev_bright is not None:
                if abs(feats["brightness"] - prev_bright) > 0.15:
                    cut_count += 1

            prev_flow_gray = curr_flow_gray
            prev_bright    = feats["brightness"]

            if timestamp <= 3.0:
                hook_frames.append(feats)

        frame_idx += 1

    cap.release()

    if not all_frames:
        return None

    result = {}
    keys   = list(all_frames[0].keys())

    for key in keys:
        vals = [f[key] for f in all_frames]
        result[f"mean_{key}"] = float(np.mean(vals))
        result[f"std_{key}"]  = float(np.std(vals))
        result[f"max_{key}"]  = float(np.max(vals))

    src = hook_frames if hook_frames else all_frames
    for key in keys:
        vals = [f[key] for f in src]
        result[f"hook_{key}"] = float(np.mean(vals))

    result["mean_motion"]      = float(np.mean(motion_scores)) if motion_scores else 0.0
    result["max_motion"]       = float(np.max(motion_scores))  if motion_scores else 0.0
    result["cuts_per_second"]  = cut_count / max(duration, 1)

    if len(all_frames) >= 2:
        mid         = len(all_frames) // 2
        first_half  = [f["brightness"] for f in all_frames[:mid]]
        second_half = [f["brightness"] for f in all_frames[mid:]]
        result["brightness_trend"] = float(np.mean(second_half) - np.mean(first_half))
    else:
        result["brightness_trend"] = 0.0

    result["total_frames_sampled"] = len(all_frames)
    return result


# ─── Worker entry point ───────────────────────────────────────────────────────

def worker_process(args):
    """Called in each worker process. Processes a chunk of video paths."""
    chunk, worker_id, tmp_path_str = args
    tmp_path  = Path(tmp_path_str)
    detector  = make_detector()

    rows   = []
    failed = 0

    for video_path in chunk:
        video_id = Path(video_path).stem
        try:
            feats = extract_features(video_path, detector)
        except Exception as e:
            failed += 1
            continue

        if feats is None:
            failed += 1
            continue

        feats["video_id"] = video_id
        rows.append(feats)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(tmp_path, index=False)

    return worker_id, len(rows), failed


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel worker processes (default: 8)")
    parser.add_argument("--video-list", default=None,
                        help="Path to a text file containing video paths to process")
    parser.add_argument("--append", action="store_true", default=False,
                        help="Append features to the output file")
    args = parser.parse_args()

    N_WORKERS = args.workers

    TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Load already-done IDs
    existing = set()
    if OUT_PATH.exists():
        try:
            existing = set(pd.read_csv(OUT_PATH)["video_id"].astype(str))
        except Exception:
            pass

    if args.video_list:
        with open(args.video_list, "r", encoding="utf-8") as f:
            video_files = [Path(line.strip()) for line in f if line.strip()]
        video_files = [v for v in video_files if v.stem not in existing]
    else:
        video_files = [v for v in RAW_DIR.glob("*.mp4") if v.stem not in existing]

    print(f"Total in raw_videos:  {len(list(RAW_DIR.glob('*.mp4')))}")
    print(f"Already extracted:    {len(existing)}")
    print(f"To process now:       {len(video_files)}")
    print(f"Workers:              {N_WORKERS}")
    print(f"Frame sample rate:    every {SAMPLE_EVERY_N_SEC}s")
    print(f"Processing res:       max {MAX_H}p")
    print(f"Optical flow res:     {FLOW_H}p")
    print()

    if not video_files:
        print("Nothing to process.")
        exit(0)

    # Split into chunks for each worker
    chunks = [[] for _ in range(N_WORKERS)]
    for i, vf in enumerate(video_files):
        chunks[i % N_WORKERS].append(str(vf))

    # Build args for each worker
    worker_args = [
        (chunks[i], i, str(TMP_DIR / f"worker_{i}.csv"))
        for i in range(N_WORKERS)
        if chunks[i]   # skip empty chunks
    ]

    print(f"Dispatching {len(worker_args)} workers...")
    print("Progress bars appear per-worker below:\n")

    # Use spawn context for Windows compatibility
    ctx  = mp.get_context("spawn")
    pool = ctx.Pool(processes=N_WORKERS)

    results = []
    with tqdm(total=len(video_files), desc="Total progress", position=0, smoothing=0.1) as pbar:
        for wid, n_ok, n_fail in pool.imap_unordered(worker_process, worker_args):
            results.append((wid, n_ok, n_fail))
            pbar.update(n_ok + n_fail)
            pbar.set_postfix({"workers_done": len(results), "ok": sum(r[1] for r in results)})

    pool.close()
    pool.join()

    # Merge tmp CSVs into main output
    print("\nMerging results...")
    tmp_dfs = []
    for i in range(N_WORKERS):
        tmp_path = TMP_DIR / f"worker_{i}.csv"
        if tmp_path.exists():
            try:
                tmp_dfs.append(pd.read_csv(tmp_path))
                tmp_path.unlink()   # clean up
            except Exception:
                pass

    if tmp_dfs:
        merged = pd.concat(tmp_dfs, ignore_index=True)
        write_header = not OUT_PATH.exists()
        merged.to_csv(OUT_PATH, mode="a", header=write_header, index=False)
        print(f"Saved {len(merged)} new rows to {OUT_PATH}")
    else:
        print("No new rows to save.")

    total_ok   = sum(r[1] for r in results)
    total_fail = sum(r[2] for r in results)
    print(f"\nDone!  Extracted: {total_ok}   Failed: {total_fail}")

    # Clean up tmp dir if empty
    try:
        TMP_DIR.rmdir()
    except Exception:
        pass
