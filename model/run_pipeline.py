"""
Full overnight pipeline:
  Step 1 — Extract visual features from already-downloaded videos (fast parallel)
  Step 2 — Download remaining fitness (16) + food (477) videos
  Step 3 — Extract visual features from the newly downloaded videos
  Step 4 — Print final status summary

Usage:
    python run_pipeline.py
    python run_pipeline.py --workers 12
    python run_pipeline.py --skip-step1   # if step 1 already done
"""

import argparse
import subprocess
import sys
import time
import random
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--workers",    type=int, default=8)
parser.add_argument("--skip-step1", action="store_true", help="Skip initial extraction (already done)")
args = parser.parse_args()

META_PATH   = Path("data/metadata.csv")
VISUAL_PATH = Path("data/features_visual.csv")
RAW_DIR     = Path("data/raw_videos")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def status_snapshot(label=""):
    meta    = pd.read_csv(META_PATH, on_bad_lines="skip", engine="python")
    vis_ids = set(pd.read_csv(VISUAL_PATH)["video_id"].astype(str)) if VISUAL_PATH.exists() else set()
    dl_ids  = {p.stem for p in RAW_DIR.glob("*.mp4")}
    print(f"\n{'='*55}  {label}")
    for niche in ["fitness", "food"]:
        niche_ids = set(meta[meta["niche"]==niche]["video_id"].astype(str))
        done      = len(niche_ids & vis_ids)
        ready     = len((niche_ids & dl_ids) - vis_ids)
        need_dl   = len(niche_ids - vis_ids - dl_ids)
        print(f"  {niche:8s}  features={done:>4}  ready_to_extract={ready:>4}  need_dl={need_dl:>4}  total={len(niche_ids)}")
    print("="*55)

def run_extraction(workers):
    print(f"\n>>> Running extract_visual_fast.py --workers {workers}")
    result = subprocess.run(
        [sys.executable, "extract_visual_fast.py", "--workers", str(workers)],
        check=False
    )
    return result.returncode

def download_remaining():
    meta    = pd.read_csv(META_PATH, on_bad_lines="skip", engine="python")
    vis_ids = set(pd.read_csv(VISUAL_PATH)["video_id"].astype(str)) if VISUAL_PATH.exists() else set()
    dl_ids  = {p.stem for p in RAW_DIR.glob("*.mp4")}

    needed = []
    for niche in ["fitness", "food"]:
        niche_ids = set(meta[meta["niche"]==niche]["video_id"].astype(str))
        batch     = list(niche_ids - vis_ids - dl_ids)
        print(f"  {niche}: {len(batch)} to download")
        needed.extend(batch)

    if not needed:
        print("  Nothing to download!")
        return 0

    print(f"\n  Total: {len(needed)} videos to download\n")
    ok = fail = 0
    for i, vid_id in enumerate(needed, 1):
        url      = f"https://www.youtube.com/watch?v={vid_id}"
        out_path = RAW_DIR / f"{vid_id}.mp4"
        print(f"  [{i}/{len(needed)}] {vid_id}...", end=" ", flush=True)

        result = subprocess.run(
            [
                "yt-dlp",
                "-o", str(RAW_DIR / "%(id)s.%(ext)s"),
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
                "--merge-output-format", "mp4",
                "--cookies-from-browser", "brave",
                "--quiet", "--no-warnings",
                url,
            ],
            capture_output=True,
        )

        if result.returncode == 0 and out_path.exists():
            print("ok")
            ok += 1
        else:
            err = result.stderr.decode(errors="ignore").strip()
            print(f"FAILED  {err[:80] if err else ''}")
            fail += 1

        time.sleep(random.uniform(2, 4))

    print(f"\n  Downloads done. ok={ok}  failed={fail}")
    return fail


# ── MAIN ─────────────────────────────────────────────────────────────────────
t_start = time.time()
status_snapshot("START")

# ── Step 1: Extract from already-downloaded videos ────────────────────────────
if not args.skip_step1:
    print("\n\n>>> STEP 1: Extract visual features (parallel)\n")
    run_extraction(args.workers)
    status_snapshot("AFTER STEP 1")
else:
    print("\n>>> STEP 1 skipped.")

# ── Step 2: Download remaining videos ────────────────────────────────────────
print("\n\n>>> STEP 2: Download remaining videos\n")
download_remaining()
status_snapshot("AFTER STEP 2")

# ── Step 3: Extract newly downloaded videos ───────────────────────────────────
print("\n\n>>> STEP 3: Extract newly downloaded videos\n")
run_extraction(args.workers)
status_snapshot("FINAL STATUS")

elapsed = (time.time() - t_start) / 3600
print(f"\nTotal pipeline time: {elapsed:.1f} hours")
print("\nAll done! Next steps:")
print("  1. Train food model:    set NICHE='food' in model_train.py  →  python model_train.py")
print("  2. Train fitness model: set NICHE='fitness' in model_train.py →  python model_train.py")
