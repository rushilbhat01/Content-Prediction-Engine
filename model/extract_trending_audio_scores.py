"""
extract_trending_audio_scores.py — Batch-score all training videos against
the trending audio database. Run this after build_trending_audio_db.py.

Output: data/features_trending_audio.csv  (video_id, trending_audio_score)

Usage:
    python extract_trending_audio_scores.py
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
from match_trending_audio import match_score, db_size

RAW_DIR  = Path("data/raw_videos")
OUT_PATH = Path("data/features_trending_audio.csv")

if db_size() == 0:
    print("ERROR: Trending audio DB is empty.")
    print("Run: python build_trending_audio_db.py")
    exit(1)

print(f"Trending audio DB: {db_size()} entries")

video_files = list(RAW_DIR.glob("*.mp4"))

if OUT_PATH.exists():
    existing = set(pd.read_csv(OUT_PATH)["video_id"])
else:
    existing = set()

to_process = [v for v in video_files if v.stem not in existing]
print(f"Total: {len(video_files)} | Done: {len(existing)} | To process: {len(to_process)}")

if not to_process:
    print("All videos already scored.")
    exit()

rows = []
for video_path in tqdm(to_process, desc="Scoring trending audio"):
    vid_id = video_path.stem
    score  = match_score(video_path)
    rows.append({"video_id": vid_id, "trending_audio_score": score})

    if len(rows) % 50 == 0:
        df = pd.DataFrame(rows)
        df.to_csv(OUT_PATH, mode="a", header=not OUT_PATH.exists(), index=False)
        rows = []

if rows:
    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, mode="a", header=not OUT_PATH.exists(), index=False)

print(f"\nDone! Saved to {OUT_PATH}")
print(f"Next: retrain with python model_train.py")
