"""
Re-download missing videos, extract visual features, then zip and move to external drive.

Usage:
    python redownload_extract.py --niche fitness
    python redownload_extract.py --niche food
    python redownload_extract.py --niche fitness --zip-to "D:/yt_shorts_backup"
    python redownload_extract.py --niche fitness --skip-download  # only extract + zip
    python redownload_extract.py --niche fitness --skip-extract   # only download + zip
"""

import argparse
import shutil
import subprocess
import time
import random
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--niche",         default="fitness")
parser.add_argument("--zip-to",        default=None, help="Path to move zip after extraction, e.g. D:/yt_shorts_backup")
parser.add_argument("--skip-download", action="store_true", help="Skip download step")
parser.add_argument("--skip-extract",  action="store_true", help="Skip extraction step")
args = parser.parse_args()

NICHE       = args.niche
RAW_DIR     = Path("data/raw_videos")
META_PATH   = Path("data/metadata.csv")
VISUAL_PATH = Path("data/features_visual.csv")

RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Step 1: Download ─────────────────────────────────────────────────────────
if not args.skip_download:
    meta = pd.read_csv(META_PATH, on_bad_lines="skip", engine="python")
    meta = meta[meta["niche"] == NICHE]

    if VISUAL_PATH.exists():
        done_ids = set(pd.read_csv(VISUAL_PATH)["video_id"].astype(str))
    else:
        done_ids = set()

    already_downloaded = {p.stem for p in RAW_DIR.glob("*.mp4")}
    needed = meta[
        ~meta["video_id"].astype(str).isin(done_ids) &
        ~meta["video_id"].astype(str).isin(already_downloaded)
    ]["video_id"].tolist()

    print(f"Niche: {NICHE}")
    print(f"Already have visual features: {len(done_ids)}")
    print(f"Already downloaded (no features yet): {len(already_downloaded)}")
    print(f"Still need to download: {len(needed)}\n")

    dl_ok = dl_fail = 0
    for i, vid_id in enumerate(needed, 1):
        url = f"https://www.youtube.com/watch?v={vid_id}"
        print(f"[{i}/{len(needed)}] {vid_id}...", end=" ", flush=True)

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

        if result.returncode == 0 and (RAW_DIR / f"{vid_id}.mp4").exists():
            print("ok")
            dl_ok += 1
        else:
            print("failed")
            dl_fail += 1

        time.sleep(random.uniform(3, 6))

    print(f"\nDownload complete. ok={dl_ok}  failed={dl_fail}\n")
else:
    print("Skipping download step.\n")

# ── Step 2: Extract visual features ──────────────────────────────────────────
if not args.skip_extract:
    downloaded = list(RAW_DIR.glob("*.mp4"))
    if downloaded:
        print(f"Extracting visual features from {len(downloaded)} videos...")
        subprocess.run(["python", "extract_visual.py"], check=False)
        print()
    else:
        print("No videos in data/raw_videos -- nothing to extract.\n")
else:
    print("Skipping extraction step.\n")

# ── Step 3: Zip and move to external drive ────────────────────────────────────
if args.zip_to:
    zip_dest = Path(args.zip_to)
    zip_dest.mkdir(parents=True, exist_ok=True)

    zip_name = f"raw_videos_{NICHE}"
    zip_path = zip_dest / zip_name

    print(f"Zipping data/raw_videos -> {zip_path}.zip ...")
    shutil.make_archive(str(zip_path), "zip", str(RAW_DIR.parent), RAW_DIR.name)
    size_gb = (zip_path.with_suffix(".zip")).stat().st_size / 1e9
    print(f"Done. Archive: {zip_path}.zip ({size_gb:.1f} GB)")
else:
    print("No --zip-to path given. Videos kept in data/raw_videos.")
    print("When ready to archive, run:")
    print(f'  python redownload_extract.py --niche {NICHE} --skip-download --skip-extract --zip-to "D:/yt_shorts_backup"')
