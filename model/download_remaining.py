"""
Download remaining videos that don't yet have visual features extracted.
Handles both fitness and food niches, skips already-downloaded files.

Usage:
    python download_remaining.py                    # all niches
    python download_remaining.py --niche fitness    # fitness only
    python download_remaining.py --niche food       # food only
    python download_remaining.py --dry-run          # just show counts
"""

import argparse
import subprocess
import time
import random
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--niche",    default="all", choices=["all", "fitness", "food"])
parser.add_argument("--dry-run",  action="store_true", help="Show what would be downloaded without doing it")
args = parser.parse_args()

META_PATH   = Path("data/metadata.csv")
VISUAL_PATH = Path("data/features_visual.csv")
RAW_DIR     = Path("data/raw_videos")
RAW_DIR.mkdir(parents=True, exist_ok=True)

meta    = pd.read_csv(META_PATH, on_bad_lines="skip", engine="python")
vis_ids = set(pd.read_csv(VISUAL_PATH)["video_id"].astype(str)) if VISUAL_PATH.exists() else set()
dl_ids  = {p.stem for p in RAW_DIR.glob("*.mp4")}

niches = ["fitness", "food"] if args.niche == "all" else [args.niche]

all_needed = []
for niche in niches:
    niche_meta = meta[meta["niche"] == niche]
    niche_ids  = set(niche_meta["video_id"].astype(str))
    needed     = list(niche_ids - vis_ids - dl_ids)
    print(f"\n[{niche.upper()}]")
    print(f"  Total in metadata:       {len(niche_meta)}")
    print(f"  Already have features:   {len(niche_ids & vis_ids)}")
    print(f"  Downloaded (no feat yet):{len(niche_ids & dl_ids - vis_ids)}")
    print(f"  Still need to download:  {len(needed)}")
    all_needed.extend(needed)

print(f"\n{'='*50}")
print(f"Total to download across all selected niches: {len(all_needed)}")

if args.dry_run or not all_needed:
    if not all_needed:
        print("\nNothing to download!")
    else:
        print("\n[DRY RUN] Would download the above. Remove --dry-run to proceed.")
    exit()

print(f"\nStarting downloads...\n")
ok = fail = 0

for i, vid_id in enumerate(all_needed, 1):
    url = f"https://www.youtube.com/watch?v={vid_id}"
    out_path = RAW_DIR / f"{vid_id}.mp4"
    print(f"[{i}/{len(all_needed)}] {vid_id}...", end=" ", flush=True)

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
        print(f"FAILED  {err[:80] if err else '(no error msg)'}")
        fail += 1

    time.sleep(random.uniform(3, 6))

print(f"\n{'='*50}")
print(f"Downloads complete.  OK: {ok}   Failed: {fail}")
print(f"\nNext step: run  python extract_visual.py  to extract features from downloaded videos.")
