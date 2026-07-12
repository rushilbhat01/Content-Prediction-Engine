"""
download_new.py — Download only videos that have never been feature-extracted.
Reads video IDs from data/to_download.txt.
"""
import os, time, random
from pathlib import Path

IDS_FILE = Path("data/to_download.txt")
RAW_DIR  = Path("data/raw_videos")
RAW_DIR.mkdir(exist_ok=True)

ids = [l.strip() for l in IDS_FILE.read_text().splitlines() if l.strip()]
print(f"To download: {len(ids)}")

success = failed = 0
for i, vid_id in enumerate(ids, 1):
    out = RAW_DIR / f"{vid_id}.mp4"
    if out.exists():
        success += 1
        continue

    url = f"https://www.youtube.com/watch?v={vid_id}"
    print(f"[{i}/{len(ids)}] {vid_id}...")
    code = os.system(
        f'yt-dlp -o "data/raw_videos/%(id)s.%(ext)s" '
        f'--js-runtimes node '
        f'--merge-output-format mp4 '
        f'--cookies-from-browser brave '
        f'--quiet --no-warnings '
        f'"{url}"'
    )
    if code == 0:
        success += 1
    else:
        failed += 1
    time.sleep(random.uniform(3, 6))

print(f"\nDone. Success: {success}  Failed: {failed}")
