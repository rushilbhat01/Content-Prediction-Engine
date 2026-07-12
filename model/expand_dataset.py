"""
expand_dataset.py
Finds all videos with metadata but no video file, downloads them,
then extracts visual / audio / embedding / hook features for ONLY the
new videos (skips anything already in the existing feature CSVs).

Run:
    python expand_dataset.py            # download + extract all missing
    python expand_dataset.py --no-download  # extract only (if already downloaded)
    python expand_dataset.py --niche food   # filter to one niche only
"""

import argparse
import os
import subprocess
import sys
import time
import random
import tempfile
import shutil
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
META_CSV   = Path("data/metadata_clean.csv")
VIDEO_DIR  = Path("data/raw_videos")
VIS_CSV    = Path("data/features_visual.csv")
AUD_CSV    = Path("data/features_audio.csv")
EMB_CSV    = Path("data/features_embeddings.csv")
HOOK_CSV   = Path("data/hook_features.csv")


def load_ids(csv_path, col="video_id"):
    if not csv_path.exists():
        return set()
    try:
        return set(pd.read_csv(csv_path, usecols=[col])[col].astype(str))
    except Exception:
        return set()


# ─────────────────────────────────────────────────────────────────────────────
def download_missing(niche_filter=None):
    """Download videos in metadata that have no video file."""
    meta = pd.read_csv(META_CSV, on_bad_lines="skip", engine="python")
    if niche_filter:
        meta = meta[meta["niche"] == niche_filter]

    meta_ids   = set(meta["video_id"].astype(str))
    downloaded = {p.stem for p in VIDEO_DIR.glob("*.mp4")}
    downloaded |= {p.stem for p in VIDEO_DIR.glob("*.mov")}
    to_dl      = list(meta_ids - downloaded)

    print(f"\n{'='*55}")
    print(f" DOWNLOAD PHASE")
    print(f"{'='*55}")
    print(f"  Need to download: {len(to_dl):,} videos")
    if not to_dl:
        print("  Nothing to download.")
        return

    success = failed = 0
    for i, vid_id in enumerate(to_dl, 1):
        out = VIDEO_DIR / f"{vid_id}.mp4"
        if out.exists():
            success += 1
            continue

        url = f"https://www.youtube.com/watch?v={vid_id}"
        print(f"  [{i}/{len(to_dl)}] {vid_id} ... ", end="", flush=True)

        ret = os.system(
            f'yt-dlp -o "data/raw_videos/%(id)s.%(ext)s" '
            f'--merge-output-format mp4 '
            f'--cookies-from-browser brave '
            f'--quiet --no-warnings '
            f'"{url}"'
        )

        if ret == 0 and out.exists():
            success += 1
            print("OK")
        else:
            failed += 1
            print("failed")

        time.sleep(random.uniform(2, 4))

    print(f"\n  Downloaded: {success}  Failed: {failed}")


# ─────────────────────────────────────────────────────────────────────────────
def run_extractor_on_new(extractor_script, feature_csv, niche_filter=None):
    """
    Run an extractor script on only the videos not yet in feature_csv.
    Creates a temporary symlink/copy directory so the extractor processes
    only the new videos.
    """
    done_ids   = load_ids(feature_csv)
    all_videos = list(VIDEO_DIR.glob("*.mp4")) + list(VIDEO_DIR.glob("*.mov"))

    if niche_filter:
        meta = pd.read_csv(META_CSV, usecols=["video_id", "niche"],
                           on_bad_lines="skip", engine="python")
        niche_ids = set(meta[meta["niche"] == niche_filter]["video_id"].astype(str))
        all_videos = [v for v in all_videos if v.stem in niche_ids]

    new_videos = [v for v in all_videos if v.stem not in done_ids]

    if not new_videos:
        print(f"  {extractor_script}: nothing new to process.")
        return

    print(f"  {extractor_script}: {len(new_videos):,} new videos to process...")

    # Create a temp dir with symlinks (or copies on Windows) to new videos only
    tmp_dir = Path(tempfile.mkdtemp(prefix="yt_new_"))
    try:
        for v in new_videos:
            dst = tmp_dir / v.name
            try:
                dst.symlink_to(v.resolve())
            except (OSError, NotImplementedError):
                # Windows: fall back to copying (slow but works)
                shutil.copy2(v, dst)

        # Patch the extractor to read from tmp_dir
        # The extractors read from data/raw_videos — we temporarily swap the dir
        # by passing an env var that we check in the extractor, OR just
        # run the extractor with a modified VIDEO_DIR env var.
        env = os.environ.copy()
        env["YT_VIDEO_DIR"] = str(tmp_dir)

        ret = subprocess.run(
            [sys.executable, extractor_script],
            env=env,
            cwd=Path.cwd(),
        )
        if ret.returncode != 0:
            print(f"  WARNING: {extractor_script} exited with code {ret.returncode}")
        else:
            print(f"  {extractor_script}: done.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
def extract_new_simple(niche_filter=None):
    """
    Simpler approach: identify new video IDs, append-extract them by
    temporarily patching the video list in each extractor.
    Uses subprocess with a helper that processes only specific IDs.
    """
    done_vis  = load_ids(VIS_CSV)
    done_aud  = load_ids(AUD_CSV)
    done_emb  = load_ids(EMB_CSV)
    done_hook = load_ids(HOOK_CSV)

    all_videos = sorted(VIDEO_DIR.glob("*.mp4"))

    if niche_filter:
        meta = pd.read_csv(META_CSV, usecols=["video_id", "niche"],
                           on_bad_lines="skip", engine="python")
        niche_ids = set(meta[meta["niche"] == niche_filter]["video_id"].astype(str))
        all_videos = [v for v in all_videos if v.stem in niche_ids]

    new_vis  = [v for v in all_videos if v.stem not in done_vis]
    new_aud  = [v for v in all_videos if v.stem not in done_aud]
    new_emb  = [v for v in all_videos if v.stem not in done_emb]
    new_hook = [v for v in all_videos if v.stem not in done_hook]

    print(f"\n{'='*55}")
    print(f" EXTRACTION PHASE")
    print(f"{'='*55}")
    print(f"  New for visual:     {len(new_vis):,}")
    print(f"  New for audio:      {len(new_aud):,}")
    print(f"  New for embeddings: {len(new_emb):,}")
    print(f"  New for hook:       {len(new_hook):,}")
    print()

    # Write a temporary list of new video paths for each stage
    def write_id_list(videos, fname):
        p = Path(f"data/_new_{fname}.txt")
        p.write_text("\n".join(str(v.resolve()) for v in videos), encoding="utf-8")
        return p

    # ── Visual ────────────────────────────────────────────────────────────────
    if new_vis:
        id_file = write_id_list(new_vis, "visual")
        print(f"[1/4] Running extract_visual_fast.py on {len(new_vis):,} new videos...")
        ret = subprocess.run(
            [sys.executable, "extract_visual_fast.py",
             "--video-list", str(id_file), "--append"],
            check=False,
        )
        id_file.unlink(missing_ok=True)
        if ret.returncode != 0:
            print("  WARNING: visual extraction had errors — continuing")

    # ── Audio ─────────────────────────────────────────────────────────────────
    if new_aud:
        id_file = write_id_list(new_aud, "audio")
        print(f"\n[2/4] Running extract_audio.py on {len(new_aud):,} new videos...")
        ret = subprocess.run(
            [sys.executable, "extract_audio.py",
             "--video-list", str(id_file), "--append"],
            check=False,
        )
        id_file.unlink(missing_ok=True)
        if ret.returncode != 0:
            print("  WARNING: audio extraction had errors — continuing")

    # ── Embeddings ────────────────────────────────────────────────────────────
    if new_emb:
        id_file = write_id_list(new_emb, "embeddings")
        print(f"\n[3/4] Running extract_embeddings.py on {len(new_emb):,} new videos...")
        ret = subprocess.run(
            [sys.executable, "extract_embeddings.py",
             "--video-list", str(id_file), "--append"],
            check=False,
        )
        id_file.unlink(missing_ok=True)
        if ret.returncode != 0:
            print("  WARNING: embedding extraction had errors — continuing")

    # ── Hook ──────────────────────────────────────────────────────────────────
    if new_hook:
        print(f"\n[4/4] Running hook extraction on {len(new_hook):,} new videos...")
        ret = subprocess.run(
            [sys.executable, "run_hook_extraction.py", "--resume"],
            check=False,
        )
        if ret.returncode != 0:
            print("  WARNING: hook extraction had errors — continuing")

    print(f"\n{'='*55}")
    print(" All extraction done.")
    print(f"{'='*55}")
    print("\nNext: run RETRAIN_AND_START.bat (or manually retrain)")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-download", action="store_true",
                        help="Skip download, only run extraction")
    parser.add_argument("--niche", default=None,
                        help="Filter to one niche (e.g. food)")
    args = parser.parse_args()

    print(f"\nExpand dataset pipeline")
    if args.niche:
        print(f"  Niche filter: {args.niche}")

    if not args.no_download:
        download_missing(niche_filter=args.niche)
    else:
        print("\nSkipping download (--no-download)")

    extract_new_simple(niche_filter=args.niche)


if __name__ == "__main__":
    main()
