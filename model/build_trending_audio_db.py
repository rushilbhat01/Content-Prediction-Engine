"""
build_trending_audio_db.py — Download trending YouTube Shorts audio and
build a fingerprint database for trending-audio detection.

How it works:
  1. yt-dlp searches for trending YT Shorts by query (audio-only download)
  2. For each track, HPSS separates the harmonic (musical) component
  3. Sliding 5-second windows → chroma (12-dim) + MFCC (13-dim) = 25-dim fingerprint
  4. Fingerprints saved to data/trending_audio_db.pkl

Usage:
    # Build with defaults (fitness niche queries, 30 videos each):
    python build_trending_audio_db.py

    # Custom queries + larger limit:
    python build_trending_audio_db.py --limit 50 --queries "gym phonk 2025" "workout music shorts"

    # Force rebuild from scratch:
    python build_trending_audio_db.py --rebuild
"""

import argparse
import subprocess
import shutil
import pickle
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm

DB_PATH  = Path("data/trending_audio_db.pkl")
TMP_DIR  = Path("data/_trending_audio_tmp")

WINDOW_SEC = 5.0
HOP_SEC    = 1.0
SR         = 16000

DEFAULT_QUERIES = [
    "fitness shorts trending 2025",
    "gym workout motivation shorts 2025",
    "phonk gym workout shorts",
    "weight loss transformation shorts viral",
    "calisthenics shorts viral 2024 2025",
    "gym music shorts trending",
    "running motivation shorts viral",
    "bodybuilding shorts trending",
]


def extract_fingerprint(audio_path: Path):
    """
    Extract a (N_windows, 25) chroma+MFCC fingerprint from an audio file.
    Uses HPSS harmonic component for key/chord stability against voice overlays.
    Returns None if audio is too short or fails to load.
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=SR, mono=True, duration=60)
    except Exception as e:
        return None

    if len(y) < SR * 3:          # need at least 3 seconds
        return None

    # Separate harmonic component — robust to voiceover on top of music
    try:
        y_harm, _ = librosa.effects.hpss(y)
    except Exception:
        y_harm = y

    window_samples = int(WINDOW_SEC * SR)
    hop_samples    = int(HOP_SEC   * SR)

    windows = []
    for start in range(0, len(y_harm) - window_samples, hop_samples):
        seg = y_harm[start:start + window_samples]

        # Chroma: captures key / pitch class — key/melody fingerprint
        try:
            chroma      = librosa.feature.chroma_cqt(y=seg, sr=SR, n_chroma=12)
            chroma_mean = chroma.mean(axis=1)          # (12,)
        except Exception:
            chroma_mean = np.zeros(12)

        # MFCC: captures timbre — distinguishes instruments / production style
        try:
            mfcc      = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=13)
            mfcc_mean = mfcc.mean(axis=1)              # (13,)
        except Exception:
            mfcc_mean = np.zeros(13)

        vec = np.concatenate([chroma_mean, mfcc_mean])  # (25,)
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        windows.append(vec)

    if not windows:
        return None

    return np.array(windows, dtype=np.float32)         # (N_windows, 25)


def download_query(query: str, limit: int, existing_ids: set):
    """
    Use yt-dlp to search for `limit` videos matching `query`, download audio,
    extract fingerprints, and return a list of DB entries.
    """
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    search_str = f"ytsearch{limit}:{query}"
    cmd = [
        "yt-dlp", search_str,
        "-x", "--audio-format", "wav",
        "--audio-quality", "5",            # 128 kbps-equivalent, fast enough
        "-o", str(TMP_DIR / "%(id)s.%(ext)s"),
        "--no-playlist",
        "--match-filter", "duration < 180",  # YT Shorts ≤ 3 min
        "--quiet", "--no-warnings",
        "--ignore-errors",
        "--socket-timeout", "30",
    ]

    print(f"  Searching: {query!r} (limit={limit})")
    try:
        subprocess.run(cmd, timeout=600, check=False)
    except subprocess.TimeoutExpired:
        print("  Download timed out — processing what we have")
    except FileNotFoundError:
        print("  ERROR: yt-dlp not found. Install with: pip install yt-dlp")
        return []

    audio_files = list(TMP_DIR.glob("*.wav"))
    new_files   = [f for f in audio_files if f.stem not in existing_ids]
    print(f"  Downloaded {len(audio_files)} files, {len(new_files)} new")

    entries = []
    for audio_file in tqdm(new_files, desc="  Fingerprinting", leave=False):
        fp = extract_fingerprint(audio_file)
        if fp is not None:
            entries.append({
                "id":          audio_file.stem,
                "query":       query,
                "n_windows":   len(fp),
                "fingerprint": fp,
            })

    # Clean up downloaded audio
    shutil.rmtree(TMP_DIR, ignore_errors=True)
    return entries


# ── CLI ───────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Build trending audio fingerprint DB")
parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES,
                    help="Search queries to use")
parser.add_argument("--limit",   type=int, default=30,
                    help="Max videos to download per query (default 30)")
parser.add_argument("--rebuild", action="store_true",
                    help="Ignore existing DB and rebuild from scratch")
args = parser.parse_args()

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Load existing DB ──────────────────────────────────────
if DB_PATH.exists() and not args.rebuild:
    with open(DB_PATH, "rb") as f:
        db = pickle.load(f)
    existing_ids = {e["id"] for e in db}
    print(f"Loaded existing DB: {len(db)} entries")
else:
    db = []
    existing_ids = set()
    print("Building fresh DB")

# ── Download + fingerprint each query ─────────────────────
total_new = 0
for query in args.queries:
    entries  = download_query(query, args.limit, existing_ids)
    new_only = [e for e in entries if e["id"] not in existing_ids]
    for e in new_only:
        existing_ids.add(e["id"])
    db.extend(new_only)
    total_new += len(new_only)
    print(f"  +{len(new_only)} new entries for: {query!r}")

# ── Save ──────────────────────────────────────────────────
with open(DB_PATH, "wb") as f:
    pickle.dump(db, f)

print(f"\nDone. DB: {len(db)} total entries (+{total_new} new)")
print(f"Saved to: {DB_PATH}")
print(f"\nNext: run python extract_trending_audio_scores.py to score training videos,")
print(f"      then retrain with python model_train.py")
