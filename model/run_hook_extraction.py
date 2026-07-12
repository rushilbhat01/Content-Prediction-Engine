"""
run_hook_extraction.py
Batch-extracts hook features (audio type + Whisper speech content) for all
videos in data/raw_videos/, saves to data/hook_features.csv.

Run:
    python run_hook_extraction.py
    python run_hook_extraction.py --no-speech   # skip Whisper (fastest)
    python run_hook_extraction.py --resume      # skip already-done videos
    python run_hook_extraction.py --workers 4
    python run_hook_extraction.py --test        # run on 5 videos first
"""

import argparse
import csv
import multiprocessing as mp
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import time
import traceback
from pathlib import Path

import pandas as pd

VIDEO_DIR  = Path("data/raw_videos")
OUTPUT_CSV = Path("data/hook_features.csv")
ERROR_LOG  = Path("data/hook_extraction_errors.txt")
EXTS       = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# ── Worker (one per process) ──────────────────────────────────────────────────
_run_whisper = True

def _worker_init(run_whisper: bool, device: str):
    """Suppress warnings + pre-load Whisper inside each worker process."""
    import warnings
    import logging

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    # Silence specific noisy loggers
    for name in ("librosa", "audioread", "numba", "transformers", "torch"):
        logging.getLogger(name).setLevel(logging.ERROR)

    global _run_whisper
    _run_whisper = run_whisper

    if run_whisper:
        try:
            import whisper
            import builtins
            builtins._hook_whisper = whisper.load_model("tiny", device=device)
        except Exception:
            pass  # Whisper unavailable — will return defaults


def _process_one(video_path: str) -> dict:
    """Extract hook features for one video. Returns flat dict."""
    import warnings
    warnings.filterwarnings("ignore")

    vid = Path(video_path).stem
    try:
        import builtins
        from extract_hook_features import (
            extract_audio_hook_features,
            extract_whisper_hook_features,
        )

        feats = {"video_id": vid}
        feats.update(extract_audio_hook_features(video_path))

        if _run_whisper:
            wm = getattr(builtins, "_hook_whisper", None)
            feats.update(extract_whisper_hook_features(video_path,
                                                        whisper_model=wm))
        return feats

    except Exception:
        return {"video_id": vid, "_error": traceback.format_exc()[:300]}


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers",   type=int,
                        default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--no-speech", action="store_true",
                        help="Skip Whisper speech features")
    parser.add_argument("--resume",    action="store_true",
                        help="Skip videos already in output CSV")
    parser.add_argument("--test",      action="store_true",
                        help="Run on first 5 videos only (sanity check)")
    parser.add_argument("--device",    default=os.getenv("YT_SHORTS_DEVICE", "cuda"),
                        choices=("cuda", "cpu"),
                        help="Device for Whisper speech features")
    args = parser.parse_args()

    run_whisper = not args.no_speech
    if run_whisper and args.device == "cuda" and args.workers > 1:
        print("CUDA Whisper uses one worker to avoid duplicate model loads in GPU memory.")
        args.workers = 1

    # ── Collect videos ────────────────────────────────────────────────────────
    all_videos = sorted(
        p for p in VIDEO_DIR.rglob("*") if p.suffix.lower() in EXTS
    )

    if args.test:
        all_videos = all_videos[:5]
        print("TEST MODE — processing 5 videos only")

    if args.resume and OUTPUT_CSV.exists():
        try:
            done = set(pd.read_csv(OUTPUT_CSV)["video_id"].astype(str))
            all_videos = [v for v in all_videos if v.stem not in done]
            print(f"Resuming: {len(all_videos):,} videos remaining")
        except Exception:
            print("Existing CSV is empty/corrupt — starting fresh")
            OUTPUT_CSV.unlink()


    total = len(all_videos)
    if total == 0:
        print("No videos to process.")
        return

    print(f"\nHook feature extraction")
    print(f"  Videos  : {total:,}")
    print(f"  Workers : {args.workers}")
    print(f"  Whisper : {'yes' if run_whisper else 'no'}")
    print(f"  Device  : {args.device if run_whisper else 'n/a'}")
    print(f"  Output  : {OUTPUT_CSV}")
    print()

    # ── Open output ───────────────────────────────────────────────────────────
    append = args.resume and OUTPUT_CSV.exists() and not args.test
    out_file = open(OUTPUT_CSV, "a" if append else "w",
                    newline="", encoding="utf-8")
    writer = None
    error_lines = []

    t0 = time.time()
    done_count = errors = 0

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(run_whisper, args.device),
    ) as pool:
        for result in pool.imap_unordered(
            _process_one,
            [str(v) for v in all_videos],
            chunksize=1,
        ):
            if "_error" in result:
                errors += 1
                error_lines.append(
                    f"{result['video_id']}: {result['_error'][:120]}\n"
                )
            else:
                if writer is None:
                    fieldnames = list(result.keys())
                    writer = csv.DictWriter(out_file, fieldnames=fieldnames,
                                           extrasaction="ignore")
                    if not append:
                        writer.writeheader()
                writer.writerow(result)
                out_file.flush()

            done_count += 1
            elapsed = time.time() - t0
            spd     = elapsed / done_count
            eta     = spd * (total - done_count)
            pct     = done_count / total * 100
            filled  = int(30 * done_count / total)
            bar     = "█" * filled + "░" * (30 - filled)

            sys.stdout.write(
                f"\r  [{bar}] {pct:5.1f}%  "
                f"{done_count}/{total}  "
                f"{spd:.1f}s/vid  "
                f"ETA {eta/60:.0f}min  "
                f"errors={errors}  "
            )
            sys.stdout.flush()

    out_file.close()

    # ── Save error log ────────────────────────────────────────────────────────
    if error_lines:
        with open(ERROR_LOG, "w", encoding="utf-8") as ef:
            ef.writelines(error_lines[:50])  # save first 50 errors
        print(f"\n\n  Errors logged to {ERROR_LOG}")

    elapsed_total = time.time() - t0
    print(f"\n\nDone in {elapsed_total/60:.1f} min")
    print(f"  {done_count - errors:,} OK  |  {errors} errors")
    print(f"  Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
