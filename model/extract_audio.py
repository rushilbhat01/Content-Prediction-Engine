import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RAW_DIR  = Path("data/raw_videos")
OUT_PATH = Path("data/features_audio.csv")

def extract_audio_features(video_path):
    try:
        y, sr = librosa.load(str(video_path), sr=16000, mono=True, duration=60)
    except Exception as e:
        return None

    if len(y) == 0:
        return None

    duration = len(y) / sr
    feats    = {}

    # ── Loudness ──────────────────────────────────────────
    rms = librosa.feature.rms(y=y)[0]
    feats["mean_loudness"]    = float(rms.mean())
    feats["max_loudness"]     = float(rms.max())
    feats["loudness_variance"] = float(rms.std())

    # ── Hook loudness (first 3 seconds) ───────────────────
    hook_samples = int(3.0 * sr)
    y_hook       = y[:hook_samples] if len(y) > hook_samples else y
    rms_hook     = librosa.feature.rms(y=y_hook)[0]
    feats["hook_loudness"] = float(rms_hook.mean())

    # ── Silence ratio ─────────────────────────────────────
    feats["silence_ratio"] = float((rms < 0.01).sum() / max(len(rms), 1))

    # ── Music score (spectral flatness) ──────────────────
    # Low flatness = tonal/harmonic signal = music
    # High flatness = noise/white noise = speech or silence
    # music_score = 1 - flatness, so high = music-like
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    feats["music_score"] = float(1.0 - flatness.mean())

    flatness_hook = librosa.feature.spectral_flatness(y=y_hook)[0]
    feats["hook_music_score"] = float(1.0 - flatness_hook.mean())

    # ── Tempo / BPM ───────────────────────────────────────
    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats["tempo_bpm"] = float(np.asarray(tempo).flat[0])
    except Exception:
        feats["tempo_bpm"] = 0.0

    # ── Zero crossing rate (speech vs music proxy) ────────
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    feats["mean_zcr"] = float(zcr.mean())
    feats["std_zcr"]  = float(zcr.std())

    # ── Spectral features ─────────────────────────────────
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats["mean_spectral_centroid"] = float(spectral_centroid.mean())
    feats["std_spectral_centroid"]  = float(spectral_centroid.std())

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats["mean_spectral_rolloff"] = float(spectral_rolloff.mean())

    # ── MFCCs (timbre fingerprint) ────────────────────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i, mfcc in enumerate(mfccs):
        feats[f"mfcc_{i+1}_mean"] = float(mfcc.mean())
        feats[f"mfcc_{i+1}_std"]  = float(mfcc.std())

    # ── Audio energy trajectory ───────────────────────────
    mid          = len(rms) // 2
    if mid > 0:
        first_half   = rms[:mid]
        second_half  = rms[mid:]
        feats["audio_energy_trend"] = float(
            second_half.mean() - first_half.mean()
        )
    else:
        feats["audio_energy_trend"] = 0.0

    # ── Harmonic vs percussive ratio ──────────────────────
    try:
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_energy    = float(np.mean(y_harm ** 2))
        perc_energy    = float(np.mean(y_perc ** 2))
        total_energy   = harm_energy + perc_energy + 1e-9
        feats["harmonic_ratio"]   = harm_energy  / total_energy
        feats["percussive_ratio"] = perc_energy  / total_energy
    except Exception:
        feats["harmonic_ratio"]   = 0.0
        feats["percussive_ratio"] = 0.0

    # ── Dynamic range ─────────────────────────────────────
    feats["dynamic_range"] = float(rms.max() - rms.min())

    feats["audio_duration"] = duration

    return feats


def load_existing():
    if not OUT_PATH.exists():
        return set()
    try:
        return set(pd.read_csv(OUT_PATH)["video_id"])
    except Exception:
        return set()


# ── Main ──────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-list", default=None,
                        help="Path to a text file containing video paths to process")
    parser.add_argument("--append", action="store_true", default=False,
                        help="Append features to the output file")
    args = parser.parse_args()

    existing = load_existing()
    if args.video_list:
        with open(args.video_list, "r", encoding="utf-8") as f:
            video_files = [Path(line.strip()) for line in f if line.strip()]
        to_process = [v for v in video_files if v.stem not in existing]
    else:
        video_files = list(RAW_DIR.glob("*.mp4"))
        to_process  = [v for v in video_files if v.stem not in existing]

print(f"Total videos:   {len(video_files)}")
print(f"Already done:   {len(existing)}")
print(f"To process:     {len(to_process)}")

if not to_process:
    print("All videos already extracted.")
    exit()

rows   = []
failed = 0

for video_path in tqdm(to_process, desc="Extracting audio features"):
    video_id = video_path.stem

    try:
        feats = extract_audio_features(video_path)
    except Exception as e:
        print(f"\n  Failed {video_id}: {e}")
        failed += 1
        continue

    if feats is None:
        failed += 1
        continue

    feats["video_id"] = video_id
    rows.append(feats)

    # Save every 20 videos
    if len(rows) % 20 == 0:
        df           = pd.DataFrame(rows)
        write_header = not OUT_PATH.exists()
        df.to_csv(OUT_PATH, mode='a', header=write_header, index=False)
        rows = []

# Save remaining
if rows:
    df           = pd.DataFrame(rows)
    write_header = not OUT_PATH.exists()
    df.to_csv(OUT_PATH, mode='a', header=write_header, index=False)

print(f"\nDone!")
print(f"  Extracted: {len(to_process) - failed}")
print(f"  Failed:    {failed}")
print(f"  Saved to:  {OUT_PATH}")