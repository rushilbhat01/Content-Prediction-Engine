"""
match_trending_audio.py — Score a video against the trending audio database.

Returns a float in [0, 1]:
  0.0 = audio does not match any trending track
  1.0 = strong match to a known trending sound

Strategy:
  - Load HPSS harmonic component from the test video
  - Extract sliding-window chroma+MFCC fingerprint (same as build_trending_audio_db.py)
  - For each DB entry, compute pairwise cosine similarities between
    test windows and reference windows
  - Take the mean of the top-5 (test, ref) window pairs as that song's similarity
  - Return the max similarity across all DB songs, normalised to [0, 1]
"""

import pickle
import numpy as np
import librosa
from pathlib import Path
from functools import lru_cache

DB_PATH    = Path("data/trending_audio_db.pkl")
WINDOW_SEC = 5.0
HOP_SEC    = 1.0
SR         = 16000

# Cosine similarity thresholds (empirical)
# < THRESH_LOW  → no match (unrelated music or speech-only)
# > THRESH_HIGH → confident match (same trending track)
THRESH_LOW  = 0.45
THRESH_HIGH = 0.85


@lru_cache(maxsize=1)
def _load_db():
    if not DB_PATH.exists():
        return []
    with open(DB_PATH, "rb") as f:
        return pickle.load(f)


def _extract_fingerprint(y: np.ndarray, sr: int):
    """Extract (N_windows, 25) chroma+MFCC fingerprint from audio array."""
    if len(y) < sr * 3:
        return None

    try:
        y_harm, _ = librosa.effects.hpss(y)
    except Exception:
        y_harm = y

    window_samples = int(WINDOW_SEC * sr)
    hop_samples    = int(HOP_SEC   * sr)

    windows = []
    for start in range(0, len(y_harm) - window_samples, hop_samples):
        seg = y_harm[start:start + window_samples]

        try:
            chroma      = librosa.feature.chroma_cqt(y=seg, sr=sr, n_chroma=12)
            chroma_mean = chroma.mean(axis=1)
        except Exception:
            chroma_mean = np.zeros(12)

        try:
            mfcc      = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
            mfcc_mean = mfcc.mean(axis=1)
        except Exception:
            mfcc_mean = np.zeros(13)

        vec  = np.concatenate([chroma_mean, mfcc_mean])
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        windows.append(vec)

    return np.array(windows, dtype=np.float32) if windows else None


def match_score(video_path) -> float:
    """
    Returns trending_audio_score ∈ [0, 1] for the given video file.
    Falls back to 0.0 if the DB is empty or audio can't be loaded.
    """
    db = _load_db()
    if not db:
        return 0.0

    try:
        y, sr = librosa.load(str(video_path), sr=SR, mono=True, duration=60)
    except Exception:
        return 0.0

    query_fp = _extract_fingerprint(y, sr)
    if query_fp is None:
        return 0.0

    best_raw = 0.0
    for entry in db:
        ref_fp = entry["fingerprint"]          # (N_ref, 25)

        # All-pairs cosine similarity — both matrices are already L2-normalised
        # Shape: (N_query, N_ref)
        sim_matrix = query_fp @ ref_fp.T

        # Take mean of top-K window pairs — robust to a few bad windows
        k       = min(5, sim_matrix.size)
        top_k   = np.partition(sim_matrix.flatten(), -k)[-k:]
        song_sim = float(top_k.mean())

        if song_sim > best_raw:
            best_raw = song_sim

    # Linear normalisation: THRESH_LOW → 0.0, THRESH_HIGH → 1.0
    score = (best_raw - THRESH_LOW) / (THRESH_HIGH - THRESH_LOW)
    return float(np.clip(score, 0.0, 1.0))


def db_size() -> int:
    """Returns number of entries in the trending audio DB."""
    return len(_load_db())
