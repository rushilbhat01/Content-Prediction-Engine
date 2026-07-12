"""
extract_hook_features.py
Extracts high-signal features from the first 3 seconds of a Short:
  - Audio:  speech vs music vs silence classification (librosa, no new deps)
  - Speech: Whisper hook transcript analysis
  - Visual: OCR text overlay detection (easyocr, optional)

Can be imported by predict.py (single video) or run_hook_extraction.py (batch).
"""

import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)

import cv2
import numpy as np
import librosa
from pathlib import Path

# Suppress librosa/audioread noise
import logging as _logging
_logging.getLogger("librosa").setLevel(_logging.ERROR)
_logging.getLogger("audioread").setLevel(_logging.ERROR)

# ── EasyOCR (optional, graceful fallback) ─────────────────────────────────────
try:
    import easyocr
    _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    HAS_OCR = True
except Exception:
    HAS_OCR = False

# ── Whisper (optional, graceful fallback) ──────────────────────────────────────
try:
    import whisper as _whisper
    _whisper_model = None  # lazy-loaded on first use to avoid slow imports in workers
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

HOOK_DURATION = 3.0   # seconds to analyse for the hook


# ─────────────────────────────────────────────────────────────────────────────
# TEXT OVERLAY FEATURES  (OCR on frames at 0.3s, 1.0s, 2.0s)
# ─────────────────────────────────────────────────────────────────────────────
def _preprocess_for_ocr(frame):
    """High-contrast greyscale — makes text pop for OCR."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def extract_text_overlay_features(video_path: str) -> dict:
    """
    Returns:
        hook_has_text_overlay   (0/1)
        hook_text_char_count    (int)
        hook_text_has_question  (0/1)
        hook_text_has_number    (0/1)
        hook_text_has_exclaim   (0/1)
    """
    defaults = {
        "hook_has_text_overlay":  0,
        "hook_text_char_count":   0,
        "hook_text_has_question": 0,
        "hook_text_has_number":   0,
        "hook_text_has_exclaim":  0,
    }

    if not HAS_OCR:
        return defaults

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return defaults

    sample_times = [0.3, 1.0, 2.0]
    all_text = ""

    for t in sample_times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            # Resize to speed up OCR — 540px wide is enough
            h, w = frame.shape[:2]
            scale = min(1.0, 540 / max(w, 1))
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            results = _ocr_reader.readtext(small, detail=0, paragraph=True)
            all_text += " ".join(results)
        except Exception:
            pass

    cap.release()

    all_text = all_text.strip()
    return {
        "hook_has_text_overlay":  int(len(all_text) > 2),
        "hook_text_char_count":   len(all_text),
        "hook_text_has_question": int("?" in all_text),
        "hook_text_has_number":   int(any(c.isdigit() for c in all_text)),
        "hook_text_has_exclaim":  int("!" in all_text),
    }


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO HOOK FEATURES  (librosa — no new deps)
# ─────────────────────────────────────────────────────────────────────────────
_SPEECH_ZCR_LO    = 0.05   # voice has moderate ZCR
_SPEECH_ZCR_HI    = 0.25
_MUSIC_FLAT_LO    = 0.05   # music = low spectral flatness (tonal)
_MUSIC_FLAT_HI    = 0.25
_SILENCE_RMS_THR  = 0.01


def _classify_audio_hook(y_hook, sr):
    """
    Classify the hook audio segment into:
      has_speech  — voice-like ZCR + mid spectral centroid
      has_music   — tonal / harmonic content (low spectral flatness)
      is_silent   — near-zero energy

    Returns (has_speech, has_music, silence_ratio)
    """
    if len(y_hook) == 0:
        return 0, 0, 1.0

    rms      = librosa.feature.rms(y=y_hook)[0]
    flatness = librosa.feature.spectral_flatness(y=y_hook)[0]
    zcr      = librosa.feature.zero_crossing_rate(y_hook)[0]

    silence_ratio = float((rms < _SILENCE_RMS_THR).sum() / max(len(rms), 1))

    # Music: low spectral flatness (tonal), non-silent
    mean_flatness = float(flatness.mean())
    has_music = int(mean_flatness < _MUSIC_FLAT_HI and silence_ratio < 0.7)

    # Speech heuristic: moderate-to-high ZCR, not purely tonal
    mean_zcr  = float(zcr.mean())
    has_speech = int(
        _SPEECH_ZCR_LO < mean_zcr < _SPEECH_ZCR_HI
        and mean_flatness > _MUSIC_FLAT_LO   # not purely tonal = not pure music
        and silence_ratio < 0.7
    )

    return has_speech, has_music, silence_ratio


def extract_audio_hook_features(video_path: str) -> dict:
    """
    Returns:
        hook_has_speech          (0/1) — voice detected in first 3s
        hook_has_bg_music        (0/1) — music detected in first 3s
        hook_speech_and_music    (0/1) — both present (ideal state)
        hook_silence_ratio_3s    (float) — fraction that is silent
        hook_audio_energy_3s     (float) — mean RMS in first 3s
        hook_music_dominance     (float) — spectral flatness inverse (tonal-ness)
    """
    defaults = {
        "hook_has_speech":       0,
        "hook_has_bg_music":     0,
        "hook_speech_and_music": 0,
        "hook_silence_ratio_3s": 1.0,
        "hook_audio_energy_3s":  0.0,
        "hook_music_dominance":  0.0,
    }

    try:
        y, sr = librosa.load(str(video_path), sr=16000, mono=True,
                             offset=0.0, duration=HOOK_DURATION)
        if len(y) == 0:
            return defaults
    except Exception:
        return defaults

    has_speech, has_music, silence_ratio = _classify_audio_hook(y, sr)

    rms = librosa.feature.rms(y=y)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]

    return {
        "hook_has_speech":       has_speech,
        "hook_has_bg_music":     has_music,
        "hook_speech_and_music": int(has_speech and has_music),
        "hook_silence_ratio_3s": round(silence_ratio, 4),
        "hook_audio_energy_3s":  round(float(rms.mean()), 6),
        "hook_music_dominance":  round(float(1.0 - flatness.mean()), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# WHISPER HOOK SPEECH FEATURES  (what is actually said in first 3s)
# ─────────────────────────────────────────────────────────────────────────────
_CURIOSITY_WORDS = {
    "secret", "trick", "mistake", "wrong", "never", "always", "why",
    "stop", "don't", "hack", "truth", "real", "actually", "honest",
    "shocking", "surprisingly", "reveal", "exposed", "myth", "lie",
}


def extract_whisper_hook_features(video_path: str,
                                  whisper_model=None) -> dict:
    """
    Transcribes first 3 seconds via Whisper and extracts semantic hook features.
    Passes the video file directly to Whisper (no soundfile dependency needed).
    """
    defaults = {
        "hook_speech_word_count":  0,
        "hook_speech_words_per_s": 0.0,
        "hook_speech_is_question": 0,
        "hook_speech_has_number":  0,
        "hook_speech_has_exclaim": 0,
        "hook_speech_curiosity":   0.0,
        "hook_speech_starts_fast": 0,
    }

    model = whisper_model
    if model is None:
        if not HAS_WHISPER:
            return defaults
        global _whisper_model
        if _whisper_model is None:
            import whisper as _w
            _whisper_model = _w.load_model("tiny", device=os.getenv("YT_SHORTS_DEVICE", "cuda"))
        model = _whisper_model

    try:
        # Load ONLY first 3 seconds, save as tmp WAV via scipy (no soundfile needed,
        # and 10-20x faster than passing the full video to Whisper)
        import tempfile, os as _os
        import numpy as _np
        from scipy.io import wavfile as _wf

        y, _sr = librosa.load(str(video_path), sr=16000, mono=True,
                              offset=0.0, duration=HOOK_DURATION)
        if len(y) == 0:
            return defaults

        _tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        _wf.write(_tmp.name, _sr, (y * 32767).astype(_np.int16))
        _tmp.close()

        result = model.transcribe(
            _tmp.name,
            fp16=False,
            language="en",
            condition_on_previous_text=False,
            verbose=False,
        )
        _os.unlink(_tmp.name)
    except Exception:
        return defaults


    segments  = result.get("segments", [])
    hook_segs = [s for s in segments if s.get("start", 99) < HOOK_DURATION]
    hook_text = " ".join(s.get("text", "") for s in hook_segs).strip().lower()

    if not hook_text:
        return defaults

    words      = hook_text.split()
    word_count = len(words)
    curiosity  = sum(1 for w in words if w.strip(".,!?") in _CURIOSITY_WORDS)
    first_start = hook_segs[0].get("start", HOOK_DURATION) if hook_segs else HOOK_DURATION

    return {
        "hook_speech_word_count":  word_count,
        "hook_speech_words_per_s": round(word_count / HOOK_DURATION, 3),
        "hook_speech_is_question": int("?" in hook_text),
        "hook_speech_has_number":  int(any(c.isdigit() for c in hook_text)),
        "hook_speech_has_exclaim": int("!" in hook_text),
        "hook_speech_curiosity":   round(curiosity / max(word_count, 1), 4),
        "hook_speech_starts_fast": int(first_start < 1.5),
    }


# ─────────────────────────────────────────────────────────────────────────────
# COMBINED EXTRACTOR  (one call for everything)
# ─────────────────────────────────────────────────────────────────────────────
def extract_all_hook_features(video_path: str,
                              whisper_model=None,
                              run_ocr: bool = True,
                              run_whisper: bool = True) -> dict:
    """Extract all hook features and return as a flat dict."""
    feats = {}

    if run_ocr:
        feats.update(extract_text_overlay_features(video_path))

    feats.update(extract_audio_hook_features(video_path))

    if run_whisper:
        feats.update(extract_whisper_hook_features(video_path, whisper_model))

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    if len(sys.argv) < 2:
        print("Usage: python extract_hook_features.py path/to/video.mp4")
        sys.exit(1)

    path = sys.argv[1]
    print(f"Extracting hook features from: {path}")
    result = extract_all_hook_features(path)
    print(json.dumps(result, indent=2))
