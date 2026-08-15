"""
_predict_infer.py — Inference-only subprocess called by predict.py.
Runs with no GPU models loaded, so no memory conflict.
"""
import argparse, json, warnings, os
warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd
import joblib
import shap
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--features',   required=True)
parser.add_argument('--niche',      default='fitness')
parser.add_argument('--title',      default='')
parser.add_argument('--hook-frame', default=None)
args  = parser.parse_args()
NICHE = args.niche

with open(args.features) as f:
    feature_dict = json.load(f)

# ── Load models ───────────────────────────────────────────
xgb_m   = joblib.load(f"models/xgb_{NICHE}.pkl")
rf_m    = joblib.load(f"models/rf_{NICHE}.pkl")
ridge_m = joblib.load(f"models/ridge_{NICHE}.pkl")
lgb_m   = joblib.load(f"models/lgb_{NICHE}.pkl")
w1, w2, w3, w4 = joblib.load(f"models/weights_{NICHE}.pkl")
imp_cont, imp_bin, cont_cols, bin_cols = joblib.load(f"models/imputer_{NICHE}.pkl")
feats          = joblib.load(f"models/features_{NICHE}.pkl")
stats          = joblib.load(f"models/niche_stats_{NICHE}.pkl")
score_dist     = joblib.load(f"models/score_dist_{NICHE}.pkl")
top_benchmarks = joblib.load(f"models/top_benchmarks_{NICHE}.pkl")
exp            = shap.TreeExplainer(xgb_m)

# ── Apply PCA transforms ──────────────────────────────────
for pca_file, cols_file, out_prefix in [
    (f"models/pca_clip_mean_{NICHE}.pkl",      f"models/clip_mean_cols_{NICHE}.pkl",      "clip_mean_pca_"),
    (f"models/pca_clip_hook_{NICHE}.pkl",      f"models/clip_hook_cols_{NICHE}.pkl",      "clip_hook_pca_"),
    (f"models/pca_vggish_mean_{NICHE}.pkl",    f"models/vggish_mean_cols_{NICHE}.pkl",    "vggish_mean_pca_"),
    (f"models/pca_vggish_hook_{NICHE}.pkl",    f"models/vggish_hook_cols_{NICHE}.pkl",    "vggish_hook_pca_"),
    (f"models/pca_title_{NICHE}.pkl",          f"models/title_cols_{NICHE}.pkl",          "title_pca_"),
    (f"models/pca_transcript_{NICHE}.pkl",     f"models/transcript_cols_{NICHE}.pkl",     "transcript_pca_"),
]:
    if not Path(pca_file).exists():
        continue
    pca      = joblib.load(pca_file)
    raw_cols = joblib.load(cols_file)
    vals     = np.array([[feature_dict.get(c, 0.0) for c in raw_cols]])
    reduced  = pca.transform(vals)[0]
    for i, v in enumerate(reduced):
        feature_dict[f"{out_prefix}{i}"] = float(v)

# ── Assign cluster ────────────────────────────────────────
km_path = Path(f"models/kmeans_{NICHE}.pkl")
if km_path.exists() and "cluster_id" in feats:
    km      = joblib.load(km_path)
    cc      = joblib.load(f"models/cluster_cols_{NICHE}.pkl")
    cc_vals = np.array([[feature_dict.get(c, 0.0) for c in cc]])
    feature_dict["cluster_id"] = int(km.predict(cc_vals)[0])

# ── Impute ────────────────────────────────────────────────
sample_df = pd.DataFrame([[feature_dict.get(f, stats.get(f, 0)) for f in feats]], columns=feats)

cont_present = [f for f in cont_cols if f in sample_df.columns]
bin_present  = [f for f in bin_cols  if f in sample_df.columns]
if cont_present:
    imputed = imp_cont.transform(sample_df[cont_present])
    for j, col in enumerate(cont_present):
        sample_df[col] = imputed[0][j]
if bin_present:
    imputed = imp_bin.transform(sample_df[bin_present])
    for j, col in enumerate(bin_present):
        sample_df[col] = imputed[0][j]

sample = sample_df[feats].values

# ── Predict ───────────────────────────────────────────────
xgb_score   = float(xgb_m.predict(sample)[0])
rf_score    = float(rf_m.predict(sample)[0])
ridge_score = float(ridge_m.predict(sample)[0])
lgb_score   = float(lgb_m.predict(sample)[0])
score       = w1*xgb_score + w2*rf_score + w3*ridge_score + w4*lgb_score

# Percentile rank — z-score mapped through normal CDF so scores spread 5–95
# rather than clustering at the high end (training data is biased toward viral videos)
from scipy.stats import norm as _norm
_score_mean = float(np.mean(score_dist))
_score_std  = float(np.std(score_dist) + 1e-9)
_z          = (score - _score_mean) / _score_std
percentile  = int(np.clip(_norm.cdf(_z) * 100, 5, 95))

# ── SHAP recommendations ──────────────────────────────────
# How to format benchmark comparison for each feature
# format_fn takes (your_val, top_val) and returns a human-readable string
BENCHMARK_FORMAT = {
    "hook_brightness":        lambda y, t: f"Top creators: {t:.0%} brightness. Yours: {y:.0%}.",
    "hook_contrast":          lambda y, t: f"Top creators have {t:.1f}x contrast. Yours: {y:.1f}x.",
    "hook_saturation":        lambda y, t: f"Top creators: {t:.0%} colour saturation. Yours: {y:.0%}.",
    "hook_centre_brightness": lambda y, t: f"Top creators: {t:.0%} centre brightness. Yours: {y:.0%}.",
    "brightness_trend":       lambda y, t: f"Top creators' videos get {'brighter' if t > 0 else 'darker'} over time. Yours gets {'brighter' if y > 0 else 'darker'}.",
    "hook_has_face":          lambda y, t: f"{int(t*100)}% of top videos show a face in the first 3 seconds.",
    "hook_face_size":         lambda y, t: f"Top creators fill {t:.0%} of the frame with their face. Yours: {y:.0%}.",
    "mean_has_face":          lambda y, t: f"Top creators are on camera {t:.0%} of the time. Yours: {y:.0%}.",
    "df_emotion_score":       lambda y, t: f"Top creators show more positive energy in their hook.",
    "df_happy_score":         lambda y, t: f"Top creators show happiness/energy in {t:.0%} of hook frames. Yours: {y:.0%}.",
    "mean_loudness":          lambda y, t: f"Top videos are {t/max(y,1e-6):.1f}x louder on average.",
    "hook_loudness":          lambda y, t: f"Top videos hit full volume in the hook. Yours is {y/max(t,1e-6):.0%} of that.",
    "silence_ratio":          lambda y, t: f"Top creators have {t:.0%} silence. Yours: {y:.0%}.",
    "dynamic_range":          lambda y, t: f"Top videos have {t:.2f} audio dynamic range. Yours: {y:.2f}.",
    "music_score":            lambda y, t: f"{int(t*100)}% of top videos have background music throughout.",
    "hook_music_score":       lambda y, t: f"Top creators start with music immediately. Yours starts quieter.",
    "tempo_bpm":              lambda y, t: f"Top videos in this niche average {t:.0f} BPM. Yours: {y:.0f} BPM.",
    "harmonic_ratio":         lambda y, t: f"Top videos use more melodic music. Yours is more percussive.",
    "mean_motion":            lambda y, t: f"Top videos have {t:.2f} motion score. Yours: {y:.2f}.",
    "temporal_hook_motion":   lambda y, t: f"Top videos start with {t:.2f} motion in the hook. Yours: {y:.2f}.",
    "temporal_buildup":       lambda y, t: f"Top creators build energy over time. Yours fades.",
    "hook_text_proxy":        lambda y, t: f"{int(t*100)}% of top videos show text in the first 2 seconds.",
    "tag_count":              lambda y, t: f"Top videos use {t:.0f} hashtags on average. Yours: {y:.0f}.",
    "title_has_number":       lambda y, t: f"{int(t*100)}% of top videos include a number in the title.",
    "desc_length":            lambda y, t: f"Top videos have {t:.0f} character descriptions. Yours: {y:.0f}.",
    "posted_prime_time":      lambda y, t: f"{int(t*100)}% of top videos are posted between 5pm–10pm.",
    "mean_edge_density":      lambda y, t: f"Top videos have more visual complexity. Yours looks simpler.",
}

RECOMMENDATION_TEXT = {
    "hook_brightness":        {"low": "Your first 3 seconds are too dark. Bright hooks retain 2x more viewers.", "high": "Hook is overexposed — too bright. Dial back your exposure so the image doesn't look blown out.", "category": "lighting"},
    "hook_contrast":          {"low": "Hook has low contrast. Make visuals pop — viewers decide in 1 second.", "high": "Hook contrast is good.", "category": "lighting"},
    "hook_saturation":        {"low": "Hook looks washed out. More vibrant colours stop the scroll.", "high": "Hook colour vivid — good.", "category": "lighting"},
    "hook_centre_brightness": {"low": "Subject underlit. Centre of frame should be brighter than background.", "high": "Subject lighting good.", "category": "lighting"},
    "brightness_trend":       {"low": "Video gets darker over time. Maintain or build brightness throughout.", "high": "Good brightness arc.", "category": "lighting"},
    "hook_has_face":          {"low": "No face in hook. Videos with a human face in the first frame retain more viewers.", "high": "Face in hook — good.", "category": "face"},
    "hook_face_size":         {"low": "Face too small in hook. Get closer to camera — fill 30%+ of the frame.", "high": "Good face framing.", "category": "face"},
    "mean_has_face":          {"low": "Stay on camera more consistently throughout the video.", "high": "Good face presence.", "category": "face"},
    "df_emotion_score":       {"low": "Expression looks flat or negative. A confident or happy expression retains more viewers.", "high": "Good expression in hook.", "category": "face"},
    "df_happy_score":         {"low": "Show more energy or a smile in the first frame — it sets the tone.", "high": "Good hook energy.", "category": "face"},
    "mean_loudness":          {"low": "Audio too quiet. Low volume = low energy = people scroll away.", "high": "Audio loudness good.", "category": "audio"},
    "hook_loudness":          {"low": "Hook audio too quiet. Hit viewers with full energy from second one.", "high": "Hook audio energy good.", "category": "audio"},
    "silence_ratio":          {"high": "Too much silence. Dead air kills retention. Cut the pauses.", "low": "Good — no excessive silence.", "category": "audio"},
    "dynamic_range":          {"low": "Audio too flat and uniform. Add peaks and drops to keep it engaging.", "high": "Good audio dynamics.", "category": "audio"},
    "music_score":            {"low": "No background music detected. Music increases saves by ~35% in this niche.", "high": "Background music — good.", "category": "music"},
    "hook_music_score":       {"low": "No music in the first 3 seconds. Start with music immediately.", "high": "Music in hook — good.", "category": "music"},
    "tempo_bpm":              {"low": "Music tempo slow for this niche. Higher BPM = more energy.", "high": "Good music tempo.", "category": "music"},
    "harmonic_ratio":         {"low": "Audio lacks melodic quality. More harmonic music increases saves.", "high": "Good harmonic audio.", "category": "music"},
    "mean_motion":            {"low": "Video too static. Add movement — camera, subject, or faster cuts.", "high": "Good motion level.", "category": "motion"},
    "temporal_hook_motion":   {"low": "Hook is too static. Start with movement — it signals energy immediately.", "high": "Good hook motion.", "category": "motion"},
    "temporal_buildup":       {"low": "Energy drops over time. Build towards a climax — don't start high and fade.", "high": "Good energy buildup.", "category": "motion"},
    "hook_text_proxy":        {"low": "No text in hook. 70% of reels watched silently — text in first 2s grabs them.", "high": "Text overlay in hook — good.", "category": "text"},
    "tag_count":              {"low": "Too few hashtags. Add 3-5 relevant ones for discoverability.", "high": "Good hashtag count.", "category": "text"},
    "title_has_number":       {"low": "Add a number to your title — '5 ways' and '30 day' perform better.", "high": "Good — number in title.", "category": "text"},
    "desc_length":            {"low": "Description too short. Longer descriptions help search discoverability.", "high": "Good description length.", "category": "text"},
    "posted_prime_time":      {"low": "Posted outside prime time. Post between 5pm-10pm for best reach.", "high": "Good posting time.", "category": "timing"},
    "mean_edge_density":      {"low": "Video looks visually simple. More visual complexity keeps eyes engaged.", "high": "Good visual complexity.", "category": "visual"},
}

s_vals = exp.shap_values(pd.DataFrame(sample, columns=feats))[0]
pairs  = sorted(zip(feats, s_vals), key=lambda x: x[1])

recs = []
seen_categories = set()
for feat, shap_val in pairs:
    if shap_val >= 0:
        break
    if feat not in RECOMMENDATION_TEXT:
        continue
    rec_info  = RECOMMENDATION_TEXT[feat]
    category  = rec_info.get("category", feat)
    val       = feature_dict.get(feat, stats.get(feat, 0))
    median    = stats.get(feat, 0)
    direction = "low" if val < median else "high"
    message   = rec_info[direction]
    _POSITIVE = {"good", "great", "solid", "detected", "present", "vibrant", "dynamic", "engaging"}
    if any(w in message.lower() for w in _POSITIVE):
        continue
    fixed    = sample.copy()
    feat_idx = list(feats).index(feat)
    fixed[0][feat_idx] = median
    fixed_df = pd.DataFrame(fixed, columns=feats)
    fixed_score = (
        w1 * float(xgb_m.predict(fixed_df)[0]) +
        w2 * float(rf_m.predict(fixed_df)[0])  +
        w3 * float(ridge_m.predict(fixed_df)[0]) +
        w4 * float(lgb_m.predict(fixed_df)[0])
    )
    impact = fixed_score - score
    if impact <= 0:
        continue
    # Only keep the highest-impact recommendation per category
    if category in seen_categories:
        continue
    seen_categories.add(category)
    # Add benchmark context if available
    benchmark_text = ""
    if feat in BENCHMARK_FORMAT and feat in top_benchmarks:
        try:
            benchmark_text = BENCHMARK_FORMAT[feat](val, top_benchmarks[feat])
        except Exception:
            pass
    recs.append({"feature": feat, "message": message, "benchmark": benchmark_text, "impact": round(impact, 3)})

recs = sorted(recs, key=lambda x: x["impact"], reverse=True)
top_recs = recs[:5]

# ── Checks: every extracted feature vs threshold ──────────
# Each entry: (label, ok, detail)
# Thresholds based on niche medians from stats dict where possible,
# otherwise empirical constants.
def _med(feat, fallback):
    return stats.get(feat, fallback)

ALL_CHECKS = [
    # ── Visual / hook ─────────────────────────────────────
    {"label": "Hook brightness",         "ok": feature_dict.get("hook_brightness", 0) > 0.35,                                           "category": "visual"},
    {"label": "Face in hook",            "ok": feature_dict.get("hook_has_face", 0) > 0.2,                                              "category": "visual"},
    {"label": "Text overlay in hook",    "ok": feature_dict.get("hook_text_proxy", 0) > 0.02,                                           "category": "visual"},
    {"label": "Hook contrast",           "ok": feature_dict.get("hook_contrast", 0) > _med("hook_contrast", 0.4),                       "category": "visual"},
    {"label": "Hook colour",             "ok": feature_dict.get("hook_saturation", 0) > _med("hook_saturation", 0.25),                  "category": "visual"},
    {"label": "Hook motion",             "ok": feature_dict.get("temporal_hook_motion", 0) > _med("temporal_hook_motion", 0.4),         "category": "visual"},
    {"label": "Face expression",         "ok": feature_dict.get("df_happy_score", 0) > 0.15 or feature_dict.get("hook_has_face", 0) < 0.5, "category": "visual"},
    # ── Visual / overall ──────────────────────────────────
    {"label": "Consistent face presence","ok": feature_dict.get("mean_has_face", 0) > _med("mean_has_face", 0.25),                      "category": "visual"},
    {"label": "Video motion",            "ok": feature_dict.get("mean_motion", 0) > _med("mean_motion", 0.4),                           "category": "visual"},
    {"label": "Brightness arc",          "ok": feature_dict.get("brightness_trend", 0) >= -0.05,                                        "category": "visual"},
    # ── Audio ─────────────────────────────────────────────
    {"label": "Audio volume",            "ok": feature_dict.get("mean_loudness", 0) > _med("mean_loudness", 0.03),                      "category": "audio"},
    {"label": "Hook audio energy",       "ok": feature_dict.get("hook_loudness", 0) > _med("hook_loudness", 0.03),                      "category": "audio"},
    {"label": "Silence",                 "ok": feature_dict.get("silence_ratio", 1) < _med("silence_ratio", 0.35),                      "category": "audio"},
    # music_score is a continuous energy value — median fallback 0.3 (not 1.0)
    # Use harmonic_ratio as primary music signal — more reliable than spectral flatness
    # when voice overlaps background music
    {"label": "Background music",        "ok": feature_dict.get("harmonic_ratio", 0) > _med("harmonic_ratio", 0.25) or feature_dict.get("music_score", 0) > _med("music_score", 0.3),  "category": "audio"},
    {"label": "Music in hook",           "ok": feature_dict.get("hook_music_score", 0) > _med("hook_music_score", 0.25) or feature_dict.get("harmonic_ratio", 0) > _med("harmonic_ratio", 0.25), "category": "audio"},
    {"label": "Music tempo",             "ok": feature_dict.get("tempo_bpm", 0) > _med("tempo_bpm", 90),                                "category": "audio"},
    {"label": "Audio dynamics",          "ok": feature_dict.get("dynamic_range", 0) > _med("dynamic_range", 0.08),                      "category": "audio"},
    # ── Pacing / editing ──────────────────────────────────
    {"label": "Cut speed",               "ok": feature_dict.get("cuts_per_second", 0) >= _med("cuts_per_second", 0.3),                   "category": "editing"},
    {"label": "Hook sharpness",          "ok": feature_dict.get("hook_sharpness", 0) >= _med("hook_sharpness", 80),                      "category": "visual"},
    {"label": "Video duration",          "ok": 10 <= feature_dict.get("duration_seconds", 30) <= 55,                                     "category": "metadata"},
    # ── Metadata (only truly actionable ones) ─────────────
    {"label": "Hashtags",                "ok": feature_dict.get("tag_count", 0) >= 3,                                                   "category": "metadata"},
    {"label": "Title length",            "ok": 20 <= feature_dict.get("title_length", 0) <= 80,                                          "category": "metadata"},
]

# Only failing checks
hook_checks = [c for c in ALL_CHECKS if not c["ok"]]

# ── Rec text for each check label ─────────────────────────
CHECK_REC_TEXT = {
    "Hook brightness":         "Your hook is too dark. Bright hooks stop the scroll — try filming in better light or boost exposure in editing.",
    "Face in hook":            "No face in the first frame. Showing your face immediately builds trust and keeps viewers from swiping.",
    "Text overlay in hook":    "No text in the first 2 seconds. Most people watch silently — a text hook grabs them before they swipe.",
    "Hook contrast":           "Low contrast in your hook — the subject blends in. Increase contrast so you pop against the background.",
    "Hook colour":             "Colours look washed out in the hook. Boost saturation slightly in editing — vibrant visuals stop the scroll.",
    "Hook motion":             "Your hook is too static. Start with movement — a camera pan, jump cut, or action in frame signals energy immediately.",
    "Face expression":         "Expression looks flat in the hook. A smile or high-energy look in the first frame sets the tone and pulls viewers in.",
    "Consistent face presence":"You disappear from frame too often. Stay on camera consistently — it builds connection and keeps viewers watching.",
    "Video motion":            "The video feels too static. Add camera movement, faster cuts, or more dynamic action to keep eyes engaged.",
    "Brightness arc":          "Your video gets darker as it goes. Maintain consistent brightness — a visual energy drop makes people scroll away.",
    "Audio volume":            "Audio is too quiet overall. Low volume = low energy — bring up your levels so it hits hard from the start.",
    "Hook audio energy":       "Hook audio is too quiet. Hit viewers with full energy from second one — don't ease in.",
    "Silence":                 "Too much dead air. Cut the pauses — silence kills retention on short-form content.",
    "Background music":        "No background music detected. Music increases saves significantly in this niche — add a track.",
    "Music in hook":           "No music in the first 3 seconds. Start your track immediately — it signals production quality right away.",
    "Music tempo":             "Music tempo is too slow for this niche. A higher BPM track creates more energy and urgency.",
    "Audio dynamics":          "Audio sounds flat and uniform throughout. Add peaks and drops — dynamic audio keeps people engaged.",
    "Hashtags":                "Too few hashtags. Add 3–5 relevant ones to improve discoverability in search and recommendations.",
    "Cut speed":               f"Editing is too slow — only {feature_dict.get('cuts_per_second', 0):.1f} cuts/sec. Top fitness Shorts cut every 1-2 seconds to maintain pace.",
    "Hook sharpness":          "First frame is out of focus. Viewers decide in 0.5 seconds — a blurry hook looks amateur and kills retention.",
    "Video duration":          f"Duration is {feature_dict.get('duration_seconds', 0):.0f}s. {'Too short — aim for 15–55 seconds for the algorithm to push it.' if feature_dict.get('duration_seconds', 30) < 10 else 'Too long — trim it down to under 55 seconds for better retention.'}",
    "Title length":            f"Title is {'too short' if feature_dict.get('title_length', 0) < 20 else 'too long'} ({feature_dict.get('title_length', 0)} chars). Aim for 20–80 characters — enough to say something meaningful without getting cut off.",
}

# Build unified issue list: check-based issues first, then SHAP-only ones not already covered
check_issues = [
    {"label": c["label"], "message": CHECK_REC_TEXT[c["label"]], "benchmark": "", "type": "check"}
    for c in hook_checks if c["label"] in CHECK_REC_TEXT
]
covered_categories = {c["category"] for c in hook_checks}
shap_issues = [
    {"label": r.get("feature", "").replace("_", " ").title(), "message": r["message"],
     "benchmark": r.get("benchmark", ""), "type": "model"}
    for r in top_recs
    if RECOMMENDATION_TEXT.get(r.get("feature", ""), {}).get("category") not in covered_categories
]
all_issues = check_issues + shap_issues

# ── Rewrite ALL issues through Groq ───────────────────────
final_issues = all_issues
groq_key = os.environ.get("GROQ_API_KEY")
if all_issues and groq_key:
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        numbered = "\n".join([
            f"{i}. [{r['label']}] {r['message']}" + (f" ({r['benchmark']})" if r.get('benchmark') else "")
            for i, r in enumerate(all_issues, 1)
        ])
        title_ctx = f'Video title: "{args.title}"' if args.title else ""
        # Build a compact video stats summary for Groq context
        _fmt = lambda k, fallback=0: feature_dict.get(k, fallback)
        video_ctx = (
            f"Hook brightness: {_fmt('hook_brightness'):.2f} | "
            f"Cuts/sec: {_fmt('cuts_per_second'):.2f} | "
            f"Duration: {_fmt('duration_seconds'):.0f}s | "
            f"Loudness: {_fmt('mean_loudness'):.3f} | "
            f"Harmonic ratio: {_fmt('harmonic_ratio'):.2f} | "
            f"Face in hook: {'yes' if _fmt('hook_has_face') > 0.2 else 'no'} | "
            f"Silence ratio: {_fmt('silence_ratio'):.2f}"
        )
        prompt = f"""You are a brutally honest but friendly YouTube Shorts coach advising a creator on how to improve their video before posting.

{title_ctx}
Niche: {NICHE}
Video stats: {video_ctx}

Issues found (each labelled in brackets):
{numbered}

Rewrite each as a sharp, specific, actionable tip tailored to THIS video. Rules:
- Reference the actual numbers where they exist (e.g. "your video is 9s long" not "your video is short")
- Be blunt but encouraging — like a mentor, not a robot
- No fluff, no filler words
- 1-2 sentences max per tip
- Each tip must feel different — no copy-paste structures
- Output ONLY a numbered list in the same order, keeping the label in brackets at the start
- Example format: 1. [Hook brightness] Your first few seconds are too dark — try..."""

        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            temperature=0.7,
        )
        llm_lines = [l.strip() for l in resp.choices[0].message.content.strip().split('\n') if l.strip()]
        llm_issues = []
        for line in llm_lines:
            line = re.sub(r'^\d+\.\s*', '', line)
            m = re.match(r'\[([^\]]+)\]\s*(.*)', line)
            if m and len(llm_issues) < len(all_issues):
                orig = all_issues[len(llm_issues)]
                llm_issues.append({"label": m.group(1), "message": m.group(2).strip(),
                                   "benchmark": orig.get("benchmark", ""), "type": orig["type"]})
            elif line and len(llm_issues) < len(all_issues):
                orig = all_issues[len(llm_issues)]
                llm_issues.append({"label": orig["label"], "message": line,
                                   "benchmark": orig.get("benchmark", ""), "type": orig["type"]})
        if len(llm_issues) >= len(all_issues) // 2:  # only use if Groq returned enough
            final_issues = llm_issues
    except Exception:
        pass  # silently fall back

# ── Output ────────────────────────────────────────────────
print(f"Percentile: {percentile}")
print(f"Score:      {score:.3f}")
print("HOOK_CHECKS:" + json.dumps(hook_checks))
print("RECOMMENDATIONS:" + json.dumps(final_issues))
