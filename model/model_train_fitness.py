import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, KFold
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import xgboost as xgb
import lightgbm as lgb
import shap

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Config ────────────────────────────────────────────────
NICHE      = "fitness"
TARGET     = "normalised_score"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ── Load data ─────────────────────────────────────────────
print("Loading data...")
meta = pd.read_csv("data/metadata_clean.csv")

visual_path = Path("data/features_visual.csv")
if visual_path.exists():
    visual = pd.read_csv(visual_path).drop_duplicates(subset="video_id")
    meta   = meta.merge(visual, on="video_id", how="left")
    print(f"Visual features:    {visual.shape[1]-1} cols")
else:
    print("No visual features found")

audio_path = Path("data/features_audio.csv")
if audio_path.exists():
    audio = pd.read_csv(audio_path).drop_duplicates(subset="video_id")
    meta  = meta.merge(audio, on="video_id", how="left")
    print(f"Audio features:     {audio.shape[1]-1} cols")
else:
    print("No audio features found")

emb_path       = Path("data/features_embeddings.csv")
has_embeddings = emb_path.exists()
if has_embeddings:
    emb = pd.read_csv(emb_path).drop_duplicates(subset="video_id")
    print(f"Embedding features: {emb.shape[1]-1} cols")
else:
    print("No embedding features found — run extract_embeddings.py first")

trending_path = Path("data/features_trending_audio.csv")
if trending_path.exists():
    trending_df = pd.read_csv(trending_path).drop_duplicates(subset="video_id")
    meta = meta.merge(trending_df, on="video_id", how="left")
    print(f"Trending audio scores: {len(trending_df)} entries")
else:
    print("No trending audio scores -- run build_trending_audio_db.py then extract_trending_audio_scores.py")

hook_path = Path("data/hook_features.csv")
if hook_path.exists():
    hook_df = pd.read_csv(hook_path).drop_duplicates(subset="video_id")
    meta = meta.merge(hook_df, on="video_id", how="left")
    print(f"Hook features:        {hook_df.shape[1]-1} cols ({len(hook_df)} videos)")
else:
    print("No hook features yet  -- run: python run_hook_extraction.py")

# ── Filter to niche ───────────────────────────────────────
df = meta[meta["niche"] == NICHE].copy()
print(f"\nNiche: {NICHE} — {len(df)} videos before content filter")

# ── Remove off-topic videos via title keyword filter ──────
# Require at least one niche keyword in the title/tags.
# Catches scraped videos that landed in the wrong niche bucket.
NICHE_KEYWORDS = {
    "fitness":   ["workout", "gym", "fitness", "exercise", "training", "muscle", "weight",
                  "squat", "deadlift", "bench", "cardio", "hiit", "strength", "lift",
                  "bodybuilding", "crossfit", "pilates", "yoga", "stretch", "abs", "glute",
                  "chest", "bicep", "tricep", "shoulder", "leg", "pull", "push", "run",
                  "protein", "calories", "fat", "body", "transformation", "calisthenics"],
    "food":      ["food", "recipe", "cook", "eat", "meal", "diet", "nutrition", "kitchen",
                  "dish", "ingredient", "bake", "grill", "fry", "breakfast", "lunch", "dinner"],
    "finance":   ["money", "finance", "invest", "stock", "crypto", "budget", "income",
                  "wealth", "saving", "trading", "bank", "loan", "debt", "profit"],
    "motivation":["motivat", "mindset", "success", "grind", "hustle", "inspire", "goal",
                  "discipline", "focus", "achieve", "dream", "positive"],
    "education": ["learn", "study", "school", "teach", "education", "science", "history",
                  "math", "explain", "fact", "how to", "why", "what is"],
    "beauty":    ["makeup", "beauty", "skin", "hair", "skincare", "cosmetic", "lipstick",
                  "foundation", "blush", "eyeshadow", "moistur", "serum", "glow"],
}

if NICHE in NICHE_KEYWORDS:
    keywords = NICHE_KEYWORDS[NICHE]
    title_col = df["title"].fillna("").str.lower()
    mask = title_col.apply(lambda t: any(kw in t for kw in keywords))
    removed = (~mask).sum()
    df = df[mask].copy()
    print(f"Removed {removed} off-topic videos — {len(df)} remaining")

# ── Cap videos per channel ────────────────────────────────
# Prevents dominant channels from overfitting model to their style.
CHANNEL_CAP = 10
before = len(df)
df = df.sample(frac=1, random_state=42).groupby("channel_id").head(CHANNEL_CAP).reset_index(drop=True)
print(f"Channel cap ({CHANNEL_CAP}/channel): {before} -> {len(df)} videos")

if len(df) < 50:
    print("ERROR: Need at least 50 videos.")
    exit()

# ── Recompute targets ─────────────────────────────────────
df["views"]             = pd.to_numeric(df["views"],             errors='coerce')
df["likes"]             = pd.to_numeric(df.get("likes",  0),    errors='coerce').fillna(0).clip(lower=0)
df["comments"]          = pd.to_numeric(df.get("comments", 0), errors='coerce').fillna(0).clip(lower=0)
df["subscriber_count"]  = pd.to_numeric(df["subscriber_count"],  errors='coerce').clip(lower=1)
df["video_age_days"]    = pd.to_numeric(df["video_age_days"],    errors='coerce').clip(lower=1)
df["channel_avg_views"] = pd.to_numeric(df["channel_avg_views"], errors='coerce').clip(lower=1)

# ── PCA composite virality target ────────────────────────
# Three engagement signals, each measured relative to the channel's own baseline.
# PCA finds the data-driven linear combo that captures the most variance across
# all three — no hand-picked weights.
from sklearn.preprocessing import StandardScaler

df["like_view_ratio"]    = df["likes"]    / df["views"].clip(lower=1)
df["comment_view_ratio"] = df["comments"] / df["views"].clip(lower=1)

# Channel-level baselines (median per channel, Winsorised at 5/95 pct)
chan_like_avg    = df.groupby("channel_id")["like_view_ratio"].transform("median").clip(
    df["like_view_ratio"].quantile(0.05), df["like_view_ratio"].quantile(0.95)
).clip(lower=1e-6)
chan_comment_avg = df.groupby("channel_id")["comment_view_ratio"].transform("median").clip(
    df["comment_view_ratio"].quantile(0.05), df["comment_view_ratio"].quantile(0.95)
).clip(lower=1e-6)

views_beat   = np.log1p(df["views"])             - np.log1p(df["channel_avg_views"])
likes_beat   = np.log1p(df["like_view_ratio"])   - np.log1p(chan_like_avg)
comments_beat = np.log1p(df["comment_view_ratio"]) - np.log1p(chan_comment_avg)

signals = pd.DataFrame({
    "views_beat":    views_beat,
    "likes_beat":    likes_beat,
    "comments_beat": comments_beat,
}).dropna()

_pca = PCA(n_components=1)
_scaled = StandardScaler().fit_transform(signals)
_pc1 = _pca.fit_transform(_scaled).flatten()
pc1_weights = _pca.components_[0]  # [w_views, w_likes, w_comments]

print(f"\nPCA virality target — PC1 explains {_pca.explained_variance_ratio_[0]*100:.1f}% of variance")
print(f"  weights: views={pc1_weights[0]:.3f}  likes={pc1_weights[1]:.3f}  comments={pc1_weights[2]:.3f}")

df["normalised_score"] = np.nan
df.loc[signals.index, "normalised_score"] = _pc1

# Channel context features — knowable at upload time
df["log1p_subscriber_count"]  = np.log1p(df["subscriber_count"])
df["log1p_channel_avg_views"] = np.log1p(df["channel_avg_views"])

# Channel-level engagement quality (also knowable at upload time via API)
df["channel_avg_like_rate"]    = df.groupby("channel_id")["like_view_ratio"].transform("median").fillna(0)
df["channel_avg_comment_rate"] = df.groupby("channel_id")["comment_view_ratio"].transform("median").fillna(0)
df["log1p_chan_like_rate"]      = np.log1p(df["channel_avg_like_rate"])
df["log1p_chan_comment_rate"]   = np.log1p(df["channel_avg_comment_rate"])

# ── Hand-crafted feature groups ───────────────────────────
# Post timing features (posted_hour, posted_weekday, posted_on_weekend,
# posted_prime_time) deliberately excluded — Shorts distribution is
# algorithm-driven, not recency-driven. These were spurious correlates
# of creator professionalism, not causal levers.
METADATA_FEATURES = [
    "duration_seconds",
    "title_length", "title_has_number",
    "title_has_emoji",
    "title_caps_ratio", "tag_count",
    "desc_length",
    "log1p_subscriber_count",
    "log1p_channel_avg_views",
    "log1p_chan_like_rate",      # channel engagement quality — knowable at upload
    "log1p_chan_comment_rate",
]

VISUAL_FEATURES = [
    "hook_brightness", "hook_contrast",
    "hook_saturation", "hook_sharpness",
    "hook_has_face", "hook_face_size",
    "hook_edge_density", "hook_hue_entropy",
    "hook_centre_brightness", "hook_text_proxy",
    "mean_brightness", "mean_contrast",
    "mean_saturation",
    "mean_has_face", "mean_face_size",
    "mean_edge_density", "mean_hue_entropy",
    "std_brightness", "std_contrast",
    "std_saturation", "mean_motion",
    "max_motion",
    "brightness_trend",
    "cuts_per_second",
]

AUDIO_FEATURES = [
    "mean_loudness", "hook_loudness",
    "loudness_variance", "silence_ratio",
    "music_score", "tempo_bpm", "mean_zcr",
    "audio_energy_trend", "hook_music_score",
    "harmonic_ratio", "percussive_ratio",
    "dynamic_range",
    "trending_audio_score",
]

TEMPORAL_FEATURES = [
    "temporal_motion_mean", "temporal_motion_max",
    "temporal_motion_std", "temporal_hook_motion",
    "temporal_end_motion", "temporal_buildup",
    "temporal_peak_time", "temporal_variance",
    "temporal_acceleration", "temporal_high_motion_ratio",
]

# Hook features — from extract_hook_features.py / run_hook_extraction.py
# Text overlay (OCR): what text is visible in the first 3 frames
# Audio hook type: speech / music / both / silent
# Whisper speech: what is said and how, in the first 3 seconds
HOOK_FEATURES = [
    # Text overlay
    "hook_has_text_overlay",
    "hook_text_char_count",
    "hook_text_has_question",
    "hook_text_has_number",
    "hook_text_has_exclaim",
    # Audio hook type
    "hook_has_speech",
    "hook_has_bg_music",
    "hook_speech_and_music",
    "hook_silence_ratio_3s",
    "hook_audio_energy_3s",
    "hook_music_dominance",
    # Whisper hook speech
    "hook_speech_word_count",
    "hook_speech_words_per_s",
    "hook_speech_is_question",
    "hook_speech_has_number",
    "hook_speech_has_exclaim",
    "hook_speech_curiosity",
    "hook_speech_starts_fast",
]

all_hand_crafted = METADATA_FEATURES + VISUAL_FEATURES + AUDIO_FEATURES + TEMPORAL_FEATURES + HOOK_FEATURES
hand_crafted = [f for f in all_hand_crafted if f in df.columns]
n_hook_avail = sum(1 for f in HOOK_FEATURES if f in df.columns)
print(f"\nHand-crafted features available: {len(hand_crafted)} / {len(all_hand_crafted)}")
print(f"  of which hook features: {n_hook_avail}/{len(HOOK_FEATURES)}"
      + ("" if n_hook_avail else "  <- run python run_hook_extraction.py"))
missing_hc = [f for f in all_hand_crafted if f not in df.columns]
if missing_hc:
    non_hook_missing = [f for f in missing_hc if f not in HOOK_FEATURES]
    if non_hook_missing:
        print(f"Missing non-hook features: {non_hook_missing}")


# ── Prepare X and y ───────────────────────────────────────
X_hc = df[hand_crafted + ["video_id"]].copy()

for col in X_hc.columns:
    if col == "video_id":
        continue
    if X_hc[col].dtype == bool:
        X_hc[col] = X_hc[col].astype(int)
    elif X_hc[col].dtype == object:
        X_hc[col] = X_hc[col].astype(str).str.lower().map(
            {'true': 1, 'false': 0, '1': 1, '0': 0}
        ).fillna(0)

y    = df[TARGET].copy()
mask = y.notna() & np.isfinite(y)
X_hc = X_hc[mask]
y    = y[mask]
print(f"Samples after dropping nulls: {len(X_hc)}")

# ── Impute hand-crafted (split by feature type) ───────────
# Binary features should use most_frequent, not median
# (median of a 0/1 column can be 0.7, which is meaningless)
BINARY_HC = {
    "posted_on_weekend", "posted_prime_time",
    "title_has_number", "title_has_emoji",
    "hook_has_face", "mean_has_face",
}
binary_cols     = [f for f in hand_crafted if f in BINARY_HC]
continuous_cols = [f for f in hand_crafted if f not in BINARY_HC]

X_hc_no_id = X_hc.drop("video_id", axis=1)

imputer_continuous = SimpleImputer(strategy='median')
imputer_binary     = SimpleImputer(strategy='most_frequent')

imp_cont = pd.DataFrame(
    imputer_continuous.fit_transform(X_hc_no_id[continuous_cols]),
    columns=continuous_cols,
    index=X_hc.index
)
imp_bin = pd.DataFrame(
    imputer_binary.fit_transform(X_hc_no_id[binary_cols]),
    columns=binary_cols,
    index=X_hc.index
)
X_hc_imp = pd.concat([imp_cont, imp_bin], axis=1)[hand_crafted]

# ── Process embeddings with PCA ───────────────────────────
X_combined = X_hc_imp.copy()

if has_embeddings:
    emb_merged = df[["video_id"]].merge(emb, on="video_id", how="left")
    emb_merged = emb_merged[mask]

    # ── CLIP mean -> PCA 50 ────────────────────────────────
    clip_mean_cols = [c for c in emb.columns if c.startswith("clip_mean_")]
    if clip_mean_cols:
        vals           = emb_merged[clip_mean_cols].fillna(0).values
        pca            = PCA(n_components=min(50, len(clip_mean_cols)), random_state=42)
        reduced        = pca.fit_transform(vals)
        for i in range(reduced.shape[1]):
            X_combined[f"clip_mean_pca_{i}"] = reduced[:, i]
        joblib.dump(pca,            f"models/pca_clip_mean_{NICHE}.pkl")
        joblib.dump(clip_mean_cols, f"models/clip_mean_cols_{NICHE}.pkl")
        print(f"CLIP mean:    {len(clip_mean_cols)} -> {reduced.shape[1]} dims")

    # ── CLIP hook -> PCA 30 ────────────────────────────────
    clip_hook_cols = [c for c in emb.columns if c.startswith("clip_hook_")]
    if clip_hook_cols:
        vals           = emb_merged[clip_hook_cols].fillna(0).values
        pca            = PCA(n_components=min(30, len(clip_hook_cols)), random_state=42)
        reduced        = pca.fit_transform(vals)
        for i in range(reduced.shape[1]):
            X_combined[f"clip_hook_pca_{i}"] = reduced[:, i]
        joblib.dump(pca,            f"models/pca_clip_hook_{NICHE}.pkl")
        joblib.dump(clip_hook_cols, f"models/clip_hook_cols_{NICHE}.pkl")
        print(f"CLIP hook:    {len(clip_hook_cols)} -> {reduced.shape[1]} dims")

    # ── VGGish mean -> PCA 20 ─────────────────────────────
    vggish_mean_cols = [c for c in emb.columns if c.startswith("vggish_mean_")]
    if vggish_mean_cols:
        vals           = emb_merged[vggish_mean_cols].fillna(0).values
        pca            = PCA(n_components=min(20, len(vggish_mean_cols)), random_state=42)
        reduced        = pca.fit_transform(vals)
        for i in range(reduced.shape[1]):
            X_combined[f"vggish_mean_pca_{i}"] = reduced[:, i]
        joblib.dump(pca,              f"models/pca_vggish_mean_{NICHE}.pkl")
        joblib.dump(vggish_mean_cols, f"models/vggish_mean_cols_{NICHE}.pkl")
        print(f"VGGish mean:  {len(vggish_mean_cols)} -> {reduced.shape[1]} dims")

    # ── VGGish hook -> PCA 10 ─────────────────────────────
    vggish_hook_cols = [c for c in emb.columns if c.startswith("vggish_hook_")]
    if vggish_hook_cols:
        vals           = emb_merged[vggish_hook_cols].fillna(0).values
        pca            = PCA(n_components=min(10, len(vggish_hook_cols)), random_state=42)
        reduced        = pca.fit_transform(vals)
        for i in range(reduced.shape[1]):
            X_combined[f"vggish_hook_pca_{i}"] = reduced[:, i]
        joblib.dump(pca,              f"models/pca_vggish_hook_{NICHE}.pkl")
        joblib.dump(vggish_hook_cols, f"models/vggish_hook_cols_{NICHE}.pkl")
        print(f"VGGish hook:  {len(vggish_hook_cols)} -> {reduced.shape[1]} dims")

    # ── DeepFace — no PCA, already low dimensional ────────
    deepface_cols = [c for c in emb.columns if c.startswith("df_") and c != "df_face_detected"]
    if deepface_cols:
        df_vals = emb_merged[deepface_cols].fillna(0).values
        for i, col in enumerate(deepface_cols):
            X_combined[col] = df_vals[:, i]
        joblib.dump(deepface_cols, f"models/deepface_cols_{NICHE}.pkl")
        print(f"DeepFace:     {len(deepface_cols)} features (no PCA)")

    # ── Temporal motion — no PCA ──────────────────────────
    temporal_cols = [c for c in emb.columns if c.startswith("temporal_")]
    if temporal_cols:
        t_vals = emb_merged[temporal_cols].fillna(0).values
        for i, col in enumerate(temporal_cols):
            X_combined[col] = t_vals[:, i]
        print(f"Temporal:     {len(temporal_cols)} features (no PCA)")

    # ── Title MiniLM -> PCA 20 ────────────────────────────
    title_cols = [c for c in emb.columns if c.startswith("title_emb_")]
    if title_cols:
        vals           = emb_merged[title_cols].fillna(0).values
        pca            = PCA(n_components=min(20, len(title_cols)), random_state=42)
        reduced        = pca.fit_transform(vals)
        for i in range(reduced.shape[1]):
            X_combined[f"title_pca_{i}"] = reduced[:, i]
        joblib.dump(pca,        f"models/pca_title_{NICHE}.pkl")
        joblib.dump(title_cols, f"models/title_cols_{NICHE}.pkl")
        print(f"Title MiniLM: {len(title_cols)} -> {reduced.shape[1]} dims")

    # ── Transcript Whisper -> PCA 20 ──────────────────────
    transcript_cols = [c for c in emb.columns if c.startswith("transcript_emb_")]
    if transcript_cols:
        vals    = emb_merged[transcript_cols].fillna(0).values
        pca     = PCA(n_components=min(20, len(transcript_cols)), random_state=42)
        reduced = pca.fit_transform(vals)
        for i in range(reduced.shape[1]):
            X_combined[f"transcript_pca_{i}"] = reduced[:, i]
        joblib.dump(pca,             f"models/pca_transcript_{NICHE}.pkl")
        joblib.dump(transcript_cols, f"models/transcript_cols_{NICHE}.pkl")
        print(f"Transcript:   {len(transcript_cols)} -> {reduced.shape[1]} dims")

print(f"\nTotal features going into model: {X_combined.shape[1]}")
features = list(X_combined.columns)

# ── Sub-niche clustering (silhouette) ─────────────────────
# Cluster on CLIP mean PCA embeddings — they capture what the video is about
# Add cluster_id as a feature so the model knows which sub-niche each video belongs to
print("\nFinding optimal sub-niche clusters...")
clip_pca_cols = [c for c in X_combined.columns if c.startswith("clip_mean_pca_")]
if clip_pca_cols:
    cluster_data = X_combined[clip_pca_cols].fillna(0).values

    best_k, best_score = 2, -1
    silhouette_scores  = {}
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(cluster_data)
        score  = silhouette_score(cluster_data, labels, sample_size=min(2000, len(cluster_data)))
        silhouette_scores[k] = score
        print(f"  k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k     = k

    print(f"\nOptimal k={best_k} (silhouette={best_score:.4f})")
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(cluster_data)
    X_combined["cluster_id"] = cluster_labels
    features = list(X_combined.columns)

    joblib.dump(km_final,      f"models/kmeans_{NICHE}.pkl")
    joblib.dump(clip_pca_cols, f"models/cluster_cols_{NICHE}.pkl")

    # Print sample titles per cluster so you can see what sub-niches emerged
    if has_embeddings:
        df_with_clusters = df[mask].copy()
        df_with_clusters["cluster_id"] = cluster_labels
        print("\nSub-niche clusters (sample titles):")
        for c in range(best_k):
            cluster_df = df_with_clusters[df_with_clusters["cluster_id"] == c]
            print(f"\n  Cluster {c} ({len(cluster_df)} videos):")
            for title in cluster_df["title"].dropna().sample(min(5, len(cluster_df)), random_state=42):
                safe_title = title[:80].encode('ascii', errors='replace').decode('ascii')
                print(f"    - {safe_title}")
else:
    print("  No CLIP embeddings found — skipping clustering")

# ── Temporal train / val / test split ────────────────────
# Sort by publish date so older videos train, newer videos test.
# This prevents leakage of time-based trends into evaluation.
if "published_at" in df.columns:
    dates = pd.to_datetime(df.loc[X_combined.index, "published_at"], errors='coerce', utc=True)
    order = dates.argsort().values
    X_combined = X_combined.iloc[order].reset_index(drop=True)
    y           = y.iloc[order].reset_index(drop=True)
    print("Temporal split: sorted by publish date")
else:
    print("Warning: published_at not found — falling back to random split")

n       = len(X_combined)
n_test  = int(n * 0.20)
n_val   = int(n * 0.20)
n_train = n - n_test - n_val

X_train = X_combined.iloc[:n_train]
X_tv    = X_combined.iloc[:n_train + n_val]
X_val   = X_combined.iloc[n_train:n_train + n_val]
X_test  = X_combined.iloc[n_train + n_val:]
y_train = y.iloc[:n_train]
y_tv    = y.iloc[:n_train + n_val]
y_val   = y.iloc[n_train:n_train + n_val]
y_test  = y.iloc[n_train + n_val:]
print(f"Train: {len(X_train)}   Val: {len(X_val)}   Test: {len(X_test)}")

# ── Tune XGBoost with Optuna ──────────────────────────────
print("\nTuning XGBoost (50 trials)...")

def xgb_objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":        trial.suggest_int("max_depth", 2, 5),
        "learning_rate":    trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 30),
        "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5.0),
        "random_state":     42,
        "verbosity":        0,
    }
    kf     = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr)
        corr, _ = spearmanr(y_val, m.predict(X_val))
        scores.append(corr)
    return np.mean(scores)

xgb_study = optuna.create_study(direction="maximize")
xgb_study.optimize(xgb_objective, n_trials=50)
best_xgb_params = xgb_study.best_params
print(f"Best XGBoost CV: {xgb_study.best_value:.3f}")
print(f"Params: {best_xgb_params}")

# ── Tune Random Forest with Optuna ────────────────────────
print("\nTuning Random Forest (30 trials)...")

def rf_objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
        "max_depth":        trial.suggest_int("max_depth", 2, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
        "max_features":     trial.suggest_float("max_features", 0.3, 0.8),
        "random_state":     42,
        "n_jobs":           -1,
    }
    kf     = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        m = RandomForestRegressor(**params)
        m.fit(X_tr, y_tr)
        corr, _ = spearmanr(y_val, m.predict(X_val))
        scores.append(corr)
    return np.mean(scores)

rf_study = optuna.create_study(direction="maximize")
rf_study.optimize(rf_objective, n_trials=30)
best_rf_params = rf_study.best_params
print(f"Best RF CV: {rf_study.best_value:.3f}")
print(f"Params: {best_rf_params}")

# ── Tune Ridge with Optuna ────────────────────────────────
print("\nTuning Ridge (20 trials)...")

def ridge_objective(trial):
    alpha  = trial.suggest_float("alpha", 0.01, 100.0, log=True)
    kf     = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        m = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge',  Ridge(alpha=alpha))
        ])
        m.fit(X_tr, y_tr)
        corr, _ = spearmanr(y_val, m.predict(X_val))
        scores.append(corr)
    return np.mean(scores)

ridge_study = optuna.create_study(direction="maximize")
ridge_study.optimize(ridge_objective, n_trials=20)
best_ridge_alpha = ridge_study.best_params["alpha"]
print(f"Best Ridge CV: {ridge_study.best_value:.3f}")
print(f"Best alpha: {best_ridge_alpha:.4f}")

# ── Tune LightGBM with Optuna ─────────────────────────────
print("\nTuning LightGBM (50 trials)...")

def lgb_objective(trial):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":        trial.suggest_int("max_depth", 2, 6),
        "learning_rate":    trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
        "num_leaves":       trial.suggest_int("num_leaves", 15, 63),
        "subsample":        trial.suggest_float("subsample", 0.5, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
        "min_child_samples":trial.suggest_int("min_child_samples", 10, 50),
        "reg_alpha":        trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda":       trial.suggest_float("reg_lambda", 0.5, 5.0),
        "random_state":     42,
        "n_jobs":           -1,
        "verbose":          -1,
    }
    kf     = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for tr_idx, val_idx in kf.split(X_train):
        X_tr, X_kval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_kval = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr, y_tr)
        corr, _ = spearmanr(y_kval, m.predict(X_kval))
        scores.append(corr)
    return np.mean(scores)

lgb_study = optuna.create_study(direction="maximize")
lgb_study.optimize(lgb_objective, n_trials=50)
best_lgb_params = lgb_study.best_params
print(f"Best LightGBM CV: {lgb_study.best_value:.3f}")
print(f"Params: {best_lgb_params}")

# ── Train final models ────────────────────────────────────
print("\nTraining final models with best params...")

xgb_model = xgb.XGBRegressor(**best_xgb_params)
xgb_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(**best_rf_params)
rf_model.fit(X_train, y_train)

ridge_model = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge',  Ridge(alpha=best_ridge_alpha))
])
ridge_model.fit(X_train, y_train)

lgb_params_final = {**best_lgb_params, "random_state": 42, "n_jobs": -1, "verbose": -1}
lgb_model = lgb.LGBMRegressor(**lgb_params_final)
lgb_model.fit(X_train, y_train)

# ── Val predictions (for weight tuning) ───────────────────
xgb_val   = xgb_model.predict(X_val)
rf_val    = rf_model.predict(X_val)
ridge_val = ridge_model.predict(X_val)
lgb_val   = lgb_model.predict(X_val)

# ── Test predictions (for final evaluation only) ──────────
xgb_pred   = xgb_model.predict(X_test)
rf_pred    = rf_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)
lgb_pred   = lgb_model.predict(X_test)

xgb_corr,   _ = spearmanr(y_val, xgb_val)
rf_corr,    _ = spearmanr(y_val, rf_val)
ridge_corr, _ = spearmanr(y_val, ridge_val)
lgb_corr,   _ = spearmanr(y_val, lgb_val)

print(f"\nIndividual model performance (val):")
print(f"  XGBoost:       {xgb_corr:.3f}")
print(f"  Random Forest: {rf_corr:.3f}")
print(f"  Ridge:         {ridge_corr:.3f}")
print(f"  LightGBM:      {lgb_corr:.3f}")

# ── Tune ensemble weights with Optuna ─────────────────────
print("\nTuning ensemble weights (100 trials)...")

def ensemble_objective(trial):
    w1    = trial.suggest_float("w_xgb",   0.0, 0.6)
    w2    = trial.suggest_float("w_rf",    0.0, 0.5)
    w3    = trial.suggest_float("w_ridge", 0.0, 0.4)
    w4    = trial.suggest_float("w_lgb",   0.0, 0.6)
    total = w1 + w2 + w3 + w4
    if total == 0:
        return -1.0
    w1, w2, w3, w4 = w1/total, w2/total, w3/total, w4/total
    blend           = w1*xgb_val + w2*rf_val + w3*ridge_val + w4*lgb_val
    corr, _         = spearmanr(y_val, blend)
    return corr

ens_study = optuna.create_study(direction="maximize")
ens_study.optimize(ensemble_objective, n_trials=100)

bw    = ens_study.best_params
total = bw["w_xgb"] + bw["w_rf"] + bw["w_ridge"] + bw["w_lgb"]
w_xgb   = bw["w_xgb"]   / total
w_rf    = bw["w_rf"]    / total
w_ridge = bw["w_ridge"] / total
w_lgb   = bw["w_lgb"]   / total

# Cap RF weight post-normalization — RF averages trees which compresses scores
if w_rf > 0.40:
    excess  = w_rf - 0.40
    w_rf    = 0.40
    # redistribute excess proportionally to other models
    others  = w_xgb + w_ridge + w_lgb
    w_xgb   += excess * (w_xgb   / others)
    w_ridge += excess * (w_ridge / others)
    w_lgb   += excess * (w_lgb   / others)

print(f"\nOptimal weights:")
print(f"  XGBoost:       {w_xgb:.3f}")
print(f"  Random Forest: {w_rf:.3f}")
print(f"  Ridge:         {w_ridge:.3f}")
print(f"  LightGBM:      {w_lgb:.3f}")

# ── Final ensemble ────────────────────────────────────────
y_pred      = w_xgb*xgb_pred + w_rf*rf_pred + w_ridge*ridge_pred + w_lgb*lgb_pred
rmse        = np.sqrt(mean_squared_error(y_test, y_pred))
spearman, _ = spearmanr(y_test, y_pred)

print(f"\nFinal ensemble:")
print(f"  Spearman: {spearman:.3f}")
print(f"  RMSE:     {rmse:.3f}")
print(f"\nGain from ensemble:")
print(f"  vs XGBoost:    +{spearman - xgb_corr:.3f}")
print(f"  vs RF:         +{spearman - rf_corr:.3f}")
print(f"  vs Ridge:      +{spearman - ridge_corr:.3f}")
print(f"  vs LightGBM:   +{spearman - lgb_corr:.3f}")

if spearman >= 0.5:
    print("\n  Strong. Good enough to build a product on.")
elif spearman >= 0.35:
    print("\n  Decent. Collect more data to improve.")
else:
    print("\n  Weak. Need more data.")

# ── Cross validation on train+val set ────────────────────
# Uses equal weights — weights were tuned on X_val so
# can't reuse them here without leakage. CV purpose is
# to check generalization stability, not get the best number.
print("\n5-fold cross validation on train+val set (equal weights)...")
kf        = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_tv), 1):
    X_tr,     X_cv_val = X_tv.iloc[tr_idx], X_tv.iloc[val_idx]
    y_tr,     y_cv_val = y_tv.iloc[tr_idx], y_tv.iloc[val_idx]

    m_xgb = xgb.XGBRegressor(**best_xgb_params)
    m_xgb.fit(X_tr, y_tr)

    m_rf = RandomForestRegressor(**best_rf_params)
    m_rf.fit(X_tr, y_tr)

    m_ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge',  Ridge(alpha=best_ridge_alpha))
    ])
    m_ridge.fit(X_tr, y_tr)

    m_lgb = lgb.LGBMRegressor(**lgb_params_final)
    m_lgb.fit(X_tr, y_tr)

    p_xgb   = m_xgb.predict(X_cv_val)
    p_rf    = m_rf.predict(X_cv_val)
    p_ridge = m_ridge.predict(X_cv_val)
    p_lgb   = m_lgb.predict(X_cv_val)
    blend   = (p_xgb + p_rf + p_ridge + p_lgb) / 4

    corr, _ = spearmanr(y_cv_val, blend)
    cv_scores.append(corr)
    print(f"  Fold {fold}: {corr:.3f}")

print(f"\n  CV Mean: {np.mean(cv_scores):.3f}")
print(f"  CV Std:  {np.std(cv_scores):.3f}")
gap = abs(spearman - np.mean(cv_scores))
print(f"\n  Gap test vs CV: {gap:.3f}")
if gap < 0.05:
    print("  Small gap — generalising well.")
elif gap < 0.10:
    print("  Moderate gap — slight overfitting.")
else:
    print("  Large gap — collect more data.")

# ── Feature importance ────────────────────────────────────
print("\nTop 20 features:")
importance = pd.Series(
    xgb_model.feature_importances_,
    index=features
).sort_values(ascending=False)

for feat, imp in importance.head(20).items():
    bar = "█" * int(imp * 400)
    print(f"  {feat:<40} {bar} {imp:.4f}")

importance.reset_index().rename(
    columns={'index': 'feature', 0: 'importance'}
).to_csv(f"models/importance_{NICHE}.csv", index=False)

# ── SHAP ──────────────────────────────────────────────────
print("\nComputing SHAP values...")
explainer = shap.TreeExplainer(xgb_model)
shap_vals = explainer.shap_values(X_test)

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_vals, X_test,
    feature_names=features,
    show=False, max_display=20
)
plt.title(f"SHAP — {NICHE} — {TARGET}")
plt.tight_layout()
plt.savefig(f"models/shap_{NICHE}.png", dpi=150, bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_vals, X_test,
    feature_names=features,
    plot_type="bar",
    show=False, max_display=20
)
plt.tight_layout()
plt.savefig(f"models/shap_bar_{NICHE}.png", dpi=150, bbox_inches='tight')
plt.close()
print("  SHAP plots saved")

# ── Save everything ───────────────────────────────────────
print("\nSaving...")
niche_stats   = X_train.median().to_dict()
# Use ensemble predictions on training data as score_dist, not raw labels.
# Labels span 0–18 but model predicts a much narrower range, so comparing
# against label dist puts everything in a tight percentile band.
# Use val set predictions — model hasn't seen these, so scores are unbiased.
# Training set predictions are inflated (RF memorises training data).
X_holdout    = pd.concat([X_val, X_test])
holdout_preds = w_xgb*xgb_model.predict(X_holdout) + w_rf*rf_model.predict(X_holdout) + \
                w_ridge*ridge_model.predict(X_holdout) + w_lgb*lgb_model.predict(X_holdout)
score_dist   = np.sort(holdout_preds)

# ── Top-video benchmarks ──────────────────────────────────
# Mean feature values for top 25% videos by score.
# Used in recommendations to show "top creators average X, you have Y"
top_threshold  = y_train.quantile(0.75)
top_mask       = y_train >= top_threshold
top_benchmarks = X_train[top_mask].mean().to_dict()

joblib.dump(xgb_model,                      f"models/xgb_{NICHE}.pkl")
joblib.dump(rf_model,                       f"models/rf_{NICHE}.pkl")
joblib.dump(ridge_model,                    f"models/ridge_{NICHE}.pkl")
joblib.dump(lgb_model,                      f"models/lgb_{NICHE}.pkl")
joblib.dump((w_xgb, w_rf, w_ridge, w_lgb),  f"models/weights_{NICHE}.pkl")
joblib.dump(
    (imputer_continuous, imputer_binary, continuous_cols, binary_cols),
    f"models/imputer_{NICHE}.pkl"
)
joblib.dump(features,               f"models/features_{NICHE}.pkl")
joblib.dump(niche_stats,            f"models/niche_stats_{NICHE}.pkl")
joblib.dump(score_dist,             f"models/score_dist_{NICHE}.pkl")
joblib.dump(top_benchmarks,         f"models/top_benchmarks_{NICHE}.pkl")
joblib.dump(explainer,              f"models/explainer_{NICHE}.pkl")

print(f"""
Saved:
  models/xgb_{NICHE}.pkl
  models/rf_{NICHE}.pkl
  models/ridge_{NICHE}.pkl
  models/lgb_{NICHE}.pkl
  models/weights_{NICHE}.pkl
  models/imputer_{NICHE}.pkl
  models/features_{NICHE}.pkl
  models/niche_stats_{NICHE}.pkl
  models/explainer_{NICHE}.pkl
  models/shap_{NICHE}.png
  models/shap_bar_{NICHE}.png
  models/importance_{NICHE}.csv
""")

# ── Recommendation engine ─────────────────────────────────
RECOMMENDATION_TEXT = {
    "hook_brightness": {
        "low":  "Your first 3 seconds are too dark. Bright hooks retain 2x more viewers.",
        "high": "Hook brightness is strong."
    },
    "hook_contrast": {
        "low":  "Hook has low contrast. Make visuals pop — viewers decide in 1 second.",
        "high": "Hook contrast is good."
    },
    "hook_has_face": {
        "low":  "No face in hook. Videos with a human face in the first frame retain more viewers.",
        "high": "Face in hook — good."
    },
    "hook_face_size": {
        "low":  "Face too small. Get closer to camera — fill 30%+ of the frame.",
        "high": "Good face framing."
    },
    "hook_text_proxy": {
        "low":  "No text in hook. 70% of reels watched silently — text in first 2s grabs them.",
        "high": "Text overlay in hook — good."
    },
    "hook_saturation": {
        "low":  "Hook looks washed out. More vibrant colours stop the scroll.",
        "high": "Hook colour vivid — good."
    },
    "hook_centre_brightness": {
        "low":  "Subject underlit. Centre of frame should be brighter than background.",
        "high": "Subject lighting good."
    },
    "hook_sharpness": {
        "low":  "Hook is out of focus. First frame must be sharp — it's your thumbnail.",
        "high": "Hook is sharp — good."
    },
    "mean_loudness": {
        "low":  "Audio too quiet. Low volume = low energy = people scroll away.",
        "high": "Audio loudness good."
    },
    "hook_loudness": {
        "low":  "Hook audio too quiet. Hit viewers with full energy from second one.",
        "high": "Hook audio energy good."
    },
    "silence_ratio": {
        "high": "Too much silence. Dead air kills retention. Cut the pauses.",
        "low":  "Good — no excessive silence."
    },
    "music_score": {
        "low":  "No background music detected. Music increases saves by ~35% in this niche.",
        "high": "Background music — good."
    },
    "hook_music_score": {
        "low":  "No music in hook. Start with music immediately — don't wait.",
        "high": "Music in hook — good."
    },
    "cuts_per_second": {
        "low":  "Pacing too slow. Top shorts average 1.5-2.5 cuts/second. Edit faster.",
        "high": "Good editing pace."
    },
    "mean_has_face": {
        "low":  "Faces appear rarely. Stay on camera more consistently.",
        "high": "Good face presence."
    },
    "mean_motion": {
        "low":  "Video too static. Add movement — camera, subject, or faster cuts.",
        "high": "Good motion level."
    },
    "temporal_buildup": {
        "low":  "Energy drops over time. Build towards a climax — don't start high and fade.",
        "high": "Good energy buildup."
    },
    "temporal_hook_motion": {
        "low":  "Hook is too static. Start with movement — it signals energy immediately.",
        "high": "Good hook motion."
    },
    "brightness_trend": {
        "low":  "Video gets darker over time. Maintain or build brightness.",
        "high": "Good brightness arc."
    },
    "tag_count": {
        "low":  "Too few hashtags. Add 3-5 relevant ones for discoverability.",
        "high": "Good hashtag count."
    },
    "title_has_number": {
        "low":  "Add a number to your title — '5 ways' and '30 day' perform better.",
        "high": "Good — number in title."
    },
    "title_has_question": {
        "low":  "Consider a question title. Questions drive curiosity and clicks.",
        "high": "Good — question in title."
    },
    "posted_prime_time": {
        "low":  "Posted outside prime time. Post between 5pm-10pm for best reach.",
        "high": "Good posting time."
    },
    "tempo_bpm": {
        "low":  "Music tempo slow for this niche. Higher BPM = more energy.",
        "high": "Good music tempo."
    },
    "harmonic_ratio": {
        "low":  "Audio lacks melodic quality. More harmonic music increases saves.",
        "high": "Good harmonic audio."
    },
    "desc_length": {
        "low":  "Description too short. Longer descriptions help search discoverability.",
        "high": "Good description length."
    },
    "df_emotion_score": {
        "low":  "Expression in hook looks flat or negative. A confident or happy expression retains more viewers.",
        "high": "Good expression in hook."
    },
    "df_happy_score": {
        "low":  "Hook expression isn't engaging enough. Smile or show high energy in the first frame.",
        "high": "Good hook energy."
    },
    "df_face_detected": {
        "low":  "No face detected in hook by DeepFace. Show your face in the first second.",
        "high": "Face detected in hook — good."
    },
    "mean_edge_density": {
        "low":  "Video looks visually simple. More complexity keeps eyes engaged.",
        "high": "Good visual complexity."
    },
    "dynamic_range": {
        "low":  "Audio too flat and uniform. Dynamic audio with peaks and drops is more engaging.",
        "high": "Good audio dynamics."
    },
}

def generate_recommendations(feature_dict, niche="fitness"):
    xgb_m   = joblib.load(f"models/xgb_{niche}.pkl")
    rf_m    = joblib.load(f"models/rf_{niche}.pkl")
    ridge_m = joblib.load(f"models/ridge_{niche}.pkl")
    lgb_m   = joblib.load(f"models/lgb_{niche}.pkl")
    weights = joblib.load(f"models/weights_{niche}.pkl")
    imp_cont, imp_bin, cont_cols, bin_cols = joblib.load(f"models/imputer_{niche}.pkl")
    feats   = joblib.load(f"models/features_{niche}.pkl")
    exp     = joblib.load(f"models/explainer_{niche}.pkl")
    stats   = joblib.load(f"models/niche_stats_{niche}.pkl")
    w1, w2, w3, w4 = weights

    # Assign cluster_id if clustering was used
    km_path = Path(f"models/kmeans_{niche}.pkl")
    if km_path.exists() and "cluster_id" in feats:
        km          = joblib.load(km_path)
        cc          = joblib.load(f"models/cluster_cols_{niche}.pkl")
        cc_vals     = np.array([[feature_dict.get(c, stats.get(c, 0)) for c in cc]])
        feature_dict["cluster_id"] = int(km.predict(cc_vals)[0])

    sample_df = pd.DataFrame(
        [[feature_dict.get(f, stats.get(f, 0)) for f in feats]],
        columns=feats
    )
    hc_feats = cont_cols + bin_cols
    hc_cols_in_feats = [f for f in hc_feats if f in sample_df.columns]
    if hc_cols_in_feats:
        s_cont = pd.DataFrame(
            imp_cont.transform(sample_df[[f for f in cont_cols if f in sample_df.columns]]),
            columns=[f for f in cont_cols if f in sample_df.columns]
        )
        s_bin = pd.DataFrame(
            imp_bin.transform(sample_df[[f for f in bin_cols if f in sample_df.columns]]),
            columns=[f for f in bin_cols if f in sample_df.columns]
        )
        for col in s_cont.columns:
            sample_df[col] = s_cont[col].values
        for col in s_bin.columns:
            sample_df[col] = s_bin[col].values
    sample = sample_df[feats].values

    xgb_score   = float(xgb_m.predict(sample)[0])
    rf_score    = float(rf_m.predict(sample)[0])
    ridge_score = float(ridge_m.predict(sample)[0])
    lgb_score   = float(lgb_m.predict(sample)[0])
    score       = w1*xgb_score + w2*rf_score + w3*ridge_score + w4*lgb_score

    s_vals = exp.shap_values(pd.DataFrame(sample, columns=feats))[0] if sample.ndim == 2 else exp.shap_values(pd.DataFrame([sample], columns=feats))[0]
    pairs  = sorted(zip(feats, s_vals), key=lambda x: x[1])

    recs = []
    for feat, shap_val in pairs:
        if shap_val >= 0:
            break
        if feat not in RECOMMENDATION_TEXT:
            continue

        val       = feature_dict.get(feat, stats.get(feat, 0))
        median    = stats.get(feat, 0)
        direction = "low" if val < median else "high"
        message   = RECOMMENDATION_TEXT[feat][direction]

        if "good" in message.lower() and message.endswith("."):
            continue

        fixed       = sample.copy()
        feat_idx    = list(feats).index(feat)
        fixed[0][feat_idx] = median
        fixed_df    = pd.DataFrame(fixed, columns=feats)

        fixed_score = (
            w1 * float(xgb_m.predict(fixed_df)[0]) +
            w2 * float(rf_m.predict(fixed_df)[0])  +
            w3 * float(ridge_m.predict(fixed_df)[0]) +
            w4 * float(lgb_m.predict(fixed_df)[0])
        )
        impact = fixed_score - score

        if impact <= 0:
            continue

        recs.append({
            "feature":  feat,
            "message":  message,
            "impact":   round(impact, 3),
            "priority": "HIGH" if abs(shap_val) > 0.3 else "MEDIUM"
        })

    recs = sorted(recs, key=lambda x: x["impact"], reverse=True)

    return {
        "predicted_score": round(score, 3),
        "xgb_score":       round(xgb_score, 3),
        "rf_score":        round(rf_score, 3),
        "ridge_score":     round(ridge_score, 3),
        "lgb_score":       round(lgb_score, 3),
        "recommendations": recs[:5]
    }

# ── Test on one sample ────────────────────────────────────
print("Testing recommendation engine...")
sample_feats = {f: float(X_test.iloc[0][f]) for f in features}
result       = generate_recommendations(sample_feats, niche=NICHE)

print(f"\nSample result:")
print(f"  Ensemble: {result['predicted_score']}")
print(f"  XGB: {result['xgb_score']}  RF: {result['rf_score']}  Ridge: {result['ridge_score']}  LGB: {result['lgb_score']}")
print(f"\nRecommendations:")
for r in result["recommendations"]:
    print(f"  [{r['priority']}] {r['message']}")
    print(f"           Impact: +{r['impact']:.3f}")

print(f"\nAll done.")
print(f"Spearman: {spearman:.3f}  |  CV Mean: {np.mean(cv_scores):.3f}")