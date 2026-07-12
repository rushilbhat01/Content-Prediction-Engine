import pandas as pd
import numpy as np
import re

df = pd.read_csv(
    "data/metadata.csv",
    on_bad_lines='skip',    # skip corrupted rows
    quoting=0,              # standard CSV quoting
    engine='python'         # more tolerant parser
)
print(f"Note: some rows may have been skipped due to formatting issues")
print(f"Before cleaning: {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# ── Remove duplicates ─────────────────────────────────────
df = df.drop_duplicates(subset="video_id")
print(f"After removing duplicates: {len(df)}")

# ── Fix views ─────────────────────────────────────────────
df["views"] = pd.to_numeric(df["views"], errors='coerce')
df = df[df["views"] > 0]
print(f"After removing 0 views: {len(df)}")

# ── Remove too-new videos ─────────────────────────────────
df["video_age_days"] = pd.to_numeric(df["video_age_days"], errors='coerce')
df = df[df["video_age_days"] >= 7]
print(f"After removing too-new videos: {len(df)}")

print(f"Max views in dataset: {df['views'].max():,.0f}")

# ── Parse duration → seconds ──────────────────────────────
def parse_duration(d):
    if not isinstance(d, str):
        return 0
    match = re.match(r'PT(?:(\d+)M)?(?:(\d+)S)?', d)
    if not match:
        return 0
    return int(match.group(1) or 0) * 60 + int(match.group(2) or 0)

df["duration_seconds"] = df["duration"].apply(parse_duration)
df = df[(df["duration_seconds"] >= 5) & (df["duration_seconds"] <= 60)]
print(f"After removing wrong duration: {len(df)}")

# ── Fix numeric columns ───────────────────────────────────
numeric_cols = [
    "likes", "comments", "subscriber_count",
    "channel_video_count", "channel_total_views",
    "channel_avg_views", "channel_age_days",
    "posted_hour", "posted_weekday",
    "posted_day", "posted_month",
    "tag_count", "desc_length",
    "title_length", "title_word_count",
    "title_caps_ratio", "desc_line_count",
    "video_age_days",
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ── Fix boolean columns ───────────────────────────────────
bool_cols = [
    "has_likes", "has_comments", "caption",
    "licensed_content", "title_has_number",
    "title_has_question", "title_has_emoji",
    "title_has_exclaim", "title_all_caps",
    "title_all_lower", "title_has_ellipsis",
    "has_hashtags", "desc_has_link",
    "has_custom_url", "channel_has_keywords",
    "is_english", "is_hindi", "is_indian_channel",
]
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower().map(
            {'true': 1, 'false': 0, '1': 1, '0': 0,
             'yes': 1, 'no': 0, 'nan': 0}
        ).fillna(0).astype(int)

# ── Clip extremes ─────────────────────────────────────────
df["subscriber_count"]  = pd.to_numeric(df["subscriber_count"],  errors='coerce').clip(lower=1)
df["channel_age_days"]  = pd.to_numeric(df["channel_age_days"],  errors='coerce').clip(lower=1)
df["video_age_days"]    = pd.to_numeric(df["video_age_days"],    errors='coerce').clip(lower=1)
df["channel_avg_views"] = pd.to_numeric(df["channel_avg_views"], errors='coerce').clip(lower=1)

# ── Definition as binary ──────────────────────────────────
if "definition" in df.columns:
    df["is_hd"] = (df["definition"] == "hd").astype(int)
else:
    df["is_hd"] = 0

# ── Training targets ──────────────────────────────────────
df["log_views"] = np.log1p(df["views"])

df["virality_score"] = np.log1p(
    df["views"] / df["subscriber_count"]
)

df["views_per_day"]     = df["views"] / df["video_age_days"]
df["log_views_per_day"] = np.log1p(df["views_per_day"])

# Best target — controls for both age AND channel size
df["normalised_score"] = np.log1p(
    df["views"] / (
        df["subscriber_count"] *
        df["video_age_days"]
    ) * 1000
)

df["vs_channel_avg"] = (
    np.log1p(df["views"]) -
    np.log1p(df["channel_avg_views"])
)

# ── Derived input features ────────────────────────────────
# Channel productivity
if "channel_video_count" in df.columns:
    df["channel_video_count"] = pd.to_numeric(
        df["channel_video_count"], errors='coerce'
    )
    df["videos_per_day"] = (
        df["channel_video_count"] /
        df["channel_age_days"].clip(lower=1)
    )

# ── Log scale channel features ────────────────────────────
# Cap subscriber count at 1M to reduce its dominance
# Difference between 1k and 10k subs matters more
# than difference between 1M and 10M
df["log_subscribers"] = np.log1p(
    df["subscriber_count"].clip(upper=1_000_000)
)
df["log_channel_avg_views"] = np.log1p(df["channel_avg_views"])

# ── Posting time ──────────────────────────────────────────
if "posted_hour" in df.columns:
    df["posted_on_weekend"]   = df["posted_weekday"].isin([5, 6]).astype(int)
    df["posted_prime_time"]   = df["posted_hour"].between(17, 22).astype(int)
    df["posted_morning"]      = df["posted_hour"].between(6, 11).astype(int)
    df["weekend_x_primetime"] = df["posted_on_weekend"] * df["posted_prime_time"]

if "posted_day" in df.columns:
    df["posted_end_of_month"] = df["posted_day"].between(25, 31).astype(int)

# ── Like and comment rates ────────────────────────────────
if "has_likes" in df.columns:
    df["like_rate"] = np.where(
        df["has_likes"] == 1,
        df["likes"] / df["views"].clip(lower=1),
        np.nan
    )

if "has_comments" in df.columns:
    df["comment_rate"] = np.where(
        df["has_comments"] == 1,
        df["comments"] / df["views"].clip(lower=1),
        np.nan
    )

# ── Title density ─────────────────────────────────────────
if "title_length" in df.columns and "title_word_count" in df.columns:
    df["title_chars_per_word"] = (
        df["title_length"] /
        df["title_word_count"].clip(lower=1)
    )

# ── Print summary ─────────────────────────────────────────
print(f"\nFinal dataset: {len(df)} rows, {len(df.columns)} columns")

if "niche" in df.columns:
    print(f"\nNiche breakdown:")
    print(df["niche"].value_counts())

print(f"\nView stats:")
print(df["views"].describe().apply(lambda x: f"{x:,.0f}"))

print(f"\nNormalised score stats:")
print(df["normalised_score"].describe().round(3))

print(f"\nlog_subscribers stats (capped at 1M):")
print(df["log_subscribers"].describe().round(3))

print(f"\nAvailable model features check:")
model_cols = [
    "video_age_days", "log_views_per_day", "posted_hour",
    "posted_weekday", "posted_on_weekend", "posted_prime_time",
    "posted_morning", "weekend_x_primetime", "duration_seconds",
    "is_hd", "caption", "title_length", "title_has_number",
    "title_has_question", "title_has_emoji", "title_caps_ratio",
    "tag_count", "desc_length", "has_hashtags", "has_likes",
    "has_comments", "channel_age_days", "log_subscribers",
    "log_channel_avg_views", "channel_video_count",
    "videos_per_day", "vs_channel_avg",
]
available = [c for c in model_cols if c in df.columns]
missing   = [c for c in model_cols if c not in df.columns]
print(f"  Available: {len(available)} / {len(model_cols)}")
if missing:
    print(f"  Missing:   {missing}")

# ── Save ──────────────────────────────────────────────────
df.to_csv("data/metadata_clean.csv", index=False)
print(f"\nSaved → data/metadata_clean.csv")