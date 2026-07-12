import os
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    print("ERROR: YOUTUBE_API_KEY not found in .env file")
    exit()

CSV_PATH = Path("data/metadata.csv")
NICHE    = "fitness"

# ── Find IDs with video but no metadata ───────────────────
meta_ids   = set(pd.read_csv(CSV_PATH, on_bad_lines='skip', engine='python')["video_id"])
downloaded = {p.stem for p in Path("data/raw_videos").glob("*.mp4")}
missing    = list(downloaded - meta_ids)
print(f"Videos missing metadata: {len(missing)}")

if not missing:
    print("Nothing to fetch.")
    exit()

youtube = build('youtube', 'v3', developerKey=API_KEY)

# ── Fetch video stats in batches of 50 ────────────────────
print("Fetching video stats...")
video_items = []
for i in range(0, len(missing), 50):
    batch = missing[i:i+50]
    try:
        resp = youtube.videos().list(
            part='statistics,contentDetails,snippet,topicDetails',
            id=','.join(batch)
        ).execute()
        video_items.extend(resp.get('items', []))
        print(f"  Fetched {len(video_items)} so far...")
    except Exception as e:
        if "quotaExceeded" in str(e):
            print(f"  Quota exceeded after {len(video_items)} videos. Saving what we have.")
            break
        print(f"  Batch failed: {e}")
        continue

print(f"Got stats for {len(video_items)} videos")

if not video_items:
    print("No stats fetched. Quota exhausted. Try tomorrow.")
    exit()

# ── Fetch channel info ────────────────────────────────────
print("Fetching channel info...")
channel_ids    = list(set(item['snippet']['channelId'] for item in video_items))
channel_lookup = {}
for i in range(0, len(channel_ids), 50):
    batch = channel_ids[i:i+50]
    try:
        resp = youtube.channels().list(
            part='snippet,statistics,brandingSettings,contentDetails',
            id=','.join(batch)
        ).execute()
        for ch in resp.get('items', []):
            channel_lookup[ch['id']] = ch
    except Exception as e:
        if "quotaExceeded" in str(e):
            print(f"  Quota hit during channel fetch.")
            break
        print(f"  Channel batch failed: {e}")
        continue

# ── Build rows ────────────────────────────────────────────
print("Processing metadata...")
rows = []
now  = datetime.now(timezone.utc)

for item in video_items:
    vid_id     = item['id']
    snippet    = item['snippet']
    stats      = item['statistics']
    details    = item['contentDetails']
    channel_id = snippet['channelId']
    channel    = channel_lookup.get(channel_id, {})
    ch_snippet = channel.get('snippet', {})
    ch_stats   = channel.get('statistics', {})

    published_at = snippet.get('publishedAt', '')
    if published_at:
        pub_dt         = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
        video_age_days = (now - pub_dt).days
        posted_hour    = pub_dt.hour
        posted_weekday = pub_dt.weekday()
    else:
        video_age_days = -1
        posted_hour    = -1
        posted_weekday = -1

    ch_created = ch_snippet.get('publishedAt', '')
    if ch_created:
        ch_dt            = datetime.fromisoformat(ch_created.replace('Z', '+00:00'))
        channel_age_days = (now - ch_dt).days
    else:
        channel_age_days = -1

    has_likes    = 'likeCount'    in stats
    has_comments = 'commentCount' in stats
    likes        = int(stats['likeCount'])    if has_likes    else None
    comments     = int(stats['commentCount']) if has_comments else None
    views        = int(stats.get('viewCount', 0))

    title              = snippet.get('title', '')
    title_length       = len(title)
    title_word_count   = len(title.split())
    title_has_number   = any(c.isdigit() for c in title)
    title_has_question = '?' in title
    title_has_exclaim  = '!' in title
    title_caps_ratio   = sum(1 for c in title if c.isupper()) / max(len(title), 1)
    title_has_emoji    = any(ord(c) > 127 for c in title)
    title_has_ellipsis = '...' in title or '…' in title

    description   = snippet.get('description', '')
    tags          = snippet.get('tags', [])
    tag_count     = len(tags)
    desc_length   = len(description)
    desc_has_link = 'http' in description.lower()
    has_hashtags  = '#' in description or '#' in title

    subscriber_count    = int(ch_stats.get('subscriberCount', 0))
    channel_video_count = int(ch_stats.get('videoCount', 0))
    channel_total_views = int(ch_stats.get('viewCount', 0))
    channel_avg_views   = channel_total_views / max(channel_video_count, 1)
    has_custom_url      = 'customUrl' in ch_snippet

    rows.append({
        "video_id":            vid_id,
        "niche":               NICHE,
        "title":               title,
        "published_at":        published_at,
        "video_age_days":      video_age_days,
        "posted_hour":         posted_hour,
        "posted_weekday":      posted_weekday,
        "duration":            details.get('duration', ''),
        "definition":          details.get('definition', ''),
        "caption":             details.get('caption', ''),
        "licensed_content":    details.get('licensedContent', False),
        "title_length":        title_length,
        "title_word_count":    title_word_count,
        "title_has_number":    title_has_number,
        "title_has_question":  title_has_question,
        "title_has_exclaim":   title_has_exclaim,
        "title_caps_ratio":    round(title_caps_ratio, 4),
        "title_has_emoji":     title_has_emoji,
        "title_has_ellipsis":  title_has_ellipsis,
        "tag_count":           tag_count,
        "desc_length":         desc_length,
        "desc_has_link":       desc_has_link,
        "has_hashtags":        has_hashtags,
        "views":               views,
        "likes":               likes,
        "has_likes":           has_likes,
        "comments":            comments,
        "has_comments":        has_comments,
        "channel_id":          channel_id,
        "channel_name":        snippet.get('channelTitle', ''),
        "channel_created":     ch_created,
        "channel_age_days":    channel_age_days,
        "subscriber_count":    subscriber_count,
        "channel_video_count": channel_video_count,
        "channel_total_views": channel_total_views,
        "channel_avg_views":   round(channel_avg_views, 2),
        "has_custom_url":      has_custom_url,
        "collected_at":        now.strftime("%Y-%m-%d %H:%M"),
    })

# ── Append to metadata.csv ────────────────────────────────
if rows:
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nAppended {len(rows)} rows to {CSV_PATH}")
else:
    print("No rows to save.")

remaining = len(missing) - len(video_items)
if remaining > 0:
    print(f"\n{remaining} videos still missing metadata — quota ran out. Re-run tomorrow.")
else:
    print(f"\nAll missing metadata fetched!")
