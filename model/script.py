import os
import csv
import time
import random
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dotenv import load_dotenv
from googleapiclient.discovery import build

# ── Load API key ──────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    print("ERROR: YOUTUBE_API_KEY not found in .env file")
    exit()

# ── Arguments ─────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--niche', default='fitness')
parser.add_argument('--count', type=int, default=100)
parser.add_argument('--download-missing', action='store_true',
                    help='Download videos that have metadata but no video file')
args  = parser.parse_args()
niche = args.niche
count = args.count

Path("data/raw_videos").mkdir(parents=True, exist_ok=True)
CSV_PATH = Path("data/metadata.csv")

# ── Download-missing mode ─────────────────────────────────
if args.download_missing:
    meta_ids   = set(pd.read_csv(CSV_PATH, on_bad_lines='skip', engine='python')["video_id"])
    downloaded = {p.stem for p in Path("data/raw_videos").glob("*.mp4")}
    to_download = list(meta_ids - downloaded)
    print(f"Videos with metadata but not downloaded: {len(to_download)}")

    success = failed = 0
    for i, vid_id in enumerate(to_download, 1):
        out_path = Path(f"data/raw_videos/{vid_id}.mp4")
        if out_path.exists():
            success += 1
            continue
        url = f"https://www.youtube.com/watch?v={vid_id}"
        print(f"[{i}/{len(to_download)}] Downloading {vid_id}...", end=' ', flush=True)
        result = subprocess.run([
            "yt-dlp",
            "-o", "data/raw_videos/%(id)s.%(ext)s",
            "--merge-output-format", "mp4",
            "--quiet", "--no-warnings",
            url
        ], capture_output=True)
        if result.returncode == 0 and out_path.exists():
            success += 1
            print("OK")
        else:
            failed += 1
            print("failed")
        time.sleep(random.uniform(2, 4))

    print(f"\nDone! Downloaded: {success}  Failed: {failed}")
    exit()

# ── Search queries ────────────────────────────────────────
NICHE_QUERIES = {
    "fitness": [
        "how to get abs shorts", "fitness meal prep shorts",
        "how to lose weight shorts", "pre workout meal shorts",
        "how to build muscle shorts", "protein meal shorts",
        "how to get fit shorts", "post workout meal shorts",
        "what i eat fitness shorts", "whey protein shorts",
        "fat to fit shorts", "cable machine shorts",
        "lose belly fat shorts", "resistance band workout shorts",
        "weight loss results shorts", "glow up fitness shorts",
        "6 month transformation shorts", "before after fitness shorts",
        "1 year transformation shorts", "body recomposition shorts",
        "30 day challenge shorts", "tone body shorts",
        "100 pushup challenge shorts", "build muscle shorts",
        "pull up challenge shorts", "get abs shorts",
        "push pull legs shorts", "body transformation shorts",
        "6 pack abs shorts", "pull up shorts",
        "back pain exercise shorts", "bench press shorts",
        "no days off shorts", "plank challenge shorts",
        "squat challenge shorts", "lower body workout shorts",
        "upper body workout shorts", "lower back workout shorts",
        "indian gym motivation shorts", "indian diet fitness shorts",
        "desi bodybuilder shorts", "indian athlete shorts",
        "desi gym bro shorts", "gym mistakes beginners shorts",
        "fitness myths debunked shorts", "running tips shorts",
        "5 minute workout shorts", "10 minute workout shorts",
        "hotel room workout shorts", "75 hard shorts",
        "no equipment workout shorts", "calisthenics skills shorts",
        "sports motivation shorts", "fat burn workout shorts",
        "athlete training shorts", "boxing training shorts",
        "full body workout shorts", "mma training shorts",
        "healthy lifestyle shorts", "posture correction shorts",
        "circuit training shorts", "leg day shorts",
        "compound exercise shorts", "strength training shorts",
        "workout for women shorts", "workout for men shorts",
        "exercise science shorts", "muscle building shorts",
        "hard work gym shorts", "weight loss shorts",
        "natural bodybuilding shorts", "fit check shorts",
        "gym diet shorts", "creatine shorts",
        "machine workout shorts", "gym day shorts",
        "kettlebell shorts", "gym vlog shorts",
        "morning workout shorts", "barbell workout shorts",
        "night workout shorts", "dumbbell workout shorts",
        "gym tips shorts", "gym mistakes shorts",
        "fitness myths shorts", "fitness transformation shorts",
        "workout mistake shorts", "gym hack shorts",
        "fitness hack shorts", "gym progress shorts",
        "fitness results shorts", "dips workout shorts",
        "squat shorts", "deadlift shorts",
        "core workout shorts", "desi gym shorts",
        "oblique workout shorts", "hindi workout shorts",
        "quad workout shorts", "hamstring workout shorts",
        "calf workout shorts", "india fitness shorts",
        "forearm workout shorts", "mumbai fitness shorts",
        "arm workout shorts", "delhi gym shorts",
        "booty workout shorts", "glute workout shorts",
        "bhai fitness shorts", "bodyweight workout shorts",
        "fitness model shorts", "fitness india shorts",
        "gym aesthetic shorts", "gym beginner shorts",
        "pilates shorts", "yoga fitness shorts",
        "quick workout shorts", "crossfit shorts",
        "park workout shorts", "powerlifting shorts",
        "outdoor workout shorts", "protein shorts fitness",
        "fitness diet shorts", "bedroom workout shorts",
        "workout routine shorts", "street workout shorts",
        "fitness challenge shorts", "functional fitness shorts",
        "gym transformation shorts", "mobility workout shorts",
        "workout tips shorts", "flexibility shorts",
        "fitness tips shorts", "stretching shorts",
        "gym motivation shorts", "hiit workout shorts",
        "tricep workout shorts", "bicep workout shorts",
        "shoulder workout shorts", "back workout shorts",
        "tabata workout shorts", "chest workout shorts",
        "abs workout shorts", "supersets workout shorts",
        "cardio workout shorts", "calisthenics shorts",
        "bodybuilding shorts", "discipline fitness shorts",
        "fitness motivation shorts", "gym rat shorts",
        "home workout shorts", "gym fail shorts",
        "gym outfit shorts", "gym workout shorts",
        "gym shorts", "workout shorts",
        "fitness shorts",
    ],
    "food": [
        "5 ingredient recipe shorts", "one pan meal shorts",
        "high protein meal shorts", "meal prep sunday shorts",
        "what i eat in a day shorts", "easy dinner recipe shorts",
        "quick lunch recipe shorts", "healthy breakfast shorts",
        "weight loss meal shorts", "budget meal prep shorts",
        "air fryer recipe shorts", "10 minute meal shorts",
        "no cook meal shorts", "protein bowl shorts",
        "smoothie recipe shorts", "overnight oats shorts",
        "egg recipe shorts", "pasta recipe shorts",
        "street food india shorts", "mumbai street food shorts",
        "indian street food shorts", "desi food shorts",
        "indian recipe shorts", "hindi cooking shorts",
        "food asmr shorts", "cooking satisfying shorts",
        "baking fails shorts", "cake decorating shorts",
        "bread recipe shorts", "cookie recipe shorts",
        "food hack shorts", "cooking tip shorts",
        "kitchen hack shorts", "food trick shorts",
        "food science shorts", "cooking mistake shorts",
        "restaurant review shorts", "food review shorts",
        "mukbang shorts", "taste test shorts",
        "vegan recipe shorts", "keto recipe shorts",
        "gluten free recipe shorts", "dairy free recipe shorts",
        "meal prep beginners shorts", "food challenge shorts",
        "food comparison shorts", "chef tips shorts",
        "cooking basics shorts", "knife skills shorts",
        "food history shorts", "spice guide shorts",
        "food shorts", "recipe shorts", "cooking shorts",
        "healthy food shorts", "quick recipe shorts",
        "easy recipe shorts", "baking shorts", "meal prep shorts",
    ],
    "comedy": [
        "comedy shorts", "funny shorts", "meme shorts",
        "stand up shorts", "prank shorts", "funny video shorts",
        "comedy skit shorts", "humor shorts",
        "relatable shorts", "viral comedy shorts",
        "indian comedy shorts", "desi comedy shorts",
        "office humor shorts", "school life shorts",
        "couple humor shorts", "parenting funny shorts",
        "sibling humor shorts", "roommate shorts",
        "awkward moments shorts", "social anxiety shorts",
        "introvert humor shorts", "millennial humor shorts",
        "gen z humor shorts", "expectation vs reality shorts",
        "this is fine shorts", "life hack fail shorts",
    ],
    "beauty": [
        "skincare routine morning shorts", "skincare routine night shorts",
        "glass skin routine shorts", "acne treatment shorts",
        "dark spot removal shorts", "hyperpigmentation shorts",
        "retinol routine shorts", "vitamin c serum shorts",
        "sunscreen shorts", "moisturiser shorts",
        "anti aging skincare shorts", "skincare for beginners shorts",
        "budget skincare shorts", "drugstore skincare shorts",
        "skincare ingredients shorts", "skincare mistakes shorts",
        "glow up transformation shorts", "before after skincare shorts",
        "30 day skincare shorts", "no makeup look shorts",
        "natural makeup shorts", "makeup for beginners shorts",
        "everyday makeup shorts", "glam makeup shorts",
        "eye makeup shorts", "lip liner shorts",
        "contour tutorial shorts", "blush placement shorts",
        "makeup hack shorts", "makeup dupe shorts",
        "hair care routine shorts", "hair growth shorts",
        "damaged hair repair shorts", "hair oiling shorts",
        "curly hair routine shorts", "hair mask shorts",
        "nail art shorts", "nail tutorial shorts",
        "body care routine shorts", "fragrance shorts",
        "beauty tips shorts", "beauty hack shorts",
        "beauty shorts", "skincare shorts", "makeup shorts",
        "makeup tutorial shorts", "beauty review shorts",
    ],
    "education": [
        "did you know shorts", "mind blowing facts shorts",
        "psychology facts shorts", "human psychology shorts",
        "brain facts shorts", "science fact shorts",
        "space facts shorts", "universe shorts",
        "history facts shorts", "ancient history shorts",
        "world war facts shorts", "untold history shorts",
        "economics explained shorts", "how money works shorts",
        "compound interest shorts", "financial literacy shorts",
        "how does it work shorts", "engineering explained shorts",
        "biology facts shorts", "chemistry facts shorts",
        "physics facts shorts", "math trick shorts",
        "language learning shorts", "english tips shorts",
        "vocabulary shorts", "grammar tip shorts",
        "study tips shorts", "memory technique shorts",
        "how to study shorts", "productivity study shorts",
        "exam tips shorts", "note taking shorts",
        "coding explained shorts", "tech explained shorts",
        "ai explained shorts", "how internet works shorts",
        "philosophy shorts", "stoicism shorts",
        "cognitive bias shorts", "logical fallacy shorts",
        "geography facts shorts", "country facts shorts",
        "animal facts shorts", "ocean facts shorts",
        "medical facts shorts", "health myth shorts",
        "educational shorts", "knowledge shorts",
        "facts shorts", "learn shorts", "science shorts",
        "history shorts", "curious facts shorts",
    ],
    "finance": [
        "how to save money shorts", "money saving tips shorts",
        "budgeting tips shorts", "50 30 20 rule shorts",
        "zero based budget shorts", "budget beginner shorts",
        "how to invest shorts", "investing for beginners shorts",
        "stock market basics shorts", "index fund shorts",
        "mutual fund shorts", "sip investment shorts",
        "compound interest shorts", "passive income shorts",
        "multiple income streams shorts", "side hustle shorts",
        "make money online shorts", "freelancing shorts",
        "how to get rich shorts", "wealth building shorts",
        "financial freedom shorts", "fire movement shorts",
        "retire early shorts", "financial independence shorts",
        "credit score shorts", "debt payoff shorts",
        "credit card tips shorts", "loan tips shorts",
        "emergency fund shorts", "financial mistakes shorts",
        "money mistake shorts", "broke to rich shorts",
        "rich habits shorts", "millionaire habits shorts",
        "money mindset shorts", "financial literacy shorts",
        "personal finance shorts", "money tips shorts",
        "finance india shorts", "indian stock market shorts",
        "nifty shorts", "sensex shorts",
        "zerodha shorts", "groww app shorts",
        "nps shorts", "ppf vs fd shorts",
        "tax saving shorts", "income tax shorts",
        "crypto explained shorts", "bitcoin shorts",
        "real estate investing shorts", "rental income shorts",
        "gold investment shorts", "bond investing shorts",
        "dividend investing shorts", "etf shorts",
        "finance shorts", "money shorts", "invest shorts",
    ],
    "motivation": [
        "motivational speech shorts", "discipline shorts",
        "hard work pays off shorts", "no excuses shorts",
        "success mindset shorts", "growth mindset shorts",
        "stoicism shorts", "stoic mindset shorts",
        "david goggins shorts", "goggins motivation shorts",
        "jocko willink shorts", "andrew tate shorts",
        "morning routine successful shorts", "5am club shorts",
        "wake up early shorts", "cold shower shorts",
        "journaling habit shorts", "meditation habit shorts",
        "atomic habits shorts", "habit building shorts",
        "tiny habits shorts", "1 percent better shorts",
        "consistency shorts", "show up every day shorts",
        "stop being lazy shorts", "self improvement shorts",
        "personal development shorts", "self discipline shorts",
        "overcome fear shorts", "comfort zone shorts",
        "rejection motivation shorts", "failure motivation shorts",
        "bounce back shorts", "never give up shorts",
        "believe in yourself shorts", "confidence shorts",
        "mindset shift shorts", "perspective shorts",
        "gratitude shorts", "positive mindset shorts",
        "entrepreneur motivation shorts", "hustle shorts",
        "grind motivation shorts", "work ethic shorts",
        "student motivation shorts", "study motivation shorts",
        "exam motivation shorts", "focus shorts",
        "deep work shorts", "flow state shorts",
        "purpose driven shorts", "find your why shorts",
        "motivation shorts", "inspire shorts",
        "mindset shorts", "success shorts", "life advice shorts",
    ],
}

# ── Setup ─────────────────────────────────────────────────

def load_existing_ids():
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, encoding='utf-8') as f:
        return {row['video_id'] for row in csv.DictReader(f)}

youtube = build('youtube', 'v3', developerKey=API_KEY)

# ── Step 1: Search (or resume from checkpoint) ────────────
CHECKPOINT = Path("data/search_checkpoint.txt")
existing   = load_existing_ids()

if CHECKPOINT.exists():
    with open(CHECKPOINT) as f:
        video_ids = [line.strip() for line in f if line.strip() and line.strip() not in existing]
    print(f"\nResuming from checkpoint: {len(video_ids)} videos to process")
else:
    print(f"\nSearching for {count} {niche} Shorts...")
    queries = NICHE_QUERIES.get(niche, [f"{niche} shorts"])
    all_ids = []

    for q in queries:
        if len(all_ids) >= count:
            break
        print(f"  Query: '{q}'...")
        try:
            results = youtube.search().list(
                part='id,snippet',
                q=q,
                type='video',
                videoDuration='short',
                maxResults=50
            ).execute()
            ids     = [r['id']['videoId'] for r in results['items']]
            new_ids = [i for i in ids if i not in existing and i not in all_ids]
            all_ids.extend(new_ids)
            print(f"    → {len(new_ids)} new (total: {len(all_ids)})")
        except Exception as e:
            if "quotaExceeded" in str(e):
                print(f"  Quota exceeded during search. Continuing with {len(all_ids)} IDs found so far.")
                break
            print(f"    → Failed: {e}")

    video_ids = all_ids[:count]
    print(f"\nNew videos to process: {len(video_ids)}")

if not video_ids:
    print("Nothing new. Quota likely exhausted — resets at 12:30pm IST tomorrow.")
    exit()

# Save checkpoint so quota isn't wasted re-searching if stats call fails
if not CHECKPOINT.exists():
    with open(CHECKPOINT, 'w') as f:
        f.write('\n'.join(video_ids))
    print(f"Checkpoint saved: {CHECKPOINT}")

# ── Step 2: Video stats ───────────────────────────────────
print("\nFetching video stats...")
video_items = []
for i in range(0, len(video_ids), 50):
    batch = video_ids[i:i+50]
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

# ── Step 3: Channel info ──────────────────────────────────
print("Fetching channel info...")
channel_ids = list(set(
    item['snippet']['channelId'] for item in video_items
))
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
            print(f"  Quota hit during channel fetch. Using {len(channel_lookup)} channels so far.")
            break
        print(f"  Channel batch failed: {e}")
        continue

# ── Step 4: Build rows ────────────────────────────────────
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

    subscriber_count     = int(ch_stats.get('subscriberCount', 0))
    channel_video_count  = int(ch_stats.get('videoCount', 0))
    channel_total_views  = int(ch_stats.get('viewCount', 0))
    channel_avg_views    = channel_total_views / max(channel_video_count, 1)
    has_custom_url       = 'customUrl' in ch_snippet

    rows.append({
        "video_id":            vid_id,
        "niche":               niche,
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

# ── Step 5: Save CSV ──────────────────────────────────────
if rows:
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {CSV_PATH}")
else:
    print("No rows to save.")
    exit()

# ── Step 6: Download videos ───────────────────────────────
print("\nDownloading videos...")
success = 0
failed  = 0

for i, row in enumerate(rows, 1):
    vid_id   = row['video_id']
    out_path = Path(f"data/raw_videos/{vid_id}.mp4")

    if out_path.exists():
        print(f"[{i}/{len(rows)}] Already have {vid_id} — skipping")
        success += 1
        continue

    url = f"https://www.youtube.com/watch?v={vid_id}"
    print(f"[{i}/{len(rows)}] Downloading {vid_id} ({row['views']:,} views)...")

    result = subprocess.run([
        "yt-dlp",
        "-o", "data/raw_videos/%(id)s.%(ext)s",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "--quiet", "--no-warnings",
        url
    ], capture_output=True)

    if result.returncode == 0:
        success += 1
        print(f"  OK")
    else:
        failed += 1
        print(f"  Failed")

    time.sleep(random.uniform(3, 6))

# ── Summary ───────────────────────────────────────────────
if CHECKPOINT.exists():
    CHECKPOINT.unlink()

print(f"""
Done!
  Downloaded:   {success}
  Failed:       {failed}
  CSV rows:     {len(rows)}
  Total in CSV: {len(load_existing_ids())}

Quota resets at 12:30pm IST tomorrow.
""")