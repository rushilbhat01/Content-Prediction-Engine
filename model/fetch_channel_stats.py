"""
fetch_channel_stats.py — Fetch subscriber_count, channel_avg_views,
channel_avg_like_rate and channel_avg_comment_rate for a YouTube channel
using yt-dlp (no API key needed).

Returns:
    {
        "subscriber_count":       int,
        "channel_avg_views":      float,   # median of last 25 videos
        "channel_avg_like_rate":  float,   # median(likes/views) across last 25
        "channel_avg_comment_rate": float, # median(comments/views) across last 25
    }

Usage:
    from fetch_channel_stats import fetch_channel_stats
    stats = fetch_channel_stats("https://www.youtube.com/@FitWithAlex")
    stats = fetch_channel_stats("@FitWithAlex")
    stats = fetch_channel_stats("UCxxxxxxxxxxxxxxxx")
"""

import subprocess
import json
import numpy as np


def _normalise_url(channel_input: str) -> str:
    """Accept handle, channel ID, or full URL — return a full channel URL."""
    s = channel_input.strip()
    if s.startswith("http"):
        return s
    if s.startswith("UC") and len(s) == 24:
        return f"https://www.youtube.com/channel/{s}"
    handle = s if s.startswith("@") else f"@{s}"
    return f"https://www.youtube.com/{handle}"


def fetch_channel_stats(channel_input: str, last_n: int = 25) -> dict:
    """
    Fetch channel stats using yt-dlp.

    Args:
        channel_input: Channel URL, handle (@name), or channel ID (UCxxx…)
        last_n:        Number of recent videos to sample for avg_views (default 25)

    Returns:
        Dict with subscriber_count (int) and channel_avg_views (float).
        Returns None on failure.
    """
    url = _normalise_url(channel_input)
    videos_url = url.rstrip("/") + "/videos"

    cmd = [
        "yt-dlp",
        "--dump-single-json",
        "--flat-playlist",
        "--playlist-end", str(last_n),
        "--quiet", "--no-warnings",
        videos_url,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        raise RuntimeError("yt-dlp not found. Install with: pip install yt-dlp")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Channel fetch timed out")

    if not proc.stdout.strip():
        raise RuntimeError(f"yt-dlp returned no output for: {url}")

    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError("Could not parse yt-dlp output as JSON")

    # ── Subscriber count ──────────────────────────────────
    sub_count = data.get("channel_follower_count") or data.get("uploader_follower_count")
    if sub_count is None:
        # Sometimes buried in the first entry
        entries = data.get("entries", [])
        if entries:
            sub_count = (entries[0] or {}).get("channel_follower_count")
    sub_count = int(sub_count) if sub_count else 0

    # ── Average views (median of last N uploads) ──────────
    entries    = data.get("entries", []) or []
    view_counts = []
    for entry in entries:
        if entry and entry.get("view_count") is not None:
            view_counts.append(int(entry["view_count"]))

    if view_counts:
        avg_views = float(np.median(view_counts))
    else:
        avg_views = 0.0

    # ── Engagement rates (like/view, comment/view per video) ─
    like_rates    = []
    comment_rates = []
    for entry in entries:
        if not entry:
            continue
        vc = entry.get("view_count") or 0
        lc = entry.get("like_count") or 0
        cc = entry.get("comment_count") or 0
        if vc > 0:
            like_rates.append(lc / vc)
            comment_rates.append(cc / vc)

    avg_like_rate    = float(np.median(like_rates))    if like_rates    else 0.0
    avg_comment_rate = float(np.median(comment_rates)) if comment_rates else 0.0

    return {
        "subscriber_count":         sub_count,
        "channel_avg_views":        avg_views,
        "channel_avg_like_rate":    avg_like_rate,
        "channel_avg_comment_rate": avg_comment_rate,
    }


# ── CLI test ──────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fetch_channel_stats.py @ChannelHandle")
        sys.exit(1)
    stats = fetch_channel_stats(sys.argv[1])
    print(f"Subscriber count:      {stats['subscriber_count']:,}")
    print(f"Channel avg views:     {stats['channel_avg_views']:,.0f}")
    print(f"Avg like rate:         {stats['channel_avg_like_rate']:.4f}")
    print(f"Avg comment rate:      {stats['channel_avg_comment_rate']:.4f}")
