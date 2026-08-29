"""
Finance Topic Research Tool
============================
Combines three legitimate, publicly-available signals to help rank which
finance/economics video topics are worth making, and to spot when a
competitor's video is breaking out (early signal a topic is having a moment).

Signals used (no scraping of private/proprietary data, no ToS violations):
  1. Google Trends interest, filtered to a country (via pytrends)         -> demand
  2. YouTube search results for the topic (via yt-dlp, no API key/quota) -> saturation
  3. Competitor channel upload history (via yt-dlp)                      -> breakout detection

None of this can tell you real CPM/RPM (that data is private to YouTube/advertisers
and not exposed by any API, official or otherwise). What it gives you is a
consistent, repeatable way to compare topics against each other using the same
proxies serious researchers use, instead of eyeballing it once.

Usage:
    pip install -r requirements.txt
    python finance_topic_research.py --config config.json

Outputs:
    topic_scores.csv        -- ranked topic opportunity scores
    competitor_breakouts.csv -- recent videos performing well above a channel's norm
"""

import argparse
import json
import time
import sys
from datetime import datetime, timezone
from statistics import median

import pandas as pd

# --------------------------------------------------------------------------
# Compatibility patch: pytrends 4.9.2 hardcodes the old urllib3 Retry
# argument name `method_whitelist` internally. urllib3 >= 2.0 renamed that
# to `allowed_methods` and removed the old name entirely, which makes every
# pytrends call raise "Retry.__init__() got an unexpected keyword argument
# 'method_whitelist'" regardless of what arguments *we* pass to TrendReq.
# This isn't fixable from the calling code otherwise, so we patch
# urllib3's Retry.__init__ to accept the old name as an alias before
# pytrends ever constructs one. Safe to remove once pytrends ships a fix.
# --------------------------------------------------------------------------
try:
    from urllib3.util.retry import Retry as _Retry
    if "method_whitelist" not in _Retry.__init__.__code__.co_varnames:
        _orig_retry_init = _Retry.__init__

        def _patched_retry_init(self, *args, **kwargs):
            if "method_whitelist" in kwargs:
                kwargs["allowed_methods"] = kwargs.pop("method_whitelist")
            _orig_retry_init(self, *args, **kwargs)

        _Retry.__init__ = _patched_retry_init
except ImportError:
    pass

try:
    from pytrends.request import TrendReq
except ImportError:
    TrendReq = None

try:
    import yt_dlp
except ImportError:
    yt_dlp = None


# --------------------------------------------------------------------------
# 1. Google Trends demand score
# --------------------------------------------------------------------------

def get_trends_scores(topics, geo="US", batch_size=5, pause=20.0, max_retries=4):
    """
    Returns {topic: avg_interest_last_90_days} using Google Trends,
    filtered to the given country. pytrends only allows 5 keywords per
    request, so topics are batched and each batch is scored on the same
    0-100 relative scale by including a fixed anchor keyword across batches
    so scores are comparable to each other.

    Google Trends is unofficial and rate-limits aggressively (HTTP 429) if
    you hit it repeatedly in a short window -- this is a per-IP throttle on
    Google's side, not a bug in this script. To work around it, each batch
    is retried with exponential backoff, and there's a long pause between
    batches by default. If you still get 429s, wait a few minutes and rerun
    with --skip-trends off, or run with fewer topics per call.
    """
    if TrendReq is None:
        raise RuntimeError("pytrends is not installed. Run: pip install pytrends")

    pytrends = TrendReq(hl="en-US", tz=360)
    anchor = topics[0]  # first topic doubles as the cross-batch anchor
    scores = {}

    batches = [topics[i:i + batch_size] for i in range(0, len(topics), batch_size)]

    for bi, batch in enumerate(batches, 1):
        kw_list = list(dict.fromkeys([anchor] + batch))[:5]  # dedupe, cap at 5
        attempt = 0
        while True:
            attempt += 1
            try:
                pytrends.build_payload(kw_list, timeframe="today 3-m", geo=geo)
                df = pytrends.interest_over_time()
                if df.empty:
                    for t in batch:
                        scores.setdefault(t, 0)
                else:
                    for t in batch:
                        scores[t] = round(df[t].mean(), 1) if t in df.columns else 0
                break
            except Exception as e:
                is_429 = "429" in str(e)
                if attempt >= max_retries:
                    print(f"  [trends] batch {bi} failed after {attempt} attempts ({e}); scoring as 0. "
                          f"If this keeps happening, Google is rate-limiting your IP -- wait a few "
                          f"minutes and rerun, or use --skip-trends for now.", file=sys.stderr)
                    for t in batch:
                        scores.setdefault(t, 0)
                    break
                wait = (30 if is_429 else 5) * attempt
                print(f"  [trends] batch {bi} attempt {attempt} failed ({e}); "
                      f"waiting {wait}s before retry...", file=sys.stderr)
                time.sleep(wait)
                continue

        if bi < len(batches):
            time.sleep(pause)  # long pause between batches -- this is what actually avoids 429s

    return scores


# --------------------------------------------------------------------------
# 2. YouTube search saturation (via yt-dlp, no API key or quota needed)
# --------------------------------------------------------------------------

def _days_since(upload_date_str):
    """upload_date_str is 'YYYYMMDD' as returned by yt-dlp."""
    if not upload_date_str:
        return None
    try:
        d = datetime.strptime(upload_date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        return max((datetime.now(timezone.utc) - d).days, 1)
    except ValueError:
        return None


def youtube_search_saturation(topic, max_results=15):
    """
    Pulls the top search results for a topic and computes:
      - avg_views: how big the existing videos on this topic already are
      - avg_views_per_day: normalizes for video age (a 3-year-old video with
        1M views is less impressive than a 2-week-old video with 200k views)
      - big_channel_count: how many results already have >250k views
        (rough proxy for "dominated by large channels already")
    Higher saturation = harder to break into with a new channel.
    """
    if yt_dlp is None:
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,  # need real view counts/upload dates, not just IDs
        "playlistend": max_results,
        # 'android' client returns view_count/upload_date/duration without
        # needing a JS runtime -- avoids the "no supported JS runtime" /
        # SABR warnings, which are just noise for our purposes since we
        # never need actual playable formats.
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }

    query = f"ytsearch{max_results}:{topic}"
    views, views_per_day, big_channel_count, durations = [], [], 0, []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(query, download=False)
        except Exception as e:
            print(f"  [yt-dlp] warning: search failed for '{topic}' ({e})", file=sys.stderr)
            return {"avg_views": 0, "avg_views_per_day": 0, "big_channel_count": 0,
                    "pct_over_8min": 0, "n_results": 0}

        entries = info.get("entries", []) if info else []
        for e in entries:
            if not e:
                continue
            v = e.get("view_count") or 0
            views.append(v)
            days = _days_since(e.get("upload_date"))
            if days:
                views_per_day.append(v / days)
            if v > 250_000:
                big_channel_count += 1
            dur = e.get("duration") or 0
            durations.append(dur)

    n = len(views) or 1
    over_8min = sum(1 for d in durations if d and d >= 480)

    return {
        "avg_views": round(sum(views) / n),
        "avg_views_per_day": round(sum(views_per_day) / len(views_per_day), 1) if views_per_day else 0,
        "big_channel_count": big_channel_count,
        "pct_over_8min": round(100 * over_8min / n, 1),
        "n_results": len(views),
    }


# --------------------------------------------------------------------------
# 3. Competitor breakout detection
# --------------------------------------------------------------------------

def competitor_breakouts(channel_url, lookback_days=60, breakout_multiplier=2.0, max_videos=50):
    """
    Pulls a competitor channel's recent uploads and flags videos whose
    views-per-day is well above that channel's own rolling median -- an
    early signal that a specific topic/format is resonating right now,
    independent of the channel's overall size.
    """
    if yt_dlp is None:
        raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
        "playlistend": max_videos,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }

    results = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(channel_url, download=False)
        except Exception as e:
            print(f"  [yt-dlp] warning: channel fetch failed for '{channel_url}' ({e})", file=sys.stderr)
            return results

        entries = info.get("entries", []) if info else []
        channel_name = info.get("channel") or info.get("uploader") or channel_url

        vids = []
        for e in entries:
            if not e:
                continue
            days = _days_since(e.get("upload_date"))
            v = e.get("view_count") or 0
            if days is None:
                continue
            vids.append({
                "title": e.get("title"),
                "url": e.get("webpage_url") or e.get("url"),
                "views": v,
                "days_since_upload": days,
                "views_per_day": round(v / days, 1),
            })

        if not vids:
            return results

        vpd_values = [x["views_per_day"] for x in vids]
        channel_median_vpd = median(vpd_values) if vpd_values else 0

        for x in vids:
            if x["days_since_upload"] <= lookback_days and channel_median_vpd > 0:
                if x["views_per_day"] >= breakout_multiplier * channel_median_vpd:
                    results.append({
                        "channel": channel_name,
                        "title": x["title"],
                        "url": x["url"],
                        "views": x["views"],
                        "days_since_upload": x["days_since_upload"],
                        "views_per_day": x["views_per_day"],
                        "channel_median_views_per_day": round(channel_median_vpd, 1),
                        "multiplier_vs_channel_norm": round(x["views_per_day"] / channel_median_vpd, 2),
                    })

    return results


# --------------------------------------------------------------------------
# Keyword Planner CSV loader (primary demand signal -- official, free, no
# rate limits, real advertiser bid data instead of a fuzzy interest score)
# --------------------------------------------------------------------------

def load_keyword_planner_csv(path):
    """
    Loads a 'Keyword Ideas' CSV exported from Google Ads Keyword Planner.

    How to get this file (~5 minutes, free, no API approval needed):
      1. Go to ads.google.com -> Tools & Settings -> Planning -> Keyword Planner
         (works with a free Google Ads account, no active campaign required)
      2. Click "Discover new keywords"
      3. Paste in your topic list (one per line or comma-separated)
      4. Set the target location to United States (or your target country)
      5. On the results page, click the download icon -> "Download keyword ideas"
         -> choose CSV
      6. Point this script at that file via config.json's "keyword_planner_csv"

    Google's export has a couple of title rows before the real header row,
    so this scans for the row starting with "Keyword" rather than assuming
    row 0. Returns a dict keyed by lowercased keyword ->
    {avg_monthly_searches, bid_low, bid_high}.
    """
    with open(path, encoding="utf-8-sig") as f:
        lines = f.readlines()

    header_idx = 0
    for i, line in enumerate(lines[:6]):
        first_field = line.split(",", 1)[0].strip().strip('"').lower()
        if first_field == "keyword":
            header_idx = i
            break

    df = pd.read_csv(path, skiprows=header_idx, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]

    def find_col(*must_contain):
        for c in df.columns:
            if all(k in c for k in must_contain):
                return c
        return None

    kw_col = find_col("keyword")
    vol_col = find_col("avg", "monthly", "search") or find_col("searches")
    bid_low_col = find_col("bid", "low")
    bid_high_col = find_col("bid", "high")

    if kw_col is None:
        raise ValueError(
            f"Couldn't find a 'Keyword' column in {path}. "
            f"Columns found: {list(df.columns)}. Make sure this is the raw "
            f"'Download keyword ideas' CSV export, not a reformatted copy."
        )

    def to_num(x):
        if pd.isna(x):
            return None
        s = str(x).replace("$", "").replace(",", "").strip()
        if s in ("", "-", "N/A"):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    result = {}
    for _, row in df.iterrows():
        kw = str(row[kw_col]).strip().lower()
        if not kw or kw == "nan":
            continue
        result[kw] = {
            "avg_monthly_searches": to_num(row[vol_col]) if vol_col else None,
            "bid_low": to_num(row[bid_low_col]) if bid_low_col else None,
            "bid_high": to_num(row[bid_high_col]) if bid_high_col else None,
        }
    return result


def match_keyword_planner_row(topic, kp_data):
    """
    Exact match first, then substring match in both directions (Keyword
    Planner often suggests close variants rather than your exact phrase).
    """
    t = topic.strip().lower()
    if t in kp_data:
        return kp_data[t]
    for kw, row in kp_data.items():
        if t in kw or kw in t:
            return row
    return None


# --------------------------------------------------------------------------
# Scoring / orchestration
# --------------------------------------------------------------------------

def normalize(values):
    """Min-max normalize a list of numbers to 0-100. Flat lists map to 50."""
    lo, hi = min(values), max(values)
    if hi == lo:
        return [50.0 for _ in values]
    return [round(100 * (v - lo) / (hi - lo), 1) for v in values]


def build_topic_scores(config, keyword_planner_csv=None, use_trends=True):
    topics = config["topics"]
    geo = config.get("trends_geo", "US")
    max_results = config.get("youtube_search_results_per_topic", 15)

    kp_data = None
    if keyword_planner_csv:
        print(f"Loading Keyword Planner data from {keyword_planner_csv}...")
        kp_data = load_keyword_planner_csv(keyword_planner_csv)
        print(f"  loaded {len(kp_data)} keyword rows")

    trends = {}
    if use_trends and kp_data is None:
        # Only bother with the flaky Trends call if we don't have the more
        # reliable Keyword Planner data -- no point risking a 429 for a
        # weaker signal when a stronger one is already available.
        print(f"Pulling Google Trends interest (geo={geo}) for {len(topics)} topics...")
        trends = get_trends_scores(topics, geo=geo)
    elif use_trends and kp_data is not None:
        print("Keyword Planner data provided -- skipping Google Trends "
              "(bid data is the stronger, more reliable signal).")

    rows = []
    for i, topic in enumerate(topics, 1):
        print(f"  [{i}/{len(topics)}] YouTube saturation check: {topic}")
        sat = youtube_search_saturation(topic, max_results=max_results)

        kp_row = match_keyword_planner_row(topic, kp_data) if kp_data else None

        rows.append({
            "topic": topic,
            "trends_interest": trends.get(topic, 0),
            "kp_avg_monthly_searches": kp_row["avg_monthly_searches"] if kp_row else None,
            "kp_bid_low": kp_row["bid_low"] if kp_row else None,
            "kp_bid_high": kp_row["bid_high"] if kp_row else None,
            "avg_views_existing": sat["avg_views"],
            "avg_views_per_day_existing": sat["avg_views_per_day"],
            "big_channel_count": sat["big_channel_count"],
            "pct_results_over_8min": sat["pct_over_8min"],
        })
        time.sleep(0.5)  # be polite to YouTube between searches

    df = pd.DataFrame(rows)

    if kp_data is not None:
        # Primary demand signal: real advertiser bid $ (what this niche is
        # actually worth to buyers) blended with real search volume (is
        # anyone looking). Missing rows fall back to 0 rather than crashing.
        bid_mid = df[["kp_bid_low", "kp_bid_high"]].mean(axis=1).fillna(0)
        volume = df["kp_avg_monthly_searches"].fillna(0)
        df["demand_score"] = [
            round(0.65 * b + 0.35 * v, 1)
            for b, v in zip(normalize(bid_mid.tolist()), normalize(volume.tolist()))
        ]
    else:
        # Fallback: Trends interest (best-effort, may be all zeros if
        # rate-limited -- in that case opportunity_score still works, it
        # just leans entirely on saturation/dominance until you add a
        # Keyword Planner export).
        df["demand_score"] = normalize(df["trends_interest"].tolist())

    df["saturation_score"] = normalize(df["avg_views_per_day_existing"].tolist())
    df["dominance_penalty"] = normalize(df["big_channel_count"].tolist())

    df["opportunity_score"] = (
        0.5 * df["demand_score"]
        - 0.3 * df["saturation_score"]
        - 0.2 * df["dominance_penalty"]
    )
    # Rescale opportunity_score to a clean 0-100 for readability
    df["opportunity_score"] = normalize(df["opportunity_score"].tolist())

    df = df.sort_values("opportunity_score", ascending=False).reset_index(drop=True)
    return df


def build_breakout_report(config):
    channels = config.get("competitor_channels", [])
    lookback = config.get("breakout_lookback_days", 60)
    mult = config.get("breakout_multiplier", 2.0)

    all_results = []
    for i, ch in enumerate(channels, 1):
        print(f"  [{i}/{len(channels)}] Scanning competitor channel: {ch}")
        all_results.extend(competitor_breakouts(ch, lookback_days=lookback, breakout_multiplier=mult))
        time.sleep(0.5)

    if not all_results:
        return pd.DataFrame(columns=[
            "channel", "title", "url", "views", "days_since_upload",
            "views_per_day", "channel_median_views_per_day", "multiplier_vs_channel_norm"
        ])

    df = pd.DataFrame(all_results)
    return df.sort_values("multiplier_vs_channel_norm", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Finance/economics YouTube topic research tool")
    parser.add_argument("--config", default="config.json", help="Path to config JSON")
    parser.add_argument("--topics-out", default="topic_scores.csv", help="Output CSV for topic ranking")
    parser.add_argument("--breakouts-out", default="competitor_breakouts.csv", help="Output CSV for breakout videos")
    parser.add_argument("--keyword-planner-csv", default=None,
                         help="Path to a Keyword Planner 'Download keyword ideas' CSV export. "
                              "Recommended -- real bid data, no rate limits, replaces Google Trends "
                              "as the demand signal when provided. See README for how to export it.")
    parser.add_argument("--skip-trends", action="store_true",
                         help="Don't fall back to Google Trends when no Keyword Planner CSV is given "
                              "(useful if you're being rate-limited and want saturation-only scoring now)")
    parser.add_argument("--skip-breakouts", action="store_true", help="Skip competitor breakout scan")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    kp_csv = args.keyword_planner_csv or config.get("keyword_planner_csv")

    print("\n=== Building topic opportunity scores ===")
    topic_df = build_topic_scores(config, keyword_planner_csv=kp_csv, use_trends=not args.skip_trends)
    topic_df.to_csv(args.topics_out, index=False)
    print(f"\nSaved {len(topic_df)} ranked topics -> {args.topics_out}")
    display_cols = ["topic", "opportunity_score", "demand_score", "big_channel_count"]
    if kp_csv:
        display_cols = ["topic", "opportunity_score", "kp_bid_low", "kp_bid_high",
                         "kp_avg_monthly_searches", "big_channel_count"]
    print(topic_df[display_cols].to_string(index=False))

    if not args.skip_breakouts:
        print("\n=== Scanning competitors for breakout videos ===")
        breakout_df = build_breakout_report(config)
        breakout_df.to_csv(args.breakouts_out, index=False)
        print(f"\nSaved {len(breakout_df)} breakout videos -> {args.breakouts_out}")
        if not breakout_df.empty:
            print(breakout_df[["channel", "title", "multiplier_vs_channel_norm"]].to_string(index=False))
    else:
        print("Skipping competitor breakout scan (--skip-breakouts).")


if __name__ == "__main__":
    main()
