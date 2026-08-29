# Finance Topic Research Tool

A data-driven way to shortlist which finance/economics video topics to make,
using signals that are free, legitimate, and publicly available. It does
**not** and cannot give you real CPM/RPM numbers — no API, official or
unofficial, exposes that data for anyone but your own channel. What it does
give you: a repeatable way to compare topics against each other instead of
guessing from a blog post once.

## What it measures

| Signal | Source | What it tells you |
|---|---|---|
| Advertiser demand | Keyword Planner CSV export (recommended) or Google Trends (automatic fallback) | Real advertiser bid $ and search volume — or, if you skip the export, a rough interest score |
| Saturation | YouTube search results (`yt-dlp`, no API key/quota needed) | How big and how recent the existing videos already are |
| Competitor breakouts | Competitor channel upload history (`yt-dlp`) | Which recent videos are outperforming a channel's own normal pace |

These combine into an **opportunity score**: topics with strong advertiser
demand that aren't already dominated by large, recent uploads score highest.

### Why Keyword Planner instead of Google Trends

Google Trends is unofficial and rate-limits hard and unpredictably (HTTP
429), especially since 2025 — it can fail regardless of how much you slow
the script down. It's also a weaker signal to begin with: "search interest"
isn't the same as "advertisers will pay real money for this." **Keyword
Planner's CSV export is the recommended path** — official, free, no rate
limit, and it hands you the actual bid range advertisers are paying per
click, which is a direct read on how much a topic is worth, not a proxy for
one. Trends is kept only as an automatic fallback if you don't provide a
Keyword Planner CSV.

### Getting the Keyword Planner CSV (~5 minutes, one-time per batch of topics)

1. Go to [ads.google.com](https://ads.google.com) → Tools & Settings →
   Planning → Keyword Planner (a free Google Ads account works, no active
   campaign or spend required)
2. Click "Discover new keywords"
3. Paste in your topic list from `config.json`
4. Set the target location to United States (or whichever country you're
   targeting)
5. On the results page, click the download icon → "Download keyword
   ideas" → CSV
6. Point the script at it — either flag or config, both work:
   ```bash
   python finance_topic_research.py --keyword-planner-csv export.csv
   ```
   or add `"keyword_planner_csv": "export.csv"` to `config.json`

## Setup

```bash
pip install -r requirements.txt
```

`yt-dlp` reads public YouTube data without hitting the quota-limited
official API. `pytrends` is only needed if you're not using the Keyword
Planner CSV.

## Configure

Edit `config.json`:
- `topics`: the finance sub-topics you want scored (be specific — "Roth IRA
  vs 401k" beats "personal finance")
- `competitor_channels`: US finance channels to benchmark against, as
  `.../videos` URLs
- `keyword_planner_csv`: path to your export (optional — can also pass via
  `--keyword-planner-csv`)
- `trends_geo`: `"US"` or `"GB"` etc. — only used if no Keyword Planner CSV
  is given
- `breakout_lookback_days` / `breakout_multiplier`: how recent and how far
  above a channel's own median a video needs to be to count as a "breakout"

## Run

```bash
# Recommended -- real bid data, no rate limits
python finance_topic_research.py --keyword-planner-csv export.csv

# Fallback -- uses Google Trends, may hit 429s
python finance_topic_research.py
```

Produces two files:
- `topic_scores.csv` — your topics ranked by opportunity score, including
  the raw bid range and search volume when Keyword Planner data is used
- `competitor_breakouts.csv` — recent competitor videos outperforming their
  own channel norm

Useful flags:
- `--keyword-planner-csv PATH` — use real bid data instead of Trends (see
  above)
- `--skip-trends` — don't even attempt Google Trends when no Keyword
  Planner CSV is given; scoring falls back to saturation/dominance only
- `--skip-breakouts` — skip the competitor scan and just re-run topic
  scoring

## Honest limitations

- **No RPM/CPM data anywhere in here.** That number is private to
  YouTube and advertisers. Nothing — no library, no paid tool, no scraper —
  legitimately exposes it for videos that aren't yours. Keyword Planner
  bids are the closest legitimate proxy that exists: real advertiser money,
  just for Search rather than YouTube specifically.
- **Google Trends will 429 you unpredictably** if you don't use the
  Keyword Planner path — this is Google's own throttling, not fixable from
  the script side. Use `--keyword-planner-csv` to sidestep it entirely.
- **`yt-dlp` search results are YouTube's ranking, not a random sample** —
  they're already influenced by what's currently popular, which is exactly
  the saturation signal you want, but it means very new or small-audience
  topics may look emptier than they really are.
- **This is a proxy, not ground truth.** The real signal only exists once
  you publish and check your own YouTube Analytics/Reporting API for actual
  RPM by topic and by country. Treat this tool as narrowing 50 ideas down to
  your best 5-10 — not as a guarantee.

## Closing the loop

Once you have videos live, the next step is pulling your own Analytics data
(RPM, CPM, and country breakdown per video via the YouTube Analytics API)
and merging it back against what this tool predicted for that topic. After
15-20 videos you'll have a model calibrated on your *actual* results, which
is worth more than any of these proxies alone. Happy to build that piece
next once you have data to feed it.
