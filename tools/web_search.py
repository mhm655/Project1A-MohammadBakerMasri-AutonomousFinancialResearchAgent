"""web_search: general web search for news, analysis and commentary.

Day 2: deterministic mock results (same query -> same results, so tests and traces are reproducible).
Day 5: live handler (DuckDuckGo via ``ddgs``, falling back to Google News RSS).
"""

from __future__ import annotations

import random
import re
import zlib
from datetime import date, timedelta
from typing import Any

TOOL = "web_search"

#: Fixed "today" for mock results so they are reproducible.
MOCK_TODAY = date(2025, 6, 30)

DATE_RANGE_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

_SOURCES = [
    ("Reuters", "reuters.com"),
    ("Bloomberg", "bloomberg.com"),
    ("CNBC", "cnbc.com"),
    ("The Wall Street Journal", "wsj.com"),
    ("Financial Times", "ft.com"),
    ("MarketWatch", "marketwatch.com"),
    ("Yahoo Finance", "finance.yahoo.com"),
    ("Seeking Alpha", "seekingalpha.com"),
]

_TITLES = [
    "{q}: what investors need to know",
    "Analysts weigh in on {q}",
    "{q} - latest developments and outlook",
    "Five takeaways on {q}",
    "Why {q} matters for the market",
    "{q}: risks and opportunities ahead",
]


def seeded_rng(*parts: object) -> random.Random:
    """A random generator seeded from the arguments, so mock output is deterministic."""
    return random.Random(zlib.crc32("|".join(map(str, parts)).encode()))


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60] or "result"


def mock_web_search(query: str, num_results: int = 10, date_range: str | None = None) -> dict[str, Any]:
    rng = seeded_rng(TOOL, query)
    max_age = DATE_RANGE_DAYS.get(date_range or "", 365)
    results = []
    for i in range(num_results):
        source, domain = _SOURCES[rng.randrange(len(_SOURCES))]
        title = _TITLES[i % len(_TITLES)].format(q=query)
        published = MOCK_TODAY - timedelta(days=rng.randrange(max_age))
        results.append({
            "title": title,
            "url": f"https://mock.{domain}/{slugify(query)}-{i + 1}",
            "snippet": f"[mock] {source} coverage of {query}, published {published.isoformat()}.",
            "source": source,
            "published": published.isoformat(),
        })
    return {"query": query, "date_range": date_range, "results": results, "notes": ["MOCK DATA - synthetic search results."]}


# --------------------------------------------------------------------------------------------------
# Live handler
# --------------------------------------------------------------------------------------------------

#: Search back ends tried in order. Verified from this network: Brave and Bing answer; DuckDuckGo's
#: own endpoint was unreachable, so it is kept last among the ddgs back ends.
SEARCH_BACKENDS = ("brave", "bing", "duckduckgo")
_TIMELIMIT = {"day": "d", "week": "w", "month": "m", "year": "y"}


def live_web_search(query: str, num_results: int = 10, date_range: str | None = None) -> dict[str, Any]:
    from urllib.parse import urlparse

    from tools.base import ToolExecutionError

    errors = []
    for backend in SEARCH_BACKENDS:
        try:
            from ddgs import DDGS

            hits = DDGS(timeout=20).text(query, max_results=num_results, backend=backend,
                                         timelimit=_TIMELIMIT.get(date_range or ""))
        except Exception as exc:  # ddgs raises its own exception types for timeouts and empty results
            errors.append(f"{backend}: {type(exc).__name__}")
            continue
        results = [{
            "title": h.get("title", ""),
            "url": h.get("href", ""),
            "snippet": h.get("body", ""),
            "source": urlparse(h.get("href", "")).hostname or "",
            "published": None,
        } for h in hits if h.get("href")]
        if results:
            return {"query": query, "date_range": date_range, "results": results[:num_results],
                    "notes": [f"Source: web search ({backend})."] + ([f"Unavailable: {', '.join(errors)}."] if errors else [])}
    # Last resort: news headlines from Google News RSS.
    from tools.news_sentiment import google_news

    try:
        articles = google_news(query, lookback_days={"day": 1, "week": 7, "month": 30}.get(date_range or "", 365),
                               limit=num_results)
    except ToolExecutionError as exc:
        raise ToolExecutionError(f"{TOOL}: every search back end failed ({', '.join(errors)}; Google News: {exc})",
                                 tool=TOOL, retryable=True) from exc
    return {"query": query, "date_range": date_range,
            "results": [{"title": a["title"], "url": a["url"], "snippet": a["title"], "source": a["source"],
                         "published": a["published"]} for a in articles],
            "notes": [f"Web search unavailable ({', '.join(errors)}); results are Google News headlines."]}
