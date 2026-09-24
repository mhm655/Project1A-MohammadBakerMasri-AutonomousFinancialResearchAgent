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
