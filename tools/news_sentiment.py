"""news_sentiment: recent news about a company or topic, scored for sentiment.

Scores follow the VADER "compound" convention: -1 (most negative) to +1 (most positive), with
|score| < 0.05 treated as neutral.

Day 2: deterministic mock articles. Day 5: live handler (Google News RSS and yfinance news, scored
with VADER).
"""

from __future__ import annotations

from datetime import timedelta
from statistics import fmean
from typing import Any

from tools.web_search import _SOURCES, MOCK_TODAY, seeded_rng, slugify

TOOL = "news_sentiment"

NEUTRAL_BAND = 0.05

_HEADLINES = [
    ("{q} beats expectations as demand stays strong", 0.6),
    ("{q} shares slip amid margin concerns", -0.4),
    ("Regulators step up scrutiny of {q}", -0.5),
    ("{q} unveils new products at annual event", 0.4),
    ("{q} faces supply-chain pressure", -0.3),
    ("Analysts raise price targets on {q}", 0.5),
    ("{q} holds steady in volatile market", 0.0),
    ("{q} announces leadership changes", 0.05),
]


def label(score: float) -> str:
    if score >= NEUTRAL_BAND:
        return "positive"
    if score <= -NEUTRAL_BAND:
        return "negative"
    return "neutral"


def aggregate(scores: list[float]) -> dict[str, Any]:
    """Overall sentiment: mean compound score plus the share of positive/negative/neutral articles."""
    if not scores:
        return {"score": 0.0, "label": "neutral", "positive": 0.0, "negative": 0.0, "neutral": 0.0}
    labels = [label(s) for s in scores]
    mean = round(fmean(scores), 4)
    return {
        "score": mean,
        "label": label(mean),
        "positive": round(labels.count("positive") / len(labels), 4),
        "negative": round(labels.count("negative") / len(labels), 4),
        "neutral": round(labels.count("neutral") / len(labels), 4),
    }


def mock_news_sentiment(query: str, num_articles: int = 20, lookback_days: int = 30) -> dict[str, Any]:
    rng = seeded_rng(TOOL, query)
    articles = []
    for i in range(num_articles):
        template, base = _HEADLINES[rng.randrange(len(_HEADLINES))]
        score = round(max(-1.0, min(1.0, base + rng.uniform(-0.15, 0.15))), 4)
        source, domain = _SOURCES[rng.randrange(len(_SOURCES))]
        published = MOCK_TODAY - timedelta(days=rng.randrange(lookback_days))
        title = template.format(q=query)
        articles.append({
            "title": title,
            "url": f"https://mock.{domain}/{slugify(title)}-{i + 1}",
            "source": source,
            "published": published.isoformat(),
            "summary": f"[mock] {title}.",
            "sentiment_score": score,
            "sentiment_label": label(score),
        })
    return {
        "query": query,
        "lookback_days": lookback_days,
        "article_count": len(articles),
        "overall": aggregate([a["sentiment_score"] for a in articles]),
        "articles": articles,
        "notes": ["MOCK DATA - synthetic articles and scores."],
    }
