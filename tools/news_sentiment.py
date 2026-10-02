"""news_sentiment: recent news about a company or topic, scored for sentiment.

Scores follow the VADER "compound" convention: -1 (most negative) to +1 (most positive), with
|score| < 0.05 treated as neutral.

Day 2: deterministic mock articles. Day 5: live handler (Google News RSS and yfinance news, scored
with VADER).
"""

from __future__ import annotations

from datetime import UTC, timedelta
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
    """Classify a compound sentiment score as positive, negative or neutral."""
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
    """Deterministic mock news articles with sentiment scores for ``query``."""
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


# --------------------------------------------------------------------------------------------------
# Live handler: Google News RSS + finance-tuned VADER
# --------------------------------------------------------------------------------------------------

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"

#: VADER is a general-purpose sentiment model: out of the box it scores "Microsoft beats expectations
#: as cloud demand soars" as negative (it reads "beats" as violence). These finance terms correct that.
#: Scores use VADER's scale (-4 to +4).
FINANCE_LEXICON = {
    "beat": 2.0, "beats": 2.0, "outperform": 2.0, "outperforms": 2.0, "outperformed": 2.0,
    "upgrade": 1.8, "upgrades": 1.8, "upgraded": 1.8, "downgrade": -1.8, "downgrades": -1.8, "downgraded": -1.8,
    "miss": -1.8, "misses": -1.8, "missed": -1.8, "soar": 2.2, "soars": 2.2, "soared": 2.2,
    "surge": 2.0, "surges": 2.0, "surged": 2.0, "jump": 1.4, "jumps": 1.4, "jumped": 1.4,
    "rally": 1.8, "rallies": 1.8, "rallied": 1.8, "record": 1.2, "bullish": 2.0, "bearish": -2.0,
    "plunge": -2.5, "plunges": -2.5, "plunged": -2.5, "tumble": -2.2, "tumbles": -2.2, "tumbled": -2.2,
    "slump": -2.0, "slumps": -2.0, "slumped": -2.0, "sink": -1.8, "sinks": -1.8, "slides": -1.5,
    "layoffs": -1.8, "lawsuit": -1.5, "probe": -1.5, "investigation": -1.5, "antitrust": -1.2,
    "recall": -1.5, "recalls": -1.5, "bankruptcy": -3.0, "default": -2.0, "fraud": -3.0,
    "cuts": -1.2, "slashes": -1.8, "raises": 1.0, "strong": 1.5, "weak": -1.5, "headwinds": -1.2, "tailwinds": 1.2,
}

_analyzer = None


def finance_sentiment(text: str) -> float:
    """VADER compound score (-1..+1) with the finance vocabulary above."""
    global _analyzer
    if _analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(FINANCE_LEXICON)
    return round(_analyzer.polarity_scores(text)["compound"], 4)


def google_news(query: str, *, lookback_days: int, limit: int) -> list[dict[str, Any]]:
    """Recent headlines from Google News RSS (free, no key), newest first."""
    from datetime import datetime
    from email.utils import parsedate_to_datetime

    import feedparser

    from tools.http import get_http

    body = get_http().get(GOOGLE_NEWS_RSS, params={"q": f"{query} when:{lookback_days}d", "hl": "en-US",
                                                    "gl": "US", "ceid": "US:en"}, ttl_s=1800)
    feed = feedparser.parse(body)
    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    articles = []
    for entry in feed.entries:
        try:
            published = parsedate_to_datetime(entry.published)
        except (AttributeError, TypeError, ValueError):
            published = None
        if published and published.tzinfo is None:
            # "-0000" or no zone at all parses to a naive datetime; read it as UTC so it can be compared.
            published = published.replace(tzinfo=UTC)
        if published and published < cutoff:
            continue
        source = entry.get("source", {}).get("title", "") if hasattr(entry, "get") else ""
        title = entry.title
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]  # Google appends " - Publisher" to every headline
        articles.append({"title": title, "url": entry.link, "source": source,
                         "published": published.date().isoformat() if published else None})
    articles.sort(key=lambda a: a["published"] or "", reverse=True)
    return articles[:limit]


def live_news_sentiment(query: str, num_articles: int = 20, lookback_days: int = 30) -> dict[str, Any]:
    """Recent Google News headlines for ``query``, scored with VADER plus a finance vocabulary."""
    articles = google_news(query, lookback_days=lookback_days, limit=num_articles)
    for article in articles:
        article["summary"] = article["title"]
        article["sentiment_score"] = finance_sentiment(article["title"])
        article["sentiment_label"] = label(article["sentiment_score"])
    notes = ["Source: Google News RSS; headlines scored with VADER plus a finance vocabulary.",
             "Headline sentiment is a rough signal of the news narrative, not of the underlying facts."]
    if not articles:
        notes.append(f"No articles found in the last {lookback_days} days.")
    return {"query": query, "lookback_days": lookback_days, "article_count": len(articles),
            "overall": aggregate([a["sentiment_score"] for a in articles]), "articles": articles, "notes": notes}
