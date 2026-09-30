"""Fallback chains per tool (spec section 9.2).

When a tool cannot deliver, the resilient dispatcher walks the chain for that tool: each
:class:`Fallback` names another tool and a function that translates the original arguments into that
tool's arguments. The mapping returns ``None`` when the input the fallback needs is missing, which
skips that step rather than sending an invalid call. Every network tool has at least two fallbacks,
and ``vector_db_search`` (already-verified findings in memory) is the last resort where it makes sense.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_TICKER = re.compile(r"^[A-Za-z][A-Za-z0-9.\-]{0,9}$")


@dataclass
class Fallback:
    tool: str
    map_args: Callable[[dict[str, Any]], dict[str, Any] | None]
    note: str


def _ticker(args: dict[str, Any]) -> str | None:
    value = args.get("ticker")
    if isinstance(value, str) and _TICKER.match(value.strip()):
        return value.strip().upper()
    return None


def _query(args: dict[str, Any], limit: int) -> str | None:
    value = args.get("query")
    if isinstance(value, str) and len(value.strip()) >= 2:
        return value.strip()[:limit]
    return None


def _statement_label(args: dict[str, Any]) -> str:
    return str(args.get("statement_type") or "financial statements").replace("_", " ")


def _vector(query: str, ticker: str | None = None) -> dict[str, Any]:
    mapped: dict[str, Any] = {"query": query[:500], "top_k": 5}
    if ticker:
        mapped["ticker"] = ticker
    return mapped


# --- sec_filing_search ------------------------------------------------------------------------------


def _sec_to_web(args: dict[str, Any]) -> dict[str, Any] | None:
    ticker = _ticker(args)
    if not ticker:
        return None
    parts = [ticker, str(args.get("filing_type") or "10-K"), "SEC filing"]
    if args.get("year"):
        parts.append(str(args["year"]))
    return {"query": " ".join(parts)}


def _sec_to_vector(args: dict[str, Any]) -> dict[str, Any] | None:
    ticker = _ticker(args)
    if not ticker:
        return None
    return _vector(f"{ticker} {args.get('filing_type') or '10-K'} filing", ticker)


# --- financial_data_api / company_profile -----------------------------------------------------------


def _to_sec_10k(args: dict[str, Any]) -> dict[str, Any] | None:
    ticker = _ticker(args)
    return {"ticker": ticker, "filing_type": "10-K"} if ticker else None


def _fin_to_web(args: dict[str, Any]) -> dict[str, Any] | None:
    ticker = _ticker(args)
    return {"query": f"{ticker} {_statement_label(args)} financials"} if ticker else None


def _fin_to_vector(args: dict[str, Any]) -> dict[str, Any] | None:
    ticker = _ticker(args)
    return _vector(f"{ticker} {_statement_label(args)}", ticker) if ticker else None


def _profile_to_web(args: dict[str, Any]) -> dict[str, Any] | None:
    ticker = _ticker(args)
    return {"query": f"{ticker} company profile"} if ticker else None


def _profile_to_vector(args: dict[str, Any]) -> dict[str, Any] | None:
    ticker = _ticker(args)
    return _vector(f"{ticker} company profile business overview", ticker) if ticker else None


# --- earnings_transcript ----------------------------------------------------------------------------


def _earnings_to_sec(args: dict[str, Any]) -> dict[str, Any] | None:
    ticker = _ticker(args)
    return {"ticker": ticker, "filing_type": "8-K"} if ticker else None


def _earnings_to_news(args: dict[str, Any]) -> dict[str, Any] | None:
    ticker = _ticker(args)
    return {"query": f"{ticker} earnings"} if ticker else None


def _earnings_to_vector(args: dict[str, Any]) -> dict[str, Any] | None:
    ticker = _ticker(args)
    if not ticker:
        return None
    period = " ".join(str(args[key]) for key in ("quarter", "year") if args.get(key))
    return _vector(f"{ticker} earnings call {period}".strip(), ticker)


# --- web_search / news_sentiment / vector_db_search -------------------------------------------------


def _web_to_news(args: dict[str, Any]) -> dict[str, Any] | None:
    query = _query(args, 200)
    return {"query": query} if query else None


def _query_to_vector(args: dict[str, Any]) -> dict[str, Any] | None:
    query = _query(args, 500)
    return _vector(query) if query else None


def _query_to_web(args: dict[str, Any]) -> dict[str, Any] | None:
    query = _query(args, 300)
    return {"query": query} if query else None


def _news_to_web(args: dict[str, Any]) -> dict[str, Any] | None:
    query = _query(args, 300)
    if not query:
        return None
    days = args.get("lookback_days")
    if isinstance(days, int) and not isinstance(days, bool):
        date_range = "day" if days <= 1 else "week" if days <= 7 else "month" if days <= 31 else "year"
    else:
        date_range = "month"
    return {"query": query, "date_range": date_range}


# --- peer_comparison --------------------------------------------------------------------------------


def _peers_to_ratios(args: dict[str, Any]) -> dict[str, Any] | None:
    ticker = _ticker(args)
    return {"ticker": ticker, "statement_type": "key_ratios"} if ticker else None


def _peers_to_profile(args: dict[str, Any]) -> dict[str, Any] | None:
    ticker = _ticker(args)
    return {"ticker": ticker} if ticker else None


def _peers_to_vector(args: dict[str, Any]) -> dict[str, Any] | None:
    ticker = _ticker(args)
    return _vector(f"{ticker} peer comparison competitors", ticker) if ticker else None


FALLBACK_CHAINS: dict[str, list[Fallback]] = {
    "sec_filing_search": [
        Fallback("web_search", _sec_to_web, "web results about the filing instead of the filing itself"),
        Fallback("vector_db_search", _sec_to_vector, "previously stored findings from this filing"),
    ],
    "financial_data_api": [
        Fallback("sec_filing_search", _to_sec_10k, "the latest 10-K, which contains the statements"),
        Fallback("web_search", _fin_to_web, "web-reported figures, lower reliability than filings"),
        Fallback("vector_db_search", _fin_to_vector, "previously stored financial findings"),
    ],
    "company_profile": [
        Fallback("sec_filing_search", _to_sec_10k, "the 10-K business description"),
        Fallback("web_search", _profile_to_web, "web-reported company overview"),
        Fallback("vector_db_search", _profile_to_vector, "previously stored company findings"),
    ],
    "earnings_transcript": [
        Fallback("sec_filing_search", _earnings_to_sec, "8-K earnings releases instead of the call transcript"),
        Fallback("news_sentiment", _earnings_to_news, "news coverage of the earnings"),
        Fallback("vector_db_search", _earnings_to_vector, "previously stored earnings findings"),
    ],
    "web_search": [
        Fallback("news_sentiment", _web_to_news, "news articles matching the query"),
        Fallback("vector_db_search", _query_to_vector, "previously stored findings matching the query"),
    ],
    "news_sentiment": [
        Fallback("web_search", _news_to_web, "web results from the same period"),
        Fallback("vector_db_search", _query_to_vector, "previously stored news findings"),
    ],
    "peer_comparison": [
        Fallback("financial_data_api", _peers_to_ratios, "the company's own key ratios, without peers"),
        Fallback("company_profile", _peers_to_profile, "the company profile, without peers"),
        Fallback("vector_db_search", _peers_to_vector, "previously stored peer findings"),
    ],
    "vector_db_search": [
        Fallback("web_search", _query_to_web, "live web results instead of stored findings"),
    ],
}


def chain_for(tool: str) -> list[Fallback]:
    """The ordered fallbacks for ``tool`` (a copy; empty for tools with none)."""
    return list(FALLBACK_CHAINS.get(tool, []))
