"""Query analysis: turn a free-text request into a profile the planner and disambiguator can act on
(architecture spec §15.1).

Why a separate module: the first version of ``analyze_query`` lived in ``agent/core.py`` and returned
only a template and entities. The disambiguation policy (§15.2) needs more: how ambiguous the request
is and *why*, which periods it names, whether it is about a sector or about memory of earlier work.
The template, ticker and name logic below is copied unchanged from core so existing callers keep
identical behaviour; the additions are computed on top of it.

Everything here is deterministic (rules and entity matching, no network, no LLM). The spec allows an
LLM tiebreaker for borderline classifications; that is deliberately out of scope for now, because a
rule-based score is reproducible and costs no free-tier quota.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------------------
# Rules copied from agent/core.py (kept identical so both analysers agree)
# --------------------------------------------------------------------------------------------------

KNOWN_COMPANIES = {
    "microsoft": "MSFT", "apple": "AAPL", "tesla": "TSLA", "nvidia": "NVDA", "amazon": "AMZN",
    "alphabet": "GOOGL", "google": "GOOGL", "palantir": "PLTR", "meta platforms": "META",
    "jpmorgan": "JPM", "goldman sachs": "GS", "morgan stanley": "MS", "oracle": "ORCL",
    # Added for the challenge set: the remaining big banks and Meta under its short name.
    "meta": "META", "jpmorgan chase": "JPM", "bank of america": "BAC", "citigroup": "C",
    "wells fargo": "WFC",
}

# Checked in order; the first match wins. The default is company_profile.
TEMPLATE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("full_report", ("investment research report", "full research report", "complete investment")),
    ("investigative", ("contradict", "investigate", "apparent contradiction")),
    ("earnings_analysis", ("earnings", "quarterly results", "consensus")),  # before "compare": "compare to consensus"
    ("comparative_analysis", ("compare", "comparison", " versus ", " vs ", " vs. ")),
    ("thematic", ("themes", "cross-cutting", "across the")),
    ("risk_assessment", ("risk assessment", "risks")),
]

_TICKER_HINT = re.compile(r"\((NASDAQ:|NYSE:)?\s*([A-Z]{1,5})\)|\bticker\s+([A-Z]{1,5})\b")

# --------------------------------------------------------------------------------------------------
# Vocabulary for the new fields
# --------------------------------------------------------------------------------------------------

SECTOR_WORDS: dict[str, str] = {
    "banks": "banks", "banking": "banks", "bank": "banks",
    "tech": "technology", "technology": "technology",
    "cloud": "cloud computing",
    "chip": "semiconductors", "chips": "semiconductors", "semiconductor": "semiconductors",
    "semiconductors": "semiconductors",
    "automaker": "automotive", "automakers": "automotive", "car makers": "automotive", "auto": "automotive",
    "oil": "energy", "energy": "energy",
}
_SECTOR_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, SECTOR_WORDS), key=len, reverse=True)) + r")\b")

_TEMPLATE_TO_TYPE = {
    "company_profile": "profile",
    "earnings_analysis": "earnings",
    "risk_assessment": "risk",
    "comparative_analysis": "comparison",
    "full_report": "full_report",
    "investigative": "investigative",
    "thematic": "thematic",
}

_CROSS_MEMORY = re.compile(
    r"already researched|previously researched|previous(ly)? (research|report|analys)|so far|"
    r"earlier research|you'?ve (already )?(researched|covered)|companies you'?ve", re.IGNORECASE)

_ASPECT = re.compile(
    r"\b(revenues?|earnings|risks?|margins?|growth|valuation|compare|comparison|profile|report|guidance|"
    r"profit|income|cash flow|debt|dividends?|executives|themes?|opportunit\w+|contradiction|competitive|"
    r"market share|sentiment|developments|financial|consensus|eps)\b", re.IGNORECASE)

_EXPLICIT_PERIOD = re.compile(
    r"\bFY\s?\d{4}\b|\bQ[1-4]\s?(?:FY)?\s?\d{4}\b|\bfiscal (?:year )?\d{4}\b|\b(?:19|20)\d{2}\b", re.IGNORECASE)
_RELATIVE_PERIOD = re.compile(
    r"\b(?:(?:most recent|latest|last|current|prior) (?:quarterly|quarter|year|fiscal year|annual|period)"
    r"|recent|latest|this year|year to date|ytd|(?:last|past|previous) \d+ (?:days|weeks|months|years))\b", re.IGNORECASE)

_FACTUAL_START = re.compile(r"^\W*(what|how much|how many|when|who)\b", re.IGNORECASE)
_INVESTIGATIVE_EXTRA = re.compile(r"explain why|why (?:did|is|are|has)", re.IGNORECASE)


@dataclass
class QueryProfile:
    """Heuristic reading of a request: report template, tickers, names, query type, complexity, periods and an ambiguity score with its signals."""
    template: str
    tickers: list[str]
    names: list[str]
    query_type: str = "profile"
    complexity: str = "single_company"
    periods: list[str] = field(default_factory=list)
    ambiguity: float = 0.0
    ambiguity_level: str = "low"
    signals: list[str] = field(default_factory=list)
    sector: str | None = None


def _resolve_entities(query: str) -> tuple[str, list[str], list[str]]:
    """Template, tickers and names, computed exactly as ``agent.core.analyze_query`` does."""
    lowered = f" {query.lower()} "
    template = next((t for t, words in TEMPLATE_RULES if any(w in lowered for w in words)), "company_profile")
    tickers: list[str] = []
    names: list[str] = []
    for name, ticker in KNOWN_COMPANIES.items():
        if re.search(rf"\b{re.escape(name)}\b", lowered) and ticker not in tickers:
            tickers.append(ticker)
            names.append(name.title())
    # A bare upper-case ticker ("NVDA", "AAPL revenue") that belongs to a company we know.
    known = set(KNOWN_COMPANIES.values())
    for m in re.finditer(r"\b[A-Z]{1,5}\b", query):
        if m.group(0) in known and m.group(0) not in tickers:
            tickers.append(m.group(0))
    for m in _TICKER_HINT.finditer(query):
        ticker = m.group(2) or m.group(3)
        before = query[: m.start()].lower().rstrip()
        # "Amazon (AWS)": brackets after a company we already know name a product or segment, not a
        # ticker, unless an exchange prefix says otherwise ("(NASDAQ: MSFT)").
        if m.group(2) and not m.group(1) and any(before.endswith(name) for name in KNOWN_COMPANIES):
            continue
        if ticker not in tickers:
            tickers.append(ticker)
    return template, tickers, names


def _find_periods(query: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for pattern in (_EXPLICIT_PERIOD, _RELATIVE_PERIOD):
        found += [(m.start(), m.group(0)) for m in pattern.finditer(query)]
    periods: list[str] = []
    for _, text in sorted(found):
        text = re.sub(r"\s+", " ", text.strip())
        text = text.upper() if re.match(r"(?i)(fy|q[1-4])", text) else text.lower()
        if text not in periods:
            periods.append(text)
    # A bare year already inside "Q3 2026" or "FY2025" is not a separate period.
    return [p for p in periods if not (re.fullmatch(r"\d{4}", p) and any(p in o and o != p for o in periods))]


def _find_sector(query: str) -> str | None:
    lowered = query.lower()
    # "Bank of America" / "Energy Transfer": a sector word inside a company name is not a sector request.
    for name in sorted(KNOWN_COMPANIES, key=len, reverse=True):
        lowered = re.sub(rf"\b{re.escape(name)}\b", " ", lowered)
    m = _SECTOR_RE.search(lowered)
    return SECTOR_WORDS[m.group(1)] if m else None


def _query_type(template: str, query: str, tickers: list[str], sector: str | None, cross_memory: bool) -> str:
    if template != "company_profile":
        return _TEMPLATE_TO_TYPE[template]
    if not tickers and not cross_memory and (not sector or not _ASPECT.search(query)):
        return "ambiguous"
    if _INVESTIGATIVE_EXTRA.search(query):
        return "investigative"
    if sector and not tickers:
        return "thematic"
    if len(tickers) == 1 and len(query.split()) <= 12 and _FACTUAL_START.match(query) and _ASPECT.search(query):
        return "factual"
    return "profile"


def analyze_query(query: str) -> QueryProfile:
    """Analyze ``query`` without calling a model: resolve entities, detect sector and periods, and score ambiguity (low below 0.3, high above 0.7)."""
    template, tickers, names = _resolve_entities(query)
    sector = _find_sector(query)
    cross_memory = bool(_CROSS_MEMORY.search(query))
    periods = _find_periods(query)

    if cross_memory:
        complexity = "cross_memory"
    elif sector and (not tickers or len(tickers) >= 4):
        complexity = "sector"
    elif len(tickers) >= 2:
        complexity = "multi_company"
    else:
        complexity = "single_company"

    score = 0.0
    signals: list[str] = []

    def add(delta: float, reason: str) -> None:
        nonlocal score
        score += delta
        signals.append(f"{delta:+.2f}: {reason}")

    if not tickers:
        add(0.45, "no company or ticker could be resolved")
        if sector:
            add(0.2, f"the request names a group ({sector}) but no companies")
    if not _ASPECT.search(query):
        add(0.15, "no explicit aspect or metric (revenue, risk, margin, ...)")
    if not periods:
        add(0.1, "no timeframe stated")
    if len(query.split()) < 6:
        add(0.1, "very short query")
    if cross_memory:
        add(-0.2, "refers to earlier research, so memory supplies the entities")
    if not signals:
        signals.append("no ambiguity signals: entity, aspect and timeframe are all stated")

    ambiguity = round(min(max(score, 0.0), 1.0), 2)
    level = "low" if ambiguity < 0.3 else "high" if ambiguity > 0.7 else "medium"
    return QueryProfile(
        template=template, tickers=tickers, names=names,
        query_type=_query_type(template, query, tickers, sector, cross_memory),
        complexity=complexity, periods=periods, ambiguity=ambiguity, ambiguity_level=level,
        signals=signals, sector=sector,
    )
