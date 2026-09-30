"""Disambiguation policy and edge-case detection (architecture spec §15.2 and §15.3).

Why: a vague request ("What's happening with the banks?") must not produce a shallow answer about
everything, nor a refusal. In autonomous mode the agent narrows the scope to the most likely reading,
writes the assumptions and exclusions into the report, and records the questions it *would* have asked.
In interactive mode the caller asks the user instead.

Edge cases are detected from what the tools actually returned, not from the query: a private company
shows up as "no EDGAR filer" failures, a recent IPO as fewer than three annual periods, and a fast-moving
situation as recent acquisition, leadership or bankruptcy news and 8-K items.

Everything is deterministic: no LLM calls (the spec's LLM tiebreaker is out of scope for now) and no
network. Dates always come from the ``today`` argument, never from the model's training data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from agent.query_analyzer import QueryProfile
from tools.base import ToolResult

# --------------------------------------------------------------------------------------------------
# Sector defaults (used when a request names a group instead of companies)
# --------------------------------------------------------------------------------------------------

SECTOR_DEFAULTS: dict[str, dict[str, Any]] = {
    "banks": {
        "tickers": ["JPM", "BAC", "C", "WFC", "GS", "MS"],
        "scope": "US large-cap banks",
        "exclusions": ("non-US banks and regional banks (the query gives no region; the six largest US banks "
                       "by assets are the most likely reading)"),
        "themes": ["earnings and net interest margin", "credit quality and provisions",
                   "capital and stress-test results", "the regulatory agenda"],
    },
    "technology": {
        "tickers": ["MSFT", "AAPL", "NVDA", "GOOGL", "AMZN", "META"],
        "scope": "US mega-cap technology companies",
        "exclusions": "smaller technology companies and non-US technology groups (the query names none)",
        "themes": ["revenue growth and margins", "AI and capital spending", "competitive position",
                   "regulatory and legal risk"],
    },
    "semiconductors": {
        "tickers": ["NVDA", "AVGO", "AMD", "QCOM", "INTC", "MU"],
        "scope": "US-listed semiconductor companies that file with the SEC",
        "exclusions": "foreign private issuers that file 20-F reports, and semiconductor equipment makers",
        "themes": ["demand from AI and data centres", "margins and pricing", "supply chain and export controls",
                   "inventory and the cycle"],
    },
    "automotive": {
        "tickers": ["TSLA", "GM", "F", "RIVN"],
        "scope": "US-listed automakers",
        "exclusions": "non-US automakers and parts suppliers (the query gives no region)",
        "themes": ["vehicle deliveries and pricing", "margins and cost of production",
                   "electric vehicle demand", "regulation and tariffs"],
    },
    "cloud computing": {
        "tickers": ["AMZN", "MSFT", "GOOGL"],
        "scope": "the three largest cloud platforms (AWS, Azure, Google Cloud)",
        "exclusions": "smaller and regional cloud providers",
        "themes": ["cloud revenue growth", "operating margin", "market share", "AI workloads and capacity"],
    },
    "energy": {
        "tickers": ["XOM", "CVX", "COP"],
        "scope": "US oil and gas majors",
        "exclusions": "non-US energy companies, utilities and renewables specialists",
        "themes": ["commodity prices and earnings", "production and reserves", "capital returns",
                   "the energy transition"],
    },
}

_ASPECT_QUESTION = "Which aspect matters most (for example earnings, risk, valuation, or recent news)?"
_TIME_QUESTION = "Which period should the research cover (the latest quarter, the last year, or a specific fiscal year)?"


@dataclass
class Disambiguation:
    level: str
    proceed_reading: str
    assumptions: list[str] = field(default_factory=list)
    clarifying_questions: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    timeframe: str = ""
    themes: list[str] = field(default_factory=list)
    mode: str = "autonomous"


def _timeframe(profile: QueryProfile, today: date) -> tuple[str, bool]:
    """The timeframe to research, and whether it was assumed rather than stated."""
    if profile.periods:
        return ", ".join(profile.periods), False
    return f"latest reported period as of {today.isoformat()}", True


def _questions(profile: QueryProfile, has_aspect: bool, has_period: bool) -> list[str]:
    if profile.sector:
        group = profile.sector if profile.sector.endswith("s") else f"{profile.sector} companies"
        who = f"Which {group} do you mean, and in which region?"
    else:
        who = "Which company or companies should the research cover?"
    questions = [who]
    if not has_aspect:
        questions.append(_ASPECT_QUESTION)
    if not has_period:
        questions.append(_TIME_QUESTION)
    return questions[:3]


def disambiguate(profile: QueryProfile, *, today: date, mode: str = "autonomous") -> Disambiguation:
    timeframe, tf_assumed = _timeframe(profile, today)
    level = profile.ambiguity_level
    has_aspect = not any("no explicit aspect" in s for s in profile.signals)
    defaults = SECTOR_DEFAULTS.get(profile.sector or "")
    tf_note = (f"Timeframe: {timeframe}, because the request names no period." if tf_assumed else "")

    if level == "low":
        assumptions = [tf_note] if tf_assumed else []
        return Disambiguation(level, "Proceeding as requested.", assumptions, [], list(profile.tickers),
                              timeframe, [], mode)

    questions = _questions(profile, has_aspect, not tf_assumed)

    if level == "high" and mode == "interactive":
        return Disambiguation(
            level, "The request is too ambiguous to research; the user is asked to clarify first.",
            [], questions, [], timeframe, [], mode)

    tickers = list(profile.tickers)
    assumptions: list[str] = []
    themes: list[str] = []
    if not tickers and defaults:
        tickers = list(defaults["tickers"])
        themes = list(defaults["themes"])
        assumptions.append(f"Scope: {defaults['scope']} ({', '.join(tickers)}).")
        assumptions.append(f"Excluded: {defaults['exclusions']}.")
        if not has_aspect:
            assumptions.append(f"Themes: {', '.join(themes)}.")
        reading = (f"Reading the request as {defaults['scope']} ({', '.join(tickers)}), for the "
                   f"{timeframe}, covering {', '.join(themes)}.")
    elif not tickers:
        return Disambiguation(
            level, "Research cannot proceed without a named company or a recognisable sector; "
                   "the questions below are what would have been asked.",
            [], questions, [], timeframe, [], mode)
    else:
        if not has_aspect:
            assumptions.append("Aspect: a general overview, because the request names no specific metric or topic.")
        reading = f"Reading the request as a general look at {', '.join(tickers)} for the {timeframe}."
    if tf_assumed:
        assumptions.append(tf_note)
    return Disambiguation(level, reading, assumptions, questions, tickers, timeframe, themes, mode)


def render_assumptions(d: Disambiguation) -> str:
    """Markdown for the report's methodology or scope note; empty when there is nothing to disclose."""
    if not d.assumptions:
        return ""
    if d.level == "low":
        lines = ["**Scope and assumptions**", ""]
    else:
        lines = [f"**Scope and assumptions** (the request was ambiguous; the reading used is: {d.proceed_reading})", ""]
    lines += [f"- {a}" for a in d.assumptions]
    if d.clarifying_questions and d.level != "low":
        lines += ["", "Questions that would have been asked:"]
        lines += [f"{i}. {q}" for i, q in enumerate(d.clarifying_questions, 1)]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
# Edge cases detected from tool outcomes (§15.3)
# --------------------------------------------------------------------------------------------------


@dataclass
class CoverageFlag:
    kind: str  # private_company | recent_ipo | non_us_filer
    entity: str
    message: str
    confidence_note: str


@dataclass
class TemporalFlag:
    entity: str
    event: str
    evidence_id: str
    date: str  # ISO date, or "" when the source has none
    headline: str
    message: str


_NO_FILER = re.compile(r"no (?:EDGAR|SEC) filer", re.IGNORECASE)
_TICKER_IN_MSG = re.compile(r"ticker '([^']+)'", re.IGNORECASE)
_ONLY_N = re.compile(r"Only (\d+) (?:annual )?periods? available", re.IGNORECASE)
_FILER_TOOLS = {"sec_filing_search", "financial_data_api", "company_profile"}


def _entity_of(args: dict[str, Any], message: str = "") -> str:
    for key in ("ticker", "company", "company_name", "name"):
        if args.get(key):
            return str(args[key]).upper() if key == "ticker" else str(args[key])
    m = _TICKER_IN_MSG.search(message)
    return m.group(1).upper() if m else ""


def _notes(result: ToolResult) -> list[str]:
    notes = result.data.get("notes") if isinstance(result.data, dict) else None
    return [str(n) for n in notes] if isinstance(notes, list) else []


def _period_label(row: dict[str, Any]) -> str:
    if row.get("fiscal_year"):
        return f"{row.get('fiscal_period') or 'FY'}{row['fiscal_year']}"
    return str(row.get("period_end") or "unknown period")


def coverage_flags(evidence: dict[str, ToolResult], failures: list[tuple[str, dict, str]]) -> list[CoverageFlag]:
    flags: list[CoverageFlag] = []
    seen: set[tuple[str, str]] = set()

    def add(flag: CoverageFlag) -> None:
        if (flag.kind, flag.entity) not in seen:
            seen.add((flag.kind, flag.entity))
            flags.append(flag)

    with_financials = {
        _entity_of({"ticker": r.data.get("ticker")} if r.data.get("ticker") else r.args)
        for r in evidence.values() if r.tool == "financial_data_api" and r.data.get("data")}

    for tool, args, message in failures:
        entity = _entity_of(args, message)
        if tool in _FILER_TOOLS and _NO_FILER.search(message) and entity not in with_financials:
            add(CoverageFlag(
                "private_company", entity,
                f"{entity or 'The company'} has no SEC filings (no EDGAR filer was found), so the filing-based "
                "sections (Tier 1 sources) are skipped; figures come from web and news sources only, and nothing "
                "is estimated to fill the gap.",
                "Lower confidence: web and news sources are not primary filings."))
        if "20-F" in message:
            add(_non_us(entity))

    for result in evidence.values():
        if result.tool == "financial_data_api":
            data = result.data
            period = data.get("period") or result.args.get("period")
            rows = data.get("data") or []
            entity = _entity_of({"ticker": data.get("ticker")} if data.get("ticker") else result.args)
            only = next((int(m.group(1)) for n in _notes(result) if (m := _ONLY_N.search(n))), None)
            requested = result.args.get("years", 3)
            short = only is not None and only < 3 or (len(rows) < 3 and requested >= 3)
            if period == "annual" and rows and short:
                labels = ", ".join(_period_label(r) for r in rows)
                add(CoverageFlag(
                    "recent_ipo", entity,
                    f"{entity or 'The company'} has only {len(rows)} annual period(s) of financial data ({labels}), "
                    "so multi-year trends and CAGR are not meaningful. The S-1 or 424B prospectus is the best "
                    "source for earlier history.",
                    "Growth rates over fewer than three annual periods are indicative only."))
        if any("20-F" in n for n in _notes(result)):
            add(_non_us(_entity_of(result.args)))
    return flags


def _non_us(entity: str) -> CoverageFlag:
    return CoverageFlag(
        "non_us_filer", entity,
        f"{entity or 'The issuer'} files 20-F or 6-K reports rather than 10-K and 10-Q, so its disclosures follow "
        "different timing and content and are not directly comparable with US domestic filers.",
        "Section structure and reporting periods differ from 10-K and 10-Q filers.")


# --------------------------------------------------------------------------------------------------
# Rapidly changing situations
# --------------------------------------------------------------------------------------------------

_EVENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("acquisition/merger", re.compile(r"\bacqui(?:re|res|red|ring|sition|sitions)\b|\bmerger\b|\bmerges?\b|"
                                      r"\btakeover\b|\bbuyout\b|\bto buy\b", re.I)),
    ("leadership change", re.compile(
        r"\b(?:ceo|cfo|chief executive|chief financial officer)\b.{0,40}\b(?:steps? down|resign\w*|appointed|ousted)\b|"
        r"\b(?:steps? down|resign\w*|appoints?|appointed|ousted)\b.{0,40}\b(?:ceo|cfo|chief executive|"
        r"chief financial officer)\b", re.I)),
    ("bankruptcy/restructuring", re.compile(r"\bbankruptcy\b|\bchapter 11\b|\brestructuring\b", re.I)),
    ("restatement", re.compile(r"\brestate\w*", re.I)),
    ("guidance change", re.compile(r"\b(?:withdraws?|cuts?|raises?|lowers?|slashes)\b.{0,25}\bguidance\b", re.I)),
    ("activist involvement", re.compile(r"\bactivist\b|\bproxy fight\b", re.I)),
]

_8K_ITEMS = {
    "1.01": "material agreement",
    "2.01": "acquisition/merger",
    "1.03": "bankruptcy/restructuring",
    "4.02": "restatement",
    "5.02": "leadership change",
}
_8K_LABELS = {
    "1.01": "material agreement", "2.01": "acquisition or disposal completed", "1.03": "bankruptcy or receivership",
    "4.02": "non-reliance on previously issued financials", "5.02": "officer or director change",
}


def _as_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def rate_of_change_flags(evidence: dict[str, ToolResult], *, today: date, lookback_days: int = 60) -> list[TemporalFlag]:
    cutoff = today - timedelta(days=lookback_days)
    found: list[TemporalFlag] = []

    def emit(entity: str, event: str, evidence_id: str, when: date | None, headline: str) -> None:
        when_s = when.isoformat() if when else ""
        found.append(TemporalFlag(
            entity, event, evidence_id, when_s, headline,
            f"{entity + ': ' if entity else ''}{event} reported{' on ' + when_s if when_s else ''} "
            f"({headline}) [{evidence_id}]."))

    for ev in evidence.values():
        entity = _entity_of(ev.args) or _entity_of({"ticker": ev.data.get("ticker")}) if ev.data else ""
        entity = entity or str(ev.args.get("query") or "")
        if ev.tool in ("news_sentiment", "web_search"):
            items = ev.data.get("articles") if ev.tool == "news_sentiment" else ev.data.get("results")
            for item in items or []:
                when = _as_date(item.get("published"))
                if when is not None and when < cutoff:
                    continue
                title = str(item.get("title") or "")
                text = f"{title} {item.get('snippet') or item.get('summary') or ''}"
                for event, pattern in _EVENT_RULES:
                    if pattern.search(text):
                        emit(entity, event, ev.evidence_id, when, title)
        elif ev.tool == "sec_filing_search" and ev.data.get("filing_type") == "8-K":
            when = _as_date(ev.data.get("filing_date"))
            if when is not None and when < cutoff:
                continue
            items_text = " ".join(n.split("8-K items:", 1)[1] for n in _notes(ev) if "8-K items:" in n)
            for item in dict.fromkeys(re.findall(r"\d\.\d{2}", items_text)):
                if item in _8K_ITEMS:
                    emit(str(ev.data.get("ticker") or entity), _8K_ITEMS[item], ev.evidence_id, when,
                         f"8-K item {item} ({_8K_LABELS[item]})")

    found.sort(key=lambda f: f.date, reverse=True)  # newest first; undated ("") last
    unique: dict[tuple[str, str], TemporalFlag] = {}
    for flag in found:
        unique.setdefault((flag.entity, flag.event), flag)
    return list(unique.values())


def render_temporal_flags(flags: list[TemporalFlag]) -> str:
    if not flags:
        return ""
    lines = ["**Temporal sensitivity:** recent events may be changing the picture quickly. "
             "Figures in this report are as of the evidence dates and may change."]
    for f in flags:
        who = f"{f.entity}: " if f.entity else ""
        lines.append(f"- {who}{f.event}, {f.date or 'date unknown'}: {f.headline} [{f.evidence_id}]")
    return "\n".join(lines)
