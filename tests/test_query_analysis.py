"""Day 10: query analysis, disambiguation and edge-case detection (no network)."""

from __future__ import annotations

from datetime import date

import pytest

from agent import core
from agent.disambiguation import (
    SECTOR_DEFAULTS,
    Disambiguation,
    coverage_flags,
    disambiguate,
    rate_of_change_flags,
    render_assumptions,
    render_temporal_flags,
)
from agent.query_analyzer import SECTOR_WORDS, analyze_query
from tools.base import ToolResult

TODAY = date(2026, 9, 30)

CHALLENGES = {
    "C1": "Create a comprehensive profile of Microsoft Corporation including business overview, financial summary, "
          "key executives, and recent developments.",
    "C2": "Analyze Apple Inc.'s most recent quarterly earnings. Compare actual results to consensus estimates and "
          "identify key takeaways from the earnings call.",
    "C3": "Produce a comprehensive risk assessment for Tesla Inc. covering financial risks, operational risks, "
          "regulatory risks, and competitive risks.",
    "C4": "Compare the cloud computing divisions of Amazon (AWS), Microsoft (Azure), and Google (GCP). Analyze "
          "revenue growth, market share, margins, and competitive advantages.",
    "C5": "Research Palantir Technologies. Note: Recent news reports suggest the company is struggling, but their "
          "financial statements show strong growth. Investigate and explain the apparent contradiction.",
    "C6": "What's happening with the banks?",
    "C7": "Based on the companies you've already researched, what themes emerge across the technology sector? "
          "Identify cross-cutting risks and opportunities.",
    "C8": "Produce a complete investment research report on NVIDIA Corporation.",
}

EXISTING_CASES = [
    "Create a comprehensive profile of Microsoft Corporation including business overview, financial summary, "
    "key executives, and recent developments.",
    "Produce a comprehensive risk assessment for Tesla Inc.",
    "Analyze Apple Inc.'s most recent quarterly earnings",
    "Compare the cloud divisions of Amazon (AWS), Microsoft (Azure), and Google (GCP)",
    "Produce a complete investment research report on NVIDIA Corporation",
    "Research the widget maker (ticker WDGT)",
    "Analyze Apple Inc.'s most recent quarterly earnings. Compare actual results to consensus estimates.",
]


def result(evidence_id: str, tool: str, data: dict, args: dict | None = None) -> ToolResult:
    return ToolResult(evidence_id=evidence_id, tool=tool, args=args or {}, data=data, source_tier=2,
                      retrieved_at="2026-09-30T00:00:00Z", latency_ms=1.0)


# ---------------------------------------------------------------- analysis


@pytest.mark.parametrize("query", EXISTING_CASES + list(CHALLENGES.values()))
def test_matches_core_analyzer(query):
    new, old = analyze_query(query), core.analyze_query(query)
    assert (new.template, new.tickers, new.names) == (old.template, old.tickers, old.names)


@pytest.mark.parametrize("cid, qtype, complexity, levels", [
    ("C1", "profile", "single_company", {"low"}),
    ("C2", "earnings", "single_company", {"low"}),
    ("C3", "risk", "single_company", {"low"}),
    ("C4", "comparison", "multi_company", {"low"}),
    ("C5", "investigative", "single_company", {"low"}),
    ("C6", "ambiguous", "sector", {"high"}),
    ("C7", "thematic", "cross_memory", {"low", "medium"}),
    ("C8", "full_report", "single_company", {"low"}),
])
def test_challenge_profiles(cid, qtype, complexity, levels):
    p = analyze_query(CHALLENGES[cid])
    assert (p.query_type, p.complexity) == (qtype, complexity)
    assert p.ambiguity_level in levels
    assert p.signals


def test_challenge_6_score_and_sector():
    p = analyze_query(CHALLENGES["C6"])
    assert p.ambiguity == 1.0 and p.sector == "banks" and p.tickers == []
    assert any(s.startswith("+0.45") for s in p.signals)


def test_periods_and_sector_words():
    assert analyze_query(CHALLENGES["C2"]).periods == ["most recent quarterly"]
    assert analyze_query("Apple revenue in FY2025 and Q3 2026").periods == ["FY2025", "Q3 2026"]
    assert analyze_query("Show the semiconductor outlook").sector == "semiconductors"
    assert analyze_query(CHALLENGES["C7"]).sector == "technology"
    assert set(SECTOR_WORDS.values()) == {"banks", "technology", "cloud computing", "semiconductors",
                                          "automotive", "energy"}


def test_factual_and_low_signal_case():
    p = analyze_query("What was Apple's revenue in FY2024?")
    assert p.query_type == "factual" and p.periods == ["FY2024"]
    assert p.ambiguity == 0.0 and "no ambiguity signals" in p.signals[0]


def test_no_entity_no_sector_is_ambiguous():
    p = analyze_query("Tell me something interesting about that thing over there")
    assert p.query_type == "ambiguous" and p.ambiguity_level in {"medium", "high"}


# ---------------------------------------------------------------- disambiguation


def test_high_ambiguity_autonomous_uses_sector_default():
    d = disambiguate(analyze_query(CHALLENGES["C6"]), today=TODAY)
    assert d.tickers == ["JPM", "BAC", "C", "WFC", "GS", "MS"] and d.mode == "autonomous"
    assert len(d.assumptions) >= 3 and any("Excluded" in a and "regional banks" in a for a in d.assumptions)
    assert 1 <= len(d.clarifying_questions) <= 3
    assert "2026-09-30" in d.timeframe and d.themes == SECTOR_DEFAULTS["banks"]["themes"]
    text = render_assumptions(d)
    assert text.startswith("**Scope and assumptions** (the request was ambiguous;")
    assert "Questions that would have been asked:" in text and "1. " in text


def test_interactive_mode_asks_instead():
    d = disambiguate(analyze_query(CHALLENGES["C6"]), today=TODAY, mode="interactive")
    assert d.tickers == [] and d.clarifying_questions and d.mode == "interactive"


def test_high_ambiguity_without_sector_cannot_proceed():
    p = analyze_query("Tell me more")
    assert p.ambiguity_level == "high" and p.sector is None
    d = disambiguate(p, today=TODAY)
    assert d.tickers == [] and "cannot proceed" in d.proceed_reading and d.clarifying_questions


def test_low_ambiguity_has_no_assumptions_when_period_is_stated():
    d = disambiguate(analyze_query("Analyze Apple Inc.'s revenue in FY2025"), today=TODAY)
    assert d.level == "low" and d.assumptions == [] and render_assumptions(d) == ""
    assert d.tickers == ["AAPL"] and d.timeframe == "FY2025"


def test_low_ambiguity_missing_timeframe_only_notes_the_timeframe():
    d = disambiguate(analyze_query(CHALLENGES["C8"]), today=TODAY)
    assert len(d.assumptions) == 1 and "2026-09-30" in d.assumptions[0]
    assert "was ambiguous" not in render_assumptions(d)


def test_medium_ambiguity_documents_assumptions():
    p = analyze_query(CHALLENGES["C7"])
    d = disambiguate(p, today=TODAY)
    assert d.level == "medium" and d.tickers == SECTOR_DEFAULTS["technology"]["tickers"] and d.assumptions


# ---------------------------------------------------------------- coverage flags


def test_private_company_flag():
    failures = [("sec_filing_search", {"ticker": "ACME"}, "sec_filing_search: no EDGAR filer found for ticker 'ACME'")]
    (flag,) = coverage_flags({}, failures)
    assert flag.kind == "private_company" and flag.entity == "ACME"
    assert "no SEC filings" in flag.message and "Tier 1" in flag.message and "nothing is estimated" in flag.message
    assert flag.confidence_note


def test_private_company_not_flagged_when_financials_exist():
    ev = {"E1": result("E1", "financial_data_api", {"ticker": "ACME", "period": "annual", "data": [
        {"fiscal_year": 2025, "fiscal_period": "FY", "values": {}}] * 3})}
    failures = [("company_profile", {"ticker": "ACME"}, "no SEC filer for this ticker")]
    assert coverage_flags(ev, failures) == []


def _annual(n: int, notes: list[str] | None = None) -> ToolResult:
    rows = [{"fiscal_year": 2025 - i, "fiscal_period": "FY", "period_end": f"{2025 - i}-12-31", "values": {}}
            for i in range(n)]
    return result("E7", "financial_data_api", {"ticker": "NEWCO", "period": "annual", "data": rows,
                                                "notes": notes or []},
                  {"ticker": "NEWCO", "period": "annual", "years": 3})


def test_recent_ipo_flag_and_not_for_three_rows():
    (flag,) = coverage_flags({"E7": _annual(2, ["Only 2 periods available."])}, [])
    assert flag.kind == "recent_ipo" and flag.entity == "NEWCO"
    assert "FY2025" in flag.message and "FY2024" in flag.message and "CAGR" in flag.message and "S-1" in flag.message
    assert coverage_flags({"E7": _annual(3)}, []) == []


def test_non_us_filer_flag():
    failures = [("sec_filing_search", {"ticker": "TSM"}, "no 10-K found; the issuer files 20-F reports")]
    (flag,) = coverage_flags({}, failures)
    assert flag.kind == "non_us_filer" and flag.entity == "TSM"


# ---------------------------------------------------------------- temporal flags


def news(evidence_id: str, articles: list[dict], query: str = "ACME") -> ToolResult:
    return result(evidence_id, "news_sentiment", {"articles": articles}, {"query": query})


def test_news_events_detected_newest_first_and_stale_ignored():
    ev = {
        "E1": news("E1", [
            {"title": "ACME to acquire Widgets Inc in $5bn deal", "published": "2026-09-20"},
            {"title": "ACME CEO resigns after probe", "published": "2026-09-25"},
            {"title": "ACME agrees old merger", "published": "2026-03-01"},  # outside the 60-day lookback
            {"title": "ACME opens new office", "published": "2026-09-28"},
        ]),
    }
    flags = rate_of_change_flags(ev, today=TODAY)
    assert [(f.event, f.date, f.evidence_id) for f in flags] == [
        ("leadership change", "2026-09-25", "E1"), ("acquisition/merger", "2026-09-20", "E1")]
    assert flags[0].entity == "ACME" and "ACME CEO resigns" in flags[0].headline


def test_web_search_and_dedup():
    web = result("E2", "web_search", {"results": [
        {"title": "Chapter 11 filing looms", "snippet": "the company withdraws guidance", "published": None},
        {"title": "Restructuring plan", "snippet": "", "published": "2026-09-01"},
    ]}, {"query": "ACME"})
    flags = rate_of_change_flags({"E2": web}, today=TODAY)
    events = [f.event for f in flags]
    assert events.count("bankruptcy/restructuring") == 1  # de-duplicated by (entity, event)
    assert "guidance change" in events and flags[0].date == "2026-09-01" and flags[-1].date == ""


def test_8k_items_flagged():
    k = result("E3", "sec_filing_search", {
        "ticker": "ACME", "filing_type": "8-K", "filing_date": "2026-09-15",
        "notes": ["Source: SEC EDGAR.", "8-K items: 2.01, 5.02, 9.01."]}, {"ticker": "ACME", "filing_type": "8-K"})
    old = result("E4", "sec_filing_search", {
        "ticker": "ACME", "filing_type": "8-K", "filing_date": "2025-01-15", "notes": ["8-K items: 1.03."]})
    earnings = result("E5", "sec_filing_search", {
        "ticker": "ACME", "filing_type": "8-K", "filing_date": "2026-09-16", "notes": ["8-K items: 2.02, 9.01."]})
    flags = rate_of_change_flags({"E3": k, "E4": old, "E5": earnings}, today=TODAY)
    assert {(f.event, f.evidence_id) for f in flags} == {("acquisition/merger", "E3"), ("leadership change", "E3")}
    assert all(f.date == "2026-09-15" and f.entity == "ACME" for f in flags)


def test_render_temporal_flags():
    assert render_temporal_flags([]) == ""
    ev = {"E1": news("E1", [{"title": "ACME CEO resigns", "published": "2026-09-25"}])}
    text = render_temporal_flags(rate_of_change_flags(ev, today=TODAY))
    assert text.startswith("**Temporal sensitivity:**") and "may change" in text
    assert "2026-09-25" in text and "[E1]" in text and "leadership change" in text


def test_disambiguation_dataclass_defaults():
    d = Disambiguation(level="low", proceed_reading="x")
    assert d.mode == "autonomous" and d.tickers == [] and d.clarifying_questions == []


# ---------------------------------------------------------------- Day 10 review regressions


@pytest.mark.parametrize("query", ["NVDA", "AAPL revenue", "What is NVDA's margin?"])
def test_bare_upper_case_ticker_is_resolved(query):
    assert analyze_query(query).tickers, query


def test_bare_ticker_does_not_match_ordinary_acronyms():
    assert analyze_query("What is the CEO of AI doing in the US").tickers == []


def test_sector_word_inside_company_name_is_not_a_sector():
    profile = analyze_query("Bank of America risks")
    assert profile.tickers == ["BAC"] and profile.sector is None
    assert analyze_query(CHALLENGES["C5"]).sector is None


@pytest.mark.parametrize("query", ["metadata and the metaverse", "pineapple prices", "automation and autonomous cars",
                                   "Oxford University", "shell company"])
def test_substring_matches_do_not_resolve_entities(query):
    assert analyze_query(query).tickers == []


def test_past_n_years_is_a_period():
    assert analyze_query("Tesla revenue over the past 3 years").periods == ["past 3 years"]


def test_dollar_amounts_and_form_names_are_not_periods():
    assert analyze_query("Apple raised $5 billion per its 10-K").periods == []


def test_acquire_customers_is_not_an_acquisition():
    ev = {"n": result("E1", "news_sentiment", {"articles": [
        {"title": "Apple to acquire customers with a new plan", "published": "2026-09-20"}]}, {"ticker": "AAPL"})}
    assert rate_of_change_flags(ev, today=TODAY) == []


def test_rfc2822_published_dates_are_parsed_and_filtered():
    def flags(published):
        ev = {"n": result("E1", "web_search", {"results": [
            {"title": "Apple to acquire Beats", "published": published}]}, {"ticker": "AAPL"})}
        return rate_of_change_flags(ev, today=TODAY)
    assert flags("Mon, 28 Sep 2024 10:00:00 GMT") == []
    assert flags("Mon, 28 Sep 2026 10:00:00 GMT")[0].date == "2026-09-28"


def test_article_about_another_company_is_not_attributed_to_the_entity():
    ev = {"n": result("E1", "news_sentiment", {"articles": [
        {"title": "Microsoft to acquire Activision", "published": "2026-09-20"}]}, {"ticker": "AAPL"})}
    assert rate_of_change_flags(ev, today=TODAY) == []


def test_recent_ipo_not_flagged_when_only_one_year_was_requested():
    ev = {"a": result("E1", "financial_data_api",
                      {"ticker": "NEW", "period": "annual", "data": [{"fiscal_year": 2025}],
                       "notes": ["Only 1 annual period available"]}, {"ticker": "NEW", "years": 1})}
    assert [f.kind for f in coverage_flags(ev, [])] == []


@pytest.mark.parametrize("tool", ["financial_data_api", "web_search", "fallback"])
def test_private_company_from_fallback_message(tool):
    msg = ("Fallback: financial_data_api failed (not_found: financial_data_api: no EDGAR filer found for ticker "
           "'XYZ'); data from web_search.")
    flags = coverage_flags({}, [(tool, {"query": "XYZ"}, msg)])
    assert [(f.kind, f.entity) for f in flags] == [("private_company", "XYZ")]
