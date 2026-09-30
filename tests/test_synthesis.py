"""Tests for the synthesis layer: conflict resolution, the claim ledger and narrative threads. No network."""

from __future__ import annotations

from synthesis.conflict_resolver import (
    DEFAULT_TIER_WEIGHTS,
    Claim,
    Period,
    ReliabilityConfig,
    confidence,
    resolve,
    tier_for_source,
)
from synthesis.engine import build_ledger, extract_claims, triangulate
from synthesis.narrative import (
    build_threads,
    conflicts_markdown,
    render_brief,
    sentiment_fact_alignment,
)
from tools import build_default_registry
from tools.base import ToolResult

CFG = ReliabilityConfig()


def claim(cid="C1", value=100e9, provider="sec_xbrl", tier=1, ev="E1", year=2024, metric="revenue",
          unit="USD", filed=None, period="FY", entity="ACME", fallback=False) -> Claim:
    return Claim(claim_id=cid, entity=entity, metric=metric, period=Period(year, period), value=value,
                 unit=unit, evidence_ids=[ev], source_tiers=[tier], provider=provider, filed=filed,
                 is_fallback=fallback)


def result(eid, tool, data, tier=1, args=None) -> ToolResult:
    return ToolResult(evidence_id=eid, tool=tool, args=args or {}, data=data, source_tier=tier,
                      retrieved_at="2026-01-01T00:00:00Z", latency_ms=1.0)


def fin(eid, ticker, name, statement, unit, rows, provider=None) -> ToolResult:
    data = {"ticker": ticker, "company_name": name, "statement_type": statement, "unit": unit,
            "data": [{"fiscal_year": fy, "fiscal_period": "FY", "period_end": f"{fy}-12-31", "values": v}
                     for fy, v in rows]}
    if provider:
        data["provider"] = provider
    return result(eid, "financial_data_api", data, args={"ticker": ticker})


def news(eid, query, score, label, count=14) -> ToolResult:
    return result(eid, "news_sentiment", {"query": query, "article_count": count, "articles": [],
                                          "overall": {"score": score, "label": label}}, tier=4)


# --- tiers and confidence -------------------------------------------------------------------------


def test_tier_for_source():
    assert tier_for_source("web_search", 5, "https://www.reuters.com/markets/x") == 4
    assert tier_for_source("news_sentiment", 4, "https://mock.wsj.com/a") == 4  # parent domain
    assert tier_for_source("web_search", 5, "https://randomblog.net/post") == 5
    assert tier_for_source("web_search", 5, "notreuters.com") == 5  # not a subdomain match
    assert tier_for_source("sec_filing_search", 1) == 1
    assert tier_for_source("financial_data_api", None) == 5


def test_weight_unknown_tier_is_least_reliable():
    assert CFG.weight(None) == DEFAULT_TIER_WEIGHTS[5]
    assert CFG.weight(9) == DEFAULT_TIER_WEIGHTS[5]
    assert CFG.weight(2) == 0.85


def test_confidence_formula():
    assert confidence(1, 1, False, False, False, CFG) == 0.95
    assert confidence(1, 2, False, False, False, CFG) == 0.99  # 1.00 capped
    assert confidence(2, 2, False, False, False, CFG) == 0.9
    assert confidence(1, 1, True, False, False, CFG) == round(0.95 * 0.9, 4)
    assert confidence(1, 1, False, True, False, CFG) == round(0.95 * 0.8, 4)
    assert confidence(1, 1, True, True, False, CFG) == round(0.95 * 0.9 * 0.8, 4)
    assert confidence(1, 3, False, False, True, CFG) == 0.5
    assert confidence(5, 1, False, False, True, CFG) == 0.3  # cap never raises


# --- resolution -----------------------------------------------------------------------------------


def test_millions_vs_base_units_is_not_a_conflict():
    a = claim("C1", 130497 * 1e6, "sec_xbrl", 1, "E1")
    b = claim("C2", 130497e6, "yfinance", 2, "E2")
    out, conflicts = resolve([a, b], CFG)
    assert conflicts == [] and len(out) == 1 and out[0].status == "verified"


def test_highest_tier_wins_on_ten_percent_gap():
    a = claim("C1", 100e9, "sec_xbrl", 1, "E1")
    b = claim("C2", 110e9, "yfinance", 2, "E2")
    out, conflicts = resolve([a, b], CFG)
    assert len(conflicts) == 1 and conflicts[0].method == "highest_tier"
    assert out[0].value == 100e9 and out[0].evidence_ids == ["E1"]
    assert conflicts[0].chosen_value == 100e9
    assert {v["evidence_id"] for v in conflicts[0].values} == {"E1", "E2"}
    assert out[0].status == "single_source"


def test_restatement_latest_filing_wins():
    old = claim("C1", 100e9, "sec_xbrl", 1, "E1", filed="2024-02-01")
    new = claim("C2", 110e9, "sec_xbrl", 1, "E2", filed="2025-02-01")
    out, conflicts = resolve([old, new], CFG)
    assert conflicts[0].method == "restatement_latest_filing"
    assert out[0].value == 110e9 and "2025-02-01" in conflicts[0].rationale


def test_triangulated_majority():
    a = claim("C1", 100e9, "sec_xbrl", 1, "E1")
    b = claim("C2", 101e9, "yfinance", 2, "E2")
    c = claim("C3", 140e9, "web", 5, "E3")
    out, conflicts = resolve([a, b, c], CFG)
    assert conflicts[0].method == "triangulated_majority"
    assert out[0].value == 100e9 and out[0].status == "verified"
    assert out[0].confidence == 0.99


def test_same_tier_disagreement_is_unresolved_and_capped():
    a = claim("C1", 100e9, "yfinance", 2, "E1")
    b = claim("C2", 130e9, "other_curated", 2, "E2")
    out, conflicts = resolve([a, b], CFG)
    assert conflicts[0].method == "unresolved_range"
    assert out[0].status == "conflicted" and out[0].confidence <= 0.5
    assert out[0].value == 100e9
    assert "$100.00 billion to $130.00 billion" in conflicts[0].rationale


def test_different_fiscal_years_and_periods_never_conflict():
    claims = [claim("C1", 100e9, year=2024), claim("C2", 200e9, year=2025),
              claim("C3", 25e9, year=2025, period="Q1", ev="E3")]
    out, conflicts = resolve(claims, CFG)
    assert conflicts == [] and len(out) == 3


def test_peer_comparison_and_xbrl_are_not_independent():
    a = claim("C1", 100e9, "sec_xbrl", 1, "E1")
    b = claim("C2", 100e9, "peer_comparison:sec_xbrl", 1, "E2")
    out, _ = resolve([a, b], CFG)
    assert out[0].status == "single_source" and out[0].evidence_ids == ["E1", "E2"]
    assert out[0].confidence == 0.95


def test_xbrl_and_yfinance_agreeing_is_verified():
    a = claim("C1", 100e9, "sec_xbrl", 1, "E1")
    b = claim("C2", 101e9, "yfinance", 2, "E2")
    out, conflicts = resolve([a, b], CFG)
    assert conflicts == [] and out[0].status == "verified"
    assert out[0].value == 100e9 and out[0].source_tiers == [1, 2]
    assert out[0].confidence == 0.99


def test_ratio_absolute_tolerance_and_fallback_flag():
    a = claim("C1", 0.0040, "sec_xbrl", 1, metric="net_margin", unit="ratio")
    b = claim("C2", 0.0046, "yfinance", 2, "E2", metric="net_margin", unit="ratio", fallback=True)
    out, conflicts = resolve([a, b], CFG)
    assert conflicts == [] and out[0].status == "verified"
    assert out[0].is_fallback is False  # one non-fallback source supports it


def test_single_source_and_fallback_penalty():
    out, _ = resolve([claim(fallback=True)], CFG)
    assert out[0].status == "single_source" and out[0].confidence == round(0.95 * 0.9, 4)


# --- extraction -----------------------------------------------------------------------------------


def test_extract_claims_from_mock_registry():
    reg = build_default_registry("mock")
    income = reg.dispatch("financial_data_api", {"ticker": "NVDA", "statement_type": "income_statement"})
    ratios = reg.dispatch("financial_data_api", {"ticker": "NVDA", "statement_type": "key_ratios"})
    evidence = {income.evidence_id: income, ratios.evidence_id: ratios}
    claims = extract_claims(evidence, CFG)
    rev = [c for c in claims if c.metric == "revenue" and c.period.fiscal_year == 2025][0]
    assert rev.value == 130497e6 and rev.unit == "USD" and rev.entity == "NVDA"
    eps = [c for c in claims if c.metric == "eps_diluted"][0]
    assert eps.unit == "USD/share" and eps.value < 100
    assert [c.claim_id for c in claims] == [f"C{i}" for i in range(1, len(claims) + 1)]

    ledger = build_ledger(evidence)
    gm = ledger.get("NVDA", "gross_margin", 2025)
    assert gm is not None and gm.unit == "ratio" and set(gm.evidence_ids) == {income.evidence_id, ratios.evidence_id}
    assert ledger.get("nvda", "revenue").period.fiscal_year == 2025
    assert ledger.summary()["conflicts"] == 0


def test_extract_claims_peer_rows_need_fiscal_year():
    reg = build_default_registry("mock")
    peers = reg.dispatch("peer_comparison", {"ticker": "NVDA"})
    assert extract_claims({peers.evidence_id: peers}, CFG) == []  # mock rows carry no fiscal year

    live_like = result("E9", "peer_comparison", {"ticker": "AAA", "matrix": [
        {"ticker": "AAA", "fiscal_year": 2024, "period_end": "2024-12-31",
         "values": {"revenue": 500, "gross_margin": 0.4, "pe_ratio": 20.0, "market_cap": 1e12}},
        {"ticker": "BBB", "values": {"revenue": 900}}]})
    claims = extract_claims({"E9": live_like}, CFG)
    assert {c.metric for c in claims} == {"revenue", "gross_margin"}
    assert claims[0].value == 500e6 and claims[0].provider == "peer_comparison:sec_xbrl"


def test_yfinance_provider_is_tier_two_and_peer_matches_are_not_independent():
    inc = fin("E1", "AAA", "Aaa Inc.", "income_statement", "millions", [(2024, {"revenue": 500})], "sec_xbrl")
    peer = result("E2", "peer_comparison", {"ticker": "AAA", "matrix": [
        {"ticker": "AAA", "fiscal_year": 2024, "values": {"revenue": 500}}]})
    yf = fin("E3", "AAA", "Aaa Inc.", "income_statement", "millions", [(2024, {"revenue": 501})], "yfinance")
    ledger = build_ledger({"E1": inc, "E2": peer})
    assert ledger.get("AAA", "revenue").status == "single_source"
    ledger = build_ledger({"E1": inc, "E2": peer, "E3": yf})
    c = ledger.get("AAA", "revenue")
    assert c.status == "verified" and c.source_tiers == [1, 2]


def test_triangulate_helper():
    out = triangulate(100.0, [(101.0, "E2"), (150.0, "E3")], 0.05)
    assert out == {"agreeing": ["E2"], "disagreeing": ["E3"], "range": (100.0, 150.0)}


# --- narrative ------------------------------------------------------------------------------------


def palantir_evidence() -> dict[str, ToolResult]:
    inc = fin("E3", "PLTR", "Palantir Technologies Inc.", "income_statement", "millions",
              [(2025, {"revenue": 3870, "net_income": 800}), (2024, {"revenue": 3000, "net_income": 460})])
    return {"E3": inc, "E7": news("E7", "Palantir", -0.21, "negative")}


def test_sentiment_fact_alignment_palantir():
    evidence = palantir_evidence()
    ledger = build_ledger(evidence)
    threads = sentiment_fact_alignment(ledger, evidence)
    assert len(threads) == 1
    t = threads[0]
    assert t.theme == "sentiment_vs_fundamentals" and t.entity == "PLTR"
    assert "negative (-0.21 across 14 articles" in t.thesis
    assert "revenue grew 29.0% in FY2025" in t.thesis and "[E3]" in t.thesis
    assert t.supporting == ["E3"] and t.contradicting == ["E7"]
    assert t.source_types == {"news_sentiment", "financial_data_api"}
    assert "valuation" in t.so_what


def test_no_divergence_when_news_agrees_or_is_unrelated():
    evidence = palantir_evidence()
    evidence["E7"] = news("E7", "Palantir", 0.4, "positive")
    assert sentiment_fact_alignment(build_ledger(evidence), evidence) == []
    evidence["E7"] = news("E7", "Tesla", -0.4, "negative")
    assert sentiment_fact_alignment(build_ledger(evidence), evidence) == []


def test_reverse_divergence_positive_news_falling_revenue():
    inc = fin("E1", "AAA", "Aaa Inc.", "income_statement", "millions",
              [(2025, {"revenue": 900}), (2024, {"revenue": 1000})])
    evidence = {"E1": inc, "E2": news("E2", "AAA stock", 0.3, "positive", count=5)}
    threads = sentiment_fact_alignment(build_ledger(evidence), evidence)
    assert len(threads) == 1 and "revenue fell 10.0%" in threads[0].thesis


def test_build_threads_use_real_evidence_ids_only():
    reg = build_default_registry("mock")
    evidence = {}
    for st in ("income_statement", "key_ratios", "cash_flow"):
        r = reg.dispatch("financial_data_api", {"ticker": "NVDA", "statement_type": st})
        evidence[r.evidence_id] = r
    peers = reg.dispatch("peer_comparison", {"ticker": "NVDA"})
    evidence[peers.evidence_id] = peers
    ledger = build_ledger(evidence)
    threads = build_threads(ledger, evidence)
    themes = {t.theme for t in threads}
    assert {"growth", "profitability", "competition", "capital_allocation"} <= themes
    valid = {r.evidence_id for r in evidence.values()}
    for t in threads:
        assert t.supporting and set(t.supporting) | set(t.contradicting) <= valid
        assert t.source_types <= {r.tool for r in evidence.values()}
    growth = next(t for t in threads if t.theme == "growth")
    assert "FY2023" in growth.thesis and "FY2025" in growth.thesis


def test_threads_only_when_data_exists():
    assert build_threads(build_ledger({}), {}) == []


def test_render_brief_respects_token_budget():
    from memory.context_manager import count_tokens

    evidence = palantir_evidence()
    evidence["E9"] = fin("E9", "PLTR", "Palantir Technologies Inc.", "income_statement", "millions",
                         [(2025, {"revenue": 4400}), (2024, {"revenue": 3000})], "yfinance")
    ledger = build_ledger(evidence)
    threads = build_threads(ledger, evidence)
    full = render_brief(ledger, threads, max_tokens=5000)
    assert full.startswith("SYNTHESIS BRIEF (resolved facts; cite the evidence ids)")
    assert "$3.87 billion" in full and "Conflicts resolved:" in full and "[sentiment_vs_fundamentals]" in full
    small = render_brief(ledger, threads, max_tokens=120)
    assert count_tokens(small) <= 120
    assert "[sentiment_vs_fundamentals]" in small or "chose" in small  # high-priority lines survive longest
    assert count_tokens(full) > count_tokens(small)


def test_conflicts_markdown():
    assert conflicts_markdown([]) == ""
    _, conflicts = resolve([claim("C1", 100e9, "sec_xbrl", 1, "E1"), claim("C2", 110e9, "yfinance", 2, "E2")], CFG)
    md = conflicts_markdown(conflicts)
    lines = md.splitlines()
    assert lines[0] == "| Entity | Metric | Period | Values (source) | Resolution | Confidence |"
    assert lines[1].startswith("|---")
    assert "ACME" in lines[2] and "FY2024" in lines[2] and "highest_tier" in lines[2]
    assert "$100.00 billion (sec_xbrl, tier 1, E1)" in lines[2] and "0.95" in lines[2]
