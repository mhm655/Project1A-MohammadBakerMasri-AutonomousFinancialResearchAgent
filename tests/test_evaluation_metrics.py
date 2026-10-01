"""Tests for evaluation/metrics.py (spec A5.2): each metric's core rule on small hand-built bundles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation import metrics as m

ROOT = Path(__file__).resolve().parent.parent


def _ev(eid: str, tool: str, text: str = "", **extra):
    rec = {"evidence_id": eid, "tool": tool, "args": {}, "data": {"text": text}, "source_tier": 1,
           "retrieved_at": "", "latency_ms": 1.0, "cache_hit": False, "is_mock": False, "is_fallback": False}
    rec.update(extra)
    return rec


def _bundle(**overrides):
    base = {
        "challenge": {"id": "C1", "number": 1, "title": "t", "expected_pages": [3, 5]},
        "query": "Profile of Microsoft",
        "template": "company_profile",
        "profile": {"tickers": ["MSFT"], "template": "company_profile"},
        "duration_s": 12.0,
        "tool_calls": 2,
        "report_markdown": "# Report\n",
        "sections": {"executive_summary": "Revenue was $245.1 billion in FY2024 [E1]."},
        "plan": {"steps": []},
        "evidence": {"E1": _ev("E1", "web_search", "Microsoft revenue was $245.1 billion in FY2024.")},
        "call_log": [],
    }
    base.update(overrides)
    return base


def _score(bundle, judgement=None):
    return {r.id: r for r in m.score_run(m.Run.from_bundle(bundle), judgement)}


def _body(text, **bundle_kw):
    return _bundle(sections={"executive_summary": text}, **bundle_kw)


def test_fa1_supported_vs_unsupported():
    b = _body("Revenue was $245.1 billion in FY2024 [E1]. Net income was $999.0 billion in FY2024 [E1].")
    r = _score(b)["FA-1"]
    assert r.value == 0.5 and r.passed is False
    assert "999.0" in r.examples[0]


def test_fa1_none_without_numeric_claims():
    r = _score(_body("The company is large and growing [E1]."))["FA-1"]
    assert r.value is None and r.passed is None


def test_fa5_hallucination_count():
    r = _score(_body("Revenue was $245.1 billion [E1]. Profit was $77.7 billion [E1]. Margin was 12.3% [E1]."))["FA-5"]
    assert r.value == 2 and r.passed is False


def test_fa5_zero_passes():
    r = _score(_body("Revenue was $245.1 billion in FY2024 [E1]."))["FA-5"]
    assert r.value == 0 and r.passed is True


def test_fa2_unknown_citation_fails():
    r = _score(_body("Revenue was $245.1 billion in FY2024 [E1]. Sales rose 5% [E9]."))["FA-2"]
    assert r.value == 0.5
    assert any("E9" in e for e in r.examples)


def test_fa3_temporal_mismatch():
    r = _score(_body("Revenue was $245.1 billion in FY2023 [E1]."))["FA-3"]
    assert r.value == 0.0 and r.passed is False
    assert _score(_body("Revenue was $245.1 billion in FY2024 [E1].", ))["FA-3"].value == 1.0


def test_fa3_quarter_accepts_three_months_ended():
    ev = {"E1": _ev("E1", "web_search", "For the three months ended Sept 2025 revenue was $5.0 billion in 2025.")}
    r = _score(_body("Q3 2025 revenue was $5.0 billion [E1].", evidence=ev))["FA-3"]
    assert r.value == 1.0


def test_fa4_entities_verified_and_not():
    ev = {"E1": _ev("E1", "web_search", "Satya Nadella is CEO of Microsoft (MSFT).")}
    ok = _score(_body("Microsoft (MSFT) is led by CEO Satya Nadella [E1].", evidence=ev))["FA-4"]
    assert ok.value == 1.0
    bad = _score(_body("Microsoft (MSFT) is led by CEO John Smithers [E1].", evidence=ev))["FA-4"]
    assert bad.value < 1.0 and any("John Smithers" in e for e in bad.examples)


def test_co1_section_coverage():
    sections = {"executive_summary": "x", "business_overview": "y", "financial_summary": " "}
    r = _score(_bundle(sections=sections))["CO-1"]
    assert r.value == pytest.approx(2 / 5)


def test_co2_source_diversity_ignores_calculation():
    ev = {"E1": _ev("E1", "web_search", "a"), "E2": _ev("E2", "sec_filing_search", "b"),
          "E3": _ev("E3", "calculation_engine", "c"), "E4": _ev("E4", "peer_comparison", "d")}
    r = _score(_body("A [E1]. B [E2]. C [E3]. D [E4].", evidence=ev))["CO-2"]
    assert r.value == 3


def test_co3_temporal_span_includes_table_header():
    body = "Revenue was $245.1 billion in FY2024 [E1].\n\n| Metric | FY2022 | FY2024 |\n|---|---|---|\n| x | 1 | 2 |"
    assert _score(_body(body))["CO-3"].value == 3


def test_co4_risk_coverage_none_without_filing():
    assert _score(_body("Competition is intense [E1]."))["CO-4"].value is None


def test_co4_risk_coverage_ratio():
    ev = {"E1": _ev("E1", "sec_filing_search", data={"risk_factors": [
        "We face competition", "cyber attacks", "climate change", "regulation"], "sections": []})}
    r = _score(_body("Competition is intense [E1]. Regulation matters [E1].", evidence=ev))["CO-4"]
    assert r.value == pytest.approx(2 / 4)  # competition, regulation/legal, cyber, climate; report has 2


def test_ad2_cross_source_count():
    ev = {"E1": _ev("E1", "web_search", "a"), "E2": _ev("E2", "sec_filing_search", "b"),
          "E3": _ev("E3", "web_search", "c")}
    r = _score(_body("Both agree [E1, E2]. Same tool [E1, E3]. Single [E1].", evidence=ev))["AD-2"]
    assert r.value == 1


def test_ad3_counts_original_calculations():
    ev = {"E1": _ev("E1", "web_search", "Revenue was $100 million.")}
    text = "Operating margin was 31.4% [E1]. Revenue growth reached 17.8% year-over-year [E1]."
    r = _score(_body(text, evidence=ev))["AD-3"]
    assert r.value == 2


def test_ad3_ignores_percent_found_in_sources():
    ev = {"E1": _ev("E1", "web_search", "Margin was 31.4% last year.")}
    r = _score(_body("Operating margin was 31.4% [E1].", evidence=ev))["AD-3"]
    assert r.value == 0


def test_ad4_forward_sections():
    sections = {"executive_summary": "x", "outlook": "y",
                "risks": "We expect growth. Management will raise guidance."}
    assert _score(_bundle(sections=sections))["AD-4"].value == 2


def test_cs2_detects_same_metric_contradiction():
    text = "Revenue was $245.1 billion in FY2024 [E1]. Revenue was $300.0 billion in FY2024 [E1]."
    r = _score(_body(text))["CS-2"]
    assert r.value == 1 and r.passed is False


def test_cs2_same_value_different_scale_is_not_contradiction():
    text = "Revenue was $245.1 billion in FY2024 [E1]. Revenue was 245,100 million in FY2024 [E1]."
    assert _score(_body(text))["CS-2"].value == 0


def test_cs2_adds_judge_contradictions():
    j = {"contradictions": [{"a": "x", "b": "y", "why": "z"}]}
    assert _score(_body("Fine [E1]."), j)["CS-2"].value == 1


def test_cs4_artefacts_fail():
    report = "# R\n\n## Executive Summary\n\nMOCK RESPONSE and None here\n\n## Sources\n"
    r = _score(_bundle(report_markdown=report, sections={"executive_summary": "MOCK RESPONSE"}))["CS-4"]
    assert any("artefacts" in e for e in r.examples)


def test_cs4_clean_report_passes_artefact_check():
    report = "# R\n\n## Executive Summary\n\nNone of the risks changed.\n\n## Sources\n"
    r = _score(_bundle(report_markdown=report))["CS-4"]
    assert not any("artefacts" in e for e in r.examples)


def test_judge_metrics_none_without_judgement():
    res = _score(_body("Fine [E1]."))
    for mid in ("AD-1", "CS-1", "CS-3", "AB-3"):
        assert res[mid].value is None and res[mid].passed is None
        assert "LLM judge not run" in res[mid].detail


def test_judgement_mapping():
    j = {"insights": ["a", "b", "c", "d"], "insight_count": 4,
         "logical_flow": {"score": 4, "rationale": "ok"},
         "exec_summary": {"score": 3, "rationale": "meh", "missed_findings": ["m"]},
         "planning": {"score": 5, "rationale": "good"}, "contradictions": []}
    res = _score(_body("Revenue was $245.1 billion in FY2024 [E1]."), j)
    assert res["AD-1"].value == 4.0 and res["AD-1"].examples == ["a", "b", "c"]  # 1 page minimum
    assert res["CS-1"].value == 4 and res["CS-1"].passed is True
    assert res["CS-3"].value == 3 and res["CS-3"].passed is False
    assert res["AB-3"].value == 5 and "coverage" in res["AB-3"].detail


def test_ab1_tool_efficiency():
    plan = {"steps": [{"id": "s1", "tool": "web_search", "evidence_ids": ["E1", "E2"], "status": "done"}]}
    r = _score(_body("Revenue was $245.1 billion [E1].", plan=plan, tool_calls=4))["AB-1"]
    assert r.value == pytest.approx(0.25)


def test_ab2_recovery_by_retry_and_fallback():
    log = [{"tool": "web_search", "args": {"q": "a"}, "ok": False}, {"tool": "web_search", "args": {"q": "a"}, "ok": True},
           {"tool": "sec_filing_search", "args": {"t": "x"}, "ok": False},
           {"tool": "company_profile", "args": {}, "ok": False}]
    ev = {"E1": _ev("E1", "web_search", "a"),
          "E2": _ev("E2", "financial_data_api", "b", is_fallback=True)}
    ev["E2"]["data"]["fallback_for"] = "sec_filing_search"
    r = _score(_body("A [E1].", call_log=log, evidence=ev))["AB-2"]
    assert r.value == pytest.approx(2 / 3, abs=1e-3)


def test_ab2_repaired_arguments_count_as_recovered():
    # The planner passed a metric name; the agent re-called the tool with the numbers.
    log = [{"tool": "calculation_engine", "args": {"current": "revenue_FY2025"}, "ok": False,
            "error_kind": "invalid_arguments"},
           {"tool": "calculation_engine", "args": {"current": 94827, "previous": 97690}, "ok": True},
           {"tool": "web_search", "args": {"q": "x"}, "ok": False, "error_kind": "http_error"},
           {"tool": "web_search", "args": {"q": "other"}, "ok": True}]
    r = _score(_body("A [E1].", call_log=log))["AB-2"]
    assert r.value == pytest.approx(0.5)  # a network error is not "repaired" by an unrelated call


def test_derived_percentages_are_traceable_but_coincidences_are_not():
    ev = {"E1": _ev("E1", "web_search", "Net income 7,091 million in FY2024; 3,794 million in FY2025.")}
    derived = _body("Net income fell from $7,091 million in FY2024 to $3,794 million in FY2025 [E1]. "
                    "That is a 46.5% decline [E1].", evidence=ev)
    scores = _score(derived)
    assert scores["FA-5"].value == 0 and scores["FA-1"].value == 1.0
    # Same claim without the input figures in its section: not traceable.
    assert _score(_body("Net income fell 46.5% [E1].", evidence=ev))["FA-5"].value == 1
    # A whole-number percentage is too coarse to prove a derivation.
    assert _score(_body("Net income was $7,091 million and $3,794 million [E1]. It fell 46% [E1].",
                        evidence=ev))["FA-5"].value == 1


def test_cs2_ignores_segment_figures_and_names_that_look_like_numbers():
    b = _body("FY2026 revenue was $331.8 billion [E1]. Azure revenue was $75.0 billion in FY2026 [E1]. "
              "Revenue in FY2026 benefited from the partnership with 3M [E1]. Revenue of $331.8 billion in FY2026 "
              "led the sector [E1].")
    assert _score(b)["CS-2"].value == 0
    clash = _body("FY2026 revenue was $331.8 billion [E1]. Revenue was $300.0 billion in FY2026 [E1].")
    assert _score(clash)["CS-2"].value == 1


def test_fa4_tolerates_irregular_spacing_in_sources():
    ev = {"E1": _ev("E1", "company_profile", "Carolina  Dybeck Happe, Executive VP & COO")}
    r = _score(_body("Carolina Dybeck Happe, the COO, leads operations [E1].", evidence=ev))["FA-4"]
    assert r.value == 1.0


def test_fa4_shortened_names_pass_but_wrong_initials_and_invented_names_fail():
    ev = {"E1": _ev("E1", "company_profile", "Jane Nind Fraser Ph.D., CEO; Brian Thomas Moynihan, Chairman & CEO")}
    ok = _score(_body("Jane Fraser, CEO, runs the bank [E1].", evidence=ev))["FA-4"]
    wrong = _score(_body("Brian M. Moynihan, Chairman & CEO, leads the bank [E1].", evidence=ev))["FA-4"]
    invented = _score(_body("Charlie Schwab, CEO, leads the bank [E1].", evidence=ev))["FA-4"]
    assert ok.passed is True
    assert wrong.passed is False and invented.passed is False


def test_ab2_none_without_errors():
    assert _score(_body("A [E1]."))["AB-2"].value is None


def test_ab4_is_ratio_not_product():
    log = [{"tool": "web_search", "args": {}, "ok": True, "cache_hit": False},
           {"tool": "sec_filing_search", "args": {}, "ok": True, "cache_hit": False},
           {"tool": "company_profile", "args": {}, "ok": True, "cache_hit": True}]
    ev = {"E1": _ev("E1", "vector_db_search", "", data={"results": [{"content": "x", "metadata": {}}]})}
    ev["E1"]["data"] = {"results": [{"content": "x", "metadata": {}}]}
    r = _score(_body("A [E1].", call_log=log, evidence=ev))["AB-4"]
    assert r.value == pytest.approx(2 / 2)  # 1 recall + 1 cache hit over 2 uncached external calls
    assert r.value <= 10


def test_ab5_latency_thresholds():
    single = _score(_body("A [E1]."), None)["AB-5"]
    assert single.value == 12.0 and single.passed is True and single.target == "< 300 s"
    multi_b = _body("A [E1].", duration_s=400.0, template="comparative_analysis",
                    profile={"tickers": ["AMZN", "MSFT", "GOOGL"]})
    multi = _score(multi_b)["AB-5"]
    assert multi.target == "< 600 s" and multi.passed is True
    slow = _score(_body("A [E1].", duration_s=400.0))["AB-5"]
    assert slow.passed is False


def test_c8_mock_bundle_end_to_end():
    path = ROOT / "results" / "dev" / "runs_mock" / "C8.json"
    if not path.exists():
        pytest.skip("mock C8 bundle not present")
    results = m.score_run(m.Run.load(path))
    ids = [r.id for r in results]
    expected = [f"{p}-{n}" for p, k in (("FA", 5), ("CO", 4), ("AD", 4), ("CS", 4), ("AB", 5)) for n in range(1, k + 1)]
    assert ids == expected
    json.dumps([r.to_dict() for r in results])


def test_benchmark_appended():
    bench = m.MetricResult("BM-1", "Bench", "Benchmark", 0.5, ">= 0.5", True, "benchmark", "ratio")
    res = m.score_run(m.Run.from_bundle(_body("A [E1].")), None, bench)
    assert res[-1].id == "BM-1" and len(res) == 23


def _statement():
    rows = [("FY", 2026, 331839e6, 133749e6), ("FY", 2025, 281724e6, 101832e6)]
    data = {"company_name": "Microsoft", "ticker": "MSFT", "statement_type": "income_statement", "period": "annual",
            "unit": "millions", "fiscal_year_end": "0630",
            "data": [{"fiscal_period": fp, "fiscal_year": y, "period_end": f"{y}-06-30",
                      "values": {"revenue": rev / 1e6, "net_income": ni / 1e6}} for fp, y, rev, ni in rows]}
    return _ev("E1", "financial_data_api", data=data)


def test_margin_and_growth_derived_from_cited_statement_rows():
    # The report states only the computed ratios; inputs sit in the cited statement (C1/C3 live bundles).
    b = _body("Net margin was 40.3 % in FY2026 [E1]. Net income grew 31.3 % in FY2026 [E1].",
              evidence={"E1": _statement()})
    s = _score(b)
    assert s["FA-1"].value == 1.0 and s["FA-5"].value == 0


def test_wrong_margin_not_derived_from_statement_rows():
    b = _body("Net margin was 52.7 % in FY2026 [E1].", evidence={"E1": _statement()})
    assert _score(b)["FA-1"].value == 0.0


def test_margin_change_in_points_derived_from_statement_rows():
    # 133,749/331,839 = 40.31 % vs 101,832/281,724 = 36.15 %: a 4.2-point change; 9.9 points is not.
    ev = {"E1": _statement()}
    ok = _score(_body("Net margin improved by 4.2 percentage points in FY2026 [E1].", evidence=ev))["FA-1"]
    bad = _score(_body("Net margin improved by 9.9 percentage points in FY2026 [E1].", evidence=ev))["FA-1"]
    assert ok.value == 1.0 and bad.value == 0.0


def test_fa3_calculation_engine_citation_is_period_agnostic():
    # A calculator record holds only inputs and the result, never fiscal years: the year is checked against
    # the data the inputs came from (C6 live bundle: "JPM CAGR 7.4% over FY2023-FY2025 [E20]").
    ev = {"E1": _ev("E1", "web_search", "JPM revenue FY2023: 158,104; FY2025: 182,447"),
          "E2": _ev("E2", "calculation_engine", "cagr(begin_value=158104, end_value=182447, years=2) = 0.0742 (= 7.42%)")}
    ok = _score(_body("JPM revenue CAGR was 7.4% over FY2023-FY2025 [E2].", evidence=ev))["FA-3"]
    bad = _score(_body("JPM revenue CAGR was 7.4% over FY2019-FY2021 [E2].", evidence=ev))["FA-3"]
    assert ok.value == 1.0 and bad.value == 0.0


def test_co3_ignores_stale_outlier_years():
    # C6 live bundle: stale peer data points (FY2014, FY2019) stretched 2023-2025 to a 12-year "history".
    text = ("Revenue was $10 billion in FY2025 [E1]. Revenue was $9 billion in FY2024 [E1]. "
            "Revenue was $8 billion in FY2023 [E1]. Revenue was $5 billion in FY2014 [E1]. "
            "Revenue was $6 billion in FY2019 [E1].")
    assert _score(_body(text))["CO-3"].value == 3


def test_cs2_possessive_company_name_does_not_hide_contradiction():
    # "Microsoft's revenue" used to get the qualifier "s", so it never met "Revenue" for the same year.
    text = ("Microsoft’s revenue was $245.1 billion in FY2024 [E1]. "
            "Total revenues were $300.0 billion in FY2024 [E1].")
    assert _score(_body(text))["CS-2"].value == 1
    segment = ("AWS’s revenue was $100.0 billion in FY2024 [E1]. Revenue was $300.0 billion in FY2024 [E1].")
    assert _score(_body(segment))["CS-2"].value == 0  # a segment is not the company total


def test_ab5_multi_company_detected_from_tool_arguments():
    # C6 ("What's happening with the banks?") resolves to several tickers only at run time: profile.tickers is
    # empty and the template is company_profile, yet five banks were researched.
    ev = {"E1": _ev("E1", "financial_data_api", args={"ticker": "JPM"}),
          "E2": _ev("E2", "financial_data_api", args={"ticker": "BAC"})}
    b = _bundle(profile={"tickers": [], "template": "company_profile"}, evidence=ev, duration_s=400.0)
    r = _score(b)["AB-5"]
    assert r.target == "< 600 s" and r.passed is True

def test_derivation_pool_is_the_paragraph_not_a_dense_table_elsewhere_in_the_section():
    # Day 11 C6: a peer table in the same section made "-1.7 pp" look derivable from unrelated pairs.
    ev = {"E1": _ev("E1", "web_search", "Revenue 100 million; costs 98.3 million.")}
    table = "| Bank | Revenue | Costs |\n|---|---|---|\n| A | $100 million | $98.3 million |"
    claim = "Margins contracted 1.7% at the bank [E1]."
    assert _score(_body(f"{table}\n\n{claim}", evidence=ev))["FA-5"].value == 1
    assert _score(_body(f"Revenue was $100 million and costs $98.3 million; {claim}", evidence=ev))["FA-5"].value == 0
