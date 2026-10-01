"""Tests for evaluation/judge.py (LLM-as-judge) and evaluation/benchmark.py (BM-1). No network, no live LLM."""

from __future__ import annotations

import json

import pytest

from agent.llm import LLMClient, MockProvider
from evaluation import judge
from evaluation.benchmark import compare, gold_for

VALID = {
    "insights": ["Margin fell while revenue rose, implying cost growth outpaced sales.", "  "],
    "insight_count": 99,
    "logical_flow": {"score": 4, "rationale": "ordered"},
    "exec_summary": {"score": 3, "rationale": "ok", "missed_findings": ["cloud growth"]},
    "planning": {"score": 4, "rationale": "covered"},
    "contradictions": [{"a": "rapidly growing", "b": "stagnating", "why": "same segment"}],
}


def bundle(report: str = "## Executive Summary\nText.\n\n## Sources\n[1] x", **kw) -> dict:
    return {"query": "Profile Microsoft", "template": "company_profile", "challenge": {"id": "C1"},
            "sections": {"executive_summary": "Text.", "sources": "[1] x"}, "report_markdown": report,
            "plan": {"steps": [{"id": "s1", "goal": "Get income statements", "tool": "financial_data_api",
                                "status": "done", "summary": "ok"}]}, **kw}


def client(script) -> tuple[LLMClient, MockProvider]:
    provider = MockProvider(script)
    return LLMClient([provider], max_retries=0, sleep=lambda s: None), provider


def test_judge_normalises_reply(tmp_path):
    llm, provider = client([json.dumps(VALID)])
    out = judge.judge_bundle(bundle(), llm=llm, cache_path=tmp_path / "c.json")
    assert out["insights"] == ["Margin fell while revenue rose, implying cost growth outpaced sales."]
    assert out["insight_count"] == 1  # recomputed, not trusted
    assert out["logical_flow"]["score"] == 4
    assert out["exec_summary"]["missed_findings"] == ["cloud growth"]
    assert out["contradictions"][0]["why"] == "same segment"
    assert out["provider"] == "mock" and out["model"] == "mock-llm"
    request = provider.requests[0]
    assert request.json_mode and request.temperature == 0
    prompt = request.messages[0]["content"]
    assert "Profile Microsoft" in prompt and "financial_data_api" in prompt
    assert "[1] x" not in prompt  # the sources section is skipped


def test_judge_clamps_scores_and_defaults_missing_keys(tmp_path):
    raw = {"logical_flow": {"score": 9}, "exec_summary": {"score": "0"}, "planning": {"score": 3.6}}
    out = judge.judge_bundle(bundle(), llm=client([json.dumps(raw)])[0], cache_path=None)
    assert out["logical_flow"]["score"] == 5
    assert out["exec_summary"]["score"] == 1
    assert out["planning"]["score"] == 4
    assert out["insights"] == [] and out["insight_count"] == 0 and out["contradictions"] == []
    empty = judge.judge_bundle(bundle(), llm=client(["{}"])[0], cache_path=None)
    assert empty["logical_flow"]["score"] == 3 and empty["exec_summary"]["missed_findings"] == []


def test_judge_retries_once_on_malformed_reply(tmp_path):
    llm, provider = client(["sorry, here is my view", json.dumps(VALID)])
    out = judge.judge_bundle(bundle(), llm=llm, cache_path=None)
    assert out["logical_flow"]["score"] == 4
    assert len(provider.requests) == 2
    assert "not valid JSON" in provider.requests[1].messages[-1]["content"]


def test_judge_raises_after_two_malformed_replies():
    llm, provider = client(["nope", "still nope"])
    with pytest.raises(judge.JudgeError):
        judge.judge_bundle(bundle(), llm=llm, cache_path=None)
    assert len(provider.requests) == 2


def test_judge_cache_hit_skips_llm_and_force_bypasses(tmp_path):
    path = tmp_path / "eval" / "cache.json"
    llm, provider = client([json.dumps(VALID)])
    first = judge.judge_bundle(bundle(), llm=llm, cache_path=path)
    assert path.exists() and not list(path.parent.glob("*.tmp"))
    again = judge.judge_bundle(bundle(), llm=llm, cache_path=path)
    assert again == first and len(provider.requests) == 1
    judge.judge_bundle(bundle(), llm=llm, cache_path=path, force=True)
    assert len(provider.requests) == 2
    # A different report is a different key.
    judge.judge_bundle(bundle("other report"), llm=llm, cache_path=path)
    assert len(provider.requests) == 3


def test_long_reports_are_cut_proportionally():
    big = {"a": "x" * 100_000, "b": "y" * 5_000, "c": "z" * 100_000}
    sections, cut = judge._fit_sections(list(big.items()), 40_000)
    assert cut and sum(len(b) for _, b in sections) <= 40_100
    assert len(dict(sections)["b"]) == 5_000  # short sections stay whole
    prompt = judge.build_prompt(bundle(sections=big))
    assert "shortened" in prompt and len(prompt) < 45_000


# --- benchmark ---------------------------------------------------------------------------------------

GOLD = {"company": "Microsoft", "ticker": "MSFT", "facts": [
    {"id": "rev", "kind": "numeric", "text": "Revenue FY2025 $281.7B", "value": 281724e6, "unit": "USD",
     "period": "FY2025", "keywords": ["revenue"]},
    {"id": "gm", "kind": "numeric", "text": "Gross margin 67.9%", "value": 67.9, "unit": "percent",
     "period": "FY2026", "keywords": ["gross margin"]},
    {"id": "eps", "kind": "numeric", "text": "EPS $17.95", "value": 17.95, "unit": "USD/share",
     "period": "FY2026", "keywords": ["EPS", "diluted"]},
    {"id": "oi", "kind": "numeric", "text": "Operating income $155.2B", "value": 155237e6, "unit": "USD",
     "period": "FY2026", "keywords": ["operating income"]},
    {"id": "ceo", "kind": "entity", "text": "CEO", "value": "Satya Nadella", "keywords": ["CEO"]},
    {"id": "cfo", "kind": "entity", "text": "CFO", "value": "Amy Hood", "keywords": ["CFO"]},
    {"id": "seg", "kind": "qualitative", "text": "segments", "value": None,
     "keywords": ["Intelligent Cloud", "Productivity", "Personal Computing", "segment"]},
]}


def test_compare_recall_scale_percent_entity_qualitative():
    report = (
        "Microsoft reported revenue of $281.7 billion in fiscal 2025.\n\n"
        "Gross margin was 67.9% in FY2026, and diluted EPS reached $17.95 for fiscal 2026.\n\n"
        "Satya Nadella is CEO; the CFO is Amy E. Hood.\n\n"
        "It reports three segments, including Intelligent Cloud and Productivity and Business Processes.\n\n"
        "| Metric | FY2026 |\n|---|---|\n| Operating income | $155,237 million |\n"
    )
    result = compare({"report_markdown": report}, GOLD)
    assert result.id == "BM-1" and result.category == "Benchmark" and result.method == "benchmark"
    assert result.unit == "ratio" and result.target == ">= 70%"
    assert result.value == 1.0 and result.passed is True
    assert result.detail.startswith("7 of 7 gold facts recalled; 0 conflicting")


def test_compare_detects_conflict_and_missing_period():
    report = (
        "Revenue was $250.0 billion in fiscal 2025.\n\n"  # 11 % off: conflict
        "Gross margin was 67.9% last quarter.\n\n"  # right value, wrong period: not recalled, not a conflict
        "Diluted EPS was $17.95 in 2026.\n"
    )
    result = compare({"report_markdown": report}, GOLD)
    assert result.value == pytest.approx(1 / 7)
    assert result.passed is False
    assert "1 conflicting" in result.detail
    assert result.examples == ["report says $250.0 billion, gold says Revenue FY2025 $281.7B [rev]"]


def test_compare_segment_figures_are_not_conflicts():
    gold = {"facts": [{"id": "g", "kind": "numeric", "text": "Q3 revenue growth 16.4%", "value": 16.4,
                       "unit": "percent", "period": "Q3 FY2026", "keywords": ["revenue", "growth"]}]}
    segment = "Mac delivered $10.4 billion in revenue for the fiscal third quarter of 2026, representing a 29% increase."
    assert compare({"report_markdown": segment}, gold).examples == []
    wrong = compare({"report_markdown": "Revenue growth was 12.0% in fiscal Q3 2026."}, gold)
    assert "1 conflicting" in wrong.detail


def test_compare_rounding_aware():
    gold = {"facts": [{"id": "x", "kind": "numeric", "text": "OI", "value": 4355e6, "unit": "USD",
                       "period": "FY2025", "keywords": ["operating income"]}]}
    hit = compare({"report_markdown": "Operating income was $4.4 billion in FY2025."}, gold)
    assert hit.value == 1.0
    miss = compare({"report_markdown": "Operating income was $4.9 billion in FY2025."}, gold)
    assert miss.value == 0.0 and "1 conflicting" in miss.detail


# --- gold facts --------------------------------------------------------------------------------------

def test_gold_facts_file_is_well_formed():
    for cid, company in (("C1", "Microsoft"), ("C2", "Apple"), ("C3", "Tesla")):
        gold = gold_for(cid)
        assert gold and gold["company"] == company and gold["ticker"]
        facts = gold["facts"]
        assert len(facts) >= 10
        assert len({f["id"] for f in facts}) == len(facts)
        for f in facts:
            assert {"id", "kind", "text", "value", "unit", "period", "keywords"} <= set(f)
            assert f["kind"] in {"numeric", "entity", "qualitative"} and f["keywords"]
            if f["kind"] == "numeric":
                assert isinstance(f["value"], (int, float)) and f["unit"] in {"USD", "percent", "USD/share"}
                assert f["period"]
            if f["kind"] == "qualitative":
                assert 3 <= len(f["keywords"]) <= 5
    assert gold_for("C99") is None


def test_compare_possessive_company_name_is_not_a_segment_qualifier():
    # "Microsoft's revenue" used to read as a segment ("s revenue") and so never counted as a conflict.
    report = "Microsoft’s revenue was $250.0 billion in fiscal 2025.\n"
    result = compare({"report_markdown": report}, GOLD)
    assert result.examples == ["report says $250.0 billion, gold says Revenue FY2025 $281.7B [rev]"]
