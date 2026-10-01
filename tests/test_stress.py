"""Tests for the Day 12 stress harness and token profiler (mock mode only; no network)."""

from __future__ import annotations

from evaluation import stress, token_profile


def _call(purpose: str, prompt: int, completion: int, reasoning: int = 0, latency: float = 100.0) -> dict:
    return {"type": "llm_call", "purpose": purpose, "provider": "p", "model": "m", "prompt_tokens": prompt,
            "completion_tokens": completion, "reasoning_tokens": reasoning, "latency_ms": latency}


def _bundle(cid: str, calls: list[dict], failures: int = 0, words: int = 100) -> dict:
    return {"challenge": {"id": cid}, "trace_events": calls, "llm_usage": {"failures": failures},
            "report_markdown": "word " * words, "stop_reason": "done"}


def test_purpose_group_normalises() -> None:
    assert token_profile.purpose_group("execute:s3") == "execute"
    assert token_profile.purpose_group("synthesize_section") == "synthesize"
    assert token_profile.purpose_group("weird") == "other"
    assert token_profile.purpose_group(None) == "other"


def test_profile_totals_shares_and_opportunities() -> None:
    bundles = [
        _bundle("C1", [_call("plan", 1000, 100), _call("execute:s1", 4000, 100), _call("execute:s2", 4000, 100),
                       _call("synthesize", 5500, 500, reasoning=400)], failures=5, words=200),
        _bundle("C2", [_call("review", 500, 50)], words=50),
    ]
    prof = token_profile.profile(bundles)
    o = prof["overall"]
    assert o["calls"] == 5
    assert o["prompt"] == 15000 and o["completion"] == 850 and o["reasoning"] == 400
    assert o["groups"]["execute"]["calls"] == 2 and o["groups"]["execute"]["prompt"] == 8000
    assert abs(sum(g["share"] for g in o["groups"].values()) - 1.0) < 1e-9
    assert prof["per_challenge"][0]["tokens_per_word"] == prof["per_challenge"][0]["total"] / 200
    assert prof["top_calls"][0]["purpose"] == "synthesize"
    text = " ".join(prof["opportunities"])
    assert "execute calls average" in text        # mean execute prompt 4000 > 3000
    assert "LLM failures" in text                 # 5 > 3
    assert "Reasoning tokens" in text             # 400 / 1250 > 30%
    assert "synthesis prompt" in text             # 5500 >= 85% of 6000


def test_render_preserves_analysis_after_marker() -> None:
    prof = token_profile.profile([_bundle("C1", [_call("plan", 10, 5)])])
    first = token_profile.render(prof)
    assert token_profile.MARKER in first and "_To be written after review._" in first
    edited = first.replace("_To be written after review._", "Hand-written conclusions.")
    again = token_profile.render(prof, existing=edited)
    assert "Hand-written conclusions." in again and "_To be written" not in again
    assert again.count(token_profile.MARKER) == 1


def test_profile_handles_no_calls() -> None:
    prof = token_profile.profile([_bundle("C1", [])])
    assert "No LLM call events" in prof["opportunities"][0]
    assert token_profile.render(prof)


def test_outage_degrades_without_crash() -> None:
    res = stress.run_outage(("C1",))
    m = res.metrics
    assert "C1.exception" not in m, res.findings
    assert m["C1.gaps"] > 0 and m["C1.gaps_section_chars"] > 0
    assert m["C1.llm_calls"] <= stress.OUTAGE_MAX_LLM_CALLS
    assert m["C1.injected_failures"] > 0
    assert res.passed, res.findings


def test_concurrency_two_tasks_no_crosstalk() -> None:
    res = stress.run_concurrency(2)
    m = res.metrics
    assert m["completed"] == 2 and m["exceptions"] == 0
    assert m["distinct_session_ids"] == 2 and m["distinct_trace_files"] == 2
    assert m["cross_talk_problems"] == 0, res.findings
    assert res.passed, res.findings


def test_context_small_filler_respects_limit() -> None:
    res = stress.run_context(100_000)
    m = res.metrics
    assert m["requests_over_limit"] == 0, res.findings
    assert m["report_produced"]
    assert res.passed, res.findings


def test_render_report_layout() -> None:
    text = stress.render([stress.StressResult("X", True, {"a": 1}, ["f"], 0.1, method="m")])
    assert "## Summary" in text and "| X | PASS |" in text and "## Live-mode note" in text and "none found" in text


def test_purpose_group_handles_continuation_and_repair_purposes() -> None:
    assert token_profile.purpose_group("synthesize:continue") == "synthesize"
    assert token_profile.purpose_group("execute:r10") == "execute"


def test_failure_opportunity_does_not_claim_rejected_calls_cost_tokens() -> None:
    prof = token_profile.profile([_bundle("C1", [_call("plan", 1000, 100)], failures=9)])
    text = " ".join(prof["opportunities"])
    assert "LLM failures" in text and "re-pay" not in text and "latency" in text


def test_env_is_restored_after_an_exception() -> None:
    import os

    os.environ["ARA_FAIL_RATE"] = "0.5"
    os.environ.pop("ARA_MODE", None)
    try:
        with stress._mock_env(ARA_FAIL_RATE="1.0"):
            assert os.environ["ARA_MODE"] == "mock" and os.environ["ARA_FAIL_RATE"] == "1.0"
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert os.environ["ARA_FAIL_RATE"] == "0.5" and "ARA_MODE" not in os.environ
    os.environ.pop("ARA_FAIL_RATE")


def test_analysis_text_survives_rerender_with_a_second_marker_and_crlf() -> None:
    prof = token_profile.profile([_bundle("C1", [_call("plan", 1000, 100)])])
    m = token_profile.MARKER
    edited = token_profile.render(prof).split(m)[0] + f"{m}\r\nMy analysis.\r\nquote: {m}\r\nmore\r\n"
    again = token_profile.render(prof, existing=edited)
    assert "My analysis." in again and "more" in again
    assert token_profile.render(prof, existing=again) == again  # stable on a further re-render
