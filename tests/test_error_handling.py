"""Tests for the Day 9 error-handling layer: classification, retries, breakers, fallbacks, injection.

Everything is deterministic: a fake clock drives the breakers, sleeps are recorded rather than slept,
and jitter comes from a stub random source. No network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agent.circuit_breaker import (
    TOOL_PROVIDERS,
    BreakerBoard,
    BreakerConfig,
    CircuitBreaker,
    provider_for,
)
from agent.error_handler import (
    ERROR_CLASSES,
    AllFallbacksFailed,
    FailureInjector,
    Gap,
    ResilientDispatcher,
    RetryPolicy,
    classify,
)
from agent.fallback_chains import FALLBACK_CHAINS, chain_for
from tools.base import (
    ToolError,
    ToolExecutionError,
    ToolLockedError,
    ToolNotFoundError,
    ToolOutputError,
    ToolResult,
    ToolUnavailableError,
    ToolValidationError,
)
from tools.tool_registry import build_default_registry

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "tools" / "schemas"
NETWORK_TOOLS = ["sec_filing_search", "financial_data_api", "company_profile", "earnings_transcript",
                 "peer_comparison", "web_search", "news_sentiment"]
LOCAL_TOOLS = ["calculation_engine", "fact_checker", "report_generator", "vector_db_search", "vector_db_store"]


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class NoJitter:
    def uniform(self, low: float, high: float) -> float:
        return 0.0


class Retryable(ToolExecutionError):
    """An error carrying a Retry-After hint, as the HTTP layer may attach."""

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message, retryable=True)
        self.retry_after = retry_after


def make_result(tool: str, args: dict[str, Any] | None = None, data: dict[str, Any] | None = None) -> ToolResult:
    return ToolResult(evidence_id="E1", tool=tool, args=args or {}, data=data if data is not None else {"ok": 1},
                      source_tier=1, retrieved_at="2026-01-01T00:00:00+00:00", latency_ms=1.0)


class Script:
    """A dispatch stub: per tool, a list of outcomes (exceptions are raised); the last one repeats."""

    def __init__(self, plan: dict[str, list[Any]] | None = None) -> None:
        self.plan = {tool: list(outcomes) for tool, outcomes in (plan or {}).items()}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, tool: str, args: dict[str, Any]) -> ToolResult:
        self.calls.append((tool, args))
        outcomes = self.plan.get(tool)
        if not outcomes:
            return make_result(tool, args)
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def tools_called(self) -> list[str]:
        return [tool for tool, _ in self.calls]


def http_503() -> ToolExecutionError:
    return ToolExecutionError("HTTP 503 from data.sec.gov", retryable=True)


def build(dispatch: Any, *, board: BreakerBoard | None = None, policy: RetryPolicy | None = None,
          **kwargs: Any) -> tuple[ResilientDispatcher, list[float], list[tuple[str, dict[str, Any]]]]:
    sleeps: list[float] = []
    events: list[tuple[str, dict[str, Any]]] = []
    dispatcher = ResilientDispatcher(
        dispatch,
        breakers=board or BreakerBoard(clock=FakeClock()),
        policy=policy or RetryPolicy(),
        sleep=sleeps.append,
        rng=NoJitter(),  # type: ignore[arg-type]
        on_event=lambda kind, payload: events.append((kind, payload)),
        **kwargs,
    )
    return dispatcher, sleeps, events


# --------------------------------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ToolExecutionError("HTTP 503 from data.sec.gov", retryable=True), "transient"),
        (ToolExecutionError("HTTP 500 from host"), "transient"),
        (ToolExecutionError("connection reset by peer"), "transient"),
        (ConnectionError("refused"), "transient"),
        (ToolExecutionError("host: boom after 4 attempts", retryable=True), "transient"),
        (ToolExecutionError("HTTP 429 from api.example.com"), "rate_limited"),
        (ToolExecutionError("HTTP 401 from host"), "auth"),
        (ToolExecutionError("HTTP 403 from host"), "auth"),
        (ToolOutputError("financial_data_api: output does not match schema"), "malformed"),
        (ToolExecutionError("request timed out after 20s", retryable=True), "timeout"),
        (TimeoutError("slow"), "timeout"),
        (ToolValidationError("web_search: invalid arguments: query"), "invalid_args"),
        (ToolExecutionError("not found: https://data.sec.gov/x"), "not_found"),
        (ToolExecutionError("HTTP 404 from host"), "not_found"),
        (ToolExecutionError("sec_filing_search: no EDGAR filer found for ticker 'ZZZ'"), "not_found"),
        (ToolExecutionError("company_profile: unknown ticker 'ZZZ'"), "not_found"),
        (ToolExecutionError("company_profile: no profile data for ticker 'ZZZ'"), "not_found"),
        (ToolExecutionError("sec_filing_search: no 10-K found in recent filings"), "not_found"),
        (ToolUnavailableError("web_search has no handler"), "unavailable"),
        (ToolLockedError("peer_comparison unlocks after challenge 4"), "unavailable"),
        (ToolExecutionError("something odd happened"), "permanent"),
        (ToolNotFoundError("no such tool"), "permanent"),
        (ValueError("bug"), "permanent"),
    ],
)
def test_classify(exc: BaseException, expected: str) -> None:
    assert classify(exc) == expected


def test_classify_covers_every_class() -> None:
    produced = {
        classify(ToolExecutionError("HTTP 503", retryable=True)),
        classify(ToolExecutionError("HTTP 429")),
        classify(ToolExecutionError("HTTP 401")),
        classify(ToolOutputError("bad")),
        classify(ToolExecutionError("timed out")),
        classify(ToolValidationError("bad")),
        classify(ToolExecutionError("not found")),
        classify(ToolUnavailableError("no handler")),
        classify(ToolExecutionError("odd")),
    }
    assert produced == set(ERROR_CLASSES)


# --------------------------------------------------------------------------------------------------
# RetryPolicy
# --------------------------------------------------------------------------------------------------


def test_retry_delays_double_without_jitter() -> None:
    policy = RetryPolicy()
    assert [policy.delay(n, NoJitter()) for n in range(5)] == [1.0, 2.0, 4.0, 8.0, 16.0]  # type: ignore[arg-type]


def test_retry_delay_adds_bounded_jitter() -> None:
    import random

    policy = RetryPolicy()
    rng = random.Random(1)
    delays = [policy.delay(0, rng) for _ in range(50)]
    assert all(1.0 <= d <= 1.5 for d in delays)
    assert len(set(delays)) > 1


def test_retries_for_each_class() -> None:
    policy = RetryPolicy()
    assert policy.retries_for("transient") == 5
    assert policy.retries_for("rate_limited") == 5
    assert policy.retries_for("timeout") == 1
    assert policy.retries_for("malformed") == 1
    for other in ("auth", "invalid_args", "not_found", "unavailable", "permanent"):
        assert policy.retries_for(other) == 0
    assert RetryPolicy(max_retries=2).retries_for("transient") == 2


# --------------------------------------------------------------------------------------------------
# Circuit breaker
# --------------------------------------------------------------------------------------------------


def test_breaker_opens_after_three_consecutive_failures() -> None:
    breaker = CircuitBreaker("p", clock=FakeClock())
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.trips == 1
    assert breaker.opened_at == 1000.0


def test_breaker_opens_at_half_failures_in_window_of_six() -> None:
    breaker = CircuitBreaker("p", clock=FakeClock())
    for failed in (True, False, True, False, True):  # never 3 in a row; window not full yet
        breaker.record_failure() if failed else breaker.record_success()
    assert breaker.state == "closed"
    breaker.record_success()  # window full: 3 failures in 6 calls = 50%
    breaker.record_failure()  # window slides to F S F S S F = 3/6 -> trips
    assert breaker.state == "open"


def test_breaker_does_not_open_below_the_ratio() -> None:
    breaker = CircuitBreaker("p", clock=FakeClock())
    for failed in (True, False, False, True, False, False, True, False):
        breaker.record_failure() if failed else breaker.record_success()
    assert breaker.state == "closed"


def test_open_breaker_refuses_then_allows_one_trial_after_cooldown() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker("p", clock=clock)
    for _ in range(3):
        breaker.record_failure()
    assert breaker.allow() is False
    clock.advance(29.9)
    assert breaker.allow() is False
    clock.advance(0.2)
    assert breaker.state == "half_open"
    assert breaker.allow() is True  # the single trial
    assert breaker.allow() is False  # further calls wait for the trial's verdict


def test_half_open_success_closes_and_failure_reopens() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker("p", clock=clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(31)
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed"
    assert breaker.allow()

    for _ in range(3):
        breaker.record_failure()
    clock.advance(31)
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state == "open"
    assert breaker.trips == 3
    assert breaker.opened_at == clock.now  # fresh cool-down
    clock.advance(29)
    assert breaker.state == "open"


def test_release_frees_a_half_open_trial() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker("p", clock=clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(31)
    assert breaker.allow()
    assert not breaker.allow()
    breaker.release()
    assert breaker.allow()


def test_board_maps_tools_to_providers() -> None:
    board = BreakerBoard(clock=FakeClock())
    assert board.for_tool("sec_filing_search") is board.for_tool("financial_data_api")
    assert board.for_tool("financial_data_api") is board.for_tool("company_profile")
    assert board.for_tool("earnings_transcript") is board.for_provider("transcripts")
    assert board.for_tool("peer_comparison") is board.for_provider("yfinance")
    assert board.for_tool("news_sentiment") is board.for_provider("google_news")
    assert board.for_tool("web_search") is board.for_provider("web_search")
    for tool in LOCAL_TOOLS:
        assert provider_for(tool) is None
        assert board.for_tool(tool) is None
    assert provider_for("no_such_tool") is None
    assert set(NETWORK_TOOLS) | set(LOCAL_TOOLS) == set(TOOL_PROVIDERS)


def test_board_system_degraded_when_three_providers_open() -> None:
    board = BreakerBoard(clock=FakeClock())
    assert board.snapshot()["sec_edgar"] == "closed"
    for provider in ("sec_edgar", "web_search"):
        for _ in range(3):
            board.for_provider(provider).record_failure()
    assert board.open_providers() == ["sec_edgar", "web_search"]
    assert not board.system_degraded()
    for _ in range(3):
        board.for_provider("yfinance").record_failure()
    assert board.system_degraded()
    assert board.snapshot()["yfinance"] == "open"


def test_board_uses_custom_config() -> None:
    board = BreakerBoard(BreakerConfig(consecutive_failures=1, max_open_providers=1), clock=FakeClock())
    board.for_provider("sec_edgar").record_failure()
    assert board.system_degraded()


# --------------------------------------------------------------------------------------------------
# Fallback chains
# --------------------------------------------------------------------------------------------------


def _schema_fallbacks() -> dict[str, list[str]]:
    out = {}
    for path in sorted(SCHEMA_DIR.glob("*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        out[schema["name"]] = schema.get("x-ara", {}).get("fallbacks", [])
    return out


def test_every_network_tool_has_two_fallbacks() -> None:
    for tool in NETWORK_TOOLS:
        assert len(chain_for(tool)) >= 2, tool


def test_every_schema_fallback_is_in_the_chain() -> None:
    declared = _schema_fallbacks()
    assert declared
    for tool, fallbacks in declared.items():
        chained = [fb.tool for fb in chain_for(tool)]
        for name in fallbacks:
            assert name in chained, f"{tool}: schema fallback {name} missing from chain"


def test_chain_for_returns_a_copy_and_handles_unknown_tools() -> None:
    chain = chain_for("web_search")
    chain.clear()
    assert chain_for("web_search")
    assert chain_for("calculation_engine") == []


SAMPLE_INPUTS: dict[str, list[dict[str, Any]]] = {
    "sec_filing_search": [{"ticker": "msft", "filing_type": "10-K"}, {"ticker": "AAPL", "filing_type": "8-K", "year": 2024}],
    "financial_data_api": [
        {"ticker": "TSLA", "statement_type": "income_statement"},
        {"ticker": "TSLA", "statement_type": "key_ratios", "period": "quarterly", "years": 2},
    ],
    "company_profile": [{"ticker": "NVDA"}],
    "earnings_transcript": [{"ticker": "AMZN"}, {"ticker": "AMZN", "quarter": "Q2", "year": 2025}],
    "web_search": [{"query": "Tesla regulatory investigations 2025"}, {"query": "x" * 400}],
    "news_sentiment": [
        {"query": "Palantir"},
        {"query": "Palantir", "lookback_days": 1},
        {"query": "Palantir", "lookback_days": 5},
        {"query": "Palantir", "lookback_days": 300},
        {"query": "y" * 250},
    ],
    "peer_comparison": [{"ticker": "MSFT"}, {"ticker": "MSFT", "num_peers": 3, "metrics": ["revenue"]}],
    "vector_db_search": [{"query": "Microsoft cloud revenue", "ticker": "MSFT"}, {"query": "z" * 600}],
}


def test_sample_inputs_cover_every_chain() -> None:
    assert set(SAMPLE_INPUTS) == set(FALLBACK_CHAINS)


def test_every_mapped_fallback_argument_set_validates() -> None:
    registry = build_default_registry("mock")
    checked = 0
    for tool, samples in SAMPLE_INPUTS.items():
        for sample in samples:
            for fb in chain_for(tool):
                mapped = fb.map_args(sample)
                assert mapped is not None, (tool, fb.tool, sample)
                registry.validate_args(fb.tool, mapped)  # raises ToolValidationError if invalid
                checked += 1
    assert checked >= 40


def test_mapping_returns_none_when_input_is_missing() -> None:
    for tool in FALLBACK_CHAINS:
        for fb in chain_for(tool):
            assert fb.map_args({}) is None, (tool, fb.tool)
    assert chain_for("sec_filing_search")[0].map_args({"ticker": "not a ticker!"}) is None
    assert chain_for("web_search")[0].map_args({"query": " "}) is None


def test_documented_mappings() -> None:
    fin = chain_for("financial_data_api")
    assert fin[0].tool == "sec_filing_search"
    assert fin[0].map_args({"ticker": "tsla", "statement_type": "cash_flow"}) == {"ticker": "TSLA", "filing_type": "10-K"}
    assert fin[-1].tool == "vector_db_search"
    assert fin[-1].map_args({"ticker": "TSLA", "statement_type": "cash_flow"})["query"] == "TSLA cash flow"
    assert chain_for("earnings_transcript")[0].map_args({"ticker": "X"}) == {"ticker": "X", "filing_type": "8-K"}
    assert chain_for("news_sentiment")[0].map_args({"query": "Palantir"}) == {"query": "Palantir", "date_range": "month"}
    assert chain_for("peer_comparison")[0].map_args({"ticker": "MSFT"}) == {
        "ticker": "MSFT", "statement_type": "key_ratios"}


# --------------------------------------------------------------------------------------------------
# Gap
# --------------------------------------------------------------------------------------------------


def test_gap_to_text_reads_as_one_sentence() -> None:
    gap = Gap(
        section="",
        missing_info="Financial statements for TSLA",
        cause="sec_edgar is unavailable (circuit open)",
        attempted_tools=["financial_data_api", "sec_filing_search", "vector_db_search"],
        suggested_manual_source="SEC EDGAR manually: https://www.sec.gov/edgar/search/",
    )
    assert gap.to_text() == (
        "Financial statements for TSLA are missing: sec_edgar is unavailable (circuit open); tried "
        "financial_data_api, sec_filing_search, vector_db_search. Check SEC EDGAR manually: "
        "https://www.sec.gov/edgar/search/"
    )


def test_gap_to_text_with_section_and_singular_subject() -> None:
    gap = Gap("Valuation", "The company profile for X", "no data exists for this request", ["company_profile"], "")
    assert gap.to_text() == "Valuation: The company profile for X is missing: no data exists for this request; tried company_profile."


# --------------------------------------------------------------------------------------------------
# ResilientDispatcher
# --------------------------------------------------------------------------------------------------


ARGS = {"ticker": "MSFT", "statement_type": "income_statement"}


def test_transient_errors_are_retried_with_backoff_then_succeed() -> None:
    script = Script({"financial_data_api": [http_503(), http_503(), make_result("financial_data_api", ARGS)]})
    dispatcher, sleeps, events = build(script)
    result = dispatcher.call("financial_data_api", ARGS)
    assert result.tool == "financial_data_api"
    assert not result.is_fallback
    assert sleeps == [1.0, 2.0]
    assert [kind for kind, _ in events] == ["retry", "retry"]
    assert script.tools_called() == ["financial_data_api"] * 3
    assert dispatcher.breakers.for_tool("financial_data_api").state == "closed"
    assert [a.error_class for a in dispatcher.last_attempts] == ["transient", "transient", None]


def test_success_records_on_breaker_and_local_tools_have_no_breaker() -> None:
    dispatcher, sleeps, _ = build(Script())
    assert dispatcher.call("calculation_engine", {}).tool == "calculation_engine"
    assert sleeps == []


def test_invalid_args_raise_immediately_without_fallback() -> None:
    script = Script({"financial_data_api": [ToolValidationError("bad args")]})
    dispatcher, sleeps, events = build(script)
    with pytest.raises(ToolValidationError):
        dispatcher.call("financial_data_api", ARGS)
    assert script.tools_called() == ["financial_data_api"]
    assert sleeps == []
    assert events == []
    assert dispatcher.breakers.for_tool("financial_data_api").trips == 0


def test_validate_hook_normalises_and_rejects_before_dispatch() -> None:
    registry = build_default_registry("mock")
    script = Script()
    dispatcher, _, _ = build(script, validate=registry.validate_args)
    dispatcher.call("company_profile", {"ticker": " msft "})
    assert script.calls[0][1]["ticker"] == "MSFT"
    with pytest.raises(ToolValidationError):
        dispatcher.call("company_profile", {"ticker": "!!"})
    assert len(script.calls) == 1


def test_not_found_goes_to_fallback_without_tripping_the_breaker() -> None:
    missing = ToolExecutionError("financial_data_api: no EDGAR filer found for ticker 'MSFT'")
    script = Script({"financial_data_api": [missing]})
    dispatcher, sleeps, _ = build(script)
    for _ in range(5):
        result = dispatcher.call("financial_data_api", ARGS)
        assert result.is_fallback
        assert result.tool == "sec_filing_search"
    assert sleeps == []
    breaker = dispatcher.breakers.for_provider("sec_edgar")
    assert breaker.state == "closed"
    assert breaker.trips == 0


def test_open_breaker_skips_straight_to_fallback() -> None:
    board = BreakerBoard(clock=FakeClock())
    for _ in range(3):
        board.for_provider("sec_edgar").record_failure()
    script = Script()
    dispatcher, sleeps, events = build(script, board=board)
    result = dispatcher.call("financial_data_api", ARGS)
    # sec_filing_search shares the open sec_edgar breaker, so the first live call is web_search.
    assert script.tools_called() == ["web_search"]
    assert result.is_fallback
    assert result.tool == "web_search"
    assert result.data["fallback_for"] == "financial_data_api"
    assert "failed (unavailable: circuit open for sec_edgar)" in result.data["notes"][0]
    assert sleeps == []
    assert dispatcher.last_attempts[0].error_class == "unavailable"
    assert [kind for kind, _ in events][:1] == ["breaker_open"]
    assert ("fallback_used", {"tool": "financial_data_api", "fallback": "web_search",
                              "error_class": "unavailable"}) in events


def test_fallback_result_is_marked_and_cached_original_is_not_mutated() -> None:
    cached = make_result("sec_filing_search", {"ticker": "MSFT"}, {"filings": [1], "notes": ["existing"]})
    script = Script({"financial_data_api": [ToolExecutionError("HTTP 401 from host")],
                     "sec_filing_search": [cached]})
    dispatcher, sleeps, events = build(script)
    result = dispatcher.call("financial_data_api", ARGS)
    assert result is not cached
    assert result.is_fallback is True
    assert result.tool == "sec_filing_search"
    assert result.data["fallback_for"] == "financial_data_api"
    assert result.data["notes"][0] == "existing"
    assert ("Fallback: financial_data_api failed (auth: HTTP 401 from host); data from sec_filing_search."
            in result.data["notes"])
    assert result.data["filings"] == [1]
    # The object the registry would keep in its cache is untouched.
    assert cached.is_fallback is False
    assert cached.data == {"filings": [1], "notes": ["existing"]}
    assert sleeps == []  # auth is never retried
    assert script.tools_called() == ["financial_data_api", "sec_filing_search"]
    assert [kind for kind, _ in events] == ["fallback_used"]


def test_fallback_skips_steps_whose_arguments_cannot_be_mapped() -> None:
    script = Script({"web_search": [http_503()]})
    dispatcher, _, events = build(script, policy=RetryPolicy(max_retries=0))
    result = dispatcher.call("web_search", {"query": "Tesla"})
    assert result.tool == "news_sentiment"
    assert result.is_fallback
    assert script.tools_called() == ["web_search", "news_sentiment"]
    assert ("fallback_used", {"tool": "web_search", "fallback": "news_sentiment", "error_class": "transient"}) in events


def test_failed_fallback_moves_to_the_next_one() -> None:
    script = Script({
        "financial_data_api": [ToolExecutionError("HTTP 401")],
        "sec_filing_search": [ToolExecutionError("HTTP 401")],
    })
    dispatcher, _, events = build(script)
    result = dispatcher.call("financial_data_api", ARGS)
    assert result.tool == "web_search"
    assert any(kind == "fallback_failed" and p["fallback"] == "sec_filing_search" for kind, p in events)


def test_all_failing_raises_with_gap_and_attempts() -> None:
    fail = http_503()
    script = Script({tool: [fail] for tool in ("financial_data_api", "sec_filing_search", "web_search",
                                                "vector_db_search")})
    dispatcher, _, events = build(script, policy=RetryPolicy(max_retries=0))
    with pytest.raises(AllFallbacksFailed) as info:
        dispatcher.call("financial_data_api", {"ticker": "TSLA", "statement_type": "income_statement"})
    error = info.value
    assert isinstance(error, ToolError)
    assert error.retryable is False
    assert [a.tool for a in error.attempts] == ["financial_data_api", "sec_filing_search", "web_search",
                                                  "vector_db_search"]
    assert all(a.error_class == "transient" for a in error.attempts)
    assert [a.fallback for a in error.attempts] == [False, True, True, True]
    gap = error.gap
    assert gap.attempted_tools == ["financial_data_api", "sec_filing_search", "web_search", "vector_db_search"]
    text = gap.to_text()
    assert "Financial statements for TSLA are missing" in text
    assert "sec_edgar is failing" in text
    for name in gap.attempted_tools:
        assert name in text
    assert "https://www.sec.gov/edgar/search/" in text
    assert events[-1][0] == "gap"
    assert str(error) == text


def test_gap_names_the_circuit_when_the_breaker_was_open() -> None:
    board = BreakerBoard(clock=FakeClock())
    for provider in ("sec_edgar", "web_search"):
        for _ in range(3):
            board.for_provider(provider).record_failure()
    script = Script({"vector_db_search": [ToolExecutionError("HTTP 503")]})
    dispatcher, _, _ = build(script, board=board)
    with pytest.raises(AllFallbacksFailed) as info:
        dispatcher.call("financial_data_api", {"ticker": "TSLA", "statement_type": "income_statement"})
    assert info.value.gap.cause == "sec_edgar is unavailable (circuit open)"
    assert "sec_edgar is unavailable (circuit open)" in info.value.gap.to_text()


def test_deadline_stops_retries() -> None:
    script = Script({"financial_data_api": [http_503()]})
    dispatcher, sleeps, events = build(script, deadline=lambda: True)
    result = dispatcher.call("financial_data_api", ARGS)
    assert script.tools_called().count("financial_data_api") == 1
    assert sleeps == []
    assert "retry" not in [kind for kind, _ in events]
    assert result.is_fallback  # the chain still gets its single attempt


def test_deadline_expiring_mid_retry_stops_further_retries() -> None:
    state = {"sleeps": 0}
    script = Script({"web_search": [http_503()]})
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        state["sleeps"] += 1

    dispatcher = ResilientDispatcher(
        script, breakers=BreakerBoard(BreakerConfig(consecutive_failures=99, window=99), clock=FakeClock()),
        policy=RetryPolicy(), sleep=sleep, rng=NoJitter(),  # type: ignore[arg-type]
        deadline=lambda: state["sleeps"] >= 2,
    )
    dispatcher.call("web_search", {"query": "Tesla"})
    assert sleeps == [1.0, 2.0]
    assert script.tools_called().count("web_search") == 3


def test_breaker_tripping_mid_retry_stops_retrying() -> None:
    script = Script({"financial_data_api": [http_503()]})
    dispatcher, sleeps, events = build(script)
    dispatcher.call("financial_data_api", ARGS)
    assert script.tools_called().count("financial_data_api") == 3  # three failures trip the breaker
    assert sleeps == [1.0, 2.0]
    assert dispatcher.breakers.for_provider("sec_edgar").state == "open"
    assert ("breaker_open", {"tool": "financial_data_api", "provider": "sec_edgar", "reason": "tripped"}) in events


def test_rate_limited_honours_retry_after() -> None:
    script = Script({"web_search": [Retryable("HTTP 429 from host", retry_after=10.0),
                                    make_result("web_search")]})
    dispatcher, sleeps, _ = build(script)
    result = dispatcher.call("web_search", {"query": "Tesla"})
    assert not result.is_fallback
    assert sleeps == [10.0]


def test_timeout_and_malformed_get_one_retry() -> None:
    for error in (ToolExecutionError("request timed out", retryable=True), ToolOutputError("bad schema")):
        script = Script({"peer_comparison": [error]})
        dispatcher, sleeps, _ = build(script)
        result = dispatcher.call("peer_comparison", {"ticker": "MSFT"})
        assert script.tools_called().count("peer_comparison") == 2
        assert sleeps == [1.0]
        assert result.is_fallback


def test_half_open_trial_is_freed_when_the_probe_ends_neutrally() -> None:
    clock = FakeClock()
    board = BreakerBoard(clock=clock)
    for _ in range(3):
        board.for_provider("sec_edgar").record_failure()
    clock.advance(31)
    script = Script({"sec_filing_search": [ToolExecutionError("no EDGAR filer found for ticker 'X'"),
                                           make_result("sec_filing_search")]})
    dispatcher, _, _ = build(script, board=board)
    assert dispatcher.call("sec_filing_search", {"ticker": "X", "filing_type": "10-K"}).tool == "web_search"
    assert board.for_provider("sec_edgar").allow()  # not stuck waiting on a trial that never reported


def test_integration_financial_api_down_falls_back_to_sec_filings() -> None:
    registry = build_default_registry("mock")

    def dispatch(name: str, args: dict[str, Any]) -> ToolResult:
        if name == "financial_data_api":
            raise ToolExecutionError("HTTP 503", retryable=True)
        return registry.dispatch(name, args)

    # The breaker is per provider and financial_data_api shares sec_edgar with sec_filing_search, so
    # with default thresholds three failures would (correctly) refuse the SEC fallback too. Loosen
    # them here to exercise the fallback path itself.
    board = BreakerBoard(BreakerConfig(consecutive_failures=99, window=99), clock=FakeClock())
    dispatcher, sleeps, _ = build(dispatch, board=board, validate=registry.validate_args)
    result = dispatcher.call("financial_data_api", {"ticker": "msft", "statement_type": "income_statement"})
    assert result.tool == "sec_filing_search"
    assert result.is_fallback
    assert result.args["ticker"] == "MSFT"
    assert result.data["fallback_for"] == "financial_data_api"
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]


def test_integration_shared_provider_outage_skips_to_web_search() -> None:
    registry = build_default_registry("mock")

    def dispatch(name: str, args: dict[str, Any]) -> ToolResult:
        if name == "financial_data_api":
            raise ToolExecutionError("HTTP 503", retryable=True)
        return registry.dispatch(name, args)

    dispatcher, _, _ = build(dispatch, validate=registry.validate_args)
    result = dispatcher.call("financial_data_api", {"ticker": "MSFT", "statement_type": "income_statement"})
    assert result.is_fallback
    assert result.tool == "web_search"  # sec_edgar's breaker opened after three failures


# --------------------------------------------------------------------------------------------------
# FailureInjector
# --------------------------------------------------------------------------------------------------


def _run_injector(injector: FailureInjector, tool: str, calls: int) -> list[str | None]:
    outcomes: list[str | None] = []
    for _ in range(calls):
        try:
            injector.maybe_fail(tool)
            outcomes.append(None)
        except ToolError as exc:
            outcomes.append(classify(exc))
    return outcomes


def test_injector_rate_is_about_the_configured_rate() -> None:
    injector = FailureInjector(rate=0.5, seed=42)
    outcomes = _run_injector(injector, "financial_data_api", 200)
    failures = sum(o is not None for o in outcomes)
    assert 80 <= failures <= 120
    assert injector.injected == {"financial_data_api": failures}


def test_injector_produces_all_three_failure_kinds_as_recoverable_errors() -> None:
    injector = FailureInjector(rate=1.0, seed=1)
    seen = set(_run_injector(injector, "web_search", 60))
    assert seen == {"timeout", "transient", "malformed"}


def test_injector_never_touches_local_tools() -> None:
    injector = FailureInjector(rate=1.0, seed=3)
    for tool in LOCAL_TOOLS:
        assert _run_injector(injector, tool, 20) == [None] * 20
    assert injector.injected == {}


def test_injector_same_seed_same_sequence() -> None:
    first = _run_injector(FailureInjector(rate=0.5, seed=7), "web_search", 100)
    second = _run_injector(FailureInjector(rate=0.5, seed=7), "web_search", 100)
    other = _run_injector(FailureInjector(rate=0.5, seed=8), "web_search", 100)
    assert first == second
    assert first != other


def test_injector_per_tool_rates_override_the_default() -> None:
    injector = FailureInjector(rate=0.0, per_tool={"financial_data_api": 1.0}, seed=5)
    assert all(_run_injector(injector, "financial_data_api", 10))
    assert _run_injector(injector, "web_search", 10) == [None] * 10


def test_injector_zero_rate_never_fails() -> None:
    assert _run_injector(FailureInjector(seed=1), "web_search", 50) == [None] * 50


def test_injector_from_env() -> None:
    injector = FailureInjector.from_env({"ARA_FAIL_RATE": "0.5", "ARA_FAIL_SEED": "42"})
    assert injector.rate == 0.5
    assert _run_injector(injector, "web_search", 30) == _run_injector(
        FailureInjector(rate=0.5, seed=42), "web_search", 30)
    assert FailureInjector.from_env({}).rate == 0.0
    with pytest.raises(ValueError, match="ARA_FAIL_RATE"):
        FailureInjector.from_env({"ARA_FAIL_RATE": "lots"})


def test_injected_failures_exercise_recovery_through_the_dispatcher() -> None:
    injector = FailureInjector(rate=0.5, seed=11)
    registry = build_default_registry("mock")

    def dispatch(name: str, args: dict[str, Any]) -> ToolResult:
        injector.maybe_fail(name)
        return registry.dispatch(name, args)

    dispatcher, _, _ = build(dispatch, validate=registry.validate_args)
    delivered = 0
    for query in ("Tesla", "Apple", "Microsoft", "Nvidia", "Amazon"):
        try:
            dispatcher.call("web_search", {"query": query})
            delivered += 1
        except AllFallbacksFailed:
            pass
    assert injector.injected.get("web_search", 0) > 0
    assert delivered >= 1


# --------------------------------------------------------------------------------------------------
# Review regressions: thread safety, classification, backoff cap, gap accuracy
# --------------------------------------------------------------------------------------------------


def _run_threads(count: int, target: Any) -> None:
    import threading

    barrier = threading.Barrier(count)

    def wrapped(index: int) -> None:
        barrier.wait()
        target(index)

    threads = [threading.Thread(target=wrapped, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_half_open_admits_exactly_one_trial_under_threads() -> None:
    clock = FakeClock()
    breaker = CircuitBreaker("p", clock=clock)
    for _ in range(3):
        breaker.record_failure()
    clock.advance(31)
    allowed: list[bool] = []
    _run_threads(8, lambda _i: allowed.append(breaker.allow()))
    assert allowed.count(True) == 1


def test_record_failure_reports_the_tripping_call_exactly_once() -> None:
    breaker = CircuitBreaker("p", BreakerConfig(consecutive_failures=50, window=1000, window_failure_ratio=2.0))
    breaker.config.consecutive_failures = 8
    tripped: list[bool] = []
    _run_threads(8, lambda _i: tripped.append(breaker.record_failure()))
    assert tripped.count(True) == 1
    assert breaker.trips == 1


def test_last_attempts_is_per_thread() -> None:
    import threading

    gate = threading.Event()
    seen: dict[str, list[str]] = {}

    def slow(tool: str, args: dict[str, Any]) -> ToolResult:
        if tool == "web_search":
            gate.wait(5)
        return make_result(tool, args)

    dispatcher, _, _ = build(slow)

    def worker(name: str, tool: str, args: dict[str, Any]) -> None:
        dispatcher.call(tool, args)
        if name == "a":
            gate.set()
        else:
            gate.wait(5)
        seen[name] = [a.tool for a in dispatcher.last_attempts]

    ta = threading.Thread(target=worker, args=("a", "company_profile", {"ticker": "MSFT"}))
    tb = threading.Thread(target=worker, args=("b", "web_search", {"query": "Tesla"}))
    tb.start()
    ta.start()
    ta.join()
    tb.join()
    assert seen == {"a": ["company_profile"], "b": ["web_search"]}


def test_unexpected_exception_frees_the_half_open_trial() -> None:
    clock = FakeClock()
    board = BreakerBoard(clock=clock)
    breaker = board.for_provider("web_search")
    for _ in range(3):
        breaker.record_failure()
    clock.advance(31)

    def boom(tool: str, args: dict[str, Any]) -> ToolResult:
        raise RuntimeError("dispatch bug")

    dispatcher, _, _ = build(boom, board=board)
    with pytest.raises(RuntimeError):
        dispatcher.call("web_search", {"query": "Tesla"})
    assert breaker.allow() is True  # the trial slot was not leaked


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("HTTP 50 results returned", "permanent"),
        ("HTTP 5000 rows scanned", "permanent"),
        ("HTTP 4040 rows", "permanent"),
        ("connection pool summary unavailable", "permanent"),
        ("no data-driven approach possible", "permanent"),
        ("cache: results not found locally; HTTP 503 from upstream", "transient"),
        ("HTTP 404 from host", "not_found"),
        ("connection reset by peer", "transient"),
    ],
)
def test_classify_avoids_loose_message_matches(message: str, expected: str) -> None:
    assert classify(ToolExecutionError(message)) == expected


def test_retryable_execution_error_with_unknown_message_is_transient() -> None:
    assert classify(ToolExecutionError("weird upstream hiccup", retryable=True)) == "transient"


def test_retry_after_is_capped_and_delays_are_never_negative() -> None:
    script = Script({"web_search": [Retryable("HTTP 429 slow down", 3600), make_result("web_search")]})
    dispatcher, sleeps, _ = build(script)
    dispatcher.call("web_search", {"query": "Tesla"})
    assert sleeps == [60.0]
    policy = RetryPolicy(base_delay_s=1.0, max_jitter_s=0.0)
    assert policy.delay(100, NoJitter()) == 60.0  # type: ignore[arg-type]
    assert RetryPolicy(base_delay_s=-5.0).delay(0, NoJitter()) == 0.0  # type: ignore[arg-type]
    script = Script({"web_search": [Retryable("HTTP 429", -30), make_result("web_search")]})
    dispatcher, sleeps, _ = build(script)
    dispatcher.call("web_search", {"query": "Tesla"})
    assert sleeps[0] >= 0


def test_no_sleep_when_the_deadline_has_already_passed() -> None:
    script = Script({"web_search": [http_503()], "news_sentiment": [http_503()],
                     "vector_db_search": [ToolExecutionError("HTTP 404")]})
    dispatcher, sleeps, _ = build(script, deadline=lambda: True)
    with pytest.raises(AllFallbacksFailed):
        dispatcher.call("web_search", {"query": "Tesla"})
    assert sleeps == []


def test_live_schema_accepts_every_mapped_fallback() -> None:
    registry = build_default_registry("live")
    extra = [{"ticker": "BRK.B", "query": "Berkshire", "filing_type": "10-K", "year": 2024, "quarter": "Q2"},
             {"ticker": " msft ", "query": " ab ", "statement_type": None, "filing_type": None}]
    for tool, samples in SAMPLE_INPUTS.items():
        for sample in [*samples, *extra]:
            for fb in chain_for(tool):
                mapped = fb.map_args(sample)
                if mapped is not None:
                    registry.validate_args(fb.tool, mapped)


def test_gap_for_an_unavailable_primary_does_not_claim_an_open_circuit() -> None:
    miss = ToolExecutionError("HTTP 404")
    script = Script({"company_profile": [ToolUnavailableError("company_profile has no handler for mode 'live'")],
                     "sec_filing_search": [miss], "web_search": [miss], "vector_db_search": [miss]})
    dispatcher, _, _ = build(script)
    with pytest.raises(AllFallbacksFailed) as info:
        dispatcher.call("company_profile", {"ticker": "MSFT"})
    assert "circuit" not in info.value.gap.cause
    assert "no handler" in info.value.gap.cause


def test_unavailable_or_open_fallback_is_skipped_and_the_chain_continues() -> None:
    board = BreakerBoard(clock=FakeClock())
    for _ in range(3):
        board.for_provider("web_search").record_failure()
    script = Script({
        "financial_data_api": [ToolExecutionError("HTTP 404")],
        "sec_filing_search": [ToolLockedError("sec_filing_search locked")],
        "vector_db_search": [make_result("vector_db_search")],
    })
    dispatcher, _, _ = build(script, board=board)
    result = dispatcher.call("financial_data_api", {"ticker": "TSLA", "statement_type": "income_statement"})
    assert result.is_fallback and result.tool == "vector_db_search"
    assert "web_search" not in script.tools_called()  # open breaker: never dispatched


def test_injector_edge_rates_and_cache_hits() -> None:
    assert _run_injector(FailureInjector(rate=1.0, seed=1), "web_search", 30) != [None] * 30
    assert all(_run_injector(FailureInjector(rate=1.0, seed=1), "web_search", 30))
    assert _run_injector(FailureInjector(rate=1.0, per_tool={"web_search": 0.0}, seed=1), "web_search", 20) == [None] * 20
    registry = build_default_registry("mock")
    first = registry.dispatch("web_search", {"query": "Tesla cache probe"})
    registry.failure_injector = FailureInjector(rate=1.0, seed=1)
    again = registry.dispatch("web_search", {"query": "Tesla cache probe"})  # served from cache
    assert again.data == first.data
    assert registry.failure_injector.injected == {}


def test_injector_counts_are_exact_under_threads() -> None:
    injector = FailureInjector(rate=1.0, seed=2)

    def hammer(_i: int) -> None:
        for _ in range(200):
            try:
                injector.maybe_fail("web_search")
            except ToolError:
                pass

    _run_threads(8, hammer)
    assert injector.injected == {"web_search": 1600}
