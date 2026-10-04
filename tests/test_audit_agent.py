"""Regression tests for bugs found in the agent/ audit (mock mode only: no network, no real LLM)."""

from __future__ import annotations

import io
import json
import sys
from types import SimpleNamespace

import pytest

from agent.llm import AllProvidersFailedError, LLMClient, LLMProviderError, LLMResponse, MockProvider, TokenUsage
from agent.parser import parse_plan, parse_sections
from agent.query_analyzer import analyze_query
from agent.references import resolve_references
from tools import build_default_registry

SECTIONS = ["executive_summary", "business_overview"]


# ==================================================================================================
# parser.py
# ==================================================================================================


def _plan_with_dependency(depends_on):
    return json.dumps({"steps": [
        {"id": "s1", "tool": "financial_data_api", "args": {"ticker": "MSFT", "statement_type": "income_statement"}},
        {"id": "s2", "tool": "calculation_engine", "args": None, "depends_on": depends_on},
    ]})


def test_a_dependency_given_as_a_string_is_kept():
    """"depends_on": "s1" was iterated character by character ('s', '1'), both dropped as unknown, so
    the calculation ran without waiting for (or seeing) the data it depends on."""
    plan = parse_plan(_plan_with_dependency("s1"), build_default_registry("mock"), max_steps=5)
    assert plan.get("s2").depends_on == ["s1"]
    assert [s.id for s in plan.ready()] == ["s1"]


@pytest.mark.parametrize("depends_on", [1, 2.5, True, {"step": "s1"}])
def test_a_dependency_of_the_wrong_type_does_not_crash_planning(depends_on):
    """"depends_on": 1 raised TypeError ('int' object is not iterable), which is not a ParseError, so
    it escaped the plan node and crashed the whole run."""
    plan = parse_plan(_plan_with_dependency(depends_on), build_default_registry("mock"), max_steps=5)
    assert [s.id for s in plan.steps] == ["s1", "s2"]
    assert plan.get("s2").depends_on in ([], ["1"])


def test_a_report_wrapped_in_a_code_fence_does_not_leak_the_fence():
    """A model that wraps its whole answer in ```markdown ... ``` left the closing fence in the last
    section, so everything after it in the report (gaps, methodology, sources) rendered as code."""
    text = ("```markdown\n=== SECTION: executive_summary ===\nA.\n"
            "=== SECTION: business_overview ===\nB.\n```")
    assert parse_sections(text, SECTIONS) == {"executive_summary": "A.", "business_overview": "B."}


def test_a_code_block_inside_a_section_is_left_alone():
    text = "=== SECTION: executive_summary ===\nA.\n```\ncode\n```\n=== SECTION: business_overview ===\nB."
    assert parse_sections(text, SECTIONS)["executive_summary"] == "A.\n```\ncode\n```"


@pytest.mark.parametrize("open_, close", [("**", "**"), ("## ", ""), ("### **", "**")])
def test_section_markers_in_bold_or_as_headings_are_recognised(open_, close):
    """'**=== SECTION: x ===**' was not seen as a marker: the whole reply, markers included, became
    the executive summary and every other section was reported missing."""
    text = (f"{open_}=== SECTION: executive_summary ==={close}\nA.\n"
            f"{open_}=== SECTION: business_overview ==={close}\nB.")
    assert parse_sections(text, SECTIONS) == {"executive_summary": "A.", "business_overview": "B."}


def test_numbered_markdown_headings_are_split_into_sections():
    """The heading fallback matched '## Executive Summary' but not '## 1. Executive Summary'."""
    text = "## 1. Executive Summary\nA.\n## 2) Business Overview\nB."
    assert parse_sections(text, SECTIONS) == {"executive_summary": "A.", "business_overview": "B."}


# ==================================================================================================
# references.py
# ==================================================================================================

INCOME = {"data": [
    {"fiscal_year": 2024, "fiscal_period": "FY", "values": {"revenue": 245122, "net_income": 88136}},
    {"fiscal_year": 2023, "fiscal_period": "FY", "values": {"revenue": 211915, "net_income": 72361}},
]}


@pytest.mark.parametrize("reference", [
    "s2.cost_of_revenue_FY2024",   # a different line item the statement does not carry
    "s2.deferred_revenue_FY2024",
    "s2.revenue_growth_FY2024",    # a rate, not an amount
    "s2.net_income_margin_FY2024",
    "revenue per share FY2024",
])
def test_a_reference_to_another_line_item_is_not_resolved_to_a_shorter_metric(reference):
    """'cost_of_revenue' was matched to 'revenue' (the only metric whose name it contains), so a
    cost-to-revenue calculation silently became revenue / revenue = 100 %."""
    args = {"calculation_type": "margin", "inputs": {"numerator": reference, "denominator": "s2.revenue_FY2024"}}
    assert resolve_references(args, [INCOME]) == (None, "")


def test_plain_references_still_resolve_after_the_qualifier_check():
    args = {"calculation_type": "margin", "inputs": {"numerator": "s2.net_income_FY2024",
                                                     "denominator": "MSFT total revenue FY2024"}}
    resolved, _ = resolve_references(args, [INCOME])
    assert resolved["inputs"] == {"numerator": 88136, "denominator": 245122}
    ratios = {"data": [{"fiscal_year": 2024, "fiscal_period": "FY", "values": {"gross_margin": 0.69}}]}
    resolved, _ = resolve_references({"inputs": {"a": "s3.gross_margin_FY2024"}}, [ratios])
    assert resolved["inputs"] == {"a": 0.69}  # the qualifier is part of the metric's own name


# ==================================================================================================
# llm.py
# ==================================================================================================


def test_an_offline_local_server_is_not_reported_as_an_exhausted_daily_quota():
    """ARA_MODE=local with Ollama not running: the first call fails with a connection error and sets
    the provider aside; every later call then said '[ollama] quota_exhausted: daily quota used up',
    which is what the report's gap text showed."""
    now = [0.0]
    ollama = MockProvider([LLMProviderError("ollama", "connection", "connection refused")], name="ollama")
    ollama.is_local = True
    client = LLMClient([ollama], max_retries=2, sleep=lambda s: None, clock=lambda: now[0])
    with pytest.raises(AllProvidersFailedError):
        client.complete([{"role": "user", "content": "plan"}])
    now[0] = 60
    with pytest.raises(AllProvidersFailedError) as info:
        client.complete([{"role": "user", "content": "synthesize"}])
    assert len(ollama.requests) == 1  # still set aside, not retried
    assert [e.kind for e in info.value.errors] == ["connection"]
    assert "quota" not in str(info.value)


def test_a_provider_out_of_daily_quota_is_still_reported_as_such():
    now = [0.0]
    gemini = MockProvider([LLMProviderError("gemini", "quota_exhausted", "exceeded your current quota")],
                          name="gemini")
    client = LLMClient([gemini], max_retries=2, sleep=lambda s: None, clock=lambda: now[0])
    with pytest.raises(AllProvidersFailedError):
        client.complete([{"role": "user", "content": "a"}])
    with pytest.raises(AllProvidersFailedError) as info:
        client.complete([{"role": "user", "content": "b"}])
    assert [e.kind for e in info.value.errors] == ["quota_exhausted"]


# ==================================================================================================
# core.py
# ==================================================================================================


def test_an_empty_execute_reply_is_a_failed_step_with_a_gap_not_a_step_that_was_not_needed():
    """Gemini returns no text and no tool call when its hidden reasoning uses up max_tokens
    (finish_reason 'length'). The step was recorded as 'skipped: not needed: no tool call made', and
    'not needed' steps add no data gap, so the planned calculation vanished from the report silently."""
    from tests.test_core import QUERY, ScriptedLLM, make_agent

    class EmptyExecute(ScriptedLLM):
        def __call__(self, request):
            if "[ARA:EXECUTE]" in request.messages[-1]["content"]:
                return LLMResponse(None, [], "mock", "mock-llm", TokenUsage(), 0.0, finish_reason="length")
            return super().__call__(request)

    result = make_agent(EmptyExecute(synthesis="=== SECTION: executive_summary ===\nNo figures."),
                        memory=None).run(QUERY)
    step = result.plan.get("s5")
    assert step.status == "failed"
    assert "not needed" not in step.summary
    assert any("FY2024 revenue growth" in gap for gap in result.gaps)


def test_a_step_done_reply_is_still_a_step_that_was_not_needed():
    from tests.test_core import QUERY, ScriptedLLM, make_agent

    class StepDone(ScriptedLLM):
        def __call__(self, request):
            if "[ARA:EXECUTE]" in request.messages[-1]["content"]:
                return "STEP_DONE: the growth rate is already in the evidence"
            return super().__call__(request)

    result = make_agent(StepDone(synthesis="=== SECTION: executive_summary ===\nNo figures."),
                        memory=None).run(QUERY)
    step = result.plan.get("s5")
    assert step.status == "skipped" and step.summary.startswith("not needed")
    assert not any("FY2024 revenue growth" in gap for gap in result.gaps)


# ==================================================================================================
# query_analyzer.py
# ==================================================================================================


@pytest.mark.parametrize("query", ["Apple vs. Microsoft revenue growth", "Tesla vs. GM: margins in FY2025"])
def test_vs_with_a_full_stop_is_a_comparison(query):
    """' vs ' selected the comparative template but 'vs.' did not: the query got a single-company
    profile template."""
    assert analyze_query(query).template == "comparative_analysis"


@pytest.mark.xfail(strict=True, reason="CONFIRMED, NOT FIXED: any bracketed capitals are read as a ticker")
@pytest.mark.parametrize("query, expected", [
    ("Analyze Microsoft's artificial intelligence (AI) strategy", ["MSFT"]),
    ("Compare Nvidia and Apple earnings per share (EPS) for fiscal year (FY) 2025", ["NVDA", "AAPL"]),
])
def test_bracketed_acronyms_are_not_tickers(query, expected):
    assert sorted(analyze_query(query).tickers) == sorted(expected)


@pytest.mark.xfail(strict=True, reason="CONFIRMED, NOT FIXED: a bare 'C' or 'MS' is read as Citigroup / Morgan Stanley")
@pytest.mark.parametrize("query, expected", [
    ("What is Apple's C-suite compensation?", ["AAPL"]),
    ("Summarize MS Office revenue for Microsoft", ["MSFT"]),
])
def test_a_capital_letter_that_happens_to_be_a_known_ticker_is_not_a_company(query, expected):
    assert analyze_query(query).tickers == expected


# ==================================================================================================
# __main__.py and trace_viewer.py (Windows: redirected output is cp1252)
# ==================================================================================================

REPORT = "# Report\n\nMargin ≈ 45 %, − 2 pts → lower [E1].\n"


def _cp1252_stdout(monkeypatch):
    buffer = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(buffer, encoding="cp1252", errors="strict"))
    return buffer


def test_printing_the_report_to_a_cp1252_pipe_does_not_crash_after_the_run(monkeypatch):
    """`python -m agent "..." > report.md` on Windows: stdout is cp1252, and the first character it
    cannot encode (a minus sign U+2212, an arrow) raised UnicodeEncodeError after the whole run."""
    import agent.__main__ as cli
    import agent.core
    import agent.tracing
    import config.settings

    result = SimpleNamespace(report_markdown=REPORT, verification={}, session_id="session-x", stop_reason="completed",
                             duration_s=1.0, tool_calls=0, plan=SimpleNamespace(steps=[], by_status=lambda *s: []),
                             llm_usage={"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0},
                             trace_path=None)
    monkeypatch.setattr(config.settings, "load_settings", lambda: SimpleNamespace(log_level="WARNING"))
    monkeypatch.setattr(agent.tracing, "configure_logging", lambda level: None)
    monkeypatch.setattr(agent.core.ResearchAgent, "from_settings",
                        classmethod(lambda cls, settings, **kw: SimpleNamespace(run=lambda query: result)))
    buffer = _cp1252_stdout(monkeypatch)
    assert cli.main(["Profile of Microsoft"]) == 0
    sys.stdout.flush()
    assert b"# Report" in buffer.getvalue() and b"45 %" in buffer.getvalue()


def test_the_trace_viewer_prints_to_a_cp1252_pipe(monkeypatch, tmp_path):
    from agent import trace_viewer

    path = tmp_path / "session-x.jsonl"
    path.write_text(json.dumps({"seq": 1, "type": "thought", "text": "margin ≈ 45 % → lower"}) + "\n",
                    encoding="utf-8")
    buffer = _cp1252_stdout(monkeypatch)
    assert trace_viewer.main([str(path)]) == 0
    sys.stdout.flush()
    assert b"45 %" in buffer.getvalue()
