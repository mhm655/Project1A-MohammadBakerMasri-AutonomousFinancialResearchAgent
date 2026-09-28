"""Tests for the Day 4 agent loop: parser, query analysis and the full graph with a scripted LLM."""

from __future__ import annotations

import json
import re

import pytest

from agent.core import ResearchAgent, analyze_query, evidence_digest
from agent.llm import LLMClient, LLMProviderError, MockProvider
from agent.parser import (
    ParseError,
    Plan,
    PlanStep,
    extract_json,
    parse_plan,
    parse_replan,
    parse_sections,
    step_done_note,
)
from config.settings import AgentSettings
from memory.vector_store import DEFAULT_STORE
from tools import build_default_registry

QUERY = ("Create a comprehensive profile of Microsoft Corporation including business overview, "
         "financial summary, key executives, and recent developments.")

GOOD_PLAN = {
    "rationale": "Profile, three years of statements, ratios, news, then growth.",
    "steps": [
        {"id": "s1", "goal": "Company profile", "tool": "company_profile", "args": {"ticker": "MSFT"}},
        {"id": "s2", "goal": "Income statements", "tool": "financial_data_api",
         "args": {"ticker": "MSFT", "statement_type": "income_statement", "years": 3}},
        {"id": "s3", "goal": "Key ratios", "tool": "financial_data_api",
         "args": {"ticker": "MSFT", "statement_type": "key_ratios", "years": 3}},
        {"id": "s4", "goal": "Recent news", "tool": "news_sentiment", "args": {"query": "Microsoft"}},
        {"id": "s5", "goal": "FY2024 revenue growth", "tool": "calculation_engine", "args": None,
         "depends_on": ["s2"]},
    ],
}


def _ids(prompt: str, tool: str) -> list[str]:
    return re.findall(rf"\[(E\d+)\] {tool}\(", prompt)


class ScriptedLLM:
    """Answers each stage the way a well-behaved model would, citing the real evidence ids."""

    def __init__(self, plan=None, replan=None, synthesis=None):
        self.plan = plan if plan is not None else json.dumps(GOOD_PLAN)
        self.replan = replan or '{"action": "done"}'
        self.synthesis = synthesis
        self.stages: list[str] = []

    def __call__(self, request):
        prompt = request.messages[-1]["content"]
        stage = re.search(r"\[ARA:(\w+)\]", prompt).group(1)
        self.stages.append(stage)
        if stage == "PLAN":
            return self.plan
        if stage == "REPLAN":
            return self.replan
        if stage == "EXECUTE":
            if "calculation" in prompt:
                return {"tool_calls": [{"name": "calculation_engine", "arguments": {
                    "calculation_type": "growth_rate", "inputs": {"current": 245122, "previous": 211915}}}]}
            return {"tool_calls": [{"name": "financial_data_api",
                                    "arguments": {"ticker": "MSFT", "statement_type": "income_statement"}}]}
        if stage == "SYNTHESIZE":
            if self.synthesis is not None:
                return self.synthesis
            fin, calc = _ids(prompt, "financial_data_api"), _ids(prompt, "calculation_engine")
            prof, news = _ids(prompt, "company_profile"), _ids(prompt, "news_sentiment")
            return (
                "=== SECTION: executive_summary ===\n"
                f"Microsoft grew FY2024 revenue 15.67% [{calc[0]}] to $245,122 million [{fin[0]}].\n"
                "=== SECTION: business_overview ===\n"
                f"Microsoft reports three segments [{prof[0]}].\n"
                "=== SECTION: financial_summary ===\n"
                f"| Metric | FY2024 |\n|---|---|\n| Revenue | $245,122 million [{fin[0]}] |\n"
                f"\nGross margin was 69.76% [{fin[1]}].\n"
                "=== SECTION: key_executives ===\n"
                f"Satya Nadella is Chairman and CEO [{prof[0]}].\n"
                "=== SECTION: recent_developments ===\n"
                f"Coverage was mixed [{news[0]}].\n"
            )
        raise AssertionError(f"unexpected stage {stage}")


def make_agent(script, *, max_tool_calls=20, clock=None, providers=None, **kw):
    llm = LLMClient(providers or [MockProvider(script)], max_retries=0, sleep=lambda s: None)
    options = {"agent_settings": AgentSettings(max_tool_calls=max_tool_calls), "max_workers": 4}
    if clock:
        options["clock"] = clock
    options.update(kw)
    return ResearchAgent(llm, lambda: build_default_registry("mock"), **options)


@pytest.fixture(autouse=True)
def _clean_memory():
    DEFAULT_STORE.clear()
    yield
    DEFAULT_STORE.clear()


# ==================================================================================================
# Parser
# ==================================================================================================


@pytest.mark.parametrize("text", [
    '{"a": 1}',
    'Here is the plan:\n```json\n{"a": 1}\n```\nDone.',
    'Sure! {"a": 1} Hope this helps.',
    '{"a": 1,}',  # trailing comma
    '{"note": "braces } inside strings { are fine", "a": 1}',
])
def test_extract_json_tolerates_common_llm_formatting(text):
    assert extract_json(text)["a"] == 1


@pytest.mark.parametrize("text", ["", "no json here", "[1, 2, 3]", '{"a": '])
def test_extract_json_rejects_unusable_replies(text):
    with pytest.raises(ParseError):
        extract_json(text)


def test_plan_validation_rejects_bad_steps_and_explains_why():
    registry = build_default_registry("mock")
    text = json.dumps({"steps": [
        {"id": "s1", "tool": "company_profile", "args": {"ticker": "MSFT"}},
        {"id": "s1", "tool": "web_search", "args": {"query": "dup id"}},
        {"id": "s2", "tool": "bloomberg_terminal", "args": {}},
        {"id": "s3", "tool": "report_generator", "args": {}},
        {"id": "s4", "tool": "financial_data_api", "args": {"ticker": "MSFT"}},  # missing statement_type
        {"id": "s5", "tool": "web_search", "args": {"query": "x y"}, "depends_on": ["s6", "ghost"]},
        {"id": "s6", "tool": "web_search", "args": {"query": "a b"}, "depends_on": ["s5"]},
    ]})
    plan = parse_plan(text, registry, max_steps=10)
    assert [s.id for s in plan.steps] == ["s1", "s4"]
    assert plan.get("s4").args is None  # kept; its arguments are repaired at run time
    reasons = " ".join(plan.rejected)
    for expected in ("duplicate step id", "unknown or unavailable tool", "run by the system",
                     "repaired at run time", "unknown dependencies", "dependency cycle"):
        assert expected in reasons


def test_plan_step_limit_and_empty_plans():
    registry = build_default_registry("mock")
    steps = [{"id": f"s{i}", "tool": "web_search", "args": {"query": f"q {i}"}} for i in range(5)]
    assert len(parse_plan(json.dumps({"steps": steps}), registry, max_steps=3).steps) == 3
    with pytest.raises(ParseError, match="no usable steps"):
        parse_plan('{"steps": [{"tool": "nope"}]}', registry, max_steps=3)


def test_replan_never_repeats_a_completed_call():
    registry = build_default_registry("mock")
    plan = Plan([PlanStep("s1", "profile", "company_profile", {"ticker": "MSFT"}, status="done")])
    text = json.dumps({"action": "continue", "add_steps": [
        {"id": "r1", "tool": "company_profile", "args": {"ticker": "msft"}},
        {"id": "r2", "tool": "web_search", "args": {"query": "Microsoft Azure growth"}},
    ]})
    decision = parse_replan(text, registry, plan, max_new_steps=5)
    assert [s.id for s in decision.new_steps] == ["r2"] and "repeats a completed call" in decision.rejected[0]
    assert parse_replan('{"action": "done"}', registry, plan, max_new_steps=5).done


def test_parse_sections():
    text = "preamble\n=== SECTION: Executive Summary ===\nA [E1].\n\n=== SECTION: key_executives ===\nB\n"
    assert parse_sections(text, ["executive_summary"]) == {"executive_summary": "A [E1].", "key_executives": "B"}
    assert parse_sections("just prose", ["executive_summary"]) == {"executive_summary": "just prose"}
    assert step_done_note("STEP_DONE: already covered by E3") == "already covered by E3"
    assert step_done_note("calling a tool") is None


@pytest.mark.parametrize("query, template, tickers", [
    (QUERY, "company_profile", ["MSFT"]),
    ("Produce a comprehensive risk assessment for Tesla Inc.", "risk_assessment", ["TSLA"]),
    ("Analyze Apple Inc.'s most recent quarterly earnings", "earnings_analysis", ["AAPL"]),
    ("Compare the cloud divisions of Amazon (AWS), Microsoft (Azure), and Google (GCP)", "comparative_analysis",
     ["MSFT", "AMZN", "GOOGL"]),
    ("Produce a complete investment research report on NVIDIA Corporation", "full_report", ["NVDA"]),
    ("Research the widget maker (ticker WDGT)", "company_profile", ["WDGT"]),
])
def test_analyze_query(query, template, tickers):
    profile = analyze_query(query)
    assert profile.template == template and sorted(profile.tickers) == sorted(tickers)


# ==================================================================================================
# The full loop
# ==================================================================================================


def test_challenge_1_end_to_end_on_mock_data():
    script = ScriptedLLM()
    result = make_agent(script).run(QUERY)

    assert result.stop_reason == "completed" and result.template == "company_profile"
    assert script.stages == ["PLAN", "EXECUTE", "REPLAN", "SYNTHESIZE"]  # 4 LLM calls for the whole report
    assert result.tool_calls == 5 and all(s.status == "done" for s in result.plan.steps)
    calc = result.plan.get("s5")
    assert result.evidence[calc.evidence_ids[0]].data["result"] == pytest.approx(0.156699, abs=1e-6)

    md = result.report_markdown
    assert md.startswith("# Microsoft: Company Profile")
    assert "MOCK data" in md and "## Research Methodology" in md
    assert result.report_checks == {"missing_sections": [], "unknown_citations": [], "uncited_sources": []}
    v = result.verification
    assert v["checked"] >= 3 and v["unsupported"] == 0 and v["uncited"] == 0
    # memory: the executive summary is stored for future sessions
    assert DEFAULT_STORE.search("Microsoft revenue", filter={"ticker": "MSFT"})["results"]


def test_the_trace_records_decisions_actions_and_llm_calls(tmp_path):
    result = make_agent(ScriptedLLM(), traces_dir=str(tmp_path)).run(QUERY)
    events = [json.loads(line) for line in open(result.trace_path, encoding="utf-8")]
    types = [e["type"] for e in events]
    assert types[0] == "session_start" and types[-1] == "session_end"
    assert types.count("llm_call") == 4
    assert sum(1 for e in events if e["type"] == "decision" and e.get("step_id")) == 5
    assert {e["tool"] for e in events if e["type"] == "action"} >= {"company_profile", "calculation_engine",
                                                                     "vector_db_search", "report_generator"}


def test_unusable_plan_falls_back_to_the_default_plan():
    script = ScriptedLLM(plan="I think we should look at Microsoft's filings.")
    result = make_agent(script).run(QUERY)
    assert script.stages[:2] == ["PLAN", "PLAN"]  # one retry with the parse error, then the default plan
    assert "Default plan" in result.plan.rationale
    assert {s.tool for s in result.plan.steps} >= {"company_profile", "financial_data_api", "sec_filing_search"}
    assert result.stop_reason == "completed" and result.report_markdown


def test_invalid_arguments_are_repaired_by_the_react_loop():
    plan = {"steps": [{"id": "s1", "goal": "Income statements", "tool": "financial_data_api",
                       "args": {"ticker": "MSFT"}}]}  # statement_type missing: the LLM repairs it
    script = ScriptedLLM(plan=json.dumps(plan), synthesis="=== SECTION: executive_summary ===\nText.")
    result = make_agent(script).run(QUERY)
    step = result.plan.get("s1")
    assert step.status == "done" and step.args["statement_type"] == "income_statement"
    assert "EXECUTE" in script.stages and result.tool_calls == 1


def test_data_errors_are_not_sent_to_the_llm_for_repair():
    """'No data for ticker' cannot be fixed by changing arguments; repairing it only burns budget."""
    plan = {"steps": [{"id": "s1", "goal": "Income statements", "tool": "financial_data_api",
                       "args": {"ticker": "ZZZZ", "statement_type": "income_statement"}}]}
    script = ScriptedLLM(plan=json.dumps(plan), synthesis="=== SECTION: executive_summary ===\nText.")
    result = make_agent(script).run(QUERY)
    assert result.plan.get("s1").status == "failed" and result.tool_calls == 1
    assert "EXECUTE" not in script.stages
    assert any("ZZZZ" in g for g in result.gaps)


def test_tool_budget_is_enforced_and_reported():
    """The planner caps the plan at the budget, so the budget can only run out at run time through
    repairs: here a repair needs two calls, leaving nothing for the second step."""
    plan = {"steps": [
        {"id": "s1", "goal": "Income", "tool": "financial_data_api", "args": {"ticker": "MSFT"}},  # needs repair
        {"id": "s2", "goal": "Profile", "tool": "company_profile", "args": {"ticker": "MSFT"}},
    ]}

    class Script(ScriptedLLM):
        def __call__(self, request):
            if "[ARA:EXECUTE]" in request.messages[-1]["content"]:
                self.stages.append("EXECUTE")
                return {"tool_calls": [  # first attempt still invalid, second one right
                    {"name": "financial_data_api", "arguments": {"ticker": "MSFT"}},
                    {"name": "financial_data_api", "arguments": {"ticker": "MSFT", "statement_type": "income_statement"}},
                ]}
            return super().__call__(request)

    script = Script(plan=json.dumps(plan), synthesis="=== SECTION: executive_summary ===\nText.")
    result = make_agent(script, max_tool_calls=2, max_workers=1).run(QUERY)  # sequential: deterministic
    assert result.tool_calls == 2 and result.stop_reason == "tool-call budget exhausted"
    assert [s.status for s in result.plan.steps] == ["done", "skipped"]
    assert any("budget ran out" in g for g in result.gaps)
    assert "## Data Gaps and Limitations" in result.report_markdown


def test_time_limit_stops_the_run():
    ticks = iter(range(0, 10_000, 100))  # every clock read advances 100 s
    result = make_agent(ScriptedLLM(), clock=lambda: next(ticks), max_seconds=250).run(QUERY)
    assert result.stop_reason == "time limit reached" and result.tool_calls < 5


def test_total_llm_outage_still_produces_a_transparent_report():
    down = MockProvider(lambda request: LLMProviderError("gemini", "server", "503 high demand"))
    result = make_agent(None, providers=[down]).run(QUERY)
    # the default plan needs no LLM, so the data is still gathered...
    assert result.tool_calls >= 4 and result.evidence
    # ...but no text could be written, and the report says so instead of crashing or inventing content
    assert result.sections == {} or all(k in ("data_gaps_and_limitations", "research_methodology") for k in result.sections)
    assert "could not be written" in result.report_markdown


def test_uncited_and_wrong_numbers_are_flagged():
    bad = ("=== SECTION: executive_summary ===\n"
           "Revenue was $999 billion [E2]. Operating margin was 12.3%.\n")
    result = make_agent(ScriptedLLM(synthesis=bad)).run(QUERY)
    v = result.verification
    assert v["uncited"] == 1 and v["unsupported"] + v["partially_supported"] >= 1
    assert "*(unverified" in result.sections["executive_summary"]
    assert "Unverified claim" in result.report_markdown


def test_missing_sections_trigger_a_gap_review():
    partial = "=== SECTION: executive_summary ===\nMicrosoft is a software company."
    script = ScriptedLLM(synthesis=partial)
    result = make_agent(script).run(QUERY)
    # reflection found four missing sections and asked for more research; the review found nothing
    # new to research, so the draft is not re-written (no wasted LLM call)
    assert script.stages == ["PLAN", "EXECUTE", "REPLAN", "SYNTHESIZE", "REPLAN"]
    assert result.report_checks["missing_sections"]  # still honestly reported as missing


def test_evidence_digest_shows_percentages_for_ratios():
    registry = build_default_registry("mock")
    ratios = registry.dispatch("financial_data_api", {"ticker": "MSFT", "statement_type": "key_ratios", "years": 1})
    growth = registry.dispatch("calculation_engine", {"calculation_type": "growth_rate",
                                                      "inputs": {"current": 245122, "previous": 211915}})
    assert "gross_margin=69.76%" in evidence_digest(ratios) and "MOCK DATA" in evidence_digest(ratios)
    assert "(= 15.67%)" in evidence_digest(growth)


# ==================================================================================================
# Regressions found in the first live run (Day 4)
# ==================================================================================================


def test_zero_width_characters_in_citations_are_normalised():
    """gpt-oss wrapped every citation in zero-width spaces, so none was recognised."""
    zw = "​"
    synthesis = f"=== SECTION: executive_summary ===\nRevenue was $245,122 million [{zw}E3{zw}] in FY2024.\n"
    sections = parse_sections(synthesis, ["executive_summary"])
    assert sections["executive_summary"] == "Revenue was $245,122 million [E3] in FY2024."


def test_table_rows_inherit_the_caption_citation_and_headers_are_not_claims():
    from agent.core import _claims, _mark

    body = ("Income statement highlights [E3]:\n\n"
            "| Fiscal Year (ended 30 Jun) | Revenue |\n|---|---|\n| 2024 | $245,122 |\n| 2023 | $211,915 [E9] |\n\n"
            "Later text 5% [E1].")
    claims = _claims(body)
    assert ("| 2024 | $245,122 |", {"E3"}) in claims
    assert ("| 2023 | $211,915 [E9] |", {"E3"}) in claims  # its own [E9] takes precedence at check time
    assert not any("Fiscal Year" in c for c, _ in claims)
    assert _mark("| 2024 | $1 |", "unverified") == "| 2024 | $1 *(unverified)* |"


def test_verification_follows_table_captions_in_a_full_run():
    def synthesis(prompt_ids):
        fin = prompt_ids
        return ("=== SECTION: financial_summary ===\n"
                f"Income statement (USD millions) [{fin}]:\n\n| FY | Revenue |\n|---|---|\n| 2024 | 245,122 |\n| 2023 | 211,915 |\n")

    class Script(ScriptedLLM):
        def __call__(self, request):
            prompt = request.messages[-1]["content"]
            if "[ARA:SYNTHESIZE]" in prompt:
                return synthesis(_ids(prompt, "financial_data_api")[0])
            return super().__call__(request)

    result = make_agent(Script()).run(QUERY)
    v = result.verification
    assert v["uncited"] == 0 and v["checked"] == 2 and v["supported"] == 2


def test_heading_nesting_is_relative():
    from tools.report_gen import nest_headings

    assert nest_headings("#### Detail\n##### Sub") == "#### Detail\n##### Sub"  # already deep enough
    assert nest_headings("# Big\n## Small") == "### Big\n#### Small"


def test_verifier_repairs_a_missing_citation_only_when_one_source_matches():
    uncited_table = ("=== SECTION: financial_summary ===\n"
                     "| FY | Revenue | Net income |\n|---|---|---|\n| 2024 | $245,122 | $88,136 |\n"
                     "Margins stayed high at 5%.\n")
    result = make_agent(ScriptedLLM(synthesis=uncited_table)).run(QUERY)
    body, v = result.sections["financial_summary"], result.verification
    income_id = next(e for e, r in result.evidence.items()
                     if r.tool == "financial_data_api" and r.args["statement_type"] == "income_statement")
    assert f"$88,136 [{income_id}] |" in body  # repaired: both figures match that one source exactly
    assert v["auto_cited"] == 1
    assert "5%. *(unverified: no source cited)*" in body  # a lone round number is never auto-cited
    assert "added by the verifier" in result.report_markdown


@pytest.mark.parametrize("query, template", [
    ("Analyze Apple Inc.'s most recent quarterly earnings. Compare actual results to consensus estimates.",
     "earnings_analysis"),
])
def test_compare_to_consensus_is_an_earnings_analysis(query, template):
    assert analyze_query(query).template == template


def test_sections_are_found_from_markdown_headings_when_markers_are_missing():
    text = "## Executive Summary\nA [E1].\n## Financial Risks\nB.\n### Detail\nC.\n## Risk Matrix\nD"
    assert parse_sections(text, ["executive_summary", "financial_risks", "risk_matrix"]) == {
        "executive_summary": "A [E1].", "financial_risks": "B.\n### Detail\nC.", "risk_matrix": "D"}


def test_fullwidth_citation_brackets_are_normalised():
    assert parse_sections("=== SECTION: x ===\nGrowth 114.2% 【E8】.", ["x"]) == {"x": "Growth 114.2% [E8]."}


def test_peer_ratios_are_shown_as_percentages_so_claims_verify():
    registry = build_default_registry("mock")
    peers = registry.dispatch("peer_comparison", {"ticker": "NVDA", "metrics": ["revenue_growth"]})
    digest = evidence_digest(peers)
    assert "revenue_growth=114.20%" in digest
    from tools.fact_checker import fact_checker
    assert fact_checker("NVIDIA revenue grew 114.2% [E1].", [digest])["status"] == "supported"


def test_a_slow_tool_call_times_out_and_the_report_is_still_written():
    """Live Day 5 run: one 10-K download took 350 s and left no time to write the report."""
    import time as _time

    plan = {"steps": [
        {"id": "s1", "goal": "Slow filing", "tool": "sec_filing_search", "args": {"ticker": "MSFT", "filing_type": "10-K"}},
        {"id": "s2", "goal": "Profile", "tool": "company_profile", "args": {"ticker": "MSFT"}},
    ]}
    script = ScriptedLLM(plan=json.dumps(plan), synthesis="=== SECTION: executive_summary ===\nMicrosoft [E2].")

    def factory():
        registry = build_default_registry("mock")
        registry.bind_handler("sec_filing_search", lambda **kw: _time.sleep(5) or {})
        return registry

    llm = LLMClient([MockProvider(script)], max_retries=0, sleep=lambda s: None)
    agent = ResearchAgent(llm, factory, agent_settings=AgentSettings(), max_seconds=4, max_workers=2)
    started = _time.monotonic()
    result = agent.run(QUERY)
    assert _time.monotonic() - started < 4.5  # the 5 s download did not hold the run hostage
    assert result.plan.get("s1").status == "failed" and "background" in result.plan.get("s1").summary
    assert result.plan.get("s2").status == "done"
    assert "SYNTHESIZE" in script.stages and result.sections.get("executive_summary")  # time was kept for writing


def test_planner_is_told_not_to_assume_the_current_year():
    from agent import prompts

    text = prompts.plan_prompt(query="q", template="company_profile", sections=["executive_summary"], entities=["MSFT"],
                               memory="", tool_names=["sec_filing_search"], max_steps=5)
    assert "Omit \"year\"" in text and "training data is older" in text
