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
from memory.episodic import EpisodicMemory
from memory.facts import FactsStore
from memory.manager import MemoryManager
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
    options = {"agent_settings": AgentSettings(max_tool_calls=max_tool_calls), "max_workers": 4,
               "memory": MemoryManager(DEFAULT_STORE, FactsStore(None), EpisodicMemory(None))}
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
    # memory: findings and source material are stored for future sessions
    assert DEFAULT_STORE.search("Microsoft revenue", filter={"ticker": "MSFT", "source_type": "analysis"})["results"]


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


# ==================================================================================================
# Regressions found in the first live Challenge 3 run (Day 7)
# ==================================================================================================


def test_peer_pe_and_market_cap_carry_their_units():
    """A bare "367.30" under a "ratios as percentages" header came out as a P/E of 36,730.00 %."""
    registry = build_default_registry("mock")
    peers = registry.dispatch("peer_comparison", {"ticker": "MSFT", "metrics": ["pe_ratio", "market_cap"]})
    digest = evidence_digest(peers)
    assert "pe_ratio=" in digest and re.search(r"pe_ratio=[\d,.]+x", digest)
    assert "market_cap=$3,100.0 bn" in digest and "ratios as percentages" not in digest
    from tools.fact_checker import fact_checker
    assert fact_checker("Microsoft's market cap is $3.1 trillion [E1].", [digest])["status"] == "supported"


def test_list_items_inherit_the_intro_citation_and_markers_are_not_figures():
    from agent.core import _claims

    body = ("Execution risks cited in the 10-K [E5]:\n"
            "1. Delays in ramping new products.\n"
            "2. Expansion of internal 4680 cell manufacturing.\n\n"
            "A new paragraph with 12% growth.\n"
            "- Unrelated bullet with $5 billion.")
    claims = _claims(body)
    assert ("Delays in ramping new products.", {"E5"}) in claims  # "1." is gone: not a figure to check
    assert ("Expansion of internal 4680 cell manufacturing.", {"E5"}) in claims
    assert ("Unrelated bullet with $5 billion.", set()) in claims  # the intro before it had no colon


def test_transcript_digest_keeps_figures_guidance_and_questions_within_budget():
    from agent.core import salient_statements

    segments = [
        {"speaker": "Travis Axelrod", "role": "Head of Investor Relations", "section": "prepared",
         "text": "Good afternoon, everyone, and welcome to the webcast. " * 30},
        {"speaker": "Vaibhav Taneja", "role": "Chief Financial Officer", "section": "prepared",
         "text": "Automotive gross margin was 17.2% this quarter. We expect capital expenditures to exceed $20 billion "
                 "next year. The team did great work across the board this quarter."},
        {"speaker": "Adam Jonas", "role": "Analyst", "section": "qa",
         "text": "Thanks for taking my question today. How should we think about robotaxi margins over time?"},
    ]
    lines = salient_statements(segments, 400)
    assert sum(len(x) + 1 for x in lines) <= 400
    assert lines[0].startswith("[prepared] Vaibhav Taneja (Chief Financial Officer): Automotive gross margin was 17.2%")
    assert any("expect capital expenditures to exceed $20 billion" in x for x in lines)
    assert any(x.startswith("[Q&A] Adam Jonas (Analyst): How should we think") for x in lines)
    assert not any("welcome to the webcast" in x for x in lines)
    everything = salient_statements(segments, 100_000)  # fact checking: every sentence is kept
    assert any("welcome to the webcast" in x for x in everything) and any("great work" in x for x in everything)


def test_abbreviations_do_not_end_a_claim():
    from agent.core import _claims

    body = "Tesla trades at 367.3x P/E vs. GM's 27.0x P/E [E8]. Ford Motor Co. earned $5 billion [E3]."
    assert [c for c, _ in _claims(body)] == ["Tesla trades at 367.3x P/E vs. GM's 27.0x P/E [E8].",
                                            "Ford Motor Co. earned $5 billion [E3]."]


def test_transcript_source_names_the_quarter_retrieved_and_its_date():
    from agent.core import source_entry

    registry = build_default_registry("mock")
    result = registry.dispatch("earnings_transcript", {"ticker": "AAPL", "quarter": "Q4", "year": 2024})
    entry = source_entry(result)
    assert entry["title"] == "earnings_transcript: AAPL Q4 FY2024 (call transcript)"
    assert entry["date"] == "2024-10-31"


def test_claims_citing_a_filing_are_checked_against_the_whole_document():
    """The 10-K's "4680" was flagged as unsupported: the verifier saw a 6,000-character digest of it."""
    from agent.core import verification_text

    registry = build_default_registry("mock")
    filing = registry.dispatch("sec_filing_search", {"ticker": "TSLA", "filing_type": "10-K"})
    last_section = filing.data["sections"][-1]
    filing.data["sections"][-1] = {**last_section, "text": "x " * 5000 + "Internal 4680 cell production grew."}
    assert "4680" not in evidence_digest(filing, 6000) and "4680" in verification_text(filing)


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


# ==================================================================================================
# Day 6: memory across runs (spec task: run Challenge 1, store, then Challenge 7 retrieves it)
# ==================================================================================================

C7_QUERY = ("Based on the companies you've already researched, what themes emerge across the technology sector? "
            "Identify cross-cutting risks and opportunities.")


class RecordingLLM(ScriptedLLM):
    """A scripted LLM that also keeps every prompt, to check what the planner was told."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompts: list[str] = []

    def __call__(self, request):
        self.prompts.append(request.messages[-1]["content"])
        return super().__call__(request)


def test_challenge_7_retrieves_what_challenge_1_stored():
    memory = MemoryManager(DEFAULT_STORE, FactsStore(None), EpisodicMemory(None))
    first = make_agent(ScriptedLLM(), memory=memory).run(QUERY)
    assert first.stop_reason == "completed"
    assert "MSFT" in memory.known_companies()
    assert memory.facts.value("MSFT", "revenue", "2024-06-30") == 245122

    thematic_plan = {"steps": [{"id": "s1", "goal": "Sector news", "tool": "web_search",
                                "args": {"query": "technology sector themes 2026"}}]}
    script = RecordingLLM(plan=json.dumps(thematic_plan),
                          synthesis="=== SECTION: executive_summary ===\nThemes from memory [E1].")
    second = make_agent(script, memory=memory).run(C7_QUERY)

    assert second.template == "thematic"
    recalled = [r for r in second.evidence.values() if r.tool == "vector_db_search"]
    assert recalled and recalled[0].args.get("ticker") == "MSFT"  # no company named, so memory supplied one
    assert any("Microsoft" in hit["content"] for r in recalled for hit in r.data["results"])
    plan_prompt = next(p for p in script.prompts if "[ARA:PLAN]" in p)
    assert "Microsoft" in plan_prompt  # the planner was shown the recalled research


def test_the_second_run_of_a_report_type_gets_lessons_from_the_first():
    memory = MemoryManager(DEFAULT_STORE, FactsStore(None), EpisodicMemory(None))
    make_agent(ScriptedLLM(), memory=memory).run(QUERY)
    script = RecordingLLM()
    make_agent(script, memory=memory).run(QUERY)
    plan_prompt = next(p for p in script.prompts if "[ARA:PLAN]" in p)
    assert "Lessons from 1 earlier company profile run" in plan_prompt
    assert "financial_data_api was useful" in plan_prompt
    assert "MSFT financial statements in memory" in plan_prompt


def test_synthesis_prompt_respects_the_token_budget():
    from memory.context_manager import count_tokens

    script = RecordingLLM()
    agent = make_agent(script, agent_settings=AgentSettings(max_prompt_tokens=1500))
    agent.run(QUERY)
    synthesis_prompt = next(p for p in script.prompts if "[ARA:SYNTHESIZE]" in p)
    total = count_tokens(agent._writing_system) + count_tokens(synthesis_prompt)
    assert total <= 1500 + 60  # only the "left out" note may overflow
    assert count_tokens(agent._writing_system) < count_tokens(agent._system) - 500  # no tool list when writing


# ==================================================================================================
# Audit regressions
# ==================================================================================================


def test_dependencies_on_steps_dropped_by_the_step_limit_or_a_cycle_are_removed():
    """A kept step must never wait for a step that validation dropped: Plan.ready() would never
    return it, so it could not run and would be skipped as if the budget had run out."""
    registry = build_default_registry("mock")
    text = json.dumps({"steps": [
        {"id": "s1", "tool": "web_search", "args": {"query": "a"}, "depends_on": ["s3"]},
        {"id": "c1", "tool": "web_search", "args": {"query": "b"}, "depends_on": ["c2"]},
        {"id": "c2", "tool": "web_search", "args": {"query": "c"}, "depends_on": ["c1"]},
        {"id": "s2", "tool": "web_search", "args": {"query": "d"}, "depends_on": ["c1"]},
        {"id": "s3", "tool": "web_search", "args": {"query": "e"}},  # over the limit of 4
    ]})
    plan = parse_plan(text, registry, max_steps=4)
    ids = {s.id for s in plan.steps}
    assert ids == {"s1", "s2"}
    assert all(set(s.depends_on) <= ids for s in plan.steps)
    assert {s.id for s in plan.ready()} == {"s1", "s2"}


def test_replan_steps_without_ids_get_fresh_ids_instead_of_being_rejected_as_duplicates():
    registry = build_default_registry("mock")
    plan = Plan([PlanStep("s1", "profile", "company_profile", {"ticker": "MSFT"}, status="done")])
    text = json.dumps({"action": "continue", "add_steps": [
        {"tool": "web_search", "args": {"query": "Microsoft Azure growth"}},
    ]})
    decision = parse_replan(text, registry, plan, max_new_steps=5)
    assert not decision.done and len(decision.new_steps) == 1
    assert decision.new_steps[0].id != "s1"


def test_budget_running_out_before_a_dependent_step_is_reported_as_the_stop_reason():
    """The repair of s1 uses the last tool call; s2 (which waits for s1) can then never run. The
    run must say it stopped because the budget ran out, not that it completed."""
    plan = {"steps": [
        {"id": "s1", "goal": "Income", "tool": "financial_data_api", "args": {"ticker": "MSFT"}},  # needs repair
        {"id": "s2", "goal": "Profile", "tool": "company_profile", "args": {"ticker": "MSFT"}, "depends_on": ["s1"]},
    ]}

    class Script(ScriptedLLM):
        def __call__(self, request):
            if "[ARA:EXECUTE]" in request.messages[-1]["content"]:
                self.stages.append("EXECUTE")
                return {"tool_calls": [
                    {"name": "financial_data_api", "arguments": {"ticker": "MSFT"}},
                    {"name": "financial_data_api", "arguments": {"ticker": "MSFT", "statement_type": "income_statement"}},
                ]}
            return super().__call__(request)

    script = Script(plan=json.dumps(plan), synthesis="=== SECTION: executive_summary ===\nText.")
    result = make_agent(script, max_tool_calls=2, max_workers=1).run(QUERY)
    assert [s.status for s in result.plan.steps] == ["done", "skipped"] and result.tool_calls == 2
    assert result.stop_reason == "tool-call budget exhausted"


def test_a_failed_gap_review_does_not_rewrite_the_same_draft():
    """Reflection found missing sections, but the review reply was unusable, so nothing new was
    researched: re-running synthesis would only spend a large LLM call re-writing the same draft
    (and a failed re-write would throw the existing draft away)."""

    class Script(ScriptedLLM):
        replans = 0

        def __call__(self, request):
            if "[ARA:REPLAN]" in request.messages[-1]["content"]:
                self.replans += 1
                self.stages.append("REPLAN")
                return '{"action": "done"}' if self.replans == 1 else "not json at all"
            return super().__call__(request)

    script = Script(synthesis="=== SECTION: executive_summary ===\nMicrosoft is a software company.")
    result = make_agent(script).run(QUERY)
    assert script.stages == ["PLAN", "EXECUTE", "REPLAN", "SYNTHESIZE", "REPLAN"]
    assert result.sections["executive_summary"].startswith("Microsoft is a software company.")


def test_a_complete_report_written_after_the_research_deadline_keeps_the_completed_stop_reason():
    """Writing is meant to happen in the reserve after the research deadline. Reflection must not
    record 'time limit reached' when no section is missing and nothing was cut short."""
    now = [0.0]

    class Script(ScriptedLLM):
        def __call__(self, request):
            if "[ARA:SYNTHESIZE]" in request.messages[-1]["content"]:
                now[0] = 800.0  # past the research deadline (750 s), within the writing reserve
            return super().__call__(request)

    script = Script()
    llm = LLMClient([MockProvider(script)], max_retries=0, sleep=lambda s: None, clock=lambda: now[0])
    agent = ResearchAgent(llm, lambda: build_default_registry("mock"), agent_settings=AgentSettings(),
                          max_seconds=1000, clock=lambda: now[0],
                          memory=MemoryManager(DEFAULT_STORE, FactsStore(None), EpisodicMemory(None)))
    result = agent.run(QUERY)
    assert result.report_checks["missing_sections"] == []
    assert result.stop_reason == "completed"


def test_a_repeated_uncited_sentence_is_marked_once_at_each_occurrence():
    body = "Microsoft had 123,456 widgets in FY2024. Microsoft had 123,456 widgets in FY2024."
    result = make_agent(ScriptedLLM(synthesis=f"=== SECTION: executive_summary ===\n{body}\n")).run(QUERY)
    text = result.sections["executive_summary"]
    assert result.verification["uncited"] == 2
    assert text.count("*(unverified: no source cited)*") == 2
    assert "*(unverified: no source cited)* *(unverified" not in text


def test_extract_json_skips_an_unbalanced_brace_in_prose_before_the_object():
    assert extract_json('I will fill in {ticker later. {"a": 1}') == {"a": 1}


def test_the_executor_refuses_system_tools_just_as_the_planner_does():
    """The planner rejects report_generator / fact_checker / vector_db_*; a model must not be able to
    reach them through a step's tool call either (e.g. writing its own 'findings' into memory)."""
    plan = {"steps": [{"id": "s1", "goal": "Recent news", "tool": "web_search", "args": None}]}

    class Script(ScriptedLLM):
        def __call__(self, request):
            if "[ARA:EXECUTE]" in request.messages[-1]["content"]:
                self.stages.append("EXECUTE")
                return {"tool_calls": [{"name": "vector_db_store", "arguments": {
                    "content": "Microsoft revenue will double next year.",
                    "metadata": {"ticker": "MSFT", "date": "2026-01-01", "source_type": "analysis"}}}]}
            return super().__call__(request)

    script = Script(plan=json.dumps(plan), synthesis="=== SECTION: executive_summary ===\nText.")
    result = make_agent(script).run(QUERY)
    assert not any(r.tool == "vector_db_store" for r in result.evidence.values())
    assert result.plan.get("s1").status != "done"


# ==================================================================================================
# Day 8: the synthesis engine in the loop
# ==================================================================================================


def test_writer_gets_the_synthesis_brief_and_the_report_lists_conflicts(monkeypatch):
    from synthesis.conflict_resolver import ConflictRecord, Period
    from synthesis.engine import build_ledger

    def with_conflict(evidence, config=None):
        ledger = build_ledger(evidence, config)
        eid = next(iter(evidence))
        ledger.conflicts.append(ConflictRecord(
            entity="MSFT", metric="revenue", period=Period(2024, "FY", "2024-06-30"),
            values=[{"value": 245122e6, "evidence_id": eid, "tier": 1, "provider": "sec_xbrl"},
                    {"value": 262000e6, "evidence_id": eid, "tier": 4, "provider": "news"}],
            chosen_value=245122e6, method="highest_tier",
            rationale="The SEC filing (tier 1) outranks the news report (tier 4).", confidence=0.95))
        return ledger

    monkeypatch.setattr("agent.core.build_ledger", with_conflict)
    script = ScriptedLLM()
    prompts_seen = []
    original = script.__call__

    class Capture:
        def __call__(self, request):
            prompts_seen.append(request.messages[-1]["content"])
            return original(request)

    result = make_agent(Capture()).run(QUERY)
    synthesis_prompt = next(p for p in prompts_seen if "[ARA:SYNTHESIZE]" in p)
    assert "SYNTHESIS BRIEF" in synthesis_prompt and "MSFT FY" in synthesis_prompt
    report = result.report_markdown
    assert "Conflicting figures between sources" in report
    assert "| MSFT | revenue |" in report.replace("Revenue", "revenue")
