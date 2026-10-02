"""Tests for the Day 13 optimisations: execute-call schemas, provider budgets, deterministic review,
ticker enforcement in plans, section-targeted memory recall and the wall-clock writing reserve."""

from __future__ import annotations

import json

import pytest
from test_core import QUERY, ScriptedLLM, make_agent

from agent import core, prompts
from agent.llm import AllProvidersFailedError, LLMClient, LLMProviderError, MockProvider
from agent.parser import Plan, PlanStep
from config.settings import AgentSettings, load_settings
from memory.vector_store import DEFAULT_STORE
from tools import build_default_registry


@pytest.fixture(autouse=True)
def _clean_memory():
    DEFAULT_STORE.clear()
    yield
    DEFAULT_STORE.clear()


def _events(result):
    return [json.loads(line) for line in open(result.trace_path, encoding="utf-8")]


# ---------------------------------------------------------------------------------------------- T1


class Recording(ScriptedLLM):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.execute_requests = []

    def __call__(self, request):
        if "[ARA:EXECUTE]" in request.messages[-1]["content"]:
            self.execute_requests.append(request)
        return super().__call__(request)


def test_execute_call_carries_only_the_planned_tools_schema():
    script = Recording()
    make_agent(script).run(QUERY)
    assert script.execute_requests
    for req in script.execute_requests:
        assert len(req.tools) == 1
        prompt = req.messages[-1]["content"]
        assert req.tools[0]["function"]["name"] in prompt
        assert "Other research tools" in prompt and "news_sentiment" in prompt


def test_unknown_planned_tool_falls_back_to_all_schemas():
    script = Recording()
    agent = make_agent(script)
    agent.run(QUERY)  # sets up the registry, deadlines and budget
    agent._deadline = agent._research_deadline = agent._clock() + 300
    agent.budget_used = 0
    script.execute_requests.clear()
    # a step whose tool has no schema among the research tools gets every schema, and no "other tools" line
    step = PlanStep(id="s9", goal="Get Microsoft revenue", tool="no_such_tool", args=None)
    agent._run_step(step, Plan(steps=[step]))
    research = [t for t in agent.registry.openai_tools() if t["function"]["name"] not in core.SYSTEM_TOOLS]
    assert script.execute_requests and len(script.execute_requests[0].tools) == len(research) > 1
    assert "Other research tools" not in script.execute_requests[0].messages[-1]["content"]


def test_execute_prompt_lists_other_tools_only_when_given():
    assert "Other research tools" not in prompts.execute_step_prompt(goal="g", tool="t", evidence="")
    text = prompts.execute_step_prompt(goal="g", tool="t", evidence="", other_tools=["a", "b"])
    assert "a, b" in text


# ---------------------------------------------------------------------------------------------- T2


def test_prompt_budget_by_provider_with_default_and_headroom():
    cfg = AgentSettings(max_prompt_tokens=6000, prompt_tokens_by_provider={"gemini": 24000, "groq": 6000})
    assert cfg.prompt_budget("gemini") == 21600
    assert cfg.prompt_budget("groq") == 5400
    assert cfg.prompt_budget("other") == 5400 and cfg.prompt_budget(None) == 5400


def test_default_config_sets_the_provider_caps():
    cfg = load_settings(env={"ARA_MODE": "mock"}).agent
    assert cfg.prompt_tokens_by_provider["gemini"] == 24000 and cfg.prompt_tokens_by_provider["groq"] == 6000


def test_next_provider_name_follows_the_cooling_order():
    now = [100.0]
    llm = LLMClient([MockProvider(name="gemini"), MockProvider(name="groq")], clock=lambda: now[0])
    assert llm.next_provider_name() == "gemini" and llm.any_provider_ready()
    llm._cooldown_until["gemini"] = 130.0
    assert llm.next_provider_name() == "groq"
    llm._cooldown_until["groq"] = 120.0
    assert llm.next_provider_name() == "gemini" and not llm.any_provider_ready()  # all cooling: chain order
    llm._cooldown_until["gemini"] = 100.0 + 3600
    llm._cooldown_until["groq"] = 100.0 + 3600
    assert llm.next_provider_name() is None
    now[0] = 100.0 + 3601
    assert llm.next_provider_name() == "gemini"


def test_synthesis_budget_is_traced_with_the_provider(tmp_path):
    agent = make_agent(ScriptedLLM(), traces_dir=str(tmp_path),
                       agent_settings=AgentSettings(prompt_tokens_by_provider={"mock": 12000}))
    result = agent.run(QUERY)
    budget = next(e for e in _events(result) if e["type"] == "metric" and e["name"] == "synthesis_prompt_budget")
    assert budget["value"]["provider"] == "mock" and budget["value"]["budget_tokens"] == 10800


class FailFirstSynthesis(ScriptedLLM):
    """The large-context provider fails on the first synthesis call (for example its quota runs out)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.synth_prompts = []

    def __call__(self, request):
        prompt = request.messages[-1]["content"]
        if "[ARA:SYNTHESIZE]" in prompt:
            self.synth_prompts.append(prompt)
            if len(self.synth_prompts) == 1:
                raise LLMProviderError("mock", "bad_request", "request too large")
        return super().__call__(request)


def test_synthesis_retries_with_the_default_budget_when_a_large_budget_call_fails(tmp_path):
    script = FailFirstSynthesis()
    agent = make_agent(script, traces_dir=str(tmp_path),
                       agent_settings=AgentSettings(prompt_tokens_by_provider={"mock": 24000}))
    result = agent.run(QUERY)
    assert len(script.synth_prompts) == 2 and result.sections  # the report was still written
    notes = [e.get("text", "") for e in _events(result) if e["type"] == "note"]
    assert any("retrying with the default 5400-token budget" in n for n in notes)


def test_synthesis_failure_with_the_default_budget_is_not_retried(tmp_path):
    script = FailFirstSynthesis()
    result = make_agent(script, traces_dir=str(tmp_path)).run(QUERY)  # mock uses the default cap
    assert len(script.synth_prompts) == 1 and not result.sections
    assert any("could not be written" in g for g in result.gaps)


# ---------------------------------------------------------------------------------------------- T3 / T4


PARTIAL = "=== SECTION: executive_summary ===" + chr(10) + "Microsoft is a software company."


def test_review_is_skipped_when_every_step_is_done_and_no_gap_is_noted(tmp_path):
    script = ScriptedLLM()
    result = make_agent(script, traces_dir=str(tmp_path)).run(QUERY)
    assert "REPLAN" not in script.stages
    decision = next(e for e in _events(result) if e["type"] == "decision" and e["chosen"] == "research_complete")
    assert "deterministic check" in decision["rationale"]


def test_review_still_runs_when_reflection_notes_a_gap(tmp_path):
    script = ScriptedLLM(synthesis=PARTIAL)  # four sections missing: reflection asks for a gap review
    make_agent(script, traces_dir=str(tmp_path)).run(QUERY)
    assert script.stages == ["PLAN", "EXECUTE", "SYNTHESIZE", "REPLAN"]


def test_review_is_skipped_when_no_llm_provider_is_available(tmp_path):
    script = ScriptedLLM(synthesis=PARTIAL)
    agent = make_agent(script, traces_dir=str(tmp_path))
    agent.llm.next_provider_name = lambda: None  # every provider out of quota or offline
    result = agent.run(QUERY)
    assert "REPLAN" not in script.stages  # without the check, the gap review above would have been called
    reasons = [e["rationale"] for e in _events(result) if e["type"] == "decision" and e["chosen"] == "research_complete"]
    assert any("no LLM provider is available" in r for r in reasons)


def test_review_still_runs_when_providers_are_only_briefly_rate_limited(tmp_path):
    script = ScriptedLLM(synthesis=PARTIAL)
    agent = make_agent(script, traces_dir=str(tmp_path))
    agent.llm.any_provider_ready = lambda: False  # all cooling for seconds, none out of quota
    agent.run(QUERY)
    assert "REPLAN" in script.stages


def test_rate_limit_waits_never_pass_the_research_deadline():
    """Day 12 stress finding: waiting out a rate limit must not eat into the time reserved for writing.
    Research calls get ``deadline=_research_deadline``; every wait is clipped to it."""
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limited = LLMProviderError("gemini", "rate_limit", "slow down", retry_after=500.0)
    llm = LLMClient([MockProvider([limited] * 10, name="gemini")], max_retries=5, sleep=sleep,
                    clock=lambda: now[0], rng=lambda: 0.0)
    research_deadline = 225.0  # 300 s budget minus the quarter kept for writing
    with pytest.raises(AllProvidersFailedError):
        llm.complete([{"role": "user", "content": "x"}], deadline=research_deadline)
    assert now[0] <= research_deadline and sum(sleeps) <= research_deadline


def test_the_writing_reserve_is_a_quarter_of_the_budget():
    agent = make_agent(ScriptedLLM(), max_seconds=300.0)
    agent.run(QUERY)
    assert agent._deadline - agent._research_deadline == 75.0


# ---------------------------------------------------------------------------------------------- P2


def _step(sid, tool, goal, args=None, depends_on=()):
    return PlanStep(id=sid, goal=goal, tool=tool, args=args, depends_on=list(depends_on))


def test_steps_without_a_ticker_are_dropped_when_no_company_is_known():
    kept, notes = core.enforce_tickers([_step("r1", "earnings_transcript", "Call earnings_transcript")], [])
    assert kept == [] and "no company is known" in notes[0]


def test_a_single_known_ticker_is_added_to_the_goal():
    kept, notes = core.enforce_tickers([_step("r1", "earnings_transcript", "Call earnings_transcript")], ["PLTR"])
    assert kept[0].goal.endswith("for PLTR") and "set to PLTR" in notes[0]


def test_ambiguous_steps_are_dropped_but_named_ones_kept():
    steps = [_step("a", "earnings_transcript", "Earnings call"), _step("b", "earnings_transcript", "Call for AAPL")]
    kept, notes = core.enforce_tickers(steps, ["AAPL", "MSFT"])
    assert [s.id for s in kept] == ["b"] and "AAPL, MSFT" in notes[0]


def test_steps_that_need_no_ticker_or_have_args_are_untouched():
    steps = [_step("a", "web_search", "Sector news"), _step("b", "company_profile", "x", args={"ticker": "MSFT"}),
             _step("c", "calculation_engine", "growth", depends_on=["b"]),
             _step("d", "financial_data_api", "needs b", depends_on=["b"])]
    kept, notes = core.enforce_tickers(steps, [])
    assert [s.id for s in kept] == ["a", "b", "c", "d"] and notes == []


def test_dependencies_on_dropped_steps_are_removed():
    steps = [_step("a", "earnings_transcript", "Call it"), _step("b", "web_search", "news", depends_on=["a"])]
    kept, _ = core.enforce_tickers(steps, [])
    assert kept[0].depends_on == []


def test_plan_node_names_memory_tickers_in_the_prompt_and_fixes_steps():
    plan = {"steps": [{"id": "s1", "goal": "Transcript", "tool": "earnings_transcript", "args": None},
                      {"id": "s2", "goal": "News", "tool": "news_sentiment", "args": {"query": "tech"}}]}
    seen = []

    class Script(ScriptedLLM):
        def __call__(self, request):
            seen.append(request.messages[-1]["content"])
            return super().__call__(request)

    agent = make_agent(Script(plan=json.dumps(plan), synthesis="=== SECTION: executive_summary ===\nText."))
    agent.registry = build_default_registry("mock")
    agent.tracer = core.TraceLogger()
    agent._research_deadline = agent._deadline = 1e12
    agent._system = agent._writing_system = "s"
    agent.llm.tracer = agent.tracer
    out = agent._node_plan({"query": "themes across tech", "profile": core.analyze_query("themes across tech"),
                            "memory_tickers": ["MSFT"], "memory_notes": ""})
    assert "Companies from memory (use these for company-specific tools): MSFT" in seen[0]
    assert out["plan"].get("s1").goal.endswith("for MSFT")


# ---------------------------------------------------------------------------------------------- M1


def _seed(store, ticker, texts):
    for i, text in enumerate(texts):
        store.store(text, {"ticker": ticker, "date": f"2025-0{i + 1}-01", "source_type": "10-K"})


def _recall(agent, monkeypatch, floor):
    monkeypatch.setattr(core, "RECALL_MIN_SCORE", floor)
    agent.registry = build_default_registry("mock")
    agent.tracer = core.TraceLogger()
    agent.evidence = {}
    profile = core.analyze_query(QUERY)
    out = agent._node_recall({"query": QUERY, "profile": profile})
    return out, agent


def test_targeted_recall_dedupes_hits_across_searches(monkeypatch):
    _seed(DEFAULT_STORE, "MSFT", ["Microsoft business overview segments cloud products customers",
                                  "Microsoft financial summary revenue margin growth cash flow"])
    out, agent = _recall(make_agent(ScriptedLLM()), monkeypatch, 0.0)
    ids = [e.data["results"] for e in agent.evidence.values()]
    flat = [h["id"] for hits in ids for h in hits]
    assert flat and len(flat) == len(set(flat))
    assert out["memory_notes"].count("Microsoft business overview") == 1


# Five records close to the query fill the query-wide top 5; two short ones match only a section topic.
EIGHT = [f"comprehensive profile Microsoft Corporation key executives recent developments {i}" for i in range(5)] + [
    "business overview", "financial summary"]


def test_targeted_recall_applies_the_relevance_floor(monkeypatch):
    _seed(DEFAULT_STORE, "MSFT", EIGHT)
    _, agent_low = _recall(make_agent(ScriptedLLM()), monkeypatch, 0.0)
    _, agent_high = _recall(make_agent(ScriptedLLM()), monkeypatch, 0.99)
    assert len(agent_high.evidence) == 1  # the floor removes every targeted hit; the query-wide search stays
    assert len(agent_low.evidence) > 1


def test_query_wide_recall_keeps_hits_with_a_negative_score(monkeypatch):
    # Cosine similarity can be below zero; the query-wide search had no floor before Day 13 and has none now.
    agent = make_agent(ScriptedLLM())
    real = build_default_registry("mock")
    hit = {"id": "r1", "content": "Microsoft note", "score": -0.05, "metadata": {"date": "2025-01-01"}}

    class Stub:
        def __getattr__(self, name):
            return getattr(real, name)

        def dispatch(self, name, args):
            result = real.dispatch(name, args)
            return result if args["query"] != QUERY else core.dataclasses.replace(
                result, data={**result.data, "results": [hit]})

    monkeypatch.setattr(core, "RECALL_MIN_SCORE", 0.3)
    agent.registry = Stub()
    agent.tracer = core.TraceLogger()
    agent.evidence = {}
    out = agent._node_recall({"query": QUERY, "profile": core.analyze_query(QUERY)})
    assert "Microsoft note" in out["memory_notes"]


def test_recall_makes_only_local_memory_calls_and_skips_companies_with_no_memory(monkeypatch):
    out, agent = _recall(make_agent(ScriptedLLM()), monkeypatch, 0.0)
    log = agent.registry.call_log
    assert {c.tool for c in log} == {"vector_db_search"}
    assert len(log) == 1  # nothing in memory for MSFT: the targeted searches are not run
    assert out["memory_tickers"] == []  # the query names a company


def test_recall_topics_come_from_the_template():
    assert core.recall_topics("company_profile") == ["Business Overview", "Financial Summary"]
    assert core.recall_topics("risk_assessment") == ["Financial Risks", "Operational Risks"]


def test_one_slow_tool_call_cannot_use_the_whole_research_window():
    # Day 14, C4: one earnings-transcript download blocked for 213 s and the calculation steps never ran.
    import time

    from tools.base import ToolExecutionError

    agent = make_agent(ScriptedLLM(), agent_settings=AgentSettings(tool_timeout_s=1.0))
    agent.run(QUERY)  # sets up the registry, dispatcher and worker pool
    agent._research_deadline = agent._clock() + 300  # plenty of research time left
    agent.dispatcher.call = lambda tool, args: time.sleep(4)
    started = time.monotonic()
    with pytest.raises(ToolExecutionError, match="allowed per tool call"):
        agent._dispatch("earnings_transcript", {"ticker": "MSFT"})
    assert time.monotonic() - started < 3
