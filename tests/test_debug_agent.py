"""Regression tests for bugs found in the agent/ debug pass (mock mode only, no network)."""

from __future__ import annotations

import json

from agent.parser import Plan, PlanStep, parse_replan
from tools import build_default_registry


def test_replan_step_depending_on_a_rejected_repeat_still_becomes_ready():
    """A review step that depends on a step rejected as a repeat used to wait for it forever."""
    registry = build_default_registry("mock")
    plan = Plan([PlanStep("s1", "income", "financial_data_api",
                          {"ticker": "MSFT", "statement_type": "income_statement"}, status="done")])
    plan.steps[0].args = registry.validate_args("financial_data_api", plan.steps[0].args)
    text = json.dumps({"action": "continue", "add_steps": [
        {"id": "r1", "tool": "financial_data_api", "args": {"ticker": "MSFT", "statement_type": "income_statement"}},
        {"id": "r2", "tool": "calculation_engine", "args": None, "depends_on": ["r1"]},
    ]})
    decision = parse_replan(text, registry, plan, max_new_steps=5)
    assert [s.id for s in decision.new_steps] == ["r2"]
    plan.steps.extend(decision.new_steps)
    # r2 must not wait for r1, which is not in the plan; it uses s1, which made the same call.
    assert [s.id for s in plan.ready()] == ["r2"]
    assert plan.get("r2").depends_on == ["s1"]


def test_all_steps_done_just_after_the_research_deadline_keeps_the_completed_stop_reason():
    """The last step finished after the research deadline; every step was done and nothing was cut
    short, yet the review's time check recorded 'time limit reached'."""
    from agent.core import ResearchAgent
    from agent.llm import LLMClient, MockProvider
    from config.settings import AgentSettings
    from tests.test_core import QUERY, ScriptedLLM

    now = [0.0]

    class Script(ScriptedLLM):
        def __call__(self, request):
            if "[ARA:EXECUTE]" in request.messages[-1]["content"]:
                now[0] = 800.0  # the calculation step's tool call lands past the research deadline (750 s)
            return super().__call__(request)

    script = Script()
    llm = LLMClient([MockProvider(script)], max_retries=0, sleep=lambda s: None, clock=lambda: now[0])
    agent = ResearchAgent(llm, lambda: build_default_registry("mock"), agent_settings=AgentSettings(),
                          max_seconds=1000, clock=lambda: now[0], max_workers=1, memory=None)
    result = agent.run(QUERY)
    assert all(s.status == "done" for s in result.plan.steps)
    assert result.report_checks["missing_sections"] == []
    assert result.stop_reason == "completed"


def test_a_claim_joined_across_an_abbreviation_with_two_spaces_is_still_marked():
    """'Corp.  Revenue ...' was rejoined with one space, so the verifier counted the uncited claim
    but could not find it in the body to mark it."""
    from tests.test_core import QUERY, ScriptedLLM, make_agent

    body = "Microsoft Corp.  Revenue was 123,456 widgets in FY2024."
    result = make_agent(ScriptedLLM(synthesis=f"=== SECTION: executive_summary ===\n{body}\n")).run(QUERY)
    assert result.verification["uncited"] == 1
    assert "*(unverified: no source cited)*" in result.sections["executive_summary"]


def test_default_plan_uses_the_companies_recalled_from_memory():
    """A cross-memory query names no company; recall supplies them. When the LLM plan was unusable,
    the default plan ignored the recalled companies and the run researched nothing."""
    from memory.episodic import EpisodicMemory
    from memory.facts import FactsStore
    from memory.manager import MemoryManager
    from memory.vector_store import DEFAULT_STORE
    from tests.test_core import C7_QUERY, QUERY, ScriptedLLM, make_agent

    DEFAULT_STORE.clear()
    try:
        memory = MemoryManager(DEFAULT_STORE, FactsStore(None), EpisodicMemory(None))
        make_agent(ScriptedLLM(), memory=memory).run(QUERY)
        script = ScriptedLLM(plan="no plan today", synthesis="=== SECTION: executive_summary ===\nThemes.")
        result = make_agent(script, memory=memory).run(C7_QUERY)
    finally:
        DEFAULT_STORE.clear()
    assert "Default plan" in result.plan.rationale
    assert result.plan.steps and all((s.args or {}).get("ticker", "MSFT") == "MSFT" for s in result.plan.steps)
    assert not any("No research plan could be made" in g for g in result.gaps)
