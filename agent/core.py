"""ARA-1's research loop: a Plan-and-Execute graph with a bounded ReAct executor (architecture spec §5, §7).

    analyze -> recall -> plan -> execute <-> review -> synthesize -> verify -> reflect -> render -> remember
                                  ^            |                                 |
                                  +------------+ (new steps)                     +--> review (gaps found)

* **plan** asks the LLM for a JSON plan of tool calls; :mod:`agent.parser` validates it. If no usable
  plan comes back, a deterministic default plan for the report type is used instead.
* **execute** runs every ready step, independent steps in parallel. A step with complete arguments is
  dispatched directly, with no LLM call. The LLM is consulted only when arguments depend on earlier
  results or a call fails. That is the bounded ReAct loop: at most ``max_calls_per_step`` tool calls.
* **review** lets the LLM add steps once the plan is complete (bounded by ``max_replans``).
* **synthesize** writes all sections in one LLM call; **verify** fact-checks every cited number;
  **reflect** sends missing sections back to review (bounded by ``max_reflections``).
* **render** assembles the report with ``report_generator``; **remember** stores the executive summary
  in long-term memory.

Budgets: at most ``max_tool_calls`` research tool calls (spec A7.1), plus a wall-clock limit. Calls
the system makes itself (memory recall and storage, report rendering) are logged but not budgeted.
Failures never crash the run: they become written data gaps in the report (spec A4.3).
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.llm import LLMClient, LLMError
from agent.parser import (
    SYSTEM_TOOLS,
    ParseError,
    Plan,
    PlanStep,
    parse_plan,
    parse_replan,
    parse_sections,
    step_done_note,
)
from agent.tracing import TraceLogger
from config.settings import AgentSettings, Settings
from tools import ToolError, ToolRegistry, ToolResult, build_default_registry
from tools.fact_checker import extract_quantities, fact_checker
from tools.report_gen import TEMPLATES, cited_ids, section_title

# --------------------------------------------------------------------------------------------------
# Query analysis (a minimal version; the full analyser and disambiguation arrive on Day 10)
# --------------------------------------------------------------------------------------------------

KNOWN_COMPANIES = {
    "microsoft": "MSFT", "apple": "AAPL", "tesla": "TSLA", "nvidia": "NVDA", "amazon": "AMZN",
    "alphabet": "GOOGL", "google": "GOOGL", "palantir": "PLTR", "meta platforms": "META",
    "jpmorgan": "JPM", "goldman sachs": "GS", "morgan stanley": "MS", "oracle": "ORCL",
}

# Checked in order; the first match wins. The default is company_profile.
TEMPLATE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("full_report", ("investment research report", "full research report", "complete investment")),
    ("investigative", ("contradict", "investigate", "apparent contradiction")),
    ("comparative_analysis", ("compare", "comparison", " versus ", " vs ")),
    ("thematic", ("themes", "cross-cutting", "across the")),
    ("risk_assessment", ("risk assessment", "risks")),
    ("earnings_analysis", ("earnings", "quarterly results", "consensus")),
]

_TICKER_HINT = re.compile(r"\((NASDAQ:|NYSE:)?\s*([A-Z]{1,5})\)|\bticker\s+([A-Z]{1,5})\b")


@dataclass
class QueryProfile:
    template: str
    tickers: list[str]
    names: list[str]


def analyze_query(query: str) -> QueryProfile:
    lowered = f" {query.lower()} "
    template = next((t for t, words in TEMPLATE_RULES if any(w in lowered for w in words)), "company_profile")
    tickers: list[str] = []
    names: list[str] = []
    for name, ticker in KNOWN_COMPANIES.items():
        if re.search(rf"\b{re.escape(name)}\b", lowered) and ticker not in tickers:
            tickers.append(ticker)
            names.append(name.title())
    for m in _TICKER_HINT.finditer(query):
        ticker = m.group(2) or m.group(3)
        before = query[: m.start()].lower().rstrip()
        # "Amazon (AWS)": brackets after a company we already know name a product or segment, not a
        # ticker, unless an exchange prefix says otherwise ("(NASDAQ: MSFT)").
        if m.group(2) and not m.group(1) and any(before.endswith(name) for name in KNOWN_COMPANIES):
            continue
        if ticker not in tickers:
            tickers.append(ticker)
    return QueryProfile(template=template, tickers=tickers, names=names)


def default_plan(profile: QueryProfile) -> Plan:
    """A deterministic plan used when the LLM does not return a usable one."""
    steps: list[PlanStep] = []
    for ticker in profile.tickers or []:
        t = ticker.lower()
        steps += [
            PlanStep(f"{t}_profile", f"Company profile of {ticker}", "company_profile", {"ticker": ticker}),
            PlanStep(f"{t}_income", f"Three years of {ticker} income statements", "financial_data_api",
                     {"ticker": ticker, "statement_type": "income_statement", "period": "annual", "years": 3}),
            PlanStep(f"{t}_ratios", f"Key ratios for {ticker}", "financial_data_api",
                     {"ticker": ticker, "statement_type": "key_ratios", "period": "annual", "years": 3}),
            PlanStep(f"{t}_10k", f"Latest {ticker} annual report", "sec_filing_search",
                     {"ticker": ticker, "filing_type": "10-K"}),
            PlanStep(f"{t}_news", f"Recent news about {ticker}", "news_sentiment",
                     {"query": ticker, "num_articles": 10, "lookback_days": 30}),
        ]
    return Plan(steps=steps, rationale="Default plan (the LLM plan was unusable).")


# --------------------------------------------------------------------------------------------------
# Evidence formatting
# --------------------------------------------------------------------------------------------------

_PERCENT_KEYS = re.compile(r"margin|growth|roe|ratio|rate", re.IGNORECASE)


def evidence_digest(result: ToolResult, limit: int = 2500) -> str:
    """A compact, citable text rendering of one tool result for LLM prompts and fact checking."""
    args = ", ".join(f"{k}={v}" for k, v in result.args.items() if k not in ("sources", "sections"))
    head = (f"[{result.evidence_id}] {result.tool}({args[:160]}) | source tier {result.source_tier or '-'}"
            f"{' | MOCK DATA' if result.is_mock else ''}")
    data = result.data
    if result.tool == "financial_data_api":
        unit = data.get("unit")
        lines = [f"{data.get('company_name', data.get('ticker'))} {data.get('statement_type')} "
                 f"({data.get('period')}, USD {unit}; fiscal year ends {data.get('fiscal_year_end')})"]
        for row in data.get("data", []):
            values = "; ".join(f"{k}={_fmt_ratio(v) if unit == 'ratio' else _fmt_amount(v)}"
                               for k, v in row["values"].items())
            lines.append(f"{row['fiscal_period']} {row['fiscal_year']} (period ended {row['period_end']}): {values}")
        body = "\n".join(lines)
    elif result.tool == "calculation_engine":
        r = data.get("result")
        inputs = ", ".join(f"{k}={v}" for k, v in data.get("inputs", {}).items())
        if isinstance(r, (int, float)):
            shown = f"{r:.6g}" + (f" (= {r:.2%})" if data["calculation_type"] in
                                  ("growth_rate", "percent_change", "cagr", "margin") else "")
        else:
            shown = prompts.compact_json(r, 400)
        body = f"{data.get('calculation_type')}({inputs}) = {shown}"
    else:
        body = prompts.compact_json(data, limit)
    text = f"{head}\n{body}"
    return text if len(text) <= limit else text[: limit - 15] + "...[truncated]"


def _fmt_amount(v: Any) -> str:
    return "n/a" if v is None else (f"{v:,.2f}" if isinstance(v, float) and not v.is_integer() else f"{v:,.0f}")


def _fmt_ratio(v: Any) -> str:
    return "n/a" if v is None else f"{v:.2%}"


SOURCE_TYPES = {
    "sec_filing_search": "SEC filing", "financial_data_api": "financial data", "company_profile": "company data",
    "earnings_transcript": "earnings call", "news_sentiment": "news", "web_search": "web",
    "peer_comparison": "peer data", "calculation_engine": "calculation", "vector_db_search": "memory",
}


def source_entry(result: ToolResult) -> dict[str, Any]:
    args = ", ".join(f"{v}" for v in result.args.values() if not isinstance(v, (dict, list)))[:80]
    entry: dict[str, Any] = {
        "id": result.evidence_id,
        "title": f"{result.tool}: {args}" if args else result.tool,
        "source_type": SOURCE_TYPES.get(result.tool, result.tool),
        "date": result.data.get("filing_date") or result.retrieved_at[:10],
        "is_mock": result.is_mock,
    }
    if isinstance(result.data.get("url"), str):
        entry["url"] = result.data["url"]
    return entry


# --------------------------------------------------------------------------------------------------
# State and result
# --------------------------------------------------------------------------------------------------


class ResearchState(TypedDict, total=False):
    query: str
    profile: QueryProfile
    memory_notes: str
    plan: Plan
    replans: int
    reflections: int
    pending_gaps: list[str]
    skip_resynthesis: bool
    sections: dict[str, str]
    verification: dict[str, Any]
    report: dict[str, Any]


@dataclass
class ResearchResult:
    session_id: str
    query: str
    template: str
    report_markdown: str
    sections: dict[str, str]
    plan: Plan
    evidence: dict[str, ToolResult]
    gaps: list[str]
    verification: dict[str, Any]
    tool_calls: int
    llm_usage: dict[str, Any]
    duration_s: float
    stop_reason: str
    trace_path: str | None = None
    report_checks: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------------------------------


class ResearchAgent:
    """One research run per :meth:`run` call. See the module docstring for the flow."""

    def __init__(
        self,
        llm: LLMClient,
        registry_factory: Callable[[], ToolRegistry],
        *,
        agent_settings: AgentSettings | None = None,
        traces_dir: str | None = None,
        max_seconds: float = 300.0,
        max_workers: int = 4,
        clock: Callable[[], float] = time.monotonic,
        today: date | None = None,
    ) -> None:
        self.llm = llm
        self.registry_factory = registry_factory
        self.cfg = agent_settings or AgentSettings()
        self.traces_dir = traces_dir
        self.max_seconds = max_seconds
        self.max_workers = max_workers
        self._clock = clock
        self._today = today
        self._graph = self._build_graph()

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> ResearchAgent:
        options: dict[str, Any] = {"agent_settings": settings.agent, "traces_dir": str(settings.paths.traces_dir)}
        options.update(overrides)
        llm = options.pop("llm", None) or LLMClient.from_settings(settings)
        return cls(llm, lambda: build_default_registry(settings.data_mode), **options)

    # ------------------------------------------------------------------ public API

    def run(self, query: str) -> ResearchResult:
        started = self._clock()
        self._deadline = started + self.max_seconds
        self.registry = self.registry_factory()
        self.tracer = TraceLogger(traces_dir=self.traces_dir)
        self.registry.add_listener(self.tracer.on_tool_call)
        self.llm.tracer = self.tracer
        self.evidence: dict[str, ToolResult] = {}
        self.gaps: list[str] = []
        self.budget_used = 0
        self._lock = threading.Lock()
        self._stop_reason = "completed"
        usage_before = self.llm.usage.totals()

        self.tracer.log("session_start", query=query, max_tool_calls=self.cfg.max_tool_calls,
                        max_seconds=self.max_seconds)
        self._system = prompts.system_prompt(self.registry.describe(exclude=SYSTEM_TOOLS), self.cfg.max_tool_calls,
                                             self._today)
        try:
            state: ResearchState = self._graph.invoke({"query": query, "replans": 0, "reflections": 0},
                                                      config={"recursion_limit": 200})
        finally:
            self.tracer.log("session_end", stop_reason=self._stop_reason, tool_calls=self.budget_used,
                            evidence=len(self.evidence))
            self.tracer.close()

        usage_after = self.llm.usage.totals()
        report = state.get("report", {})
        return ResearchResult(
            session_id=self.tracer.session_id,
            query=query,
            template=state["profile"].template,
            report_markdown=report.get("markdown", ""),
            sections=state.get("sections", {}),
            plan=state["plan"],
            evidence=dict(self.evidence),
            gaps=list(self.gaps),
            verification=state.get("verification", {}),
            tool_calls=self.budget_used,
            llm_usage={
                "calls": usage_after.calls - usage_before.calls,
                "failures": usage_after.failures - usage_before.failures,
                "prompt_tokens": usage_after.prompt_tokens - usage_before.prompt_tokens,
                "completion_tokens": usage_after.completion_tokens - usage_before.completion_tokens,
                "reasoning_tokens": usage_after.reasoning_tokens - usage_before.reasoning_tokens,
            },
            duration_s=round(self._clock() - started, 2),
            stop_reason=self._stop_reason,
            trace_path=str(self.tracer.path) if self.tracer.path else None,
            report_checks={k: report.get(k) for k in ("missing_sections", "unknown_citations", "uncited_sources")},
        )

    # ------------------------------------------------------------------ graph

    def _build_graph(self) -> Any:
        g = StateGraph(ResearchState)
        for name in ("analyze", "recall", "plan", "execute", "review", "synthesize", "verify", "reflect",
                     "render", "remember"):
            g.add_node(name, getattr(self, f"_node_{name}"))
        g.add_edge(START, "analyze")
        g.add_edge("analyze", "recall")
        g.add_edge("recall", "plan")
        g.add_edge("plan", "execute")
        g.add_conditional_edges("execute", self._route_after_execute, {"execute": "execute", "review": "review"})
        g.add_conditional_edges("review", self._route_after_review,
                                {"execute": "execute", "synthesize": "synthesize", "render": "render"})
        g.add_edge("synthesize", "verify")
        g.add_edge("verify", "reflect")
        g.add_conditional_edges("reflect", self._route_after_reflect, {"review": "review", "render": "render"})
        g.add_edge("render", "remember")
        g.add_edge("remember", END)
        return g.compile()

    # ------------------------------------------------------------------ budgets

    def _time_left(self) -> bool:
        if self._clock() < self._deadline:
            return True
        self._stop("time limit reached")
        return False

    def _take_call(self) -> bool:
        with self._lock:
            if self.budget_used >= self.cfg.max_tool_calls:
                self._stop("tool-call budget exhausted")
                return False
            self.budget_used += 1
            return True

    def _budget_left(self) -> bool:
        return self.budget_used < self.cfg.max_tool_calls

    def _stop(self, reason: str) -> None:
        if self._stop_reason == "completed":
            self._stop_reason = reason
            self.tracer.log("note", text=f"Stopping early: {reason}")

    def _llm(self, user: str, *, purpose: str, **kwargs: Any) -> Any:
        return self.llm.complete([{"role": "system", "content": self._system}, {"role": "user", "content": user}],
                                 purpose=purpose, **kwargs)

    # ------------------------------------------------------------------ nodes

    def _node_analyze(self, state: ResearchState) -> ResearchState:
        profile = analyze_query(state["query"])
        self.tracer.thought(f"Report type '{profile.template}'; companies {profile.tickers or 'not identified'}.",
                            node="analyze")
        return {"profile": profile}

    def _node_recall(self, state: ResearchState) -> ResearchState:
        """Memory first (principle P3): look for prior findings before any external call."""
        profile, notes = state["profile"], []
        for ticker in profile.tickers or [None]:
            args: dict[str, Any] = {"query": state["query"][:400], "top_k": 5}
            if ticker:
                args["ticker"] = ticker
            try:
                result = self.registry.dispatch("vector_db_search", args)
            except ToolError as exc:
                self.tracer.log("note", text=f"Memory recall failed: {exc}")
                continue
            hits = result.data.get("results", [])
            if hits:
                self.evidence[result.evidence_id] = result
                notes += [f"[{result.evidence_id}] {h['metadata'].get('date')}: {h['content'][:200]}" for h in hits]
        self.tracer.thought(f"Memory recall found {len(notes)} prior finding(s).", node="recall")
        return {"memory_notes": "\n".join(notes)}

    def _node_plan(self, state: ResearchState) -> ResearchState:
        profile = state["profile"]
        sections = TEMPLATES[profile.template]
        tool_names = [s.name for s in self.registry.available() if s.name not in SYSTEM_TOOLS]
        # Leave about a fifth of the budget for repairs and follow-up steps.
        max_steps = max(1, self.cfg.max_tool_calls - self.cfg.max_tool_calls // 5)
        prompt = prompts.plan_prompt(query=state["query"], template=profile.template, sections=sections,
                                     entities=profile.tickers, memory=state.get("memory_notes", ""),
                                     tool_names=tool_names, max_steps=max_steps)
        plan: Plan | None = None
        problem = ""
        for attempt in range(2):
            try:
                reply = self._llm(prompt + problem, purpose="plan", json_mode=True, reasoning_effort="low",
                                  max_tokens=2048)
                self.tracer.log("note", text="raw plan reply", raw=(reply.content or "")[:6000])
                plan = parse_plan(reply.content, self.registry, max_steps=max_steps)
                break
            except ParseError as exc:
                problem = f"\n\nYour previous reply could not be used ({exc}). Return valid JSON only."
                self.tracer.log("error", source="parser", error_kind="plan_parse", message=str(exc), attempt=attempt + 1)
            except LLMError as exc:
                self.tracer.log("error", source="llm", error_kind="plan_failed", message=str(exc)[:300])
                break
        if plan is None:
            plan = default_plan(profile)
            if not plan.steps:
                self.gaps.append("No research plan could be made: the LLM failed and no company was identified.")
        self.tracer.log("decision", chosen="plan", rationale=plan.rationale, steps=len(plan.steps))
        for step in plan.steps:
            self.tracer.decision(step.tool, step.reason or step.goal, step_id=step.id, args=step.args,
                                 depends_on=step.depends_on)
        for note in plan.rejected:
            self.tracer.log("note", text=f"Plan validation: {note}")
        return {"plan": plan}

    def _node_execute(self, state: ResearchState) -> ResearchState:
        plan = state["plan"]
        ready = plan.ready()
        for step in ready:
            step.status = "running"
        if len(ready) == 1 or self.max_workers <= 1:
            for step in ready:
                self._run_step(step, plan)
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(ready))) as pool:
                list(pool.map(lambda s: self._run_step(s, plan), ready))
        return {"plan": plan}

    def _run_step(self, step: PlanStep, plan: Plan) -> None:
        """Run one plan step: a direct call if the arguments are known, else a bounded ReAct loop."""
        problem = ""
        if step.args is not None:
            if not (self._time_left() and self._take_call()):
                return self._finish(step, "skipped", "not run: the time or tool budget ran out")
            step.attempts += 1
            try:
                result = self.registry.dispatch(step.tool, step.args)
                return self._finish(step, "done", f"{step.tool} returned {result.evidence_id}", result)
            except ToolError as exc:
                problem = str(exc)

        dependencies = "\n\n".join(
            evidence_digest(self.evidence[eid], 1800)
            for dep in step.depends_on for eid in plan.get(dep).evidence_ids if eid in self.evidence
        )
        tools = [t for t in self.registry.openai_tools() if t["function"]["name"] not in SYSTEM_TOOLS]
        while step.attempts < self.cfg.max_calls_per_step and self._time_left() and self._budget_left():
            prompt = prompts.execute_step_prompt(goal=step.goal, tool=step.tool, evidence=dependencies, problem=problem)
            try:
                reply = self._llm(prompt, purpose=f"execute:{step.id}", tools=tools, tool_choice="auto", max_tokens=1024)
            except LLMError as exc:
                return self._finish(step, "failed", f"the LLM could not be reached: {str(exc)[:200]}")
            if not reply.tool_calls:
                note = step_done_note(reply.content) or (reply.content or "no tool call made").strip()[:300]
                return self._finish(step, "skipped", f"not needed: {note}")
            self.tracer.thought(reply.content or f"Calling {[c.name for c in reply.tool_calls]}", node="execute",
                                step_id=step.id)
            for call in reply.tool_calls:
                if step.attempts >= self.cfg.max_calls_per_step or not self._take_call():
                    break
                step.attempts += 1
                try:
                    result = self.registry.dispatch(call.name, call.arguments)
                    step.args = result.args
                    return self._finish(step, "done", f"{call.name} returned {result.evidence_id}", result)
                except ToolError as exc:
                    problem = str(exc)
        if not self._budget_left():
            self._stop("tool-call budget exhausted")
        reason = problem or "the call and step budgets ran out"
        return self._finish(step, "failed", f"failed after {step.attempts} attempt(s): {reason[:300]}")

    def _finish(self, step: PlanStep, status: str, summary: str, result: ToolResult | None = None) -> None:
        step.status, step.summary = status, summary
        if result is not None:
            with self._lock:
                self.evidence[result.evidence_id] = result
            step.evidence_ids.append(result.evidence_id)
        if status in ("failed", "skipped") and not summary.startswith("not needed"):
            with self._lock:
                self.gaps.append(f"Step '{step.goal}' ({step.tool}) {summary}.")
        self.tracer.log("note", text=f"step {step.id} {status}: {summary}", step_id=step.id)

    def _node_review(self, state: ResearchState) -> ResearchState:
        """Once the plan is complete, let the LLM add steps for real gaps (bounded by max_replans)."""
        plan, replans = state["plan"], state.get("replans", 0)
        gaps = state.get("pending_gaps", [])
        if replans >= self.cfg.max_replans or not self._budget_left() or not self._time_left():
            return {"pending_gaps": []}
        profile = state["profile"]
        completed = "\n".join(f"- {s.id} {s.goal}: {s.summary}" for s in plan.by_status("done"))
        failed = "\n".join(f"- {s.id} {s.goal}: {s.summary}" for s in plan.by_status("failed", "skipped"))
        prompt = prompts.replan_prompt(query=state["query"], template=profile.template,
                                       sections=TEMPLATES[profile.template], completed=completed, failed=failed,
                                       gaps=gaps, used=self.budget_used, budget=self.cfg.max_tool_calls)
        try:
            reply = self._llm(prompt, purpose="review", json_mode=True, max_tokens=1536)
            decision = parse_replan(reply.content, self.registry, plan,
                                    max_new_steps=self.cfg.max_tool_calls - self.budget_used)
        except (ParseError, LLMError) as exc:
            self.tracer.log("error", source="review", error_kind=type(exc).__name__, message=str(exc)[:300])
            return {"replans": replans + 1, "pending_gaps": []}
        for note in decision.rejected:
            self.tracer.log("note", text=f"Review validation: {note}")
        if decision.new_steps:
            plan.steps.extend(decision.new_steps)
            self.tracer.decision("add_steps", decision.rationale, steps=[s.id for s in decision.new_steps])
        else:
            self.tracer.decision("research_complete", decision.rationale or "no further steps needed")
        # A gap-driven review that finds nothing new to research would only re-write the same draft.
        return {"plan": plan, "replans": replans + 1, "pending_gaps": [],
                "skip_resynthesis": bool(gaps) and not decision.new_steps}

    def _node_synthesize(self, state: ResearchState) -> ResearchState:
        sections_needed = TEMPLATES[state["profile"].template]
        if not self.evidence:
            self.gaps.append("No evidence was gathered, so no section could be written.")
            return {"sections": {}}
        digest = "\n\n".join(evidence_digest(r) for r in self.evidence.values())
        prompt = prompts.synthesize_prompt(query=state["query"], evidence=digest, sections=sections_needed)
        try:
            reply = self._llm(prompt, purpose="synthesize", reasoning_effort="low", max_tokens=6000)
        except LLMError as exc:
            self.gaps.append(f"The report text could not be written: {str(exc)[:200]}")
            return {"sections": {}}
        return {"sections": parse_sections(reply.content, sections_needed)}

    def _node_verify(self, state: ResearchState) -> ResearchState:
        """Fact-check every sentence that contains a number (spec A4.3 verification pipeline)."""
        sections = dict(state.get("sections", {}))
        tally = {"checked": 0, "supported": 0, "partially_supported": 0, "unsupported": 0, "unverifiable": 0,
                 "uncited": 0, "auto_cited": 0}
        flagged: list[str] = []
        digests = {eid: evidence_digest(r, 6000) for eid, r in self.evidence.items()}
        for key, body in sections.items():
            for claim, inherited in _claims(body):
                quantities = extract_quantities(claim)
                if not quantities:
                    continue
                ids = [i for i in (cited_ids(claim) or inherited) if i in self.evidence]
                source = None if ids else _find_single_source(claim, quantities, digests)
                if source:
                    # Citation repair: every figure in the claim matches this one source exactly, so
                    # citing it is accurate. Counted separately so the report can disclose it.
                    body = body.replace(claim, _cite(claim, source), 1)
                    tally["auto_cited"] += 1
                    tally["checked"] += 1
                    tally["supported"] += 1
                    continue
                if not ids:
                    tally["uncited"] += 1
                    flagged.append(f"{section_title(key)}: uncited figure: {claim[:140]}")
                    body = body.replace(claim, _mark(claim, "unverified: no source cited"), 1)
                    continue
                verdict = fact_checker(claim, [evidence_digest(self.evidence[i], 6000) for i in ids])
                tally["checked"] += 1
                tally[verdict["status"]] = tally.get(verdict["status"], 0) + 1
                if verdict["status"] in ("unsupported", "partially_supported"):
                    flagged.append(f"{section_title(key)}: {verdict['status'].replace('_', ' ')}: {claim[:140]}")
                    body = body.replace(claim, _mark(claim, "unverified"), 1)
            sections[key] = body
        self.tracer.metric("verification", tally)
        return {"sections": sections, "verification": {**tally, "flagged": flagged}}

    def _node_reflect(self, state: ResearchState) -> ResearchState:
        needed = TEMPLATES[state["profile"].template]
        missing = [s for s in needed if s not in state.get("sections", {})]
        reflections = state.get("reflections", 0)
        self.tracer.thought(f"Reflection: {len(missing)} section(s) missing {missing or ''}".strip(), node="reflect")
        can_research = state.get("replans", 0) < self.cfg.max_replans and self._budget_left() and self._time_left()
        if missing and reflections < self.cfg.max_reflections and can_research:
            return {"pending_gaps": [f"Section '{section_title(s)}' has no supporting evidence." for s in missing],
                    "reflections": reflections + 1}
        return {"pending_gaps": []}

    def _node_render(self, state: ResearchState) -> ResearchState:
        profile, plan = state["profile"], state["plan"]
        verification = state.get("verification", {})
        gaps = list(dict.fromkeys(self.gaps))  # de-duplicate, keep order
        gaps += [f"Unverified claim in {f}" for f in verification.get("flagged", [])]
        sections = dict(state.get("sections", {}))
        sections["data_gaps_and_limitations"] = "\n".join(f"- {g}" for g in gaps) or ""
        sections["research_methodology"] = self._methodology(plan, verification)
        company = profile.names[0] if len(profile.names) == 1 else ", ".join(profile.tickers) or "Research"
        result = self.registry.dispatch("report_generator", {
            "template": profile.template,
            "title": f"{company}: {section_title(profile.template)}",
            "sections": sections,
            "sources": [source_entry(r) for r in self.evidence.values()],
        })
        return {"report": result.data}

    def _methodology(self, plan: Plan, verification: dict[str, Any]) -> str:
        done = plan.by_status("done")
        tools_used = sorted({s.tool for s in done})
        providers = sorted({k.split("/")[0] for k, v in self.llm.usage.by_model.items() if v.calls})
        lines = [
            f"- Plan: {len(plan.steps)} step(s), {len(done)} completed. {plan.rationale}".rstrip(),
            f"- Tools used: {', '.join(tools_used) or 'none'}; {self.budget_used} of {self.cfg.max_tool_calls} "
            "research tool calls.",
            f"- Language models: {', '.join(providers) or 'none'}.",
            f"- Verification: {verification.get('checked', 0)} cited numeric claim(s) checked against their sources; "
            f"{verification.get('supported', 0)} supported, {verification.get('unsupported', 0)} unsupported, "
            f"{verification.get('uncited', 0)} without a citation. "
            f"{verification.get('auto_cited', 0)} citation(s) were added by the verifier where every figure in a "
            "claim matched exactly one retrieved source.",
            f"- Stop reason: {self._stop_reason}.",
        ]
        if any(r.is_mock for r in self.evidence.values()):
            lines.append("- **Mock data was used; figures are approximations for development only.**")
        return "\n".join(lines)

    def _node_remember(self, state: ResearchState) -> ResearchState:
        summary = state.get("sections", {}).get("executive_summary")
        tickers = state["profile"].tickers
        if summary and len(tickers) == 1:
            try:
                self.registry.dispatch("vector_db_store", {"content": summary[:4000], "metadata": {
                    "ticker": tickers[0], "date": date.today().isoformat(), "source_type": "analysis",
                    "confidence": 0.5, "verified": False}})
            except ToolError as exc:
                self.tracer.log("note", text=f"Could not store the summary in memory: {exc}")
        return {}

    # ------------------------------------------------------------------ routing

    def _route_after_execute(self, state: ResearchState) -> str:
        plan = state["plan"]
        if plan.ready() and self._budget_left() and self._time_left():
            return "execute"
        for step in plan.by_status("pending"):  # unreachable, or the budget ran out
            self._finish(step, "skipped", "not run: the time or tool budget ran out")
        return "review"

    def _route_after_review(self, state: ResearchState) -> str:
        if state["plan"].ready():
            return "execute"
        return "render" if state.get("skip_resynthesis") else "synthesize"

    def _route_after_reflect(self, state: ResearchState) -> str:
        return "review" if state.get("pending_gaps") else "render"


def _find_single_source(claim: str, quantities: list[Any], digests: dict[str, str]) -> str | None:
    """The one evidence item that supports every figure in an uncited claim, if exactly one does.

    Only precise claims qualify: two or more figures, or one figure that is large or has decimals.
    A lone round number such as "5%" could match an unrelated source by coincidence.
    """
    precise = len(quantities) >= 2 or any(abs(q.value) >= 1000 or not float(q.value).is_integer()
                                          for q in quantities if q.kind == "amount")
    if not precise:
        return None
    matches = [eid for eid, text in digests.items()
               if fact_checker(claim, [text])["status"] == "supported"]
    return matches[0] if len(matches) == 1 else None


def _cite(claim: str, evidence_id: str) -> str:
    """Add a citation: inside the last cell of a table row, or before a sentence's final full stop."""
    if claim.startswith("|") and claim.endswith("|"):
        return f"{claim[:-1].rstrip()} [{evidence_id}] |"
    if claim.endswith("."):
        return f"{claim[:-1]} [{evidence_id}]."
    return f"{claim} [{evidence_id}]"


def _mark(claim: str, note: str) -> str:
    """Append a visible verification note; inside the last cell for table rows, so the table survives."""
    if claim.startswith("|") and claim.endswith("|"):
        return f"{claim[:-1].rstrip()} *({note})* |"
    return f"{claim} *({note})*"


_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z*(\[])")


def _claims(body: str) -> list[tuple[str, set[str]]]:
    """The units that get fact-checked: sentences and table rows, each with inherited citations.

    Analysts (and models) usually cite a table once, in its caption or header, not in every row. So
    a table row without its own citation inherits those of the table's header and of the line just
    before the table. Header and separator rows are not claims.
    """
    claims: list[tuple[str, set[str]]] = []
    lines = [line.strip() for line in body.splitlines()]
    previous_text = ""
    in_table, table_ids = False, set()
    for i, line in enumerate(lines):
        if not line:
            in_table = False
            continue
        if line.startswith("|"):
            is_separator = set(line) <= set("|-: ")
            next_is_separator = i + 1 < len(lines) and lines[i + 1].startswith("|") and set(lines[i + 1]) <= set("|-: ")
            if not in_table:
                in_table, table_ids = True, cited_ids(previous_text)
            if next_is_separator:  # header row
                table_ids |= cited_ids(line)
            elif not is_separator:
                claims.append((line, table_ids))
            continue
        in_table = False
        previous_text = line
        claims += [(s.strip(), set()) for s in _SENTENCE.split(line) if s.strip()]
    return claims
