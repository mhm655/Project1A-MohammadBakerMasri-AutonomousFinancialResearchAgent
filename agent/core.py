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

import dataclasses
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
from agent.circuit_breaker import BreakerBoard
from agent.disambiguation import (
    Disambiguation,
    coverage_flags,
    disambiguate,
    rate_of_change_flags,
    render_assumptions,
    render_temporal_flags,
)
from agent.error_handler import AllFallbacksFailed, FailureInjector, ResilientDispatcher, RetryPolicy
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
from agent.query_analyzer import KNOWN_COMPANIES, TEMPLATE_RULES, QueryProfile, analyze_query  # noqa: F401
from agent.references import resolve_references
from agent.tracing import TraceLogger
from config.settings import AgentSettings, ResilienceSettings, RetrySettings, Settings, SynthesisSettings
from memory.context_manager import count_tokens, fit_evidence
from synthesis.engine import Ledger, build_ledger
from synthesis.narrative import build_threads, conflicts_markdown, render_brief
from tools import ToolError, ToolExecutionError, ToolRegistry, ToolResult, ToolValidationError, build_default_registry
from tools.fact_checker import extract_quantities, fact_checker, unsupported_names
from tools.report_gen import TEMPLATES, cited_ids, section_title


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
    elif result.tool == "sec_filing_search":
        # A 10-K runs to hundreds of thousands of characters: show what the report needs most, namely
        # the risk-factor headings and the opening of each section, within the digest's budget.
        lines = [f"{data.get('company_name')} Form {data.get('filing_type')} for the period ended "
                 f"{data.get('period_of_report')}, filed {data.get('filing_date')} (accession {data.get('accession_number')})"]
        risks = data.get("risk_factors") or []
        if risks:
            lines.append(f"Risk factors ({len(risks)}): " + " | ".join(r[:140] for r in risks[:40]))
        sections = data.get("sections", [])
        share = max(300, (limit - sum(len(x) for x in lines) - 200) // max(1, len(sections)))
        for s in sections:
            text = " ".join(s["text"].split())
            lines.append(f"Item {s['item']} {s['title']}: {text[:share]}")
        body = "\n".join(lines)
    elif result.tool == "peer_comparison":
        # Every value carries its unit: a bare "367.30" under a "ratios as percentages" header was read
        # as a P/E of 36,730 %.
        lines = [f"Peer comparison for {data.get('ticker')} (growth, margins and ROE in percent; revenue in "
                 "USD millions; P/E in times earnings; market cap in USD billions)"]
        for row in data.get("matrix", []):
            values = "; ".join(f"{k}={_fmt_peer_value(k, v)}" for k, v in row["values"].items())
            period = f", FY{row['fiscal_year']} ended {row['period_end']}" if row.get("fiscal_year") else ""
            lines.append(f"{row['ticker']} ({row.get('name', '')}{period}): {values}")
        lines += [n for n in data.get("notes", []) if n.startswith(("Peers:", "P/E")) or " left out: " in n]
        body = "\n".join(lines)
    elif result.tool == "earnings_transcript":
        kind = "earnings call transcript" if data.get("source_kind") == "transcript" else "earnings press release"
        lines = [f"{data.get('ticker')} {data.get('quarter')} FY{data.get('fiscal_year')} {kind}, "
                 f"{data.get('call_date') or 'date unknown'} ({data.get('source_url', '')})"]
        lines += [n for n in data.get("notes", []) if n.startswith(("No quarter", "Call transcript unavailable"))]
        budget = limit - len(head) - sum(len(x) + 1 for x in lines) - 20
        body = "\n".join(lines + salient_statements(data.get("segments", []), budget))
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


_FIGURE = re.compile(r"\d[\d,.]*\s*(?:%|percent|basis points|billion|million)|\$\s?\d", re.IGNORECASE)
_GUIDANCE = re.compile(r"\b(?:expect|guidance|outlook|anticipate|forecast|next quarter|full[- ]year|going forward)",
                       re.IGNORECASE)
_RISK = re.compile(r"\b(?:headwind|pressure|tariff|competition|competitive|constrain|shortage|decline|slow|"
                   r"uncertain|regulat|recall|risk)", re.IGNORECASE)
MAX_STATEMENT_CHARS = 320
#: Follow-up calls allowed when the report text stops at the model's output limit.
MAX_CONTINUATIONS = 2


def salient_statements(segments: list[dict[str, Any]], budget: int) -> list[str]:
    """The most informative sentences of an earnings call, in call order, within ``budget`` characters.

    A call runs to 50,000+ characters and a prompt digest gets about 2,500. Sentences are scored by
    what an analysis needs: figures, guidance, risks, and analysts' questions (they show what the
    market is worried about); management's sentences rank above the IR host's. When everything
    fits (fact checking reads whole transcripts), every sentence is kept.
    """
    candidates = []  # (score, position, line)
    for seg in segments:
        role = seg.get("role") or ""
        tag = f"[{'Q&A' if seg.get('section') == 'qa' else 'prepared'}] {seg.get('speaker', '')}" + \
              (f" ({role})" if role else "")
        for sentence in re.split(r"(?<=[.!?])\s+", " ".join(seg.get("text", "").split())):
            if len(sentence) < 25:
                continue
            score = 2 * bool(_FIGURE.search(sentence)) + 2 * bool(_GUIDANCE.search(sentence)) + \
                bool(_RISK.search(sentence))
            if role.lower() == "analyst":
                score += 1.5 if sentence.endswith("?") else -1
            elif "chief" in role.lower() or role in ("CEO", "CFO", "Company"):
                score += 0.5
            candidates.append((score, len(candidates), f"{tag}: {sentence[:MAX_STATEMENT_CHARS]}"))
    if sum(len(c[2]) + 1 for c in candidates) > budget:
        chosen, used = [], 0
        for score, position, line in sorted(candidates, key=lambda c: (-c[0], c[1])):
            if score <= 0:
                break
            if used + len(line) + 1 <= budget:
                chosen.append((score, position, line))
                used += len(line) + 1
        candidates = chosen
    return [line for _, _, line in sorted(candidates, key=lambda c: c[1])]


#: Fact checking reads long documents whole: a claim may come from any part of a 10-K or a call
#: transcript, not only from the opening that fits in a prompt digest.
VERIFY_LIMITS = {"sec_filing_search": 400_000, "earnings_transcript": 250_000}


def verification_text(result: ToolResult) -> str:
    return evidence_digest(result, VERIFY_LIMITS.get(result.tool, 6000))


def _args_text(args: Any) -> str:
    return ", ".join(str(v) for v in args.values() if not isinstance(v, (dict, list)))[:80] \
        if isinstance(args, dict) else ""


#: Tools that take a company ticker. A step for one of them with no ticker cannot run (Day 11, C7).
TICKER_TOOLS = frozenset({"company_profile", "financial_data_api", "sec_filing_search", "earnings_transcript",
                          "peer_comparison"})


#: Section-targeted recall keeps only hits at or above this similarity (1 - cosine distance); see
#: results/runs_day13/memory_probe.md for how it was chosen.
RECALL_MIN_SCORE = 0.30


def recall_topics(template: str, n: int = 2) -> list[str]:
    """Titles of the template's first ``n`` content sections, used as extra memory queries."""
    skip = {"executive_summary", "methodology", "conclusion"}
    return [section_title(k) for k in TEMPLATES.get(template, []) if k not in skip][:n]


def enforce_tickers(steps: list[PlanStep], tickers: list[str]) -> tuple[list[PlanStep], list[str]]:
    """Make every ticker-requiring step name its company, or drop it.

    Day 11, C7 ("the companies you've already researched"): the planner and the review added
    ``earnings_transcript`` steps with no ticker; the executor model could only answer "no ticker was
    specified", so 12 of 21 steps were skipped. A step with open arguments (``args`` None) must name one
    of ``tickers`` in its goal. One candidate ticker is added to the goal; with several candidates and no
    choice made, or none at all, the step is dropped. Returns ``(kept steps, notes for the trace)``.
    """
    kept: list[PlanStep] = []
    notes: list[str] = []
    for step in steps:
        if step.tool not in TICKER_TOOLS or step.args is not None or step.depends_on:
            kept.append(step)
            continue
        named = [t for t in tickers if re.search(rf"(?<![A-Za-z]){re.escape(t)}(?![A-Za-z])", step.goal)]
        if named:
            kept.append(step)
        elif len(tickers) == 1:
            step.goal = f"{step.goal} for {tickers[0]}"
            notes.append(f"{step.id} ({step.tool}): no ticker given; set to {tickers[0]}")
            kept.append(step)
        else:
            why = "no company is known" if not tickers else f"it does not say which of {', '.join(tickers)}"
            notes.append(f"{step.id} ({step.tool}): dropped, it needs a ticker and {why}")
    dropped = {s.id for s in steps} - {s.id for s in kept}
    for step in kept:
        step.depends_on = [d for d in step.depends_on if d not in dropped]
    return kept, notes


def _failure_text(exc: ToolError) -> str:
    """A failed step's reason for the report: the Gap sentence when every fallback failed."""
    return exc.gap.to_text() if isinstance(exc, AllFallbacksFailed) else str(exc)


def _fmt_amount(v: Any) -> str:
    return "n/a" if v is None else (f"{v:,.2f}" if isinstance(v, float) and not v.is_integer() else f"{v:,.0f}")


def _fmt_ratio(v: Any) -> str:
    return "n/a" if v is None else f"{v:.2%}"


def _fmt_peer_value(metric: str, v: Any) -> str:
    if v is None:
        return "n/a"
    if metric == "pe_ratio":
        return f"{v:,.1f}x"
    if metric == "market_cap":
        return f"${v / 1e9:,.1f} bn"
    return _fmt_ratio(v) if _PERCENT_KEYS.search(metric) else _fmt_amount(v)


SOURCE_TYPES = {
    "sec_filing_search": "SEC filing", "financial_data_api": "financial data", "company_profile": "company data",
    "earnings_transcript": "earnings call", "news_sentiment": "news", "web_search": "web",
    "peer_comparison": "peer data", "calculation_engine": "calculation", "vector_db_search": "memory",
}


def source_entry(result: ToolResult) -> dict[str, Any]:
    args = ", ".join(f"{v}" for v in result.args.values() if not isinstance(v, (dict, list)))[:80]
    data = result.data
    if result.tool == "earnings_transcript" and data.get("quarter"):
        # The quarter actually retrieved (it is chosen by the tool when the call omitted it).
        kind = "call transcript" if data.get("source_kind") == "transcript" else "press release, no call transcript"
        args = f"{data.get('ticker')} {data['quarter']} FY{data.get('fiscal_year')} ({kind})"
    entry: dict[str, Any] = {
        "id": result.evidence_id,
        "title": f"{result.tool}: {args}" if args else result.tool,
        "source_type": SOURCE_TYPES.get(result.tool, result.tool),
        "date": data.get("filing_date") or data.get("call_date") or result.retrieved_at[:10],
        "is_mock": result.is_mock,
    }
    url = data.get("url") or data.get("source_url")
    if isinstance(url, str) and url:
        entry["url"] = url
    return entry


# --------------------------------------------------------------------------------------------------
# State and result
# --------------------------------------------------------------------------------------------------


class ResearchState(TypedDict, total=False):
    query: str
    profile: QueryProfile
    memory_notes: str
    memory_tickers: list[str]
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
        memory: Any = None,
        synthesis_settings: SynthesisSettings | None = None,
        resilience_settings: ResilienceSettings | None = None,
        retry_settings: RetrySettings | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.llm = llm
        self.registry_factory = registry_factory
        self.cfg = agent_settings or AgentSettings()
        self.synthesis_cfg = synthesis_settings or SynthesisSettings()
        self.resilience_cfg = resilience_settings or ResilienceSettings()
        self.retry_cfg = retry_settings or RetrySettings()
        self._sleep = sleep
        self.traces_dir = traces_dir
        self.max_seconds = max_seconds
        self.max_workers = max_workers
        self._clock = clock
        self._today = today
        #: :class:`memory.manager.MemoryManager`; None disables memory writes (recall still uses the tools).
        self.memory = memory
        self._graph = self._build_graph()
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ara-tool")  # bounded tool calls

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> ResearchAgent:
        options: dict[str, Any] = {"agent_settings": settings.agent, "traces_dir": str(settings.paths.traces_dir),
                                   "synthesis_settings": settings.synthesis, "resilience_settings": settings.resilience,
                                   "retry_settings": settings.retry}
        options.update(overrides)
        llm = options.pop("llm", None) or LLMClient.from_settings(settings)
        if "memory" not in options:
            from memory.manager import MemoryManager

            options["memory"] = MemoryManager.for_settings(settings)
        return cls(llm, lambda: build_default_registry(settings.data_mode), **options)

    # ------------------------------------------------------------------ public API

    def run(self, query: str) -> ResearchResult:
        started = self._clock()
        self._deadline = started + self.max_seconds
        # Keep time to write the report with whatever was gathered: a quarter of the budget, at
        # least a minute (but never more than half).
        self._research_deadline = self._deadline - min(self.max_seconds / 2, max(60.0, self.max_seconds / 4))
        self.registry = self.registry_factory()
        if self.resilience_cfg.fail_rate > 0:
            rc = self.resilience_cfg
            per_tool = {t: rc.fail_rate for t in rc.fail_tools}
            self.registry.failure_injector = FailureInjector(rate=0.0 if per_tool else rc.fail_rate,
                                                             per_tool=per_tool, seed=rc.fail_seed)
        self.breakers = BreakerBoard(self.resilience_cfg.breaker_config(), clock=self._clock)
        self.resilience_events: list[tuple[str, dict[str, Any]]] = []
        self.dispatcher = ResilientDispatcher(
            self.registry.dispatch, breakers=self.breakers,
            policy=RetryPolicy(base_delay_s=self.retry_cfg.base_delay_s, max_retries=self.retry_cfg.max_retries,
                               max_jitter_s=self.retry_cfg.max_jitter_s),
            validate=self.registry.validate_args, sleep=self._sleep, clock=self._clock,
            deadline=lambda: self._clock() >= self._research_deadline, on_event=self._on_resilience_event)
        self.tracer = TraceLogger(traces_dir=self.traces_dir)
        self.registry.add_listener(self.tracer.on_tool_call)
        self.llm.tracer = self.tracer
        self.evidence: dict[str, ToolResult] = {}
        self.gaps: list[str] = []
        self.ledger = Ledger(claims=[], conflicts=[])
        self.failures: list[tuple[str, dict[str, Any], str]] = []  # (tool, args, message), for coverage flags
        self.disambiguation: Disambiguation | None = None
        self.budget_used = 0
        self._lock = threading.Lock()
        self._stop_reason = "completed"
        usage_before = self.llm.usage.totals()

        self.tracer.log("session_start", query=query, max_tool_calls=self.cfg.max_tool_calls,
                        max_seconds=self.max_seconds)
        self._writing_system = prompts.system_prompt("", self.cfg.max_tool_calls, self._today)
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
        result = ResearchResult(
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
        if self.memory is not None:
            try:
                self.memory.record_episode(result, state["profile"].tickers)
            except Exception as exc:  # memory must never break a finished run
                self.tracer.log("note", text=f"Could not record the episode: {exc}")
        return result

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
        """True while there is time for more research; the rest is kept for writing the report."""
        if self._clock() < self._research_deadline:
            return True
        self._stop("time limit reached")
        return False

    def _dispatch(self, tool: str, args: Any) -> ToolResult:
        """A research tool call bounded by the research deadline.

        On a slow link one download (a 10-K can take minutes) could otherwise consume the whole run.
        If the call overruns, the step fails and research moves on. The call keeps running in the
        background, and whatever it downloads is cached for the next run.
        """
        from concurrent.futures import TimeoutError as FutureTimeout

        timeout = max(1.0, self._research_deadline - self._clock())
        # Retries, circuit breakers and fallback chains (spec A4.3, architecture section 14).
        future = self._pool.submit(self.dispatcher.call, tool, args)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            raise ToolExecutionError(f"{tool} did not finish within the {timeout:.0f} s left for research; the "
                                     "download continues in the background and will be cached for the next run",
                                     tool=tool) from None

    def _on_resilience_event(self, kind: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self.resilience_events.append((kind, payload))
        details = ", ".join(f"{k}={str(v)[:160]}" for k, v in payload.items())
        self.tracer.log("note", text=f"resilience {kind}: {details}")

    def _take_call(self) -> bool:
        if self._providers_down():
            return False
        with self._lock:
            if self.budget_used >= self.cfg.max_tool_calls:
                self._stop("tool-call budget exhausted")
                return False
            self.budget_used += 1
            return True

    def _budget_left(self) -> bool:
        return self.budget_used < self.cfg.max_tool_calls and not self._providers_down()

    def _providers_down(self) -> bool:
        """Several providers down at once: more calls would only fail (cascading failure), so stop
        researching and write the report with what was gathered."""
        if not self.breakers.system_degraded():
            return False
        self._stop(f"several data providers are down ({', '.join(self.breakers.open_providers())})")
        return True

    def _stop(self, reason: str) -> None:
        if self._stop_reason == "completed":
            self._stop_reason = reason
            self.tracer.log("note", text=f"Stopping early: {reason}")

    def _llm(self, user: str, *, purpose: str, writing: bool = False, **kwargs: Any) -> Any:
        """An LLM call bounded by the research deadline, or by the final deadline when writing."""
        deadline = self._deadline if writing else self._research_deadline
        system = self._writing_system if writing else self._system
        return self.llm.complete([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                 purpose=purpose, deadline=deadline, **kwargs)

    # ------------------------------------------------------------------ nodes

    def _node_analyze(self, state: ResearchState) -> ResearchState:
        profile = analyze_query(state["query"])
        self.tracer.thought(f"Report type '{profile.template}' ({profile.query_type}, {profile.complexity}); "
                            f"companies {profile.tickers or 'not identified'}; ambiguity {profile.ambiguity:.2f} "
                            f"({profile.ambiguity_level}: {'; '.join(profile.signals) or 'clear'}).", node="analyze")
        # Spec A7.3: an autonomous agent does not stop to ask. It picks the most likely reading,
        # documents it, and lists the questions it would have asked.
        self.disambiguation = disambiguate(profile, today=self._today or date.today())
        d = self.disambiguation
        # A query about earlier research gets its companies from memory (recall), not from sector defaults.
        if not profile.tickers and d.tickers and profile.complexity != "cross_memory":
            profile.tickers = list(d.tickers)
        if d.assumptions:
            self.tracer.decision("disambiguate", d.proceed_reading, assumptions=d.assumptions,
                                 questions=d.clarifying_questions)
        return {"profile": profile}

    def _node_recall(self, state: ResearchState) -> ResearchState:
        """Memory first (principle P3): look for prior findings before any external call."""
        profile, notes = state["profile"], []
        targets: list[str | None] = list(profile.tickers)
        if not targets and self.memory is not None:
            # "Based on the companies you've already researched..." (Challenge 7): recall each known company.
            targets = self.memory.known_companies()
            if targets:
                self.tracer.thought(f"No company named; memory holds research on {', '.join(targets)}.", node="recall")
        seen: set[str] = set()  # record ids already shown to the planner, across all searches
        topics = recall_topics(profile.template)
        for ticker in targets or [None]:
            top_k = 5 if len(targets) <= 2 else 3
            # The query-wide search first, then section-targeted ones (local searches: no API cost). The
            # targeted ones keep only hits above a relevance floor so they add findings, not noise.
            # No floor on the query-wide search (cosine scores can be negative): it keeps its pre-Day-13 behaviour.
            queries = [(state["query"][:400], float("-inf"), top_k)] + [
                (f"{ticker} {t}" if ticker else t, RECALL_MIN_SCORE, min(top_k, 3)) for t in topics]
            for n, (query, floor, k) in enumerate(queries):
                args: dict[str, Any] = {"query": query, "top_k": k}
                if ticker:
                    args["ticker"] = ticker
                try:
                    result = self.registry.dispatch("vector_db_search", args)
                except ToolError as exc:
                    self.tracer.log("note", text=f"Memory recall failed: {exc}")
                    continue
                found = result.data.get("results", [])
                if not found and n == 0:
                    break  # memory holds nothing for this company: the targeted searches would find nothing
                hits = [h for h in found if h["id"] not in seen and h.get("score", 1.0) >= floor]
                seen.update(h["id"] for h in hits)
                if hits:
                    if len(hits) != len(found):
                        result = dataclasses.replace(result, data={**result.data, "results": hits})
                    self.evidence[result.evidence_id] = result
                    notes += [f"[{result.evidence_id}] {h['metadata'].get('date')}: {h['content'][:200]}"
                              for h in hits]
        # Tickers for company-specific tools when the query names no company (C7). "MULTI" is the label under
        # which multi-company analyses are filed, not a ticker.
        memory_tickers = [t for t in targets if t and t != "MULTI"] if not profile.tickers else []
        planning = self.memory.planning_notes(profile.template, profile.tickers) if self.memory else ""
        self.tracer.thought(f"Memory recall found {len(notes)} prior finding(s)"
                            f"{' and strategy notes from earlier runs' if planning else ''}.", node="recall")
        return {"memory_notes": "\n".join([*notes, planning]).strip(), "memory_tickers": memory_tickers}

    def _node_plan(self, state: ResearchState) -> ResearchState:
        profile = state["profile"]
        sections = TEMPLATES[profile.template]
        tool_names = [s.name for s in self.registry.available() if s.name not in SYSTEM_TOOLS]
        # Leave about a fifth of the budget for repairs and follow-up steps.
        max_steps = max(1, self.cfg.max_tool_calls - self.cfg.max_tool_calls // 5)
        memory_tickers = state.get("memory_tickers", [])
        prompt = prompts.plan_prompt(query=state["query"], template=profile.template, sections=sections,
                                     entities=profile.tickers, memory=state.get("memory_notes", ""),
                                     tool_names=tool_names, max_steps=max_steps, memory_tickers=memory_tickers)
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
        if plan is not None:
            plan.steps, notes = enforce_tickers(plan.steps, profile.tickers or memory_tickers)
            plan.rejected += notes
            if not plan.steps:
                plan = None  # nothing runnable was left: use the default plan, as for an unusable reply
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
        if len(ready) <= 1 or self.max_workers <= 1:
            for step in ready:
                self._run_step(step, plan)
        else:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(ready))) as pool:
                list(pool.map(lambda s: self._run_step(s, plan), ready))
        return {"plan": plan}

    def _run_step(self, step: PlanStep, plan: Plan) -> None:
        """Run one plan step: a direct call if the arguments are known, else a bounded ReAct loop."""
        problem, resolved_note = "", ""
        if step.args is not None and step.tool == "calculation_engine":
            # Planned before the data existed, calculation inputs are often names ("s2.revenue_FY2025");
            # look them up deterministically instead of spending LLM calls on a repair (Day 11 finding).
            resolved, resolved_note = resolve_references(step.args, self._financial_data_for(step, plan))
            if resolved is not None:
                step.args = resolved
                self.tracer.decision("resolve_inputs", resolved_note, step_id=step.id)
        if step.args is not None:
            if not (self._time_left() and self._take_call()):
                return self._finish(step, "skipped", "not run: the time or tool budget ran out")
            step.attempts += 1
            try:
                result = self._dispatch(step.tool, step.args)
                return self._finish(step, "done", f"{step.tool} returned {result.evidence_id}"
                                    + (f"; {resolved_note}" if resolved_note else ""), result)
            except ToolValidationError as exc:
                problem = str(exc)  # wrong arguments: the LLM can repair these
            except ToolError as exc:
                # The call was well-formed but the data is not there ("no filings for PLTR"), or
                # every fallback failed too. Changing the arguments cannot fix that, so do not spend
                # LLM calls and budget trying.
                return self._finish(step, "failed", _failure_text(exc))

        dependencies = "\n\n".join(
            evidence_digest(self.evidence[eid], 1800)
            for dep in step.depends_on for eid in plan.get(dep).evidence_ids if eid in self.evidence
        )
        research_tools = [t for t in self.registry.openai_tools() if t["function"]["name"] not in SYSTEM_TOOLS]
        # Only the planned tool's schema goes with the call (the full list was 56% of all tokens, Day 12);
        # the other tool names are listed in the prompt so the model can still say another tool is needed.
        # An unknown planned tool falls back to every schema. Dispatch validates whatever the model calls.
        tools = [t for t in research_tools if t["function"]["name"] == step.tool] or research_tools
        other_tools = ([t["function"]["name"] for t in research_tools if t["function"]["name"] != step.tool]
                       if tools is not research_tools else [])
        while step.attempts < self.cfg.max_calls_per_step and self._time_left() and self._budget_left():
            prompt = prompts.execute_step_prompt(goal=step.goal, tool=step.tool, evidence=dependencies, problem=problem,
                                                 other_tools=other_tools)
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
                if call.name in SYSTEM_TOOLS:
                    # Same rule as the planner: the system runs these (a model must not, for example,
                    # write its own "findings" into memory). Counts as an attempt so the loop ends.
                    step.attempts += 1
                    problem = f"{call.name} is run by the system and cannot be called in a research step"
                    continue
                if step.attempts >= self.cfg.max_calls_per_step or not self._take_call():
                    break
                step.attempts += 1
                try:
                    result = self._dispatch(call.name, call.arguments)
                    step.args = result.args
                    return self._finish(step, "done", f"{call.name} returned {result.evidence_id}", result)
                except ToolError as exc:
                    problem = str(exc)
        if not self._budget_left():
            self._stop("tool-call budget exhausted")
        reason = problem or "the call and step budgets ran out"
        return self._finish(step, "failed", f"failed after {step.attempts} attempt(s): {reason[:300]}")

    def _financial_data_for(self, step: PlanStep, plan: Plan) -> list[dict[str, Any]]:
        """Structured financial data a calculation step can draw on: its dependencies' results,
        or every financial_data_api result so far when the step declared no dependencies."""
        ids = [eid for dep in step.depends_on if dep in {s.id for s in plan.steps}
               for eid in plan.get(dep).evidence_ids]
        with self._lock:
            records = [self.evidence[e] for e in ids if e in self.evidence] or list(self.evidence.values())
        return [r.data for r in records if r.tool == "financial_data_api"]

    def _finish(self, step: PlanStep, status: str, summary: str, result: ToolResult | None = None) -> None:
        step.status, step.summary = status, summary
        if result is not None:
            with self._lock:
                self.evidence[result.evidence_id] = result
            step.evidence_ids.append(result.evidence_id)
            if result.is_fallback:  # graceful degradation is disclosed, never silent (spec A4.3)
                note = next((n for n in result.data.get("notes", []) if n.startswith("Fallback:")), "fallback data")
                with self._lock:
                    self.failures.append((result.data.get("fallback_for", step.tool), dict(step.args or {}), note))
                    self.gaps.append(f"Step '{step.goal}' ({step.tool} {_args_text(step.args)}): {note} "
                                     f"[{result.evidence_id}] is less reliable than the primary source.")
        if status == "failed":
            with self._lock:
                self.failures.append((step.tool, dict(step.args or {}), summary))
        if status in ("failed", "skipped") and not summary.startswith("not needed"):
            with self._lock:
                self.gaps.append(f"Step '{step.goal}' ({step.tool}) {summary}.")
        self.tracer.log("note", text=f"step {step.id} {status}: {summary}", step_id=step.id)

    def _node_review(self, state: ResearchState) -> ResearchState:
        """Once the plan is complete, let the LLM add steps for real gaps (bounded by max_replans)."""
        plan, replans = state["plan"], state.get("replans", 0)
        gaps = state.get("pending_gaps", [])
        if replans >= self.cfg.max_replans or not self._budget_left() or not self._time_left():
            # Nothing new can be researched, so a gap-driven review must not re-write the same draft.
            return {"pending_gaps": [], "skip_resynthesis": bool(gaps)}
        if not self.llm.any_provider_ready():
            # Every provider is cooling down or out of quota: a review call would fail or wait, and any step
            # it added could need the LLM to run (Day 12 stress finding).
            self.tracer.decision("research_complete", "review skipped: no LLM provider is available right now "
                                 "(all are rate-limited or out of quota); writing the report with what was gathered")
            return {"plan": plan, "replans": replans + 1, "pending_gaps": [], "skip_resynthesis": bool(gaps)}
        if not gaps and not plan.by_status("failed", "skipped"):
            # Day 12: the review answered "done" in most runs. With every step done and no gap noted there is
            # nothing for it to add, so skip the call.
            self.tracer.decision("research_complete", f"deterministic check: all {len(plan.steps)} planned step(s) "
                                 "done, no failed or skipped step, no gap noted; review call skipped")
            return {"plan": plan, "replans": replans + 1, "pending_gaps": [], "skip_resynthesis": False}
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
            # As below: no new steps, so re-running synthesis would only re-write the same draft.
            return {"replans": replans + 1, "pending_gaps": [], "skip_resynthesis": bool(gaps)}
        for note in decision.rejected:
            self.tracer.log("note", text=f"Review validation: {note}")
        decision.new_steps, notes = enforce_tickers(decision.new_steps,
                                                    profile.tickers or state.get("memory_tickers", []))
        for note in notes:
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
        # Spec A6: reconcile the sources before writing, so the writer works from resolved figures.
        self.ledger = build_ledger(self.evidence, self.synthesis_cfg.reliability())
        threads = build_threads(self.ledger, self.evidence)
        brief = render_brief(self.ledger, threads, max_tokens=self.synthesis_cfg.brief_tokens)
        self.tracer.metric("synthesis_ledger", {**self.ledger.summary(), "threads": [t.theme for t in threads]})
        for c in self.ledger.conflicts:
            self.tracer.decision(f"conflict:{c.method}", c.rationale, entity=c.entity, metric=c.metric)
        overhead = count_tokens(self._writing_system) + count_tokens(
            prompts.synthesize_prompt(query=state["query"], evidence="", sections=sections_needed, brief=brief))
        # The provider that will answer sets the cap: Gemini takes a far bigger prompt than Groq's free tier.
        provider = self.llm.next_provider_name()
        budget = self.cfg.prompt_budget(provider)
        self.tracer.metric("synthesis_prompt_budget", {"provider": provider, "budget_tokens": budget,
                                                       "overhead_tokens": overhead})
        fitted = fit_evidence(self.evidence, max(300, budget - overhead), evidence_digest)
        if fitted.dropped:
            self.tracer.log("note", text=f"Context budget: left {len(fitted.dropped)} evidence item(s) out of the "
                                         f"synthesis prompt: {', '.join(fitted.dropped)}")
        prompt = prompts.synthesize_prompt(query=state["query"], evidence=fitted.text, sections=sections_needed,
                                           brief=brief)
        try:
            reply = self._llm(prompt, purpose="synthesize", writing=True, reasoning_effort="low", max_tokens=6000)
        except LLMError as exc:
            small = self.cfg.prompt_budget(None)
            if budget <= small:
                self.gaps.append(f"The report text could not be written: {str(exc)[:200]}")
                return {"sections": {}}
            # The prompt was sized for a large-context provider that then failed; the fallback (Groq's free
            # tier) rejects a prompt that big. Re-fit the evidence to the default cap and try once more.
            self.tracer.log("note", text=f"Synthesis failed with a {budget}-token budget for {provider}; "
                                         f"retrying with the default {small}-token budget")
            fitted = fit_evidence(self.evidence, max(300, small - overhead), evidence_digest)
            prompt = prompts.synthesize_prompt(query=state["query"], evidence=fitted.text,
                                               sections=sections_needed, brief=brief)
            try:
                reply = self._llm(prompt, purpose="synthesize:refit", writing=True, reasoning_effort="low",
                                  max_tokens=6000)
            except LLMError as exc2:
                self.gaps.append(f"The report text could not be written: {str(exc2)[:200]}")
                return {"sections": {}}
        sections = parse_sections(reply.content, sections_needed)
        # A fallback model with a small output cap (Groq's Qwen: 1,000 tokens) stops mid-report; Day 12
        # reruns lost 2-4 sections that way. Ask it to carry on: the cut section is rewritten in full.
        for attempt in range(MAX_CONTINUATIONS):
            missing = [s for s in sections_needed if s not in sections]
            written = [s for s in sections_needed if s in sections]
            # A cut can also land inside the final section, leaving nothing "missing" but it unfinished.
            if reply.finish_reason != "length" or not (missing or written) or not self._time_left_to_write():
                break
            todo = written[-1:] + missing  # the last section was probably cut off: write it again
            self.tracer.thought(f"Synthesis was cut off by the output limit; continuing with {todo}",
                                node="synthesize", attempt=attempt + 1)
            try:
                reply = self._llm(prompts.continue_prompt(prompt, written=written, todo=todo),
                                  purpose="synthesize:continue", writing=True, reasoning_effort="low",
                                  max_tokens=6000)
            except LLMError as exc:
                self.gaps.append(f"The report was cut off and could not be completed: {str(exc)[:200]}")
                break
            # Only the sections asked for: a model that also repeats a finished section must not replace
            # good text, and a rewrite cut off again must not replace a longer one.
            for key, text in parse_sections(reply.content, todo).items():
                if key in todo and not (key in sections and reply.finish_reason == "length"
                                        and len(text) < len(sections[key])):
                    sections[key] = text
        return {"sections": sections}

    def _time_left_to_write(self) -> bool:
        return self._clock() < self._deadline - 15  # a continuation call needs a few seconds

    def _node_verify(self, state: ResearchState) -> ResearchState:
        """Fact-check every sentence that contains a number (spec A4.3 verification pipeline)."""
        sections = dict(state.get("sections", {}))
        tally = {"checked": 0, "supported": 0, "partially_supported": 0, "unsupported": 0, "unverifiable": 0,
                 "uncited": 0, "auto_cited": 0}
        flagged: list[str] = []
        digests = {eid: verification_text(r) for eid, r in self.evidence.items()}
        all_sources = list(digests.values())
        for key, body in sections.items():
            # Claims come in document order; each edit is made at the claim's own position. A plain
            # str.replace(claim, ..., 1) hits the first occurrence, so a repeated sentence would be
            # marked twice at its first occurrence and never at the second.
            cursor = 0
            for claim, inherited in _claims(body):
                at = body.find(claim, cursor)
                if at != -1:
                    cursor = at + len(claim)
                # A model can invent a plausible executive ("Charlie Schwab, CEO" of Wells Fargo), so a
                # named officer must appear in some source, just as a figure must.
                invented = unsupported_names(claim, all_sources)
                if invented:
                    tally["unsupported_names"] = tally.get("unsupported_names", 0) + len(invented)
                    flagged.append(f"{section_title(key)}: name not in any source ({', '.join(invented)}): "
                                   f"{claim[:120]}")
                    marked = _mark(claim, f"unverified: {', '.join(invented)} not found in sources")
                    body, cursor = _replace_at(body, at, claim, marked, cursor)
                    claim = marked
                quantities = extract_quantities(claim)
                if not quantities:
                    continue
                ids = [i for i in (cited_ids(claim) or inherited) if i in self.evidence]
                source = None if ids else _find_single_source(claim, quantities, digests)
                if source:
                    # Citation repair: every figure in the claim matches this one source exactly, so
                    # citing it is accurate. Counted separately so the report can disclose it.
                    body, cursor = _replace_at(body, at, claim, _cite(claim, source), cursor)
                    tally["auto_cited"] += 1
                    tally["checked"] += 1
                    tally["supported"] += 1
                    continue
                if not ids:
                    tally["uncited"] += 1
                    flagged.append(f"{section_title(key)}: uncited figure: {claim[:140]}")
                    body, cursor = _replace_at(body, at, claim, _mark(claim, "unverified: no source cited"), cursor)
                    continue
                verdict = fact_checker(claim, [digests[i] for i in ids])
                tally["checked"] += 1
                tally[verdict["status"]] = tally.get(verdict["status"], 0) + 1
                if verdict["status"] in ("unsupported", "partially_supported"):
                    flagged.append(f"{section_title(key)}: {verdict['status'].replace('_', ' ')}: {claim[:140]}")
                    body, cursor = _replace_at(body, at, claim, _mark(claim, "unverified"), cursor)
            sections[key] = body
        self.tracer.metric("verification", tally)
        return {"sections": sections, "verification": {**tally, "flagged": flagged}}

    def _node_reflect(self, state: ResearchState) -> ResearchState:
        needed = TEMPLATES[state["profile"].template]
        missing = [s for s in needed if s not in state.get("sections", {})]
        reflections = state.get("reflections", 0)
        self.tracer.thought(f"Reflection: {len(missing)} section(s) missing {missing or ''}".strip(), node="reflect")
        # Checked only when sections are missing: _time_left() records "time limit reached", which
        # is wrong for a complete report written in the reserve after the research deadline.
        if (missing and reflections < self.cfg.max_reflections and state.get("replans", 0) < self.cfg.max_replans
                and self._budget_left() and self._time_left()):
            return {"pending_gaps": [f"Section '{section_title(s)}' has no supporting evidence." for s in missing],
                    "reflections": reflections + 1}
        return {"pending_gaps": []}

    def _node_render(self, state: ResearchState) -> ResearchState:
        profile, plan = state["profile"], state["plan"]
        verification = state.get("verification", {})
        gaps = list(dict.fromkeys(self.gaps))  # de-duplicate, keep order
        gaps += [f"Unverified claim in {f}" for f in verification.get("flagged", [])]
        sections = dict(state.get("sections", {}))
        today = self._today or date.today()
        gaps = [f.message for f in coverage_flags(self.evidence, self.failures)] + gaps  # spec 15.3 edge cases
        sections["data_gaps_and_limitations"] = "\n".join(f"- {g}" for g in gaps) or ""
        temporal = render_temporal_flags(rate_of_change_flags(self.evidence, today=today))
        if temporal and sections.get("executive_summary"):
            sections["executive_summary"] = f"{temporal}\n\n{sections['executive_summary']}"
        elif temporal:
            sections["data_gaps_and_limitations"] = f"{temporal}\n\n{sections['data_gaps_and_limitations']}"
        if self.ledger.conflicts:  # spec A6.3 step 6 / P6: every conflict and how it was resolved
            sections["data_gaps_and_limitations"] += (
                "\n\nConflicting figures between sources, and how each was resolved:\n\n"
                + conflicts_markdown(self.ledger.conflicts)).lstrip()
        sections["research_methodology"] = self._methodology(plan, verification)
        company = profile.names[0] if len(profile.names) == 1 else ", ".join(profile.tickers) or "Research"
        result = self.registry.dispatch("report_generator", {
            "template": profile.template,
            "title": f"{company}: {section_title(profile.template)}",
            "sections": sections,
            "sources": [source_entry(r) for r in self.evidence.values()],
        })
        return {"report": result.data}

    def _resilience_lines(self) -> list[str]:
        """Graceful degradation (spec A4.3): say which data came from fallbacks and what failed."""
        lines = []
        fallbacks = [r for r in self.evidence.values() if r.is_fallback]
        if fallbacks:
            used = "; ".join(f"[{r.evidence_id}] {r.tool} instead of {r.data.get('fallback_for', '?')}"
                             for r in fallbacks)
            lines.append(f"- Fallback data (lower confidence than the primary source): {used}.")
        counts: dict[str, int] = {}
        for kind, _ in self.resilience_events:
            counts[kind] = counts.get(kind, 0) + 1
        if counts.get("retry") or counts.get("breaker_open"):
            lines.append(f"- Resilience: {counts.get('retry', 0)} retried call(s); circuit breakers opened for "
                         f"{', '.join(self.breakers.open_providers()) or 'no provider'} at the end of research.")
        injector = self.registry.failure_injector
        if injector is not None:
            lines.append(f"- **Failure injection test:** {sum(injector.injected.values())} tool failure(s) were "
                         f"injected on purpose (rate {self.resilience_cfg.fail_rate:.0%}"
                         + (f" on {', '.join(self.resilience_cfg.fail_tools)}" if self.resilience_cfg.fail_tools else "")
                         + ").")
        return lines

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
        lines += self._resilience_lines()
        assumptions = render_assumptions(self.disambiguation) if self.disambiguation else ""
        if assumptions:
            lines.append("\n" + assumptions)
        if any(r.is_mock for r in self.evidence.values()):
            lines.append("- **Mock data was used; figures are approximations for development only.**")
        return "\n".join(lines)

    def _node_remember(self, state: ResearchState) -> ResearchState:
        """Write this run to long-term memory: source material, statement facts, and the verified findings."""
        if self.memory is None:
            return {}
        tickers = state["profile"].tickers
        research = {e: r for e, r in self.evidence.items() if r.tool != "vector_db_search"}
        try:
            counts = self.memory.ingest_evidence(research, self.tracer.session_id, tickers)
            counts["findings"] = self.memory.ingest_findings(
                {k: v for k, v in state.get("sections", {}).items()}, state.get("verification", {}), tickers,
                self.tracer.session_id)
            self.tracer.log("note", text="Stored in memory", **counts)
        except Exception as exc:  # memory must never break a finished run
            self.tracer.log("note", text=f"Could not write to memory: {exc}")
        return {}

    # ------------------------------------------------------------------ routing

    def _route_after_execute(self, state: ResearchState) -> str:
        plan = state["plan"]
        if plan.ready() and self._budget_left() and self._time_left():
            return "execute"
        if plan.ready() and not self._budget_left():
            # Steps could still run but the last call went to an earlier step: record why they will not.
            self._stop("tool-call budget exhausted")
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


def _replace_at(body: str, at: int, claim: str, new: str, cursor: int) -> tuple[str, int]:
    """Replace the claim that starts at ``at``; return the new body and the position after the edit.

    ``at`` is -1 only if the claim was not found (it always should be); nothing changes then.
    """
    if at < 0:
        return body, cursor
    return body[:at] + new + body[at + len(claim):], at + len(new)


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
_LIST_ITEM = re.compile(r"^(?:[-*+]|\d{1,2}[.)])\s+")
_ABBREVIATION = re.compile(r"\b(?:vs|e\.g|i\.e|approx|Inc|Corp|Co|Ltd|No|U\.S)\.$", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    """Sentences of ``text``; "vs.", "e.g." and "Inc." do not end a sentence (a split there left
    "GM's 27.0x P/E ..." as a separate, uncited claim)."""
    parts: list[str] = []
    for part in _SENTENCE.split(text):
        if parts and _ABBREVIATION.search(parts[-1]):
            parts[-1] = f"{parts[-1]} {part}"
        else:
            parts.append(part)
    return [p.strip() for p in parts if p.strip()]


def _claims(body: str) -> list[tuple[str, set[str]]]:
    """The units that get fact-checked: sentences and table rows, each with inherited citations.

    Analysts (and models) usually cite a table once, in its caption or header, not in every row. So
    a table row without its own citation inherits those of the table's header and of the line just
    before the table. Header and separator rows are not claims. Likewise, the items of a list inherit
    the citations of the line that introduces the list ("Risks cited in the 10-K [E5]:"), and a list
    marker ("1.") is not a figure.
    """
    claims: list[tuple[str, set[str]]] = []
    lines = [line.strip() for line in body.splitlines()]
    previous_text = ""
    in_table, table_ids = False, set()
    list_ids: set[str] = set()
    for i, line in enumerate(lines):
        if not line:
            in_table = False
            continue
        if marker := _LIST_ITEM.match(line):
            item = line[marker.end():]
            claims += [(s, list_ids) for s in _sentences(item)]
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
        list_ids = cited_ids(line) if line.rstrip("*_ ").endswith(":") else set()
        claims += [(s, set()) for s in _sentences(line)]
    return claims
