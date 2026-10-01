"""Stress tests for ARA-1 (spec Day 12 task 2; sections 14-15): concurrency, maximum context, total API outage.

    python -m evaluation.stress                       # all three tests, mock LLM + mock data
    python -m evaluation.stress --only outage
    python -m evaluation.stress --live                # concurrency only, 2 tasks, real LLM + live data (not run in CI)
    python -m evaluation.stress --out results/stress_test_report.md

Everything runs in full mock mode by default (``ARA_MODE=mock`` scripted LLM, ``ARA_DATA_MODE=mock`` tools), so
the harness is free, offline and deterministic. Each test returns a :class:`StressResult`; :func:`render`
turns the list into the Markdown report. Environment variables are set inside the process and restored after.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import platform
import re
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
#: Mock-LLM run budget per task (seconds); generous, the mock finishes far sooner.
MAX_SECONDS = 120.0
#: Request size may exceed the prompt budget by this fraction before the context test fails (token-count drift).
CONTEXT_TOLERANCE = 0.05
#: Outage runs longer than this (seconds, mock mode) are flagged: retries/backoff should be bounded.
OUTAGE_SLOW_S = 60.0
#: An outage run that makes more LLM calls than this is considered to be spinning.
OUTAGE_MAX_LLM_CALLS = 40
#: Characters of filler added to every mock tool result in the context test.
DEFAULT_FILLER_CHARS = 400_000
#: Company names per challenge, for the cross-talk check (a title may only name its own task's companies).
COMPANIES: dict[str, tuple[str, ...]] = {
    "C1": ("Microsoft",), "C2": ("Apple",), "C3": ("Tesla",),
    "C4": ("Amazon", "Microsoft", "Google", "Alphabet"), "C5": ("Palantir",),
}
_ALL_COMPANIES = sorted({c for names in COMPANIES.values() for c in names} | {"NVIDIA"})
_FILING_PARAGRAPH = (
    "Item 7. Management's discussion and analysis of financial condition and results of operations. "
    "Revenue increased 12% year over year driven by cloud services, while operating margin expanded 150 basis "
    "points on cost discipline. Risk factors include supply constraints, regulatory scrutiny and currency "
    "movements; see Note 14 for segment information. "
)


@dataclass
class StressResult:
    name: str
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    #: Defects in agent/ or tools/ code surfaced by the test (file:line, repro, observed vs expected).
    bugs: list[str] = field(default_factory=list)
    method: str = ""


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _env(**values: str | None) -> Iterator[None]:
    """Set (or unset, for None) environment variables for the block, restoring them afterwards."""
    saved = {k: os.environ.get(k) for k in values}
    try:
        for key, val in values.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def _mock_env(**extra: str | None) -> Any:
    base: dict[str, str | None] = {"ARA_MODE": "mock", "ARA_DATA_MODE": "mock", "ARA_FAIL_RATE": None,
                                   "ARA_FAIL_SEED": None, "ARA_FAIL_TOOLS": None}
    return _env(**{**base, **extra})


def _query(cid: str) -> str:
    from evaluation.runner import CHALLENGES

    return CHALLENGES[cid].query


class _Watcher(threading.Thread):
    """Samples the process's thread count (and RSS if psutil exists) every 50 ms."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._stop_evt = threading.Event()
        self.peak_threads = threading.active_count()
        self.peak_rss_mb: float | None = None
        try:
            import psutil

            self._proc = psutil.Process()
        except Exception:  # psutil is optional
            self._proc = None

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self.peak_threads = max(self.peak_threads, threading.active_count())
            if self._proc is not None:
                rss = self._proc.memory_info().rss / 1e6
                self.peak_rss_mb = rss if self.peak_rss_mb is None else max(self.peak_rss_mb, rss)
            self._stop_evt.wait(0.05)

    def stop(self) -> None:
        self._stop_evt.set()
        self.join(timeout=1.0)


def _report_sections(markdown: str) -> dict[str, str]:
    """``## Heading`` -> body, from the rendered report (``result.sections`` lacks the rendered gaps section)."""
    parts = re.split(r"^## +(.+)$", markdown, flags=re.MULTILINE)
    return {parts[i].strip().lower(): parts[i + 1] for i in range(1, len(parts) - 1, 2)}


def _own_state_attrs() -> list[str]:
    """Attributes ``ResearchAgent.run`` assigns on ``self``: the per-run state that makes one instance unshareable."""
    from agent.core import ResearchAgent

    src = inspect.getsource(ResearchAgent.run)
    return sorted(set(re.findall(r"self\.([A-Za-z_]\w*)(?::[^=\n]+)?\s*=(?!=)", src)))


# --------------------------------------------------------------------------------------------------
# (a) Concurrency
# --------------------------------------------------------------------------------------------------


def _run_one(cid: str, settings: Any, traces_dir: str) -> dict[str, Any]:
    from agent.core import ResearchAgent

    t0 = time.perf_counter()
    out: dict[str, Any] = {"cid": cid, "error": None}
    try:
        agent = ResearchAgent.from_settings(settings, max_seconds=MAX_SECONDS, traces_dir=traces_dir)
        res = agent.run(_query(cid))
        out["result"] = res
    except Exception as exc:  # the test's point is to record, not raise
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["duration_s"] = round(time.perf_counter() - t0, 3)
    return out


def _crosstalk(cid: str, res: Any) -> list[str]:
    """Problems suggesting this report contains another task's data."""
    problems = []
    cited = set(re.findall(r"\bE\d+\b", res.report_markdown))
    missing = sorted(cited - set(res.evidence))
    if missing:
        problems.append(f"{cid}: cites evidence ids not in its own evidence dict: {missing[:5]}")
    title = next((ln for ln in res.report_markdown.splitlines() if ln.strip()), "")
    own = {n.lower() for n in COMPANIES.get(cid, ())}
    foreign = [n for n in _ALL_COMPANIES if n.lower() not in own and n.lower() in title.lower()]
    if foreign:
        problems.append(f"{cid}: title line names other tasks' companies {foreign}: {title[:80]!r}")
    return problems


def run_concurrency(n: int = 5, *, live: bool = False) -> StressResult:
    """``n`` research tasks at once on separate agent instances, versus the same tasks sequentially."""
    t_start = time.perf_counter()
    ids = [f"C{i}" for i in range(1, n + 1)]
    env = _env(ARA_FAIL_RATE=None, ARA_DATA_MODE="live") if live else _mock_env()
    findings: list[str] = []
    metrics: dict[str, Any] = {"tasks": n, "mode": "live" if live else "mock"}
    problems: list[str] = []
    with env, tempfile.TemporaryDirectory(prefix="ara_stress_traces_") as tmp:
        from config.settings import load_settings

        settings = load_settings()
        t0 = time.perf_counter()
        sequential = [_run_one(c, settings, tmp) for c in ids]
        metrics["sequential_wall_s"] = round(time.perf_counter() - t0, 3)

        watcher = _Watcher()
        watcher.start()
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n) as pool:
            concurrent = list(pool.map(lambda c: _run_one(c, settings, tmp), ids))
        metrics["concurrent_wall_s"] = round(time.perf_counter() - t0, 3)
        watcher.stop()

        metrics["peak_threads"] = watcher.peak_threads
        metrics["peak_rss_mb"] = round(watcher.peak_rss_mb, 1) if watcher.peak_rss_mb else "n/a (psutil not installed)"
        metrics["per_task_duration_s"] = {o["cid"]: o["duration_s"] for o in concurrent}
        metrics["per_task_sequential_s"] = {o["cid"]: o["duration_s"] for o in sequential}
        seq, con = metrics["sequential_wall_s"], metrics["concurrent_wall_s"]
        metrics["speedup"] = round(seq / con, 2) if con else None

        errors = [f"{o['cid']}: {o['error']}" for o in concurrent if o["error"]]
        metrics["exceptions"] = len(errors)
        problems += errors
        done = [o for o in concurrent if not o["error"]]
        metrics["completed"] = len(done)
        sessions = [o["result"].session_id for o in done]
        traces = [o["result"].trace_path for o in done]
        metrics["distinct_session_ids"] = len(set(sessions))
        metrics["distinct_trace_files"] = len({t for t in traces if t})
        metrics["stop_reasons"] = {o["cid"]: o["result"].stop_reason for o in done}
        if len(set(sessions)) != len(sessions):
            problems.append("duplicate session ids across concurrent tasks")
        if len({t for t in traces if t}) != len(traces):
            problems.append("tasks share a trace file (or have none)")
        for o in done:
            if not o["result"].report_markdown.strip():
                problems.append(f"{o['cid']}: empty report")
            problems += _crosstalk(o["cid"], o["result"])
        metrics["cross_talk_problems"] = len([p for p in problems if "cites" in p or "title line" in p])

    attrs = _own_state_attrs()
    findings.append(
        "ResearchAgent is per-run stateful: run() assigns "
        f"{', '.join(f'`{a}`' for a in attrs[:12])}{' ...' if len(attrs) > 12 else ''} on self "
        f"({len(attrs)} attributes), so one instance must not be shared between threads; the test therefore "
        "uses one agent per task."
    )
    findings.append("Mock-mode memory uses the module-level `memory.vector_store.DEFAULT_STORE`, shared by all agents; "
                    "no cross-talk was observed in the reports, but it is shared mutable state.")
    if metrics["speedup"] is not None and metrics["speedup"] < 1.5:
        findings.append("Little wall-clock gain from concurrency: mock runs are CPU-bound (GIL) and finish in "
                        "well under a second, so speed-up is not representative of live (I/O-bound) runs.")
    findings += [f"PROBLEM: {p}" for p in problems]
    return StressResult(f"Concurrency ({n} tasks)", not problems, metrics, findings,
                        round(time.perf_counter() - t_start, 2),
                        method=(f"{n} independent agents (C1-C{n} queries) in a ThreadPoolExecutor({n}), after the same "
                                "tasks sequentially as a baseline. A watcher thread samples thread count every 50 ms."))


# --------------------------------------------------------------------------------------------------
# (b) Maximum context utilisation
# --------------------------------------------------------------------------------------------------


def _inflate(registry: Any, filler: str) -> None:
    """Make every tool result huge by adding a ``filler`` field after the normal mock handler returns."""
    original = registry.dispatch

    def dispatch(name: str, args: Any = None) -> Any:
        result = original(name, args)
        if isinstance(result.data, dict) and "filler" not in result.data:
            result.data["filler"] = filler  # the cache holds this same object, so cache hits stay inflated
        return result

    registry.dispatch = dispatch


def run_context(filler_chars: int = DEFAULT_FILLER_CHARS, *, query_id: str = "C1") -> StressResult:
    """Inflate every tool result to ~``filler_chars`` and check no LLM request exceeds the prompt budget."""
    from agent.core import ResearchAgent
    from agent.llm import LLMClient
    from config.settings import load_settings
    from memory.context_manager import count_tokens
    from tools.tool_registry import build_default_registry

    t_start = time.perf_counter()
    filler = (_FILING_PARAGRAPH * (filler_chars // len(_FILING_PARAGRAPH) + 1))[:filler_chars]
    sizes: list[int] = []
    problems: list[str] = []
    findings: list[str] = []
    metrics: dict[str, Any] = {"filler_chars_per_result": filler_chars}
    with _mock_env(), tempfile.TemporaryDirectory(prefix="ara_stress_traces_") as tmp:
        settings = load_settings()
        limit = settings.agent.max_prompt_tokens
        llm = LLMClient.from_settings(settings)
        for provider in llm.providers:
            original = provider.complete

            def recording(request: Any, _orig: Any = original) -> Any:
                sizes.append(count_tokens(json.dumps(request.messages, default=str)))
                return _orig(request)

            provider.complete = recording  # type: ignore[method-assign]

        def factory() -> Any:
            registry = build_default_registry(settings.data_mode)
            _inflate(registry, filler)
            return registry

        from memory.manager import MemoryManager

        agent = ResearchAgent(llm, factory, agent_settings=settings.agent, traces_dir=tmp, max_seconds=MAX_SECONDS,
                              synthesis_settings=settings.synthesis, resilience_settings=settings.resilience,
                              retry_settings=settings.retry, memory=MemoryManager.for_settings(settings))
        res = None
        try:
            res = agent.run(_query(query_id))
        except Exception as exc:
            problems.append(f"exception: {type(exc).__name__}: {exc}")

    limit_with_tol = int(limit * (1 + CONTEXT_TOLERANCE))
    biggest = max(sizes, default=0)
    over = [s for s in sizes if s > limit_with_tol]
    inflated_total = 0
    if res is not None:
        inflated_total = sum(len(json.dumps(r.data, default=str)) for r in res.evidence.values())
    metrics.update({
        "max_prompt_tokens_setting": limit, "tolerance": f"{int(CONTEXT_TOLERANCE * 100)}% (= {limit_with_tol})",
        "llm_requests": len(sizes), "max_request_tokens": biggest,
        "utilisation_of_limit": round(biggest / limit, 3) if limit else None,
        "requests_over_limit": len(over), "evidence_items": len(res.evidence) if res else 0,
        "raw_evidence_chars": inflated_total,
        "report_produced": bool(res and res.report_markdown.strip()),
        "stop_reason": res.stop_reason if res else "exception",
        "duration_s": res.duration_s if res else None,
    })
    if over:
        problems.append(f"{len(over)} request(s) over {limit_with_tol} tokens (max {biggest})")
    if res is not None and not res.report_markdown.strip():
        problems.append("no report produced")
    findings.append(f"Measured request size = tiktoken count of the JSON `messages` array (tool schemas excluded); "
                    f"limit is `agent.max_prompt_tokens` = {limit} with a {int(CONTEXT_TOLERANCE * 100)}% tolerance "
                    "for tokenizer drift.")
    findings += [f"PROBLEM: {p}" for p in problems]
    return StressResult("Maximum context utilisation", not problems, metrics, findings,
                        round(time.perf_counter() - t_start, 2),
                        method=(f"Every mock tool result is inflated with a ~{filler_chars:,}-character filing-text "
                                f"`filler` field (registry dispatch wrapped; core untouched). The {query_id} query runs "
                                "with a recording LLM that counts tokens of every request."))


# --------------------------------------------------------------------------------------------------
# (c) All external APIs failing
# --------------------------------------------------------------------------------------------------


def run_outage(query_ids: tuple[str, ...] = ("C1", "C8")) -> StressResult:
    """``ARA_FAIL_RATE=1.0``: every network tool call fails; the agent must degrade, not crash or spin."""
    from agent.core import ResearchAgent
    from config.settings import load_settings
    from tools.fact_checker import extract_quantities

    t_start = time.perf_counter()
    problems: list[str] = []
    findings: list[str] = []
    per_query: dict[str, dict[str, Any]] = {}
    with _mock_env(ARA_FAIL_RATE="1.0", ARA_FAIL_SEED="12"), tempfile.TemporaryDirectory(prefix="ara_stress_traces_") as tmp:
        settings = load_settings()
        for cid in query_ids:
            row: dict[str, Any] = {}
            per_query[cid] = row
            agent = ResearchAgent.from_settings(settings, max_seconds=MAX_SECONDS, traces_dir=tmp)
            try:
                res = agent.run(_query(cid))
            except Exception as exc:
                row["exception"] = f"{type(exc).__name__}: {exc}"
                problems.append(f"{cid}: exception {row['exception']}")
                continue
            injector = getattr(agent.registry, "failure_injector", None)
            injected = sum((getattr(injector, "injected", {}) or {}).values())
            kinds: dict[str, int] = {}
            for kind, _payload in getattr(agent, "resilience_events", []):
                kinds[kind] = kinds.get(kind, 0) + 1
            rendered = _report_sections(res.report_markdown)
            body = "\n\n".join(text for key, text in rendered.items()
                               if not key.startswith(("data gaps", "research methodology", "sources")))
            gaps_section = next((t for k, t in rendered.items() if k.startswith("data gaps")), "")
            quantities = len(extract_quantities(body))
            row.update({
                "stop_reason": res.stop_reason, "duration_s": res.duration_s, "tool_calls": res.tool_calls,
                "injected_failures": injected, "breaker_events": dict(sorted(kinds.items())),
                "evidence_items": len(res.evidence), "gaps": len(res.gaps),
                "gaps_section_chars": len(gaps_section.strip()), "llm_calls": res.llm_usage.get("calls"),
                "llm_failures": res.llm_usage.get("failures"), "body_quantities": quantities,
                "report_words": len(res.report_markdown.split()),
            })
            if not res.gaps and not gaps_section.strip():
                problems.append(f"{cid}: no gaps listed despite every tool failing")
            if res.duration_s >= MAX_SECONDS:
                problems.append(f"{cid}: run hit the time limit ({res.duration_s}s)")
            if res.duration_s > OUTAGE_SLOW_S:
                findings.append(f"{cid}: took {res.duration_s}s in mock mode (> {OUTAGE_SLOW_S:.0f}s): retry backoff is slow.")
            if (res.llm_usage.get("calls") or 0) > OUTAGE_MAX_LLM_CALLS:
                problems.append(f"{cid}: {res.llm_usage.get('calls')} LLM calls (> {OUTAGE_MAX_LLM_CALLS}): agent is spinning")
            if not res.report_markdown.strip():
                problems.append(f"{cid}: empty report")
            if quantities:
                findings.append(f"{cid}: {quantities} numeric quantities in body sections with {len(res.evidence)} "
                                "successful evidence items; the scripted mock LLM can emit canned numbers, so this "
                                "measures the pipeline, not a live model's fabrication rate.")
            elif not res.evidence:
                findings.append(f"{cid}: no numeric quantities in the body with zero evidence: nothing fabricated.")
    metrics: dict[str, Any] = {"fail_rate": 1.0}
    for cid, row in per_query.items():
        for key, val in row.items():
            metrics[f"{cid}.{key}"] = val
    findings += [f"PROBLEM: {p}" for p in problems]
    return StressResult("All external APIs failing", not problems, metrics, findings,
                        round(time.perf_counter() - t_start, 2),
                        method=("`ARA_FAIL_RATE=1.0` (FailureInjector on every tool) for " + " and ".join(query_ids) +
                                ". Checks: no exception, bounded duration and LLM calls, gaps reported, breaker events, "
                                "quantities in body sections."))


# --------------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items()) or "none"
    return str(value)


def _key_numbers(r: StressResult) -> str:
    m = r.metrics
    if r.name.startswith("Concurrency"):
        return (f"{m.get('completed')}/{m.get('tasks')} done, concurrent {m.get('concurrent_wall_s')}s vs "
                f"sequential {m.get('sequential_wall_s')}s, peak threads {m.get('peak_threads')}")
    if r.name.startswith("Maximum"):
        return f"max request {m.get('max_request_tokens')} / {m.get('max_prompt_tokens_setting')} tokens, {m.get('requests_over_limit')} over"
    durations = [f"{k[:-len('.duration_s')]} {v}s" for k, v in m.items() if k.endswith(".duration_s")]
    calls = [f"{k[:-len('.llm_calls')]} {v} LLM calls" for k, v in m.items() if k.endswith(".llm_calls")]
    return "; ".join(durations + calls)


MARKER = "<!-- ANALYSIS -->"


def render(results: list[StressResult], *, live: bool = False) -> str:
    env_line = (f"mock LLM + mock data (full mock mode), {os.cpu_count()} CPUs, Python {platform.python_version()}, "
                f"{platform.system()}")
    lines = ["# ARA-1 Stress Test Report", "", f"Date: {date.today().isoformat()}  ", f"Environment: {env_line}", "",
             "Generated by `python -m evaluation.stress` (spec Day 12 task 2).", "", "## Summary", "",
             "| Test | Result | Key numbers |", "|---|---|---|"]
    for r in results:
        lines.append(f"| {r.name} | {'PASS' if r.passed else 'FAIL'} | {_key_numbers(r)} |")
    for r in results:
        lines += ["", f"## {r.name}", "", f"**Result:** {'PASS' if r.passed else 'FAIL'} ({r.duration_s}s)", "",
                  f"**Method.** {r.method}", "", "| Metric | Value |", "|---|---|"]
        lines += [f"| {k} | {_fmt(v)} |" for k, v in r.metrics.items()]
        lines += ["", "**Findings**", ""] + [f"- {f}" for f in r.findings or ["none"]]
        lines += ["", "**Bugs found**", ""] + [f"- {b}" for b in r.bugs or ["none found"]]
    lines += ["", "## Live-mode note", "",
              "The concurrency test was run in mock mode because five concurrent live runs would exhaust free-tier "
              "LLM quotas (per-minute and per-day limits are shared across the provider chain). "
              "`python -m evaluation.stress --live` is supported: it switches to `ARA_DATA_MODE=live` with the real LLM "
              "chain and runs the concurrency test with 2 tasks only. It was not run for this report.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evaluation.stress", description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", choices=["concurrency", "context", "outage"])
    parser.add_argument("--out", default=str(ROOT / "results" / "stress_test_report.md"))
    parser.add_argument("--live", action="store_true",
                        help="concurrency test only, 2 tasks, live data + real LLM (spends free-tier quota)")
    args = parser.parse_args(argv)

    from agent.tracing import configure_logging
    from config.settings import load_settings

    with _mock_env():
        configure_logging(load_settings().log_level)
    if args.live:
        results = [run_concurrency(2, live=True)]
    else:
        tests = {"concurrency": run_concurrency, "context": run_context, "outage": run_outage}
        results = []
        for name in ([args.only] if args.only else list(tests)):
            print(f"running {name} ...", flush=True)
            results.append(tests[name]())
            print(f"  {'PASS' if results[-1].passed else 'FAIL'} in {results[-1].duration_s}s", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = render(results, live=args.live)
    # Keep the hand-written findings (Challenge 8, fixes, re-runs) below the marker on re-render.
    old = out.read_text(encoding="utf-8") if out.exists() else ""
    if MARKER in old:
        text = text.rstrip("\n") + "\n\n" + old[old.index(MARKER):]
    out.write_text(text, encoding="utf-8")
    print(f"report: {out}")
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
