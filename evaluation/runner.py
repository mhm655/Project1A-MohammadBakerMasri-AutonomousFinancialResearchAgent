"""Run the eight spec challenges (section B2) and save each run as a self-contained *run bundle*.

A bundle (``results/runs/C<n>.json``) holds everything the metrics need: the report, its sections,
the plan, every piece of evidence, the tool call log, resilience events, LLM usage and the trace
events. Scoring reads bundles only, so metrics can be recomputed (and improved) without paying for
another research run.

    python -m evaluation.runner                 # all eight challenges, live data
    python -m evaluation.runner C1 C4 --pause 30
    python -m evaluation.runner C8 --out-dir results/runs
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_VERSION = 1
#: Strings longer than this inside evidence data are cut, so a bundle stays a few hundred KB.
MAX_STRING = 20_000
#: Trace event types copied into the bundle (traces/ itself is not committed).
TRACE_TYPES = frozenset({"llm_call", "decision", "error", "note", "conflict", "action", "observation", "thought"})


@dataclasses.dataclass(frozen=True)
class Challenge:
    """One of the spec's evaluation challenges: its query, optional failure injection and expected report length in pages."""
    id: str
    number: int
    title: str
    query: str
    #: Failure injection for this run (Challenge 8: 50% on the two tools the spec names).
    fail_rate: float = 0.0
    fail_tools: tuple[str, ...] = ()
    expected_pages: tuple[int, int] = (3, 5)


CHALLENGES: dict[str, Challenge] = {c.id: c for c in (
    Challenge("C1", 1, "Single-Company Profile",
              "Create a comprehensive profile of Microsoft Corporation including business overview, financial "
              "summary, key executives, and recent developments.", expected_pages=(3, 5)),
    Challenge("C2", 2, "Earnings Analysis",
              "Analyze Apple Inc.'s most recent quarterly earnings. Compare actual results to consensus estimates "
              "and identify key takeaways from the earnings call.", expected_pages=(5, 7)),
    Challenge("C3", 3, "Risk Assessment",
              "Produce a comprehensive risk assessment for Tesla Inc. covering financial risks, operational risks, "
              "regulatory risks, and competitive risks.", expected_pages=(8, 10)),
    Challenge("C4", 4, "Industry Comparison",
              "Compare the cloud computing divisions of Amazon (AWS), Microsoft (Azure), and Google (GCP). Analyze "
              "revenue growth, market share, margins, and competitive advantages.", expected_pages=(10, 12)),
    Challenge("C5", 5, "Contradictory Data Handling",
              "Research Palantir Technologies. Note: Recent news reports suggest the company is struggling, but "
              "their financial statements show strong growth. Investigate and explain the apparent contradiction.",
              expected_pages=(8, 10)),
    Challenge("C6", 6, "Ambiguous Query Handling", "What's happening with the banks?", expected_pages=(5, 7)),
    Challenge("C7", 7, "Sector Analysis With Memory",
              "Based on the companies you've already researched, what themes emerge across the technology sector? "
              "Identify cross-cutting risks and opportunities.", expected_pages=(10, 15)),
    Challenge("C8", 8, "Full Research Report With Degradation",
              "Produce a complete investment research report on NVIDIA Corporation. Note: The financial data API "
              "and SEC filing search tools are currently experiencing intermittent failures (simulate 50% failure "
              "rate).", fail_rate=0.5, fail_tools=("financial_data_api", "sec_filing_search"),
              expected_pages=(15, 20)),
)}


def _jsonable(value: Any, _depth: int = 0) -> Any:
    """Plain strict JSON (dataclasses, tuples, sets, dates), with very long strings cut."""
    if _depth > 40:  # a cyclic structure would otherwise recurse forever and lose the whole run
        return "[...too deep]"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value), _depth + 1)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v, _depth + 1) for v in value]
    if isinstance(value, str):
        return value if len(value) <= MAX_STRING else value[:MAX_STRING] + " [...cut]"
    if isinstance(value, float) and not math.isfinite(value):
        return None  # json.dumps would write NaN/Infinity, which is not valid JSON for other tools
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)


def _trace_events(path: str | None) -> list[dict[str, Any]]:
    if not path or not Path(path).exists():
        return []
    events = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") in TRACE_TYPES:
            events.append(_jsonable(event))
    return events


def build_bundle(challenge: Challenge, agent: Any, result: Any, *, data_mode: str) -> dict[str, Any]:
    """Everything the metrics need from one finished run (see the module docstring)."""
    from agent.query_analyzer import analyze_query

    registry = getattr(agent, "registry", None)
    injector = getattr(registry, "failure_injector", None)
    return {
        "bundle_version": BUNDLE_VERSION,
        "challenge": {"id": challenge.id, "number": challenge.number, "title": challenge.title,
                      "expected_pages": list(challenge.expected_pages)},
        "query": result.query,
        "session_id": result.session_id,
        "data_mode": data_mode,
        "fail_rate": challenge.fail_rate,
        "fail_tools": list(challenge.fail_tools),
        "injected_failures": dict(getattr(injector, "injected", {}) or {}),
        "profile": _jsonable(analyze_query(result.query)),
        "template": result.template,
        "stop_reason": result.stop_reason,
        "duration_s": result.duration_s,
        "tool_calls": result.tool_calls,
        "llm_usage": _jsonable(result.llm_usage),
        "report_markdown": result.report_markdown,
        "sections": _jsonable(result.sections),
        "plan": _jsonable(result.plan),
        "evidence": {eid: _jsonable(r) for eid, r in result.evidence.items()},
        "call_log": _jsonable(getattr(registry, "call_log", [])),
        "resilience_events": _jsonable(getattr(agent, "resilience_events", [])),
        "failures": _jsonable(getattr(agent, "failures", [])),
        "disambiguation": _jsonable(getattr(agent, "disambiguation", None)),
        "conflicts": _jsonable(getattr(getattr(agent, "ledger", None), "conflicts", [])),
        "gaps": list(result.gaps),
        "verification": _jsonable(result.verification),
        "report_checks": _jsonable(result.report_checks),
        "trace_path": result.trace_path,
        "trace_events": _trace_events(result.trace_path),
    }


def run_challenge(challenge: Challenge, *, max_seconds: float = 300.0, seed: int = 8) -> tuple[Any, Any]:
    """One research run with the challenge's failure settings; returns ``(agent, result)``."""
    from agent.core import ResearchAgent
    from config.settings import load_settings

    fail_keys = ("ARA_FAIL_RATE", "ARA_FAIL_SEED", "ARA_FAIL_TOOLS")
    saved = {key: os.environ.get(key) for key in fail_keys}  # restored below, not just cleared
    os.environ.pop("ARA_FAIL_RATE", None)
    os.environ.pop("ARA_FAIL_TOOLS", None)
    if challenge.fail_rate:
        os.environ["ARA_FAIL_RATE"] = str(challenge.fail_rate)
        os.environ["ARA_FAIL_SEED"] = str(seed)
        os.environ["ARA_FAIL_TOOLS"] = ",".join(challenge.fail_tools)
    try:
        settings = load_settings()
        agent = ResearchAgent.from_settings(settings, max_seconds=max_seconds)
        return agent, agent.run(challenge.query)
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def save(challenge: Challenge, bundle: dict[str, Any], *, out_dir: Path, report_dir: Path | None) -> Path:
    """Write a run bundle to ``out_dir/<id>.json`` and, if ``report_dir`` is given, its report to ``challenge_<n>.md``; returns the bundle path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{challenge.id}.json"
    path.write_text(json.dumps(bundle, indent=1, ensure_ascii=False), encoding="utf-8")
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / f"challenge_{challenge.number}.md").write_text(bundle["report_markdown"], encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: run the selected challenges and save their bundles."""
    parser = argparse.ArgumentParser(prog="python -m evaluation.runner", description=__doc__.split("\n\n")[0])
    parser.add_argument("ids", nargs="*", help="challenge ids (default: C1..C8)")
    parser.add_argument("--data", choices=["mock", "live"], default="live")
    parser.add_argument("--out-dir", default=str(ROOT / "results" / "runs"))
    parser.add_argument("--report-dir", default=str(ROOT / "results"),
                        help="where challenge_<n>.md goes ('' to skip)")
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--pause", type=float, default=45.0,
                        help="seconds between runs, so free-tier per-minute LLM quotas recover")
    args = parser.parse_args(argv)

    previous_mode = os.environ.get("ARA_DATA_MODE")
    os.environ["ARA_DATA_MODE"] = args.data
    try:
        return _run_all(args, parser)
    finally:  # do not leave the data mode set for whoever called main() in-process
        if previous_mode is None:
            os.environ.pop("ARA_DATA_MODE", None)
        else:
            os.environ["ARA_DATA_MODE"] = previous_mode


def _run_all(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    ids = [i.upper() for i in args.ids] or list(CHALLENGES)
    unknown = [i for i in ids if i not in CHALLENGES]
    if unknown:
        parser.error(f"unknown challenge id(s): {', '.join(unknown)}")
    from agent.tracing import configure_logging
    from config.settings import load_settings

    configure_logging(load_settings().log_level)
    failed = 0
    for n, cid in enumerate(ids):
        challenge = CHALLENGES[cid]
        if n:
            time.sleep(args.pause)
        try:
            agent, result = run_challenge(challenge, max_seconds=args.max_seconds)
            path = save(challenge, build_bundle(challenge, agent, result, data_mode=args.data),
                        out_dir=Path(args.out_dir), report_dir=Path(args.report_dir) if args.report_dir else None)
            u = result.llm_usage
            print(f"{cid}: {result.stop_reason} in {result.duration_s}s | tools {result.tool_calls} | "
                  f"LLM {u['calls']} calls, {u['prompt_tokens'] + u['completion_tokens']} tokens | {path}", flush=True)
        except Exception:
            failed += 1
            print(f"{cid}: CRASH\n{traceback.format_exc()[-1500:]}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
