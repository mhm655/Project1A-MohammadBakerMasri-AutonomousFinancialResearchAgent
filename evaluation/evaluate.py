"""Score saved run bundles and write scores.json, the HTML dashboard and the Markdown report (spec A5.2, Day 11).

    python -m evaluation.runner                      # produce results/runs/C*.json
    python -m evaluation.evaluate                    # score + judge + dashboard + report
    python -m evaluation.evaluate --ids C1 C4 --no-judge
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from evaluation import dashboard


def _backends() -> SimpleNamespace:
    """Import scoring backends lazily (they are separate modules; tests substitute fakes here)."""
    from evaluation import benchmark, judge, metrics
    return SimpleNamespace(Run=metrics.Run, score_run=metrics.score_run, judge_bundle=judge.judge_bundle,
                           JudgeError=judge.JudgeError, gold_for=benchmark.gold_for, compare=benchmark.compare)


def aggregate(runs: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Per-metric aggregate across challenges plus the overall / per-category summary (pure)."""
    agg: dict[str, dict[str, Any]] = {}
    for run in runs:
        for m in run.get("metrics", []):
            a = agg.setdefault(m["id"], {"name": m.get("name"), "category": m.get("category"),
                                         "target": m.get("target"), "unit": m.get("unit"),
                                         "_vals": [], "passed": 0, "measured": 0, "n": 0})
            a["n"] += 1
            if m.get("value") is not None:
                a["_vals"].append(float(m["value"]))
            if m.get("passed") is not None and m.get("value") is not None:
                a["measured"] += 1
                a["passed"] += bool(m["passed"])
    by_cat: dict[str, dict[str, int]] = {}
    for a in agg.values():
        vals = a.pop("_vals")
        a["mean"] = sum(vals) / len(vals) if vals else None
        c = by_cat.setdefault(a["category"], {"passed": 0, "measured": 0})
        c["passed"] += a["passed"]
        c["measured"] += a["measured"]
    passed = sum(c["passed"] for c in by_cat.values())
    measured = sum(c["measured"] for c in by_cat.values())
    summary = {"metrics_total": len(agg), "pass_rate": passed / measured if measured else 0.0,
               "by_category": by_cat}
    return agg, summary


def _run_entry(bundle: dict[str, Any], metrics: list[dict[str, Any]]) -> dict[str, Any]:
    ch = bundle.get("challenge", {})
    usage = bundle.get("llm_usage", {}) or {}
    return {"id": ch.get("id"), "number": ch.get("number"), "title": ch.get("title"),
            "query": bundle.get("query"), "template": bundle.get("template"),
            "data_mode": bundle.get("data_mode"), "stop_reason": bundle.get("stop_reason"),
            "duration_s": bundle.get("duration_s"), "tool_calls": bundle.get("tool_calls"),
            "llm_calls": usage.get("calls", 0),
            "tokens": (usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0),
            "fail_rate": bundle.get("fail_rate"), "injected_failures": bundle.get("injected_failures") or {},
            "metrics": metrics}


def evaluate_dir(runs_dir: Path, ids: list[str] | None, *, use_judge: bool, force_judge: bool,
                 pause: float) -> dict[str, Any]:
    be = _backends()
    runs: list[tuple[int, Any]] = []
    for path in sorted(runs_dir.glob("C*.json")):
        run = be.Run.load(path)
        ch = run.bundle.get("challenge", {})
        if ids and ch.get("id") not in ids:
            continue
        runs.append((ch.get("number", 0), run))
    runs.sort(key=lambda t: t[0])

    entries: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    need_pause = False
    for _n, run in runs:
        bundle = run.bundle
        cid = bundle["challenge"]["id"]
        judgement = None
        if use_judge:
            if need_pause and pause > 0:
                time.sleep(pause)  # free-tier quota: space out real LLM calls
            try:
                judgement = be.judge_bundle(bundle, force=force_judge)
                need_pause = not (judgement or {}).get("cached", False)
            except Exception as exc:  # JudgeError or any provider failure: keep scoring without the judge
                errors[cid] = f"{type(exc).__name__}: {exc}"
                need_pause = True
        gold = be.gold_for(cid)
        bench = be.compare(bundle, gold) if gold else None
        results = be.score_run(run, judgement, bench)
        entries.append(_run_entry(bundle, [m.to_dict() for m in results]))

    agg, summary = aggregate(entries)
    return {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "judge": {"enabled": use_judge, "errors": errors}, "runs": entries,
            "aggregate": agg, "summary": summary}


def _print_summary(scores: dict[str, Any]) -> None:
    for r in scores["runs"]:
        ms = [m for m in r["metrics"] if m.get("value") is not None and m.get("passed") is not None]
        fails = [m["id"] for m in ms if not m["passed"]]
        ok = sum(1 for m in ms if m["passed"])
        print(f"{r['id']}: pass {ok}/{len(ms)}" + (f"  failing: {', '.join(fails)}" if fails else ""))
    s = scores["summary"]
    print(f"overall pass rate {100 * s['pass_rate']:.0f}%")
    for cid, err in scores["judge"]["errors"].items():
        print(f"judge error {cid}: {err}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score run bundles and render the dashboard and report")
    p.add_argument("--runs", default="results/runs")
    p.add_argument("--ids", nargs="*")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--force-judge", action="store_true")
    p.add_argument("--out", default="results/eval/scores.json")
    p.add_argument("--dashboard", default="results/evaluation_dashboard.html")
    p.add_argument("--report", default="results/evaluation_report.md")
    p.add_argument("--pause", type=float, default=20.0, help="seconds between live judge calls")
    a = p.parse_args(argv)

    scores = evaluate_dir(Path(a.runs), a.ids, use_judge=not a.no_judge, force_judge=a.force_judge,
                          pause=a.pause)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=1), encoding="utf-8")
    dashboard.write_outputs(scores, Path(a.dashboard), Path(a.report))
    _print_summary(scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
