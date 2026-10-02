"""Token-usage profile of the challenge runs (spec Day 12 task 3; sections 14-15).

    python -m evaluation.token_profile [--runs results/runs] [--out results/token_usage_analysis.md]

Reads run bundles (``C*.json``, see ``evaluation.runner``), aggregates the ``llm_call`` trace events per
challenge and per purpose group, and renders a Markdown analysis with data-driven optimisation hints.
The text after the ``<!-- ANALYSIS -->`` marker is hand-written and preserved when the file is re-rendered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
MARKER = "<!-- ANALYSIS -->"
PLACEHOLDER = "## Recommendations\n_To be written after review._\n"
GROUPS = ("plan", "execute", "review", "synthesize", "verify", "reflect", "other")
#: Heuristic thresholds for the optimisation list.
GROUP_SHARE_LIMIT = 0.40
EXECUTE_PROMPT_LIMIT = 3000
REASONING_SHARE_LIMIT = 0.30
FAILURE_LIMIT = 3
SYNTH_NEAR_LIMIT = 0.85
DEFAULT_MAX_PROMPT_TOKENS = 6000


def purpose_group(purpose: str | None) -> str:
    """``execute:s3`` -> ``execute``; unknown purposes -> ``other``."""
    base = (purpose or "").split(":", 1)[0].strip().lower()
    if base.startswith("synth"):
        return "synthesize"
    return base if base in GROUPS else "other"


def load_bundles(runs_dir: Path) -> list[dict[str, Any]]:
    """Load the run bundles (C*.json) in ``runs_dir``, skipping unreadable or malformed files."""
    bundles = []
    for path in sorted(runs_dir.glob("C*.json"), key=lambda p: (len(p.stem), p.stem)):
        try:
            bundles.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return bundles


def _calls(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    return [e for e in bundle.get("trace_events", []) if e.get("type") == "llm_call"]


def _tokens(call: dict[str, Any]) -> int:
    return (call.get("prompt_tokens") or 0) + (call.get("completion_tokens") or 0) + (call.get("reasoning_tokens") or 0)


def _empty_group() -> dict[str, Any]:
    return {"calls": 0, "prompt": 0, "completion": 0, "reasoning": 0, "latency_ms": 0.0}


def _aggregate(calls: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {g: _empty_group() for g in GROUPS}
    mix: dict[str, int] = {}
    for c in calls:
        g = groups[purpose_group(c.get("purpose"))]
        g["calls"] += 1
        g["prompt"] += c.get("prompt_tokens") or 0
        g["completion"] += c.get("completion_tokens") or 0
        g["reasoning"] += c.get("reasoning_tokens") or 0
        g["latency_ms"] += c.get("latency_ms") or 0.0
        key = f"{c.get('provider', '?')}/{c.get('model', '?')}"
        mix[key] = mix.get(key, 0) + 1
    prompt = sum(g["prompt"] for g in groups.values())
    completion = sum(g["completion"] for g in groups.values())
    reasoning = sum(g["reasoning"] for g in groups.values())
    total = prompt + completion + reasoning
    for g in groups.values():
        g["total"] = g["prompt"] + g["completion"] + g["reasoning"]
        g["share"] = g["total"] / total if total else 0.0
        g["mean_latency_ms"] = g["latency_ms"] / g["calls"] if g["calls"] else 0.0
        g["mean_prompt"] = g["prompt"] / g["calls"] if g["calls"] else 0.0
    return {"calls": len(calls), "prompt": prompt, "completion": completion, "reasoning": reasoning, "total": total,
            "groups": groups, "provider_mix": mix,
            "prompt_completion_ratio": prompt / completion if completion else None}


def profile(bundles: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure aggregation of run bundles into the numbers the report shows."""
    per: list[dict[str, Any]] = []
    all_calls: list[dict[str, Any]] = []
    failures = 0
    for b in bundles:
        calls = _calls(b)
        agg = _aggregate(calls)
        words = len((b.get("report_markdown") or "").split())
        agg.update({
            "id": (b.get("challenge") or {}).get("id", "?"), "words": words,
            "tokens_per_word": agg["total"] / words if words else None,
            "failures": (b.get("llm_usage") or {}).get("failures", 0) or 0,
            "stop_reason": b.get("stop_reason"),
        })
        failures += agg["failures"]
        per.append(agg)
        all_calls += [dict(c, challenge=agg["id"]) for c in calls]
    overall = _aggregate(all_calls)
    words = sum(p["words"] for p in per)
    overall.update({"words": words, "tokens_per_word": overall["total"] / words if words else None,
                    "failures": failures, "challenges": len(per)})
    top = sorted(all_calls, key=_tokens, reverse=True)[:5]
    top_calls = [{"challenge": c["challenge"], "purpose": c.get("purpose"), "tokens": _tokens(c),
                  "prompt": c.get("prompt_tokens") or 0, "completion": c.get("completion_tokens") or 0,
                  "model": c.get("model")} for c in top]
    result = {"per_challenge": per, "overall": overall, "top_calls": top_calls,
              "max_prompt_seen": max((c.get("prompt_tokens") or 0 for c in all_calls), default=0),
              "synth_max_prompt": max((c.get("prompt_tokens") or 0 for c in all_calls
                                       if purpose_group(c.get("purpose")) == "synthesize"), default=0)}
    result["opportunities"] = opportunities(result)
    return result


def opportunities(prof: dict[str, Any], max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS) -> list[str]:
    """Heuristic hints, each quoting the numbers behind it."""
    out: list[str] = []
    overall = prof["overall"]
    if not overall["total"]:
        return ["No LLM call events found in the bundles; nothing to analyse."]
    for name, g in overall["groups"].items():
        if g["share"] > GROUP_SHARE_LIMIT:
            out.append(f"**{name}** uses {g['share']:.0%} of all tokens ({g['total']:,} over {g['calls']} calls); "
                       f"it is the first place to trim (shorter prompts, fewer calls, or a smaller model).")
    ex = overall["groups"]["execute"]
    if ex["calls"] and ex["mean_prompt"] > EXECUTE_PROMPT_LIMIT:
        out.append(f"execute calls average {ex['mean_prompt']:,.0f} prompt tokens (> {EXECUTE_PROMPT_LIMIT:,}): "
                   "evidence is probably re-sent on every step; send digests or only new evidence.")
    if overall["failures"] > FAILURE_LIMIT:
        out.append(f"{overall['failures']} LLM failures/retries across {overall['challenges']} runs "
                   f"(> {FAILURE_LIMIT}): rate-limit rejections (HTTP 429) are not billed, so this costs latency rather than tokens; check provider limits and backoff.")
    visible = overall["completion"] + overall["reasoning"]
    if visible and overall["reasoning"] / visible > REASONING_SHARE_LIMIT:
        out.append(f"Reasoning tokens are {overall['reasoning'] / visible:.0%} of output "
                   f"({overall['reasoning']:,} of {visible:,}); lower `reasoning_effort` on plan/review/verify calls.")
    if prof["synth_max_prompt"] >= SYNTH_NEAR_LIMIT * max_prompt_tokens:
        out.append(f"A synthesis prompt reached {prof['synth_max_prompt']:,} tokens, "
                   f"{prof['synth_max_prompt'] / max_prompt_tokens:.0%} of max_prompt_tokens ({max_prompt_tokens:,}): "
                   "evidence fitting is close to truncating useful material.")
    if not out:
        out.append("No heuristic threshold was exceeded; token use looks balanced.")
    return out


def _fmt(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def render(prof: dict[str, Any], existing: str | None = None) -> str:
    """Markdown report; everything after an existing ``<!-- ANALYSIS -->`` marker is kept as written."""
    o = prof["overall"]
    lines = ["# Token Usage Analysis", "",
             "Generated by `python -m evaluation.token_profile` from the run bundles (spec Day 12 task 3).", "",
             "## Overview per challenge", "",
             "| Challenge | Calls | Prompt | Completion | Reasoning | Total | Report words | Tokens/word | Prompt:completion | LLM failures |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for p in prof["per_challenge"] + [dict(o, id="**All**")]:
        lines.append(f"| {p['id']} | {p['calls']} | {p['prompt']:,} | {p['completion']:,} | {p['reasoning']:,} | "
                     f"{p['total']:,} | {p['words']:,} | {_fmt(p['tokens_per_word'])} | "
                     f"{_fmt(p['prompt_completion_ratio'])} | {p['failures']} |")
    lines += ["", "## Tokens by purpose (all challenges)", "",
              "| Purpose | Calls | Prompt | Completion | Reasoning | Total | Share | Mean latency (ms) |",
              "|---|---|---|---|---|---|---|---|"]
    for name, g in o["groups"].items():
        if g["calls"]:
            lines.append(f"| {name} | {g['calls']} | {g['prompt']:,} | {g['completion']:,} | {g['reasoning']:,} | "
                         f"{g['total']:,} | {g['share']:.1%} | {g['mean_latency_ms']:,.0f} |")
    lines += ["", "## Provider / model mix", ""]
    lines += [f"- `{k}`: {v} calls" for k, v in sorted(o["provider_mix"].items(), key=lambda kv: -kv[1])] or ["- none"]
    lines += ["", "## Largest single calls", "", "| # | Challenge | Purpose | Prompt | Completion | Total | Model |",
              "|---|---|---|---|---|---|---|"]
    for i, c in enumerate(prof["top_calls"], 1):
        lines.append(f"| {i} | {c['challenge']} | {c['purpose']} | {c['prompt']:,} | {c['completion']:,} | "
                     f"{c['tokens']:,} | {c['model']} |")
    lines += ["", "## Optimisation opportunities", ""] + [f"- {t}" for t in prof["opportunities"]]
    head = "\n".join(lines) + "\n\n"
    tail = PLACEHOLDER
    if existing and MARKER in existing:
        tail = existing.split(MARKER, 1)[1].lstrip("\n")
    return f"{head}{MARKER}\n{tail}"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: profile token usage of the run bundles and write the report."""
    parser = argparse.ArgumentParser(prog="python -m evaluation.token_profile", description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", default=str(ROOT / "results" / "runs"))
    parser.add_argument("--out", default=str(ROOT / "results" / "token_usage_analysis.md"))
    args = parser.parse_args(argv)
    bundles = load_bundles(Path(args.runs))
    if not bundles:
        print(f"no bundles found in {args.runs}", file=sys.stderr)
        return 1
    out = Path(args.out)
    existing = out.read_text(encoding="utf-8") if out.exists() else None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(profile(bundles), existing), encoding="utf-8")
    print(f"{len(bundles)} bundles -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
