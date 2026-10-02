"""Repeatable token measurement for before/after comparisons (spec Day 13 tasks 4-6).

    python -m evaluation.token_compare measure results/runs_day13/baseline_mock     # 8 challenges, scripted LLM, no network
    python -m evaluation.token_compare table results/runs_day13/baseline_mock [results/runs_day13/after_mock]

``measure`` runs C1..C8 with a scripted LLM (below) and mock data in a throw-away data directory (so the live memory
store is untouched and every measurement starts from the same empty state), then saves run bundles.
``table`` prints calls and prompt/completion tokens per challenge and per call purpose from bundles; with
two directories it also prints the change. Mock numbers are deterministic, so they are exact between runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from agent import llm as llm_module
from evaluation import token_profile

ROOT = Path(__file__).resolve().parent.parent
PURPOSES = ("plan", "execute", "review", "synthesize", "verify", "reflect", "other")


# ---------------------------------------------------------------------------------------------
# Scripted LLM
#
# The stock mock provider answers "{}" to everything, so no plan step ever needs a ReAct execute call and
# the biggest cost (tool schemas in every execute call) would not show up. This provider plays a plausible
# model: a plan with two direct and two ReAct steps per company, a tool call for each ReAct step, "done" for
# every review and one short section per required section. It also counts the tool schemas in the prompt
# size, as a real provider does (the stock mock counts messages only).
# ---------------------------------------------------------------------------------------------

_ARGS = {"financial_data_api": {"statement_type": "income_statement"}, "earnings_transcript": {},
         "news_sentiment": {}, "web_search": {}}


def _plan(prompt: str) -> dict[str, Any]:
    line = re.search(r"Companies identified: (.*)", prompt)
    tickers = [t.strip() for t in (line.group(1) if line else "").split(",") if re.fullmatch(r"[A-Z.]{1,6}", t.strip())]
    steps: list[dict[str, Any]] = []
    n = 0

    def add(goal: str, tool: str, args: dict[str, Any] | None) -> None:
        nonlocal n
        n += 1
        steps.append({"id": f"s{n}", "goal": goal, "tool": tool, "args": args, "depends_on": [], "reason": goal})

    for t in tickers[:3]:
        add(f"Company facts for {t}", "company_profile", {"ticker": t})
        add(f"Income statement for {t}", "financial_data_api", None)
        add(f"Earnings call for {t}", "earnings_transcript", None)
        add(f"Latest 10-K for {t}", "sec_filing_search", {"ticker": t, "filing_type": "10-K"})
    if not tickers:
        add("Recent news", "news_sentiment", {"query": "technology sector"})
        add("Sector commentary", "web_search", {"query": "technology sector outlook"})
    return {"rationale": "scripted", "steps": steps}


class ScriptedProvider(llm_module.MockProvider):
    """Deterministic stand-in for a real model; see the comment above."""

    def _script(self, request: Any) -> Any:
        prompt = str(request.messages[-1]["content"])
        if prompt.startswith("[ARA:PLAN]"):
            return json.dumps(_plan(prompt))
        if prompt.startswith("[ARA:EXECUTE]"):
            tool = re.search(r"Preferred tool: (\w+)", prompt).group(1)  # type: ignore[union-attr]
            who = re.search(r"for ([A-Z.]{1,6})\b", prompt)
            args = dict(_ARGS.get(tool, {}))
            if who:
                args["query" if tool in ("news_sentiment", "web_search") else "ticker"] = who.group(1)
            return {"tool_calls": [{"name": tool, "arguments": args}]}
        if prompt.startswith("[ARA:REPLAN]"):
            return json.dumps({"action": "done"})
        if prompt.startswith("[ARA:SYNTHESIZE]"):
            keys = re.findall(r"^- (\w+):", prompt.split("Sections, in order:", 1)[-1], re.MULTILINE)
            return "\n".join(f"=== SECTION: {k} ===\nScripted text for {k}. [E1]" for k in keys)
        return "{}" if request.json_mode else "MOCK RESPONSE"

    def complete(self, request: Any) -> Any:
        """Complete the request with the scripted responder, then set prompt tokens to include the tool schemas."""
        self._responder = self._script
        response = super().complete(request)
        # Real providers bill the tool schemas as prompt tokens; the stock mock counts the messages only.
        text = json.dumps(request.messages, default=str) + json.dumps(request.tools or [], default=str)
        response.usage.prompt_tokens = llm_module._estimate_tokens(text)
        return response


def measure(out_dir: Path) -> int:
    """Run all eight challenges offline and save the bundles to ``out_dir``."""
    from evaluation import runner

    with tempfile.TemporaryDirectory(prefix="ara_tokens_") as tmp:
        env = {"ARA_MODE": "mock", "ARA_DATA_DIR": tmp, "ARA_CHROMA_DIR": str(Path(tmp) / "chroma"),
               "ARA_EPISODIC_DB": str(Path(tmp) / "episodic.sqlite"), "ARA_TRACES_DIR": str(Path(tmp) / "traces")}
        saved = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        stock_mock = llm_module.MockProvider
        llm_module.MockProvider = ScriptedProvider  # build_provider looks the class up at call time
        try:
            return runner.main(["--data", "mock", "--out-dir", str(out_dir), "--report-dir", "", "--pause", "0"])
        finally:
            llm_module.MockProvider = stock_mock
            for key, old in saved.items():
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old


def rows(bundles: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """``{challenge id or 'ALL': {'calls', 'prompt', 'completion', 'groups': {purpose: (calls, prompt, completion)}}}``."""
    prof = token_profile.profile(bundles)
    out: dict[str, dict[str, Any]] = {}
    for p in prof["per_challenge"] + [dict(prof["overall"], id="ALL")]:
        out[p["id"]] = {"calls": p["calls"], "prompt": p["prompt"], "completion": p["completion"] + p["reasoning"],
                        "groups": {g: (v["calls"], v["prompt"], v["completion"] + v["reasoning"])
                                   for g, v in p["groups"].items() if v["calls"]}}
    return out


def _pct(before: int, after: int) -> str:
    return "n/a" if not before else f"{(after - before) / before:+.1%}"


def render_table(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]] | None = None) -> str:
    """Markdown tables: per challenge totals, then per challenge x purpose."""
    head = "| Challenge | Calls | Prompt | Completion | Total |" if after is None else \
        "| Challenge | Calls before | Calls after | Prompt before | Prompt after | Completion before | Completion after | Total before | Total after | Total change |"
    lines = ["### Tokens per challenge", "", head, "|" + "---|" * (head.count("|") - 1)]
    for cid, b in before.items():
        if after is None:
            lines.append(f"| {cid} | {b['calls']} | {b['prompt']:,} | {b['completion']:,} | {b['prompt'] + b['completion']:,} |")
            continue
        a = after.get(cid) or {"calls": 0, "prompt": 0, "completion": 0, "groups": {}}
        tb, ta = b["prompt"] + b["completion"], a["prompt"] + a["completion"]
        lines.append(f"| {cid} | {b['calls']} | {a['calls']} | {b['prompt']:,} | {a['prompt']:,} | {b['completion']:,} | "
                     f"{a['completion']:,} | {tb:,} | {ta:,} | {_pct(tb, ta)} |")
    lines += ["", "### Calls and tokens per challenge and purpose (calls / prompt / completion)", ""]
    head = "| Challenge | Purpose | Calls | Prompt | Completion |" if after is None else \
        "| Challenge | Purpose | Calls before | Calls after | Prompt before | Prompt after | Completion before | Completion after |"
    lines += [head, "|" + "---|" * (head.count("|") - 1)]
    for cid, b in before.items():
        a = (after or {}).get(cid, {"groups": {}})
        for purpose in PURPOSES:
            bc, bp, bo = b["groups"].get(purpose, (0, 0, 0))
            ac, ap, ao = a["groups"].get(purpose, (0, 0, 0))
            if not (bc or ac):
                continue
            if after is None:
                lines.append(f"| {cid} | {purpose} | {bc} | {bp:,} | {bo:,} |")
            else:
                lines.append(f"| {cid} | {purpose} | {bc} | {ac} | {bp:,} | {ap:,} | {bo:,} | {ao:,} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point with ``measure`` (run and save bundles) and ``table`` (print the token table) subcommands."""
    parser = argparse.ArgumentParser(prog="python -m evaluation.token_compare", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure", help="run C1..C8 offline and save bundles")
    m.add_argument("out_dir")
    t = sub.add_parser("table", help="print the token table (and the change versus a second directory)")
    t.add_argument("before")
    t.add_argument("after", nargs="?")
    t.add_argument("--out", help="also write the table to this file")
    args = parser.parse_args(argv)
    if args.cmd == "measure":
        return measure(Path(args.out_dir))
    before = token_profile.load_bundles(Path(args.before))
    if not before:
        print(f"no bundles in {args.before}", file=sys.stderr)
        return 1
    after = token_profile.load_bundles(Path(args.after)) if args.after else None
    text = render_table(rows(before), rows(after) if after else None)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
