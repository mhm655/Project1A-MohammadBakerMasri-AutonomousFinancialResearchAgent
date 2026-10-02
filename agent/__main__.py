"""Command line: ``python -m agent "research request" [--out report.md]``.

Examples:
    python -m agent "Create a comprehensive profile of Microsoft Corporation" --data mock
    python -m agent "..." --mode mock          # no network at all (scripted LLM, mock data)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: run one research query and write or print the Markdown report; returns the exit code."""
    parser = argparse.ArgumentParser(prog="python -m agent", description="Run an ARA-1 research task.")
    parser.add_argument("query", help="the research request")
    parser.add_argument("--out", help="write the report to this Markdown file")
    parser.add_argument("--mode", choices=["mock", "local", "free"], help="override ARA_MODE")
    parser.add_argument("--data", choices=["mock", "live"], help="override ARA_DATA_MODE")
    parser.add_argument("--max-seconds", type=float, default=300.0, help="wall-clock limit (default 300)")
    parser.add_argument("--fail-rate", type=float, help="inject tool failures at this rate (0-1), to test recovery")
    parser.add_argument("--fail-seed", type=int, help="seed for --fail-rate, for reproducible runs")
    parser.add_argument("--fail-tools", help="comma-separated tools to inject failures into (default: all network tools)")
    args = parser.parse_args(argv)

    if args.mode:
        os.environ["ARA_MODE"] = args.mode
    if args.data:
        os.environ["ARA_DATA_MODE"] = args.data
    if args.fail_rate is not None:
        os.environ["ARA_FAIL_RATE"] = str(args.fail_rate)
    if args.fail_seed is not None:
        os.environ["ARA_FAIL_SEED"] = str(args.fail_seed)
    if args.fail_tools:
        os.environ["ARA_FAIL_TOOLS"] = args.fail_tools

    from agent.core import ResearchAgent
    from agent.tracing import configure_logging
    from config.settings import load_settings

    settings = load_settings()
    configure_logging(settings.log_level)
    agent = ResearchAgent.from_settings(settings, max_seconds=args.max_seconds)
    result = agent.run(args.query)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(result.report_markdown, encoding="utf-8")
    else:
        print(result.report_markdown)

    v, u = result.verification, result.llm_usage
    print(
        f"\n--- {result.session_id}: {result.stop_reason} in {result.duration_s}s | "
        f"tool calls {result.tool_calls} | steps done {len(result.plan.by_status('done'))}/{len(result.plan.steps)} | "
        f"LLM calls {u['calls']} ({u['prompt_tokens']}+{u['completion_tokens']}+{u['reasoning_tokens']} tokens) | "
        f"claims checked {v.get('checked', 0)}, unsupported {v.get('unsupported', 0)}, uncited {v.get('uncited', 0)}"
        + (f" | report: {args.out}" if args.out else "")
        + (f" | trace: {result.trace_path}" if result.trace_path else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
