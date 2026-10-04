"""Pretty-print an ARA-1 reasoning trace.

Usage::

    python -m agent.trace_viewer <trace.jsonl | run-bundle.json> \
        [--types decision,action,...] [--step s3] [--limit N]

A run bundle is a JSON file with a ``trace_events`` list (see ``results/runs``).
Each event is rendered with :func:`agent.tracing.format_event` and coloured by type.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.text import Text

from agent.tracing import format_event, read_trace

_STYLES = {
    "session_start": "bold", "session_end": "bold", "thought": "cyan",
    "decision": "bold green", "action": "yellow", "observation": "dim",
    "error": "red", "llm_call": "magenta", "note": "blue", "metric": "white",
}


def load_events(path: str | Path) -> list[dict[str, Any]]:
    """Load events from a ``.jsonl`` trace or from the ``trace_events`` of a bundle."""
    p = Path(path)
    if p.suffix == ".jsonl":
        return read_trace(p)
    bundle = json.loads(p.read_text(encoding="utf-8"))
    events = bundle.get("trace_events") if isinstance(bundle, dict) else None
    if not isinstance(events, list):
        raise ValueError(f"{p} has no 'trace_events' list")
    return events


def filter_events(events: Sequence[dict[str, Any]], types: Sequence[str] | None = None,
                  step: str | None = None) -> list[dict[str, Any]]:
    """Keep events whose type is in ``types`` and whose ``step_id`` equals ``step``."""
    wanted = set(types) if types else None
    return [e for e in events
            if (wanted is None or e.get("type") in wanted)
            and (step is None or e.get("step_id") == step)]


def render(events: Sequence[dict[str, Any]], console: Console | None = None) -> None:
    """Print events, one line each, coloured by event type."""
    console = console or Console()
    for event in events:
        console.print(Text(format_event(event), style=_STYLES.get(event.get("type", ""), "")))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(prog="agent.trace_viewer", description=__doc__.splitlines()[0])
    parser.add_argument("path", help="trace .jsonl or run-bundle .json")
    parser.add_argument("--types", help="comma-separated event types to keep")
    parser.add_argument("--step", help="only events with this step_id, e.g. s3")
    parser.add_argument("--limit", type=int, help="print at most N events")
    args = parser.parse_args(argv)
    try:
        events = load_events(args.path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    types = [t.strip() for t in args.types.split(",") if t.strip()] if args.types else None
    events = filter_events(events, types, args.step)
    if args.limit:
        events = events[: args.limit]
    # Redirected output on Windows is cp1252; trace text (model replies) often has characters outside it.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    render(events, Console(width=200, soft_wrap=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
