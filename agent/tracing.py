"""Tracing and logging: an append-only JSONL record of everything an agent session does.

Every event carries a sequence number, a UTC timestamp and the session id. Event types follow the
architecture spec (section 16): ``thought``, ``decision``, ``action``, ``observation``, ``error``,
``conflict``, ``llm_call``, ``metric``, plus ``session_start`` / ``session_end`` and free-form
``note``. Traces feed the Trace Gallery and several evaluation metrics, so they are machine-readable
first and human-readable second (see :func:`format_event`).

Fields whose names look like credentials are redacted before anything is written.
"""

from __future__ import annotations

import itertools
import json
import logging
import threading
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVENT_TYPES = frozenset({
    "session_start", "session_end", "thought", "decision", "action", "observation",
    "error", "conflict", "llm_call", "metric", "note",
})

_SECRET_MARKERS = ("api_key", "apikey", "authorization", "password", "secret", "token_value")
REDACTED = "[REDACTED]"

log = logging.getLogger("ara")


def new_session_id() -> str:
    """``session-YYYYMMDD-HHMMSS-xxxx``: sortable by time, unique enough for one machine."""
    now = datetime.now(UTC)
    return f"session-{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def redact(value: Any) -> Any:
    """Recursively replace values stored under credential-like keys."""
    if isinstance(value, dict):
        return {
            # Hyphens count as underscores: HTTP headers spell keys "X-Api-Key", "x-goog-api-key".
            k: REDACTED if any(m in str(k).lower().replace("-", "_") for m in _SECRET_MARKERS) else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


class TraceLogger:
    """Collects trace events in memory and, if ``traces_dir`` is given, appends them to a JSONL file.

    Thread-safe, so concurrent plan steps can log to the same session.
    """

    def __init__(self, session_id: str | None = None, traces_dir: str | Path | None = None) -> None:
        self.session_id = session_id or new_session_id()
        self.events: list[dict[str, Any]] = []
        self.path: Path | None = None
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self._fh = None
        if traces_dir is not None:
            directory = Path(traces_dir)
            directory.mkdir(parents=True, exist_ok=True)
            self.path = directory / f"{self.session_id}.jsonl"
            self._fh = self.path.open("a", encoding="utf-8")

    # ------------------------------------------------------------------ core

    def log(self, event_type: str, **fields: Any) -> dict[str, Any]:
        """Append an event of ``event_type`` to the trace with a sequence number, timestamp and ``fields``; raises ValueError for an unknown type."""
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown trace event type {event_type!r}")
        with self._lock:
            event = {
                "seq": next(self._seq),
                "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
                "session_id": self.session_id,
                "type": event_type,
            }
            event.update({k: v for k, v in redact(fields).items() if k not in event})  # fields never replace the envelope
            self.events.append(event)
            if self._fh is not None:
                self._fh.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")
                self._fh.flush()
        log.debug(format_event(event))
        return event

    def close(self) -> None:
        """Close the trace file; later events are kept in memory only."""
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None

    def __enter__(self) -> TraceLogger:
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ convenience wrappers

    def thought(self, text: str, **fields: Any) -> dict[str, Any]:
        """Log a ``thought`` event."""
        return self.log("thought", text=text, **fields)

    def decision(self, chosen: str, rationale: str, **fields: Any) -> dict[str, Any]:
        """Log a ``decision`` event recording the choice and its rationale."""
        return self.log("decision", chosen=chosen, rationale=rationale, **fields)

    def metric(self, name: str, value: Any, **fields: Any) -> dict[str, Any]:
        """Log a ``metric`` event."""
        return self.log("metric", name=name, value=value, **fields)

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        """Events of the given type, in order."""
        return [e for e in self.events if e["type"] == event_type]

    # ------------------------------------------------------------------ integrations

    def on_tool_call(self, record: Any) -> None:
        """Listener for :meth:`tools.tool_registry.ToolRegistry.add_listener`."""
        self.log("action", tool=record.tool, args=record.args)
        if record.ok:
            self.log("observation", tool=record.tool, evidence_id=record.evidence_id,
                     cache_hit=record.cache_hit, latency_ms=record.latency_ms)
        else:
            self.log("error", source="tool", tool=record.tool, error_kind=record.error_kind,
                     message=record.error_message, latency_ms=record.latency_ms)


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL trace file."""
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def iter_traces(traces_dir: str | Path) -> Iterator[Path]:
    """Trace files in a directory, oldest first."""
    yield from sorted(Path(traces_dir).glob("session-*.jsonl"))


def format_event(event: dict[str, Any]) -> str:
    """One-line human-readable rendering of an event."""
    t = event["type"]
    if t == "thought":
        body = event.get("text", "")
    elif t == "action":
        body = f"{event.get('tool')}({json.dumps(event.get('args', {}), default=str)})"
    elif t == "observation":
        body = f"{event.get('tool')} -> {event.get('evidence_id')}{' (cached)' if event.get('cache_hit') else ''}"
    elif t == "error":
        body = f"{event.get('source', '')}:{event.get('tool') or event.get('provider', '')} {event.get('error_kind')}: {event.get('message')}"
    elif t == "llm_call":
        body = (f"{event.get('provider')}/{event.get('model')} "
                f"{event.get('prompt_tokens')}+{event.get('completion_tokens')} tok, {event.get('latency_ms')} ms")
    else:
        body = json.dumps({k: v for k, v in event.items() if k not in ("seq", "ts", "session_id", "type")}, default=str)
    return f"#{event['seq']:>4} {t:<12} {body}"


def configure_logging(level: str = "INFO") -> None:
    """Console logging for the ``ara`` logger, using rich if it is installed."""
    logger = logging.getLogger("ara")
    logger.setLevel(level)
    if logger.handlers:
        return
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(show_path=False, markup=False)
        handler.setFormatter(logging.Formatter("%(message)s"))
    except ImportError:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
