"""Core types shared by every tool: the tool specification, the result envelope, and errors.

Every tool result is wrapped in a :class:`ToolResult` that carries provenance (which tool, which
arguments, the reliability tier of the source, when it was retrieved, whether it came from cache or a
fallback). Downstream components — the synthesis engine, the fact checker, the evaluation metrics —
rely on this envelope rather than on the raw payload, so every claim in a report can be traced back
to the call that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

ToolHandler = Callable[..., dict[str, Any]]


# --------------------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------------------


class ToolError(Exception):
    """Base class for every error raised by the tool layer."""

    #: Short machine-readable category, used in traces and by the error handler (Day 9).
    kind: str = "tool_error"
    #: Whether retrying the same call could plausibly succeed.
    retryable: bool = False

    def __init__(self, message: str, *, tool: str | None = None) -> None:
        super().__init__(message)
        self.tool = tool


class ToolNotFoundError(ToolError):
    """The requested tool name is not in the registry."""

    kind = "not_found"


class ToolLockedError(ToolError):
    """The tool exists but has not been unlocked yet (spec B4.4 progression unlocks)."""

    kind = "locked"


class ToolUnavailableError(ToolError):
    """The tool has no handler for the current run mode (e.g. no live implementation yet)."""

    kind = "unavailable"


class ToolValidationError(ToolError):
    """The arguments do not match the tool's input schema. Not retryable: the caller must fix them."""

    kind = "invalid_arguments"


class ToolOutputError(ToolError):
    """The tool returned data that does not match its output schema (a "malformed response")."""

    kind = "malformed_response"
    retryable = True


class ToolExecutionError(ToolError):
    """The tool ran but failed (API error, missing data, ...)."""

    kind = "execution_error"

    def __init__(self, message: str, *, tool: str | None = None, retryable: bool = False) -> None:
        super().__init__(message, tool=tool)
        self.retryable = retryable


# --------------------------------------------------------------------------------------------------
# Specification and result
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class UnlockRule:
    """A tool becomes available once ``after_challenge`` challenges have been completed (spec B4.4)."""

    after_challenge: int

    def is_met(self, completed_challenges: int) -> bool:
        return completed_challenges >= self.after_challenge


@dataclass
class ToolSpec:
    """Everything the registry knows about one tool.

    ``name``, ``description`` and ``parameters`` follow the OpenAI function-calling format and are
    what the LLM sees. The remaining fields are ARA-1 metadata that the LLM never sees.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler | None = None
    handler_mode: str | None = None  # which handler was bound: "mock", "live" or "any"
    output_schema: dict[str, Any] | None = None
    source_tier: int | None = None  # reliability tier of the data this tool returns (spec A6.2)
    cache_ttl_s: int = 0  # 0 disables caching
    fallbacks: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)
    unlock: UnlockRule | None = None

    def to_openai(self) -> dict[str, Any]:
        """Return the tool definition in OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    """The envelope around every successful tool call.

    ``evidence_id`` is the handle the rest of the system uses to cite this result (``[E3]``).
    """

    evidence_id: str
    tool: str
    args: dict[str, Any]
    data: dict[str, Any]
    source_tier: int | None
    retrieved_at: str  # ISO-8601 UTC
    latency_ms: float
    cache_hit: bool = False
    is_mock: bool = False
    is_fallback: bool = False


@dataclass
class CallRecord:
    """One line of the registry's call log, used for tracing and for the AB-1/AB-2 metrics."""

    tool: str
    args: dict[str, Any]
    ok: bool
    latency_ms: float
    evidence_id: str | None = None
    cache_hit: bool = False
    error_kind: str | None = None
    error_message: str | None = None
