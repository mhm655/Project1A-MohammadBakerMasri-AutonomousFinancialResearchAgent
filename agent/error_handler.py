"""Error classification, retry policy and the resilient dispatcher (spec section 14).

The tool layer already retries HTTP 429/5xx inside ``HttpClient``; what reaches this module has
survived that. So the retries here are deliberately modest and error-class specific, and they sit
behind a per-provider circuit breaker so one dead provider cannot consume the whole retry budget.
When a tool cannot deliver, the dispatcher walks its fallback chain and, if everything fails, raises
:class:`AllFallbacksFailed` carrying a :class:`Gap`, so the report can say what is missing and why
instead of inventing content.

Sleep, clock and random source are all injectable to keep tests deterministic.
"""

from __future__ import annotations

import dataclasses
import math
import random
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agent.circuit_breaker import BreakerBoard, CircuitBreaker, provider_for
from agent.fallback_chains import FALLBACK_CHAINS, Fallback
from tools.base import (
    ToolError,
    ToolExecutionError,
    ToolLockedError,
    ToolNotFoundError,
    ToolOutputError,
    ToolResult,
    ToolUnavailableError,
    ToolValidationError,
)

# --------------------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------------------

TRANSIENT = "transient"
RATE_LIMITED = "rate_limited"
AUTH = "auth"
MALFORMED = "malformed"
TIMEOUT = "timeout"
INVALID_ARGS = "invalid_args"
NOT_FOUND = "not_found"
UNAVAILABLE = "unavailable"
PERMANENT = "permanent"

ERROR_CLASSES = (TRANSIENT, RATE_LIMITED, AUTH, MALFORMED, TIMEOUT, INVALID_ARGS, NOT_FOUND, UNAVAILABLE,
                 PERMANENT)

#: Classes that say the provider itself is unhealthy. Bad input or absent data does not.
BREAKER_COUNTED = frozenset({TRANSIENT, RATE_LIMITED, TIMEOUT, MALFORMED, AUTH})

_TIMEOUT = re.compile(r"timed out|timeout", re.I)
_RATE_LIMITED = re.compile(r"HTTP 429|rate.?limit|too many requests", re.I)
_AUTH = re.compile(r"HTTP 40[13]\b|unauthori[sz]ed|forbidden", re.I)
_HTTP_STATUS = re.compile(r"\bHTTP (\d{3})\b", re.I)
_NOT_FOUND = re.compile(
    r"HTTP 404\b|not found|no EDGAR filer|unknown ticker|\bno (?:\w+ )?data\b(?!-)|\bno .{1,40}? found\b", re.I
)
_SERVER = re.compile(
    r"HTTP 5\d\d\b|connection (?:reset|refused|aborted|closed|error|failed|timed)|connect(?:ion)? error"
    r"|unreachable|temporarily unavailable", re.I)


def classify(exc: BaseException) -> str:
    """Map an exception to one of the section 14.1 error classes.

    Exception type wins over message text; message patterns matter for ``ToolExecutionError``, whose
    text is the only place ``HttpClient`` and the handlers say what went wrong.
    """
    if isinstance(exc, ToolValidationError):
        return INVALID_ARGS
    if isinstance(exc, ToolOutputError):
        return MALFORMED
    if isinstance(exc, (ToolUnavailableError, ToolLockedError)):
        return UNAVAILABLE
    if isinstance(exc, ToolNotFoundError):  # an unknown tool *name*: a bug, not missing data
        return PERMANENT
    if isinstance(exc, TimeoutError):
        return TIMEOUT
    if isinstance(exc, ConnectionError):
        return TRANSIENT

    message = str(exc)
    if _TIMEOUT.search(message):
        return TIMEOUT
    status = _HTTP_STATUS.search(message)  # an explicit status code beats loose wording elsewhere
    if status:
        code = int(status.group(1))
        if code == 429:
            return RATE_LIMITED
        if code in (401, 403):
            return AUTH
        if code == 404:
            return NOT_FOUND
        if 500 <= code < 600:
            return TRANSIENT
    if _RATE_LIMITED.search(message):
        return RATE_LIMITED
    if _AUTH.search(message):
        return AUTH
    if _NOT_FOUND.search(message):
        return NOT_FOUND
    if _SERVER.search(message):
        return TRANSIENT
    if getattr(exc, "retryable", False):  # e.g. HttpClient's "... after N attempts"
        return TRANSIENT
    return PERMANENT


# --------------------------------------------------------------------------------------------------
# Retry policy
# --------------------------------------------------------------------------------------------------


@dataclass
class RetryPolicy:
    """Exponential backoff (spec 14.2) with a per-class retry allowance."""

    base_delay_s: float = 1.0
    factor: float = 2.0
    max_retries: int = 5
    max_jitter_s: float = 0.5
    #: No single wait may exceed this, whatever the backoff or a server's Retry-After says.
    max_delay_s: float = 60.0

    def retries_for(self, error_class: str) -> int:
        """Maximum retries for ``error_class``: ``max_retries`` for transient and rate-limited errors, 1 for timeouts and malformed output, otherwise 0."""
        if error_class in (TRANSIENT, RATE_LIMITED):
            return self.max_retries
        if error_class in (TIMEOUT, MALFORMED):
            return 1
        return 0

    def delay(self, attempt: int, rng: random.Random | None = None) -> float:
        """Seconds to wait before retry number ``attempt`` (0-based): base * factor**attempt + jitter."""
        jitter = (rng or random).uniform(0, self.max_jitter_s)
        try:
            raw = self.base_delay_s * self.factor**attempt + jitter
        except OverflowError:
            raw = self.max_delay_s
        return self.clamp(raw)

    def clamp(self, delay: float) -> float:
        """Bound a wait to ``[0, max_delay_s]`` (NaN counts as 0)."""
        if math.isnan(delay):
            return 0.0
        return max(0.0, min(delay, self.max_delay_s))


# --------------------------------------------------------------------------------------------------
# Gaps and attempts
# --------------------------------------------------------------------------------------------------

SUGGESTED_SOURCES: dict[str, str] = {
    "sec_filing_search": "SEC EDGAR manually: https://www.sec.gov/edgar/search/",
    "financial_data_api": "SEC EDGAR manually: https://www.sec.gov/edgar/search/",
    "company_profile": "the company's investor-relations website, or SEC EDGAR manually: "
                       "https://www.sec.gov/edgar/search/",
    "earnings_transcript": "the company's investor-relations website (earnings call transcripts and webcasts)",
    "peer_comparison": "Yahoo Finance manually: https://finance.yahoo.com/",
    "web_search": "a web search engine manually: https://duckduckgo.com/",
    "news_sentiment": "Google News manually: https://news.google.com/",
    "vector_db_search": "the original sources (SEC EDGAR: https://www.sec.gov/edgar/search/)",
}
_DEFAULT_SOURCE = "the original primary sources"

_LABELS = {
    "sec_filing_search": "SEC filings",
    "financial_data_api": "Financial statements",
    "company_profile": "The company profile",
    "earnings_transcript": "The earnings call transcript",
    "peer_comparison": "The peer comparison",
    "web_search": "Web search results",
    "news_sentiment": "News sentiment",
    "vector_db_search": "Stored research findings",
}


@dataclass
class Gap:
    """A piece of information the agent could not obtain (spec 14.4)."""

    section: str
    missing_info: str
    cause: str
    attempted_tools: list[str] = field(default_factory=list)
    suggested_manual_source: str = ""

    def to_text(self) -> str:
        """One readable sentence for the report's "Data Gaps and Limitations" section."""
        head = self.missing_info.split(" for ")[0].rstrip()
        verb = "are" if head.endswith("s") and not head.endswith("ss") else "is"
        text = f"{self.missing_info} {verb} missing: {self.cause}"
        if self.attempted_tools:
            text += f"; tried {', '.join(self.attempted_tools)}"
        text += "."
        if self.suggested_manual_source:
            text += f" Check {self.suggested_manual_source}"
        if self.section:
            text = f"{self.section}: {text}"
        return text


@dataclass
class Attempt:
    """One dispatch attempt (successful or not), for traces and gap reports."""

    tool: str
    args: dict[str, Any]
    error_class: str | None
    message: str
    fallback: bool
    elapsed_s: float


class AllFallbacksFailed(ToolExecutionError):
    """The primary tool and every fallback failed; ``gap`` says what is missing."""

    kind = "all_fallbacks_failed"

    def __init__(self, message: str, *, gap: Gap, attempts: list[Attempt], tool: str | None = None) -> None:
        super().__init__(message, tool=tool, retryable=False)
        self.gap = gap
        self.attempts = attempts


# --------------------------------------------------------------------------------------------------
# Resilient dispatcher
# --------------------------------------------------------------------------------------------------

Dispatch = Callable[[str, dict[str, Any]], ToolResult]


@dataclass
class _Failure:
    error_class: str
    exc: ToolError | None
    message: str


class ResilientDispatcher:
    """Wraps a ``dispatch(name, args)`` callable with retries, breakers, fallbacks and gap reporting."""

    def __init__(
        self,
        dispatch: Dispatch,
        *,
        breakers: BreakerBoard,
        policy: RetryPolicy,
        chains: Mapping[str, list[Fallback]] = FALLBACK_CHAINS,
        validate: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        deadline: Callable[[], bool] | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._dispatch = dispatch
        self.breakers = breakers
        self.policy = policy
        self.chains = chains
        self._validate = validate
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._deadline = deadline
        self._on_event = on_event
        self._clock = clock
        #: Per-thread attempts of the most recent ``call`` (also carried by AllFallbacksFailed).
        self._local = threading.local()

    @property
    def last_attempts(self) -> list[Attempt]:
        """Attempts made by the calling thread's most recent ``call`` (calls run on worker threads)."""
        return getattr(self._local, "attempts", [])

    def call(self, tool: str, args: dict[str, Any]) -> ToolResult:
        """Call ``tool`` with retries and the fallback chain; raises ``AllFallbacksFailed`` when nothing succeeds, or the original error for invalid arguments."""
        attempts: list[Attempt] = []
        self._local.attempts = attempts

        if self._validate is not None:
            try:
                args = self._validate(tool, args)
            except ToolError as exc:  # bad arguments: the caller must fix them, no fallback can help
                logged = dict(args) if isinstance(args, dict) else {"_raw": args}  # models may send a JSON string
                attempts.append(Attempt(tool, logged, classify(exc), str(exc)[:300], False, 0.0))
                raise

        outcome = self._attempt(tool, args, attempts, fallback=False)
        if isinstance(outcome, ToolResult):
            return outcome
        if outcome.error_class == INVALID_ARGS and outcome.exc is not None:
            raise outcome.exc
        primary = outcome

        for fb in self.chains.get(tool, []):
            fb_args = fb.map_args(args)
            if fb_args is None:
                continue
            result = self._attempt(fb.tool, fb_args, attempts, fallback=True)
            if isinstance(result, ToolResult):
                self._emit("fallback_used", {"tool": tool, "fallback": fb.tool,
                                             "error_class": primary.error_class})
                return self._mark_fallback(result, tool, primary.error_class, fb, primary.message)
            self._emit("fallback_failed", {"tool": tool, "fallback": fb.tool,
                                           "error_class": result.error_class, "message": result.message})

        gap = self._build_gap(tool, args, primary, attempts)
        self._emit("gap", {"tool": tool, "gap": gap.to_text()})
        raise AllFallbacksFailed(gap.to_text(), gap=gap, attempts=attempts, tool=tool)

    # ------------------------------------------------------------------ one tool, with retries

    def _attempt(self, tool: str, args: dict[str, Any], attempts: list[Attempt], *,
                 fallback: bool) -> ToolResult | _Failure:
        breaker = self.breakers.for_tool(tool)
        provider = provider_for(tool)
        retry = 0
        last: _Failure | None = None
        while True:
            if breaker is not None and not breaker.allow():
                self._emit("breaker_open", {"tool": tool, "provider": provider, "reason": "refused"})
                if last is None:  # never got a real answer from this provider
                    message = f"circuit open for {provider}"
                    attempts.append(Attempt(tool, args, UNAVAILABLE, message, fallback, 0.0))
                    return _Failure(UNAVAILABLE, None, message)
                return last

            started = self._clock()
            try:
                result = self._dispatch(tool, args)
            except ToolError as exc:
                error_class = classify(exc)
                message = str(exc)[:300]
                attempts.append(Attempt(tool, args, error_class, message, fallback, self._clock() - started))
                last = _Failure(error_class, exc, message)
                self._record(breaker, error_class, tool, provider)
                if not self._may_retry(breaker, error_class, retry):
                    return last
                delay = self._retry_delay(exc, error_class, retry)
                self._emit("retry", {"tool": tool, "attempt": retry + 1, "error_class": error_class,
                                     "delay_s": delay})
                self._sleep(delay)
                retry += 1
                continue
            except BaseException:  # a bug or interrupt: free a half-open trial slot, then propagate
                if breaker is not None:
                    breaker.release()
                raise

            if breaker is not None:
                breaker.record_success()
            attempts.append(Attempt(tool, args, None, "ok", fallback, self._clock() - started))
            return result

    def _record(self, breaker: CircuitBreaker | None, error_class: str, tool: str, provider: str | None) -> None:
        if breaker is None:
            return
        if error_class in BREAKER_COUNTED:
            if breaker.record_failure():
                self._emit("breaker_open", {"tool": tool, "provider": provider, "reason": "tripped"})
        else:
            breaker.release()  # no evidence about provider health; free a half-open trial slot

    def _may_retry(self, breaker: CircuitBreaker | None, error_class: str, retry: int) -> bool:
        if retry >= self.policy.retries_for(error_class):
            return False
        if self._deadline is not None and self._deadline():
            return False
        return breaker is None or breaker.state != "open"

    def _retry_delay(self, exc: BaseException, error_class: str, retry: int) -> float:
        delay = self.policy.delay(retry, self._rng)
        if error_class == RATE_LIMITED:
            try:
                delay = max(delay, float(getattr(exc, "retry_after", None) or 0.0))
            except (TypeError, ValueError):
                pass
        return self.policy.clamp(delay)

    # ------------------------------------------------------------------ results and gaps

    @staticmethod
    def _mark_fallback(result: ToolResult, primary: str, error_class: str, fb: Fallback,
                       message: str = "") -> ToolResult:
        # Copy: the registry may hand back its cached object, which other callers still hold.
        data = dict(result.data)
        notes = data.get("notes")
        notes = list(notes) if isinstance(notes, list) else ([notes] if notes else [])
        reason = f"{error_class}: {message[:160]}" if message else error_class
        notes.append(f"Fallback: {primary} failed ({reason}); data from {fb.tool}.")
        if fb.note:
            notes.append(f"Fallback data is {fb.note}.")
        data["notes"] = notes
        data["fallback_for"] = primary
        return dataclasses.replace(result, data=data, is_fallback=True)

    @staticmethod
    def _build_gap(tool: str, args: dict[str, Any], primary: _Failure, attempts: list[Attempt]) -> Gap:
        subject = args.get("ticker") or (f'"{args["query"]}"' if args.get("query") else None)
        label = _LABELS.get(tool, f"Data from {tool}")
        missing = f"{label} for {subject}" if subject else label
        provider = provider_for(tool) or tool
        causes = {
            UNAVAILABLE: f"{provider} is unavailable (circuit open)",
            TRANSIENT: f"{provider} is failing (server or connection errors)",
            RATE_LIMITED: f"{provider} is rate limiting requests",
            TIMEOUT: f"{provider} timed out",
            AUTH: f"{provider} rejected our credentials",
            MALFORMED: f"{provider} returned malformed data",
            NOT_FOUND: "no data exists for this request",
        }
        if primary.error_class == UNAVAILABLE and primary.exc is not None:  # no handler / locked, not a breaker
            causes[UNAVAILABLE] = f"{tool} is unavailable ({primary.message[:120]})"
        cause = causes.get(primary.error_class, f"{tool} failed ({primary.error_class}): {primary.message[:120]}")
        tried = list(dict.fromkeys(a.tool for a in attempts))
        return Gap(
            section="",
            missing_info=missing,
            cause=cause,
            attempted_tools=tried,
            suggested_manual_source=SUGGESTED_SOURCES.get(tool, _DEFAULT_SOURCE),
        )

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self._on_event is not None:
            self._on_event(kind, payload)


# --------------------------------------------------------------------------------------------------
# Failure injection (Challenge 8)
# --------------------------------------------------------------------------------------------------


class FailureInjector:
    """Raises simulated provider failures at a seeded rate.

    Installed *below* the retry and breaker layers (inside the registry's handler call) so injected
    failures exercise the real recovery path. Local tools are never injected.
    """

    def __init__(self, rate: float = 0.0, per_tool: dict[str, float] | None = None,
                 kinds: tuple[str, ...] = ("timeout", "http_503", "malformed"), seed: int | None = None) -> None:
        self.rate = rate
        self.per_tool = dict(per_tool or {})
        self.kinds = tuple(kinds)
        self._rng = random.Random(seed)
        self.injected: dict[str, int] = {}
        self._lock = threading.Lock()  # the registry runs handlers on several worker threads

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> FailureInjector:
        """Build from ``ARA_FAIL_RATE`` (float) and ``ARA_FAIL_SEED`` (int); both optional."""
        raw_rate = environ.get("ARA_FAIL_RATE", "").strip()
        raw_seed = environ.get("ARA_FAIL_SEED", "").strip()
        try:
            rate = float(raw_rate) if raw_rate else 0.0
            seed = int(raw_seed) if raw_seed else None
        except ValueError as exc:
            raise ValueError(f"ARA_FAIL_RATE must be a number and ARA_FAIL_SEED an integer ({exc})") from exc
        return cls(rate=rate, seed=seed)

    def maybe_fail(self, tool: str) -> None:
        """Raise a simulated failure (timeout, HTTP 503 or malformed output) for network tools at the configured rate; local tools never fail."""
        if provider_for(tool) is None:
            return
        rate = self.per_tool.get(tool, self.rate)
        if rate <= 0:
            return
        with self._lock:
            if self._rng.random() >= rate:
                return
            kind = self._rng.choice(self.kinds)
            self.injected[tool] = self.injected.get(tool, 0) + 1
        if kind == "timeout":
            raise ToolExecutionError(f"{tool}: request timed out (injected)", tool=tool, retryable=True)
        if kind == "http_503":
            raise ToolExecutionError(f"HTTP 503 from injected failure for {tool}", tool=tool, retryable=True)
        raise ToolOutputError(f"{tool}: response malformed (injected)", tool=tool)
