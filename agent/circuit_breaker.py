"""Per-provider circuit breakers (spec section 14.3).

There is one breaker per *provider* (for example ``sec_edgar``), not per tool, because one outage
affects every tool that talks to that provider. Without this, a dead provider would eat the whole
retry budget once per tool. An open breaker makes calls fail immediately so the caller goes straight
to the fallback chain; after a cool-down a single trial call decides whether the provider recovered.

The clock is injectable so state transitions can be tested without sleeping.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

#: Which upstream provider each tool depends on. ``None`` marks a local tool that can never be
#: "down" in the outage sense, so it is never broken or failure-injected.
TOOL_PROVIDERS: dict[str, str | None] = {
    "sec_filing_search": "sec_edgar",
    "financial_data_api": "sec_edgar",
    "company_profile": "sec_edgar",
    "earnings_transcript": "transcripts",
    "peer_comparison": "yfinance",
    "web_search": "web_search",
    "news_sentiment": "google_news",
    "calculation_engine": None,
    "fact_checker": None,
    "report_generator": None,
    "vector_db_search": None,
    "vector_db_store": None,
}

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


def provider_for(tool: str) -> str | None:
    """The provider behind ``tool``, or ``None`` for local (and unknown) tools."""
    return TOOL_PROVIDERS.get(tool)


@dataclass
class BreakerConfig:
    """Thresholds for a circuit breaker: trip on ``consecutive_failures`` in a row or when the failure ratio over the last ``window`` calls reaches ``window_failure_ratio``; ``cooldown_s`` before a trial call; ``max_open_providers`` open at once marks the system degraded."""
    consecutive_failures: int = 3
    window: int = 6
    window_failure_ratio: float = 0.5
    cooldown_s: float = 30.0
    #: Global guard: this many open providers at once means the system as a whole is degraded.
    max_open_providers: int = 3


class CircuitBreaker:
    """Closed -> open on repeated failure; open -> half-open after the cool-down; one trial decides."""

    def __init__(self, name: str, config: BreakerConfig | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.name = name
        self.config = config or BreakerConfig()
        self._clock = clock
        self._lock = threading.RLock()  # one breaker is shared by every worker thread
        self._state = CLOSED
        self._consecutive = 0
        self._window: deque[bool] = deque(maxlen=self.config.window)  # True = the call failed
        self._trial_pending = False
        self.opened_at: float | None = None
        self.trips = 0

    @property
    def state(self) -> str:
        """Current state; reports ``half_open`` as soon as the cool-down has elapsed."""
        with self._lock:
            self._refresh()
            return self._state

    def allow(self) -> bool:
        """May a call go out now? In half-open state exactly one trial call is allowed at a time."""
        with self._lock:
            self._refresh()
            if self._state == CLOSED:
                return True
            if self._state == HALF_OPEN and not self._trial_pending:
                self._trial_pending = True
                return True
            return False

    def record_success(self) -> None:
        """Record a successful call: closes a half-open breaker, or resets the failure streak when closed."""
        with self._lock:
            self._refresh()
            if self._state == HALF_OPEN:
                self._close()
            elif self._state == CLOSED:
                self._consecutive = 0
                self._window.append(False)

    def record_failure(self) -> bool:
        """Record a provider failure; returns True if this very failure tripped the breaker."""
        with self._lock:
            self._refresh()
            if self._state == HALF_OPEN:
                self._trip()
                return True
            if self._state == CLOSED:
                self._consecutive += 1
                self._window.append(True)
                failures = sum(self._window)
                window_full = len(self._window) >= max(1, self.config.window)  # window=0 keeps no history
                if (
                    self._consecutive >= self.config.consecutive_failures
                    or (window_full and failures / len(self._window) >= self.config.window_failure_ratio)
                ):
                    self._trip()
                    return True
            return False

    def release(self) -> None:
        """Give back a half-open trial that ended without evidence either way.

        Bad arguments or a "no such company" answer say nothing about provider health, but the trial
        slot was consumed by ``allow()``; without this the breaker would refuse calls forever.
        """
        with self._lock:
            self._trial_pending = False

    def _refresh(self) -> None:
        if (
            self._state == OPEN
            and self.opened_at is not None
            and self._clock() - self.opened_at >= self.config.cooldown_s
        ):
            self._state = HALF_OPEN
            self._trial_pending = False

    def _trip(self) -> None:
        self._state = OPEN
        self.opened_at = self._clock()
        self.trips += 1
        self._trial_pending = False

    def _close(self) -> None:
        self._state = CLOSED
        self._consecutive = 0
        self._window.clear()
        self._trial_pending = False


class BreakerBoard:
    """All breakers for a run, keyed by provider."""

    def __init__(self, config: BreakerConfig | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.config = config or BreakerConfig()
        self._clock = clock
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()
        for provider in sorted({p for p in TOOL_PROVIDERS.values() if p}):
            self.for_provider(provider)

    def for_provider(self, name: str) -> CircuitBreaker:
        """The breaker for provider ``name``, created on first use."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, self.config, self._clock)
            return self._breakers[name]

    def for_tool(self, tool: str) -> CircuitBreaker | None:
        """The breaker for the provider behind ``tool``, or ``None`` for local tools."""
        provider = provider_for(tool)
        return self.for_provider(provider) if provider else None

    def open_providers(self) -> list[str]:
        """Sorted names of the providers whose breaker is currently open."""
        with self._lock:
            breakers = list(self._breakers.items())
        return sorted(name for name, breaker in breakers if breaker.state == OPEN)

    def system_degraded(self) -> bool:
        """Several providers failing at once: stop external research and write with what we have."""
        return len(self.open_providers()) >= self.config.max_open_providers

    def snapshot(self) -> dict[str, str]:
        """Map of provider name to current breaker state."""
        with self._lock:
            breakers = sorted(self._breakers.items())
        return {name: breaker.state for name, breaker in breakers}
