"""Shared HTTP layer for the live tools: rate limiting, a disk cache, and retries.

* **Rate limiting** per host with a token bucket. SEC EDGAR allows 10 requests per second; ARA-1 stays
  below that (``sec.max_requests_per_second``).
* **Disk cache** under ``data/cache/http`` with a per-call time-to-live, so repeated research on the
  same company costs nothing and survives restarts. This matters on free tiers and slow links.
* **Retries** for transient failures (timeouts, 429, 5xx) with exponential backoff and jitter,
  honouring ``Retry-After`` (spec A4.3). A 404 is not retried.
* **Privacy**: the SEC requires a contact email in the User-Agent, so that header goes to sec.gov
  hosts only. Every other site gets a generic User-Agent without the email.

Failures are raised as :class:`~tools.base.ToolExecutionError`, marked ``retryable`` when a later
attempt could succeed. The Day 9 error handler uses that to choose between retrying and falling back.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse

import httpx

from tools.base import ToolExecutionError

GENERIC_USER_AGENT = "Mozilla/5.0 (compatible; ARA-1 research agent)"
SEC_HOSTS = ("sec.gov",)


class RateLimiter:
    """Token bucket: at most ``rate`` requests per second on average, bursts up to ``burst``."""

    def __init__(self, rate: float, burst: int = 1, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.rate, self.burst = rate, burst
        self._tokens = float(burst)
        self._last = clock()
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()

    def acquire(self) -> float:
        """Take one token, waiting if necessary. Returns the seconds waited."""
        with self._lock:
            now = self._clock()
            self._tokens = min(self.burst, self._tokens + (now - self._last) * self.rate)
            self._last = now
            wait = 0.0 if self._tokens >= 1 else (1 - self._tokens) / self.rate
            self._tokens -= 1  # may go negative: the wait below pays it back
        if wait > 0:
            self._sleep(wait)
        return wait


class DiskCache:
    """A tiny file cache: one JSON metadata file plus one body file per key."""

    def __init__(self, directory: str | Path, *, clock: Callable[[], float] = time.time) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._clock = clock

    def _paths(self, key: str) -> tuple[Path, Path]:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.dir / f"{digest}.json", self.dir / f"{digest}.body"

    def get(self, key: str, ttl_s: float) -> bytes | None:
        """Cached body for ``key`` if present and younger than ``ttl_s`` seconds, else None."""
        meta_path, body_path = self._paths(key)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if self._clock() - meta["stored_at"] > ttl_s:
                return None
            return body_path.read_bytes()
        except (OSError, ValueError, KeyError, TypeError):  # TypeError: a damaged metadata file
            return None

    def discard(self, key: str) -> None:
        """Remove ``key`` from the cache (no error if it is absent)."""
        for path in self._paths(key):
            path.unlink(missing_ok=True)

    def put(self, key: str, body: bytes) -> None:
        """Write ``body`` and its timestamp metadata under ``key``."""
        meta_path, body_path = self._paths(key)
        body_path.write_bytes(body)
        meta_path.write_text(json.dumps({"key": key, "stored_at": self._clock(), "bytes": len(body)}), encoding="utf-8")


class HttpClient:
    """HTTP client with per-host rate limiting, retries with backoff, and an optional disk cache. ``sec_user_agent`` is the User-Agent required for SEC hosts; ``rate_limits`` maps host to requests per second; ``transport`` and ``sleep`` can be replaced in tests."""
    def __init__(
        self,
        *,
        sec_user_agent: str | None,
        cache_dir: str | Path | None = None,
        rate_limits: dict[str, float] | None = None,
        timeout_s: float = 30.0,
        max_retries: int = 3,
        base_delay_s: float = 1.0,
        max_jitter_s: float = 0.5,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.sec_user_agent = sec_user_agent
        self.cache = DiskCache(cache_dir) if cache_dir else None
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.base_delay_s, self.max_jitter_s = base_delay_s, max_jitter_s
        self._sleep = sleep
        self._limits = {host: RateLimiter(rps, burst=max(1, int(rps)), sleep=sleep)
                        for host, rps in (rate_limits or {}).items()}
        self._client = httpx.Client(transport=transport, follow_redirects=True,
                                    headers={"Accept-Encoding": "gzip, deflate"},
                                    event_hooks={"request": [self._keep_contact_email_on_sec_hosts]})
        self.requests_made = 0
        self.cache_hits = 0

    def _headers_for(self, host: str) -> dict[str, str]:
        if any(host == h or host.endswith("." + h) for h in SEC_HOSTS):
            if not self.sec_user_agent:
                raise ToolExecutionError("SEC_CONTACT_EMAIL is not set; EDGAR requires a contact email in the "
                                         "User-Agent. Add it to .env.")
            return {"User-Agent": self.sec_user_agent}
        return {"User-Agent": GENERIC_USER_AGENT}

    def _keep_contact_email_on_sec_hosts(self, request: httpx.Request) -> None:
        # A redirect re-sends the request headers: the SEC User-Agent (with the contact email) must not
        # follow a redirect to another site.
        if self.sec_user_agent and request.headers.get("User-Agent") == self.sec_user_agent:
            host = request.url.host or ""
            if not any(host == h or host.endswith("." + h) for h in SEC_HOSTS):
                request.headers["User-Agent"] = GENERIC_USER_AGENT

    def _limiter(self, host: str) -> RateLimiter | None:
        for pattern, limiter in self._limits.items():
            if host == pattern or host.endswith("." + pattern):
                return limiter
        return None

    def get(self, url: str, *, params: dict[str, Any] | None = None, ttl_s: float = 0,
            timeout_s: float | None = None) -> bytes:
        """GET ``url`` and return the body, from cache when a fresh copy exists."""
        full = f"{url}?{urlencode(params)}" if params else url
        if self.cache and ttl_s > 0:
            cached = self.cache.get(full, ttl_s)
            if cached is not None:
                self.cache_hits += 1
                return cached

        host = urlparse(full).hostname or ""
        headers = self._headers_for(host)
        limiter = self._limiter(host)
        last_error = ""
        for attempt in range(self.max_retries + 1):
            if limiter:
                limiter.acquire()
            retry_after: float | None = None
            try:
                self.requests_made += 1
                response = self._client.get(full, headers=headers, timeout=timeout_s or self.timeout_s)
            except httpx.TimeoutException:
                last_error = "timed out"
            except httpx.HTTPError as exc:
                last_error = f"connection failed ({type(exc).__name__})"
            else:
                if response.status_code == 200:
                    body = response.content
                    if self.cache and ttl_s > 0:
                        self.cache.put(full, body)
                    return body
                if response.status_code == 404:
                    raise ToolExecutionError(f"not found: {full}")
                if response.status_code in (408, 429, 500, 502, 503, 504):
                    last_error = f"HTTP {response.status_code}"
                    try:
                        retry_after = float(response.headers.get("retry-after", ""))
                    except ValueError:
                        retry_after = None
                else:
                    raise ToolExecutionError(f"HTTP {response.status_code} from {host}")
            if attempt < self.max_retries:
                delay = self.base_delay_s * 2 ** attempt + random.random() * self.max_jitter_s
                self._sleep(min(60.0, max(delay, retry_after or 0.0)))
        raise ToolExecutionError(f"{host}: {last_error} after {self.max_retries + 1} attempts", retryable=True)

    def get_json(self, url: str, **kwargs: Any) -> Any:
        """GET ``url`` and parse the response as JSON; raises a retryable ToolExecutionError on invalid JSON."""
        body = self.get(url, **kwargs)
        try:
            return json.loads(body)
        except ValueError as exc:
            # A 200 with a non-JSON body (an HTML error page) was already cached by get(); drop it so the
            # bad body is not served for the rest of its TTL.
            if self.cache:
                params = kwargs.get("params")
                self.cache.discard(f"{url}?{urlencode(params)}" if params else url)
            raise ToolExecutionError(f"invalid JSON from {url}: {exc}", retryable=True) from exc


# --------------------------------------------------------------------------------------------------
# Process-wide client
# --------------------------------------------------------------------------------------------------

_client: HttpClient | None = None
_client_lock = threading.Lock()


def get_http() -> HttpClient:
    """The shared client, configured from settings on first use."""
    global _client
    with _client_lock:
        if _client is None:
            from config.settings import load_settings

            s = load_settings()
            try:
                user_agent = s.sec.user_agent
            except ValueError:
                user_agent = None  # only SEC requests fail, with a clear message
            _client = HttpClient(
                sec_user_agent=user_agent,
                cache_dir=s.paths.data_dir / "cache" / "http",
                rate_limits={"sec.gov": s.sec.max_requests_per_second, "news.google.com": 2.0},
                max_retries=s.retry.max_retries,
                base_delay_s=s.retry.base_delay_s,
                max_jitter_s=s.retry.max_jitter_s,
            )
        return _client


def set_http(client: HttpClient | None) -> None:
    """Replace the shared client (tests inject one with a mock transport)."""
    global _client
    with _client_lock:
        _client = client
