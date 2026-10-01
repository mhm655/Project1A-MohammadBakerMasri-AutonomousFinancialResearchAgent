"""The LLM layer: one client over a fallback chain of free providers.

``LLMClient.complete`` sends a chat request to the first provider in the chain. Transient failures
(rate limits, timeouts, connection errors, 5xx) are retried with exponential backoff and jitter;
permanent failures (bad key, unknown model, rejected request) and exhausted retries move on to the
next provider. Every attempt is recorded in the trace, and token usage is tracked per provider/model.

Messages and tool definitions use the OpenAI chat format throughout. Gemini, Groq and Ollama all
expose OpenAI-compatible endpoints, so one adapter (:class:`OpenAICompatibleProvider`, built on the
official ``openai`` SDK) serves all three; only the base URL, key and model differ. An Anthropic
adapter is included for reviewers who have a key, and :class:`MockProvider` serves tests.
"""

from __future__ import annotations

import copy
import json
import random
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

from config.settings import ConfigError, LLMSettings, ProviderSettings, Settings

# --------------------------------------------------------------------------------------------------
# Data types
# --------------------------------------------------------------------------------------------------


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, exactly as the model produced it; the tool registry validates it
    #: Provider-specific fields that must be sent back unchanged on the next turn. Gemini 3 puts a
    #: ``thought_signature`` here and rejects the follow-up request if it is missing.
    extra: dict[str, Any] | None = None


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Hidden "thinking" tokens. Billed and counted against quotas, but not part of the visible reply.
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.reasoning_tokens


@dataclass
class LLMRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None  # "auto", "required" or "none"
    json_mode: bool = False
    temperature: float = 0.0
    max_tokens: int = 4096
    #: "none", "minimal", "low", "medium" or "high"; None = the provider's configured default.
    reasoning_effort: str | None = None


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall]
    provider: str
    model: str
    usage: TokenUsage
    latency_ms: float
    finish_reason: str | None = None
    #: How many providers failed before this one answered (0 = the primary provider).
    fallback_depth: int = 0
    #: Errors from earlier attempts during this call, oldest first.
    errors: list[LLMProviderError] = field(default_factory=list)


class LLMError(Exception):
    """Base class for LLM-layer errors."""


class LLMProviderError(LLMError):
    """One failed attempt against one provider."""

    RETRYABLE_KINDS = frozenset({"rate_limit", "timeout", "connection", "server", "bad_response"})

    def __init__(self, provider: str, kind: str, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(f"[{provider}] {kind}: {message}")
        self.provider = provider
        self.kind = kind
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.kind in self.RETRYABLE_KINDS


class AllProvidersFailedError(LLMError):
    """Every provider in the chain failed. ``errors`` holds every attempt, oldest first."""

    def __init__(self, errors: list[LLMProviderError]) -> None:
        summary = "; ".join(str(e) for e in errors[-4:]) or "no providers configured"
        super().__init__(f"All LLM providers failed ({len(errors)} attempts): {summary}")
        self.errors = errors


class Provider(Protocol):
    name: str
    model: str

    def complete(self, request: LLMRequest) -> LLMResponse: ...


# --------------------------------------------------------------------------------------------------
# Error classification and backoff
# --------------------------------------------------------------------------------------------------


def classify_exception(provider: str, exc: BaseException) -> LLMProviderError:
    """Map an SDK exception (openai or anthropic, which share a design) to an LLMProviderError."""
    if isinstance(exc, LLMProviderError):
        return exc
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    message = str(exc)[:500]
    if "Timeout" in name:
        return LLMProviderError(provider, "timeout", message)
    if "Connection" in name:
        return LLMProviderError(provider, "connection", message)
    if status == 429:
        lowered = message.lower()
        if "request too large" in lowered:  # larger than the per-minute allowance: retrying cannot help
            return LLMProviderError(provider, "request_too_large", message)
        # Gemini says "exceeded your current quota" for per-minute limits too; its quota id tells them apart.
        per_minute = any(marker in lowered.replace(" ", "") for marker in _PER_MINUTE_MARKERS)
        if not per_minute and any(marker in lowered for marker in _DAILY_QUOTA_MARKERS):
            return LLMProviderError(provider, "quota_exhausted", message)
        return LLMProviderError(provider, "rate_limit", message, retry_after=_retry_after(exc))
    if status in (401, 403):
        return LLMProviderError(provider, "auth", message)
    if status == 404:
        return LLMProviderError(provider, "not_found", message)
    if status in (400, 409, 413, 422):
        return LLMProviderError(provider, "bad_request", message)
    if isinstance(status, int) and status >= 500:
        return LLMProviderError(provider, "server", message)
    return LLMProviderError(provider, "other", f"{name}: {message}")


# Phrases that mean a daily (not per-minute) quota is used up; it will not clear by waiting seconds.
_DAILY_QUOTA_MARKERS = ("tokens per day", "requests per day", "(tpd)", "(rpd)", "exceeded your current quota",
                        "per day")
# Quota ids and phrases of per-minute limits (spaces removed before matching): these clear within a minute.
_PER_MINUTE_MARKERS = ("perminute", "(tpm)", "(rpm)")


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    try:
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def backoff_delay(
    attempt: int,
    *,
    base_s: float = 1.0,
    max_jitter_s: float = 0.5,
    retry_after: float | None = None,
    cap_s: float = 60.0,
    rng: Callable[[], float] = random.random,
) -> float:
    """Exponential backoff with jitter (spec A4.3): base x 2^attempt + U(0, jitter), capped.

    A server-provided ``Retry-After`` is honoured when it asks for a longer wait.
    """
    delay = base_s * (2 ** attempt) + rng() * max_jitter_s
    if retry_after is not None:
        delay = max(delay, retry_after)
    return min(delay, cap_s)


# --------------------------------------------------------------------------------------------------
# Tool-schema dialects
# --------------------------------------------------------------------------------------------------

# JSON Schema keywords Gemini's function declarations do not accept.
_GEMINI_UNSUPPORTED = frozenset({"additionalProperties", "uniqueItems", "$schema", "$id", "examples", "const"})


def adapt_tools(tools: list[dict[str, Any]] | None, dialect: str) -> list[dict[str, Any]] | None:
    """Return tool definitions adapted to a provider's schema dialect (never mutates the input)."""
    if not tools:
        return None
    adapted = copy.deepcopy(tools)
    if dialect == "gemini":
        for tool in adapted:
            fn = tool["function"]
            fn["parameters"] = _gemini_schema(fn["parameters"], top_level=True)
    return adapted


def _gemini_schema(node: Any, top_level: bool = False) -> Any:
    if not isinstance(node, dict):
        return node
    if node.get("type") == "object" and not node.get("properties") and not top_level:
        # Gemini requires OBJECT schemas to list their properties. Free-form objects are sent as
        # JSON-encoded strings instead; ToolRegistry.validate_args decodes them again.
        description = node.get("description", "").rstrip()
        hint = "Pass a JSON object encoded as a string."
        return {"type": "string", "description": f"{description} {hint}".strip()}
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key in _GEMINI_UNSUPPORTED:
            continue
        if key == "properties":
            out[key] = {name: _gemini_schema(sub) for name, sub in value.items()}
        elif key == "items":
            out[key] = _gemini_schema(value)
        else:
            out[key] = value
    return out


# --------------------------------------------------------------------------------------------------
# Message helpers (OpenAI chat format)
# --------------------------------------------------------------------------------------------------


def assistant_message(response: LLMResponse) -> dict[str, Any]:
    """The assistant turn to append to the conversation after a response."""
    message: dict[str, Any] = {"role": "assistant", "content": response.content}
    if response.tool_calls:
        message["tool_calls"] = [
            {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": tc.arguments},
             **(tc.extra or {})}  # e.g. Gemini's thought signature, echoed back verbatim
            for tc in response.tool_calls
        ]
    return message


def tool_result_message(tool_call_id: str, content: Any) -> dict[str, Any]:
    """A tool-result turn answering the tool call with ``tool_call_id``."""
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    return {"role": "tool", "tool_call_id": tool_call_id, "content": text}


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _usage_from_openai(usage: Any) -> TokenUsage:
    """Token usage, including hidden reasoning tokens.

    Providers report reasoning differently: in ``completion_tokens_details.reasoning_tokens``, or
    (Gemini) only as the gap between ``total_tokens`` and prompt + completion. Both are handled.
    """
    if usage is None:
        return TokenUsage()
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", 0) or 0
    if reasoning:
        completion = max(0, completion - reasoning)  # some providers include reasoning in completion_tokens
    else:
        reasoning = max(0, (getattr(usage, "total_tokens", 0) or 0) - prompt - completion)
    return TokenUsage(prompt, completion, reasoning)


# --------------------------------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------------------------------


class OpenAICompatibleProvider:
    """Gemini, Groq, Ollama (and OpenAI itself) through the official ``openai`` SDK."""

    def __init__(self, settings: ProviderSettings, *, timeout_s: float = 60.0, client: Any = None) -> None:
        self.name = settings.name
        self.model = settings.model
        self.dialect = settings.schema_dialect
        # Only sent to providers configured with a default, since others reject the parameter.
        self.reasoning_effort = settings.reasoning_effort or None
        # Some free tiers reject any request asking for more output than their per-minute allowance.
        self.max_output_tokens = settings.max_output_tokens or None
        self._uses_openai_api = "api.openai.com" in (settings.base_url or "")
        #: A local server (Ollama) that refuses connections is not running; it will not appear in seconds.
        self.is_local = any(h in (settings.base_url or "") for h in ("localhost", "127.0.0.1", "[::1]"))
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=settings.api_key or "not-needed",  # Ollama ignores the key but the SDK requires one
                base_url=settings.base_url or None,
                timeout=timeout_s,
                max_retries=0,  # retries are handled by LLMClient so every attempt is traced
            )
        self._client = client

    def list_models(self) -> list[str]:
        """Model ids the server offers. A listed model may still be retired for new keys (Gemini)."""
        return [m.id.removeprefix("models/") for m in self._client.models.list()]

    def complete(self, request: LLMRequest) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": request.messages,
            "temperature": request.temperature,
        }
        # OpenAI's own API has renamed the parameter; compatible servers still use max_tokens.
        max_tokens = min(request.max_tokens, self.max_output_tokens) if self.max_output_tokens else request.max_tokens
        kwargs["max_completion_tokens" if self._uses_openai_api else "max_tokens"] = max_tokens
        tools = adapt_tools(request.tools, self.dialect)
        if tools:
            kwargs["tools"] = tools
            if request.tool_choice:
                kwargs["tool_choice"] = request.tool_choice
        if request.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = request.reasoning_effort or self.reasoning_effort

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise classify_exception(self.name, exc) from exc
        latency = round((time.perf_counter() - started) * 1000, 1)

        if not getattr(response, "choices", None):
            raise LLMProviderError(self.name, "bad_response", "response contained no choices")
        choice = response.choices[0]
        message = choice.message
        tool_calls = [
            ToolCall(
                id=tc.id or f"call_{uuid.uuid4().hex[:12]}",
                name=tc.function.name,
                arguments=tc.function.arguments or "{}",
                extra=dict(tc.model_extra) if getattr(tc, "model_extra", None) else None,
            )
            for tc in (message.tool_calls or [])
        ]
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            provider=self.name,
            model=getattr(response, "model", None) or self.model,
            usage=_usage_from_openai(getattr(response, "usage", None)),
            latency_ms=latency,
            finish_reason=choice.finish_reason,
        )


class AnthropicProvider:
    """Optional adapter for Anthropic's Messages API (paid; not in the default chain)."""

    _TOOL_CHOICE = {"auto": {"type": "auto"}, "required": {"type": "any"}, "none": {"type": "none"}}

    def __init__(self, settings: ProviderSettings, *, timeout_s: float = 60.0, client: Any = None) -> None:
        self.name = settings.name
        self.model = settings.model
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=settings.api_key, timeout=timeout_s, max_retries=0)
        self._client = client

    @staticmethod
    def convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
        """Convert OpenAI-format messages into (system prompt, Anthropic messages)."""
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                system_parts.append(msg["content"])
            elif role == "user":
                converted.append({"role": "user", "content": msg["content"]})
            elif role == "assistant":
                blocks: list[dict[str, Any]] = []
                if msg.get("content"):
                    blocks.append({"type": "text", "text": msg["content"]})
                for tc in msg.get("tool_calls", []):
                    blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"]["arguments"] or "{}"),
                    })
                converted.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                block = {"type": "tool_result", "tool_use_id": msg["tool_call_id"], "content": msg["content"]}
                # Anthropic expects all results for one assistant turn in a single user message.
                if converted and converted[-1]["role"] == "user" and isinstance(converted[-1]["content"], list):
                    converted[-1]["content"].append(block)
                else:
                    converted.append({"role": "user", "content": [block]})
            else:
                raise ValueError(f"Unsupported message role {role!r}")
        return "\n\n".join(system_parts), converted

    def complete(self, request: LLMRequest) -> LLMResponse:
        system, messages = self.convert_messages(request.messages)
        if request.json_mode:
            system = (system + "\n\n" if system else "") + "Respond with a single JSON object and nothing else."
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if request.tools:
            kwargs["tools"] = [
                {"name": t["function"]["name"], "description": t["function"]["description"],
                 "input_schema": t["function"]["parameters"]}
                for t in request.tools
            ]
            if request.tool_choice:
                kwargs["tool_choice"] = self._TOOL_CHOICE[request.tool_choice]

        started = time.perf_counter()
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            raise classify_exception(self.name, exc) from exc
        latency = round((time.perf_counter() - started) * 1000, 1)

        texts, tool_calls = [], []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=json.dumps(block.input)))
        return LLMResponse(
            content="".join(texts) or None,
            tool_calls=tool_calls,
            provider=self.name,
            model=response.model,
            usage=TokenUsage(response.usage.input_tokens, response.usage.output_tokens),
            latency_ms=latency,
            finish_reason=response.stop_reason,
        )


MockItem = Any  # LLMResponse | LLMProviderError | str | dict | Callable[[LLMRequest], MockItem]


class MockProvider:
    """A scripted provider for tests and offline runs.

    ``script`` is either a callable ``request -> item`` or an iterable of items consumed in order.
    An item may be a string (reply text), a dict ``{"content": ..., "tool_calls": [{"name", "arguments"}]}``,
    an :class:`LLMResponse`, or an exception instance to raise (to simulate failures). When the
    script runs out, the provider replies ``"{}"`` in JSON mode and ``"MOCK RESPONSE"`` otherwise.
    """

    def __init__(self, script: Iterable[MockItem] | Callable[[LLMRequest], MockItem] | None = None,
                 *, name: str = "mock", model: str = "mock-llm") -> None:
        self.name = name
        self.model = model
        self._responder = script if callable(script) else None
        self._queue: deque[MockItem] = deque() if callable(script) or script is None else deque(script)
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self._responder is not None:
            item = self._responder(request)
        elif self._queue:
            item = self._queue.popleft()
        else:
            item = "{}" if request.json_mode else "MOCK RESPONSE"

        if isinstance(item, BaseException):
            raise classify_exception(self.name, item)
        if isinstance(item, LLMResponse):
            return item
        if isinstance(item, str):
            item = {"content": item}
        tool_calls = [
            ToolCall(
                id=tc.get("id") or f"call_{i + 1}",
                name=tc["name"],
                arguments=tc["arguments"] if isinstance(tc.get("arguments"), str) else json.dumps(tc.get("arguments", {})),
                extra=tc.get("extra"),
            )
            for i, tc in enumerate(item.get("tool_calls", []))
        ]
        prompt_text = json.dumps(request.messages, default=str)
        return LLMResponse(
            content=item.get("content"),
            tool_calls=tool_calls,
            provider=self.name,
            model=self.model,
            usage=TokenUsage(_estimate_tokens(prompt_text), _estimate_tokens(item.get("content") or "") + 10 * len(tool_calls)),
            latency_ms=0.0,
            finish_reason="tool_calls" if tool_calls else "stop",
        )


def build_provider(settings: ProviderSettings, *, timeout_s: float = 60.0) -> Provider:
    if settings.kind == "openai_compatible":
        return OpenAICompatibleProvider(settings, timeout_s=timeout_s)
    if settings.kind == "anthropic":
        return AnthropicProvider(settings, timeout_s=timeout_s)
    if settings.kind == "mock":
        return MockProvider(name=settings.name, model=settings.model)
    raise ConfigError(f"Unknown provider kind {settings.kind!r}")


# --------------------------------------------------------------------------------------------------
# Usage tracking
# --------------------------------------------------------------------------------------------------


@dataclass
class UsageStats:
    calls: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens + self.reasoning_tokens


class UsageTracker:
    """Token and latency accounting per ``provider/model`` (feeds token_usage_analysis.md on Day 12)."""

    def __init__(self) -> None:
        self.by_model: dict[str, UsageStats] = {}

    def _stats(self, provider: str, model: str) -> UsageStats:
        return self.by_model.setdefault(f"{provider}/{model}", UsageStats())

    def record_success(self, response: LLMResponse) -> None:
        stats = self._stats(response.provider, response.model)
        stats.calls += 1
        stats.prompt_tokens += response.usage.prompt_tokens
        stats.completion_tokens += response.usage.completion_tokens
        stats.reasoning_tokens += response.usage.reasoning_tokens
        stats.latency_ms += response.latency_ms

    def record_failure(self, provider: str, model: str) -> None:
        self._stats(provider, model).failures += 1

    def totals(self) -> UsageStats:
        total = UsageStats()
        for s in self.by_model.values():
            total.calls += s.calls
            total.failures += s.failures
            total.prompt_tokens += s.prompt_tokens
            total.completion_tokens += s.completion_tokens
            total.reasoning_tokens += s.reasoning_tokens
            total.latency_ms += s.latency_ms
        return total

    def summary(self) -> dict[str, Any]:
        rows = {k: {**vars(v), "total_tokens": v.total_tokens} for k, v in self.by_model.items()}
        t = self.totals()
        return {"by_model": rows, "total": {**vars(t), "total_tokens": t.total_tokens}}


# --------------------------------------------------------------------------------------------------
# The client
# --------------------------------------------------------------------------------------------------


class LLMClient:
    """Calls providers in order, with per-provider retries, tracing and usage tracking."""

    def __init__(
        self,
        providers: list[Provider],
        *,
        max_retries: int = 2,
        base_delay_s: float = 1.0,
        max_jitter_s: float = 0.5,
        temperature: float = 0.0,
        max_output_tokens: int = 4096,
        tracer: Any = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[], float] = random.random,
        clock: Callable[[], float] = time.monotonic,
        cooldown_s: float = 30.0,
    ) -> None:
        if not providers:
            raise ConfigError("LLMClient needs at least one provider")
        self.providers = providers
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s
        self.max_jitter_s = max_jitter_s
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.tracer = tracer
        self.usage = UsageTracker()
        self._sleep = sleep
        self._rng = rng
        self._clock = clock
        #: A rate-limited provider is skipped until this time while another provider is available.
        self.cooldown_s = cooldown_s
        #: A provider whose daily quota is exhausted is set aside for this long.
        self.quota_cooldown_s = 3600.0
        self._cooldown_until: dict[str, float] = {}

    @classmethod
    def from_settings(cls, settings: Settings, *, tracer: Any = None, **overrides: Any) -> LLMClient:
        llm: LLMSettings = settings.llm
        configured = llm.configured_chain()
        if not configured:
            missing = ", ".join(p.api_key_env for p in llm.chain() if p.needs_key)
            raise ConfigError(
                "No LLM provider is configured. Set one of these in .env: "
                f"{missing} (Gemini keys are free at https://aistudio.google.com/apikey), "
                "or install Ollama and set ARA_MODE=local, or set ARA_MODE=mock."
            )
        providers = [build_provider(p, timeout_s=llm.timeout_s) for p in configured]
        options: dict[str, Any] = {
            "max_retries": llm.max_retries,
            "base_delay_s": settings.retry.base_delay_s,
            "max_jitter_s": settings.retry.max_jitter_s,
            "temperature": llm.temperature,
            "max_output_tokens": llm.max_output_tokens,
            "tracer": tracer,
        }
        options.update(overrides)
        return cls(providers, **options)

    def _ready_and_cooling(self, now: float) -> tuple[list[Provider], list[Provider]]:
        # Providers cooling down after a rate limit or an unreachable server go last rather than being
        # dropped: per-minute limits clear within seconds, and a last resort beats no answer.
        ready = [p for p in self.providers if self._cooldown_until.get(p.name, 0.0) <= now]
        # Cooling providers go last, except those out of daily quota (cooled for longer than a minute
        # limit could explain), which are skipped until their cooldown ends.
        cooling = [p for p in self.providers if p not in ready
                   and self._cooldown_until[p.name] - now <= self.cooldown_s + 60]
        return ready, cooling

    def next_provider_name(self) -> str | None:
        """Name of the provider :meth:`complete` would try first right now (None if all are set aside)."""
        ready, cooling = self._ready_and_cooling(self._clock())
        queue = ready + cooling
        return queue[0].name if queue else None

    def any_provider_ready(self) -> bool:
        """True if at least one provider is not cooling down (rate limit, outage or exhausted quota)."""
        return bool(self._ready_and_cooling(self._clock())[0])

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        purpose: str = "",
        deadline: float | None = None,
    ) -> LLMResponse:
        """Send one chat request, falling back through the provider chain.

        Args:
            deadline: a ``clock()`` time after which no further attempt or wait is started, so the
                agent's time limit also bounds retries and backoff inside this call.

        Raises:
            AllProvidersFailedError: if every provider failed or the deadline passed.
        """
        request = LLMRequest(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            json_mode=json_mode,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
        errors: list[LLMProviderError] = []
        now = self._clock()
        ready, cooling = self._ready_and_cooling(now)
        queue = ready + cooling
        if not queue:
            raise AllProvidersFailedError([LLMProviderError(p.name, "quota_exhausted", "daily quota used up; skipped")
                                           for p in self.providers])
        final_pass: set[str] = set()  # providers cooled during this call get one last full try
        i = 0
        while i < len(queue):
            provider = queue[i]
            depth = self.providers.index(provider)
            is_last = i == len(queue) - 1
            final = provider.name in final_pass
            i += 1
            for attempt in range(self.max_retries + 1):
                if deadline is not None and self._clock() >= deadline:
                    errors.append(LLMProviderError(provider.name, "deadline", "time limit reached before this attempt"))
                    raise AllProvidersFailedError(errors)
                try:
                    response = provider.complete(request)
                except Exception as exc:  # providers raise LLMProviderError; anything else is classified
                    error = classify_exception(provider.name, exc)
                    errors.append(error)
                    self.usage.record_failure(provider.name, provider.model)
                    # Cooldowns run from the failure, not from the start of this call: earlier providers
                    # may already have spent a minute or more timing out.
                    failed_at = self._clock()
                    # A rate limit or an unreachable server rarely clears within a few seconds: with
                    # another provider still to try, move on at once and give this one a cooling-off
                    # period (a light circuit breaker; the full breaker arrives on Day 9).
                    if error.kind == "connection" and getattr(provider, "is_local", False):
                        # A local model server that refuses connections is not running: set it aside
                        # for ten minutes instead of retrying it on every call.
                        self._cooldown_until[provider.name] = failed_at + 600
                        self._trace("error", source="llm", provider=provider.name, model=provider.model,
                                    error_kind=error.kind, message=str(error), attempt=attempt + 1,
                                    will_retry=False, cooling_down=True, purpose=purpose)
                        break
                    if error.kind == "quota_exhausted":
                        # A daily quota: set the provider aside for an hour and do not retry it.
                        self._cooldown_until[provider.name] = failed_at + self.quota_cooldown_s
                        self._trace("error", source="llm", provider=provider.name, model=provider.model,
                                    error_kind=error.kind, message=str(error), attempt=attempt + 1,
                                    will_retry=False, cooling_down=True, purpose=purpose)
                        break
                    cool = error.kind in ("rate_limit", "connection") and not is_last and not final
                    if cool:
                        self._cooldown_until[provider.name] = failed_at + max(error.retry_after or 0.0,
                                                                              self.cooldown_s)
                        final_pass.add(provider.name)
                        queue.append(provider)
                    will_retry = error.retryable and attempt < self.max_retries and not cool
                    self._trace("error", source="llm", provider=provider.name, model=provider.model,
                                error_kind=error.kind, message=str(error), attempt=attempt + 1,
                                will_retry=will_retry, cooling_down=cool, purpose=purpose)
                    if not will_retry:
                        break  # permanent error or retries exhausted: next provider
                    wait = backoff_delay(attempt, base_s=self.base_delay_s, max_jitter_s=self.max_jitter_s,
                                         retry_after=error.retry_after, rng=self._rng)
                    if deadline is not None:
                        wait = min(wait, max(0.0, deadline - self._clock()))
                    self._sleep(wait)
                    continue

                response.fallback_depth = depth
                response.errors = list(errors)
                self.usage.record_success(response)
                self._trace(
                    "llm_call", provider=response.provider, model=response.model, purpose=purpose,
                    prompt_tokens=response.usage.prompt_tokens, completion_tokens=response.usage.completion_tokens,
                    reasoning_tokens=response.usage.reasoning_tokens,
                    latency_ms=response.latency_ms, fallback_depth=depth, attempt=attempt + 1,
                    finish_reason=response.finish_reason, tool_calls=[tc.name for tc in response.tool_calls],
                    preview=(response.content or "")[:200],
                )
                return response
        raise AllProvidersFailedError(errors)

    def _trace(self, event_type: str, **fields: Any) -> None:
        if self.tracer is not None:
            self.tracer.log(event_type, **fields)
