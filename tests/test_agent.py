"""Tests for Day 3 infrastructure: configuration, the LLM client, and tracing.

No test here touches the network or needs an API key.
"""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import httpx
import openai
import pytest

from agent.llm import (
    AllProvidersFailedError,
    AnthropicProvider,
    LLMClient,
    LLMProviderError,
    LLMRequest,
    MockProvider,
    OpenAICompatibleProvider,
    adapt_tools,
    assistant_message,
    backoff_delay,
    classify_exception,
    tool_result_message,
)
from agent.tracing import REDACTED, TraceLogger, format_event, read_trace
from config.settings import CONFIG_DIR, ConfigError, ProviderSettings, load_settings
from tools import ToolExecutionError, build_default_registry

# ==================================================================================================
# Configuration
# ==================================================================================================


def test_defaults_without_any_environment():
    s = load_settings(env={})
    assert s.mode == "free" and s.data_mode == "live"
    assert s.llm.provider_order == ("gemini", "groq", "groq_qwen", "ollama")
    assert [p.name for p in s.llm.configured_chain()] == ["ollama"]  # the only provider needing no key
    assert s.agent.max_tool_calls == 20
    assert (s.retry.base_delay_s, s.retry.max_retries, s.retry.max_jitter_s) == (1.0, 5, 0.5)
    assert s.paths.chroma_dir.is_absolute() and s.paths.chroma_dir.parts[-2:] == ("data", "chroma")


def test_mode_selects_provider_chain_and_data_mode():
    mock = load_settings(env={"ARA_MODE": "mock"})
    assert mock.llm.provider_order == ("mock",) and mock.data_mode == "mock"
    local = load_settings(env={"ARA_MODE": "local"})
    assert local.llm.provider_order == ("ollama",) and local.data_mode == "live"
    assert load_settings(env={"ARA_MODE": "free", "ARA_DATA_MODE": "mock"}).data_mode == "mock"


def test_keys_come_from_the_environment_and_never_appear_in_repr():
    s = load_settings(env={"GEMINI_API_KEY": "AIza-secret-value"})
    gemini = s.llm.providers["gemini"]
    assert gemini.is_configured and gemini.api_key == "AIza-secret-value"
    assert "AIza-secret-value" not in repr(s)


def test_overrides_for_order_and_model():
    s = load_settings(env={"ARA_PROVIDER_ORDER": "groq, ollama", "ARA_GROQ_MODEL": "some-model"})
    assert s.llm.provider_order == ("groq", "ollama")
    assert s.llm.providers["groq"].model == "some-model"


@pytest.mark.parametrize(
    "env, message",
    [
        ({"ARA_MODE": "production"}, "ARA_MODE"),
        ({"ARA_DATA_MODE": "half"}, "ARA_DATA_MODE"),
        ({"ARA_PROVIDER_ORDER": "gemini,bard"}, "unknown providers"),
    ],
)
def test_invalid_configuration_is_rejected(env, message):
    with pytest.raises(ConfigError, match=message):
        load_settings(env=env)


def test_mode_file_overrides_defaults(tmp_path):
    shutil.copy(CONFIG_DIR / "default.toml", tmp_path / "default.toml")
    (tmp_path / "local.toml").write_text("[llm]\ntemperature = 0.3\n\n[llm.providers.ollama]\nmodel = \"llama3.2:3b\"\n")
    s = load_settings(env={"ARA_MODE": "local"}, config_dir=tmp_path)
    assert s.llm.temperature == 0.3
    assert s.llm.providers["ollama"].model == "llama3.2:3b"
    assert s.llm.providers["ollama"].base_url == "http://localhost:11434/v1"  # untouched keys survive the merge


def test_sec_user_agent_requires_contact_email():
    with pytest.raises(ConfigError, match="SEC_CONTACT_EMAIL"):
        _ = load_settings(env={}).sec.user_agent
    assert load_settings(env={"SEC_CONTACT_EMAIL": "me@example.com"}).sec.user_agent == "ARA-1 Research Agent me@example.com"


def test_client_explains_how_to_get_a_free_key_when_nothing_is_configured():
    s = load_settings(env={"ARA_PROVIDER_ORDER": "gemini,groq"})
    with pytest.raises(ConfigError, match="GEMINI_API_KEY.*aistudio"):
        LLMClient.from_settings(s)


def test_mock_mode_runs_end_to_end_without_network_or_keys():
    s = load_settings(env={"ARA_MODE": "mock"})
    tracer = TraceLogger()
    client = LLMClient.from_settings(s, tracer=tracer)
    registry = build_default_registry(s.data_mode)
    registry.add_listener(tracer.on_tool_call)
    assert client.complete([{"role": "user", "content": "hi"}], tools=registry.openai_tools()).provider == "mock"
    assert registry.dispatch("company_profile", {"ticker": "NVDA"}).is_mock
    assert [e["type"] for e in tracer.events] == ["llm_call", "action", "observation"]


def test_client_from_settings_builds_the_configured_chain():
    s = load_settings(env={"GROQ_API_KEY": "gsk-test"})
    client = LLMClient.from_settings(s)
    assert [p.name for p in client.providers] == ["groq", "groq_qwen", "ollama"]  # gemini skipped: no key
    assert client.max_retries == s.llm.max_retries


# ==================================================================================================
# Backoff and error classification
# ==================================================================================================


def test_backoff_is_exponential_with_jitter_and_cap():
    assert [backoff_delay(a, rng=lambda: 0.0) for a in range(4)] == [1.0, 2.0, 4.0, 8.0]
    assert backoff_delay(0, rng=lambda: 1.0) == 1.5  # max jitter 0.5 s
    assert backoff_delay(10, rng=lambda: 0.0) == 60.0  # capped
    assert backoff_delay(0, retry_after=7, rng=lambda: 0.0) == 7.0  # server asked for longer


def _status_error(cls, status, headers=None):
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    return cls("boom", response=httpx.Response(status, request=request, headers=headers or {}), body=None)


@pytest.mark.parametrize(
    "exc, kind, retryable",
    [
        (_status_error(openai.RateLimitError, 429, {"retry-after": "7"}), "rate_limit", True),
        (_status_error(openai.AuthenticationError, 401), "auth", False),
        (_status_error(openai.NotFoundError, 404), "not_found", False),
        (_status_error(openai.BadRequestError, 400), "bad_request", False),
        (_status_error(openai.InternalServerError, 503), "server", True),
        (openai.APITimeoutError(request=httpx.Request("POST", "https://example.invalid")), "timeout", True),
        (openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid")), "connection", True),
        (ValueError("weird"), "other", False),
    ],
)
def test_sdk_exceptions_are_classified(exc, kind, retryable):
    error = classify_exception("gemini", exc)
    assert (error.kind, error.retryable, error.provider) == (kind, retryable, "gemini")
    if kind == "rate_limit":
        assert error.retry_after == 7.0


# ==================================================================================================
# LLMClient: retries, fallback, usage, tracing
# ==================================================================================================


def make_client(*providers, tracer=None, max_retries=2):
    sleeps: list[float] = []
    client = LLMClient(list(providers), max_retries=max_retries, tracer=tracer, sleep=sleeps.append, rng=lambda: 0.0)
    return client, sleeps


def rate_limited(provider="gemini"):
    return LLMProviderError(provider, "rate_limit", "quota exceeded")


def test_success_on_primary_provider():
    client, sleeps = make_client(MockProvider(["hello"], name="gemini"))
    response = client.complete([{"role": "user", "content": "hi"}])
    assert response.content == "hello" and response.provider == "gemini"
    assert response.fallback_depth == 0 and response.errors == [] and sleeps == []


def test_transient_errors_are_retried_with_backoff_then_succeed():
    gemini = MockProvider([rate_limited(), rate_limited(), "third time lucky"], name="gemini")
    client, sleeps = make_client(gemini)
    response = client.complete([{"role": "user", "content": "hi"}])
    assert response.content == "third time lucky"
    assert sleeps == [1.0, 2.0]
    assert len(response.errors) == 2 and response.fallback_depth == 0


def test_exhausted_retries_fall_back_to_the_next_provider():
    # server errors (503) are retried; rate limits move on at once (see the cooldown test below)
    gemini = MockProvider([LLMProviderError("gemini", "server", "503 high demand")] * 3, name="gemini")
    groq = MockProvider(["from groq"], name="groq")
    client, sleeps = make_client(gemini, groq)
    response = client.complete([{"role": "user", "content": "hi"}])
    assert response.provider == "groq" and response.fallback_depth == 1
    assert len(gemini.requests) == 3 and len(sleeps) == 2
    assert client.usage.by_model["gemini/mock-llm"].failures == 3


def test_permanent_errors_skip_retries():
    gemini = MockProvider([LLMProviderError("gemini", "auth", "bad key")], name="gemini")
    groq = MockProvider(["ok"], name="groq")
    client, sleeps = make_client(gemini, groq)
    assert client.complete([{"role": "user", "content": "hi"}]).provider == "groq"
    assert len(gemini.requests) == 1 and sleeps == []


def test_all_providers_failing_raises_with_every_attempt():
    client, _ = make_client(
        MockProvider([LLMProviderError("gemini", "auth", "x")], name="gemini"),
        MockProvider([ConnectionError("down")] * 3, name="ollama"),
    )
    with pytest.raises(AllProvidersFailedError) as info:
        client.complete([{"role": "user", "content": "hi"}])
    assert [e.provider for e in info.value.errors] == ["gemini", "ollama", "ollama", "ollama"]


def test_tool_calls_and_usage_are_tracked():
    script = [{"content": None, "tool_calls": [{"name": "company_profile", "arguments": {"ticker": "MSFT"}}]}]
    client, _ = make_client(MockProvider(script))
    response = client.complete([{"role": "user", "content": "profile MSFT"}], tools=[])
    call = response.tool_calls[0]
    assert (call.name, json.loads(call.arguments)) == ("company_profile", {"ticker": "MSFT"})
    totals = client.usage.totals()
    assert totals.calls == 1 and totals.prompt_tokens > 0 and totals.total_tokens > totals.prompt_tokens


def test_llm_calls_and_errors_are_traced():
    tracer = TraceLogger()
    client, _ = make_client(MockProvider([rate_limited(), "ok"], name="gemini"), tracer=tracer)
    client.complete([{"role": "user", "content": "hi"}], purpose="plan")
    error, call = tracer.of_type("error")[0], tracer.of_type("llm_call")[0]
    assert error["source"] == "llm" and error["will_retry"] is True and error["error_kind"] == "rate_limit"
    assert call["purpose"] == "plan" and call["attempt"] == 2 and call["provider"] == "gemini"


def test_json_mode_default_reply():
    client, _ = make_client(MockProvider())
    assert client.complete([{"role": "user", "content": "x"}], json_mode=True).content == "{}"


# ==================================================================================================
# Provider adapters
# ==================================================================================================


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.kwargs = response, error, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def fake_openai_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def chat_completion(message, finish_reason="stop"):
    return openai.types.chat.ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "gemini-2.5-flash",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", **message}}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    })


GEMINI = ProviderSettings(name="gemini", kind="openai_compatible", model="gemini-2.5-flash",
                          base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                          api_key_env="GEMINI_API_KEY", schema_dialect="gemini", api_key="k")


def test_openai_compatible_provider_request_and_response():
    completions = FakeCompletions(chat_completion(
        {"content": None, "tool_calls": [{"id": "", "type": "function",
                                          "function": {"name": "company_profile", "arguments": '{"ticker":"MSFT"}'}}]},
        finish_reason="tool_calls"))
    provider = OpenAICompatibleProvider(GEMINI, client=fake_openai_client(completions))
    tools = build_default_registry("mock").openai_tools()
    response = provider.complete(LLMRequest(messages=[{"role": "user", "content": "x"}], tools=tools,
                                            tool_choice="auto", json_mode=True, max_tokens=256))

    sent = completions.kwargs
    assert sent["model"] == "gemini-2.5-flash" and sent["max_tokens"] == 256 and "max_completion_tokens" not in sent
    assert sent["response_format"] == {"type": "json_object"} and sent["tool_choice"] == "auto"
    assert "additionalProperties" not in json.dumps(sent["tools"])  # gemini dialect applied
    assert response.tool_calls[0].name == "company_profile" and response.tool_calls[0].id.startswith("call_")
    assert (response.usage.prompt_tokens, response.usage.completion_tokens) == (12, 3)
    assert response.finish_reason == "tool_calls"


def test_gemini_thought_signatures_survive_the_round_trip():
    """Gemini 3 rejects a follow-up turn whose tool calls lack the thought signature it issued."""
    signature = {"extra_content": {"google": {"thought_signature": "EtsCCtgC-opaque"}}}
    completions = FakeCompletions(chat_completion(
        {"content": None, "tool_calls": [{"id": "call_1", "type": "function",
                                          "function": {"name": "company_profile", "arguments": '{"ticker":"MSFT"}'},
                                          **signature}]},
        finish_reason="tool_calls"))
    response = OpenAICompatibleProvider(GEMINI, client=fake_openai_client(completions)).complete(
        LLMRequest(messages=[{"role": "user", "content": "x"}]))
    assert response.tool_calls[0].extra == signature
    echoed = assistant_message(response)["tool_calls"][0]
    assert echoed["extra_content"] == signature["extra_content"] and echoed["function"]["name"] == "company_profile"


@pytest.mark.parametrize(
    "usage, expected",
    [
        # Gemini: hidden thinking shows up only in total_tokens
        ({"prompt_tokens": 6, "completion_tokens": 0, "total_tokens": 22}, (6, 0, 16)),
        # OpenAI style: reasoning reported inside completion_tokens
        ({"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60,
          "completion_tokens_details": {"reasoning_tokens": 30}}, (10, 20, 30)),
        # no reasoning at all
        ({"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}, (12, 3, 0)),
    ],
)
def test_reasoning_tokens_are_counted(usage, expected):
    response = openai.types.chat.ChatCompletion.model_validate({
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK"}}],
        "usage": usage,
    })
    out = OpenAICompatibleProvider(GEMINI, client=fake_openai_client(FakeCompletions(response))).complete(
        LLMRequest(messages=[{"role": "user", "content": "x"}]))
    u = out.usage
    assert (u.prompt_tokens, u.completion_tokens, u.reasoning_tokens) == expected
    assert u.total_tokens == usage["total_tokens"]


def test_reasoning_effort_is_only_sent_to_providers_configured_for_it():
    thinking = ProviderSettings(**{**vars(GEMINI), "reasoning_effort": "minimal"})
    completions = FakeCompletions(chat_completion({"content": "OK"}))
    provider = OpenAICompatibleProvider(thinking, client=fake_openai_client(completions))
    provider.complete(LLMRequest(messages=[{"role": "user", "content": "x"}]))
    assert completions.kwargs["reasoning_effort"] == "minimal"
    provider.complete(LLMRequest(messages=[{"role": "user", "content": "x"}], reasoning_effort="medium"))
    assert completions.kwargs["reasoning_effort"] == "medium"  # per-call override, e.g. for planning

    plain = FakeCompletions(chat_completion({"content": "OK"}))
    OpenAICompatibleProvider(GEMINI, client=fake_openai_client(plain)).complete(
        LLMRequest(messages=[{"role": "user", "content": "x"}], reasoning_effort="medium"))
    assert "reasoning_effort" not in plain.kwargs  # providers that do not support it never see it


def test_reasoning_effort_setting_and_override():
    assert load_settings(env={}).llm.providers["gemini"].reasoning_effort == "minimal"
    assert load_settings(env={}).llm.providers["groq"].reasoning_effort == "low"
    assert load_settings(env={}).llm.providers["groq_qwen"].reasoning_effort == ""
    s = load_settings(env={"ARA_GEMINI_REASONING_EFFORT": "low"})
    assert s.llm.providers["gemini"].reasoning_effort == "low"


def test_openai_compatible_provider_maps_sdk_errors():
    completions = FakeCompletions(error=_status_error(openai.RateLimitError, 429))
    provider = OpenAICompatibleProvider(GEMINI, client=fake_openai_client(completions))
    with pytest.raises(LLMProviderError) as info:
        provider.complete(LLMRequest(messages=[{"role": "user", "content": "x"}]))
    assert info.value.kind == "rate_limit"


def test_official_openai_endpoint_uses_max_completion_tokens():
    settings = ProviderSettings(name="openai", kind="openai_compatible", model="gpt-4o-mini",
                                base_url="https://api.openai.com/v1", api_key_env="OPENAI_API_KEY", api_key="k")
    completions = FakeCompletions(chat_completion({"content": "hi"}))
    OpenAICompatibleProvider(settings, client=fake_openai_client(completions)).complete(
        LLMRequest(messages=[{"role": "user", "content": "x"}], max_tokens=99))
    assert completions.kwargs["max_completion_tokens"] == 99 and "max_tokens" not in completions.kwargs


def test_gemini_dialect_turns_free_form_objects_into_json_strings():
    registry = build_default_registry("mock")
    original = registry.openai_tools()
    adapted = {t["function"]["name"]: t["function"]["parameters"] for t in adapt_tools(original, "gemini")}

    assert adapted["calculation_engine"]["properties"]["inputs"]["type"] == "string"
    assert adapted["report_generator"]["properties"]["sections"]["type"] == "string"
    assert adapted["vector_db_search"]["properties"]["filter"]["type"] == "object"  # has declared properties
    assert "additionalProperties" not in json.dumps(adapted) and "uniqueItems" not in json.dumps(adapted)
    assert "additionalProperties" in json.dumps(original)  # the input was not mutated
    assert adapt_tools(original, "openai") == original


def test_registry_decodes_json_string_arguments_sent_by_gemini():
    registry = build_default_registry("mock")
    result = registry.dispatch("calculation_engine", {"calculation_type": "ratio",
                                                      "inputs": '{"numerator": 1, "denominator": 4}'})
    assert result.data["result"] == 0.25
    with pytest.raises(Exception, match="inputs"):
        registry.dispatch("calculation_engine", {"calculation_type": "ratio", "inputs": "not json"})


def test_anthropic_message_conversion_round_trip():
    tool_turn = LLMClient([MockProvider([{"tool_calls": [{"id": "t1", "name": "company_profile",
                                                          "arguments": {"ticker": "MSFT"}}]}])]).complete([])
    messages = [
        {"role": "system", "content": "You are ARA-1."},
        {"role": "user", "content": "Profile MSFT"},
        assistant_message(tool_turn),
        tool_result_message("t1", {"name": "Microsoft Corporation"}),
    ]
    system, converted = AnthropicProvider.convert_messages(messages)
    assert system == "You are ARA-1."
    assert converted[1]["content"][0] == {"type": "tool_use", "id": "t1", "name": "company_profile",
                                          "input": {"ticker": "MSFT"}}
    assert converted[2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": '{"name": "Microsoft Corporation"}'}]}


def test_anthropic_provider_parses_blocks():
    response = SimpleNamespace(
        model="claude-sonnet-5", stop_reason="tool_use", usage=SimpleNamespace(input_tokens=20, output_tokens=5),
        content=[SimpleNamespace(type="text", text="Looking up."),
                 SimpleNamespace(type="tool_use", id="t9", name="web_search", input={"query": "nvidia"})])
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return response

    settings = ProviderSettings(name="anthropic", kind="anthropic", model="claude-sonnet-5", api_key="k")
    provider = AnthropicProvider(settings, client=SimpleNamespace(messages=SimpleNamespace(create=create)))
    tools = [t for t in build_default_registry("mock").openai_tools() if t["function"]["name"] == "web_search"]
    out = provider.complete(LLMRequest(messages=[{"role": "user", "content": "x"}], tools=tools, tool_choice="required"))
    assert captured["tool_choice"] == {"type": "any"} and captured["tools"][0]["input_schema"]["type"] == "object"
    assert out.content == "Looking up." and out.tool_calls[0].name == "web_search"
    assert json.loads(out.tool_calls[0].arguments) == {"query": "nvidia"}


def test_message_helpers():
    msg = assistant_message(LLMClient([MockProvider(["plain answer"])]).complete([]))
    assert msg == {"role": "assistant", "content": "plain answer"}
    assert tool_result_message("c1", "text") == {"role": "tool", "tool_call_id": "c1", "content": "text"}


# ==================================================================================================
# Tracing
# ==================================================================================================


def test_trace_is_written_as_jsonl_and_redacts_secrets(tmp_path):
    with TraceLogger(traces_dir=tmp_path) as tracer:
        tracer.thought("Start with the 10-K.", node="plan")
        tracer.log("note", config={"api_key": "AIza-secret", "model": "m"}, headers={"Authorization": "Bearer x"})
    events = read_trace(tracer.path)
    assert [e["seq"] for e in events] == [1, 2] and events == tracer.events
    assert events[1]["config"] == {"api_key": REDACTED, "model": "m"} and events[1]["headers"]["Authorization"] == REDACTED
    assert "AIza-secret" not in tracer.path.read_text(encoding="utf-8")
    assert tracer.path.name.startswith("session-") and tracer.path.suffix == ".jsonl"


def test_unknown_event_type_is_rejected():
    with pytest.raises(ValueError):
        TraceLogger().log("gossip")


def test_tool_calls_flow_into_the_trace():
    tracer = TraceLogger()
    registry = build_default_registry("mock")
    registry.add_listener(tracer.on_tool_call)
    registry.dispatch("company_profile", {"ticker": "MSFT"})
    with pytest.raises(ToolExecutionError):
        registry.dispatch("company_profile", {"ticker": "ZZZZ"})
    assert [e["type"] for e in tracer.events] == ["action", "observation", "action", "error"]
    assert tracer.events[1]["evidence_id"] == "E1" and tracer.events[3]["error_kind"] == "execution_error"
    assert "company_profile" in format_event(tracer.events[0])


def test_a_broken_listener_does_not_break_tool_calls():
    registry = build_default_registry("mock")
    registry.add_listener(lambda record: 1 / 0)
    assert registry.dispatch("company_profile", {"ticker": "MSFT"}).data["ticker"] == "MSFT"


def test_cached_results_are_isolated_from_caller_mutation():
    registry = build_default_registry("mock")
    first = registry.dispatch("company_profile", {"ticker": "MSFT"})
    first.data["name"] = "tampered"
    assert registry.dispatch("company_profile", {"ticker": "MSFT"}).data["name"] == "Microsoft Corporation"


def test_rate_limited_provider_cools_down_while_others_are_available():
    now = [0.0]
    gemini = MockProvider([rate_limited(), "gemini is back"], name="gemini")
    groq = MockProvider(["groq 1", "groq 2"], name="groq")
    sleeps: list[float] = []
    client = LLMClient([gemini, groq], max_retries=2, sleep=sleeps.append, rng=lambda: 0.0,
                       clock=lambda: now[0], cooldown_s=30)
    assert client.complete([{"role": "user", "content": "a"}]).provider == "groq"
    assert len(gemini.requests) == 1 and sleeps == []  # no retries or waiting on the rate-limited provider
    now[0] = 10
    assert client.complete([{"role": "user", "content": "b"}]).provider == "groq"  # gemini still cooling
    assert len(gemini.requests) == 1
    now[0] = 31
    assert client.complete([{"role": "user", "content": "c"}]).provider == "gemini"  # cooled down


def test_the_last_provider_is_still_retried_on_rate_limits():
    client, sleeps = make_client(MockProvider([rate_limited(), "ok"], name="groq"))
    assert client.complete([{"role": "user", "content": "x"}]).content == "ok" and sleeps == [1.0]


def test_cooling_providers_remain_a_last_resort():
    """Every provider is rate-limited except an unreachable one: the cooled providers are retried last."""
    now = [0.0]
    groq = MockProvider([rate_limited("groq"), rate_limited("groq"), "groq recovered"], name="groq")
    ollama = MockProvider([ConnectionError("refused")] * 5, name="ollama")
    sleeps: list[float] = []
    client = LLMClient([groq, ollama], max_retries=2, sleep=sleeps.append, rng=lambda: 0.0,
                       clock=lambda: now[0], cooldown_s=30)
    assert client.complete([{"role": "user", "content": "a"}]).content == "groq recovered"
    # groq cooled down at once, ollama was tried and cooled too, then groq was retried as the last resort
    assert len(ollama.requests) == 1 and len(groq.requests) == 3


# ==================================================================================================
# Free-tier quota handling (found when the daily quotas ran out during the Day 4 evaluation)
# ==================================================================================================


@pytest.mark.parametrize("message, kind", [
    ("Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000", "quota_exhausted"),
    ("You exceeded your current quota, please check your plan and billing details.", "quota_exhausted"),
    ("Request too large for model `qwen/qwen3.8-27b` on output tokens per minute (OTPM): Limit 1000", "request_too_large"),
    ("Rate limit reached on tokens per minute (TPM): Limit 8000", "rate_limit"),
    ("You exceeded your current quota. Quota exceeded for metric: generate_content_free_tier_requests, "
     "quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "rate_limit"),
    ("You exceeded your current quota. Quota exceeded for metric: generate_content_free_tier_requests, "
     "quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quota_exhausted"),
])
def test_429s_are_told_apart(message, kind):
    error = classify_exception("groq", type("E", (Exception,), {"status_code": 429, "__str__": lambda s: message})())
    assert error.kind == kind and error.retryable == (kind == "rate_limit")


def test_a_provider_out_of_daily_quota_is_skipped_not_retried():
    now = [0.0]
    daily = LLMProviderError("gemini", "quota_exhausted", "exceeded your current quota")
    gemini = MockProvider([daily, "should not be reached"], name="gemini")
    groq = MockProvider(["a", "b"], name="groq")
    client = LLMClient([gemini, groq], max_retries=2, sleep=lambda s: None, clock=lambda: now[0])
    assert client.complete([{"role": "user", "content": "1"}]).provider == "groq"
    now[0] = 600  # ten minutes later: still out of quota, so not even tried
    assert client.complete([{"role": "user", "content": "2"}]).provider == "groq"
    assert len(gemini.requests) == 1


def test_the_deadline_bounds_retries_and_waiting():
    now = [0.0]
    waits: list[float] = []

    def sleep(seconds):
        waits.append(seconds)
        now[0] += seconds

    busy = MockProvider([LLMProviderError("groq", "server", "503")] * 10, name="groq")
    client = LLMClient([busy], max_retries=5, sleep=sleep, clock=lambda: now[0], base_delay_s=4.0)
    with pytest.raises(AllProvidersFailedError) as info:
        client.complete([{"role": "user", "content": "x"}], deadline=6.0)
    assert info.value.errors[-1].kind == "deadline"
    assert now[0] <= 6.0 and len(busy.requests) == 2  # waited 4 s, then only the 2 s left, then stopped


def test_output_cap_shrinks_requests_to_the_free_tier_allowance():
    qwen = ProviderSettings(name="groq_qwen", kind="openai_compatible", model="qwen/qwen3.8-27b",
                            base_url="https://api.groq.com/openai/v1", api_key="k", max_output_tokens=1000)
    completions = FakeCompletions(chat_completion({"content": "OK"}))
    OpenAICompatibleProvider(qwen, client=fake_openai_client(completions)).complete(
        LLMRequest(messages=[{"role": "user", "content": "x"}], max_tokens=6000))
    assert completions.kwargs["max_tokens"] == 1000
    assert load_settings(env={}).llm.providers["groq_qwen"].max_output_tokens == 1000


def test_a_local_server_that_is_not_running_is_set_aside():
    ollama_settings = ProviderSettings(name="ollama", kind="openai_compatible", model="qwen2.5:3b",
                                       base_url="http://localhost:11434/v1")
    completions = FakeCompletions(error=openai.APIConnectionError(request=httpx.Request("POST", "http://localhost")))
    ollama = OpenAICompatibleProvider(ollama_settings, client=fake_openai_client(completions))
    assert ollama.is_local
    now = [0.0]
    client = LLMClient([ollama], max_retries=3, sleep=lambda s: None, clock=lambda: now[0])
    with pytest.raises(AllProvidersFailedError) as info:
        client.complete([{"role": "user", "content": "x"}])
    assert len(info.value.errors) == 1  # tried once, not retried
    now[0] = 60
    with pytest.raises(AllProvidersFailedError):
        client.complete([{"role": "user", "content": "y"}])  # still set aside: not even attempted


# ==================================================================================================
# Audit regressions
# ==================================================================================================


def test_cooldowns_start_when_the_provider_fails_not_when_the_call_started():
    """gemini spends 100 s timing out before groq is rate-limited. groq's 30 s cooldown must run
    from its failure (t=100), so a call at t=110 still tries it last, not second."""
    now = [0.0]

    def slow_timeout(request):
        now[0] += 100
        return LLMProviderError("gemini", "timeout", "read timed out")

    gemini = MockProvider(slow_timeout, name="gemini")
    groq = MockProvider([rate_limited("groq"), "groq answer"], name="groq")
    ollama = MockProvider(["ollama 1", "ollama 2"], name="ollama")
    client = LLMClient([gemini, groq, ollama], max_retries=0, sleep=lambda s: None, clock=lambda: now[0],
                       cooldown_s=30)
    assert client.complete([{"role": "user", "content": "a"}]).provider == "ollama"
    assert now[0] == 100 and len(groq.requests) == 1
    now[0] = 110  # groq's cooldown runs until t=130; gemini is still slow
    assert client.complete([{"role": "user", "content": "b"}]).provider == "ollama"
    assert len(groq.requests) == 1


def test_hyphenated_credential_keys_are_redacted():
    tracer = TraceLogger()
    event = tracer.log("note", headers={"X-Api-Key": "sk-live-123", "x-goog-api-key": "AIza-456", "accept": "json"})
    assert event["headers"] == {"X-Api-Key": REDACTED, "x-goog-api-key": REDACTED, "accept": "json"}
