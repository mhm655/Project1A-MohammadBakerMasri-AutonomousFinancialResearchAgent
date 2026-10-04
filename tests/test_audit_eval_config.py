"""Audit regression tests for evaluation/ and config/ (offline, no network, no real LLM)."""

from __future__ import annotations

from config import doctor

# --- doctor: fallback downgrade ---------------------------------------------------------------------------

def _run_doctor_with(monkeypatch, per_provider):
    monkeypatch.setenv("ARA_MODE", "free")
    monkeypatch.setenv("ARA_PROVIDER_ORDER", ",".join(per_provider))
    monkeypatch.setattr(doctor, "check_vector_db", lambda offline: ("ok", "stubbed"))
    monkeypatch.setattr(doctor, "check_provider", lambda name: [doctor.Check(c.name, c.status, c.detail)
                                                               for c in per_provider[name]])
    return {c.name: c for c in doctor.run(offline=False)}


def test_doctor_two_providers_that_both_reject_tool_schemas_still_fail(monkeypatch):
    """Both providers chat fine but neither accepts the real tool schemas: nothing covers for either, so
    the tool failures must stay failures (they used to be downgraded to warnings, exit code 0)."""
    checks = _run_doctor_with(monkeypatch, {
        "gemini": [doctor.Check("llm:gemini chat", "ok", "OK"), doctor.Check("llm:gemini tools", "fail", "rejected")],
        "groq": [doctor.Check("llm:groq chat", "ok", "OK"), doctor.Check("llm:groq tools", "fail", "rejected")],
    })
    assert checks["llm:gemini tools"].status == "fail"
    assert checks["llm:groq tools"].status == "fail"


def test_doctor_fully_working_provider_still_excuses_a_broken_fallback(monkeypatch):
    checks = _run_doctor_with(monkeypatch, {
        "gemini": [doctor.Check("llm:gemini chat", "ok", "OK"), doctor.Check("llm:gemini tools", "ok", "fine")],
        "groq": [doctor.Check("llm:groq chat", "ok", "OK"), doctor.Check("llm:groq tools", "fail", "rejected")],
    })
    assert checks["llm:gemini tools"].status == "ok"
    assert checks["llm:groq tools"].status == "warn"


# --- settings: whitespace around environment values ----------------------------------------------------------

def _settings(env):
    from config.settings import load_settings

    return load_settings(env, load_dotenv_file=False)


def test_whitespace_only_api_key_or_email_counts_as_unset():
    s = _settings({"GEMINI_API_KEY": "   ", "SEC_CONTACT_EMAIL": "  "})
    assert s.llm.providers["gemini"].api_key is None and not s.llm.providers["gemini"].is_configured
    assert s.sec.contact_email is None


def test_env_values_are_stripped():
    s = _settings({"ARA_MODE": "mock ", "ARA_LOG_LEVEL": " debug ", "GROQ_API_KEY": " k \r",
                   "ARA_PROVIDER_ORDER": " groq ", "ARA_FAIL_RATE": " 0.5 "})
    assert s.mode == "mock" and s.log_level == "DEBUG"
    s = _settings({"GROQ_API_KEY": " k \r", "ARA_DATA_MODE": "mock "})
    assert s.llm.providers["groq"].api_key == "k" and s.data_mode == "mock"


# --- dashboard: a ratio below 100% must not be shown as 100% ----------------------------------------------------

def test_ratio_just_below_one_is_not_displayed_as_100_percent():
    from evaluation import dashboard

    assert dashboard.fmt_value(0.995, "ratio") != "100%"  # was "100%" next to a failing mark
    assert dashboard.fmt_value(0.9996, "ratio") != "100%"
    assert dashboard.fmt_value(1.0, "ratio") == "100%" and dashboard.fmt_value(0.97, "ratio") == "97%"
    cell = dashboard._cell_text({"value": 0.995, "unit": "ratio", "passed": False})
    assert not cell.startswith("100%")


# --- runner: the console summary must count reasoning tokens like the scores do -----------------------------------

def test_runner_console_tokens_include_reasoning_tokens(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace

    from evaluation import runner

    result = SimpleNamespace(stop_reason="done", duration_s=1.0, tool_calls=2,
                             llm_usage={"calls": 3, "prompt_tokens": 100, "completion_tokens": 20,
                                        "reasoning_tokens": 30})
    monkeypatch.setattr(runner, "run_challenge", lambda *a, **k: (object(), result))
    monkeypatch.setattr(runner, "build_bundle", lambda *a, **k: {})
    monkeypatch.setattr(runner, "save", lambda *a, **k: tmp_path / "C1.json")
    assert runner.main(["C1", "--data", "mock", "--out-dir", str(tmp_path), "--report-dir=", "--pause", "0"]) == 0
    assert "150 tokens" in capsys.readouterr().out  # 100 + 20 + 30, not 120


# --- benchmark: short keywords must match whole words ---------------------------------------------------------

def test_qualitative_fact_with_short_keyword_needs_the_word_not_a_substring():
    from evaluation import benchmark

    gold = {"facts": [{"id": "ai", "kind": "qualitative", "text": "AI risk", "keywords": ["AI", "cloud", "demand", "invest"]}]}
    # "ai" is inside "maintain" / "against" / "retail": one more keyword used to be enough for a false recall.
    no_ai = {"report_markdown": "We maintain strong retail cloud capacity against rivals."}
    with_ai = {"report_markdown": "AI demand drives cloud capacity."}
    assert benchmark.compare(no_ai, gold).value == 0.0
    assert benchmark.compare(with_ai, gold).value == 1.0
