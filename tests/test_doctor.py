"""Tests for the environment doctor (config/doctor.py). Offline only; no keys needed."""

from __future__ import annotations

import json

from config import doctor


def test_offline_doctor_runs_every_local_check(monkeypatch):
    monkeypatch.setenv("ARA_MODE", "mock")
    checks = {c.name: c for c in doctor.run(offline=True)}
    for name in ("python", "packages", "settings", "tool registry", "vector db"):
        assert checks[name].status == "ok", (name, checks[name].detail)
    assert checks["llm providers"].status == "skip"
    assert "12 tools registered" in checks["tool registry"].detail


def test_a_crashing_check_is_reported_not_raised():
    def boom():
        raise RuntimeError("disk on fire")

    check = doctor._run("explodes", boom)
    assert check.status == "fail" and "disk on fire" in check.detail


def test_unconfigured_provider_is_skipped(monkeypatch):
    monkeypatch.setenv("ARA_MODE", "free")
    monkeypatch.setenv("GROQ_API_KEY", "")
    [check] = doctor.check_provider("groq")
    assert check.status == "skip" and "GROQ_API_KEY" in check.detail


def test_json_output_and_exit_code(monkeypatch, capsys):
    monkeypatch.setenv("ARA_MODE", "mock")
    code = doctor.main(["--offline", "--json"])
    rows = json.loads(capsys.readouterr().out)
    assert code == 0 and {"name", "status", "detail"} <= set(rows[0])


def test_a_failure_of_the_only_working_provider_is_not_excused_as_a_fallback(monkeypatch):
    """The fail -> warn downgrade is for broken fallbacks. A provider whose chat works but which
    rejects the tool schemas is not 'a fallback provider' covered by another one. No network: the
    provider and vector-db checks are replaced."""
    monkeypatch.setenv("ARA_MODE", "free")
    monkeypatch.setenv("ARA_PROVIDER_ORDER", "gemini,groq")
    monkeypatch.setattr(doctor, "check_vector_db", lambda offline: ("ok", "stubbed"))

    def fake_check_provider(name):
        if name == "gemini":
            return [doctor.Check("llm:gemini (m) chat", "ok", "replied 'OK'"),
                    doctor.Check("llm:gemini (m) tools", "fail", "provider rejected the tool schemas")]
        return [doctor.Check("llm:groq (m) chat", "fail", "auth error")]

    monkeypatch.setattr(doctor, "check_provider", fake_check_provider)
    checks = {c.name: c for c in doctor.run(offline=False)}
    assert checks["llm:groq (m) chat"].status == "warn"  # a broken fallback: gemini works
    assert checks["llm:gemini (m) tools"].status == "fail"
