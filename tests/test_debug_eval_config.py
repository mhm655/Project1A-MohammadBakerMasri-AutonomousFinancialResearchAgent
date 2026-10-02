"""Debug-pass regression tests for evaluation/ and config/ (edge cases found by the Day 14 review)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from config import doctor
from config.settings import ConfigError, load_settings
from evaluation import evaluate, judge, memory_probe, runner
from evaluation import metrics as m
from tests.test_evaluation_metrics import _bundle, _ev

ROOT = Path(__file__).resolve().parent.parent


# --- metrics: None values inside a bundle -------------------------------------------------------------

def test_metrics_survive_none_query_and_none_evidence_fields():
    b = _bundle(query=None, evidence={"E1": _ev("E1", "vector_db_search", "x", data=None, args=None)})
    results = m.score_run(m.Run.from_bundle(b))
    assert len(results) == 22


def test_ad1_judge_insight_count_none_falls_back_to_list_length():
    j = {"insight_count": None, "insights": ["a", "b"]}
    r = {x.id: x for x in m.score_run(m.Run.from_bundle(_bundle()), j)}["AD-1"]
    assert r.value is not None


# --- judge: malformed contradictions ------------------------------------------------------------------

def test_judge_normalise_non_list_contradictions():
    out = judge.normalise({"contradictions": 3, "insights": "none"}, provider="p", model="x")
    assert out["contradictions"] == []


# --- evaluate: token total includes reasoning tokens --------------------------------------------------

def test_run_entry_tokens_include_reasoning():
    bundle = {"challenge": {"id": "C1"}, "llm_usage": {"calls": 1, "prompt_tokens": 10, "completion_tokens": 5,
                                                       "reasoning_tokens": 7}}
    assert evaluate._run_entry(bundle, [])["tokens"] == 22


# --- runner: environment restored after a failed run --------------------------------------------------

def test_run_challenge_restores_preexisting_fail_env(monkeypatch):
    monkeypatch.setenv("ARA_FAIL_RATE", "0.25")
    monkeypatch.setenv("ARA_FAIL_TOOLS", "web_search")
    monkeypatch.delenv("ARA_FAIL_SEED", raising=False)

    def boom(*_a, **_k):
        raise RuntimeError("agent failed")

    import agent.core as core
    monkeypatch.setattr(core.ResearchAgent, "from_settings", classmethod(lambda cls, *a, **k: boom()))
    with pytest.raises(RuntimeError):
        runner.run_challenge(runner.CHALLENGES["C8"])
    assert os.environ.get("ARA_FAIL_RATE") == "0.25"
    assert os.environ.get("ARA_FAIL_TOOLS") == "web_search"
    assert "ARA_FAIL_SEED" not in os.environ


# --- memory_probe: tiny / empty stores ----------------------------------------------------------------

def test_make_probes_terminates_when_fewer_distinct_phrases_than_requested():
    text = " ".join(f"w{i}" for i in range(200))
    records = [{"id": "a", "content": text, "meta": {"ticker": "X", "source_type": "10-K"}}]
    probes = memory_probe.make_probes(records, 48)
    assert 1 <= len(probes) < 48


def test_hit_at_k_and_floor_table_handle_no_probes():
    col = SimpleNamespace(query=lambda **_k: {"documents": [[]], "distances": [[]]})
    assert memory_probe.hit_at_k(col, []) == {"hit@3": 0.0, "hit@5": 0.0}
    assert all(r["hits"] == 0 for r in memory_probe.floor_table(col, []))


# --- settings -----------------------------------------------------------------------------------------

def _cfg(tmp_path, extra: str) -> Path:
    text = (ROOT / "config" / "default.toml").read_text(encoding="utf-8") + "\n" + extra
    (tmp_path / "default.toml").write_text(text, encoding="utf-8")
    return tmp_path


def test_unknown_agent_key_is_a_config_error(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    bad = (ROOT / "config" / "default.toml").read_text(encoding="utf-8").replace(
        "max_replans = 3", "max_replans = 3\nmax_replan = 9")
    (d / "default.toml").write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings({}, config_dir=d, project_root=tmp_path, load_dotenv_file=False)


def test_provider_without_model_is_a_config_error(tmp_path):
    d = tmp_path / "c"
    d.mkdir()
    bad = (ROOT / "config" / "default.toml").read_text(encoding="utf-8").replace(
        'model = "qwen2.5:3b"\n', "")
    (d / "default.toml").write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings({}, config_dir=d, project_root=tmp_path, load_dotenv_file=False)


# --- doctor -------------------------------------------------------------------------------------------

def test_doctor_table_survives_bracketed_error_text(capsys):
    doctor._print_table([doctor.Check("llm:x chat", "fail", "bad reply [/foo] and [bold]text[/bold]")])
    out = capsys.readouterr().out
    assert "[/foo]" in out and "[bold]text[/bold]" in out
