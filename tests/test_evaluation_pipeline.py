"""Regression tests for evaluation/runner.py, judge.py, evaluate.py and dashboard.py edge cases (Day 11 debug).

No network and no live LLM: the judge uses ``MockProvider`` and evaluate.py uses fake backends.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.llm import LLMClient, MockProvider
from evaluation import dashboard, evaluate, judge, runner


# --------------------------------------------------------------------------- evaluate.py
class _Metric:
    def __init__(self, d): self.d = d
    def to_dict(self): return self.d


class _Run:
    def __init__(self, bundle): self.bundle = bundle

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def _bundle(n: int) -> dict:
    return {"challenge": {"id": f"C{n}", "number": n, "title": f"t{n}"}, "query": "q"}


def _backends(judge_fn=None, is_cached=None):
    metric = {"id": "FA-1", "name": "n", "category": "Factual Accuracy", "value": 1.0, "target": ">= 1",
              "passed": True, "unit": "ratio"}
    ns = dict(Run=_Run, JudgeError=RuntimeError, judge_bundle=judge_fn or (lambda b, **kw: {"x": 1}),
              score_run=lambda run, j, b: [_Metric(metric)], gold_for=lambda cid: None, compare=lambda b, g: None)
    if is_cached:
        ns["is_cached"] = is_cached
    return SimpleNamespace(**ns)


def _write_runs(tmp_path, n=2):
    runs = tmp_path / "runs"
    runs.mkdir()
    for i in range(1, n + 1):
        (runs / f"C{i}.json").write_text(json.dumps(_bundle(i)), encoding="utf-8")
    return runs


def test_ids_filter_is_case_insensitive(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "_backends", lambda: _backends())
    runs = _write_runs(tmp_path)
    scores = evaluate.evaluate_dir(runs, ["c1"], use_judge=False, force_judge=False, pause=0)
    assert [r["id"] for r in scores["runs"]] == ["C1"]  # lower-case ids used to select nothing, silently


def test_malformed_run_file_is_skipped_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate, "_backends", lambda: _backends())
    runs = _write_runs(tmp_path)
    (runs / "C3.json").write_text("", encoding="utf-8")
    (runs / "C4.json").write_text("{not json", encoding="utf-8")
    (runs / "C5.json").write_text("[]", encoding="utf-8")
    (runs / "C6.json").write_text("{}", encoding="utf-8")  # no challenge block
    scores = evaluate.evaluate_dir(runs, None, use_judge=False, force_judge=False, pause=0)
    assert [r["id"] for r in scores["runs"]] == ["C1", "C2"]
    assert set(scores["skipped"]) == {"C3.json", "C4.json", "C5.json", "C6.json"}


def test_pause_only_after_uncached_judge_calls(tmp_path, monkeypatch):
    runs = _write_runs(tmp_path, 3)
    slept: list[float] = []
    monkeypatch.setattr(evaluate.time, "sleep", slept.append)
    cached_ids = {"C1", "C2"}
    monkeypatch.setattr(evaluate, "_backends", lambda: _backends(
        is_cached=lambda b: b["challenge"]["id"] in cached_ids))
    evaluate.evaluate_dir(runs, None, use_judge=True, force_judge=False, pause=7)
    assert slept == []  # C1, C2 cached, C3 is last: nothing to space out
    slept.clear()
    cached_ids.clear()
    evaluate.evaluate_dir(runs, None, use_judge=True, force_judge=False, pause=7)
    assert slept == [7, 7]
    slept.clear()
    cached_ids.update({"C1", "C2", "C3"})
    evaluate.evaluate_dir(runs, None, use_judge=True, force_judge=True, pause=7)  # --force-judge: all real calls
    assert slept == [7, 7]


def test_judge_provider_errors_are_recorded(tmp_path, monkeypatch):
    def boom(bundle, **kw):
        raise ConnectionError("network down")

    monkeypatch.setattr(evaluate, "_backends", lambda: _backends(judge_fn=boom))
    scores = evaluate.evaluate_dir(_write_runs(tmp_path), None, use_judge=True, force_judge=False, pause=0)
    assert len(scores["runs"]) == 2 and set(scores["judge"]["errors"]) == {"C1", "C2"}


def test_aggregate_edge_cases():
    runs = [{"metrics": [{"id": "A", "category": "X", "value": None, "passed": None}]},
            {"metrics": [{"id": "A", "category": "X", "value": None, "passed": None},
                         {"id": "B", "category": "Y", "value": 0, "passed": False}]},
            {"metrics": []}]
    agg, summ = evaluate.aggregate(runs)
    assert agg["A"]["mean"] is None and agg["A"]["measured"] == 0 and agg["B"]["n"] == 1
    assert summ["pass_rate"] == 0.0
    assert evaluate.aggregate([])[1]["pass_rate"] == 0.0  # no division by zero


def test_old_bundle_without_optional_keys_renders():
    entry = evaluate._run_entry({"challenge": {"id": "C1", "number": 1}}, [])
    scores = {"runs": [entry], "aggregate": {}, "summary": {"pass_rate": 0, "by_category": {}},
              "judge": {"enabled": False, "errors": {}}}
    assert "C1" in dashboard.render_html(scores) and "C1" in dashboard.render_markdown(scores)


# --------------------------------------------------------------------------- dashboard.py
def _scores_with(detail: str, examples=None) -> dict:
    m = {"id": "FA-1", "name": "Na|me", "category": "Factual Accuracy", "value": 0.5, "target": ">= 90% | x",
         "passed": False, "method": "auto", "unit": "ratio", "detail": detail, "examples": examples or []}
    run = {"id": "C1", "number": 1, "title": "T", "query": "q", "template": "t", "data_mode": "mock",
           "stop_reason": "done", "duration_s": 1, "tool_calls": 1, "llm_calls": 1, "tokens": 1,
           "injected_failures": {}, "metrics": [m]}
    agg, summ = evaluate.aggregate([run])
    return {"generated_at": "g", "judge": {"enabled": False, "errors": {}}, "runs": [run], "aggregate": agg,
            "summary": summ}


def test_markdown_details_with_pipes_and_newlines_keep_one_line_per_item():
    md = dashboard.render_markdown(_scores_with("| Revenue | $96,773 |\r\nline2\nline3"))
    bullet = [ln for ln in md.splitlines() if ln.startswith("- ✗ **FA-1**")]
    assert len(bullet) == 1 and "\r" not in md.split(dashboard.ANALYSIS_MARKER)[0]
    assert "\\| Revenue \\| $96,773 \\|" in bullet[0] and "line3" in bullet[0]
    row = next(ln for ln in md.splitlines() if ln.startswith("| FA-1 |"))
    assert row.count("|") - row.count("\\|") == 5  # id, metric, target, C1 cell: intact column count


def test_html_attribute_quotes_and_markup_are_escaped():
    nasty = "it's \"quoted\" ’curly” <b>&amp; \nnewline"
    page = dashboard.render_html(_scores_with(nasty, [nasty]))
    assert "<b>&" not in page and "title='it's" not in page
    assert "title='it&#x27;s &quot;quoted&quot;" in page


def test_analysis_survives_double_marker_crlf_bom_and_no_trailing_newline():
    scores = _scores_with("d")
    marker = dashboard.ANALYSIS_MARKER
    tail = f"\n## Analysis\n\nMine.\n{marker}\nsecond marker text"  # marker twice, no trailing newline
    out = dashboard.render_markdown(scores, "old\r\n" + marker + tail.replace("\n", "\r\n"))
    assert out.split(marker, 1)[1] == tail.replace("\n", "\r\n") and "Mine." in out
    out = dashboard.render_markdown(scores, "﻿old\n" + marker + tail)
    assert out.endswith(marker + tail) and not out.startswith("﻿")


def test_write_outputs_preserves_analysis_in_cp1252_report(tmp_path):
    report = tmp_path / "r.md"
    report.write_bytes(("x\n" + dashboard.ANALYSIS_MARKER + "\n## Analysis\n\nIt’s “good”.\n")
                       .encode("cp1252"))  # saved by Notepad: not valid UTF-8
    dashboard.write_outputs(_scores_with("d"), tmp_path / "d.html", report)
    assert "It’s “good”." in report.read_text(encoding="utf-8")


def test_write_outputs_roundtrip_is_stable(tmp_path):
    report = tmp_path / "r.md"
    dashboard.write_outputs(_scores_with("d"), tmp_path / "d.html", report)
    first = report.read_bytes()
    dashboard.write_outputs(_scores_with("d"), tmp_path / "d.html", report)
    assert report.read_bytes() == first


def test_fmt_value_edge_cases():
    f = dashboard.fmt_value
    assert f(0, "ratio") == "0%" and f(0, "seconds") == "0 s" and f(-0.25, "ratio") == "-25%"
    assert f(1.5, "ratio") == "150%" and f(3, None) == "3" and f(2, "") == "2"
    assert f("n/a", "ratio") == "n/a"
    assert dashboard._facts({"tokens": None, "duration_s": None, "tool_calls": None})  # no crash on None


# --------------------------------------------------------------------------- judge.py
def _client(script):
    return LLMClient([MockProvider(script)], max_retries=0, sleep=lambda s: None)


def _jb(**kw):
    return {"query": "q", "sections": {"summary": "Text."}, "report_markdown": "Text.",
            "challenge": {"id": "C1"}, **kw}


def test_judge_score_formats():
    assert judge._score({"score": float("inf")})[0] == 3  # OverflowError used to escape
    assert judge._score({"score": float("nan")})[0] == 3
    assert judge._score({"score": "4/5"})[0] == 4
    assert judge._score({"score": " 4 "})[0] == 4
    assert judge._score({"score": 4.5})[0] in (4, 5)


def test_judge_strings_skip_none_and_blanks():
    assert judge._strings(["a", None, "", "  ", 5]) == ["a", "5"]  # None used to become the text "None"


def test_judge_fenced_reply_and_messy_blocks():
    raw = {"insights": ["i", None], "logical_flow": {"score": "4"}, "exec_summary": "oops",
           "planning": None, "contradictions": [{"a": "x"}, "junk", {"why": "only why"}, {"b": None, "a": "y"}]}
    reply = "```json\n" + json.dumps(raw) + "\n```"
    out = judge.judge_bundle(_jb(), llm=_client([reply]), cache_path=None)
    assert out["insights"] == ["i"] and out["logical_flow"]["score"] == 4
    assert out["exec_summary"]["score"] == 3 and out["planning"]["score"] == 3
    assert [c["a"] for c in out["contradictions"]] == ["x", "y"] and out["contradictions"][0]["b"] == ""


def test_judge_corrupt_cache_is_treated_as_empty(tmp_path):
    path = tmp_path / "c.json"
    path.write_text("{corrupt", encoding="utf-8")
    out = judge.judge_bundle(_jb(), llm=_client(['{"insights": ["a"]}']), cache_path=path)
    assert out["insights"] == ["a"] and json.loads(path.read_text(encoding="utf-8"))  # healed


def test_is_cached_reflects_cache(tmp_path):
    path = tmp_path / "c.json"
    assert not judge.is_cached(_jb(), path)
    judge.judge_bundle(_jb(), llm=_client(["{}"]), cache_path=path)
    assert judge.is_cached(_jb(), path) and not judge.is_cached(_jb(query="other"), path)


def test_save_cache_retries_when_windows_has_the_file_open(tmp_path, monkeypatch):
    real, calls = os.replace, {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "file in use")
        return real(src, dst)

    monkeypatch.setattr(judge.os, "replace", flaky)
    monkeypatch.setattr(judge.time, "sleep", lambda s: None)
    path = tmp_path / "c.json"
    judge._save_cache(path, {"k": {"v": 1}})
    assert json.loads(path.read_text(encoding="utf-8")) == {"k": {"v": 1}} and not list(tmp_path.glob("*.tmp"))


def test_truncated_report_keeps_section_order_and_says_so():
    secs = {"alpha": "a" * 60_000, "beta": "b" * 60_000, "gamma": "c" * 100}
    prompt = judge.build_prompt(_jb(sections=secs))
    assert prompt.index("## Alpha") < prompt.index("## Beta") < prompt.index("## Gamma")
    assert "shortened" in prompt and "c" * 100 in prompt


def test_provider_order_env_restored_even_on_error(monkeypatch):
    monkeypatch.setenv("ARA_JUDGE_PROVIDER_ORDER", "groq")
    monkeypatch.setenv("ARA_PROVIDER_ORDER", "gemini")
    import config.settings as settings

    def boom():
        raise RuntimeError("bad settings")

    monkeypatch.setattr(settings, "load_settings", boom)
    with pytest.raises(RuntimeError):
        judge._default_llm()
    assert os.environ["ARA_PROVIDER_ORDER"] == "gemini"
    monkeypatch.delenv("ARA_PROVIDER_ORDER")
    with pytest.raises(RuntimeError):
        judge._default_llm()
    assert "ARA_PROVIDER_ORDER" not in os.environ


# --------------------------------------------------------------------------- runner.py
def test_jsonable_makes_strict_json():
    cyc: list = [1]
    cyc.append(cyc)
    d: dict = {"a": float("nan"), "b": float("inf"), "c": -math.inf, "d": 1.5, "e": cyc, "f": {1, 2}}
    out = runner._jsonable(d)
    text = json.dumps(out, allow_nan=False)  # raises on NaN/Infinity
    back = json.loads(text)
    assert back["a"] is None and back["b"] is None and back["c"] is None and back["d"] == 1.5


def test_trace_events_tolerates_non_object_lines(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_bytes(b'{"type": "error", "x": 1}\n5\n[1]\n"s"\nnull\n\xff\xfe broken\n{"type": "decision"}\n')
    assert [e["type"] for e in runner._trace_events(str(p))] == ["error", "decision"]
    assert runner._trace_events(None) == [] and runner._trace_events(str(tmp_path / "missing")) == []


class _FakeAgent:
    seen: list[dict] = []

    @classmethod
    def from_settings(cls, settings, max_seconds=0):
        return cls()

    def run(self, query):
        _FakeAgent.seen.append({k: os.environ.get(k) for k in
                                ("ARA_FAIL_RATE", "ARA_FAIL_SEED", "ARA_FAIL_TOOLS")})
        if query == "boom":
            raise RuntimeError("agent crashed")
        return "result"


def test_failure_injection_env_never_leaks_between_challenges(monkeypatch):
    import agent.core
    monkeypatch.setattr(agent.core, "ResearchAgent", _FakeAgent)
    for k in ("ARA_FAIL_RATE", "ARA_FAIL_SEED", "ARA_FAIL_TOOLS"):
        monkeypatch.delenv(k, raising=False)
    _FakeAgent.seen = []
    runner.run_challenge(runner.CHALLENGES["C8"])
    runner.run_challenge(runner.CHALLENGES["C1"])
    c8, c1 = _FakeAgent.seen
    assert c8["ARA_FAIL_RATE"] == "0.5" and c8["ARA_FAIL_SEED"] == "8" and "financial_data_api" in c8["ARA_FAIL_TOOLS"]
    assert c1 == {"ARA_FAIL_RATE": None, "ARA_FAIL_SEED": None, "ARA_FAIL_TOOLS": None}
    boom = runner.Challenge("CX", 99, "t", "boom", fail_rate=0.5, fail_tools=("web_search",))
    with pytest.raises(RuntimeError):
        runner.run_challenge(boom)
    assert not any(k in os.environ for k in ("ARA_FAIL_RATE", "ARA_FAIL_SEED", "ARA_FAIL_TOOLS"))


def test_main_restores_data_mode_env_and_reports_exit_code(tmp_path, monkeypatch):
    monkeypatch.setenv("ARA_DATA_MODE", "live")
    monkeypatch.setattr(runner, "run_challenge", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    rc = runner.main(["C1", "--data", "mock", "--out-dir", str(tmp_path), "--report-dir=", "--pause", "0"])
    assert rc == 1 and os.environ["ARA_DATA_MODE"] == "live"  # used to stay "mock" for the rest of the process
    assert not list(tmp_path.iterdir())


def test_main_lowercase_ids_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "run_challenge", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert runner.main(["c1", "--data", "mock", "--out-dir", str(tmp_path), "--report-dir=", "--pause", "0"]) == 1
