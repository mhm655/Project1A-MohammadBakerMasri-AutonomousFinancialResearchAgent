"""Tests for the evaluation entry point and dashboard (spec A5.2, Day 11). No LLM, no network."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaluation import dashboard, evaluate

ROOT = Path(__file__).resolve().parent.parent


def _m(mid: str, cat: str, value, passed, unit="ratio", detail="d", examples=None, method="auto") -> dict:
    return {"id": mid, "name": f"name {mid}", "category": cat, "value": value, "target": ">= 90%",
            "passed": passed, "method": method, "unit": unit, "detail": detail, "examples": examples or []}


def _runs() -> list[dict]:
    base = {"template": "profile", "data_mode": "live", "stop_reason": "done", "duration_s": 100,
            "tool_calls": 5, "llm_calls": 3, "tokens": 1000, "fail_rate": 0, "injected_failures": {}}
    r1 = {**base, "id": "C1", "number": 1, "title": "Profile", "query": "q <script>alert(1)</script>",
          "metrics": [_m("FA-1", "Factual Accuracy", 0.97, True, examples=["ex <b>1</b>"]),
                      _m("FA-2", "Factual Accuracy", 0.5, False, detail="bad <script>x</script>"),
                      _m("CS-1", "Coherence and Structure", 4.0, True, "score_1_5", method="judge")]}
    r2 = {**base, "id": "C2", "number": 2, "title": "Earnings", "query": "q2", "duration_s": 200,
          "tokens": 500, "injected_failures": {"web_search": 2},
          "metrics": [_m("FA-1", "Factual Accuracy", None, None),
                      _m("FA-2", "Factual Accuracy", 0.95, True),
                      _m("CS-1", "Coherence and Structure", 2.0, False, "score_1_5", method="judge")]}
    return [r1, r2]


def _scores() -> dict:
    runs = _runs()
    agg, summ = evaluate.aggregate(runs)
    return {"generated_at": "2026-01-01T00:00:00+00:00", "judge": {"enabled": True, "errors": {}},
            "runs": runs, "aggregate": agg, "summary": summ}


def test_aggregate_and_summary():
    agg, summ = evaluate.aggregate(_runs())
    assert agg["FA-1"]["mean"] == pytest.approx(0.97)
    assert (agg["FA-1"]["passed"], agg["FA-1"]["measured"], agg["FA-1"]["n"]) == (1, 1, 2)
    assert agg["FA-2"]["mean"] == pytest.approx(0.725)
    assert summ["metrics_total"] == 3
    assert summ["by_category"]["Factual Accuracy"] == {"passed": 2, "measured": 3}
    assert summ["pass_rate"] == pytest.approx(3 / 5)


def test_fmt_value():
    f = dashboard.fmt_value
    assert f(0.97, "ratio") == "97%"
    assert f(4, "score_1_5") == "4/5"
    assert f(142, "seconds") == "142 s"
    assert f(3.14, "per_page") == "3.1/pg"
    assert f(7.0, "count") == "7"
    assert f(None, "ratio") == "—"


def test_html_escapes_and_cells():
    page = dashboard.render_html(_scores())
    assert "<script>" not in page
    assert "&lt;script&gt;" in page
    assert "class='c pass'" in page and "class='c fail'" in page and "class='c na'" in page
    assert "97% ✓" in page and "50% ✗" in page and "4/5 ✓" in page
    assert ">—<" in page
    for cat in ("Factual Accuracy", "Coherence and Structure"):
        assert f"<td colspan='6'>{cat}</td>" in page
    assert "http://" not in page.replace("http://www.w3.org", "") and "https://" not in page


def test_markdown_sections_and_glyphs():
    md = dashboard.render_markdown(_scores())
    for h in ("# ARA-1 Evaluation Report", "## Summary", "## Metric Matrix", "## Per-Challenge Results",
              "## Method", dashboard.ANALYSIS_MARKER, "## Analysis", dashboard.ANALYSIS_PLACEHOLDER):
        assert h in md
    assert "✓" in md and "✗" in md and "| — |" in md
    assert "ERROR_LOG.md Error 1" in md


def test_analysis_preserved():
    existing = dashboard.render_markdown(_scores()) .replace(
        dashboard.ANALYSIS_PLACEHOLDER, "My hand-written analysis.")
    again = dashboard.render_markdown(_scores(), existing)
    assert "My hand-written analysis." in again
    assert dashboard.ANALYSIS_PLACEHOLDER not in again
    assert again.count(dashboard.ANALYSIS_MARKER) == 1
    # No marker in the existing file -> placeholder.
    assert dashboard.ANALYSIS_PLACEHOLDER in dashboard.render_markdown(_scores(), "old text")


def test_evaluate_main_writes_files(tmp_path, monkeypatch):
    src = ROOT / "results" / "dev" / "runs_mock" / "C1.json"
    if not src.exists():
        pytest.skip("mock bundle missing")
    runs = tmp_path / "runs"
    runs.mkdir()
    shutil.copy(src, runs / "C1.json")

    class FakeMetric:
        def __init__(self, d): self.d = d
        def to_dict(self): return self.d

    class FakeRun:
        def __init__(self, bundle): self.bundle = bundle
        @classmethod
        def load(cls, path):
            import json
            return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    class FakeJudgeError(Exception):
        pass

    def judge_bundle(bundle, **kw):
        raise FakeJudgeError("quota")

    fake = SimpleNamespace(
        Run=FakeRun, JudgeError=FakeJudgeError, judge_bundle=judge_bundle,
        score_run=lambda run, j, b: [FakeMetric(_m("FA-1", "Factual Accuracy", 1.0, True))],
        gold_for=lambda cid: None, compare=lambda b, g: None)
    monkeypatch.setattr(evaluate, "_backends", lambda: fake)
    out, dash, rep = tmp_path / "scores.json", tmp_path / "d.html", tmp_path / "r.md"
    rc = evaluate.main(["--runs", str(runs), "--out", str(out), "--dashboard", str(dash),
                        "--report", str(rep), "--pause", "0"])
    assert rc == 0
    import json
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["runs"][0]["id"] == "C1" and "C1" in data["judge"]["errors"]
    assert "ARA-1 Evaluation Dashboard" in dash.read_text(encoding="utf-8")
    assert "## Analysis" in rep.read_text(encoding="utf-8")
