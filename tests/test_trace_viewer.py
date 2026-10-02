"""Tests for agent.trace_viewer."""

import json

import pytest

from agent import trace_viewer

EVENTS = [
    {"seq": 1, "type": "thought", "text": "hi", "step_id": "s1"},
    {"seq": 2, "type": "action", "tool": "t", "args": {}, "step_id": "s2"},
    {"seq": 3, "type": "decision", "chosen": "x", "step_id": "s2"},
]


def test_filter_by_type_and_step():
    assert [e["seq"] for e in trace_viewer.filter_events(EVENTS, ["action", "decision"])] == [2, 3]
    assert [e["seq"] for e in trace_viewer.filter_events(EVENTS, step="s2")] == [2, 3]
    assert [e["seq"] for e in trace_viewer.filter_events(EVENTS, ["thought"], "s2")] == []


def test_load_jsonl_and_bundle(tmp_path):
    jl = tmp_path / "a.jsonl"
    jl.write_text("\n".join(json.dumps(e) for e in EVENTS), encoding="utf-8")
    bundle = tmp_path / "b.json"
    bundle.write_text(json.dumps({"trace_events": EVENTS}), encoding="utf-8")
    assert trace_viewer.load_events(jl) == EVENTS
    assert trace_viewer.load_events(bundle) == EVENTS


def test_bundle_without_events_rejected(tmp_path):
    bad = tmp_path / "c.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        trace_viewer.load_events(bad)


def test_main_prints_filtered(tmp_path, capsys):
    jl = tmp_path / "a.jsonl"
    jl.write_text("\n".join(json.dumps(e) for e in EVENTS), encoding="utf-8")
    assert trace_viewer.main([str(jl), "--types", "decision"]) == 0
    out = capsys.readouterr().out
    assert "decision" in out and "thought" not in out
    assert trace_viewer.main([str(tmp_path / "missing.json")]) == 2
