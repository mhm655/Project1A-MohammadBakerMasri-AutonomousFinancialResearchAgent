"""Episodic memory: what worked in past research sessions (spec A3.2, architecture spec section 10.4).

After every run, an *episode* records the query type, the plan, how useful each tool was (were its
results cited in the report?), the errors met, and quality scores. Before the next run of the same
report type, the best episodes become *strategy hints* for the planner. This is how the agent improves
from challenge to challenge (spec B2.1).

Lessons are derived by rules from the run's own numbers rather than by an extra LLM call, to save the
free-tier quota. They are factual observations such as "earnings_transcript was called twice and never
cited".
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tools.report_gen import cited_ids

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    session_id TEXT PRIMARY KEY,
    created_at TEXT,
    query TEXT,
    template TEXT,
    tickers TEXT,
    plan_json TEXT,
    tool_stats_json TEXT,
    errors_json TEXT,
    metrics_json TEXT,
    lessons_json TEXT,
    score REAL
)
"""


def episode_from_result(result: Any) -> dict[str, Any]:
    """Summarise a :class:`agent.core.ResearchResult` as an episode."""
    cited = cited_ids(result.report_markdown)
    tool_stats: dict[str, dict[str, int]] = {}
    for step in result.plan.steps:
        stats = tool_stats.setdefault(step.tool, {"steps": 0, "done": 0, "failed": 0, "cited": 0})
        stats["steps"] += 1
        stats["done"] += step.status == "done"
        stats["failed"] += step.status == "failed"
        stats["cited"] += any(e in cited for e in step.evidence_ids)
    errors = [{"tool": s.tool, "summary": s.summary[:200]} for s in result.plan.by_status("failed")]
    v = result.verification or {}
    claims = v.get("checked", 0) + v.get("uncited", 0)
    supported_share = v.get("supported", 0) / claims if claims else 0.0
    done_share = len(result.plan.by_status("done")) / max(1, len(result.plan.steps))
    score = round(0.6 * supported_share + 0.3 * done_share + 0.1 * (result.stop_reason == "completed"), 3)
    metrics = {
        "duration_s": result.duration_s, "tool_calls": result.tool_calls, "llm_calls": result.llm_usage.get("calls"),
        "supported": v.get("supported", 0), "unsupported": v.get("unsupported", 0), "uncited": v.get("uncited", 0),
        "stop_reason": result.stop_reason, "missing_sections": len((result.report_checks or {}).get("missing_sections") or []),
    }
    return {
        "session_id": result.session_id, "query": result.query, "template": result.template,
        "plan": [{"tool": s.tool, "goal": s.goal, "status": s.status} for s in result.plan.steps],
        "tool_stats": tool_stats, "errors": errors, "metrics": metrics,
        "lessons": derive_lessons(tool_stats, errors, metrics), "score": score,
    }


def derive_lessons(tool_stats: dict[str, dict[str, int]], errors: list[dict[str, str]],
                   metrics: dict[str, Any]) -> list[str]:
    """Turn a finished run's tool statistics, errors and metrics into short planner lessons."""
    lessons = []
    for tool, s in sorted(tool_stats.items()):
        if s["done"] and not s["cited"]:
            lessons.append(f"{tool} returned data {s['done']} time(s) but was never cited: consider skipping it.")
        elif s["cited"]:
            lessons.append(f"{tool} was useful ({s['cited']} of {s['steps']} step(s) cited).")
    for err in errors:
        if "did not finish within" in err["summary"]:
            lessons.append(f"{err['tool']} was too slow on this connection: prefer the latest (cached) filing.")
        elif "no data" in err["summary"] or "not found" in err["summary"] or "no EDGAR" in err["summary"]:
            lessons.append(f"{err['tool']} had no data for this company: {err['summary'][:120]}")
    if metrics.get("stop_reason") not in (None, "completed"):
        lessons.append(f"The run stopped early ({metrics['stop_reason']}): plan fewer steps.")
    if metrics.get("uncited"):
        lessons.append(f"{metrics['uncited']} figure(s) were left uncited: cite every figure, including table captions.")
    return lessons


class EpisodicMemory:
    """SQLite log of past research runs (episodes); ``path=None`` keeps it in memory."""
    def __init__(self, path: str | Path | None = None) -> None:
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(_SCHEMA)
            self._db.commit()

    def record(self, episode: dict[str, Any]) -> None:
        """Insert or replace the episode for its ``session_id``, stamped with the current UTC time."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (episode["session_id"], datetime.now(UTC).isoformat(timespec="seconds"), episode["query"],
                 episode["template"], json.dumps(episode.get("tickers", [])), json.dumps(episode["plan"]),
                 json.dumps(episode["tool_stats"]), json.dumps(episode["errors"]), json.dumps(episode["metrics"]),
                 json.dumps(episode["lessons"]), episode["score"]))
            self._db.commit()

    def episodes(self, template: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """Past episodes, best score first (newest first on ties), optionally only for one report ``template``."""
        sql, params = "SELECT * FROM episodes", []
        if template:
            sql, params = sql + " WHERE template = ?", [template]
        with self._lock:
            rows = self._db.execute(sql + " ORDER BY score DESC, created_at DESC LIMIT ?", [*params, limit]).fetchall()
        return [{**dict(r), **{k: json.loads(r[f"{k}_json"]) for k in ("plan", "tool_stats", "errors", "metrics", "lessons")}}
                for r in rows]

    def hints(self, template: str, limit: int = 2) -> str:
        """Planner hints from the best past episodes of the same report type ("" if there are none)."""
        best = self.episodes(template, limit)
        if not best:
            return ""
        lines = [f"Lessons from {len(best)} earlier {template.replace('_', ' ')} run(s):"]
        for ep in best:
            useful = [t for t, s in ep["tool_stats"].items() if s["cited"]]
            lines.append(f"- score {ep['score']:.2f}; tools that produced cited evidence: {', '.join(useful) or 'none'}")
            lines += [f"  - {lesson}" for lesson in ep["lessons"][:5]]
        return "\n".join(lines)
