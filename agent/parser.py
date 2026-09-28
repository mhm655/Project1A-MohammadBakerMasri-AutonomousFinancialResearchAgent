"""Parsing and validating LLM output.

LLM output is untrusted text. Everything the agent acts on passes through here first:

* :func:`extract_json` finds the JSON object in a reply, tolerating code fences and surrounding prose.
* :func:`parse_plan` turns a plan into validated :class:`PlanStep` objects: unknown tools, invalid
  arguments, duplicate ids, missing dependencies and dependency cycles are rejected with a note
  explaining why, rather than executed.
* :func:`parse_sections` splits a synthesis reply into report sections.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from tools import ToolRegistry, ToolValidationError

# Tools the system runs itself; the planner must not schedule them.
SYSTEM_TOOLS = frozenset({"report_generator", "fact_checker", "vector_db_search", "vector_db_store"})


class ParseError(ValueError):
    """The LLM output could not be understood."""


# Models sometimes emit invisible or typographic characters: gpt-oss wraps citations in zero-width
# spaces ("[​E2​]"), which look right but defeat every citation check.
_TRANSLATION = {
    **dict.fromkeys(map(ord, "​‌‍⁠﻿"), None),  # zero-width characters
    **dict.fromkeys(map(ord, "    "), " "),  # non-breaking and thin spaces
    **dict.fromkeys(map(ord, "‐‑"), "-"),  # typographic hyphens
    ord("【"): "[", ord("】"): "]",  # fullwidth brackets: gpt-oss writes citations as 【E8】
}


def normalize_text(text: str | None) -> str:
    """LLM text with invisible characters removed and unusual spaces and hyphens made plain."""
    return (text or "").translate(_TRANSLATION)


@dataclass
class PlanStep:
    id: str
    goal: str
    tool: str
    args: dict[str, Any] | None  # None = fill in from dependency results at execution time
    depends_on: list[str] = field(default_factory=list)
    reason: str = ""
    status: str = "pending"  # pending | running | done | failed | skipped
    evidence_ids: list[str] = field(default_factory=list)
    summary: str = ""
    attempts: int = 0


@dataclass
class Plan:
    steps: list[PlanStep]
    rationale: str = ""
    #: Why steps were dropped during validation; surfaced in the trace.
    rejected: list[str] = field(default_factory=list)

    def get(self, step_id: str) -> PlanStep:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(step_id)

    def ready(self) -> list[PlanStep]:
        """Pending steps whose dependencies have all finished (successfully or not)."""
        finished = {s.id for s in self.steps if s.status in ("done", "failed", "skipped")}
        return [s for s in self.steps if s.status == "pending" and set(s.depends_on) <= finished]

    def by_status(self, *statuses: str) -> list[PlanStep]:
        return [s for s in self.steps if s.status in statuses]

    @property
    def complete(self) -> bool:
        return not self.by_status("pending", "running")


# --------------------------------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------------------------------

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str | None) -> dict[str, Any]:
    """Return the first JSON object in ``text``.

    Handles code fences, prose before or after the object, and a trailing comma before a closing
    bracket (a common model slip).
    """
    text = normalize_text(text)
    if not text.strip():
        raise ParseError("empty reply")
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for candidate in candidates:
        start = candidate.find("{")
        while start != -1:
            end = _matching_brace(candidate, start)
            if end == -1:
                break
            blob = candidate[start : end + 1]
            for attempt in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
                try:
                    value = json.loads(attempt)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    return value
            start = candidate.find("{", start + 1)
    raise ParseError(f"no JSON object found in reply: {text[:120]!r}")


def _matching_brace(text: str, start: int) -> int:
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


# --------------------------------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------------------------------


def parse_steps(raw_steps: Any, registry: ToolRegistry, *, existing_ids: set[str] | None = None,
                max_steps: int | None = None) -> tuple[list[PlanStep], list[str]]:
    """Validate raw step dicts. Returns (valid steps, reasons for rejected steps)."""
    if not isinstance(raw_steps, list):
        raise ParseError("'steps' must be a list")
    taken = set(existing_ids or ())
    steps: list[PlanStep] = []
    rejected: list[str] = []
    for i, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            rejected.append(f"step {i + 1}: not an object")
            continue
        step_id = str(raw.get("id") or f"s{i + 1}")
        tool = str(raw.get("tool") or "")
        label = f"{step_id} ({tool or 'no tool'})"
        if step_id in taken:
            rejected.append(f"{label}: duplicate step id")
            continue
        if tool in SYSTEM_TOOLS:
            rejected.append(f"{label}: run by the system, not planned")
            continue
        if tool not in registry or not registry.is_available(tool):
            rejected.append(f"{label}: unknown or unavailable tool")
            continue
        args = raw.get("args")
        if args is not None:
            try:
                args = registry.validate_args(tool, args)
            except ToolValidationError as exc:
                # Keep the step: the executor asks the LLM to repair the arguments.
                rejected.append(f"{label}: arguments will be repaired at run time ({exc})")
                args = None
        depends_on = [str(d) for d in raw.get("depends_on") or [] if d is not None]
        steps.append(PlanStep(id=step_id, goal=str(raw.get("goal") or f"Call {tool}"), tool=tool, args=args,
                              depends_on=depends_on, reason=str(raw.get("reason") or "")))
        taken.add(step_id)

    if max_steps is not None and len(steps) > max_steps:
        rejected += [f"{s.id} ({s.tool}): over the step limit of {max_steps}" for s in steps[max_steps:]]
        steps = steps[:max_steps]

    # Dependencies must exist (in this batch or earlier ones); unknown ones are dropped.
    known = taken | {s.id for s in steps}
    for step in steps:
        missing = [d for d in step.depends_on if d not in known]
        if missing:
            rejected.append(f"{step.id}: ignored unknown dependencies {missing}")
            step.depends_on = [d for d in step.depends_on if d in known]
    cyclic = _cyclic_steps(steps)
    if cyclic:
        rejected += [f"{sid}: part of a dependency cycle" for sid in sorted(cyclic)]
        steps = [s for s in steps if s.id not in cyclic]
    return steps, rejected


def _cyclic_steps(steps: list[PlanStep]) -> set[str]:
    graph = {s.id: [d for d in s.depends_on if d in {t.id for t in steps}] for s in steps}
    state: dict[str, int] = {}  # 1 = visiting, 2 = done
    cyclic: set[str] = set()

    def visit(node: str, path: list[str]) -> None:
        if state.get(node) == 2:
            return
        if state.get(node) == 1:
            cyclic.update(path[path.index(node):])
            return
        state[node] = 1
        for dep in graph[node]:
            visit(dep, [*path, dep])
        state[node] = 2

    for sid in graph:
        visit(sid, [sid])
    return cyclic


def parse_plan(text: str | None, registry: ToolRegistry, *, max_steps: int) -> Plan:
    data = extract_json(text)
    steps, rejected = parse_steps(data.get("steps"), registry, max_steps=max_steps)
    if not steps:
        raise ParseError("the plan has no usable steps: " + "; ".join(rejected or ["no steps given"]))
    return Plan(steps=steps, rationale=str(data.get("rationale") or ""), rejected=rejected)


@dataclass
class ReplanDecision:
    done: bool
    new_steps: list[PlanStep] = field(default_factory=list)
    rationale: str = ""
    rejected: list[str] = field(default_factory=list)


def parse_replan(text: str | None, registry: ToolRegistry, plan: Plan, *, max_new_steps: int) -> ReplanDecision:
    data = extract_json(text)
    action = str(data.get("action") or "done").lower()
    if action != "continue":
        return ReplanDecision(done=True, rationale=str(data.get("rationale") or ""))
    steps, rejected = parse_steps(data.get("add_steps") or [], registry,
                                  existing_ids={s.id for s in plan.steps}, max_steps=max_new_steps)
    # Never repeat a call that already succeeded.
    done_calls = {(s.tool, json.dumps(s.args, sort_keys=True)) for s in plan.by_status("done") if s.args}
    fresh = []
    for step in steps:
        if step.args is not None and (step.tool, json.dumps(step.args, sort_keys=True)) in done_calls:
            rejected.append(f"{step.id}: repeats a completed call")
        else:
            fresh.append(step)
    return ReplanDecision(done=not fresh, new_steps=fresh, rationale=str(data.get("rationale") or ""),
                          rejected=rejected)


# --------------------------------------------------------------------------------------------------
# Sections and step replies
# --------------------------------------------------------------------------------------------------

_SECTION_MARKER = re.compile(r"^\s*={2,}\s*SECTION\s*:\s*([A-Za-z0-9_ -]+?)\s*={2,}\s*$", re.MULTILINE)


def parse_sections(text: str | None, expected: list[str]) -> dict[str, str]:
    """Split a synthesis reply on ``=== SECTION: key ===`` markers.

    Keys are normalised (``Executive Summary`` -> ``executive_summary``). Unexpected keys are kept;
    the report generator lists them as extra sections.
    """
    text = normalize_text(text)
    if not text:
        return {}
    matches = list(_SECTION_MARKER.finditer(text))
    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        key = re.sub(r"[^a-z0-9]+", "_", m.group(1).lower()).strip("_")
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()
        if body:
            sections[key] = body
    if len(sections) <= 1 and expected:
        # The model ignored the markers and used Markdown headings ("## Financial Risks"): split on
        # headings that name an expected section, so the text lands in the right sections.
        by_heading = _split_on_headings(text, expected)
        if len(by_heading) > len(sections):
            return by_heading
    if not sections and text.strip() and expected:
        # No structure at all: keep the text rather than lose it, under the first section.
        sections[expected[0]] = text.strip()
    return sections


def _split_on_headings(text: str, expected: list[str]) -> dict[str, str]:
    wanted = {key.replace("_", " "): key for key in expected}
    heading = re.compile(r"^\s*#{1,4}\s*(.+?)\s*#*\s*$", re.MULTILINE)
    starts = []
    for m in heading.finditer(text):
        name = re.sub(r"[^a-z0-9 ]+", " ", m.group(1).lower())
        name = re.sub(r"\s+", " ", name).strip()
        if name in wanted:
            starts.append((m.start(), m.end(), wanted[name]))
    sections: dict[str, str] = {}
    for i, (_, end, key) in enumerate(starts):
        stop = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        body = text[end:stop].strip()
        if body:
            sections[key] = body
    return sections


def step_done_note(text: str | None) -> str | None:
    """The explanation after ``STEP_DONE:``, or None if the reply is not a step-done message."""
    text = normalize_text(text)
    if not text:
        return None
    m = re.search(r"STEP_DONE\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else None
