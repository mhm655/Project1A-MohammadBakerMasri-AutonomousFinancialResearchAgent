"""Gold-standard benchmark comparison (spec Day 11 task 3; metric BM-1 "Benchmark Fact Recall").

``evaluation/benchmarks/gold_facts.json`` holds the machine-checkable facts behind the gold summaries
(Microsoft C1, Apple C2, Tesla C3). :func:`compare` checks how many of them a run's report states
correctly and lists any numeric fact the report gets wrong ("conflicts").

Fact kinds:
  numeric      recalled when a quantity in the report matches within 1 % (or within the rounding of the
               digits the report printed, so "$4.4 billion" matches 4,355 million), and the same sentence
               or table row names a keyword and the period's year. Percent facts hold percentage points.
  entity       recalled when the name appears (case-insensitive), or all its words appear in one paragraph.
  qualitative  recalled when at least half of the keywords appear in one paragraph.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GOLD_PATH = Path(__file__).resolve().parent / "benchmarks" / "gold_facts.json"
RECALL_TOLERANCE = 0.01
CONFLICT_TOLERANCE = 0.02
TARGET_RECALL = 0.70

_gold_cache: dict[str, Any] | None = None


def _metric_result_cls() -> Any:
    try:
        from evaluation.metrics import MetricResult
        return MetricResult
    except ImportError:  # metrics module not available (yet): same fields, so callers can still read it
        @dataclass
        class MetricResult:  # type: ignore[no-redef]
            id: str
            name: str
            category: str
            value: float | None
            target: str
            passed: bool | None
            method: str
            unit: str
            detail: str = ""
            examples: list[Any] = field(default_factory=list)
        return MetricResult


def gold_for(challenge_id: str) -> dict[str, Any] | None:
    """Return the gold facts for a challenge id such as "C1", or None if it has no gold standard."""
    global _gold_cache
    if _gold_cache is None:
        try:
            _gold_cache = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _gold_cache = {}
    return _gold_cache.get(challenge_id)


# --- text units --------------------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z$(\"])")


def _units(text: str) -> list[tuple[str, str]]:
    """Split a report into (unit, context) pairs: sentences, bullets and table rows.

    For a table row the context also holds the table's header row, so a year given only in a column
    heading still counts for recall (conflict checks use the unit alone).
    """
    out: list[tuple[str, str]] = []
    header = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("|"):
            if not header:
                header = line
            if not re.fullmatch(r"[|\s:\-]+", line):
                out.append((line, f"{line} {header}"))
            continue
        header = ""
        for sentence in _SENTENCE_END.split(line):
            if sentence.strip():
                out.append((sentence, sentence))
    return out


def _paragraphs(text: str) -> list[str]:
    return [p.lower() for p in re.split(r"\n\s*\n", text) if p.strip()]


# --- numeric matching --------------------------------------------------------------------------------

def _half_ulp(q: Any) -> float:
    """Half the last printed digit of a quantity, in the same units as ``q.value``."""
    m = re.search(r"(\d[\d,]*)(?:\.(\d+))?", q.text)
    if not m:
        return 0.0
    number = float((m.group(1) + ("." + m.group(2) if m.group(2) else "")).replace(",", ""))
    if number == 0:
        return 0.0
    return 0.5 * 10 ** -len(m.group(2) or "") * abs(q.value / number)


def _candidates(q: Any, fact: dict[str, Any]) -> list[tuple[float, float]]:
    """(value, rounding slack) readings of a quantity in the units of the gold fact."""
    unit = fact.get("unit")
    slack = _half_ulp(q)
    if unit == "percent":
        return [(q.value * 100, slack * 100)] if q.kind == "percent" else []
    if q.kind != "amount":
        return []
    if unit == "USD/share":
        return [(q.value, slack)]
    # Dollar totals: an unscaled figure may come from a table headed "$ in millions" or "in thousands".
    scales = (1.0,) if q.scaled else (1.0, 1e3, 1e6, 1e9)
    return [(q.value * s, slack * s) for s in scales]


def _diff(reported: float, gold: float, unit: str | None) -> float:
    a, b = (abs(reported), abs(gold)) if unit == "percent" else (reported, gold)
    return abs(a - b)


def _matches(fact: dict[str, Any], q: Any, tolerance: float) -> bool:
    gold = float(fact["value"])
    return any(_diff(v, gold, fact.get("unit")) <= max(tolerance * abs(gold), slack)
               for v, slack in _candidates(q, fact))


def _plausible(fact: dict[str, Any], q: Any) -> bool:
    """A quantity of the right kind and order of magnitude, so unrelated numbers are not 'conflicts'."""
    gold = abs(float(fact["value"]))
    if gold == 0:
        return False
    return any(0.5 <= abs(v) / gold <= 2.0 for v, _ in _candidates(q, fact))


def _year(fact: dict[str, Any]) -> str:
    m = re.search(r"(?:19|20)\d{2}", str(fact.get("period") or ""))
    return m.group(0) if m else ""


def _mentions(text: str, fact: dict[str, Any]) -> bool:
    low = text.lower()
    year = _year(fact)
    return any(k.lower() in low for k in fact.get("keywords", [])) and (not year or year in low)


def _check_numeric(fact: dict[str, Any], units: list[tuple[str, str]]) -> tuple[bool, list[str]]:
    from tools.fact_checker import extract_quantities

    for unit, context in units:
        if _mentions(context, fact) and any(_matches(fact, q, RECALL_TOLERANCE) for q in extract_quantities(unit)):
            return True, []
    from evaluation.metrics import _stated_value

    conflicts: list[str] = []
    kind = "percent" if fact.get("unit") == "percent" else "amount"
    for unit, _ in units:
        if not _mentions(unit, fact):
            continue
        # A conflict needs the figure stated for the metric itself ("revenue was $X"), with no
        # segment qualifier: "Mac revenue ... up 29%" is not Apple's revenue growth.
        for keyword in fact.get("keywords") or []:
            stated = _stated_value(unit, keyword.lower(), kind)
            if stated is None or stated[0]:
                continue
            q = next((q for q in extract_quantities(unit) if q.kind == kind and q.value == stated[1]), None)
            if q is not None and _plausible(fact, q) and not _matches(fact, q, CONFLICT_TOLERANCE):
                conflicts.append(f"report says {q.text}, gold says {fact['text']} [{fact['id']}]")
            break
    return False, conflicts[:1]


# --- entity / qualitative ----------------------------------------------------------------------------

def _check_entity(fact: dict[str, Any], text: str, paragraphs: list[str]) -> bool:
    name = str(fact["value"]).lower()
    if name in text.lower():
        return True
    words = [w for w in re.findall(r"[\w'-]+", name) if len(w) > 1]  # "Amy E. Hood" still matches "Amy Hood"
    return bool(words) and any(all(w in p for w in words) for p in paragraphs)


def _check_qualitative(fact: dict[str, Any], paragraphs: list[str]) -> bool:
    keywords = [k.lower() for k in fact.get("keywords", [])]
    need = math.ceil(len(keywords) / 2)
    return bool(keywords) and any(sum(k in p for k in keywords) >= need for p in paragraphs)


# --- public ------------------------------------------------------------------------------------------

def compare(bundle: dict[str, Any], gold: dict[str, Any]) -> Any:
    """Score a run bundle against a gold fact set (BM-1). Returns an ``evaluation.metrics.MetricResult``."""
    text = str(bundle.get("report_markdown") or "")
    units, paragraphs = _units(text), _paragraphs(text)
    facts = gold.get("facts", [])
    recalled = 0
    conflicts: list[str] = []
    missed: list[str] = []
    for fact in facts:
        kind = fact.get("kind")
        if kind == "numeric":
            ok, found = _check_numeric(fact, units)
            conflicts += found
        elif kind == "entity":
            ok = _check_entity(fact, text, paragraphs)
        else:
            ok = _check_qualitative(fact, paragraphs)
        if ok:
            recalled += 1
        else:
            missed.append(fact["id"])
    n = len(facts)
    value = recalled / n if n else 0.0
    detail = f"{recalled} of {n} gold facts recalled; {len(conflicts)} conflicting"
    if missed:
        detail += f" (missed: {', '.join(missed)})"
    return _metric_result_cls()(
        id="BM-1", name="Benchmark Fact Recall", category="Benchmark", value=value,
        target=f">= {TARGET_RECALL:.0%}", passed=value >= TARGET_RECALL, method="benchmark", unit="ratio",
        detail=detail, examples=conflicts[:10],
    )
