"""Resolve symbolic calculation inputs ("s2.revenue_FY2025") to the numbers they name.

Planners write calculation steps before any data exists, so they refer to figures by name:
``{"current": "s2.revenue_FY2026", "previous": "s2.revenue_FY2025"}``. The calculation engine
needs numbers (principle P2: numbers are computed, not generated). Day 11 evaluation found that
leaving the repair to the LLM failed in two of eight live runs, so derived figures went missing.

This module looks each name up in the structured financial data the step depends on. If the named
fiscal year is not reported yet (FY2026 before the 10-K is filed), every year in the step is moved
back by the same number of years, so a growth rate stays a like-for-like comparison, and the note
says so. Anything it cannot resolve unambiguously is left to the LLM, as before.
"""

from __future__ import annotations

import math
import re
from typing import Any

_STEP_PREFIX = re.compile(r"^\s*[a-z]+\d+\s*[.:]\s*", re.IGNORECASE)
_YEAR = re.compile(r"(?:fy|fiscal(?:\s+year)?)?[\s_\-]*((?:19|20)\d{2})\b", re.IGNORECASE)
_QUARTER = re.compile(r"\bq([1-4])\b", re.IGNORECASE)
#: Words that name a segment or another line item, so the consolidated figure would be wrong.
_TOO_SPECIFIC = re.compile(r"\b(?:segment|aws|azure|cloud|services|iphone|mac|ipad|automotive|energy|division)\b",
                           re.IGNORECASE)


def _records(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Period rows of a financial_data_api result: ``{"fiscal_year", "fiscal_period", "values"}``."""
    rows = data.get("data") if isinstance(data, dict) else None
    return [r for r in rows or [] if isinstance(r, dict) and isinstance(r.get("values"), dict)]


def _parse(ref: str, metrics: set[str]) -> tuple[str, int, str] | None:
    """``(metric, year, quarter)`` named by ``ref``, or None if it is not a plain reference."""
    text = _STEP_PREFIX.sub("", ref).replace("_", " ").strip()
    if _TOO_SPECIFIC.search(text):
        return None
    year = _YEAR.search(text)
    if not year:
        return None
    quarter = _QUARTER.search(text)
    words = f" {re.sub(r'[^a-z ]', ' ', text.lower())} "
    named = [m for m in metrics if f" {m.replace('_', ' ')} " in words]
    if not named:
        return None
    metric = max(named, key=len)  # "gross profit" rather than "profit"
    return metric, int(year.group(1)), f"Q{quarter.group(1)}" if quarter else "FY"


def resolve_references(args: dict[str, Any], sources: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    """Replace named inputs in calculation_engine ``args`` with numbers from ``sources``.

    ``sources`` are the data dicts of the financial_data_api results the step depends on. Returns
    ``(new_args, note)``; ``new_args`` is None when nothing needed or could be resolved.
    """
    inputs = args.get("inputs") if isinstance(args, dict) else None
    if not isinstance(inputs, dict):
        return None, ""
    named = {k: v for k, v in inputs.items() if isinstance(v, str)}
    if not named:
        return None, ""

    # One table of (metric, year, period) -> value; a figure two sources disagree on is ambiguous.
    table: dict[tuple[str, int, str], float | None] = {}
    for data in sources:
        for row in _records(data):
            key_year, period = row.get("fiscal_year"), str(row.get("fiscal_period") or "FY").upper()
            for metric, value in row["values"].items():
                if not isinstance(value, (int, float)) or not isinstance(key_year, int):
                    continue
                key = (metric, key_year, period)
                # Float noise between providers (4475.45 vs 4475.450000001) is the same figure, not a conflict.
                seen = table.get(key, value)
                table[key] = seen if seen is not None and math.isclose(seen, value, rel_tol=1e-9) else None
    if not table:
        return None, ""
    metrics = {m for m, _, _ in table}

    parsed = {k: _parse(v, metrics) for k, v in named.items()}
    if any(p is None for p in parsed.values()):
        return None, ""

    # Shift every year by the same amount when the newest one asked for is not reported yet.
    shift = 0
    for metric, year, period in parsed.values():  # type: ignore[misc]
        if (metric, year, period) not in table:
            latest = max((y for m, y, p in table if m == metric and p == period), default=None)
            if latest is None or latest >= year:
                return None, ""
            shift = max(shift, year - latest)
    resolved, used = dict(inputs), []
    for name, (metric, year, period) in parsed.items():  # type: ignore[misc]
        value = table.get((metric, year - shift, period))
        if value is None:
            return None, ""
        resolved[name] = value
        used.append(f"{name} = {metric} {'FY' if period == 'FY' else period + ' '}{year - shift}")
    note = "inputs taken from the financial data: " + ", ".join(used)
    if shift:
        note += f" (the requested year is not reported yet, so every year was moved back {shift})"
    return {**args, "inputs": resolved}, note
