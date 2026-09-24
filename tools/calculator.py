"""calculation_engine: deterministic financial calculations with visible intermediate steps.

Research principle P2 ("numbers are computed, not generated"): the LLM never does arithmetic for the
report. It calls this tool, and the returned ``steps`` make every derived figure auditable.

This tool has no external dependencies, so the same implementation serves mock and live modes.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from typing import Any

from tools.base import ToolValidationError

TOOL = "calculation_engine"


def _num(inputs: dict[str, Any], key: str, *, positive: bool = False, nonzero: bool = False) -> float:
    if key not in inputs:
        raise ToolValidationError(f"{TOOL}: missing input {key!r}", tool=TOOL)
    value = inputs[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ToolValidationError(f"{TOOL}: input {key!r} must be a finite number, got {value!r}", tool=TOOL)
    if positive and value <= 0:
        raise ToolValidationError(f"{TOOL}: input {key!r} must be > 0, got {value}", tool=TOOL)
    if nonzero and value == 0:
        raise ToolValidationError(f"{TOOL}: input {key!r} must not be zero", tool=TOOL)
    return float(value)


def _num_list(inputs: dict[str, Any], key: str, *, min_len: int = 1) -> list[float]:
    values = inputs.get(key)
    if not isinstance(values, list) or len(values) < min_len:
        raise ToolValidationError(f"{TOOL}: input {key!r} must be a list of at least {min_len} numbers", tool=TOOL)
    return [_num({key: v}, key) for v in values]


def _growth(inputs: dict[str, Any]) -> tuple[float, list[str]]:
    current, previous = _num(inputs, "current"), _num(inputs, "previous", nonzero=True)
    result = (current - previous) / abs(previous)
    return result, [
        f"change = current - previous = {current:g} - {previous:g} = {current - previous:g}",
        f"growth = change / |previous| = {current - previous:g} / {abs(previous):g} = {result:.6f}",
    ]


def _cagr(inputs: dict[str, Any]) -> tuple[float, list[str]]:
    begin = _num(inputs, "begin_value", positive=True)
    end = _num(inputs, "end_value")
    years = _num(inputs, "years", positive=True)
    if end < 0:
        raise ToolValidationError(f"{TOOL}: CAGR is undefined for a negative end_value", tool=TOOL)
    result = (end / begin) ** (1 / years) - 1
    return result, [
        f"ratio = end / begin = {end:g} / {begin:g} = {end / begin:.6f}",
        f"CAGR = ratio^(1/{years:g}) - 1 = {result:.6f}",
    ]


def _ratio(inputs: dict[str, Any], label: str = "ratio", denominator_key: str = "denominator") -> tuple[float, list[str]]:
    numerator = _num(inputs, "numerator")
    denominator = _num(inputs, denominator_key, nonzero=True)
    result = numerator / denominator
    return result, [f"{label} = {numerator:g} / {denominator:g} = {result:.6f}"]


def _margin(inputs: dict[str, Any]) -> tuple[float, list[str]]:
    return _ratio(inputs, "margin", "revenue")


def _dcf(inputs: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    flows = _num_list(inputs, "cash_flows")
    rate = _num(inputs, "discount_rate")
    growth = _num(inputs, "terminal_growth_rate")
    if rate <= -1:
        raise ToolValidationError(f"{TOOL}: discount_rate must be greater than -1", tool=TOOL)
    if rate <= growth:
        raise ToolValidationError(
            f"{TOOL}: discount_rate ({rate}) must exceed terminal_growth_rate ({growth}) for a finite terminal value",
            tool=TOOL,
        )
    steps = []
    pv_flows = 0.0
    for t, cf in enumerate(flows, start=1):
        pv = cf / (1 + rate) ** t
        pv_flows += pv
        steps.append(f"PV(CF{t}) = {cf:g} / (1 + {rate:g})^{t} = {pv:.4f}")
    n = len(flows)
    terminal = flows[-1] * (1 + growth) / (rate - growth)
    pv_terminal = terminal / (1 + rate) ** n
    enterprise_value = pv_flows + pv_terminal
    steps += [
        f"terminal value = CF{n} x (1 + g) / (r - g) = {flows[-1]:g} x {1 + growth:g} / {rate - growth:g} = {terminal:.4f}",
        f"PV(terminal) = {terminal:.4f} / (1 + {rate:g})^{n} = {pv_terminal:.4f}",
        f"enterprise value = sum PV(CF) + PV(terminal) = {pv_flows:.4f} + {pv_terminal:.4f} = {enterprise_value:.4f}",
    ]
    result = {
        "pv_cash_flows": pv_flows,
        "terminal_value": terminal,
        "pv_terminal_value": pv_terminal,
        "enterprise_value": enterprise_value,
    }
    if "net_debt" in inputs:
        equity = enterprise_value - _num(inputs, "net_debt")
        result["equity_value"] = equity
        steps.append(f"equity value = EV - net debt = {enterprise_value:.4f} - {inputs['net_debt']:g} = {equity:.4f}")
        if "shares_outstanding" in inputs:
            shares = _num(inputs, "shares_outstanding", positive=True)
            result["value_per_share"] = equity / shares
            steps.append(f"value per share = {equity:.4f} / {shares:g} = {equity / shares:.4f}")
    return result, steps


def _stats(inputs: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    values = _num_list(inputs, "values", min_len=1)
    result = {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }
    return result, [f"n = {len(values)}", "stdev is the sample standard deviation (n - 1)"]


_CALCULATIONS: dict[str, Callable[[dict[str, Any]], tuple[Any, list[str]]]] = {
    "growth_rate": _growth,
    "percent_change": _growth,
    "cagr": _cagr,
    "margin": _margin,
    "ratio": _ratio,
    "dcf": _dcf,
    "descriptive_stats": _stats,
}

#: Which inputs each calculation type expects; also used in the tool description.
INPUT_SPEC: dict[str, str] = {
    "growth_rate": "current, previous",
    "percent_change": "current, previous",
    "cagr": "begin_value, end_value, years",
    "margin": "numerator, revenue",
    "ratio": "numerator, denominator",
    "dcf": "cash_flows[], discount_rate, terminal_growth_rate, net_debt?, shares_outstanding?",
    "descriptive_stats": "values[]",
}


def calculation_engine(calculation_type: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Run one calculation. Rates are fractions: 0.12 means 12 %."""
    try:
        fn = _CALCULATIONS[calculation_type]
    except KeyError:
        raise ToolValidationError(f"{TOOL}: unknown calculation_type {calculation_type!r}", tool=TOOL) from None
    result, steps = fn(inputs)
    return {"calculation_type": calculation_type, "inputs": inputs, "result": result, "steps": steps}
