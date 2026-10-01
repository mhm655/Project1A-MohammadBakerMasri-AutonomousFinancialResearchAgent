"""Symbolic calculation inputs are resolved from the financial data (Day 12 fix, agent/references.py)."""

from __future__ import annotations

from agent.references import resolve_references

PLTR = {"data": [
    {"fiscal_year": 2025, "fiscal_period": "FY", "values": {"revenue": 4475.45, "gross_profit": 3686.27}},
    {"fiscal_year": 2024, "fiscal_period": "FY", "values": {"revenue": 2865.51, "gross_profit": 2299.52}},
]}


def _calc(**inputs):
    return {"calculation_type": "growth_rate", "inputs": inputs}


def test_names_are_replaced_by_the_figures_they_name():
    args, note = resolve_references(_calc(current="s2.revenue_FY2025", previous="revenue FY2024"), [PLTR])
    assert args["inputs"] == {"current": 4475.45, "previous": 2865.51}
    assert "revenue FY2025" in note and "moved back" not in note


def test_unreported_year_moves_every_year_back_together():
    args, note = resolve_references(
        {"calculation_type": "margin", "inputs": {"numerator": "s2.gross_profit_FY2026", "revenue": "s2.revenue_FY2026"}},
        [PLTR])
    assert args["inputs"] == {"numerator": 3686.27, "revenue": 4475.45}
    assert "moved back 1" in note


def test_numbers_segments_and_unknown_metrics_are_left_alone():
    assert resolve_references(_calc(current=1.0, previous=2.0), [PLTR]) == (None, "")
    assert resolve_references(_calc(current="AWS FY2025 revenue", previous="AWS FY2024 revenue"), [PLTR])[0] is None
    assert resolve_references(_calc(current="s2.free_cash_flow_FY2025", previous=1.0), [PLTR])[0] is None
    assert resolve_references(_calc(current="revenue", previous="revenue FY2024"), [PLTR])[0] is None  # no year


def test_two_sources_disagreeing_on_a_figure_is_not_guessed():
    other = {"data": [{"fiscal_year": 2025, "fiscal_period": "FY", "values": {"revenue": 99.0}},
                      {"fiscal_year": 2024, "fiscal_period": "FY", "values": {"revenue": 2865.51}}]}
    assert resolve_references(_calc(current="revenue FY2025", previous="revenue FY2024"), [PLTR, other])[0] is None


def test_float_noise_between_sources_is_not_a_disagreement():
    noisy = {"data": [{"fiscal_year": 2025, "fiscal_period": "FY", "values": {"revenue": 4475.450000001}},
                      {"fiscal_year": 2024, "fiscal_period": "FY", "values": {"revenue": 2865.51}}]}
    args, _ = resolve_references(_calc(current="revenue FY2025", previous="revenue FY2024"), [PLTR, noisy])
    assert args["inputs"]["current"] == 4475.45


def test_quarterly_negative_and_metric_name_references():
    data = {"data": [
        {"fiscal_year": 2025, "fiscal_period": "FY", "values": {"net_income": -5.0, "operating_income": 20.0}},
        {"fiscal_year": 2024, "fiscal_period": "FY", "values": {"net_income": -10.0}},
        {"fiscal_year": 2025, "fiscal_period": "Q3", "values": {"revenue": 30.0}},
        {"fiscal_year": 2025, "fiscal_period": "Q2", "values": {"revenue": 25.0}}]}
    q, _ = resolve_references(_calc(current="s2.revenue_Q3_FY2025", previous="revenue Q2 FY2025"), [data])
    assert q["inputs"] == {"current": 30.0, "previous": 25.0}
    neg, _ = resolve_references(_calc(current="net income (FY2025)", previous="FY2024 net income"), [data])
    assert neg["inputs"] == {"current": -5.0, "previous": -10.0}
    # "income" alone could be operating or net income: never guess.
    assert resolve_references(_calc(current="income FY2025", previous="income FY2024"), [data])[0] is None
