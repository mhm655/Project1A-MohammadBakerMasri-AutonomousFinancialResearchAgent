"""Regression tests for defects that span two packages (found in the supervised audit)."""
from __future__ import annotations

from synthesis.conflict_resolver import ReliabilityConfig, resolve
from synthesis.engine import extract_claims
from tools.base import ToolResult


def test_unnumbered_yfinance_quarters_are_not_one_conflicting_period():
    quarters = [("2025-09-30", 100.0), ("2025-06-30", 90.0), ("2025-03-31", 80.0)]
    rows = [{"fiscal_year": 2025, "fiscal_period": "Q", "period_end": end, "values": {"revenue": value}}
            for end, value in quarters]
    data = {"ticker": "XYZ", "statement_type": "income_statement", "period": "quarterly", "unit": "millions",
            "provider": "yfinance", "data": rows}
    evidence = {"E1": ToolResult(evidence_id="E1", tool="financial_data_api", args={"ticker": "XYZ"}, data=data,
                                 source_tier=2, retrieved_at="2026-10-04T00:00:00Z", latency_ms=1.0,
                                 is_fallback=True)}
    config = ReliabilityConfig()

    resolved, conflicts = resolve(extract_claims(evidence, config), config)

    assert conflicts == []
    assert sorted(c.value for c in resolved if c.metric == "revenue") == [80e6, 90e6, 100e6]
