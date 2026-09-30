"""engine: build the claim ledger from tool evidence.

Why this exists: the writer must not pick numbers out of raw tool output. This module extracts every
structured figure (financial_data_api rows, peer_comparison matrix rows) into normalised claims: money
in base USD, ratios as fractions, one explicit fiscal period each. It then hands them to the
conflict resolver, so the report is written from one reconciled, confidence-scored ledger.

Only structured sources are extracted. Free text (news, web) is never parsed for numbers here; that
keeps the ledger deterministic and stops a headline's "$3bn" from masquerading as a filed figure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from synthesis.conflict_resolver import (
    Claim,
    ConflictRecord,
    Period,
    ReliabilityConfig,
    resolve,
    tier_for_source,
    values_agree,
)
from tools.base import ToolResult

_PEER_MONEY = ("revenue",)  # peer_comparison reports revenue in USD millions
_PEER_RATIOS = ("revenue_growth", "gross_margin", "operating_margin", "net_margin", "roe")
_DERIVED_MARGINS = (("gross_margin", "gross_profit"), ("operating_margin", "operating_income"),
                    ("net_margin", "net_income"))

_MILLIONS = 1e6


def _period(row: dict[str, Any]) -> Period:
    return Period(fiscal_year=row.get("fiscal_year"), fiscal_period=row.get("fiscal_period") or "FY",
                  period_end=row.get("period_end"))


def _tier(result: ToolResult, provider: str) -> int:
    tier = tier_for_source(result.tool, result.source_tier)
    # yfinance repackages market data: never let it claim the filing-grade tier of the tool default.
    return max(tier, 2) if provider == "yfinance" else tier


def extract_claims(evidence: dict[str, ToolResult], config: ReliabilityConfig) -> list[Claim]:
    """Deterministic claims from structured tool results, ids C1..Cn in evidence order."""
    del config  # tiers come from the tool envelope; kept in the signature for the caller's symmetry
    raw: list[Claim] = []

    def add(entity: str, metric: str, period: Period, value: float, unit: str, result: ToolResult,
            provider: str) -> None:
        raw.append(Claim(
            claim_id="", entity=entity.upper(), metric=metric, period=period, value=value, unit=unit,
            evidence_ids=[result.evidence_id], source_tiers=[_tier(result, provider)], provider=provider,
            is_fallback=result.is_fallback))

    for result in evidence.values():
        data = result.data
        if result.tool == "financial_data_api":
            ticker = data.get("ticker") or result.args.get("ticker")
            if not ticker:
                continue
            provider = data.get("provider") or "mock"
            unit = data.get("unit")
            for row in data.get("data", []):
                period = _period(row)
                values = row.get("values", {})
                for metric, v in values.items():
                    if v is None:
                        continue
                    if unit == "millions" and metric == "eps_diluted":
                        add(ticker, metric, period, float(v), "USD/share", result, provider)
                    elif unit == "millions":
                        add(ticker, metric, period, float(v) * _MILLIONS, "USD", result, provider)
                    elif unit == "ratio":
                        add(ticker, metric, period, float(v), "ratio", result, provider)
                    else:
                        add(ticker, metric, period, float(v), "count", result, provider)
                revenue = values.get("revenue")
                if data.get("statement_type") == "income_statement" and revenue:
                    for margin, numerator in _DERIVED_MARGINS:
                        if values.get(numerator) is not None:
                            add(ticker, margin, period, round(values[numerator] / revenue, 4), "ratio",
                                result, provider)
        elif result.tool == "peer_comparison":
            for row in data.get("matrix", []):
                if not row.get("ticker") or row.get("fiscal_year") is None:
                    continue  # never guess a period
                period = Period(fiscal_year=row["fiscal_year"], fiscal_period="FY",
                                period_end=row.get("period_end"))
                provider = "peer_comparison:" + (row.get("provider") or "sec_xbrl")
                values = row.get("values", {})
                for metric in _PEER_MONEY:
                    if values.get(metric) is not None:
                        add(row["ticker"], metric, period, float(values[metric]) * _MILLIONS, "USD", result,
                            provider)
                for metric in _PEER_RATIOS:
                    if values.get(metric) is not None:
                        add(row["ticker"], metric, period, float(values[metric]), "ratio", result, provider)

    for n, claim in enumerate(raw, start=1):
        claim.claim_id = f"C{n}"
    return raw


@dataclass
class Ledger:
    claims: list[Claim] = field(default_factory=list)
    conflicts: list[ConflictRecord] = field(default_factory=list)

    def for_entity(self, ticker: str) -> list[Claim]:
        return [c for c in self.claims if c.entity == ticker.upper()]

    def get(self, entity: str, metric: str, fiscal_year: int | None = None) -> Claim | None:
        """The annual claim for a metric: the given fiscal year, or the latest one when omitted."""
        found = [c for c in self.for_entity(entity)
                 if c.metric == metric and c.period.fiscal_period == "FY" and c.period.fiscal_year is not None
                 and (fiscal_year is None or c.period.fiscal_year == fiscal_year)]
        return max(found, key=lambda c: c.period.fiscal_year or 0) if found else None

    def summary(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for c in self.claims:
            by_status[c.status] = by_status.get(c.status, 0) + 1
        return {"claims": len(self.claims), "by_status": by_status, "conflicts": len(self.conflicts),
                "conflict_methods": sorted({c.method for c in self.conflicts})}


def build_ledger(evidence: dict[str, ToolResult], config: ReliabilityConfig | None = None) -> Ledger:
    config = config or ReliabilityConfig()
    claims, conflicts = resolve(extract_claims(evidence, config), config)
    return Ledger(claims=claims, conflicts=conflicts)


def triangulate(value: float, sources: list[tuple[float, str]], tolerance: float) -> dict[str, Any]:
    """Compare ``value`` with other sources' values: who agrees, who does not, and the overall range."""
    agreeing = [sid for v, sid in sources if values_agree(value, v, "USD", tolerance)]
    disagreeing = [sid for v, sid in sources if not values_agree(value, v, "USD", tolerance)]
    everything = [value] + [v for v, _ in sources]
    return {"agreeing": agreeing, "disagreeing": disagreeing, "range": (min(everything), max(everything))}
