"""financial_data_api: structured financial statements and key ratios.

Day 2: mock handler backed by :mod:`tools.mock_data`.
Day 5: live handler (SEC XBRL company facts, with yfinance as the secondary source).
"""

from __future__ import annotations

from typing import Any

from tools.base import ToolExecutionError
from tools.mock_data import ANNUAL, MOCK_NOTICE, PROFILES, QUARTERLY_INCOME

TOOL = "financial_data_api"


def key_ratios(income: dict[str, float], balance: dict[str, float] | None = None,
               cash: dict[str, float] | None = None) -> dict[str, float | None]:
    """Derive standard ratios from statement line items (fractions, rounded to 4 d.p.)."""
    revenue = income.get("revenue")

    def safe(numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or not denominator:
            return None
        return round(numerator / denominator, 4)

    ratios = {
        "gross_margin": safe(income.get("gross_profit"), revenue),
        "operating_margin": safe(income.get("operating_income"), revenue),
        "net_margin": safe(income.get("net_income"), revenue),
    }
    if balance:
        ratios["roe"] = safe(income.get("net_income"), balance.get("shareholders_equity"))
        ratios["debt_to_equity_proxy"] = safe(balance.get("total_liabilities"), balance.get("shareholders_equity"))
    if cash:
        ratios["fcf_margin"] = safe(cash.get("free_cash_flow"), revenue)
    return ratios


def mock_financial_data_api(ticker: str, statement_type: str, period: str = "annual", years: int = 3) -> dict[str, Any]:
    ticker = ticker.upper()
    if ticker not in ANNUAL:
        raise ToolExecutionError(f"{TOOL}: no financial data for ticker {ticker!r}", tool=TOOL)

    notes = [MOCK_NOTICE]
    if period == "annual":
        periods = ANNUAL[ticker][:years]
        if len(periods) < years:
            notes.append(f"Only {len(periods)} annual periods available.")
        rows = []
        for p in periods:
            if statement_type == "key_ratios":
                values = key_ratios(p["income_statement"], p["balance_sheet"], p["cash_flow"])
            else:
                values = dict(p[statement_type])
            rows.append({"fiscal_year": p["fiscal_year"], "fiscal_period": "FY", "period_end": p["period_end"], "values": values})
    else:
        if statement_type not in ("income_statement", "key_ratios") or ticker not in QUARTERLY_INCOME:
            raise ToolExecutionError(
                f"{TOOL}: quarterly {statement_type} not available for {ticker} in mock mode", tool=TOOL
            )
        quarters = QUARTERLY_INCOME[ticker][: years * 4]
        rows = [
            {
                "fiscal_year": q["fiscal_year"],
                "fiscal_period": q["fiscal_period"],
                "period_end": q["period_end"],
                "values": key_ratios(q["income_statement"]) if statement_type == "key_ratios" else dict(q["income_statement"]),
            }
            for q in quarters
        ]

    return {
        "ticker": ticker,
        "company_name": PROFILES[ticker]["name"],
        "statement_type": statement_type,
        "period": period,
        "currency": "USD",
        "unit": "ratio" if statement_type == "key_ratios" else "millions",
        "fiscal_year_end": PROFILES[ticker]["fiscal_year_end"],
        "data": rows,
        "notes": notes,
    }
