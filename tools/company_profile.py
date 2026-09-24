"""company_profile: basic company facts (sector, industry, size, executives, fiscal calendar).

Day 2: mock handler. Day 7: live handler (yfinance, falling back to EDGAR submissions metadata).
"""

from __future__ import annotations

from typing import Any

from tools.base import ToolExecutionError
from tools.mock_data import MOCK_NOTICE, PROFILES, STATIC_METRICS

TOOL = "company_profile"


def mock_company_profile(ticker: str) -> dict[str, Any]:
    ticker = ticker.upper()
    if ticker in PROFILES:
        return {"ticker": ticker, **PROFILES[ticker], "notes": [MOCK_NOTICE]}
    if ticker in STATIC_METRICS:
        # Peers only have summary data in mock mode.
        return {
            "ticker": ticker,
            "name": STATIC_METRICS[ticker]["name"],
            "market_cap": STATIC_METRICS[ticker]["market_cap"],
            "notes": [MOCK_NOTICE, "Partial profile: only name and market cap available in mock mode."],
        }
    raise ToolExecutionError(f"{TOOL}: unknown ticker {ticker!r}", tool=TOOL)
