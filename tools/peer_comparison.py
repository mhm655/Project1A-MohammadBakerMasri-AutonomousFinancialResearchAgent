"""peer_comparison: identify peers and compare them on standard metrics.

Day 2: mock handler. Day 7: live handler (peers from the EDGAR SIC code, metrics from XBRL/yfinance).
"""

from __future__ import annotations

from typing import Any

from tools.base import ToolExecutionError
from tools.financial_api import key_ratios
from tools.mock_data import ANNUAL, MOCK_NOTICE, PEER_MAP, PROFILES, STATIC_METRICS

TOOL = "peer_comparison"

METRICS = (
    "revenue",
    "revenue_growth",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "roe",
    "pe_ratio",
    "market_cap",
)


def _metrics_for(ticker: str) -> tuple[str, dict[str, float | None]]:
    if ticker in ANNUAL:
        latest, prior = ANNUAL[ticker][0], ANNUAL[ticker][1]
        income = latest["income_statement"]
        ratios = key_ratios(income, latest["balance_sheet"])
        market_cap = PROFILES[ticker]["market_cap"]
        net_income_usd = income["net_income"] * 1e6
        values = {
            "revenue": income["revenue"],
            "revenue_growth": round(income["revenue"] / prior["income_statement"]["revenue"] - 1, 4),
            "gross_margin": ratios["gross_margin"],
            "operating_margin": ratios["operating_margin"],
            "net_margin": ratios["net_margin"],
            "roe": ratios["roe"],
            "pe_ratio": round(market_cap / net_income_usd, 1) if net_income_usd > 0 else None,
            "market_cap": market_cap,
        }
        return PROFILES[ticker]["name"], values
    if ticker in STATIC_METRICS:
        data = STATIC_METRICS[ticker]
        return data["name"], {m: data[m] for m in METRICS}
    raise ToolExecutionError(f"{TOOL}: no data for ticker {ticker!r}", tool=TOOL)


def mock_peer_comparison(ticker: str, num_peers: int = 5, metrics: list[str] | None = None) -> dict[str, Any]:
    ticker = ticker.upper()
    if ticker not in PEER_MAP:
        raise ToolExecutionError(f"{TOOL}: no peer set for ticker {ticker!r}", tool=TOOL)
    metrics = list(metrics or METRICS)
    peers = PEER_MAP[ticker][:num_peers]

    matrix = []
    for t in [ticker, *peers]:
        name, values = _metrics_for(t)
        matrix.append({"ticker": t, "name": name, "values": {m: values[m] for m in metrics}})

    # Highest value first; companies with no value (e.g. P/E with negative earnings) go last.
    rankings = {
        m: [row["ticker"] for row in sorted(matrix, key=lambda r: (r["values"][m] is None, -(r["values"][m] or 0)))]
        for m in metrics
    }
    notes = [MOCK_NOTICE, "Rankings are ordered highest value first."]
    if len(peers) < num_peers:
        notes.append(f"Only {len(peers)} peers available.")
    return {"ticker": ticker, "peers": peers, "metrics": metrics, "matrix": matrix, "rankings": rankings, "notes": notes}
