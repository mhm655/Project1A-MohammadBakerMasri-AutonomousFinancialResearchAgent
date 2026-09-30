"""peer_comparison: identify peers and compare them on standard metrics.

Live handler (Day 7), free and keyless:

* **Peers** are the largest companies in the same yfinance industry (``yf.Industry(...).top_companies``,
  ordered by market weight), or the tickers the caller passes in ``peers`` (for comparisons the
  industry classification does not capture, such as Amazon against Microsoft in cloud computing).
* **Metrics** come from each company's latest annual SEC XBRL figures (through ``financial_data_api``)
  and its market capitalisation from yfinance. Companies' fiscal years end on different dates, so every
  row says which fiscal year it is.

A mock handler with canned data is kept for tests and offline runs.
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
EXTRA_CANDIDATES = 3
MIN_RELATIVE_WEIGHT = 0.01
UNITS_NOTE = ("Units: revenue in USD millions, market_cap in USD, pe_ratio in times earnings; growth, margins "
              "and ROE are fractions.")


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


def _rankings(matrix: list[dict[str, Any]], metrics: list[str]) -> dict[str, list[str]]:
    # Highest value first; companies with no value (e.g. P/E with negative earnings) go last.
    return {m: [row["ticker"] for row in sorted(matrix, key=lambda r: (r["values"][m] is None, -(r["values"][m] or 0)))]
            for m in metrics}


def _explicit_peers(ticker: str, peers: list[str] | None, num_peers: int) -> list[str]:
    chosen: list[str] = []
    for p in peers or []:
        p = p.upper()
        if p != ticker and p not in chosen:
            chosen.append(p)
    return chosen[:num_peers]


def mock_peer_comparison(ticker: str, num_peers: int = 5, metrics: list[str] | None = None,
                         peers: list[str] | None = None) -> dict[str, Any]:
    ticker = ticker.upper()
    if peers:
        peers = _explicit_peers(ticker, peers, num_peers)
    elif ticker in PEER_MAP:
        peers = PEER_MAP[ticker][:num_peers]
    else:
        raise ToolExecutionError(f"{TOOL}: no peer set for ticker {ticker!r}", tool=TOOL)
    metrics = list(metrics or METRICS)

    matrix = []
    for t in [ticker, *peers]:
        name, values = _metrics_for(t)
        matrix.append({"ticker": t, "name": name, "values": {m: values[m] for m in metrics}})

    notes = [MOCK_NOTICE, "Rankings are ordered highest value first."]
    if len(peers) < num_peers:
        notes.append(f"Only {len(peers)} peers available.")
    return {"ticker": ticker, "peers": peers, "metrics": metrics, "matrix": matrix,
            "rankings": _rankings(matrix, metrics), "notes": notes}


# --------------------------------------------------------------------------------------------------
# Live handler
# --------------------------------------------------------------------------------------------------


def industry_peers(ticker: str, num_peers: int) -> tuple[list[str], str]:
    """``(peers, description)``: the largest other companies in ``ticker``'s yfinance industry."""
    from tools.company_profile import yfinance_info

    info = yfinance_info(ticker)
    key = info.get("industryKey")
    if not key:
        raise ToolExecutionError(f"{TOOL}: yfinance has no industry for {ticker}; pass the peers explicitly",
                                 tool=TOOL, retryable=True)
    try:
        import yfinance as yf

        top = yf.Industry(key).top_companies
    except Exception as exc:  # unofficial API
        raise ToolExecutionError(f"{TOOL}: yfinance industry lookup failed ({type(exc).__name__}); "
                                 "pass the peers explicitly", tool=TOOL, retryable=True) from exc
    if top is None or top.empty:
        raise ToolExecutionError(f"{TOOL}: no companies listed for industry {key!r}; pass the peers explicitly",
                                 tool=TOOL)
    own_name = (info.get("longName") or "").strip().lower()
    weights = {str(s).upper(): _weight(row.get("market weight")) for s, row in top.iterrows()}
    own_weight = weights.get(ticker)
    peers: list[str] = []
    for symbol, row in top.iterrows():
        symbol = str(symbol).upper()
        name = str(row.get("name") or "").strip().lower()
        # Skip the company itself, including its other share classes (GOOG for GOOGL).
        if symbol == ticker or (own_name and name == own_name) or symbol in peers:
            continue
        # Skip companies too small to compare with: no market weight at all (yfinance lists China
        # Yuchai, a diesel-engine maker, among auto manufacturers without one), or under 1 % of the
        # company's own weight.
        weight = weights[symbol]
        if weight is None or (own_weight and weight < own_weight * MIN_RELATIVE_WEIGHT):
            continue
        peers.append(symbol)
    industry = info.get("industry") or key
    # A few spares, in case some candidates have no usable financial data.
    return peers[:num_peers + EXTRA_CANDIDATES], \
        f"Peers: the largest companies by market weight in the yfinance industry '{industry}' " \
        "(companies under 1 % of this company's weight are left out)."


def _weight(value: Any) -> float | None:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return None
    return None if weight != weight or weight <= 0 else weight  # NaN != NaN


def live_metrics(ticker: str) -> dict[str, Any]:
    """Latest annual metrics of one company: XBRL statements plus the yfinance market cap."""
    from tools.company_profile import yfinance_info
    from tools.financial_api import live_financial_data_api

    income = live_financial_data_api(ticker, "income_statement", "annual", 2)
    balance = live_financial_data_api(ticker, "balance_sheet", "annual", 1)
    rows = income["data"]
    if not rows or rows[0]["values"].get("revenue") is None:
        raise ToolExecutionError(f"{TOOL}: no annual revenue for {ticker}", tool=TOOL)
    latest = rows[0]["values"]
    bal = next((r["values"] for r in balance["data"] if r["period_end"] == rows[0]["period_end"]), {})
    ratios = key_ratios(latest, bal)
    prior_revenue = rows[1]["values"].get("revenue") if len(rows) > 1 else None
    info = yfinance_info(ticker)
    market_cap = float(info["marketCap"]) if isinstance(info.get("marketCap"), int | float) and info["marketCap"] > 0 \
        else None
    net_income = latest.get("net_income")
    values = {
        "revenue": latest["revenue"],
        "revenue_growth": round(latest["revenue"] / prior_revenue - 1, 4) if prior_revenue else None,
        "gross_margin": ratios["gross_margin"],
        "operating_margin": ratios["operating_margin"],
        "net_margin": ratios["net_margin"],
        "roe": ratios.get("roe"),
        # Trailing P/E on the last full fiscal year's earnings, not the last twelve months.
        "pe_ratio": round(market_cap / (net_income * 1e6), 1) if market_cap and net_income and net_income > 0 else None,
        "market_cap": market_cap,
    }
    name = info.get("longName") if income.get("provider") == "yfinance" else income["company_name"]
    return {"ticker": ticker, "name": name or ticker, "fiscal_year": rows[0]["fiscal_year"],
            "period_end": rows[0]["period_end"], "provider": income.get("provider", ""), "values": values}


def live_peer_comparison(ticker: str, num_peers: int = 5, metrics: list[str] | None = None,
                         peers: list[str] | None = None) -> dict[str, Any]:
    ticker = ticker.upper()
    metrics = list(metrics or METRICS)
    if peers:
        peers = _explicit_peers(ticker, peers, num_peers)
        how = "Peers: chosen by the caller."
    else:
        peers, how = industry_peers(ticker, num_peers)

    notes = [how]
    matrix = []
    for t in [ticker, *peers]:
        if len(matrix) > num_peers:
            break
        try:
            row = live_metrics(t)
        except ToolExecutionError as exc:
            if t == ticker:
                raise
            notes.append(f"{t} left out: {str(exc).split(': ', 1)[-1]}")
            continue
        row["values"] = {m: row["values"][m] for m in metrics}
        matrix.append(row)
    if len(matrix) < 2:
        raise ToolExecutionError(f"{TOOL}: no peer of {ticker} has usable financial data", tool=TOOL)

    notes += [
        "Sources: SEC XBRL annual figures (tier 1; yfinance where a company has none) and yfinance market caps.",
        "Each row is the company's latest fiscal year: " + ", ".join(
            f"{r['ticker']} FY{r['fiscal_year']} (ended {r['period_end']})" for r in matrix) + ".",
        "P/E is market cap over the last fiscal year's net income.",
        UNITS_NOTE,
        "Rankings are ordered highest value first.",
    ]
    return {"ticker": ticker, "peers": [r["ticker"] for r in matrix[1:]], "metrics": metrics, "matrix": matrix,
            "rankings": _rankings(matrix, metrics), "notes": notes}
