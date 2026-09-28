"""financial_data_api: structured financial statements and key ratios.

Live handler (Day 5): **SEC XBRL**, the figures companies file with their 10-K and 10-Q reports
(source tier 1). It is free and keyless. One small request per concept goes to
``data.sec.gov/api/xbrl/companyconcept``, and each response is cached for a day. The all-in-one
company-facts file is avoided because it runs to many megabytes.

* Annual values are durations of about a year from 10-K filings; quarterly values are durations of
  about three months. The fiscal Q4 is derived as FY - (Q1 + Q2 + Q3) and flagged as derived.
* When several filings report the same period, the **latest filing wins**, and any difference is
  reported as a restatement (spec A6.3, step 4).
* If a company has no usable XBRL data, **yfinance** (tier 2) is used instead, and the result says so.

A mock handler with canned data is kept for tests and offline runs.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from tools.base import ToolExecutionError
from tools.mock_data import ANNUAL, MOCK_NOTICE, PROFILES, QUARTERLY_INCOME

TOOL = "financial_data_api"

CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"

# Our line item -> XBRL concepts, in order of preference. Companies change concepts over the years
# (Microsoft reported "Revenues" until 2017), so all candidates are merged, the first one winning.
CONCEPTS: dict[str, list[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "shareholders_equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash_and_equivalents": ["CashAndCashEquivalentsAtCarryingValue",
                             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capital_expenditure": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
}
STATEMENTS = {
    "income_statement": ["revenue", "gross_profit", "operating_income", "net_income", "eps_diluted"],
    "balance_sheet": ["total_assets", "total_liabilities", "shareholders_equity", "cash_and_equivalents"],
    "cash_flow": ["operating_cash_flow", "capital_expenditure"],
}
INSTANT_ITEMS = set(STATEMENTS["balance_sheet"])  # point-in-time values, not durations
PER_SHARE_ITEMS = {"eps_diluted"}
RESTATEMENT_THRESHOLD = 0.005  # report a restatement when a later filing differs by more than 0.5 %


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


# --------------------------------------------------------------------------------------------------
# Live handler: SEC XBRL
# --------------------------------------------------------------------------------------------------


def _facts(cik: str, item: str) -> list[dict[str, Any]]:
    """Every reported value for a line item, merged across its candidate concepts."""
    from tools.http import get_http

    merged: dict[tuple[str | None, str, str], dict[str, Any]] = {}
    for rank, tag in enumerate(CONCEPTS[item]):
        try:
            data = get_http().get_json(CONCEPT_URL.format(cik=cik, tag=tag), ttl_s=86_400)
        except ToolExecutionError as exc:
            if "not found" in str(exc):
                continue  # this company does not use this concept
            raise
        unit = "USD/shares" if item in PER_SHARE_ITEMS else "USD"
        for fact in data.get("units", {}).get(unit, []):
            key = (fact.get("start"), fact["end"], fact["accn"])
            if key not in merged:  # an earlier (preferred) concept already supplied this fact
                merged[key] = {**fact, "concept": tag, "rank": rank}
    return list(merged.values())


def _days(fact: dict[str, Any]) -> int | None:
    if not fact.get("start"):
        return None
    return (date.fromisoformat(fact["end"]) - date.fromisoformat(fact["start"])).days


def _latest_by_period(facts: list[dict[str, Any]], keep, notes: list[str], label: str) -> dict[str, float]:
    """``period end -> value`` from the latest filing; differing earlier values are reported."""
    by_end: dict[str, list[dict[str, Any]]] = {}
    for fact in facts:
        if keep(fact):
            by_end.setdefault(fact["end"], []).append(fact)
    values: dict[str, float] = {}
    for end, group in by_end.items():
        group.sort(key=lambda f: (f["filed"], -f["rank"]))
        latest, first = group[-1], group[0]
        values[end] = latest["val"]
        if first["val"] and abs(latest["val"] - first["val"]) / abs(first["val"]) > RESTATEMENT_THRESHOLD:
            why = " (per-share figures also change after a stock split)" if label.startswith("eps") else ""
            notes.append(f"@{end}@Restated{why}: {label} for the period ended {end} was {first['val']:,} in the "
                         f"filing of {first['filed']} and {latest['val']:,} in the filing of {latest['filed']}; "
                         "the later figure is used.")
    return values


def _notes_for(notes: list[str], period_ends: set[str]) -> list[str]:
    """Keep general notes, and restatement notes only for the periods actually returned."""
    kept = []
    for note in notes:
        if note.startswith("@"):
            end, _, text = note[1:].partition("@")
            if end in period_ends:
                kept.append(text)
        else:
            kept.append(note)
    return kept


def _is_annual(f: dict[str, Any]) -> bool:
    days = _days(f)
    return f.get("form", "").startswith("10-K") and days is not None and 330 <= days <= 380


def _is_quarter(f: dict[str, Any]) -> bool:
    days = _days(f)
    return days is not None and 80 <= days <= 100


def _quarter_label(end: date, fy_ends: list[date]) -> tuple[int, str]:
    """(fiscal year, "Q1".."Q4") for a quarter ending on ``end``."""
    fy_end = next((e for e in sorted(fy_ends) if e >= end - timedelta(days=5)), None)
    if fy_end is None:  # a quarter of the fiscal year still in progress
        last = max(fy_ends)
        fy_end = date(last.year + 1, last.month, min(last.day, 28))
    quarters_before = round((fy_end - end).days / 91.3)
    return fy_end.year, f"Q{max(1, 4 - quarters_before)}"


def live_financial_data_api(ticker: str, statement_type: str, period: str = "annual", years: int = 3) -> dict[str, Any]:
    from tools.sec_edgar import resolve_cik, submissions

    ticker = ticker.upper()
    try:
        cik, title = resolve_cik(ticker)
    except ToolExecutionError:
        return _yfinance_statements(ticker, statement_type, period, years, reason="no SEC filer for this ticker")
    notes = ["Source: SEC XBRL financial data filed with the company's 10-K and 10-Q reports."]
    items = (STATEMENTS["income_statement"] + STATEMENTS["balance_sheet"] + STATEMENTS["cash_flow"]
             if statement_type == "key_ratios" else STATEMENTS[statement_type])
    if period == "quarterly" and statement_type in ("balance_sheet", "cash_flow"):
        raise ToolExecutionError(f"{TOOL}: quarterly {statement_type} is not supported; use period='annual'", tool=TOOL)

    revenue_facts = _facts(cik, "revenue")
    fy_ends = sorted({date.fromisoformat(e) for e in _latest_by_period(revenue_facts, _is_annual, [], "revenue")},
                     reverse=True)
    if not fy_ends:
        return _yfinance_statements(ticker, statement_type, period, years, reason="no annual revenue in SEC XBRL data")

    series: dict[str, dict[str, float]] = {}
    for item in items:
        facts = revenue_facts if item == "revenue" else _facts(cik, item)
        label = item.replace("_", " ")
        if item in INSTANT_ITEMS:
            series[item] = _latest_by_period(facts, lambda f: not f.get("start"), notes, label)
        elif period == "annual":
            series[item] = _latest_by_period(facts, _is_annual, notes, label)
        else:
            quarterly = _latest_by_period(facts, _is_quarter, notes, label)
            annual = _latest_by_period(facts, _is_annual, [], label)
            series[item] = _with_derived_q4(quarterly, annual, fy_ends, notes)

    if period == "annual":
        ends = fy_ends[:years]
    else:
        ends = sorted({date.fromisoformat(e) for e in series["revenue"]}, reverse=True)[: years * 4]
    rows = []
    for end in ends:
        key = end.isoformat()
        values = {item: _scale(item, series[item].get(key)) for item in items}
        if values.get("operating_cash_flow") is not None and values.get("capital_expenditure") is not None:
            values["free_cash_flow"] = round(values["operating_cash_flow"] - values["capital_expenditure"], 2)
        if statement_type == "key_ratios":
            income = {k: values.get(k) for k in STATEMENTS["income_statement"]}
            balance = {k: values.get(k) for k in STATEMENTS["balance_sheet"]} if period == "annual" else None
            cash = {"free_cash_flow": values.get("free_cash_flow")} if period == "annual" else None
            values = key_ratios(income, balance, cash)
        fiscal_year, fiscal_period = (end.year, "FY") if period == "annual" else _quarter_label(end, fy_ends)
        rows.append({"fiscal_year": fiscal_year, "fiscal_period": fiscal_period, "period_end": key, "values": values})

    missing = sorted({k for row in rows for k, v in row["values"].items() if v is None})
    if missing:
        notes.append(f"Not reported in XBRL for some periods: {', '.join(missing)}.")
    if len(rows) < (years if period == "annual" else years * 4):
        notes.append(f"Only {len(rows)} periods available.")
    notes = _notes_for(notes, {row["period_end"] for row in rows})
    fye = submissions(cik).get("fiscalYearEnd")
    return {
        "ticker": ticker, "company_name": title, "statement_type": statement_type, "period": period,
        "currency": "USD", "unit": "ratio" if statement_type == "key_ratios" else "millions",
        "fiscal_year_end": fye or f"{fy_ends[0].month:02d}{fy_ends[0].day:02d}",
        "provider": "sec_xbrl", "data": rows, "notes": notes,
    }


def _with_derived_q4(quarters: dict[str, float], annual: dict[str, float], fy_ends: list[date],
                     notes: list[str]) -> dict[str, float]:
    """Add fiscal Q4 = FY - (Q1 + Q2 + Q3) where the three quarters are reported."""
    result = dict(quarters)
    for fy_end in fy_ends:
        key = fy_end.isoformat()
        if key in result or key not in annual:
            continue
        window = [v for e, v in quarters.items()
                  if fy_end - timedelta(days=370) < date.fromisoformat(e) < fy_end - timedelta(days=20)]
        if len(window) == 3:
            result[key] = annual[key] - sum(window)
            note = "Fiscal Q4 values are derived as full year minus Q1-Q3 (Q4 is not filed separately)."
            if note not in notes:
                notes.append(note)
    return result


def _scale(item: str, value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4) if item in PER_SHARE_ITEMS else round(value / 1e6, 2)


# --------------------------------------------------------------------------------------------------
# Secondary source: yfinance (tier 2)
# --------------------------------------------------------------------------------------------------

_YF_ROWS = {
    "revenue": "Total Revenue", "gross_profit": "Gross Profit", "operating_income": "Operating Income",
    "net_income": "Net Income", "eps_diluted": "Diluted EPS", "total_assets": "Total Assets",
    "total_liabilities": "Total Liabilities Net Minority Interest", "shareholders_equity": "Stockholders Equity",
    "cash_and_equivalents": "Cash And Cash Equivalents", "operating_cash_flow": "Operating Cash Flow",
    "capital_expenditure": "Capital Expenditure",
}


def _yfinance_statements(ticker: str, statement_type: str, period: str, years: int, *, reason: str) -> dict[str, Any]:
    import math

    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        attributes = {"income_statement": ("income_stmt", "quarterly_income_stmt"),
                      "balance_sheet": ("balance_sheet", "quarterly_balance_sheet"),
                      "cash_flow": ("cashflow", "quarterly_cashflow")}
        wanted = ["income_statement", "balance_sheet", "cash_flow"] if statement_type == "key_ratios" else [statement_type]
        # Each statement is a separate network call, so only fetch the ones needed.
        frames = {name: getattr(t, attributes[name][0 if period == "annual" else 1]) for name in wanted}
    except Exception as exc:  # yfinance scrapes an unofficial API and fails in many ways
        raise ToolExecutionError(f"{TOOL}: {reason}, and yfinance failed: {type(exc).__name__}: {exc}", tool=TOOL,
                                 retryable=True) from exc
    for frame_name in wanted:
        if frames[frame_name] is None or frames[frame_name].empty:
            raise ToolExecutionError(f"{TOOL}: {reason}, and yfinance has no {frame_name} for {ticker}", tool=TOOL)
    columns = list(frames[wanted[0]].columns)[: years if period == "annual" else years * 4]
    rows = []
    for col in columns:
        values: dict[str, float | None] = {}
        for frame_name in wanted:
            frame = frames[frame_name]
            for item in STATEMENTS[frame_name]:
                raw = frame.at[_YF_ROWS[item], col] if _YF_ROWS[item] in frame.index and col in frame.columns else None
                if raw is None or (isinstance(raw, float) and math.isnan(raw)):
                    values[item] = None
                else:
                    raw = abs(float(raw)) if item == "capital_expenditure" else float(raw)
                    values[item] = _scale(item, raw)
        if values.get("operating_cash_flow") is not None and values.get("capital_expenditure") is not None:
            values["free_cash_flow"] = round(values["operating_cash_flow"] - values["capital_expenditure"], 2)
        if statement_type == "key_ratios":
            values = key_ratios({k: values.get(k) for k in STATEMENTS["income_statement"]},
                                {k: values.get(k) for k in STATEMENTS["balance_sheet"]},
                                {"free_cash_flow": values.get("free_cash_flow")})
        end = col.date() if hasattr(col, "date") else date.fromisoformat(str(col)[:10])
        rows.append({"fiscal_year": end.year, "fiscal_period": "FY" if period == "annual" else "Q",
                     "period_end": end.isoformat(), "values": values})
    return {
        "ticker": ticker, "company_name": ticker, "statement_type": statement_type, "period": period,
        "currency": "USD", "unit": "ratio" if statement_type == "key_ratios" else "millions",
        "fiscal_year_end": "", "provider": "yfinance", "data": rows,
        "notes": [f"Source: yfinance (tier 2), because {reason}. Figures are not from the filings themselves."],
    }
