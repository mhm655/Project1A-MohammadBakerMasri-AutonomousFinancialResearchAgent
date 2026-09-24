"""sec_filing_search: SEC EDGAR filings (10-K, 10-Q, 8-K, DEF 14A).

Day 2: mock handler producing section-structured filings from :mod:`tools.mock_data`.
Day 5: live handler (EDGAR submissions API + archive documents, full-text search fallback).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from tools.base import ToolExecutionError
from tools.mock_data import ANNUAL, MOCK_NOTICE, PROFILES, RISK_FACTORS

TOOL = "sec_filing_search"

# Typical number of days between period end and filing date, by form type (mock approximation).
_FILING_LAG_DAYS = {"10-K": 35, "10-Q": 35, "8-K": 28, "DEF 14A": 110}
_FORM_SEQ = {"10-K": 1, "10-Q": 2, "8-K": 3, "DEF 14A": 4}


def _mdna(ticker: str, fy_index: int) -> str:
    years = ANNUAL[ticker]
    cur = years[fy_index]["income_statement"]
    text = f"Fiscal {years[fy_index]['fiscal_year']} revenue was ${cur['revenue']:,} million"
    if fy_index + 1 < len(years):
        prev = years[fy_index + 1]["income_statement"]
        change = (cur["revenue"] - prev["revenue"]) / prev["revenue"]
        text += f", {'up' if change >= 0 else 'down'} {abs(change):.1%} from the prior year"
    return text + (
        f". Operating income was ${cur['operating_income']:,} million and net income ${cur['net_income']:,} million."
    )


def mock_sec_filing_search(ticker: str, filing_type: str, year: int | None = None) -> dict[str, Any]:
    ticker = ticker.upper()
    if ticker not in ANNUAL:
        raise ToolExecutionError(f"{TOOL}: no EDGAR filer found for ticker {ticker!r}", tool=TOOL)

    fiscal_years = [p["fiscal_year"] for p in ANNUAL[ticker]]
    if year is None:
        idx = 0
    elif year in fiscal_years:
        idx = fiscal_years.index(year)
    else:
        raise ToolExecutionError(
            f"{TOOL}: no {filing_type} for {ticker} fiscal year {year} (available: {fiscal_years})", tool=TOOL
        )

    profile, fy = PROFILES[ticker], ANNUAL[ticker][idx]
    period_end = date.fromisoformat(fy["period_end"])
    if filing_type == "10-Q":
        period_end -= timedelta(days=91)  # the third-quarter report of that fiscal year
    filed = period_end + timedelta(days=_FILING_LAG_DAYS[filing_type])
    cik = profile["cik"]
    accession = f"{cik}-{filed.year % 100:02d}-{_FORM_SEQ[filing_type] * 1000 + idx:06d}"
    risks = RISK_FACTORS.get(ticker, [])

    if filing_type == "10-K":
        sections = [
            {"item": "1", "title": "Business", "text": profile["description"]},
            {"item": "1A", "title": "Risk Factors", "text": "\n".join(f"- {r}" for r in risks)},
            {"item": "7", "title": "Management's Discussion and Analysis", "text": _mdna(ticker, idx)},
            {"item": "8", "title": "Financial Statements and Supplementary Data",
             "text": "Audited consolidated financial statements (see financial_data_api for line items)."},
        ]
    elif filing_type == "10-Q":
        sections = [
            {"item": "Part I, Item 2", "title": "Management's Discussion and Analysis",
             "text": "Unaudited interim results; see financial_data_api (period='quarterly')."},
            {"item": "Part II, Item 1A", "title": "Risk Factors",
             "text": "No material changes from the risk factors in the most recent Form 10-K."},
        ]
    elif filing_type == "8-K":
        sections = [
            {"item": "2.02", "title": "Results of Operations and Financial Condition",
             "text": "The company announced financial results; the press release is furnished as Exhibit 99.1."},
            {"item": "Exhibit 99.1", "title": "Press release", "text": _mdna(ticker, idx)},
        ]
    else:  # DEF 14A
        officers = "; ".join(f"{e['name']} ({e['title']})" for e in profile["executives"])
        sections = [
            {"item": "Executive Compensation", "title": "Named executive officers", "text": officers},
            {"item": "Proposals", "title": "Matters for shareholder vote",
             "text": "Election of directors, ratification of auditors, advisory vote on executive compensation."},
        ]

    return {
        "ticker": ticker,
        "cik": cik,
        "company_name": profile["name"],
        "filing_type": filing_type,
        "filing_date": filed.isoformat(),
        "period_of_report": period_end.isoformat(),
        "accession_number": accession,
        "url": f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/mock.htm",
        "sections": sections,
        "notes": [MOCK_NOTICE],
    }
