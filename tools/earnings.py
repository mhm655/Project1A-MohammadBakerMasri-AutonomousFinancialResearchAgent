"""earnings_transcript: earnings-call transcript for a company, quarter and fiscal year.

Free full transcripts are often unavailable, so the result says what kind of source it is:
``source_kind`` is ``"transcript"`` for a real call transcript, or ``"press_release"`` when the tool
fell back to the earnings press release (8-K Exhibit 99.1). Reports must tell the reader which one
was used (architecture spec, section 9.2).

Day 2: mock handler. Day 7: live handler (public transcripts via web search, 8-K Ex-99.1 fallback).
"""

from __future__ import annotations

from typing import Any

from tools.base import ToolExecutionError
from tools.mock_data import ANNUAL, MOCK_NOTICE, PROFILES, QUARTERLY_INCOME, TRANSCRIPTS

TOOL = "earnings_transcript"


def mock_earnings_transcript(ticker: str, quarter: str, year: int) -> dict[str, Any]:
    ticker = ticker.upper()
    key = (ticker, quarter, year)
    if key in TRANSCRIPTS:
        t = TRANSCRIPTS[key]
        return {
            "ticker": ticker,
            "quarter": quarter,
            "fiscal_year": year,
            "call_date": t["call_date"],
            "source_kind": "transcript",
            "segments": t["segments"],
            "notes": [MOCK_NOTICE, "Speaker turns are paraphrased summaries, not verbatim quotes."],
        }

    if ticker not in PROFILES:
        raise ToolExecutionError(f"{TOOL}: unknown ticker {ticker!r}", tool=TOOL)

    # Fallback shape: an earnings press release instead of a call transcript.
    quarter_row = next(
        (q for q in QUARTERLY_INCOME.get(ticker, []) if q["fiscal_year"] == year and q["fiscal_period"] == quarter),
        None,
    )
    if quarter_row:
        inc = quarter_row["income_statement"]
        text = f"Quarterly revenue of ${inc['revenue']:,} million and net income of ${inc['net_income']:,} million."
    elif any(p["fiscal_year"] == year for p in ANNUAL[ticker]):
        text = "The company reported quarterly results; see financial_data_api for figures."
    else:
        raise ToolExecutionError(f"{TOOL}: no earnings material for {ticker} {quarter} FY{year}", tool=TOOL)

    cfo = next((e["name"] for e in PROFILES[ticker]["executives"] if "Financial" in e["title"]), "CFO")
    return {
        "ticker": ticker,
        "quarter": quarter,
        "fiscal_year": year,
        "call_date": None,
        "source_kind": "press_release",
        "segments": [{"speaker": cfo, "role": "CFO", "section": "prepared", "text": text}],
        "notes": [MOCK_NOTICE, "Call transcript unavailable; content is from the earnings press release."],
    }
