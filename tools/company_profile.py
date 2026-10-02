"""company_profile: basic company facts (sector, industry, size, executives, fiscal calendar).

Live handler (Day 7), free and keyless, from two sources:

* **SEC EDGAR** submissions metadata (tier 1): legal name, CIK, SIC code, exchange, business address
  and fiscal-year end. These are what the company itself registered with the SEC.
* **yfinance** (tier 2): sector, industry, employees, market capitalisation, business description and
  executives, which EDGAR does not publish in structured form.

Either source alone is enough for a (partial) profile; the notes say which fields came from where.
A mock handler with canned data is kept for tests and offline runs.
"""

from __future__ import annotations

import re
from typing import Any

from tools.base import ToolExecutionError
from tools.mock_data import MOCK_NOTICE, PROFILES, STATIC_METRICS

TOOL = "company_profile"

MAX_EXECUTIVES = 8
MAX_DESCRIPTION_CHARS = 1500
_HONORIFIC = re.compile(r"^(?:mr|mrs|ms|dr|prof|sir)\.?\s+", re.IGNORECASE)


def mock_company_profile(ticker: str) -> dict[str, Any]:
    """Company profile from bundled mock data; peers get only name and market cap. Raises ToolExecutionError for an unknown ticker."""
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


# --------------------------------------------------------------------------------------------------
# Live handler
# --------------------------------------------------------------------------------------------------


def yfinance_info(ticker: str) -> dict[str, Any]:
    """yfinance's quote summary for ``ticker`` ({} if yfinance fails: it scrapes an unofficial API)."""
    try:
        import yfinance as yf

        return dict(yf.Ticker(ticker).info or {})
    except Exception:
        return {}


def live_company_profile(ticker: str) -> dict[str, Any]:
    """Company profile built from live SEC EDGAR data."""
    from tools.sec_edgar import resolve_cik, submissions

    ticker = ticker.upper()
    profile: dict[str, Any] = {"ticker": ticker}
    notes: list[str] = []
    sources: list[str] = []

    subs = None
    try:
        cik, title = resolve_cik(ticker)
        profile.update({"name": title, "cik": cik})
        subs = submissions(cik)
    except ToolExecutionError as exc:
        notes.append(f"No SEC EDGAR {'filing metadata' if 'cik' in profile else 'data'} ({exc}).")
    if "cik" in profile:
        sources.append("SEC EDGAR (name, CIK" + (", SIC code, exchange, address, fiscal-year end)" if subs else ")"))
    if subs is not None:
        profile["name"] = subs.get("name") or profile["name"]
        if subs.get("exchanges"):
            profile["exchange"] = ", ".join(e for e in subs["exchanges"] if e)
        if subs.get("sic"):
            profile["sic_code"] = str(subs["sic"])
            if subs.get("sicDescription"):
                profile["industry"] = subs["sicDescription"]
        if re.fullmatch(r"\d{4}", str(subs.get("fiscalYearEnd") or "")):
            profile["fiscal_year_end"] = subs["fiscalYearEnd"]
        address = (subs.get("addresses") or {}).get("business") or {}
        place = [address.get("city"), address.get("stateOrCountryDescription") or address.get("stateOrCountry")]
        if all(place):
            profile["headquarters"] = ", ".join(p.title() if p.isupper() and len(p) > 2 else p for p in place)

    info = yfinance_info(ticker)
    if info.get("longName") or info.get("shortName"):
        sources.append("yfinance (sector, industry, employees, market cap, description, executives)")
        profile.setdefault("name", info.get("longName") or info.get("shortName"))
        if info.get("sector"):
            profile["sector"] = info["sector"]
        if info.get("industry"):  # yfinance's industry names are more specific than SIC descriptions
            profile["industry"] = info["industry"]
        if isinstance(info.get("fullTimeEmployees"), int | float):
            profile["employees"] = int(info["fullTimeEmployees"])
        if isinstance(info.get("marketCap"), int | float) and info["marketCap"] > 0:
            profile["market_cap"] = float(info["marketCap"])
            notes.append("Market capitalisation is in US dollars at the latest yfinance quote.")
        if info.get("longBusinessSummary"):
            profile["description"] = info["longBusinessSummary"][:MAX_DESCRIPTION_CHARS]
        if "headquarters" not in profile:
            place = [info.get("city"), info.get("state") or info.get("country")]
            if all(place):
                profile["headquarters"] = ", ".join(place)
        executives = [{"name": _HONORIFIC.sub("", o["name"]).strip(), "title": o.get("title") or ""}
                      for o in info.get("companyOfficers") or [] if o.get("name")]
        if executives:
            profile["executives"] = executives[:MAX_EXECUTIVES]
    else:
        notes.append("yfinance returned no quote data, so sector, employees, market cap and executives are missing.")

    if "name" not in profile:
        raise ToolExecutionError(f"{TOOL}: no profile data for ticker {ticker!r} from SEC EDGAR or yfinance",
                                 tool=TOOL, retryable=True)
    profile["notes"] = [f"Sources: {'; '.join(sources)}."] + notes
    return profile
