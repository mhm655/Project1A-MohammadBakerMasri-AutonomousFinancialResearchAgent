"""sec_filing_search: SEC EDGAR filings (10-K, 10-Q, 8-K, DEF 14A).

Live handler (Day 5), all free and keyless:

1. ticker -> CIK from ``www.sec.gov/files/company_tickers.json`` (cached for a week);
2. the company's filing list from ``data.sec.gov/submissions/CIK##########.json`` (cached for a day);
3. the filing's primary document from the EDGAR archive (cached for 30 days: filings never change),
   parsed into sections (Item 1 Business, 1A Risk Factors, 7 MD&A, ...) plus the individual risk
   factor headings, which the risk-coverage metric (CO-4) needs;
4. for an 8-K, the press release in Exhibit 99.1, when there is one.

A mock handler with canned data is kept for tests and offline runs.
"""

from __future__ import annotations

import re
import warnings
from datetime import date, timedelta
from typing import Any

from tools.base import ToolExecutionError
from tools.mock_data import ANNUAL, MOCK_NOTICE, PROFILES, RISK_FACTORS

TOOL = "sec_filing_search"

SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

DAY, WEEK, MONTH = 86_400, 7 * 86_400, 30 * 86_400
MAX_SECTION_CHARS = 60_000

ITEM_TITLES = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments", "1C": "Cybersecurity",
    "2": "Properties", "3": "Legal Proceedings", "5": "Market for Registrant's Common Equity",
    "7": "Management's Discussion and Analysis", "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data", "9A": "Controls and Procedures",
}
# Which items each form returns, in order.
FORM_ITEMS = {"10-K": ["1", "1A", "7", "7A", "8"], "10-Q": ["2", "1A"]}


# --------------------------------------------------------------------------------------------------
# EDGAR lookups
# --------------------------------------------------------------------------------------------------


def _http():
    from tools.http import get_http

    return get_http()


def resolve_cik(ticker: str) -> tuple[str, str]:
    """``(10-digit CIK, company title)`` for a ticker, or a ToolExecutionError if EDGAR has no filer."""
    table = _http().get_json(TICKERS_URL, ttl_s=WEEK)
    wanted = ticker.upper().replace(".", "-")
    for row in table.values():
        if row["ticker"].upper() == wanted:
            return f"{int(row['cik_str']):010d}", row["title"]
    raise ToolExecutionError(f"{TOOL}: no EDGAR filer found for ticker {ticker!r}", tool=TOOL)


def submissions(cik: str) -> dict[str, Any]:
    return _http().get_json(SUBMISSIONS_URL.format(cik=cik), ttl_s=DAY)


def find_filing(subs: dict[str, Any], form: str, year: int | None) -> dict[str, str]:
    """The most recent filing of ``form`` (for ``year``, the fiscal year of the report, if given)."""
    recent = subs["filings"]["recent"]
    for i, kind in enumerate(recent["form"]):
        if kind != form:
            continue
        report_date = recent["reportDate"][i] or recent["filingDate"][i]
        if year is not None and not report_date.startswith(str(year)):
            continue
        return {key: recent[key][i] for key in ("accessionNumber", "filingDate", "reportDate", "primaryDocument", "items")
                if key in recent}
    detail = f" for fiscal year {year}" if year else ""
    raise ToolExecutionError(f"{TOOL}: no {form} found{detail} in the company's recent EDGAR filings", tool=TOOL)


# --------------------------------------------------------------------------------------------------
# Document parsing
# --------------------------------------------------------------------------------------------------

_BLOCKS = ["p", "div", "li", "td", "h1", "h2", "h3", "h4", "h5", "h6"]
_BOLD_STYLE = re.compile(r"font-weight:\s*(bold|[6-9]00)|font-style:\s*italic", re.IGNORECASE)
_ITEM_HEADING = re.compile(r"\bitem\s*(\d{1,2}[a-c]?)\s*\.", re.IGNORECASE)
_PAGE_NOISE = re.compile(r"^(?:\d{1,3}|part\s+[iv]+|item\s+\d{1,2}[a-c]?(?:\s*,\s*\d{1,2}[a-c]?)*|table of contents)$",
                         re.IGNORECASE)


def document_text(html: bytes) -> tuple[str, list[tuple[int, str]]]:
    """Plain text of a filing plus ``(offset, text)`` for each run of bold or italic text.

    Filings are inline-XBRL HTML: headings are bold or italic spans, often split across several
    elements. Adjacent emphasised strings are merged into one run.
    """
    from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "ix:header"]):
        tag.decompose()

    parts: list[str] = []
    runs: list[tuple[int, str]] = []
    offset, run_start, run_text, run_block = 0, None, [], None
    for node in soup.find_all(string=True):
        text = re.sub(r"[\s\xa0]+", " ", str(node))
        if not text.strip():
            continue
        block = node.find_parent(_BLOCKS)
        if run_start is not None and block is not run_block:
            # A new paragraph ends the run: consecutive bold paragraphs are separate headings.
            runs.append((run_start, "".join(run_text).strip()))
            run_start, run_text = None, []
        emphasised = any(
            parent.name in ("b", "strong", "i", "em") or _BOLD_STYLE.search(parent.get("style", "") or "")
            for parent in node.parents if getattr(parent, "name", None) not in (None, "[document]", "body", "html")
            and parent.name in ("b", "strong", "i", "em", "span", "font", "p", "div")
        )
        if emphasised:
            if run_start is None:
                run_start, run_block = offset, block
            run_text.append(text)
        elif run_start is not None:
            runs.append((run_start, "".join(run_text).strip()))
            run_start, run_text = None, []
        parts.append(text + "\n")
        offset += len(text) + 1
    if run_start is not None:
        runs.append((run_start, "".join(run_text).strip()))
    return "".join(parts), runs


def split_items(text: str, wanted: list[str]) -> dict[str, tuple[int, int]]:
    """``item -> (start, end)`` offsets of each wanted item's body.

    Headings are "Item 1A." with a full stop (running page headers have none). The table of contents
    lists every item too, so for each item the occurrence followed by the longest text is taken.
    """
    marks = [(m.start(), m.group(1).upper()) for m in _ITEM_HEADING.finditer(text)]
    best: dict[str, tuple[int, int]] = {}
    for i, (start, item) in enumerate(marks):
        end = next((s for s, other in marks[i + 1:] if other != item), len(text))
        if item in wanted and (item not in best or end - start > best[item][1] - best[item][0]):
            best[item] = (start, end)
    return best


def _clean(section: str) -> str:
    lines = [line.strip() for line in section.splitlines()]
    kept = [line for line in lines if line and not _PAGE_NOISE.match(line)]
    return re.sub(r"\n{2,}", "\n", "\n".join(kept)).strip()


def risk_headings(runs: list[tuple[int, str]], span: tuple[int, int]) -> list[str]:
    """The individual risk-factor headings: bold or italic sentences inside Item 1A."""
    start, end = span
    headings = []
    for offset, text in runs:
        text = re.sub(r"\s+", " ", text).strip()
        if start < offset < end and 25 <= len(text) <= 400 and not _ITEM_HEADING.match(text):
            if text not in headings:
                headings.append(text)
    return headings


# --------------------------------------------------------------------------------------------------
# Live handler
# --------------------------------------------------------------------------------------------------


def live_sec_filing_search(ticker: str, filing_type: str, year: int | None = None) -> dict[str, Any]:
    cik, title = resolve_cik(ticker)
    subs = submissions(cik)
    filing = find_filing(subs, filing_type, year)
    accession = filing["accessionNumber"]
    folder = f"{SEC_ARCHIVES}/{int(cik)}/{accession.replace('-', '')}"
    url = f"{folder}/{filing['primaryDocument']}"
    html = _http().get(url, ttl_s=MONTH, timeout_s=300)  # filings never change; large 10-Ks are slow
    text, runs = document_text(html)

    sections: list[dict[str, str]] = []
    risk_list: list[str] = []
    notes = ["Source: SEC EDGAR (primary filing document)."]
    if filing_type in FORM_ITEMS:
        spans = split_items(text, FORM_ITEMS[filing_type])
        for item in FORM_ITEMS[filing_type]:
            if item not in spans:
                continue
            body = _clean(text[spans[item][0]:spans[item][1]])
            if len(body) > MAX_SECTION_CHARS:
                body = body[:MAX_SECTION_CHARS] + " ...[truncated]"
            sections.append({"item": item, "title": ITEM_TITLES.get(item, f"Item {item}"), "text": body})
        if "1A" in spans:
            risk_list = risk_headings(runs, spans["1A"])
        missing = [i for i in FORM_ITEMS[filing_type] if i not in spans]
        if missing:
            notes.append(f"Items not found in the document: {', '.join(missing)}.")
    else:
        sections.append({"item": filing_type, "title": filing_type, "text": _clean(text)[:MAX_SECTION_CHARS]})

    if filing_type == "8-K":
        items = filing.get("items") or ""
        if items:
            notes.append(f"8-K items: {items}.")
        release = _exhibit_99(folder)
        if release:
            sections.append({"item": "Exhibit 99.1", "title": "Press release", "text": release})

    return {
        "ticker": ticker.upper(),
        "cik": cik,
        "company_name": subs.get("name") or title,
        "filing_type": filing_type,
        "filing_date": filing["filingDate"],
        "period_of_report": filing.get("reportDate") or filing["filingDate"],
        "accession_number": accession,
        "url": url,
        "sections": sections,
        "risk_factors": risk_list,
        "fiscal_year_end": subs.get("fiscalYearEnd"),
        "notes": notes,
    }


def _exhibit_99(folder: str) -> str | None:
    """Text of the first Exhibit 99 document in a filing folder (earnings releases live there)."""
    try:
        index = _http().get_json(f"{folder}/index.json", ttl_s=MONTH)
    except ToolExecutionError:
        return None
    names = [item["name"] for item in index.get("directory", {}).get("item", [])]
    exhibit = next((n for n in names if re.search(r"ex[-_]?99", n, re.IGNORECASE) and n.lower().endswith((".htm", ".html"))), None)
    if not exhibit:
        return None
    text, _ = document_text(_http().get(f"{folder}/{exhibit}", ttl_s=MONTH, timeout_s=120))
    return _clean(text)[:MAX_SECTION_CHARS]


# --------------------------------------------------------------------------------------------------
# Mock handler
# --------------------------------------------------------------------------------------------------

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
        "risk_factors": list(risks) if filing_type == "10-K" else [],
        "notes": [MOCK_NOTICE],
    }
