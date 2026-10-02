"""earnings_transcript: earnings-call transcript for a company, quarter and fiscal year.

Free full transcripts are often unavailable, so the result says what kind of source it is:
``source_kind`` is ``"transcript"`` for a real call transcript, or ``"press_release"`` when the tool
fell back to the earnings press release (8-K Exhibit 99.1). Reports must tell the reader which one
was used (architecture spec, section 9.2).

Live handler (Day 7), free and keyless: the call transcript from The Motley Fool (found through web
search, parsed into speaker turns and prepared/Q&A sections), falling back to the earnings press
release in the quarter's 8-K. A mock handler with canned data is kept for tests and offline runs.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any

from tools.base import ToolExecutionError
from tools.mock_data import ANNUAL, MOCK_NOTICE, PROFILES, QUARTERLY_INCOME, TRANSCRIPTS

TOOL = "earnings_transcript"


def _check_period(quarter: str | None, year: int | None) -> None:
    if (quarter is None) != (year is None):
        raise ToolExecutionError(f"{TOOL}: give both quarter and year, or neither for the latest quarter", tool=TOOL)


def mock_earnings_transcript(ticker: str, quarter: str | None = None, year: int | None = None) -> dict[str, Any]:
    """Earnings transcript from bundled mock data; the latest reported quarter when ``quarter`` is omitted."""
    ticker = ticker.upper()
    _check_period(quarter, year)
    if quarter is None:
        reported = [(q["fiscal_year"], q["fiscal_period"]) for q in QUARTERLY_INCOME.get(ticker, [])]
        reported += [(y, q) for t, q, y in TRANSCRIPTS if t == ticker]
        if not reported:
            raise ToolExecutionError(f"{TOOL}: no earnings material for {ticker}", tool=TOOL)
        year, quarter = max(reported)
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


# --------------------------------------------------------------------------------------------------
# Live handler
# --------------------------------------------------------------------------------------------------

MONTH = 30 * 86_400
#: The Motley Fool publishes free call transcripts at URLs like
#: /earnings/call-transcripts/2026/08/07/apple-aapl-q3-2026-earnings-call-transcript/ (robots.txt allows them).
_FOOL_URL = re.compile(r"^https?://(?:www\.)?fool\.com/earnings/call-transcripts/\d{4}/\d{2}/\d{2}/([a-z0-9-]+)/?$")
_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
# "July 30, 2026" (current pages) or "Jan 24, 2024" (older pages).
_CALL_DATE = re.compile(rf"\b({'|'.join(_MONTHS)})[a-z]*\.?\s+(\d{{1,2}}),\s+(\d{{4}})", re.IGNORECASE)
_FIRM_ANALYST = re.compile(r"^(.+?)\s+--\s+analyst$", re.IGNORECASE)
#: Turns shorter than this ("Thank you.", "Yeah.") carry no content and would start empty Q&A pairs.
MIN_TURN_CHARS = 40
#: Earnings 8-Ks (Item 2.02) are filed this many days after the quarter ends.
RELEASE_WINDOW_DAYS = (5, 80)


def live_earnings_transcript(ticker: str, quarter: str | None = None, year: int | None = None) -> dict[str, Any]:
    """Earnings transcript for a quarter (latest when omitted), falling back to the press release if no transcript is found."""
    ticker = ticker.upper()
    _check_period(quarter, year)
    latest_note = None
    if quarter is None:
        quarter, year = latest_reported_quarter(ticker)
        latest_note = (f"No quarter was requested: {quarter} FY{year} is the latest quarter with an earnings "
                       "release (8-K Item 2.02) on SEC EDGAR.")
    try:
        result = fool_transcript(ticker, quarter, year)
    except ToolExecutionError as exc:
        result = press_release(ticker, quarter, year, reason=str(exc).removeprefix(f"{TOOL}: "))
    if latest_note:
        result["notes"].insert(0, latest_note)
    return result


def latest_reported_quarter(ticker: str) -> tuple[str, int]:
    """``(quarter, fiscal year)`` of the company's most recent earnings release on EDGAR."""
    from tools.sec_edgar import resolve_cik, submissions

    subs = submissions(resolve_cik(ticker)[0])
    recent = subs["filings"]["recent"]
    items = recent.get("items") or [""] * len(recent["form"])
    released = [date.fromisoformat(recent["filingDate"][i]) for i, form in enumerate(recent["form"])
                if form == "8-K" and "2.02" in (items[i] or "")]
    if not released:
        raise ToolExecutionError(f"{TOOL}: no earnings release (8-K Item 2.02) for {ticker} on EDGAR; "
                                 "give the quarter and year", tool=TOOL)
    last = max(released)
    fye = subs.get("fiscalYearEnd") or "1231"
    # The latest fiscal quarter that had ended at least a few days before that release.
    ended = [(_quarter_end(fye, f"Q{q}", y), f"Q{q}", y) for y in range(last.year - 1, last.year + 2)
             for q in range(1, 5)]
    _, quarter, year = max(e for e in ended if e[0] + timedelta(days=RELEASE_WINDOW_DAYS[0]) <= last)
    return quarter, year


def fool_transcript(ticker: str, quarter: str, year: int) -> dict[str, Any]:
    """The call transcript from The Motley Fool, found through web search."""
    from tools.sec_edgar import _http
    from tools.web_search import live_web_search

    slug_part = f"-{ticker.lower().replace('.', '')}-{quarter.lower()}-{year}-earnings"
    hits = live_web_search(f"{ticker} {quarter} {year} earnings call transcript fool.com", num_results=10)["results"]
    url = next((h["url"] for h in hits if (m := _FOOL_URL.match(h["url"])) and slug_part in m.group(1)), None)
    if url is None:
        raise ToolExecutionError(f"{TOOL}: no Motley Fool transcript found for {ticker} {quarter} {year}", tool=TOOL)
    call_date, segments = parse_fool_transcript(_http().get(url, ttl_s=MONTH, timeout_s=60))
    if not segments:
        raise ToolExecutionError(f"{TOOL}: could not parse the transcript at {url}", tool=TOOL)
    notes = [f"Source: The Motley Fool call transcript ({url}), tier 3.",
             "Operator turns and one-line courtesies are left out."]
    if not any(s["section"] == "qa" for s in segments):
        notes.append("The Q&A section could not be identified; all turns are marked as prepared remarks.")
    return {"ticker": ticker, "quarter": quarter, "fiscal_year": year, "call_date": call_date,
            "source_kind": "transcript", "source_url": url, "segments": segments, "notes": notes}


def parse_fool_transcript(html: bytes) -> tuple[str | None, list[dict[str, str]]]:
    """``(call date, segments)`` from a Motley Fool transcript page.

    Two page layouts exist. Current pages list participants as "Title - Name" and start each turn with
    "<strong>Name:</strong> text"; there is no Q&A heading, so Q&A starts with the first speaker who is
    not on the participant list (an analyst). Older pages put each speaker on a line of their own
    ("<strong>Name</strong> -- <em>Title</em>") and have "Prepared Remarks" and "Questions & Answers"
    headings.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one(".article-body") or soup.find("article") or soup
    participants = _participants(body)
    call_date = None
    turns: list[list[str]] = []  # [speaker, role, section, text]
    started = qa = headed = False  # headed: the page has "Prepared Remarks" / "Questions & Answers" headings
    speaker, role = "", ""
    for el in body.find_all(["h2", "p"]):
        if el.name == "h2":
            heading = el.get_text(" ", strip=True).lower()
            if "full conference call transcript" in heading or "prepared remarks" in heading:
                started = True
                headed = "prepared remarks" in heading
            elif "questions" in heading and "answers" in heading and started:
                qa = True
            elif started:
                break  # "Call participants", "More AAPL analysis", ...
            continue
        text = el.get_text(" ", strip=True)
        if not started:
            if call_date is None and (m := _CALL_DATE.search(text)):
                call_date = f"{m.group(3)}-{_MONTHS.index(m.group(1)[:3].lower()) + 1:02d}-{int(m.group(2)):02d}"
            continue
        strong = el.find("strong")
        label = strong.get_text(" ", strip=True) if strong else ""
        name = label.rstrip(":").strip()
        if name and text.startswith(label):
            rest = text[len(label):].strip()
            em = el.find("em")
            if not rest or (em and rest.startswith("--") and rest.lstrip("- ").strip() == em.get_text(" ", strip=True)):
                # A speaker line of its own (older layout); the turn's text follows.
                speaker, role = name, em.get_text(" ", strip=True) if em else participants.get(name, "")
                if firm := _FIRM_ANALYST.match(role):  # "Morgan Stanley -- Analyst"
                    speaker, role = f"{name} ({firm.group(1)})", "Analyst"
                continue
            if label.endswith(":"):
                speaker, role, text = name, participants.get(name, ""), rest
        if not speaker or not text:
            continue
        if speaker.lower() == "operator":
            role = "Operator"
        elif role == "Analyst" or (not role and not headed and participants):
            # The first analyst starts the Q&A (some older pages have no "Questions & Answers" heading;
            # current pages mark nobody, but analysts are the speakers missing from the participant list).
            qa = True
        if qa and not role and not headed:  # headed pages label analysts themselves
            role = "Analyst"
        section = "qa" if qa else "prepared"
        if turns and turns[-1][0] == speaker and turns[-1][2] == section:
            turns[-1][3] += "\n" + text
        else:
            turns.append([speaker, role, section, text])
    segments = [{"speaker": s, "role": r, "section": sec, "text": t} for s, r, sec, t in turns
                if r != "Operator" and len(t) >= MIN_TURN_CHARS]
    return call_date, segments


def _participants(body: Any) -> dict[str, str]:
    """Speaker name -> title, from the page's participant list ("Title - Name" or "Name -- Title")."""
    result: dict[str, str] = {}
    for h2 in body.find_all("h2"):
        if "participants" not in h2.get_text(strip=True).lower():
            continue
        for sib in h2.find_next_siblings():
            if sib.name == "h2":
                break
            for line in [li.get_text(" ", strip=True) for li in sib.find_all("li")] or [sib.get_text(" ", strip=True)]:
                if " -- " in line:
                    name, title = line.split(" -- ", 1)
                elif " - " in line:
                    title, name = line.rsplit(" - ", 1)
                else:
                    continue
                result[name.strip()] = title.strip()
    return result


def press_release(ticker: str, quarter: str, year: int, *, reason: str) -> dict[str, Any]:
    """Fallback: the earnings press release (Exhibit 99 of the quarter's Item 2.02 8-K)."""
    from tools.sec_edgar import SEC_ARCHIVES, _exhibit_99, resolve_cik, submissions

    cik, title = resolve_cik(ticker)
    subs = submissions(cik)
    quarter_end = _quarter_end(subs.get("fiscalYearEnd") or "1231", quarter, year)
    recent = subs["filings"]["recent"]
    items = recent.get("items") or [""] * len(recent["form"])
    lo, hi = (quarter_end + timedelta(days=d) for d in RELEASE_WINDOW_DAYS)
    candidates = sorted((recent["filingDate"][i], i) for i, form in enumerate(recent["form"])
                        if form == "8-K" and "2.02" in (items[i] or "")
                        and lo <= date.fromisoformat(recent["filingDate"][i]) <= hi)
    for _, i in candidates:
        folder = f"{SEC_ARCHIVES}/{int(cik)}/{recent['accessionNumber'][i].replace('-', '')}"
        release = _exhibit_99(folder)
        if release:
            return {
                "ticker": ticker, "quarter": quarter, "fiscal_year": year,
                "call_date": recent["reportDate"][i] or recent["filingDate"][i],
                "source_kind": "press_release", "source_url": f"{folder}/",
                "segments": [{"speaker": subs.get("name") or title, "role": "Company", "section": "prepared",
                              "text": release}],
                "notes": [f"Call transcript unavailable ({reason}); content is the earnings press release "
                          "(8-K Exhibit 99, SEC EDGAR, tier 1). There is no management Q&A.",
                          "call_date is the date of the press release."],
            }
    raise ToolExecutionError(f"{TOOL}: no transcript ({reason}) and no earnings press release (8-K Item 2.02) "
                             f"filed {lo} to {hi} for {ticker} {quarter} FY{year}", tool=TOOL)


def _quarter_end(fiscal_year_end: str, quarter: str, year: int) -> date:
    """Approximate end of fiscal ``quarter`` of fiscal ``year`` (52/53-week years shift it by days)."""
    month, day = int(fiscal_year_end[:2]), int(fiscal_year_end[2:])
    # A fiscal year ending in the first days of January is named after the previous calendar year.
    fy_end = date(year + 1 if month == 1 and day <= 7 else year, month, min(day, 28))
    return fy_end - timedelta(days=round((4 - int(quarter[1])) * 91.3))
