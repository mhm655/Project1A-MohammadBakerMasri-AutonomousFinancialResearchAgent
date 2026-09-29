"""Tests for the Day 5 live tools, against a fake network built from real EDGAR/Google News shapes.

No test here touches the internet. ``test_network_smoke`` (marked ``network``) does, and is skipped by
default; run it with ``python -m pytest -m network``.
"""

from __future__ import annotations

import json
from datetime import UTC

import httpx
import pytest

from tools import ToolExecutionError, build_default_registry
from tools.http import DiskCache, HttpClient, RateLimiter, set_http
from tools.news_sentiment import finance_sentiment

UA = "ARA-1 Research Agent test@example.com"

TICKERS = {"0": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
           "1": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}

SUBMISSIONS = {
    "cik": "789019", "name": "MICROSOFT CORP", "fiscalYearEnd": "0630", "sic": "7372",
    "filings": {"recent": {
        "accessionNumber": ["0001193125-26-400001", "0001193125-26-323660", "0000950170-25-100235"],
        "filingDate": ["2026-08-15", "2026-07-29", "2025-07-30"],
        "reportDate": ["2026-08-14", "2026-06-30", "2025-06-30"],
        "form": ["8-K", "10-K", "10-K"],
        "primaryDocument": ["msft-8k.htm", "msft-20260630.htm", "msft-20250630.htm"],
        "items": ["2.02,9.01", "", ""],
    }},
}

TEN_K = b"""<html><body>
<p>Table of Contents</p>
<p>Item 1. Business 3</p><p>Item 1A. Risk Factors 14</p><p>Item 7. Management's Discussion 34</p>
<p>Item 8. Financial Statements 50</p>
<p>PART I</p><p>Item 1</p>
<p><b>ITEM 1. BUSINESS</b></p><p>Microsoft is a technology company. It builds cloud and AI platforms.</p>
<p>12</p><p>Item 1A</p>
<p><b>ITEM 1A. RISK FACTORS</b></p>
<p><span style="font-weight:700">We face intense competition across all markets for our products.</span></p>
<p>Competition could reduce our revenue and margins over time and affect results.</p>
<p><span style="font-style:italic">Cyberattacks could harm our reputation and our results of operations.</span></p>
<p>Security incidents may be costly and disruptive to customers and partners.</p>
<p><b>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</b></p><p>Revenue increased 18% to $331.8 billion.</p>
<p><b>ITEM 8. FINANCIAL STATEMENTS</b></p><p>Income statements (in millions).</p>
</body></html>"""


def concept(values: list[tuple[str | None, str, float, str, str]], unit: str = "USD") -> dict:
    """values: (start, end, val, form, filed)."""
    return {"units": {unit: [{"start": s, "end": e, "val": v, "form": f, "filed": d, "accn": f"acc-{d}-{e}",
                              "fy": int(e[:4]), "fp": "FY"} for s, e, v, f, d in values]}}


REVENUE = concept([
    ("2023-07-01", "2024-06-30", 245122e6, "10-K", "2024-07-30"),
    ("2024-07-01", "2025-06-30", 281724e6, "10-K", "2025-07-30"),
    ("2025-07-01", "2026-06-30", 331839e6, "10-K", "2026-07-29"),
    # a later filing restates FY2025
    ("2024-07-01", "2025-06-30", 285000e6, "10-K", "2026-07-29"),
    # quarters of FY2026 (Q4 is not filed separately)
    ("2025-07-01", "2025-09-30", 77000e6, "10-Q", "2025-10-29"),
    ("2025-10-01", "2025-12-31", 81000e6, "10-Q", "2026-01-28"),
    ("2026-01-01", "2026-03-31", 83000e6, "10-Q", "2026-04-29"),
])
NET_INCOME = concept([
    ("2023-07-01", "2024-06-30", 88136e6, "10-K", "2024-07-30"),
    ("2024-07-01", "2025-06-30", 101832e6, "10-K", "2025-07-30"),
    ("2025-07-01", "2026-06-30", 133749e6, "10-K", "2026-07-29"),
])
EQUITY = concept([(None, "2026-06-30", 442000e6, "10-K", "2026-07-29"), (None, "2025-06-30", 343000e6, "10-K", "2025-07-30")])

RSS = """<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Microsoft beats expectations as cloud demand soars - Reuters</title><link>https://news.example/1</link>
<pubDate>{recent}</pubDate><source url="https://reuters.com">Reuters</source></item>
<item><title>Microsoft faces antitrust probe - Financial Times</title><link>https://news.example/2</link>
<pubDate>{recent}</pubDate><source url="https://ft.com">Financial Times</source></item>
<item><title>Old story - CNBC</title><link>https://news.example/3</link>
<pubDate>Mon, 01 Jan 2024 10:00:00 GMT</pubDate><source url="https://cnbc.com">CNBC</source></item>
</channel></rss>"""


class FakeNetwork:
    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.fail_next: dict[str, list[int]] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)
        for fragment, codes in self.fail_next.items():
            if fragment in url and codes:
                return httpx.Response(codes.pop(0), headers={"retry-after": "0"})
        if url.endswith("company_tickers.json"):
            return httpx.Response(200, json=TICKERS)
        if "submissions/CIK0000789019" in url:
            return httpx.Response(200, json=SUBMISSIONS)
        if "companyconcept" in url:
            tag = url.rsplit("/", 1)[-1].removesuffix(".json")
            body = {"RevenueFromContractWithCustomerExcludingAssessedTax": REVENUE, "NetIncomeLoss": NET_INCOME,
                    "StockholdersEquity": EQUITY}.get(tag)
            return httpx.Response(200, json=body) if body else httpx.Response(404)
        if url.endswith(("msft-20260630.htm", "msft-20250630.htm")):
            return httpx.Response(200, content=TEN_K)
        if url.endswith("index.json"):
            return httpx.Response(200, json={"directory": {"item": [{"name": "msft-8k.htm"}, {"name": "msft-ex991.htm"}]}})
        if url.endswith("msft-ex991.htm"):
            return httpx.Response(200, content=b"<html><body><p>Microsoft reports record revenue of $77.0 billion.</p></body></html>")
        if url.endswith("msft-8k.htm"):
            return httpx.Response(200, content=b"<html><body><p>Item 2.02 Results of Operations.</p></body></html>")
        if "news.google.com" in url:
            from datetime import datetime
            from email.utils import format_datetime

            return httpx.Response(200, text=RSS.format(recent=format_datetime(datetime.now(UTC))))
        return httpx.Response(404)


@pytest.fixture
def network(tmp_path):
    fake = FakeNetwork()
    client = HttpClient(sec_user_agent=UA, cache_dir=tmp_path / "cache", transport=httpx.MockTransport(fake),
                        rate_limits={"sec.gov": 1000}, sleep=lambda s: None)
    set_http(client)
    yield fake, client
    set_http(None)


@pytest.fixture
def live():
    return build_default_registry("live")


# ==================================================================================================
# HTTP layer
# ==================================================================================================


def test_sec_gets_the_contact_user_agent_and_nobody_else_does(network):
    fake, client = network
    client.get("https://www.sec.gov/files/company_tickers.json")
    client.get("https://news.google.com/rss/search", params={"q": "x"})
    sec, google = fake.requests
    assert sec.headers["user-agent"] == UA
    assert "@" not in google.headers["user-agent"]  # the email goes to the SEC only


def test_responses_are_cached_on_disk(network):
    fake, client = network
    for _ in range(3):
        client.get_json("https://www.sec.gov/files/company_tickers.json", ttl_s=60)
    assert len(fake.requests) == 1 and client.cache_hits == 2


def test_transient_errors_are_retried_and_404_is_not(network):
    fake, client = network
    fake.fail_next["company_tickers"] = [503, 429]
    assert client.get_json("https://www.sec.gov/files/company_tickers.json")["0"]["ticker"] == "MSFT"
    assert len(fake.requests) == 3
    with pytest.raises(ToolExecutionError, match="not found"):
        client.get("https://www.sec.gov/missing")


def test_missing_sec_email_gives_a_clear_error():
    client = HttpClient(sec_user_agent=None, transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    with pytest.raises(ToolExecutionError, match="SEC_CONTACT_EMAIL"):
        client.get("https://data.sec.gov/anything")


def test_rate_limiter_spaces_requests():
    now, waits = [0.0], []
    limiter = RateLimiter(rate=4, burst=1, clock=lambda: now[0], sleep=waits.append)
    for _ in range(3):
        limiter.acquire()
    assert waits == pytest.approx([0.25, 0.5])


def test_disk_cache_expiry(tmp_path):
    t = [100.0]
    cache = DiskCache(tmp_path, clock=lambda: t[0])
    cache.put("k", b"v")
    assert cache.get("k", ttl_s=10) == b"v"
    t[0] += 11
    assert cache.get("k", ttl_s=10) is None


# ==================================================================================================
# sec_filing_search
# ==================================================================================================


def test_10k_is_split_into_items_with_risk_headings(network, live):
    data = live.dispatch("sec_filing_search", {"ticker": "msft", "filing_type": "10-K"}).data
    assert data["cik"] == "0000789019" and data["accession_number"] == "0001193125-26-323660"
    assert data["period_of_report"] == "2026-06-30" and data["fiscal_year_end"] == "0630"
    items = {s["item"]: s["text"] for s in data["sections"]}
    assert set(items) == {"1", "1A", "7", "8"}  # 7A is absent in this filing and noted as missing
    assert "technology company" in items["1"] and "Table of Contents" not in items["1A"]
    assert data["risk_factors"] == ["We face intense competition across all markets for our products.",
                                    "Cyberattacks could harm our reputation and our results of operations."]
    assert any("7A" in n for n in data["notes"])


def test_filing_year_selects_the_right_report(network, live):
    data = live.dispatch("sec_filing_search", {"ticker": "MSFT", "filing_type": "10-K", "year": 2025}).data
    assert data["accession_number"] == "0000950170-25-100235"
    with pytest.raises(ToolExecutionError, match="no 10-K found for fiscal year 2019"):
        live.dispatch("sec_filing_search", {"ticker": "MSFT", "filing_type": "10-K", "year": 2019})


def test_8k_includes_the_press_release_exhibit(network, live):
    data = live.dispatch("sec_filing_search", {"ticker": "MSFT", "filing_type": "8-K"}).data
    release = next(s for s in data["sections"] if s["item"] == "Exhibit 99.1")
    assert "record revenue of $77.0 billion" in release["text"]
    assert any("2.02" in n for n in data["notes"])


def test_unknown_ticker(network, live):
    with pytest.raises(ToolExecutionError, match="no EDGAR filer"):
        live.dispatch("sec_filing_search", {"ticker": "ZZZZ", "filing_type": "10-K"})


# ==================================================================================================
# financial_data_api (SEC XBRL)
# ==================================================================================================


def test_annual_statements_from_xbrl_with_restatement_noted(network, live):
    data = live.dispatch("financial_data_api", {"ticker": "MSFT", "statement_type": "income_statement"}).data
    assert data["provider"] == "sec_xbrl" and data["unit"] == "millions"
    rows = {r["fiscal_year"]: r["values"] for r in data["data"]}
    assert list(rows) == [2026, 2025, 2024]
    assert rows[2026]["revenue"] == 331839.0 and rows[2024]["net_income"] == 88136.0
    assert rows[2025]["revenue"] == 285000.0  # the later filing's restated figure wins
    assert any("Restated" in n and "2025-06-30" in n for n in data["notes"])
    assert any("gross_profit" in n for n in data["notes"])  # concepts this company does not report


def test_quarterly_statements_derive_q4(network, live):
    data = live.dispatch("financial_data_api", {"ticker": "MSFT", "statement_type": "income_statement",
                                                "period": "quarterly", "years": 1}).data
    labels = [(r["fiscal_period"], r["fiscal_year"], r["values"]["revenue"]) for r in data["data"]]
    assert labels[0] == ("Q4", 2026, 331839.0 - 77000 - 81000 - 83000)
    assert ("Q1", 2026, 77000.0) in labels
    assert any("derived" in n for n in data["notes"])


def test_key_ratios_from_xbrl(network, live):
    row = live.dispatch("financial_data_api", {"ticker": "MSFT", "statement_type": "key_ratios", "years": 1}).data["data"][0]
    assert row["values"]["net_margin"] == pytest.approx(133749 / 331839, abs=1e-4)
    assert row["values"]["roe"] == pytest.approx(133749 / 442000, abs=1e-4)


def test_non_sec_ticker_falls_back_to_yfinance(network, live, monkeypatch):
    import pandas as pd

    frame = pd.DataFrame({pd.Timestamp("2026-03-31"): [1_000e6, 400e6, 200e6, 150e6, 1.5]},
                         index=["Total Revenue", "Gross Profit", "Operating Income", "Net Income", "Diluted EPS"])

    class FakeTicker:
        def __init__(self, symbol):
            self.income_stmt = frame

    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    data = live.dispatch("financial_data_api", {"ticker": "SHOP", "statement_type": "income_statement", "years": 1}).data
    assert data["provider"] == "yfinance" and data["data"][0]["values"]["revenue"] == 1000.0
    assert "tier 2" in data["notes"][0]


# ==================================================================================================
# news_sentiment and web_search
# ==================================================================================================


@pytest.mark.parametrize("headline, sign", [
    ("Microsoft beats expectations as cloud demand soars", 1),  # plain VADER scores this negative
    ("Tesla shares plunge after deliveries miss", -1),
    ("Analysts upgrade Nvidia on record data-center sales", 1),
    ("Apple faces antitrust probe in the EU", -1),
])
def test_finance_sentiment(headline, sign):
    assert finance_sentiment(headline) * sign > 0.2


def test_news_from_google_rss(network, live):
    data = live.dispatch("news_sentiment", {"query": "Microsoft", "lookback_days": 30}).data
    assert data["article_count"] == 2  # the 2024 story is outside the window
    first = data["articles"][0]
    assert first["title"] == "Microsoft beats expectations as cloud demand soars"  # publisher suffix removed
    assert first["source"] == "Reuters" and first["sentiment_label"] == "positive"
    assert data["overall"]["label"] in ("positive", "neutral", "negative")


def test_web_search_uses_the_first_backend_that_answers(live, monkeypatch):
    calls = []

    class FakeDDGS:
        def __init__(self, timeout):
            pass

        def text(self, query, max_results, backend, timelimit):
            calls.append((backend, timelimit))
            if backend == "brave":
                raise TimeoutError("slow")
            return [{"title": "Azure grows", "href": "https://example.com/a", "body": "Azure revenue grew."}]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    data = live.dispatch("web_search", {"query": "Azure growth", "num_results": 3, "date_range": "month"}).data
    assert calls == [("brave", "m"), ("bing", "m")]
    assert data["results"][0]["source"] == "example.com" and "bing" in data["notes"][0]


def test_web_search_falls_back_to_google_news(network, live, monkeypatch):
    class DeadDDGS:
        def __init__(self, timeout):
            pass

        def text(self, *args, **kwargs):
            raise TimeoutError("down")

    monkeypatch.setattr("ddgs.DDGS", DeadDDGS)
    data = live.dispatch("web_search", {"query": "Microsoft", "num_results": 5}).data
    assert data["results"] and "Google News" in data["notes"][0]


# ==================================================================================================
# Real network (skipped by default)
# ==================================================================================================


@pytest.mark.network
def test_network_smoke():
    registry = build_default_registry("live")
    profile = registry.dispatch("financial_data_api", {"ticker": "MSFT", "statement_type": "income_statement", "years": 1})
    assert profile.data["provider"] == "sec_xbrl" and profile.data["data"][0]["values"]["revenue"] > 100_000
    assert registry.dispatch("news_sentiment", {"query": "Microsoft", "num_articles": 3}).data["article_count"] > 0
    json.dumps(profile.data)


# ==================================================================================================
# Audit regressions
# ==================================================================================================


def test_52_53_week_year_ending_in_early_january_keeps_its_fiscal_year(monkeypatch):
    # Kellogg-style calendar: fiscal 2021 ended on Saturday 1 January 2022, fiscal 2022 on 31 December 2022.
    from tools import financial_api

    def fact(start, end, val, form, filed):
        return {"start": start, "end": end, "val": val, "form": form, "filed": filed,
                "accn": f"acc-{filed}", "rank": 0}

    revenue = [
        fact("2019-12-29", "2021-01-02", 13770e6, "10-K", "2021-02-22"),
        fact("2021-01-03", "2022-01-01", 14181e6, "10-K", "2022-02-22"),
        fact("2022-01-02", "2022-12-31", 15315e6, "10-K", "2023-02-21"),
        fact("2021-01-03", "2021-04-03", 3600e6, "10-Q", "2021-05-06"),
        fact("2021-04-04", "2021-07-03", 3500e6, "10-Q", "2021-08-05"),
        fact("2021-07-04", "2021-10-02", 3600e6, "10-Q", "2021-11-04"),
    ]
    monkeypatch.setattr(financial_api, "_facts", lambda cik, item: revenue if item == "revenue" else [])
    monkeypatch.setattr("tools.sec_edgar.resolve_cik", lambda ticker: ("0000055067", "KELLOGG CO"))
    monkeypatch.setattr("tools.sec_edgar.submissions", lambda cik: {"fiscalYearEnd": "1231"})

    annual = financial_api.live_financial_data_api("K", "income_statement", "annual", 3)
    assert [(r["fiscal_year"], r["period_end"]) for r in annual["data"]] == [
        (2022, "2022-12-31"), (2021, "2022-01-01"), (2020, "2021-01-02")]
    quarterly = financial_api.live_financial_data_api("K", "income_statement", "quarterly", 1)
    labels = {r["period_end"]: (r["fiscal_period"], r["fiscal_year"]) for r in quarterly["data"]}
    assert labels["2022-01-01"] == ("Q4", 2021) and labels["2021-04-03"] == ("Q1", 2021)


def test_filing_year_matches_a_52_53_week_year_ending_in_early_january():
    from tools.sec_edgar import find_filing

    subs = {"filings": {"recent": {
        "accessionNumber": ["acc-2023", "acc-2022"], "filingDate": ["2023-02-21", "2022-02-22"],
        "reportDate": ["2022-12-31", "2022-01-01"], "form": ["10-K", "10-K"],
        "primaryDocument": ["k-20221231.htm", "k-20220101.htm"]}}}
    assert find_filing(subs, "10-K", 2021)["accessionNumber"] == "acc-2022"
    assert find_filing(subs, "10-K", 2022)["accessionNumber"] == "acc-2023"


def test_8k_press_release_named_exhibit991_is_found(network, live, monkeypatch):
    fake, _ = network
    original = fake.__call__

    def with_long_exhibit_name(request):
        url = str(request.url)
        if url.endswith("index.json"):
            return httpx.Response(200, json={"directory": {"item": [{"name": "msft-8k.htm"},
                                                                    {"name": "exhibit991.htm"}]}})
        if url.endswith("exhibit991.htm"):
            return httpx.Response(200, content=b"<html><body><p>Record revenue of $77.0 billion.</p></body></html>")
        return original(request)

    monkeypatch.setattr(FakeNetwork, "__call__", lambda self, request: with_long_exhibit_name(request))
    data = live.dispatch("sec_filing_search", {"ticker": "MSFT", "filing_type": "8-K"}).data
    release = next(s for s in data["sections"] if s["item"] == "Exhibit 99.1")
    assert "Record revenue of $77.0 billion" in release["text"]


def test_rss_dates_without_a_known_timezone_do_not_crash(network, live, monkeypatch):
    # RFC 2822 "-0000" (and a missing zone) parse to a naive datetime, which cannot be compared
    # with the aware cutoff.
    from datetime import datetime, timedelta

    fake, _ = network
    original = fake.__call__
    recent = (datetime.now(UTC) - timedelta(days=1)).strftime("%a, %d %b %Y %H:%M:%S -0000")

    def rss(request):
        if "news.google.com" in str(request.url):
            return httpx.Response(200, text=RSS.format(recent=recent))
        return original(request)

    monkeypatch.setattr(FakeNetwork, "__call__", lambda self, request: rss(request))
    data = live.dispatch("news_sentiment", {"query": "Microsoft", "lookback_days": 30}).data
    assert data["article_count"] == 2
