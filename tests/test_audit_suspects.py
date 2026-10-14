"""Regression tests for the audit's "suspected" items that were reproduced and fixed (offline, mocked)."""
from __future__ import annotations

import json
import sys
import types
from datetime import datetime

import httpx
import pytest

from agent.circuit_breaker import BreakerConfig, CircuitBreaker
from agent.parser import parse_steps
from agent.tracing import TraceLogger
from tools import financial_api
from tools.fact_checker import extract_quantities, fact_checker
from tools.http import GENERIC_USER_AGENT, DiskCache, HttpClient
from tools.tool_registry import build_default_registry

SEC_AGENT = "ARA-1 test someone@example.com"


def _client(handler, **kwargs) -> HttpClient:
    return HttpClient(sec_user_agent=SEC_AGENT, transport=httpx.MockTransport(handler), sleep=lambda s: None, **kwargs)


def test_sec_contact_email_is_not_forwarded_on_a_redirect_to_another_site():
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.headers["user-agent"]))
        if request.url.host == "www.sec.gov":
            return httpx.Response(302, headers={"location": "https://elsewhere.example.com/x"})
        return httpx.Response(200, text="ok")

    assert _client(handler).get("https://www.sec.gov/redirect") == b"ok"
    assert seen == [("www.sec.gov", SEC_AGENT), ("elsewhere.example.com", GENERIC_USER_AGENT)]


def test_a_request_timeout_status_is_retried():
    replies = iter([httpx.Response(408), httpx.Response(200, text="ok")])
    client = _client(lambda request: next(replies))

    assert client.get("https://example.com/x") == b"ok"
    assert client.requests_made == 2


@pytest.mark.parametrize("metadata", ["[1, 2]", json.dumps({"stored_at": "yesterday"})])
def test_a_damaged_cache_metadata_file_is_a_cache_miss(tmp_path, metadata):
    cache = DiskCache(tmp_path)
    meta_path, body_path = cache._paths("k")
    meta_path.write_text(metadata, encoding="utf-8")
    body_path.write_bytes(b"x")

    assert cache.get("k", 100) is None


def test_accounting_brackets_do_not_cut_a_number_off_from_its_unit():
    assert [q.value for q in extract_quantities("net loss of $(5.2) billion")] == [5.2e9]
    assert fact_checker("Net loss was $5.2 million", ["net loss of $(5.2) billion"])["status"] != "supported"
    assert fact_checker("Net loss was $5.2 billion", ["net loss of $(5.2) billion"])["status"] == "supported"
    assert extract_quantities("as reported (2024)") == []  # a bracketed year is still not a quantity


def test_between_x_and_y_gives_the_unit_to_both_ends():
    assert [q.value for q in extract_quantities("capex between 5 and 7 billion")] == [5e9, 7e9]
    assert [q.value for q in extract_quantities("between 2023 and 2025")] == []


def test_yfinance_year_ending_in_the_first_days_of_january_is_named_like_the_sec_path(monkeypatch):
    class Frame:
        columns = [datetime(2022, 1, 1)]
        index = ["Total Revenue"]
        empty = False
        at = {("Total Revenue", columns[0]): 14_181_000_000.0}

    fake = types.ModuleType("yfinance")
    fake.Ticker = lambda ticker: types.SimpleNamespace(income_stmt=Frame())
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    data = financial_api._yfinance_statements("K", "income_statement", "annual", 1, reason="test")

    assert [(r["fiscal_year"], r["period_end"]) for r in data["data"]] == [(2021, "2022-01-01")]


def test_a_step_that_lists_itself_as_a_dependency_is_kept():
    raw = [{"id": "s1", "tool": "company_profile", "args": {"ticker": "AAPL"}, "depends_on": ["s1"]}]

    steps, _rejected = parse_steps(raw, build_default_registry("mock"))

    assert [(s.id, s.depends_on) for s in steps] == [("s1", [])]


def test_a_breaker_with_a_zero_window_does_not_divide_by_zero():
    breaker = CircuitBreaker("provider", BreakerConfig(window=0))

    assert breaker.record_failure() is False


def test_trace_fields_cannot_replace_the_event_envelope():
    event = TraceLogger().log("note", text="x", **{"type": "other", "seq": 99})

    assert event["type"] == "note" and event["seq"] != 99 and event["text"] == "x"


def test_a_transcript_url_that_starts_with_the_ticker_is_found(monkeypatch):
    from tools import earnings, sec_edgar, web_search

    url = "https://www.fool.com/earnings/call-transcripts/2026/04/29/bip-q1-2026-earnings-call-transcript/"
    fetched: list[str] = []
    monkeypatch.setattr(web_search, "live_web_search", lambda query, num_results=10: {"results": [{"url": url}]})
    monkeypatch.setattr(sec_edgar, "_http", lambda: types.SimpleNamespace(get=lambda u, **kw: fetched.append(u) or b""))
    monkeypatch.setattr(earnings, "parse_fool_transcript", lambda body: ("2026-04-29", []))

    with pytest.raises(Exception, match="could not parse"):  # found and fetched; the empty fake page has no turns
        earnings.fool_transcript("BIP", "Q1", 2026)
    assert fetched == [url]


def test_the_dashboard_shows_every_target_a_metric_was_scored_against():
    from evaluation.dashboard import _rows

    def run(target: str) -> dict:
        return {"metrics": [{"id": "AB-5", "name": "Latency", "category": "Agent Behavior", "target": target,
                             "unit": "seconds", "value": 50.0, "passed": True}]}

    scores = {"runs": [run("< 300 s"), run("< 600 s"), run("< 300 s")], "aggregate": {}}

    targets = [m["target"] for _category, metrics in _rows(scores) for m in metrics if m["id"] == "AB-5"]
    assert targets == ["< 300 s / < 600 s"]


def test_two_claims_from_one_provider_do_not_outvote_a_filing():
    from synthesis.conflict_resolver import Claim, Period, ReliabilityConfig, resolve

    def claim(n: int, value: float, provider: str, tier: int) -> Claim:
        return Claim(claim_id="", entity="XYZ", metric="revenue", period=Period(2024, "FY"), value=value, unit="USD",
                     evidence_ids=[f"E{n}"], source_tiers=[tier], provider=provider)

    claims = [claim(1, 120e6, "yfinance", 2), claim(2, 120e6, "peer_comparison:yfinance", 2),
              claim(3, 100e6, "sec_xbrl", 1)]

    resolved, conflicts = resolve(claims, ReliabilityConfig())

    assert resolved[0].value == 100e6 and conflicts[0].method == "highest_tier"
