"""Debug-pass regression tests for the tool layer (tools/). Mock mode and pure functions only."""

from __future__ import annotations

import inspect
import json

import httpx
import pytest

from tools.http import HttpClient, ToolExecutionError
from tools.tool_registry import SCHEMA_DIR, resolve_handler


def _client(tmp_path, responses):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return responses[min(len(calls), len(responses)) - 1]

    client = HttpClient(sec_user_agent="t t@example.com", cache_dir=tmp_path, max_retries=0,
                        transport=httpx.MockTransport(handler), sleep=lambda s: None)
    return client, calls


def test_a_200_response_that_is_not_json_is_not_cached(tmp_path):
    # A transient HTML error page served with status 200 must not poison the disk cache for its whole TTL.
    client, calls = _client(tmp_path, [httpx.Response(200, content=b"<html>busy</html>"),
                                       httpx.Response(200, content=b'{"ok": true}')])
    url = "https://data.sec.gov/x.json"
    try:
        client.get_json(url, ttl_s=3600)
    except ToolExecutionError:
        pass
    else:
        raise AssertionError("invalid JSON should raise")
    assert client.get_json(url, ttl_s=3600) == {"ok": True}
    assert len(calls) == 2


def test_10q_fiscal_year_follows_the_company_fiscal_calendar():
    from tools.sec_edgar import find_filing

    # Microsoft (fiscal year ends 30 June): the quarter ended 2025-12-31 is fiscal 2026 Q2.
    subs = {"fiscalYearEnd": "0630", "filings": {"recent": {
        "accessionNumber": ["a2", "a1"], "filingDate": ["2026-01-28", "2025-04-30"],
        "reportDate": ["2025-12-31", "2025-03-31"], "form": ["10-Q", "10-Q"],
        "primaryDocument": ["q2.htm", "q1.htm"]}}}
    assert find_filing(subs, "10-Q", 2026)["accessionNumber"] == "a2"
    assert find_filing(subs, "10-Q", 2025)["accessionNumber"] == "a1"
    # A calendar-year filer is unaffected.
    subs["fiscalYearEnd"] = "1231"
    assert find_filing(subs, "10-Q", 2025)["accessionNumber"] == "a2"


@pytest.mark.parametrize("path", sorted(SCHEMA_DIR.glob("*.json")), ids=lambda p: p.stem)
def test_every_schema_matches_the_signature_of_each_of_its_handlers(path):
    schema = json.loads(path.read_text(encoding="utf-8"))
    props, required = schema["parameters"]["properties"], set(schema["parameters"].get("required", []))
    handlers = {mode: dotted for mode, dotted in schema["x-ara"]["handlers"].items() if dotted}
    assert handlers
    for mode, dotted in handlers.items():
        params = inspect.signature(resolve_handler(dotted)).parameters
        assert set(props) <= set(params), f"{mode}: schema properties the handler cannot accept"
        for name, param in params.items():
            assert name in props, f"{mode}: handler parameter {name!r} is not in the schema"
            if param.default is inspect.Parameter.empty:
                assert name in required, f"{mode}: {name!r} is required by the handler but optional in the schema"
            else:
                assert name not in required, f"{mode}: {name!r} has a handler default but is required"
                if "default" in props[name]:
                    assert props[name]["default"] == param.default, f"{mode}: default of {name!r} differs"
        for name in required:
            assert name in params
