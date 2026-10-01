"""Tests for the tool registry (Day 2) and the tool implementations.

Run with:  python -m pytest
"""

from __future__ import annotations

import json
import re
from datetime import date

import pytest
from jsonschema import Draft202012Validator

from memory.vector_store import DEFAULT_STORE
from tools import (
    SCHEMA_DIR,
    ToolExecutionError,
    ToolLockedError,
    ToolNotFoundError,
    ToolOutputError,
    ToolRegistry,
    ToolValidationError,
    build_default_registry,
)
from tools.fact_checker import extract_quantities
from tools.web_search import MOCK_TODAY

EXPECTED_TOOLS = {
    "calculation_engine",
    "company_profile",
    "earnings_transcript",
    "fact_checker",
    "financial_data_api",
    "news_sentiment",
    "peer_comparison",
    "report_generator",
    "sec_filing_search",
    "vector_db_search",
    "vector_db_store",
    "web_search",
}
SCHEMA_FILES = sorted(SCHEMA_DIR.glob("*.json"))


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def registry() -> ToolRegistry:
    return build_default_registry("mock")


@pytest.fixture(autouse=True)
def _empty_memory():
    DEFAULT_STORE.clear()
    yield
    DEFAULT_STORE.clear()


# ==================================================================================================
# Schemas
# ==================================================================================================


def test_all_twelve_tools_are_registered(registry):
    assert set(registry.names) == EXPECTED_TOOLS
    assert len(registry.available()) == 12


@pytest.mark.parametrize("path", SCHEMA_FILES, ids=lambda p: p.stem)
def test_schema_follows_openai_function_format(path):
    schema = json.loads(path.read_text(encoding="utf-8"))
    assert path.stem == schema["name"], "file name must match the tool name"
    assert re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", schema["name"])
    assert 20 <= len(schema["description"]) <= 1024

    params = schema["parameters"]
    Draft202012Validator.check_schema(params)
    assert params["type"] == "object"
    assert params.get("additionalProperties") is False, "reject arguments the tool does not understand"
    assert set(params["required"]) <= set(params["properties"])
    for name, prop in params["properties"].items():
        assert prop.get("description"), f"property {name!r} needs a description for the LLM"

    meta = schema["x-ara"]
    assert meta["handlers"], "at least one handler is required"
    Draft202012Validator.check_schema(meta["output_schema"])


def test_openai_payload_exposes_only_llm_fields(registry):
    for tool in registry.openai_tools():
        assert tool["type"] == "function"
        assert set(tool["function"]) == {"name", "description", "parameters"}
        assert "x-ara" not in json.dumps(tool)


def test_openai_payload_is_a_copy(registry):
    registry.openai_tools()[0]["function"]["parameters"]["type"] = "tampered"
    assert registry.openai_tools()[0]["function"]["parameters"]["type"] == "object"


def test_describe_marks_optional_arguments(registry):
    text = registry.describe()
    line = next(row for row in text.splitlines() if row.startswith("- sec_filing_search("))
    assert "ticker: string" in line and "year?: integer" in line


def test_every_network_tool_declares_fallbacks(registry):
    for name in ("sec_filing_search", "financial_data_api", "web_search", "news_sentiment",
                 "earnings_transcript", "company_profile", "peer_comparison"):
        spec = registry.get(name)
        assert len(spec.fallbacks) >= 2, name
        assert set(spec.fallbacks) <= EXPECTED_TOOLS, f"{name} falls back to an unknown tool"


# ==================================================================================================
# Argument validation
# ==================================================================================================


def test_missing_required_argument(registry):
    with pytest.raises(ToolValidationError, match="filing_type"):
        registry.dispatch("sec_filing_search", {"ticker": "AAPL"})


def test_enum_violation(registry):
    with pytest.raises(ToolValidationError, match="10-X"):
        registry.dispatch("sec_filing_search", {"ticker": "AAPL", "filing_type": "10-X"})


def test_unknown_argument_is_rejected(registry):
    with pytest.raises(ToolValidationError, match="limit"):
        registry.dispatch("web_search", {"query": "nvidia", "limit": 3})


def test_wrong_type_and_range(registry):
    with pytest.raises(ToolValidationError):
        registry.dispatch("web_search", {"query": "nvidia", "num_results": "ten"})
    with pytest.raises(ToolValidationError):
        registry.dispatch("web_search", {"query": "nvidia", "num_results": 500})


def test_ticker_pattern(registry):
    with pytest.raises(ToolValidationError):
        registry.dispatch("company_profile", {"ticker": "DROP TABLE"})


def test_defaults_are_filled(registry):
    args = registry.validate_args("financial_data_api", {"ticker": "MSFT", "statement_type": "income_statement"})
    assert args == {"ticker": "MSFT", "statement_type": "income_statement", "period": "annual", "years": 3}


def test_harmless_llm_argument_mistakes_are_repaired(registry):
    # null for an optional argument means "not given"
    args = {"ticker": "AAPL", "filing_type": "10-K", "year": None}
    assert registry.validate_args("sec_filing_search", args) == {"ticker": "AAPL", "filing_type": "10-K"}
    assert registry.dispatch("sec_filing_search", args).data["period_of_report"] == "2024-09-28"
    # scalars sent as strings, including inside nested objects
    assert registry.validate_args("web_search", {"query": "nvidia", "num_results": " 5 "})["num_results"] == 5
    nested = registry.validate_args("vector_db_search", {"query": "supply chain",
                                                         "filter": {"verified": "TRUE", "min_confidence": "0.8", "ticker": None}})
    assert nested["filter"] == {"verified": True, "min_confidence": 0.8}
    # tickers are normalised, so both spellings share one cache entry
    registry.dispatch("company_profile", {"ticker": " msft "})
    assert registry.dispatch("company_profile", {"ticker": "MSFT"}).cache_hit


@pytest.mark.parametrize(
    "tool, args, message",
    [
        ("web_search", {"query": "nvidia", "num_results": "five"}, "num_results"),  # not a number
        ("web_search", {"query": "nvidia", "num_results": "5.5"}, "num_results"),  # not an integer
        ("sec_filing_search", {"ticker": None, "filing_type": "10-K"}, "ticker"),  # null for a required argument
        ("web_search", {"query": "nvidia", "bogus": None}, "bogus"),  # unknown arguments are still rejected
    ],
)
def test_unrepairable_arguments_are_still_rejected(registry, tool, args, message):
    with pytest.raises(ToolValidationError, match=message):
        registry.validate_args(tool, args)


def test_json_string_arguments_from_llm(registry):
    result = registry.dispatch("company_profile", '{"ticker": "MSFT"}')
    assert result.data["name"] == "Microsoft Corporation"
    with pytest.raises(ToolValidationError, match="not valid JSON"):
        registry.dispatch("company_profile", '{"ticker": ')
    with pytest.raises(ToolValidationError, match="JSON object"):
        registry.dispatch("company_profile", "[1, 2]")


# ==================================================================================================
# Dispatch, provenance, caching, errors
# ==================================================================================================


def test_result_envelope_carries_provenance(registry):
    result = registry.dispatch("financial_data_api", {"ticker": "MSFT", "statement_type": "income_statement"})
    assert result.evidence_id == "E1"
    assert result.tool == "financial_data_api"
    assert result.source_tier == 1
    assert result.is_mock is True
    assert result.cache_hit is False
    assert result.args["years"] == 3
    assert result.retrieved_at.endswith("+00:00")


def test_local_tools_are_not_flagged_as_mock(registry):
    result = registry.dispatch("calculation_engine", {"calculation_type": "ratio", "inputs": {"numerator": 1, "denominator": 4}})
    assert result.is_mock is False


def test_cache_hit_then_expiry():
    clock = FakeClock()
    registry = build_default_registry("mock", clock=clock)
    args = {"ticker": "MSFT", "statement_type": "income_statement"}

    first = registry.dispatch("financial_data_api", args)
    second = registry.dispatch("financial_data_api", dict(args, years=3))  # same call once defaults are filled
    assert second.cache_hit and second.evidence_id == first.evidence_id

    clock.now += registry.get("financial_data_api").cache_ttl_s + 1
    third = registry.dispatch("financial_data_api", args)
    assert not third.cache_hit and third.evidence_id != first.evidence_id


def test_uncached_tool_runs_every_time(registry):
    args = {"calculation_type": "ratio", "inputs": {"numerator": 1, "denominator": 2}}
    assert registry.dispatch("calculation_engine", args).evidence_id == "E1"
    assert registry.dispatch("calculation_engine", args).evidence_id == "E2"


def test_unknown_tool(registry):
    with pytest.raises(ToolNotFoundError):
        registry.dispatch("bloomberg_terminal", {})
    assert registry.call_log[-1].error_kind == "not_found"


def test_malformed_output_is_caught(registry):
    registry.bind_handler("company_profile", lambda ticker: {"ticker": ticker})  # missing "name"
    with pytest.raises(ToolOutputError, match="name"):
        registry.dispatch("company_profile", {"ticker": "MSFT"})
    registry.bind_handler("company_profile", lambda ticker: ["not", "an", "object"])
    with pytest.raises(ToolOutputError):
        registry.dispatch("company_profile", {"ticker": "MSFT"})


def test_handler_crash_becomes_execution_error(registry):
    def broken(ticker):
        raise ConnectionError("upstream reset")

    registry.bind_handler("company_profile", broken)
    with pytest.raises(ToolExecutionError, match="ConnectionError"):
        registry.dispatch("company_profile", {"ticker": "MSFT"})


def test_call_log_records_successes_and_failures(registry):
    registry.dispatch("company_profile", {"ticker": "MSFT"})
    with pytest.raises(ToolExecutionError):
        registry.dispatch("company_profile", {"ticker": "ZZZZ"})
    ok, failed = registry.call_log
    assert ok.ok and ok.evidence_id == "E1" and ok.error_kind is None
    assert not failed.ok and failed.error_kind == "execution_error" and "ZZZZ" in failed.error_message


def test_every_tool_has_a_live_handler():
    # Day 7 completed the set: no tool falls back to canned data in live mode.
    live = build_default_registry("live")
    assert {s.name for s in live.available()} == set(live.names)
    assert all(live.get(name).handler_mode in ("live", "any") for name in live.names)


def test_invalid_mode():
    with pytest.raises(ValueError):
        ToolRegistry("production")


# ==================================================================================================
# Unlocks and dynamic registration (spec B4.4)
# ==================================================================================================


def test_unlocks_are_off_by_default(registry):
    assert registry.is_available("earnings_transcript")


def test_progression_unlocks_when_enforced():
    registry = build_default_registry("mock", enforce_unlocks=True)
    locked = {"earnings_transcript", "peer_comparison", "calculation_engine"}
    assert {s.name for s in registry.available()} == EXPECTED_TOOLS - locked
    assert all(t["function"]["name"] not in locked for t in registry.openai_tools())
    with pytest.raises(ToolLockedError, match="after challenge 2"):
        registry.dispatch("earnings_transcript", {"ticker": "AAPL", "quarter": "Q4", "year": 2024})

    assert registry.set_progress(2) == ["earnings_transcript"]
    assert registry.set_progress(4) == ["peer_comparison"]
    assert registry.set_progress(6) == ["calculation_engine"]
    assert len(registry.available()) == 12


def test_new_tool_added_from_a_schema_file_without_code_changes(tmp_path, registry):
    schema = {
        "name": "sentiment_aggregate",
        "description": "Aggregate a list of sentiment scores into an overall label.",
        "parameters": {
            "type": "object",
            "properties": {"scores": {"type": "array", "items": {"type": "number"}, "description": "Scores in [-1, 1]."}},
            "required": ["scores"],
            "additionalProperties": False,
        },
        "x-ara": {"handlers": {"any": "tools.news_sentiment:aggregate"}},
    }
    (tmp_path / "sentiment_aggregate.json").write_text(json.dumps(schema), encoding="utf-8")

    assert registry.load_directory(tmp_path) == ["sentiment_aggregate"]
    assert registry.is_available("sentiment_aggregate")
    result = registry.dispatch("sentiment_aggregate", {"scores": [0.5, 0.3, -0.1]})
    assert result.data["label"] == "positive"


@pytest.mark.parametrize(
    "broken, message",
    [
        ({"name": "bad name!", "description": "x", "parameters": {"type": "object"}}, "Invalid tool name"),
        ({"name": "t", "description": "", "parameters": {"type": "object"}}, "no description"),
        ({"name": "t", "description": "d", "parameters": {"type": "array"}}, "type 'object'"),
        ({"name": "t", "description": "d", "parameters": {"type": "object", "properties": {}, "required": ["x"]}}, "not declared"),
        ({"name": "t", "description": "d", "parameters": {"type": "object"}, "x-ara": {"handlers": {"any": "nope"}}}, "module:function"),
    ],
)
def test_bad_tool_definitions_are_rejected(registry, broken, message):
    with pytest.raises(ValueError, match=message):
        registry.register_schema(broken)


# ==================================================================================================
# calculation_engine
# ==================================================================================================


def calc(registry, calculation_type, **inputs):
    return registry.dispatch("calculation_engine", {"calculation_type": calculation_type, "inputs": inputs}).data


def test_growth_rate_uses_absolute_base(registry):
    assert calc(registry, "growth_rate", current=245122, previous=211915)["result"] == pytest.approx(0.156699, abs=1e-6)
    # Improving from a loss of -100 to -50 is +50 %, not -50 %.
    assert calc(registry, "percent_change", current=-50, previous=-100)["result"] == pytest.approx(0.5)


def test_cagr(registry):
    out = calc(registry, "cagr", begin_value=198270, end_value=245122, years=2)
    assert out["result"] == pytest.approx((245122 / 198270) ** 0.5 - 1)
    assert len(out["steps"]) == 2


def test_margin_and_ratio(registry):
    assert calc(registry, "margin", numerator=109433, revenue=245122)["result"] == pytest.approx(0.446443, abs=1e-6)
    assert calc(registry, "ratio", numerator=3, denominator=4)["result"] == 0.75


def test_dcf_matches_hand_calculation(registry):
    out = calc(registry, "dcf", cash_flows=[100, 100, 100], discount_rate=0.10, terminal_growth_rate=0.02,
               net_debt=200, shares_outstanding=10)
    r = out["result"]
    pv_flows = 100 / 1.1 + 100 / 1.1**2 + 100 / 1.1**3
    pv_terminal = (100 * 1.02 / 0.08) / 1.1**3
    assert r["pv_cash_flows"] == pytest.approx(pv_flows)
    assert r["enterprise_value"] == pytest.approx(pv_flows + pv_terminal)
    assert r["value_per_share"] == pytest.approx((pv_flows + pv_terminal - 200) / 10)


def test_descriptive_stats(registry):
    r = calc(registry, "descriptive_stats", values=[1, 2, 3, 4])["result"]
    assert r["mean"] == 2.5 and r["median"] == 2.5 and r["count"] == 4


@pytest.mark.parametrize(
    "calculation_type, inputs, message",
    [
        ("growth_rate", {"current": 5, "previous": 0}, "must not be zero"),
        ("growth_rate", {"current": 5}, "missing input 'previous'"),
        ("cagr", {"begin_value": -1, "end_value": 5, "years": 2}, "must be > 0"),
        ("dcf", {"cash_flows": [1], "discount_rate": 0.02, "terminal_growth_rate": 0.03}, "must exceed"),
        ("ratio", {"numerator": True, "denominator": 1}, "finite number"),
        ("descriptive_stats", {"values": []}, "at least 1"),
    ],
)
def test_calculation_input_errors(registry, calculation_type, inputs, message):
    with pytest.raises(ToolValidationError, match=message):
        registry.dispatch("calculation_engine", {"calculation_type": calculation_type, "inputs": inputs})


def test_numbers_sent_as_strings_inside_calculation_inputs(registry):
    assert calc(registry, "ratio", numerator="3", denominator=" 4 ")["result"] == 0.75


@pytest.mark.parametrize(
    "calculation_type, inputs, message",
    [
        ("ratio", {"numerator": 1e300, "denominator": 1e-300}, "non-finite"),  # overflow must not return inf
        ("ratio", {"numerator": "245,122", "denominator": 1}, "plain number"),  # ambiguous separator
        ("dcf", {"cash_flows": [100], "discount_rate": 9, "terminal_growth_rate": 0.02}, "fraction"),  # 9 meant 9 %
        ("dcf", {"cash_flows": [100], "discount_rate": -0.5, "terminal_growth_rate": -0.6}, "between 0 and 1"),
    ],
)
def test_calculation_guards_against_unit_mistakes(registry, calculation_type, inputs, message):
    with pytest.raises(ToolValidationError, match=message):
        registry.dispatch("calculation_engine", {"calculation_type": calculation_type, "inputs": inputs})


# ==================================================================================================
# financial_data_api, company_profile, peer_comparison
# ==================================================================================================


def test_financial_data_annual_income(registry):
    data = registry.dispatch("financial_data_api", {"ticker": "msft", "statement_type": "income_statement"}).data
    assert [row["fiscal_year"] for row in data["data"]] == [2024, 2023, 2022]
    assert data["data"][0]["values"]["revenue"] == 245122
    assert data["unit"] == "millions" and data["fiscal_year_end"] == "0630"


def test_financial_data_key_ratios_are_derived(registry):
    row = registry.dispatch("financial_data_api", {"ticker": "NVDA", "statement_type": "key_ratios", "years": 1}).data["data"][0]
    assert row["period_end"] == "2025-01-26"  # NVIDIA's fiscal year ends in late January
    assert row["values"]["gross_margin"] == pytest.approx(97858 / 130497, abs=1e-4)
    assert row["values"]["fcf_margin"] == pytest.approx((64089 - 3236) / 130497, abs=1e-4)


def test_financial_data_quarterly(registry):
    rows = registry.dispatch("financial_data_api", {"ticker": "AAPL", "statement_type": "income_statement",
                                                     "period": "quarterly", "years": 1}).data["data"]
    assert [r["fiscal_period"] for r in rows] == ["Q4", "Q3", "Q2", "Q1"]
    assert sum(r["values"]["revenue"] for r in rows) == 391035  # quarters add up to the fiscal year
    with pytest.raises(ToolExecutionError, match="quarterly"):
        registry.dispatch("financial_data_api", {"ticker": "MSFT", "statement_type": "balance_sheet", "period": "quarterly"})


def test_financial_data_unknown_ticker(registry):
    with pytest.raises(ToolExecutionError, match="ZZZZ"):
        registry.dispatch("financial_data_api", {"ticker": "ZZZZ", "statement_type": "income_statement"})


def test_company_profile(registry):
    msft = registry.dispatch("company_profile", {"ticker": "MSFT"}).data
    assert msft["fiscal_year_end"] == "0630" and msft["executives"][0]["name"] == "Satya Nadella"
    partial = registry.dispatch("company_profile", {"ticker": "GOOGL"}).data
    assert partial["name"] == "Alphabet Inc." and "sector" not in partial


def test_peer_comparison(registry):
    data = registry.dispatch("peer_comparison", {"ticker": "NVDA", "num_peers": 3,
                                                 "metrics": ["operating_margin", "pe_ratio"]}).data
    assert data["peers"] == ["AMD", "AVGO", "QCOM"]
    assert data["rankings"]["operating_margin"][0] == "NVDA"
    assert set(data["matrix"][0]["values"]) == {"operating_margin", "pe_ratio"}


def test_peer_ranking_puts_missing_values_last(registry):
    data = registry.dispatch("peer_comparison", {"ticker": "TSLA", "metrics": ["pe_ratio"]}).data
    assert data["rankings"]["pe_ratio"][-1] == "RIVN"  # negative earnings -> no P/E


# ==================================================================================================
# sec_filing_search, earnings_transcript
# ==================================================================================================


def test_sec_10k_has_structured_risk_factors(registry):
    data = registry.dispatch("sec_filing_search", {"ticker": "TSLA", "filing_type": "10-K"}).data
    risk = next(s for s in data["sections"] if s["item"] == "1A")
    assert len(risk["text"].splitlines()) >= 10
    assert data["period_of_report"] == "2024-12-31"
    assert date.fromisoformat(data["filing_date"]) > date.fromisoformat(data["period_of_report"])


def test_sec_filing_for_missing_year(registry):
    with pytest.raises(ToolExecutionError, match="available"):
        registry.dispatch("sec_filing_search", {"ticker": "AAPL", "filing_type": "10-K", "year": 2015})


def test_earnings_transcript_has_qa(registry):
    data = registry.dispatch("earnings_transcript", {"ticker": "AAPL", "quarter": "Q4", "year": 2024}).data
    assert data["source_kind"] == "transcript"
    assert {s["section"] for s in data["segments"]} == {"prepared", "qa"}


def test_earnings_transcript_falls_back_to_press_release(registry):
    data = registry.dispatch("earnings_transcript", {"ticker": "MSFT", "quarter": "Q2", "year": 2024}).data
    assert data["source_kind"] == "press_release"
    assert any("unavailable" in n for n in data["notes"])


# ==================================================================================================
# web_search, news_sentiment
# ==================================================================================================


def test_web_search_is_deterministic_and_respects_limits(registry):
    a = registry.dispatch("web_search", {"query": "Palantir growth", "num_results": 4, "date_range": "week"}).data
    registry.clear_cache()
    b = registry.dispatch("web_search", {"query": "Palantir growth", "num_results": 4, "date_range": "week"}).data
    assert a == b and len(a["results"]) == 4
    for r in a["results"]:
        assert (MOCK_TODAY - date.fromisoformat(r["published"])).days < 7


def test_news_sentiment_overall_is_consistent(registry):
    data = registry.dispatch("news_sentiment", {"query": "Tesla", "num_articles": 15}).data
    scores = [a["sentiment_score"] for a in data["articles"]]
    assert data["overall"]["score"] == pytest.approx(sum(scores) / len(scores), abs=1e-4)
    assert data["overall"]["positive"] + data["overall"]["negative"] + data["overall"]["neutral"] == pytest.approx(1)


# ==================================================================================================
# fact_checker
# ==================================================================================================


def check(registry, claim, *sources):
    return registry.dispatch("fact_checker", {"claim": claim, "sources": list(sources)}).data


def test_fact_check_across_scales(registry):
    out = check(registry, "Microsoft revenue was $245.1 billion in fiscal 2024.",
                "Total revenue was $245,122 million, up 16%.")
    assert out["status"] == "supported"
    assert out["confidence"] == pytest.approx(0.9)
    assert not out["evidence"][0]["scale_inferred"]


def test_fact_check_unitless_table_value_gets_lower_confidence(registry):
    out = check(registry, "Revenue was $245.1 billion.", "Revenue | 245,122 | 211,915")
    assert out["status"] == "supported"
    assert out["evidence"][0]["scale_inferred"] and out["confidence"] < 0.9


def test_fact_check_independent_sources_raise_confidence(registry):
    out = check(registry, "Revenue was $245.1 billion.", "Revenue of $245.1bn.", "Sales reached 245,122 million.")
    assert out["confidence"] == pytest.approx(0.95)


def test_fact_check_outside_tolerance(registry):
    out = check(registry, "Operating margin was 52%.", "Operating margin was 44.6%.")
    assert out["status"] == "unsupported"


def test_fact_check_partial_and_unverifiable(registry):
    partial = check(registry, "Revenue was $245.1 billion and net income $95 billion.", "Revenue $245,122 million; net income $88,136 million.")
    assert partial["status"] == "partially_supported" and 0 < partial["confidence"] < 0.9
    assert check(registry, "Management sounded optimistic.", "Some text")["status"] == "unverifiable"
    assert registry.dispatch("fact_checker", {"claim": "Revenue was $5 billion."}).data["status"] == "unverifiable"


def test_quantity_extraction_ignores_labels():
    quantities = extract_quantities("Q4 FY24 revenue per the 10-K and 8-K was $94.9 billion.")
    assert [q.value for q in quantities] == [pytest.approx(94.9e9)]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Operating margin expanded 150 bps.", [("percent", 0.015)]),
        ("up 25 basis points", [("percent", 0.0025)]),
        ("margin rose 1.5 percentage points", [("percent", 0.015)]),
        ("Guidance of 10-12% growth", [("percent", 0.10), ("percent", 0.12)]),
        ("growth of 10 to 12 percent", [("percent", 0.10), ("percent", 0.12)]),
        ("guidance of $10-12 billion", [("amount", 10e9), ("amount", 12e9)]),
        ("150 to 200 bps", [("percent", 0.015), ("percent", 0.02)]),
        ("filed on 2024-09-28", []),
        ("FY2023-2024 revenue", []),
        ("a 3-for-1 stock split", []),
        ("margin of -5%", [("percent", -0.05)]),
    ],
)
def test_quantity_units(text, expected):
    got = [(q.kind, q.value) for q in extract_quantities(text)]
    assert [k for k, _ in got] == [k for k, _ in expected]
    assert [v for _, v in got] == pytest.approx([v for _, v in expected])


def test_a_range_claim_matches_a_spelled_out_range(registry):
    assert check(registry, "Guidance was $10-12 billion.", "Guided to $10 billion to $12 billion.")["status"] == "supported"


def test_basis_points_and_percentage_points_agree(registry):
    out = check(registry, "Operating margin expanded 150 bps.", "Operating margin rose 1.5 percentage points.")
    assert out["status"] == "supported"


def test_quantity_extraction_keeps_numbers_before_punctuation():
    quantities = extract_quantities("Sales rose to 500, while costs fell to 1,200. Margin was 12.5%.")
    assert [q.value for q in quantities] == [500, 1200, pytest.approx(0.125)]


def test_labels_do_not_weaken_a_correct_claim(registry):
    out = check(registry, "Q4 revenue per the 10-K was $94.9 billion.", "Revenue $94,930 million.")
    assert out["status"] == "supported"


def test_quantity_extraction_ignores_years():
    quantities = extract_quantities("In 2024 revenue grew 12.5% to $3.2 billion from 2,850 million.")
    assert [(q.kind, q.value) for q in quantities] == [
        ("percent", pytest.approx(0.125)),
        ("amount", pytest.approx(3.2e9)),
        ("amount", pytest.approx(2.85e9)),
    ]


# ==================================================================================================
# report_generator
# ==================================================================================================


def test_report_flags_missing_sections_and_bad_citations(registry):
    data = registry.dispatch("report_generator", {
        "template": "company_profile",
        "title": "Microsoft Corporation: Company Profile",
        "sections": {
            "executive_summary": "Revenue grew 15.7% [E1], per the 10-K [E9].",
            "business_overview": "Three segments [E2].",
        },
        "sources": [
            {"id": "E1", "title": "Income statement FY2024", "source_type": "financial_data", "date": "2024-06-30", "is_mock": True},
            {"id": "E2", "title": "Form 10-K", "url": "https://www.sec.gov/example"},
            {"id": "E3", "title": "Unused source"},
        ],
    }).data
    md = data["markdown"]
    assert md.startswith("# Microsoft Corporation: Company Profile")
    assert data["missing_sections"] == ["financial_summary", "key_executives", "recent_developments"]
    assert "- Section not completed: Key Executives." in md
    assert data["unknown_citations"] == ["E9"]
    assert data["uncited_sources"] == ["E3"]
    assert data["contains_mock_data"] and "MOCK data" in md
    assert "- [E2] Form 10-K. https://www.sec.gov/example" in md


def test_citation_lists_in_every_llm_format_are_recognised(registry):
    data = registry.dispatch("report_generator", {
        "template": "company_profile",
        "sections": {"executive_summary": "Grew [E1, E2]; margin [E3][E4]; see [E5; E6] and [E7 E8]. Not a cite: [Exhibit 1]."},
        "sources": [{"id": f"E{i}", "title": "t"} for i in range(1, 9)],
    }).data
    assert data["uncited_sources"] == [] and data["unknown_citations"] == []


def test_headings_inside_sections_are_nested_under_the_section(registry):
    md = registry.dispatch("report_generator", {
        "template": "company_profile", "title": "Report",
        "sections": {"executive_summary": "# Big claim\nText [E1]\n## Detail\nMore"},
        "sources": [{"id": "E1", "title": "t"}],
    }).data["markdown"]
    assert md.count("\n# ") + md.startswith("# ") == 1  # only the report title is an H1
    assert "### Big claim" in md and "#### Detail" in md


def test_complete_report_has_no_gaps(registry):
    sections = {s: f"Body of {s} [E1]." for s in
                ["executive_summary", "financial_risks", "operational_risks", "regulatory_risks", "competitive_risks", "risk_matrix"]}
    data = registry.dispatch("report_generator", {"template": "risk_assessment", "sections": sections,
                                                  "sources": [{"id": "E1", "title": "10-K"}]}).data
    assert data["missing_sections"] == [] and "No data gaps were recorded." in data["markdown"]


# ==================================================================================================
# vector_db_store / vector_db_search
# ==================================================================================================


def test_memory_round_trip_with_filters(registry):
    stored = registry.dispatch("vector_db_store", {
        "content": "Tesla faces supply chain risks from battery cell suppliers.",
        "metadata": {"ticker": "tsla", "date": "2025-01-29", "source_type": "10-K", "verified": True, "confidence": 0.9},
    }).data
    assert stored["id"] == "tsla-10-k-2025-01-29-001"
    registry.dispatch("vector_db_store", {
        "content": "Apple supply chain concentrated in China.",
        "metadata": {"ticker": "AAPL", "date": "2024-11-01", "source_type": "10-K"},
    })

    hits = registry.dispatch("vector_db_search", {"query": "supply chain risks"}).data["results"]
    assert hits[0]["id"] == stored["id"] and len(hits) == 2

    only_tesla = registry.dispatch("vector_db_search", {"query": "supply chain", "filter": {"ticker": "TSLA"}}).data
    assert [h["metadata"]["ticker"] for h in only_tesla["results"]] == ["TSLA"]
    verified = registry.dispatch("vector_db_search", {"query": "supply chain", "filter": {"verified": True}}).data
    assert len(verified["results"]) == 1


def test_memory_search_accepts_a_top_level_ticker(registry):
    for ticker in ("TSLA", "AAPL"):
        registry.dispatch("vector_db_store", {"content": f"{ticker} supply chain finding.",
                                              "metadata": {"ticker": ticker, "date": "2025-01-01", "source_type": "news"}})
    hits = registry.dispatch("vector_db_search", {"query": "supply chain", "ticker": "tsla"}).data["results"]
    assert [h["metadata"]["ticker"] for h in hits] == ["TSLA"]


def test_memory_store_requires_provenance(registry):
    with pytest.raises(ToolValidationError, match="source_type"):
        registry.dispatch("vector_db_store", {"content": "A finding without a source.", "metadata": {"ticker": "X", "date": "2025-01-01"}})


# ==================================================================================================
# End to end: evidence ids flow from retrieval to the final report
# ==================================================================================================


def test_evidence_flows_from_tools_into_a_cited_report(registry):
    fin = registry.dispatch("financial_data_api", {"ticker": "MSFT", "statement_type": "income_statement", "years": 2})
    fy24, fy23 = (row["values"]["revenue"] for row in fin.data["data"])
    growth = registry.dispatch("calculation_engine", {"calculation_type": "growth_rate",
                                                      "inputs": {"current": fy24, "previous": fy23}})
    claim = f"Revenue grew {growth.data['result']:.1%} to ${fy24 / 1000:.1f} billion."
    verdict = registry.dispatch("fact_checker", {"claim": claim, "sources": [f"Revenue {fy24} million", "growth 15.7%"]})
    assert verdict.data["status"] == "supported"

    report = registry.dispatch("report_generator", {
        "template": "company_profile",
        "sections": {"executive_summary": f"{claim} [{fin.evidence_id}][{growth.evidence_id}]"},
        "sources": [
            {"id": fin.evidence_id, "title": "Income statement", "is_mock": fin.is_mock},
            {"id": growth.evidence_id, "title": "Growth calculation"},
        ],
    }).data
    assert report["unknown_citations"] == [] and report["uncited_sources"] == []
    assert [r.tool for r in registry.call_log] == ["financial_data_api", "calculation_engine", "fact_checker", "report_generator"]


def test_a_claim_in_billions_matches_a_source_in_millions(registry):
    out = check(registry, "| FY2022 | 198.3 |", "FY2022 revenue=198,270 (USD millions)")
    assert out["status"] == "supported" and out["evidence"][0]["scale_inferred"]


# ==================================================================================================
# Audit regressions
# ==================================================================================================


def test_percent_range_with_the_unit_on_both_ends_is_not_negative(registry):
    # "30%-35%": the hyphen is a range dash, not a minus sign on 35 %.
    assert [(q.kind, q.value) for q in extract_quantities("Guidance of 30%-35% margin")] == [
        ("percent", pytest.approx(0.30)), ("percent", pytest.approx(0.35))]
    out = check(registry, "Operating margin guidance is 30%-35%.", "Management guided to 30% to 35% operating margin.")
    assert out["status"] == "supported"


def test_numeric_dates_are_not_quantities(registry):
    assert [q.value for q in extract_quantities("As of 09/28/2024 revenue was $391 billion.")] == [pytest.approx(391e9)]
    out = check(registry, "As of 9/28/2024 revenue was $391 billion.", "Revenue was $391,035 million.")
    assert out["status"] == "supported"


def test_fiscal_period_labels_glued_to_digits_are_not_quantities(registry):
    # "3Q24" and "2H25" are period labels, like "Q3" and "FY24".
    assert [q.value for q in extract_quantities("3Q24 revenue rose 5% and 2H25 margins widened 1.5 pp")] == [
        pytest.approx(0.05), pytest.approx(0.015)]
    out = check(registry, "3Q24 revenue rose 5%.", "Revenue rose 5% year over year.")
    assert out["status"] == "supported"


def test_dcf_accepts_net_debt_sent_as_a_string(registry):
    out = calc(registry, "dcf", cash_flows=[100], discount_rate=0.1, terminal_growth_rate=0.0, net_debt="50",
               shares_outstanding="10")
    ev = 100 / 1.1 + (100 / 0.1) / 1.1
    assert out["result"]["equity_value"] == pytest.approx(ev - 50)
    assert out["result"]["value_per_share"] == pytest.approx((ev - 50) / 10)


def test_integral_floats_for_integer_arguments_become_ints(registry):
    # Some providers deliver every JSON number as a float ("years": 2.0).
    args = registry.validate_args("financial_data_api", {"ticker": "MSFT", "statement_type": "income_statement",
                                                         "years": 2.0})
    assert args["years"] == 2 and type(args["years"]) is int
    assert len(registry.dispatch("financial_data_api", args).data["data"]) == 2
    assert len(registry.dispatch("web_search", {"query": "nvidia", "num_results": 3.0}).data["results"]) == 3
    with pytest.raises(ToolValidationError, match="num_results"):
        registry.validate_args("web_search", {"query": "nvidia", "num_results": 5.5})


def test_name_verifier_ignores_business_phrases_and_companies_next_to_titles():
    from tools.fact_checker import unsupported_names
    src = ["nothing relevant"]
    for text in ("Executive Vice President Worldwide Sales rose.", "The Commercial Business President grew.",
                 "Tesla's Full Self-Driving, said the CEO.", "Wells Fargo & Company CEO"):
        assert unsupported_names(text, src) == [], text
    # The company before the title is not a person; the person after it still is.
    assert unsupported_names("Goldman Sachs CEO David Solomon said", src) == ["David Solomon"]
    assert unsupported_names("Goldman Sachs CEO David Solomon said", ["David Solomon leads the bank"]) == []


def test_name_verifier_handles_titles_inside_runs_hyphens_and_apostrophes():
    from tools.fact_checker import unsupported_names
    src = ["nothing relevant"]
    assert unsupported_names("Bank of America Chairman Brian Moynihan", src) == ["Brian Moynihan"]
    assert unsupported_names("CEO Jean-Pierre Lacroix said", src) == ["Jean-Pierre Lacroix"]
    assert unsupported_names("CEO Pat O'Brien said", src) == ["Pat O'Brien"]
    assert unsupported_names("CEO Zoë Smith said", src) == ["Zoë Smith"]
    assert unsupported_names("CEO Zoë Smith said", ["zoë smith, chief executive"]) == []
    assert unsupported_names("Dr. Lisa Su, CEO of AMD", ["Lisa Su"]) == []

def test_name_verifier_accepts_familiar_first_names_but_not_other_surnames():
    # Day 11 C6: sources list "James  Dimon" and "Charles W. Scharf"; the report wrote "Jamie Dimon".
    from tools.fact_checker import unsupported_names
    src = ["executives: James  Dimon, Chairman & CEO; Charles W. Scharf, CEO; David M. Solomon, CEO"]
    assert unsupported_names("Jamie Dimon, CEO, leads the bank.", src) == []
    assert unsupported_names("Charlie Scharf, CEO, runs it.", src) == []
    assert unsupported_names("Charlie Schwab, CEO, runs it.", src) == ["Charlie Schwab"]
    assert unsupported_names("David M. Sullivan, Chairman & CEO, said so.", src) == ["David M. Sullivan"]
