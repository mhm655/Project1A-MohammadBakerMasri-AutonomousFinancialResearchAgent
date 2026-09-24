"""Tests for long-term memory (Chroma). Uses the offline hashing embedding, so no model download."""

from __future__ import annotations

import pytest

from memory.vector_store import (
    ChromaVectorStore,
    get_live_store,
    hashing_embedding_function,
    set_live_store,
)
from tools import ToolValidationError, build_default_registry


def make_store(path):
    return ChromaVectorStore(path, embedding_model="hashing", embedding_function=hashing_embedding_function())


FINDINGS = [
    ("Tesla faces supply chain risks from battery cell suppliers.",
     {"ticker": "tsla", "date": "2025-01-29", "source_type": "10-K", "confidence": 0.9, "verified": True}),
    ("Tesla automotive gross margin compressed after price cuts.",
     {"ticker": "TSLA", "date": "2024-10-23", "source_type": "earnings_call", "confidence": 0.7}),
    ("Apple supply chain is concentrated in China and India.",
     {"ticker": "AAPL", "date": "2024-11-01", "source_type": "10-K", "confidence": 0.8, "verified": True}),
]


@pytest.fixture
def store(tmp_path):
    s = make_store(tmp_path / "chroma")
    for content, meta in FINDINGS:
        s.store(content, meta)
    return s


def test_records_follow_the_spec_schema(store):
    hit = store.search("battery supply chain", top_k=1)["results"][0]
    assert hit["id"] == "tsla-10-k-2025-01-29-001"
    meta = hit["metadata"]
    assert meta["ticker"] == "TSLA" and meta["source_type"] == "10-K" and meta["date"] == "2025-01-29"
    assert meta["confidence"] == 0.9 and meta["verified"] is True
    assert "date_ord" not in meta  # internal filter field is hidden
    assert 0 < hit["score"] <= 1


def test_ids_are_sequential_per_ticker_source_and_date(store):
    again = store.store("Another Tesla 10-K finding about battery suppliers.",
                        {"ticker": "TSLA", "date": "2025-01-29", "source_type": "10-K"})
    assert again["id"] == "tsla-10-k-2025-01-29-002" and again["total_records"] == 4


def test_semantic_ranking_puts_the_relevant_record_first(store):
    results = store.search("Tesla gross margin price cuts")["results"]
    assert results[0]["metadata"]["source_type"] == "earnings_call"
    assert results[0]["score"] > results[-1]["score"]


@pytest.mark.parametrize(
    "flt, expected",
    [
        ({"ticker": "tsla"}, {"TSLA"}),
        ({"verified": True}, {"TSLA", "AAPL"}),
        ({"source_type": "earnings_call"}, {"TSLA"}),
        ({"min_confidence": 0.85}, {"TSLA"}),
        ({"date_from": "2024-11-01", "date_to": "2024-12-31"}, {"AAPL"}),
    ],
)
def test_metadata_filters(store, flt, expected):
    results = store.search("supply chain margin", top_k=10, filter=flt)["results"]
    assert {r["metadata"]["ticker"] for r in results} == expected


def test_memory_persists_across_restarts(tmp_path):
    make_store(tmp_path / "db").store("NVIDIA data-center revenue grew strongly.",
                                      {"ticker": "NVDA", "date": "2025-02-26", "source_type": "10-K"})
    reopened = make_store(tmp_path / "db")
    assert reopened.count() == 1
    assert reopened.search("NVIDIA data center")["results"][0]["metadata"]["ticker"] == "NVDA"


def test_empty_store_and_bad_input(tmp_path):
    empty = make_store(tmp_path / "empty")
    assert empty.search("anything")["results"] == []
    with pytest.raises(ToolValidationError):
        empty.search("!!!")
    with pytest.raises(ToolValidationError, match="ISO"):
        empty.store("text", {"ticker": "X", "date": "yesterday", "source_type": "news"})


def test_collections_are_separated_by_embedding_model(tmp_path):
    assert make_store(tmp_path).collection_name == "ara_findings__hashing"


def test_live_tool_handlers_use_chroma(tmp_path):
    set_live_store(make_store(tmp_path / "live"))
    try:
        registry = build_default_registry("live")
        stored = registry.dispatch("vector_db_store", {
            "content": "Microsoft Azure growth drove Intelligent Cloud revenue.",
            "metadata": {"ticker": "MSFT", "date": "2024-07-30", "source_type": "earnings_call"},
        })
        assert stored.is_mock is False and stored.data["id"] == "msft-earnings-call-2024-07-30-001"
        found = registry.dispatch("vector_db_search", {"query": "Azure cloud growth", "filter": {"ticker": "MSFT"}})
        assert found.data["results"][0]["id"] == stored.data["id"]
        assert get_live_store().count() == 1
    finally:
        set_live_store(None)
