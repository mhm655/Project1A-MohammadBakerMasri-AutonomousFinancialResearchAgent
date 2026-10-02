"""Regression tests from the debug pass over memory/ and synthesis/. Offline: hashing embeddings, tmp_path."""

from __future__ import annotations

import random
import re

import pytest

from memory.chunking import MAX_CHARS, MemoryRecord, chunk_news, split_paragraphs
from memory.context_manager import fit_evidence
from memory.vector_store import ChromaVectorStore, InMemoryVectorStore, hashing_embedding_function
from synthesis.conflict_resolver import format_money
from synthesis.narrative import _news_for
from tools.base import ToolResult


def chroma(tmp_path):
    return ChromaVectorStore(tmp_path / "c", embedding_model="hashing", embedding_function=hashing_embedding_function())


def rec(text, **meta):
    base = {"ticker": "msft", "date": "2025-06-30", "source_type": "10-K"}
    return MemoryRecord(text, {**base, **meta})


def test_inmemory_date_filters_match_chroma_for_datetime_strings(tmp_path):
    flt = {"date_from": "2025-01-29T15:00:00Z", "date_to": "2025-01-29T15:00:00Z"}
    results = []
    for store in (InMemoryVectorStore(), chroma(tmp_path)):
        store.store("battery supply risk", {"ticker": "TSLA", "date": "2025-01-29", "source_type": "10-K"})
        results.append(len(store.search("battery supply", filter=flt)["results"]))
    assert results[0] == results[1] == 1


def test_inmemory_verified_filter_is_coerced_like_chroma():
    s = InMemoryVectorStore()
    s.store("battery supply risk", {"ticker": "TSLA", "date": "2025-01-29", "source_type": "10-K", "verified": True})
    assert len(s.search("battery supply", filter={"verified": 1})["results"]) == 1


def test_chroma_add_many_duplicates_and_known_ids(tmp_path):
    s = chroma(tmp_path)
    batch = [rec(f"unique chunk number {i} about revenue") for i in range(250)] + [rec("unique chunk number 3 about revenue")]
    assert s.add_many(batch) == 250 and s.count() == 250
    # Same content again with new metadata: no duplicates, metadata refreshed, nothing re-embedded.
    assert s.add_many([rec("unique chunk number 3 about revenue", confidence=0.9, verified=True)]) == 1
    assert s.count() == 250
    # (read back with get(): a filtered ANN query can miss a lone match among near-identical vectors)
    meta = s._collection.get(where={"verified": True}, include=["metadatas"])["metadatas"]
    assert len(meta) == 1 and meta[0]["confidence"] == 0.9


def test_chunk_news_tolerates_missing_sentiment_and_date():
    data = {"articles": [{"title": "Acme beats estimates", "source": "x", "published": None, "sentiment_score": None}]}
    out = chunk_news(data, {"ticker": "ACME", "date": "2025-01-01"})
    assert len(out) == 1 and out[0].metadata["date"] == "2025-01-01"


def test_chunk_news_bad_published_date_falls_back_to_base_date():
    data = {"articles": [{"title": "Acme beats estimates", "published": "yesterday"}]}
    out = chunk_news(data, {"ticker": "ACME", "date": "2025-01-01"})
    assert out[0].metadata["date"] == "2025-01-01"


def test_split_paragraphs_is_lossless_and_bounded():
    rnd = random.Random(7)
    words = ["a", "bb", "revenue", "x" * 40, "y" * 700, "end."]
    for _ in range(200):
        text = "\n".join(" ".join(rnd.choice(words) for _ in range(rnd.randint(1, 120))) for _ in range(rnd.randint(1, 4)))
        chunks = split_paragraphs(text)
        assert all(0 < len(c) <= MAX_CHARS for c in chunks)
        assert re.sub(r"\s+", "", "".join(chunks)) == re.sub(r"\s+", "", text)


def _result(eid, tool, text):
    return ToolResult(evidence_id=eid, tool=tool, args={}, data={"t": text}, source_tier=1,
                      retrieved_at="2026-01-01T00:00:00Z", latency_ms=1.0)


@pytest.mark.parametrize("budget", [-5, 0, 1, 50, 400, 5000])
def test_fit_evidence_never_overruns_and_accounts_for_every_id(budget):
    ev = {f"E{i}": _result(f"E{i}", t, "word " * 400) for i, t in
          enumerate(["financial_data_api", "news_sentiment", "sec_filing_search", "web_search"], 1)}
    ctx = fit_evidence(ev, budget, lambda r, n: r.data["t"][:n])
    assert sorted(ctx.included + ctx.dropped) == sorted(ev)
    if ctx.included:
        assert ctx.tokens <= max(budget, 0) + 60  # the "left out" note is the only overhead


def test_format_money_rounding_boundaries_and_negative_zero():
    assert format_money(-0.4) == "$0"
    assert format_money(999_999.6) == "$1.0 million"
    assert format_money(999_960_000) == "$1.00 billion"


def test_news_weighted_score_stays_within_bounds():
    ev = {"E1": ToolResult(evidence_id="E1", tool="news_sentiment", args={"query": "Acme"}, source_tier=4,
                           data={"query": "Acme", "article_count": 0, "articles": [], "overall": {"score": 1.0, "label": "positive"}},
                           retrieved_at="2026-01-01T00:00:00Z", latency_ms=1.0),
          "E2": ToolResult(evidence_id="E2", tool="news_sentiment", args={"query": "Acme"}, source_tier=4,
                           data={"query": "Acme", "article_count": 5, "articles": [], "overall": {"score": 1.0, "label": "positive"}},
                           retrieved_at="2026-01-01T00:00:00Z", latency_ms=1.0)}
    ev["F"] = ToolResult(evidence_id="F", tool="financial_data_api", args={}, source_tier=1,
                         data={"ticker": "ACME", "company_name": "Acme Inc"}, retrieved_at="x", latency_ms=1.0)
    assert _news_for(ev, "ACME")["score"] <= 1.0


def test_facts_store_ignores_missing_values_instead_of_overwriting():
    from memory.facts import FactsStore

    f = FactsStore()
    row = {"period_end": "2024-12-31", "fiscal_year": 2024, "fiscal_period": "FY"}
    f.store_statement({"ticker": "a", "data": [{**row, "values": {"revenue": 5.0}}]})
    f.store_statement({"ticker": "a", "data": [{**row, "values": {"revenue": None, "net_income": None}}]})
    assert f.value("A", "revenue", "2024-12-31") == 5.0
    assert "1 line items" in f.summary("a")
