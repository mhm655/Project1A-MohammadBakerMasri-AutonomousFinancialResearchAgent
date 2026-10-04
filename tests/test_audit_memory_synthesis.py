"""Audit regression tests for memory/ and synthesis/. Offline; no network, no LLM."""

from __future__ import annotations

import sys
import threading

from memory.vector_store import InMemoryVectorStore
from synthesis.narrative import _news_for
from tools.base import ToolResult


def test_inmemory_store_is_thread_safe():
    """Tool calls run in parallel threads: concurrent store() must not raise or reuse an id."""
    store = InMemoryVectorStore()
    errors: list[str] = []

    def work(k: int) -> None:
        for i in range(100):
            try:
                store.store(f"doc {k} {i}", {"ticker": "T", "date": "2025-01-01", "source_type": "10-K"})
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old)
    assert not errors
    assert len(store.records) == 800


def _news(eid: str, score: float, count: int) -> ToolResult:
    return ToolResult(evidence_id=eid, tool="news_sentiment", args={"query": "Acme"}, source_tier=4,
                      data={"query": "Acme", "article_count": count, "articles": [],
                            "overall": {"score": score, "label": "positive"}},
                      retrieved_at="2026-01-01T00:00:00Z", latency_ms=1.0)


def test_empty_news_search_does_not_dilute_the_weighted_sentiment():
    fin = ToolResult(evidence_id="F", tool="financial_data_api", args={}, source_tier=1,
                     data={"ticker": "ACME", "company_name": "Acme Inc"}, retrieved_at="x", latency_ms=1.0)
    ev = {"E1": _news("E1", 0.5, 20), "E2": _news("E2", 0.0, 0), "F": fin}
    assert _news_for(ev, "ACME")["score"] == 0.5
