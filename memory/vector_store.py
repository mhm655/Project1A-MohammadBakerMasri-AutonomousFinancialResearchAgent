"""Long-term memory: the vector_db_search and vector_db_store tools.

Day 2: an in-memory stand-in with keyword-overlap scoring, so the tool contract (record schema from
spec A3.3, filters, similarity scores) can be exercised before a real vector database exists.
Day 6: replaced by a persistent Chroma collection with real embeddings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from tools.base import ToolValidationError

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@dataclass
class InMemoryVectorStore:
    """Stores records with the A3.3 schema; similarity is word-overlap (Jaccard), not embeddings."""

    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def store(self, content: str, metadata: dict[str, Any]) -> dict[str, Any]:
        ticker = metadata["ticker"].upper()
        prefix = f"{ticker.lower()}-{metadata['source_type'].lower()}-{metadata['date'][:10]}"
        seq = sum(1 for rid in self.records if rid.startswith(prefix)) + 1
        record_id = f"{prefix}-{seq:03d}"
        self.records[record_id] = {
            "id": record_id,
            "content": content,
            "metadata": {
                "confidence": 0.5,
                "verified": False,
                **metadata,
                "ticker": ticker,
            },
        }
        return {"id": record_id, "stored": True, "total_records": len(self.records)}

    def search(self, query: str, top_k: int = 5, filter: dict[str, Any] | None = None) -> dict[str, Any]:
        filter = filter or {}
        query_tokens = _tokens(query)
        if not query_tokens:
            raise ToolValidationError("vector_db_search: query has no searchable words", tool="vector_db_search")

        scored = []
        for record in self.records.values():
            if not _passes(record["metadata"], filter):
                continue
            doc_tokens = _tokens(record["content"])
            union = query_tokens | doc_tokens
            score = len(query_tokens & doc_tokens) / len(union) if union else 0.0
            if score > 0:
                scored.append({**record, "score": round(score, 4)})
        scored.sort(key=lambda r: r["score"], reverse=True)
        return {"query": query, "results": scored[:top_k], "searched_records": len(self.records)}

    def clear(self) -> None:
        self.records.clear()


def _passes(meta: dict[str, Any], flt: dict[str, Any]) -> bool:
    if "ticker" in flt and meta.get("ticker") != str(flt["ticker"]).upper():
        return False
    if "source_type" in flt and meta.get("source_type") != flt["source_type"]:
        return False
    if "verified" in flt and meta.get("verified") != flt["verified"]:
        return False
    if "date_from" in flt and meta.get("date", "") < flt["date_from"]:
        return False
    if "date_to" in flt and meta.get("date", "") > flt["date_to"]:
        return False
    if "min_confidence" in flt and meta.get("confidence", 0) < flt["min_confidence"]:
        return False
    return True


#: Process-wide store used by the tool handlers (Day 6 swaps in Chroma).
DEFAULT_STORE = InMemoryVectorStore()


def vector_db_search(query: str, top_k: int = 5, filter: dict[str, Any] | None = None) -> dict[str, Any]:
    return DEFAULT_STORE.search(query, top_k, filter)


def vector_db_store(content: str, metadata: dict[str, Any]) -> dict[str, Any]:
    return DEFAULT_STORE.store(content, metadata)
