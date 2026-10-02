"""Long-term memory: the vector_db_search and vector_db_store tools.

Two interchangeable stores implement the same contract (record schema from spec A3.3, metadata
filters, similarity scores):

* :class:`ChromaVectorStore` — the real store (Day 3): a persistent local Chroma collection with
  cosine similarity. Used by the live tool handlers.
* :class:`InMemoryVectorStore` — a dependency-free stand-in with keyword-overlap scoring, used by the
  mock tool handlers and tests.

Chunking, the embedding pipeline for whole documents, and memory-first retrieval arrive on Day 6.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from tools.base import ToolValidationError

_WORD = re.compile(r"[a-z0-9]+")

#: Bump when the record schema changes; stored in the collection metadata.
SCHEMA_VERSION = 1


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def record_id_prefix(metadata: dict[str, Any]) -> str:
    """``tsla-10-k-2025-01-29``: the readable part of a record id (spec A3.3 example style)."""
    source = re.sub(r"[^a-z0-9]+", "-", str(metadata["source_type"]).lower()).strip("-")
    return f"{str(metadata['ticker']).lower()}-{source}-{str(metadata['date'])[:10]}"


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        raise ToolValidationError(f"metadata date must be ISO format (YYYY-MM-DD), got {value!r}") from None


@dataclass
class InMemoryVectorStore:
    """Stores records with the A3.3 schema; similarity is word-overlap (Jaccard), not embeddings."""

    records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def store(self, content: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """Store a document and return its generated id, ``stored`` flag and the record total."""
        ticker = metadata["ticker"].upper()
        stored_on = _parse_date(metadata["date"]).isoformat()  # "2025-01-29T15:00Z" is stored as the day, as in Chroma
        prefix = record_id_prefix(metadata)
        seq = sum(1 for rid in self.records if rid.startswith(prefix + "-")) + 1
        record_id = f"{prefix}-{seq:03d}"
        self.records[record_id] = {
            "id": record_id,
            "content": content,
            "metadata": {
                "confidence": 0.5,
                "verified": False,
                **metadata,
                "ticker": ticker,
                "date": stored_on,
            },
        }
        return {"id": record_id, "stored": True, "total_records": len(self.records)}

    def search(self, query: str, top_k: int = 5, filter: dict[str, Any] | None = None) -> dict[str, Any]:
        """Rank stored records by token overlap with ``query``; ``filter`` narrows by metadata. Raises ToolValidationError for a query with no words."""
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
        return {"query": query, "results": scored[:max(0, top_k)], "searched_records": len(self.records)}

    def add_many(self, records: list[Any]) -> int:
        """Store many chunks at once; content-hash ids make re-storing a chunk an update."""
        written = set()
        for record in records:
            meta = {"confidence": 0.5, "verified": False, **record.metadata,
                    "ticker": str(record.metadata["ticker"]).upper()}
            meta["date"] = _parse_date(meta["date"]).isoformat()
            rid = content_id(record.content, meta)
            self.records[rid] = {"id": rid, "content": record.content, "metadata": meta}
            written.add(rid)
        return len(written)

    def companies(self) -> dict[str, int]:
        """Record count per ticker, largest first."""
        counts: dict[str, int] = {}
        for record in self.records.values():
            counts[record["metadata"]["ticker"]] = counts.get(record["metadata"]["ticker"], 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def clear(self) -> None:
        """Remove every stored record."""
        self.records.clear()


def _passes(meta: dict[str, Any], flt: dict[str, Any]) -> bool:
    if "ticker" in flt and meta.get("ticker") != str(flt["ticker"]).upper():
        return False
    if "source_type" in flt and meta.get("source_type") != flt["source_type"]:
        return False
    if "verified" in flt and meta.get("verified") != flt["verified"]:
        return False
    # Compare parsed days, as Chroma does: "2025-01-29T15:00Z" as a bound must equal the day 2025-01-29.
    if "date_from" in flt and meta.get("date", "") < _parse_date(flt["date_from"]).isoformat():
        return False
    if "date_to" in flt and meta.get("date", "") > _parse_date(flt["date_to"]).isoformat():
        return False
    if "min_confidence" in flt and meta.get("confidence", 0) < flt["min_confidence"]:
        return False
    return True


# --------------------------------------------------------------------------------------------------
# Chroma
# --------------------------------------------------------------------------------------------------


def _hashing_embedding_class() -> type:
    """Build (once) a Chroma-registered embedding function that needs no model download."""
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
    from chromadb.utils.embedding_functions import register_embedding_function

    @register_embedding_function
    class HashingEmbeddingFunction(EmbeddingFunction[Documents]):
        """Feature-hashed bag of words, L2-normalised. Offline and deterministic, but only lexical:
        use it for tests or when the ONNX model cannot be downloaded, not for real research."""

        def __init__(self, dimensions: int = 256) -> None:
            self.dimensions = dimensions

        def __call__(self, input: Documents) -> Embeddings:
            vectors = []
            for text in input:
                vec = [0.0] * self.dimensions
                for token in _WORD.findall(text.lower()):
                    digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                    vec[int.from_bytes(digest, "big") % self.dimensions] += 1.0
                norm = math.sqrt(sum(v * v for v in vec)) or 1.0
                vectors.append([v / norm for v in vec])
            return vectors

        @staticmethod
        def name() -> str:
            return "ara_hashing"

        def get_config(self) -> dict[str, Any]:
            return {"dimensions": self.dimensions}

        @staticmethod
        def build_from_config(config: dict[str, Any]) -> HashingEmbeddingFunction:
            return HashingEmbeddingFunction(config.get("dimensions", 256))

    return HashingEmbeddingFunction


_HASHING_CLASS: type | None = None


def hashing_embedding_function(dimensions: int = 256) -> Any:
    """A deterministic offline embedding function that hashes words into ``dimensions`` buckets."""
    global _HASHING_CLASS
    if _HASHING_CLASS is None:
        _HASHING_CLASS = _hashing_embedding_class()
    return _HASHING_CLASS(dimensions)


def make_embedding_function(model: str) -> Any:
    """``"default"`` = Chroma's ONNX all-MiniLM-L6-v2 (downloaded once, ~80 MB); ``"hashing"`` = offline."""
    if model == "hashing":
        return hashing_embedding_function()
    if model == "default":
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        return DefaultEmbeddingFunction()
    raise ValueError(f"Unknown embedding model {model!r}; use 'default' or 'hashing'")


class ChromaVectorStore:
    """Persistent long-term memory in a local Chroma collection (cosine similarity).

    One collection per embedding model (``<prefix>__<model>``), so vectors of different dimensions
    are never mixed (architecture spec, section 10.3). Metadata follows spec A3.3; dates are also
    stored as an integer ``date_ord`` because Chroma range filters only work on numbers.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        collection_prefix: str = "ara_findings",
        embedding_model: str = "default",
        embedding_function: Any = None,
        client: Any = None,
    ) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        if client is None:
            chroma_settings = ChromaSettings(anonymized_telemetry=False)
            client = (chromadb.PersistentClient(path=str(path), settings=chroma_settings) if path
                      else chromadb.EphemeralClient(settings=chroma_settings))
        self._client = client
        self.embedding_model = embedding_model
        self.collection_name = f"{collection_prefix}__{re.sub(r'[^a-zA-Z0-9_-]', '_', embedding_model)}"
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=embedding_function or make_embedding_function(embedding_model),
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"schema_version": SCHEMA_VERSION, "embedding_model": embedding_model},
        )
        self._lock = threading.Lock()  # id allocation must be atomic

    def count(self) -> int:
        """Number of stored records."""
        return self._collection.count()

    def store(self, content: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """Store a document under a new readable id and return its id, ``stored`` flag and the record total."""
        meta = _normalize_meta(metadata)
        prefix = record_id_prefix(meta)
        with self._lock:
            # Ids are counted by their readable prefix, not by exact metadata: "10-K" and "10-k" share
            # the prefix, and Chroma silently ignores an ``add`` whose id already exists.
            taken = set(self._collection.get(where={"date": meta["date"]}, include=[])["ids"])
            seq = sum(1 for rid in taken if rid.startswith(prefix + "-")) + 1
            while f"{prefix}-{seq:03d}" in taken:
                seq += 1
            record_id = f"{prefix}-{seq:03d}"
            self._collection.add(ids=[record_id], documents=[content], metadatas=[meta])
        return {"id": record_id, "stored": True, "total_records": self.count()}

    def search(self, query: str, top_k: int = 5, filter: dict[str, Any] | None = None) -> dict[str, Any]:
        """Semantic search over stored records; ``filter`` narrows by metadata. Raises ToolValidationError for a query with no words."""
        if not _tokens(query):
            raise ToolValidationError("vector_db_search: query has no searchable words", tool="vector_db_search")
        total = self.count()
        if total == 0 or top_k < 1:  # Chroma rejects n_results < 1
            return {"query": query, "results": [], "searched_records": total}

        kwargs: dict[str, Any] = {"query_texts": [query], "n_results": min(top_k, total)}
        where = _chroma_where(filter or {})
        if where:
            kwargs["where"] = where
        found = self._collection.query(**kwargs, include=["documents", "metadatas", "distances"])

        results = []
        for rid, doc, meta, dist in zip(found["ids"][0], found["documents"][0], found["metadatas"][0], found["distances"][0], strict=True):
            clean_meta = {k: v for k, v in meta.items() if k != "date_ord"}
            results.append({"id": rid, "content": doc, "score": round(1.0 - float(dist), 4), "metadata": clean_meta})
        return {"query": query, "results": results, "searched_records": total}

    def add_many(self, records: list[Any]) -> int:
        """Store many chunks at once. Ids are content hashes, so storing the same chunk again updates
        it instead of duplicating it. Returns the number of records written."""
        if not records:
            return 0
        ids, docs, metas = [], [], []
        for record in records:
            meta = _normalize_meta(record.metadata)
            rid = content_id(record.content, meta)
            if rid in ids:
                continue
            ids.append(rid)
            docs.append(record.content)
            metas.append(meta)
        with self._lock:
            for start in range(0, len(ids), 200):  # keep batches small for the embedding model
                batch = range(start, min(start + 200, len(ids)))
                # Ids hash the content, so a stored id already has the same text and vector. Re-embedding it
                # cost minutes per run once filings were cut into 500-character chunks (Day 14, C4: 2,174
                # chunks); update only its metadata, and embed the new chunks.
                known = set(self._collection.get(ids=[ids[i] for i in batch], include=[])["ids"])
                new = [i for i in batch if ids[i] not in known]
                old = [i for i in batch if ids[i] in known]
                if new:
                    self._collection.upsert(ids=[ids[i] for i in new], documents=[docs[i] for i in new],
                                            metadatas=[metas[i] for i in new])
                if old:
                    self._collection.update(ids=[ids[i] for i in old], metadatas=[metas[i] for i in old])
        return len(ids)

    def companies(self) -> dict[str, int]:
        """``ticker -> number of records``: which companies memory knows about (Challenge 7)."""
        counts: dict[str, int] = {}
        for meta in self._collection.get(include=["metadatas"])["metadatas"]:
            counts[meta["ticker"]] = counts.get(meta["ticker"], 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def clear(self) -> None:
        """Remove every stored record."""
        ids = self._collection.get(include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)


#: Metadata keys stored as plain strings when present (Chroma accepts only flat scalar values).
_OPTIONAL_KEYS = ("source_url", "evidence_id", "section", "kind")


def _normalize_meta(metadata: dict[str, Any]) -> dict[str, Any]:
    stored_on = _parse_date(metadata["date"])
    meta: dict[str, Any] = {
        "ticker": str(metadata["ticker"]).upper(),
        "source_type": str(metadata["source_type"]),
        "date": stored_on.isoformat(),
        "date_ord": stored_on.toordinal(),
        "confidence": float(metadata.get("confidence", 0.5)),
        "verified": bool(metadata.get("verified", False)),
        "researcher_session": str(metadata.get("researcher_session", "")),
        "stored_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    for key in _OPTIONAL_KEYS:
        if metadata.get(key):
            meta[key] = str(metadata[key])
    return meta


def content_id(content: str, meta: dict[str, Any]) -> str:
    """Readable prefix plus a hash of the content, e.g. ``msft-10-k-2026-06-30-3f2a9c1b04``."""
    digest = hashlib.sha1(f"{meta['ticker']}|{meta['source_type']}|{content}".encode()).hexdigest()[:10]
    return f"{record_id_prefix(meta)}-{digest}"


def _chroma_where(flt: dict[str, Any]) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if "ticker" in flt:
        clauses.append({"ticker": str(flt["ticker"]).upper()})
    if "source_type" in flt:
        clauses.append({"source_type": flt["source_type"]})
    if "verified" in flt:
        clauses.append({"verified": bool(flt["verified"])})
    if "min_confidence" in flt:
        clauses.append({"confidence": {"$gte": float(flt["min_confidence"])}})
    if "date_from" in flt:
        clauses.append({"date_ord": {"$gte": _parse_date(flt["date_from"]).toordinal()}})
    if "date_to" in flt:
        clauses.append({"date_ord": {"$lte": _parse_date(flt["date_to"]).toordinal()}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# --------------------------------------------------------------------------------------------------
# Tool handlers
# --------------------------------------------------------------------------------------------------

#: Process-wide store used by the mock tool handlers.
DEFAULT_STORE = InMemoryVectorStore()

_live_store: ChromaVectorStore | None = None
_live_lock = threading.Lock()


def get_live_store() -> ChromaVectorStore:
    """The persistent Chroma store configured in settings, created on first use."""
    global _live_store
    with _live_lock:
        if _live_store is None:
            from config.settings import load_settings

            s = load_settings()
            _live_store = ChromaVectorStore(
                s.paths.chroma_dir,
                collection_prefix=s.memory.collection_prefix,
                embedding_model=s.memory.embedding_model,
            )
        return _live_store


def set_live_store(store: ChromaVectorStore | None) -> None:
    """Replace the live store (tests use an in-memory Chroma client)."""
    global _live_store
    with _live_lock:
        _live_store = store


def _with_ticker(filter: dict[str, Any] | None, ticker: str | None) -> dict[str, Any] | None:
    """Merge the top-level ``ticker`` shortcut into the filter (models reach for it naturally)."""
    return {**(filter or {}), "ticker": ticker} if ticker else filter


def vector_db_search(query: str, top_k: int = 5, filter: dict[str, Any] | None = None,
                     ticker: str | None = None) -> dict[str, Any]:
    """Search the shared in-memory store (mock mode); ``ticker`` is a shortcut for a ticker filter."""
    return DEFAULT_STORE.search(query, top_k, _with_ticker(filter, ticker))


def vector_db_store(content: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Store a document in the shared in-memory store (mock mode)."""
    return DEFAULT_STORE.store(content, metadata)


def live_vector_db_search(query: str, top_k: int = 5, filter: dict[str, Any] | None = None,
                          ticker: str | None = None) -> dict[str, Any]:
    """Search the persistent Chroma store (live mode); ``ticker`` is a shortcut for a ticker filter."""
    return get_live_store().search(query, top_k, _with_ticker(filter, ticker))


def live_vector_db_store(content: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Store a document in the persistent Chroma store (live mode)."""
    return get_live_store().store(content, metadata)
