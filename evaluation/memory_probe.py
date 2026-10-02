"""Retrieval probe for the long-term memory (spec Day 13 task 3).

    python -m evaluation.memory_probe [--out results/runs_day13/memory_probe.md] [--probes 48]

Measures, on the records the live Chroma store already holds:

* hit@3 and hit@5 of a fixed probe set for the current chunk size (``memory.chunking.MAX_CHARS``) and for
  smaller chunks and chunks with overlap (stored chunks are re-split, so only sizes below the stored one
  can be tried);
* the score of section-targeted queries ("<TICKER> Business Overview") against a keyword relevance label, to
  pick the relevance floor used by the targeted recall in ``agent.core``.

A probe is a short phrase taken from the middle of a stored chunk; the expected answer is "a retrieved chunk
that contains that phrase", so the check works for any chunking. The phrase is lexically close to its chunk,
which makes absolute hit rates optimistic; the comparison between configurations is what matters.
The store is only read. There is no hybrid re-ranker (and so no alpha) in the code, so none is swept.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

from memory.chunking import MAX_CHARS, split_paragraphs

ROOT = Path(__file__).resolve().parent.parent
SEED = 13
PHRASE_WORDS = 9
FLOORS = (0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
#: Keywords that make a retrieved chunk relevant to a section-targeted query (a proxy label).
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "business overview": ("business", "segment", "products", "services", "customers", "operat"),
    "financial summary": ("revenue", "income", "margin", "cash flow", "earnings", "growth"),
    "financial risks": ("risk", "debt", "liquidity", "interest", "currency", "credit"),
    "operational risks": ("risk", "supply", "operat", "manufactur", "cyber", "disruption"),
    "results vs expectations": ("revenue", "eps", "expect", "estimate", "quarter", "results"),
    "segment performance": ("segment", "revenue", "growth", "margin", "cloud", "services"),
    "revenue growth": ("revenue", "growth", "increase", "year"),
    "the contradiction": ("growth", "decline", "revenue", "risk", "sentiment"),
    "themes": ("ai", "cloud", "demand", "growth", "regulat", "competition"),
    "company overview": ("business", "segment", "products", "services", "customers"),
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def load_records(store: Any) -> list[dict[str, Any]]:
    """Read every stored chunk (id, content, metadata) from the vector store's collection."""
    raw = store._collection.get(include=["documents", "metadatas"])
    return [{"id": i, "content": d, "meta": m} for i, d, m in zip(raw["ids"], raw["documents"], raw["metadatas"],
                                                                      strict=True)]


def make_probes(records: list[dict[str, Any]], n: int, seed: int = SEED) -> list[dict[str, str]]:
    """``n`` probes spread over (ticker, source type) groups: a phrase from the middle of a long chunk."""
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in sorted(records, key=lambda r: r["id"]):
        if len(r["content"]) >= 500:
            groups.setdefault((r["meta"]["ticker"], r["meta"]["source_type"]), []).append(r)
    keys = sorted(groups)
    probes: list[dict[str, str]] = []
    i = 0
    # Bounded: a small store has fewer distinct phrases than ``n``, and an unbounded loop never ended.
    while len(probes) < n and keys and i < n * 50:
        key = keys[i % len(keys)]
        i += 1
        words = rng.choice(groups[key])["content"].split()
        start = len(words) // 2 + rng.randint(-5, 5)
        phrase = " ".join(words[max(0, start): start + PHRASE_WORDS])
        if len(phrase.split()) >= 5 and _norm(phrase) not in {_norm(p["phrase"]) for p in probes}:
            probes.append({"ticker": key[0], "phrase": phrase})
    return probes


def rechunk(records: list[dict[str, Any]], max_chars: int, overlap: int = 0) -> list[dict[str, Any]]:
    """Split each stored chunk again. ``max_chars`` at or above the stored size leaves it unchanged."""
    out = []
    for r in records:
        text = r["content"]
        if len(text) <= max_chars:
            pieces = [text]
        elif overlap:
            step = max_chars - overlap
            pieces = [text[i: i + max_chars] for i in range(0, len(text), step) if len(text[i: i + max_chars]) > overlap]
        else:
            pieces = split_paragraphs(text, max_chars)
        for j, piece in enumerate(pieces):
            out.append({"id": f"{r['id']}#{j}", "content": piece, "meta": r["meta"]})
    return out


def build_collection(client: Any, name: str, chunks: list[dict[str, Any]], embedding_function: Any) -> Any:
    """Create an in-memory Chroma collection (cosine space) and add ``chunks`` to it in batches of 200."""
    col = client.create_collection(name=name, embedding_function=embedding_function,
                                   configuration={"hnsw": {"space": "cosine"}})
    for start in range(0, len(chunks), 200):
        batch = chunks[start: start + 200]
        col.add(ids=[c["id"] for c in batch], documents=[c["content"] for c in batch],
                metadatas=[{"ticker": c["meta"]["ticker"], "source_type": c["meta"]["source_type"]} for c in batch])
    return col


def hit_at_k(col: Any, probes: list[dict[str, str]], ks: tuple[int, ...] = (3, 5)) -> dict[str, float]:
    """Share of probes whose phrase appears in the top-k results for its ticker, keyed ``hit@k``."""
    hits = dict.fromkeys(ks, 0)
    for p in probes:
        found = col.query(query_texts=[p["phrase"]], n_results=max(ks), where={"ticker": p["ticker"]},
                          include=["documents"])["documents"][0]
        needle = _norm(p["phrase"])
        for k in ks:
            if any(needle in _norm(d) for d in found[:k]):
                hits[k] += 1
    return {f"hit@{k}": hits[k] / len(probes) if probes else 0.0 for k in ks}


def floor_table(col: Any, tickers: list[str]) -> list[dict[str, Any]]:
    """Share of top-3 hits kept and share of those that are relevant (keyword proxy), per floor."""
    scored: list[tuple[float, bool]] = []
    for ticker in tickers:
        for topic, words in TOPIC_KEYWORDS.items():
            res = col.query(query_texts=[f"{ticker} {topic}"], n_results=3, where={"ticker": ticker},
                            include=["documents", "distances"])
            for doc, dist in zip(res["documents"][0], res["distances"][0], strict=True):
                scored.append((1.0 - float(dist), any(w in doc.lower() for w in words)))
    rows = []
    for floor in FLOORS:
        kept = [rel for score, rel in scored if score >= floor]
        rows.append({"floor": floor, "kept_share": len(kept) / len(scored) if scored else 0.0,
                     "relevant_share": (sum(kept) / len(kept)) if kept else None, "hits": len(kept)})
    return rows


def run(n_probes: int = 24) -> dict[str, Any]:
    """Run the probe against the live store: compare chunking configurations and build the relevance-floor table."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    from memory.vector_store import get_live_store, make_embedding_function

    store = get_live_store()
    records = load_records(store)
    probes = make_probes(records, n_probes)
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True))
    embed = make_embedding_function(store.embedding_model)
    configs = [(f"current ({MAX_CHARS} chars, as stored)", MAX_CHARS, 0), ("800 chars", 800, 0),
               ("500 chars", 500, 0), ("350 chars", 350, 0), ("500 chars, 100 overlap", 500, 100),
               ("800 chars, 150 overlap", 800, 150)]
    results = []
    cols: dict[str, Any] = {}
    for i, (label, size, overlap) in enumerate(configs):
        chunks = rechunk(records, size, overlap)
        col = build_collection(client, f"probe{i}", chunks, embed)
        cols[label] = col
        results.append({"config": label, "chunks": len(chunks), **hit_at_k(col, probes)})
    tickers = sorted({r["meta"]["ticker"] for r in records if r["meta"]["ticker"] != "MULTI"})
    return {"records": len(records), "probes": len(probes), "results": results,
            "floors": floor_table(cols[configs[0][0]], tickers), "tickers": tickers}


def render(res: dict[str, Any]) -> str:
    """Render the probe results as a Markdown report."""
    lines = ["# Memory retrieval probe", "",
             "Generated by `python -m evaluation.memory_probe` (spec Day 13 task 3). "
             f"{res['probes']} probes over {res['records']} stored chunks.", "",
             "## Chunk size", "", "| Configuration | Chunks | hit@3 | hit@5 |", "|---|---|---|---|"]
    for r in res["results"]:
        lines.append(f"| {r['config']} | {r['chunks']:,} | {r['hit@3']:.0%} | {r['hit@5']:.0%} |")
    lines += ["", "## Relevance floor for section-targeted queries", "",
              f"Queries are \"<TICKER> <section topic>\" for {len(res['tickers'])} tickers and "
              f"{len(TOPIC_KEYWORDS)} topics, top 3 hits each. Relevant = the chunk contains a topic keyword.", "",
              "| Floor (1 - cosine distance) | Hits kept | Share of hits kept | Relevant among kept |", "|---|---|---|---|"]
    for f in res["floors"]:
        rel = "n/a" if f["relevant_share"] is None else f"{f['relevant_share']:.0%}"
        lines.append(f"| {f['floor']:.2f} | {f['hits']} | {f['kept_share']:.0%} | {rel} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: run the probe and write the Markdown and JSON results."""
    parser = argparse.ArgumentParser(prog="python -m evaluation.memory_probe", description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(ROOT / "results" / "runs_day13" / "memory_probe.md"))
    parser.add_argument("--probes", type=int, default=48)
    args = parser.parse_args(argv)
    res = run(args.probes)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(res), encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(render(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
