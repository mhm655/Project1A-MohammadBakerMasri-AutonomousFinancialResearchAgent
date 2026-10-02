"""The agent's memory, in one place: what is written after a run and what is recalled before one.

Layers (architecture spec, section 10.1):

=================  ==============================  ============================================
Layer              Store                           Used for
=================  ==============================  ============================================
Short-term         :mod:`memory.context_manager`   fitting evidence into each prompt's budget
Long-term          Chroma (``vector_store``)       filing, transcript and news chunks; report findings
Structured facts   SQLite (``facts``)              financial statement line items, by exact lookup
Episodic           SQLite (``episodic``)           what worked before: tools, errors, lessons
=================  ==============================  ============================================

**Write policy** (spec section 10.3): source material is stored with a confidence set by its
reliability tier. Report paragraphs flagged by the verifier are stored as unverified, with low
confidence, so the agent never "remembers" its own mistakes as facts. Mock data is never written to the
persistent stores: in mock mode every store is in-memory.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from memory.chunking import (
    TIER_CONFIDENCE,
    MemoryRecord,
    chunk_filing,
    chunk_findings,
    chunk_news,
    chunk_transcript,
)
from memory.episodic import EpisodicMemory, episode_from_result
from memory.facts import FactsStore

#: source_type recorded for each tool's material.
SOURCE_TYPES = {"sec_filing_search": None, "earnings_transcript": "earnings_call", "news_sentiment": "news"}


class MemoryManager:
    """Bundles the vector, facts and episodic stores so the agent can share one set of memories."""
    def __init__(self, vectors: Any, facts: FactsStore, episodes: EpisodicMemory) -> None:
        self.vectors, self.facts, self.episodes = vectors, facts, episodes

    @classmethod
    def for_settings(cls, settings: Any) -> MemoryManager:
        """Persistent stores for live data; throw-away in-memory stores for mock data."""
        if settings.data_mode == "mock":
            from memory.vector_store import DEFAULT_STORE

            return cls(DEFAULT_STORE, FactsStore(None), EpisodicMemory(None))
        from memory.vector_store import get_live_store

        return cls(get_live_store(), FactsStore(settings.paths.data_dir / "facts.sqlite"),
                   EpisodicMemory(settings.paths.episodic_db))

    # ------------------------------------------------------------------ recall (before planning)

    def planning_notes(self, template: str, tickers: list[str]) -> str:
        """Strategy hints and stored facts for the planner."""
        parts = []
        hints = self.episodes.hints(template)
        if hints:
            parts.append(hints)
        for ticker in tickers:
            summary = self.facts.summary(ticker)
            if summary:
                parts.append(summary + " (refreshing is cheap: responses are cached)")
        return "\n".join(parts)

    def known_companies(self, limit: int = 8) -> list[str]:
        """Companies with the most material in long-term memory (for cross-company questions, C7)."""
        return list(self.vectors.companies())[:limit]

    # ------------------------------------------------------------------ write (after a run)

    def ingest_evidence(self, evidence: dict[str, Any], session_id: str, tickers: list[str]) -> dict[str, int]:
        """Chunk and store retrieved source material; statements go to the facts store.

        Tools called with a ticker are filed under it. Tools searched by name (news) are filed under
        the research's company when there is exactly one, and otherwise are not stored, because a
        wrongly attributed chunk is worse than none (cross-company contamination, spec A8.4).
        """
        counts = {"chunks": 0, "facts": 0}
        records: list[MemoryRecord] = []
        for eid, result in evidence.items():
            data = result.data
            if result.tool == "financial_data_api":
                counts["facts"] += self.facts.store_statement(data, evidence_id=eid, session=session_id)
                continue
            if result.tool not in SOURCE_TYPES:
                continue  # calculations, web snippets and memory hits are not source material
            ticker = str(result.args.get("ticker") or data.get("ticker") or "").upper()
            if not ticker and len(tickers) == 1:
                ticker = tickers[0]
            if not ticker:
                continue
            base = {
                "ticker": ticker,
                "source_type": SOURCE_TYPES[result.tool] or data.get("filing_type", "filing"),
                "date": _date_of(data, result),
                "confidence": TIER_CONFIDENCE.get(result.source_tier or 5, 0.3),
                "verified": False,
                "researcher_session": session_id,
                "evidence_id": eid,
                "source_url": data.get("url", ""),
            }
            if result.tool == "sec_filing_search":
                records += chunk_filing(data, base)
            elif result.tool == "earnings_transcript":
                records += chunk_transcript(data, base)
            else:
                records += chunk_news(data, base)
        counts["chunks"] = self.vectors.add_many(records)
        return counts

    def ingest_findings(self, sections: dict[str, str], verification: dict[str, Any], tickers: list[str],
                        session_id: str) -> int:
        """Store the finished report's paragraphs as findings for future sessions."""
        if not sections:
            return 0
        clean = verification.get("unsupported", 0) == 0 and verification.get("supported", 0) > 0
        base = {"ticker": tickers[0] if len(tickers) == 1 else "MULTI", "source_type": "analysis",
                "date": date.today().isoformat(), "confidence": 0.6 if clean else 0.5, "verified": clean,
                "researcher_session": session_id}
        return self.vectors.add_many(chunk_findings(sections, base))

    def record_episode(self, result: Any, tickers: list[str]) -> dict[str, Any]:
        """Build the episode from a run result, save it to episodic memory and return it."""
        episode = {**episode_from_result(result), "tickers": tickers}
        self.episodes.record(episode)
        return episode


def _date_of(data: dict[str, Any], result: Any) -> str:
    for key in ("period_of_report", "filing_date", "call_date"):
        if data.get(key):
            return str(data[key])[:10]
    return result.retrieved_at[:10]
