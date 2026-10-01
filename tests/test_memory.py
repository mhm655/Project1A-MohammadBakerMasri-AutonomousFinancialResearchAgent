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


# ==================================================================================================
# Day 6: chunking, facts, episodic memory, context budget, and the memory manager
# ==================================================================================================

from memory.chunking import (  # noqa: E402
    MAX_CHARS,
    chunk_filing,
    chunk_findings,
    chunk_news,
    chunk_transcript,
    split_paragraphs,
)
from memory.context_manager import count_tokens, fit_evidence  # noqa: E402
from memory.episodic import EpisodicMemory, derive_lessons  # noqa: E402
from memory.facts import FactsStore  # noqa: E402
from memory.manager import MemoryManager  # noqa: E402
from memory.vector_store import InMemoryVectorStore  # noqa: E402

BASE = {"ticker": "MSFT", "source_type": "10-K", "date": "2026-06-30"}


def test_long_text_is_split_at_paragraphs_and_sentences():
    text = "\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(6))
    chunks = split_paragraphs(text, 1500)  # the algorithm test uses the pre-Day-13 chunk size its text was written for
    assert all(len(c) <= 1500 for c in chunks) and len(chunks) >= 3
    assert all(c.startswith("Paragraph") for c in chunks)  # no chunk starts mid-paragraph


def test_filing_risk_factors_become_one_chunk_per_risk_even_with_split_words():
    filing = {"company_name": "MICROSOFT CORP", "filing_type": "10-K", "period_of_report": "2026-06-30",
              "risk_factors": ["We face intense competition across all markets.", "Cyberattacks could harm our reputation."],
              "sections": [
                  {"item": "1", "title": "Business", "text": "Microsoft builds cloud and AI platforms for everyone."},
                  # the filing splits "competition" across elements: "compe tition"
                  {"item": "1A", "title": "Risk Factors", "text": "Intro text about risks in general here.\n"
                   "We face intense compe tition across all markets. Rivals may cut prices and gain share.\n"
                   "Cyberattacks could harm our reputation. " + "Breaches are costly. " * 120},
              ]}
    risks = [r for r in chunk_filing(filing, BASE) if r.metadata.get("kind") == "risk_factor"]
    assert risks[0].content.endswith("gain share.") and "compe tition" in risks[0].content
    assert len(risks) >= 3  # the long cyber risk is split, nothing truncated away
    assert all("Cyberattacks could harm our reputation." in r.content for r in risks[1:])  # heading on every piece
    assert all(r.metadata["section"] == "Item 1A" for r in risks)


def test_transcript_keeps_each_question_with_its_answer():
    transcript = {"ticker": "AAPL", "quarter": "Q4", "fiscal_year": 2025, "source_kind": "transcript", "segments": [
        {"speaker": "Tim Cook", "role": "CEO", "section": "prepared", "text": "Record quarter."},
        {"speaker": "Analyst", "role": "Analyst", "section": "qa", "text": "What about margins?"},
        {"speaker": "CFO", "role": "CFO", "section": "qa", "text": "Margins will be stable."},
        {"speaker": "Analyst", "role": "Analyst", "section": "qa", "text": "And China?"},
    ]}
    records = chunk_transcript(transcript, {"ticker": "AAPL", "source_type": "earnings_call", "date": "2025-10-30"})
    assert [r.metadata["kind"] for r in records] == ["prepared", "qa", "qa"]
    assert "What about margins?" in records[1].content and "Margins will be stable." in records[1].content
    assert "And China?" in records[2].content  # an unanswered question is still kept


def test_news_and_findings_chunks():
    news = chunk_news({"articles": [{"title": "Microsoft beats", "source": "Reuters", "published": "2026-09-20",
                                     "url": "https://x", "sentiment_score": 0.69}]},
                      {"ticker": "MSFT", "source_type": "news", "date": "2026-09-28"})
    assert news[0].metadata["date"] == "2026-09-20" and "+0.69" in news[0].content
    findings = chunk_findings({"executive_summary": "Revenue grew 17.8% [E2].\n\nMargins were 99% [E3]. *(unverified)*",
                               "research_methodology": "not stored"},
                              {"ticker": "MSFT", "source_type": "analysis", "date": "2026-09-28", "verified": True})
    assert [f.metadata["verified"] for f in findings] == [True, False]  # a flagged paragraph is never "verified"


def test_facts_store_upserts_and_queries_by_exact_period():
    facts = FactsStore(None)
    statement = {"ticker": "MSFT", "period": "annual", "unit": "millions", "provider": "sec_xbrl", "data": [
        {"fiscal_year": 2026, "fiscal_period": "FY", "period_end": "2026-06-30",
         "values": {"revenue": 331839.0, "eps_diluted": 17.95}},
        {"fiscal_year": 2025, "fiscal_period": "FY", "period_end": "2025-06-30",
         "values": {"revenue": 281724.0, "eps_diluted": 13.64}},
    ]}
    assert facts.store_statement(statement, evidence_id="E2") == 4
    statement["data"][1]["values"]["revenue"] = 285000.0  # a restatement in a newer filing
    facts.store_statement(statement, evidence_id="E9")
    assert facts.value("MSFT", "revenue", "2025-06-30") == 285000.0
    rows = facts.query("msft", items=["eps_diluted"], limit_periods=1)
    assert [(r["period_end"], r["value"], r["unit"]) for r in rows] == [("2026-06-30", 17.95, "per_share")]
    assert "annual FY2025-FY2026" in facts.summary("MSFT") and facts.tickers() == ["MSFT"]
    assert facts.summary("TSLA") is None


def test_facts_persist_on_disk(tmp_path):
    path = tmp_path / "facts.sqlite"
    FactsStore(path).store_statement({"ticker": "NVDA", "data": [
        {"fiscal_year": 2026, "fiscal_period": "FY", "period_end": "2026-01-25", "values": {"revenue": 1.0}}]})
    assert FactsStore(path).value("NVDA", "revenue", "2026-01-25") == 1.0


def test_episodic_lessons_and_hints(tmp_path):
    lessons = derive_lessons(
        {"financial_data_api": {"steps": 2, "done": 2, "failed": 0, "cited": 2},
         "web_search": {"steps": 1, "done": 1, "failed": 0, "cited": 0}},
        [{"tool": "sec_filing_search", "summary": "sec_filing_search did not finish within the 200 s left"}],
        {"stop_reason": "completed", "uncited": 2})
    text = " ".join(lessons)
    assert "financial_data_api was useful (2 of 2" in text
    assert "web_search returned data 1 time(s) but was never cited" in text
    assert "too slow on this connection" in text and "2 figure(s) were left uncited" in text

    memory = EpisodicMemory(tmp_path / "ep.sqlite")
    for sid, score in (("s-low", 0.4), ("s-high", 0.9)):
        memory.record({"session_id": sid, "query": "q", "template": "company_profile", "plan": [], "errors": [],
                       "metrics": {}, "lessons": [f"lesson from {sid}"], "score": score,
                       "tool_stats": {"financial_data_api": {"steps": 1, "done": 1, "failed": 0, "cited": 1}}})
    hints = EpisodicMemory(tmp_path / "ep.sqlite").hints("company_profile")
    assert hints.index("s-high") < hints.index("s-low")  # the best episode first, and it persisted on disk
    assert EpisodicMemory(tmp_path / "ep.sqlite").hints("risk_assessment") == ""


class _Result:
    def __init__(self, tool, eid, tier=1):
        self.tool, self.evidence_id, self.source_tier = tool, eid, tier


def test_context_budget_shrinks_then_drops_the_least_important_evidence():
    evidence = {"E1": _Result("financial_data_api", "E1"), "E2": _Result("sec_filing_search", "E2"),
                "E3": _Result("news_sentiment", "E3", 4), "E4": _Result("web_search", "E4", 4)}

    def digest(result, limit):
        return f"[{result.evidence_id}] {result.tool} " + ("data " * 2000)[: limit - 30]

    roomy = fit_evidence(evidence, 5000, digest)
    assert roomy.dropped == [] and roomy.tokens <= 5000
    shrunk = fit_evidence(evidence, 450, digest)  # shrinking every digest is enough here
    assert shrunk.dropped == [] and shrunk.tokens <= 450
    tight = fit_evidence(evidence, 200, digest)  # too small even at minimum size: items must go
    assert tight.tokens <= 200 + 60  # the note about dropped items is the only overflow
    assert tight.included == ["E1", "E3"]  # the financial statement and the news survive
    assert set(tight.dropped) == {"E2", "E4"}  # the long filing and the web snippet go first
    assert "do not cite" in tight.text


def test_count_tokens_is_close_to_reality():
    assert 10 <= count_tokens("Microsoft revenue was $245,122 million in FY2024 [E3].") <= 20


def test_mock_mode_memory_is_never_persistent(tmp_path):
    from config.settings import load_settings

    manager = MemoryManager.for_settings(load_settings(env={"ARA_MODE": "mock", "ARA_DATA_DIR": str(tmp_path)}))
    assert isinstance(manager.vectors, InMemoryVectorStore)
    assert not any(tmp_path.iterdir())  # nothing was written to disk


def _tesla_evidence():
    from tools import build_default_registry

    registry = build_default_registry("mock")
    return {r.evidence_id: r for r in (
        registry.dispatch("sec_filing_search", {"ticker": "TSLA", "filing_type": "10-K"}),
        registry.dispatch("financial_data_api", {"ticker": "TSLA", "statement_type": "income_statement"}),
        registry.dispatch("news_sentiment", {"query": "Tesla", "num_articles": 3}),
        registry.dispatch("calculation_engine", {"calculation_type": "ratio", "inputs": {"numerator": 1, "denominator": 2}}),
    )}


def test_manager_ingests_sources_and_facts_and_files_news_under_the_company():
    store = InMemoryVectorStore()
    manager = MemoryManager(store, FactsStore(None), EpisodicMemory(None))
    counts = manager.ingest_evidence(_tesla_evidence(), "s1", ["TSLA"])
    assert counts["facts"] == 15 and counts["chunks"] > 5
    assert set(store.companies()) == {"TSLA"}  # news searched as "Tesla" is filed under TSLA, not "TESLA"
    risk = store.search("battery cell suppliers", filter={"ticker": "TSLA", "source_type": "10-K"})["results"][0]
    assert risk["metadata"]["confidence"] == 0.95 and risk["metadata"]["evidence_id"]
    assert "TSLA financial statements in memory" in manager.planning_notes("risk_assessment", ["TSLA"])


def test_news_is_not_stored_when_the_company_is_ambiguous():
    store = InMemoryVectorStore()
    MemoryManager(store, FactsStore(None), EpisodicMemory(None)).ingest_evidence(_tesla_evidence(), "s1", ["TSLA", "F"])
    assert store.search("Tesla", filter={"source_type": "news"})["results"] == []  # better none than misattributed


def test_storing_the_same_material_twice_does_not_duplicate_it():
    store = InMemoryVectorStore()
    manager = MemoryManager(store, FactsStore(None), EpisodicMemory(None))
    evidence = _tesla_evidence()
    manager.ingest_evidence(evidence, "s1", ["TSLA"])
    size = len(store.records)
    manager.ingest_evidence(evidence, "s2", ["TSLA"])
    assert len(store.records) == size


# ==================================================================================================
# Regression tests: chunk boundaries, metadata and store edge cases
# ==================================================================================================


def test_split_paragraphs_never_truncates_a_sentence_longer_than_a_chunk():
    text = " ".join(f"w{i}" for i in range(600))  # ~2,900 characters and no sentence punctuation
    chunks = split_paragraphs(text)
    assert all(len(c) <= MAX_CHARS for c in chunks)
    assert " ".join(chunks).split() == text.split()  # every word survives, in order


def test_split_paragraphs_keeps_a_short_tail_of_a_long_text():
    text = "Revenue rose sharply. " * 68 + "\nYes, indeed."  # the tail does not fit in the first chunk
    chunks = split_paragraphs(text, 1500)
    assert chunks[-1] == "Yes, indeed."


def test_risk_body_starts_right_after_a_heading_whose_whitespace_differs_from_the_text():
    heading = "Cyberattacks could harm our reputation and our busi ness res ults badly."  # split words
    text = "Cyberattacks could harm our reputation and our business results badly. " + "Breaches are costly. " * 100
    filing = {"company_name": "MICROSOFT CORP", "filing_type": "10-K", "period_of_report": "2026-06-30",
              "risk_factors": [heading], "sections": [{"item": "1A", "title": "Risk Factors", "text": text}]}
    risks = [r for r in chunk_filing(filing, BASE) if r.metadata.get("kind") == "risk_factor"]
    assert f"{heading} Breaches are costly." in risks[0].content  # not "reaches are costly."


def test_filing_text_before_the_first_found_risk_heading_is_kept():
    filing = {"company_name": "MICROSOFT CORP", "filing_type": "10-K", "period_of_report": "2026-06-30",
              "risk_factors": ["Cyberattacks could harm our reputation."],
              "sections": [{"item": "1A", "title": "Risk Factors", "text":
                            "Our AI investments may not pay off and could reduce margins.\n"
                            "Cyberattacks could harm our reputation. Breaches are costly."}]}
    contents = " ".join(r.content for r in chunk_filing(filing, BASE))
    assert "Our AI investments may not pay off" in contents  # a risk whose heading was not matched
    risks = [r for r in chunk_filing(filing, BASE) if r.metadata.get("kind") == "risk_factor"]
    assert len(risks) == 1 and risks[0].content.endswith("Breaches are costly.")


def test_transcript_keeps_every_answer_with_its_question():
    transcript = {"ticker": "AAPL", "quarter": "Q4", "fiscal_year": 2025, "source_kind": "transcript", "segments": [
        {"speaker": "Analyst", "role": "Analyst", "section": "qa", "text": "What about margins?"},
        {"speaker": "Tim Cook", "role": "CEO", "section": "qa", "text": "Margins will be stable."},
        {"speaker": "Kevan Parekh", "role": "CFO", "section": "qa", "text": "Services mix helps too."},
    ]}
    records = chunk_transcript(transcript, {"ticker": "AAPL", "source_type": "earnings_call", "date": "2025-10-30"})
    assert [r.metadata["kind"] for r in records] == ["qa"]
    assert "What about margins?" in records[0].content and "Services mix helps too." in records[0].content


def test_long_transcript_turns_are_split_not_truncated():
    long_text = "Our installed base reached a new record. " * 60 + "FINAL WORDS."
    transcript = {"ticker": "AAPL", "quarter": "Q4", "fiscal_year": 2025, "source_kind": "transcript", "segments": [
        {"speaker": "Tim Cook", "role": "CEO", "section": "prepared", "text": long_text},
        {"speaker": "Analyst", "role": "Analyst", "section": "qa", "text": "How is China?"},
        {"speaker": "Tim Cook", "role": "CEO", "section": "qa", "text": long_text},
    ]}
    records = chunk_transcript(transcript, {"ticker": "AAPL", "source_type": "earnings_call", "date": "2025-10-30"})
    prepared = [r for r in records if r.metadata["kind"] == "prepared"]
    qa = [r for r in records if r.metadata["kind"] == "qa"]
    assert len(prepared) >= 2 and "FINAL WORDS." in prepared[-1].content
    assert len(qa) >= 2 and "FINAL WORDS." in qa[-1].content
    assert all("How is China?" in r.content for r in qa)  # every piece keeps its question
    assert all(len(r.content) <= MAX_CHARS for r in records)


def test_every_piece_of_a_long_flagged_paragraph_is_unverified():
    paragraph = " ".join(f"Sentence {i} states a fact about revenue." for i in range(60))
    paragraph += " Margin was 99% [E3]. *(unverified)*"
    findings = chunk_findings({"risk_analysis": paragraph},
                              {"ticker": "MSFT", "source_type": "analysis", "date": "2026-09-28", "verified": True})
    assert len(findings) >= 2
    assert all(f.metadata["verified"] is False and f.metadata["confidence"] == 0.4 for f in findings)


def test_chroma_store_never_reuses_an_id(tmp_path):
    s = make_store(tmp_path / "ids")
    first = s.store("Tesla battery supply risk grows.", {"ticker": "TSLA", "date": "2025-01-29", "source_type": "10-K"})
    second = s.store("Tesla margin compression continues.",
                     {"ticker": "TSLA", "date": "2025-01-29", "source_type": "10-k"})  # same id prefix
    assert first["id"] != second["id"] and s.count() == 2


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_returns_no_results(tmp_path, top_k):
    chroma = make_store(tmp_path / "topk")
    memory = InMemoryVectorStore()
    for s in (chroma, memory):
        s.store("Tesla battery supply risk grows.", {"ticker": "TSLA", "date": "2025-01-29", "source_type": "10-K"})
        s.store("Tesla battery margins shrink.", {"ticker": "TSLA", "date": "2025-01-30", "source_type": "10-K"})
        assert s.search("tesla battery", top_k=top_k)["results"] == []


def test_in_memory_store_normalizes_timestamps_to_dates():
    s = InMemoryVectorStore()
    s.store("Tesla battery supply risk.", {"ticker": "TSLA", "date": "2025-01-29T15:00:00Z", "source_type": "news"})
    hits = s.search("tesla battery", filter={"date_from": "2025-01-29", "date_to": "2025-01-29"})["results"]
    assert len(hits) == 1 and hits[0]["metadata"]["date"] == "2025-01-29"  # same as the Chroma store


def test_facts_period_limit_counts_only_periods_that_have_the_requested_items():
    facts = FactsStore(None)
    facts.store_statement({"ticker": "MSFT", "data": [
        {"fiscal_year": 2026, "period_end": "2026-06-30", "values": {"revenue": 1.0}},
        {"fiscal_year": 2025, "period_end": "2025-06-30", "values": {"revenue": 2.0, "eps_diluted": 3.0}}]})
    rows = facts.query("MSFT", items=["eps_diluted"], limit_periods=1)
    assert [(r["period_end"], r["value"]) for r in rows] == [("2025-06-30", 3.0)]


def test_fitted_evidence_stays_within_the_budget_including_separators():
    evidence = {f"E{i}": _Result(["financial_data_api", "news_sentiment", "sec_filing_search", "web_search"][i % 4],
                                 f"E{i}") for i in range(1, 13)}

    def digest(result, limit):
        return (f"[{result.evidence_id}] " + "Revenue rose 12% to $4.2bn. " * 200)[:limit]

    fitted = fit_evidence(evidence, 1043, digest)
    body = fitted.text.split("\n\n(Also retrieved")[0]  # the dropped-items note is the only allowed overflow
    assert count_tokens(body) <= 1043
