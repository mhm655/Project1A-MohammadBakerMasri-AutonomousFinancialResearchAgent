"""Semantic chunking of research material before it goes into long-term memory (spec A3.3).

Fixed-size chunking cuts risk factors and Q&A exchanges in half, so each source type is split along its
own structure:

* **SEC filings**: one chunk per item. Within Risk Factors, one chunk per individual risk (split at
  the risk headings the filing parser extracted); long items are split at paragraph boundaries.
* **Earnings calls**: one chunk per prepared-remarks turn; each analyst question is kept together
  with the management answers that follow it.
* **News**: one chunk per article (headline, source, date, sentiment).
* **Report findings**: one chunk per paragraph of a finished report section.
* **Financial statements** are *not* chunked or embedded. They go to the structured facts store
  (:mod:`memory.facts`), because numbers are retrieved by exact metadata, not by similarity.

Every chunk carries the metadata of spec A3.3 (ticker, source type, date, confidence, verified) plus
its provenance (evidence id, source URL, section).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

MAX_CHARS = 500  # about 125 tokens; Day 13 probe: hit@3 35% -> 52% versus 1500 (results/runs_day13/memory_probe.md)
MIN_CHARS = 20  # short findings such as "Revenue grew 17.8% [E2]." still carry information

#: Confidence by the source reliability tier (architecture spec, section 12.3).
TIER_CONFIDENCE = {1: 0.95, 2: 0.85, 3: 0.70, 4: 0.60, 5: 0.30}


@dataclass
class MemoryRecord:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def split_paragraphs(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Group paragraphs into chunks of at most ``max_chars``; oversize paragraphs are split at sentences,
    and oversize sentences between words, so no text is cut away."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        pieces = [para] if len(para) <= max_chars else [
            part for sentence in re.split(r"(?<=[.!?])\s+", para) for part in _wrap_words(sentence, max_chars)]
        for piece in pieces:
            if current and len(current) + len(piece) + 1 > max_chars:
                chunks.append(current)
                current = ""
            current = f"{current}\n{piece}".strip() if current else piece
    if current:
        chunks.append(current)
    # MIN_CHARS drops texts that are only noise ("None."), not the short tail of a longer text.
    return chunks if sum(len(c) for c in chunks) >= MIN_CHARS else []


def _wrap_words(sentence: str, max_chars: int) -> list[str]:
    """``sentence`` in pieces of at most ``max_chars``, broken between words (inside a word only when
    the word itself is longer than a chunk)."""
    if len(sentence) <= max_chars:
        return [sentence]
    pieces: list[str] = []
    line = ""
    for word in sentence.split():
        while len(word) > max_chars:
            if line:
                pieces.append(line)
                line = ""
            pieces.append(word[:max_chars])
            word = word[max_chars:]
        if line and len(line) + 1 + len(word) > max_chars:
            pieces.append(line)
            line = ""
        line = f"{line} {word}" if line else word
    if line:
        pieces.append(line)
    return pieces


def chunk_filing(data: dict[str, Any], base: dict[str, Any]) -> list[MemoryRecord]:
    """Chunks of a ``sec_filing_search`` result."""
    header = f"{data.get('company_name', data.get('ticker'))} Form {data.get('filing_type')} " \
             f"(period ended {data.get('period_of_report')})"
    records: list[MemoryRecord] = []
    for section in data.get("sections", []):
        label = f"{header}, Item {section['item']} {section['title']}"
        meta = {**base, "section": f"Item {section['item']}"}
        if section["item"] == "1A" and data.get("risk_factors"):
            preamble, risks = _split_risks(section["text"], data["risk_factors"])
            for chunk in preamble:
                records.append(MemoryRecord(f"{label}: {chunk}", meta))
            for risk_text in risks:
                records.append(MemoryRecord(f"{label}: {risk_text}", {**meta, "kind": "risk_factor"}))
        else:
            for chunk in split_paragraphs(section["text"]):
                records.append(MemoryRecord(f"{label}: {chunk}", meta))
    return records


def _split_risks(section_text: str, headings: list[str]) -> tuple[list[str], list[str]]:
    """Chunks per risk factor: each heading with the text up to the next heading.

    Filings often split words across HTML elements ("RIS K FACTORS"), so headings are located while
    ignoring whitespace. A long risk becomes several chunks, each prefixed with its heading so it still
    stands alone; nothing is truncated away. Returns ``(preamble, risks)``: the text before the first
    heading found (an introduction, or a risk whose heading was not matched) is kept as plain chunks.
    """
    flat = " ".join(section_text.split())
    squeezed_chars, index = [], []  # whitespace-free text, and each character's offset in ``flat``
    for i, ch in enumerate(flat):
        if not ch.isspace():
            squeezed_chars.append(ch)
            index.append(i)
    squeezed = "".join(squeezed_chars)
    positions = []
    for heading in headings:
        needle = "".join(heading.split())[:60]
        pos = squeezed.find(needle)
        if pos != -1:
            positions.append((index[pos], pos, heading))
    positions.sort()
    if not positions:
        return [], split_paragraphs(section_text)
    preamble = split_paragraphs(flat[:positions[0][0]])
    chunks = []
    for i, (pos, squeezed_pos, heading) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(flat)
        risk = flat[pos:end].strip()
        if len(risk) < MIN_CHARS:
            continue
        if len(risk) <= MAX_CHARS:
            chunks.append(risk)
            continue
        label = " ".join(heading.split())[:200]
        # Where the label ends in ``flat``, found through the whitespace-free text: the heading and the
        # filing may split words differently, so ``len(label)`` is not a reliable offset.
        squeezed_label = "".join(label.split())
        if squeezed_label and squeezed.startswith(squeezed_label, squeezed_pos):
            body = flat[index[squeezed_pos + len(squeezed_label) - 1] + 1:end].strip()
        else:
            body = risk
        for j, piece in enumerate(split_paragraphs(body, MAX_CHARS - len(label) - 20)):
            chunks.append(f"{label} {'(continued) ' if j else ''}{piece}")
    return preamble, chunks


def chunk_transcript(data: dict[str, Any], base: dict[str, Any]) -> list[MemoryRecord]:
    """Chunks of an ``earnings_transcript`` result: prepared turns, and Q&A question-answer pairs."""
    header = f"{data.get('ticker')} {data.get('quarter')} FY{data.get('fiscal_year')} " \
             f"{'earnings call' if data.get('source_kind') == 'transcript' else 'earnings release'}"
    records: list[MemoryRecord] = []
    question: str | None = None
    answers: list[str] = []  # every management answer to ``question`` (CEO, then CFO, ...)
    for seg in data.get("segments", []):
        turn = f"{seg['speaker']} ({seg.get('role', '')}): {seg['text']}"
        if seg.get("section") == "qa" and seg.get("role", "").lower() == "analyst":
            if question:  # a question with no answer still gets stored
                records += _qa_records(header, question, answers, base)
            question, answers = turn, []
        elif seg.get("section") == "qa" and question:
            answers.append(turn)
        else:
            records += [MemoryRecord(text, {**base, "kind": "prepared"})
                        for text in _prefixed_chunks(f"{header}, prepared remarks: ", turn)]
    if question:
        records += _qa_records(header, question, answers, base)
    return records


def _qa_records(header: str, question: str, answers: list[str], base: dict[str, Any]) -> list[MemoryRecord]:
    """One question with all its answers; a long exchange is split and every piece repeats the question."""
    if not answers:
        return [MemoryRecord(f"{header}, Q&A: {question}", {**base, "kind": "qa"})]
    return [MemoryRecord(text, {**base, "kind": "qa"})
            for text in _prefixed_chunks(f"{header}, Q&A: {question} | ", " | ".join(answers))]


def _prefixed_chunks(prefix: str, text: str) -> list[str]:
    """``prefix + text`` in pieces of at most MAX_CHARS (unless ``prefix`` alone is nearly that long),
    each starting with ``prefix``, so a long turn is split rather than truncated."""
    if len(prefix) + len(text) <= MAX_CHARS:
        return [prefix + text]
    room = max(MAX_CHARS - len(prefix) - len("(continued) "), 200)
    return [f"{prefix}{'(continued) ' if j else ''}{piece}" for j, piece in enumerate(split_paragraphs(text, room))]


def chunk_news(data: dict[str, Any], base: dict[str, Any]) -> list[MemoryRecord]:
    records = []
    for article in data.get("articles", []):
        text = (f"News ({article.get('source') or 'unknown source'}, {article.get('published') or 'undated'}): "
                f"{article['title']} [sentiment {article.get('sentiment_score', 0):+.2f}]")
        meta = {**base, "date": article.get("published") or base["date"], "source_url": article.get("url", "")}
        records.append(MemoryRecord(text, meta))
    return records


def chunk_findings(sections: dict[str, str], base: dict[str, Any]) -> list[MemoryRecord]:
    """Paragraphs of a finished report. Paragraphs flagged by the verifier are stored as unverified."""
    records = []
    for key, body in sections.items():
        if key in ("data_gaps_and_limitations", "research_methodology"):
            continue
        # One chunk per paragraph, never merged: merging a verified paragraph with a flagged one
        # would make the verified one look unverified (or the reverse).
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", body) if len(p.strip()) >= MIN_CHARS]
        for paragraph in paragraphs:
            # Judged on the whole paragraph: when a long flagged paragraph is split, the pieces
            # without the flag still belong to a flagged paragraph.
            unverified = "*(unverified" in paragraph
            meta = {**base, "section": key, "verified": not unverified and base.get("verified", False),
                    "confidence": 0.4 if unverified else base.get("confidence", 0.6)}
            for chunk in split_paragraphs(paragraph) if len(paragraph) > MAX_CHARS else [paragraph]:
                records.append(MemoryRecord(chunk, dict(meta)))
    return records
