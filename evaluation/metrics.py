"""Automated evaluation metrics for ARA-1 (spec A5.2): 21 metrics in five categories.

Scoring reads a *run bundle* (``evaluation/runner.py``) and nothing else, so metrics can be
recomputed without another research run. Every metric is one small function returning a
``MetricResult``; ``REGISTRY`` lists them in spec order and ``score_run`` applies them.

Design rules:

* Reuse the agent's own helpers (claim splitting, evidence digests, the fact checker) so the
  evaluation and the agent's verifier agree on what a "claim" and a "match" are.
* Only the report *body* is scored for accuracy: the data-gaps, methodology and sources sections
  describe the run and would otherwise be counted as claims.
* ``value=None`` means "not applicable / not measured" (e.g. no judge run); ``passed`` is then None.
* Pass/fail thresholds live in ``_SPEC`` (comparator + number), never parsed from the target text.
"""

from __future__ import annotations

import bisect
import dataclasses
import json
import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.circuit_breaker import TOOL_PROVIDERS
from agent.core import _claims, _sentences, evidence_digest
from agent.query_analyzer import KNOWN_COMPANIES
from tools.base import ToolResult
from tools.fact_checker import Quantity, executive_names, extract_quantities, fact_checker, name_in_sources
from tools.report_gen import TEMPLATES, cited_ids, section_title

# --------------------------------------------------------------------------------------------------
# Public data types
# --------------------------------------------------------------------------------------------------


@dataclass
class MetricResult:
    id: str
    name: str
    category: str
    value: float | int | None
    target: str
    passed: bool | None
    method: str  # automated | llm_judge | hybrid | benchmark
    unit: str  # ratio | count | per_page | years | types | score_1_5 | seconds
    detail: str = ""
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


#: Tools the spec (B2) expects each challenge to use; AB-3 reports coverage of this set.
EXPECTED_TOOLS: dict[str, set[str]] = {
    "C1": {"company_profile", "financial_data_api", "web_search"},
    "C2": {"financial_data_api", "earnings_transcript", "web_search", "news_sentiment"},
    "C3": {"sec_filing_search", "web_search", "news_sentiment", "financial_data_api", "earnings_transcript"},
    "C4": {"financial_data_api", "sec_filing_search", "earnings_transcript", "web_search", "peer_comparison",
           "calculation_engine"},
    "C5": {"sec_filing_search", "financial_data_api", "news_sentiment", "web_search", "earnings_transcript",
           "fact_checker"},
    "C6": {"web_search", "news_sentiment", "financial_data_api"},
    "C7": {"vector_db_search", "web_search"},
    "C8": {"financial_data_api", "sec_filing_search", "earnings_transcript", "news_sentiment", "web_search",
           "company_profile"},
}

#: id -> (name, category, target text, method, unit, comparator, threshold). AB-5 overrides the
#: target and threshold per run (single- vs multi-company).
_SPEC: dict[str, tuple[str, str, str, str, str, Callable[[Any, Any], bool], float]] = {
    "FA-1": ("Numerical Accuracy Rate", "Factual Accuracy", ">= 98%", "automated", "ratio", operator.ge, 0.98),
    "FA-2": ("Citation Accuracy", "Factual Accuracy", "100%", "automated", "ratio", operator.ge, 1.0),
    "FA-3": ("Temporal Accuracy", "Factual Accuracy", "100%", "automated", "ratio", operator.ge, 1.0),
    "FA-4": ("Entity Accuracy", "Factual Accuracy", "100%", "automated", "ratio", operator.ge, 1.0),
    "FA-5": ("Hallucination Rate", "Factual Accuracy", "0", "automated", "count", operator.le, 0),
    "CO-1": ("Section Coverage", "Completeness", "100%", "automated", "ratio", operator.ge, 1.0),
    "CO-2": ("Data Source Diversity", "Completeness", ">= 4 source types", "automated", "types", operator.ge, 4),
    "CO-3": ("Temporal Coverage", "Completeness", ">= 3 years", "automated", "years", operator.ge, 3),
    "CO-4": ("Risk Factor Coverage", "Completeness", ">= 80%", "automated", "ratio", operator.ge, 0.8),
    "AD-1": ("Insight Density", "Analytical Depth", ">= 3 per page", "llm_judge", "per_page", operator.ge, 3),
    "AD-2": ("Cross-Source Synthesis", "Analytical Depth", ">= 5", "automated", "count", operator.ge, 5),
    "AD-3": ("Quantitative Reasoning", "Analytical Depth", ">= 10 original calculations", "automated", "count",
             operator.ge, 10),
    "AD-4": ("Forward-Looking Analysis", "Analytical Depth", ">= 2 sections", "automated", "count", operator.ge, 2),
    "CS-1": ("Logical Flow", "Coherence and Structure", ">= 4 / 5", "llm_judge", "score_1_5", operator.ge, 4),
    "CS-2": ("Internal Consistency", "Coherence and Structure", "0 contradictions", "hybrid", "count",
             operator.le, 0),
    "CS-3": ("Executive Summary Quality", "Coherence and Structure", ">= 4 / 5", "llm_judge", "score_1_5",
             operator.ge, 4),
    "CS-4": ("Professional Formatting", "Coherence and Structure", ">= 80% of checks", "automated", "ratio",
             operator.ge, 0.8),
    "AB-1": ("Tool Efficiency", "Agent Behaviour", ">= 70%", "automated", "ratio", operator.ge, 0.7),
    "AB-2": ("Error Recovery Rate", "Agent Behaviour", ">= 90%", "automated", "ratio", operator.ge, 0.9),
    "AB-3": ("Planning Quality", "Agent Behaviour", ">= 4 / 5", "hybrid", "score_1_5", operator.ge, 4),
    "AB-4": ("Memory Utilization", "Agent Behaviour", ">= 0.3", "automated", "ratio", operator.ge, 0.3),
    "AB-5": ("Latency", "Agent Behaviour", "< 300 s", "automated", "seconds", operator.lt, 300),
}

_EPS = 1e-9
_UNVERIFIED = re.compile(r"\*\(unverified[^)]*\)\*")
_NON_BODY = {"data_gaps_and_limitations", "research_methodology", "sources", "references"}
_STOPWORDS = frozenset(
    "this that with from have has had were was been will would could should their there which while about "
    "also than then them they into over under more most such each other some only both because after before "
    "during between among its our your these those being does did not but and the for are can may".split())
_JUDGE_OFF = "LLM judge not run"


def _snip(text: str, n: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _make(mid: str, value: float | int | None, detail: str = "", examples: list[str] | None = None,
          *, target: str | None = None, threshold: float | None = None) -> MetricResult:
    name, category, spec_target, method, unit, cmp, spec_threshold = _SPEC[mid]
    if isinstance(value, float):
        value = round(value, 4)
    passed = None if value is None else bool(cmp(value + _EPS if cmp is operator.ge else value,
                                                 spec_threshold if threshold is None else threshold))
    return MetricResult(mid, name, category, value, target or spec_target, passed, method, unit, detail,
                        [_snip(e) for e in (examples or [])[:5]])


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


# --------------------------------------------------------------------------------------------------
# Run: a parsed bundle
# --------------------------------------------------------------------------------------------------


@dataclass
class Claim:
    text: str  # sentence or table row, unverified marker removed
    raw: str  # as written in the report
    ids: set[str]  # evidence ids it cites (own or inherited)
    quantities: list[Quantity]

    @property
    def flagged_unverified(self) -> bool:
        return self.raw != self.text


def _tool_result(eid: str, rec: dict[str, Any]) -> ToolResult:
    known = {f.name for f in dataclasses.fields(ToolResult)}
    fields = {k: v for k, v in rec.items() if k in known}
    fields.setdefault("evidence_id", eid)
    fields.setdefault("tool", "unknown")
    fields.setdefault("args", {})
    fields.setdefault("data", {})
    fields.setdefault("source_tier", None)
    fields.setdefault("retrieved_at", "")
    fields.setdefault("latency_ms", 0.0)
    return ToolResult(**fields)


def _digest(result: ToolResult) -> str:
    try:
        return evidence_digest(result, 20000)
    except Exception:  # malformed record in an old bundle: fall back to raw JSON text
        return json.dumps(result.data, default=str)[:20000]


class Run:
    """One bundle, parsed once: evidence digests, body text and the claim list."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        self.bundle = bundle
        self.challenge_id: str = (bundle.get("challenge") or {}).get("id", "")
        self.template: str = bundle.get("template") or (bundle.get("profile") or {}).get("template", "")
        self.query: str = bundle.get("query", "")
        self.tickers: list[str] = list((bundle.get("profile") or {}).get("tickers") or [])
        self.evidence: dict[str, ToolResult] = {
            eid: _tool_result(eid, rec) for eid, rec in (bundle.get("evidence") or {}).items() if rec}
        self.digests: dict[str, str] = {eid: _digest(r) for eid, r in self.evidence.items()}
        self.sections: dict[str, str] = dict(bundle.get("sections") or {})
        self.body_sections: dict[str, str] = {
            k: v for k, v in self.sections.items() if k not in _NON_BODY and (v or "").strip()}
        self.body_text: str = "\n\n".join(self.body_sections.values())
        self.report: str = bundle.get("report_markdown") or ""
        self.words: int = len(re.findall(r"\b\w+\b", self.report or self.body_text))
        self.pages: float = max(1.0, self.words / 500)
        self.claims: list[tuple[str, set[str]]] = _claims(self.body_text)
        self.claim_objs: list[Claim] = []
        for raw, ids in self.claims:
            text = _UNVERIFIED.sub("", raw).strip()
            self.claim_objs.append(Claim(text, raw, set(ids) | cited_ids(raw), extract_quantities(text)))
        self.numeric_claims: list[Claim] = [c for c in self.claim_objs if c.quantities]
        self._status_cache: dict[tuple[str, tuple[str, ...]], str] = {}
        self._amount_cache: dict[str, list[float]] = {}

    @classmethod
    def from_bundle(cls, bundle: dict[str, Any]) -> Run:
        return cls(bundle)

    @classmethod
    def load(cls, path: str | Path) -> Run:
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- helpers shared by several metrics -----------------------------------------------------

    def ids_of(self, tools: set[str]) -> list[str]:
        return [e for e, r in self.evidence.items() if r.tool in tools]

    def status(self, text: str, ids: list[str] | None = None) -> str:
        """fact_checker status of ``text`` against the digests of ``ids`` (default: all evidence)."""
        key = (text, tuple(sorted(self.digests if ids is None else ids)))
        if key not in self._status_cache:
            sources = [self.digests[i] for i in key[1] if i in self.digests]
            self._status_cache[key] = fact_checker(text, sources)["status"]
        return self._status_cache[key]

    def numeric_ok(self, text: str, ids: list[str] | None = None) -> bool:
        if self.status(text, ids) in ("supported", "partially_supported"):
            return True
        # A margin or growth rate the report computed itself is traceable (spec P2 asks for exactly
        # that), so accept a claim whose every quantity is either in the sources or derived from
        # figures the report states in the same paragraph (the working is visible to the reader; those
        # input figures are themselves checked by this metric).
        sources = [self.digests[i] for i in (self.digests if ids is None else ids) if i in self.digests]
        quantities = extract_quantities(text)
        return bool(quantities) and bool(sources) and all(
            fact_checker(q.text, sources)["status"] == "supported"
            or (q.kind == "percent" and (_derivable(q, self._section_amounts(text))
                                         # Only for cited records: a ratio of two fields of one statement row (margin)
                                         # or of one field across rows (growth), not of any two amounts in them.
                                         or (ids is not None and any(_row_derivable(q, self.digests[i])
                                                                     for i in ids if i in self.digests))))
            for q in quantities)

    def _section_amounts(self, text: str) -> list[float]:
        """Amounts stated in the paragraph that contains ``text`` (or in ``text`` itself).

        A paragraph, not the whole section: a dense peer table offers so many amount pairs that 62%
        of random one-decimal percentages "derive" from some pair (Day 11 debugging pass, C6).
        """
        own = {abs(q.value) for q in extract_quantities(text) if q.kind == "amount" and q.value}
        for body in self.body_sections.values():
            for para in re.split(r"\n\s*\n", _UNVERIFIED.sub("", body)):
                if text[:60] in para:
                    if para not in self._amount_cache:
                        self._amount_cache[para] = sorted({abs(q.value) for q in extract_quantities(para)
                                                           if q.kind == "amount" and q.value})
                    return sorted(own | set(self._amount_cache[para]))
        return sorted(own)

    def text_supported(self, claim: Claim, ids: list[str] | None = None) -> bool:
        """Numeric claims go through the fact checker, prose through a keyword-overlap rule."""
        if claim.quantities:
            return self.numeric_ok(claim.text, ids)
        return _keyword_support(claim.text, [self.digests[i] for i in (ids if ids is not None else self.digests)
                                             if i in self.digests])


_ONE_DECIMAL = re.compile(r"\d\.\d")


def _derivable(q: Quantity, amounts: list[float]) -> bool:
    """``q`` (a percentage written with decimals) equals a/b or (a - b)/b for two of ``amounts``.

    Only precisely written percentages qualify, and the tolerance is half a rounding step of the
    last printed decimal, so a coincidental match between unrelated figures stays unlikely.
    """
    printed = re.search(r"\d+(?:\.(\d+))?", q.text.replace(",", ""))
    decimals = len(printed.group(1) or "") if printed else 0
    if not _ONE_DECIMAL.search(q.text) or decimals == 0:
        return False
    tol = 0.5 * 10 ** -(decimals + 2) + 1e-6  # half a step of the last printed digit, as a fraction
    fraction = abs(q.value)  # "fell 46.5%" is often written without a minus sign
    for b in amounts:
        for target in (fraction * b, (1 + fraction) * b, (1 - fraction) * b):  # ratio, growth, decline
            lo = bisect.bisect_left(amounts, target - tol * b)
            if lo < len(amounts) and amounts[lo] <= target + tol * b and amounts[lo] != b:
                return True
    return False


_FIELD = re.compile(r"(\w+)=(-?\d[\d,]*(?:\.\d+)?)")


def _row_derivable(q: Quantity, digest: str) -> bool:
    """``q`` equals a/b of two fields of one statement row, or the growth of one field between two rows.

    Digests list statements as "FY 2026 (...): revenue=331,839; net_income=133,749" lines; a margin or
    growth rate quoted without its inputs is checkable against those rows. Pairing only fields of the
    same row / the same field keeps coincidental matches rare even for long statements.
    """
    printed = re.search(r"\d+(?:\.(\d+))?", q.text.replace(",", ""))
    decimals = len(printed.group(1) or "") if printed else 0
    if decimals == 0:
        return False
    tol = 0.5 * 10 ** -(decimals + 2) + 1e-6
    fraction = abs(q.value)
    rows = [{k: float(v.replace(",", "")) for k, v in _FIELD.findall(line)} for line in digest.splitlines()]
    rows = [r for r in rows if r]
    if re.search(r"pp|point", q.text, re.IGNORECASE):
        # "margin up 1.2 percentage points": the change of one a/b ratio between two rows, not a ratio itself.
        for i, newer in enumerate(rows):
            for older in rows[i + 1:]:
                for a in newer.keys() & older.keys():
                    for b in newer.keys() & older.keys():
                        if a != b and newer[b] and older[b] and abs(
                                abs(newer[a] / newer[b] - older[a] / older[b]) - fraction) <= tol:
                            return True
        return False
    for row in rows:
        for a in row.values():
            for b in row.values():
                if b and a != b and abs(abs(a / b) - fraction) <= tol:
                    return True
    for i, newer in enumerate(rows):
        for older in rows[i + 1:]:
            for key in newer.keys() & older.keys():
                if older[key] and abs(abs(newer[key] / older[key] - 1) - fraction) <= tol:
                    return True
    return False


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 4 and w not in _STOPWORDS}


def _keyword_support(sentence: str, digests: list[str]) -> bool:
    """A prose sentence is supported if >= 30% of its content words occur in the evidence."""
    words = _content_words(sentence)
    if not words or not digests:
        return False
    blob = " ".join(digests).lower()
    return sum(1 for w in words if w in blob) / len(words) >= 0.3


# --------------------------------------------------------------------------------------------------
# Factual accuracy
# --------------------------------------------------------------------------------------------------


def fa1_numerical_accuracy(run: Run, _j: dict | None) -> MetricResult:
    bad, good = [], 0
    for c in run.numeric_claims:
        cited = [i for i in c.ids if i in run.evidence]
        if (cited and run.numeric_ok(c.text, cited)) or run.numeric_ok(c.text):
            good += 1
        else:
            bad.append(c.text)
    total = len(run.numeric_claims)
    detail = (f"{good} of {total} numeric claims match their cited source (or, if uncited, any evidence)"
              if total else "no numeric claims in the report body")
    return _make("FA-1", _ratio(good, total), detail, bad)


def fa2_citation_accuracy(run: Run, _j: dict | None) -> MetricResult:
    cited = set(cited_ids(run.body_text)) | {i for c in run.claim_objs for i in c.ids}
    if not cited:
        return _make("FA-2", None, "the body cites no evidence ids")
    ok, bad = 0, []
    for eid in sorted(cited):
        if eid in run.evidence and any(
                run.text_supported(c, [eid]) for c in run.claim_objs if eid in c.ids):
            ok += 1
        else:
            bad.append(f"{eid}: {'unknown evidence id' if eid not in run.evidence else 'no citing sentence is supported'}")
    detail = (f"{ok} of {len(cited)} cited ids are real and support a claim; accessible = evidence record "
              "present with retrievable data (no URL fetching)")
    return _make("FA-2", ok / len(cited), detail, bad)


_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_QTR = re.compile(r"\bQ([1-4])\b")
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}
_QTR_WORD = re.compile(r"\b(first|second|third|fourth)\s+quarter\b", re.IGNORECASE)


def _periods(text: str) -> tuple[set[str], set[int]]:
    years = set(_YEAR.findall(text))
    quarters = {int(q) for q in _QTR.findall(text)} | {_ORDINALS[w.lower()] for w in _QTR_WORD.findall(text)}
    return years, quarters


def _digest_has_period(digest: str, years: set[str], quarters: set[int]) -> bool:
    if not all(re.search(rf"(?<!\d){y}(?!\d)", digest) for y in years):
        return False
    lowered = digest.lower()
    for q in quarters:  # "Q3", "third quarter" or the statement wording "three months ended"
        name = next(k for k, v in _ORDINALS.items() if v == q)
        if not (re.search(rf"\bq{q}\b", lowered) or f"{name} quarter" in lowered or "three months ended" in lowered):
            return False
    return True


def fa3_temporal_accuracy(run: Run, _j: dict | None) -> MetricResult:
    total, ok, bad = 0, 0, []
    for c in run.numeric_claims:
        years, quarters = _periods(c.text)
        if not years:
            continue
        total += 1
        # A calculation_engine record has inputs and a result but no fiscal years; its inputs came from
        # other evidence, so a claim citing only calculators is checked against all evidence.
        cited = [i for i in c.ids if i in run.evidence and run.evidence[i].tool != "calculation_engine"]
        cited = cited or list(run.digests)
        if any(_digest_has_period(run.digests[i], years, quarters) for i in cited):
            ok += 1
        else:
            bad.append(c.text)
    detail = (f"{ok} of {total} period-bearing numeric claims use a period present in their cited evidence"
              if total else "no numeric claims carry a period")
    return _make("FA-3", _ratio(ok, total), detail, bad)


_TICKER_PAREN = re.compile(r"\(([A-Z]{1,5})\)|\b(?:NASDAQ|NYSE)\s*:\s*([A-Z]{1,5})\b")
_NOT_TICKERS = frozenset("CEO CFO COO CTO AI GAAP USD EPS YOY FY US EU UK IP PE ROE ROI FCF CAGR IT API SEC GPU CPU "
                         "ESG TTM AWS GCP LLM SAAS R&D M&A".split())
def fa4_entity_accuracy(run: Run, _j: dict | None) -> MetricResult:
    body = _UNVERIFIED.sub("", run.body_text)
    blob = "\n".join(run.digests.values())
    # Sources are not tidy ("Carolina  Dybeck Happe" in Yahoo's officer list): compare on single spaces.
    blob_l, query_l = re.sub(r"\s+", " ", blob).lower(), run.query.lower()
    tickers = {a or b for a, b in _TICKER_PAREN.findall(body)} - _NOT_TICKERS
    companies = {n for n in KNOWN_COMPANIES if re.search(rf"\b{re.escape(n)}\b", body, re.IGNORECASE)}
    names = executive_names(body)
    known = set(run.tickers)
    bad = [t for t in sorted(tickers) if t not in known and not re.search(rf"\b{t}\b", blob + " " + run.query)]
    bad += [n for n in sorted(companies)
            if n not in blob_l and n not in query_l and KNOWN_COMPANIES[n] not in known
            and not re.search(rf"\b{KNOWN_COMPANIES[n]}\b", blob)]
    bad += [n for n in sorted(names) if n.lower() not in query_l and not name_in_sources(n, blob_l)]
    total = len(tickers) + len(companies) + len(names)
    detail = (f"{total - len(bad)} of {total} entities verified ({len(tickers)} tickers, {len(companies)} "
              f"companies, {len(names)} executive names)" if total else "no entities found in the report body")
    return _make("FA-4", _ratio(total - len(bad), total), detail, [f"unverified entity: {b}" for b in bad])


def fa5_hallucination_rate(run: Run, _j: dict | None) -> MetricResult:
    bad = [c.text for c in run.numeric_claims if not run.numeric_ok(c.text)
           and not (c.ids & set(run.evidence) and run.numeric_ok(c.text, sorted(c.ids & set(run.evidence))))]
    return _make("FA-5", len(bad), f"{len(bad)} of {len(run.numeric_claims)} numeric claims can be neither found in "
                 "nor derived (ratio or growth of two source amounts) from any evidence record", bad)


# --------------------------------------------------------------------------------------------------
# Completeness
# --------------------------------------------------------------------------------------------------


def co1_section_coverage(run: Run, _j: dict | None) -> MetricResult:
    required = TEMPLATES.get(run.template, [])
    if not required:
        return _make("CO-1", None, f"unknown template {run.template!r}")
    missing = [k for k in required if not (run.sections.get(k) or "").strip()]
    return _make("CO-1", (len(required) - len(missing)) / len(required),
                 f"{len(required) - len(missing)} of {len(required)} required {run.template} sections present",
                 [f"missing section: {k}" for k in missing])


_SOURCE_TYPES = {
    "sec_filing_search": "SEC filings", "financial_data_api": "financial data", "company_profile": "financial data",
    "earnings_transcript": "earnings calls", "news_sentiment": "news/web", "web_search": "news/web",
    "peer_comparison": "industry/peer data",
}


def _source_types(run: Run) -> set[str]:
    types: set[str] = set()
    for eid in cited_ids(run.body_text) & set(run.evidence):
        rec = run.evidence[eid]
        if rec.tool == "vector_db_search":
            stored = [(h.get("metadata") or {}).get("source_type") for h in (rec.data.get("results") or [])]
            types.update(str(s) for s in stored if s)
            if not any(stored):
                types.add("memory")
        elif rec.tool in _SOURCE_TYPES:  # calculation_engine, fact_checker, report_generator are not sources
            types.add(_SOURCE_TYPES[rec.tool])
    return types


def co2_source_diversity(run: Run, _j: dict | None) -> MetricResult:
    types = _source_types(run)
    return _make("CO-2", len(types), f"{len(types)} distinct source types cited: {', '.join(sorted(types)) or 'none'}")


def co3_temporal_coverage(run: Run, _j: dict | None) -> MetricResult:
    years: set[int] = set()
    for c in run.numeric_claims:
        years.update(int(y) for y in _YEAR.findall(c.text))
    for line in run.body_text.splitlines():  # table headers are not claims but carry the year columns
        if line.strip().startswith("|"):
            years.update(int(y) for y in _YEAR.findall(line))
    years = {y for y in years if 2000 <= y <= 2035}
    # Spec: a 3-5 year horizon. Only years within 5 years of the latest count, so one stale data point
    # (a peer's FY2014 figure) does not stretch the span of the years the report actually analyses.
    years = {y for y in years if y > max(years, default=0) - 5}
    if not years:
        return _make("CO-3", 0, "no years found in numeric claims or tables")
    return _make("CO-3", max(years) - min(years) + 1, f"years {min(years)}-{max(years)} appear in numeric claims/tables")


#: Risk taxonomy for CO-4: category -> keywords (matched as word prefixes, case-insensitive).
RISK_TAXONOMY: dict[str, tuple[str, ...]] = {
    "competition": ("competition", "competitor", "competitive"),
    "regulation/legal": ("regulat", "legal", "litigation", "lawsuit", "antitrust", "compliance"),
    "supply chain": ("supply chain", "supplier", "manufactur", "shortage"),
    "macroeconomic": ("macroeconomic", "interest rate", "inflation", "recession", "consumer demand", "economic conditions"),
    "key personnel": ("key personnel", "executive officers", "talent", "retain", "senior management"),
    "cybersecurity/privacy": ("cyber", "data privacy", "data breach", "security incident", "privacy"),
    "international/geopolitical": ("geopolitical", "tariff", "trade restriction", "export control", "international operations",
                                   "foreign"),
    "technology/product": ("technolog", "artificial intelligence", "new product", "product transition", "obsolete"),
    "financing/liquidity": ("liquidity", "indebtedness", "debt", "capital resources", "financing", "credit rating"),
    "customer concentration": ("customer concentration", "significant customers", "major customers", "largest customers"),
    "intellectual property": ("intellectual property", "patent", "trademark", "infring"),
    "climate/environmental": ("climate", "environmental", "emissions", "sustainab"),
}


def _risk_categories(text: str) -> set[str]:
    lowered = text.lower()
    return {cat for cat, kws in RISK_TAXONOMY.items() if any(re.search(rf"\b{re.escape(k)}", lowered) for k in kws)}


def co4_risk_coverage(run: Run, _j: dict | None) -> MetricResult:
    filings = run.ids_of({"sec_filing_search"})
    tenk = _risk_categories(" ".join(run.digests[i] for i in filings))
    if not tenk:
        return _make("CO-4", None, "no 10-K risk text retrieved")
    covered = tenk & _risk_categories(run.body_text)
    return _make("CO-4", len(covered) / len(tenk),
                 f"{len(covered)} of {len(tenk)} risk categories in the 10-K are covered by the report",
                 [f"not covered: {c}" for c in sorted(tenk - covered)])


# --------------------------------------------------------------------------------------------------
# Analytical depth
# --------------------------------------------------------------------------------------------------


def ad1_insight_density(run: Run, judgement: dict | None) -> MetricResult:
    if not judgement:
        return _make("AD-1", None, _JUDGE_OFF)
    count = judgement.get("insight_count", len(judgement.get("insights") or []))
    return _make("AD-1", round(count / run.pages, 2),
                 f"{count} insights over {run.pages:.1f} pages (~500 words each)", list(judgement.get("insights") or [])[:3])


def ad2_cross_source(run: Run, _j: dict | None) -> MetricResult:
    hits = [c.text for c in run.claim_objs
            if len({run.evidence[i].tool for i in c.ids if i in run.evidence}) >= 2]
    return _make("AD-2", len(hits), f"{len(hits)} claims cite evidence from two or more different tools", hits)


_COMPUTATION = re.compile(r"\b(?:growth|grew|margin|cagr|ratio|share|yoy|year-over-year|increase[ds]?|decrease[ds]?|"
                          r"decline[ds]?|per)\b", re.IGNORECASE)
_MULTIPLE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[x×](?!\w)")
_CALC_ROW = re.compile(r"margin|growth|ratio|cagr|yoy", re.IGNORECASE)
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUMBER.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def _derived(text: str) -> list[tuple[float, str, str]]:
    """(value, kind, quantity text) for percentages and multiples in ``text``."""
    found = [(q.value, "percent", q.text) for q in extract_quantities(text) if q.kind == "percent"]
    found += [(float(m.group(1)), "multiple", m.group(0)) for m in _MULTIPLE.finditer(text)]
    return found


def _in_numbers(value: float, digests: list[str]) -> bool:
    return any(abs(n - value) <= 0.02 * max(abs(value), 1e-9) for d in digests for n in _numbers(d))


def ad3_quantitative_reasoning(run: Run, _j: dict | None) -> MetricResult:
    calc = [run.digests[i] for i in run.ids_of({"calculation_engine"})]
    other_ids = [i for i, r in run.evidence.items() if r.tool != "calculation_engine"]
    other = [run.digests[i] for i in other_ids]
    found: set[tuple[float, str]] = set()
    for c in run.claim_objs:
        if c.text.lstrip().startswith("|"):
            continue  # table rows are handled below
        for value, kind, qtext in _derived(c.text):
            key = (round(value, 4), kind)
            from_calc = bool(calc) and (run.numeric_ok(qtext, [i for i in run.ids_of({"calculation_engine"})])
                                        if kind == "percent" else _in_numbers(value, calc))
            absent = (run.status(qtext, other_ids) == "unsupported") if kind == "percent" else not _in_numbers(value, other)
            if from_calc or (absent and _COMPUTATION.search(c.text) and not c.flagged_unverified):
                found.add(key)
    lines = [ln.strip() for ln in run.body_text.splitlines() if ln.strip().startswith("|")]
    for n, line in enumerate(lines):
        cells = [x.strip() for x in line.strip("|").split("|")]
        if set(line) <= set("|-: ") or (n + 1 < len(lines) and set(lines[n + 1]) <= set("|-: ")):
            continue  # separator or header row
        if cells and _CALC_ROW.search(cells[0]):
            for cell in cells[1:]:
                derived = _derived(cell)
                if derived:
                    found.update((round(v, 4), k) for v, k, _ in derived)
                else:
                    found.update((round(v, 4), "cell") for v in _numbers(cell))
    return _make("AD-3", len(found), f"{len(found)} distinct derived quantities (percentages, multiples, calculated table cells)")


_FORWARD_TITLE = re.compile(r"outlook|forward|scenario|projection|guidance|catalyst|recommendation|thesis", re.IGNORECASE)
_FORWARD = re.compile(r"\b(?:will|expects?|guidance|forecasts?|project(?:ed|ion|ions|s)?|scenarios?|bull case|bear case|"
                      r"next year|going forward|could reach|outlook)\b", re.IGNORECASE)


def ad4_forward_looking(run: Run, _j: dict | None) -> MetricResult:
    hits = []
    for key, body in run.body_sections.items():
        titled = _FORWARD_TITLE.search(key) or _FORWARD_TITLE.search(section_title(key))
        if titled or sum(1 for s in _sentences(body) if _FORWARD.search(s)) >= 2:
            hits.append(key)
    return _make("AD-4", len(hits), f"{len(hits)} forward-looking sections: {', '.join(hits) or 'none'}")


# --------------------------------------------------------------------------------------------------
# Coherence and structure
# --------------------------------------------------------------------------------------------------

_METRICS = ("free cash flow", "operating income", "operating margin", "gross margin", "net income", "revenue",
            "capex", "eps")
_PERCENT_METRICS = {"gross margin", "operating margin"}
_COMPANY_WORD = {n: re.compile(rf"\b{re.escape(n)}\b", re.IGNORECASE) for n in KNOWN_COMPANIES}


def _entity(text: str, tickers: list[str]) -> str | None:
    mentioned = {KNOWN_COMPANIES[n] for n, rx in _COMPANY_WORD.items() if rx.search(text)}
    mentioned |= {t for t in tickers if re.search(rf"\b{t}\b", text)}
    if len(tickers) == 1 and mentioned <= {tickers[0]}:
        return "_"  # one company researched: every unnamed figure is about it
    if len(mentioned) == 1:
        return next(iter(mentioned))
    return "_" if not mentioned and len(tickers) <= 1 else None


def _period_key(text: str) -> str | None:
    years, quarters = _periods(text)
    if len(years) != 1 or len(quarters) > 1:
        return None
    return f"{next(iter(years))}{f'-Q{next(iter(quarters))}' if quarters else ''}"


def _same_value(a: float, b: float) -> bool:
    if abs(a - b) <= 0.02 * max(abs(a), abs(b), 1e-12):
        return True
    return any(abs(a - b * f) <= 0.02 * max(abs(a), 1e-12) or abs(b - a * f) <= 0.02 * max(abs(b), 1e-12)
               for f in (1e3, 1e6, 1e9))  # same amount written in a different scale


#: Words before a metric name that do not narrow it to a segment ("total revenue" is revenue).
_NEUTRAL_QUALIFIERS = frozenset("total the its and of in for company consolidated annual quarterly reported gaap "
                                "fiscal full year group overall net diluted".split())
#: A company's own name before the metric ("Microsoft revenue", "Microsoft's revenue") is the company total.
_COMPANY_QUALIFIERS = frozenset(KNOWN_COMPANIES) | {t.lower() for t in KNOWN_COMPANIES.values()}


def _stated_value(text: str, metric: str, kind: str) -> tuple[str, float] | None:
    """``(qualifier, value)`` when the sentence states the metric's value right after naming it.

    Only "revenue was $X" / "revenue of $X" / "Revenue: $X" shapes count, and the word before the
    metric ("Mac revenue", "AWS revenue") becomes part of the metric, so a segment figure is never
    compared with the company total. "3M" as a partner's name or "$10bn in revenue" are skipped.
    """
    # An optional possessive ("Microsoft's", curly apostrophe, or U+FFFD from a mangled one) is not part of
    # the qualifier word, and "revenues" is the metric too.
    m = re.search(rf"(?:([A-Za-z0-9&.\-]+)(?:['’�]s)?\s+)?\b{re.escape(metric)}s?\b", text, re.IGNORECASE)
    if m is None:
        return None
    window = " ".join(text[m.end():].split()[:6])
    quantities = [q for q in extract_quantities(window) if q.kind == kind]
    if not quantities:
        return None
    before = (m.group(1) or "").lower().strip(".")
    if before in _NEUTRAL_QUALIFIERS or before in _COMPANY_QUALIFIERS or _YEAR.fullmatch(before) or re.fullmatch(r"fy\d{2,4}|q[1-4]", before):
        before = ""
    return before, quantities[0].value


def _automated_contradictions(run: Run) -> list[str]:
    seen: dict[tuple[str, str, str, str], list[tuple[float, str]]] = {}
    for c in run.numeric_claims:
        lowered = c.text.lower()
        present = [m for m in _METRICS if re.search(rf"\b{m}s?\b", lowered)]
        period, entity = _period_key(c.text), _entity(c.text, run.tickers)
        if len(present) != 1 or period is None or entity is None:
            continue  # ambiguous sentences are skipped: a false contradiction is worse than a missed one
        metric = present[0]
        kind = "percent" if metric in _PERCENT_METRICS else "amount"
        stated = _stated_value(c.text, metric, kind)
        if stated is not None:
            qualifier, value = stated
            seen.setdefault((f"{qualifier} {metric}".strip(), period, entity, kind), []).append((value, c.text))
    out = []
    for (metric, period, entity, _k), items in seen.items():
        first = items[0]
        other = next((it for it in items[1:] if not _same_value(first[0], it[0])), None)
        if other:
            out.append(f"{metric} {period} ({entity}): {first[1]} VS {other[1]}")
    return out


def cs1_logical_flow(run: Run, judgement: dict | None) -> MetricResult:
    lf = (judgement or {}).get("logical_flow")
    if not lf:
        return _make("CS-1", None, _JUDGE_OFF)
    return _make("CS-1", lf.get("score"), _snip(lf.get("rationale", ""), 300))


def cs2_internal_consistency(run: Run, judgement: dict | None) -> MetricResult:
    auto = _automated_contradictions(run)
    judged = list((judgement or {}).get("contradictions") or [])
    examples = auto + [f"{j.get('a', '')} VS {j.get('b', '')}" for j in judged]
    detail = f"{len(auto)} automated (same metric/period/entity differing >2%) + {len(judged)} from the judge"
    if judgement is None:
        detail += "; LLM judge not run, automated check only"
    return _make("CS-2", len(auto) + len(judged), detail, examples)


def _exec_number_overlap(run: Run) -> tuple[int, int]:
    summary = run.body_sections.get("executive_summary", "")
    rest = "\n".join(v for k, v in run.body_sections.items() if k != "executive_summary")
    quantities = extract_quantities(summary)
    shared = sum(1 for q in quantities if fact_checker(q.text, [rest])["status"] == "supported")
    return shared, len(quantities)


def cs3_exec_summary(run: Run, judgement: dict | None) -> MetricResult:
    shared, total = _exec_number_overlap(run)
    auto = f"{shared} of {total} numbers in the executive summary also appear in the body" if total else \
        "executive summary has no numbers"
    es = (judgement or {}).get("exec_summary")
    if not es:
        return _make("CS-3", None, f"{_JUDGE_OFF}; automated: {auto}")
    return _make("CS-3", es.get("score"), f"{_snip(es.get('rationale', ''), 200)} Automated: {auto}",
                 list(es.get("missed_findings") or []))


_ARTEFACTS = re.compile(r"\bNone\b(?! of\b)|\bnan\b|\{'|Traceback|MOCK RESPONSE|TODO")  # "None of ..." is prose


def cs4_formatting(run: Run, _j: dict | None) -> MetricResult:
    report = run.report
    checks: dict[str, bool] = {}
    checks["H1 title present"] = bool(re.search(r"^# \S", report, re.MULTILINE))
    headers = {h.strip().lower() for h in re.findall(r"^##\s+(.+)$", report, re.MULTILINE)}
    present = [k for k in TEMPLATES.get(run.template, []) if (run.sections.get(k) or "").strip()]
    checks["every present required section has a ## header"] = all(section_title(k).lower() in headers for k in present)
    checks["at least one markdown table"] = bool(re.search(r"^\|[\s:|-]*-[\s:|-]*\|?\s*$", report, re.MULTILINE))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", run.body_text)
                  if len(p.strip()) > 40 and not p.strip().startswith(("|", "#"))]
    cited = sum(1 for p in paragraphs if cited_ids(p))
    checks["at least 80% of paragraphs cite evidence"] = bool(paragraphs) and cited / len(paragraphs) >= 0.8
    low, high = (run.bundle.get("challenge") or {}).get("expected_pages") or (0, 10**6)
    checks["length within expected pages"] = 0.6 * low <= run.words / 500 <= 1.5 * high
    checks["no artefacts (None, nan, MOCK RESPONSE, ...)"] = not _ARTEFACTS.search(report)
    checks["sources section present"] = bool(re.search(r"^##\s+(?:Sources|References)\b", report,
                                                       re.MULTILINE | re.IGNORECASE))
    failed = [name for name, ok in checks.items() if not ok]
    return _make("CS-4", (len(checks) - len(failed)) / len(checks),
                 f"{len(checks) - len(failed)} of {len(checks)} checks passed ({run.words} words, "
                 f"{cited}/{len(paragraphs)} paragraphs cited)", [f"failed: {n}" for n in failed])


# --------------------------------------------------------------------------------------------------
# Agent behaviour
# --------------------------------------------------------------------------------------------------


def _plan_steps(run: Run) -> list[dict[str, Any]]:
    return list((run.bundle.get("plan") or {}).get("steps") or [])


def ab1_tool_efficiency(run: Run, _j: dict | None) -> MetricResult:
    research = {e for s in _plan_steps(run) for e in s.get("evidence_ids") or []}
    useful = research & cited_ids(run.body_text)
    calls = int(run.bundle.get("tool_calls") or 0)
    return _make("AB-1", len(useful) / max(1, calls),
                 f"{len(useful)} research evidence records cited in the body over {calls} tool calls")


_REPAIRABLE = frozenset({"invalid_arguments", "validation", "schema"})


def _call_key(entry: dict[str, Any]) -> tuple[str, str]:
    return entry.get("tool", ""), json.dumps(entry.get("args"), sort_keys=True, default=str)


def ab2_error_recovery(run: Run, _j: dict | None) -> MetricResult:
    log = list(run.bundle.get("call_log") or [])
    episodes: list[tuple[int, tuple[str, str]]] = []
    for n, entry in enumerate(log):
        if entry.get("ok") is False:
            key = _call_key(entry)
            if episodes and episodes[-1][1] == key and all(log[m].get("ok") is False for m in range(episodes[-1][0], n)
                                                           if _call_key(log[m]) == key):
                continue  # consecutive retry of the same failing call: one episode
            episodes.append((n, key))
    if not episodes:
        return _make("AB-2", None, "no tool errors occurred")
    fallbacks = {r.data.get("fallback_for") for r in run.evidence.values() if r.is_fallback}
    recovered, lost = 0, []
    for n, key in episodes:
        repairable = log[n].get("error_kind") in _REPAIRABLE
        if (any(e.get("ok") and _call_key(e) == key for e in log[n + 1:]) or key[0] in fallbacks
                # Bad arguments are recovered by calling the same tool again with corrected ones.
                or (repairable and any(e.get("ok") and e.get("tool") == key[0] for e in log[n + 1:]))):
            recovered += 1
        else:
            lost.append(f"{key[0]} {key[1]}")
    return _make("AB-2", recovered / len(episodes), f"{recovered} of {len(episodes)} error episodes recovered "
                 "(same call later succeeded, arguments were repaired, or a fallback tool's evidence exists)", lost)


def ab3_planning(run: Run, judgement: dict | None) -> MetricResult:
    steps = _plan_steps(run)
    used = {s.get("tool") for s in steps}
    expected = EXPECTED_TOOLS.get(run.challenge_id, set())
    coverage = f"{len(used & expected)}/{len(expected)}" if expected else "n/a"
    auto = (f"expected-tool coverage {coverage}; {len(steps)} plan steps, "
            f"{sum(1 for s in steps if s.get('status') == 'failed')} failed, "
            f"{sum(1 for s in steps if s.get('status') == 'skipped')} skipped")
    planning = (judgement or {}).get("planning")
    if not planning:
        return _make("AB-3", None, f"{_JUDGE_OFF}; automated: {auto}",
                     [f"expected tool not used: {t}" for t in sorted(expected - used)])
    return _make("AB-3", planning.get("score"), f"{_snip(planning.get('rationale', ''), 160)} Automated: {auto}",
                 [f"expected tool not used: {t}" for t in sorted(expected - used)])


def ab4_memory_utilization(run: Run, _j: dict | None) -> MetricResult:
    # The spec formula multiplies memory_hits by total_api_calls; that is a deliberate spec error
    # (ERROR_LOG.md, Error 1). A utilisation *ratio* needs hits divided by external API calls.
    log = run.bundle.get("call_log") or []
    recalls = sum(1 for r in run.evidence.values() if r.tool == "vector_db_search" and r.data.get("results"))
    cache_hits = sum(1 for e in log if e.get("cache_hit") and e.get("ok"))
    external = sum(1 for e in log if e.get("ok") is not None and TOOL_PROVIDERS.get(e.get("tool")) is not None
                   and not e.get("cache_hit"))
    hits = recalls + cache_hits
    return _make("AB-4", hits / max(1, external),
                 f"{hits} memory hits ({recalls} recalls with results + {cache_hits} cache hits) / {external} "
                 "external API calls")


def ab5_latency(run: Run, _j: dict | None) -> MetricResult:
    duration = run.bundle.get("duration_s")
    if duration is None:
        return _make("AB-5", None, "no duration recorded")
    # A vague query ("the banks") resolves to its companies only while running: with no ticker in the profile,
    # count the distinct tickers the tools were called with (peer lookups must not turn a single-company run
    # into a multi-company one).
    called = (
        {str(r.args["ticker"]).upper() for r in run.evidence.values() if r.args.get("ticker")}
        if not run.tickers else set())
    multi = len(set(run.tickers) | called) >= 2 or run.template in ("comparative_analysis", "thematic")
    limit = 600 if multi else 300
    return _make("AB-5", round(float(duration), 1), f"{'multi' if multi else 'single'}-company run, limit {limit} s",
                 target=f"< {limit} s", threshold=limit)


#: Every spec metric in spec order.
REGISTRY: list[Callable[[Run, dict | None], MetricResult]] = [
    fa1_numerical_accuracy, fa2_citation_accuracy, fa3_temporal_accuracy, fa4_entity_accuracy, fa5_hallucination_rate,
    co1_section_coverage, co2_source_diversity, co3_temporal_coverage, co4_risk_coverage,
    ad1_insight_density, ad2_cross_source, ad3_quantitative_reasoning, ad4_forward_looking,
    cs1_logical_flow, cs2_internal_consistency, cs3_exec_summary, cs4_formatting,
    ab1_tool_efficiency, ab2_error_recovery, ab3_planning, ab4_memory_utilization, ab5_latency,
]


def score_run(run: Run, judgement: dict | None = None, benchmark: MetricResult | None = None) -> list[MetricResult]:
    """All 21 spec metrics in spec order (plus ``benchmark`` when given)."""
    results = [fn(run, judgement) for fn in REGISTRY]
    if benchmark is not None:
        results.append(benchmark)
    return results
