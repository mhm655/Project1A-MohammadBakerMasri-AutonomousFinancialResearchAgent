"""fact_checker: check the numbers in a claim against source texts.

Implements the numeric core of the verification pipeline (spec A4.3): extract every quantity from
the claim, look for the same quantity in the sources (accepting different scales, e.g. "$245.1
billion" vs "245,122 million"), and accept matches within a 5 % relative tolerance.

Day 2: works on source texts passed by the caller. Day 7: when ``sources`` is omitted, it also
re-queries independent tools (XBRL, market data) and can then return ``contradicted``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

TOOL = "fact_checker"

TOLERANCE = 0.05  # spec A4.3: flag disagreements above 5 %

_SCALES = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "trillion": 1e12, "tn": 1e12, "t": 1e12,
}

# The lookbehind skips numbers glued to letters or hyphens ("Q4", "FY24", "3-for-1"); the lookahead
# skips form names such as "10-K" and "8-K" and period labels such as "3Q24" and "2H25", and stops the
# engine backtracking to a partial number ("1" of "10-K"). Percentage points count as percentages;
# basis points are hundredths of a percent.
_QUANTITY = re.compile(
    r"(?<![\w.\-])(?P<neg>-)?\$?(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?!\d|[.,]\d|-[A-Za-z]|[A-Za-z]+\d)\s*"
    r"(?:(?P<pct>%|percent\b|percentage\s+points?\b|pp\b|ppts?\b)"
    r"|(?P<bps>basis\s+points?|bps|bp)\b"
    r"|(?P<scale>thousand|million|billion|trillion|mn|bn|tn|[kmbt])\b)?",
    re.IGNORECASE,
)

# "10-12%", "$10-12 billion", "150 to 200 bps": the unit after a range applies to both ends, so it is
# copied to the first number ("10% to 12%"). A unit is required, so dates such as 2024-09-28 are
# left alone (the hyphen lookbehind above then keeps their parts from being read as numbers).
_UNIT = (r"%|percent\b|percentage\s+points?\b|pp\b|ppts?\b|basis\s+points?\b|bps\b|bp\b"
         r"|thousand\b|million\b|billion\b|trillion\b|mn\b|bn\b|tn\b|[kmbt]\b")
_RANGE = re.compile(rf"(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*\$?(\d+(?:\.\d+)?)\s*({_UNIT})", re.IGNORECASE)
# "30%-35%": a dash right after a percent sign is a range dash, not a minus sign on the next number.
_PERCENT_DASH = re.compile(r"%\s*[-–—]\s*(?=\$?\d)")

# Numbers that are almost always labels rather than measured quantities.
_YEAR = re.compile(r"^(19|20)\d{2}$")


@dataclass(frozen=True)
class Quantity:
    value: float  # absolute value in base units (a percentage is stored as a fraction)
    kind: str  # "percent" or "amount"
    scaled: bool  # True if an explicit scale word/suffix was present
    text: str


_MONTH = (r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
          r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")
# "June 1 2025", "June 1st, 2025", "1 June 2025", "June 2025", "09/28/2024": dates, not quantities.
_DATE = re.compile(
    rf"\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}\b|\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}\s+\d{{4}}\b"
    rf"|\b{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?!\s*(?:%|million|billion|thousand))"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    re.IGNORECASE,
)


def extract_quantities(text: str) -> list[Quantity]:
    """Find numeric quantities in text, skipping dates and bare years such as 2024."""
    text = _DATE.sub(" ", text)
    text = _PERCENT_DASH.sub("% to ", text)
    text = _RANGE.sub(lambda m: m.group(0) if _YEAR.match(m.group(1)) else f"{m.group(1)} {m.group(3)} to {m.group(2)} {m.group(3)}", text)
    found = []
    for m in _QUANTITY.finditer(text):
        raw = m.group("num")
        if not (m.group("pct") or m.group("bps") or m.group("scale")) and _YEAR.match(raw):
            continue
        value = float(raw.replace(",", ""))
        if m.group("neg"):
            value = -value
        if m.group("pct"):
            found.append(Quantity(value / 100, "percent", True, m.group(0).strip()))
        elif m.group("bps"):
            found.append(Quantity(value / 10_000, "percent", True, m.group(0).strip()))
        else:
            scale = m.group("scale")
            factor = _SCALES[scale.lower()] if scale else 1.0
            found.append(Quantity(value * factor, "amount", bool(scale), m.group(0).strip()))
    return found


def _relative_diff(a: float, b: float) -> float:
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b))


def _match(claim_q: Quantity, source_q: Quantity) -> tuple[float, bool] | None:
    """Return (relative difference, exact_scale) if the quantities agree within tolerance."""
    if claim_q.kind != source_q.kind:
        return None
    diff = _relative_diff(claim_q.value, source_q.value)
    if diff <= TOLERANCE:
        return diff, True
    # Tables often omit the unit ("245,122" meaning millions). Try the common scales, but mark the
    # match as scale-inferred so its confidence is lower.
    if claim_q.kind == "amount" and not (source_q.scaled and claim_q.scaled):
        # One side lacks a unit: tables often state "USD millions" or "USD billions" once, in a
        # header. Try the usual scale steps in both directions; the match is marked as inferred.
        for factor in (1e3, 1e6, 1e9):
            for a, b in ((claim_q.value, source_q.value * factor), (claim_q.value * factor, source_q.value)):
                diff = _relative_diff(a, b)
                if diff <= TOLERANCE:
                    return diff, False
    return None


def fact_checker(claim: str, sources: list[str] | None = None) -> dict[str, Any]:
    claim_quantities = extract_quantities(claim)
    sources = sources or []

    if not claim_quantities:
        return _result(claim, "unverifiable", 0.0, [], ["The claim contains no numeric quantity; qualitative claims are checked by the verifier LLM."])
    if not sources:
        return _result(claim, "unverifiable", 0.0, [], ["No sources supplied to check against."])

    evidence = []
    supported_quantities = 0
    for cq in claim_quantities:
        matches = []
        for idx, text in enumerate(sources):
            for sq in extract_quantities(text):
                hit = _match(cq, sq)
                if hit:
                    diff, exact_scale = hit
                    matches.append({
                        "source_index": idx,
                        "claim_text": cq.text,
                        "source_text": sq.text,
                        "relative_difference": round(diff, 4),
                        "scale_inferred": not exact_scale,
                    })
        if matches:
            supported_quantities += 1
            best = min(matches, key=lambda m: (m["scale_inferred"], m["relative_difference"]))
            best["supporting_sources"] = len({m["source_index"] for m in matches})
            evidence.append(best)

    if supported_quantities == len(claim_quantities):
        status = "supported"
    elif supported_quantities:
        status = "partially_supported"
    else:
        status = "unsupported"

    confidence = 0.0
    if evidence:
        per_quantity = []
        for e in evidence:
            base = 0.7 if e["scale_inferred"] else 0.9
            per_quantity.append(min(0.99, base + 0.05 * (e["supporting_sources"] - 1)))
        confidence = round(min(per_quantity) * supported_quantities / len(claim_quantities), 4)

    notes = []
    if status == "unsupported":
        notes.append("None of the claim's numbers were found in the sources (within 5 %).")
    return _result(claim, status, confidence, evidence, notes)


def _result(claim: str, status: str, confidence: float, evidence: list, notes: list[str]) -> dict[str, Any]:
    return {"claim": claim, "status": status, "confidence": confidence, "evidence": evidence, "notes": notes}


# --------------------------------------------------------------------------------------------------
# People: executive names must come from the sources too (a model can invent a plausible CEO)
# --------------------------------------------------------------------------------------------------

_TITLE = re.compile(r"\b(?:CEO|CFO|COO|CTO|Chief\s+(?:\w+\s+)?Officer|Chairman|Chairwoman|President|founder|"
                    r"Founder)\b")
_NAME_RUN = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+){1,2}\b")
_NAME_SKIP = frozenset("The Its Our In As At Chief Executive Officer Financial Operating Technology President Chairman "
                       "Founder And Of For With Board Directors Company Corporation Inc Vice Senior".split())


def executive_names(text: str) -> set[str]:
    """Two or three capitalised words within six tokens of a title word."""
    names: set[str] = set()
    for m in _NAME_RUN.finditer(text):
        words = [w for w in m.group(0).split() if w not in _NAME_SKIP]
        if len(words) < 2:
            continue
        before = text[: m.start()].split()[-6:]
        after = text[m.end():].split()[:6]
        if _TITLE.search(" ".join(before)) or _TITLE.search(" ".join(after)):
            names.add(" ".join(words))
    return names


def name_in_sources(name: str, blob_l: str) -> bool:
    """First and last name appear together in the sources; a written middle initial must agree.

    Sources give full names ("Jane Nind Fraser", "Brian Thomas Moynihan"), reports usually shorten
    them, so middle names may be dropped, but "Brian M. Moynihan" against "Brian Thomas" is wrong.
    """
    parts = name.lower().replace(".", "").split()
    first, last = parts[0], parts[-1]
    initial = parts[1][0] if len(parts) == 3 else ""
    for m in re.finditer(rf"\b{re.escape(first)}\b((?:\s+[a-z][\w.\-]*){{0,2}}?)\s+{re.escape(last)}\b", blob_l):
        middle = m.group(1).split()
        if not initial or not middle or middle[0][0] == initial:
            return True
    return False

def unsupported_names(text: str, sources: list[str]) -> list[str]:
    """Executive names in ``text`` that no source mentions (see :func:`name_in_sources`)."""
    blob_l = re.sub(r"\s+", " ", "\n".join(sources)).lower()
    return sorted(n for n in executive_names(text) if not name_in_sources(n, blob_l))
