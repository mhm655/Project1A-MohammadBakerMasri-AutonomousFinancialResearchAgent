"""report_generator: render researched sections into a structured Markdown report.

Each template defines the sections a report of that type must contain (metric CO-1). Missing
sections are rendered as explicit gaps rather than silently dropped (spec A4.3, graceful
degradation), and every citation such as ``[E3]`` must refer to a supplied source (metric FA-2).
Citations are rendered from the source list, never written freely, so URLs cannot be invented.
"""

from __future__ import annotations

import re
from typing import Any

TOOL = "report_generator"

TEMPLATES: dict[str, list[str]] = {
    "company_profile": ["executive_summary", "business_overview", "financial_summary", "key_executives", "recent_developments"],
    "earnings_analysis": ["executive_summary", "results_vs_expectations", "segment_performance", "earnings_call_takeaways", "guidance_and_outlook"],
    "risk_assessment": ["executive_summary", "financial_risks", "operational_risks", "regulatory_risks", "competitive_risks", "risk_matrix"],
    "comparative_analysis": ["executive_summary", "methodology", "revenue_growth", "market_share", "margins", "competitive_advantages", "conclusion"],
    "investigative": ["executive_summary", "the_contradiction", "evidence_review", "resolution", "implications"],
    "thematic": ["executive_summary", "themes", "cross_cutting_risks", "opportunities", "company_notes"],
    "full_report": ["executive_summary", "company_overview", "financial_analysis", "risk_assessment", "competitive_position", "outlook", "valuation_scenarios"],
}

#: Appended to every report. Rendered with a default text when not supplied.
STANDARD_SECTIONS = ["data_gaps_and_limitations", "research_methodology"]

_LOWERCASE_WORDS = {"vs", "and", "the", "of"}

# A citation group: [E1], [E1, E2], [E1; E2] or [E1 E2]. LLMs use all of these.
_CITATION_GROUP = re.compile(r"\[\s*E\d+(?:\s*[,;]?\s*E\d+)*\s*\]")
_EVIDENCE_ID = re.compile(r"E\d+")


def cited_ids(text: str) -> set[str]:
    """Every evidence id cited anywhere in ``text``."""
    return {eid for group in _CITATION_GROUP.findall(text) for eid in _EVIDENCE_ID.findall(group)}


def section_title(key: str) -> str:
    words = key.split("_")
    return " ".join(w if (i and w in _LOWERCASE_WORDS) else w.capitalize() for i, w in enumerate(words))


def _evidence_order(evidence_id: str) -> tuple[int, str]:
    digits = evidence_id[1:]
    return (int(digits) if digits.isdigit() else 0, evidence_id)


def report_generator(
    template: str,
    sections: dict[str, str],
    sources: list[dict[str, Any]],
    title: str | None = None,
) -> dict[str, Any]:
    required = TEMPLATES[template]
    provided = {k: v.strip() for k, v in sections.items() if isinstance(v, str) and v.strip()}
    missing = [s for s in required if s not in provided]
    extra = [s for s in provided if s not in required and s not in STANDARD_SECTIONS]
    source_ids = {s["id"] for s in sources}
    contains_mock = any(s.get("is_mock") for s in sources)

    lines = [f"# {title or section_title(template)}", ""]
    if contains_mock:
        lines += ["> **Warning:** this report was generated from MOCK data and must not be used or cited.", ""]

    for key in required + extra:
        lines += [f"## {section_title(key)}", ""]
        if key in provided:
            lines += [provided[key], ""]
        else:
            lines += ["_This section could not be completed. See Data Gaps and Limitations._", ""]

    gaps = provided.get("data_gaps_and_limitations", "")
    if missing:
        auto = "\n".join(f"- Section not completed: {section_title(s)}." for s in missing)
        gaps = f"{gaps}\n{auto}".strip()
    lines += ["## Data Gaps and Limitations", "", gaps or "No data gaps were recorded.", ""]
    lines += ["## Research Methodology", "", provided.get("research_methodology", "Not provided."), ""]

    lines += ["## Sources", ""]
    for s in sources:
        detail = ", ".join(str(s[k]) for k in ("source_type", "date") if s.get(k))
        entry = f"- [{s['id']}] {s['title']}"
        if detail:
            entry += f" ({detail})"
        if s.get("url"):
            entry += f". {s['url']}"
        lines.append(entry)
    if not sources:
        lines.append("_No sources._")

    markdown = "\n".join(lines).rstrip() + "\n"
    cited = cited_ids(" ".join(provided.values()))
    return {
        "template": template,
        "markdown": markdown,
        "sections_rendered": required + extra + STANDARD_SECTIONS,
        "missing_sections": missing,
        "extra_sections": extra,
        "unknown_citations": sorted(cited - source_ids, key=_evidence_order),
        "uncited_sources": sorted(source_ids - cited, key=_evidence_order),
        "word_count": len(re.findall(r"\b\w+\b", markdown)),
        "contains_mock_data": contains_mock,
    }
