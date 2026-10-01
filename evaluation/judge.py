"""LLM-as-judge for report quality (spec A5.2 metrics AD-1, CS-1, CS-2, CS-3, AB-3).

One LLM call per report returns every rubric as a single JSON object (free-tier quota is tight):
non-obvious insights (AD-1), logical flow (CS-1), executive-summary accuracy (CS-2), planning quality
(CS-3) and internal contradictions (AB-3). ``evaluation.metrics`` turns that object into scores.

Judge bias: the judge may share a model family with the generator (a free-tier constraint). This is
mitigated by temperature 0, anchored 1-5 rubrics, and a requirement to quote contradictions exactly,
so a claim can be checked against the report text. Set ``ARA_JUDGE_PROVIDER_ORDER`` (for example
``groq,gemini``) to override the provider order for the judge only.

Results are cached on disk by sha256(JUDGE_VERSION + query + report_markdown), so re-scoring an
unchanged report costs no quota. Bump ``JUDGE_VERSION`` whenever ``JUDGE_PROMPT`` changes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

JUDGE_VERSION = "1"
DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "results" / "eval" / "judge_cache.json"
MAX_REPORT_CHARS = 40_000  # keeps one call inside free-tier context and token limits
_SKIP_SECTIONS = ("source", "reference")

JUDGE_PROMPT = """You are a strict senior equity-research editor reviewing an AI-written research report.
Judge only what is on the page. Be sceptical: do not reward length, confident tone, or numbers that are merely restated.

Research query:
{query}

Research plan (tool, goal, status):
{plan}

Report body{truncation_note}:
{report}

Score the report on these rubrics.

1. insights: list every NON-OBVIOUS analytical observation. It must go beyond restating a retrieved figure, for example
   connect two facts, explain a cause, draw an implication, or quantify a trade-off. Restated numbers, definitions and
   generic statements ("competition is intense") do NOT count. Each insight is at most 25 words.
   insight_count must equal the length of the list.

2. logical_flow, integer 1-5.
   5 = sections progress logically and every conclusion follows from evidence presented in the report.
   3 = mostly ordered, but some conclusions are unsupported or sections repeat each other.
   1 = disorganised, or conclusions contradict the evidence.

3. exec_summary, integer 1-5: does the executive summary accurately capture the key findings of the body, with no
   finding that is absent from the body and no important body finding missing?
   5 = faithful and complete; 3 = broadly right but omits or overstates something notable; 1 = misleading or empty.
   missed_findings lists important body findings that the summary leaves out (empty list if none).

4. planning, integer 1-5: did the plan cover the research steps this query needs (financials, filings, transcript,
   news, peers, as relevant) without redundant steps, and were failed steps handled sensibly?
   5 = complete, efficient, failures handled; 3 = some gaps or redundancy; 1 = plan misses the point of the query.

5. contradictions: only genuine self-contradictions inside the report, such as "rapidly growing" versus "stagnating"
   for the same thing and period, or two different values for the same metric and period. Quote both passages
   exactly (at most 30 words each). Do not flag differences between different periods or segments.
   Use an empty list if there are none.

Respond with one JSON object only, exactly this shape:
{{"insights": ["..."], "insight_count": 0,
 "logical_flow": {{"score": 1, "rationale": "..."}},
 "exec_summary": {{"score": 1, "rationale": "...", "missed_findings": ["..."]}},
 "planning": {{"score": 1, "rationale": "..."}},
 "contradictions": [{{"a": "exact quote", "b": "exact quote", "why": "..."}}]}}"""

RETRY_PROMPT = "Your last reply was not valid JSON. Reply again with the one JSON object only, no prose or code fences."


class JudgeError(RuntimeError):
    """The judge could not produce a usable verdict (provider failure or unparseable reply)."""


# --- prompt assembly -------------------------------------------------------------------------------

def _fit_sections(sections: list[tuple[str, str]], limit: int) -> tuple[list[tuple[str, str]], bool]:
    """Cut the longest sections so the total fits ``limit``; short sections stay whole.

    A per-section cap is found by bisection so the cut is shared among the longest sections
    proportionally to how far they exceed the cap.
    """
    total = sum(len(body) for _, body in sections)
    if total <= limit:
        return sections, False
    lo, hi = 0, max(len(body) for _, body in sections)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(len(body), mid) for _, body in sections) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return [(name, body if len(body) <= lo else body[:lo].rstrip() + " [...cut]") for name, body in sections], True


def _report_text(bundle: dict[str, Any]) -> tuple[str, bool]:
    sections = [(str(name), str(body or "")) for name, body in (bundle.get("sections") or {}).items()
                if not any(skip in str(name).lower() for skip in _SKIP_SECTIONS)]
    if not sections:
        sections = [("report", str(bundle.get("report_markdown") or ""))]
    fitted, cut = _fit_sections(sections, MAX_REPORT_CHARS)
    return "\n\n".join(f"## {name.replace('_', ' ').title()}\n{body}" for name, body in fitted), cut


def _plan_text(bundle: dict[str, Any]) -> str:
    steps = (bundle.get("plan") or {}).get("steps") or []
    if not steps:
        return "(no plan recorded)"
    return "\n".join(f"- {s.get('tool', '?')}: {s.get('goal', '')} [{s.get('status', '?')}]"
                     + (f" ({s['summary']})" if s.get("summary") and s.get("status") not in ("done", "ok") else "")
                     for s in steps)


def build_prompt(bundle: dict[str, Any]) -> str:
    report, cut = _report_text(bundle)
    note = (f" (the longest sections were shortened to fit {MAX_REPORT_CHARS:,} characters; do not penalise the cuts)"
            if cut else "")
    return JUDGE_PROMPT.format(query=bundle.get("query", ""), plan=_plan_text(bundle), report=report,
                               truncation_note=note)


# --- normalisation ----------------------------------------------------------------------------------

def _score(block: Any) -> tuple[int, str]:
    """Clamp a rubric score to an int in 1-5; a missing score becomes a neutral 3 and says so."""
    if not isinstance(block, dict):
        return 3, "The judge did not return this rubric."
    try:
        score = int(round(float(block.get("score"))))
    except (TypeError, ValueError):
        return 3, str(block.get("rationale") or "The judge returned no usable score.")
    return max(1, min(5, score)), str(block.get("rationale") or "")


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


def normalise(raw: dict[str, Any], *, provider: str, model: str) -> dict[str, Any]:
    flow, flow_why = _score(raw.get("logical_flow"))
    summ, summ_why = _score(raw.get("exec_summary"))
    plan, plan_why = _score(raw.get("planning"))
    insights = _strings(raw.get("insights"))
    summary_block = raw.get("exec_summary") if isinstance(raw.get("exec_summary"), dict) else {}
    contradictions = [
        {"a": str(c.get("a", "")).strip(), "b": str(c.get("b", "")).strip(), "why": str(c.get("why", "")).strip()}
        for c in (raw.get("contradictions") or []) if isinstance(c, dict) and (c.get("a") or c.get("b"))
    ]
    return {
        "insights": insights,
        "insight_count": len(insights),
        "logical_flow": {"score": flow, "rationale": flow_why},
        "exec_summary": {"score": summ, "rationale": summ_why,
                         "missed_findings": _strings(summary_block.get("missed_findings"))},
        "planning": {"score": plan, "rationale": plan_why},
        "contradictions": contradictions,
        "provider": provider,
        "model": model,
    }


# --- cache -------------------------------------------------------------------------------------------

def cache_key(bundle: dict[str, Any]) -> str:
    text = JUDGE_VERSION + str(bundle.get("query", "")) + str(bundle.get("report_markdown", ""))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}  # missing or corrupt cache: start fresh rather than fail the evaluation


def _save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, path)  # atomic: a crash never leaves a half-written cache
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# --- LLM ---------------------------------------------------------------------------------------------

def _default_llm() -> Any:
    from agent.llm import LLMClient
    from config.settings import load_settings

    order = os.environ.get("ARA_JUDGE_PROVIDER_ORDER")
    if not order:
        return LLMClient.from_settings(load_settings())
    # load_settings reads ARA_PROVIDER_ORDER from the environment, so override it only around the call.
    previous = os.environ.get("ARA_PROVIDER_ORDER")
    os.environ["ARA_PROVIDER_ORDER"] = order
    try:
        return LLMClient.from_settings(load_settings())
    finally:
        if previous is None:
            os.environ.pop("ARA_PROVIDER_ORDER", None)
        else:
            os.environ["ARA_PROVIDER_ORDER"] = previous


def _ask(llm: Any, messages: list[dict[str, Any]]) -> Any:
    try:
        return llm.complete(messages, json_mode=True, temperature=0, max_tokens=4096,
                            purpose="judge", reasoning_effort="low")
    except Exception as exc:  # every provider failed: surface one error type to the evaluator
        raise JudgeError(f"judge LLM call failed: {exc}") from exc


def judge_bundle(bundle: dict[str, Any], *, llm: Any | None = None, cache_path: Path | None = DEFAULT_CACHE,
                 force: bool = False) -> dict[str, Any]:
    """Judge one run bundle with a single LLM call (plus one retry if the reply is not JSON).

    Raises:
        JudgeError: the LLM call failed, or the reply was not valid JSON twice.
    """
    from agent.parser import extract_json

    key = cache_key(bundle)
    cache = _load_cache(cache_path) if cache_path else {}
    if key in cache and not force:
        return cache[key]

    llm = llm or _default_llm()
    messages: list[dict[str, Any]] = [{"role": "user", "content": build_prompt(bundle)}]
    response = _ask(llm, messages)
    try:
        raw = extract_json(response.content)
    except Exception:
        messages += [{"role": "assistant", "content": response.content or ""},
                     {"role": "user", "content": RETRY_PROMPT}]
        response = _ask(llm, messages)
        try:
            raw = extract_json(response.content)
        except Exception as exc:
            raise JudgeError(f"judge reply was not valid JSON after a retry: {exc}") from exc

    result = normalise(raw, provider=response.provider, model=response.model)
    if cache_path:
        cache[key] = result
        _save_cache(cache_path, cache)
    return result
