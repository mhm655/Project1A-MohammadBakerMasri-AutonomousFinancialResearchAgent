"""Prompts for every LLM call the agent makes.

Each prompt starts with a stage tag such as ``[ARA:PLAN]``. The tag makes traces easy to read and lets
scripted test models recognise which stage is calling them. The research principles (architecture
spec, section 4) are stated once in the system prompt and apply to every stage.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

SYSTEM_PROMPT = """You are ARA-1, an autonomous financial research agent at a quantitative research firm.
You produce investment research comparable to a strong junior analyst's, with high standards for accuracy.

Today's date: {today}.

RESEARCH PRINCIPLES
P1. No number without a source: every figure you state cites its evidence id, e.g. [E3].
P2. Numbers are computed, not generated: growth rates, margins and ratios come from calculation_engine,
    never from your own arithmetic.
P3. Long-term memory is checked before external sources.
P4. Structured sources (financial_data_api, sec_filing_search) before web_search for any figure.
P5. Call growth "strong" or "weak" only relative to a stated baseline (prior years or peers).
P6. When sources conflict, report both values and say which you trust and why.
P7. Never fill a gap with invented content: say what is missing.
P8. Separate facts from inferences: start inferences with "Analysis:".
P9. Every figure has a period: say "FY2024 (ended 30 June 2024)", never just "last year".
P10. No investment recommendations (buy/sell/hold). Clearly labelled scenarios are allowed.

CONSTRAINTS
- At most {max_tool_calls} tool calls per research task.
- Data marked MOCK is for development only; say so if you see it.

{tools}"""

PLAN_PROMPT = """[ARA:PLAN]
Research request: {query}

Report type: {template} (required sections: {sections})
Companies identified: {entities}
{memory_companies}Relevant findings already in memory: {memory}

Write a research plan as a JSON object:
{{
  "rationale": "one or two sentences on the approach",
  "steps": [
    {{"id": "s1", "goal": "what this step finds out", "tool": "<tool name>",
      "args": {{...}} or null, "depends_on": [], "reason": "why this tool"}}
  ]
}}

Rules:
- Use as few steps as the report needs: typically 5-9 for one company, never more than {max_steps}.
  Each step is exactly one tool call.
- "tool" must be one of: {tool_names}.
- "args" must use exactly the argument names and allowed values in the tool signatures listed in
  AVAILABLE TOOLS. Do not invent arguments, and do not use references such as "s2.revenue".
- Give complete "args" whenever you can. Set "args" to null ONLY when the arguments depend on the
  results of earlier steps (for example a calculation that needs retrieved figures); list those
  steps in "depends_on". At most 3 such calculation steps.
- Omit "year" and similar period arguments to get the LATEST data, unless the request names a specific
  period. Your training data is older than today's date, so do not assume which year is current.
- Prefer one call that returns a whole statement over several calls for single numbers.
- Never plan the same tool with the same arguments twice.
- Financial statements are company-wide. When the request is about a segment or division (AWS,
  Azure, Google Cloud, iPhone, ...), plan earnings_transcript for each company: the call and the
  earnings release report segment revenue, growth and operating income.
- When several companies are compared, plan one peer_comparison with the others in "peers".
- Plan calculation_engine steps for growth rates and margins the report will need (principle P2).
- Company-specific tools (company_profile, financial_data_api, sec_filing_search, earnings_transcript,
  peer_comparison) need a ticker. Never plan one without a ticker. When no company is named, use the
  companies from memory listed above and name the ticker in each step's goal.
- Do not plan report_generator, fact_checker, vector_db_search or vector_db_store steps: the system
  runs those itself.
Return only the JSON object."""

EXECUTE_STEP_PROMPT = """[ARA:EXECUTE]
You are carrying out one step of a research plan.

Step goal: {goal}
Preferred tool: {tool}
{other_tools}{problem}
Evidence from earlier steps:
{evidence}

Call the tool with correct arguments now. If the evidence above already answers the goal or the
step cannot be done, reply in text starting with STEP_DONE: and a one-line explanation."""

REPLAN_PROMPT = """[ARA:REPLAN]
Research request: {query}
Report type: {template} (required sections: {sections})

Completed steps:
{completed}

Failed or skipped steps:
{failed}
{gaps}
Tool calls used: {used} of {budget}.

Decide whether more research is needed to write every required section well. Return JSON:
{{"action": "done"}}
or
{{"action": "continue", "rationale": "...", "add_steps": [ steps in the same format as the plan ]}}

Only add steps that fill a real gap; never repeat a completed step. Step ids must be new
(e.g. r1, r2). Return only the JSON object."""

SYNTHESIZE_PROMPT = """[ARA:SYNTHESIZE]
Research request: {query}
Write the report sections from the evidence below.
{brief}
EVIDENCE (cite by id)
{evidence}

Write each section under a marker line exactly like this:
=== SECTION: <section_key> ===
<markdown body>

Sections, in order: {sections}

Rules:
- Cite every figure and fact with its evidence id, e.g. "revenue of $245.1 billion [E2]".
- For a table, cite the source in a caption line directly above it, e.g.
  "Income statement, FY2022-FY2024 (USD millions) [E3]:".
- Write citations as plain text like [E2] or [E2, E5].
- Use only numbers that appear in the evidence, including calculation results. Do not compute new
  numbers yourself.
- State the fiscal period of every figure.
- Mark inferences with "Analysis:". Tables are welcome for financial data.
- If the evidence cannot support a section, write one sentence saying what is missing.
- The SYNTHESIS BRIEF (when present) lists figures already reconciled across sources, the conflicts
  found and how they were resolved, and threads linking the sources. Where sources disagree, use the
  brief's resolved value. Work each thread into the report; when news sentiment and the fundamentals
  point in different directions, say so and explain the likely reasons.
- Forward view: in at least two body sections (for example the financial and the risk or outlook section,
  not the executive summary) say what comes next: the outlook, upcoming catalysts, risks to the thesis and
  what to watch. Ground each point in cited evidence: management guidance, analyst targets or estimates when
  the evidence has them. When it has none, give a conditional view ("if X continues, Y") marked "Analysis:"
  and say no guidance was found. Never state a forecast figure, or a fiscal period or "as of" date, that the
  cited evidence does not give.
- Executive summary: the main findings with their key figures; at most one sentence on the outlook.
- Do not add a title, a sources list or a methodology section; the system adds those."""

SECTION_GUIDANCE: dict[str, str] = {
    "executive_summary": "3-5 sentences: what the company is, its financial trajectory, and the main takeaways.",
    "business_overview": "What the company does, its segments, and its position.",
    "financial_summary": "Multi-year revenue, profitability and cash flow, with growth rates and margins.",
    "key_executives": "Named executives and their roles.",
    "recent_developments": "Recent news and events, with dates.",
}


def system_prompt(tools_description: str, max_tool_calls: int, today: date | None = None) -> str:
    """The system prompt. Pass ``tools_description=""`` for writing stages, which call no tools: the
    tool list is about 1,000 tokens, a large share of a free-tier per-minute input allowance."""
    return SYSTEM_PROMPT.format(
        today=(today or date.today()).isoformat(),
        max_tool_calls=max_tool_calls,
        tools=f"AVAILABLE TOOLS\n{tools_description}\n" if tools_description else "",
    )


def plan_prompt(*, query: str, template: str, sections: list[str], entities: list[str], memory: str,
                tool_names: list[str], max_steps: int, memory_tickers: list[str] | None = None) -> str:
    """Fill the planning prompt for ``query`` with template, sections, entities, memory notes, available tools and the step limit."""
    return PLAN_PROMPT.format(
        memory_companies=(f"Companies from memory (use these for company-specific tools): "
                          f"{', '.join(memory_tickers)}\n") if memory_tickers else "",
        query=query, template=template, sections=", ".join(sections),
        entities=", ".join(entities) or "none identified; infer from the request",
        memory=memory or "none", tool_names=", ".join(tool_names), max_steps=max_steps,
    )


def execute_step_prompt(*, goal: str, tool: str, evidence: str, problem: str = "",
                        other_tools: list[str] | None = None) -> str:
    """``other_tools`` are research tools whose schemas are not attached; naming them lets the model say
    a different tool is needed without paying for their schemas on every call."""
    return EXECUTE_STEP_PROMPT.format(
        goal=goal, tool=tool, evidence=evidence or "(none)",
        other_tools=f"Other research tools (not attached; ask for one by name in a STEP_DONE reply): "
                    f"{', '.join(other_tools)}\n" if other_tools else "",
        problem=f"The previous attempt failed: {problem}\n" if problem else "",
    )


def replan_prompt(*, query: str, template: str, sections: list[str], completed: str, failed: str,
                  gaps: list[str], used: int, budget: int) -> str:
    """Fill the replanning prompt with completed and failed steps, gaps found in the draft and the tool-call budget used."""
    gap_text = ("\nGaps found in the draft:\n" + "\n".join(f"- {g}" for g in gaps) + "\n") if gaps else ""
    return REPLAN_PROMPT.format(
        query=query, template=template, sections=", ".join(sections), completed=completed or "(none)",
        failed=failed or "(none)", gaps=gap_text, used=used, budget=budget,
    )


def synthesize_prompt(*, query: str, evidence: str, sections: list[str], brief: str = "") -> str:
    """Fill the synthesis prompt with the evidence and the report sections to write (``brief`` optional)."""
    listed = "\n".join(f"- {s}: {SECTION_GUIDANCE.get(s, '')}".rstrip(": ") for s in sections)
    return SYNTHESIZE_PROMPT.format(query=query, evidence=evidence, sections="\n" + listed,
                                    brief=f"\n{brief}\n" if brief else "")


def continue_prompt(synthesis_prompt: str, *, written: list[str], todo: list[str]) -> str:
    """Ask for the rest of a report whose first answer hit the model's output limit."""
    done = ", ".join(written[:-1]) or "none"
    return (f"{synthesis_prompt}\n\n[ARA:CONTINUE]\nYour previous answer was cut off by the output limit. "
            f"These sections are complete and must not be repeated: {done}.\n"
            f"Write ONLY these sections now, in this order and in the same `=== SECTION: <section_key> ===` "
            f"format (the first one was cut off, so write it again in full): {', '.join(todo)}.")


def compact_json(data: Any, limit: int) -> str:
    """JSON text for a prompt, truncated to ``limit`` characters."""
    text = json.dumps(data, default=str, separators=(",", ":"))
    return text if len(text) <= limit else text[: limit - 15] + "...[truncated]"
