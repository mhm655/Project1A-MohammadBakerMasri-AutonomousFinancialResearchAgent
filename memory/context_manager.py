"""Short-term memory: fitting evidence into the LLM's prompt budget (architecture spec, section 10.2).

Every prompt the agent sends has a token budget. This is a hard constraint on free tiers: Groq accepts
about 7,000 input tokens per minute per model, and a synthesis prompt with ten full evidence items is
far larger.

:func:`fit_evidence` builds the evidence block for a prompt:

1. Evidence is split into **primary** data (filings, financial statements, calculations, profiles,
   peers) and **supporting** data (news, web, transcripts, memory), in the spec's 40:30 proportion.
   Unused budget in one group flows to the other.
2. Within a group, each item's digest is **shrunk evenly** until the group fits, down to a floor.
3. If it still does not fit, the **lowest-priority items are dropped**. Their ids are listed so that
   the model knows they exist but cannot be cited from this prompt.

Compression is extractive (the digests already put the most important content first), so it costs
no LLM calls.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

#: Spec section 10.2: 40 % primary data, 30 % supporting evidence (the other 30 % is instructions and output).
PRIMARY_SHARE, SUPPORTING_SHARE = 40, 30

PRIMARY_TOOLS = ("financial_data_api", "calculation_engine", "sec_filing_search", "company_profile", "peer_comparison")
#: Lower number = dropped last.
PRIORITY = {"financial_data_api": 0, "calculation_engine": 1, "company_profile": 2, "sec_filing_search": 3,
            "peer_comparison": 4, "earnings_transcript": 5, "news_sentiment": 6, "web_search": 7, "vector_db_search": 8}

MIN_ITEM_CHARS = 350
MAX_ITEM_CHARS = 2500
#: Items are joined with a blank line; each separator costs tokens too.
SEPARATOR = "\n\n"


@lru_cache(maxsize=1)
def _encoding() -> Any:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # tiktoken downloads its vocabulary on first use; fall back offline
        return None


def count_tokens(text: str) -> int:
    """Token count with tiktoken's cl100k encoding: an approximation for Gemini, Llama and Qwen
    tokenizers, which is all a budget needs. Falls back to ~4 characters per token."""
    enc = _encoding()
    return len(enc.encode(text, disallowed_special=())) if enc else max(1, len(text) // 4)


@dataclass
class FittedContext:
    text: str
    included: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    tokens: int = 0


def fit_evidence(evidence: dict[str, Any], budget_tokens: int,
                 digest: Callable[[Any, int], str]) -> FittedContext:
    """Render ``evidence`` (evidence id -> ToolResult) within ``budget_tokens``.

    ``digest(result, char_limit)`` renders one item; it should put the most important content first.
    """
    groups: dict[str, list[tuple[str, Any]]] = {"primary": [], "supporting": []}
    for eid, result in evidence.items():
        groups["primary" if result.tool in PRIMARY_TOOLS else "supporting"].append((eid, result))
    for items in groups.values():
        items.sort(key=lambda kv: (PRIORITY.get(kv[1].tool, 9), int(kv[0][1:]) if kv[0][1:].isdigit() else 0))

    total_share = PRIMARY_SHARE + SUPPORTING_SHARE
    budgets = {"primary": budget_tokens * PRIMARY_SHARE // total_share,
               "supporting": budget_tokens * SUPPORTING_SHARE // total_share}
    # A group that needs less than its share passes the rest to the other group.
    natural = {g: sum(count_tokens(digest(r, MAX_ITEM_CHARS)) for _, r in items) for g, items in groups.items()}
    for group, other in (("primary", "supporting"), ("supporting", "primary")):
        spare = budgets[group] - natural[group]
        if spare > 0:
            budgets[group] -= spare
            budgets[other] += spare

    rendered: list[tuple[str, str]] = []
    dropped: list[str] = []
    for group, items in groups.items():
        kept, lost = _fit_group(items, budgets[group], digest)
        rendered += kept
        dropped += lost

    text = SEPARATOR.join(t for _, t in rendered)
    if dropped:
        text += ("\n\n(Also retrieved but left out of this prompt for length, so do not cite: "
                 + ", ".join(dropped) + ".)")
    return FittedContext(text=text, included=[e for e, _ in rendered], dropped=dropped, tokens=count_tokens(text))


def _fit_group(items: list[tuple[str, Any]], budget: int,
               digest: Callable[[Any, int], str]) -> tuple[list[tuple[str, str]], list[str]]:
    kept = list(items)
    while kept:
        limit = MAX_ITEM_CHARS
        while True:
            texts = [(eid, digest(r, limit)) for eid, r in kept]
            # One separator per item: an upper bound on the joins, including the one between the groups.
            used = sum(count_tokens(t) for _, t in texts) + len(texts) * count_tokens(SEPARATOR)
            if used <= budget or limit <= MIN_ITEM_CHARS:
                break
            limit = max(MIN_ITEM_CHARS, int(limit * 0.8))
        if used <= budget:
            return texts, [eid for eid, _ in items if eid not in {e for e, _ in texts}]
        kept = kept[:-1]  # drop the lowest-priority item and try again
    return [], [eid for eid, _ in items]
