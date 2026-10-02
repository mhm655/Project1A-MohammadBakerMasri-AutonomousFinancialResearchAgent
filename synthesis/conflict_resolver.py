"""conflict_resolver: turn raw claims from several tools into one reconciled, confidence-scored set.

Why this exists: the same figure (say FY2024 revenue) can arrive from SEC XBRL, yfinance and a peer
table, in different units and with small or large disagreements. The report must state ONE value per
(entity, metric, period), say how sure we are, and show its working when sources disagree. The rules
are the corrected tier table and confidence formula of spec section 12.3 / 12.4. Everything here is
deterministic: no network and no LLM, so the same evidence always produces the same ledger.

Two ideas keep the arithmetic honest:

* Grouping is by (entity, metric, fiscal year, fiscal period). Different periods are never a conflict.
* "Independent" means a different *base* provider. peer_comparison rebuilds its figures from SEC XBRL,
  so ``peer_comparison:sec_xbrl`` and ``sec_xbrl`` are the same provider and do not corroborate each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal
from urllib.parse import urlparse

from tools.fact_checker import TOLERANCE

DEFAULT_TIER_WEIGHTS: dict[int, float] = {1: 0.95, 2: 0.85, 3: 0.70, 4: 0.60, 5: 0.30}

MAJOR_NEWS_DOMAINS: tuple[str, ...] = (
    "reuters.com", "bloomberg.com", "ft.com", "wsj.com", "cnbc.com",
    "apnews.com", "nytimes.com", "marketwatch.com", "barrons.com", "economist.com",
)

#: Ratios (margins, growth) are also compared by absolute difference: 0.0050 vs 0.0052 is a 4 % relative
#: gap on a value that is really zero, and nobody would call that a disagreement.
RATIO_ABS_TOLERANCE = 0.005

_NEWS_TOOLS = ("web_search", "news_sentiment")

Status = Literal["verified", "single_source", "conflicted", "unverified"]


@dataclass
class ReliabilityConfig:
    """Tunable weights and penalties used when scoring claim reliability and resolving conflicts."""
    weights: dict[int, float] = field(default_factory=lambda: dict(DEFAULT_TIER_WEIGHTS))
    major_news_domains: tuple[str, ...] = MAJOR_NEWS_DOMAINS
    tolerance: float = TOLERANCE
    fallback_penalty: float = 0.90
    stale_penalty: float = 0.80
    unresolved_cap: float = 0.50
    agreement_bonus: float = 0.05
    max_confidence: float = 0.99

    def weight(self, tier: int | None) -> float:
        """Base weight of a tier; an unknown tier is treated as the least reliable one."""
        if tier is None or tier not in self.weights:
            return self.weights[5]
        return self.weights[tier]


def _host(url_or_host: str) -> str:
    text = (url_or_host or "").strip().lower()
    if "://" in text:
        text = urlparse(text).hostname or ""
    return text.split("/")[0].removeprefix("www.")


def tier_for_source(tool: str, source_tier: int | None, url_or_host: str = "") -> int:
    """Tier of one item. Web and news items are tiered by publisher; structured tools keep their own tier."""
    if tool in _NEWS_TOOLS:
        host = _host(url_or_host)
        major = any(host == d or host.endswith("." + d) for d in MAJOR_NEWS_DOMAINS)
        return 4 if major else 5
    return source_tier if source_tier is not None else 5


@dataclass
class Period:
    """A fiscal period: year, period label (``FY`` or ``Q1``..``Q4``) and optional period end date."""
    fiscal_year: int | None
    fiscal_period: str = "FY"  # "FY", "Q1".."Q4"
    period_end: str | None = None

    def key(self) -> tuple[int | None, str]:
        """Hashable (fiscal year, fiscal period) identity of the period."""
        return (self.fiscal_year, self.fiscal_period)

    def label(self) -> str:
        """Human-readable label such as ``FY2024`` or ``Q3 FY2024``."""
        if self.fiscal_year is None:
            return self.fiscal_period
        if self.fiscal_period == "FY":
            return f"FY{self.fiscal_year}"
        return f"{self.fiscal_period} FY{self.fiscal_year}"


@dataclass
class Claim:
    """One value for one metric of one entity and period, with its evidence, source tiers and provider."""
    claim_id: str
    entity: str
    metric: str
    period: Period
    value: float | str
    unit: str  # "USD", "ratio", "USD/share", "count"
    evidence_ids: list[str]
    source_tiers: list[int]
    provider: str  # "sec_xbrl", "yfinance", "peer_comparison:sec_xbrl", "news", ...
    filed: str | None = None  # ISO date of the filing, when known (used to spot restatements)
    is_fallback: bool = False
    stale: bool = False
    confidence: float = 0.0
    status: Status = "unverified"


@dataclass
class ConflictRecord:
    """A disagreement between sources for one metric and period, how it was resolved and the confidence."""
    entity: str
    metric: str
    period: Period
    values: list[dict[str, Any]]  # each {value, evidence_id, tier, provider}
    chosen_value: float | str
    method: str  # restatement_latest_filing | highest_tier | triangulated_majority | unresolved_range
    rationale: str
    confidence: float
    unit: str = "USD"  # lets the report format the values above


# --------------------------------------------------------------------------------------------------
# Formatting (shared with the narrative module so a figure reads the same everywhere)
# --------------------------------------------------------------------------------------------------


def format_money(usd: float) -> str:
    """Format USD as dollars, in millions or billions for large amounts."""
    sign = "-" if usd < 0 else ""
    a = abs(usd)
    if a >= 1e9:
        return f"{sign}${a / 1e9:,.2f} billion"
    if a >= 1e6:
        return f"{sign}${a / 1e6:,.1f} million"
    return f"{sign}${a:,.0f}"


def format_value(value: float | str, unit: str) -> str:
    """Format a value according to its unit (USD, USD/share, ratio or plain count)."""
    if isinstance(value, str):
        return value
    if unit == "USD":
        return format_money(value)
    if unit == "USD/share":
        return f"${value:,.2f} per share"
    if unit == "ratio":
        return f"{value * 100:.1f}%"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


# --------------------------------------------------------------------------------------------------
# Agreement and confidence
# --------------------------------------------------------------------------------------------------


def values_agree(a: float | str, b: float | str, unit: str = "USD", tolerance: float = TOLERANCE) -> bool:
    """True when two values are the same fact: within tolerance relative to the larger, or (ratios) 0.005 apart."""
    if isinstance(a, str) or isinstance(b, str):
        return str(a).strip().casefold() == str(b).strip().casefold()
    diff = abs(a - b)
    if unit == "ratio" and diff <= RATIO_ABS_TOLERANCE:
        return True
    denom = max(abs(a), abs(b))
    return denom == 0 or diff / denom <= tolerance


def confidence(
    best_tier: int | None,
    independent_agreeing: int,
    is_fallback: bool,
    stale: bool,
    unresolved: bool,
    config: ReliabilityConfig,
) -> float:
    """Spec 12.4: tier weight plus an agreement bonus, then penalties, then the unresolved cap."""
    c = config.weight(best_tier) + config.agreement_bonus * (max(1, independent_agreeing) - 1)
    c = min(c, config.max_confidence)
    if is_fallback:
        c *= config.fallback_penalty
    if stale:
        c *= config.stale_penalty
    if unresolved:
        c = min(c, config.unresolved_cap)
    return round(c, 4)


def base_provider(provider: str) -> str:
    """"peer_comparison:sec_xbrl" -> "sec_xbrl": who originally produced the number."""
    return provider.rsplit(":", 1)[-1]


def _tier(c: Claim) -> int:
    return min(c.source_tiers) if c.source_tiers else 5


def _independent(members: list[Claim]) -> int:
    return len({base_provider(m.provider) for m in members})


def _union(members: list[Claim]) -> tuple[list[str], list[int]]:
    ids: list[str] = []
    tiers: set[int] = set()
    for m in members:
        ids.extend(e for e in m.evidence_ids if e not in ids)
        tiers.update(m.source_tiers)
    return ids, sorted(tiers)


def _best(members: list[Claim]) -> Claim:
    return min(members, key=_tier)  # min() keeps the first of equals


def _build(members: list[Claim], value: float | str, config: ReliabilityConfig, *,
           unresolved: bool = False, all_members: list[Claim] | None = None) -> Claim:
    """One claim standing for ``members`` (which all support ``value``); evidence covers ``all_members``."""
    winner = _best(members)
    ids, tiers = _union(all_members or members)
    n_ind = _independent(members)
    fallback = all(m.is_fallback for m in members)
    stale = all(m.stale for m in members)
    conf = confidence(_tier(winner), n_ind, fallback, stale, unresolved, config)
    if unresolved:
        status: Status = "conflicted"
    else:
        status = "verified" if n_ind >= 2 else "single_source"
    return replace(winner, value=value, evidence_ids=ids, source_tiers=tiers,
                   is_fallback=fallback, stale=stale, confidence=conf, status=status)


def _clusters(members: list[Claim], config: ReliabilityConfig) -> list[list[Claim]]:
    clusters: list[list[Claim]] = []
    for m in members:
        for cl in clusters:
            if values_agree(cl[0].value, m.value, m.unit, config.tolerance):
                cl.append(m)
                break
        else:
            clusters.append([m])
    return clusters


def _values(members: list[Claim]) -> list[dict[str, Any]]:
    return [{"value": m.value, "evidence_id": m.evidence_ids[0] if m.evidence_ids else "",
             "tier": _tier(m), "provider": m.provider} for m in members]


def _range_text(members: list[Claim]) -> str:
    nums = [m.value for m in members if not isinstance(m.value, str)]
    unit = members[0].unit
    if not nums:
        return ", ".join(sorted({str(m.value) for m in members}))
    if min(nums) == max(nums):
        return format_value(nums[0], unit)
    return f"{format_value(min(nums), unit)} to {format_value(max(nums), unit)}"


def _resolve_conflict(members: list[Claim], config: ReliabilityConfig) -> tuple[Claim, ConflictRecord]:
    first = members[0]
    label = f"{first.entity} {first.metric} for {first.period.label()}"

    # Restatement: one provider, several filings of the same period -> the latest filing wins.
    filed = [m.filed for m in members]
    if len({base_provider(m.provider) for m in members}) == 1 and all(filed) and len(set(filed)) > 1:
        winner = max(members, key=lambda m: m.filed or "")
        supporters = [m for m in members if values_agree(m.value, winner.value, m.unit, config.tolerance)]
        claim = _build(supporters, winner.value, config)
        method = "restatement_latest_filing"
        why = (f"{label} appears in several {base_provider(winner.provider)} filings with different values "
               f"({_range_text(members)}); the filing dated {winner.filed} is the latest, so it supersedes "
               f"the earlier ones.")
    else:
        clusters = sorted(_clusters(members, config), key=len, reverse=True)
        top = clusters[0]
        best_tier = min(_tier(m) for m in members)
        best_tier_members = [m for m in members if _tier(m) == best_tier]
        tiers_differ = len({_tier(m) for m in members}) > 1
        if len(members) >= 3 and len(top) >= 2 and (len(clusters) == 1 or len(top) > len(clusters[1])):
            claim = _build(top, _best(top).value, config)
            method = "triangulated_majority"
            outliers = [m for m in members if m not in top]
            why = (f"{len(top)} of {len(members)} sources agree on {label} within "
                   f"{config.tolerance:.0%}; the majority value is used and the outlying "
                   f"{'value' if len(outliers) == 1 else 'values'} ({_range_text(outliers)}) set aside.")
        elif tiers_differ and len(_clusters(best_tier_members, config)) == 1:
            winner = _best(members)
            supporters = [m for m in members if values_agree(m.value, winner.value, m.unit, config.tolerance)]
            claim = _build(supporters, winner.value, config)
            method = "highest_tier"
            why = (f"Sources disagree on {label} ({_range_text(members)}); the tier-{best_tier} source "
                   f"({winner.provider}) is the most reliable, so its value is used.")
        else:
            claim = _build(best_tier_members, best_tier_members[0].value, config, unresolved=True,
                           all_members=members)
            method = "unresolved_range"
            why = (f"Sources of similar reliability disagree on {label}, ranging from {_range_text(members)}; "
                   f"none can be preferred, so the range is reported and confidence is capped at "
                   f"{config.unresolved_cap:.2f}.")

    record = ConflictRecord(entity=first.entity, metric=first.metric, period=first.period,
                            values=_values(members), chosen_value=claim.value, method=method,
                            rationale=why, confidence=claim.confidence, unit=first.unit)
    return claim, record


def resolve(claims: list[Claim], config: ReliabilityConfig) -> tuple[list[Claim], list[ConflictRecord]]:
    """Reconcile claims per (entity, metric, period). Inputs are not mutated."""
    groups: dict[tuple, list[Claim]] = {}
    for c in claims:
        groups.setdefault((c.entity, c.metric, c.period.key()), []).append(c)

    resolved: list[Claim] = []
    conflicts: list[ConflictRecord] = []
    for members in groups.values():
        if len(members) == 1:
            m = members[0]
            resolved.append(replace(m, status="single_source", confidence=confidence(
                _tier(m), 1, m.is_fallback, m.stale, False, config)))
        elif all(values_agree(a.value, b.value, a.unit, config.tolerance)
                 for i, a in enumerate(members) for b in members[i + 1:]):
            resolved.append(_build(members, _best(members).value, config))
        else:
            claim, record = _resolve_conflict(members, config)
            resolved.append(claim)
            conflicts.append(record)
    return resolved, conflicts
