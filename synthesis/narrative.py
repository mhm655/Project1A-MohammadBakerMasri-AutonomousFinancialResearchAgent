"""narrative: turn the resolved ledger into themes the report can be written around.

Why this exists: a report that lists numbers from each tool in turn is not synthesis. Here the ledger
and the news evidence are compared to find where sources tell different stories, chiefly news
sentiment against reported fundamentals (Challenge 5: "news says struggling, financials show strong
growth"). Each theme becomes a :class:`Thread` with a thesis, the evidence for and against it, and a
"so what" the writer must address. Threads that draw on two or more tools count as cross-source
synthesis. Deterministic: no LLM, so every figure in a thesis is taken from the ledger or evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from memory.context_manager import count_tokens
from synthesis.conflict_resolver import Claim, ConflictRecord, format_money, format_value
from synthesis.engine import Ledger
from tools.base import ToolResult
from tools.news_sentiment import label as sentiment_label

_COMPANY_SUFFIXES = re.compile(r"[,.]?\s+(inc|incorporated|corp|corporation|co|company|ltd|plc|holdings|group)\.?$",
                               re.IGNORECASE)


@dataclass
class Thread:
    theme: str  # growth | profitability | sentiment_vs_fundamentals | competition | risk | capital_allocation
    entity: str
    thesis: str
    supporting: list[str]  # evidence ids
    contradicting: list[str]
    so_what: str
    source_types: set[str] = field(default_factory=set)  # tool names used


# --------------------------------------------------------------------------------------------------
# Ledger helpers
# --------------------------------------------------------------------------------------------------


def _annual(ledger: Ledger, entity: str, metric: str) -> list[Claim]:
    """Annual claims for a metric, latest fiscal year first."""
    found = [c for c in ledger.for_entity(entity)
             if c.metric == metric and c.period.fiscal_period == "FY" and c.period.fiscal_year is not None
             and not isinstance(c.value, str)]
    return sorted(found, key=lambda c: c.period.fiscal_year or 0, reverse=True)


def _consecutive(claims: list[Claim]) -> list[Claim]:
    """The run of consecutive fiscal years starting from the latest (a gap breaks comparability)."""
    run = claims[:1]
    for c in claims[1:]:
        if (run[-1].period.fiscal_year or 0) - (c.period.fiscal_year or 0) == 1:
            run.append(c)
        else:
            break
    return run


def _f(c: Claim) -> float:
    """Numeric value of a claim. Callers only pass claims from :func:`_annual` (never text values)."""
    return float(c.value)


def _ids(*claims: Claim | None) -> list[str]:
    out: list[str] = []
    for c in claims:
        if c:
            out.extend(e for e in c.evidence_ids if e not in out)
    return out


def _tools(evidence: dict[str, ToolResult], ids: list[str]) -> set[str]:
    by_id = {r.evidence_id: r.tool for r in evidence.values()}
    return {by_id[i] for i in ids if i in by_id}


def _cite(ids: list[str]) -> str:
    return "".join(f"[{i}]" for i in ids[:3])


def _entities(ledger: Ledger) -> list[str]:
    seen: list[str] = []
    for c in ledger.claims:
        if c.entity not in seen:
            seen.append(c.entity)
    return seen


def _growth(ledger: Ledger, entity: str) -> tuple[float, int, list[str]] | None:
    """(latest-year revenue growth, fiscal year, evidence ids) from two consecutive years, else the stated claim."""
    run = _consecutive(_annual(ledger, entity, "revenue"))
    if len(run) >= 2 and _f(run[1]):
        return _f(run[0]) / _f(run[1]) - 1, run[0].period.fiscal_year or 0, _ids(run[0], run[1])
    stated = _annual(ledger, entity, "revenue_growth")
    if stated:
        return _f(stated[0]), stated[0].period.fiscal_year or 0, _ids(stated[0])
    return None


def _net_income_direction(ledger: Ledger, entity: str) -> tuple[str, float, int, list[str]] | None:
    """("grew"/"fell"/"was positive"/"was negative", figure, year, ids): direction of net income."""
    run = _consecutive(_annual(ledger, entity, "net_income"))
    if not run:
        return None
    latest = run[0]
    if len(run) >= 2 and _f(run[1]) > 0:
        change = _f(latest) / _f(run[1]) - 1
        return ("grew" if change > 0 else "fell"), change, latest.period.fiscal_year or 0, _ids(latest, run[1])
    return ("was positive" if _f(latest) > 0 else "was negative"), _f(latest), latest.period.fiscal_year or 0, _ids(latest)


# --------------------------------------------------------------------------------------------------
# News
# --------------------------------------------------------------------------------------------------


def _company_names(evidence: dict[str, ToolResult], ticker: str) -> set[str]:
    names: set[str] = set()
    for r in evidence.values():
        if r.tool == "financial_data_api" and str(r.data.get("ticker", "")).upper() == ticker:
            names.add(str(r.data.get("company_name") or ""))
        elif r.tool == "peer_comparison":
            names.update(str(row.get("name") or "") for row in r.data.get("matrix", [])
                         if str(row.get("ticker", "")).upper() == ticker)
    return {_COMPANY_SUFFIXES.sub("", n).strip().lower() for n in names if n}


#: Tickers that are also everyday words ("ALL" Allstate, "ON", "NOW"): matched only when the query
#: writes them in capitals, so "all time highs" or "what is on the menu" are not taken for a company.
_WORD_TICKERS = frozenset({"a", "all", "are", "now", "on", "one", "it", "so", "can", "for", "has", "see", "well",
                           "real", "key", "big", "cost", "post", "best", "good", "love", "pay", "run", "fast"})


def _mentions(query: str, ticker: str, names: set[str]) -> bool:
    q = query.lower()
    boundary = r"(?<![\w-])" + re.escape(ticker) + r"(?![\w-])"  # "F-150" is not the ticker F
    if len(ticker) <= 2 or ticker.lower() in _WORD_TICKERS:
        if re.search(boundary, query):  # case-sensitive: tickers are upper case
            return True
    elif re.search(boundary, query, re.IGNORECASE):
        return True
    stripped = q.strip()
    for n in names:
        if not n:
            continue
        if re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", q):
            return True
        if len(stripped) >= 4 and re.search(r"(?<!\w)" + re.escape(stripped) + r"(?!\w)", n):
            return True
    return False


def _news_for(evidence: dict[str, ToolResult], ticker: str) -> dict[str, Any] | None:
    """Aggregate sentiment (article-weighted) over news_sentiment results about this company."""
    names = _company_names(evidence, ticker)
    hits = []
    for r in evidence.values():
        if r.tool != "news_sentiment":
            continue
        query = str(r.data.get("query") or r.args.get("query") or "")
        overall = r.data.get("overall")
        if overall and _mentions(query, ticker, names):
            count = int(r.data.get("article_count") or len(r.data.get("articles", [])) or 0)
            hits.append((r.evidence_id, float(overall.get("score", 0.0)), str(overall.get("label", "")), count))
    if not hits:
        return None
    total = sum(h[3] for h in hits)
    if len(hits) == 1:
        score, label = hits[0][1], hits[0][2] or sentiment_label(hits[0][1])
    else:
        score = sum(h[1] * (h[3] or 1) for h in hits) / (total or len(hits))
        label = sentiment_label(score)
    return {"score": score, "label": label, "articles": total, "ids": [h[0] for h in hits]}


# --------------------------------------------------------------------------------------------------
# Threads
# --------------------------------------------------------------------------------------------------


def sentiment_fact_alignment(ledger: Ledger, evidence: dict[str, ToolResult]) -> list[Thread]:
    """A thread per entity whose news tone points the opposite way to its reported fundamentals."""
    threads: list[Thread] = []
    for entity in _entities(ledger):
        news = _news_for(evidence, entity)
        growth = _growth(ledger, entity)
        income = _net_income_direction(ledger, entity)
        if not news or news["label"] not in ("positive", "negative") or (growth is None and income is None):
            continue
        # Revenue is the primary fundamental; net income decides only when revenue is unavailable.
        if growth is not None:
            rate, year, fids = growth
            direction = "positive" if rate > 0 else "negative" if rate < 0 else "neutral"
            fact = f"revenue {'grew' if rate > 0 else 'fell'} {abs(rate) * 100:.1f}% in FY{year}"
        else:
            assert income is not None
            verb, _figure, year, fids = income
            direction = "positive" if verb in ("grew", "was positive") else "negative"
            fact = f"net income {verb} in FY{year}"
        if direction == "neutral" or direction == news["label"]:
            continue
        extra = ""
        if growth is not None and income is not None:
            verb, figure, _y, ids2 = income
            fids = fids + [i for i in ids2 if i not in fids]
            extra = (f", and net income {verb} {abs(figure) * 100:.1f}%" if verb in ("grew", "fell")
                     else f", and net income {verb}")
        count = f" across {news['articles']} articles" if news["articles"] else ""
        thesis = (f"News sentiment is {news['label']} ({news['score']:+.2f}{count}) while {fact}{extra} "
                  f"{_cite(fids)}; sentiment from {_cite(news['ids'])}.")
        so_what = ("Do not repeat the headline tone as if it were the fundamentals. Explain the gap: valuation "
                   "and expectations already priced in, one-off events, or timing differences between the "
                   "news window and the reporting period.")
        threads.append(Thread(
            theme="sentiment_vs_fundamentals", entity=entity, thesis=thesis, supporting=fids,
            contradicting=list(news["ids"]), so_what=so_what,
            source_types={"news_sentiment"} | _tools(evidence, fids)))
    return threads


def _growth_thread(ledger: Ledger, evidence: dict[str, ToolResult], entity: str) -> Thread | None:
    run = _consecutive(_annual(ledger, entity, "revenue"))
    if len(run) < 2:
        return None
    chron = list(reversed(run))
    yoy = [(b.period.fiscal_year, _f(b) / _f(a) - 1, b) for a, b in zip(chron, chron[1:], strict=False) if _f(a)]
    if not yoy:
        return None
    overall = _f(chron[-1]) - _f(chron[0])
    contra = [b for _y, g, b in yoy if (g < 0) != (overall < 0)]
    path = " -> ".join(f"{format_money(_f(c))} (FY{c.period.fiscal_year})" for c in chron)
    latest_year, latest_g, _ = yoy[-1]
    thesis = (f"{entity} revenue {'rose' if overall >= 0 else 'fell'} over {len(chron)} fiscal years: {path}; "
              f"latest-year growth {latest_g * 100:+.1f}% in FY{latest_year} {_cite(_ids(*run))}.")
    so_what = "Explain what drove the trend and whether the latest pace is sustainable."
    supporting = _ids(*[c for c in chron if c not in contra])
    return Thread("growth", entity, thesis, supporting, _ids(*contra), so_what, _tools(evidence, _ids(*run)))


def _profitability_thread(ledger: Ledger, evidence: dict[str, ToolResult], entity: str) -> Thread | None:
    for metric in ("operating_margin", "net_margin", "gross_margin"):
        run = _consecutive(_annual(ledger, entity, metric))
        if len(run) >= 2:
            break
    else:
        return None
    name = metric.replace("_", " ")
    latest, prior = run[0], run[1]
    delta = (_f(latest) - _f(prior)) * 100
    verb = "expanded" if delta > 0 else "contracted" if delta < 0 else "was flat"
    others = []
    for m, n in (("gross_margin", "gross"), ("net_margin", "net")):
        c = ledger.get(entity, m, latest.period.fiscal_year)
        if c and m != metric:
            others.append(f"{n} {format_value(c.value, c.unit)}")
    tail = f" (also {', '.join(others)})" if others else ""
    thesis = (f"{entity} {name} {verb} from {format_value(prior.value, 'ratio')} in FY{prior.period.fiscal_year} "
              f"to {format_value(latest.value, 'ratio')} in FY{latest.period.fiscal_year} "
              f"({delta:+.1f} percentage points){tail} {_cite(_ids(latest, prior))}.")
    so_what = "Say whether margin change comes from pricing and mix or from cost control, and how durable it is."
    return Thread("profitability", entity, thesis, _ids(latest, prior), [], so_what,
                  _tools(evidence, _ids(latest, prior)))


def _competition_thread(evidence: dict[str, ToolResult], entity: str) -> Thread | None:
    for r in evidence.values():
        if r.tool != "peer_comparison" or str(r.data.get("ticker", "")).upper() != entity:
            continue
        matrix = r.data.get("matrix", [])
        if len(matrix) < 2:
            continue
        parts, weak = [], []
        for metric, name in (("revenue", "revenue"), ("operating_margin", "operating margin"),
                             ("revenue_growth", "revenue growth")):
            scored = [(row["ticker"].upper(), row["values"].get(metric)) for row in matrix
                      if row.get("values", {}).get(metric) is not None]
            order = [t for t, _ in sorted(scored, key=lambda x: -x[1])]
            if entity in order:
                rank = order.index(entity) + 1
                parts.append(f"{rank} of {len(order)} on {name}")
                if rank > len(order) / 2:
                    weak.append(name)
        if not parts:
            continue
        peers = [row["ticker"] for row in matrix if row["ticker"].upper() != entity]
        thesis = f"Against peers ({', '.join(peers)}), {entity} ranks {', '.join(parts)} {_cite([r.evidence_id])}."
        so_what = "Explain what sets the company apart from, or holds it behind, its closest competitors."
        return Thread("competition", entity, thesis, [r.evidence_id], [r.evidence_id] if weak else [], so_what,
                      {"peer_comparison"})
    return None


def _capital_thread(ledger: Ledger, evidence: dict[str, ToolResult], entity: str) -> Thread | None:
    fcf = _annual(ledger, entity, "free_cash_flow")
    if not fcf:
        return None
    latest = fcf[0]
    capex = ledger.get(entity, "capital_expenditure", latest.period.fiscal_year)
    ocf = ledger.get(entity, "operating_cash_flow", latest.period.fiscal_year)
    text = f"{entity} generated free cash flow of {format_money(_f(latest))} in FY{latest.period.fiscal_year}"
    if capex:
        text += f" after {format_money(_f(capex))} of capital expenditure"
        if ocf and _f(ocf):
            text += f" ({_f(capex) / _f(ocf) * 100:.1f}% of operating cash flow)"
    prior = _consecutive(fcf)
    if len(prior) >= 2 and _f(prior[1]) > 0:
        text += f"; free cash flow changed {(_f(latest) / _f(prior[1]) - 1) * 100:+.1f}% year on year"
    elif len(prior) >= 2:  # a percentage change from a zero or negative base is meaningless
        text += f"; free cash flow moved from {format_money(_f(prior[1]))} the year before"
    ids = _ids(latest, capex, ocf, *(prior[1:2]))
    so_what = "Explain how the cash is being deployed (reinvestment, buybacks, dividends, balance sheet)."
    return Thread("capital_allocation", entity, f"{text} {_cite(ids)}.", ids, [], so_what, _tools(evidence, ids))


def build_threads(ledger: Ledger, evidence: dict[str, ToolResult]) -> list[Thread]:
    """All threads whose data exists, sentiment-vs-fundamentals first."""
    threads = sentiment_fact_alignment(ledger, evidence)
    for entity in _entities(ledger):
        for t in (_growth_thread(ledger, evidence, entity), _profitability_thread(ledger, evidence, entity),
                  _competition_thread(evidence, entity), _capital_thread(ledger, evidence, entity)):
            if t:
                threads.append(t)
    return threads


# --------------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------------

_HEADER = "SYNTHESIS BRIEF (resolved facts; cite the evidence ids)"


def _item(c: Claim, label: str) -> str:
    flag = ", conflicted" if c.status == "conflicted" else ""
    return f"{label} {format_value(c.value, c.unit)} {_cite(c.evidence_ids)} (conf {c.confidence:.2f}{flag})"


def _figure_lines(ledger: Ledger) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    for entity in _entities(ledger):
        years = sorted({c.period.fiscal_year for c in ledger.for_entity(entity)
                        if c.period.fiscal_period == "FY" and c.period.fiscal_year is not None}, reverse=True)[:2]
        for rank, year in enumerate(years):
            for metrics, prio in ((("revenue", "net_income"), 80 - 30 * rank),
                                  (("gross_margin", "operating_margin", "net_margin"), 70 - 30 * rank)):
                items = []
                for m in metrics:
                    c = ledger.get(entity, m, year)
                    if c:
                        items.append(_item(c, m.replace("_", " ")))
                if items:
                    lines.append((prio, f"- {entity} FY{year}: " + "; ".join(items)))
    return lines


def _conflict_line(c: ConflictRecord) -> str:
    vals = "; ".join(f"{format_value(v['value'], c.unit)} [{v['evidence_id']}] tier {v['tier']}" for v in c.values)
    return (f"- {c.entity} {c.metric} {c.period.label()}: chose {format_value(c.chosen_value, c.unit)} via "
            f"{c.method} (conf {c.confidence:.2f}); sources: {vals}")


def render_brief(ledger: Ledger, threads: list[Thread], max_tokens: int = 900) -> str:
    """Compact writer-prompt block; lowest-priority lines are dropped until it fits ``max_tokens``."""
    sections: list[tuple[str, list[tuple[int, str]]]] = [
        ("Key figures:", _figure_lines(ledger)),
        ("Conflicts resolved:", [(90, _conflict_line(c)) for c in ledger.conflicts]),
        ("Narrative threads:", [(95 if t.theme == "sentiment_vs_fundamentals" else 60,
                                 f"- [{t.theme}] {t.thesis} So what: {t.so_what}") for t in threads]),
    ]

    def render() -> str:
        out = [_HEADER]
        for title, lines in sections:
            if lines:
                out.append(title)
                out.extend(text for _p, text in lines)
        return "\n".join(out)

    if not any(lines for _t, lines in sections):
        return ""  # nothing resolved: a header-only brief would only mislead the writer
    text = render()
    while count_tokens(text) > max_tokens:
        candidates = [(p, si, li) for si, (_t, lines) in enumerate(sections) for li, (p, _x) in enumerate(lines)]
        if not candidates:
            break
        # Lowest priority first; among equals, drop the one that appears last.
        _p, si, li = min(candidates, key=lambda x: (x[0], -x[1], -x[2]))
        del sections[si][1][li]
        text = render()
    return text if any(lines for _t, lines in sections) else ""


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def conflicts_markdown(conflicts: list[ConflictRecord]) -> str:
    """Markdown table for the report's conflicts section; empty string when there is nothing to show."""
    if not conflicts:
        return ""
    rows = ["| Entity | Metric | Period | Values (source) | Resolution | Confidence |",
            "|---|---|---|---|---|---|"]
    for c in conflicts:
        vals = "; ".join(f"{format_value(v['value'], c.unit)} ({v['provider']}, tier {v['tier']}, "
                         f"{v['evidence_id']})" for v in c.values)
        resolution = f"{c.method}: {format_value(c.chosen_value, c.unit)}. {c.rationale}"
        rows.append(f"| {_cell(c.entity)} | {_cell(c.metric)} | {_cell(c.period.label())} | {_cell(vals)} | "
                    f"{_cell(resolution)} | {c.confidence:.2f} |")
    return "\n".join(rows)
