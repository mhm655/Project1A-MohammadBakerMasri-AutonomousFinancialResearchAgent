"""Render the evaluation scores as a self-contained HTML dashboard and a Markdown report (spec A5.2, Day 11).

Everything here is a pure function of the ``scores`` dict written by ``evaluation.evaluate`` (no imports of
metrics/judge), so both files can be re-rendered from ``scores.json`` alone:

    python -m evaluation.dashboard results/eval/scores.json
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any

CATEGORIES = ("Factual Accuracy", "Completeness", "Analytical Depth", "Coherence and Structure",
              "Agent Behaviour", "Benchmark")
ANALYSIS_MARKER = "<!-- ANALYSIS -->"
ANALYSIS_PLACEHOLDER = "_To be written after review._"
TITLE = "ARA-1 Evaluation Dashboard"

CSS = """
:root{--bg:#f6f7f9;--fg:#1b1f24;--muted:#6a737d;--card:#fff;--line:#d8dde3;--pass-bg:#dcf3e2;--pass-fg:#14602b;
--fail-bg:#fbdcdc;--fail-fg:#8f1d1d;--na-bg:#eceef1;--na-fg:#6a737d;--head:#e9edf2}
@media (prefers-color-scheme:dark){:root{--bg:#14171b;--fg:#e6e8eb;--muted:#9aa4af;--card:#1d2227;--line:#343b43;
--pass-bg:#173a25;--pass-fg:#7fdc9f;--fail-bg:#441c1c;--fail-fg:#ff9b9b;--na-bg:#262c33;--na-fg:#8b95a0;
--head:#262c33}}
*{box-sizing:border-box}
body{margin:0;padding:24px;background:var(--bg);color:var(--fg);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
h1{margin:0 0 4px;font-size:24px}h2{margin:32px 0 12px;font-size:18px}
.sub{color:var(--muted);margin:0 0 20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
.tile .v{font-size:26px;font-weight:650}.tile .l{color:var(--muted);font-size:12px}
.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;min-width:980px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}
thead th{background:var(--head)}
tr.cat td{background:var(--head);font-weight:650}
td.c,th.c{text-align:center;white-space:nowrap;font-variant-numeric:tabular-nums;min-width:72px}
td.pass{background:var(--pass-bg);color:var(--pass-fg)}td.fail{background:var(--fail-bg);color:var(--fail-fg)}
td.na{background:var(--na-bg);color:var(--na-fg)}
.badge{display:inline-block;padding:1px 7px;border-radius:10px;border:1px solid var(--line);font-size:11px;color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px}
.card h3{margin:0 0 6px;font-size:15px}.card p{margin:4px 0}.facts{color:var(--muted);font-size:12px}
.card ul{margin:6px 0 0;padding-left:18px}.ex{color:var(--muted);font-size:12px}
.err{color:var(--fail-fg)}
code{font-family:ui-monospace,Consolas,monospace;font-size:12px}
"""


# --------------------------------------------------------------------------- formatting helpers
def fmt_value(value: Any, unit: str) -> str:
    """Format a metric value for a matrix cell; ``None`` (not measured) is an em dash."""
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if unit == "ratio":
        pct = v * 100
        if 99.5 <= pct < 100:  # 99.5-99.99% must not read as a (failed) 100%
            return f"{math.floor(pct * 10) / 10:.1f}%"
        return f"{pct:.0f}%"
    if unit == "score_1_5":
        return f"{v:.1f}/5".replace(".0/5", "/5")
    if unit == "seconds":
        return f"{v:.0f} s"
    if unit == "per_page":
        return f"{v:.1f}/pg"
    return f"{v:.0f}"


def _state(metric: dict[str, Any] | None) -> str:
    if not metric or metric.get("value") is None or metric.get("passed") is None:
        return "na"
    return "pass" if metric["passed"] else "fail"


def _cell_text(metric: dict[str, Any] | None) -> str:
    """Value plus a glyph, so colour is never the only signal; unmeasured is a bare dash."""
    if metric is None or metric.get("value") is None:
        return "—"
    glyph = {"pass": "✓", "fail": "✗"}.get(_state(metric), "")
    return f"{fmt_value(metric.get('value'), metric.get('unit', ''))} {glyph}".strip()


def _rows(scores: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Metric definitions grouped by category, in canonical order (first-seen order inside a category)."""
    seen: dict[str, dict[str, Any]] = {}
    for run in scores.get("runs", []):
        for m in run.get("metrics", []):
            first = seen.setdefault(m["id"], m)
            if m.get("target") and first.get("target") and m["target"] not in first["target"].split(" / "):
                # AB-5 has a 300 s or 600 s limit per run: show every target in use, not only the first run's.
                seen[m["id"]] = {**first, "target": f"{first['target']} / {m['target']}"}
    for mid, a in scores.get("aggregate", {}).items():
        seen.setdefault(mid, {"id": mid, **{k: a.get(k) for k in ("name", "category", "target", "unit")}})
    cats = list(CATEGORIES) + sorted({m.get("category") for m in seen.values()} - set(CATEGORIES))
    return [(c, [m for m in seen.values() if m.get("category") == c]) for c in cats
            if any(m.get("category") == c for m in seen.values())]


def _by_id(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {m["id"]: m for m in run.get("metrics", [])}


def _pct(passed: int, measured: int) -> str:
    return f"{100 * passed / measured:.0f}%" if measured else "—"


def _mean_latency(scores: dict[str, Any]) -> float | None:
    xs = [r.get("duration_s") for r in scores.get("runs", []) if r.get("duration_s") is not None]
    return sum(xs) / len(xs) if xs else None


def _total_tokens(scores: dict[str, Any]) -> int:
    return int(sum(r.get("tokens") or 0 for r in scores.get("runs", [])))


def _judge_status(scores: dict[str, Any]) -> str:
    j = scores.get("judge", {})
    if not j.get("enabled"):
        return "disabled (judge metrics not measured)"
    n = len(j.get("errors", {}))
    return "enabled" + (f", {n} error(s)" if n else "")


def _failing(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in run.get("metrics", []) if _state(m) == "fail"]


def _facts(run: dict[str, Any]) -> str:
    parts = [f"stop: {run.get('stop_reason')}", f"{run.get('duration_s') or 0:.0f} s",
             f"{run.get('tool_calls', 0)} tool calls", f"{run.get('llm_calls', 0)} LLM calls",
             f"{run.get('tokens') or 0:,} tokens"]
    if run.get("injected_failures"):
        parts.append("injected failures: " + ", ".join(f"{k}={v}" for k, v in run["injected_failures"].items()))
    return " · ".join(parts)


def _modes(runs: list[dict[str, Any]]) -> str:
    return ", ".join(sorted({str(r.get("data_mode")) for r in runs})) or "n/a"


# --------------------------------------------------------------------------- HTML
def render_html(scores: dict[str, Any]) -> str:
    """Render the evaluation scores as a self-contained HTML dashboard."""
    esc = html.escape
    runs = scores.get("runs", [])
    summary = scores.get("summary", {})
    out: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{TITLE}</title><style>{CSS}</style></head><body>",
        f"<h1>{TITLE}</h1>",
        f"<p class='sub'>Generated {esc(str(scores.get('generated_at', '')))} · data mode: "
        f"{esc(_modes(runs))} · LLM judge: {esc(_judge_status(scores))}</p>",
    ]
    lat = _mean_latency(scores)
    tiles = [(f"{100 * summary.get('pass_rate', 0):.0f}%", "Overall pass rate")]
    for cat, c in summary.get("by_category", {}).items():
        tiles.append((_pct(c.get("passed", 0), c.get("measured", 0)),
                      f"{cat} ({c.get('passed', 0)}/{c.get('measured', 0)})"))
    tiles.append((f"{_total_tokens(scores):,}", "Total LLM tokens"))
    tiles.append((f"{lat:.0f} s" if lat is not None else "—", "Mean latency"))
    out.append("<div class='tiles'>" + "".join(
        f"<div class='tile'><div class='v'>{esc(v)}</div><div class='l'>{esc(label)}</div></div>"
        for v, label in tiles) + "</div>")

    out.append("<h2>Metric matrix</h2><div class='wrap'><table><thead><tr><th>ID</th><th>Metric</th>"
               "<th>Target</th><th>Method</th>")
    out.extend(f"<th class='c' title='{esc(str(r.get('title', '')), quote=True)}'>{esc(str(r['id']))}</th>"
               for r in runs)
    out.append("</tr></thead><tbody>")
    by_run = [_by_id(r) for r in runs]
    for cat, metrics in _rows(scores):
        out.append(f"<tr class='cat'><td colspan='{4 + len(runs)}'>{esc(cat)}</td></tr>")
        for m in metrics:
            out.append(f"<tr><td style='white-space:nowrap'><code>{esc(m['id'])}</code></td><td>{esc(str(m.get('name', '')))}</td>"
                       f"<td>{esc(str(m.get('target', '')))}</td>"
                       f"<td><span class='badge'>{esc(str(m.get('method', '')))}</span></td>")
            for b in by_run:
                cell = b.get(m["id"])
                tip = ""
                if cell:
                    tip = str(cell.get("detail", ""))
                    if cell.get("examples"):
                        tip += "\nExample: " + str(cell["examples"][0])
                out.append(f"<td class='c {_state(cell)}' title='{esc(tip, quote=True)}'>"
                           f"{esc(_cell_text(cell))}</td>")
            out.append("</tr>")
    out.append("</tbody></table></div>")

    out.append("<h2>Per-challenge results</h2><div class='cards'>")
    for r in runs:
        measured = [m for m in r.get("metrics", []) if _state(m) != "na"]
        passed = sum(1 for m in measured if m.get("passed"))
        out.append(f"<div class='card'><h3>{esc(str(r['id']))} — {esc(str(r.get('title', '')))} "
                   f"<span class='badge'>{passed}/{len(measured)} passed</span></h3>"
                   f"<p>{esc(str(r.get('query', '')))}</p>"
                   f"<p class='facts'>template: {esc(str(r.get('template')))} · {esc(_facts(r))}</p>")
        fails = _failing(r)
        if fails:
            out.append("<ul>")
            for m in fails:
                out.append(f"<li class='err'>✗ <b>{esc(m['id'])}</b> {esc(str(m.get('name', '')))}: "
                           f"{esc(fmt_value(m.get('value'), m.get('unit', '')))} "
                           f"(target {esc(str(m.get('target', '')))})"
                           f"<br><span class='ex'>{esc(str(m.get('detail', '')))}</span>")
                for ex in (m.get("examples") or [])[:3]:
                    out.append(f"<br><span class='ex'>• {esc(str(ex))}</span>")
                out.append("</li>")
            out.append("</ul>")
        else:
            out.append("<p>No failing metrics.</p>")
        out.append("</div>")
    out.append("</div></body></html>")
    return "".join(out)


# --------------------------------------------------------------------------- Markdown
def _md(text: Any) -> str:
    # CR is a line break in Markdown too, so drop it along with LF (real details are multi-line table rows).
    return str(text).replace("|", "\\|").replace("\r", "").replace("\n", " ")


def render_markdown(scores: dict[str, Any], existing: str | None = None) -> str:
    """Render the evaluation scores as the Markdown report; ``existing`` is the previous report text, if any."""
    runs = scores.get("runs", [])
    summary = scores.get("summary", {})
    lines: list[str] = [
        "# ARA-1 Evaluation Report", "",
        f"Generated: {scores.get('generated_at', '')}  ",
        f"Data mode: {_modes(runs)}  ",
        f"LLM judge: {_judge_status(scores)}", "",
        "Reproduce: `python -m evaluation.runner`, then `python -m evaluation.evaluate`.", "",
        "## Summary", "",
        f"Overall pass rate: **{100 * summary.get('pass_rate', 0):.0f}%** "
        f"({summary.get('metrics_total', 0)} metrics x {len(runs)} challenges). "
        f"Total LLM tokens: {_total_tokens(scores):,}.", "",
        "| Category | Passed | Measured | Rate |", "|---|---|---|---|",
    ]
    for cat, c in summary.get("by_category", {}).items():
        lines.append(f"| {_md(cat)} | {c.get('passed', 0)} | {c.get('measured', 0)} | "
                     f"{_pct(c.get('passed', 0), c.get('measured', 0))} |")
    lines += ["", "## Metric Matrix", "", "Legend: ✓ meets target, ✗ misses target, — not measured.", ""]
    by_run = [_by_id(r) for r in runs]
    for cat, metrics in _rows(scores):
        lines += [f"### {cat}", "", "| ID | Metric | Target | " + " | ".join(str(r["id"]) for r in runs) + " |",
                  "|---|---|---|" + "---|" * len(runs)]
        for m in metrics:
            cells = " | ".join(_md(_cell_text(b.get(m["id"]))) for b in by_run)
            lines.append(f"| {m['id']} | {_md(m.get('name', ''))} | {_md(m.get('target', ''))} | {cells} |")
        lines.append("")
    lines += ["## Per-Challenge Results", ""]
    for r in runs:
        lines += [f"### {r['id']} — {r.get('title', '')}", "", f"Query: {r.get('query', '')}", "",
                  f"Run: template `{r.get('template')}`, {_facts(r)}", ""]
        fails = _failing(r)
        if fails:
            lines.append("Failing metrics:")
            for m in fails:
                lines.append(f"- ✗ **{m['id']}** {m.get('name', '')}: "
                             f"{fmt_value(m.get('value'), m.get('unit', ''))} (target {m.get('target', '')}) — "
                             f"{_md(m.get('detail', ''))}")
        else:
            lines.append("No failing metrics.")
        lines.append("")
    lines += ["## Method", ""]
    for _cat, metrics in _rows(scores):
        for m in metrics:
            lines.append(f"- **{m['id']}** {m.get('name', '')} — method: {m.get('method', '')}; "
                         f"target: {m.get('target', '')}")
    lines += ["", "Note: AB-4 uses memory_hits ÷ external_api_calls; the spec's \"multiplied by\" is a "
              "documented spec error (see ERROR_LOG.md Error 1).", ""]
    head = "\n".join(lines) + "\n"
    if existing and ANALYSIS_MARKER in existing:
        # The lead writes the Analysis by hand; keep everything after the marker verbatim.
        return head + ANALYSIS_MARKER + existing.split(ANALYSIS_MARKER, 1)[1]
    return head + ANALYSIS_MARKER + "\n## Analysis\n\n" + ANALYSIS_PLACEHOLDER + "\n"


def _read_report(path: Path) -> str | None:
    if not path.exists():
        return None
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Windows editors (Notepad) may save the hand-written Analysis as cp1252; decoding it as UTF-8 would
        # crash the re-render, and the Analysis must survive.
        text = raw.decode("cp1252", errors="replace")
    return text.replace("\r\n", "\n")  # text-mode write below re-adds CRs on Windows; avoid CR CR LF


def write_outputs(scores: dict[str, Any], html_path: Path, report_path: Path) -> None:
    """Write both files, preserving a hand-written Analysis section in an existing report."""
    existing = _read_report(report_path)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(render_html(scores), encoding="utf-8")
    report_path.write_text(render_markdown(scores, existing), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: re-render the dashboard and report from a scores.json file."""
    p = argparse.ArgumentParser(description="Re-render the dashboard and report from scores.json")
    p.add_argument("scores", nargs="?", default="results/eval/scores.json")
    p.add_argument("--dashboard", default="results/evaluation_dashboard.html")
    p.add_argument("--report", default="results/evaluation_report.md")
    a = p.parse_args(argv)
    write_outputs(json.loads(Path(a.scores).read_text(encoding="utf-8")), Path(a.dashboard), Path(a.report))
    print(f"wrote {a.dashboard} and {a.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
