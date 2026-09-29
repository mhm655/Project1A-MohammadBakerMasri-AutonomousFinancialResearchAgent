"""Structured facts store: financial statement line items in SQLite (architecture spec, section 10.1).

Spec A3.3: financial statements should be stored as structured data, "using metadata filtering rather
than semantic search for numerical data retrieval". Asking a vector database for "Microsoft FY2025
revenue" can return FY2024. A SQL lookup on ``(ticker, period_end, item)`` cannot.

One row per ``(ticker, period, period_end, item)``. Storing a statement again replaces its values, so
restated figures from a newer filing overwrite older ones.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    ticker TEXT NOT NULL,
    period TEXT NOT NULL,          -- annual | quarterly
    period_end TEXT NOT NULL,
    fiscal_year INTEGER,
    fiscal_period TEXT,
    item TEXT NOT NULL,            -- revenue, net_income, gross_margin, ...
    value REAL,
    unit TEXT,                     -- millions | ratio | per_share
    provider TEXT,                 -- sec_xbrl | yfinance | mock
    evidence_id TEXT,
    session TEXT,
    stored_at TEXT,
    PRIMARY KEY (ticker, period, period_end, item)
)
"""


class FactsStore:
    def __init__(self, path: str | Path | None = None) -> None:
        """``path=None`` keeps the store in memory (tests and mock runs)."""
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path) if path else ":memory:", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(_SCHEMA)
            self._db.commit()

    def store_statement(self, data: dict[str, Any], *, evidence_id: str = "", session: str = "") -> int:
        """Store every value of a ``financial_data_api`` result; returns the number of values written."""
        ticker, period = data["ticker"].upper(), data.get("period", "annual")
        unit = data.get("unit", "millions")
        provider = data.get("provider") or ("mock" if any("MOCK" in n for n in data.get("notes", [])) else "")
        now = datetime.now(UTC).isoformat(timespec="seconds")
        rows = []
        for row in data.get("data", []):
            for item, value in row["values"].items():
                item_unit = "per_share" if item == "eps_diluted" else unit
                rows.append((ticker, period, row["period_end"], row.get("fiscal_year"), row.get("fiscal_period"),
                             item, value, item_unit, provider, evidence_id, session, now))
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            self._db.commit()
        return len(rows)

    def query(self, ticker: str, *, items: list[str] | None = None, period: str = "annual",
              limit_periods: int = 5) -> list[dict[str, Any]]:
        """The latest ``limit_periods`` periods for a ticker, newest first."""
        # The items filter applies inside the period subquery too: otherwise a newer period that lacks
        # the requested items uses up the limit and the periods that have them are cut off.
        item_filter = f" AND item IN ({','.join('?' * len(items))})" if items else ""
        sql = (f"SELECT * FROM facts WHERE ticker = ? AND period = ?{item_filter} AND period_end IN "
               f"(SELECT DISTINCT period_end FROM facts WHERE ticker = ? AND period = ?{item_filter} "
               "ORDER BY period_end DESC LIMIT ?)")
        params: list[Any] = [ticker.upper(), period, *(items or []), ticker.upper(), period, *(items or []),
                             limit_periods]
        with self._lock:
            return [dict(r) for r in self._db.execute(sql + " ORDER BY period_end DESC, item", params)]

    def value(self, ticker: str, item: str, period_end: str, period: str = "annual") -> float | None:
        with self._lock:
            row = self._db.execute("SELECT value FROM facts WHERE ticker=? AND item=? AND period_end=? AND period=?",
                                   (ticker.upper(), item, period_end, period)).fetchone()
        return row["value"] if row else None

    def summary(self, ticker: str) -> str | None:
        """A one-line description of what is stored for a ticker, for the planner."""
        with self._lock:
            rows = self._db.execute(
                "SELECT period, MIN(fiscal_year) lo, MAX(fiscal_year) hi, COUNT(DISTINCT item) items, "
                "MAX(stored_at) stored, GROUP_CONCAT(DISTINCT provider) providers FROM facts "
                "WHERE ticker = ? GROUP BY period", (ticker.upper(),)).fetchall()
        if not rows:
            return None
        parts = [f"{r['period']} FY{r['lo']}-FY{r['hi']} ({r['items']} line items, {r['providers']}, "
                 f"stored {r['stored'][:10]})" for r in rows]
        return f"{ticker.upper()} financial statements in memory: " + "; ".join(parts)

    def tickers(self) -> list[str]:
        with self._lock:
            return [r[0] for r in self._db.execute("SELECT DISTINCT ticker FROM facts ORDER BY ticker")]
