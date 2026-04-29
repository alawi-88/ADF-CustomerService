"""Recommendation snapshots — persistent audit log of every insight run.

Designed to plug straight into the existing SQLite database used by
`src/tickets.py`. Same DB file (`data/processed/tickets.db`), same
connection pattern, same threading discipline.

A *snapshot* is the full output of an insights generation event:
  - the filter set used (date range, categories, severities, etc.)
  - the LLM provider + model that produced it
  - the structured insights JSON (recommendations the dashboard shows)
  - the per-ticket recommended actions associated with this run
  - free-text notes a manager can add when they "lock" a snapshot

Every uploaded data refresh creates a new snapshot. Past snapshots are
read-only after they have been locked. Two snapshots can be diffed
side-by-side so management can answer "what changed between April 15
and April 22, and why?".

Public API:
    init_db()                              -- idempotent; creates tables
    create_snapshot(payload) -> int        -- writes a new run, returns id
    list_snapshots(limit=50) -> list[Row]  -- newest first
    get_snapshot(snapshot_id) -> dict      -- full snapshot incl. items
    lock_snapshot(snapshot_id, by_id, note) -> None
    diff_snapshots(a_id, b_id) -> dict     -- structural diff
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "tickets.db"

_lock = threading.Lock()


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON;")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db() -> None:
    """Idempotent — safe to call on every process start."""
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS recommendation_runs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at     REAL NOT NULL,
                created_by_id  TEXT,
                trigger        TEXT NOT NULL,            -- 'upload' | 'manual' | 'scheduled'
                source_file    TEXT,                     -- xlsx filename if trigger='upload'
                row_count      INTEGER NOT NULL,
                date_from      TEXT,
                date_to        TEXT,
                filters_json   TEXT NOT NULL,            -- full filter set
                provider       TEXT NOT NULL,            -- 'groq' | 'ollama' | 'rule'
                model          TEXT,                     -- e.g. 'llama-3.3-70b-versatile'
                prompt_version TEXT NOT NULL,            -- e.g. 'enrich_v1'
                language       TEXT NOT NULL,            -- 'ar' | 'en'
                insights_json  TEXT NOT NULL,            -- full insights payload
                kpis_json      TEXT NOT NULL,            -- KPIs at snapshot time
                forecast_json  TEXT,                     -- forecast at snapshot time
                locked         INTEGER NOT NULL DEFAULT 0,
                locked_at      REAL,
                locked_by_id   TEXT,
                manager_note   TEXT
            );

            CREATE TABLE IF NOT EXISTS recommendation_items (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id         INTEGER NOT NULL,
                request_id     INTEGER,                  -- nullable for cluster-level items
                cluster_id     TEXT,                     -- nullable for ticket-level items
                kind           TEXT NOT NULL,            -- 'ticket' | 'cluster' | 'topic'
                category       TEXT,
                subcategory    TEXT,                     -- new sub-taxonomy (see prompts)
                severity_ai    TEXT,
                severity_reason TEXT,
                topic_label    TEXT,
                action         TEXT,
                evidence       TEXT,                     -- short quoted body / cluster sample
                FOREIGN KEY (run_id) REFERENCES recommendation_runs(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS ix_rec_items_run ON recommendation_items(run_id);
            CREATE INDEX IF NOT EXISTS ix_rec_items_request ON recommendation_items(request_id);
            CREATE INDEX IF NOT EXISTS ix_rec_runs_created ON recommendation_runs(created_at);
            """
        )


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------


def create_snapshot(payload: dict[str, Any]) -> int:
    """Persist a full recommendation run.

    payload must include:
      trigger, row_count, filters, provider, prompt_version, language,
      insights, kpis
    Optional:
      created_by_id, source_file, date_from, date_to, model, forecast,
      items (list of dicts with kind, request_id?, cluster_id?, ...)

    Returns the new run id.
    """
    items = payload.get("items") or []
    now = time.time()
    with _lock, _conn() as c:
        cur = c.execute(
            """
            INSERT INTO recommendation_runs (
                created_at, created_by_id, trigger, source_file, row_count,
                date_from, date_to, filters_json, provider, model,
                prompt_version, language, insights_json, kpis_json, forecast_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now,
                payload.get("created_by_id"),
                payload["trigger"],
                payload.get("source_file"),
                int(payload["row_count"]),
                payload.get("date_from"),
                payload.get("date_to"),
                json.dumps(payload.get("filters") or {}, ensure_ascii=False),
                payload["provider"],
                payload.get("model"),
                payload["prompt_version"],
                payload["language"],
                json.dumps(payload["insights"], ensure_ascii=False),
                json.dumps(payload["kpis"], ensure_ascii=False),
                json.dumps(payload.get("forecast")) if payload.get("forecast") else None,
            ),
        )
        run_id = int(cur.lastrowid)
        if items:
            c.executemany(
                """
                INSERT INTO recommendation_items (
                    run_id, request_id, cluster_id, kind, category, subcategory,
                    severity_ai, severity_reason, topic_label, action, evidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        it.get("request_id"),
                        it.get("cluster_id"),
                        it["kind"],
                        it.get("category"),
                        it.get("subcategory"),
                        it.get("severity_ai"),
                        it.get("severity_reason"),
                        it.get("topic_label"),
                        it.get("action"),
                        it.get("evidence"),
                    )
                    for it in items
                ],
            )
        return run_id


def lock_snapshot(snapshot_id: int, by_id: str, note: str = "") -> None:
    """Make a snapshot read-only and stamp it with the locking manager."""
    with _lock, _conn() as c:
        c.execute(
            """
            UPDATE recommendation_runs
            SET locked = 1, locked_at = ?, locked_by_id = ?, manager_note = ?
            WHERE id = ? AND locked = 0
            """,
            (time.time(), by_id, note, int(snapshot_id)),
        )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def list_snapshots(limit: int = 50) -> list[dict[str, Any]]:
    with _conn() as c:
        rows = c.execute(
            """
            SELECT id, created_at, trigger, source_file, row_count,
                   date_from, date_to, provider, model, prompt_version,
                   language, locked, locked_at, locked_by_id, manager_note
            FROM recommendation_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]


def get_snapshot(snapshot_id: int) -> dict[str, Any] | None:
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM recommendation_runs WHERE id = ?", (int(snapshot_id),)
        ).fetchone()
        if not row:
            return None
        items = c.execute(
            "SELECT * FROM recommendation_items WHERE run_id = ?", (int(snapshot_id),)
        ).fetchall()
        d = dict(row)
        d["filters"] = json.loads(d.pop("filters_json") or "{}")
        d["insights"] = json.loads(d.pop("insights_json") or "{}")
        d["kpis"] = json.loads(d.pop("kpis_json") or "{}")
        fcst = d.pop("forecast_json", None)
        d["forecast"] = json.loads(fcst) if fcst else None
        d["items"] = [dict(i) for i in items]
        return d


def diff_snapshots(a_id: int, b_id: int) -> dict[str, Any]:
    """Compute a structural diff between two snapshots.

    Returns:
      {
        "kpi_deltas":    { metric: {a, b, delta, pct_change} },
        "insights_added":   [insight_dicts in b but not a],
        "insights_removed": [insight_dicts in a but not b],
        "tickets_severity_changed": [
            {request_id, a_sev, b_sev, a_reason, b_reason}, ...
        ],
        "meta": {
            "a": { id, created_at, provider, model, prompt_version, locked },
            "b": { ... }
        }
      }
    """
    a = get_snapshot(a_id)
    b = get_snapshot(b_id)
    if a is None or b is None:
        raise ValueError("One or both snapshots not found")

    # KPI deltas — operate on flat numeric KPIs
    a_kpis = {k: v for k, v in (a["kpis"] or {}).items() if isinstance(v, (int, float))}
    b_kpis = {k: v for k, v in (b["kpis"] or {}).items() if isinstance(v, (int, float))}
    kpi_deltas: dict[str, Any] = {}
    for k in sorted(set(a_kpis) | set(b_kpis)):
        av, bv = a_kpis.get(k), b_kpis.get(k)
        if av is None or bv is None:
            kpi_deltas[k] = {"a": av, "b": bv, "delta": None, "pct_change": None}
        else:
            delta = bv - av
            pct = (delta / av * 100.0) if av else None
            kpi_deltas[k] = {"a": av, "b": bv, "delta": delta, "pct_change": pct}

    # Insights diff — match by 'id' field if present, else by 'title'
    def _key(ins: dict) -> str:
        return str(ins.get("id") or ins.get("title") or json.dumps(ins, sort_keys=True))

    a_ins = (a["insights"] or {}).get("items") or []
    b_ins = (b["insights"] or {}).get("items") or []
    a_keys = {_key(i): i for i in a_ins}
    b_keys = {_key(i): i for i in b_ins}
    insights_added = [b_keys[k] for k in b_keys if k not in a_keys]
    insights_removed = [a_keys[k] for k in a_keys if k not in b_keys]

    # Per-ticket severity changes — join items by request_id
    a_items = {it["request_id"]: it for it in a["items"] if it.get("request_id")}
    b_items = {it["request_id"]: it for it in b["items"] if it.get("request_id")}
    sev_changes = []
    for rid in sorted(set(a_items) & set(b_items)):
        if a_items[rid].get("severity_ai") != b_items[rid].get("severity_ai"):
            sev_changes.append(
                {
                    "request_id": rid,
                    "a_sev": a_items[rid].get("severity_ai"),
                    "b_sev": b_items[rid].get("severity_ai"),
                    "a_reason": a_items[rid].get("severity_reason"),
                    "b_reason": b_items[rid].get("severity_reason"),
                }
            )

    return {
        "kpi_deltas": kpi_deltas,
        "insights_added": insights_added,
        "insights_removed": insights_removed,
        "tickets_severity_changed": sev_changes,
        "meta": {
            "a": {k: a[k] for k in ("id", "created_at", "provider", "model", "prompt_version", "locked")},
            "b": {k: b[k] for k in ("id", "created_at", "provider", "model", "prompt_version", "locked")},
        },
    }


# Module-import side-effect — same pattern as src/tickets.py
init_db()
