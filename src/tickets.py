"""Ticket store — local SQLite-backed CRUD for case management on top of
the AI-classified records. Each request_id from the dataset becomes a
ticket on first interaction; the ticket carries status, assignee,
priority, comments, and a full audit history of status changes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "tickets.db"

# A small hardcoded directory of users — wire to LDAP / IdP in production.
USERS = [
    {"id": "u_layla",  "name": "ليلى محمد",   "name_en": "Layla Mohammed",  "role": "fpoint"},
    {"id": "u_ahmed",  "name": "أحمد العمري",  "name_en": "Ahmed Al-Omari",  "role": "lead"},
    {"id": "u_sara",   "name": "سارة القحطاني","name_en": "Sara Al-Qahtani", "role": "specialist"},
    {"id": "u_khaled", "name": "خالد الشهري", "name_en": "Khaled Al-Shehri","role": "specialist"},
    {"id": "u_nora",   "name": "نورة العتيبي","name_en": "Nora Al-Otaibi",  "role": "manager"},
]

STATUSES = [
    {"id": "open",        "label_ar": "مفتوح",         "label_en": "Open",         "color": "info"},
    {"id": "in_progress", "label_ar": "قيد المعالجة",  "label_en": "In progress",  "color": "warning"},
    {"id": "pending",     "label_ar": "بانتظار رد",    "label_en": "Awaiting reply","color": "warning"},
    {"id": "resolved",    "label_ar": "محلول",         "label_en": "Resolved",     "color": "success"},
    {"id": "closed",      "label_ar": "مغلق",          "label_en": "Closed",       "color": "neutral"},
]
DEFAULT_STATUS = "open"

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
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS tickets (
                request_id  INTEGER PRIMARY KEY,
                status      TEXT NOT NULL DEFAULT 'open',
                assignee_id TEXT,
                priority    TEXT,
                created_at  REAL NOT NULL,
                updated_at  REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS comments (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id INTEGER NOT NULL,
                author_id  TEXT NOT NULL,
                body       TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (request_id) REFERENCES tickets(request_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS status_history (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id  INTEGER NOT NULL,
                from_status TEXT,
                to_status   TEXT NOT NULL,
                by_id       TEXT,
                at          REAL NOT NULL,
                FOREIGN KEY (request_id) REFERENCES tickets(request_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS ix_comments_req ON comments(request_id);
            CREATE INDEX IF NOT EXISTS ix_history_req  ON status_history(request_id);
            CREATE INDEX IF NOT EXISTS ix_tickets_assn ON tickets(assignee_id);
        """)


def _ensure_ticket(c: sqlite3.Connection, request_id: int) -> sqlite3.Row:
    """Create a ticket row for this request_id if missing, return it."""
    row = c.execute("SELECT * FROM tickets WHERE request_id=?", (request_id,)).fetchone()
    if row:
        return row
    now = time.time()
    c.execute(
        "INSERT INTO tickets(request_id, status, assignee_id, priority, created_at, updated_at) "
        "VALUES (?, ?, NULL, NULL, ?, ?)",
        (request_id, DEFAULT_STATUS, now, now),
    )
    return c.execute("SELECT * FROM tickets WHERE request_id=?", (request_id,)).fetchone()


def get_ticket(request_id: int) -> dict:
    with _lock, _conn() as c:
        t = _ensure_ticket(c, request_id)
        comments = [dict(r) for r in c.execute(
            "SELECT * FROM comments WHERE request_id=? ORDER BY created_at ASC",
            (request_id,),
        ).fetchall()]
        history = [dict(r) for r in c.execute(
            "SELECT * FROM status_history WHERE request_id=? ORDER BY at ASC",
            (request_id,),
        ).fetchall()]
        return {
            "request_id":  t["request_id"],
            "status":      t["status"],
            "assignee_id": t["assignee_id"],
            "priority":    t["priority"],
            "created_at":  t["created_at"],
            "updated_at":  t["updated_at"],
            "comments":    comments,
            "history":     history,
        }


def get_tickets_summary(request_ids: list[int]) -> dict[int, dict]:
    """Lightweight bulk fetch — just status + assignee + comment count."""
    if not request_ids:
        return {}
    with _conn() as c:
        placeholders = ",".join("?" * len(request_ids))
        rows = c.execute(
            f"SELECT request_id, status, assignee_id FROM tickets WHERE request_id IN ({placeholders})",
            request_ids,
        ).fetchall()
        ccounts = c.execute(
            f"SELECT request_id, COUNT(*) AS n FROM comments WHERE request_id IN ({placeholders}) GROUP BY request_id",
            request_ids,
        ).fetchall()
        out = {r["request_id"]: {"status": r["status"], "assignee_id": r["assignee_id"], "comments": 0}
               for r in rows}
        for r in ccounts:
            if r["request_id"] in out:
                out[r["request_id"]]["comments"] = r["n"]
        return out


def set_status(request_id: int, status: str, by_id: str | None) -> dict:
    if status not in {s["id"] for s in STATUSES}:
        raise ValueError(f"Unknown status: {status}")
    with _lock, _conn() as c:
        t = _ensure_ticket(c, request_id)
        prev = t["status"]
        if prev == status:
            return get_ticket(request_id)
        now = time.time()
        c.execute("UPDATE tickets SET status=?, updated_at=? WHERE request_id=?",
                  (status, now, request_id))
        c.execute("INSERT INTO status_history(request_id, from_status, to_status, by_id, at) "
                  "VALUES (?, ?, ?, ?, ?)",
                  (request_id, prev, status, by_id, now))
    return get_ticket(request_id)


def set_assignee(request_id: int, assignee_id: str | None, by_id: str | None) -> dict:
    valid = {u["id"] for u in USERS}
    if assignee_id and assignee_id not in valid:
        raise ValueError(f"Unknown user: {assignee_id}")
    with _lock, _conn() as c:
        _ensure_ticket(c, request_id)
        now = time.time()
        c.execute("UPDATE tickets SET assignee_id=?, updated_at=? WHERE request_id=?",
                  (assignee_id, now, request_id))
        # Surface assignment as a comment for visibility in the timeline.
        if assignee_id:
            user = next((u for u in USERS if u["id"] == assignee_id), {"name": assignee_id})
            c.execute("INSERT INTO comments(request_id, author_id, body, created_at) "
                      "VALUES (?, ?, ?, ?)",
                      (request_id, by_id or "system",
                       f"تم إسناد التذكرة إلى {user['name']}", now))
    return get_ticket(request_id)


def add_comment(request_id: int, author_id: str, body: str) -> dict:
    body = (body or "").strip()
    if not body:
        raise ValueError("Empty comment")
    with _lock, _conn() as c:
        _ensure_ticket(c, request_id)
        now = time.time()
        c.execute("INSERT INTO comments(request_id, author_id, body, created_at) "
                  "VALUES (?, ?, ?, ?)",
                  (request_id, author_id, body, now))
        c.execute("UPDATE tickets SET updated_at=? WHERE request_id=?",
                  (now, request_id))
    return get_ticket(request_id)


def stats() -> dict:
    with _conn() as c:
        rows = c.execute("SELECT status, COUNT(*) AS n FROM tickets GROUP BY status").fetchall()
        out = {s["id"]: 0 for s in STATUSES}
        for r in rows:
            out[r["status"]] = r["n"]
        return out


# Initialise on import — cheap, idempotent.
init_db()
