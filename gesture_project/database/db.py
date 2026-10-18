"""
database/db.py
--------------
SQLite schema creation, seed data, and thin query helpers.
All other modules import from here — no raw SQL elsewhere.
"""

import sqlite3
import json
import os
import sys
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import DB_PATH


# ── Schema ─────────────────────────────────────────────────────────────────────

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS users (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username   TEXT    NOT NULL UNIQUE,
    created_at TEXT    NOT NULL,
    enabled    INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS biometric_profiles (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL REFERENCES users(id),
    biometric_type TEXT    NOT NULL DEFAULT 'FACE',
    embedding      TEXT    NOT NULL,   -- JSON-encoded float list
    created_at     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS states (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    state_name  TEXT    NOT NULL UNIQUE,
    description TEXT,
    protected   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS transitions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source_state        TEXT    NOT NULL,
    gesture             TEXT    NOT NULL,
    destination_state   TEXT    NOT NULL,
    risk_level          TEXT    NOT NULL DEFAULT 'LOW',
    biometric_required  INTEGER NOT NULL DEFAULT 0,
    biometric_type      TEXT    NOT NULL DEFAULT 'FACE',
    confidence_threshold REAL   NOT NULL DEFAULT 0.80,
    timeout             REAL    NOT NULL DEFAULT 15.0,
    action              TEXT    NOT NULL DEFAULT 'NONE',
    enabled             INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT    NOT NULL UNIQUE,
    started_at TEXT    NOT NULL,
    ended_at   TEXT,
    user_id    INTEGER REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS transition_log (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp            TEXT    NOT NULL,
    session_id           TEXT    NOT NULL,
    user_id              INTEGER,
    previous_state       TEXT    NOT NULL,
    gesture              TEXT    NOT NULL,
    gesture_confidence   REAL    NOT NULL,
    destination_state    TEXT    NOT NULL,
    risk_level           TEXT    NOT NULL,
    biometric_required   INTEGER NOT NULL,
    biometric_type       TEXT,
    biometric_confidence REAL,
    authentication_result TEXT,
    authorization_result  TEXT   NOT NULL,
    final_state          TEXT    NOT NULL,
    action               TEXT    NOT NULL,
    reason               TEXT
);

CREATE TABLE IF NOT EXISTS system_config (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# ── Seed data ──────────────────────────────────────────────────────────────────

_STATES_SEED = [
    ("LOCKED",        "System locked, awaiting authentication",  1),
    ("AUTH_PENDING",  "Biometric authentication in progress",    1),
    ("AUTHENTICATED", "User authenticated, selecting mode",      0),
    ("PRESENTATION",  "Presentation control mode",               0),
    ("SETTINGS",      "Settings mode — protected",               1),
    ("SENSITIVE_MODE","Sensitive operations — every action gated",1),
    ("ERROR",         "Error state",                             0),
]

# Transition table — the policy lives here, not in code
_TRANSITIONS_SEED = [
    # source_state, gesture, dest_state, risk, bio_req, bio_type, conf_thresh, timeout, action
    # ── LOCKED ────────────────────────────────────────────────────────────────
    ("LOCKED",        "Start",      "AUTH_PENDING",  "MEDIUM", 0, "FACE", 0.80, 15.0, "REQUEST_AUTH"),
    # ── AUTH_PENDING (handled by biometric module, not gesture) ───────────────
    # ── AUTHENTICATED ─────────────────────────────────────────────────────────
    ("AUTHENTICATED", "Start",      "PRESENTATION",  "MEDIUM", 0, "FACE", 0.80, 15.0, "START_PRESENTATION"),
    ("AUTHENTICATED", "Grab",       "SETTINGS",      "HIGH",   1, "FACE", 0.90, 15.0, "OPEN_SETTINGS"),
    ("AUTHENTICATED", "Release",    "SENSITIVE_MODE","CRITICAL",1,"FACE", 0.95, 10.0, "ENTER_SENSITIVE"),
    # ── PRESENTATION ──────────────────────────────────────────────────────────
    ("PRESENTATION",  "Turn Left",  "PRESENTATION",  "LOW",    0, "FACE", 0.80, 15.0, "PREV_SLIDE"),
    ("PRESENTATION",  "Turn Right", "PRESENTATION",  "LOW",    0, "FACE", 0.80, 15.0, "NEXT_SLIDE"),
    ("PRESENTATION",  "Upward",     "PRESENTATION",  "LOW",    0, "FACE", 0.80, 15.0, "VOLUME_UP"),
    ("PRESENTATION",  "Downward",   "PRESENTATION",  "LOW",    0, "FACE", 0.80, 15.0, "VOLUME_DOWN"),
    ("PRESENTATION",  "Turn Up",    "PRESENTATION",  "LOW",    0, "FACE", 0.80, 15.0, "CONFIRM"),
    ("PRESENTATION",  "Turn Down",  "PRESENTATION",  "LOW",    0, "FACE", 0.80, 15.0, "CANCEL"),
    ("PRESENTATION",  "Stop",       "LOCKED",        "HIGH",   1, "FACE", 0.90, 15.0, "STOP_PRESENTATION"),
    ("PRESENTATION",  "Grab",       "SETTINGS",      "HIGH",   1, "FACE", 0.90, 15.0, "OPEN_SETTINGS"),
    # ── SETTINGS ──────────────────────────────────────────────────────────────
    ("SETTINGS",      "Turn Right", "SETTINGS",      "MEDIUM", 0, "FACE", 0.80, 15.0, "SETTINGS_NEXT"),
    ("SETTINGS",      "Turn Left",  "SETTINGS",      "MEDIUM", 0, "FACE", 0.80, 15.0, "SETTINGS_PREV"),
    ("SETTINGS",      "Turn Up",    "SETTINGS",      "HIGH",   1, "FACE", 0.90, 15.0, "SETTINGS_CONFIRM"),
    ("SETTINGS",      "Stop",       "AUTHENTICATED", "MEDIUM", 0, "FACE", 0.80, 15.0, "CLOSE_SETTINGS"),
    ("SETTINGS",      "Release",    "AUTHENTICATED", "MEDIUM", 0, "FACE", 0.80, 15.0, "CLOSE_SETTINGS"),
    # ── SENSITIVE_MODE ────────────────────────────────────────────────────────
    ("SENSITIVE_MODE","Turn Right", "SENSITIVE_MODE","CRITICAL",1,"FACE", 0.95, 10.0, "SENSITIVE_NEXT"),
    ("SENSITIVE_MODE","Turn Left",  "SENSITIVE_MODE","CRITICAL",1,"FACE", 0.95, 10.0, "SENSITIVE_PREV"),
    ("SENSITIVE_MODE","Stop",       "LOCKED",        "CRITICAL",1,"FACE", 0.95, 10.0, "EXIT_SENSITIVE"),
    ("SENSITIVE_MODE","Release",    "AUTHENTICATED", "CRITICAL",1,"FACE", 0.95, 10.0, "LEAVE_SENSITIVE"),
]


def get_connection() -> sqlite3.Connection:
    """Return a WAL-mode connection to the SQLite database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialise() -> None:
    """Create schema, run migration, and seed tables if they are empty."""
    conn = get_connection()
    conn.executescript(_SCHEMA)
    _migrate(conn)

    # Seed states
    if conn.execute("SELECT COUNT(*) FROM states").fetchone()[0] == 0:
        conn.executemany(
            "INSERT INTO states (state_name, description, protected) VALUES (?,?,?)",
            _STATES_SEED,
        )

    # Seed transitions
    if conn.execute("SELECT COUNT(*) FROM transitions").fetchone()[0] == 0:
        conn.executemany(
            """INSERT INTO transitions
               (source_state, gesture, destination_state, risk_level,
                biometric_required, biometric_type, confidence_threshold,
                timeout, action)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            _TRANSITIONS_SEED,
        )

    conn.commit()
    conn.close()


_MIGRATION_COLUMNS = [
    # (table, column, definition)
    ("transition_log", "authorization_score",      "REAL"),
    ("transition_log", "gesture_authenticity_score","REAL"),
    ("transition_log", "temporal_binding_score",    "REAL"),
    ("transition_log", "risk_score",                "REAL"),
    ("transition_log", "biometric_threshold_used",  "REAL"),
    ("transition_log", "authorization_threshold",   "REAL"),
    ("transition_log", "authorization_level",       "TEXT"),
    ("transition_log", "security_state",            "TEXT"),
    ("transition_log", "security_escalation_level", "INTEGER"),
    ("transition_log", "token_id",                  "TEXT"),
    ("transition_log", "authentication_age",        "REAL"),
    ("transition_log", "identity_continuity",       "REAL"),
    ("transition_log", "decision_reason",           "TEXT"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    """Safe ALTER TABLE migration — adds new columns if they don't exist."""
    existing: dict[str, set] = {}
    for table, col, _ in _MIGRATION_COLUMNS:
        if table not in existing:
            rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            existing[table] = {r["name"] for r in rows}
        if col not in existing[table]:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {_}")
            existing[table].add(col)
    conn.commit()


# ── User helpers ───────────────────────────────────────────────────────────────

def create_user(username: str) -> int:
    conn = get_connection()
    conn.execute(
        "INSERT OR IGNORE INTO users (username, created_at) VALUES (?,?)",
        (username, datetime.utcnow().isoformat()),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return row["id"]


def get_user(username: str) -> Optional[sqlite3.Row]:
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE username=? AND enabled=1", (username,)
    ).fetchone()
    conn.close()
    return row


def list_users() -> list:
    conn = get_connection()
    rows = conn.execute("SELECT id, username, created_at FROM users WHERE enabled=1").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Biometric profile helpers ──────────────────────────────────────────────────

def save_embedding(user_id: int, embedding: list[float]) -> None:
    conn = get_connection()
    # Replace any existing profile for this user
    conn.execute(
        "DELETE FROM biometric_profiles WHERE user_id=? AND biometric_type='FACE'",
        (user_id,),
    )
    conn.execute(
        """INSERT INTO biometric_profiles
           (user_id, biometric_type, embedding, created_at)
           VALUES (?,?,?,?)""",
        (user_id, "FACE", json.dumps(embedding), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def load_embeddings() -> list[dict]:
    """Return all enrolled face embeddings with their user_id."""
    conn = get_connection()
    rows = conn.execute(
        """SELECT bp.user_id, u.username, bp.embedding
           FROM biometric_profiles bp
           JOIN users u ON u.id = bp.user_id
           WHERE bp.biometric_type='FACE' AND u.enabled=1"""
    ).fetchall()
    conn.close()
    return [
        {"user_id": r["user_id"], "username": r["username"],
         "embedding": json.loads(r["embedding"])}
        for r in rows
    ]


# ── Transition helpers ─────────────────────────────────────────────────────────

def lookup_transition(source_state: str, gesture: str) -> Optional[sqlite3.Row]:
    conn = get_connection()
    row = conn.execute(
        """SELECT * FROM transitions
           WHERE source_state=? AND gesture=? AND enabled=1
           LIMIT 1""",
        (source_state, gesture),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_transitions() -> list[dict]:
    conn = get_connection()
    rows = conn.execute("SELECT * FROM transitions WHERE enabled=1").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Session helpers ────────────────────────────────────────────────────────────

def create_session(session_id: str) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO sessions (session_id, started_at) VALUES (?,?)",
        (session_id, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def close_session(session_id: str) -> None:
    conn = get_connection()
    conn.execute(
        "UPDATE sessions SET ended_at=? WHERE session_id=?",
        (datetime.utcnow().isoformat(), session_id),
    )
    conn.commit()
    conn.close()


# ── Audit log helper ───────────────────────────────────────────────────────────

def insert_log(entry: dict) -> None:
    conn = get_connection()
    # Build column list dynamically so old and new entries both work
    base_cols = [
        "timestamp", "session_id", "user_id", "previous_state", "gesture",
        "gesture_confidence", "destination_state", "risk_level",
        "biometric_required", "biometric_type", "biometric_confidence",
        "authentication_result", "authorization_result", "final_state",
        "action", "reason",
    ]
    extra_cols = [
        "authorization_score", "gesture_authenticity_score", "temporal_binding_score",
        "risk_score", "biometric_threshold_used", "authorization_threshold",
        "authorization_level", "security_state", "security_escalation_level",
        "token_id", "authentication_age", "identity_continuity", "decision_reason",
    ]
    cols = base_cols + [c for c in extra_cols if c in entry]
    placeholders = ", ".join(f":{c}" for c in cols)
    col_str = ", ".join(cols)
    conn.execute(f"INSERT INTO transition_log ({col_str}) VALUES ({placeholders})", entry)
    conn.commit()
    conn.close()


def recent_logs(n: int = 20) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM transition_log ORDER BY id DESC LIMIT ?", (n,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
