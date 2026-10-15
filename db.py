"""SQLite persistence for TruthLens.

Tables:
  claims       — every claim checked (keyed by sha256 of normalized text)
  verdicts     — saved result per claim (exact-duplicate cache, skips the LLM)
  evidence     — sources behind each verdict
  agent_traces — audit log of every node run {step, input, output, latency_ms}
  feedback     — user thumbs-up/down on results
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_BASE_DIR = Path(__file__).resolve().parent
load_dotenv(_BASE_DIR / ".env")  # WHY: secrets/config live only in .env, next to the code

DB_PATH = Path(os.getenv("DB_PATH", str(_BASE_DIR / "truthlens.db")))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_hash TEXT UNIQUE NOT NULL,
    claim_text TEXT NOT NULL,
    language   TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS verdicts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_hash TEXT UNIQUE NOT NULL REFERENCES claims(claim_hash),
    verdict    TEXT NOT NULL,
    confidence REAL,
    explanation TEXT,
    correct_answer TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS evidence (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_hash TEXT NOT NULL REFERENCES claims(claim_hash),
    url        TEXT NOT NULL,
    title      TEXT,
    snippet    TEXT
);
CREATE TABLE IF NOT EXISTS agent_traces (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    step       TEXT NOT NULL,
    input      TEXT,
    output     TEXT,
    latency_ms REAL,
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS feedback (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_hash TEXT,
    rating     TEXT,
    comment    TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    # WHY: sqlite3's `with conn:` only commits — on Windows an unclosed handle
    # keeps the file locked, so we always close explicitly.
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        # WHY: existing installs made before `correct_answer` existed — a plain
        # CREATE TABLE IF NOT EXISTS would never add the new column to them.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(verdicts)")}
        if "correct_answer" not in cols:
            conn.execute("ALTER TABLE verdicts ADD COLUMN correct_answer TEXT")


# ------------------------------------------------------------------ cache key
def normalize_claim(claim: str) -> str:
    """Stable text form: trim + collapse whitespace + lowercase."""
    return " ".join(claim.split()).strip().lower()


def claim_hash(claim: str) -> str:
    """Exact-duplicate key: sha256 of the normalized claim text."""
    return hashlib.sha256(normalize_claim(claim).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------- traces
def log_trace(step: str, inp: Any, out: Any, latency_ms: float) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO agent_traces (step, input, output, latency_ms) VALUES (?, ?, ?, ?)",
            (step, str(inp)[:2000], str(out)[:2000], latency_ms),
        )


# ------------------------------------------------------------- save / lookup
def save_result(
    claim_text: str,
    language: str,
    verdict: str,
    confidence: float,
    citations: list[str],
    evidence: list[dict[str, str]],
    explanation: str,
    correct_answer: str = "",
) -> None:
    """Persist a finished fact-check. Called only after the critic passed."""
    chash = claim_hash(claim_text)
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO claims (claim_hash, claim_text, language) VALUES (?, ?, ?)",
            (chash, claim_text, language),
        )
        conn.execute(
            """INSERT INTO verdicts (claim_hash, verdict, confidence, explanation, correct_answer)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(claim_hash) DO UPDATE SET verdict=excluded.verdict,
                 confidence=excluded.confidence, explanation=excluded.explanation,
                 correct_answer=excluded.correct_answer""",
            (chash, verdict, confidence, explanation, correct_answer),
        )
        conn.execute("DELETE FROM evidence WHERE claim_hash = ?", (chash,))
        conn.executemany(
            "INSERT INTO evidence (claim_hash, url, title, snippet) VALUES (?, ?, ?, ?)",
            [(chash, e.get("url", ""), e.get("title", ""), e.get("snippet", "")) for e in evidence],
        )


def lookup_cached(claim_text: str) -> dict[str, Any] | None:
    """Exact-duplicate check by hash. Returns the saved verdict, no LLM needed."""
    chash = claim_hash(claim_text)
    with _connect() as conn:
        row = conn.execute(
            """SELECT c.claim_text, c.language, v.verdict, v.confidence, v.explanation, v.correct_answer
               FROM verdicts v JOIN claims c ON c.claim_hash = v.claim_hash
               WHERE v.claim_hash = ?""",
            (chash,),
        ).fetchone()
        if row is None:
            return None
        citations = [
            r["url"]
            for r in conn.execute(
                "SELECT url FROM evidence WHERE claim_hash = ? ORDER BY id", (chash,)
            ).fetchall()
        ]
    return {
        "claim_text": row["claim_text"],
        "language": row["language"] or "en",
        "verdict": row["verdict"],
        "confidence": row["confidence"],
        "explanation": row["explanation"] or "",
        "correct_answer": row["correct_answer"] or "",
        "citations": citations,
    }


def save_feedback(claim_text: str, rating: str, comment: str = "") -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO feedback (claim_hash, rating, comment) VALUES (?, ?, ?)",
            (claim_hash(claim_text), rating, comment),
        )


# ------------------------------------------------- async wrappers (sqlite is sync)
async def init_db_async() -> None:
    await asyncio.to_thread(init_db)


async def log_trace_async(step: str, inp: Any, out: Any, latency_ms: float) -> None:
    await asyncio.to_thread(log_trace, step, inp, out, latency_ms)


async def save_result_async(*args: Any, **kwargs: Any) -> None:
    await asyncio.to_thread(save_result, *args, **kwargs)


async def lookup_cached_async(claim_text: str) -> dict[str, Any] | None:
    return await asyncio.to_thread(lookup_cached, claim_text)


def _self_test() -> None:
    """__main__ test: schema, hash dedupe, save/lookup roundtrip, feedback."""
    global DB_PATH
    DB_PATH = _BASE_DIR / "truthlens_test.db"  # WHY: never touch the real db in tests
    DB_PATH.unlink(missing_ok=True)  # WHY: a previous failed run may have left data
    try:
        init_db()
        text = "  India's   Chandrayaan-3 landed on the Moon. "
        h1, h2 = claim_hash(text), claim_hash("india's chandrayaan-3 landed on the moon.")
        assert h1 == h2, "normalized hash must match across spacing/case"
        assert lookup_cached(text) is None, "fresh cache must be empty"
        save_result(text, "en", "True", 0.91, ["https://example.com/isro"],
                    [{"url": "https://example.com/isro", "title": "ISRO", "snippet": "landed"}],
                    "It landed in August 2023.", "India's PM is Narendra Modi.")
        hit = lookup_cached(text)
        assert hit and hit["verdict"] == "True" and hit["citations"] == ["https://example.com/isro"]
        save_feedback(text, "up", "useful")
        log_trace("self_test", {"in": 1}, {"out": 2}, 1.23)
        with _connect() as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"claims", "verdicts", "evidence", "agent_traces", "feedback"} <= tables
        assert lookup_cached(text)["correct_answer"] == "India's PM is Narendra Modi."
        print(f"db.py self-test OK (db file: {DB_PATH})")
    finally:
        DB_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    _self_test()
