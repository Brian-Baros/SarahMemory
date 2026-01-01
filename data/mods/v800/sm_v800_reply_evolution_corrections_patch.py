# --==The SarahMemory Project==--
# File: ../data/mods/v800/sm_v800_reply_evolution_corrections_patch.py
# Patch: v8.0.0 Evolution Corrections Hook
#
# Goal:
# - Add a safe "response_corrections" store (no history rewrites).
# - Serve corrected answers when the same/similar question is asked again.
# - Headless-safe (PythonAnywhere) and Windows-safe.
# - No new dependencies.

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("sm_v800_reply_evolution_corrections_patch")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

_APPLIED = False

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")

def _norm_q(text: str) -> str:
    if not text:
        return ""
    t = text.strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^a-z0-9\s\.\,\?\!\'\"\-:;/\(\)]", "", t)
    return t[:800]

def _get_paths() -> Tuple[Path, Path]:
    """
    Returns:
      datasets_dir, corrections_db_path
    """
    try:
        import SarahMemoryGlobals as config  # type: ignore
        base_dir = Path(getattr(config, "BASE_DIR", Path(__file__).resolve().parents[4]))
        data_dir = Path(getattr(config, "DATA_DIR", base_dir / "data"))
        datasets_dir = Path(getattr(config, "DATASETS_DIR", data_dir / "memory" / "datasets"))
    except Exception:
        # conservative fallback
        base_dir = Path(__file__).resolve().parents[4]
        datasets_dir = base_dir / "data" / "memory" / "datasets"

    datasets_dir.mkdir(parents=True, exist_ok=True)

    # Keep this separate so we don't risk schema conflicts with existing DBs:
    corrections_db = datasets_dir / "evolution_corrections.db"
    return datasets_dir, corrections_db

def _db() -> sqlite3.Connection:
    _, corrections_db = _get_paths()
    conn = sqlite3.connect(str(corrections_db), timeout=5.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
    except Exception:
        pass
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS response_corrections (
            question_norm TEXT PRIMARY KEY,
            better_answer TEXT NOT NULL,
            ts TEXT,
            source TEXT,
            confidence REAL,
            notes TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS response_quality_flags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT,
            answer TEXT,
            ts TEXT,
            flags TEXT
        )
        """
    )
    conn.commit()
    return conn

def get_correction(question: str) -> Optional[str]:
    qn = _norm_q(question)
    if not qn:
        return None
    try:
        with _db() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT better_answer FROM response_corrections WHERE question_norm=?",
                (qn,),
            )
            row = cur.fetchone()
            return row[0] if row and row[0] else None
    except Exception:
        return None

def put_correction(question: str, better_answer: str, source: str = "evolution", confidence: float = 0.75, notes: str = "") -> bool:
    qn = _norm_q(question)
    if not qn or not better_answer:
        return False
    try:
        with _db() as conn:
            conn.execute(
                """
                INSERT INTO response_corrections(question_norm, better_answer, ts, source, confidence, notes)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(question_norm) DO UPDATE SET
                    better_answer=excluded.better_answer,
                    ts=excluded.ts,
                    source=excluded.source,
                    confidence=excluded.confidence,
                    notes=excluded.notes
                """,
                (qn, better_answer.strip(), _now(), source, float(confidence), notes[:800]),
            )
            conn.commit()
        return True
    except Exception:
        return False

def _flag_bad_answer(question: str, answer: str, flags: str) -> None:
    try:
        with _db() as conn:
            conn.execute(
                "INSERT INTO response_quality_flags(question, answer, ts, flags) VALUES(?,?,?,?)",
                (question[:1200], (answer or "")[:5000], _now(), flags[:500]),
            )
            conn.commit()
    except Exception:
        pass

def _looks_low_quality(answer: str) -> Optional[str]:
    """
    Heuristics only (fast, local, no API call):
    returns flags string if low quality, else None
    """
    if not answer:
        return "empty_answer"
    a = answer.lower()

    flags = []
    # classic "LLM disclaimers" that you explicitly *don't* want
    if "as a language model" in a or "trained on data" in a or "i don't have a version number" in a:
        flags.append("llm_disclaimer")

    # overly short / non-answer
    if len(answer.strip()) < 25:
        flags.append("too_short")

    # identity confusion patterns you mentioned before
    if "gpt-3" in a or "openai" in a and "i am sarah" not in a:
        flags.append("identity_drift")

    return ",".join(flags) if flags else None

def apply() -> bool:
    global _APPLIED
    if _APPLIED:
        return True

    try:
        # Ensure DB exists and schema ready
        with _db():
            pass

        # Monkey-patch SarahMemoryReply to consult corrections FIRST.
        import SarahMemoryReply as SMR  # type: ignore

        # Find a likely reply entrypoint
        target_name = None
        for name in ("generate_reply", "get_reply", "reply", "handle_user_message"):
            if hasattr(SMR, name) and callable(getattr(SMR, name)):
                target_name = name
                break

        if not target_name:
            logger.warning("No known reply function found to patch in SarahMemoryReply.")
            _APPLIED = True
            return True

        original = getattr(SMR, target_name)

        def wrapped(*args, **kwargs):
            # Try to locate the question text from args/kwargs (best-effort).
            question = ""
            for key in ("message", "prompt", "text", "user_input", "query"):
                if key in kwargs and isinstance(kwargs[key], str):
                    question = kwargs[key]
                    break
            if not question:
                for a in args:
                    if isinstance(a, str) and len(a) > 0:
                        question = a
                        break

            # If we have a correction, use it immediately.
            corr = get_correction(question) if question else None
            if corr:
                return corr

            # Otherwise call original
            ans = original(*args, **kwargs)

            # If answer looks low-quality, flag it (Evolution can fix later)
            try:
                flags = _looks_low_quality(ans if isinstance(ans, str) else str(ans))
                if flags and question:
                    _flag_bad_answer(question, ans if isinstance(ans, str) else str(ans), flags)
            except Exception:
                pass

            return ans

        setattr(SMR, target_name, wrapped)
        logger.info("Patched SarahMemoryReply.%s with Evolution corrections hook.", target_name)

        _APPLIED = True
        return True

    except Exception as e:
        logger.error("Patch apply failed: %s", e)
        return False

# auto-apply on import
try:
    apply()
except Exception:
    pass
