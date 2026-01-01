# --==The SarahMemory Project==--
# File: /home/Softdev0/SarahMemory/data/mods/v800/sm_v800_database_stability_patch.py
# Patch: v8.0.0 Database Stability (Consolidated)
#
# Goal:
# - Consolidate the v8.0.0 "database stability" patch stubs into ONE patch file.
# - Provide a real fix for the observed runtime error:
#       "table traits has no column named last_updated"
# - Remain headless-safe (Linux/PythonAnywhere) and Windows-safe.
# - Monkey-patch only (no core file modifications).
#
# Notes:
# - This patch is safe to import multiple times (idempotent).
# - If the target DB file(s) are missing, this patch will not crash.
# - No new third-party dependencies are introduced.

"""Owner-only monkey patch for SarahMemory.

RULES:
- DO NOT modify core files directly.
- This patch should be safe to import multiple times.
- It may add wrappers / monkey-patch functions in-memory at runtime.
- No new dependencies.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import traceback

logger = logging.getLogger("sm_v800_database_stability_patch")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

_APPLIED = False


# =============================================================================
# Helpers
# =============================================================================

def _find_datasets_dir() -> str:
    """Locate the datasets directory in a safe way."""
    # 1) Preferred: SarahMemoryGlobals.DATASETS_DIR
    try:
        import SarahMemoryGlobals as G  # type: ignore
        ds = getattr(G, "DATASETS_DIR", None)
        if isinstance(ds, str) and ds.strip():
            return ds
    except Exception:
        pass

    # 2) Common relative fallback: ./data/memory/datasets
    try:
        base_dir = os.getcwd()
        guess = os.path.join(base_dir, "data", "memory", "datasets")
        return guess
    except Exception:
        return "data/memory/datasets"


def _safe_connect(db_path: str) -> sqlite3.Connection | None:
    try:
        if not db_path or not isinstance(db_path, str):
            return None
        if not os.path.exists(db_path):
            return None
        conn = sqlite3.connect(db_path, timeout=5.0)
        return conn
    except Exception:
        return None


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (table,))
        return cur.fetchone() is not None
    except Exception:
        return False


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table});")
        rows = cur.fetchall() or []
        for r in rows:
            # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
            if len(r) >= 2 and str(r[1]).lower() == str(column).lower():
                return True
        return False
    except Exception:
        return False


def _ensure_column(db_path: str, table: str, column: str, column_def_sql: str) -> bool:
    """Ensure a column exists. Returns True if column exists/created."""
    conn = _safe_connect(db_path)
    if conn is None:
        return False

    try:
        if not _has_table(conn, table):
            return False

        if _has_column(conn, table, column):
            return True

        cur = conn.cursor()
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def_sql};")
        conn.commit()
        logger.info("[DBSTAB] Added missing column: %s.%s (%s) in %s", table, column, column_def_sql, db_path)
        return True
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        logger.warning("[DBSTAB] Failed ensuring column %s.%s in %s: %s", table, column, db_path, e)
        logger.debug(traceback.format_exc())
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _ensure_personality_traits_last_updated() -> bool:
    """Fix: traits.last_updated missing in personality1.db (seen in System.log on 2026-01-01)."""
    datasets_dir = _find_datasets_dir()
    db_path = os.path.join(datasets_dir, "personality1.db")

    # Column type choice:
    # - Use TEXT to stay compatible with any code writing ISO strings or timestamps.
    # - Default NULL is fine; no default constraint needed.
    return _ensure_column(db_path, "traits", "last_updated", "TEXT")


# =============================================================================
# Main Patch Apply
# =============================================================================

def apply() -> bool:
    """Apply the monkey patch (idempotent)."""
    global _APPLIED
    if _APPLIED:
        return True

    try:
        # 1) Schema guardrails for known runtime errors
        _ensure_personality_traits_last_updated()

        # 2) (Optional) Soft wrappers:
        # We keep this conservative. If you later identify a specific function
        # that writes to traits, we can wrap it here WITHOUT changing core files.

        _APPLIED = True
        return True

    except Exception as e:
        logger.error("Patch apply failed: %s", e)
        logger.debug(traceback.format_exc())
        return False


# Self-apply on import (fallback)
try:
    apply()
except Exception:
    pass
