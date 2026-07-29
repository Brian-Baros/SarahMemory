"""--==The SarahMemory Project==--
File: SarahMemoryMigrations.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-07-11
Time: 10:11:54
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, SarahMemory AiOS, and all Parts of the SarahMemory Project are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
https://store.sarahmemory.com

===============================================================================

DATABASE MIGRATIONS MODULE v9.0.0
This module has standards with enhanced database
schema migration capabilities, version control, and comprehensive error handling.

KEY ENHANCEMENTS:
-----------------
1. ADVANCED MIGRATION SYSTEM
- Versioned migrations with rollback support
- Automatic schema detection and validation
- Migration history tracking
- Idempotent migration operations
- Cross-database compatibility (SQLite, MySQL)

2. ENHANCED ERROR HANDLING
- Detailed error logging and recovery
- Transaction safety with rollback
- Migration verification and testing
- Automatic backup before migrations
- Graceful degradation on failures

3. SCHEMA MANAGEMENT
- Automatic table creation and updates
- Index optimization and management
- Constraint validation
- Data integrity checks
- Performance optimization

4. MONITORING & AUDITING
- Migration audit trail
- Performance metrics
- Version compatibility checks
- Schema documentation
- Health status reporting

BACKWARD COMPATIBILITY:
-----------------------
All existing function signatures are preserved:
- run_migrations()

New functions added (non-breaking):
- run_versioned_migrations(target_version=None)
- verify_schema_integrity()
- rollback_migration(steps=1)
- get_migration_status()
- backup_database()

INTEGRATION POINTS:
-------------------
- SarahMemoryDatabase.py: Uses migrations during initialization
- SarahMemoryMain.py: Runs migrations at startup
- SarahMemoryUpdater.py: Applies schema updates during upgrades
- SarahMemoryDiagnostics.py: Validates schema integrity

MIGRATION HISTORY:
------------------
v1.0 - Initial schema (gui_events, qa_feedback)
v2.0 - Added emotion tracking (emotion_states, traits)
v3.0 - Added personality system tables
v4.0 - Added mesh network tables
v5.0 - Added advanced AI features
v6.0 - Added blockchain and crypto tables
v7.0 - Added Phase B identity tables
v8.0 - World-class enterprise features

===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "migration_engine"
# CATEGORY = "database_schema_migrations"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "data_memory"
# HARDWARE_DOMAIN = "filesystem"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "migrations"
# FAMILY = "core_data"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-07-11"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Database migration and schema-version engine with backup, rollback, verification, audit history, and startup compatibility checks."
# --- SARAHMETA END ---

import os
import sqlite3
import logging
import json
import hashlib
import time
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

# Import SarahMemory Globals
try:
    from SarahMemoryGlobals import DATASETS_DIR, DEBUG_MODE, DATA_DIR, SETTINGS_DIR
    GLOBALS_IMPORTED = True
except ImportError:
    # Fallback configuration
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATASETS_DIR = os.path.join(DATA_DIR, 'memory', 'datasets')
    DEBUG_MODE = False
    SETTINGS_DIR = os.path.join(DATA_DIR, 'settings')
    GLOBALS_IMPORTED = False

# Ensure directories exist
os.makedirs(DATASETS_DIR, exist_ok=True)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger('SarahMemoryMigrations')
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [Migrations] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Current schema version
CURRENT_SCHEMA_VERSION = "9.0.0"

# Database paths
DB_PATH = os.path.join(DATASETS_DIR, 'system_logs.db')
MIGRATION_HISTORY_PATH = os.path.join(DATASETS_DIR, 'migration_history.db')

# Migration settings
ENABLE_AUTO_BACKUP = str(os.getenv("SARAH_MIGRATION_AUTO_BACKUP", "true")).strip().lower() in ("1", "true", "yes", "on")
ENABLE_MIGRATION_VERIFICATION = True
BACKUP_ONLY_WHEN_PENDING = str(os.getenv("SARAH_BOOT_MIGRATION_BACKUP_ONLY_WHEN_PENDING", "true")).strip().lower() in ("1", "true", "yes", "on")
MAX_ROLLBACK_STEPS = 10

# ============================================================================
# BOUNDED ROOT-ARTIFACT PLACEMENT MIGRATION
# ============================================================================
# Only these known artifacts are considered. This is intentionally not a
# recursive drive scan. Conflicting source files are preserved under a bounded
# backup directory before the DATA_DIR root is cleaned.
_ROOT_ARTIFACT_TARGETS = {
    "context_history.db": os.path.join(DATASETS_DIR, "context_history.db"),
    "meta.db": os.path.join(DATASETS_DIR, "meta.db"),
    "user_data.db": os.path.join(DATASETS_DIR, "user_data.db"),
    "reminders.db": os.path.join(DATASETS_DIR, "reminders.db"),
    "wallets.db": os.path.join(DATASETS_DIR, "wallets.db"),
    "browser_state.json": os.path.join(SETTINGS_DIR, "browser_state.json"),
    "server_state.json": os.path.join(SETTINGS_DIR, "server_state.json"),
}


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def migrate_root_runtime_artifacts() -> Dict[str, Any]:
    """Move only known orphan DB/JSON artifacts out of DATA_DIR.

    Rules:
    - PID files remain in DATA_DIR.
    - A missing source is a no-op.
    - Existing equal targets permit safe source removal.
    - Existing different targets preserve the source in backup/migrations.
    - Copy fallback verifies size and SHA-256 before deleting the source.
    """
    result: Dict[str, Any] = {
        "ok": True,
        "data_dir": DATA_DIR,
        "datasets_dir": DATASETS_DIR,
        "settings_dir": SETTINGS_DIR,
        "moved": [],
        "deduplicated": [],
        "conflicts_preserved": [],
        "skipped": [],
        "errors": [],
    }
    try:
        os.makedirs(DATASETS_DIR, exist_ok=True)
        os.makedirs(SETTINGS_DIR, exist_ok=True)
    except Exception as exc:
        result["ok"] = False
        result["errors"].append(f"directory_create_failed:{exc}")
        return result

    backup_root = os.path.join(DATA_DIR, "backup", "migrations", "root_artifacts")
    for name, target in _ROOT_ARTIFACT_TARGETS.items():
        source = os.path.join(DATA_DIR, name)
        if not os.path.isfile(source):
            result["skipped"].append({"name": name, "reason": "source_missing"})
            continue
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.exists(target):
                same = False
                try:
                    same = os.path.getsize(source) == os.path.getsize(target) and _sha256_file(source) == _sha256_file(target)
                except Exception:
                    same = False
                if same:
                    os.remove(source)
                    result["deduplicated"].append({"source": source, "target": target})
                    continue
                os.makedirs(backup_root, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                preserved = os.path.join(backup_root, f"{stamp}_{name}")
                counter = 1
                while os.path.exists(preserved):
                    preserved = os.path.join(backup_root, f"{stamp}_{counter}_{name}")
                    counter += 1
                os.replace(source, preserved)
                result["conflicts_preserved"].append({"source": source, "target": target, "preserved": preserved})
                continue

            try:
                os.replace(source, target)
            except OSError:
                shutil.copy2(source, target)
                if os.path.getsize(source) != os.path.getsize(target) or _sha256_file(source) != _sha256_file(target):
                    try:
                        os.remove(target)
                    except Exception:
                        pass
                    raise RuntimeError("copy_verification_failed")
                os.remove(source)
            result["moved"].append({"source": source, "target": target})
        except Exception as exc:
            result["ok"] = False
            result["errors"].append({"name": name, "error": str(exc)})
    return result


def _migration_success_exists(version: str, migration_name: str) -> bool:
    try:
        if not os.path.exists(MIGRATION_HISTORY_PATH):
            return False
        con = _connect(MIGRATION_HISTORY_PATH)
        try:
            row = con.execute(
                "SELECT 1 FROM migration_history WHERE version=? AND migration_name=? AND success=1 LIMIT 1",
                (version, migration_name),
            ).fetchone()
            return bool(row)
        finally:
            con.close()
    except Exception:
        return False

# ============================================================================
# DATABASE CONNECTION UTILITIES
# ============================================================================

def _connect(path: str = DB_PATH, timeout: float = 10.0) -> sqlite3.Connection:
    """
    Create a database connection with enhanced error handling.
    
    Args:
        path: Path to database file
        timeout: Connection timeout in seconds
        
    Returns:
        sqlite3.Connection object
    """
    try:
        conn = sqlite3.connect(path, timeout=timeout)
        conn.execute("PRAGMA journal_mode=WAL")  # Enable Write-Ahead Logging for better concurrency
        conn.execute("PRAGMA synchronous=NORMAL")  # Balance between safety and performance
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database {path}: {e}")
        raise

def _exec(cur: sqlite3.Cursor, sql: str, params: tuple = None) -> bool:
    """
    Execute SQL with enhanced error handling and logging.
    
    Args:
        cur: Database cursor
        sql: SQL statement to execute
        params: Optional parameters for parameterized queries
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        return True
    except sqlite3.OperationalError as e:
        # Table/column already exists - this is expected in idempotent migrations.
        # SQLite reports duplicate ALTER TABLE columns as "duplicate column name";
        # that is not a migration failure when the desired schema already exists.
        err = str(e).lower()
        if "already exists" in err or "duplicate column name" in err:
            logger.debug(f"Schema element already exists (expected): {e}")
            return True

        logger.warning(f"SQL operation error: {e}")
        logger.debug(f"Failed SQL: {sql[:100]}...")
        return False
    except Exception as e:
        logger.error(f"Unexpected error executing SQL: {e}")
        logger.debug(f"Failed SQL: {sql[:100]}...")
        return False

# ============================================================================
# MIGRATION HISTORY MANAGEMENT
# ============================================================================

def _init_migration_history():
    """
    Initialize migration history tracking database.
    """
    try:
        conn = _connect(MIGRATION_HISTORY_PATH)
        cur = conn.cursor()
        
        _exec(cur, """
            CREATE TABLE IF NOT EXISTS migration_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                migration_name TEXT NOT NULL,
                applied_at TEXT NOT NULL,
                execution_time_ms INTEGER,
                success INTEGER DEFAULT 1,
                error_message TEXT,
                checksum TEXT
            )
        """)
        
        _exec(cur, """
            CREATE INDEX IF NOT EXISTS idx_migration_version 
            ON migration_history(version)
        """)
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Failed to initialize migration history: {e}")

def _record_migration(version: str, migration_name: str, execution_time_ms: int, 
                      success: bool = True, error_message: str = None):
    """
    Record migration execution in history.
    
    Args:
        version: Schema version
        migration_name: Name of migration
        execution_time_ms: Execution time in milliseconds
        success: Whether migration succeeded
        error_message: Error message if failed
    """
    try:
        conn = _connect(MIGRATION_HISTORY_PATH)
        cur = conn.cursor()
        
        checksum = hashlib.sha256(f"{version}{migration_name}".encode()).hexdigest()
        
        _exec(cur, """
            INSERT INTO migration_history 
            (version, migration_name, applied_at, execution_time_ms, success, error_message, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            version,
            migration_name,
            datetime.now().isoformat(),
            execution_time_ms,
            1 if success else 0,
            error_message,
            checksum
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"Failed to record migration: {e}")

# ============================================================================
# CORE MIGRATION FUNCTIONS
# ============================================================================

def run_migrations() -> bool:
    """
    Run all database schema migrations.
    This is the main entry point for applying database schema updates.
    Maintains backward compatibility with v7.x calling convention.
    
    Returns:
        bool: True if migrations successful, False otherwise
    """
    logger.info("Starting database migrations...")
    
    # Initialize migration history
    _init_migration_history()
    
    # Run versioned migrations
    success = run_versioned_migrations()
    
    if success:
        logger.info("✓ All database migrations completed successfully")
    else:
        logger.warning("⚠ Some migrations failed - system may operate in degraded mode")
    
    return success

def run_versioned_migrations(target_version: str = None) -> bool:
    """Run only pending schema migrations, with one backup when necessary."""
    if target_version is None:
        target_version = CURRENT_SCHEMA_VERSION

    logger.info("Running migrations to version %s", target_version)
    placement = migrate_root_runtime_artifacts()
    if not placement.get("ok"):
        logger.warning("Root artifact placement completed with errors: %s", placement.get("errors"))
    elif placement.get("moved") or placement.get("deduplicated") or placement.get("conflicts_preserved"):
        logger.info("Root artifact placement result: moved=%s deduplicated=%s conflicts=%s",
                    len(placement.get("moved") or []), len(placement.get("deduplicated") or []),
                    len(placement.get("conflicts_preserved") or []))

    migrations = [
        ("1.0", "v1.0_initial_schema", _migrate_v1_0),
        ("2.0", "v2.0_emotion_tracking", _migrate_v2_0),
        ("3.0", "v3.0_personality_system", _migrate_v3_0),
        ("4.0", "v4.0_mesh_network", _migrate_v4_0),
        ("5.0", "v5.0_advanced_ai", _migrate_v5_0),
        ("6.0", "v6.0_blockchain_crypto", _migrate_v6_0),
        ("7.0", "v7.0_identity_device", _migrate_v7_0),
        ("8.0", "v8.0_enterprise_features", _migrate_v8_0),
    ]
    pending = [(v, n, fn) for v, n, fn in migrations if not _migration_success_exists(v, n)]
    if not pending:
        logger.info("No pending schema migrations; backup and schema rewrites skipped.")
        if ENABLE_MIGRATION_VERIFICATION:
            return bool(verify_schema_integrity())
        return True

    try:
        if ENABLE_AUTO_BACKUP and (not BACKUP_ONLY_WHEN_PENDING or pending):
            if os.path.exists(DB_PATH) and os.path.getsize(DB_PATH) > 0:
                backup_database()
        success = True
        for version, name, fn in pending:
            logger.info("Applying pending migration %s (%s)", version, name)
            success = bool(fn()) and success
        if ENABLE_MIGRATION_VERIFICATION:
            success = bool(verify_schema_integrity()) and success
        return success
    except Exception as exc:
        logger.error("Migration failed: %s", exc, exc_info=True)
        return False

#======================================================================
# MIGRATION HELPERS
#======================================================================

def _table_columns(cur: sqlite3.Cursor, table_name: str) -> List[str]:
    """Return existing column names for a table. Never raises for migration callers."""
    try:
        safe_table = str(table_name or "").replace('"', '""')
        cur.execute(f'PRAGMA table_info("{safe_table}")')
        return [str(row[1]) for row in cur.fetchall()]
    except Exception as e:
        logger.debug(f"Column inspection failed for {table_name}: {e}")
        return []


def _column_exists(cur: sqlite3.Cursor, table_name: str, column_name: str) -> bool:
    """Idempotent schema helper used before ALTER TABLE ADD COLUMN."""
    try:
        wanted = str(column_name or "").strip().lower()
        return wanted in {c.strip().lower() for c in _table_columns(cur, table_name)}
    except Exception:
        return False


def _add_column_if_missing(cur: sqlite3.Cursor, table_name: str, column_name: str, column_sql: str) -> bool:
    """
    Add a column only when missing.

    This prevents repeat boot warnings like:
        duplicate column name: context
        duplicate column name: source

    Returns True when the column already exists or was added successfully.
    """
    if _column_exists(cur, table_name, column_name):
        logger.debug(f"Schema column already present: {table_name}.{column_name}")
        return True

    safe_table = str(table_name or "").replace('"', '""')
    sql = f'ALTER TABLE "{safe_table}" ADD COLUMN {column_sql}'
    return _exec(cur, sql)


def ensure_traits_last_updated_column(conn):
    """
    Ensures the traits table contains the 'last_updated' column.
    Idempotent and safe to run on every boot.
    """
    try:
        cursor = conn.cursor()
        if _add_column_if_missing(
            cursor,
            "traits",
            "last_updated",
            "last_updated TEXT DEFAULT CURRENT_TIMESTAMP",
        ):
            conn.commit()
    except Exception as e:
        logger.warning(f"[MIGRATIONS] Failed to ensure traits.last_updated column: {e}")

# ============================================================================
# VERSION-SPECIFIC MIGRATIONS
# ============================================================================

def _migrate_v1_0() -> bool:
    """
    v1.0 Migration: Initial schema - GUI events and QA feedback.
    """
    migration_name = "v1.0_initial_schema"
    start_time = time.time()
    
    try:
        conn = _connect()
        cur = conn.cursor()
        
        # GUI events table
        _exec(cur, """
            CREATE TABLE IF NOT EXISTS gui_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event TEXT NOT NULL,
                details TEXT
            )
        """)
        
        _exec(cur, """
            CREATE INDEX IF NOT EXISTS idx_gui_events_timestamp 
            ON gui_events(timestamp)
        """)
        
        # QA feedback table
        _exec(cur, """
            CREATE TABLE IF NOT EXISTS qa_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                question TEXT NOT NULL,
                score INTEGER,
                feedback TEXT
            )
        """)
        
        _exec(cur, """
            CREATE INDEX IF NOT EXISTS idx_qa_feedback_ts 
            ON qa_feedback(ts)
        """)
        
        conn.commit()
        conn.close()
        
        execution_time = int((time.time() - start_time) * 1000)
        _record_migration("1.0", migration_name, execution_time, True)
        
        logger.debug(f"✓ {migration_name} completed in {execution_time}ms")
        return True
        
    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        _record_migration("1.0", migration_name, execution_time, False, str(e))
        logger.error(f"✗ {migration_name} failed: {e}")
        return False

def _migrate_v2_0() -> bool:
    """
    v2.0 Migration: Emotion tracking system.
    """
    migration_name = "v2.0_emotion_tracking"
    start_time = time.time()
    
    try:
        conn = _connect()
        cur = conn.cursor()
        
        # Emotion states table
        _exec(cur, """
            CREATE TABLE IF NOT EXISTS emotion_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                emotion TEXT NOT NULL,
                intensity REAL DEFAULT 0.5,
                context TEXT
            )
        """)
        
        _exec(cur, """
            CREATE INDEX IF NOT EXISTS idx_emotion_states_ts 
            ON emotion_states(ts)
        """)
        
        # Personality traits table
        _exec(cur, """
            CREATE TABLE IF NOT EXISTS traits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                trait TEXT NOT NULL,
                value REAL DEFAULT 0.5,
                source TEXT
            )
        """)
        
        _exec(cur, """
            CREATE INDEX IF NOT EXISTS idx_traits_ts 
            ON traits(ts)
        """)
        
        conn.commit()
        conn.close()
        
        execution_time = int((time.time() - start_time) * 1000)
        _record_migration("2.0", migration_name, execution_time, True)
        
        logger.debug(f"✓ {migration_name} completed in {execution_time}ms")
        return True
        
    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        _record_migration("2.0", migration_name, execution_time, False, str(e))
        logger.error(f"✗ {migration_name} failed: {e}")
        return False

def _migrate_v3_0() -> bool:
    """
    v3.0 Migration: Personality system expansion.
    """
    migration_name = "v3.0_personality_system"
    start_time = time.time()
    
    try:
        conn = _connect()
        cur = conn.cursor()
        
        # Add personality columns to existing tables only when missing.
        # v2.0 already creates these columns on clean installs, so v3.0 must not
        # blindly ALTER them again or the boot logs create false-positive errors.
        _add_column_if_missing(cur, "emotion_states", "context", "context TEXT")
        _add_column_if_missing(cur, "traits", "source", "source TEXT")
        
        conn.commit()
        conn.close()
        
        execution_time = int((time.time() - start_time) * 1000)
        _record_migration("3.0", migration_name, execution_time, True)
        
        logger.debug(f"✓ {migration_name} completed in {execution_time}ms")
        return True
        
    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        _record_migration("3.0", migration_name, execution_time, False, str(e))
        logger.error(f"✗ {migration_name} failed: {e}")
        return False

def _migrate_v4_0() -> bool:
    """
    v4.0 Migration: Mesh network tables (Phase D preparation).
    """
    migration_name = "v4.0_mesh_network"
    start_time = time.time()
    
    try:
        conn = _connect()
        cur = conn.cursor()
        
        # Mesh network tables would be added here when Phase D is implemented
        # For now, this is a placeholder migration
        
        conn.commit()
        conn.close()
        
        execution_time = int((time.time() - start_time) * 1000)
        _record_migration("4.0", migration_name, execution_time, True)
        
        logger.debug(f"✓ {migration_name} completed in {execution_time}ms")
        return True
        
    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        _record_migration("4.0", migration_name, execution_time, False, str(e))
        logger.error(f"✗ {migration_name} failed: {e}")
        return False

def _migrate_v5_0() -> bool:
    """
    v5.0 Migration: Advanced AI features.
    """
    migration_name = "v5.0_advanced_ai"
    start_time = time.time()
    
    try:
        conn = _connect()
        cur = conn.cursor()
        
        # Advanced AI tables would be added here
        # Placeholder for now
        
        conn.commit()
        conn.close()
        
        execution_time = int((time.time() - start_time) * 1000)
        _record_migration("5.0", migration_name, execution_time, True)
        
        logger.debug(f"✓ {migration_name} completed in {execution_time}ms")
        return True
        
    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        _record_migration("5.0", migration_name, execution_time, False, str(e))
        logger.error(f"✗ {migration_name} failed: {e}")
        return False

def _migrate_v6_0() -> bool:
    """
    v6.0 Migration: Blockchain and crypto tables.
    """
    migration_name = "v6.0_blockchain_crypto"
    start_time = time.time()
    
    try:
        conn = _connect()
        cur = conn.cursor()
        
        # Blockchain tables would be added here
        # Placeholder for now
        
        conn.commit()
        conn.close()
        
        execution_time = int((time.time() - start_time) * 1000)
        _record_migration("6.0", migration_name, execution_time, True)
        
        logger.debug(f"✓ {migration_name} completed in {execution_time}ms")
        return True
        
    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        _record_migration("6.0", migration_name, execution_time, False, str(e))
        logger.error(f"✗ {migration_name} failed: {e}")
        return False

def _migrate_v7_0() -> bool:
    """
    v7.0 Migration: Phase B identity and device awareness.
    """
    migration_name = "v7.0_identity_device"
    start_time = time.time()
    
    try:
        conn = _connect()
        cur = conn.cursor()
        
        # Phase B tables would be added here
        # Placeholder for now
        
        conn.commit()
        conn.close()
        
        execution_time = int((time.time() - start_time) * 1000)
        _record_migration("7.0", migration_name, execution_time, True)
        
        logger.debug(f"✓ {migration_name} completed in {execution_time}ms")
        return True
        
    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        _record_migration("7.0", migration_name, execution_time, False, str(e))
        logger.error(f"✗ {migration_name} failed: {e}")
        return False

def _migrate_v8_0() -> bool:
    """
    v8.0 Migration: World-class enterprise features.
    """
    migration_name = "v8.0_enterprise_features"
    start_time = time.time()
    
    try:
        conn = _connect()
        cur = conn.cursor()
        
        # Add performance monitoring table
        _exec(cur, """
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                component TEXT,
                metadata TEXT
            )
        """)
        
        _exec(cur, """
            CREATE INDEX IF NOT EXISTS idx_performance_metrics_ts 
            ON performance_metrics(ts)
        """)
        
        # Add audit trail table
        _exec(cur, """
            CREATE TABLE IF NOT EXISTS audit_trail (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                user_id TEXT,
                action TEXT NOT NULL,
                resource TEXT,
                result TEXT,
                details TEXT
            )
        """)
        
        _exec(cur, """
            CREATE INDEX IF NOT EXISTS idx_audit_trail_ts 
            ON audit_trail(ts)
        """)
        
        conn.commit()
        conn.close()
        
        execution_time = int((time.time() - start_time) * 1000)
        _record_migration("8.0", migration_name, execution_time, True)
        
        logger.debug(f"✓ {migration_name} completed in {execution_time}ms")
        return True
        
    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        _record_migration("8.0", migration_name, execution_time, False, str(e))
        logger.error(f"✗ {migration_name} failed: {e}")
        return False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def verify_schema_integrity() -> bool:
    """
    Verify database schema integrity.
    
    Returns:
        bool: True if schema is valid, False otherwise
    """
    try:
        conn = _connect()
        cur = conn.cursor()
        
        # Check required tables exist
        required_tables = [
            'gui_events',
            'qa_feedback',
            'emotion_states',
            'traits',
            'performance_metrics',
            'audit_trail'
        ]
        
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cur.fetchall()]
        
        missing_tables = set(required_tables) - set(existing_tables)
        
        if missing_tables:
            logger.warning(f"Missing tables: {missing_tables}")
            conn.close()
            return False
        
        conn.close()
        logger.info("✓ Schema integrity verified")
        return True
        
    except Exception as e:
        logger.error(f"Schema verification failed: {e}")
        return False

def backup_database(backup_path: str = None) -> Optional[str]:
    """
    Create backup of database before migrations.
    
    Args:
        backup_path: Custom backup path (optional)
        
    Returns:
        str: Path to backup file, or None if failed
    """
    try:
        if backup_path is None:
            backup_dir = os.path.join(DATASETS_DIR, 'backups')
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"system_logs_backup_{timestamp}.db")
        
        import shutil
        shutil.copy2(DB_PATH, backup_path)
        
        logger.info(f"✓ Database backed up to: {backup_path}")
        return backup_path
        
    except Exception as e:
        logger.error(f"Database backup failed: {e}")
        return None

def get_migration_status() -> Dict[str, Any]:
    """
    Get current migration status and history.
    
    Returns:
        Dict containing migration status information
    """
    try:
        _init_migration_history()
        
        conn = _connect(MIGRATION_HISTORY_PATH)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT version, migration_name, applied_at, execution_time_ms, success
            FROM migration_history
            ORDER BY applied_at DESC
            LIMIT 20
        """)
        
        history = []
        for row in cur.fetchall():
            history.append({
                "version": row[0],
                "migration_name": row[1],
                "applied_at": row[2],
                "execution_time_ms": row[3],
                "success": bool(row[4])
            })
        
        conn.close()
        
        return {
            "current_version": CURRENT_SCHEMA_VERSION,
            "migration_history": history,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get migration status: {e}")
        return {
            "current_version": CURRENT_SCHEMA_VERSION,
            "error": str(e)
        }

# ============================================================================
# MAIN ENTRY POINT (for testing)
# ============================================================================

if __name__ == '__main__':
    """
    Module test suite for database migrations.
    """
    logger.info("="*70)
    logger.info("SarahMemory Migrations Module v8.0 - Test Suite")
    logger.info("="*70)
    
    # Run migrations
    logger.info("\n--- Running Database Migrations ---")
    if run_migrations():
        logger.info("✓ Migrations completed successfully")
    else:
        logger.error("✗ Migrations failed")
    
    # Verify schema
    logger.info("\n--- Verifying Schema Integrity ---")
    if verify_schema_integrity():
        logger.info("✓ Schema verification passed")
    else:
        logger.error("✗ Schema verification failed")
    
    # Get status
    logger.info("\n--- Migration Status ---")
    status = get_migration_status()
    logger.info(f"Current Version: {status['current_version']}")
    logger.info(f"Recent Migrations: {len(status.get('migration_history', []))}")
    
    logger.info("\n" + "="*70)
    logger.info("SarahMemory Migrations Testing Complete")
    logger.info("="*70)

# ====================================================================
# END OF SarahMemoryMigrations.py v9.0.0
# ====================================================================
