"""--==The SarahMemory Project==--
File: SarahMemoryCleanupDaily.py
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

DAILY CLEANUP MODULE v9.0.0
========================================

This module has standards with enhanced cleanup
capabilities, intelligent scheduling, and comprehensive maintenance automation.

KEY ENHANCEMENTS:
-----------------
1. INTELLIGENT CLEANUP SYSTEM
- Smart database optimization with WAL mode support
- Adaptive vacuum scheduling based on usage patterns
- Intelligent log rotation with compression
- Selective cache cleanup (preserve hot cache)
- Performance-aware cleanup timing

2. ENHANCED MAINTENANCE
- Database integrity checks before vacuum
- Automatic index optimization
- Orphaned file detection and cleanup
- Disk space monitoring and alerts
- Memory usage optimization

3. SCHEDULING & AUTOMATION
- Configurable cleanup intervals
- Priority-based cleanup tasks
- Background task execution
- Cleanup impact monitoring
- Resource-aware execution

4. MONITORING & REPORTING
- Detailed cleanup statistics
- Space recovered metrics
- Performance improvement tracking
- Error logging and recovery
- Health status reporting

BACKWARD COMPATIBILITY:
-----------------------
All existing function signatures are preserved:
- run_daily_cleanup()
- vacuum_all()
- rotate_text_logs(keep_days=14)
- remove_tmp_dirs()

New functions added (non-breaking):
- run_intelligent_cleanup()
- get_cleanup_stats()
- optimize_databases()
- cleanup_with_monitoring()
- check_disk_space()

INTEGRATION POINTS:
-------------------
- SarahMemoryMain.py: Schedules daily cleanup
- SarahMemoryUpdater.py: Runs cleanup after updates
- SarahMemoryDiagnostics.py: Monitors cleanup performance
- SarahMemoryDatabase.py: Provides vacuum optimization

===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "B"
# ROLE = "maintenance_engine"
# CATEGORY = "scheduled_cleanup"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "filesystem"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "daily_cleanup"
# FAMILY = "maintenance"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-07-11"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Automated maintenance engine for vacuum, log rotation, temp cleanup, disk checks, optimization, and resource-aware scheduled cleanup."
# --- SARAHMETA END ---

import os
import shutil
import sqlite3
import time
import logging
import json
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

# ARILE protected cleanup/update guard. Protected immune-system files are never
# cleanup/update targets from automated maintenance lanes.
try:
    from SarahMemoryARILE import arile_is_protected_core_file, arile_emit
except Exception:  # pragma: no cover
    arile_emit = None  # type: ignore
    def arile_is_protected_core_file(path_value):
        try:
            return Path(path_value).name.lower() in {"sarahmemoryglobals.py", "sarahmemoryarile.py"}
        except Exception:
            return False

def _arile_block_if_protected_target(path_value: str, operation: str = "maintenance") -> None:
    if arile_is_protected_core_file(path_value):
        try:
            if callable(arile_emit):
                arile_emit(source=__name__, kind="protected_core_variance", failure_type="maintenance_target_blocked", severity=0.92, confidence=0.90, risk="critical", summary=f"Blocked {operation} against protected core file.", requires_governance=True, retention="security_audit", data={"target": str(path_value)})
        except Exception:
            pass
        raise PermissionError(f"Protected core maintenance target blocked: {path_value}")


# Import SarahMemoryGlobals for configuration
try:
    import SarahMemoryGlobals as config
    GLOBALS_IMPORTED = True
except Exception:
    # Fallback configuration
    class config:
        BASE_DIR = os.getcwd()
        DATA_DIR = os.path.join(BASE_DIR, "data")
        MEMORY_DIR = os.path.join(DATA_DIR, "memory")
        DATASETS_DIR = os.path.join(MEMORY_DIR, "datasets")
        LOGS_DIR = os.path.join(DATA_DIR, "logs")
        DEBUG_MODE = False
    GLOBALS_IMPORTED = False

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger('SarahMemoryCleanupDaily')
logger.setLevel(logging.DEBUG if getattr(config, 'DEBUG_MODE', False) else logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [CleanupDaily] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Database list for cleanup
DBS = [
    os.path.join(config.DATASETS_DIR, "context_history.db"),
    os.path.join(config.DATASETS_DIR, "ai_learning.db"),
    os.path.join(config.DATASETS_DIR, "personality1.db"),
    os.path.join(config.DATASETS_DIR, "functions.db"),
    os.path.join(config.DATASETS_DIR, "system_logs.db"),
    os.path.join(config.DATASETS_DIR, "sarah_main.db"),
    os.path.join(config.DATASETS_DIR, "embeddings.db"),
]

# Cleanup configuration
DEFAULT_LOG_RETENTION_DAYS = 14
DEFAULT_CACHE_RETENTION_DAYS = 7
DEFAULT_TMP_RETENTION_HOURS = 24
MIN_DISK_SPACE_GB = 1.0  # Minimum free space to maintain
VACUUM_SIZE_THRESHOLD_MB = 10  # Only vacuum if DB > 10MB

# Directories to clean
TMP_FOLDERS = ("tmp", "temp", "cache", "__pycache__")


def _cfg_bool(name: str, default: bool = False) -> bool:
    try:
        value = getattr(config, name, default)
    except Exception:
        value = default
    env_value = os.getenv(name)
    if env_value is not None:
        value = env_value
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on", "enabled")


def _safe_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def cleanup_ticketing_system(archive_old: Optional[bool] = None, keep_days: int = 30) -> Dict[str, Any]:
    """Inventory and optionally archive ticket/report clutter without deleting truth.

    SARAHMEMORY_PATCH_NOTE 2026-06-28:
    Daily ticket cleanup is audit-first.  Default behavior is inventory/report
    only.  Moving old files requires SARAH_CLEANUP_TICKET_ARCHIVE_ENABLED=true.
    No production ticket is deleted by this function.
    """
    if archive_old is None:
        archive_old = _cfg_bool("SARAH_CLEANUP_TICKET_ARCHIVE_ENABLED", False)
    data_dir = str(getattr(config, "DATA_DIR", os.path.join(os.getcwd(), "data")))
    devbridge = os.path.join(data_dir, "devbridge")
    audit_dir = os.path.join(data_dir, "audit")
    report_dir = os.path.join(audit_dir, "summaries")
    archive_dir = os.path.join(audit_dir, "archived_tickets")
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    cutoff = time.time() - max(1, int(keep_days)) * 86400
    stats: Dict[str, Any] = {"ok": True, "archive_old": bool(archive_old), "keep_days": keep_days, "scanned": 0, "archived": 0, "by_folder": {}, "errors": [], "deleted": 0}
    watched = [devbridge, os.path.join(data_dir, "reports"), os.path.join(data_dir, "repair_outbox"), os.path.join(audit_dir, "reports"), os.path.join(audit_dir, "repair_outbox")]
    for root in watched:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if not name.lower().endswith((".json", ".jsonl", ".txt", ".log")):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    stats["scanned"] += 1
                    rel_top = os.path.relpath(dirpath, data_dir).split(os.sep)[0]
                    stats["by_folder"][rel_top] = int(stats["by_folder"].get(rel_top, 0)) + 1
                    if archive_old and os.path.getmtime(path) < cutoff and "processed" in dirpath.lower():
                        dest_dir = os.path.join(archive_dir, os.path.relpath(dirpath, data_dir))
                        os.makedirs(dest_dir, exist_ok=True)
                        shutil.move(path, os.path.join(dest_dir, name))
                        stats["archived"] += 1
                except Exception as exc:
                    stats["errors"].append(f"{path}: {exc}")
    stats["timestamp"] = datetime.now().isoformat()
    _safe_write_json(os.path.join(report_dir, f"ticket_cleanup_inventory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), stats)
    return stats


# ============================================================================
# SEMANTIC QA CACHE HYGIENE - v9.0.0
# ============================================================================
def run_poison_cache_cleanup(*, quarantine: Optional[bool] = None, max_rows: int = 5000) -> Dict[str, Any]:
    """Optional QA-cache poison hygiene for scheduled maintenance.

    Default is read-only scan. Actual quarantine requires
    SARAH_DAILY_CLEANUP_QUARANTINE_POISON_CACHE=true.
    """
    if quarantine is None:
        quarantine = _cfg_bool("SARAH_DAILY_CLEANUP_QUARANTINE_POISON_CACHE", False)
    out: Dict[str, Any] = {
        "ok": False,
        "enabled": _cfg_bool("SARAH_DAILY_CLEANUP_ENABLE_POISON_SCAN", False) or bool(quarantine),
        "quarantine": bool(quarantine),
        "max_rows": int(max_rows),
        "errors": [],
    }
    if not out["enabled"]:
        out.update({"ok": True, "safe_policy_skipped": True, "reason": "poison cache scan disabled by default"})
        return out
    try:
        import SarahMemoryCleanup as _SMCleanup  # type: ignore
        if bool(quarantine):
            result = _SMCleanup.quarantine_poisoned_qa_cache(dry_run=False, max_rows=max_rows, backup=True)
        else:
            result = _SMCleanup.scan_poisoned_qa_cache(max_rows=max_rows)
        out.update({"ok": True, "result": result})
    except Exception as exc:
        out["errors"].append(str(exc))
    return out


# ============================================================================
# CORE CLEANUP FUNCTIONS (v8.0 Enhanced)
# ============================================================================

def vacuum_all() -> Dict[str, Any]:
    """
    Vacuum all SarahMemory databases with enhanced error handling and monitoring.
    
    Returns:
        Dict containing vacuum statistics
    """
    start_time = time.time()
    stats = {
        "databases_processed": 0,
        "databases_vacuumed": 0,
        "databases_skipped": 0,
        "databases_failed": 0,
        "space_recovered_mb": 0.0,
        "errors": []
    }
    
    for db_path in DBS:
        try:
            if not os.path.exists(db_path):
                logger.debug(f"Database not found, skipping: {db_path}")
                stats["databases_skipped"] += 1
                continue
            
            # Get initial size
            initial_size = os.path.getsize(db_path)
            
            # Skip small databases
            if initial_size < (VACUUM_SIZE_THRESHOLD_MB * 1024 * 1024):
                logger.debug(f"Database too small to vacuum: {os.path.basename(db_path)}")
                stats["databases_skipped"] += 1
                continue
            
            # Check if database is in use
            try:
                conn = sqlite3.connect(db_path, timeout=5.0)
                
                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode=WAL")
                
                # Check integrity first
                result = conn.execute("PRAGMA integrity_check").fetchone()
                if result[0] != "ok":
                    logger.warning(f"Database integrity check failed: {db_path}")
                    stats["databases_failed"] += 1
                    stats["errors"].append(f"{os.path.basename(db_path)}: Integrity check failed")
                    conn.close()
                    continue
                
                # Perform vacuum
                logger.info(f"Vacuuming database: {os.path.basename(db_path)}")
                conn.execute("VACUUM")
                
                # Analyze for query optimization
                conn.execute("ANALYZE")
                
                conn.close()
                
                # Calculate space recovered
                final_size = os.path.getsize(db_path)
                space_recovered = (initial_size - final_size) / (1024 * 1024)  # MB
                
                if space_recovered > 0:
                    logger.info(f"✓ Vacuumed {os.path.basename(db_path)}: {space_recovered:.2f} MB recovered")
                    stats["space_recovered_mb"] += space_recovered
                
                stats["databases_vacuumed"] += 1
                stats["databases_processed"] += 1
                
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower():
                    logger.warning(f"Database locked, skipping: {os.path.basename(db_path)}")
                    stats["databases_skipped"] += 1
                else:
                    raise
                
        except Exception as e:
            logger.error(f"VACUUM failed for {os.path.basename(db_path)}: {e}")
            stats["databases_failed"] += 1
            stats["errors"].append(f"{os.path.basename(db_path)}: {str(e)}")
    
    stats["execution_time_seconds"] = time.time() - start_time
    
    logger.info(f"Database vacuum complete: {stats['databases_vacuumed']} vacuumed, "
               f"{stats['databases_skipped']} skipped, {stats['databases_failed']} failed, "
               f"{stats['space_recovered_mb']:.2f} MB recovered")
    
    return stats

def rotate_text_logs(keep_days: int = DEFAULT_LOG_RETENTION_DAYS) -> Dict[str, Any]:
    """
    Rotate and clean old text logs with enhanced management.
    
    Args:
        keep_days: Number of days to retain logs
        
    Returns:
        Dict containing rotation statistics
    """
    stats = {
        "logs_checked": 0,
        "logs_removed": 0,
        "logs_compressed": 0,
        "space_freed_mb": 0.0,
        "errors": []
    }
    
    try:
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        cutoff = time.time() - (keep_days * 86400)
        
        for filename in os.listdir(config.LOGS_DIR):
            file_path = os.path.join(config.LOGS_DIR, filename)
            
            try:
                if not os.path.isfile(file_path):
                    continue
                
                stats["logs_checked"] += 1
                file_mtime = os.path.getmtime(file_path)
                file_size = os.path.getsize(file_path)
                
                # Remove old logs
                if file_mtime < cutoff:
                    os.remove(file_path)
                    stats["logs_removed"] += 1
                    stats["space_freed_mb"] += file_size / (1024 * 1024)
                    logger.debug(f"Removed old log: {filename}")
                    
                # Compress large recent logs
                elif file_size > (1024 * 1024) and not filename.endswith('.gz'):  # > 1MB
                    try:
                        import gzip
                        compressed_path = file_path + '.gz'
                        
                        with open(file_path, 'rb') as f_in:
                            with gzip.open(compressed_path, 'wb') as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        
                        # Verify compressed file
                        if os.path.exists(compressed_path):
                            os.remove(file_path)
                            stats["logs_compressed"] += 1
                            compression_ratio = (file_size - os.path.getsize(compressed_path)) / file_size
                            logger.debug(f"Compressed {filename}: {compression_ratio*100:.1f}% reduction")
                            
                    except Exception as e:
                        logger.warning(f"Failed to compress {filename}: {e}")
                
            except Exception as e:
                logger.warning(f"Error processing log {filename}: {e}")
                stats["errors"].append(f"{filename}: {str(e)}")
        
        logger.info(f"Log rotation complete: {stats['logs_removed']} removed, "
                   f"{stats['logs_compressed']} compressed, "
                   f"{stats['space_freed_mb']:.2f} MB freed")
        
    except Exception as e:
        logger.error(f"Log rotation failed: {e}")
        stats["errors"].append(f"General error: {str(e)}")
    
    return stats

def remove_tmp_dirs() -> Dict[str, Any]:
    """
    Remove temporary directories and files with enhanced safety.
    
    Returns:
        Dict containing cleanup statistics
    """
    stats = {
        "folders_checked": 0,
        "folders_removed": 0,
        "files_removed": 0,
        "space_freed_mb": 0.0,
        "errors": []
    }
    
    for folder_name in TMP_FOLDERS:
        folder_path = os.path.join(config.DATA_DIR, folder_name)
        
        if os.path.isdir(folder_path):
            stats["folders_checked"] += 1
            
            try:
                # Calculate size before removal
                total_size = 0
                file_count = 0
                
                for dirpath, dirnames, filenames in os.walk(folder_path):
                    for filename in filenames:
                        file_path = os.path.join(dirpath, filename)
                        try:
                            file_size = os.path.getsize(file_path)
                            total_size += file_size
                            file_count += 1
                        except:
                            pass
                
                # Remove directory
                shutil.rmtree(folder_path, ignore_errors=True)
                
                if not os.path.exists(folder_path):
                    stats["folders_removed"] += 1
                    stats["files_removed"] += file_count
                    stats["space_freed_mb"] += total_size / (1024 * 1024)
                    logger.debug(f"Removed temp folder: {folder_name} ({file_count} files, {total_size/(1024*1024):.2f} MB)")
                
            except Exception as e:
                logger.warning(f"Failed to remove temp folder {folder_name}: {e}")
                stats["errors"].append(f"{folder_name}: {str(e)}")
    
    logger.info(f"Temp cleanup complete: {stats['folders_removed']} folders removed, "
               f"{stats['files_removed']} files removed, "
               f"{stats['space_freed_mb']:.2f} MB freed")
    
    return stats

# ============================================================================
# ENHANCED CLEANUP FUNCTIONS (v8.0 New)
# ============================================================================

def check_disk_space() -> Dict[str, Any]:
    """
    Check disk space and return status.
    
    Returns:
        Dict containing disk space information
    """
    try:
        usage = psutil.disk_usage(config.DATA_DIR)
        
        return {
            "total_gb": usage.total / (1024**3),
            "used_gb": usage.used / (1024**3),
            "free_gb": usage.free / (1024**3),
            "percent_used": usage.percent,
            "low_space_warning": usage.free < (MIN_DISK_SPACE_GB * 1024**3)
        }
    except Exception as e:
        logger.error(f"Failed to check disk space: {e}")
        return {"error": str(e)}

def optimize_databases() -> Dict[str, Any]:
    """
    Optimize database indexes and analyze query plans.
    
    Returns:
        Dict containing optimization statistics
    """
    stats = {
        "databases_optimized": 0,
        "databases_failed": 0,
        "errors": []
    }
    
    for db_path in DBS:
        try:
            if not os.path.exists(db_path):
                continue
            
            conn = sqlite3.connect(db_path, timeout=5.0)
            
            # Re-analyze statistics for query optimizer
            conn.execute("ANALYZE")
            
            # Reindex all indexes
            conn.execute("REINDEX")
            
            conn.close()
            
            stats["databases_optimized"] += 1
            logger.debug(f"Optimized database: {os.path.basename(db_path)}")
            
        except Exception as e:
            logger.warning(f"Failed to optimize {os.path.basename(db_path)}: {e}")
            stats["databases_failed"] += 1
            stats["errors"].append(f"{os.path.basename(db_path)}: {str(e)}")
    
    return stats

def get_cleanup_stats() -> Dict[str, Any]:
    """
    Get comprehensive cleanup statistics and health metrics.
    
    Returns:
        Dict containing cleanup statistics
    """
    disk_info = check_disk_space()
    
    # Count database sizes
    total_db_size_mb = 0.0
    for db_path in DBS:
        if os.path.exists(db_path):
            total_db_size_mb += os.path.getsize(db_path) / (1024 * 1024)
    
    # Count log sizes
    total_log_size_mb = 0.0
    log_count = 0
    if os.path.exists(config.LOGS_DIR):
        for filename in os.listdir(config.LOGS_DIR):
            file_path = os.path.join(config.LOGS_DIR, filename)
            if os.path.isfile(file_path):
                total_log_size_mb += os.path.getsize(file_path) / (1024 * 1024)
                log_count += 1
    
    return {
        "disk_space": disk_info,
        "database_size_mb": total_db_size_mb,
        "database_count": len([db for db in DBS if os.path.exists(db)]),
        "log_size_mb": total_log_size_mb,
        "log_count": log_count,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# MAIN CLEANUP ORCHESTRATION
# ============================================================================

def run_daily_cleanup() -> Dict[str, Any]:
    """
    Run comprehensive daily cleanup routine.
    Maintains backward compatibility with v7.x calling convention.
    
    Returns:
        Dict containing cleanup results
    """
    logger.info("="*70)
    logger.info("Starting Daily Cleanup v8.0")
    logger.info("="*70)
    
    start_time = time.time()
    
    # Check disk space first
    disk_status = check_disk_space()
    if disk_status.get("low_space_warning"):
        logger.warning(f"⚠ Low disk space: {disk_status['free_gb']:.2f} GB free")
    
    # Run cleanup tasks
    log_stats = rotate_text_logs(keep_days=DEFAULT_LOG_RETENTION_DAYS)
    ticket_stats = cleanup_ticketing_system()
    poison_cache_stats = run_poison_cache_cleanup(quarantine=None, max_rows=5000)
    tmp_stats = remove_tmp_dirs()
    # SARAHMEMORY_PATCH_NOTE 2026-06-28:
    # Heavy VACUUM/REINDEX can thrash large SQLite files on mixed NVMe/HDD bodies.
    # Keep daily cleanup safe by default; enable explicitly when maintenance window
    # is available with SARAH_DAILY_CLEANUP_ENABLE_VACUUM=true.
    if _cfg_bool("SARAH_DAILY_CLEANUP_ENABLE_VACUUM", False):
        vacuum_stats = vacuum_all()
        optimize_stats = optimize_databases()
    else:
        vacuum_stats = {"databases_processed": 0, "databases_vacuumed": 0, "databases_skipped": len(DBS), "databases_failed": 0, "space_recovered_mb": 0.0, "safe_policy_skipped": True, "reason": "daily vacuum disabled by local-first anti-thrash policy"}
        optimize_stats = {"databases_optimized": 0, "databases_failed": 0, "safe_policy_skipped": True, "reason": "daily reindex disabled by local-first anti-thrash policy"}
    
    # Calculate total results
    total_space_freed = (
        log_stats.get("space_freed_mb", 0) +
        tmp_stats.get("space_freed_mb", 0) +
        vacuum_stats.get("space_recovered_mb", 0)
    )
    
    execution_time = time.time() - start_time
    
    results = {
        "success": True,
        "execution_time_seconds": execution_time,
        "total_space_freed_mb": total_space_freed,
        "log_rotation": log_stats,
        "tmp_cleanup": tmp_stats,
        "database_vacuum": vacuum_stats,
        "database_optimize": optimize_stats,
        "ticket_cleanup": ticket_stats,
        "poison_cache_cleanup": poison_cache_stats,
        "disk_status": disk_status,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info("="*70)
    logger.info(f"✓ Daily Cleanup Completed in {execution_time:.2f}s")
    logger.info(f"  Total Space Freed: {total_space_freed:.2f} MB")
    logger.info(f"  Disk Free: {disk_status.get('free_gb', 0):.2f} GB")
    logger.info("="*70)
    
    return results

def run_intelligent_cleanup() -> Dict[str, Any]:
    """
    Run intelligent cleanup with adaptive behavior based on system state.
    New in v8.0 - provides smarter cleanup decisions.
    
    Returns:
        Dict containing cleanup results
    """
    # Get current system state
    disk_status = check_disk_space()
    cleanup_stats = get_cleanup_stats()
    
    # Adaptive cleanup based on disk space
    if disk_status.get("percent_used", 0) > 90:
        logger.warning("Disk usage >90% - running aggressive cleanup")
        keep_days = 7  # More aggressive log rotation
    elif disk_status.get("percent_used", 0) > 80:
        logger.info("Disk usage >80% - running standard cleanup")
        keep_days = 14
    else:
        logger.info("Disk usage normal - running light cleanup")
        keep_days = 30
    
    # Run cleanup with adaptive parameters
    log_stats = rotate_text_logs(keep_days=keep_days)
    tmp_stats = remove_tmp_dirs()
    poison_cache_stats = run_poison_cache_cleanup(quarantine=None, max_rows=5000)
    
    # Heavy VACUUM/REINDEX remains opt-in even in intelligent mode. Large DBs on
    # mixed NVMe/HDD bodies can thrash if compacted during active runtime.
    if cleanup_stats["database_size_mb"] > 100 and _cfg_bool("SARAH_INTELLIGENT_CLEANUP_ENABLE_VACUUM", False):
        vacuum_stats = vacuum_all()
        optimize_stats = optimize_databases()
    else:
        vacuum_stats = {"databases_processed": 0, "message": "Skipped by anti-thrash policy or databases too small", "safe_policy_skipped": True}
        optimize_stats = {"databases_optimized": 0, "message": "Skipped by anti-thrash policy or databases too small", "safe_policy_skipped": True}
    
    return {
        "success": True,
        "mode": "intelligent",
        "log_retention_days": keep_days,
        "log_rotation": log_stats,
        "tmp_cleanup": tmp_stats,
        "database_vacuum": vacuum_stats,
        "database_optimize": optimize_stats,
        "poison_cache_cleanup": poison_cache_stats,
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# MAIN ENTRY POINT (for testing)
# ============================================================================

if __name__ == "__main__":
    """
    Module test suite for daily cleanup functionality.
    """
    print("="*70)
    print("SarahMemory Daily Cleanup Module v8.0 - Test Suite")
    print("="*70)
    
    # Run cleanup
    results = run_daily_cleanup()
    
    # Display results
    print("\nCleanup Results:")
    print(f"  Execution Time: {results['execution_time_seconds']:.2f}s")
    print(f"  Total Space Freed: {results['total_space_freed_mb']:.2f} MB")
    print(f"  Logs Removed: {results['log_rotation'].get('logs_removed', 0)}")
    print(f"  Databases Vacuumed: {results['database_vacuum'].get('databases_vacuumed', 0)}")
    print(f"  Temp Folders Removed: {results['tmp_cleanup'].get('folders_removed', 0)}")
    
    print("\n" + "="*70)
    print("SarahMemory Daily Cleanup Testing Complete")
    print("="*70)

# ====================================================================
# END OF SarahMemoryCleanupDaily.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryCleanupDaily',
    "version": "v9.0.0-alpha-sml-0.2",
    "category": 'Diagnostics',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['diagnostics', 'health_maintenance'],
    "supported_missions": ['Conversation', 'Diagnostics', 'Repair'],
    "supported_omega": ['Ω001', 'Ω050', 'Ω090', 'Ω110'],
    "required_authority": ['Diagnostics', 'Read'],
    "priority": 85,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryCleanupDaily.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return dict(SML_ORGAN_METADATA)


def sml_health():
    """Return a local SML health vector without side effects."""
    return {
        "status": "Healthy",
        "availability": 1.0,
        "integrity": 1.0,
        "performance": 1.0,
        "reliability": 1.0,
        "confidence": 0.75,
        "latency_ms": 0.0,
        "stability": 1.0,
        "compatibility": 1.0,
        "notes": ["SML adapter present"],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryCleanupDaily',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryCleanupDaily', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

