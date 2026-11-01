"""--==The SarahMemory Project==--
File: SarahMemoryCleanupDaily.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.1.0a (Enterprise)
Date: 2026-11-01
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
# VALIDATION_DATE = "2026-11-01"
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
import stat, uuid, gzip, hashlib, copy, threading, contextlib
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
    """Unique temporary path prevents multiple maintenance workers overwriting .tmp."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp = path + '.' + uuid.uuid4().hex + '.tmp'
    try:
        with open(temp, 'x', encoding='utf-8') as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.remove(temp)




# v9.1.0a conservative retention contract: daily scheduled calls inspect by
# default; a human-triggered approved call may archive eligible artifacts.
_DAILY_LOCK = threading.RLock()
_MAX_FILES_PER_PASS = 300
_MAX_DIRECT_FILE_BYTES = 64 * 1024 * 1024
_MAX_WALK_DEPTH = 3
_MIN_RETENTION_SECONDS = DEFAULT_TMP_RETENTION_HOURS * 3600
_PROTECTED_LOG_TOKENS = ('audit', 'security', 'governance', 'operator', 'assurance', 'ledger', 'truth', 'memory', 'arile')


def _safe_beneath(path: str, root: str, *, require_file: bool = False) -> str:
    root = os.path.abspath(root)
    path = os.path.abspath(path)
    if path == root or os.path.commonpath((root, path)) != root:
        raise PermissionError('maintenance_root_boundary_denied')
    if os.path.islink(root):
        raise PermissionError('maintenance_symlink_root_denied')
    relative = os.path.relpath(path, root)
    walk = root
    for item in relative.split(os.sep):
        walk = os.path.join(walk, item)
        if os.path.islink(walk):
            raise PermissionError('maintenance_symlink_denied')
    if require_file:
        st = os.lstat(path)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink > 1:
            raise PermissionError('maintenance_non_regular_file_denied')
    return path


def _authorize_daily(operation: str, target: str, *, user_consented: bool):
    if not user_consented:
        raise PermissionError('explicit_operator_confirmation_required')
    try:
        from SarahMemoryOperatorCore import authorize_external_domain_action
        decision = authorize_external_domain_action(
            'SarahMemory governed daily maintenance: ' + operation,
            origin='SarahMemoryCleanupDaily',
            proposed_action={'action_type': operation, 'target': target, 'target_ref': target, 'capability_name': 'filesystem_maintenance', 'required_permissions': ['filesystem_maintenance', 'operator_approval'], 'risk_level': 'high'},
            meta={'scope': 'bounded_local_retention'}, user_consented=True,
        )
    except Exception as exc:
        raise PermissionError('operator_core_authorization_unavailable') from exc
    if not isinstance(decision, dict) or not (decision.get('ok') is True and decision.get('allow') is True and decision.get('domain_execution_authorized') is True and str(decision.get('decision')).upper() == 'ALLOW'):
        raise PermissionError('operator_core_denied_maintenance')


def _is_log_candidate(name: str) -> bool:
    lowered = name.lower()
    return (lowered.endswith(('.log', '.txt')) and not any(token in lowered for token in _PROTECTED_LOG_TOKENS) and not lowered.startswith('.'))


def _file_hash(path: str) -> str:
    dig = hashlib.sha256()
    with open(path, 'rb') as src:
        for block in iter(lambda: src.read(1024 * 1024), b''):
            dig.update(block)
    return dig.hexdigest()


def _safe_archive_move(src: str, dst_dir: str, *, root: str) -> str:
    """Recoverable same-filesystem archive; never overwrite another artifact."""
    _safe_beneath(src, root, require_file=True)
    os.makedirs(dst_dir, exist_ok=True)
    _safe_beneath(dst_dir, root)
    dst = os.path.join(dst_dir, Path(src).name + '.' + uuid.uuid4().hex[:12])
    if os.path.lexists(dst):
        raise FileExistsError('archive_destination_collision')
    shutil.move(src, dst)
    return dst


def _daily_error_result(stats, error):
    stats['errors'].append(str(error))
    stats['ok'] = False
    return stats


def _daily_tasks_ok(*items) -> bool:
    for item in items:
        if not isinstance(item, dict):
            return False
        if item.get('ok') is False or item.get('success') is False or item.get('errors') or item.get('error'):
            return False
    return True



def cleanup_ticketing_system(archive_old: Optional[bool] = None, keep_days: int = 30, *, user_consented: bool = False) -> Dict[str, Any]:
    """Bounded ticket inventory; archive only with independently approved consent."""
    if archive_old is None:
        archive_old = _cfg_bool('SARAH_CLEANUP_TICKET_ARCHIVE_ENABLED', False)
    keep_days = max(1, min(3650, int(keep_days)))
    data_dir = os.path.abspath(getattr(config, 'DATA_DIR', os.path.join(os.getcwd(), 'data')))
    audit_dir = os.path.join(data_dir, 'audit')
    stats = {'ok': True, 'archive_old': bool(archive_old), 'keep_days': keep_days, 'scanned': 0, 'archived': 0, 'by_folder': {}, 'errors': [], 'deleted': 0, 'safe_policy_skipped': not bool(archive_old), 'scan_limited': False}
    if archive_old:
        _authorize_daily('archive_tickets', data_dir, user_consented=user_consented)
        from SarahMemoryCleanup import _require_audit_event
        _require_audit_event('archive_tickets_authorized', {'root': data_dir})
    watched = [os.path.join(data_dir, x) for x in ('devbridge', 'reports', 'repair_outbox', 'audit/reports', 'audit/repair_outbox')]
    cutoff = time.time() - keep_days * 86400
    budget = _MAX_FILES_PER_PASS
    with _DAILY_LOCK:
        for root in watched:
            if not os.path.isdir(root):
                continue
            try:
                _safe_beneath(root, data_dir)
            except PermissionError as exc:
                stats['errors'].append(str(exc))
                continue
            for current, dirs, names in os.walk(root, topdown=True, followlinks=False):
                depth = len(Path(current).relative_to(root).parts)
                dirs[:] = [d for d in sorted(dirs) if not os.path.islink(os.path.join(current, d)) and depth < _MAX_WALK_DEPTH][:24]
                for name in sorted(names):
                    if budget <= 0:
                        stats['scan_limited'] = True
                        break
                    if not name.lower().endswith(('.json', '.jsonl', '.txt', '.log')):
                        continue
                    budget -= 1
                    path = os.path.join(current, name)
                    try:
                        _safe_beneath(path, data_dir, require_file=True)
                        stats['scanned'] += 1
                        top = os.path.relpath(current, data_dir).split(os.sep)[0]
                        stats['by_folder'][top] = stats['by_folder'].get(top, 0) + 1
                        if archive_old and 'processed' in Path(current).parts and os.path.getmtime(path) < cutoff:
                            _safe_archive_move(path, os.path.join(audit_dir, 'archived_tickets', top), root=data_dir)
                            stats['archived'] += 1
                    except Exception as exc:
                        stats['errors'].append(f'{name}: {exc}')
                if budget <= 0:
                    break
            if budget <= 0:
                break
    stats['timestamp'] = datetime.now().isoformat()
    stats['ok'] = not stats['errors']
    # One daily summary filename instead of a rapidly growing report storm.
    if os.path.isdir(audit_dir):
        try:
            _safe_write_json(os.path.join(audit_dir, 'summaries', 'ticket_cleanup_inventory_' + datetime.now().strftime('%Y%m%d') + '.json'), stats)
        except Exception as exc:
            stats['errors'].append(str(exc))
            stats['ok'] = False
    return stats



# ============================================================================
# SEMANTIC QA CACHE HYGIENE - v9.0.0
# ============================================================================
def run_poison_cache_cleanup(*, quarantine: Optional[bool] = None, max_rows: int = 5000) -> Dict[str, Any]:
    """Daily inspection only; never derive human consent from an environment flag."""
    if quarantine is None:
        quarantine = _cfg_bool('SARAH_DAILY_CLEANUP_QUARANTINE_POISON_CACHE', False)
    enabled = _cfg_bool('SARAH_DAILY_CLEANUP_ENABLE_POISON_SCAN', False) or bool(quarantine)
    out = {'ok': True, 'enabled': enabled, 'quarantine': bool(quarantine), 'max_rows': min(5000, max(0, int(max_rows))), 'errors': []}
    if not enabled:
        out.update({'safe_policy_skipped': True, 'reason': 'scan disabled by default'})
        return out
    try:
        import SarahMemoryCleanup as cleanup
        result = cleanup.scan_poisoned_qa_cache(max_rows=out['max_rows'])
        out['result'] = result
        out['ok'] = bool(result.get('ok')) and not bool(result.get('errors'))
        if quarantine:
            # Do not report the requested quarantine as performed: scheduled
            # invocation cannot establish an authenticated human approval.
            out.update({'ok': False, 'quarantine_denied': True})
            out['errors'].append('scheduled_quarantine_requires_operator_approval')
    except Exception as exc:
        out['ok'] = False
        out['errors'].append(str(exc))
    return out



# ============================================================================
# CORE CLEANUP FUNCTIONS (v8.0 Enhanced)
# ============================================================================

def vacuum_all(*, user_consented: bool = False) -> Dict[str, Any]:
    """Explicitly authorized bounded vacuum; preserve existing journal modes."""
    stats = {'ok': True, 'databases_processed': 0, 'databases_vacuumed': 0, 'databases_skipped': 0, 'databases_failed': 0, 'space_recovered_mb': 0.0, 'errors': []}
    if not user_consented:
        stats.update({'safe_policy_skipped': True, 'reason': 'operator consent required', 'databases_skipped': len(DBS)})
        return stats
    _authorize_daily('vacuum_sqlite', config.DATASETS_DIR, user_consented=user_consented)
    from SarahMemoryCleanup import _require_audit_event
    _require_audit_event('vacuum_authorized', {'root': config.DATASETS_DIR})
    start = time.monotonic()
    with _DAILY_LOCK:
        for path in DBS:
            if time.monotonic() - start > 30:
                stats['errors'].append('maintenance_time_budget_exceeded')
                break
            if not os.path.isfile(path):
                stats['databases_skipped'] += 1
                continue
            try:
                _safe_beneath(path, config.DATASETS_DIR, require_file=True)
                before = os.path.getsize(path)
                if before < VACUUM_SIZE_THRESHOLD_MB * 1024**2:
                    stats['databases_skipped'] += 1
                    continue
                if shutil.disk_usage(config.DATA_DIR).free < max(before * 3, MIN_DISK_SPACE_GB * 1024**3):
                    stats['databases_skipped'] += 1
                    continue
                from SarahMemoryCleanup import _backup_core
                _backup_core(names=[os.path.basename(path)])
                with sqlite3.connect(path, timeout=1.5) as con:
                    con.execute('PRAGMA busy_timeout=1500')
                    if con.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                        raise ValueError('database_integrity_failed')
                    con.execute('VACUUM')
                    con.execute('PRAGMA optimize')
                after = os.path.getsize(path)
                stats['space_recovered_mb'] += max(0, before - after) / 1024**2
                stats['databases_vacuumed'] += 1
                stats['databases_processed'] += 1
            except Exception as exc:
                stats['databases_failed'] += 1
                stats['errors'].append(f'{os.path.basename(path)}: {exc}')
    stats['execution_time_seconds'] = time.monotonic() - start
    stats['ok'] = not stats['errors']
    return stats


def rotate_text_logs(keep_days: int = DEFAULT_LOG_RETENTION_DAYS, *, user_consented: bool = False) -> Dict[str, Any]:
    """Approved log-only retention, with reversible archival and no evidence deletion."""
    stats = {'ok': True, 'logs_checked': 0, 'logs_removed': 0, 'logs_compressed': 0, 'space_freed_mb': 0.0, 'errors': [], 'archived': 0}
    try:
        days = int(keep_days)
        if not 1 <= days <= 3650:
            raise ValueError('invalid_log_retention_days')
    except Exception as exc:
        return _daily_error_result(stats, exc)
    if not user_consented:
        stats.update({'safe_policy_skipped': True, 'reason': 'scheduled retention cannot assume human approval'})
        return stats
    _authorize_daily('rotate_log_files', config.LOGS_DIR, user_consented=user_consented)
    from SarahMemoryCleanup import _require_audit_event
    _require_audit_event('log_rotation_authorized', {'root': config.LOGS_DIR, 'days': days})
    if not os.path.isdir(config.LOGS_DIR):
        return stats
    cutoff = time.time() - days * 86400
    archive_root = os.path.join(config.LOGS_DIR, '_retired')
    with _DAILY_LOCK:
        for name in sorted(os.listdir(config.LOGS_DIR))[:_MAX_FILES_PER_PASS]:
            if not _is_log_candidate(name):
                continue
            source = os.path.join(config.LOGS_DIR, name)
            try:
                _safe_beneath(source, config.LOGS_DIR, require_file=True)
                size = os.path.getsize(source)
                stats['logs_checked'] += 1
                if size > _MAX_DIRECT_FILE_BYTES:
                    stats['errors'].append(f'{name}: oversized_log_requires_manual_review')
                    continue
                if os.path.getmtime(source) < cutoff:
                    _safe_archive_move(source, archive_root, root=config.LOGS_DIR)
                    stats['archived'] += 1
                    stats['logs_removed'] += 1
                elif size > 1024**2:
                    archive_path = os.path.join(archive_root, name + '.' + uuid.uuid4().hex[:12] + '.gz')
                    os.makedirs(archive_root, exist_ok=True)
                    _safe_beneath(archive_path, config.LOGS_DIR)
                    with open(source, 'rb') as src, gzip.open(archive_path, 'xb') as dst:
                        shutil.copyfileobj(src, dst, length=1024**2)
                    with gzip.open(archive_path, 'rb') as check:
                        h = hashlib.sha256()
                        for block in iter(lambda: check.read(1024**2), b''):
                            h.update(block)
                    if h.hexdigest() != _file_hash(source):
                        os.remove(archive_path)
                        raise ValueError('compressed_archive_integrity_failure')
                    stats['logs_compressed'] += 1
                    # Original remains in place; archive is additional recoverability.
            except Exception as exc:
                stats['errors'].append(f'{name}: {exc}')
    stats['ok'] = not stats['errors']
    return stats


def remove_tmp_dirs(*, user_consented: bool = False) -> Dict[str, Any]:
    """Bounded age-aware quarantine of temp files; never recursively rmtree."""
    stats = {'ok': True, 'folders_checked': 0, 'folders_removed': 0, 'files_removed': 0, 'space_freed_mb': 0.0, 'errors': [], 'files_quarantined': 0, 'bytes_quarantined': 0}
    if not user_consented:
        stats.update({'safe_policy_skipped': True, 'reason': 'scheduled temporary cleanup cannot assume human approval'})
        return stats
    _authorize_daily('quarantine_expired_temp', config.DATA_DIR, user_consented=user_consented)
    from SarahMemoryCleanup import _require_audit_event
    _require_audit_event('temp_quarantine_authorized', {'root': config.DATA_DIR})
    cutoff = time.time() - _MIN_RETENTION_SECONDS
    dest_root = os.path.join(config.DATA_DIR, 'maintenance_quarantine', 'temp')
    budget = _MAX_FILES_PER_PASS
    with _DAILY_LOCK:
        for folder in TMP_FOLDERS:
            source_root = os.path.join(config.DATA_DIR, folder)
            if not os.path.isdir(source_root):
                continue
            try:
                _safe_beneath(source_root, config.DATA_DIR)
            except Exception as exc:
                stats['errors'].append(f'{folder}: {exc}')
                continue
            stats['folders_checked'] += 1
            # Do not walk recursively: only one level of explicit temporary files.
            for name in sorted(os.listdir(source_root)):
                if budget <= 0:
                    stats['scan_limited'] = True
                    break
                budget -= 1
                path = os.path.join(source_root, name)
                try:
                    if not os.path.isfile(path) or os.path.islink(path):
                        continue
                    _safe_beneath(path, config.DATA_DIR, require_file=True)
                    if name.lower().endswith(('.db', '.sqlite', '.db-wal', '.db-shm', '.jsonl', '.py', '.lock')):
                        continue
                    if os.path.getmtime(path) >= cutoff:
                        continue
                    bytes_moved = os.path.getsize(path)
                    if bytes_moved > _MAX_DIRECT_FILE_BYTES:
                        continue
                    _safe_archive_move(path, dest_root, root=config.DATA_DIR)
                    stats['files_removed'] += 1
                    stats['files_quarantined'] += 1
                    stats['bytes_quarantined'] += bytes_moved
                except Exception as exc:
                    stats['errors'].append(f'{name}: {exc}')
            if budget <= 0:
                break
    stats['ok'] = not stats['errors']
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

def optimize_databases(*, user_consented: bool = False) -> Dict[str, Any]:
    """Opt-in PRAGMA optimize; REINDEX is excluded from scheduled operations."""
    stats = {'ok': True, 'databases_optimized': 0, 'databases_failed': 0, 'errors': []}
    if not user_consented:
        stats.update({'safe_policy_skipped': True, 'reason': 'operator consent required'})
        return stats
    _authorize_daily('optimize_sqlite', config.DATASETS_DIR, user_consented=user_consented)
    from SarahMemoryCleanup import _require_audit_event
    _require_audit_event('sqlite_optimize_authorized', {'root': config.DATASETS_DIR})
    with _DAILY_LOCK:
        for path in DBS:
            if not os.path.isfile(path):
                continue
            try:
                _safe_beneath(path, config.DATASETS_DIR, require_file=True)
                from SarahMemoryCleanup import _backup_core
                _backup_core(names=[os.path.basename(path)])
                with sqlite3.connect(path, timeout=1.5) as con:
                    if con.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                        raise sqlite3.DatabaseError('database_integrity_failed')
                    con.execute('PRAGMA optimize')
                stats['databases_optimized'] += 1
            except Exception as exc:
                stats['databases_failed'] += 1
                stats['errors'].append(f'{os.path.basename(path)}: {exc}')
    stats['ok'] = not stats['errors']
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
    """Non-destructive daily inventory. Never assumes unattended delete approval."""
    start = time.monotonic()
    disk = check_disk_space()
    logs = rotate_text_logs(keep_days=DEFAULT_LOG_RETENTION_DAYS)
    tickets = cleanup_ticketing_system(archive_old=False)
    poison = run_poison_cache_cleanup(quarantine=False, max_rows=5000)
    tmp = remove_tmp_dirs()
    vacuum = vacuum_all()
    optimize = optimize_databases()
    ok = _daily_tasks_ok(disk, logs, tickets, poison, tmp, vacuum, optimize)
    return {'success': ok, 'execution_time_seconds': time.monotonic() - start, 'total_space_freed_mb': 0.0, 'log_rotation': logs, 'tmp_cleanup': tmp, 'database_vacuum': vacuum, 'database_optimize': optimize, 'ticket_cleanup': tickets, 'poison_cache_cleanup': poison, 'disk_status': disk, 'timestamp': datetime.now().isoformat(), 'mode': 'scheduled_safe_inventory', 'destructive_actions_performed': False}


def run_intelligent_cleanup() -> Dict[str, Any]:
    """Resource-aware, non-destructive inventory; no aggressive silent retention."""
    disk = check_disk_space()
    stats = get_cleanup_stats()
    used = float(disk.get('percent_used') or 0)
    keep_days = 7 if used > 90 else 14 if used > 80 else 30
    logs = rotate_text_logs(keep_days=keep_days)
    tmp = remove_tmp_dirs()
    poison = run_poison_cache_cleanup(quarantine=False, max_rows=5000)
    vacuum = vacuum_all()
    optimize = optimize_databases()
    return {'success': _daily_tasks_ok(disk, stats, logs, tmp, poison, vacuum, optimize), 'mode': 'intelligent_safe_inventory', 'log_retention_days': keep_days, 'log_rotation': logs, 'tmp_cleanup': tmp, 'database_vacuum': vacuum, 'database_optimize': optimize, 'poison_cache_cleanup': poison, 'disk_status': disk, 'timestamp': datetime.now().isoformat(), 'destructive_actions_performed': False}


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
    "version": "v9.1.0a-alpha-sml-0.2",
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
    return copy.deepcopy(SML_ORGAN_METADATA)


def sml_health():
    """Report inspected paths, not invented performance or reliability metrics."""
    root_ok = os.path.isdir(config.DATA_DIR)
    datasets_ok = os.path.isdir(config.DATASETS_DIR)
    return {
        'status': 'Ready' if root_ok and datasets_ok else 'Degraded',
        'availability': 1.0 if root_ok and datasets_ok else 0.0,
        'integrity': None, 'performance': None, 'reliability': None,
        'confidence': 0.5, 'latency_ms': None, 'stability': None, 'compatibility': None,
        'notes': ['Scheduled maintenance is inventory-only without consent', 'Retained log and cache data are never silently deleted'],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryCleanupDaily',
        "sml_adapter": True,
        "metadata": copy.deepcopy(SML_ORGAN_METADATA),
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

