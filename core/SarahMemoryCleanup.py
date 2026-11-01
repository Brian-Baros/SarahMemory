"""--==The SarahMemory Project==--
File: SarahMemoryCleanup.py
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
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "C"
# ROLE = "maintenance_tool"
# CATEGORY = "cleanup_and_restore"
# USER_FACING = True
# UI_EXPOSURE = "direct_screen_candidate"
# DEPLOYMENT_TARGET = "classic_ui"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "filesystem"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "cleanup_gui"
# FAMILY = "maintenance"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = True
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-11-01"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "User-facing cleanup, backup, restore, and log maintenance GUI for databases and local data retention management."
# --- SARAHMETA END ---

import os, sqlite3, shutil, time, traceback, json, re, sys
import hashlib, uuid, threading, copy, stat, contextlib, logging, zipfile
from datetime import datetime, timedelta
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

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


try:
    from PIL import Image, ImageTk  # optional for icons
except Exception:
    Image = ImageTk = None

try:
    import SarahMemoryGlobals as config
except Exception:
    class config:
        BASE_DIR = os.getcwd()
        DATA_DIR = os.path.join(BASE_DIR, "data")
        MEMORY_DIR = os.path.join(DATA_DIR, "memory")
        DATASETS_DIR = os.path.join(MEMORY_DIR, "datasets")
        LOGS_DIR = os.path.join(DATA_DIR, "logs")

BACKUP_DIR = os.path.join(config.DATASETS_DIR, "_backups")
# Backup directories are created only for an explicitly requested backup.

DBS = {
    "context_history.db": {
        "path": os.path.join(config.DATASETS_DIR, "context_history.db"),
        "ranges": [("context_history","timestamp")]
    },
    "ai_learning.db": {
        "path": os.path.join(config.DATASETS_DIR, "ai_learning.db"),
        "ranges": [("intent_logs","timestamp")]
    },
    "personality1.db": {
        "path": os.path.join(config.DATASETS_DIR, "personality1.db"),
        "ranges": [("emotion_states","timestamp")]  # `responses` may lack timestamp; cleared on ALL only
    },
    "functions.db": {
        "path": os.path.join(config.DATASETS_DIR, "functions.db"),
        "ranges": [("dl_cache","ts")]
    },
    "system_logs.db": {
        "path": os.path.join(config.DATASETS_DIR, "system_logs.db"),
        "ranges": [("events","timestamp"), ("response","timestamp"), ("responses","timestamp")]
    },
    # Additional known datasets: backup/restore only, never blanket SQL clear.
    "sarah_main.db": {"path": os.path.join(config.DATASETS_DIR, "sarah_main.db"), "ranges": []},
    "embeddings.db": {"path": os.path.join(config.DATASETS_DIR, "embeddings.db"), "ranges": []},
    "ailearning.db": {"path": os.path.join(config.DATASETS_DIR, "ailearning.db"), "ranges": []},
}



# v9.1.0a: this organ owns local maintenance only, not authorization.
_MAINTENANCE_LOCK = threading.RLock()
_MAX_BACKUP_BYTES = 4 * 1024**3
_MAX_SCAN_ROWS = 5000
_MAX_QA_RESULTS = 500
_MAINTENANCE_TIMEOUT_SECONDS = 8.0


def _within_root(path: str, root: str, *, allow_root: bool = False) -> str:
    """Reject symlink traversal, traversal escape and non-regular targets."""
    root_abs = os.path.abspath(root)
    path_abs = os.path.abspath(path)
    if not (path_abs == root_abs if allow_root else os.path.commonpath((root_abs, path_abs)) == root_abs and path_abs != root_abs):
        raise PermissionError("maintenance_path_outside_approved_root")
    relative = os.path.relpath(path_abs, root_abs)
    current = root_abs
    if os.path.islink(root_abs):
        raise PermissionError("maintenance_root_symlink_denied")
    for component in (() if relative == '.' else relative.split(os.sep)):
        current = os.path.join(current, component)
        if os.path.islink(current):
            raise PermissionError("maintenance_symlink_denied")
    return path_abs


def _regular_file(path: str, root: str) -> str:
    path = _within_root(path, root)
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode) or st.st_nlink > 1:
        raise PermissionError("maintenance_non_regular_or_linked_file_denied")
    return path


def _sha256_file(path: str) -> str:
    dig = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            dig.update(chunk)
    return dig.hexdigest()


def _sqlite_snapshot(src: str, dst: str) -> dict:
    """SQLite online backup captures WAL transactions; never copy a live .db raw."""
    _regular_file(src, config.DATASETS_DIR)
    if os.path.getsize(src) > _MAX_BACKUP_BYTES:
        raise ValueError("database_exceeds_backup_budget")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    source = target = None
    try:
        source = sqlite3.connect('file:' + Path(src).resolve().as_posix() + '?mode=ro', uri=True, timeout=2.0)
        target = sqlite3.connect(dst, timeout=2.0)
        started = time.monotonic()
        def progress(_status, _remaining, _total):
            if time.monotonic() - started > _MAINTENANCE_TIMEOUT_SECONDS:
                raise TimeoutError("sqlite_backup_time_budget_exceeded")
        source.backup(target, pages=128, sleep=0.04, progress=progress)
        check = target.execute('PRAGMA quick_check').fetchone()
        if not check or check[0] != 'ok':
            raise sqlite3.DatabaseError("backup_integrity_check_failed")
        target.commit()
    finally:
        if target is not None:
            target.close()
        if source is not None:
            source.close()
    return {'bytes': os.path.getsize(dst), 'sha256': _sha256_file(dst)}


def _operator_authorize(action: str, target: str, *, user_consented: bool) -> dict:
    if not user_consented:
        raise PermissionError("explicit_operator_confirmation_required")
    try:
        from SarahMemoryOperatorCore import authorize_external_domain_action
        decision = authorize_external_domain_action(
            'Governed SarahMemory maintenance: ' + action,
            origin='SarahMemoryCleanup',
            proposed_action={
                'action_type': action,
                'capability_name': 'filesystem_maintenance',
                'target': target,
                'target_ref': target,
                'required_permissions': ['filesystem_maintenance', 'operator_approval'],
                'risk_level': 'high',
            },
            meta={'scope': 'local_known_maintenance_targets', 'authority_owner': 'SarahMemoryOperatorCore'},
            user_consented=True,
        )
    except Exception as exc:
        raise PermissionError('operator_core_authorization_unavailable') from exc
    if not isinstance(decision, dict) or not all((decision.get('ok') is True, decision.get('allow') is True, decision.get('domain_execution_authorized') is True, str(decision.get('decision')).upper() == 'ALLOW')):
        raise PermissionError('operator_core_denied_maintenance')
    return decision


@contextlib.contextmanager
def _exclusive_maintenance():
    """Serialize maintenance within this process; no cross-process ownership claim."""
    with _MAINTENANCE_LOCK:
        yield


def _snapshot_logs(bundle: str, manifest: dict) -> None:
    """Optional bounded logs.zip companion, preserving the legacy archive name."""
    if not os.path.isdir(config.LOGS_DIR):
        return
    path = os.path.join(bundle, 'logs.zip')
    files = []
    bytes_total = 0
    for name in sorted(os.listdir(config.LOGS_DIR))[:200]:
        candidate = os.path.join(config.LOGS_DIR, name)
        if not name.lower().endswith(('.log', '.txt')) or not os.path.isfile(candidate):
            continue
        try:
            _regular_file(candidate, config.LOGS_DIR)
            size = os.path.getsize(candidate)
            if size > 32 * 1024**2 or bytes_total + size > 256 * 1024**2:
                manifest.setdefault('warnings', []).append('log_archive_budget_reached')
                break
            files.append((name, candidate))
            bytes_total += size
        except (OSError, PermissionError):
            continue
    if not files:
        return
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
        for name, candidate in files:
            archive.write(candidate, arcname=name)
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError('log_backup_archive_corrupt')
    manifest['supplementary'] = [{'name': 'logs.zip', 'bytes': os.path.getsize(path), 'sha256': _sha256_file(path), 'files': len(files)}]


def _backup_core(*, names=None, audit=True) -> str:
    """Build a recoverable verified bundle.  Never touches live production files."""
    with _exclusive_maintenance():
        os.makedirs(BACKUP_DIR, exist_ok=True)
        _within_root(BACKUP_DIR, config.DATASETS_DIR)
        bundle = os.path.join(BACKUP_DIR, 'backup-' + datetime.utcnow().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:12])
        os.mkdir(bundle)
        manifest = {'schema': 'SarahMemory.cleanup.backup.v1', 'created_at': datetime.utcnow().isoformat(), 'files': [], 'purpose': 'SQLite-consistent pre-change backup', 'complete': False}
        try:
            source_names = list(DBS) if names is None else list(names)
            if len(source_names) > 12:
                raise ValueError('backup_file_count_budget')
            for name in source_names:
                if name not in DBS:
                    raise ValueError('unknown_database_name')
                source = DBS[name]['path']
                if not os.path.exists(source):
                    continue
                dst = os.path.join(bundle, name)
                evidence = _sqlite_snapshot(source, dst)
                if evidence['bytes'] > _MAX_BACKUP_BYTES:
                    raise ValueError('snapshot_exceeds_backup_budget')
                manifest['files'].append({'name': name, **evidence})
            if not manifest['files']:
                raise FileNotFoundError('no_known_databases_to_back_up')
            if names is None:
                _snapshot_logs(bundle, manifest)
            manifest['complete'] = True
            temp = os.path.join(bundle, 'backup_manifest.json.tmp')
            with open(temp, 'w', encoding='utf-8') as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
            os.replace(temp, os.path.join(bundle, 'backup_manifest.json'))
        except Exception:
            shutil.rmtree(bundle, ignore_errors=True)
            raise
        if audit:
            _require_audit_event('backup_created', {'bundle': bundle, 'database_count': len(manifest['files'])})
        return bundle


def _verified_bundle(folder: str) -> list:
    folder = _within_root(folder, BACKUP_DIR)
    if not os.path.isdir(folder):
        raise ValueError('backup_folder_missing')
    manifest_path = _regular_file(os.path.join(folder, 'backup_manifest.json'), folder)
    if os.path.getsize(manifest_path) > 128 * 1024:
        raise ValueError('manifest_too_large')
    with open(manifest_path, 'r', encoding='utf-8') as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict) or manifest.get('schema') != 'SarahMemory.cleanup.backup.v1' or manifest.get('complete') is not True:
        raise ValueError('untrusted_or_incomplete_backup_manifest')
    files = manifest.get('files')
    if not isinstance(files, list) or not files or len(files) > 12:
        raise ValueError('invalid_backup_file_inventory')
    supplements = manifest.get('supplementary') or []
    if not isinstance(supplements, list) or len(supplements) > 1:
        raise ValueError('invalid_supplementary_backup_inventory')
    for item in supplements:
        if not isinstance(item, dict) or item.get('name') != 'logs.zip':
            raise ValueError('invalid_backup_supplement')
        path = _regular_file(os.path.join(folder, 'logs.zip'), folder)
        if os.path.getsize(path) != item.get('bytes') or _sha256_file(path) != item.get('sha256'):
            raise ValueError('backup_supplement_checksum_mismatch')
    verified = []
    seen = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError('invalid_backup_entry')
        name = entry.get('name')
        if name not in DBS or name in seen:
            raise ValueError('unapproved_or_duplicate_backup_file')
        seen.add(name)
        path = _regular_file(os.path.join(folder, name), folder)
        if os.path.getsize(path) != entry.get('bytes') or _sha256_file(path) != entry.get('sha256'):
            raise ValueError('backup_checksum_mismatch')
        with sqlite3.connect('file:' + Path(path).resolve().as_posix() + '?mode=ro', uri=True, timeout=1.0) as db:
            if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError('backup_database_integrity_failure')
        verified.append((name, path))
    return verified


def _restore_bundle(folder: str, *, user_consented: bool) -> dict:
    """Verify all sources then use SQLite online restore, retaining recovery snapshots."""
    _operator_authorize('restore_sqlite_backup', str(folder), user_consented=user_consented)
    with _exclusive_maintenance():
        verified = _verified_bundle(folder)
        _require_audit_event('restore_preflight_authorized', {'bundle': folder, 'count': len(verified)})
        # Create a new, independent recovery point before modifying any live DB.
        recovery = _backup_core(names=[name for name, _ in verified], audit=True)
        restored = []
        try:
            for name, src in verified:
                dst = DBS[name]['path']
                if not os.path.isfile(dst):
                    raise FileNotFoundError('live_database_missing_for_restore:' + name)
                _regular_file(dst, config.DATASETS_DIR)
                with sqlite3.connect('file:' + Path(src).resolve().as_posix() + '?mode=ro', uri=True, timeout=2.0) as source:
                    with sqlite3.connect(dst, timeout=2.0) as target:
                        source.backup(target, pages=128, sleep=0.05)
                restored.append(name)
            _cleanup_audit_event('backup_restored', {'bundle': folder, 'recovery': recovery, 'restored': restored})
            return {'ok': True, 'restored': restored, 'recovery_bundle': recovery}
        except Exception as exc:
            # Recovery backup is never erased.  Attempts to restore are reported.
            recovered = []
            errors = []
            for name, path in _verified_bundle(recovery):
                # Also restore the in-progress database, whose pages may have
                # changed before the failing backup() call raised.
                if name not in [item_name for item_name, _ in verified]:
                    continue
                try:
                    with sqlite3.connect('file:' + Path(path).resolve().as_posix() + '?mode=ro', uri=True, timeout=2.0) as source:
                        with sqlite3.connect(DBS[name]['path'], timeout=2.0) as target:
                            source.backup(target, pages=128, sleep=0.05)
                    recovered.append(name)
                except Exception as rollback_exc:
                    errors.append(str(rollback_exc))
            _cleanup_audit_event('restore_failed', {'error': str(exc), 'recovery': recovery, 'rolled_back': recovered, 'rollback_errors': errors})
            raise RuntimeError('restore_failed_recovery_preserved:' + recovery) from exc


def _cleanup_audit_event(event_type: str, payload: dict) -> bool:
    """Append a durable local record. Return false if the audit failed."""
    try:
        data_dir = os.path.abspath(getattr(config, 'DATA_DIR', os.getcwd()))
        audit_dir = os.path.join(data_dir, 'audit', 'cleanup')
        os.makedirs(audit_dir, exist_ok=True)
        _within_root(audit_dir, data_dir)
        path = os.path.join(audit_dir, 'cleanup_events.jsonl')
        _within_root(path, data_dir)
        if os.path.lexists(path):
            _regular_file(path, data_dir)
        row = {'event_type': event_type, 'timestamp': datetime.utcnow().isoformat(), **(payload or {})}
        serialized = (json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + '\n').encode('utf-8')
        if len(serialized) > 16384:
            raise ValueError('audit_event_budget_exceeded')
        flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY
        if hasattr(os, 'O_NOFOLLOW'):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, 'ab') as fh:
            fh.write(serialized)
            fh.flush()
            os.fsync(fh.fileno())
        return True
    except Exception:
        return False


def _require_audit_event(event_type: str, payload: dict) -> None:
    if not _cleanup_audit_event(event_type, payload):
        raise PermissionError('required_maintenance_audit_unavailable')


def _safe_sql_identifier(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


# ---------------------------------------------------------------------------
# QA CACHE POISON QUARANTINE - v9 Semantic Memory Hygiene
# ---------------------------------------------------------------------------
# These helpers do not perform broad filesystem scans. They inspect known
# project SQLite DBs only, backup before mutation, quarantine before delete, and
# write an audit event.  This is cleanup, not answer routing authority.

def _qa_cache_tables(cur) -> list[str]:
    try:
        rows = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    except Exception:
        return []
    names = []
    for r in rows or []:
        name = str(r[0] or "")
        if name.lower() in {"qa_cache", "qacache", "sm_qa_cache"}:
            names.append(name)
    return names


def _qa_cache_columns(cur, table: str) -> dict:
    try:
        cols = [str(r[1]) for r in cur.execute(f"PRAGMA table_info({_safe_sql_identifier(table)})").fetchall()]
    except Exception:
        cols = []
    low = {c.lower(): c for c in cols}
    return {
        "all": cols,
        "query": low.get("query") or low.get("question") or low.get("prompt") or low.get("user_input") or low.get("input"),
        "answer": low.get("ai_answer") or low.get("answer") or low.get("response") or low.get("reply") or low.get("content") or low.get("output"),
        "timestamp": low.get("timestamp") or low.get("created_at") or low.get("ts"),
    }


def _normal_text(value) -> str:
    try:
        return re.sub(r"\s+", " ", str(value or "").replace("\x00", " ")).strip()
    except Exception:
        return str(value or "").strip()


def _definition_subject(query: str) -> str:
    q = _normal_text(query).lower().strip(" ?.!")
    for pat in (
        r"^what\s+is\s+an?\s+(.+)$",
        r"^what\s+is\s+(.+)$",
        r"^what\s+are\s+(.+)$",
        r"^define\s+(.+)$",
        r"^explain\s+(.+)$",
        r"^describe\s+(.+)$",
        r"^tell\s+me\s+about\s+(.+)$",
    ):
        m = re.match(pat, q)
        if m:
            return re.sub(r"\b(the|a|an)\b", " ", m.group(1)).strip()
    return ""


def _qa_poison_reason(query: str, answer: str) -> str:
    """Conservative, explainable classifier. Human review required before quarantine."""
    raw = str(answer or '').strip()
    low = raw.lower()
    if not raw:
        return 'empty_answer'
    if any(x in low for x in ('request denied by policy', 'user confirmation required', 'unable to answer')):
        return 'cached_failure_or_policy_text'
    q = _normal_text(query).lower()
    if 'python' in q and (low.startswith('emscripten') or 'llvm-to-javascript compiler' in low):
        return 'wrong_topic_python_emscripten_blob'
    # Generic academic terms alone do not establish that a response is poisoned.
    return ''



def scan_poisoned_qa_cache(*, db_names: list[str] | None = None, max_rows: int = 5000) -> dict:
    """Bounded read-only inspection of known database names only."""
    names = db_names if db_names is not None else ['ai_learning.db', 'ailearning.db', 'personality1.db', 'context_history.db', 'functions.db']
    report = {'ok': True, 'mode': 'read_only', 'poisoned': [], 'checked_rows': 0, 'errors': [], 'timestamp': datetime.utcnow().isoformat()}
    remaining = min(_MAX_SCAN_ROWS, max(0, int(max_rows)))
    for dbname in list(names)[:8]:
        if remaining <= 0:
            break
        if dbname not in DBS and dbname != 'ailearning.db':
            if db_names is not None:
                report['errors'].append('unsupported_database_name')
            continue
        path = os.path.join(config.DATASETS_DIR, dbname)
        if not os.path.isfile(path):
            continue
        try:
            _regular_file(path, config.DATASETS_DIR)
            with sqlite3.connect('file:' + Path(path).resolve().as_posix() + '?mode=ro', uri=True, timeout=1.0) as con:
                cur = con.cursor()
                for table in _qa_cache_tables(cur):
                    if remaining <= 0:
                        break
                    columns = _qa_cache_columns(cur, table)
                    if not columns['query'] or not columns['answer']:
                        continue
                    statement = 'SELECT rowid, ' + _safe_sql_identifier(columns['query']) + ', ' + _safe_sql_identifier(columns['answer']) + ' FROM ' + _safe_sql_identifier(table) + ' LIMIT ?'
                    for rowid, question, answer in cur.execute(statement, (remaining,)):
                        report['checked_rows'] += 1
                        remaining -= 1
                        reason = _qa_poison_reason(question, answer)
                        if reason and len(report['poisoned']) < _MAX_QA_RESULTS:
                            report['poisoned'].append({'db': dbname, 'table': table, 'rowid': int(rowid), 'query': _normal_text(question)[:240], 'reason': reason, 'answer_preview': _normal_text(answer)[:260]})
        except Exception as exc:
            report['errors'].append(f'{dbname}: {exc}')
    report['poison_count'] = len(report['poisoned'])
    report['ok'] = not report['errors']
    return report



def _backup_db_for_quarantine(db_path: str) -> str:
    """Legacy compatibility: online snapshot of a known database."""
    path = _regular_file(db_path, config.DATASETS_DIR)
    name = os.path.basename(path)
    if name not in DBS:
        raise ValueError('quarantine_unknown_database')
    bundle = _backup_core(names=[name])
    return os.path.join(bundle, name)



def quarantine_poisoned_qa_cache(*, dry_run: bool = True, max_rows: int = 5000, backup: bool = True, user_consented: bool = False) -> dict:
    """Candidate quarantine. Explicit consent and OperatorCore ALLOW mandatory."""
    scan = scan_poisoned_qa_cache(max_rows=max_rows)
    out = {'ok': bool(scan.get('ok')), 'dry_run': bool(dry_run), 'scan': scan, 'quarantined': 0, 'deleted': 0, 'backups': [], 'errors': list(scan.get('errors') or []), 'timestamp': datetime.utcnow().isoformat()}
    if dry_run or not scan.get('poisoned'):
        _cleanup_audit_event('qa_poison_scan', {'dry_run': True, 'poison_count': scan.get('poison_count', 0)})
        return out
    if not backup:
        raise PermissionError('quarantine_requires_verified_backup')
    if not scan.get('ok'):
        raise RuntimeError('quarantine_scan_incomplete')
    _operator_authorize('qa_cache_quarantine', config.DATASETS_DIR, user_consented=user_consented)
    _require_audit_event('qa_cache_quarantine_authorized', {'count': len(scan['poisoned'])})
    with _exclusive_maintenance():
        by_db = {}
        for item in scan['poisoned']:
            by_db.setdefault(item['db'], []).append(item)
        for dbname, items in by_db.items():
            path = _regular_file(os.path.join(config.DATASETS_DIR, dbname), config.DATASETS_DIR)
            backup_path = _backup_db_for_quarantine(path)
            out['backups'].append(backup_path)
            con = None
            try:
                con = sqlite3.connect(path, timeout=2.0)
                con.execute('BEGIN IMMEDIATE')
                con.execute('CREATE TABLE IF NOT EXISTS qa_cache_poison_quarantine (id INTEGER PRIMARY KEY AUTOINCREMENT, source_db TEXT, source_table TEXT, source_rowid INTEGER, query TEXT, ai_answer TEXT, reason TEXT, quarantined_at TEXT)')
                for item in items[:_MAX_QA_RESULTS]:
                    table = item['table']
                    if table not in _qa_cache_tables(con.cursor()):
                        continue
                    cols = _qa_cache_columns(con.cursor(), table)
                    if not cols['query'] or not cols['answer']:
                        continue
                    stmt = 'SELECT ' + _safe_sql_identifier(cols['query']) + ', ' + _safe_sql_identifier(cols['answer']) + ' FROM ' + _safe_sql_identifier(table) + ' WHERE rowid=?'
                    row = con.execute(stmt, (int(item['rowid']),)).fetchone()
                    if not row or _qa_poison_reason(row[0], row[1]) != item['reason']:
                        continue
                    con.execute('INSERT INTO qa_cache_poison_quarantine (source_db, source_table, source_rowid, query, ai_answer, reason, quarantined_at) VALUES (?, ?, ?, ?, ?, ?, ?)', (dbname, table, item['rowid'], str(row[0] or ''), str(row[1] or ''), item['reason'], datetime.utcnow().isoformat()))
                    con.execute('DELETE FROM ' + _safe_sql_identifier(table) + ' WHERE rowid=?', (item['rowid'],))
                    out['quarantined'] += 1
                    out['deleted'] += 1
                con.commit()
            except Exception as exc:
                if con is not None:
                    con.rollback()
                out['ok'] = False
                out['errors'].append(str(exc))
            finally:
                if con is not None:
                    con.close()
    _cleanup_audit_event('qa_poison_quarantine', {'quarantined': out['quarantined'], 'backups': out['backups'], 'errors': out['errors']})
    return out



RANGES = [
    ("5 minutes", 5*60),
    ("10 minutes", 10*60),
    ("30 minutes", 30*60),
    ("1 hour", 60*60),
    ("3 hours", 3*60*60),
    ("5 hours", 5*60*60),
    ("12 hours", 12*60*60),
    ("1 day", 24*60*60),
    ("3 days", 3*24*60*60),
    ("1 week", 7*24*60*60),
    ("1 month (~30d)", 30*24*60*60),
    ("3 months", 90*24*60*60),
    ("6 months", 180*24*60*60),
    ("1 year", 365*24*60*60),
    ("ALL DATA", None),
]

def backup_all():
    """Create a verified consistent snapshot; UI alerts best effort."""
    bundle = _backup_core()
    try:
        messagebox.showinfo('Backup', f'Verified backup created: {bundle}')
    except Exception:
        pass
    return bundle


def restore_backup():
    """Interactive restore only. Path selection alone never constitutes permission."""
    folder = filedialog.askdirectory(initialdir=BACKUP_DIR, title='Select verified SarahMemory backup')
    if not folder:
        return {'ok': False, 'cancelled': True}
    if not messagebox.askyesno('Confirm restore', 'Restore a verified backup? A separate recovery snapshot will be created first.'):
        return {'ok': False, 'cancelled': True}
    try:
        result = _restore_bundle(folder, user_consented=True)
    except Exception as exc:
        messagebox.showerror('Restore denied or failed', str(exc))
        return {'ok': False, 'error': str(exc)}
    messagebox.showinfo('Restore', 'Restored: ' + ', '.join(result['restored']))
    return result


def restore_latest(*, user_consented: bool = False):
    """Bridge-compatible restore; GET request alone has no consent authority."""
    _operator_authorize('restore_latest_database_backup', BACKUP_DIR, user_consented=user_consented)
    if not os.path.isdir(BACKUP_DIR):
        raise FileNotFoundError('no_verified_backups')
    candidates = []
    for name in sorted(os.listdir(BACKUP_DIR), reverse=True)[:100]:
        if not name.startswith('backup-'):
            continue
        path = os.path.join(BACKUP_DIR, name)
        try:
            _verified_bundle(path)
            candidates.append(path)
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError('no_valid_verified_backup')
    return _restore_bundle(candidates[0], user_consented=user_consented)


def clear_range(seconds=None, cutoff_seconds=None, *, user_consented: bool = False):
    """Governed selected-DB clear with legacy Bridge two-positional support.

    Bare calls, including legacy GET/POST API requests without consent, fail closed.
    """
    if isinstance(seconds, str):
        db_name, seconds = seconds, cutoff_seconds
    else:
        db_name = None
        if cutoff_seconds is not None:
            raise ValueError('invalid_clear_range_arguments')
    if seconds is not None and (not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds <= 0 or seconds > 10 * 365 * 86400):
        raise ValueError('invalid_retention_seconds')
    names = [db_name] if db_name else list(DBS)
    if any(name not in DBS for name in names):
        raise ValueError('unapproved_database_name')
    _operator_authorize('clear_database_range', ','.join(names), user_consented=user_consented)
    _require_audit_event('clear_range_authorized', {'names': names, 'seconds': seconds})
    cutoff = None if seconds is None else datetime.utcnow() - timedelta(seconds=float(seconds))
    backup = _backup_core(names=names)
    result = {'ok': True, 'backup': backup, 'cutoff': cutoff.isoformat() if cutoff else None, 'deleted': 0, 'errors': []}
    with _exclusive_maintenance():
        for name in names:
            path = DBS[name]['path']
            if not os.path.isfile(path):
                continue
            _regular_file(path, config.DATASETS_DIR)
            try:
                with sqlite3.connect(path, timeout=2.0) as con:
                    con.execute('BEGIN IMMEDIATE')
                    available_tables = set(r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'"))
                    for table, tscol in DBS[name]['ranges']:
                        if table not in available_tables:
                            continue
                        if seconds is None:
                            cur = con.execute('DELETE FROM ' + _safe_sql_identifier(table))
                        else:
                            cols = set(r[1] for r in con.execute('PRAGMA table_info(' + _safe_sql_identifier(table) + ')'))
                            if tscol not in cols:
                                continue
                            cur = con.execute('DELETE FROM ' + _safe_sql_identifier(table) + ' WHERE ' + _safe_sql_identifier(tscol) + ' >= ?', (cutoff.isoformat(),))
                        result['deleted'] += max(0, cur.rowcount)
            except Exception as exc:
                result['ok'] = False
                result['errors'].append(f'{name}: {exc}')
    _cleanup_audit_event('clear_range', {'names': names, 'cutoff': result['cutoff'], 'backup': backup, 'deleted': result['deleted'], 'ok': result['ok'], 'errors': result['errors']})
    if not result['ok']:
        raise RuntimeError('clear_range_incomplete_backup_preserved:' + backup)
    return result


def tidy_logs(*, user_consented: bool = False):
    """Bounded log truncation with root protection, copy-on-write and approval."""
    _operator_authorize('tidy_logs', config.LOGS_DIR, user_consented=user_consented)
    _require_audit_event('tidy_logs_authorized', {'root': config.LOGS_DIR})
    checked = truncated = 0
    errors = []
    with _exclusive_maintenance():
        if not os.path.isdir(config.LOGS_DIR):
            return {'ok': True, 'checked': 0, 'truncated': 0}
        for name in sorted(os.listdir(config.LOGS_DIR))[:300]:
            if not name.lower().endswith('.log') or any(x in name.lower() for x in ('audit', 'security', 'governance')):
                continue
            path = os.path.join(config.LOGS_DIR, name)
            try:
                _regular_file(path, config.LOGS_DIR)
                checked += 1
                if os.path.getsize(path) <= 5 * 1024 * 1024:
                    continue
                # Preserve existing history; rotate into uniquely named archival copy.
                unique = path + '.preserved-' + uuid.uuid4().hex[:12]
                shutil.copy2(path, unique)
                if _sha256_file(path) != _sha256_file(unique):
                    os.remove(unique)
                    raise OSError('archive_copy_integrity_failed')
                with open(path, 'rb') as fh:
                    fh.seek(-1024 * 1024, os.SEEK_END)
                    tail = fh.read()
                tmp = path + '.staged-' + uuid.uuid4().hex
                try:
                    with open(tmp, 'xb') as fh:
                        fh.write(b'...[prior log preserved in adjacent archive]\n' + tail)
                    os.replace(tmp, path)
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                truncated += 1
            except (OSError, PermissionError) as exc:
                errors.append(f'{name}: {exc}')
                _cleanup_audit_event('log_tidy_rejected', {'name': name, 'reason': str(exc)})
                continue
    _cleanup_audit_event('tidy_logs', {'checked': checked, 'truncated': truncated, 'errors': errors})
    try:
        messagebox.showinfo('Logs', f'Log files checked: {checked}; safely rotated: {truncated}')
    except Exception:
        pass
    return {'ok': not errors, 'checked': checked, 'truncated': truncated, 'errors': errors}


def launch_cleanup_gui():
    root = tk.Tk()
    root.title("SarahMemory Cleanup & Restore")
    root.geometry("520x520")
    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="Select a time range to clear across DBs").pack(pady=6)

    range_var = tk.StringVar(value=RANGES[0][0])
    combo = ttk.Combobox(frm, textvariable=range_var, values=[r[0] for r in RANGES], state="readonly")
    combo.pack(pady=6, fill="x")

    btns = ttk.Frame(frm)
    btns.pack(fill="x", pady=10)
    ttk.Button(btns, text="Create Backup", command=backup_all).pack(side="left", padx=4)
    ttk.Button(btns, text="Restore Backup", command=restore_backup).pack(side="left", padx=4)
    def on_tidy():
        if messagebox.askyesno('Confirm log maintenance', 'Preserve large logs before truncating?'):
            try:
                result = tidy_logs(user_consented=True)
                if not result.get('ok'):
                    messagebox.showerror('Log maintenance', '; '.join(result.get('errors') or []))
            except Exception as exc:
                messagebox.showerror('Log maintenance denied', str(exc))
    ttk.Button(btns, text="Tidy Logs", command=on_tidy).pack(side="left", padx=4)

    def on_clear():
        label = range_var.get()
        seconds = next((s for lbl, s in RANGES if lbl == label), None)
        if messagebox.askyesno('Confirm', f'Proceed to clear: {label}?'):
            try:
                result = clear_range(seconds, user_consented=True)
                messagebox.showinfo('Cleanup', f"Deleted {result['deleted']} rows; recovery: {result['backup']}")
            except Exception as exc:
                messagebox.showerror('Cleanup denied or failed', str(exc))

    def on_scan_poison():
        report = scan_poisoned_qa_cache(max_rows=5000)
        messagebox.showinfo("Poison Cache Scan", f"Checked rows: {report.get('checked_rows', 0)}\nPoison rows: {report.get('poison_count', 0)}")

    def on_quarantine_poison():
        if messagebox.askyesno("Confirm", "Backup and quarantine poisoned QA cache rows? This does not vacuum databases."):
            try:
                report = quarantine_poisoned_qa_cache(dry_run=False, max_rows=5000, backup=True, user_consented=True)
                messagebox.showinfo('Poison Cache Quarantine', f"Quarantined: {report.get('quarantined', 0)}\nBackups: {len(report.get('backups') or [])}\nErrors: {len(report.get('errors') or [])}")
            except Exception as exc:
                messagebox.showerror('QA quarantine denied', str(exc))

    ttk.Button(frm, text="SCAN POISONED QA CACHE", command=on_scan_poison).pack(pady=4, fill="x")
    ttk.Button(frm, text="QUARANTINE POISONED QA CACHE", command=on_quarantine_poison).pack(pady=4, fill="x")

    ttk.Button(frm, text="CLEAR SELECTED RANGE", command=on_clear).pack(pady=16, fill="x")

    ttk.Label(frm, text="DB Directory: " + config.DATASETS_DIR).pack(anchor="w", pady=6)
    ttk.Label(frm, text="Logs Directory: " + config.LOGS_DIR).pack(anchor="w")

    root.mainloop()

if __name__ == "__main__":
    if "--scan-poison" in sys.argv:
        print(json.dumps(scan_poisoned_qa_cache(max_rows=5000), indent=2, ensure_ascii=False, default=str))
    elif "--quarantine-poison" in sys.argv:
        print(json.dumps(quarantine_poisoned_qa_cache(dry_run=False, max_rows=5000, backup=True, user_consented=True), indent=2, ensure_ascii=False, default=str))
    elif "--dry-run-quarantine-poison" in sys.argv:
        print(json.dumps(quarantine_poisoned_qa_cache(dry_run=True, max_rows=5000, backup=True), indent=2, ensure_ascii=False, default=str))
    else:
        launch_cleanup_gui()

# ====================================================================
# END OF SarahMemoryCleanup.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryCleanup',
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
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryCleanup.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return copy.deepcopy(SML_ORGAN_METADATA)


def sml_health():
    """No fabricated maintenance results: report only local readiness evidence."""
    datasets = bool(os.path.isdir(config.DATASETS_DIR))
    backup_root = bool(os.path.isdir(BACKUP_DIR))
    return {
        'status': 'Ready' if datasets else 'Degraded',
        'availability': 1.0 if datasets else 0.0,
        'integrity': None, 'performance': None, 'reliability': None,
        'confidence': 0.5, 'latency_ms': None, 'stability': None, 'compatibility': None,
        'notes': ['Known SQLite paths configured', 'Backups exist' if backup_root else 'No backup directory available', 'Destructive actions require explicit OperatorCore ALLOW'],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryCleanup',
        "sml_adapter": True,
        "metadata": copy.deepcopy(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryCleanup', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

