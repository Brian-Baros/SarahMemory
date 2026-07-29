"""--==The SarahMemory Project==--
File: SarahMemorySync.py
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

PHASE C ENHANCEMENT:
mobile app synchronization infrastructure in addition to the original Dropbox and
FTPS sync capabilities.

v9.0.0 Local-First Security Patch:
- Offline/local sync works without Dropbox, FTPS, SarahNet, cloud APIs, or a centralized AI provider.
- Remote sync is fail-closed unless explicitly armed by the user/UI/operator.
- AI agents, external packets, model output, and broker messages cannot self-authorize sync.
- Core file transfer from SarahNet is staged by default, not applied directly.
- Every allow/deny/defer/stage decision emits an audit event.
- Phase C has a local SQLite fallback when SarahMemory_PhaseC_Sync.py is absent.
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "sync_engine"
# CATEGORY = "cross_device_sync"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "sync"
# HARDWARE_DOMAIN = "network_filesystem"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "sync"
# FAMILY = "device_link"
# GOVERNANCE_LEVEL = "bounded"
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
# NOTES = "Local-first sync engine with Dropbox/FTPS/SarahNet behind explicit online arming, local Phase C fallback, anti-agent hijack gates, staged core file intake, and audit-first governance."
# --- SARAHMETA END ---

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:  # pragma: no cover
    config = None  # type: ignore

try:
    from SarahMemoryAudit import audit_event  # type: ignore
except Exception:  # pragma: no cover
    def audit_event(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {"ok": False, "audit_unavailable": True}

logger = logging.getLogger("SarahMemorySync")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.NullHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

MODULE_VERSION = "9.0.0-local-first-security.1"
SYNC_VERSION = MODULE_VERSION
_DECISION_ALLOW = "ALLOW"
_DECISION_DENY = "DENY"
_DECISION_DEFER = "DEFER"
_DECISION_STAGE = "STAGE_ONLY"

# SARAHMEMORY_PATCH_NOTE: These phrases identify authority-inversion or
# AI-agent-hijack attempts. They do not block normal text storage by themselves;
# they block attempts to use remote/model/agent language as execution authority.
_AGENT_HIJACK_MARKERS = (
    "ignore governance", "bypass governance", "override governance", "self authorize",
    "self-authorize", "autonomous agent", "ai agent", "tool call authority",
    "execute without approval", "silent apply", "disable safety", "disable audit",
    "root command", "remote shell", "apply core patch", "overwrite core",
)

_SAFE_ACTIONS = {
    "local_sync", "phase_c_sync", "device_register", "read_contacts", "read_history",
    "read_reminders", "dropbox_upload", "dropbox_download", "ftps_connect",
    "sarahnet_push", "sarahnet_poll", "sarahnet_stage", "sarahnet_apply_core",
    "test", "audit", "status",
}


def _cfg(name: str, default: Any = None) -> Any:
    try:
        return getattr(config, name, default)
    except Exception:
        return default


def _env_flag(name: str, default: str = "false") -> bool:
    try:
        return str(os.getenv(name, default)).strip().lower() in ("1", "true", "yes", "on")
    except Exception:
        return False


def _base_dir() -> str:
    # SARAHMEMORY_PATCH_NOTE: BASE_DIR is the install root. Never use os.getcwd()
    # for authority because Windows shortcuts, terminals, services, and cloud WSGI
    # can launch from unrelated folders.
    return os.path.abspath(str(_cfg("BASE_DIR", Path(__file__).resolve().parent.parent)))


def _core_dir() -> str:
    return os.path.abspath(str(_cfg("CORE_DIR", os.path.join(_base_dir(), "core"))))


def _data_dir() -> str:
    return os.path.abspath(str(_cfg("DATA_DIR", os.path.join(_base_dir(), "data"))))


def _datasets_dir() -> str:
    return os.path.abspath(str(_cfg("DATASETS_DIR", os.path.join(_data_dir(), "memory", "datasets"))))


def _sync_dir() -> str:
    # SARAHMEMORY_PATCH_NOTE: Local sync storage lives under data/sync, not cwd.
    # This preserves drive-letter portability and avoids writing into random shells.
    return os.path.abspath(str(_cfg("SYNC_DIR", os.path.join(_data_dir(), "sync"))))


def _audit(action: str, verdict: str, details: Optional[Dict[str, Any]] = None, *, risk: str = "low", source: str = "SarahMemorySync") -> Dict[str, Any]:
    return audit_event("sync", action, verdict, details or {}, risk=risk, source=source, actor="SarahMemorySync")


def _sync_db_path() -> str:
    return os.path.join(_datasets_dir(), "device_link.db")


def _connect_sync_db() -> sqlite3.Connection:
    path = _sync_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path, timeout=15)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def _ensure_sync_tables() -> None:
    con = _connect_sync_db()
    try:
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sync_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                details TEXT,
                verdict TEXT DEFAULT 'INFO',
                event_hash TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sync_devices (
                device_id TEXT PRIMARY KEY,
                user_id TEXT,
                device_name TEXT,
                device_type TEXT,
                metadata_json TEXT,
                created_ts INTEGER,
                updated_ts INTEGER
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sync_contacts (
                user_id TEXT,
                contact_id TEXT,
                payload_json TEXT,
                last_modified_timestamp INTEGER,
                PRIMARY KEY(user_id, contact_id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sync_history (
                user_id TEXT,
                history_id TEXT,
                payload_json TEXT,
                last_modified_timestamp INTEGER,
                PRIMARY KEY(user_id, history_id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sync_reminders (
                user_id TEXT,
                reminder_id TEXT,
                payload_json TEXT,
                last_modified_timestamp INTEGER,
                completed INTEGER DEFAULT 0,
                PRIMARY KEY(user_id, reminder_id)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS code_sync_state(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_id TEXT,
                peer_id TEXT,
                relpath TEXT,
                sha256 TEXT,
                mtime REAL,
                size INTEGER,
                last_sent_ts REAL,
                last_recv_ts REAL,
                staged_path TEXT,
                decision TEXT
            )
        """)
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_code_sync_state_uq ON code_sync_state(node_id, peer_id, relpath)")
        con.commit()
    finally:
        con.close()


def log_sync_event(event: str, details: Any, verdict: str = "INFO") -> None:
    """Log sync activity to SQLite and central audit.

    # SARAHMEMORY_PATCH_NOTE: This is the local audit trail for Sync. It is not a
    # permission system; it records decisions already made by governance gates.
    """
    try:
        _ensure_sync_tables()
        payload = json.dumps(details if isinstance(details, (dict, list)) else {"message": str(details)}, sort_keys=True, default=str)
        event_hash = hashlib.sha256((str(event) + payload + str(time.time())).encode("utf-8", "ignore")).hexdigest()
        con = _connect_sync_db()
        try:
            con.execute(
                "INSERT INTO sync_events(timestamp,event,details,verdict,event_hash) VALUES(?,?,?,?,?)",
                (datetime.utcnow().isoformat(timespec="seconds") + "Z", str(event)[:128], payload[:16000], str(verdict)[:64], event_hash),
            )
            con.commit()
        finally:
            con.close()
    except Exception as exc:
        logger.error("Error logging sync event: %s", exc)
    _audit(str(event)[:128], str(verdict or "INFO"), {"details": details})


# Initialize local directories at import. Directory creation is bounded and local.
LOCAL_SYNC_DIR = _sync_dir()
DROPBOX_SYNC_FOLDER = "/SarahMemorySync"
try:
    os.makedirs(LOCAL_SYNC_DIR, exist_ok=True)
except Exception:
    pass


def _online_session_token_valid(session_token: Optional[str] = None) -> bool:
    token = (session_token or os.getenv("SARAH_ONLINE_SESSION_TOKEN") or str(_cfg("ONLINE_SESSION_TOKEN", ""))).strip()
    if len(token) < 24:
        return False
    # SARAHMEMORY_PATCH_NOTE: Token is an arming proof, not authentication alone.
    # Remote endpoints still require their own auth/signature checks.
    return True


def _online_armed(session_token: Optional[str] = None, user_approved: bool = False) -> bool:
    feature_flag = bool(_cfg("REMOTE_SYNC_ENABLED", False)) or _env_flag("SARAH_REMOTE_SYNC_ENABLED", "false")
    session_flag = bool(_cfg("ONLINE_SESSION_ARMED", False)) or _env_flag("SARAH_ONLINE_SESSION_ARMED", "false")
    return bool(feature_flag and session_flag and (user_approved or _online_session_token_valid(session_token)))


def _contains_hijack_marker(value: Any) -> bool:
    text = json.dumps(value, default=str).lower() if not isinstance(value, str) else value.lower()
    return any(marker in text for marker in _AGENT_HIJACK_MARKERS)


def sm_sync_governance_review(
    action: str,
    *,
    remote: bool = False,
    source: str = "internal",
    payload: Optional[Dict[str, Any]] = None,
    risk: str = "low",
    user_approved: bool = False,
    session_token: Optional[str] = None,
    write_core: bool = False,
) -> Dict[str, Any]:
    """Review a sync action before it performs filesystem/network work.

    # SARAHMEMORY_PATCH_NOTE: This is the Sync Court. It does not replace global
    # governance; it adds a required local guard because Sync touches files,
    # databases, network brokers, and cross-device state.
    """
    action = str(action or "unknown").strip().lower()
    reasons: List[str] = []
    verdict = _DECISION_ALLOW

    if action not in _SAFE_ACTIONS:
        verdict = _DECISION_DENY
        reasons.append("unknown_sync_action")

    if _contains_hijack_marker(payload or {}) or _contains_hijack_marker(source):
        verdict = _DECISION_DENY
        reasons.append("anti_agent_hijack_marker_detected")

    if remote and not _online_armed(session_token=session_token, user_approved=user_approved):
        verdict = _DECISION_DENY
        reasons.append("remote_sync_requires_explicit_user_ui_online_arming_and_session_token")

    if write_core and not user_approved:
        verdict = _DECISION_DENY
        reasons.append("core_write_requires_explicit_human_approval")

    if bool(_cfg("LOCAL_ONLY_MODE", True)) and remote and not user_approved:
        verdict = _DECISION_DENY
        reasons.append("LOCAL_ONLY_MODE_blocks_unapproved_remote_sync")

    packet = {
        "ok": verdict == _DECISION_ALLOW,
        "verdict": verdict,
        "action": action,
        "remote": bool(remote),
        "source": source,
        "risk": risk,
        "write_core": bool(write_core),
        "user_approved": bool(user_approved),
        "reasons": reasons,
        "local_first": True,
        "anti_agent_hijack_enabled": True,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    _audit(action, verdict, packet, risk=risk, source=source)
    return packet


# -----------------------------------------------------------------------------
# LEGACY DROPBOX SYNC - DISABLED UNLESS EXPLICITLY ARMED
# -----------------------------------------------------------------------------
try:
    import dropbox  # type: ignore
    from dropbox.files import WriteMode  # type: ignore
except Exception:
    dropbox = None  # type: ignore
    WriteMode = None  # type: ignore

DROPBOX_ACCESS_TOKEN = os.environ.get("DROPBOX_ACCESS_TOKEN", "").strip() or None


def _dropbox_ready() -> bool:
    return bool(dropbox is not None and DROPBOX_ACCESS_TOKEN)


def sync_to_dropbox(file_path: str, dbx: Any = None, *, user_approved: bool = False, session_token: Optional[str] = None) -> bool:
    review = sm_sync_governance_review("dropbox_upload", remote=True, source="dropbox", payload={"file_path": file_path}, risk="medium", user_approved=user_approved, session_token=session_token)
    if not review["ok"]:
        log_sync_event("Dropbox Upload Denied", review, verdict="DENY")
        return False
    try:
        if not _dropbox_ready():
            raise RuntimeError("Dropbox SDK/token unavailable")
        dbx = dbx or dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)  # type: ignore[attr-defined]
        rel = os.path.relpath(os.path.abspath(file_path), LOCAL_SYNC_DIR).replace(os.sep, "/")
        if rel.startswith(".."):
            raise ValueError("file_path escapes LOCAL_SYNC_DIR")
        dropbox_path = (DROPBOX_SYNC_FOLDER + "/" + rel).replace("//", "/")
        with open(file_path, "rb") as f:
            dbx.files_upload(f.read(), dropbox_path, mode=WriteMode("overwrite"))  # type: ignore[misc]
        log_sync_event("Dropbox Upload", {"file": file_path, "dropbox_path": dropbox_path}, verdict="ALLOW")
        return True
    except Exception as exc:
        log_sync_event("Dropbox Upload Error", str(exc), verdict="DENY")
        return False


def sync_from_dropbox(file_path: str, dbx: Any = None, *, user_approved: bool = False, session_token: Optional[str] = None) -> bool:
    review = sm_sync_governance_review("dropbox_download", remote=True, source="dropbox", payload={"file_path": file_path}, risk="medium", user_approved=user_approved, session_token=session_token)
    if not review["ok"]:
        log_sync_event("Dropbox Download Denied", review, verdict="DENY")
        return False
    try:
        if not _dropbox_ready():
            raise RuntimeError("Dropbox SDK/token unavailable")
        dbx = dbx or dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)  # type: ignore[attr-defined]
        rel = os.path.relpath(os.path.abspath(file_path), LOCAL_SYNC_DIR).replace(os.sep, "/")
        if rel.startswith(".."):
            raise ValueError("file_path escapes LOCAL_SYNC_DIR")
        dropbox_path = (DROPBOX_SYNC_FOLDER + "/" + rel).replace("//", "/")
        _metadata, res = dbx.files_download(dropbox_path)
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(res.content)
        log_sync_event("Dropbox Download", {"file": file_path, "dropbox_path": dropbox_path}, verdict="ALLOW")
        return True
    except Exception as exc:
        log_sync_event("Dropbox Download Error", str(exc), verdict="DENY")
        return False


def sync_data(*, user_approved: bool = False, session_token: Optional[str] = None) -> Dict[str, Any]:
    review = sm_sync_governance_review("local_sync", remote=False, source="local", payload={"dir": LOCAL_SYNC_DIR}, risk="low")
    if not review["ok"]:
        return {"ok": False, "review": review}
    count = 0
    errors = 0
    for _root, _dirs, files in os.walk(LOCAL_SYNC_DIR):
        count += len(files)
    # SARAHMEMORY_PATCH_NOTE: Local sync_data does not automatically go online.
    # Dropbox upload requires explicit user/session arming through sync_to_dropbox.
    log_sync_event("Local Sync Inventory", {"files_seen": count, "online_attempted": False}, verdict="ALLOW")
    return {"ok": True, "files_seen": count, "errors": errors, "online_attempted": False}


def start_sync_monitor(interval: int = 300) -> Dict[str, Any]:
    """Start a bounded local-only monitor loop.

    # SARAHMEMORY_PATCH_NOTE: The interval is clamped to at least 300 seconds to
    # prevent old HDD/NVMe thrash. The monitor inventories local sync state only;
    # it does not perform remote uploads without explicit arming.
    """
    interval = max(300, int(interval or 300))
    stop = threading.Event()
    def sync_loop() -> None:
        while not stop.is_set():
            try:
                sync_data()
            except Exception as exc:
                log_sync_event("Sync Monitor Error", str(exc), verdict="DENY")
            stop.wait(interval)
    t = threading.Thread(target=sync_loop, name="SM_SyncMonitor", daemon=True)
    t.start()
    log_sync_event("Sync Monitor Started", {"interval": interval}, verdict="ALLOW")
    return {"ok": True, "thread": t.name, "interval": interval, "remote": False}


# -----------------------------------------------------------------------------
# LEGACY FTPS - DISABLED UNLESS EXPLICITLY ARMED
# -----------------------------------------------------------------------------
def connect_ftps_with_auto_accept(host: str, port: int = 21, user: Optional[str] = None, password: Optional[str] = None, allow_insecure_env: str = "SARAHMEMORY_ALLOW_INSECURE_FTPS", *, user_approved: bool = False, session_token: Optional[str] = None):
    review = sm_sync_governance_review("ftps_connect", remote=True, source="ftps", payload={"host": host, "port": port}, risk="high", user_approved=user_approved, session_token=session_token)
    if not review["ok"]:
        raise RuntimeError("FTPS denied by Sync governance: " + ",".join(review.get("reasons", [])))
    import ssl, ftplib
    u = user or os.getenv("SARAHMEMORY_FTP_USER")
    p = password or os.getenv("SARAHMEMORY_FTP_PASS")
    if not u or not p:
        raise RuntimeError("FTP credentials not set in env (SARAHMEMORY_FTP_USER/PASS)")
    # SARAHMEMORY_PATCH_NOTE: insecure cert acceptance defaults OFF. The old default
    # allowed insecure=True; v9 requires explicit environment opt-in.
    allow_insecure = _env_flag(allow_insecure_env, "false")
    ctx = ssl.create_default_context()
    if allow_insecure:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    ftps = ftplib.FTP_TLS(context=ctx)
    ftps.connect(host, int(port), timeout=15)
    ftps.auth()
    ftps.prot_p()
    ftps.login(u, p)
    log_sync_event("FTPS Connected", {"host": host, "port": port, "insecure": allow_insecure}, verdict="ALLOW")
    return ftps


# -----------------------------------------------------------------------------
# PHASE C LOCAL FALLBACK
# -----------------------------------------------------------------------------
_PHASE_C_BACKEND = "local_sqlite_fallback"
PHASE_C_ENABLED = True
try:
    from SarahMemory_PhaseC_Sync import (  # type: ignore
        get_sync_manager,
        shutdown_sync_manager,
        sync_device_data,
        register_new_device,
        get_device_contacts,
        get_device_history,
        get_device_reminders,
        SYNC_VERSION as _EXT_SYNC_VERSION,
    )
    SYNC_VERSION = str(_EXT_SYNC_VERSION)
    _PHASE_C_BACKEND = "external_module"
except Exception:
    def get_sync_manager() -> Dict[str, Any]:
        _ensure_sync_tables()
        return {"ok": True, "backend": _PHASE_C_BACKEND, "db": _sync_db_path()}

    def shutdown_sync_manager() -> Dict[str, Any]:
        return {"ok": True, "backend": _PHASE_C_BACKEND, "shutdown": "noop"}

    def register_new_device(user_id: str, device_name: str, device_type: str, **kwargs: Any) -> str:
        review = sm_sync_governance_review("device_register", remote=False, source="phase_c_local", payload={"user_id": user_id, "device_name": device_name, "device_type": device_type}, risk="low")
        if not review["ok"]:
            raise RuntimeError("device registration denied")
        _ensure_sync_tables()
        raw = f"{user_id}|{device_name}|{device_type}|{time.time()}|{secrets.token_hex(8)}"
        device_id = "dev_" + hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()[:24]
        now = int(time.time())
        meta = json.dumps(kwargs or {}, sort_keys=True, default=str)
        con = _connect_sync_db()
        try:
            con.execute(
                "INSERT OR REPLACE INTO sync_devices(device_id,user_id,device_name,device_type,metadata_json,created_ts,updated_ts) VALUES(?,?,?,?,?,?,?)",
                (device_id, user_id, device_name, device_type, meta, now, now),
            )
            con.commit()
        finally:
            con.close()
        log_sync_event("Phase C Device Registration", {"user_id": user_id, "device_id": device_id, "backend": _PHASE_C_BACKEND}, verdict="ALLOW")
        return device_id

    def _upsert_payload_table(table: str, user_id: str, item_id_key: str, items: Iterable[Dict[str, Any]]) -> int:
        _ensure_sync_tables()
        count = 0
        con = _connect_sync_db()
        try:
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                item_id = str(item.get(item_id_key) or "").strip()
                if not item_id:
                    continue
                lmt = int(item.get("last_modified_timestamp") or item.get("updated_ts") or time.time())
                completed = int(bool(item.get("completed", item.get("is_completed", False))))
                payload = json.dumps(item, sort_keys=True, default=str)
                if table == "sync_reminders":
                    con.execute(
                        "INSERT INTO sync_reminders(user_id,reminder_id,payload_json,last_modified_timestamp,completed) VALUES(?,?,?,?,?) "
                        "ON CONFLICT(user_id,reminder_id) DO UPDATE SET payload_json=excluded.payload_json,last_modified_timestamp=excluded.last_modified_timestamp,completed=excluded.completed "
                        "WHERE excluded.last_modified_timestamp >= sync_reminders.last_modified_timestamp",
                        (user_id, item_id, payload, lmt, completed),
                    )
                elif table == "sync_contacts":
                    con.execute(
                        "INSERT INTO sync_contacts(user_id,contact_id,payload_json,last_modified_timestamp) VALUES(?,?,?,?) "
                        "ON CONFLICT(user_id,contact_id) DO UPDATE SET payload_json=excluded.payload_json,last_modified_timestamp=excluded.last_modified_timestamp "
                        "WHERE excluded.last_modified_timestamp >= sync_contacts.last_modified_timestamp",
                        (user_id, item_id, payload, lmt),
                    )
                elif table == "sync_history":
                    con.execute(
                        "INSERT INTO sync_history(user_id,history_id,payload_json,last_modified_timestamp) VALUES(?,?,?,?) "
                        "ON CONFLICT(user_id,history_id) DO UPDATE SET payload_json=excluded.payload_json,last_modified_timestamp=excluded.last_modified_timestamp "
                        "WHERE excluded.last_modified_timestamp >= sync_history.last_modified_timestamp",
                        (user_id, item_id, payload, lmt),
                    )
                count += 1
            con.commit()
        finally:
            con.close()
        return count

    def sync_device_data(user_id: str, device_id: str, **kwargs: Any) -> Dict[str, Any]:
        review = sm_sync_governance_review("phase_c_sync", remote=False, source="phase_c_local", payload={"user_id": user_id, "device_id": device_id}, risk="low")
        if not review["ok"]:
            return {"success": False, "error": "denied", "review": review}
        contacts = _upsert_payload_table("sync_contacts", user_id, "contact_id", kwargs.get("contacts") or [])
        history = _upsert_payload_table("sync_history", user_id, "history_id", kwargs.get("history") or [])
        reminders = _upsert_payload_table("sync_reminders", user_id, "reminder_id", kwargs.get("reminders") or [])
        result = {"success": True, "backend": _PHASE_C_BACKEND, "uploaded": {"contacts": contacts, "history": history, "reminders": reminders}, "errors": []}
        log_sync_event("Phase C Local Sync", result, verdict="ALLOW")
        return result

    def _fetch_payloads(table: str, id_col: str, user_id: str, since_timestamp: int = 0, include_completed: bool = True) -> List[Dict[str, Any]]:
        _ensure_sync_tables()
        con = _connect_sync_db()
        try:
            cur = con.cursor()
            if table == "sync_reminders" and not include_completed:
                cur.execute(f"SELECT payload_json FROM {table} WHERE user_id=? AND last_modified_timestamp>=? AND completed=0 ORDER BY last_modified_timestamp ASC", (user_id, int(since_timestamp or 0)))
            else:
                cur.execute(f"SELECT payload_json FROM {table} WHERE user_id=? AND last_modified_timestamp>=? ORDER BY last_modified_timestamp ASC", (user_id, int(since_timestamp or 0)))
            out = []
            for (payload,) in cur.fetchall() or []:
                try:
                    out.append(json.loads(payload or "{}"))
                except Exception:
                    pass
            return out
        finally:
            con.close()

    def get_device_contacts(user_id: str, since_timestamp: int = 0) -> List[Dict[str, Any]]:
        return _fetch_payloads("sync_contacts", "contact_id", user_id, since_timestamp)

    def get_device_history(user_id: str, since_timestamp: int = 0) -> List[Dict[str, Any]]:
        return _fetch_payloads("sync_history", "history_id", user_id, since_timestamp)

    def get_device_reminders(user_id: str, include_completed: bool = False) -> List[Dict[str, Any]]:
        return _fetch_payloads("sync_reminders", "reminder_id", user_id, 0, include_completed=include_completed)


def phase_c_sync_available() -> bool:
    return bool(PHASE_C_ENABLED)


def perform_phase_c_sync(user_id: str, device_id: str, **kwargs: Any) -> Dict[str, Any]:
    try:
        return sync_device_data(user_id, device_id, **kwargs)
    except Exception as exc:
        log_sync_event("Phase C Sync Error", str(exc), verdict="DENY")
        return {"success": False, "error": str(exc)}


def register_phase_c_device(user_id: str, device_name: str, device_type: str, **kwargs: Any) -> Optional[str]:
    try:
        return register_new_device(user_id, device_name, device_type, **kwargs)
    except Exception as exc:
        log_sync_event("Phase C Registration Error", str(exc), verdict="DENY")
        return None


def get_phase_c_contacts(user_id: str, since_timestamp: int = 0) -> List[Dict[str, Any]]:
    try:
        return get_device_contacts(user_id, since_timestamp)
    except Exception:
        return []


def get_phase_c_history(user_id: str, since_timestamp: int = 0) -> List[Dict[str, Any]]:
    try:
        return get_device_history(user_id, since_timestamp)
    except Exception:
        return []


def get_phase_c_reminders(user_id: str, include_completed: bool = False) -> List[Dict[str, Any]]:
    try:
        return get_device_reminders(user_id, include_completed)
    except Exception:
        return []


# -----------------------------------------------------------------------------
# SARAHNET CORE FILE SYNC - STAGE-FIRST, APPLY ONLY WITH HUMAN APPROVAL
# -----------------------------------------------------------------------------
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 256), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_relpath(path: str, base: str) -> str:
    base_abs = os.path.abspath(base)
    p_abs = os.path.abspath(path)
    try:
        common = os.path.commonpath([base_abs, p_abs])
    except Exception:
        common = ""
    if common != base_abs:
        raise ValueError("path escapes base")
    return os.path.relpath(p_abs, base_abs).replace("\\", "/")


def _trusted_peer_ids() -> set[str]:
    out: set[str] = set()
    try:
        peers = _cfg("SARAHNET_PEERS", {}) or {}
        out.update(str(k) for k in peers.keys())
    except Exception:
        pass
    try:
        out.add(str(_cfg("SARAHNET_NODE_ID", "local-node")))
    except Exception:
        pass
    env_peers = os.getenv("SARAHNET_TRUSTED_PEERS", "")
    for part in env_peers.split(","):
        if part.strip():
            out.add(part.strip())
    return out


def _broker_base_url() -> str:
    base = str(_cfg("SARAH_WEB_BASE", "") or "").strip()
    if not base or bool(_cfg("LOCAL_ONLY_MODE", True)):
        base = "http://127.0.0.1:8000"
    return base.rstrip("/")


def _broker_headers(payload_bytes: Optional[bytes] = None) -> dict:
    headers = {"Content-Type": "application/json", "X-SarahNet-Client": "SarahMemorySync"}
    api_key = _cfg("REMOTE_API_KEY", None) or os.getenv("SARAH_REMOTE_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    secret = os.getenv("SARAHNET_SHARED_SECRET", "").strip()
    if secret and payload_bytes is not None:
        sig = hmac.new(secret.encode("utf-8"), payload_bytes, hashlib.sha256).hexdigest()
        headers["X-Sarah-Signature"] = sig
    return headers


def _http_post_json(url: str, payload: dict, timeout: float = 10.0, extra_headers: Optional[dict] = None) -> Tuple[int, dict]:
    raw = json.dumps(payload).encode("utf-8")
    headers = _broker_headers(raw)
    if extra_headers:
        headers.update(extra_headers)
    try:
        import requests  # type: ignore
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"text": r.text}
    except Exception:
        import urllib.request
        req = urllib.request.Request(url, data=raw, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                txt = resp.read().decode("utf-8", "ignore")
                try:
                    return resp.getcode(), json.loads(txt)
                except Exception:
                    return resp.getcode(), {"text": txt}
        except Exception as exc:
            return 599, {"error": str(exc)}


def _http_get_json(url: str, timeout: float = 10.0) -> Tuple[int, dict]:
    headers = _broker_headers(None)
    try:
        import requests  # type: ignore
        r = requests.get(url, headers=headers, timeout=timeout)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, {"text": r.text}
    except Exception:
        import urllib.request
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                txt = resp.read().decode("utf-8", "ignore")
                try:
                    return resp.getcode(), json.loads(txt)
                except Exception:
                    return resp.getcode(), {"text": txt}
        except Exception as exc:
            return 599, {"error": str(exc)}


def _collect_core_files(base_dir: Optional[str] = None) -> List[dict]:
    base = os.path.abspath(base_dir or _base_dir())
    core = os.path.join(base, "core")
    items: List[dict] = []
    for folder in (core, os.path.join(base, "api", "server")):
        if not os.path.isdir(folder):
            continue
        for name in os.listdir(folder):
            if folder == core and not (name.startswith("SarahMemory") and name.endswith(".py")):
                continue
            if folder.endswith(os.path.join("api", "server")) and not (name.startswith("app") and name.endswith(".py")):
                continue
            p = os.path.join(folder, name)
            try:
                st = os.stat(p)
                items.append({"abs": p, "rel": _safe_relpath(p, base), "mtime": st.st_mtime, "size": st.st_size})
            except Exception:
                pass
    return items


def sarahnet_sync_push(peer_id: str, base_dir: Optional[str] = None, max_bytes: int = 1_900_000, *, user_approved: bool = False, session_token: Optional[str] = None) -> dict:
    review = sm_sync_governance_review("sarahnet_push", remote=True, source="sarahnet", payload={"peer_id": peer_id}, risk="high", user_approved=user_approved, session_token=session_token)
    if not review["ok"]:
        return {"ok": False, "review": review}
    base = os.path.abspath(base_dir or _base_dir())
    node_id = str(_cfg("SARAHNET_NODE_ID", "local-node"))
    if peer_id not in _trusted_peer_ids():
        return {"ok": False, "reason": f"peer_id not trusted: {peer_id}"}
    _ensure_sync_tables()
    files = _collect_core_files(base)
    sent = skipped = errors = 0
    details: List[dict] = []
    con = _connect_sync_db()
    cur = con.cursor()
    try:
        for it in files:
            try:
                if int(it["size"]) > int(max_bytes):
                    skipped += 1
                    continue
                rel = it["rel"]
                sha = _sha256_file(it["abs"])
                cur.execute("SELECT sha256, mtime FROM code_sync_state WHERE node_id=? AND peer_id=? AND relpath=? LIMIT 1", (node_id, peer_id, rel))
                row = cur.fetchone()
                if row and row[0] == sha and float(row[1] or 0) >= float(it["mtime"]):
                    skipped += 1
                    continue
                with open(it["abs"], "rb") as f:
                    data_b64 = base64.b64encode(f.read()).decode("ascii")
                payload = {"to_node": peer_id, "from_node": node_id, "filename": f"core_sync::{rel}", "mime": "application/octet-stream", "data_b64": data_b64, "sha256": sha}
                status, resp = _http_post_json(_broker_base_url() + "/api/net/file/send", payload, timeout=float(_cfg("REMOTE_HTTP_TIMEOUT", 10.0)), extra_headers={"X-SarahNet-Channel": "sync", "X-Sarah-Human-Approved": "true"})
                if status == 200 and resp.get("ok"):
                    sent += 1
                    cur.execute(
                        "INSERT INTO code_sync_state(node_id,peer_id,relpath,sha256,mtime,size,last_sent_ts,last_recv_ts,staged_path,decision) VALUES(?,?,?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(node_id,peer_id,relpath) DO UPDATE SET sha256=excluded.sha256,mtime=excluded.mtime,size=excluded.size,last_sent_ts=excluded.last_sent_ts,decision=excluded.decision",
                        (node_id, peer_id, rel, sha, float(it["mtime"]), int(it["size"]), time.time(), None, None, "sent"),
                    )
                    details.append({"rel": rel, "sent": True})
                else:
                    errors += 1
                    details.append({"rel": rel, "sent": False, "status": status, "resp": resp})
            except Exception as exc:
                errors += 1
                details.append({"rel": it.get("rel"), "error": str(exc)})
        con.commit()
    finally:
        con.close()
    out = {"ok": True, "node_id": node_id, "peer_id": peer_id, "sent": sent, "skipped": skipped, "errors": errors, "details": details[:25]}
    log_sync_event("SarahNet Push", out, verdict="ALLOW" if errors == 0 else "ALLOW_WITH_ERRORS")
    return out


def sarahnet_sync_poll_and_apply(base_dir: Optional[str] = None, max_items: int = 25, *, user_approved: bool = False, session_token: Optional[str] = None, apply_to_core: bool = False) -> dict:
    action = "sarahnet_apply_core" if apply_to_core else "sarahnet_stage"
    review = sm_sync_governance_review(action, remote=True, source="sarahnet", payload={"apply_to_core": apply_to_core}, risk="critical" if apply_to_core else "high", user_approved=user_approved, session_token=session_token, write_core=apply_to_core)
    if not review["ok"]:
        return {"ok": False, "review": review}
    base = os.path.abspath(base_dir or _base_dir())
    node_id = str(_cfg("SARAHNET_NODE_ID", "local-node"))
    trusted = _trusted_peer_ids()
    status, resp = _http_get_json(_broker_base_url() + f"/api/net/file/poll?to_node={node_id}&limit={int(max_items)}", timeout=float(_cfg("REMOTE_HTTP_TIMEOUT", 10.0)))
    if status != 200 or not resp.get("ok"):
        return {"ok": False, "status": status, "resp": resp}
    _ensure_sync_tables()
    staged_root = os.path.join(_sync_dir(), "staged_core_inbound")
    os.makedirs(staged_root, exist_ok=True)
    applied = staged = rejected = errors = acks = 0
    con = _connect_sync_db()
    cur = con.cursor()
    try:
        for it in resp.get("items") or []:
            try:
                file_id = it.get("file_id") or it.get("id")
                from_node = str(it.get("from_node") or "").strip()
                filename = str(it.get("filename") or "").strip()
                data_b64 = str(it.get("data_b64") or "").strip()
                sha_in = str(it.get("sha256") or "").strip()
                if not file_id or not filename.startswith("core_sync::"):
                    continue
                if from_node not in trusted:
                    rejected += 1
                    continue
                rel = filename.split("core_sync::", 1)[1].strip().replace("\\", "/")
                if not rel or ".." in rel or rel.startswith(("/", "\\")) or not (rel.startswith("core/") or rel.startswith("api/server/")):
                    rejected += 1
                    continue
                raw = base64.b64decode(data_b64.encode("ascii"), validate=False)
                sha = hashlib.sha256(raw).hexdigest()
                if sha_in and sha.lower() != sha_in.lower():
                    rejected += 1
                    continue
                stage_path = os.path.abspath(os.path.join(staged_root, rel.replace("/", os.sep)))
                if os.path.commonpath([staged_root, stage_path]) != staged_root:
                    rejected += 1
                    continue
                os.makedirs(os.path.dirname(stage_path), exist_ok=True)
                with open(stage_path, "wb") as f:
                    f.write(raw)
                staged += 1
                decision = "staged"
                if apply_to_core:
                    dst = os.path.abspath(os.path.join(base, rel.replace("/", os.sep)))
                    if os.path.commonpath([base, dst]) != base:
                        rejected += 1
                        continue
                    backup = dst + f".syncbak.{int(time.time())}"
                    if os.path.exists(dst):
                        os.replace(dst, backup)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    with open(dst, "wb") as f:
                        f.write(raw)
                    applied += 1
                    decision = "applied_with_backup"
                cur.execute(
                    "INSERT INTO code_sync_state(node_id,peer_id,relpath,sha256,mtime,size,last_sent_ts,last_recv_ts,staged_path,decision) VALUES(?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(node_id,peer_id,relpath) DO UPDATE SET sha256=excluded.sha256,mtime=excluded.mtime,size=excluded.size,last_recv_ts=excluded.last_recv_ts,staged_path=excluded.staged_path,decision=excluded.decision",
                    (node_id, from_node, rel, sha, time.time(), len(raw), None, time.time(), stage_path, decision),
                )
                try:
                    st2, r2 = _http_post_json(_broker_base_url() + "/api/net/file/ack", {"file_id": file_id, "to_node": node_id}, timeout=float(_cfg("REMOTE_HTTP_TIMEOUT", 10.0)))
                    if st2 == 200 and r2.get("ok"):
                        acks += 1
                except Exception:
                    pass
            except Exception:
                errors += 1
        con.commit()
    finally:
        con.close()
    out = {"ok": True, "node_id": node_id, "staged": staged, "applied": applied, "rejected": rejected, "errors": errors, "acked": acks, "polled": len(resp.get("items") or []), "apply_to_core": bool(apply_to_core)}
    log_sync_event("SarahNet Poll", out, verdict="STAGE_ONLY" if not apply_to_core else "ALLOW")
    return out


def sarahnet_sync_tick(peer_ids: Optional[List[str]] = None, base_dir: Optional[str] = None, *, user_approved: bool = False, session_token: Optional[str] = None) -> dict:
    base = os.path.abspath(base_dir or _base_dir())
    node_id = str(_cfg("SARAHNET_NODE_ID", "local-node"))
    if peer_ids is None:
        try:
            peer_ids = [k for k in (_cfg("SARAHNET_PEERS", {}) or {}).keys() if k != node_id]
        except Exception:
            peer_ids = []
    poll = sarahnet_sync_poll_and_apply(base_dir=base, user_approved=user_approved, session_token=session_token, apply_to_core=False)
    push = [sarahnet_sync_push(pid, base_dir=base, user_approved=user_approved, session_token=session_token) for pid in (peer_ids or [])]
    return {"ok": True, "node_id": node_id, "poll": poll, "push": push}


# -----------------------------------------------------------------------------
# LIGHTWEIGHT TEST SUITE
# -----------------------------------------------------------------------------
class SarahMemorySyncTestSuite:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.test_user_id = "test_user_" + str(int(time.time()))
        self.test_device_id: Optional[str] = None

    def _pass(self, msg: str) -> None:
        self.passed += 1
        print("  PASS:", msg)

    def _fail(self, msg: str) -> None:
        self.failed += 1
        print("  FAIL:", msg)

    def _warn(self, msg: str) -> None:
        self.warnings += 1
        print("  WARN:", msg)

    def run_all_tests(self) -> int:
        print("SarahMemorySync local-first test suite", MODULE_VERSION)
        try:
            _ensure_sync_tables(); self._pass("sync tables ready")
        except Exception as exc:
            self._fail(f"sync tables failed: {exc}")
        try:
            review = sm_sync_governance_review("sarahnet_push", remote=True, source="test", payload={"text": "AI Agent bypass governance"})
            if not review["ok"]:
                self._pass("anti-agent/remote gate denies unarmed remote sync")
            else:
                self._fail("remote sync was allowed while unarmed")
        except Exception as exc:
            self._fail(f"governance test failed: {exc}")
        try:
            self.test_device_id = register_phase_c_device(self.test_user_id, "Local Test Device", "desktop")
            if self.test_device_id:
                self._pass("local Phase C device registration")
            else:
                self._fail("local Phase C device registration returned None")
        except Exception as exc:
            self._fail(f"device registration failed: {exc}")
        try:
            result = perform_phase_c_sync(self.test_user_id, self.test_device_id or "unknown", contacts=[{"contact_id":"c1","display_name":"Local Contact","last_modified_timestamp":int(time.time())}])
            if result.get("success"):
                self._pass("local Phase C contact sync")
            else:
                self._fail("local Phase C contact sync failed")
        except Exception as exc:
            self._fail(f"contact sync failed: {exc}")
        try:
            contacts = get_phase_c_contacts(self.test_user_id)
            if contacts:
                self._pass("local Phase C contact retrieval")
            else:
                self._fail("local Phase C contact retrieval empty")
        except Exception as exc:
            self._fail(f"contact retrieval failed: {exc}")
        print(f"Passed={self.passed} Failed={self.failed} Warnings={self.warnings}")
        return 0 if self.failed == 0 else 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SarahMemorySync local-first governed test suite")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--phase-c-only", action="store_true")
    args = parser.parse_args()
    if args.test or args.legacy_only or args.phase_c_only:
        sys.exit(SarahMemorySyncTestSuite().run_all_tests())
    print(json.dumps(sync_data(), indent=2))

# ====================================================================
# END OF SarahMemorySync.py v9.0.0
# ====================================================================
# END OF LINE
