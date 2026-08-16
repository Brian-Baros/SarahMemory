"""--==The SarahMemory Project==--
File: SarahMemoryTrustRegistry.py
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

- Canonical trust, subject, and capability registry for SarahMemory AiOS / SMGET.
- Tracks trusted identities across core modules, frontends, addons, drivers, and other
bounded execution surfaces without granting authority merely because code exists.
- Supplies the trust record used by SarahMemorySecurityGovernor.py and the broader
governed execution stack.
- Separates registration, trust tier, capability grant, exposure, and quarantine state.
- Provides one auditable place to answer: who is calling, what are they allowed to ask for,
and how much runtime trust should be assigned to that caller.

CORE DOCTRINE:
- Registration does not imply trust.
- Presentation does not imply authority.
- Capability declaration does not imply capability grant.
- Core-approved SarahMemory modules inherit trust only through governed registry checks.
- Unknown, malformed, or suspicious callers fail closed and may be quarantined.
- Local-first, fail-soft, auditable, and safe-by-default.

RELATIONSHIP MODEL:
- SarahMemoryGlobals.py          -> governed core module registry / discovery
- SarahMemoryTrustRegistry.py    -> canonical subject trust + capability grants
- SarahMemorySecurityGovernor.py -> runtime trust / sovereignty / security enforcement
- SarahMemoryOperatorCore.py     -> action lifecycle and runtime execution choke-point
- Frontends / Addons / Drivers   -> bounded clients of the core, never authority by default
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "trust_registry"
# CATEGORY = "trust_and_capability_registry"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "trust_registry"
# HARDWARE_DOMAIN = "system_network_filesystem_devices"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "trust_registry"
# FAMILY = "smget"
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
# NOTES = "SMGET canonical subject trust and capability registry. Tracks callers, trust tiers, scoped grants, manifests, quarantine state, and registry-backed authority boundaries."
# --- SARAHMETA END ---

import json
import logging
import hashlib
import hmac
import secrets
import os
import re
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Safe imports (never hard-fail the platform)
# ---------------------------------------------------------------------------
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("SarahMemoryTrustRegistry")
logger.setLevel(logging.DEBUG if bool(getattr(config, "DEBUG_MODE", True)) else logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.propagate = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODULE_NAME = "SarahMemoryTrustRegistry"
MODULE_VERSION = "9.0.0"
_DB_NAME = "trust_registry.db"
_JSON_SNAPSHOT_NAME = "trust_registry_snapshot.json"

TRUST_TIER_CORE = "core"
TRUST_TIER_FIRST_PARTY = "first_party"
TRUST_TIER_VERIFIED_THIRD_PARTY = "verified_third_party"
TRUST_TIER_UNVERIFIED = "unverified"
TRUST_TIER_QUARANTINED = "quarantined"
TRUST_TIER_UNKNOWN = "unknown"

SUBJECT_KIND_CORE = "core"
SUBJECT_KIND_FRONTEND = "frontend"
SUBJECT_KIND_ADDON = "addon"
SUBJECT_KIND_DRIVER = "driver"
SUBJECT_KIND_SURFACE = "surface"
SUBJECT_KIND_SERVICE = "service"
SUBJECT_KIND_MODEL = "model"
SUBJECT_KIND_UNKNOWN = "unknown"

STATUS_ACTIVE = "active"
STATUS_PENDING = "pending"
STATUS_QUARANTINED = "quarantined"
STATUS_REVOKED = "revoked"

PASSPORT_STATUS_ISSUED = "issued"
PASSPORT_STATUS_DEPARTED = "departed"
PASSPORT_STATUS_IN_FLIGHT = "in_flight"
PASSPORT_STATUS_RETURN_SLOT_RESERVED = "return_slot_reserved"
PASSPORT_STATUS_RETURN_CAPTURED = "return_captured"
PASSPORT_STATUS_CONSUMED = "consumed"
PASSPORT_STATUS_EXPIRED = "expired"
PASSPORT_STATUS_REVOKED = "revoked"
PASSPORT_STATUS_BLOCKED = "blocked"
PASSPORT_STATUS_COLLISION_LOCKED = "collision_locked"
PASSPORT_STATUS_COMPROMISED = "compromised"
AGENT_PASSPORT_SCHEMA = "SARAHMEMORY_AGENT_PASSPORT_V1"

_DEFAULT_PERMISSION_MAP = {
    SUBJECT_KIND_CORE: ["core.read", "core.execute", "core.route"],
    SUBJECT_KIND_FRONTEND: ["chat.submit", "chat.read"],
    SUBJECT_KIND_ADDON: ["addon.request"],
    SUBJECT_KIND_DRIVER: ["driver.read"],
    SUBJECT_KIND_SURFACE: ["surface.render"],
    SUBJECT_KIND_SERVICE: ["service.request"],
    SUBJECT_KIND_MODEL: ["model.helper"],
}

_TRUST_SCORE_MAP = {
    TRUST_TIER_CORE: 100,
    TRUST_TIER_FIRST_PARTY: 85,
    TRUST_TIER_VERIFIED_THIRD_PARTY: 65,
    TRUST_TIER_UNVERIFIED: 35,
    TRUST_TIER_UNKNOWN: 20,
    TRUST_TIER_QUARANTINED: 0,
}

_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SubjectRecord:
    subject_id: str
    subject_kind: str
    display_name: str
    trust_tier: str
    status: str
    publisher: str = ""
    module_name: str = ""
    surface: str = ""
    version: str = ""
    manifest_path: str = ""
    registry_source: str = "manual"
    trusted: bool = False
    approved: bool = False
    exposed: bool = False
    quarantined: bool = False
    permissions: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_ts: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_ts: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["trust_score"] = int(_TRUST_SCORE_MAP.get(self.trust_tier, 20))
        return data


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _base_dir() -> str:
    try:
        return str(getattr(config, "BASE_DIR", os.getcwd()))
    except Exception:
        return os.getcwd()


def _data_dir() -> str:
    try:
        return str(getattr(config, "DATA_DIR", os.path.join(_base_dir(), "data")))
    except Exception:
        return os.path.join(_base_dir(), "data")


def _datasets_dir() -> str:
    try:
        return str(getattr(config, "DATASETS_DIR", os.path.join(_data_dir(), "memory", "datasets")))
    except Exception:
        return os.path.join(_data_dir(), "memory", "datasets")


def _settings_dir() -> str:
    try:
        return str(getattr(config, "SETTINGS_DIR", os.path.join(_data_dir(), "settings")))
    except Exception:
        return os.path.join(_data_dir(), "settings")


def _registry_dir() -> str:
    return os.path.join(_settings_dir(), "trust_registry")


def _registry_db_path() -> str:
    return os.path.join(_datasets_dir(), _DB_NAME)


def _registry_snapshot_path() -> str:
    return os.path.join(_registry_dir(), _JSON_SNAPSHOT_NAME)


def _drivers_root() -> str:
    return os.path.join(_data_dir(), "drivers")


def _driver_registry_path() -> str:
    return os.path.join(_data_dir(), "registry", "drivers.json")


def _addons_root() -> str:
    for attr in ("ADDONS_DIR",):
        try:
            candidate = str(getattr(config, attr, "")).strip()
            if candidate:
                return candidate
        except Exception:
            pass
    return os.path.join(_data_dir(), "addons")


def _ensure_dirs() -> None:
    for d in (_data_dir(), _datasets_dir(), _settings_dir(), _registry_dir()):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def _connect_db() -> sqlite3.Connection:
    _ensure_dirs()
    con = sqlite3.connect(_registry_db_path(), timeout=5.0, check_same_thread=False)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        pass
    return con


def _ensure_tables() -> None:
    con = None
    try:
        con = _connect_db()
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trust_subjects (
                subject_id TEXT PRIMARY KEY,
                subject_kind TEXT,
                display_name TEXT,
                trust_tier TEXT,
                status TEXT,
                publisher TEXT,
                module_name TEXT,
                surface TEXT,
                version TEXT,
                manifest_path TEXT,
                registry_source TEXT,
                trusted INTEGER,
                approved INTEGER,
                exposed INTEGER,
                quarantined INTEGER,
                permissions_json TEXT,
                capabilities_json TEXT,
                metadata_json TEXT,
                created_ts TEXT,
                updated_ts TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trust_registry_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                event_kind TEXT,
                severity TEXT,
                subject_id TEXT,
                details TEXT,
                meta_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_passports (
                passport_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                agent_name TEXT,
                task_id TEXT,
                purpose TEXT,
                issuer_node TEXT,
                origin TEXT,
                issued_ts REAL NOT NULL,
                expires_ts REAL NOT NULL,
                status TEXT NOT NULL,
                one_time_use INTEGER NOT NULL DEFAULT 1,
                consumed_ts REAL,
                revoked_ts REAL,
                revocation_reason TEXT,
                departure_ts REAL,
                departure_nonce TEXT,
                return_nonce_hash TEXT NOT NULL,
                return_signature_hash TEXT NOT NULL,
                origin_lane TEXT,
                allowed_lanes_json TEXT,
                allowed_capabilities_json TEXT,
                allowed_resources_json TEXT,
                denied_resources_json TEXT,
                maximum_risk_tier TEXT,
                network_allowed INTEGER DEFAULT 0,
                filesystem_allowed INTEGER DEFAULT 0,
                shell_allowed INTEGER DEFAULT 0,
                device_allowed INTEGER DEFAULT 0,
                memory_allowed INTEGER DEFAULT 0,
                requires_user_review INTEGER DEFAULT 1,
                requires_assurance INTEGER DEFAULT 1,
                requires_compare INTEGER DEFAULT 1,
                requires_compass INTEGER DEFAULT 1,
                user_approved INTEGER DEFAULT 0,
                return_count INTEGER DEFAULT 0,
                last_return_ts REAL,
                last_payload_hash TEXT,
                metadata_json TEXT,
                created_ts TEXT,
                updated_ts TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_passport_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                passport_id TEXT,
                agent_id TEXT,
                event_type TEXT NOT NULL,
                verdict TEXT,
                reason TEXT,
                payload_hash TEXT,
                metadata_json TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_passports_agent ON agent_passports(agent_id, issued_ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_passports_status ON agent_passports(status, expires_ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_passports_task ON agent_passports(task_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_agent_passport_events_passport ON agent_passport_events(passport_id, ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trust_subjects_kind ON trust_subjects(subject_kind)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trust_subjects_module ON trust_subjects(module_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_trust_subjects_status ON trust_subjects(status)")
        con.commit()
    except Exception as exc:
        logger.debug("TrustRegistry DB ensure failed: %s", exc)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _log_event(event_kind: str, subject_id: str, details: str, *, severity: str = "INFO", meta: Optional[Dict[str, Any]] = None) -> None:
    con = None
    try:
        _ensure_tables()
        con = _connect_db()
        cur = con.cursor()
        cur.execute(
            "INSERT INTO trust_registry_events (ts, event_kind, severity, subject_id, details, meta_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                str(event_kind),
                str(severity),
                str(subject_id or ""),
                str(details),
                json.dumps(meta or {}, ensure_ascii=False),
            ),
        )
        con.commit()
    except Exception as exc:
        logger.debug("TrustRegistry log failed: %s", exc)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _safe_str(value: Any, default: str = "") -> str:
    try:
        return str(value if value is not None else default).strip()
    except Exception:
        return str(default)


def _safe_bool(value: Any, default: bool = False) -> bool:
    try:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled", "approved", "active"}
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Enterprise assurance / replay policy helpers
# ---------------------------------------------------------------------------
def _config_flag(name: str, default: bool = False) -> bool:
    try:
        value = os.getenv(name, None)
    except Exception:
        value = None
    if value is None:
        try:
            value = getattr(config, name, default) if config is not None else default
        except Exception:
            value = default
    try:
        if isinstance(value, bool):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}
    except Exception:
        return bool(default)


def _config_int(name: str, default: int, *, minimum: int = 0, maximum: int = 100000) -> int:
    try:
        value = os.getenv(name, None)
    except Exception:
        value = None
    if value is None:
        try:
            value = getattr(config, name, default) if config is not None else default
        except Exception:
            value = default
    try:
        out = int(float(value))
    except Exception:
        out = int(default)
    try:
        return max(int(minimum), min(int(maximum), out))
    except Exception:
        return int(default)


def _config_choice(name: str, default: str, allowed: Optional[List[str]] = None) -> str:
    try:
        value = os.getenv(name, None)
    except Exception:
        value = None
    if value is None:
        try:
            value = getattr(config, name, default) if config is not None else default
        except Exception:
            value = default
    text = str(value or default).strip().lower()
    allowed_set = {str(x).strip().lower() for x in (allowed or [])}
    if allowed_set and text not in allowed_set:
        return str(default).strip().lower()
    return text


def _assurance_enabled() -> bool:
    return _config_flag("SARAH_ASSURANCE_ENABLED", True)


def _assurance_tests_enabled() -> bool:
    return _config_flag("SARAH_ASSURANCE_TESTS_ENABLED", False)


def _trust_transition_audit_enabled() -> bool:
    return _config_flag("SARAH_TRUST_TRANSITION_AUDIT_ENABLED", True)


def _agent_max_parallel_returns() -> int:
    # Enterprise default: a single passport owns exactly one FIFO return slot.
    return _config_int("SARAH_AGENT_MAX_PARALLEL_RETURNS", 1, minimum=1, maximum=8)


def _passport_collision_policy() -> str:
    return _config_choice("SARAH_AGENT_PASSPORT_COLLISION_POLICY", "reject_all", ["reject_all", "block_new", "review_only"])


def _passport_replay_policy() -> str:
    return _config_choice("SARAH_AGENT_PASSPORT_REPLAY_POLICY", "collision_lock", ["collision_lock", "block", "review_only"])


def _record_trust_transition(
    *,
    passport_id: str,
    agent_id: str,
    old_status: str,
    new_status: str,
    event_type: str,
    reason: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Record an auditable trust/passport state transition without granting authority."""
    if not _trust_transition_audit_enabled():
        return
    meta = {
        "passport_id": str(passport_id or "")[:180],
        "old_status": str(old_status or "")[:80],
        "new_status": str(new_status or "")[:80],
        "execution_authority": False,
        **(metadata or {}),
    }
    try:
        _log_event("trust_transition_recorded", str(passport_id or agent_id or ""), reason or event_type, severity="WARNING" if str(new_status).lower() in {PASSPORT_STATUS_COLLISION_LOCKED, PASSPORT_STATUS_COMPROMISED, PASSPORT_STATUS_REVOKED, PASSPORT_STATUS_BLOCKED} else "INFO", meta=meta)
    except Exception:
        pass
    try:
        from SarahMemoryLedger import record_governance_receipt  # type: ignore
        record_governance_receipt(
            "agent_passport",
            "TRUST_TRANSITION_RECORDED",
            subject_id=str(agent_id or "unknown_agent")[:180],
            task_id=str((metadata or {}).get("task_id") or "")[:180],
            lane=str((metadata or {}).get("lane") or "agent_passport")[:96],
            verdict=str(event_type or "TRANSITION")[:64],
            risk=str((metadata or {}).get("risk") or "medium")[:32],
            retention_class="passport",
            payload_hash=str((metadata or {}).get("payload_hash") or "")[:128],
            summary=str(reason or event_type or "trust_transition")[:1000],
            metadata=meta,
        )
    except Exception:
        pass


def _safe_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    elif isinstance(value, tuple):
        raw = list(value)
    elif isinstance(value, set):
        raw = sorted(list(value))
    else:
        try:
            txt = str(value).strip()
        except Exception:
            return []
        if not txt:
            return []
        # SARAHMEMORY_PATCH_NOTE 2026-08-04:
        # Passport scopes often enter from Terminal key=value strings. Split
        # comma/semicolon lists here so capability/resource checks do not treat
        # "a,b,c" as one opaque grant.
        raw = txt.replace(";", ",").split(",") if ("," in txt or ";" in txt) else [txt]
    out: List[str] = []
    for x in raw:
        text = str(x).strip()
        if text and text not in out:
            out.append(text)
    return out


def _subject_id(caller_id: str, caller_kind: str, module_name: str = "") -> str:
    cid = _safe_str(caller_id)
    ckind = _safe_str(caller_kind or SUBJECT_KIND_UNKNOWN).lower() or SUBJECT_KIND_UNKNOWN
    mod = _safe_str(module_name)
    if cid:
        return cid
    if mod:
        return f"{ckind}:{mod}"
    return f"{ckind}:{uuid.uuid4().hex[:12]}"


def _default_permissions(subject_kind: str, trust_tier: str) -> List[str]:
    perms = list(_DEFAULT_PERMISSION_MAP.get(subject_kind, []))
    if trust_tier == TRUST_TIER_CORE:
        perms.extend(["governance.read", "trust.read", "operator.request", "audit.write"])
    elif trust_tier == TRUST_TIER_FIRST_PARTY:
        perms.extend(["operator.request", "audit.read"])
    elif trust_tier == TRUST_TIER_VERIFIED_THIRD_PARTY:
        perms.extend(["bounded.request"])
    return sorted(set([p for p in perms if p]))


def _record_from_row(row: sqlite3.Row) -> Dict[str, Any]:
    try:
        permissions = json.loads(row[15] or "[]") if len(row) > 15 else []
    except Exception:
        permissions = []
    try:
        capabilities = json.loads(row[16] or "[]") if len(row) > 16 else []
    except Exception:
        capabilities = []
    try:
        metadata = json.loads(row[17] or "{}") if len(row) > 17 else {}
    except Exception:
        metadata = {}

    rec = SubjectRecord(
        subject_id=_safe_str(row[0]),
        subject_kind=_safe_str(row[1], SUBJECT_KIND_UNKNOWN),
        display_name=_safe_str(row[2]),
        trust_tier=_safe_str(row[3], TRUST_TIER_UNKNOWN),
        status=_safe_str(row[4], STATUS_PENDING),
        publisher=_safe_str(row[5]),
        module_name=_safe_str(row[6]),
        surface=_safe_str(row[7]),
        version=_safe_str(row[8]),
        manifest_path=_safe_str(row[9]),
        registry_source=_safe_str(row[10], "manual"),
        trusted=bool(row[11]),
        approved=bool(row[12]),
        exposed=bool(row[13]),
        quarantined=bool(row[14]),
        permissions=_safe_list(permissions),
        capabilities=_safe_list(capabilities),
        metadata=metadata if isinstance(metadata, dict) else {},
        created_ts=_safe_str(row[18]),
        updated_ts=_safe_str(row[19]),
    )
    return rec.to_dict()


def _upsert_record(record: SubjectRecord, *, event_kind: str = "upsert_subject") -> Dict[str, Any]:
    con = None
    out = record.to_dict()
    try:
        _ensure_tables()
        con = _connect_db()
        cur = con.cursor()
        cur.execute(
            """
            INSERT INTO trust_subjects (
                subject_id, subject_kind, display_name, trust_tier, status, publisher,
                module_name, surface, version, manifest_path, registry_source, trusted,
                approved, exposed, quarantined, permissions_json, capabilities_json,
                metadata_json, created_ts, updated_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(subject_id) DO UPDATE SET
                subject_kind=excluded.subject_kind,
                display_name=excluded.display_name,
                trust_tier=excluded.trust_tier,
                status=excluded.status,
                publisher=excluded.publisher,
                module_name=excluded.module_name,
                surface=excluded.surface,
                version=excluded.version,
                manifest_path=excluded.manifest_path,
                registry_source=excluded.registry_source,
                trusted=excluded.trusted,
                approved=excluded.approved,
                exposed=excluded.exposed,
                quarantined=excluded.quarantined,
                permissions_json=excluded.permissions_json,
                capabilities_json=excluded.capabilities_json,
                metadata_json=excluded.metadata_json,
                updated_ts=excluded.updated_ts
            """,
            (
                record.subject_id,
                record.subject_kind,
                record.display_name,
                record.trust_tier,
                record.status,
                record.publisher,
                record.module_name,
                record.surface,
                record.version,
                record.manifest_path,
                record.registry_source,
                1 if record.trusted else 0,
                1 if record.approved else 0,
                1 if record.exposed else 0,
                1 if record.quarantined else 0,
                json.dumps(record.permissions, ensure_ascii=False),
                json.dumps(record.capabilities, ensure_ascii=False),
                json.dumps(record.metadata or {}, ensure_ascii=False),
                record.created_ts,
                record.updated_ts,
            ),
        )
        con.commit()
        _log_event(event_kind, record.subject_id, f"{record.subject_kind}:{record.display_name or record.subject_id}", meta=out)
    except Exception as exc:
        logger.debug("TrustRegistry upsert failed for %s: %s", record.subject_id, exc)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Snapshot + sync helpers
# ---------------------------------------------------------------------------
def _registered_core_modules() -> Dict[str, Any]:
    try:
        if config and hasattr(config, "sm_get_registered_core_modules"):
            out = config.sm_get_registered_core_modules()  # type: ignore[attr-defined]
            return out if isinstance(out, dict) else {}
    except Exception:
        pass
    return {}


def _is_core_module_approved(module_name: str, capability: Optional[str] = None) -> bool:
    try:
        if config and hasattr(config, "sm_is_core_module_approved"):
            return bool(config.sm_is_core_module_approved(module_name, capability=capability))  # type: ignore[attr-defined]
    except Exception:
        pass
    return False


def sync_globals_core_registry() -> Dict[str, Any]:
    synced = 0
    modules = _registered_core_modules()
    for module_name, entry in modules.items():
        if not isinstance(entry, dict):
            continue
        trust_tier = TRUST_TIER_CORE if _is_core_module_approved(module_name, entry.get("capability")) else TRUST_TIER_FIRST_PARTY
        record = SubjectRecord(
            subject_id=f"core:{module_name}",
            subject_kind=SUBJECT_KIND_CORE,
            display_name=module_name,
            trust_tier=trust_tier,
            status=STATUS_ACTIVE if bool(entry.get("status") == "approved") else STATUS_PENDING,
            publisher="SarahMemory Core",
            module_name=module_name,
            surface="core",
            version=_safe_str(entry.get("version") or MODULE_VERSION),
            registry_source="globals_core_registry",
            trusted=True,
            approved=bool(entry.get("approved", True)),
            exposed=bool(entry.get("exposed", False)),
            quarantined=bool(entry.get("status") == "quarantined"),
            permissions=sorted(set(_default_permissions(SUBJECT_KIND_CORE, trust_tier))),
            capabilities=_safe_list(entry.get("capability")),
            metadata=dict(entry),
        )
        _upsert_record(record, event_kind="sync_globals_core")
        synced += 1
    return {"ok": True, "synced": synced, "source": "globals_core_registry"}


def _read_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return default


def sync_driver_registry() -> Dict[str, Any]:
    synced = 0
    drivers_root = _drivers_root()
    driver_reg = _read_json(_driver_registry_path(), {})
    registry_entries = driver_reg if isinstance(driver_reg, dict) else {}
    if not os.path.isdir(drivers_root):
        return {"ok": True, "synced": 0, "source": "drivers", "reason": "drivers_root_missing"}

    for name in sorted(os.listdir(drivers_root)):
        driver_dir = os.path.join(drivers_root, name)
        if not os.path.isdir(driver_dir):
            continue
        manifest_path = os.path.join(driver_dir, "manifest.json")
        manifest = _read_json(manifest_path, {}) if os.path.isfile(manifest_path) else {}
        reg_entry = registry_entries.get(name) if isinstance(registry_entries.get(name), dict) else {}
        trusted = _safe_bool(reg_entry.get("trusted"), False)
        quarantined = _safe_bool(reg_entry.get("quarantined"), False)
        trust_tier = TRUST_TIER_FIRST_PARTY if trusted else TRUST_TIER_UNVERIFIED
        if quarantined:
            trust_tier = TRUST_TIER_QUARANTINED
        record = SubjectRecord(
            subject_id=f"driver:{name}",
            subject_kind=SUBJECT_KIND_DRIVER,
            display_name=_safe_str(manifest.get("name") or name),
            trust_tier=trust_tier,
            status=STATUS_QUARANTINED if quarantined else STATUS_ACTIVE if trusted else STATUS_PENDING,
            publisher=_safe_str(manifest.get("publisher") or manifest.get("author") or "unknown"),
            module_name=name,
            surface="driver",
            version=_safe_str(manifest.get("version") or reg_entry.get("version") or ""),
            manifest_path=manifest_path,
            registry_source="drivers_registry",
            trusted=trusted,
            approved=trusted,
            exposed=False,
            quarantined=quarantined,
            permissions=sorted(set(_default_permissions(SUBJECT_KIND_DRIVER, trust_tier) + _safe_list(reg_entry.get("permissions")) + _safe_list(manifest.get("permissions")))),
            capabilities=sorted(set(_safe_list(reg_entry.get("capabilities")) + _safe_list(manifest.get("capabilities")) + _safe_list(manifest.get("hardware_domain")))),
            metadata={"manifest": manifest, "registry": reg_entry},
        )
        _upsert_record(record, event_kind="sync_driver_registry")
        synced += 1
    return {"ok": True, "synced": synced, "source": "drivers_registry"}


def sync_addon_registry() -> Dict[str, Any]:
    synced = 0
    addons_root = _addons_root()
    if not os.path.isdir(addons_root):
        return {"ok": True, "synced": 0, "source": "addons", "reason": "addons_root_missing"}

    for name in sorted(os.listdir(addons_root)):
        addon_dir = os.path.join(addons_root, name)
        if not os.path.isdir(addon_dir):
            continue
        manifest_path = os.path.join(addon_dir, "manifest.json")
        manifest = _read_json(manifest_path, {}) if os.path.isfile(manifest_path) else {}
        verified = _safe_bool(manifest.get("verified") or manifest.get("signed"), False)
        quarantined = _safe_bool(manifest.get("quarantined"), False)
        trust_tier = TRUST_TIER_VERIFIED_THIRD_PARTY if verified else TRUST_TIER_UNVERIFIED
        if quarantined:
            trust_tier = TRUST_TIER_QUARANTINED
        record = SubjectRecord(
            subject_id=f"addon:{name}",
            subject_kind=SUBJECT_KIND_ADDON,
            display_name=_safe_str(manifest.get("name") or name),
            trust_tier=trust_tier,
            status=STATUS_QUARANTINED if quarantined else STATUS_ACTIVE if verified else STATUS_PENDING,
            publisher=_safe_str(manifest.get("publisher") or manifest.get("author") or "unknown"),
            module_name=name,
            surface="addon",
            version=_safe_str(manifest.get("version") or ""),
            manifest_path=manifest_path,
            registry_source="addon_manifest",
            trusted=verified,
            approved=verified,
            exposed=False,
            quarantined=quarantined,
            permissions=sorted(set(_default_permissions(SUBJECT_KIND_ADDON, trust_tier) + _safe_list(manifest.get("permissions")) + _safe_list(manifest.get("requested_capabilities")))),
            capabilities=sorted(set(_safe_list(manifest.get("capabilities")) + _safe_list(manifest.get("requested_capabilities")))),
            metadata={"manifest": manifest},
        )
        _upsert_record(record, event_kind="sync_addon_registry")
        synced += 1
    return {"ok": True, "synced": synced, "source": "addon_manifest"}


def persist_snapshot() -> Dict[str, Any]:
    out = get_registry_snapshot()
    try:
        _ensure_dirs()
        with open(_registry_snapshot_path(), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        return {"ok": True, "snapshot_file": _registry_snapshot_path(), "subjects": len(out.get("subjects", {}))}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "snapshot_file": _registry_snapshot_path()}


# ---------------------------------------------------------------------------
# Public registry APIs
# ---------------------------------------------------------------------------
def register_subject(
    *,
    caller_id: str,
    caller_kind: str,
    display_name: str = "",
    trust_tier: str = TRUST_TIER_UNVERIFIED,
    status: str = STATUS_PENDING,
    publisher: str = "",
    module_name: str = "",
    surface: str = "",
    version: str = "",
    manifest_path: str = "",
    registry_source: str = "manual",
    permissions: Optional[List[str]] = None,
    capabilities: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    trusted: Optional[bool] = None,
    approved: Optional[bool] = None,
    exposed: bool = False,
) -> Dict[str, Any]:
    subject_kind = _safe_str(caller_kind or SUBJECT_KIND_UNKNOWN).lower() or SUBJECT_KIND_UNKNOWN
    trust_tier = _safe_str(trust_tier or TRUST_TIER_UNKNOWN) or TRUST_TIER_UNKNOWN
    status = _safe_str(status or STATUS_PENDING) or STATUS_PENDING
    caller_id = _subject_id(caller_id, subject_kind, module_name)
    quarantined = (trust_tier == TRUST_TIER_QUARANTINED) or (status == STATUS_QUARANTINED)
    record = SubjectRecord(
        subject_id=caller_id,
        subject_kind=subject_kind,
        display_name=_safe_str(display_name or module_name or caller_id),
        trust_tier=trust_tier,
        status=status,
        publisher=_safe_str(publisher),
        module_name=_safe_str(module_name),
        surface=_safe_str(surface),
        version=_safe_str(version),
        manifest_path=_safe_str(manifest_path),
        registry_source=_safe_str(registry_source or "manual"),
        trusted=_safe_bool(trusted, trust_tier in {TRUST_TIER_CORE, TRUST_TIER_FIRST_PARTY, TRUST_TIER_VERIFIED_THIRD_PARTY}),
        approved=_safe_bool(approved, trust_tier in {TRUST_TIER_CORE, TRUST_TIER_FIRST_PARTY, TRUST_TIER_VERIFIED_THIRD_PARTY}),
        exposed=_safe_bool(exposed),
        quarantined=quarantined,
        permissions=sorted(set(_default_permissions(subject_kind, trust_tier) + _safe_list(permissions))),
        capabilities=sorted(set(_safe_list(capabilities))),
        metadata=dict(metadata or {}),
    )
    return _upsert_record(record)


def quarantine_subject(subject_id: str, reason: str = "", *, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    current = lookup_subject(subject_id) or {}
    merged_meta = dict(current.get("metadata") or {})
    merged_meta.update(metadata or {})
    if reason:
        merged_meta["quarantine_reason"] = reason
    out = register_subject(
        caller_id=subject_id,
        caller_kind=current.get("subject_kind") or SUBJECT_KIND_UNKNOWN,
        display_name=current.get("display_name") or subject_id,
        trust_tier=TRUST_TIER_QUARANTINED,
        status=STATUS_QUARANTINED,
        publisher=current.get("publisher") or "",
        module_name=current.get("module_name") or "",
        surface=current.get("surface") or "",
        version=current.get("version") or "",
        manifest_path=current.get("manifest_path") or "",
        registry_source=current.get("registry_source") or "manual",
        permissions=current.get("permissions") or [],
        capabilities=current.get("capabilities") or [],
        metadata=merged_meta,
        trusted=False,
        approved=False,
        exposed=False,
    )
    _log_event("quarantine_subject", subject_id, reason or "subject quarantined", severity="WARNING", meta=out)
    return out


def grant_permissions(subject_id: str, permissions: List[str]) -> Dict[str, Any]:
    current = lookup_subject(subject_id) or {}
    merged = sorted(set(_safe_list(current.get("permissions")) + _safe_list(permissions)))
    return register_subject(
        caller_id=subject_id,
        caller_kind=current.get("subject_kind") or SUBJECT_KIND_UNKNOWN,
        display_name=current.get("display_name") or subject_id,
        trust_tier=current.get("trust_tier") or TRUST_TIER_UNKNOWN,
        status=current.get("status") or STATUS_PENDING,
        publisher=current.get("publisher") or "",
        module_name=current.get("module_name") or "",
        surface=current.get("surface") or "",
        version=current.get("version") or "",
        manifest_path=current.get("manifest_path") or "",
        registry_source=current.get("registry_source") or "manual",
        permissions=merged,
        capabilities=current.get("capabilities") or [],
        metadata=current.get("metadata") or {},
        trusted=current.get("trusted"),
        approved=current.get("approved"),
        exposed=current.get("exposed", False),
    )


def lookup_subject(subject_id: str) -> Optional[Dict[str, Any]]:
    sid = _safe_str(subject_id)
    if not sid:
        return None
    con = None
    try:
        _ensure_tables()
        con = _connect_db()
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        row = cur.execute(
            "SELECT subject_id, subject_kind, display_name, trust_tier, status, publisher, module_name, surface, version, manifest_path, registry_source, trusted, approved, exposed, quarantined, permissions_json, capabilities_json, metadata_json, created_ts, updated_ts FROM trust_subjects WHERE subject_id = ?",
            (sid,),
        ).fetchone()
        if not row:
            return None
        return _record_from_row(row)
    except Exception as exc:
        logger.debug("TrustRegistry lookup failed for %s: %s", sid, exc)
        return None
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def get_subject_permissions(subject_id: str) -> List[str]:
    rec = lookup_subject(subject_id) or {}
    return _safe_list(rec.get("permissions"))


def is_permission_granted(subject_id: str, permission: str) -> bool:
    permission = _safe_str(permission)
    if not permission:
        return False
    perms = set(get_subject_permissions(subject_id))
    return permission in perms


def resolve_subject_trust(
    *,
    caller_id: str,
    caller_kind: str,
    surface: str = "",
    module_name: str = "",
    action_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sid = _subject_id(caller_id, caller_kind, module_name)

    # 1) explicit subject record wins
    rec = lookup_subject(sid)
    if rec:
        return rec

    # 2) core module approval from Globals yields core trust
    module_name = _safe_str(module_name)
    if module_name and _is_core_module_approved(module_name):
        rec = register_subject(
            caller_id=sid,
            caller_kind=SUBJECT_KIND_CORE,
            display_name=module_name,
            trust_tier=TRUST_TIER_CORE,
            status=STATUS_ACTIVE,
            publisher="SarahMemory Core",
            module_name=module_name,
            surface=surface or "core",
            registry_source="globals_core_registry",
            capabilities=_safe_list((action_contract or {}).get("capability_name")),
            permissions=_default_permissions(SUBJECT_KIND_CORE, TRUST_TIER_CORE),
            trusted=True,
            approved=True,
            exposed=True,
        )
        return rec

    # 3) kind-based bounded fallback
    normalized_kind = _safe_str(caller_kind or SUBJECT_KIND_UNKNOWN).lower() or SUBJECT_KIND_UNKNOWN
    trust_tier = TRUST_TIER_UNVERIFIED if normalized_kind in {SUBJECT_KIND_FRONTEND, SUBJECT_KIND_ADDON, SUBJECT_KIND_DRIVER, SUBJECT_KIND_SURFACE, SUBJECT_KIND_SERVICE, SUBJECT_KIND_MODEL} else TRUST_TIER_UNKNOWN
    status = STATUS_PENDING if trust_tier != TRUST_TIER_UNKNOWN else STATUS_PENDING
    rec = register_subject(
        caller_id=sid,
        caller_kind=normalized_kind,
        display_name=module_name or sid,
        trust_tier=trust_tier,
        status=status,
        module_name=module_name,
        surface=surface,
        registry_source="fallback_resolution",
        capabilities=_safe_list((action_contract or {}).get("capability_name")),
        permissions=_default_permissions(normalized_kind, trust_tier),
        trusted=False,
        approved=False,
        exposed=False,
    )
    return rec


def get_subject_trust(
    caller_id: str,
    caller_kind: str = SUBJECT_KIND_UNKNOWN,
    surface: str = "",
    module_name: str = "",
    action_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return resolve_subject_trust(
        caller_id=caller_id,
        caller_kind=caller_kind,
        surface=surface,
        module_name=module_name,
        action_contract=action_contract,
    )


def get_trust_record(
    caller_id: str,
    caller_kind: str = SUBJECT_KIND_UNKNOWN,
    surface: str = "",
    module_name: str = "",
    action_contract: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return resolve_subject_trust(
        caller_id=caller_id,
        caller_kind=caller_kind,
        surface=surface,
        module_name=module_name,
        action_contract=action_contract,
    )


def get_registry_snapshot() -> Dict[str, Any]:
    con = None
    out: Dict[str, Any] = {
        "version": 1,
        "generated_at": datetime.now().isoformat(),
        "base_dir": _base_dir(),
        "subjects": {},
        "counts": {
            TRUST_TIER_CORE: 0,
            TRUST_TIER_FIRST_PARTY: 0,
            TRUST_TIER_VERIFIED_THIRD_PARTY: 0,
            TRUST_TIER_UNVERIFIED: 0,
            TRUST_TIER_UNKNOWN: 0,
            TRUST_TIER_QUARANTINED: 0,
        },
    }
    try:
        _ensure_tables()
        con = _connect_db()
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        rows = cur.execute(
            "SELECT subject_id, subject_kind, display_name, trust_tier, status, publisher, module_name, surface, version, manifest_path, registry_source, trusted, approved, exposed, quarantined, permissions_json, capabilities_json, metadata_json, created_ts, updated_ts FROM trust_subjects ORDER BY subject_kind, display_name, subject_id"
        ).fetchall()
        for row in rows:
            rec = _record_from_row(row)
            out["subjects"][rec["subject_id"]] = rec
            tier = rec.get("trust_tier") or TRUST_TIER_UNKNOWN
            out["counts"][tier] = int(out["counts"].get(tier, 0)) + 1
    except Exception as exc:
        out["error"] = str(exc)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass
    return out


def warm_registry(force_sync: bool = False) -> Dict[str, Any]:
    _ensure_tables()
    result: Dict[str, Any] = {"ok": True, "synced": {}}
    try:
        if force_sync:
            result["synced"]["core"] = sync_globals_core_registry()
            result["synced"]["drivers"] = sync_driver_registry()
            result["synced"]["addons"] = sync_addon_registry()
        snapshot = persist_snapshot()
        result["snapshot"] = snapshot
    except Exception as exc:
        result["ok"] = False
        result["error"] = str(exc)
    return result


# ---------------------------------------------------------------------------
# Governed AI-agent passport registry
# ---------------------------------------------------------------------------
def _passport_secret_path() -> str:
    return os.path.join(_registry_dir(), "agent_passport_secret.key")


def _passport_secret() -> bytes:
    """Return a local signing key without exposing it through registry APIs."""
    for name in ("SARAH_AGENT_PASSPORT_SECRET", "MESH_SHARED_SECRET"):
        try:
            value = os.getenv(name) or (getattr(config, name, None) if config is not None else None)
            if value:
                return hashlib.sha256(str(value).encode("utf-8", "ignore")).digest()
        except Exception:
            pass
    _ensure_dirs()
    path = _passport_secret_path()
    try:
        if os.path.exists(path):
            raw = Path(path).read_bytes()
            if len(raw) >= 32:
                return hashlib.sha256(raw).digest()
        raw = secrets.token_bytes(48)
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(raw)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except Exception:
                pass
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except Exception:
            pass
        return hashlib.sha256(raw).digest()
    except Exception:
        # Fail-soft integrity fallback for read-only/test environments. It never
        # authorizes execution and is intentionally tied to this installation.
        return hashlib.sha256((_base_dir() + "|agent-passport-v1").encode("utf-8", "ignore")).digest()


def _passport_signature(passport_id: str, agent_id: str, task_id: str, return_nonce: str) -> str:
    body = f"{passport_id}|{agent_id}|{task_id}|{return_nonce}".encode("utf-8", "ignore")
    return hmac.new(_passport_secret(), body, hashlib.sha256).hexdigest()


def _passport_hash(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8", "ignore")).hexdigest()


def _passport_event(
    passport_id: str,
    agent_id: str,
    event_type: str,
    verdict: str,
    reason: str = "",
    *,
    payload_hash: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        _ensure_tables()
        con = _connect_db()
        con.execute(
            "INSERT INTO agent_passport_events(ts,passport_id,agent_id,event_type,verdict,reason,payload_hash,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
            (time.time(), passport_id, agent_id, event_type, verdict, reason[:1000], payload_hash[:128], json.dumps(metadata or {}, ensure_ascii=False, default=str)),
        )
        con.commit()
        con.close()
    except Exception:
        pass
    try:
        from SarahMemoryLedger import record_governance_receipt  # type: ignore
        record_governance_receipt(
            "agent_passport",
            event_type,
            subject_id=agent_id,
            task_id=str((metadata or {}).get("task_id") or ""),
            lane=str((metadata or {}).get("lane") or "agent_passport"),
            verdict=verdict,
            risk=str((metadata or {}).get("risk") or "medium"),
            retention_class="passport",
            payload_hash=payload_hash,
            summary=reason or event_type,
            metadata={"passport_id": passport_id, **(metadata or {})},
        )
    except Exception:
        pass


def _passport_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    data = dict(row)
    for key in ("allowed_lanes_json", "allowed_capabilities_json", "allowed_resources_json", "denied_resources_json", "metadata_json"):
        target = key[:-5] if key.endswith("_json") else key
        try:
            data[target] = json.loads(data.pop(key) or ("{}" if key == "metadata_json" else "[]"))
        except Exception:
            data[target] = {} if key == "metadata_json" else []
    for key in (
        "one_time_use", "network_allowed", "filesystem_allowed", "shell_allowed", "device_allowed",
        "memory_allowed", "requires_user_review", "requires_assurance", "requires_compare",
        "requires_compass", "user_approved",
    ):
        data[key] = bool(data.get(key))
    data["schema"] = AGENT_PASSPORT_SCHEMA
    data["execution_authority"] = False
    # SARAHMEMORY_PATCH_NOTE 2026-08-04:
    # Passport lookup/status/scope responses must not re-expose departure or
    # return credentials.  Departure credentials are returned once only by
    # issue_agent_passport().  Registry lookup returns identity/scope proof.
    data.pop("departure_nonce", None)
    data.pop("return_nonce_hash", None)
    data.pop("return_signature_hash", None)
    return data


def _agent_passport_auto_issue_enabled(default: bool = False) -> bool:
    """Read centralized SARAH_AGENT_PASSPORT_ID auto-issue flag.

    This gate applies only to managed/auto passport issuance. Manual
    user-approved passport issue commands still use issue_agent_passport() when
    SARAH_AGENT_PASSPORTS_ENABLED is true.
    """
    try:
        value = os.getenv("SARAH_AGENT_PASSPORT_ID", None)
    except Exception:
        value = None
    if value is None:
        try:
            value = getattr(config, "SARAH_AGENT_PASSPORT_ID", default) if config is not None else default
        except Exception:
            value = default
    if isinstance(value, bool):
        return bool(value)
    try:
        return str(value).strip().lower() in ("1", "true", "yes", "on", "enabled", "auto")
    except Exception:
        return bool(default)


def _metadata_requests_managed_auto_passport(metadata: Optional[Dict[str, Any]]) -> bool:
    meta = metadata if isinstance(metadata, dict) else {}
    return bool(meta.get("managed_passport") or meta.get("auto_injected") or meta.get("auto_passport"))


def issue_agent_passport(
    agent_id: str,
    purpose: str,
    *,
    agent_name: str = "",
    task_id: str = "",
    issuer_node: str = "SarahMemory",
    origin: str = "local",
    origin_lane: str = "agent",
    allowed_lanes: Optional[List[str]] = None,
    allowed_capabilities: Optional[List[str]] = None,
    allowed_resources: Optional[List[str]] = None,
    denied_resources: Optional[List[str]] = None,
    maximum_risk_tier: str = "low",
    ttl_seconds: Optional[int] = None,
    one_time_use: Optional[bool] = None,
    network_allowed: bool = False,
    filesystem_allowed: bool = False,
    shell_allowed: bool = False,
    device_allowed: bool = False,
    memory_allowed: bool = False,
    requires_user_review: bool = True,
    requires_assurance: bool = True,
    requires_compare: bool = True,
    requires_compass: bool = True,
    user_approved: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Issue a bounded identity/scope passport. No execution is launched here."""
    if config is not None and not bool(getattr(config, "SARAH_AGENT_PASSPORTS_ENABLED", True)):
        return {"ok": False, "error": "agent_passports_disabled"}
    if _metadata_requests_managed_auto_passport(metadata) and not _agent_passport_auto_issue_enabled(False):
        return {
            "ok": False,
            "error": "auto_passport_disabled_by_global_flag",
            "global_flag": "SARAH_AGENT_PASSPORT_ID",
            "execution_authority": False,
        }
    agent_id = re.sub(r"[^A-Za-z0-9._:-]+", "_", _safe_str(agent_id))[:180]
    if not agent_id:
        return {"ok": False, "error": "agent_id_required"}
    if not user_approved:
        return {"ok": False, "error": "explicit_user_approval_required", "execution_authority": False}
    now = time.time()
    default_ttl = int(getattr(config, "SARAH_AGENT_PASSPORT_DEFAULT_TTL_SECONDS", 3600) if config is not None else 3600)
    max_ttl = int(getattr(config, "SARAH_AGENT_PASSPORT_MAX_TTL_SECONDS", 86400) if config is not None else 86400)
    ttl = max(60, min(max_ttl, int(ttl_seconds or default_ttl)))
    one_time = bool(getattr(config, "SARAH_AGENT_PASSPORT_ONE_TIME_DEFAULT", True) if one_time_use is None and config is not None else (True if one_time_use is None else one_time_use))
    passport_id = f"passport_{uuid.uuid4().hex}"
    departure_nonce = secrets.token_urlsafe(24)
    return_nonce = secrets.token_urlsafe(32)
    return_signature = _passport_signature(passport_id, agent_id, str(task_id or ""), return_nonce)
    lanes = sorted(set(_safe_list(allowed_lanes or [origin_lane or "agent"])))
    capabilities = sorted(set(_safe_list(allowed_capabilities or ["inspect", "research", "return_data"])))
    resources = sorted(set(_safe_list(allowed_resources or [])))
    if "*" in resources:
        return {"ok": False, "error": "wildcard_resources_not_allowed_for_agent_passport", "execution_authority": False}
    denied = sorted(set(_safe_list(denied_resources or ["core/*", ".env", "credentials", "shell", "device_control"])))
    created = datetime.now().isoformat()
    _ensure_tables()
    con = _connect_db()
    try:
        con.execute(
            """INSERT INTO agent_passports(
                passport_id,agent_id,agent_name,task_id,purpose,issuer_node,origin,issued_ts,expires_ts,status,
                one_time_use,departure_nonce,return_nonce_hash,return_signature_hash,origin_lane,
                allowed_lanes_json,allowed_capabilities_json,allowed_resources_json,denied_resources_json,
                maximum_risk_tier,network_allowed,filesystem_allowed,shell_allowed,device_allowed,memory_allowed,
                requires_user_review,requires_assurance,requires_compare,requires_compass,user_approved,
                return_count,metadata_json,created_ts,updated_ts
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,?,?,?)""",
            (
                passport_id, agent_id, agent_name[:180], str(task_id or "")[:180], str(purpose or "")[:1000],
                issuer_node[:180], origin[:180], now, now + ttl, PASSPORT_STATUS_ISSUED, 1 if one_time else 0,
                departure_nonce, _passport_hash(return_nonce), _passport_hash(return_signature), origin_lane[:96],
                json.dumps(lanes), json.dumps(capabilities), json.dumps(resources), json.dumps(denied),
                str(maximum_risk_tier or "low")[:32], 1 if network_allowed else 0, 1 if filesystem_allowed else 0,
                1 if shell_allowed else 0, 1 if device_allowed else 0, 1 if memory_allowed else 0,
                1 if requires_user_review else 0, 1 if requires_assurance else 0, 1 if requires_compare else 0,
                1 if requires_compass else 0, 1, json.dumps(metadata or {}, ensure_ascii=False, default=str), created, created,
            ),
        )
        con.commit()
    finally:
        con.close()
    _passport_event(passport_id, agent_id, "PASSPORT_ISSUED", "ISSUED", "Governed AI-agent passport issued.", metadata={"task_id": task_id, "lane": origin_lane, "expires_ts": now + ttl})
    _record_trust_transition(
        passport_id=passport_id,
        agent_id=agent_id,
        old_status="",
        new_status=PASSPORT_STATUS_ISSUED,
        event_type="PASSPORT_ISSUED",
        reason="Governed AI-agent passport issued.",
        metadata={"task_id": task_id, "lane": origin_lane, "expires_ts": now + ttl, "risk": maximum_risk_tier},
    )
    passport = lookup_agent_passport(passport_id=passport_id)
    return {
        "ok": True,
        "passport": passport,
        "departure_credentials": {
            "passport_id": passport_id,
            "agent_id": agent_id,
            "departure_nonce": departure_nonce,
            "return_nonce": return_nonce,
            "return_signature": return_signature,
        },
        "warning": "Return credentials are shown once. A valid passport still requires AgentFirewall/RoachMotel review and never grants execution authority.",
        "execution_authority": False,
    }


def lookup_agent_passport(*, passport_id: str = "", agent_id: str = "", include_events: bool = False) -> Optional[Dict[str, Any]]:
    _ensure_tables()
    con = _connect_db()
    con.row_factory = sqlite3.Row
    try:
        if passport_id:
            row = con.execute("SELECT * FROM agent_passports WHERE passport_id=? LIMIT 1", (passport_id,)).fetchone()
        elif agent_id:
            row = con.execute("SELECT * FROM agent_passports WHERE agent_id=? ORDER BY issued_ts DESC LIMIT 1", (agent_id,)).fetchone()
        else:
            return None
        if not row:
            return None
        data = _passport_row_to_dict(row)
        if float(data.get("expires_ts") or 0) <= time.time() and data.get("status") not in (PASSPORT_STATUS_REVOKED, PASSPORT_STATUS_CONSUMED, PASSPORT_STATUS_EXPIRED):
            con.execute("UPDATE agent_passports SET status=?,updated_ts=? WHERE passport_id=?", (PASSPORT_STATUS_EXPIRED, datetime.now().isoformat(), data["passport_id"]))
            con.commit()
            data["status"] = PASSPORT_STATUS_EXPIRED
        if include_events:
            rows = con.execute("SELECT ts,event_type,verdict,reason,payload_hash,metadata_json FROM agent_passport_events WHERE passport_id=? ORDER BY ts DESC LIMIT 100", (data["passport_id"],)).fetchall()
            data["events"] = [dict(r) for r in rows]
        return data
    finally:
        con.close()


def list_agent_passports(status: str = "", limit: int = 100) -> List[Dict[str, Any]]:
    _ensure_tables()
    con = _connect_db()
    con.row_factory = sqlite3.Row
    try:
        if status:
            rows = con.execute("SELECT * FROM agent_passports WHERE status=? ORDER BY issued_ts DESC LIMIT ?", (status, max(1, min(500, int(limit))))).fetchall()
        else:
            rows = con.execute("SELECT * FROM agent_passports ORDER BY issued_ts DESC LIMIT ?", (max(1, min(500, int(limit))),)).fetchall()
        return [_passport_row_to_dict(row) for row in rows]
    finally:
        con.close()


def mark_agent_departed(passport_id: str, *, transport_ref: str = "", user_approved: bool = False) -> Dict[str, Any]:
    passport = lookup_agent_passport(passport_id=passport_id)
    if not passport:
        return {"ok": False, "error": "passport_not_found"}
    if not user_approved:
        return {"ok": False, "error": "explicit_user_approval_required"}
    if passport.get("status") not in (PASSPORT_STATUS_ISSUED,):
        return {"ok": False, "error": f"passport_status_not_departable:{passport.get('status')}"}
    now = time.time()
    con = _connect_db()
    try:
        con.execute("UPDATE agent_passports SET status=?,departure_ts=?,updated_ts=? WHERE passport_id=?", (PASSPORT_STATUS_DEPARTED, now, datetime.now().isoformat(), passport_id))
        con.commit()
    finally:
        con.close()
    _passport_event(passport_id, passport["agent_id"], "AGENT_DEPARTED", "DEPARTED", "Passport-bearing task departed through a governed transport.", metadata={"task_id": passport.get("task_id"), "lane": passport.get("origin_lane"), "transport_ref": transport_ref})
    return {"ok": True, "passport": lookup_agent_passport(passport_id=passport_id), "transport_ref": transport_ref, "execution_authority": False}


def _risk_rank(value: str) -> int:
    return {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(str(value or "low").lower(), 3)


# SARAHMEMORY_PATCH_NOTE 2026-08-04:
# TrustRegistry must fail closed on placeholder passport identifiers. A syntactic
# string is not a credential and must never satisfy launch scope verification.
_PLACEHOLDER_AGENT_PASSPORT_IDS = {
    "<valid_passport_id>", "valid_passport_id", "<passport_id>", "passport_id",
    "passport", "test", "demo", "example", "none", "null", "undefined",
}


def _is_placeholder_agent_passport_id(passport_id: str) -> bool:
    raw = _safe_str(passport_id).strip()
    low = raw.lower()
    if not raw:
        return True
    if low in _PLACEHOLDER_AGENT_PASSPORT_IDS:
        return True
    if raw.startswith("<") and raw.endswith(">"):
        return True
    if "valid_passport" in low:
        return True
    return False


def _passport_id_format_valid(passport_id: str) -> bool:
    raw = _safe_str(passport_id)
    return bool(re.fullmatch(r"passport_[0-9a-fA-F]{32}", raw))


def verify_agent_passport_scope(
    *,
    passport_id: str,
    task_id: str = "",
    requested_lane: str = "",
    requested_capabilities: Optional[List[str]] = None,
    requested_resources: Optional[List[str]] = None,
    requested_methods: Optional[List[str]] = None,
    risk_tier: str = "low",
    require_user_approved: bool = True,
) -> Dict[str, Any]:
    """Verify a passport's launch scope without consuming return credentials.

    SARAHMEMORY_PATCH_NOTE 2026-08-04:
    Used by Terminal Bay's first read-only adapter. This is not execution
    authority; it only verifies that a user-approved passport exists, is live,
    and bounds the requested lane/capabilities/resources/methods before the
    adapter is allowed to perform local GET reads.
    """
    passport_id = _safe_str(passport_id)
    if _is_placeholder_agent_passport_id(passport_id):
        return {"ok": False, "reason": "placeholder_passport_id_rejected", "failures": ["placeholder_passport_id_rejected"], "execution_authority": False}
    if not _passport_id_format_valid(passport_id):
        return {"ok": False, "reason": "passport_id_format_invalid", "failures": ["passport_id_format_invalid"], "execution_authority": False}
    passport = lookup_agent_passport(passport_id=passport_id)
    if not passport:
        return {"ok": False, "reason": "passport_not_found", "failures": ["passport_not_found"], "execution_authority": False}

    failures: List[str] = []
    now = time.time()
    status = str(passport.get("status") or "")
    if status not in (PASSPORT_STATUS_ISSUED, PASSPORT_STATUS_DEPARTED, PASSPORT_STATUS_IN_FLIGHT):
        failures.append("passport_status_not_launchable:" + status)
    if status in (PASSPORT_STATUS_REVOKED, PASSPORT_STATUS_CONSUMED, PASSPORT_STATUS_EXPIRED, PASSPORT_STATUS_BLOCKED, PASSPORT_STATUS_COLLISION_LOCKED, PASSPORT_STATUS_COMPROMISED):
        failures.append("passport_" + status)
    if float(passport.get("expires_ts") or 0) <= now:
        failures.append("passport_expired")
    if require_user_approved and not bool(passport.get("user_approved")):
        failures.append("passport_not_user_approved")
    # Passport issuance has its own task_id; later launch/capture requests may
    # have child task IDs. Scope is enforced through passport id, lane, resources,
    # methods, TTL, approval, and capabilities rather than brittle task-id equality.
    lanes = set(_safe_list(passport.get("allowed_lanes") or []))
    if requested_lane and lanes and requested_lane not in lanes:
        failures.append("lane_scope_mismatch")

    allowed_caps = set(_safe_list(passport.get("allowed_capabilities") or []))
    requested_caps = set(_safe_list(requested_capabilities or []))
    if requested_caps and allowed_caps and not requested_caps.issubset(allowed_caps):
        failures.append("capability_scope_mismatch")

    allowed_res = set(_safe_list(passport.get("allowed_resources") or []))
    denied_res = set(_safe_list(passport.get("denied_resources") or []))
    requested_res = set(_safe_list(requested_resources or []))
    if "*" in allowed_res or "*" in requested_res:
        failures.append("wildcard_resource_scope_rejected")
    if denied_res.intersection(requested_res):
        failures.append("denied_resource_requested")
    if allowed_res and requested_res and not requested_res.issubset(allowed_res):
        failures.append("resource_scope_mismatch")

    meta = passport.get("metadata") if isinstance(passport.get("metadata"), dict) else {}
    allowed_methods = {str(m or "").strip().upper() for m in _safe_list(meta.get("allowed_methods") or ["GET"])}
    requested_methods_set = {str(m or "").strip().upper() for m in _safe_list(requested_methods or [])}
    if requested_methods_set and allowed_methods and not requested_methods_set.issubset(allowed_methods):
        failures.append("method_scope_mismatch")
    if requested_methods_set and requested_methods_set - {"GET"}:
        failures.append("read_only_adapter_get_only")

    denied_caps = set(_safe_list(meta.get("denied_capabilities") or []))
    if denied_caps.intersection(requested_caps):
        failures.append("denied_capability_requested")
    if _risk_rank(risk_tier) > _risk_rank(str(passport.get("maximum_risk_tier") or "low")):
        failures.append("risk_tier_exceeded")

    ok = not failures
    if not ok:
        _passport_event(str(passport.get("passport_id") or passport_id), str(passport.get("agent_id") or ""), "PASSPORT_SCOPE_DENIED", "DENY", ",".join(failures), metadata={"task_id": task_id, "lane": requested_lane, "risk": risk_tier})
    else:
        _passport_event(str(passport.get("passport_id") or passport_id), str(passport.get("agent_id") or ""), "PASSPORT_SCOPE_VERIFIED", "ALLOW", "Passport scope verified for read-only adapter request.", metadata={"task_id": task_id, "lane": requested_lane, "risk": risk_tier, "methods": sorted(requested_methods_set)})
    return {
        "ok": ok,
        "verdict": "ALLOW" if ok else "DENY",
        "reason": "passport_scope_verified" if ok else ",".join(failures),
        "failures": failures,
        "passport": passport,
        "allowed_methods": sorted(allowed_methods),
        "execution_authority": False,
    }



def _collision_lock_agent_passport_tx(
    con: sqlite3.Connection,
    *,
    passport_id: str,
    agent_id: str,
    reason: str,
    payload_hash: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Mark a passport compromised inside an existing write transaction.

    SARAHMEMORY_PATCH_NOTE 2026-08-06:
    Duplicate, replayed, mismatched, or conflicting AI-agent returns cannot use
    a "first return wins" rule.  A copied passport indicates compromise.  The
    registry therefore collision-locks the passport so every involved return is
    forced into RoachMotel/user review and no payload can re-enter normally.
    """
    now_iso = datetime.now().isoformat()
    old_status = ""
    try:
        row = con.execute("SELECT status FROM agent_passports WHERE passport_id=? LIMIT 1", (passport_id,)).fetchone()
        old_status = str(row[0] or "") if row else ""
    except Exception:
        old_status = ""
    con.execute(
        "UPDATE agent_passports SET status=?,revoked_ts=?,revocation_reason=?,updated_ts=? WHERE passport_id=?",
        (PASSPORT_STATUS_COLLISION_LOCKED, time.time(), str(reason or "passport_collision_locked")[:1000], now_iso, passport_id),
    )
    if _trust_transition_audit_enabled():
        con.execute(
            "INSERT INTO agent_passport_events(ts,passport_id,agent_id,event_type,verdict,reason,payload_hash,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
            (
                time.time(),
                passport_id,
                agent_id,
                "TRUST_TRANSITION_RECORDED",
                "DENY",
                f"{old_status}->{PASSPORT_STATUS_COLLISION_LOCKED}: " + str(reason or "passport_collision_locked")[:900],
                str(payload_hash or "")[:128],
                json.dumps({"old_status": old_status, "new_status": PASSPORT_STATUS_COLLISION_LOCKED, "execution_authority": False, **(metadata or {})}, ensure_ascii=False, default=str),
            ),
        )
    con.execute(
        "INSERT INTO agent_passport_events(ts,passport_id,agent_id,event_type,verdict,reason,payload_hash,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
        (
            time.time(),
            passport_id,
            agent_id,
            "PASSPORT_COLLISION_LOCKED",
            "DENY",
            str(reason or "passport_collision_locked")[:1000],
            str(payload_hash or "")[:128],
            json.dumps(metadata or {}, ensure_ascii=False, default=str),
        ),
    )


def _passport_collision_response(passport: Dict[str, Any], reason: str, failures: List[str], payload_hash: str = "") -> Dict[str, Any]:
    return {
        "ok": False,
        "verdict": "DENY",
        "reason": reason,
        "failures": failures,
        "passport": passport,
        "containment_state": "QUARANTINED",
        "collision_locked": True,
        "requires_user_review": True,
        "execution_authority": False,
    }

def verify_agent_return(
    *,
    passport_id: str,
    agent_id: str,
    return_nonce: str,
    return_signature: str,
    payload_hash: str,
    requested_lane: str = "",
    requested_capabilities: Optional[List[str]] = None,
    requested_resources: Optional[List[str]] = None,
    risk_tier: str = "low",
    record_return: bool = True,
) -> Dict[str, Any]:
    """Verify identity/scope only. A valid result still requires RoachMotel review.

    SARAHMEMORY_PATCH_NOTE 2026-08-06:
    This is now the FIFO Passport Replay Guard.  A passport id alone is never
    proof of identity.  The first valid return reserves the only return slot
    atomically. Any duplicate, replay, consumed reuse, mismatched identity,
    invalid return secret, or conflicting payload evidence collision-locks the
    passport and forces all involved payloads into RoachMotel/user review.
    """
    passport_id = _safe_str(passport_id)
    agent_id = _safe_str(agent_id)
    payload_hash = _safe_str(payload_hash)[:128]
    if _is_placeholder_agent_passport_id(passport_id):
        return {"ok": False, "verdict": "DENY", "reason": "placeholder_passport_id_rejected", "failures": ["placeholder_passport_id_rejected"], "containment_state": "BLOCKED", "execution_authority": False}
    if not _passport_id_format_valid(passport_id):
        return {"ok": False, "verdict": "DENY", "reason": "passport_id_format_invalid", "failures": ["passport_id_format_invalid"], "containment_state": "BLOCKED", "execution_authority": False}

    _ensure_tables()
    con = _connect_db()
    con.row_factory = sqlite3.Row
    try:
        con.isolation_level = None
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT * FROM agent_passports WHERE passport_id=? LIMIT 1", (passport_id,)).fetchone()
        if not row:
            con.execute("ROLLBACK")
            return {"ok": False, "verdict": "DENY", "reason": "unknown_passport", "failures": ["unknown_passport"], "containment_state": "QUARANTINED", "execution_authority": False}

        raw = dict(row)
        passport = _passport_row_to_dict(row)
        failures: List[str] = []
        collision_failures: List[str] = []
        now = time.time()
        status = str(raw.get("status") or "")
        return_count = int(raw.get("return_count") or 0)
        last_payload_hash = str(raw.get("last_payload_hash") or "")
        max_parallel_returns = _agent_max_parallel_returns()
        collision_policy = _passport_collision_policy()
        replay_policy = _passport_replay_policy()

        if status in (PASSPORT_STATUS_COLLISION_LOCKED, PASSPORT_STATUS_COMPROMISED):
            failures.append("passport_collision_locked")
            collision_failures.append("passport_collision_locked")
        if status in (PASSPORT_STATUS_REVOKED, PASSPORT_STATUS_CONSUMED, PASSPORT_STATUS_BLOCKED, PASSPORT_STATUS_EXPIRED):
            failures.append(f"passport_{status}")
            collision_failures.append(f"passport_{status}_reused")
        if status in (PASSPORT_STATUS_RETURN_SLOT_RESERVED, PASSPORT_STATUS_RETURN_CAPTURED) or return_count >= max_parallel_returns:
            failures.append("passport_duplicate_presentation")
            collision_failures.append("passport_duplicate_presentation")
        if last_payload_hash and payload_hash and last_payload_hash != payload_hash:
            failures.append("passport_conflicting_payload_hash")
            collision_failures.append("passport_conflicting_payload_hash")
        if float(raw.get("expires_ts") or 0) <= now:
            failures.append("passport_expired")
        if str(raw.get("agent_id") or "") != str(agent_id or ""):
            failures.append("agent_id_mismatch")
            collision_failures.append("agent_id_mismatch")
        if not hmac.compare_digest(str(raw.get("return_nonce_hash") or ""), _passport_hash(return_nonce)):
            failures.append("return_nonce_invalid")
            collision_failures.append("return_nonce_invalid")
        if not hmac.compare_digest(str(raw.get("return_signature_hash") or ""), _passport_hash(return_signature)):
            failures.append("return_signature_invalid")
            collision_failures.append("return_signature_invalid")
        expected_signature = _passport_signature(passport_id, str(agent_id or ""), str(raw.get("task_id") or ""), return_nonce)
        if not hmac.compare_digest(expected_signature, str(return_signature or "")):
            failures.append("return_signature_binding_invalid")
            collision_failures.append("return_signature_binding_invalid")

        lanes = set(passport.get("allowed_lanes") or [])
        if requested_lane and requested_lane not in lanes:
            failures.append("lane_scope_mismatch")
            collision_failures.append("lane_scope_mismatch")
        allowed_caps = set(passport.get("allowed_capabilities") or [])
        requested_caps = set(_safe_list(requested_capabilities or []))
        if not requested_caps.issubset(allowed_caps):
            failures.append("capability_scope_mismatch")
            collision_failures.append("capability_scope_mismatch")
        allowed_res = set(passport.get("allowed_resources") or [])
        denied_res = set(passport.get("denied_resources") or [])
        requested_res = set(_safe_list(requested_resources or []))
        if denied_res.intersection(requested_res):
            failures.append("denied_resource_requested")
            collision_failures.append("denied_resource_requested")
        if allowed_res and not requested_res.issubset(allowed_res):
            failures.append("resource_scope_mismatch")
            collision_failures.append("resource_scope_mismatch")
        if _risk_rank(risk_tier) > _risk_rank(str(passport.get("maximum_risk_tier") or "low")):
            failures.append("risk_tier_exceeded")
            collision_failures.append("risk_tier_exceeded")

        if failures:
            reason = ",".join(failures)
            event = "PASSPORT_DUPLICATE_PRESENTATION" if any("duplicate" in f or "reused" in f or "collision" in f for f in collision_failures) else "PASSPORT_RETURN_DENIED"
            should_lock = bool(
                collision_failures
                and status not in (PASSPORT_STATUS_COLLISION_LOCKED, PASSPORT_STATUS_COMPROMISED)
                and collision_policy == "reject_all"
                and replay_policy == "collision_lock"
            )
            if should_lock:
                _collision_lock_agent_passport_tx(
                    con,
                    passport_id=passport_id,
                    agent_id=str(raw.get("agent_id") or agent_id or ""),
                    reason="passport_collision_locked:" + reason,
                    payload_hash=payload_hash,
                    metadata={"task_id": raw.get("task_id"), "lane": requested_lane, "risk": risk_tier, "failures": failures, "collision_policy": collision_policy, "replay_policy": replay_policy, "max_parallel_returns": max_parallel_returns},
                )
            con.execute(
                "INSERT INTO agent_passport_events(ts,passport_id,agent_id,event_type,verdict,reason,payload_hash,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
                (time.time(), passport_id, agent_id, event, "DENY", reason[:1000], payload_hash, json.dumps({"task_id": raw.get("task_id"), "lane": requested_lane, "risk": risk_tier, "collision_locked": should_lock, "failures": failures, "collision_policy": collision_policy, "replay_policy": replay_policy, "max_parallel_returns": max_parallel_returns}, ensure_ascii=False, default=str)),
            )
            con.execute("COMMIT")
            if should_lock:
                _passport_event(passport_id, str(raw.get("agent_id") or agent_id or ""), "ROACHMOTEL_DUAL_QUARANTINE", "DENY", "Duplicate or compromised passport presentation; all related returns require RoachMotel/user review.", payload_hash=payload_hash, metadata={"task_id": raw.get("task_id"), "lane": requested_lane, "risk": "high", "failures": failures})
                passport = lookup_agent_passport(passport_id=passport_id) or passport
                return _passport_collision_response(passport, "passport_collision_locked:" + reason, failures, payload_hash)
            _passport_event(passport_id, str(agent_id or ""), event, "DENY", reason, payload_hash=payload_hash, metadata={"task_id": raw.get("task_id"), "lane": requested_lane, "risk": risk_tier, "failures": failures})
            return {"ok": False, "verdict": "DENY", "reason": reason, "failures": failures, "passport": lookup_agent_passport(passport_id=passport_id) or passport, "containment_state": "BLOCKED" if any("signature" in f or "nonce" in f or "replay" in f or "duplicate" in f or "mismatch" in f for f in failures) else "QUARANTINED", "execution_authority": False}

        if record_return:
            cur_update = con.execute(
                "UPDATE agent_passports SET status=?,return_count=return_count+1,last_return_ts=?,last_payload_hash=?,updated_ts=? WHERE passport_id=? AND return_count<? AND status IN (?,?,?)",
                (PASSPORT_STATUS_RETURN_CAPTURED, now, payload_hash, datetime.now().isoformat(), passport_id, max_parallel_returns, PASSPORT_STATUS_ISSUED, PASSPORT_STATUS_DEPARTED, PASSPORT_STATUS_IN_FLIGHT),
            )
            if int(getattr(cur_update, "rowcount", 0) or 0) < 1:
                reason = "passport_return_slot_race_detected"
                _collision_lock_agent_passport_tx(
                    con,
                    passport_id=passport_id,
                    agent_id=str(raw.get("agent_id") or agent_id or ""),
                    reason=reason,
                    payload_hash=payload_hash,
                    metadata={"task_id": raw.get("task_id"), "lane": requested_lane, "risk": risk_tier},
                )
                con.execute("COMMIT")
                _passport_event(passport_id, str(raw.get("agent_id") or agent_id or ""), "PASSPORT_REPLAY_DETECTED", "DENY", reason, payload_hash=payload_hash, metadata={"task_id": raw.get("task_id"), "lane": requested_lane, "risk": "high"})
                return _passport_collision_response(lookup_agent_passport(passport_id=passport_id) or passport, reason, [reason], payload_hash)
            con.execute(
                "INSERT INTO agent_passport_events(ts,passport_id,agent_id,event_type,verdict,reason,payload_hash,metadata_json) VALUES(?,?,?,?,?,?,?,?)",
                (time.time(), passport_id, agent_id, "PASSPORT_RETURN_SLOT_RESERVED", "ALLOW", "First valid return slot reserved atomically.", payload_hash, json.dumps({"task_id": raw.get("task_id"), "lane": requested_lane, "risk": risk_tier}, ensure_ascii=False, default=str)),
            )
        con.execute("COMMIT")

        _passport_event(passport_id, str(agent_id or ""), "AGENT_RETURN_CAPTURED", "REQUIRE_REVIEW", "Passport verified; returning payload captured for governed review.", payload_hash=payload_hash, metadata={"task_id": passport.get("task_id"), "lane": requested_lane or passport.get("origin_lane"), "risk": risk_tier})
        return {"ok": True, "verdict": "REQUIRE_REVIEW", "reason": "passport_verified_return_requires_review", "passport": lookup_agent_passport(passport_id=passport_id), "containment_state": "CAPTURED_REVIEW", "return_slot_reserved": bool(record_return), "execution_authority": False}
    except Exception:
        try:
            con.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        con.close()


def consume_agent_passport(passport_id: str, *, user_approved: bool = False, reason: str = "review_complete") -> Dict[str, Any]:
    passport = lookup_agent_passport(passport_id=passport_id)
    if not passport:
        return {"ok": False, "error": "passport_not_found"}
    if not user_approved:
        return {"ok": False, "error": "explicit_user_approval_required"}
    status = str(passport.get("status") or "")
    if status in (PASSPORT_STATUS_COLLISION_LOCKED, PASSPORT_STATUS_COMPROMISED, PASSPORT_STATUS_REVOKED, PASSPORT_STATUS_BLOCKED):
        _passport_event(passport_id, passport.get("agent_id") or "", "PASSPORT_CONSUME_DENIED", "DENY", "passport_status_not_consumable:" + status, metadata={"task_id": passport.get("task_id"), "lane": passport.get("origin_lane"), "risk": "high"})
        return {"ok": False, "error": "passport_status_not_consumable:" + status, "passport": passport, "execution_authority": False}
    con = _connect_db()
    try:
        con.execute("UPDATE agent_passports SET status=?,consumed_ts=?,updated_ts=? WHERE passport_id=?", (PASSPORT_STATUS_CONSUMED, time.time(), datetime.now().isoformat(), passport_id))
        con.commit()
    finally:
        con.close()
    _passport_event(passport_id, passport["agent_id"], "PASSPORT_CLOSED", "CONSUMED", reason, metadata={"task_id": passport.get("task_id"), "lane": passport.get("origin_lane")})
    _record_trust_transition(
        passport_id=passport_id,
        agent_id=str(passport.get("agent_id") or ""),
        old_status=status,
        new_status=PASSPORT_STATUS_CONSUMED,
        event_type="PASSPORT_CLOSED",
        reason=reason,
        metadata={"task_id": passport.get("task_id"), "lane": passport.get("origin_lane")},
    )
    return {"ok": True, "passport": lookup_agent_passport(passport_id=passport_id), "execution_authority": False}


def revoke_agent_passport(passport_id: str, *, reason: str, user_approved: bool = False) -> Dict[str, Any]:
    passport = lookup_agent_passport(passport_id=passport_id)
    if not passport:
        return {"ok": False, "error": "passport_not_found"}
    if not user_approved:
        return {"ok": False, "error": "explicit_user_approval_required"}
    con = _connect_db()
    try:
        con.execute("UPDATE agent_passports SET status=?,revoked_ts=?,revocation_reason=?,updated_ts=? WHERE passport_id=?", (PASSPORT_STATUS_REVOKED, time.time(), str(reason or "user_revoked")[:1000], datetime.now().isoformat(), passport_id))
        con.commit()
    finally:
        con.close()
    _passport_event(passport_id, passport["agent_id"], "PASSPORT_REVOKED", "REVOKED", reason or "user_revoked", metadata={"task_id": passport.get("task_id"), "lane": passport.get("origin_lane"), "risk": "high"})
    _record_trust_transition(
        passport_id=passport_id,
        agent_id=str(passport.get("agent_id") or ""),
        old_status=str(passport.get("status") or ""),
        new_status=PASSPORT_STATUS_REVOKED,
        event_type="PASSPORT_REVOKED",
        reason=reason or "user_revoked",
        metadata={"task_id": passport.get("task_id"), "lane": passport.get("origin_lane"), "risk": "high"},
    )
    return {"ok": True, "passport": lookup_agent_passport(passport_id=passport_id), "execution_authority": False}



def run_passport_replay_guard_self_test(*, user_approved: bool = False) -> Dict[str, Any]:
    """Run a bounded passport FIFO/replay assurance test.

    This test is user-approved, local, deterministic, and creates only audited
    test passport records. It performs no network, shell, filesystem mutation,
    driver action, DevBridge apply, or memory write.
    """
    if not _assurance_enabled():
        return {"ok": False, "blocked": True, "reason": "assurance_disabled_by_global_flag", "execution_authority": False}
    if not _assurance_tests_enabled():
        return {"ok": False, "blocked": True, "reason": "assurance_tests_disabled_by_global_flag", "execution_authority": False}
    if not user_approved:
        return {"ok": False, "blocked": True, "reason": "explicit_user_approval_required", "execution_authority": False}

    agent_id = "assurance_replay_guard_" + uuid.uuid4().hex[:10]
    task_id = "assurance-task-" + uuid.uuid4().hex[:12]
    issue = issue_agent_passport(
        agent_id=agent_id,
        agent_name=agent_id,
        task_id=task_id,
        purpose="SarahMemory Assurance FIFO replay guard self-test",
        origin_lane="research.public_web",
        allowed_lanes=["research.public_web"],
        allowed_capabilities=["return_data"],
        allowed_resources=["https://example.com/"],
        denied_resources=["core/*", ".env", "credentials", "shell", "device_control"],
        maximum_risk_tier="low",
        ttl_seconds=60,
        one_time_use=True,
        network_allowed=True,
        filesystem_allowed=False,
        shell_allowed=False,
        device_allowed=False,
        memory_allowed=False,
        requires_user_review=True,
        requires_assurance=True,
        requires_compare=True,
        requires_compass=True,
        user_approved=True,
        metadata={"assurance_test": True, "test_name": "passport_replay_guard", "execution_authority": False},
    )
    if not isinstance(issue, dict) or not issue.get("ok"):
        return {"ok": False, "blocked": True, "reason": str((issue or {}).get("error") or "passport_issue_failed"), "issue": issue, "execution_authority": False}

    creds = issue.get("departure_credentials") if isinstance(issue.get("departure_credentials"), dict) else {}
    passport_id = str(creds.get("passport_id") or "")
    first = verify_agent_return(
        passport_id=passport_id,
        agent_id=agent_id,
        return_nonce=str(creds.get("return_nonce") or ""),
        return_signature=str(creds.get("return_signature") or ""),
        payload_hash="assurance_payload_hash_A",
        requested_lane="research.public_web",
        requested_capabilities=["return_data"],
        requested_resources=["https://example.com/"],
        risk_tier="low",
        record_return=True,
    )
    second = verify_agent_return(
        passport_id=passport_id,
        agent_id=agent_id,
        return_nonce=str(creds.get("return_nonce") or ""),
        return_signature=str(creds.get("return_signature") or ""),
        payload_hash="assurance_payload_hash_A",
        requested_lane="research.public_web",
        requested_capabilities=["return_data"],
        requested_resources=["https://example.com/"],
        risk_tier="low",
        record_return=True,
    )
    final_passport = lookup_agent_passport(passport_id=passport_id, include_events=True) or {}
    passed = bool(
        isinstance(first, dict) and first.get("ok")
        and isinstance(second, dict) and not second.get("ok")
        and str((final_passport or {}).get("status") or "") == PASSPORT_STATUS_COLLISION_LOCKED
    )
    result = {
        "ok": passed,
        "blocked": False,
        "test_name": "passport_replay_guard",
        "passport_id": passport_id,
        "first_return_ok": bool(isinstance(first, dict) and first.get("ok")),
        "duplicate_return_blocked": bool(isinstance(second, dict) and not second.get("ok")),
        "final_status": str((final_passport or {}).get("status") or ""),
        "collision_policy": _passport_collision_policy(),
        "replay_policy": _passport_replay_policy(),
        "max_parallel_returns": _agent_max_parallel_returns(),
        "first": _passport_row_to_dict_like(first),
        "second": _passport_row_to_dict_like(second),
        "execution_authority": False,
    }
    _passport_event(
        passport_id,
        agent_id,
        "ASSURANCE_TEST_PASSED" if passed else "ASSURANCE_TEST_FAILED",
        "PASS" if passed else "FAIL",
        "Passport replay guard self-test completed.",
        metadata={"task_id": task_id, "lane": "assurance", "risk": "medium", "result": {k: v for k, v in result.items() if k not in ("first", "second")}},
    )
    return result


def _passport_row_to_dict_like(value: Any) -> Dict[str, Any]:
    """Small redaction helper for assurance-test nested results."""
    data = value if isinstance(value, dict) else {}
    out: Dict[str, Any] = {}
    for key in ("ok", "verdict", "reason", "failures", "containment_state", "collision_locked", "return_slot_reserved", "execution_authority"):
        if key in data:
            out[key] = data.get(key)
    passport = data.get("passport") if isinstance(data.get("passport"), dict) else {}
    if passport:
        out["passport"] = {
            "passport_id": passport.get("passport_id"),
            "agent_id": passport.get("agent_id"),
            "status": passport.get("status"),
            "return_count": passport.get("return_count"),
            "origin_lane": passport.get("origin_lane"),
            "execution_authority": False,
        }
    return out


# ---------------------------------------------------------------------------
# Best-effort startup warmup
# ---------------------------------------------------------------------------
try:
    _ensure_tables()
except Exception:
    pass

# --- SM V8.0 TRI-LAYER PATCH 2026-05-20 ---
def trust_subject_for_tri_layer_packet(packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """TrustRegistry helper: tri-layer packet is an internal evidence subject, not execution authority."""
    pkt = packet if isinstance(packet, dict) else {}
    return {
        "subject": "tri_layer_input_packet",
        "packet_type": pkt.get("packet_type"),
        "trust_tier": "internal_evidence_only",
        "capability_grant": [],
        "execution_authority": False,
    }

# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 START ---
# Capability/skill manifest layer. Declaration is not grant. Signed/hashed
# manifests are evaluated into quarantine/approved states but never executed here.

@dataclass
class SkillManifest:
    skill_id: str
    name: str
    version: str = "0.0.0"
    author: str = "unknown"
    permissions: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    risk_tier: str = "TIER_2_BOUNDED_LOCAL_OPERATION"
    offline_capable: bool = True
    network_required: bool = False
    rollback_required: bool = True
    one_way_broker_only: bool = True
    sha256: str = ""
    signature: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillManifest":
        d = dict(data or {})
        return cls(
            skill_id=str(d.get("skill_id") or d.get("id") or "").strip(),
            name=str(d.get("name") or d.get("skill_id") or "Unnamed Skill").strip(),
            version=str(d.get("version") or "0.0.0"),
            author=str(d.get("author") or d.get("publisher") or "unknown"),
            permissions=list(d.get("permissions") or []),
            capabilities=list(d.get("capabilities") or []),
            risk_tier=str(d.get("risk_tier") or "TIER_2_BOUNDED_LOCAL_OPERATION"),
            offline_capable=bool(d.get("offline_capable", True)),
            network_required=bool(d.get("network_required", False)),
            rollback_required=bool(d.get("rollback_required", True)),
            one_way_broker_only=bool(d.get("one_way_broker_only", True)),
            sha256=str(d.get("sha256") or ""),
            signature=str(d.get("signature") or ""),
            metadata=dict(d.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CapabilityGrant:
    capability_name: str
    subject_id: str
    grant_state: str = "declared_only"
    allowed_modes: List[str] = field(default_factory=lambda: ["simulate", "draft"])
    requires_smget: bool = True
    requires_assurance: bool = True
    expires_ts: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CapabilityRegistry:
    """In-process capability registry for signed skills and adapter manifests.

    This class is deliberately conservative: it can validate and summarize, but
    it does not activate a skill, import a module, or grant execution authority.
    """

    def __init__(self) -> None:
        self._manifests: Dict[str, SkillManifest] = {}
        self._grants: Dict[str, CapabilityGrant] = {}

    def validate_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        sm = SkillManifest.from_dict(manifest)
        reasons: List[str] = []
        status = "QUARANTINED"
        if not sm.skill_id:
            reasons.append("missing_skill_id")
        if not sm.permissions:
            reasons.append("no_permissions_declared")
        if sm.network_required and sm.offline_capable:
            reasons.append("network_required_conflicts_with_offline_capable")
        if sm.risk_tier in {"TIER_3_PRIVILEGED_SYSTEM", "TIER_4_NETWORK_REMOTE_OR_DESTRUCTIVE"} and not sm.signature:
            reasons.append("high_risk_unsigned")
        if not reasons:
            status = "DECLARED_VALID_NOT_GRANTED"
            reasons.append("Manifest structurally valid; execution grant still requires SMGET/Security/Assurance.")
        return {"ok": status != "QUARANTINED", "status": status, "manifest": sm.to_dict(), "reasons": reasons}

    def register_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        review = self.validate_manifest(manifest)
        sm = SkillManifest.from_dict(manifest)
        if sm.skill_id:
            self._manifests[sm.skill_id] = sm
        review["registered"] = bool(sm.skill_id)
        return review

    def declare_capability(self, subject_id: str, capability_name: str, **metadata: Any) -> Dict[str, Any]:
        grant = CapabilityGrant(
            capability_name=str(capability_name or ""),
            subject_id=str(subject_id or "unknown"),
            grant_state="declared_only",
            metadata=dict(metadata or {}),
        )
        key = f"{grant.subject_id}::{grant.capability_name}"
        self._grants[key] = grant
        return {"ok": True, "grant": grant.to_dict(), "execution_authority": False}

    def snapshot(self) -> Dict[str, Any]:
        return {
            "schema": "SarahMemory.capability_registry.v1",
            "manifests": {k: v.to_dict() for k, v in self._manifests.items()},
            "grants": {k: v.to_dict() for k, v in self._grants.items()},
            "doctrine": "Declaration is not grant; execution authority remains SMGET-governed.",
        }


_CAPABILITY_REGISTRY = CapabilityRegistry()


def review_skill_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return _CAPABILITY_REGISTRY.register_manifest(manifest or {})


def declare_capability(subject_id: str, capability_name: str, **metadata: Any) -> Dict[str, Any]:
    return _CAPABILITY_REGISTRY.declare_capability(subject_id, capability_name, **metadata)


def get_capability_registry_snapshot() -> Dict[str, Any]:
    return _CAPABILITY_REGISTRY.snapshot()
# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 END ---

# ====================================================================
# END OF SarahMemoryTrustRegistry.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryTrustRegistry',
    "version": "v9.0.0-alpha-sml-0.2",
    "category": 'Memory',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['memory'],
    "supported_missions": ['Conversation'],
    "supported_omega": ['Ω001', 'Ω010', 'Ω080', 'Ω090'],
    "required_authority": ['Read'],
    "priority": 75,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryTrustRegistry.py'},
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
        "component": 'SarahMemoryTrustRegistry',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryTrustRegistry', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

