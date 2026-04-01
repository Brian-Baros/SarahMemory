"""--==The SarahMemory Project==--
File: SarahMemoryTrustRegistry.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2026-03-31
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

PURPOSE:
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
# NOTES = "SMGET canonical subject trust and capability registry. Tracks callers, trust tiers, scoped grants, manifests, quarantine state, and registry-backed authority boundaries."
# --- SARAHMETA END ---

import json
import logging
import os
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
MODULE_VERSION = "8.0.0"
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


def _safe_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, tuple):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, set):
        return sorted([str(x).strip() for x in value if str(x).strip()])
    try:
        txt = str(value).strip()
        return [txt] if txt else []
    except Exception:
        return []


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
# Best-effort startup warmup
# ---------------------------------------------------------------------------
try:
    _ensure_tables()
except Exception:
    pass

