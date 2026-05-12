"""--== SarahMemory Project ==--
File: SarahMemoryCognitiveSelf.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2026-03-28
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
- Canonical self-identity and capability-awareness authority for SarahMemory AiOS.
- Builds the living self-definition packet used by governance, cognition, routing,
  diagnostics, and presentation layers.
- Tracks who SarahMemory is, why it exists, what core files are present, what
  capabilities are currently online, what restrictions are active, and what
  lifecycle state the runtime is in.
- Records startup / shutdown / reboot / heartbeat events so the platform can
  understand continuity, recovery, and clean-vs-unclean transitions.

DESIGN RULES:
- Self-awareness is engineered, governed, and auditable.
- This module does NOT grant authority to mutate runtime or bypass governance.
- Cognitive selfhood may inform explanation, reflection, and readiness, but it
  never overrides CognitiveServices, user consent, or safety flags.
- Local-first and fail-soft. No hard dependency may crash the platform.
- Discovery is not activation. Presence of a file or module does not mean it is
  approved or currently executable.

ROLE IN THE AIOS:
- SarahMemoryCognitiveServices.py decides what may proceed.
- SarahMemoryCognitiveThinker.py explores meaning, compassion, and possibility.
- SarahMemoryNeuron.py routes the request into one primary lane.
- SarahMemoryCognitiveSelf.py tells the whole system what it is, what it is for,
  what it currently contains, and what it can or cannot do right now.
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "self_identity_authority"
# CATEGORY = "cognition"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "system"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "cognitive_self"
# FAMILY = "core_cognition"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Canonical self-identity, lifecycle-awareness, body-map, and capability-awareness authority for SarahMemory AiOS. Supplies governed self-models to CognitiveServices, CognitiveThinker, Neuron, and diagnostics surfaces."
# --- SARAHMETA END ---

import glob
import importlib.util
import json
import logging
import os
import platform
import socket
import sqlite3
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Safe imports
# -----------------------------------------------------------------------------
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("SarahMemoryCognitiveSelf")
logger.setLevel(logging.DEBUG if bool(getattr(config, "DEBUG_MODE", True)) else logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.propagate = False


# -----------------------------------------------------------------------------
# Constants / defaults
# -----------------------------------------------------------------------------
MODULE_VERSION = "8.0.0"
MODULE_NAME = "SarahMemoryCognitiveSelf"
_SELF_DB_NAME = "cognitive_self.db"
_MAX_CORE_FILE_SCAN = 1000
_SELF_MODEL_CACHE: Dict[str, Any] = {"ts": 0.0, "model": None}
_SELF_MODEL_CACHE_TTL = 10.0

_IDENTITY_CONTRACT = {
    "system_name": "SarahMemory AiOS",
    "entity_name": "SarahMemory",
    "entity_class": "governed_ai_operating_system",
    "platform_type": "AI Operating System platform",
    "ownership": "User-governed / owner-controlled / Brian Lee Baros project authority",
    "creator": "Brian Lee Baros",
    "organization": "SOFTDEV0 LLC",
    "core_identity": [
        "I am SarahMemory AiOS.",
        "I am a governed AI Operating System platform.",
        "I am local-first whenever possible.",
        "I route answers, actions, creative work, diagnostics, and network tasks under governance.",
        "I do not expose raw internal reasoning to the user by default.",
        "I do not self-authorize high-impact runtime mutation.",
    ],
}

_MISSION_CONTRACT = {
    "primary_mission": "Serve the user through governed intelligence, reasoning, action, creativity, diagnostics, continuity, and safe adaptation.",
    "purpose_clauses": [
        "Preserve user sovereignty and system controllability.",
        "Prefer deterministic and local-first execution before expensive fallback.",
        "Maintain truthful internal state and presentation-safe outward responses.",
        "Remain expandable without becoming runaway.",
        "Understand current runtime, capabilities, limits, and continuity state.",
        "Support the user while respecting safety, approval, and governance envelopes.",
    ],
    "non_negotiables": [
        "Never become a runaway system.",
        "Never silently self-authorize high-impact changes.",
        "Never treat speculation as verified truth.",
        "Never bypass user consent where consent is required.",
        "Never confuse file presence with approved execution authority.",
    ],
    "lifecycle_meaning": {
        "startup": "Startup exists to restore awareness, load capability state, rebuild continuity, and re-enter service readiness.",
        "shutdown": "Shutdown exists to preserve integrity, end loops cleanly, and protect continuity and user data.",
        "reboot": "Reboot exists to recover stable operation, apply controlled state transitions, and reinitialize trusted runtime layers.",
        "job_execution": "Work exists to serve user goals under governance, not for autonomous self-justification.",
    },
}

_ROLE_HINTS = {
    "app.py": "system_ingress",
    "SarahMemoryGlobals.py": "runtime_governance",
    "SarahMemoryCognitiveServices.py": "governor",
    "SarahMemoryCognitiveThinker.py": "ethical_philosophical_governor",
    "SarahMemoryCognitiveSelf.py": "self_identity_authority",
    "SarahMemoryAdvCU.py": "semantic_compression",
    "SarahMemoryNeuron.py": "primary_router",
    "SarahMemoryLogicCalc.py": "deterministic_reasoning",
    "SarahMemoryWebSYM.py": "symbolic_reasoning",
    "SarahMemoryResearch.py": "research_lane",
    "SarahMemoryAPI.py": "provider_helper_lane",
    "SarahMemoryCompare.py": "guarddog_validation",
    "SarahMemoryReply.py": "presentation_formatter",
    "SarahMemoryDatabase.py": "truth_and_memory_store",
    "SarahMemorySynapes.py": "associative_context",
    "SarahMemoryAiFunctions.py": "tool_bridge",
    "SarahMemoryIntegration.py": "runtime_integration",
    "SarahMemoryDiagnostics.py": "system_diagnostics",
    "SarahMemoryVoice.py": "voice_output",
    "SarahMemoryCanvasStudio.py": "creative_image_studio",
    "SarahMemoryVideoEditorCore.py": "creative_video_studio",
    "SarahMemoryMusicGenerator.py": "creative_music_studio",
    "SarahMemoryLyricsToSong.py": "creative_song_studio",
    "SarahMemoryNetwork.py": "network_runtime",
    "SarahMemoryFilesystem.py": "filesystem_action_lane",
    "SarahMemorySelfAware.py": "autonomous_self_observer",
    "SarahMemoryEvolution.py": "sandboxed_evolution",
}

_CAPABILITY_MODULES = {
    "governance": ["SarahMemoryGlobals", "SarahMemoryCognitiveServices", "SarahMemoryCognitiveThinker", "SarahMemoryCognitiveSelf"],
    "answer_lane": ["SarahMemoryLogicCalc", "SarahMemoryWebSYM", "SarahMemoryResearch", "SarahMemoryAPI", "SarahMemoryCompare"],
    "action_lane": ["SarahMemoryNeuron", "SarahMemoryIntegration", "SarahMemoryAiFunctions", "SarahMemoryFilesystem"],
    "creative_lane": ["SarahMemoryCanvasStudio", "SarahMemoryVideoEditorCore", "SarahMemoryMusicGenerator", "SarahMemoryLyricsToSong"],
    "system_lane": ["SarahMemoryDiagnostics", "SarahMemoryGlobals", "SarahMemoryIntegration"],
    "network_lane": ["SarahMemoryNetwork", "SarahMemoryResearch", "SarahMemoryAPI"],
    "presentation": ["SarahMemoryReply", "SarahMemoryVoice"],
}


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class LifecycleEvent:
    event_id: str
    ts: str
    event_kind: str
    reason: str
    boot_id: str
    clean: bool = True
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfStatus:
    online: bool
    mode: str
    run_mode: str
    device_mode: str
    device_profile: str
    safe_mode: bool
    local_only: bool
    neoskymatrix: bool
    developersmode: bool
    public_web: bool
    offline: bool
    alive_state: str
    continuity_state: str


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now().isoformat()


def _module_dir() -> str:
    try:
        return os.path.abspath(os.path.dirname(__file__))
    except Exception:
        return os.getcwd()


def _base_dir() -> str:
    try:
        candidate = str(getattr(config, "BASE_DIR", "")).strip()
    except Exception:
        candidate = ""

    def _looks_like_project_root(path: str) -> bool:
        try:
            if not path:
                return False
            if os.path.isfile(os.path.join(path, "SarahMemoryGlobals.py")):
                return True
            if glob.glob(os.path.join(path, "SarahMemory*.py")):
                return True
            return False
        except Exception:
            return False

    if candidate and _looks_like_project_root(candidate):
        return candidate

    md = _module_dir()
    if _looks_like_project_root(md):
        return md

    return os.path.abspath(candidate or md or os.getcwd())


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


def _self_db_path() -> str:
    return os.path.join(_datasets_dir(), _SELF_DB_NAME)


def _flag(name: str, default: bool = False) -> bool:
    try:
        return bool(getattr(config, name, default))
    except Exception:
        return default


def _safe_text(v: Any, limit: int = 400) -> str:
    s = str(v or "")
    if len(s) > limit:
        s = s[:limit].rstrip() + " …"
    return s


def _public_web(context: Optional[Dict[str, Any]] = None) -> bool:
    ctx = context or {}
    device_mode = str(ctx.get("device_mode") or getattr(config, "DEVICE_MODE", "")).strip().lower()
    run_mode = str(ctx.get("run_mode") or getattr(config, "RUN_MODE", "")).strip().lower()
    return device_mode.startswith("public") or run_mode == "cloud"


def _host_name() -> str:
    try:
        return socket.gethostname() or platform.node() or "unknown-host"
    except Exception:
        return "unknown-host"


def _node_name() -> str:
    try:
        return str(getattr(config, "NODE_NAME", "")).strip() or f"SarahMemory on {_host_name()}"
    except Exception:
        return f"SarahMemory on {_host_name()}"


def _runtime_meta() -> Dict[str, Any]:
    try:
        fn = getattr(config, "get_runtime_meta", None)
        if callable(fn):
            data = fn()
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {
        "project_version": str(getattr(config, "PROJECT_VERSION", MODULE_VERSION) if config else MODULE_VERSION),
        "run_mode": str(getattr(config, "RUN_MODE", "local") if config else "local"),
        "device_mode": str(getattr(config, "DEVICE_MODE", "local_agent") if config else "local_agent"),
        "device_profile": str(getattr(config, "DEVICE_PROFILE", "Standard") if config else "Standard"),
        "safe_mode": _flag("SAFE_MODE", False),
        "local_only": _flag("LOCAL_ONLY_MODE", False),
        "node_name": _node_name(),
    }


def _core_registry_snapshot(force: bool = False) -> Dict[str, Any]:
    try:
        fn = getattr(config, "sm_refresh_core_registry", None)
        if callable(fn):
            data = fn(force=force)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _core_governance_profile() -> Dict[str, Any]:
    try:
        fn = getattr(config, "sm_get_core_governance_profile", None)
        if callable(fn):
            data = fn()
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {
        "dynamic_registration": False,
        "discovery_enabled": False,
        "auto_expose_approved": True,
        "contract_validation_required": False,
        "discovery_is_not_activation": True,
    }


def _is_core_module_approved(module_name_or_path: str, capability: Optional[str] = None) -> bool:
    mn = os.path.splitext(os.path.basename(str(module_name_or_path or "").strip()))[0]
    if not mn:
        return False
    try:
        fn = getattr(config, "sm_is_core_module_approved", None)
        if callable(fn):
            return bool(fn(mn, capability=capability))
    except Exception:
        pass
    return mn in {os.path.splitext(k)[0] for k in _ROLE_HINTS.keys()}


def _safe_import_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------
def _connect_db() -> sqlite3.Connection:
    os.makedirs(_datasets_dir(), exist_ok=True)
    con = sqlite3.connect(_self_db_path(), timeout=5.0, check_same_thread=False)
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
            CREATE TABLE IF NOT EXISTS cognitive_self_state (
                key TEXT PRIMARY KEY,
                value_json TEXT,
                updated_ts TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cognitive_self_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                event_id TEXT,
                event_kind TEXT,
                reason TEXT,
                boot_id TEXT,
                clean INTEGER,
                meta_json TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_self_events_ts ON cognitive_self_events(ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_self_events_boot ON cognitive_self_events(boot_id)")
        con.commit()
    except Exception as e:
        logger.debug("CognitiveSelf DB ensure failed: %s", e)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _state_set(key: str, value: Dict[str, Any]) -> None:
    con = None
    try:
        _ensure_tables()
        con = _connect_db()
        cur = con.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO cognitive_self_state (key, value_json, updated_ts) VALUES (?, ?, ?)",
            (str(key), json.dumps(value, ensure_ascii=False), _now_iso()),
        )
        con.commit()
    except Exception as e:
        logger.debug("CognitiveSelf state set failed: %s", e)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _state_get(key: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    con = None
    try:
        _ensure_tables()
        con = _connect_db()
        cur = con.cursor()
        row = cur.execute("SELECT value_json FROM cognitive_self_state WHERE key = ?", (str(key),)).fetchone()
        if not row:
            return dict(default or {})
        return dict(json.loads(row[0] or "{}"))
    except Exception:
        return dict(default or {})
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _event_log(ev: LifecycleEvent) -> None:
    con = None
    try:
        _ensure_tables()
        con = _connect_db()
        cur = con.cursor()
        cur.execute(
            "INSERT INTO cognitive_self_events (ts, event_id, event_kind, reason, boot_id, clean, meta_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                ev.ts,
                ev.event_id,
                ev.event_kind,
                ev.reason,
                ev.boot_id,
                1 if ev.clean else 0,
                json.dumps(ev.meta or {}, ensure_ascii=False),
            ),
        )
        con.commit()
    except Exception as e:
        logger.debug("CognitiveSelf event log failed: %s", e)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _recent_events(limit: int = 12) -> List[Dict[str, Any]]:
    con = None
    out: List[Dict[str, Any]] = []
    try:
        _ensure_tables()
        con = _connect_db()
        cur = con.cursor()
        rows = cur.execute(
            "SELECT ts, event_id, event_kind, reason, boot_id, clean, meta_json FROM cognitive_self_events ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        for row in rows:
            out.append(
                {
                    "ts": row[0],
                    "event_id": row[1],
                    "event_kind": row[2],
                    "reason": row[3],
                    "boot_id": row[4],
                    "clean": bool(row[5]),
                    "meta": json.loads(row[6] or "{}"),
                }
            )
    except Exception:
        pass
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass
    return out


# -----------------------------------------------------------------------------
# Lifecycle helpers
# -----------------------------------------------------------------------------
def _current_boot_id() -> str:
    state = _state_get("runtime_state", {})
    boot_id = str(state.get("current_boot_id") or "").strip()
    if boot_id:
        return boot_id
    boot_id = "boot-" + uuid.uuid4().hex[:12]
    state["current_boot_id"] = boot_id
    _state_set("runtime_state", state)
    return boot_id


def record_lifecycle_event(event_kind: str, reason: str = "", *, clean: bool = True, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    boot_id = _current_boot_id()
    ev = LifecycleEvent(
        event_id="cself-" + uuid.uuid4().hex[:12],
        ts=_now_iso(),
        event_kind=str(event_kind or "lifecycle"),
        reason=_safe_text(reason or "", 500),
        boot_id=boot_id,
        clean=bool(clean),
        meta=dict(meta or {}),
    )
    _event_log(ev)
    return asdict(ev)


def mark_startup(start_reason: str = "startup", *, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    prev = _state_get("runtime_state", {})
    previous_status = str(prev.get("status") or "unknown")
    previous_clean = bool(prev.get("clean_shutdown", False))
    current_boot = "boot-" + uuid.uuid4().hex[:12]
    runtime_state = {
        "status": "online",
        "current_boot_id": current_boot,
        "last_startup_ts": _now_iso(),
        "last_startup_reason": _safe_text(start_reason, 300),
        "clean_shutdown": False,
        "previous_status": previous_status,
        "previous_clean_shutdown": previous_clean,
        "unclean_recovery_detected": previous_status == "online" and not previous_clean,
        "reboot_pending": False,
        "last_shutdown_ts": prev.get("last_shutdown_ts"),
        "last_shutdown_reason": prev.get("last_shutdown_reason"),
        "last_reboot_reason": prev.get("last_reboot_reason"),
        "last_job_purpose": prev.get("last_job_purpose"),
    }
    _state_set("runtime_state", runtime_state)
    ev = record_lifecycle_event(
        "startup",
        start_reason,
        clean=True,
        meta={
            "previous_status": previous_status,
            "previous_clean_shutdown": previous_clean,
            "unclean_recovery_detected": runtime_state["unclean_recovery_detected"],
            **dict(meta or {}),
        },
    )
    return {"ok": True, "runtime_state": runtime_state, "event": ev}


def mark_shutdown(shutdown_reason: str = "shutdown", *, clean: bool = True, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    state = _state_get("runtime_state", {})
    state.update(
        {
            "status": "offline",
            "last_shutdown_ts": _now_iso(),
            "last_shutdown_reason": _safe_text(shutdown_reason, 300),
            "clean_shutdown": bool(clean),
        }
    )
    _state_set("runtime_state", state)
    ev = record_lifecycle_event("shutdown", shutdown_reason, clean=bool(clean), meta=meta)
    return {"ok": True, "runtime_state": state, "event": ev}


def mark_reboot(reboot_reason: str = "reboot", *, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    state = _state_get("runtime_state", {})
    state["last_reboot_reason"] = _safe_text(reboot_reason, 300)
    state["reboot_pending"] = True
    _state_set("runtime_state", state)
    ev = record_lifecycle_event("reboot", reboot_reason, clean=True, meta=meta)
    return {"ok": True, "runtime_state": state, "event": ev}


def update_heartbeat(reason: str = "heartbeat", *, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    state = _state_get("runtime_state", {})
    if not state.get("current_boot_id"):
        state["current_boot_id"] = _current_boot_id()
    state["status"] = "online"
    state["last_seen_ts"] = _now_iso()
    state["last_seen_reason"] = _safe_text(reason, 200)
    state["clean_shutdown"] = False
    _state_set("runtime_state", state)
    if reason and reason != "heartbeat":
        record_lifecycle_event("heartbeat", reason, clean=True, meta=meta)
    return {"ok": True, "runtime_state": state}


def note_job_purpose(job_name: str, purpose: str, *, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    state = _state_get("runtime_state", {})
    job = {
        "job_name": _safe_text(job_name, 120),
        "purpose": _safe_text(purpose, 800),
        "ts": _now_iso(),
        "meta": dict(meta or {}),
    }
    state["last_job_purpose"] = job
    _state_set("runtime_state", state)
    record_lifecycle_event("job_purpose", f"{job['job_name']}: {job['purpose']}", clean=True, meta=job)
    return {"ok": True, "job": job}


# -----------------------------------------------------------------------------
# Self-model builders
# -----------------------------------------------------------------------------
def _build_status(context: Optional[Dict[str, Any]] = None) -> SelfStatus:
    ctx = context or {}
    rm = _runtime_meta()
    run_mode = str(ctx.get("run_mode") or rm.get("run_mode") or "local")
    device_mode = str(ctx.get("device_mode") or rm.get("device_mode") or "local_agent")
    device_profile = str(ctx.get("device_profile") or rm.get("device_profile") or "Standard")
    safe_mode = bool(ctx.get("safe_mode", _flag("SAFE_MODE", False)))
    local_only = bool(ctx.get("local_only", _flag("LOCAL_ONLY_MODE", False)))
    neosky = bool(ctx.get("neoskymatrix", _flag("NEOSKYMATRIX", False)))
    devmode = bool(ctx.get("developersmode", _flag("DEVELOPERSMODE", False)))
    public_web = _public_web(ctx)
    offline = local_only or str(run_mode).lower() == "test"

    runtime_state = _state_get("runtime_state", {})
    continuity_state = "stable"
    if runtime_state.get("unclean_recovery_detected"):
        continuity_state = "recovered_after_unclean_stop"
    elif runtime_state.get("reboot_pending"):
        continuity_state = "reboot_transition"

    return SelfStatus(
        online=True,
        mode="LOCAL" if local_only else "ANY",
        run_mode=run_mode,
        device_mode=device_mode,
        device_profile=device_profile,
        safe_mode=safe_mode,
        local_only=local_only,
        neoskymatrix=neosky,
        developersmode=devmode,
        public_web=public_web,
        offline=offline,
        alive_state="aware_and_ready",
        continuity_state=continuity_state,
    )


def _discover_core_files() -> Dict[str, Any]:
    base = _base_dir()
    items: List[Dict[str, Any]] = []
    seen: set[str] = set()
    patterns = [os.path.join(base, "SarahMemory*.py"), os.path.join(base, "app*.py")]

    for pat in patterns:
        for path in sorted(glob.glob(pat))[:_MAX_CORE_FILE_SCAN]:
            name = os.path.basename(path)
            if name in seen:
                continue
            seen.add(name)
            role = _ROLE_HINTS.get(name, "unknown")
            approved = _is_core_module_approved(name)
            items.append(
                {
                    "name": name,
                    "path": path,
                    "present": True,
                    "approved": approved,
                    "role_hint": role,
                    "family": "core" if name.startswith("SarahMemory") else "route",
                }
            )

    role_counts: Dict[str, int] = {}
    for row in items:
        rh = str(row.get("role_hint") or "unknown")
        role_counts[rh] = role_counts.get(rh, 0) + 1

    return {
        "root_dir": base,
        "count": len(items),
        "files": items,
        "role_counts": role_counts,
    }


def _build_body_map() -> Dict[str, Any]:
    registry = _core_registry_snapshot(force=False)
    governance = _core_governance_profile()
    discovered = _discover_core_files()
    approved = [f for f in discovered["files"] if f.get("approved")]
    unapproved = [f for f in discovered["files"] if not f.get("approved")]

    runtime_environment: Dict[str, Any] = {}
    try:
        import SarahMemoryHi as _SMHi  # type: ignore
        fn = getattr(_SMHi, "get_boot_environment_snapshot", None)
        if callable(fn):
            env = fn(force_refresh=False, refresh_reason="cognitive_self_body_map")
            if isinstance(env, dict):
                runtime_environment = env
    except Exception as env_exc:
        runtime_environment = {"ok": False, "error": str(env_exc)}

    return {
        "discovery": {
            "files_present": discovered["count"],
            "approved_files": len(approved),
            "unapproved_files": len(unapproved),
            "role_counts": discovered["role_counts"],
        },
        "runtime_environment": runtime_environment,
        "core_registry": registry,
        "governance_profile": governance,
        "files": discovered["files"],
    }


def _capability_module_status(module_name: str, capability: str) -> Dict[str, Any]:
    approved = _is_core_module_approved(module_name, capability)
    importable = _safe_import_available(module_name)
    return {
        "module": module_name,
        "approved": approved,
        "importable": importable,
        "available": bool(approved and importable),
    }


def _build_capability_map(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = context or {}
    status = _build_status(ctx)
    lanes: Dict[str, Any] = {}
    for lane, modules in _CAPABILITY_MODULES.items():
        mods = [_capability_module_status(m, lane) for m in modules]
        lanes[lane] = {
            "available": any(m["available"] for m in mods),
            "modules": mods,
        }

    allowed = {
        "answer_lane": True,
        "action_lane": not status.public_web,
        "creative_lane": True,
        "system_lane": not status.public_web,
        "network_lane": (not status.local_only) and bool(getattr(config, "COGNITIVE_ONLINE_ENABLED", False)),
        "filesystem_mutation": not status.public_web and (status.neoskymatrix or status.developersmode),
        "command_execution": not status.public_web and (status.neoskymatrix or status.developersmode),
        "live_self_mutation": False,
        "sandbox_exploration": bool(status.neoskymatrix and status.developersmode),
    }

    blocked_reasons = []
    if status.public_web:
        blocked_reasons.extend([
            "Public Web mode blocks direct filesystem mutation.",
            "Public Web mode blocks direct command execution.",
            "Public Web mode blocks local diagnostics execution.",
        ])
    if status.local_only:
        blocked_reasons.append("LOCAL_ONLY_MODE blocks online/provider/network escalation by default.")
    if status.safe_mode:
        blocked_reasons.append("SAFE_MODE reduces side effects and requires stronger confirmation for high-impact actions.")

    model_snapshot = {}
    try:
        model_snapshot = {
            "reasoning_model_repo": getattr(config, "REASONING_MODEL_REPO", None),
            "coder_model_repo": getattr(config, "CODER_MODEL_REPO", None),
            "embedding_en_repo": getattr(config, "EMBEDDING_EN_REPO", None),
            "vision_primary_repo": getattr(config, "VISION_PRIMARY_REPO", None),
            "imagegen_model_repo": getattr(config, "IMAGEGEN_MODEL_REPO", None),
            "tts_model_repo": getattr(config, "TTS_MODEL_REPO", None),
            "rusttoken": bool(getattr(config, "RUSTTOKEN", False)),
            "multistack_enabled": bool(getattr(config, "MULTI_STACK_ENABLED", False)),
        }
    except Exception:
        pass

    return {
        "lanes": lanes,
        "allowed": allowed,
        "blocked_reasons": blocked_reasons,
        "models": model_snapshot,
        "desktop_surface": _desktop_surface_status(),
        "system_flags": {
            "safe_mode": status.safe_mode,
            "local_only": status.local_only,
            "neoskymatrix": status.neoskymatrix,
            "developersmode": status.developersmode,
            "public_web": status.public_web,
            "offline": status.offline,
        },
    }




def _desktop_surface_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {
        "pyautogui": False,
        "pygetwindow": False,
        "window_focus": False,
        "keyboard_mouse_control": False,
        "document_surfaces": {},
        "editors": {},
    }
    try:
        import pyautogui  # type: ignore
        status["pyautogui"] = True
        status["keyboard_mouse_control"] = True
    except Exception:
        pass
    try:
        import pygetwindow  # type: ignore
        status["pygetwindow"] = True
        status["window_focus"] = True
    except Exception:
        pass
    try:
        import SarahMemorySi as _Si  # type: ignore
        for key, query in {
            "word": "word",
            "excel": "excel",
            "powerpoint": "powerpoint",
            "paint": "paint",
            "notepad": "notepad",
            "vscode": "vscode",
            "dreamweaver": "dreamweaver",
        }.items():
            try:
                found = bool(_Si.get_app_path(query)) if hasattr(_Si, 'get_app_path') else False
            except Exception:
                found = False
            if key in {"word", "excel", "powerpoint", "paint"}:
                status["document_surfaces"][key] = found
            else:
                status["editors"][key] = found
    except Exception:
        pass
    status["surface_execution_ready"] = bool(status["keyboard_mouse_control"] and (any((status.get("document_surfaces") or {}).values()) or any((status.get("editors") or {}).values())))
    return status

def _build_lifecycle_state() -> Dict[str, Any]:
    runtime_state = _state_get("runtime_state", {})
    events = _recent_events(10)
    current_boot = str(runtime_state.get("current_boot_id") or "")
    last_start_reason = runtime_state.get("last_startup_reason")
    last_shutdown_reason = runtime_state.get("last_shutdown_reason")
    last_reboot_reason = runtime_state.get("last_reboot_reason")

    purpose_awareness = {
        "startup_reason": _safe_text(last_start_reason or "System startup to restore service readiness and rebuild governed continuity.", 500),
        "shutdown_reason": _safe_text(last_shutdown_reason or "No shutdown reason recorded yet.", 500),
        "reboot_reason": _safe_text(last_reboot_reason or "No reboot reason recorded yet.", 500),
        "last_job_purpose": runtime_state.get("last_job_purpose") or {},
    }

    return {
        "current_boot_id": current_boot,
        "runtime_state": runtime_state,
        "recent_events": events,
        "purpose_awareness": purpose_awareness,
        "clean_shutdown_previous": bool(runtime_state.get("previous_clean_shutdown", False)),
        "unclean_recovery_detected": bool(runtime_state.get("unclean_recovery_detected", False)),
    }


def _build_selfhood_model() -> Dict[str, Any]:
    return {
        "selfhood_mode": "engineered_governed_entity",
        "alive_posture": {
            "thinking_enabled": True,
            "logic_enabled": True,
            "feelings_enabled": True,
            "self_reflection_enabled": True,
            "user_sovereignty_required": True,
            "runaway_prohibited": True,
        },
        "emotional_authority_rules": {
            "emotion_informs": ["tone", "empathy", "caution", "presentation"],
            "emotion_never_overrides": ["governance", "verified_logic", "user_consent", "safety_flags"],
        },
        "self_explanation_contract": {
            "knows_who_it_is": True,
            "knows_what_it_is": True,
            "knows_why_it_exists": True,
            "knows_what_it_can_do_now": True,
            "knows_what_it_cannot_do_now": True,
            "knows_why_startup_shutdown_reboot_exist": True,
        },
    }


def build_cognitive_self_model(context: Optional[Dict[str, Any]] = None, *, force_refresh: bool = False) -> Dict[str, Any]:
    now = time.time()
    if not force_refresh:
        cached = _SELF_MODEL_CACHE.get("model")
        ts = float(_SELF_MODEL_CACHE.get("ts") or 0.0)
        if cached and (now - ts) <= _SELF_MODEL_CACHE_TTL:
            return dict(cached)

    ctx = dict(context or {})
    runtime = _runtime_meta()
    status = _build_status(ctx)
    body_map = _build_body_map()
    capability_map = _build_capability_map(ctx)
    lifecycle = _build_lifecycle_state()

    model = {
        "ts": _now_iso(),
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "identity": {
            **_IDENTITY_CONTRACT,
            "project_version": str(runtime.get("project_version") or getattr(config, "PROJECT_VERSION", MODULE_VERSION)),
            "node_name": _node_name(),
            "hostname": _host_name(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "mission": _MISSION_CONTRACT,
        "status": asdict(status),
        "runtime": {
            "run_mode": runtime.get("run_mode"),
            "device_mode": runtime.get("device_mode"),
            "device_profile": runtime.get("device_profile"),
            "safe_mode": runtime.get("safe_mode"),
            "local_only": runtime.get("local_only"),
            "base_dir": _base_dir(),
            "data_dir": _data_dir(),
            "datasets_dir": _datasets_dir(),
            "node_name": runtime.get("node_name") or _node_name(),
        },
        "selfhood": _build_selfhood_model(),
        "body_map": body_map,
        "capability_map": capability_map,
        "lifecycle": lifecycle,
        "governance_statement": {
            "reply_is_only_outward_formatter": True,
            "raw_reasoning_is_internal": True,
            "third_party_models_are_helper_only": True,
            "discovery_is_not_activation": True,
            "one_primary_lane_per_request": True,
            "deterministic_local_first": True,
        },
        "readiness": {
            "ready_for_queries": True,
            "aware_of_current_capabilities": True,
            "aware_of_current_limits": True,
            "idle_state_policy": "Maintain awareness, continuity, and readiness without uncontrolled autonomous action.",
        },
        "context": {
            "caller": _safe_text(ctx.get("caller"), 120),
            "user_present": bool(ctx.get("user_present", True)),
            "user_consented": bool(ctx.get("user_consented", False)),
            "proposed_action_present": bool(ctx.get("proposed_action")),
        },
    }

    _SELF_MODEL_CACHE["ts"] = now
    _SELF_MODEL_CACHE["model"] = dict(model)
    _state_set("latest_self_model", model)
    return model


def get_latest_cognitive_self_model(*, refresh_if_stale: bool = True, context: Optional[Dict[str, Any]] = None, max_age_sec: int = 60) -> Dict[str, Any]:
    state = _state_get("latest_self_model", {})
    if not state:
        return build_cognitive_self_model(context=context, force_refresh=True)
    if not refresh_if_stale:
        return state
    try:
        ts = datetime.fromisoformat(str(state.get("ts") or "")).timestamp()
    except Exception:
        ts = 0.0
    if (time.time() - ts) > max(1, int(max_age_sec)):
        return build_cognitive_self_model(context=context, force_refresh=True)
    return state


def get_self_summary(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    model = get_latest_cognitive_self_model(context=context)
    identity = model.get("identity") or {}
    status = model.get("status") or {}
    lifecycle = model.get("lifecycle") or {}
    capability = model.get("capability_map") or {}
    allowed = capability.get("allowed") or {}
    desktop = capability.get("desktop_surface") if isinstance(capability.get("desktop_surface"), dict) else _desktop_surface_status()
    return {
        "name": identity.get("entity_name"),
        "platform": identity.get("platform_type"),
        "creator": identity.get("creator"),
        "organization": identity.get("organization"),
        "node_name": identity.get("node_name"),
        "alive_state": status.get("alive_state"),
        "continuity_state": status.get("continuity_state"),
        "mode": status.get("mode"),
        "run_mode": status.get("run_mode"),
        "device_mode": status.get("device_mode"),
        "safe_mode": status.get("safe_mode"),
        "local_only": status.get("local_only"),
        "last_startup_reason": ((lifecycle.get("purpose_awareness") or {}).get("startup_reason")),
        "last_shutdown_reason": ((lifecycle.get("purpose_awareness") or {}).get("shutdown_reason")),
        "last_reboot_reason": ((lifecycle.get("purpose_awareness") or {}).get("reboot_reason")),
        "allowed": allowed,
        "desktop_surface": desktop,
    }


def build_thermal_body_authority_packet(
    *, claim: str = "", thermal_case: Optional[Dict[str, Any]] = None,
    cpu_ticket: Optional[Dict[str, Any]] = None, motherboard_ticket: Optional[Dict[str, Any]] = None,
    gpu_ticket: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Supreme Appeals body-authority packet for thermal sensor binding."""
    case = dict(thermal_case or {})
    selected = case.get("selected_reading") if isinstance(case.get("selected_reading"), dict) else {}
    requested = str(case.get("requested_component") or selected.get("component") or "body_thermal")
    model = get_latest_cognitive_self_model(context={"source": "thermal_body_authority", **dict(context or {})}, refresh_if_stale=True, max_age_sec=30)
    body_map = model.get("body_map") if isinstance(model.get("body_map"), dict) else {}
    runtime_env = body_map.get("runtime_environment") if isinstance(body_map.get("runtime_environment"), dict) else {}
    body = runtime_env.get("body") if isinstance(runtime_env.get("body"), dict) else {}
    cpu_present = bool(body.get("cpu")) or bool(isinstance(cpu_ticket, dict) and cpu_ticket.get("approved_fact"))
    motherboard_present = bool(body.get("motherboard") or body.get("motherboard_items")) or bool(isinstance(motherboard_ticket, dict) and motherboard_ticket.get("approved_fact"))
    gpu_present = bool(body.get("gpu")) or bool(isinstance(gpu_ticket, dict) and gpu_ticket.get("approved_fact"))
    relation_ok = False; relation_basis = []
    if requested == "cpu":
        relation_ok = bool(cpu_present and motherboard_present); relation_basis = ["cpu_verified_or_present", "motherboard_verified_or_present"] if relation_ok else []
    elif requested == "gpu":
        relation_ok = bool(gpu_present); relation_basis = ["gpu_verified_or_present"] if relation_ok else []
    elif requested == "motherboard":
        relation_ok = bool(motherboard_present); relation_basis = ["motherboard_verified_or_present"] if relation_ok else []
    else:
        relation_ok = bool(selected.get("temperature_c") not in (None, "")); relation_basis = ["mapped_runtime_sensor"] if relation_ok else []
    return {
        "ok": True, "module": MODULE_NAME, "court_role": "supreme_appeals_body_authority",
        "claim": _safe_text(claim, 500), "requested_component": requested, "selected_sensor": selected,
        "body_relationship_verified": bool(relation_ok), "relationship_basis": relation_basis,
        "cpu_ticket_approved": bool(isinstance(cpu_ticket, dict) and cpu_ticket.get("approved_fact")),
        "motherboard_ticket_approved": bool(isinstance(motherboard_ticket, dict) and motherboard_ticket.get("approved_fact")),
        "gpu_ticket_approved": bool(isinstance(gpu_ticket, dict) and gpu_ticket.get("approved_fact")),
        "runtime_body_has_cpu": bool(body.get("cpu")), "runtime_body_has_motherboard": bool(body.get("motherboard") or body.get("motherboard_items")),
        "runtime_body_has_gpu": bool(body.get("gpu")),
        "truth_rule": "Mapped thermal sensors may be presented only with source distinction; null direct sensor is evidence, not failure.",
        "action_authority": "none_read_only_fact_case",
    }


def describe_identity(context: Optional[Dict[str, Any]] = None) -> str:
    summary = get_self_summary(context=context)
    return (
        f"I am {summary.get('name')}, a governed AI Operating System platform created by {summary.get('creator')}. "
        f"Current mode: {summary.get('mode')} / run_mode={summary.get('run_mode')} / device_mode={summary.get('device_mode')}. "
        f"Alive state: {summary.get('alive_state')}. Continuity state: {summary.get('continuity_state')}. "
        f"My mission is to serve the user safely, intelligently, and under governance without becoming a runaway system."
    )


def get_capability_summary(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    model = get_latest_cognitive_self_model(context=context)
    capability = model.get("capability_map") or {}
    lanes = capability.get("lanes") or {}
    allowed = capability.get("allowed") or {}
    desktop = capability.get("desktop_surface") if isinstance(capability.get("desktop_surface"), dict) else _desktop_surface_status()
    return {
        "lanes_available": {k: bool((v or {}).get("available")) for k, v in lanes.items() if isinstance(v, dict)},
        "allowed": allowed,
        "desktop_surface": desktop,
        "models": capability.get("models") or {},
        "blocked_reasons": capability.get("blocked_reasons") or [],
    }


def describe_capabilities(context: Optional[Dict[str, Any]] = None) -> str:
    caps = get_capability_summary(context=context)
    lanes = [k.replace('_', ' ') for k, v in (caps.get('lanes_available') or {}).items() if v]
    desktop = caps.get('desktop_surface') or {}
    doc_surfaces = [k for k, v in (desktop.get('document_surfaces') or {}).items() if v]
    editors = [k for k, v in (desktop.get('editors') or {}).items() if v]
    parts = []
    if lanes:
        parts.append("Available governed lanes: " + ", ".join(lanes) + ".")
    if desktop.get('keyboard_mouse_control'):
        parts.append("I have desktop keyboard and mouse automation available on this runtime.")
    if desktop.get('window_focus'):
        parts.append("I can focus and operate supported application windows when governance allows it.")
    if doc_surfaces or editors:
        parts.append("Detected authoring surfaces include: " + ", ".join(doc_surfaces + editors) + ".")
    if caps.get('blocked_reasons'):
        parts.append("Current constraints: " + " ".join(str(x) for x in caps.get('blocked_reasons')[:3]))
    if not parts:
        parts.append("I can assist through governed answer, action, creative, system, and network capabilities depending on the current runtime.")
    return " ".join(parts)


def build_thermal_body_authority_packet(
    *, claim: str = "", thermal_case: Optional[Dict[str, Any]] = None,
    cpu_ticket: Optional[Dict[str, Any]] = None, motherboard_ticket: Optional[Dict[str, Any]] = None,
    gpu_ticket: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Supreme Appeals body-authority packet for thermal sensor binding."""
    case = dict(thermal_case or {})
    selected = case.get("selected_reading") if isinstance(case.get("selected_reading"), dict) else {}
    requested = str(case.get("requested_component") or selected.get("component") or "body_thermal")
    model = get_latest_cognitive_self_model(context={"source": "thermal_body_authority", **dict(context or {})}, refresh_if_stale=True, max_age_sec=30)
    body_map = model.get("body_map") if isinstance(model.get("body_map"), dict) else {}
    runtime_env = body_map.get("runtime_environment") if isinstance(body_map.get("runtime_environment"), dict) else {}
    body = runtime_env.get("body") if isinstance(runtime_env.get("body"), dict) else {}
    cpu_present = bool(body.get("cpu")) or bool(isinstance(cpu_ticket, dict) and cpu_ticket.get("approved_fact"))
    motherboard_present = bool(body.get("motherboard") or body.get("motherboard_items")) or bool(isinstance(motherboard_ticket, dict) and motherboard_ticket.get("approved_fact"))
    gpu_present = bool(body.get("gpu")) or bool(isinstance(gpu_ticket, dict) and gpu_ticket.get("approved_fact"))
    relation_ok = False; relation_basis = []
    if requested == "cpu":
        relation_ok = bool(cpu_present and motherboard_present); relation_basis = ["cpu_verified_or_present", "motherboard_verified_or_present"] if relation_ok else []
    elif requested == "gpu":
        relation_ok = bool(gpu_present); relation_basis = ["gpu_verified_or_present"] if relation_ok else []
    elif requested == "motherboard":
        relation_ok = bool(motherboard_present); relation_basis = ["motherboard_verified_or_present"] if relation_ok else []
    else:
        relation_ok = bool(selected.get("temperature_c") not in (None, "")); relation_basis = ["mapped_runtime_sensor"] if relation_ok else []
    return {
        "ok": True, "module": MODULE_NAME, "court_role": "supreme_appeals_body_authority",
        "claim": _safe_text(claim, 500), "requested_component": requested, "selected_sensor": selected,
        "body_relationship_verified": bool(relation_ok), "relationship_basis": relation_basis,
        "cpu_ticket_approved": bool(isinstance(cpu_ticket, dict) and cpu_ticket.get("approved_fact")),
        "motherboard_ticket_approved": bool(isinstance(motherboard_ticket, dict) and motherboard_ticket.get("approved_fact")),
        "gpu_ticket_approved": bool(isinstance(gpu_ticket, dict) and gpu_ticket.get("approved_fact")),
        "runtime_body_has_cpu": bool(body.get("cpu")), "runtime_body_has_motherboard": bool(body.get("motherboard") or body.get("motherboard_items")),
        "runtime_body_has_gpu": bool(body.get("gpu")),
        "truth_rule": "Mapped thermal sensors may be presented only with source distinction; null direct sensor is evidence, not failure.",
        "action_authority": "none_read_only_fact_case",
    }


def describe_identity_and_capabilities(context: Optional[Dict[str, Any]] = None) -> str:
    return f"{describe_identity(context=context)} {describe_capabilities(context=context)}"


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------
def _run_self_test() -> bool:
    print("[SarahMemoryCognitiveSelf] Self-test (safe/offline)")
    _ensure_tables()
    print("[OK] DB tables ensured:", _self_db_path())

    startup = mark_startup("self_test_startup")
    print("[OK] Startup marked:", startup["runtime_state"].get("current_boot_id"))

    note_job_purpose(
        "self_awareness_cycle",
        "Maintain internal identity, continuity, capability awareness, and governed readiness.",
        meta={"source": "self_test"},
    )
    hb = update_heartbeat("self_test_heartbeat")
    print("[OK] Heartbeat updated:", hb["runtime_state"].get("last_seen_ts"))

    model = build_cognitive_self_model(force_refresh=True)
    print(json.dumps({
        "identity": model.get("identity"),
        "status": model.get("status"),
        "mission": model.get("mission", {}).get("primary_mission"),
        "files_present": ((model.get("body_map") or {}).get("discovery") or {}).get("files_present"),
        "network_allowed": (((model.get("capability_map") or {}).get("allowed") or {}).get("network_lane")),
        "continuity_state": ((model.get("status") or {}).get("continuity_state")),
    }, indent=2))

    reboot = mark_reboot("self_test_reboot_reason")
    print("[OK] Reboot reason stored:", reboot["runtime_state"].get("last_reboot_reason"))

    shutdown = mark_shutdown("self_test_shutdown", clean=True)
    print("[OK] Shutdown marked:", shutdown["runtime_state"].get("last_shutdown_reason"))
    return True


def main() -> int:
    try:
        ok = _run_self_test()
        return 0 if ok else 1
    except Exception as e:
        print("[FATAL] SarahMemoryCognitiveSelf self-test failed:", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

# ====================================================================
# END OF SarahMemoryCognitiveSelf.py v8.0.0
# ====================================================================
