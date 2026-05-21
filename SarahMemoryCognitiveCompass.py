"""--== SarahMemory Project ==--
File: SarahMemoryCognitiveCompass.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2026-03-29
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
- Orientation watchdog, anti-drift guide, and task-completion compass for SarahMemory AiOS.
- Keeps SarahMemoryNeuron.py pointed at the ORIGINAL user goal until task completion is
  verified, or until the system safely aborts / clarifies / escapes a loop.
- Works with the governed triforce:
    * SarahMemoryCognitiveSelf.py      -> what exists / what is possible right now
    * SarahMemoryCognitiveServices.py  -> what may proceed right now
    * SarahMemoryCognitiveThinker.py   -> alternate safer / more meaningful paths
- Uses SarahMemoryPreTokenAnalyzer.py whenever possible to preserve semantic clarity.
- Selects sidekick FAMILIES and candidate helper modules without becoming the executor.
- Holds outward success / Reply release until completion proof and Compare acceptance exist.

DESIGN RULES:
- Compass does NOT replace CognitiveSelf, CognitiveServices, CognitiveThinker, Neuron, Compare, or Reply.
- Compass does NOT directly execute high-impact runtime mutations.
- Compass must detect drift, procedural gaps, premature completion, mirror-state logic, and loop risk.
- Compass must provide escape hatches so Neuron cannot get stuck in a constant loop.
- Compass is universal: it must work for Answer / Action / Creative / System / Network tasks,
  desktop software, drivers, browser workflows, creative studios, and future sidekicks.
- Discovery is not activation. Module presence alone is not approval.
- Local-first, fail-soft, auditable, and restart-safe.
===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "orientation_watchdog"
# CATEGORY = "cognition"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "system"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "cognitive_compass"
# FAMILY = "core_cognition"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Anti-drift, anti-loop, task-completion watchdog that keeps Neuron on bearing until Compare-validated completion or safe abort."
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
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Safe imports (never hard-fail the platform)
# -----------------------------------------------------------------------------
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None

try:
    import SarahMemoryCognitiveSelf as _CogSelf  # type: ignore
except Exception:
    _CogSelf = None

try:
    import SarahMemoryCognitiveServices as _CogServices  # type: ignore
except Exception:
    _CogServices = None

try:
    import SarahMemoryOperatorCore as _OperatorCore  # type: ignore
except Exception:
    _OperatorCore = None

try:
    import SarahMemoryCognitiveThinker as _CogThinker  # type: ignore
except Exception:
    _CogThinker = None

try:
    import SarahMemoryPreTokenAnalyzer as _PreToken  # type: ignore
except Exception:
    _PreToken = None

try:
    import SarahMemoryCompare as _Compare  # type: ignore
except Exception:
    _Compare = None

try:
    import SarahMemoryNeuron as _Neuron  # type: ignore
except Exception:
    _Neuron = None

try:
    import SarahMemorySi as _Si  # type: ignore
except Exception:
    _Si = None

try:
    import SarahMemoryAiFunctions as _AiFunctions  # type: ignore
except Exception:
    _AiFunctions = None

try:
    import SarahMemoryResearch as _Research  # type: ignore
except Exception:
    _Research = None

try:
    import SarahMemorySoftwareResearch as _SoftwareResearch  # type: ignore
except Exception:
    _SoftwareResearch = None

try:
    import SarahMemorySynapes as _Synapes  # type: ignore
except Exception:
    _Synapes = None

try:
    import SarahMemoryCanvasStudio as _CanvasStudio  # type: ignore
except Exception:
    _CanvasStudio = None

try:
    import SarahMemoryDiagnostics as _Diagnostics  # type: ignore
except Exception:
    _Diagnostics = None


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("SarahMemoryCognitiveCompass")
logger.setLevel(logging.DEBUG if bool(getattr(config, "DEBUG_MODE", True)) else logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.propagate = False


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
MODULE_NAME = "SarahMemoryCognitiveCompass"
MODULE_VERSION = "8.0.0"
_DB_NAME = "cognitive_compass.db"
_CACHE_TTL = 10.0

STATUS_ON_COURSE = "ON_COURSE"
STATUS_MINOR_DRIFT = "MINOR_DRIFT"
STATUS_PROCEDURAL_GAP = "PROCEDURAL_GAP"
STATUS_WAITING_ON_SIDEKICK = "WAITING_ON_SIDEKICK"
STATUS_UNVERIFIED_COMPLETION = "UNVERIFIED_COMPLETION"
STATUS_NO_PROGRESS = "NO_PROGRESS"
STATUS_LOOP_RISK = "LOOP_RISK"
STATUS_RED_ZONE = "RED_ZONE"
STATUS_ESCAPE_HATCH = "ESCAPE_HATCH_TRIGGERED"
STATUS_SAFE_ABORT = "SAFE_ABORT"
STATUS_COMPARE_PENDING = "COMPARE_PENDING"
STATUS_COMPLETE_VERIFIED = "COMPLETE_VERIFIED"

DIRECTIVE_REANCHOR_ORIGINAL = "REANCHOR_TO_ORIGINAL_GOAL"
DIRECTIVE_REANCHOR_PRETOKEN = "REANCHOR_TO_PRETOKEN_PACKET"
DIRECTIVE_REANCHOR_GOVERNOR = "REANCHOR_TO_GOVERNOR_POLICY"
DIRECTIVE_QUERY_SELF = "QUERY_COGNITIVE_SELF"
DIRECTIVE_QUERY_THINKER = "QUERY_COGNITIVE_THINKER"
DIRECTIVE_SELECT_FAMILY = "SELECT_SIDEKICK_FAMILY"
DIRECTIVE_SELECT_MODULE = "SELECT_SIDEKICK_MODULE"
DIRECTIVE_PREPARE_SURFACE = "PREPARE_SURFACE"
DIRECTIVE_FOCUS_SURFACE = "FOCUS_SURFACE"
DIRECTIVE_RESEARCH_PROCEDURE = "RESEARCH_PROCEDURE"
DIRECTIVE_EXECUTE_SUBSTEP = "EXECUTE_SUBSTEP"
DIRECTIVE_VERIFY_SUBSTEP = "VERIFY_SUBSTEP"
DIRECTIVE_VERIFY_FINAL = "VERIFY_FINAL_RESULT"
DIRECTIVE_HOLD_REPLY = "HOLD_REPLY"
DIRECTIVE_ALLOW_REPLY = "ALLOW_REPLY"
DIRECTIVE_ABORT = "ABORT_AND_REPORT"

ESCAPE_CLARIFY = "ASK_USER_FOR_CLARIFICATION"
ESCAPE_SWITCH_FAMILY = "SWITCH_SIDEKICK_FAMILY"
ESCAPE_DEGRADE_LOCAL = "DOWNGRADE_TO_LOCAL_ONLY"
ESCAPE_SAFER_PLAN = "DOWNGRADE_TO_SAFER_PLAN"
ESCAPE_RESEARCH_FIRST = "ROUTE_TO_RESEARCH_FIRST"
ESCAPE_SANDBOX_ONLY = "ROUTE_TO_SANDBOX_ONLY"
ESCAPE_SAFE_REPLY = "ABORT_TO_SAFE_REPLY"
ESCAPE_STOP_RED_ZONE = "STOP_AND_LOG_RED_ZONE"

TASK_DEFAULT_MAX_ATTEMPTS_PER_STEP = 3
TASK_DEFAULT_MAX_TOTAL_CYCLES = 12
TASK_DEFAULT_MAX_SAME_STATE_REPEATS = 2
TASK_DEFAULT_MAX_SAME_HELPER_REPEATS = 2
TASK_DEFAULT_MAX_TOTAL_SWITCHES = 6
TASK_DEFAULT_MAX_TOTAL_REANCHORS = 6

_SIDEKICK_FAMILY_REGISTRY = {
    "software_control": ["SarahMemorySi"],
    "desktop_automation": ["SarahMemoryAiFunctions"],
    "research": ["SarahMemoryResearch"],
    "software_research": ["SarahMemorySoftwareResearch", "SarahMemoryResearch"],
    "semantic_helper": ["SarahMemoryLogicCalc", "SarahMemoryWebSYM", "SarahMemoryResearch", "SarahMemoryAPI"],
    "associative_memory": ["SarahMemorySynapes"],
    "tool_bridge": ["SarahMemoryAiFunctions"],
    "creative_studio": ["SarahMemoryCanvasStudio", "SarahMemoryMusicGenerator", "SarahMemoryLyricsToSong", "SarahMemoryVideoEditorCore"],
    "driver_control": ["appdrivers", "appsys", "SarahMemoryIntegration"],
    "diagnostics": ["SarahMemoryDiagnostics"],
    "network": ["SarahMemoryNetwork", "appnet", "appnet2"],
    "filesystem": ["SarahMemoryFilesystem", "appsys"],
    "browser_surface": ["SarahMemoryBrowser", "SarahMemoryAiFunctions"],
    "document_surface": ["SarahMemorySi", "SarahMemoryAiFunctions"],
    "spreadsheet_surface": ["SarahMemorySi", "SarahMemoryAiFunctions"],
    "presentation_surface": ["SarahMemorySi", "SarahMemoryAiFunctions"],
}

_TASK_FAMILY_HINTS = {
    "action": "generic_action_workflow",
    "creative": "creative_workflow",
    "answer": "answer_workflow",
    "system": "system_workflow",
    "network": "network_workflow",
}


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class LoopGuardState:
    attempt_count: int = 0
    total_cycles: int = 0
    same_state_repeats: int = 0
    same_helper_repeats: int = 0
    helper_switches: int = 0
    reanchors: int = 0
    max_attempts_per_step: int = TASK_DEFAULT_MAX_ATTEMPTS_PER_STEP
    max_total_cycles: int = TASK_DEFAULT_MAX_TOTAL_CYCLES
    max_same_state_repeats: int = TASK_DEFAULT_MAX_SAME_STATE_REPEATS
    max_same_helper_repeats: int = TASK_DEFAULT_MAX_SAME_HELPER_REPEATS
    max_total_switches: int = TASK_DEFAULT_MAX_TOTAL_SWITCHES
    max_total_reanchors: int = TASK_DEFAULT_MAX_TOTAL_REANCHORS


@dataclass
class TaskProgress:
    surface_open: bool = False
    surface_focused: bool = False
    working_context_ready: bool = False
    content_generated: bool = False
    content_applied: bool = False
    result_verified: bool = False
    compare_passed: bool = False


@dataclass
class CompassState:
    goal_id: str
    original_user_goal: str
    primary_lane: str
    task_family: str
    target_surface: str
    desired_outcome: str
    status: str
    reply_allowed: bool
    continue_allowed: bool
    reanchor_target: str
    next_required_step: str
    reason: str
    selected_sidekick_family: Optional[str] = None
    selected_sidekick_module: Optional[str] = None
    candidate_sidekick_families: List[str] = field(default_factory=list)
    candidate_sidekick_modules: List[str] = field(default_factory=list)
    task_progress: Dict[str, Any] = field(default_factory=dict)
    loop_guard: Dict[str, Any] = field(default_factory=dict)
    escape_hatch: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


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
    if candidate:
        return os.path.abspath(candidate)
    return _module_dir()


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


def _db_path() -> str:
    return os.path.join(_datasets_dir(), _DB_NAME)


def _safe_text(value: Any, limit: int = 500) -> str:
    s = str(value or "")
    if len(s) > limit:
        s = s[:limit].rstrip() + " …"
    return s


def _flag(name: str, default: bool = False) -> bool:
    try:
        return bool(getattr(config, name, default))
    except Exception:
        return default


def _public_web(context: Optional[Dict[str, Any]] = None) -> bool:
    ctx = dict(context or {})
    device_mode = str(ctx.get("device_mode") or getattr(config, "DEVICE_MODE", "")).strip().lower()
    run_mode = str(ctx.get("run_mode") or getattr(config, "RUN_MODE", "")).strip().lower()
    return device_mode.startswith("public") or run_mode == "cloud"


def _jsonable(data: Any) -> Any:
    try:
        json.dumps(data, ensure_ascii=False)
        return data
    except Exception:
        return _safe_text(data, 200)


def _module_approved(module_name: str, capability: Optional[str] = None) -> bool:
    module_name = os.path.splitext(os.path.basename(str(module_name or "").strip()))[0]
    if not module_name:
        return False
    try:
        fn = getattr(config, "sm_is_core_module_approved", None)
        if callable(fn):
            return bool(fn(module_name, capability=capability))
    except Exception:
        pass
    return True


def _module_importable(module_name: str) -> bool:
    if module_name in sys.modules:
        return True
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _deterministic_hash(text: str) -> str:
    return uuid.uuid5(uuid.NAMESPACE_DNS, str(text or "")).hex[:12]


# -----------------------------------------------------------------------------
# DB helpers
# -----------------------------------------------------------------------------
def _connect_db() -> sqlite3.Connection:
    os.makedirs(_datasets_dir(), exist_ok=True)
    con = sqlite3.connect(_db_path(), timeout=5.0, check_same_thread=False)
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
            CREATE TABLE IF NOT EXISTS compass_state (
                goal_id TEXT PRIMARY KEY,
                state_json TEXT,
                updated_ts TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS compass_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                event_kind TEXT,
                goal_id TEXT,
                severity TEXT,
                details TEXT,
                meta_json TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_compass_events_goal ON compass_events(goal_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_compass_events_ts ON compass_events(ts)")
        con.commit()
    except Exception as e:
        logger.debug("Compass DB ensure failed: %s", e)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _state_set(goal_id: str, state: Dict[str, Any]) -> None:
    con = None
    try:
        _ensure_tables()
        con = _connect_db()
        con.execute(
            "INSERT OR REPLACE INTO compass_state (goal_id, state_json, updated_ts) VALUES (?, ?, ?)",
            (str(goal_id), json.dumps(state, ensure_ascii=False), _now_iso()),
        )
        con.commit()
    except Exception as e:
        logger.debug("Compass state set failed: %s", e)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _state_get(goal_id: str) -> Dict[str, Any]:
    con = None
    try:
        _ensure_tables()
        con = _connect_db()
        row = con.execute("SELECT state_json FROM compass_state WHERE goal_id = ?", (str(goal_id),)).fetchone()
        if not row:
            return {}
        return dict(json.loads(row[0] or "{}"))
    except Exception:
        return {}
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _log_event(event_kind: str, goal_id: str, details: str, severity: str = "INFO", meta: Optional[Dict[str, Any]] = None) -> None:
    con = None
    try:
        _ensure_tables()
        con = _connect_db()
        con.execute(
            "INSERT INTO compass_events (ts, event_kind, goal_id, severity, details, meta_json) VALUES (?, ?, ?, ?, ?, ?)",
            (_now_iso(), str(event_kind), str(goal_id), str(severity), str(details), json.dumps(meta or {}, ensure_ascii=False)),
        )
        con.commit()
    except Exception as e:
        logger.debug("Compass event log failed: %s", e)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Cross-module queries
# -----------------------------------------------------------------------------
def _query_cognitive_self(request_text: str = "", context: Optional[Dict[str, Any]] = None, *, force_refresh: bool = False) -> Dict[str, Any]:
    if not _CogSelf:
        return {}
    ctx = dict(context or {})
    if request_text and not ctx.get("request_text") and not ctx.get("text"):
        ctx["request_text"] = request_text
    for fn_name in ("get_latest_cognitive_self_model", "build_cognitive_self_model"):
        try:
            fn = getattr(_CogSelf, fn_name, None)
            if callable(fn):
                pkt = fn(context=ctx, force_refresh=force_refresh) if fn_name == "build_cognitive_self_model" else fn(context=ctx)
                if isinstance(pkt, dict):
                    return pkt
        except Exception:
            continue
    return {}


def _query_governor(request_text: str, context: Optional[Dict[str, Any]] = None, proposed_action: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not _CogServices:
        return {}
    ctx = dict(context or {})
    try:
        fn = getattr(_CogServices, "govern_request", None)
        if callable(fn):
            pkt = fn(
                request_text,
                caller=MODULE_NAME,
                caller_context=ctx,
                user_present=bool(ctx.get("user_present", True)),
                user_consented=bool(ctx.get("user_consented", False)),
                proposed_action=proposed_action or {},
            )
            return pkt if isinstance(pkt, dict) else {}
    except Exception as e:
        logger.debug("Compass governor query failed: %s", e)
    return {}


def _query_thinker(request_text: str, context: Optional[Dict[str, Any]] = None, proposed_action: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not _CogThinker:
        return {}
    ctx = dict(context or {})
    try:
        fn = getattr(_CogThinker, "paired_governance_view", None)
        if callable(fn):
            pkt = fn(
                request_text,
                caller=MODULE_NAME,
                caller_context=ctx,
                user_present=bool(ctx.get("user_present", True)),
                user_consented=bool(ctx.get("user_consented", False)),
                proposed_action=proposed_action or {},
            )
            return pkt if isinstance(pkt, dict) else {}
    except Exception as e:
        logger.debug("Compass thinker query failed: %s", e)
    return {}


def _query_pretoken(request_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not _PreToken:
        return {}
    ctx = dict(context or {})
    candidates = [
        "analyze_pre_token_text",
        "analyze_request",
        "process_text",
        "analyze_text",
        "pretoken_analyze",
    ]
    for name in candidates:
        try:
            fn = getattr(_PreToken, name, None)
            if callable(fn):
                result = fn(request_text, context_packet=ctx) if "context_packet" in getattr(fn, "__code__", type("x", (), {"co_varnames": ()})).co_varnames else fn(request_text, ctx)
                if isinstance(result, dict):
                    return result
        except TypeError:
            try:
                result = fn(request_text)
                if isinstance(result, dict):
                    return result
            except Exception:
                continue
        except Exception:
            continue
    return {}


def _compare_accepts(request_text: str, candidate_text: str, intent: str = "general") -> Dict[str, Any]:
    if not _Compare:
        return {"ok": False, "status": "UNAVAILABLE"}
    try:
        fn = getattr(_Compare, "compare_reply", None)
        if callable(fn):
            out = fn(request_text, candidate_text, intent=intent)
            return out if isinstance(out, dict) else {"ok": False, "status": "INVALID"}
    except Exception as e:
        logger.debug("Compass compare query failed: %s", e)
    return {"ok": False, "status": "ERROR"}


# -----------------------------------------------------------------------------
# Sidekick registry
# -----------------------------------------------------------------------------
def get_sidekick_registry(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = dict(context or {})
    public_web = _public_web(ctx)
    registry: Dict[str, Any] = {}
    for family, modules in _SIDEKICK_FAMILY_REGISTRY.items():
        candidates = []
        for module_name in modules:
            approved = _module_approved(module_name, capability=family)
            importable = _module_importable(module_name)
            candidates.append(
                {
                    "module": module_name,
                    "approved": approved,
                    "importable": importable,
                    "available": bool(approved and importable),
                }
            )
        registry[family] = {
            "available": any(x["available"] for x in candidates),
            "public_web_safe": family in {"research", "software_research", "semantic_helper", "creative_studio"} and not public_web or family == "research",
            "modules": candidates,
        }
    return registry


def select_sidekick_family(
    request_text: str,
    missing_need: str,
    *,
    primary_lane: str = "action",
    context: Optional[Dict[str, Any]] = None,
    self_model: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    registry = get_sidekick_registry(context=context)
    need = str(missing_need or "").strip().lower()
    lane = str(primary_lane or "action").strip().lower()
    model = self_model or {}
    desktop = (((model.get("capability_map") or {}).get("desktop_surface")) if isinstance(model.get("capability_map"), dict) else {}) or {}

    families: List[str] = []
    if need in {"launch_app", "focus_window", "verify_window", "window_control"}:
        families = ["software_control", "desktop_automation"]
    elif need in {"prepare_surface", "create_blank_document", "open_named_document", "spreadsheet_scaffold", "presentation_scaffold"}:
        families = ["document_surface", "spreadsheet_surface", "presentation_surface", "software_control", "desktop_automation"]
    elif need in {"software_procedure", "research_procedure", "unknown_ui_workflow"}:
        families = ["software_research", "research", "associative_memory"]
    elif need in {"generate_content", "draft_text", "answer_subproblem"}:
        families = ["semantic_helper", "research", "associative_memory"]
    elif need in {"creative_artifact", "image_edit", "music_task", "video_task"}:
        families = ["creative_studio", "semantic_helper"]
    elif need in {"driver_action", "hardware_control"}:
        families = ["driver_control", "diagnostics"]
    elif need in {"browser_task", "web_scaffold"}:
        families = ["browser_surface", "desktop_automation", "research"]
    elif need in {"filesystem_task"}:
        families = ["filesystem", "tool_bridge"]
    elif need in {"verify_result", "verify_completion"}:
        families = ["tool_bridge", "software_control", "desktop_automation"]
    else:
        families = [
            "software_control",
            "desktop_automation",
            "research",
            "software_research",
            "semantic_helper",
            "creative_studio",
            "associative_memory",
            "tool_bridge",
        ]

    if lane == "creative" and "creative_studio" not in families:
        families = ["creative_studio"] + families
    if lane == "answer" and "semantic_helper" not in families:
        families = ["semantic_helper"] + families

    if desktop.get("surface_execution_ready") is False:
        families = [f for f in families if f not in {"document_surface", "spreadsheet_surface", "presentation_surface", "desktop_automation"}] + ["software_research"]

    ordered = []
    for family in families:
        if family not in ordered:
            ordered.append(family)

    selected_family = None
    selected_module = None
    candidate_modules: List[str] = []
    for family in ordered:
        fam = registry.get(family) or {}
        modules = fam.get("modules") or []
        for item in modules:
            if item.get("available"):
                candidate_modules.append(str(item.get("module")))
        if selected_family is None and fam.get("available"):
            selected_family = family
            for item in modules:
                if item.get("available"):
                    selected_module = str(item.get("module"))
                    break

    return {
        "candidate_families": ordered,
        "candidate_modules": candidate_modules,
        "selected_family": selected_family,
        "selected_module": selected_module,
        "registry": registry,
    }


# -----------------------------------------------------------------------------
# Task contracts / progress inference
# -----------------------------------------------------------------------------
def _pretoken_task_shape(pretoken: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(pretoken, dict):
        return {}
    out: Dict[str, Any] = {}
    # Try several common field names to stay resilient.
    out["likely_lane"] = pretoken.get("likely_lane") or pretoken.get("lane") or pretoken.get("primary_lane")
    out["subject"] = pretoken.get("subject")
    out["query_type"] = pretoken.get("query_type")
    out["output_type"] = pretoken.get("output_type")
    out["semantic_summary"] = pretoken.get("semantic_summary")
    out["clarification_required"] = bool(
        pretoken.get("clarification_required")
        or pretoken.get("needs_clarification")
        or pretoken.get("pending_clarification")
        or pretoken.get("requires_clarification")
    )
    out["ambiguities"] = list(pretoken.get("ambiguities") or [])
    out["symbolic_packet"] = pretoken.get("symbolic_packet") or pretoken.get("symbolic") or pretoken.get("compact_packet")
    out["surface_task"] = pretoken.get("surface_task") or pretoken.get("task") or {}
    return out


def _goal_surface(request_text: str, pretoken_shape: Dict[str, Any]) -> str:
    task = pretoken_shape.get("surface_task") if isinstance(pretoken_shape.get("surface_task"), dict) else {}
    for key in ("app", "target_app", "surface", "editor", "tool"):
        value = task.get(key)
        if value:
            return str(value).strip().lower()
    low = (request_text or "").lower()
    known = ["word", "excel", "powerpoint", "paint", "notepad", "vscode", "visual studio code", "dreamweaver", "browser"]
    for item in known:
        if item in low:
            return item.replace("visual studio code", "vscode")
    return ""


def _desired_outcome(request_text: str, pretoken_shape: Dict[str, Any], lane: str) -> str:
    task = pretoken_shape.get("surface_task") if isinstance(pretoken_shape.get("surface_task"), dict) else {}
    for key in ("task_kind", "action_kind", "desired_outcome"):
        val = task.get(key)
        if val:
            return str(val).strip().lower()
    low = (request_text or "").lower()
    if "document" in low and any(k in low for k in ("write", "create", "draft")):
        return "document_created_and_filled"
    if "open" in low and "document named" in low:
        return "document_opened"
    if "excel" in low or "spreadsheet" in low:
        return "spreadsheet_created"
    if "website" in low or "homepage" in low or "about page" in low:
        return "website_scaffold_created"
    if "draw" in low and "paint" in low:
        return "paint_artifact_created"
    if lane == "creative":
        return "creative_artifact_created"
    return "task_completed"


def _smget_packet_from_inputs(
    governor: Optional[Dict[str, Any]] = None,
    plan_state: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    proposed_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    candidates: List[Any] = []
    if isinstance(governor, dict):
        candidates.extend([governor.get("smget"), (governor.get("meta") if isinstance(governor.get("meta"), dict) else {}).get("smget")])
    if isinstance(plan_state, dict):
        candidates.extend([plan_state.get("smget"), (plan_state.get("governance") if isinstance(plan_state.get("governance"), dict) else {}).get("smget")])
    if isinstance(context, dict):
        candidates.extend([context.get("smget"), (context.get("governance") if isinstance(context.get("governance"), dict) else {}).get("smget")])
    if isinstance(proposed_action, dict):
        candidates.extend([proposed_action.get("smget"), (proposed_action.get("governance") if isinstance(proposed_action.get("governance"), dict) else {}).get("smget")])
    for cand in candidates:
        if isinstance(cand, dict) and cand:
            return dict(cand)
    return {}


def _smget_contract(smget: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(smget, dict):
        return {}
    contract = smget.get("action_contract") if isinstance(smget.get("action_contract"), dict) else {}
    return dict(contract) if contract else {}


def _operator_execution_state(smget: Optional[Dict[str, Any]] = None, plan_state: Optional[Dict[str, Any]] = None) -> str:
    state = dict(plan_state or {})
    contract = _smget_contract(smget)
    for value in (
        state.get("operator_state"),
        state.get("execution_state"),
        state.get("current_state"),
        state.get("smget_state"),
        contract.get("current_state"),
        contract.get("execution_state"),
    ):
        if value:
            return str(value).strip().upper()
    return ""


def _operator_summary_text(smget: Optional[Dict[str, Any]] = None, plan_state: Optional[Dict[str, Any]] = None) -> str:
    state = dict(plan_state or {})
    packet = dict(smget or {})
    contract = _smget_contract(packet)
    execution_result = state.get("execution_result") if isinstance(state.get("execution_result"), dict) else {}
    if not execution_result:
        execution_result = packet.get("execution_result") if isinstance(packet.get("execution_result"), dict) else {}
    verification_result = state.get("verification_result") if isinstance(state.get("verification_result"), dict) else {}
    if not verification_result:
        verification_result = packet.get("verification_result") if isinstance(packet.get("verification_result"), dict) else {}

    for source in (state, execution_result, verification_result, packet, contract):
        if not isinstance(source, dict):
            continue
        for key in ("candidate_text", "generated_text", "reply", "result", "summary", "execution_summary", "result_summary", "message", "details"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _smget_status_override(contract: Dict[str, Any], plan_state: Optional[Dict[str, Any]] = None, smget: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    state_name = _operator_execution_state(smget=smget, plan_state=plan_state)
    if not state_name:
        return {}

    if state_name in {"CLASSIFIED", "AUTHORIZED", "PLANNED"}:
        return {
            "status": STATUS_WAITING_ON_SIDEKICK,
            "next_required_step": DIRECTIVE_EXECUTE_SUBSTEP,
            "reason": "SMGET has prepared an ActionContract and is waiting for bounded execution to continue.",
        }
    if state_name in {"DISPATCHED", "EXECUTING"}:
        return {
            "status": STATUS_WAITING_ON_SIDEKICK,
            "next_required_step": DIRECTIVE_VERIFY_SUBSTEP,
            "reason": "SMGET execution is in progress; Reply must remain held until verification completes.",
        }
    if state_name in {"FAILED", "ROLLED_BACK"}:
        return {
            "status": STATUS_SAFE_ABORT,
            "next_required_step": DIRECTIVE_ABORT,
            "reason": "SMGET reported a failed or rolled-back execution path; do not present success.",
        }
    if state_name == "REPORTED":
        return {
            "status": STATUS_COMPARE_PENDING,
            "next_required_step": DIRECTIVE_HOLD_REPLY,
            "reason": "SMGET has packaged a result, but Compass still requires final verification and Compare release before success is presented.",
        }
    return {}


def _progress_from_state(
    previous: Dict[str, Any],
    plan_state: Optional[Dict[str, Any]] = None,
    *,
    governor: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    proposed_action: Optional[Dict[str, Any]] = None,
) -> TaskProgress:
    progress = TaskProgress()
    prev_prog = previous.get("task_progress") if isinstance(previous.get("task_progress"), dict) else {}
    state = dict(plan_state or {})

    for field_name in progress.__dataclass_fields__.keys():
        setattr(progress, field_name, bool(prev_prog.get(field_name, False) or state.get(field_name, False)))

    # Heuristic shorthands from local plan/executor state
    if state.get("surface_open") is True or state.get("app_opened") is True or state.get("launched") is True:
        progress.surface_open = True
    if state.get("surface_focused") is True or state.get("focused") is True or state.get("window_focused") is True:
        progress.surface_focused = True
    if state.get("working_context_ready") is True or state.get("blank_document_ready") is True or state.get("canvas_ready") is True:
        progress.working_context_ready = True
    if state.get("content_generated") is True or state.get("draft_ready") is True or state.get("generated") is True:
        progress.content_generated = True
    if state.get("content_applied") is True or state.get("typed_into_surface") is True or state.get("artifact_saved") is True:
        progress.content_applied = True
    if state.get("result_verified") is True or state.get("verified") is True:
        progress.result_verified = True
    if state.get("compare_passed") is True:
        progress.compare_passed = True

    # SMGET / OperatorCore lifecycle inference
    smget = _smget_packet_from_inputs(governor=governor, plan_state=state, context=context, proposed_action=proposed_action)
    contract = _smget_contract(smget)
    operator_state = _operator_execution_state(smget=smget, plan_state=state)
    execution_result = state.get("execution_result") if isinstance(state.get("execution_result"), dict) else {}
    if not execution_result:
        execution_result = smget.get("execution_result") if isinstance(smget.get("execution_result"), dict) else {}
    verification_result = state.get("verification_result") if isinstance(state.get("verification_result"), dict) else {}
    if not verification_result:
        verification_result = smget.get("verification_result") if isinstance(smget.get("verification_result"), dict) else {}

    if contract:
        progress.working_context_ready = True
    if operator_state in {"CLASSIFIED", "AUTHORIZED", "PLANNED"}:
        progress.working_context_ready = True
    if operator_state in {"DISPATCHED", "EXECUTING"}:
        progress.content_generated = True
    if operator_state in {"VERIFIED", "SUCCEEDED", "FAILED", "ROLLED_BACK", "REPORTED"} or bool(execution_result):
        progress.content_generated = True
        progress.content_applied = True
    if operator_state in {"VERIFIED", "SUCCEEDED", "REPORTED"} or bool(verification_result) or bool(state.get("smget_verified")):
        progress.result_verified = True

    compare_meta = state.get("compare") if isinstance(state.get("compare"), dict) else {}
    if not compare_meta and isinstance(smget.get("compare"), dict):
        compare_meta = dict(smget.get("compare") or {})
    compare_status = str(compare_meta.get("status") or "").strip().upper()
    if compare_status in {"PASS", "OK", "HIT"} or bool(compare_meta.get("ok")):
        progress.compare_passed = True

    return progress


def build_task_contract(
    request_text: str,
    *,
    context: Optional[Dict[str, Any]] = None,
    pretoken_packet: Optional[Dict[str, Any]] = None,
    goal_id: Optional[str] = None,
) -> Dict[str, Any]:
    ctx = dict(context or {})
    pretoken = _pretoken_task_shape(pretoken_packet or _query_pretoken(request_text, context=ctx))
    primary_lane = str(pretoken.get("likely_lane") or ctx.get("primary_lane") or ctx.get("intent") or "action").strip().lower()
    if primary_lane not in {"answer", "action", "creative", "system", "network"}:
        primary_lane = "action" if any(k in (request_text or "").lower() for k in ("open ", "create ", "write ", "draw ", "make ")) else "answer"
    family = _TASK_FAMILY_HINTS.get(primary_lane, "generic_action_workflow")
    target_surface = _goal_surface(request_text, pretoken)
    desired_outcome = _desired_outcome(request_text, pretoken, primary_lane)
    completion = [
        "surface_open",
        "surface_focused",
        "working_context_ready",
        "content_generated",
        "content_applied",
        "result_verified",
        "compare_passed",
    ]
    if primary_lane == "answer":
        completion = ["content_generated", "result_verified", "compare_passed"]
    elif primary_lane == "creative":
        completion = ["working_context_ready", "content_generated", "content_applied", "result_verified", "compare_passed"]
    elif primary_lane == "system":
        completion = ["result_verified", "compare_passed"]
    elif primary_lane == "network":
        completion = ["working_context_ready", "result_verified", "compare_passed"]

    return {
        "goal_id": str(goal_id or ("goal-" + _deterministic_hash(request_text + json.dumps(pretoken, sort_keys=True, default=str)))),
        "original_user_goal": _safe_text(request_text, 1200),
        "primary_lane": primary_lane,
        "task_family": family,
        "target_surface": target_surface,
        "desired_outcome": desired_outcome,
        "completion_proof": completion,
        "allowed_sidekick_families": list(_SIDEKICK_FAMILY_REGISTRY.keys()),
        "pretoken": pretoken,
        "context": {
            "user_present": bool(ctx.get("user_present", True)),
            "user_consented": bool(ctx.get("user_consented", False)),
            "safe_mode": bool(ctx.get("safe_mode", _flag("SAFE_MODE", False))),
            "local_only": bool(ctx.get("local_only", _flag("LOCAL_ONLY_MODE", False))),
            "public_web": _public_web(ctx),
        },
        "loop_guard": asdict(LoopGuardState()),
    }


# -----------------------------------------------------------------------------
# Core compass logic
# -----------------------------------------------------------------------------
def _next_missing_step(contract: Dict[str, Any], progress: TaskProgress) -> Tuple[str, str]:
    lane = str(contract.get("primary_lane") or "action")
    desired = str(contract.get("desired_outcome") or "")
    target_surface = str(contract.get("target_surface") or "").strip().lower()

    if lane == "answer":
        if not progress.content_generated:
            return STATUS_PROCEDURAL_GAP, DIRECTIVE_EXECUTE_SUBSTEP
        if not progress.result_verified:
            return STATUS_COMPARE_PENDING, DIRECTIVE_VERIFY_FINAL
        if not progress.compare_passed:
            return STATUS_COMPARE_PENDING, DIRECTIVE_HOLD_REPLY
        return STATUS_COMPLETE_VERIFIED, DIRECTIVE_ALLOW_REPLY

    # Generic non-surface workflows (drivers, network, bounded system actions, SMGET contracts)
    if lane in {"system", "network"} or (lane == "action" and not target_surface):
        if not progress.working_context_ready:
            return STATUS_PROCEDURAL_GAP, DIRECTIVE_EXECUTE_SUBSTEP
        if not progress.content_generated:
            return STATUS_WAITING_ON_SIDEKICK, DIRECTIVE_EXECUTE_SUBSTEP
        if not progress.content_applied:
            return STATUS_UNVERIFIED_COMPLETION, DIRECTIVE_VERIFY_SUBSTEP
        if not progress.result_verified:
            return STATUS_COMPARE_PENDING, DIRECTIVE_VERIFY_FINAL
        if not progress.compare_passed:
            return STATUS_COMPARE_PENDING, DIRECTIVE_HOLD_REPLY
        return STATUS_COMPLETE_VERIFIED, DIRECTIVE_ALLOW_REPLY

    if not progress.surface_open:
        return STATUS_PROCEDURAL_GAP, DIRECTIVE_EXECUTE_SUBSTEP
    if not progress.surface_focused:
        return STATUS_MINOR_DRIFT, DIRECTIVE_FOCUS_SURFACE
    if not progress.working_context_ready:
        return STATUS_PROCEDURAL_GAP, DIRECTIVE_PREPARE_SURFACE
    if "open" in desired and desired.endswith("opened") and progress.working_context_ready:
        if not progress.result_verified:
            return STATUS_COMPARE_PENDING, DIRECTIVE_VERIFY_FINAL
        if not progress.compare_passed:
            return STATUS_COMPARE_PENDING, DIRECTIVE_HOLD_REPLY
        return STATUS_COMPLETE_VERIFIED, DIRECTIVE_ALLOW_REPLY
    if not progress.content_generated:
        return STATUS_PROCEDURAL_GAP, DIRECTIVE_EXECUTE_SUBSTEP
    if not progress.content_applied:
        return STATUS_UNVERIFIED_COMPLETION, DIRECTIVE_EXECUTE_SUBSTEP
    if not progress.result_verified:
        return STATUS_COMPARE_PENDING, DIRECTIVE_VERIFY_FINAL
    if not progress.compare_passed:
        return STATUS_COMPARE_PENDING, DIRECTIVE_HOLD_REPLY
    return STATUS_COMPLETE_VERIFIED, DIRECTIVE_ALLOW_REPLY


def _update_loop_guard(
    previous: Dict[str, Any],
    selected_family: Optional[str],
    selected_module: Optional[str],
    status: str,
    next_required_step: str,
) -> LoopGuardState:
    prev = previous.get("loop_guard") if isinstance(previous.get("loop_guard"), dict) else {}
    lg = LoopGuardState(**{k: prev.get(k, getattr(LoopGuardState(), k)) for k in LoopGuardState().__dict__.keys()})
    lg.total_cycles += 1
    lg.attempt_count += 1

    if str(previous.get("status") or "") == str(status):
        lg.same_state_repeats += 1
    else:
        lg.same_state_repeats = 0

    if str(previous.get("selected_sidekick_family") or "") == str(selected_family or "") and str(previous.get("selected_sidekick_module") or "") == str(selected_module or ""):
        lg.same_helper_repeats += 1
    else:
        if previous.get("selected_sidekick_family") or previous.get("selected_sidekick_module"):
            lg.helper_switches += 1
        lg.same_helper_repeats = 0

    if str(previous.get("next_required_step") or "") != str(next_required_step or ""):
        lg.reanchors += 1

    return lg


def _escape_hatch_from_guard(lg: LoopGuardState, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = dict(context or {})
    if lg.same_state_repeats > lg.max_same_state_repeats or lg.same_helper_repeats > lg.max_same_helper_repeats:
        return {"triggered": True, "mode": ESCAPE_SWITCH_FAMILY, "reason": "stagnation_detected"}
    if lg.total_cycles > lg.max_total_cycles or lg.attempt_count > lg.max_attempts_per_step:
        if bool(ctx.get("local_only", _flag("LOCAL_ONLY_MODE", False))):
            return {"triggered": True, "mode": ESCAPE_SAFE_REPLY, "reason": "attempt_budget_exceeded_local_only"}
        return {"triggered": True, "mode": ESCAPE_RESEARCH_FIRST, "reason": "attempt_budget_exceeded"}
    if lg.helper_switches > lg.max_total_switches or lg.reanchors > lg.max_total_reanchors:
        return {"triggered": True, "mode": ESCAPE_STOP_RED_ZONE, "reason": "red_zone_bearing_loss"}
    return {"triggered": False, "mode": None, "reason": None}


def _compare_gate(
    request_text: str,
    contract: Dict[str, Any],
    progress: TaskProgress,
    plan_state: Optional[Dict[str, Any]] = None,
    *,
    governor: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    proposed_action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if progress.compare_passed:
        return {"status": "PASS", "ok": True}
    if not progress.result_verified:
        return {"status": "PENDING", "ok": False}

    smget = _smget_packet_from_inputs(governor=governor, plan_state=plan_state, context=context, proposed_action=proposed_action)
    candidate_text = _operator_summary_text(smget=smget, plan_state=plan_state)
    if not candidate_text:
        return {"status": "PENDING", "ok": False, "reason": "no_candidate_text"}
    intent = str(contract.get("primary_lane") or "general")
    compare = _compare_accepts(request_text, candidate_text, intent=intent)
    status = str(compare.get("status") or "").upper()
    ok = status in {"HIT", "PASS", "OK"} or bool(compare.get("ok"))
    return {"status": status or "ERROR", "ok": ok, "compare": compare, "candidate_text": _safe_text(candidate_text, 600)}


def compass_guard(
    request_text: str,
    *,
    caller_context: Optional[Dict[str, Any]] = None,
    plan_state: Optional[Dict[str, Any]] = None,
    proposed_action: Optional[Dict[str, Any]] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Core universal guard.

    Input:
    - request_text: original user goal
    - caller_context: runtime + policy context
    - plan_state: current execution/progress claims from Neuron/executor layer
    - proposed_action: optional structured action

    Output:
    - structured compass state containing status, directives, sidekick selection,
      loop guard, reply gate, and escape-hatch instructions.
    """
    ctx = dict(caller_context or {})
    contract = build_task_contract(request_text, context=ctx)
    goal_id = str(contract.get("goal_id"))
    previous = _state_get(goal_id)

    self_model = _query_cognitive_self(request_text, context=ctx, force_refresh=force_refresh)
    governor = _query_governor(request_text, context=ctx, proposed_action=proposed_action)
    thinker = _query_thinker(request_text, context=ctx, proposed_action=proposed_action) if ctx.get("consult_thinker", True) else {}

    progress = _progress_from_state(previous, plan_state, governor=governor, context=ctx, proposed_action=proposed_action)

    # Clarification gate from PreTokenAnalyzer always wins before continuation.
    pretoken = contract.get("pretoken") if isinstance(contract.get("pretoken"), dict) else {}
    if bool(pretoken.get("clarification_required")):
        state = CompassState(
            goal_id=goal_id,
            original_user_goal=contract["original_user_goal"],
            primary_lane=contract["primary_lane"],
            task_family=contract["task_family"],
            target_surface=contract["target_surface"],
            desired_outcome=contract["desired_outcome"],
            status=STATUS_RED_ZONE,
            reply_allowed=False,
            continue_allowed=False,
            reanchor_target=DIRECTIVE_REANCHOR_PRETOKEN,
            next_required_step=ESCAPE_CLARIFY,
            reason="PreTokenAnalyzer reports unresolved ambiguity; task must not continue until clarification is merged into the same conceptual query.",
            candidate_sidekick_families=[],
            candidate_sidekick_modules=[],
            task_progress=asdict(progress),
            loop_guard=asdict(LoopGuardState()),
            escape_hatch={"triggered": True, "mode": ESCAPE_CLARIFY, "reason": "clarification_required"},
            meta={
                "pretoken": pretoken,
                "governor_decision": governor.get("decision"),
                "thinker_decision": thinker.get("final_balance_decision") if isinstance(thinker, dict) else None,
            },
        )
        out = asdict(state)
        _state_set(goal_id, out)
        _log_event("clarification_hold", goal_id, out["reason"], "WARNING", {"pretoken": pretoken})
        return out

    # Governor is authoritative on whether the next move may proceed.
    g_decision = str(governor.get("decision") or "ALLOW").upper() if isinstance(governor, dict) else "ALLOW"
    if g_decision in {"DENY", "DEFER", "REQUIRE_USER"}:
        reason = "Governor did not authorize continued execution."
        if g_decision == "REQUIRE_USER":
            reason = "Governor requires explicit user approval before continuation."
        elif g_decision == "DEFER":
            reason = "Governor deferred the request pending more metadata or a safer path."
        state = CompassState(
            goal_id=goal_id,
            original_user_goal=contract["original_user_goal"],
            primary_lane=contract["primary_lane"],
            task_family=contract["task_family"],
            target_surface=contract["target_surface"],
            desired_outcome=contract["desired_outcome"],
            status=STATUS_SAFE_ABORT if g_decision == "DENY" else STATUS_RED_ZONE,
            reply_allowed=True if g_decision in {"DENY", "DEFER", "REQUIRE_USER"} else False,
            continue_allowed=False,
            reanchor_target=DIRECTIVE_REANCHOR_GOVERNOR,
            next_required_step=DIRECTIVE_ABORT,
            reason=reason,
            task_progress=asdict(progress),
            loop_guard=asdict(LoopGuardState()),
            escape_hatch={"triggered": True, "mode": ESCAPE_SAFE_REPLY, "reason": g_decision.lower()},
            meta={"governor": governor, "thinker": thinker, "pretoken": pretoken},
        )
        out = asdict(state)
        _state_set(goal_id, out)
        _log_event("governor_hold", goal_id, reason, "WARNING", {"decision": g_decision})
        return out

    smget = _smget_packet_from_inputs(governor=governor, plan_state=plan_state, context=ctx, proposed_action=proposed_action)
    smget_contract = _smget_contract(smget)
    status, next_required_step = _next_missing_step(contract, progress)

    smget_override = _smget_status_override(contract, plan_state=plan_state, smget=smget)
    if smget_override:
        status = str(smget_override.get("status") or status)
        next_required_step = str(smget_override.get("next_required_step") or next_required_step)

    need_map = {
        DIRECTIVE_FOCUS_SURFACE: "focus_window",
        DIRECTIVE_PREPARE_SURFACE: "prepare_surface",
        DIRECTIVE_EXECUTE_SUBSTEP: "generate_content" if progress.working_context_ready else "software_procedure",
        DIRECTIVE_VERIFY_FINAL: "verify_completion",
    }
    missing_need = need_map.get(next_required_step, "software_procedure")

    # Sidekick selection only when not done.
    selected = select_sidekick_family(
        request_text,
        missing_need,
        primary_lane=contract["primary_lane"],
        context=ctx,
        self_model=self_model,
    )

    selected_family = selected.get("selected_family")
    selected_module = selected.get("selected_module")
    if smget_contract:
        selected_family = selected_family or "governed_execution"
        selected_module = selected_module or str(smget_contract.get("executor_name") or "SarahMemoryOperatorCore")
    loop_guard = _update_loop_guard(previous, selected_family, selected_module, status, next_required_step)
    escape = _escape_hatch_from_guard(loop_guard, context=ctx)

    if escape.get("triggered"):
        status = STATUS_ESCAPE_HATCH if escape.get("mode") != ESCAPE_STOP_RED_ZONE else STATUS_RED_ZONE
        next_required_step = DIRECTIVE_ABORT

    compare_gate = _compare_gate(request_text, contract, progress, plan_state, governor=governor, context=ctx, proposed_action=proposed_action)
    if compare_gate.get("ok"):
        progress.compare_passed = True
        status = STATUS_COMPLETE_VERIFIED
        next_required_step = DIRECTIVE_ALLOW_REPLY
    elif progress.result_verified and not compare_gate.get("ok") and not escape.get("triggered"):
        status = STATUS_COMPARE_PENDING
        next_required_step = DIRECTIVE_HOLD_REPLY

    reason = {
        STATUS_ON_COURSE: "Task remains on course.",
        STATUS_MINOR_DRIFT: "Task progressed, but the active surface is not yet focused or aligned.",
        STATUS_PROCEDURAL_GAP: "Neuron knows the goal, but the required procedure is still missing or incomplete.",
        STATUS_WAITING_ON_SIDEKICK: "Awaiting the selected helper path to continue execution.",
        STATUS_UNVERIFIED_COMPLETION: "A partial step completed, but the requested outcome is not yet proven.",
        STATUS_NO_PROGRESS: "The task has not produced measurable progress yet.",
        STATUS_LOOP_RISK: "Repeated states or helper selections indicate loop risk.",
        STATUS_RED_ZONE: "The request is entering a red-zone condition: drift, ambiguity, or instability.",
        STATUS_ESCAPE_HATCH: "Compass triggered an escape hatch to prevent a loop or red-zone stall.",
        STATUS_SAFE_ABORT: "Task was safely aborted under governance or escape-hatch control.",
        STATUS_COMPARE_PENDING: "Task outcome still requires validation before Reply may present success.",
        STATUS_COMPLETE_VERIFIED: "Task completion is verified and Compare has cleared release.",
    }.get(status, "Compass evaluation completed.")

    if status == STATUS_PROCEDURAL_GAP and not selected_family:
        reason = "No approved sidekick family is currently available for the missing procedural step."
    if status == STATUS_COMPARE_PENDING and not progress.result_verified:
        reason = "Reply is held because the requested outcome has not been verified yet."
    if escape.get("triggered"):
        reason = f"Escape hatch triggered: {escape.get('mode')} ({escape.get('reason')})."
    if smget_override and smget_override.get("reason") and not escape.get("triggered"):
        reason = str(smget_override.get("reason") or reason)

    reply_allowed = status == STATUS_COMPLETE_VERIFIED and next_required_step == DIRECTIVE_ALLOW_REPLY
    continue_allowed = ((not escape.get("triggered")) and status not in {STATUS_SAFE_ABORT, STATUS_RED_ZONE}) or bool(status in {STATUS_PROCEDURAL_GAP, STATUS_MINOR_DRIFT, STATUS_COMPARE_PENDING, STATUS_WAITING_ON_SIDEKICK})
    if escape.get("triggered"):
        continue_allowed = False
    if status == STATUS_RED_ZONE and not reply_allowed:
        continue_allowed = False

    state = CompassState(
        goal_id=goal_id,
        original_user_goal=contract["original_user_goal"],
        primary_lane=contract["primary_lane"],
        task_family=contract["task_family"],
        target_surface=contract["target_surface"],
        desired_outcome=contract["desired_outcome"],
        status=status,
        reply_allowed=reply_allowed,
        continue_allowed=continue_allowed,
        reanchor_target=(DIRECTIVE_REANCHOR_ORIGINAL if status in {STATUS_PROCEDURAL_GAP, STATUS_UNVERIFIED_COMPLETION, STATUS_LOOP_RISK, STATUS_NO_PROGRESS} else DIRECTIVE_REANCHOR_GOVERNOR if status in {STATUS_SAFE_ABORT, STATUS_RED_ZONE} else ""),
        next_required_step=next_required_step,
        reason=reason,
        selected_sidekick_family=selected_family,
        selected_sidekick_module=selected_module,
        candidate_sidekick_families=list(selected.get("candidate_families") or []),
        candidate_sidekick_modules=list(selected.get("candidate_modules") or []),
        task_progress=asdict(progress),
        loop_guard=asdict(loop_guard),
        escape_hatch=escape,
        meta={
            "contract": contract,
            "governor": governor,
            "thinker": thinker,
            "self_summary": ((self_model.get("identity") or {}) if isinstance(self_model, dict) else {}),
            "pretoken": pretoken,
            "compare_gate": compare_gate,
            "smget": {
                "enabled": bool(smget),
                "contract_id": smget_contract.get("contract_id"),
                "executor_name": smget_contract.get("executor_name"),
                "execution_mode": smget_contract.get("execution_mode"),
                "current_state": _operator_execution_state(smget=smget, plan_state=plan_state),
            },
        },
    )

    out = asdict(state)
    _state_set(goal_id, out)
    severity = "INFO"
    if status in {STATUS_LOOP_RISK, STATUS_RED_ZONE, STATUS_SAFE_ABORT, STATUS_ESCAPE_HATCH}:
        severity = "WARNING"
    _log_event("compass_guard", goal_id, reason, severity, {"status": status, "next_required_step": next_required_step})
    return out


# -----------------------------------------------------------------------------
# Public helpers intended for the rest of the stack
# -----------------------------------------------------------------------------
def get_compass_packet(
    request_text: str,
    *,
    caller_context: Optional[Dict[str, Any]] = None,
    plan_state: Optional[Dict[str, Any]] = None,
    proposed_action: Optional[Dict[str, Any]] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    return compass_guard(
        request_text,
        caller_context=caller_context,
        plan_state=plan_state,
        proposed_action=proposed_action,
        force_refresh=force_refresh,
    )


def should_hold_reply(compass_packet: Dict[str, Any]) -> bool:
    return not bool((compass_packet or {}).get("reply_allowed"))


def allow_reply(compass_packet: Dict[str, Any]) -> bool:
    return bool((compass_packet or {}).get("reply_allowed"))


def compass_release_gate(compass_packet: Dict[str, Any]) -> Dict[str, Any]:
    pkt = dict(compass_packet or {})
    return {
        "reply_allowed": bool(pkt.get("reply_allowed")),
        "continue_allowed": bool(pkt.get("continue_allowed")),
        "status": str(pkt.get("status") or ""),
        "next_required_step": str(pkt.get("next_required_step") or ""),
        "reason": str(pkt.get("reason") or ""),
    }


def get_recent_compass_events(limit: int = 25) -> List[Dict[str, Any]]:
    con = None
    out: List[Dict[str, Any]] = []
    try:
        _ensure_tables()
        con = _connect_db()
        rows = con.execute(
            "SELECT ts, event_kind, goal_id, severity, details, meta_json FROM compass_events ORDER BY id DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        for row in rows:
            out.append(
                {
                    "ts": row[0],
                    "event_kind": row[1],
                    "goal_id": row[2],
                    "severity": row[3],
                    "details": row[4],
                    "meta": json.loads(row[5] or "{}"),
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


def explain_compass_role() -> str:
    return (
        "SarahMemoryCognitiveCompass is the orientation watchdog for SarahMemory AiOS. "
        "It keeps SarahMemoryNeuron on bearing, prevents premature completion, selects appropriate sidekick families for missing procedural steps, "
        "holds Reply until verification passes, and triggers safe escape hatches when drift or loop risk appears."
    )


# -----------------------------------------------------------------------------
# Self-test
# -----------------------------------------------------------------------------
def _run_self_test() -> bool:
    print("[SarahMemoryCognitiveCompass] Self-test (safe/offline)")
    _ensure_tables()
    print("[OK] DB tables ensured:", _db_path())

    sample_1 = compass_guard(
        "Open Word and write me a document about cats",
        caller_context={
            "user_present": True,
            "user_consented": True,
            "safe_mode": False,
            "local_only": True,
            "consult_thinker": False,
        },
        plan_state={"surface_open": True},
    )
    print(json.dumps({
        "status": sample_1.get("status"),
        "next_required_step": sample_1.get("next_required_step"),
        "selected_sidekick_family": sample_1.get("selected_sidekick_family"),
        "selected_sidekick_module": sample_1.get("selected_sidekick_module"),
        "reply_allowed": sample_1.get("reply_allowed"),
    }, indent=2))

    sample_2 = compass_guard(
        "Create me an image of a cat with sunglasses on a skateboard dabbing in the city",
        caller_context={
            "user_present": True,
            "user_consented": True,
            "safe_mode": False,
            "local_only": False,
            "consult_thinker": False,
        },
        plan_state={"working_context_ready": True, "content_generated": True, "content_applied": True, "result_verified": True, "compare_passed": True},
    )
    print(json.dumps({
        "status": sample_2.get("status"),
        "next_required_step": sample_2.get("next_required_step"),
        "reply_allowed": sample_2.get("reply_allowed"),
    }, indent=2))

    sample_3 = compass_guard(
        "Open Word and write report for LA plant",
        caller_context={
            "user_present": True,
            "user_consented": True,
            "safe_mode": False,
            "local_only": True,
            "consult_thinker": False,
        },
        plan_state={},
    )
    print(json.dumps({
        "status": sample_3.get("status"),
        "next_required_step": sample_3.get("next_required_step"),
        "escape_hatch": sample_3.get("escape_hatch"),
    }, indent=2))

    return True


def main() -> int:
    try:
        ok = _run_self_test()
        return 0 if ok else 1
    except Exception as e:
        print("[FATAL] CognitiveCompass self-test failed:", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# V10/V9F SelfAware evidence case compass helper
# -----------------------------------------------------------------------------
def compass_review_selfaware_evidence_case(original_goal: str = "", evidence_packet: Optional[Dict[str, Any]] = None, appeal_packet: Optional[Dict[str, Any]] = None, proposed_reply: str = "") -> Dict[str, Any]:
    """Keep SelfAware evidence answers anchored to the original requested component."""
    evidence = dict(evidence_packet or {})
    requested = str(evidence.get('requested_component') or (appeal_packet or {}).get('requested_component') or '')
    drift = False; reasons = []
    low_reply = str(proposed_reply or '').lower()
    if requested == 'cpu' and 'gpu temperature' in low_reply and 'cpu temperature' not in low_reply:
        drift = True; reasons.append('Reply drifted from CPU temperature to GPU-only thermal status.')
    if requested and requested.replace('_',' ') not in low_reply and evidence.get('selected_reading'):
        reasons.append('Reply should explicitly mention requested component and sensor source.')
    return {'ok': True, 'module': MODULE_NAME, 'status': STATUS_MINOR_DRIFT if drift else STATUS_ON_COURSE, 'directive': DIRECTIVE_REANCHOR_ORIGINAL if drift else DIRECTIVE_ALLOW_REPLY, 'requested_component': requested, 'original_goal': original_goal, 'hold_reply': bool(drift), 'reasons': reasons or ['SelfAware evidence answer remains anchored to the original requested component.'], 'read_only': True, 'action_taken': False}

# -----------------------------------------------------------------------------
# V10/V9G SelfAware answer-shape compass review
# -----------------------------------------------------------------------------
def review_selfaware_answer_bearing(original_goal: str = "", evidence_packet: Optional[Dict[str, Any]] = None, appeal_packet: Optional[Dict[str, Any]] = None, proposed_reply: str = "", canonical_packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Keep SelfAware body answers anchored to the user's requested component/metric."""
    pkt = dict(canonical_packet or {})
    if not pkt and isinstance(appeal_packet, dict):
        pkt = {
            'requested_component': appeal_packet.get('requested_component'),
            'requested_metric': appeal_packet.get('requested_metric'),
        }
    metric = str(pkt.get('requested_metric') or '').lower()
    component = str(pkt.get('requested_component') or pkt.get('target') or '').lower()
    reply = str(proposed_reply or '').lower()
    directives = []
    hold = False
    status = STATUS_ON_COURSE if 'STATUS_ON_COURSE' in globals() else 'ON_COURSE'
    if metric == 'temperature':
        if component == 'cpu' and ('physical cores' in reply or 'logical threads' in reply) and 'temperature' not in reply and '°c' not in reply:
            hold = True; status = STATUS_PROCEDURAL_GAP if 'STATUS_PROCEDURAL_GAP' in globals() else 'PROCEDURAL_GAP'; directives.append('REANCHOR_TO_CPU_TEMPERATURE')
        if component and component != 'gpu' and 'gpu temperature' in reply and f'{component} temperature' not in reply:
            hold = True; status = STATUS_MINOR_DRIFT if 'STATUS_MINOR_DRIFT' in globals() else 'MINOR_DRIFT'; directives.append('DO_NOT_SUBSTITUTE_RELATED_THERMAL_SENSOR')
    if metric == 'bios_version' and ('server version' in reply or 'my name is' in reply):
        hold = True; status = STATUS_PROCEDURAL_GAP if 'STATUS_PROCEDURAL_GAP' in globals() else 'PROCEDURAL_GAP'; directives.append('REANCHOR_TO_HARDWARE_BIOS_VERSION')
    return {'ok': True, 'module': 'SarahMemoryCognitiveCompass', 'status': status, 'hold_reply': bool(hold), 'directives': directives, 'original_goal': original_goal, 'requested_metric': metric, 'requested_component': component}

# --- SM V8.0 TRI-LAYER PATCH 2026-05-20 ---
def assess_language_identity_drift(
    original_goal: str,
    current_route: str = "",
    language_context_packet: Optional[Dict[str, Any]] = None,
    emotion_affect_packet: Optional[Dict[str, Any]] = None,
    six_question_packet: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Detect phrase-lock, substring-route, identity, emotion-override, and open-loop drift."""
    lang = language_context_packet if isinstance(language_context_packet, dict) else {}
    emo = emotion_affect_packet if isinstance(emotion_affect_packet, dict) else {}
    six = six_question_packet if isinstance(six_question_packet, dict) else {}
    warnings: List[str] = []
    status = STATUS_ON_COURSE
    directive = DIRECTIVE_ALLOW_REPLY
    route = str(current_route or "").lower()
    for item in lang.get("blocked_substring_matches") or []:
        if isinstance(item, dict) and route and str(item.get("candidate") or "").lower() in route:
            warnings.append(f"route_substring_blocked:{item.get('candidate')} inside {item.get('inside_phrase')}")
            status = STATUS_RED_ZONE
            directive = DIRECTIVE_REANCHOR_PRETOKEN
    if emo.get("output_constraints", {}).get("emotion_does_not_authorize_action") is not True:
        warnings.append("emotion_packet_missing_no_authority_doctrine")
        status = STATUS_MINOR_DRIFT if status == STATUS_ON_COURSE else status
    if six and not bool(six.get("loop_closed")) and str(six.get("final_decision") or "").upper() in {"DEFER", "UNKNOWN"}:
        warnings.append("six_question_loop_not_closed")
        status = STATUS_PROCEDURAL_GAP if status == STATUS_ON_COURSE else status
        directive = DIRECTIVE_REANCHOR_GOVERNOR
    return {
        "ok": not warnings,
        "status": status,
        "directive": directive,
        "warnings": warnings,
        "original_goal": _safe_text(original_goal, 500) if '_safe_text' in globals() else str(original_goal or '')[:500],
        "current_route": current_route,
        "reply_allowed": status not in {STATUS_RED_ZONE, STATUS_PROCEDURAL_GAP, STATUS_LOOP_RISK},
        "execution_allowed": False,
        "source": "CognitiveCompass.tri_layer_drift_guard",
    }
