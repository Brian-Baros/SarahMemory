"""--==The SarahMemory Project==--
File: SarahMemoryNeuron.py
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

==============================================================================================================================================================

--------
SarahMemoryNeuron is a cognitive axis module.
It consolidates:
- Meta-cognition (confidence, self-check, contradiction detection)
- Cross-domain synthesis (math/chem/physics/code/system constraints)
- Curiosity engine (gap detection + safe experiment proposals)
- Cognitive graph core (MeaningGraph-like lightweight memory links)
- Hybrid routing (deterministic first, API second, sandbox optional)
- Parallel thought architecture (Analyst/Skeptic/Optimizer/Engineer/Governor)
- AdvCU delegation for intent + command parsing (better routing immediately).
- Research lane insertion (Tier-2 evidence-backed answers).
- Creative job ticket output (standardizes creative requests for Studio).
- Compare-based QA gate (confidence calibration + consensus).
- SarahMemoryGlobals.py: identity + mode flags + paths + safety envelope
- SarahMemoryLogicCalc.py: deterministic scientific reasoning (Tier-0)
- SarahMemoryWebSYM.py: symbolic router (Tier-1)
- SarahMemoryResearch.py: evidence lane (Tier-2)
- SarahMemoryCanvasStudio.py: creative job ticketing (Creative lane)
- SarahMemoryCompare.py: QA gate (post-check)
- SarahMemoryAPI.py: multi-provider LLM routing (Tier-3)
- SarahMemoryCognitiveServices.py: orchestration/awareness bridge (optional)
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "router"
# CATEGORY = "cognition"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "neuron_axis"
# FAMILY = "core_cognition"
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
# NOTES = "Central cognitive axis and lane selector. Owns primary routing, deterministic-first reasoning, helper delegation, research insertion, creative tickets, and QA gating."
# --- SARAHMETA END ---

import os
import re
import shutil
import subprocess
import platform
import sys
import time
import json
import queue
import sqlite3
import logging
import threading
import atexit
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ARILE organ sentinel helper. This file reports local variance to the central
# SarahMemoryARILE.py engine without owning ARILE authority.
try:
    from SarahMemoryARILE import ARILESentinelBase, arile_emit, arile_should_run
except Exception:  # pragma: no cover
    ARILESentinelBase = object  # type: ignore
    arile_emit = None  # type: ignore
    def arile_should_run(lane: str, source: str = "unknown", default: bool = True) -> bool:
        return bool(default)

class LocalARILESentinel(ARILESentinelBase):
    organ_name = __name__

    def report(self, failure_type: str, summary: str, severity: float = 0.50, **data) -> None:
        try:
            if callable(arile_emit):
                arile_emit(source=__name__, organ=self.organ_name, kind="organ_variance", failure_type=failure_type, severity=severity, confidence=0.82, risk="high" if severity >= 0.75 else "medium", summary=summary, requires_governance=severity >= 0.75, retention="security_audit" if severity >= 0.75 else "diagnostic", data=data)
        except Exception:
            pass

_local_arile_sentinel = LocalARILESentinel()


# -----------------------------------------------------------------------------
# Safe imports (never hard-fail the platform)
# -----------------------------------------------------------------------------
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None

# Deterministic core (Tier-0)
try:
    from SarahMemoryLogicCalc import LogicCalc as _LogicCalc  # type: ignore
except Exception:
    _LogicCalc = None

# Symbolic router (Tier-1)
try:
    import SarahMemoryWebSYM as _WebSYM  # type: ignore
except Exception:
    _WebSYM = None

# Synapses governance + sandbox (Tier-2 / optional)
try:
    import SarahMemorySynapes as _Syn  # type: ignore
except Exception:
    _Syn = None

# Multi-provider LLM API (Tier-3)
try:
    import SarahMemoryAPI as _SMAPI  # type: ignore
except Exception:
    _SMAPI = None

# Optional orchestrator/awareness layer
try:
    import SarahMemoryCognitiveServices as _Cog  # type: ignore
except Exception:
    _Cog = None

# Governed execution choke-point
try:
    import SarahMemoryOperatorCore as _OperatorCore  # type: ignore
except Exception:
    _OperatorCore = None

# Tier-0.5: Advanced command understanding
try:
    import SarahMemoryAdvCU as _AdvCU  # type: ignore
except Exception:
    _AdvCU = None

# Tier-2: Evidence-backed research
try:
    import SarahMemoryResearch as _Research  # type: ignore
except Exception:
    _Research = None

# Creative lane: ticketing / directories / ids
try:
    import SarahMemoryCanvasStudio as _CanvasStudio  # type: ignore
except Exception:
    _CanvasStudio = None

# QA gate: compare / consensus
try:
    import SarahMemoryCompare as _Compare  # type: ignore
except Exception:
    _Compare = None

# Optional Vision workers (helper-only; never hard-fail the platform)
try:
    import SarahMemorySOBJE as _SOBJE  # type: ignore
except Exception:
    _SOBJE = None

try:
    import SarahMemoryFacialRecognition as _FaceRec  # type: ignore
except Exception:
    _FaceRec = None

try:
    import base64 as _sm_base64  # type: ignore
except Exception:
    _sm_base64 = None

try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None

try:
    import cv2 as _cv2  # type: ignore
except Exception:
    _cv2 = None


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("SarahMemoryNeuron")
logger.setLevel(logging.DEBUG if bool(getattr(config, "DEBUG_MODE", True)) else logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.propagate = False


# -----------------------------------------------------------------------------
# Deterministic bounded-call execution and circuit breakers
# -----------------------------------------------------------------------------
@dataclass
class _CircuitRecord:
    failures: int = 0
    opened_until: float = 0.0
    in_flight: bool = False
    generation: int = 0
    timed_out_generation: int = 0
    last_error: str = ""
    last_duration_s: float = 0.0


_CIRCUIT_LOCK = threading.RLock()
_CIRCUITS: Dict[str, _CircuitRecord] = {}


def _runtime_float(name: str, default: float) -> float:
    try:
        return max(0.01, float(getattr(config, name, default) if config else default))
    except Exception:
        return max(0.01, float(default))


def _runtime_int(name: str, default: int) -> int:
    try:
        return max(1, int(getattr(config, name, default) if config else default))
    except Exception:
        return max(1, int(default))


def _bounded_call(key: str, fn, *, timeout_s: float, default: Any = None) -> Dict[str, Any]:
    """Run one helper call behind a hard caller deadline and bounded circuit.

    Python cannot forcibly terminate an arbitrary in-process thread. To prevent
    UI/terminal lockups without spawning unbounded workers, each circuit permits
    only one in-flight daemon call. A timed-out worker is isolated; subsequent
    requests fail fast until it exits and the cooldown expires.
    """
    circuit_key = str(key or "helper").strip().lower() or "helper"
    timeout = max(0.01, float(timeout_s))
    now = time.monotonic()
    with _CIRCUIT_LOCK:
        rec = _CIRCUITS.setdefault(circuit_key, _CircuitRecord())
        if rec.opened_until > now:
            return {
                "ok": False,
                "value": default,
                "error": "circuit_open",
                "circuit": circuit_key,
                "retry_after_s": round(rec.opened_until - now, 3),
            }
        if rec.in_flight:
            return {
                "ok": False,
                "value": default,
                "error": "call_already_in_flight",
                "circuit": circuit_key,
            }
        rec.in_flight = True
        rec.generation += 1
        generation = rec.generation

    result_q: "queue.Queue[Tuple[str, Any, float]]" = queue.Queue(maxsize=1)
    started = time.monotonic()

    def _worker() -> None:
        status = "ok"
        payload: Any = None
        try:
            payload = fn()
        except BaseException as exc:  # keep helper failure isolated from request thread
            status = "error"
            payload = f"{type(exc).__name__}: {exc}"
        duration = time.monotonic() - started
        try:
            result_q.put_nowait((status, payload, duration))
        except Exception:
            pass
        with _CIRCUIT_LOCK:
            current = _CIRCUITS.setdefault(circuit_key, _CircuitRecord())
            if current.generation == generation:
                current.in_flight = False
                current.last_duration_s = duration
                if status == "error":
                    current.last_error = str(payload)
                if current.timed_out_generation != generation:
                    if status == "ok":
                        current.failures = 0
                        current.opened_until = 0.0
                        current.last_error = ""
                    else:
                        current.failures += 1

    threading.Thread(
        target=_worker,
        name=f"SMNeuronBounded-{circuit_key[:32]}",
        daemon=True,
    ).start()

    try:
        status, payload, duration = result_q.get(timeout=timeout)
    except queue.Empty:
        threshold = _runtime_int("NEURON_CIRCUIT_FAILURE_THRESHOLD", 2)
        cooldown = _runtime_float("NEURON_CIRCUIT_COOLDOWN_SECONDS", 30.0)
        with _CIRCUIT_LOCK:
            rec = _CIRCUITS.setdefault(circuit_key, _CircuitRecord())
            if rec.generation == generation:
                rec.timed_out_generation = generation
                rec.failures += 1
                rec.last_error = f"timeout_after_{timeout:.3f}s"
                if rec.failures >= threshold:
                    rec.opened_until = time.monotonic() + cooldown
        return {
            "ok": False,
            "value": default,
            "error": "timeout",
            "timed_out": True,
            "timeout_s": timeout,
            "circuit": circuit_key,
        }

    if status == "error":
        threshold = _runtime_int("NEURON_CIRCUIT_FAILURE_THRESHOLD", 2)
        cooldown = _runtime_float("NEURON_CIRCUIT_COOLDOWN_SECONDS", 30.0)
        with _CIRCUIT_LOCK:
            rec = _CIRCUITS.setdefault(circuit_key, _CircuitRecord())
            if rec.failures >= threshold:
                rec.opened_until = time.monotonic() + cooldown
        return {
            "ok": False,
            "value": default,
            "error": str(payload),
            "duration_s": duration,
            "circuit": circuit_key,
        }
    return {
        "ok": True,
        "value": payload,
        "error": None,
        "duration_s": duration,
        "circuit": circuit_key,
    }


def neuron_circuit_status() -> Dict[str, Any]:
    now = time.monotonic()
    with _CIRCUIT_LOCK:
        return {
            key: {
                "failures": rec.failures,
                "open": bool(rec.opened_until > now),
                "retry_after_s": round(max(0.0, rec.opened_until - now), 3),
                "in_flight": rec.in_flight,
                "last_error": rec.last_error,
                "last_duration_s": round(rec.last_duration_s, 4),
            }
            for key, rec in _CIRCUITS.items()
        }


# -----------------------------------------------------------------------------
# Paths / DB (best-effort, non-fatal)
# -----------------------------------------------------------------------------
def _base_dir() -> str:
    try:
        base = str(getattr(config, "BASE_DIR", "") or "").strip()
        if base:
            return base
    except Exception:
        pass
    try:
        here = os.path.abspath(os.path.dirname(__file__))
        if os.path.basename(here).lower() == "core":
            return os.path.abspath(os.path.join(here, ".."))
        return here
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

def _neuron_db_path() -> str:
    return os.path.join(_datasets_dir(), "neuron_axis.db")

def _ensure_dirs() -> None:
    try:
        os.makedirs(_datasets_dir(), exist_ok=True)
    except Exception:
        pass

def _connect_db() -> Optional[sqlite3.Connection]:
    try:
        _ensure_dirs()
        con = sqlite3.connect(_neuron_db_path(), check_same_thread=False, timeout=10.0)
        try:
            if config and hasattr(config, "apply_sqlite_pragmas"):
                config.apply_sqlite_pragmas(con)  # type: ignore[attr-defined]
            else:
                con.execute("PRAGMA journal_mode=WAL;")
                con.execute("PRAGMA synchronous=NORMAL;")
                con.execute("PRAGMA busy_timeout=10000;")
                con.execute("PRAGMA foreign_keys=ON;")
                con.execute("PRAGMA wal_autocheckpoint=1000;")
        except Exception:
            pass
        return con
    except Exception:
        return None

_DB: Optional[sqlite3.Connection] = None

_EVENT_LOCK = threading.RLock()
_EVENT_QUEUE: List[Tuple[float, str, str, float, str, str]] = []
_EVENT_LAST_FLUSH = 0.0
_EVENT_ATEXIT_REGISTERED = False


def _neuron_event_logging_enabled() -> bool:
    try:
        return bool(getattr(config, "NEURON_EVENT_LOG_ENABLED", True))
    except Exception:
        return True


def _flush_event_queue(force: bool = False) -> None:
    """Batch neuron route telemetry so chat traffic does not commit per request."""
    global _EVENT_LAST_FLUSH
    if _DB is None:
        return
    try:
        batch_size = int(getattr(config, "NEURON_EVENT_BATCH_SIZE", 12) if config else 12)
    except Exception:
        batch_size = 12
    try:
        flush_seconds = float(getattr(config, "NEURON_EVENT_FLUSH_SECONDS", 5.0) if config else 5.0)
    except Exception:
        flush_seconds = 5.0
    now = time.time()
    with _EVENT_LOCK:
        if not _EVENT_QUEUE:
            return
        if not force and len(_EVENT_QUEUE) < max(1, batch_size) and (now - _EVENT_LAST_FLUSH) < max(1.0, flush_seconds):
            return
        rows = list(_EVENT_QUEUE)
        _EVENT_QUEUE.clear()
    try:
        cur = _DB.cursor()
        cur.executemany(
            "INSERT INTO neuron_events (ts, kind, intent, confidence, source, payload) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        _DB.commit()
        _EVENT_LAST_FLUSH = now
    except Exception:
        # If the DB is temporarily locked, keep the hot path quiet and do not
        # spin-write retries.  Route answers remain unaffected.
        pass

def _shutdown_neuron_runtime() -> None:
    """Flush telemetry, checkpoint WAL, and stop the background lane safely."""
    global _DB
    try:
        _NEURON_STOP.set()
    except Exception:
        pass
    try:
        thread = globals().get("_NEURON_THREAD")
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=1.5)
    except Exception:
        pass
    try:
        _flush_event_queue(force=True)
    except Exception:
        pass
    con = _DB
    _DB = None
    if con is not None:
        try:
            con.execute("PRAGMA wal_checkpoint(PASSIVE);")
        except Exception:
            pass
        try:
            con.close()
        except Exception:
            pass


def _init_db() -> None:
    global _DB
    if _DB is not None:
        return
    _DB = _connect_db()
    if _DB is None:
        return
    try:
        cur = _DB.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS neuron_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                kind TEXT,
                intent TEXT,
                confidence REAL,
                source TEXT,
                payload TEXT
            )
            """
        )
        _DB.commit()
        global _EVENT_ATEXIT_REGISTERED
        if not _EVENT_ATEXIT_REGISTERED:
            try:
                atexit.register(_shutdown_neuron_runtime)
                _EVENT_ATEXIT_REGISTERED = True
            except Exception:
                pass
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Safety envelope
# -----------------------------------------------------------------------------
def _flag(name: str, default: bool = False) -> bool:
    try:
        return bool(getattr(config, name, default))
    except Exception:
        return default

def _is_safe_mode() -> bool:
    # SAFE_MODE is the master "no autonomous execution" gate
    return _flag("SAFE_MODE", True)

def _is_local_only() -> bool:
    return _flag("LOCAL_ONLY_MODE", False)

def _neosky_armed() -> bool:
    # Dual-key arm: NEOSKYMATRIX + DEVELOPERSMODE
    return _flag("NEOSKYMATRIX", False) and _flag("DEVELOPERSMODE", False)

def _device_profile() -> Dict[str, Any]:
    return {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "base_dir": _base_dir(),
        "core_dir": str(getattr(config, "CORE_DIR", os.path.join(_base_dir(), "core"))),
        "local_only": _is_local_only(),
        "safe_mode": _is_safe_mode(),
        "neosky_armed": _neosky_armed(),
    }

def _budget_limits() -> Dict[str, Any]:
    return {
        "max_parallel": int(getattr(config, "NEURON_MAX_PARALLEL", 4) if config else 4),
        "max_curiosity": int(getattr(config, "NEURON_MAX_CURIOSITY", 2) if config else 2),
        "max_trace_kb": int(getattr(config, "NEURON_MAX_TRACE_KB", 64) if config else 64),
        "max_links": int(getattr(config, "NEURON_MAX_LINKS", 32) if config else 32),
    }


def _core_registry_snapshot(force: bool = False) -> Dict[str, Any]:
    try:
        if config and hasattr(config, "sm_refresh_core_registry"):
            data = config.sm_refresh_core_registry(force=force)  # type: ignore[attr-defined]
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _core_module_allowed(module_name: str, capability: Optional[str] = None, import_obj: Any = None) -> bool:
    key = os.path.splitext(os.path.basename(str(module_name or "").strip()))[0]
    # Read-only answer helpers must remain available when the governance registry
    # is incomplete or temporarily out of sync. This is not an action bypass: the
    # list is limited to local reasoning/retrieval/validation helpers and requires
    # the module to be importable in-process.
    safe_local_answer_helpers = {
        "SarahMemoryAdvCU",
        "SarahMemoryLogicCalc",
        "SarahMemoryWebSYM",
        "SarahMemoryResearch",
        "SarahMemoryCompare",
        "SarahMemoryDatabase",
    }
    try:
        if config and hasattr(config, "sm_is_core_module_approved"):
            approved = bool(config.sm_is_core_module_approved(key, capability=capability))  # type: ignore[attr-defined]
            if approved:
                return True
            cap = str(capability or "").strip().lower()
            if key in safe_local_answer_helpers and import_obj is not None and cap in {"reasoning", "helper", "utility", "retrieval", "data"}:
                return True
            return False
    except Exception:
        pass
    return bool(import_obj is not None)


def _vision_helper_allowed(module_name: str, import_obj: Any) -> bool:
    """Allow read-only local vision helpers even if registry approval is incomplete.

    This is intentionally narrower than general core-module bypass:
    - only applies to vision helper modules
    - only when the module is already imported locally
    - used by the Answer/Vision helper lane, not action lanes
    """
    if _core_module_allowed(module_name, "vision", import_obj):
        return True
    try:
        return bool(import_obj is not None and str(module_name or "") in ("SarahMemorySOBJE", "SarahMemoryFacialRecognition"))
    except Exception:
        return False


def _core_governance_trace() -> Dict[str, Any]:
    try:
        if config and hasattr(config, "sm_get_core_governance_profile"):
            profile = config.sm_get_core_governance_profile()  # type: ignore[attr-defined]
            if isinstance(profile, dict):
                return {
                    "discovery_enabled": bool(profile.get("discovery_enabled", False)),
                    "registered_count": int(profile.get("registered_count", 0) or 0),
                    "quarantined_count": int(profile.get("quarantined_count", 0) or 0),
                    "ignored_count": int(profile.get("ignored_count", 0) or 0),
                }
    except Exception:
        pass
    return {
        "discovery_enabled": False,
        "registered_count": 0,
        "quarantined_count": 0,
        "ignored_count": 0,
    }


def _approved_lane_modules() -> Dict[str, bool]:
    return {
        "advcu": _core_module_allowed("SarahMemoryAdvCU", "helper", _AdvCU),
        "logiccalc": _core_module_allowed("SarahMemoryLogicCalc", "reasoning", _LogicCalc),
        "websym": _core_module_allowed("SarahMemoryWebSYM", "reasoning", _WebSYM),
        "research": _core_module_allowed("SarahMemoryResearch", "reasoning", _Research),
        "api": _core_module_allowed("SarahMemoryAPI", "helper", _SMAPI),
        "compare": _core_module_allowed("SarahMemoryCompare", "utility", _Compare),
        "canvas": _core_module_allowed("SarahMemoryCanvasStudio", "creative", _CanvasStudio),
        "filesystem": _core_module_allowed("SarahMemoryFilesystem", "action", True),
        "network": _core_module_allowed("SarahMemoryNetwork", "network", True),
        "cognitive": _core_module_allowed("SarahMemoryCognitiveServices", "diagnostics", _Cog),
    }


def _trace_primary_lane(trace: Dict[str, Any], lane: str, owner: str) -> None:
    try:
        trace["primary_lane"] = str(lane or "general")
        trace["primary_owner"] = str(owner or "neuron")
    except Exception:
        pass


def _logiccalc_neuron_axis_guard(
    requested_lane: str,
    *,
    current_lane: str = "answer",
    risk_hint: str = "low",
    user_present: bool = True,
    user_consented: bool = False,
    route_confidence: float = 0.0,
) -> Dict[str, Any]:
    """WAVE7 Neuron Axis guard using SarahMemoryLogicCalc advisory math.

    This function does not replace CognitiveServices, SMGET, SafetyPolicies, or
    user authority. It adds a deterministic, auditable lane-collapse packet so
    Neuron does not silently jump from answer/reasoning into file/network/device
    or action lanes.
    """
    lane = str(requested_lane or "answer").strip().lower()
    risk = str(risk_hint or "low").strip().lower()
    try:
        if _LogicCalc and _core_module_allowed("SarahMemoryLogicCalc", "reasoning", _LogicCalc):
            engine = _LogicCalc() if isinstance(_LogicCalc, type) else _LogicCalc
            gate = getattr(engine, "neuron_axis_gate", None)
            if callable(gate):
                route_validity = max(0.0, min(1.0, float(route_confidence or 0.0)))
                if lane == "answer":
                    values = (0.93, max(0.91, route_validity), 0.88, 0.04)
                elif lane == "system":
                    values = (0.90, max(0.84, route_validity), 0.90 if user_present else 0.72, 0.10)
                elif lane == "network":
                    values = (0.86, max(0.72, route_validity), 0.92 if user_consented else 0.74, 0.28 if risk != "high" else 0.48)
                elif lane in {"action", "creative"}:
                    # Neuron authorizes bounded routing/ticketing here, not final
                    # execution. OperatorCore/SMGET still owns consent and action.
                    governance = 0.94 if user_consented else (0.82 if user_present else 0.62)
                    values = (0.90, max(0.78, route_validity), governance, 0.18 if risk != "high" else 0.45)
                else:
                    values = (0.76, max(0.50, route_validity), 0.55, 0.50)
                try:
                    out = gate(
                        current_lane_confidence=values[0],
                        requested_lane_validity=values[1],
                        governance_score=values[2],
                        risk_score=values[3],
                        threshold=0.50,
                    )
                except TypeError:
                    # Backward compatibility for any older LogicCalc build still
                    # using the former keyword names. Do not bypass the gate.
                    out = gate(
                        current_lane_confidence=values[0],
                        requested_lane_validity=values[1],
                        governance_modifier=values[2],
                        risk_penalty=values[3],
                        threshold=0.50,
                    )
                if isinstance(out, dict):
                    out.setdefault("requested_lane", lane)
                    out.setdefault("current_lane", current_lane)
                    out.setdefault("risk_hint", risk)
                    out.setdefault("authority", "advisory_math_only")
                    out.setdefault("user_present", bool(user_present))
                    out.setdefault("user_consented", bool(user_consented))
                    out.setdefault("route_confidence", float(route_confidence or 0.0))
                    return out
    except Exception as exc:
        risky = lane in {"action", "network", "system"}
        return {"ok": False, "decision": 0 if risky else 1, "verdict": "DEFER_NO_LOGICCALC" if risky else "READ_ONLY_FALLBACK_NO_LOGICCALC", "error": str(exc), "requested_lane": lane, "execution_authority": False}
    risky = lane in {"action", "network", "system"}
    return {"ok": False, "decision": 0 if risky else 1, "verdict": "DEFER_NO_LOGICCALC" if risky else "READ_ONLY_FALLBACK_NO_LOGICCALC", "error": "LogicCalc unavailable", "requested_lane": lane, "execution_authority": False}


# -----------------------------------------------------------------------------
# Cognitive graph core (lightweight, local)
# -----------------------------------------------------------------------------
@dataclass
class GraphEdge:
    src: str
    dst: str
    rel: str
    w: float = 0.5
    meta: Dict[str, Any] = field(default_factory=dict)

class MeaningGraph:
    def __init__(self) -> None:
        self.edges: List[GraphEdge] = []

    def link(self, src: str, dst: str, rel: str, w: float = 0.5, meta: Optional[Dict[str, Any]] = None) -> None:
        self.edges.append(GraphEdge(src=src, dst=dst, rel=rel, w=float(w), meta=meta or {}))

_GRAPH = MeaningGraph()


# -----------------------------------------------------------------------------
# Structured I/O
# -----------------------------------------------------------------------------
@dataclass
class NeuronInput:
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NeuronResult:
    ok: bool
    reply: str
    confidence: float = 0.5
    intent: str = "general"
    source: str = "neuron"
    artifacts: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "reply": str(self.reply),
            "confidence": float(self.confidence),
            "intent": str(self.intent),
            "source": str(self.source),
            "artifacts": self.artifacts,
            "trace": self.trace,
            "actions": list(self.actions or []),
        }


# -----------------------------------------------------------------------------
# Thought agents (parallel-calibration layer)
# -----------------------------------------------------------------------------
class ThoughtAgent:
    name = "agent"

    def evaluate(self, inp: NeuronInput, draft: str) -> Tuple[str, float, Dict[str, Any]]:
        return draft, 0.0, {}

class AnalystAgent(ThoughtAgent):
    name = "Analyst"

    def evaluate(self, inp: NeuronInput, draft: str) -> Tuple[str, float, Dict[str, Any]]:
        if not draft or len(draft.strip()) < 10:
            return "I need a bit more detail to answer precisely. What constraints matter most?", -0.05, {"reason": "too_short"}
        return draft, 0.01, {"reason": "ok"}

class SkepticAgent(ThoughtAgent):
    name = "Skeptic"

    def evaluate(self, inp: NeuronInput, draft: str) -> Tuple[str, float, Dict[str, Any]]:
        t = (draft or "").lower()
        if any(k in t for k in ("not sure", "can't", "unknown", "maybe", "might be")):
            return draft, -0.08, {"reason": "uncertainty_markers"}
        return draft, 0.0, {"reason": "ok"}

class OptimizerAgent(ThoughtAgent):
    name = "Optimizer"

    def evaluate(self, inp: NeuronInput, draft: str) -> Tuple[str, float, Dict[str, Any]]:
        if len(draft) > 2200:
            return draft[:2200].rstrip() + "\n\n[Truncated for performance.]", -0.03, {"reason": "truncated"}
        return draft, 0.0, {"reason": "ok"}

class EngineerAgent(ThoughtAgent):
    name = "Engineer"

    def evaluate(self, inp: NeuronInput, draft: str) -> Tuple[str, float, Dict[str, Any]]:
        if _is_safe_mode() and ("autonomous" in (draft or "").lower() or "self-evolve" in (draft or "").lower()):
            return "SAFE_MODE: Autonomous operations are gated. Provide explicit user authorization.", -0.25, {"policy": "safe_mode_gate"}
        return draft, 0.0, {"policy": "ok"}

class GovernorAgent(ThoughtAgent):
    name = "Governor"

    def evaluate(self, inp: NeuronInput, draft: str) -> Tuple[str, float, Dict[str, Any]]:
        return draft, 0.0, {"governance": "ok"}


# -----------------------------------------------------------------------------
# Intent heuristics (fallback if AdvCU not available / not confident)
# -----------------------------------------------------------------------------
def _classify_intent(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return "empty"

    def _has_phrase(*phrases: str) -> bool:
        try:
            for phrase in phrases:
                p = str(phrase or "").strip().lower()
                if not p:
                    continue
                if re.search(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])", t):
                    return True
        except Exception:
            pass
        return False

    def _looks_like_story_math() -> bool:
        has_count_question = _has_phrase("how many", "what is the total", "what's the total", "how much in total", "altogether", "combined", "in all", "total")
        has_quantities = bool(re.search(r"\b\d+\b", t)) or _has_phrase(
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
            "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",
        )
        has_countable_noun = _has_phrase(
            "apple", "apples", "orange", "oranges", "banana", "bananas", "item", "items",
            "bottle", "bottles", "book", "books", "key", "keys", "coin", "coins",
        )
        return bool(has_count_question and has_quantities and has_countable_noun)

    if _has_phrase("calculate", "solve", "convert", "unit", "derivative", "integral", "matrix", "vector", "sqrt", "square root", "plus", "minus", "times", "multiplied by", "divide", "divided by", "sum of"):
        return "math"
    if re.search(r"\d", t) and any(op in t for op in ("+", "-", "*", "/")):
        return "math"
    if re.search(r"\b[a-z]\s*=\s*[-+]?\d+(?:\.\d+)?", t):
        return "math"
    if _looks_like_story_math():
        return "math"

    if _has_phrase("diagnos", "self-test", "self test", "health check"):
        return "diagnostics"
    if _has_phrase("gpu", "vram", "cuda", "disk space", "free disk", "free space", "storage", "drive space", "cpu", "ram", "memory usage", "system stats", "system status", "hardware stats"):
        return "device_query"
    if _has_phrase("chem", "molar", "stoichi", "compound", "element", "reaction", "acid", "base") or re.search(r"\bph\b", t):
        return "chemistry"
    if _has_phrase("optimize", "speed up", "performance", "refactor", "bug", "error", "traceback"):
        return "engineering"
    if _has_phrase(
        "who are you", "what is your name", "your name",
        "who made you", "who created you", "who built you",
        "who designed you", "who engineered you",
        "creator", "brian", "softdev0", "what version are you",
        "your version", "sarahmemory version", "server version"
    ):
        return "identity"
    if _has_phrase("research", "look up", "browse", "latest", "verify", "sources", "citation"):
        return "research"
    if _has_phrase("generate", "create", "make", "draw", "song", "music", "video", "avatar", "image"):
        return "creative"
    return "general"


def _normalize_terminal_ai_directive(text: str) -> Dict[str, Any]:
    """Normalize terminal AI directives before cognitive routing.

    The React terminal strips /ai before POST /api/chat, but CLI/backend callers
    may still pass the directive through. Keeping this normalization inside
    Neuron preserves the governed AI-query workflow even when ingress differs.
    """
    raw = str(text or "").strip()
    m = re.match(r"^/(ai|task)\s+(.+)$", raw, flags=re.IGNORECASE)
    if not m:
        return {"changed": False, "directive": "", "raw": raw, "text": raw}
    routed_text = str(m.group(2) or "").strip()
    return {
        "changed": bool(routed_text),
        "directive": str(m.group(1) or "").lower(),
        "raw": raw,
        "text": routed_text or raw,
    }


def _is_greeting_text(text: str) -> bool:
    t = re.sub(r"[^a-z0-9'\s]", " ", str(text or "").lower()).strip()
    t = re.sub(r"\s+", " ", t)
    if not t:
        return False
    greetings = {
        "hi", "hello", "hey", "yo", "hiya", "howdy", "greetings",
        "hi sarah", "hello sarah", "hey sarah",
        "good morning", "good afternoon", "good evening",
        "good morning sarah", "good afternoon sarah", "good evening sarah",
    }
    if t in greetings:
        return True
    return bool(re.match(r"^(hi|hello|hey|greetings)( sarah)?$", t))


def _try_greeting_reply(text: str, intent: str, adv: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Presentation-only greeting adapter; Neuron does not own greeting text."""
    adv_intent = ""
    try:
        adv_intent = str((adv or {}).get("intent") or "").strip().lower()
    except Exception:
        adv_intent = ""
    if str(intent or "").lower() not in {"greeting", "salutation", "chat"} and adv_intent not in {"greeting", "salutation"} and not _is_greeting_text(text):
        return None
    if not _is_greeting_text(text) and str(intent or "").lower() not in {"greeting", "salutation"}:
        return None
    try:
        import SarahMemoryPersonality as _Personality  # presentation layer compatibility
        fn = getattr(_Personality, "get_greeting_response", None)
        if callable(fn):
            out = str(fn() or "").strip()
            if out:
                return out
    except Exception:
        pass
    return None


def _websym_symbolic_query_allowed(text: str) -> bool:
    """Keep WebSYM on symbolic/math lanes, not general chat or /ai greetings."""
    q = str(text or "").strip().lower()
    if not q or q.startswith("/"):
        return False
    symbolic_words = (
        "calculate", "calc", "solve", "convert", "unit", "units",
        "derivative", "integral", "matrix", "vector", "sqrt", "square root",
        "equation", "factor", "simplify", "percentage", "percent",
    )
    if any(w in q for w in symbolic_words):
        return True
    if re.search(r"\d", q) and re.search(r"[+*/=^%]|\bminus\b|\bplus\b|\btimes\b|\bdivided by\b|\bmultiplied by\b", q):
        return True
    if re.search(r"\b[a-z]\s*=\s*[-+]?\d", q):
        return True
    return False


def _websym_reply_usable(reply: Any) -> Optional[str]:
    out = str(reply or "").strip()
    if not out:
        return None
    bad_markers = (
        "i'm sorry, i couldn't solve that problem",
        "i could not solve that problem",
        "please try rephrasing or provide more details",
        "no engine produced an answer",
    )
    low = out.lower()
    if any(marker in low for marker in bad_markers):
        return None
    return out


# -----------------------------------------------------------------------------
# Tier-0.5: AdvCU delegation (intent + command parsing)
# -----------------------------------------------------------------------------

def _advcu_analyze(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "intent": None,
        "confidence": None,
        "command": None,
        "semantic_packet": {},
        "helper_payload": {},
        "entities": {},
        "raw": {},
    }
    if not _AdvCU:
        return out
    if not _core_module_allowed("SarahMemoryAdvCU", "helper", _AdvCU):
        out["raw"]["governance"] = "advcu_not_registered"
        return out

    try:
        parse_fn = getattr(_AdvCU, "parse_command", None)
        if callable(parse_fn):
            cmd = parse_fn(text)  # type: ignore
            cmd_dict: Dict[str, Any] = {}
            semantic_packet: Dict[str, Any] = {}
            if isinstance(cmd, dict):
                cmd_dict = dict(cmd)
                semantic_packet = dict(cmd_dict.get("semantic_packet") or {})
            else:
                try:
                    if hasattr(cmd, "to_dict") and callable(getattr(cmd, "to_dict")):
                        d = cmd.to_dict()  # type: ignore[attr-defined]
                        if isinstance(d, dict):
                            cmd_dict = dict(d)
                except Exception:
                    cmd_dict = {}
                try:
                    if hasattr(cmd, "to_semantic_packet") and callable(getattr(cmd, "to_semantic_packet")):
                        sp = cmd.to_semantic_packet()  # type: ignore[attr-defined]
                        if isinstance(sp, dict):
                            semantic_packet = dict(sp)
                except Exception:
                    semantic_packet = {}

            if cmd_dict or semantic_packet:
                out["command"] = cmd_dict or semantic_packet
                out["semantic_packet"] = semantic_packet or dict((cmd_dict.get("extra") or {}).get("semantic_packet") or {})
                helper_payload = dict(
                    (cmd_dict.get("helper_payload") or {})
                    or (out["semantic_packet"].get("helper_payload") if isinstance(out["semantic_packet"], dict) else {})
                    or {}
                )
                out["helper_payload"] = helper_payload
                out["intent"] = (
                    cmd_dict.get("intent")
                    or out["semantic_packet"].get("intent")
                    or cmd_dict.get("action")
                    or cmd_dict.get("type")
                )
                out["entities"] = {
                    "subject": cmd_dict.get("subject") or out["semantic_packet"].get("subject"),
                    "query_type": cmd_dict.get("query_type") or out["semantic_packet"].get("query_type"),
                    "attributes": cmd_dict.get("attributes") or out["semantic_packet"].get("attributes") or [],
                    "keywords": cmd_dict.get("keywords") or out["semantic_packet"].get("keywords") or [],
                    "module_hints": cmd_dict.get("module_hints") or out["semantic_packet"].get("module_hints") or [],
                    "action_expectation": cmd_dict.get("action_expectation") or out["semantic_packet"].get("action_expectation"),
                    "helper_payload": helper_payload,
                }
                out["raw"]["parse_command"] = cmd_dict or out["semantic_packet"]
    except Exception:
        pass

    try:
        clf_fn = getattr(_AdvCU, "classify_intent_with_confidence", None)
        if callable(clf_fn):
            intent, conf = clf_fn(text)  # type: ignore
            if intent:
                out["raw"]["classify_intent_with_confidence"] = {"intent": intent, "confidence": conf}
                if not out["intent"]:
                    out["intent"] = intent
                try:
                    out["confidence"] = float(conf) if conf is not None else out["confidence"]
                except Exception:
                    pass
    except Exception:
        pass

    return out


# -----------------------------------------------------------------------------
# Creative lane: job ticket output (unifies Studio contract)
# -----------------------------------------------------------------------------
def _is_creative_intent(intent: str, text: str, adv: Optional[Dict[str, Any]] = None) -> bool:
    t = (text or "").lower()
    i = (intent or "").lower()
    if i in ("image", "music", "song", "lyrics_to_song", "video", "avatar", "creative"):
        return True
    try:
        cmd = (adv or {}).get("command") or {}
        act = str(cmd.get("action") or cmd.get("intent") or "").lower()
        if any(k in act for k in ("image", "music", "song", "video", "avatar", "render", "generate")):
            return True
    except Exception:
        pass
    return any(k in t for k in (
        "generate an image", "make an image", "create an image", "draw ",
        "generate a song", "make a song", "lyrics to song", "compose music",
        "generate a video", "make a video", "create a video",
        "make an avatar", "create an avatar"
    ))

def _creative_kind(intent: str, text: str, adv: Optional[Dict[str, Any]] = None) -> str:
    t = (text or "").lower()
    i = (intent or "").lower()
    if i in ("image",):
        return "image"
    if i in ("music", "song", "lyrics_to_song"):
        return "music"
    if i in ("video",):
        return "video"
    if i in ("avatar",):
        return "avatar"
    cmd = (adv or {}).get("command") or {}
    act = str(cmd.get("action") or cmd.get("intent") or "").lower()
    for k in ("image", "music", "song", "video", "avatar"):
        if k in act:
            return "music" if k == "song" else k
    if "lyrics to song" in t or "song" in t or "music" in t:
        return "music"
    if "video" in t:
        return "video"
    if "avatar" in t:
        return "avatar"
    return "image"

def _make_creative_job_ticket(prompt: str, kind: str, meta: Dict[str, Any], adv: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        if _CanvasStudio and hasattr(_CanvasStudio, "ensure_canvas_directories"):
            _CanvasStudio.ensure_canvas_directories()  # type: ignore
    except Exception:
        pass

    job_id = None
    try:
        if _CanvasStudio and hasattr(_CanvasStudio, "generate_unique_id"):
            job_id = _CanvasStudio.generate_unique_id()  # type: ignore
    except Exception:
        job_id = None

    job_id = job_id or f"job_{int(time.time()*1000)}"
    cmd = (adv or {}).get("command") or {}
    entities = (adv or {}).get("entities") or {}
    style = entities.get("style") or cmd.get("style") or meta.get("style") or ""
    size = entities.get("size") or cmd.get("size") or meta.get("size") or ""
    duration = entities.get("duration") or cmd.get("duration") or meta.get("duration") or ""
    format_hint = entities.get("format") or cmd.get("format") or meta.get("format") or ""

    return {
        "job_id": job_id,
        "kind": kind,  # image|music|video|avatar
        "prompt": prompt,
        "params": {"style": style, "size": size, "duration": duration, "format": format_hint},
        "routing": {
            "requested_by": "neuron",
            "intent": meta.get("intent"),
            "offline": bool(meta.get("offline")),
            "local_only": bool(_is_local_only()),
        },
        "ts": time.time(),
    }


# -----------------------------------------------------------------------------
# Action lane: strict local executor ticketing (Filesystem/System operations)
# Neuron ONLY emits tickets; Kernel/Integration performs execution.
# -----------------------------------------------------------------------------
def _is_action_intent(intent: str, text: str, adv: Optional[Dict[str, Any]] = None) -> bool:
    cmd = (adv or {}).get("command") or {}
    act = str(cmd.get("action") or cmd.get("intent") or cmd.get("type") or "").lower()
    t = (text or "").lower()

    if act and any(k in act for k in (
        "copy", "move", "rename", "delete",
        "backup", "restore", "scan", "quarantine",
        "file_", "filesystem", "snapshot", "capture", "zoom"
    )):
        return True

    if any(k in t for k in (
        "take a snapshot", "capture this", "capture it", "take a picture", "take a photo",
        "save it", "save this", "zoom in", "zoom on", "magnify", "focus on"
    )) and (_contains_any_phrase_token(t, _VISUAL_SUBJECT_HINTS) or _has_deictic_visual_reference(t)):
        return True

    return any(k in t for k in (
        "copy file", "move file", "rename file", "delete file",
        "create backup", "restore backup", "scan file", "scan directory",
        "quarantine", "restore from quarantine"
    ))


def _map_action_to_executor_action(adv: Optional[Dict[str, Any]] = None, text: str = "") -> str:
    cmd = (adv or {}).get("command") or {}
    act = str(cmd.get("action") or cmd.get("intent") or cmd.get("type") or "").lower()
    t = (text or "").lower()

    if any(k in act for k in ("snapshot", "capture")) or any(k in t for k in ("take a snapshot", "capture this", "capture it", "take a picture", "take a photo")):
        if any(k in t for k in ("save it", "save this", "store it", "store this", "download this")):
            return "vision_snapshot_save"
        return "vision_snapshot"
    if "zoom" in act or any(k in t for k in ("zoom in", "zoom on", "magnify", "closer look", "focus on")):
        return "vision_zoom_focus"
    if "copy" in act or "copy" in t:
        return "file_copy"
    if "move" in act or "move" in t:
        return "file_move"
    if "rename" in act or "rename" in t:
        return "file_rename"
    if "delete" in act or "delete" in t:
        return "file_delete"
    if "attribute" in act or "attributes" in act:
        return "file_attrs"
    if "incremental" in act:
        return "backup_incremental"
    if "backup" in act or "create backup" in t:
        return "backup_full"
    if "restore" in act or "restore backup" in t:
        return "backup_restore"
    if "rotate" in act:
        return "backup_rotate"
    if "scan directory" in act or "scan_dir" in act or "scan directory" in t:
        return "scan_dir"
    if "scan" in act or "scan file" in t:
        return "scan_file"
    if "quarantine" in act and "restore" in act:
        return "quarantine_restore"
    if "restore from quarantine" in t:
        return "quarantine_restore"
    return act or "unknown"


def _infer_safety_level(executor_action: str) -> str:
    a = (executor_action or "").lower()
    if a in ("scan_file", "scan_dir"):
        return "low"
    if a in ("file_copy", "file_move", "file_rename", "backup_full", "backup_incremental", "backup_rotate"):
        return "medium"
    if a in ("file_delete", "backup_restore", "file_attrs", "quarantine_restore"):
        return "high"
    return "medium"

def _make_action_ticket(text: str, meta: Dict[str, Any], adv: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cmd = (adv or {}).get("command") or {}
    entities = (adv or {}).get("entities") or {}

    executor_action = _map_action_to_executor_action(adv=adv, text=text)
    safety_level = _infer_safety_level(executor_action)

    args: Dict[str, Any] = {}
    if isinstance(cmd, dict):
        args.update(cmd.get("args") or {})
    if isinstance(entities, dict):
        for k, v in (entities or {}).items():
            if k not in args:
                args[k] = v

    if "src" in args and "source" not in args:
        args["source"] = args["src"]
    if "dst" in args and "destination" not in args:
        args["destination"] = args["dst"]
    if "path" in args and "file_path" not in args and executor_action in ("file_delete", "scan_file", "file_attrs"):
        args["file_path"] = args["path"]
    if "dir" in args and "directory" not in args and executor_action == "scan_dir":
        args["directory"] = args["dir"]

    vision_request = meta.get("vision_request") if isinstance(meta.get("vision_request"), dict) else None
    if vision_request:
        args.setdefault("vision_request", vision_request)

    requires_confirm = True
    ticket_id = f"act_{int(time.time()*1000)}"
    executor_name = "SarahMemoryVision" if str(executor_action).startswith("vision_") else "SarahMemoryFilesystem"
    return {
        "ticket_id": ticket_id,
        "action": executor_action,
        "args": args,
        "safety_level": safety_level,
        "requires_confirm": requires_confirm,
        "executor": executor_name,
        "meta": {
            "intent": meta.get("intent"),
            "session_id": meta.get("session_id") or (meta.get("meta", {}) or {}).get("session_id") if isinstance(meta.get("meta"), dict) else None,
            "user_id": meta.get("user_id") or (meta.get("meta", {}) or {}).get("user_id") if isinstance(meta.get("meta"), dict) else None,
            "offline": bool(meta.get("offline") or meta.get("LOCAL_ONLY_MODE") or meta.get("local_only")),
        },
        "ts": time.time(),
    }

# -----------------------------------------------------------------------------
# Tier-2: Evidence-backed research lane
# -----------------------------------------------------------------------------
_RESEARCH_REQUEST_CACHE: Dict[Tuple[str, bool, str], Tuple[float, Optional[Dict[str, Any]]]] = {}
_RESEARCH_REQUEST_CACHE_TTL = 30.0
_RESEARCH_REQUEST_CACHE_MAX = 64

def _try_research(text: str, *, local_only: bool = False, intent: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Call SarahMemoryResearch with correct local-first semantics.

    local_only=True means local datasets/QA/vector/local LLM are allowed;
    web research and third-party API research are not allowed.
    """
    if not _Research:
        return None
    if not _core_module_allowed("SarahMemoryResearch", "reasoning", _Research):
        return None

    cache_key = (str(text or "").strip().lower(), bool(local_only), str(intent or "question").strip().lower())
    now = time.time()
    try:
        cached = _RESEARCH_REQUEST_CACHE.get(cache_key)
        if cached and (now - float(cached[0])) <= _RESEARCH_REQUEST_CACHE_TTL:
            cached_data = cached[1]
            return dict(cached_data) if isinstance(cached_data, dict) else None
    except Exception:
        pass

    result: Optional[Dict[str, Any]] = None

    def _research_call() -> Optional[Dict[str, Any]]:
        if bool(local_only):
            fn_local = getattr(_Research, "get_local_research_data", None)
            if not callable(fn_local):
                return None
            data = fn_local(text, intent=intent or "question")  # type: ignore
            if isinstance(data, dict):
                out = dict(data)
                out.setdefault("metadata", {})
                if isinstance(out.get("metadata"), dict):
                    out["metadata"].setdefault("neuron_lane", "local_research")
                    out["metadata"].setdefault("local_only", True)
                return out
            return {"raw": data, "source": "local_raw", "confidence": 0.0}

        fn = getattr(_Research, "get_research_data", None)
        if not callable(fn):
            return None
        data = fn(text)  # type: ignore
        out = data if isinstance(data, dict) else {"raw": data}
        out = dict(out)
        out.setdefault("metadata", {})
        if isinstance(out.get("metadata"), dict):
            out["metadata"].setdefault("neuron_lane", "research")
            out["metadata"].setdefault("local_only", False)
        return out

    call = _bounded_call(
        "research_local" if bool(local_only) else "research_web",
        _research_call,
        timeout_s=_runtime_float("NEURON_RESEARCH_TIMEOUT_SECONDS", 8.0),
        default=None,
    )
    if call.get("ok") and isinstance(call.get("value"), dict):
        result = dict(call.get("value") or {})
        result.setdefault("metadata", {})
        if isinstance(result.get("metadata"), dict):
            result["metadata"]["bounded_call"] = {k: v for k, v in call.items() if k != "value"}

    try:
        if len(_RESEARCH_REQUEST_CACHE) >= _RESEARCH_REQUEST_CACHE_MAX:
            oldest = min(_RESEARCH_REQUEST_CACHE.items(), key=lambda kv: kv[1][0])[0]
            _RESEARCH_REQUEST_CACHE.pop(oldest, None)
        _RESEARCH_REQUEST_CACHE[cache_key] = (time.time(), dict(result) if isinstance(result, dict) else None)
    except Exception:
        pass
    return result


def _research_has_usable_content(research_data: Optional[Dict[str, Any]], *, min_confidence: float = 0.01) -> bool:
    """Return True only when the research dict carries usable answer text."""
    if not isinstance(research_data, dict):
        return False
    try:
        conf = float(research_data.get("confidence") or 0.0)
    except Exception:
        conf = 0.0
    text = str(
        research_data.get("data")
        or research_data.get("snippet")
        or research_data.get("summary")
        or research_data.get("answer")
        or research_data.get("result")
        or ""
    ).strip()
    if not text or conf < float(min_confidence):
        return False
    bad = (
        "sorry, i was unable to find any reliable information",
        "sorry, i couldn't find any reliable information",
        "i'm sorry, i couldn't solve that problem",
        "i’m sorry, i couldn’t solve that problem",
        "i could not solve that problem",
        "please try rephrasing or provide more details",
        "i couldn't find reliable information",
        "i could not find a vetted local cached answer",
        "local llm did not return a usable response",
        "no engine produced an answer",
        "local research failed",
        "research failed:",
        "no usable local answer",
        "no reliable information was found",
    )
    return not any(marker in text.lower() for marker in bad)


def _looks_like_general_knowledge_query(text: str, intent: str = "") -> bool:
    """Detect general answer questions that should hit local knowledge/research."""
    t = str(text or "").strip().lower()
    if not t:
        return False
    i = str(intent or "").strip().lower()
    if i in {"identity", "identity_query", "selfaware_body", "device_query", "diagnostics", "action", "creative"}:
        return False
    if i in {"research", "question", "general", "chat", "unknown", "conversation", "explanation", "history", "science", "ai", "business", "technology"}:
        return True
    if re.search(r"^\s*(who|what|when|where|why|how)\s+", t):
        return True
    if re.search(r"^\s*(tell me about|explain|define|describe|give me information on)\b", t):
        return True
    return False

def _synthesize_evidence_reply(base_reply: str, research_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    # Support both legacy research keys and ResearchResult.to_dict() keys.
    # Local research returns data/snippet; older synthesis only checked summary/answer/result.
    summ = (
        research_data.get("summary")
        or research_data.get("answer")
        or research_data.get("result")
        or research_data.get("data")
        or research_data.get("snippet")
        or ""
    )
    links = research_data.get("links") or research_data.get("sources") or []
    if isinstance(links, str):
        links = [links]

    appended: List[str] = []
    if isinstance(summ, str) and summ.strip():
        appended.append(summ.strip())
    if isinstance(links, list):
        cleaned = [str(x) for x in links if str(x).strip()][:6]
        if cleaned:
            appended.append("Sources: " + " | ".join(cleaned))

    if not appended:
        return base_reply, {"research": research_data}

    merged = (base_reply.rstrip() + "\n\n" + "\n".join(appended)).strip()
    return merged, {"research": research_data, "research_summary": appended[0] if appended else ""}


# -----------------------------------------------------------------------------
# Compare-based QA gate
# -----------------------------------------------------------------------------
def _qa_compare_gate(user_text: str, draft: str, intent: str) -> Tuple[float, Dict[str, Any]]:
    if not _Compare:
        return 0.0, {}
    if not _core_module_allowed("SarahMemoryCompare", "utility", _Compare):
        return 0.0, {"compare": {"status": "BYPASS", "reason": "compare_not_registered"}}

    fn = getattr(_Compare, "compare_reply", None)
    if not callable(fn):
        return 0.0, {"compare": {"status": "UNAVAILABLE", "reason": "compare_callable_missing"}}

    call = _bounded_call(
        "compare_qa",
        lambda: fn(user_text, draft, intent=intent),  # type: ignore[misc]
        timeout_s=_runtime_float("NEURON_COMPARE_TIMEOUT_SECONDS", 1.5),
        default={"status": "TIMEOUT", "score": 0.0},
    )
    result = call.get("value") if isinstance(call.get("value"), dict) else {"status": "ERROR", "score": 0.0}
    result = dict(result)
    result["bounded_call"] = {k: v for k, v in call.items() if k != "value"}
    if not call.get("ok"):
        return -0.02, {"compare": result}

    try:
        status = str(result.get("status") or "").upper()
        score = float(result.get("score") or result.get("similarity") or 0.0)
    except Exception:
        status = "ERROR"
        score = 0.0
    if status in ("HIT", "PASS", "OK") and score >= 0.55:
        return +0.08, {"compare": result}
    if status in ("MISS", "FAIL") or score < 0.35:
        return -0.14, {"compare": result}
    return -0.05, {"compare": result}



# -----------------------------------------------------------------------------
# Deterministic tiers
# -----------------------------------------------------------------------------
def _try_logiccalc(text: str) -> Optional[Dict[str, Any]]:
    """Attempt deterministic evaluation.

    Priority:
      1) SarahMemoryLogicCalc.route(...) using the exported singleton or class
      2) SarahMemoryLogicCalc answer/solve aliases if present
      3) Safe fallback parser for basic arithmetic + sqrt
    """
    raw = (text or "").strip()
    if not raw:
        return None

    def _logiccalc_payload_to_neuron(out: Any, engine_name: str) -> Optional[Dict[str, Any]]:
        if not isinstance(out, dict):
            return None
        if not bool(out.get("ok")):
            return None

        kind = str(out.get("canonical_type") or out.get("kind") or "").strip().lower()
        allowed_kinds = {
            "calc", "convert", "solve", "vector", "tensor", "calculus",
            "chemistry", "nuclear", "constants",
        }
        # LogicCalc intentionally has an explain fallback for general language.
        # Neuron Tier-0 must not accept that as a math/science truth answer.
        if kind not in allowed_kinds:
            return None

        value = out.get("canonical_answer")
        if value is None:
            value = out.get("value")
        if value is None:
            value = out.get("raw_answer")
        text_out = str(out.get("presentation_hint") or out.get("text") or "").strip()
        if value in (None, "") and not text_out:
            return None

        meta = out.get("meta") if isinstance(out.get("meta"), dict) else {}
        expr = meta.get("expression") or meta.get("formula") or meta.get("equation")
        return {
            "ok": True,
            "engine": engine_name,
            "expr": expr,
            "value": value,
            "text": text_out,
            "meta": {**meta, "normalized_from": raw, "intent": "calc" if kind == "calc" else kind},
            "meaning": out.get("meaning"),
            "canonical_type": kind,
            "deterministic": True,
        }

    # Primary: project engine.  Current SarahMemoryLogicCalc exports LogicCalc as
    # a singleton instance, not a callable class.  Support both contracts without
    # changing the public module name.
    if _LogicCalc and _core_module_allowed("SarahMemoryLogicCalc", "reasoning", _LogicCalc):
        try:
            engine = _LogicCalc() if isinstance(_LogicCalc, type) else _LogicCalc
            for method_name in ("route", "answer", "solve"):
                fn = getattr(engine, method_name, None)
                if callable(fn):
                    out = fn(raw)  # type: ignore[misc]
                    normalized = _logiccalc_payload_to_neuron(out, f"LogicCalc.{method_name}")
                    if normalized:
                        return normalized
        except Exception:
            pass

    # No duplicate math fallback exists in Neuron.  LogicCalc/QuantumSafe own
    # deterministic computation.  If LogicCalc cannot answer or is unavailable,
    # routing continues to another legal evidence source rather than reimplementing
    # arithmetic inside the activation organ.
    return None

# -----------------------------------------------------------------------------
# Tier-0: System / Diagnostics lane (safe reads)
# -----------------------------------------------------------------------------
def _is_public_device(meta: Dict[str, Any]) -> bool:
    dm = str((meta or {}).get("device_mode") or "").strip().lower()
    return dm.startswith("public")


def _detect_system_kind(text: str, intent: str = "") -> Optional[str]:
    """Delegate self/runtime interpretation to CognitiveSelf."""
    try:
        import SarahMemoryCognitiveSelf as _SMCognitiveSelf  # type: ignore
        fn = getattr(_SMCognitiveSelf, "classify_runtime_system_question", None)
        if callable(fn):
            out = str(fn(text, intent) or "").strip()
            return out or None
    except Exception:
        return None
    return None


def _disk_usage_summary(path: str) -> Dict[str, Any]:
    try:
        du = shutil.disk_usage(path)
        total, used, free = int(du.total), int(du.used), int(du.free)
        gb = 1024 ** 3
        return {
            "ok": True,
            "path": path,
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "total_gb": round(total / gb, 2),
            "used_gb": round(used / gb, 2),
            "free_gb": round(free / gb, 2),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "path": path}


def _gpu_stats_summary() -> Dict[str, Any]:
    # Try torch first (if available)
    try:
        import torch  # type: ignore
        out: Dict[str, Any] = {"ok": True, "backend": "torch", "cuda_available": bool(torch.cuda.is_available())}
        if torch.cuda.is_available():
            idx = 0
            try:
                idx = int(torch.cuda.current_device())
            except Exception:
                idx = 0
            out["device_index"] = idx
            try:
                out["name"] = torch.cuda.get_device_name(idx)
            except Exception:
                out["name"] = None
            try:
                props = torch.cuda.get_device_properties(idx)
                out["total_vram_bytes"] = int(getattr(props, "total_memory", 0) or 0)
                out["total_vram_gb"] = round(out["total_vram_bytes"] / (1024 ** 3), 2) if out["total_vram_bytes"] else None
            except Exception:
                pass
            try:
                free_b, total_b = torch.cuda.mem_get_info(idx)
                out["free_vram_bytes"] = int(free_b)
                out["free_vram_gb"] = round(int(free_b) / (1024 ** 3), 2)
                out["used_vram_gb"] = round((int(total_b) - int(free_b)) / (1024 ** 3), 2)
            except Exception:
                pass
        return out
    except Exception:
        pass

    # Fallback: nvidia-smi if installed
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        raw = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=3).strip()
        if not raw:
            return {"ok": False, "backend": "nvidia-smi", "error": "No output"}
        parts = [p.strip() for p in raw.split(",")]
        out = {"ok": True, "backend": "nvidia-smi", "raw": raw}
        if len(parts) >= 5:
            out.update(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "total_vram_mb": float(parts[2]),
                    "free_vram_mb": float(parts[3]),
                    "gpu_util_pct": float(parts[4]),
                }
            )
        return out
    except Exception as e:
        return {"ok": False, "backend": "nvidia-smi", "error": str(e)}


def _quick_system_stats() -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": True, "platform": platform.platform(), "python": platform.python_version()}
    # psutil is optional
    try:
        import psutil  # type: ignore

        out["cpu_percent"] = float(psutil.cpu_percent(interval=0.2))
        vm = psutil.virtual_memory()
        out["ram_total_gb"] = round(float(vm.total) / (1024 ** 3), 2)
        out["ram_available_gb"] = round(float(vm.available) / (1024 ** 3), 2)
        out["ram_percent"] = float(vm.percent)
    except Exception:
        pass
    return out


def _boot_environment_snapshot() -> Dict[str, Any]:
    """Read the unified SarahMemory body map captured during boot."""
    try:
        import SarahMemoryHi as _SMHi  # type: ignore
        fn = getattr(_SMHi, "get_boot_environment_snapshot", None)
        if callable(fn):
            snap = fn(force_refresh=False, refresh_reason="neuron_device_query")
            if isinstance(snap, dict):
                return snap
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "SarahMemoryHi unified environment snapshot unavailable"}


def _environment_body() -> Dict[str, Any]:
    snap = _boot_environment_snapshot()
    body = snap.get("body") if isinstance(snap.get("body"), dict) else {}
    return body if isinstance(body, dict) else {}


def _run_quick_diagnostics() -> Dict[str, Any]:
    # Always provide a safe, fast snapshot for the chat endpoint.
    base_dir = os.getcwd()
    try:
        import SarahMemoryGlobals as _G  # type: ignore
        base_dir = str(getattr(_G, "BASE_DIR", base_dir) or base_dir)
    except Exception:
        pass

    out: Dict[str, Any] = {
        "ok": True,
        "engine": "quick",
        "system": _quick_system_stats(),
        "gpu": _gpu_stats_summary(),
        "disk": _disk_usage_summary(base_dir),
    }

    # Optional: include slower full diagnostics if explicitly enabled.
    if str(os.environ.get("SM_NEURON_FULL_DIAGNOSTICS", "")).strip().lower() in {"1", "true", "yes"}:
        try:
            import SarahMemoryDiagnostics as _D  # type: ignore

            sysr = getattr(_D, "run_system_diagnostics", None)
            hwr = getattr(_D, "run_hardware_diagnostics", None)
            out["engine"] = "quick+SarahMemoryDiagnostics"
            if callable(sysr):
                out["full_system"] = sysr()
            if callable(hwr):
                out["full_hardware"] = hwr()
        except Exception as e:
            out["full_error"] = str(e)

    return out

def _try_websym(text: str) -> Optional[str]:
    if not _WebSYM:
        return None
    if not _core_module_allowed("SarahMemoryWebSYM", "reasoning", _WebSYM):
        return None
    if not _websym_symbolic_query_allowed(text):
        return None
    try:
        for fn_name in ("route_query", "handle_query", "process_query", "websym_query"):
            fn = getattr(_WebSYM, fn_name, None)
            if callable(fn):
                out = _websym_reply_usable(fn(text))
                if out:
                    return out
        synth = getattr(_WebSYM, "WebSemanticSynthesizer", None)
        if synth is not None:
            is_math = getattr(synth, "is_math_query", None)
            calculator = getattr(synth, "sarah_calculator", None)
            standalone_is_math = getattr(_WebSYM, "is_math_expression", None)
            q = str(text or "").strip()
            if callable(calculator) and q:
                allowed = True
                if callable(is_math):
                    allowed = bool(is_math(q))
                elif callable(standalone_is_math):
                    allowed = bool(standalone_is_math(q))
                if allowed:
                    out = _websym_reply_usable(calculator(q, original_query=q))
                    if out:
                        return out
    except Exception:
        return None
    return None

def _try_api(text: str, meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if _is_local_only():
        return None
    if not _SMAPI:
        return None
    if not _core_module_allowed("SarahMemoryAPI", "helper", _SMAPI):
        return None

    meta2 = dict(meta or {})
    helper_payload = meta2.get("sm_helper_payload")
    user_input = str(helper_payload or text or "").strip()
    if not user_input:
        return None

    try:
        if config and hasattr(config, "resolve_model"):
            rm = config.resolve_model("reasoning", text=text, meta=meta2)  # type: ignore[attr-defined]
            meta2.setdefault("sm_model_resolve", {})["reasoning"] = rm
            if isinstance(rm, dict) and rm.get("selected"):
                meta2.setdefault("preferred_model_repo", rm.get("selected"))
    except Exception:
        pass

    fn = getattr(_SMAPI, "send_to_api", None)
    if not callable(fn):
        return None

    call = _bounded_call(
        "api_generation",
        lambda: fn(user_input, **meta2),
        timeout_s=_runtime_float("NEURON_API_TIMEOUT_SECONDS", 20.0),
        default=None,
    )
    if not call.get("ok"):
        try:
            _local_arile_sentinel.report(
                "api_timeout_or_failure",
                "API helper failed or exceeded the Neuron caller deadline.",
                severity=0.55,
                circuit=call.get("circuit"),
                error=call.get("error"),
            )
        except Exception:
            pass
        return None

    resp = call.get("value")
    if isinstance(resp, str) and resp.strip():
        return resp.strip()
    if isinstance(resp, dict):
        if resp.get("reply"):
            return str(resp["reply"]).strip()
        if resp.get("data"):
            return str(resp["data"]).strip()
    return None



# -----------------------------------------------------------------------------
# Curiosity engine (safe proposals; never executes without gate)
# -----------------------------------------------------------------------------
def _curiosity_prompts(intent: str, text: str, budget: Dict[str, Any]) -> List[str]:
    max_c = int(budget.get("max_curiosity", 2))
    t = (text or "").lower()
    prompts: List[str] = []

    if intent in ("chemistry", "math", "engineering"):
        prompts.append("If you want, I can propose 2-3 safe sandbox experiments (no execution) to explore variations.")
    if "mix" in t and "element" in t:
        prompts.append("Do you want a stoichiometry sandbox plan: balance reaction → compute yields → propose constraints?")
    if "optimize" in t or "speed" in t:
        prompts.append("Do you want a performance audit plan: profile → bottleneck map → safe patch proposal?")
    if "?" in t and "constraints" not in t:
        prompts.append("What constraints matter most (accuracy, speed, offline, local-only, safety gating)?")

    return [p for p in prompts if p][:max_c]


# -----------------------------------------------------------------------------
# Event logging
# -----------------------------------------------------------------------------
def _log_event(kind: str, intent: str, confidence: float, source: str, payload: Dict[str, Any]) -> None:
    if _DB is None or not _neuron_event_logging_enabled():
        return
    try:
        # Compress route telemetry.  Full traces are expensive and can create DB
        # churn during active chat.  Keep the audit shape, but drop raw bulky data.
        compact_payload = {}
        if isinstance(payload, dict):
            compact_payload = {
                "input_preview": str(payload.get("input") or "")[:512],
                "artifacts_keys": list(payload.get("artifacts_keys") or [])[:32],
            }
            trace = payload.get("trace") if isinstance(payload.get("trace"), dict) else {}
            if trace:
                compact_payload["trace"] = {
                    "primary_lane": trace.get("primary_lane"),
                    "primary_owner": trace.get("primary_owner"),
                    "tiers_count": len(trace.get("tiers") or []) if isinstance(trace.get("tiers"), list) else 0,
                    "approved_modules": trace.get("approved_modules"),
                }
        else:
            compact_payload = {"payload_preview": str(payload)[:512]}

        s = json.dumps(compact_payload, ensure_ascii=False, default=str)
        max_kb = int(getattr(config, "NEURON_EVENT_MAX_KB", 24) if config else 24)
        if len(s) > max_kb * 1024:
            s = s[: max_kb * 1024] + "…"
        row = (time.time(), str(kind), str(intent), float(confidence), str(source), s)
        with _EVENT_LOCK:
            _EVENT_QUEUE.append(row)
        _flush_event_queue(force=False)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Helper semantic-gap payloads (compressed helper-only LLM method)
# -----------------------------------------------------------------------------
def _build_helper_payload(user_text: str, intent: str, adv: Optional[Dict[str, Any]] = None) -> str:
    try:
        adv = adv or {}
        cmd = adv.get("command") or {}
        entities = adv.get("entities") or {}
        sem = adv.get("semantic_packet") or {}
        parts: List[str] = []
        if intent:
            parts.append(f"intent={intent}")
        query_type = cmd.get("query_type") or sem.get("query_type") or entities.get("query_type") or ""
        if query_type:
            parts.append(f"query_type={query_type}")
        action = cmd.get("action") or cmd.get("intent") or sem.get("action") or ""
        if action:
            parts.append(f"action={action}")
        for key in ("subject", "topic", "object", "target"):
            val = entities.get(key) or cmd.get(key) or sem.get(key)
            if val:
                parts.append(f"{key}={val}")
        attrs = entities.get("attributes") or sem.get("attributes") or entities.get("mods") or cmd.get("attributes") or []
        if isinstance(attrs, (list, tuple)) and attrs:
            cleaned = [str(x).strip() for x in attrs if str(x).strip()]
            if cleaned:
                parts.append("attributes=" + ",".join(cleaned[:8]))
        elif isinstance(attrs, str) and attrs.strip():
            parts.append(f"attributes={attrs.strip()}")
        keywords = entities.get("keywords") or sem.get("keywords") or cmd.get("keywords") or []
        if isinstance(keywords, (list, tuple)) and keywords:
            cleaned = [str(x).strip() for x in keywords if str(x).strip()]
            if cleaned:
                parts.append("keywords=" + ",".join(cleaned[:8]))
        if not parts:
            return str(user_text or "").strip()
        return " | ".join(parts)
    except Exception:
        return str(user_text or "").strip()



# -----------------------------------------------------------------------------
# Vision helper routing (category-driven, helper-only)
# -----------------------------------------------------------------------------
_VISUAL_QUERY_TYPES = {
    "identify_color",
    "read_text",
    "detect_objects",
    "detect_faces",
    "identify_person",
    "locate_subject",
    "track_motion",
    "estimate_distance",
    "inspect_safety_zone",
    "compare_before_after",
    "assess_style_or_fit",
    "scene_summary",
    "held_object",
}

_VISUAL_STRONG_CUE_PHRASES = (
    "what color", "what colour", "color of", "colour of",
    "what does", "read this", "read the text", "text on", "say on", "ocr",
    "what do you see", "what can you see", "describe what you see", "scene summary", "summarize the scene",
    "can you see me", "do you see me", "look at me", "look at this", "look at that",
    "analyze this image", "analyse this image", "show me what you see",
    "track", "follow", "moving", "motion",
    "danger zone", "hazard zone", "safe zone",
    "how far", "how close", "too close", "distance",
    "before and after", "compare these", "what changed",
    "look fat", "look good", "look okay", "does this shirt make me",
    "who is this person", "is this", "do you recognize", "who is this", "where is my",
    "behind me", "in front of me", "left of me", "right of me", "next to me",
    "in my hand", "in my hands", "what is in my hand", "what am i holding", "what's in my hand", "what object is in my hand",
    "take a snapshot", "capture this", "take a picture", "take a photo",
    "zoom in", "zoom on", "magnify", "closer look", "focus on",
)

_VISUAL_SUBJECT_HINTS = (
    "shirt", "pants", "trousers", "jeans", "shorts", "dress", "skirt", "jacket", "coat",
    "hat", "cap", "glasses", "eyes", "eye", "face", "beard", "mouth", "nose", "hair", "hand", "hands",
    "employee", "worker", "person", "people", "woman", "man", "girl", "boy",
    "wife", "daughter", "son", "child", "mother", "father", "sister", "brother",
    "machine", "forklift", "car", "road", "controller", "keys", "key",
    "desk", "screen", "monitor", "door", "animal", "dog", "cat",
    "photo", "image", "picture", "camera", "frame", "webcam", "pc", "computer",
    "waterbottle", "water bottle", "bottle", "logo", "paper", "document", "page", "note",
)

_VISUAL_ATTRIBUTE_HINTS = (
    "color", "colour", "text", "distance", "motion", "fit", "style", "objects", "faces",
    "safety", "identity", "position", "location", "logo", "label",
)

_VISUAL_ACTION_CUE_PHRASES = (
    "take a snapshot", "capture this", "capture it", "take a picture", "take a photo",
    "save it", "save this", "store it", "store this", "download this",
    "zoom in", "zoom on", "magnify", "closer look", "focus on",
)

_VISUAL_DEICTIC_REFERENCES = (
    "this", "that", "these", "those", "here", "holding up", "in my hand", "in my hands", "holding", "on this",
)


def _contains_phrase_token(text: str, phrase: str) -> bool:
    t = str(text or "").strip().lower()
    p = str(phrase or "").strip().lower()
    if not t or not p:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])"
    try:
        return re.search(pattern, t) is not None
    except Exception:
        return p in t


def _contains_any_phrase_token(text: str, phrases: Tuple[str, ...]) -> bool:
    try:
        return any(_contains_phrase_token(text, p) for p in (phrases or ()))
    except Exception:
        return False


def _has_visual_media(meta: Optional[Dict[str, Any]]) -> bool:
    meta = meta or {}
    payload_meta = _meta_payload_block(meta)
    try:
        for key in ("frame", "current_frame", "latest_frame", "image", "image_data"):
            if meta.get(key) is not None or payload_meta.get(key) is not None:
                return True
        for key in ("images", "video"):
            if isinstance(meta.get(key), list) and meta.get(key):
                return True
            if isinstance(payload_meta.get(key), list) and payload_meta.get(key):
                return True
        files: List[Any] = []
        if isinstance(meta.get("files"), list):
            files.extend(meta.get("files") or [])
        if isinstance(payload_meta.get("files"), list):
            files.extend(payload_meta.get("files") or [])
        return any(_looks_like_image_path(x) for x in files)
    except Exception:
        return False


def _has_deictic_visual_reference(text: str) -> bool:
    return _contains_any_phrase_token(text, _VISUAL_DEICTIC_REFERENCES)


def _looks_like_person_identity_prompt(text: str) -> bool:
    t = str(text or "").strip().lower()
    if re.search(r"\bis this\s+[a-z][a-z'\-]+\s+or\s+[a-z][a-z'\-]+\b", t):
        return True
    return _contains_any_phrase_token(t, (
        "who is this person", "who is this", "do you recognize", "who am i",
        "this is my wife", "this is my daughter", "this is my son", "this is my child",
    ))
def _looks_like_spatial_subject_query(text: str) -> bool:
    t = str(text or "").strip().lower()
    if not t:
        return False
    has_relation = _contains_any_phrase_token(t, ("behind me", "in front of me", "left of me", "right of me", "next to me", "near me"))
    has_subject = _contains_any_phrase_token(t, ("wife", "daughter", "son", "child", "person", "worker", "employee", "keys", "key"))
    return bool(has_relation and has_subject)


def _looks_like_knowledge_query(text: str) -> bool:
    t = str(text or "").strip().lower()
    if not t:
        return False
    if _contains_any_phrase_token(t, _VISUAL_STRONG_CUE_PHRASES) or _contains_any_phrase_token(t, _VISUAL_SUBJECT_HINTS) or _has_deictic_visual_reference(t):
        return False
    return bool(
        re.search(r"^\s*(who|what|when|where|why|how)\s+is\b", t)
        or re.search(r"^\s*tell me about\b", t)
        or re.search(r"^\s*explain\b", t)
        or re.search(r"^\s*define\b", t)
    )


def _meta_context_packet(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        cp = (meta or {}).get("context_packet") or {}
        return cp if isinstance(cp, dict) else {}
    except Exception:
        return {}


def _meta_payload_block(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cp = _meta_context_packet(meta)
    block = cp.get("meta") if isinstance(cp, dict) else {}
    return block if isinstance(block, dict) else {}


def _looks_like_image_path(value: Any) -> bool:
    try:
        s = str(value or "").strip().lower()
        return bool(s) and any(s.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    except Exception:
        return False


def _decode_visual_frame(candidate: Any):
    if candidate is None or _np is None or _cv2 is None:
        return None
    try:
        if isinstance(candidate, _np.ndarray):
            return candidate.copy()
    except Exception:
        pass
    try:
        if isinstance(candidate, (bytes, bytearray)):
            arr = _np.frombuffer(candidate, dtype=_np.uint8)
            if arr.size:
                frame = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
    except Exception:
        pass
    try:
        if isinstance(candidate, dict):
            for key in (
                "frame", "current_frame", "latest_frame", "vision_frame",
                "bgr", "ndarray", "image", "image_data", "imageData",
                "data", "data_url", "dataUrl", "image_b64", "imageBase64",
                "base64", "bytes", "path",
            ):
                if key in candidate:
                    frame = _decode_visual_frame(candidate.get(key))
                    if frame is not None:
                        return frame
            return None
    except Exception:
        pass
    try:
        s = str(candidate or "").strip()
        if not s:
            return None
        if _looks_like_image_path(s) and os.path.exists(s):
            frame = _cv2.imread(s)
            if frame is not None:
                return frame
        if s.startswith("data:image") and "," in s and _sm_base64 is not None:
            s = s.split(",", 1)[1]
        if _sm_base64 is not None:
            blob = _sm_base64.b64decode(s, validate=False)
            arr = _np.frombuffer(blob, dtype=_np.uint8)
            if arr.size:
                frame = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
    except Exception:
        return None
    return None


def _extract_appvision_frame_from_loaded_runtime():
    """Read the live mounted appvision frame cache without creating a second module.

    Resolution order is deterministic:
      1) known loaded module names,
      2) Flask application globals that retain the mounted module,
      3) blueprint view-function global namespaces.

    The function is read-only, bounded, never imports appvision.py directly, and
    never opens camera hardware.
    """
    class _NamespaceProxy:
        __slots__ = ("_namespace", "_identity")

        def __init__(self, namespace: Dict[str, Any], identity: str) -> None:
            object.__setattr__(self, "_namespace", namespace)
            object.__setattr__(self, "_identity", identity)

        def __getattr__(self, name: str) -> Any:
            namespace = object.__getattribute__(self, "_namespace")
            if name in namespace:
                return namespace[name]
            raise AttributeError(name)

        def __repr__(self) -> str:
            return f"<_NamespaceProxy {object.__getattribute__(self, '_identity')}>"

    candidates: List[Any] = []

    def _append(candidate: Any) -> None:
        if candidate is None:
            return
        try:
            if hasattr(candidate, "get_latest_cached_frame_for_chat") or hasattr(candidate, "_FRAME_CACHE"):
                candidates.append(candidate)
        except Exception:
            pass

    for name in ("appvision", "api.server.appvision", "server.appvision"):
        try:
            _append(sys.modules.get(name))
        except Exception:
            pass

    try:
        from flask import current_app, has_app_context  # type: ignore
        if has_app_context():
            app_globals = getattr(current_app, "__dict__", {}) or {}
            for key in ("_appvision", "appvision", "vision_module"):
                _append(app_globals.get(key) if isinstance(app_globals, dict) else None)

            for endpoint, fn in list((getattr(current_app, "view_functions", {}) or {}).items()):
                namespace = getattr(fn, "__globals__", None)
                if not isinstance(namespace, dict):
                    continue
                endpoint_name = str(endpoint or "").lower()
                schema_ok = str(namespace.get("SMHUD_SCHEMA_VERSION") or "") == "SMHUD_PACKET_V1"
                cache_ok = isinstance(namespace.get("_FRAME_CACHE"), dict)
                helper_ok = callable(namespace.get("get_latest_cached_frame_for_chat"))
                if ("appvision" in endpoint_name or schema_ok) and (cache_ok or helper_ok):
                    _append(_NamespaceProxy(namespace, f"flask:{endpoint_name}"))
    except Exception:
        pass

    seen: set[int] = set()
    max_age_s = max(1, int(getattr(config, "VISION_FRAME_MAX_AGE_SECONDS", 45) if config else 45))
    for mod in candidates:
        ident = id(mod)
        if ident in seen:
            continue
        seen.add(ident)

        helper = None
        try:
            helper = getattr(mod, "get_latest_cached_frame_for_chat", None)
        except Exception:
            helper = None
        if callable(helper):
            call = _bounded_call(
                "appvision_frame_helper",
                lambda h=helper: h(max_age_s=max_age_s),
                timeout_s=0.75,
                default=None,
            )
            rec = call.get("value") if call.get("ok") else None
            if isinstance(rec, dict) and bool(rec.get("ok", True)):
                for payload_key in ("frame", "data_url", "image_b64"):
                    payload_value = rec.get(payload_key)
                    if payload_value is None:
                        continue
                    frame = _decode_visual_frame(payload_value)
                    if frame is not None:
                        return frame

        try:
            cache = getattr(mod, "_FRAME_CACHE", None)
            lock = getattr(mod, "_FRAME_LOCK", None)
            if not isinstance(cache, dict):
                continue
            if lock is not None and callable(getattr(lock, "acquire", None)):
                acquired = bool(lock.acquire(timeout=0.20))
                if not acquired:
                    continue
                try:
                    rec = dict(cache)
                finally:
                    lock.release()
            else:
                rec = dict(cache)
            if not bool(rec.get("has_frame")):
                continue
            ts_value = rec.get("image_cached_ts") or rec.get("ts")
            ts_epoch = 0.0
            try:
                ts_epoch = float(ts_value)
            except Exception:
                try:
                    from datetime import datetime as _dt
                    ts_text = str(ts_value or "").strip()
                    if ts_text.endswith("Z"):
                        ts_text = ts_text[:-1] + "+00:00"
                    ts_epoch = _dt.fromisoformat(ts_text).timestamp() if ts_text else 0.0
                except Exception:
                    ts_epoch = 0.0
            if ts_epoch and (time.time() - ts_epoch) > max_age_s:
                continue
            for payload_key in ("data_url", "image_b64", "frame"):
                payload_value = rec.get(payload_key)
                if payload_value is None:
                    continue
                frame = _decode_visual_frame(payload_value)
                if frame is not None:
                    return frame
        except Exception:
            continue
    return None


def _extract_visual_frame(meta: Optional[Dict[str, Any]]):
    meta = meta or {}
    payload_meta = _meta_payload_block(meta)
    candidates: List[Any] = []
    frame_keys = (
        "frame", "current_frame", "latest_frame", "image", "image_data", "imageData",
        "data_url", "dataUrl", "image_b64", "imageBase64", "vision_frame",
    )
    for key in frame_keys:
        if key in meta:
            candidates.append(meta.get(key))
    for key in frame_keys:
        if key in payload_meta:
            candidates.append(payload_meta.get(key))

    # Common nested blocks used by app.py and appvision.py.
    for block in (meta.get("vision_frame"), payload_meta.get("vision_frame"), meta.get("ingress_meta"), payload_meta.get("ingress_meta")):
        if isinstance(block, dict):
            for key in frame_keys:
                if key in block:
                    candidates.append(block.get(key))

    for key in ("images", "video", "files"):
        val = meta.get(key)
        if isinstance(val, list) and val:
            candidates.extend(val[:2])
        val2 = payload_meta.get(key)
        if isinstance(val2, list) and val2:
            candidates.extend(val2[:2])
    for cand in candidates:
        frame = _decode_visual_frame(cand)
        if frame is not None:
            return frame

    return _extract_appvision_frame_from_loaded_runtime()

def _infer_visual_query_type(text: str) -> str:
    t = str(text or "").strip().lower()
    if not t:
        return ""

    if _contains_any_phrase_token(t, ("what color", "what colour", "color of", "colour of")):
        return "identify_color"
    if _contains_any_phrase_token(t, ("what is in my hand", "what's in my hand", "what is in my hands", "what am i holding", "what object is in my hand", "in my hand", "in my hands")):
        return "held_object"
    if _contains_any_phrase_token(t, ("what does", "read this", "read the text", "text on", "say on", "ocr", "read what is on")):
        return "read_text"
    if _looks_like_person_identity_prompt(t):
        return "identify_person"
    if _looks_like_spatial_subject_query(t):
        return "locate_subject"
    if _contains_any_phrase_token(t, ("track", "follow", "moving", "motion")):
        return "track_motion"
    if _contains_any_phrase_token(t, ("danger zone", "hazard zone", "safe zone")):
        return "inspect_safety_zone"
    if _contains_any_phrase_token(t, ("distance", "how far", "how close", "too close", "far away")):
        return "estimate_distance"
    if _contains_any_phrase_token(t, ("look fat", "look good", "look okay", "outfit", "style", "fit", "does this shirt make me")):
        return "assess_style_or_fit"
    if _contains_any_phrase_token(t, ("what do you see", "what can you see", "describe what you see", "scene summary", "summarize the scene", "can you see me", "do you see me", "am i visible", "am i in frame", "look at me", "look at this", "look at that", "around me", "environment")):
        return "scene_summary"
    if _contains_any_phrase_token(t, ("color", "colour")) and _contains_any_phrase_token(t, _VISUAL_SUBJECT_HINTS):
        return "identify_color"
    if _contains_any_phrase_token(t, ("face", "eyes", "eye", "nose", "mouth", "beard", "glasses", "hair")):
        return "detect_faces"
    if _contains_any_phrase_token(t, ("detect", "identify", "what is in", "what objects", "what's in", "do you see my", "where are my")):
        return "detect_objects"

    has_visual_subject = _contains_any_phrase_token(t, _VISUAL_SUBJECT_HINTS)
    has_visual_attr = _contains_any_phrase_token(t, _VISUAL_ATTRIBUTE_HINTS)
    has_visual_action = _contains_any_phrase_token(t, _VISUAL_ACTION_CUE_PHRASES)
    if has_visual_subject and has_visual_attr:
        if _contains_any_phrase_token(t, ("color", "colour")):
            return "identify_color"
        if _contains_any_phrase_token(t, ("text", "read", "logo", "label")):
            return "read_text"
        if _contains_any_phrase_token(t, ("distance", "close", "far", "near")):
            return "estimate_distance"
        if _contains_any_phrase_token(t, ("fit", "style")):
            return "assess_style_or_fit"
        if _contains_any_phrase_token(t, ("identity", "recognize")):
            return "identify_person"
        return "scene_summary"
    if has_visual_action and (has_visual_subject or _has_deictic_visual_reference(t)):
        return "scene_summary"

    return ""


def _infer_visual_subject(text: str, query_type: str = "") -> str:
    t = f" {str(text or '').strip().lower()} "
    candidates = (
        "water bottle", "waterbottle", "bottle", "logo", "paper", "document", "page", "note",
        "shirt", "pants", "trousers", "jeans", "shorts", "dress", "skirt", "jacket", "coat",
        "hat", "cap", "glasses", "eyes", "eye", "face", "beard", "mouth", "nose", "hair", "hand", "hands",
        "employee", "worker", "person", "wife", "daughter", "son", "child", "mother", "father",
        "machine", "forklift", "car", "road", "controller", "keys", "key",
        "desk", "screen", "monitor", "door", "animal", "dog", "cat", "webcam", "camera", "pc", "computer",
    )
    for c in candidates:
        if f" {c} " in t:
            return c.replace(" ", "_") if " " in c else c
    if query_type in ("detect_faces", "identify_person"):
        return "person"
    if query_type == "locate_subject":
        return "subject"
    if query_type in ("scene_summary", "detect_objects"):
        return "scene"
    return ""


def _infer_visual_attributes(text: str, query_type: str = "") -> List[str]:
    t = str(text or "").strip().lower()
    attrs: List[str] = []
    if "color" in t or "colour" in t:
        attrs.append("color")
    if any(s in t for s in ("text", "read", "say on", "what does", "logo", "label")):
        attrs.append("text")
    if any(s in t for s in ("distance", "close", "far", "near", "behind", "front of")):
        attrs.append("distance")
    if "motion" in t or "moving" in t or "track" in t:
        attrs.append("motion")
    if query_type == "assess_style_or_fit":
        attrs.extend(["fit", "style"])
    if query_type == "identify_person":
        attrs.append("identity")
    if query_type == "locate_subject":
        attrs.append("position")
    out: List[str] = []
    for a in attrs:
        if a not in out:
            out.append(a)
    return out


def _infer_visual_action_expectation(text: str, query_type: str = "") -> str:
    t = str(text or "").strip().lower()
    if _contains_any_phrase_token(t, ("take a snapshot", "capture this", "capture it", "take a picture", "take a photo")):
        if _contains_any_phrase_token(t, ("save it", "save this", "store it", "store this", "download this")):
            return "capture_and_save"
        return "capture_only"
    if _contains_any_phrase_token(t, ("zoom in", "zoom on", "magnify", "closer look", "focus on")):
        return "zoom_focus"
    return "answer_only"


def _build_vision_request(text: str, meta: Dict[str, Any], adv: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    adv = adv or {}
    cmd = adv.get("command") or {}
    sem = adv.get("semantic_packet") or {}
    helper_payload = adv.get("helper_payload") or {}
    vision = {}
    if isinstance(helper_payload, dict):
        vision = dict(helper_payload.get("vision") or {})
    inferred_query_type = _infer_visual_query_type(text)
    semantic_query_type = str(vision.get("query_type") or sem.get("query_type") or cmd.get("query_type") or "").strip().lower()
    query_type = semantic_query_type if semantic_query_type in _VISUAL_QUERY_TYPES else inferred_query_type
    if not query_type and (_has_visual_media(meta) and _has_deictic_visual_reference(text)):
        query_type = "scene_summary"
    generic_subject = str(sem.get("subject") or cmd.get("subject") or "").strip().lower()
    requested_subject = vision.get("requested_subject") or (
        generic_subject if generic_subject and generic_subject not in {"color", "text", "question", "statement", "scene", "object", "read", "detect", "track"} else ""
    ) or _infer_visual_subject(text, query_type) or ""
    requested_attributes = vision.get("requested_attributes") or sem.get("attributes") or cmd.get("attributes") or []
    if isinstance(requested_attributes, str):
        requested_attributes = [requested_attributes]
    requested_attributes = [str(x).strip().lower() for x in list(requested_attributes or []) if str(x).strip()]
    valid_attribute_names = {"color", "text", "distance", "motion", "fit", "style", "objects", "faces", "count", "safety", "identity", "position"}
    if (not requested_attributes) or all(a not in valid_attribute_names for a in requested_attributes):
        requested_attributes = _infer_visual_attributes(text, query_type) or requested_attributes
    module_hints = vision.get("module_hints") or sem.get("module_hints") or cmd.get("module_hints") or []
    if isinstance(module_hints, str):
        module_hints = [module_hints]
    action_expectation = vision.get("action_expectation") or sem.get("action_expectation") or cmd.get("action_expectation") or _infer_visual_action_expectation(text, query_type)
    return {
        "text": str(text or "").strip(),
        "query_type": query_type,
        "requested_subject": requested_subject,
        "subject": requested_subject,
        "requested_attributes": [str(x).strip() for x in list(requested_attributes or []) if str(x).strip()],
        "attributes": [str(x).strip() for x in list(requested_attributes or []) if str(x).strip()],
        "action_expectation": action_expectation or "answer_only",
        "module_hints": [str(x).strip() for x in list(module_hints or []) if str(x).strip()] or ["SarahMemorySOBJE", "SarahMemoryFacialRecognition"],
        "intent": adv.get("intent") or sem.get("intent") or cmd.get("intent") or meta.get("intent") or "vision",
        "helper_payload": {
            "vision": vision,
            "semantic_packet": sem,
        },
        "sensor_context": {
            "ui": meta.get("ui") or _meta_context_packet(meta).get("ui"),
            "has_images": bool((meta.get("images") or []) or (_meta_payload_block(meta).get("images") or [])),
            "has_video": bool((meta.get("video") or []) or (_meta_payload_block(meta).get("video") or [])),
        },
    }


def _is_visual_request(intent: str, text: str, meta: Optional[Dict[str, Any]] = None, adv: Optional[Dict[str, Any]] = None) -> bool:
    adv = adv or {}
    sem = adv.get("semantic_packet") or {}
    entities = adv.get("entities") or {}
    helper_payload = adv.get("helper_payload") or {}
    vision_helper = {}
    if isinstance(helper_payload, dict) and isinstance(helper_payload.get("vision"), dict):
        vision_helper = dict(helper_payload.get("vision") or {})

    t = str(text or "").strip().lower()
    helper_query_type = str(vision_helper.get("query_type") or "").strip().lower()
    helper_subject = str(vision_helper.get("requested_subject") or vision_helper.get("subject") or "").strip().lower()
    helper_attrs = vision_helper.get("requested_attributes") or vision_helper.get("attributes") or []
    if isinstance(helper_attrs, str):
        helper_attrs = [helper_attrs]
    helper_attrs = [str(x).strip().lower() for x in list(helper_attrs or []) if str(x).strip()]

    sem_query_type = str(sem.get("query_type") or entities.get("query_type") or "").strip().lower()
    inferred_query_type = _infer_visual_query_type(t)
    candidate_query_type = ""
    for candidate in (helper_query_type, sem_query_type, inferred_query_type):
        if candidate in _VISUAL_QUERY_TYPES:
            candidate_query_type = candidate
            break
    if not candidate_query_type:
        return False

    has_visual_media = _has_visual_media(meta)
    has_visual_subject = _contains_any_phrase_token(t, _VISUAL_SUBJECT_HINTS)
    has_visual_phrase = _contains_any_phrase_token(t, _VISUAL_STRONG_CUE_PHRASES)
    has_deictic_reference = _has_deictic_visual_reference(t)
    helper_has_signal = bool(helper_subject or helper_attrs)

    if _looks_like_knowledge_query(t) and not (has_visual_media or has_visual_phrase or has_visual_subject or has_deictic_reference or helper_has_signal):
        return False

    if candidate_query_type in {"identify_person", "locate_subject"}:
        return bool(
            _looks_like_person_identity_prompt(t)
            or _looks_like_spatial_subject_query(t)
            or has_visual_phrase
            or (has_visual_media and (has_deictic_reference or has_visual_subject or helper_has_signal))
        )

    if candidate_query_type in {"scene_summary", "detect_objects", "held_object"}:
        return bool(
            has_visual_phrase
            or has_visual_subject
            or helper_has_signal
            or (has_visual_media and (has_deictic_reference or helper_has_signal))
        )

    if candidate_query_type in {"identify_color", "read_text", "detect_faces", "estimate_distance", "track_motion", "inspect_safety_zone", "compare_before_after", "assess_style_or_fit"}:
        return bool(has_visual_phrase or has_visual_subject or helper_has_signal or (has_visual_media and has_deictic_reference))

    return False
def _augment_face_findings(frame: Any, findings: Dict[str, Any]) -> Dict[str, Any]:
    if frame is None or not _FaceRec or not _vision_helper_allowed("SarahMemoryFacialRecognition", _FaceRec):
        return findings
    try:
        if hasattr(_FaceRec, "detect_faces_dnn"):
            faces = _FaceRec.detect_faces_dnn(frame)  # type: ignore[attr-defined]
            count = len(faces) if isinstance(faces, (list, tuple)) else 0
            face_block = dict(findings.get("faces") or {})
            if count and int(face_block.get("count") or 0) <= 0:
                face_block["count"] = count
                findings["faces"] = face_block
        if hasattr(_FaceRec, "get_user_fer_state"):
            fer = _FaceRec.get_user_fer_state(frame=frame)  # type: ignore[attr-defined]
            if isinstance(fer, dict):
                findings.setdefault("faces", {}).setdefault("fer", fer)
    except Exception:
        pass
    return findings


def _legacy_visual_text(text: str, request: Dict[str, Any]) -> str:
    qtype = str(request.get("query_type") or "").strip().lower()
    subj = str(request.get("requested_subject") or request.get("subject") or "").strip().lower()
    if qtype == "read_text" and subj == "shirt":
        return "what does my shirt say"
    if qtype == "identify_color" and subj:
        return f"what color is my {subj}"
    if qtype == "detect_faces":
        return "what's on my face"
    return str(text or "")


def _try_vision_lane(text: str, meta: Optional[Dict[str, Any]] = None, adv: Optional[Dict[str, Any]] = None) -> Optional[NeuronResult]:
    if not _SOBJE:
        return None
    if not _vision_helper_allowed("SarahMemorySOBJE", _SOBJE):
        return None
    if not _is_visual_request(str((meta or {}).get("intent") or ""), text, meta=meta, adv=adv):
        return None

    request = _build_vision_request(text, meta or {}, adv)
    frame = _extract_visual_frame(meta)
    worker_payload = dict(request)
    worker_payload["subject"] = request.get("requested_subject") or request.get("subject")
    worker_payload["attributes"] = list(request.get("requested_attributes") or request.get("attributes") or [])
    helper_block = dict((worker_payload.get("helper_payload") or {}).get("vision") or {})
    helper_block.setdefault("query_type", worker_payload.get("query_type"))
    helper_block.setdefault("requested_subject", worker_payload.get("subject"))
    helper_block.setdefault("requested_attributes", worker_payload.get("attributes") or [])
    helper_block.setdefault("action_expectation", worker_payload.get("action_expectation"))
    helper_block.setdefault("module_hints", worker_payload.get("module_hints") or [])
    worker_payload["helper_payload"] = {"vision": helper_block}
    legacy_mode = False
    first_error = None
    try:
        out = _SOBJE.answer_visual_question(worker_payload, frame)  # type: ignore[attr-defined]
    except Exception as e:
        first_error = str(e)
        try:
            out = _SOBJE.answer_visual_question(_legacy_visual_text(text, request), frame)  # type: ignore[attr-defined]
            legacy_mode = True
        except Exception as e2:
            return NeuronResult(
                ok=False,
                reply=f"Vision worker error: {e2}",
                confidence=0.2,
                intent=str(request.get("query_type") or "vision"),
                source="vision",
                artifacts={"vision_error": str(e2), "vision_request": request, "vision_worker_error": first_error},
                trace={"tiers": [{"tier": "vision", "engine": "SOBJE", "ok": False, "error": str(e2)}]},
            )

    reply = str(out.get("answer") or "").strip() if isinstance(out, dict) else str(out or "").strip()
    if not reply:
        reply = "I processed the visual request."
    details = dict(out.get("details") or {}) if isinstance(out, dict) else {}
    findings = dict(details.get("findings") or {})
    if legacy_mode and not findings:
        findings = {
            "vision_request": request,
            "resolved_subject": request.get("requested_subject") or request.get("subject"),
            "resolved_attributes": {a: details.get(a) for a in (request.get("requested_attributes") or []) if details.get(a) is not None},
            "confidence": float(details.get("confidence") or 0.45),
            "legacy_mode": True,
        }
    findings = _augment_face_findings(frame, findings)
    if findings:
        details["findings"] = findings

    conf = 0.55
    try:
        conf = float((findings or {}).get("confidence") or details.get("confidence") or 0.55)
    except Exception:
        conf = 0.55
    if frame is None:
        conf = min(conf, 0.35)

    artifacts = {
        "vision": {
            "request": request,
            "details": details,
            "frame_available": bool(frame is not None),
            "legacy_mode": bool(legacy_mode),
        }
    }
    if findings:
        artifacts["vision_findings"] = findings

    return NeuronResult(
        ok=True,
        reply=reply,
        confidence=max(0.25, min(0.95, conf)),
        intent=str(request.get("query_type") or "vision"),
        source="vision",
        artifacts=artifacts,
        trace={"tiers": [{"tier": "vision", "engine": "SOBJE", "ok": True, "frame": bool(frame is not None)}]},
    )


# -----------------------------------------------------------------------------
# Public router surface
# -----------------------------------------------------------------------------


def _normalize_ingress_route(meta: Optional[Dict[str, Any]], user_text: str) -> Dict[str, Any]:
    meta = dict(meta or {})
    route = dict(meta.get("ingress_route") or {})
    if not isinstance(route, dict):
        route = {}
    route_id = str(route.get("route_id") or "chat.general").strip() or "chat.general"
    domain = str(route.get("domain") or "chat").strip() or "chat"
    action = str(route.get("action") or "general_reply").strip() or "general_reply"
    target_module = str(route.get("target_module") or "SarahMemoryReply").strip() or "SarahMemoryReply"
    transport_target = str(route.get("transport_target") or "/api/chat").strip() or "/api/chat"
    confidence = route.get("confidence")
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    entities = route.get("entities") if isinstance(route.get("entities"), dict) else {}
    intent_hint = str(route.get("intent_hint") or "").strip().lower()
    if not intent_hint:
        if route_id.startswith("research.weather") or domain == "research":
            intent_hint = "research"
        elif route_id.startswith("avatar."):
            intent_hint = "creative"
        elif route_id.startswith("reminder."):
            intent_hint = "time"
        elif domain in {"drivers", "system", "documents", "email", "network", "communication"}:
            intent_hint = "action"
        else:
            intent_hint = domain.lower()
    return {"route_id": route_id, "domain": domain, "action": action, "target_module": target_module, "transport_target": transport_target, "confidence": max(0.0, min(0.99, confidence)), "entities": entities, "intent_hint": intent_hint, "normalized_text": str(route.get("normalized_text") or user_text or ""), "needs_discovery": bool(route.get("needs_discovery")), "source": str(route.get("source") or "semantic_ingress_router")}


def _datasets_db_path(name: str) -> str:
    candidates = [
        os.path.join(_datasets_dir(), name),
        os.path.join(_data_dir(), 'memory', 'datasets', name),
        os.path.join(_base_dir(), 'data', 'memory', 'datasets', name),
        os.path.join(getattr(config, 'CORE_DIR', os.path.join(_base_dir(), 'core')), name),
    ]
    best = None
    best_size = -1
    for cand in candidates:
        try:
            if os.path.exists(cand):
                sz = os.path.getsize(cand)
                if sz > best_size:
                    best = cand
                    best_size = sz
        except Exception:
            pass
    return best or candidates[0]


def _parse_version_from_text(value: str) -> str:
    text = str(value or "")
    m = re.search(r"(?:version|ue[_\- ]?)(\d+(?:\.\d+){0,2})", text, re.I)
    if m:
        return m.group(1)
    m = re.search(r"(\d+(?:\.\d+){1,2})", text)
    return m.group(1) if m else ""


def _system_index_lookup(terms: List[str], limit: int = 12) -> Dict[str, Any]:
    out = {"files": [], "registry": [], "db": _datasets_db_path("system_index.db")}
    db_path = out["db"]
    if not os.path.exists(db_path):
        return out
    clean_terms = [str(t).strip().lower() for t in (terms or []) if str(t).strip()]
    if not clean_terms:
        return out
    con = None
    try:
        con = sqlite3.connect(db_path, timeout=2.0)
        cur = con.cursor()
        file_hits = []
        reg_hits = []
        for term in clean_terms:
            like = f"%{term}%"
            try:
                cur.execute("SELECT file_path, file_type FROM file_index WHERE lower(coalesce(file_path,'')) LIKE ? LIMIT ?", (like, int(limit)))
                for row in cur.fetchall():
                    item = {"file_path": row[0], "file_type": row[1], "term": term}
                    if item not in file_hits:
                        file_hits.append(item)
            except Exception:
                pass
            try:
                cur.execute("SELECT root_key, key_path, value_name, value_data FROM registry_index WHERE lower(coalesce(value_data,'')) LIKE ? OR lower(coalesce(value_name,'')) LIKE ? OR lower(coalesce(key_path,'')) LIKE ? LIMIT ?", (like, like, like, int(limit)))
                for row in cur.fetchall():
                    item = {"root_key": row[0], "key_path": row[1], "value_name": row[2], "value_data": row[3], "term": term}
                    if item not in reg_hits:
                        reg_hits.append(item)
            except Exception:
                pass
        out["files"] = file_hits[:limit]
        out["registry"] = reg_hits[:limit]
    except Exception:
        pass
    finally:
        try:
            if con: con.close()
        except Exception: pass
    return out


def _software_db_lookup(terms: List[str], limit: int = 8) -> List[Dict[str, Any]]:
    db_path = _datasets_db_path("software.db")
    if not os.path.exists(db_path):
        return []
    clean_terms = [str(t).strip().lower() for t in (terms or []) if str(t).strip()]
    if not clean_terms:
        return []
    hits = []
    con = None
    try:
        con = sqlite3.connect(db_path, timeout=2.0)
        cur = con.cursor()
        cur.execute("PRAGMA table_info(software_apps)")
        cols = {str(r[1]).lower() for r in cur.fetchall()}
        has_name = "name" in cols
        has_app_name = "app_name" in cols
        has_path = "path" in cols
        has_platform = "platform" in cols
        has_last_used = "last_used" in cols
        has_usage_count = "usage_count" in cols
        has_version = "version" in cols
        has_category = "category" in cols
        has_is_installed = "is_installed" in cols
        if not (has_name or has_app_name or has_path):
            return []
        select_name = "coalesce(name, app_name)" if (has_name and has_app_name) else ("name" if has_name else ("app_name" if has_app_name else "''"))
        select_path = "path" if has_path else "''"
        select_platform = "platform" if has_platform else "''"
        select_last_used = "last_used" if has_last_used else "''"
        select_usage_count = "usage_count" if has_usage_count else "0"
        select_version = "version" if has_version else "''"
        select_category = "category" if has_category else "''"
        select_installed = "is_installed" if has_is_installed else "NULL"
        for term in clean_terms:
            like = f"%{term}%"
            where_parts = []
            params = []
            if has_name:
                where_parts.append("lower(coalesce(name,'')) LIKE ?")
                params.append(like)
            if has_app_name:
                where_parts.append("lower(coalesce(app_name,'')) LIKE ?")
                params.append(like)
            if has_path:
                where_parts.append("lower(coalesce(path,'')) LIKE ?")
                params.append(like)
            if has_category:
                where_parts.append("lower(coalesce(category,'')) LIKE ?")
                params.append(like)
            if not where_parts:
                continue
            sql = f"SELECT {select_name} AS resolved_name, {select_path} AS path, {select_platform} AS platform, {select_last_used} AS last_used, {select_usage_count} AS usage_count, {select_version} AS version, {select_category} AS category, {select_installed} AS is_installed FROM software_apps WHERE ({' OR '.join(where_parts)}) ORDER BY coalesce(usage_count,0) DESC, coalesce(last_used,'') DESC LIMIT ?"
            params.append(int(limit))
            try:
                cur.execute(sql, tuple(params))
                for row in cur.fetchall():
                    item = {"name": row[0], "path": row[1], "platform": row[2], "last_used": row[3], "usage_count": row[4], "version": row[5], "category": row[6], "is_installed": row[7], "term": term}
                    if item not in hits:
                        hits.append(item)
            except Exception:
                pass
    except Exception:
        return []
    finally:
        try:
            if con: con.close()
        except Exception: pass
    return hits[:limit]


def _si_lookup(candidates: List[str]) -> List[Dict[str, Any]]:
    out = []
    try:
        import SarahMemorySi as _Si  # type: ignore
    except Exception:
        return out
    for name in candidates or []:
        try:
            path = _Si.get_app_path(name)
        except Exception:
            path = None
        if path:
            out.append({"name": name, "path": path, "source": "SarahMemorySi", "version": _parse_version_from_text(path)})
    return out


def _software_research_lookup(candidates: List[str], limit: int = 12) -> List[str]:
    try:
        import SarahMemorySoftwareResearch as _SSR  # type: ignore
        items = _SSR.list_installed_software()
    except Exception:
        items = []
    if not isinstance(items, list):
        return []
    out = []
    for item in [str(x) for x in items if isinstance(x, str)]:
        if any(str(c).lower() in item.lower() for c in (candidates or [])):
            out.append(item)
    return out[:limit]


def _discover_driver_capabilities(route: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    entities = dict(route.get("entities") or {})
    requested_device = str(entities.get("device_type") or "").strip().lower()
    requested_vendor = str(entities.get("vendor") or "").strip().lower()
    out = {"requested_device": requested_device, "requested_vendor": requested_vendor, "driver_ids": [], "matches": [], "available": False}
    try:
        import appdrivers as _AppDrivers  # type: ignore
        ids = list(_AppDrivers._discover_driver_ids())
    except Exception:
        ids = []
        return out
    out["driver_ids"] = ids
    for did in ids:
        try:
            manifest = _AppDrivers._load_manifest(did) or {}
        except Exception:
            manifest = {}
        blob = json.dumps(manifest, ensure_ascii=False).lower()
        if requested_device and requested_device not in blob and requested_device not in str(did).lower():
            if not (requested_device == 'webcam' and ('camera' in blob or 'cam' in blob)):
                continue
        if requested_vendor and requested_vendor not in blob and requested_vendor not in str(did).lower():
            continue
        actions = manifest.get('actions') if isinstance(manifest.get('actions'), list) else []
        out['matches'].append({'driver_id': did, 'manifest': manifest, 'actions': actions})
    out['available'] = bool(out['matches'])
    return out


def _derive_schedule_spec(user_text: str, route: Dict[str, Any]) -> Dict[str, Any]:
    text = str(user_text or '').lower()
    if any(p in text for p in ['daily','every day','everyday']):
        return {'requested': True, 'pattern': 'daily', 'time_hint': '09:00', 'summary': 'Recurring daily task requested'}
    if 'weekly' in text or 'every week' in text:
        return {'requested': True, 'pattern': 'weekly', 'time_hint': '09:00', 'summary': 'Recurring weekly task requested'}
    if 'monthly' in text or 'every month' in text:
        return {'requested': True, 'pattern': 'monthly', 'time_hint': '09:00', 'summary': 'Recurring monthly task requested'}
    if str(route.get('route_id') or '').startswith('reminder.'):
        return {'requested': True, 'pattern': 'one_shot_or_parse', 'summary': 'Reminder or calendar action requested'}
    return {'requested': False}


def _driver_action_hint(route: Dict[str, Any], user_text: str) -> str:
    text = str(user_text or '').lower()
    entities = dict(route.get('entities') or {})
    control_name = str(entities.get('control_name') or entities.get('key_name') or '').lower()
    state = str(entities.get('requested_state') or '').lower()
    device_type = str(entities.get('device_type') or '').lower()
    if control_name in {'caps_lock', 'num_lock', 'scroll_lock'} or any(k in text for k in ('caps lock', 'num lock', 'scroll lock')):
        return 'keyboard_lock_set'
    if device_type == 'keyboard' and (entities.get('value') or any(k in text for k in ('color','light','lights','led','rgb','backlight'))):
        return 'keyboard_rgb_set'
    if state in {'on','open','enable','activate','start'}:
        return 'power_on'
    if state in {'off','disable','stop','close'}:
        return 'power_off'
    if entities.get('value') or any(k in text for k in ('color','light','lights','led','rgb')):
        return 'set_value'
    return 'generic_action'


def _unwrap_flaskish_response(obj: Any) -> Dict[str, Any]:
    try:
        if isinstance(obj, tuple) and len(obj) >= 1:
            resp = obj[0]
            status = obj[1] if len(obj) > 1 else None
            if hasattr(resp, 'get_json'):
                data = resp.get_json(silent=True) or {}
            else:
                data = resp if isinstance(resp, dict) else {'result': str(resp)}
            if status is not None and isinstance(data, dict):
                data.setdefault('status', status)
            return data if isinstance(data, dict) else {'result': data}
        if hasattr(obj, 'get_json'):
            data = obj.get_json(silent=True) or {}
            return data if isinstance(data, dict) else {'result': data}
        if isinstance(obj, dict):
            return obj
        return {'result': obj}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def _discover_runtime_capabilities(route: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    route_id = str(route.get('route_id') or '')
    entities = dict(route.get('entities') or {})
    software_terms = []
    capability_type = 'generic'
    if route_id == 'avatar.create.activate':
        capability_type = 'avatar_engine'
        engine_pref = str(entities.get('engine_preference') or '').lower()
        software_terms = [engine_pref] if engine_pref else ['unreal engine','unreal','blender']
    elif route_id == 'documents.office.write':
        capability_type = 'office_document'
        software_terms = ['microsoft word','word','winword','libreoffice writer']
    elif route_id == 'system.application.control':
        capability_type = 'application_control'
        target = str(entities.get('target_app') or '').strip().lower()
        software_terms = [target] if target else []
    elif route_id == 'email.mail.automation':
        capability_type = 'mail_automation'
        software_terms = ['outlook','thunderbird']
    elif route_id == 'drivers.device.control':
        capability_type = 'driver_control'
    clean_terms = [t for t in software_terms if t]
    system_index = _system_index_lookup(clean_terms)
    software_db = _software_db_lookup(clean_terms)
    si_hits = _si_lookup(clean_terms)
    software_research = _software_research_lookup(clean_terms)
    driver_caps = _discover_driver_capabilities(route, user_text) if route_id == 'drivers.device.control' else {}
    schedule_spec = _derive_schedule_spec(user_text, route) if route_id in {'email.mail.automation','reminder.schedule.task'} else {'requested': False}
    discovered_versions = []
    for coll in (si_hits, software_db, system_index.get('files', []), system_index.get('registry', [])):
        for item in coll:
            raw = json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
            ver = _parse_version_from_text(raw)
            if ver and ver not in discovered_versions:
                discovered_versions.append(ver)
    return {'capability_type': capability_type, 'software_terms': clean_terms, 'system_index': system_index, 'software_db': software_db, 'si_hits': si_hits, 'software_research': software_research, 'driver_capabilities': driver_caps, 'schedule_spec': schedule_spec, 'discovered_versions': discovered_versions[:8], 'has_discovery_hits': bool(si_hits or software_db or system_index.get('files') or system_index.get('registry') or software_research or driver_caps.get('matches'))}


def _make_execution_plan(route: Dict[str, Any], discovery: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    route_id = str(route.get('route_id') or 'chat.general')
    entities = dict(route.get('entities') or {})
    plan = {'route_id': route_id, 'transport_target': str(route.get('transport_target') or ''), 'target_module': str(route.get('target_module') or ''), 'requires_confirmation': True, 'steps': [], 'endpoint_calls': [], 'adapter_gap': None}
    if route_id == 'drivers.device.control':
        matches = ((discovery.get('driver_capabilities') or {}).get('matches') or [])
        if matches:
            driver_id = str(matches[0].get('driver_id') or '')
            action_id = _driver_action_hint(route, user_text)
            payload = {'payload': {'requested_action': action_id, 'entities': entities, 'user_text': user_text}}
            plan['steps'] = [f'Discover governed driver capabilities for {driver_id}.', f'Connect driver session for {driver_id} if not already active.', f'Send action {action_id} with extracted entities.']
            plan['endpoint_calls'] = [{'method': 'POST', 'path': f'/api/drivers/{driver_id}/discover', 'payload': payload['payload']}, {'method': 'POST', 'path': f'/api/drivers/{driver_id}/connect', 'payload': payload}, {'method': 'POST', 'path': f'/api/drivers/{driver_id}/actions/{action_id}', 'payload': payload}]
            plan['requires_confirmation'] = False
        else:
            plan['steps'] = ['Run governed driver discovery and manifest scan before attempting hardware control.']
            plan['adapter_gap'] = 'driver_or_mapping_missing'
    elif route_id == 'email.mail.automation':
        sched = discovery.get('schedule_spec') or {}
        plan['steps'] = ['List inbox/spam messages through the communications lane.', 'Prepare unsubscribe actions only for messages positively classified as spam or promotional.']
        plan['endpoint_calls'] = [{'method': 'POST', 'path': '/api/comm/email/list', 'payload': {'folder': entities.get('target_folder') or 'spam', 'limit': 50, 'source': 'neuron_ingress'}}]
        if sched.get('requested'):
            plan['steps'].append('Create recurring reminder/scheduler entry for ongoing cleanup.')
            plan['endpoint_calls'].append({'method': 'POST', 'path': '/api/comm/reminders/save', 'payload': {'title': 'Empty spam trash', 'body': 'Governed recurring spam-trash cleanup task requested from chat ingress.', 'status': 'active', 'source': 'neuron_ingress', 'extra': {'pattern': sched.get('pattern'), 'time_hint': sched.get('time_hint')}}})
    elif route_id == 'documents.office.write':
        plan['steps'] = ['Discover a word-processing runtime from indexed/system software data.', 'Generate requested document content through the writing lane.', 'Open the document in the discovered software under user authority.']
        if discovery.get('si_hits') or discovery.get('software_db') or discovery.get('system_index', {}).get('registry'):
            plan['requires_confirmation'] = False
        else:
            plan['adapter_gap'] = 'software_runtime_not_confirmed'
    elif route_id == 'avatar.create.activate':
        plan['steps'] = ['Discover the preferred 3D engine/runtime from indexed system knowledge.', 'Create or extend the governed avatar adapter if the engine capability is incomplete.', 'Publish the avatar into the Avatar Panel without requiring a frontend rebuild.']
        if not discovery.get('has_discovery_hits'):
            plan['adapter_gap'] = 'avatar_engine_adapter_missing_or_unconfirmed'
    elif route_id == 'system.application.control':
        target_app = str(entities.get('target_app') or 'requested application')
        requested_state = str(entities.get('requested_state') or 'open')
        plan['steps'] = [f'Resolve the runtime path for {target_app} from indexed software sources.', f'Perform governed application state action: {requested_state}.']
        if discovery.get('si_hits') or discovery.get('software_db') or discovery.get('system_index', {}).get('registry'):
            plan['requires_confirmation'] = False
        else:
            plan['adapter_gap'] = 'app_runtime_not_confirmed'
    return plan




def _generic_set_lock_key_state(key_name: str, requested_state: str) -> Dict[str, Any]:
    key_name = str(key_name or '').strip().lower()
    requested_state = str(requested_state or 'on').strip().lower()
    vk_map = {'caps_lock': 0x14, 'num_lock': 0x90, 'scroll_lock': 0x91}
    if key_name not in vk_map:
        return {'ok': False, 'error': 'unsupported_lock_key', 'key_name': key_name}
    if os.name != 'nt':
        return {'ok': False, 'error': 'unsupported_os', 'os': os.name, 'key_name': key_name}
    try:
        import ctypes, time as _time
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        desired_on = requested_state != 'off'
        vk = vk_map[key_name]
        KEYEVENTF_EXTENDEDKEY = 0x0001
        KEYEVENTF_KEYUP = 0x0002
        changed = False
        for _ in range(4):
            current_on = bool(user32.GetKeyState(vk) & 1)
            if current_on == desired_on:
                return {'ok': True, 'key_name': key_name, 'requested_state': requested_state, 'final_state': 'on' if current_on else 'off', 'changed': changed}
            user32.keybd_event(vk, 0x45, KEYEVENTF_EXTENDEDKEY, 0)
            user32.keybd_event(vk, 0x45, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
            changed = True
            _time.sleep(0.05)
        final_on = bool(user32.GetKeyState(vk) & 1)
        return {'ok': final_on == desired_on, 'key_name': key_name, 'requested_state': requested_state, 'final_state': 'on' if final_on else 'off', 'changed': changed}
    except Exception as e:
        return {'ok': False, 'error': str(e), 'key_name': key_name, 'requested_state': requested_state}


def _generic_keyboard_rgb_set(color_value: str) -> Dict[str, Any]:
    try:
        import shutil, subprocess
        op = shutil.which('openrgb') or shutil.which('OpenRGB')
        if not op:
            return {'ok': False, 'error': 'openrgb_not_found'}
        color = str(color_value or '').strip().lower() or 'white'
        color_map = {'red':'FF0000','green':'00FF00','blue':'0000FF','purple':'800080','yellow':'FFFF00','white':'FFFFFF','orange':'FFA500','pink':'FFC0CB'}
        cmd = [op, '--mode', 'static']
        if color in color_map:
            cmd.extend(['--color', color_map[color]])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        return {'ok': proc.returncode == 0, 'color': color, 'stdout': proc.stdout[-500:], 'stderr': proc.stderr[-500:], 'returncode': proc.returncode}
    except Exception as e:
        return {'ok': False, 'error': str(e), 'color': str(color_value or '')}




def _canonical_app_name(app_name: str) -> str:
    aliases = {
        'microsoft word': 'winword', 'ms word': 'winword', 'word': 'winword', 'winword': 'winword',
        'microsoft paint': 'mspaint', 'paint': 'mspaint', 'mspaint': 'mspaint',
        'microsoft excel': 'excel', 'ms excel': 'excel', 'excel': 'excel',
        'microsoft powerpoint': 'powerpnt', 'ms powerpoint': 'powerpnt', 'powerpoint': 'powerpnt', 'powerpnt': 'powerpnt',
        'calculator': 'calc', 'calc': 'calc', 'outlook': 'outlook', 'notepad': 'notepad',
        'visual studio code': 'code', 'vs code': 'code', 'vscode': 'code', 'code': 'code',
        'dreamweaver': 'dreamweaver', 'edge': 'msedge', 'microsoft edge': 'msedge', 'msedge': 'msedge',
        'chrome': 'chrome', 'google chrome': 'chrome', 'firefox': 'firefox', 'brave': 'brave', 'opera': 'opera',
        'explorer': 'explorer', 'file explorer': 'explorer',
    }
    app = str(app_name or '').strip().lower().replace('.exe', '')
    return aliases.get(app, app)


def _derive_surface_task(canonical_app: str, entities: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    task = dict(entities.get('surface_task') or {}) if isinstance(entities.get('surface_task'), dict) else {}
    try:
        import SarahMemoryPreTokenAnalyzer as _PTA  # type: ignore
        fn = getattr(_PTA, 'extract_surface_task', None)
        if callable(fn):
            data = fn(user_text, preferred_app=canonical_app or entities.get('target_app') or entities.get('target_app_exec'))
            if isinstance(data, dict):
                task.update({k: v for k, v in data.items() if v not in (None, '', [], {})})
        elif hasattr(_PTA, 'analyze_text'):
            analysis = _PTA.analyze_text(user_text, context_packet={})  # type: ignore[attr-defined]
            if isinstance(analysis, dict) and isinstance(analysis.get('surface_task'), dict):
                task.update({k: v for k, v in dict(analysis.get('surface_task') or {}).items() if v not in (None, '', [], {})})
    except Exception:
        pass
    for key in ('target_app', 'target_app_exec', 'requested_app', 'requested_app_exec'):
        if entities.get(key) and not task.get(key):
            task[key] = entities.get(key)
    for key in ('topic', 'title', 'document_text', 'draw_subject', 'document_name', 'pages', 'template_kind', 'search_query', 'target_url', 'headers'):
        if entities.get(key) and not task.get(key):
            task[key] = entities.get(key)
    follow = str(entities.get('followup_action') or '').strip().lower()
    if follow and not task.get('task_kind'):
        task['task_kind'] = follow
    app = _canonical_app_name(task.get('requested_app_exec') or task.get('requested_app') or canonical_app or entities.get('target_app_exec') or entities.get('target_app'))
    if app:
        task['requested_app'] = app
        task['requested_app_exec'] = app
    if app == 'winword' and not task.get('task_kind'):
        task['task_kind'] = 'document_write'
    return task


def _compass_packet_for_execution(user_text: str, plan_state: Dict[str, Any], meta: Optional[Dict[str, Any]] = None, proposed_action: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        import SarahMemoryCognitiveCompass as _Compass  # type: ignore
        fn = getattr(_Compass, 'get_compass_packet', None)
        if callable(fn):
            return fn(user_text, caller_context=dict(meta or {}), plan_state=plan_state, proposed_action=proposed_action or {})
    except Exception:
        pass
    return {}



def _smget_execution_mode(meta: Optional[Dict[str, Any]] = None) -> str:
    meta = dict(meta or {})
    governor = meta.get('governor') if isinstance(meta.get('governor'), dict) else {}
    if bool(meta.get('safe_mode') or governor.get('require_user')):
        return 'draft'
    if not bool(meta.get('user_present', True)) or not bool(meta.get('user_consented', False)):
        return 'draft'
    return 'apply'


def _smget_open_surface(canonical: str, route_id: str, entities: Dict[str, Any], user_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if _OperatorCore is None:
        return {'ok': False, 'error': 'operator_core_unavailable'}
    try:
        execution_mode = _smget_execution_mode(meta)
        proposed_action = {'route_id': route_id, 'action': 'open', 'action_type': 'open_app', 'target': canonical, 'entities': dict(entities or {})}
        op_meta = {
            'session_id': str((meta or {}).get('session_id') or ''),
            'surface': 'neuron_ingress',
            'source_surface': 'neuron_ingress',
            'execution_mode': execution_mode,
            'user_present': bool((meta or {}).get('user_present', True)),
            'user_consented': bool((meta or {}).get('user_consented', False)),
        }
        fn = getattr(_OperatorCore, 'process_action_request', None)
        if not callable(fn):
            return {'ok': False, 'error': 'operator_core_callable_missing'}
        out = fn(f'open {canonical}', origin='neuron_ingress', meta=op_meta, proposed_action=proposed_action, execution_mode=execution_mode)
        return out if isinstance(out, dict) else {'ok': False, 'error': 'invalid_operator_packet'}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def _execute_ingress_plan(route: Dict[str, Any], discovery: Dict[str, Any], user_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = dict(meta or {})
    route_id = str(route.get('route_id') or '')
    entities = dict(route.get('entities') or {})
    result = {'attempted': False, 'executed': False, 'mode': 'plan_only', 'details': {}}
    action_routes = {'system.application.control', 'documents.office.write', 'email.mail.automation', 'drivers.device.control', 'avatar.create.activate'}
    if route_id in action_routes and _smget_execution_mode(meta) != 'apply':
        reason = 'safe_mode_gate' if bool(meta.get('safe_mode')) else 'explicit_user_consent_required'
        result['attempted'] = True
        result['mode'] = 'governed_draft'
        result['details'] = {
            'ok': False,
            'reason': reason,
            'requires_confirmation': True,
            'user_present': bool(meta.get('user_present', True)),
            'user_consented': bool(meta.get('user_consented', False)),
            'safe_mode': bool(meta.get('safe_mode', False)),
        }
        return result
    try:
        if route_id in {'system.application.control', 'documents.office.write'}:
            preferred = entities.get('target_app_exec') or entities.get('target_app') or ('winword' if route_id == 'documents.office.write' else '')
            task = _derive_surface_task(_canonical_app_name(str(preferred or '')), entities, user_text)
            canonical = _canonical_app_name(task.get('requested_app_exec') or task.get('requested_app') or preferred)
            requested_state = str(entities.get('requested_state') or 'open').strip().lower()
            explicit_followup = bool(entities.get('followup_action') or entities.get('task_kind') or entities.get('surface_task'))
            if route_id == 'system.application.control' and requested_state == 'open' and not explicit_followup:
                task['task_kind'] = ''
            if canonical:
                operator_packet = {}
                ok = False
                plan_state = {'surface_open': False, 'launched': False, 'focused': False}
                if requested_state == 'open':
                    operator_packet = _smget_open_surface(canonical, route_id, entities, user_text, meta)
                    result = dict(operator_packet.get('result') or {}) if isinstance(operator_packet, dict) else {}
                    execution = dict(result.get('execution_result') or {}) if isinstance(result, dict) else {}
                    verification = dict(result.get('verification_result') or {}) if isinstance(result, dict) else {}
                    info = dict(verification.get('info') or {}) if isinstance(verification, dict) else {}
                    executed = bool(execution.get('executed'))
                    verified = bool(verification.get('ok')) and bool(result.get('success'))
                    ok = bool(operator_packet.get('ok')) and executed and verified
                    plan_state.update({'surface_open': bool(ok), 'launched': bool(executed), 'focused': bool(info.get('focused') or info.get('is_focused')), 'surface_focused': bool(info.get('focused') or info.get('is_focused')), 'result_verified': bool(verified)})
                else:
                    operator_packet = {'ok': False, 'reason': 'unsupported_requested_state'}

                followup = {'ok': True, 'skipped': True, 'reason': 'no_followup', 'task_kind': task.get('task_kind')}
                if ok and requested_state == 'open' and task.get('task_kind'):
                    try:
                        import SarahMemoryAiFunctions as _AF  # type: ignore
                        fn = getattr(_AF, 'execute_surface_task', None)
                        if callable(fn):
                            followup = fn(canonical, task, user_text=user_text)
                        else:
                            followup = {'ok': False, 'task_kind': task.get('task_kind'), 'reason': 'surface_executor_unavailable'}
                    except Exception as e:
                        followup = {'ok': False, 'task_kind': task.get('task_kind'), 'error': str(e)}
                    if isinstance(followup, dict):
                        plan_state.update({k: v for k, v in followup.items() if isinstance(v, (bool, str, int, float, dict, list))})
                compass = _compass_packet_for_execution(user_text, plan_state=plan_state, meta=meta, proposed_action={'route_id': route_id, 'entities': entities, 'task': task, 'operator_packet': operator_packet})
                executed = bool(ok and (not task.get('task_kind') or (isinstance(followup, dict) and bool(followup.get('ok')) and bool((compass or {}).get('reply_allowed', True)))))
                return {'attempted': True, 'executed': executed, 'mode': 'smget_dispatch', 'details': {'ok': ok, 'target_app': canonical, 'requested_state': requested_state, 'task': task, 'followup': followup, 'compass': compass, 'operator_packet': operator_packet}}
        if route_id == 'email.mail.automation':
            import appcomm as _AppComm  # type: ignore
            folder = str(entities.get('target_folder') or 'spam')
            listed = _unwrap_flaskish_response(_AppComm._email_poll({'folder': folder, 'limit': 50, 'source': 'neuron_ingress'})) if hasattr(_AppComm, '_email_poll') else {'ok': False, 'error': 'email_poll_unavailable'}
            details = {'listed': listed}
            executed = bool(listed.get('ok'))
            sched = discovery.get('schedule_spec') or {}
            if sched.get('requested') and hasattr(_AppComm, '_reminder_upsert'):
                reminder = _AppComm._reminder_upsert({'title': 'Empty spam trash', 'body': 'Governed recurring spam-trash cleanup task requested from chat ingress.', 'status': 'active', 'source': 'neuron_ingress', 'extra': {'pattern': sched.get('pattern'), 'time_hint': sched.get('time_hint')}})
                details['scheduler'] = reminder if isinstance(reminder, dict) else {'result': reminder}
                executed = executed or bool(details['scheduler'].get('ok'))
            return {'attempted': True, 'executed': executed, 'mode': 'direct_internal_dispatch', 'details': details}
        if route_id == 'drivers.device.control':
            action_id = _driver_action_hint(route, user_text)
            control_name = str(entities.get('control_name') or entities.get('key_name') or '').lower()
            if action_id == 'keyboard_lock_set' and control_name in {'caps_lock', 'num_lock', 'scroll_lock'}:
                action_res = _generic_set_lock_key_state(control_name, str(entities.get('requested_state') or 'on'))
                return {'attempted': True, 'executed': bool(action_res.get('ok')), 'mode': 'generic_local_control', 'details': {'action': action_res, 'action_id': action_id}}
            if action_id == 'keyboard_rgb_set':
                action_res = _generic_keyboard_rgb_set(str(entities.get('value') or 'white'))
                if bool(action_res.get('ok')):
                    return {'attempted': True, 'executed': True, 'mode': 'generic_local_control', 'details': {'action': action_res, 'action_id': action_id}}
            import appdrivers as _AppDrivers  # type: ignore
            matches = ((discovery.get('driver_capabilities') or {}).get('matches') or [])
            if matches and hasattr(_AppDrivers, '_driver_discover') and hasattr(_AppDrivers, '_driver_connect'):
                driver_id = str(matches[0].get('driver_id') or '')
                discovered = _unwrap_flaskish_response(_AppDrivers._driver_discover(driver_id, payload={'source': 'neuron_ingress', 'entities': entities}))
                connected = _unwrap_flaskish_response(_AppDrivers._driver_connect(driver_id, cfg={}, connect_payload={'source': 'neuron_ingress', 'entities': entities}))
                action_res = {'ok': False, 'skipped': True}
                try:
                    mod, err = _AppDrivers._load_driver_module(driver_id)
                    if not err and mod is not None:
                        context = _AppDrivers._build_driver_context(driver_id, instance_id=(_AppDrivers._session_get(driver_id) or {}).get('instance_id'), extra={'action_id': action_id})
                        payload = {'requested_action': action_id, 'entities': entities, 'user_text': user_text}
                        if hasattr(mod, 'driver_action'):
                            action_res = _unwrap_flaskish_response(mod.driver_action(action_id=action_id, context=context, payload=payload))
                        elif hasattr(mod, f'action_{action_id}'):
                            action_res = _unwrap_flaskish_response(getattr(mod, f'action_{action_id}')(context=context, payload=payload))
                except Exception as e:
                    action_res = {'ok': False, 'error': str(e)}
                executed = bool(discovered.get('ok')) or bool(connected.get('ok')) or bool(action_res.get('ok'))
                return {'attempted': True, 'executed': executed, 'mode': 'direct_internal_dispatch', 'details': {'driver_id': driver_id, 'discover': discovered, 'connect': connected, 'action': action_res, 'action_id': action_id}}
            return {'attempted': True, 'executed': False, 'mode': 'direct_internal_dispatch', 'details': {'ok': False, 'error': 'driver_unavailable', 'action_id': action_id, 'entities': entities}}
    except Exception as e:
        return {'attempted': True, 'executed': False, 'mode': 'direct_internal_dispatch', 'details': {'ok': False, 'error': str(e)}}
    return result



def _ingress_execution_ticket(route: Dict[str, Any], trace: Dict[str, Any], user_text: str = '', meta: Optional[Dict[str, Any]] = None) -> NeuronResult:
    route_id = str(route.get('route_id') or 'chat.general')
    target_module = str(route.get('target_module') or '')
    entities = dict(route.get('entities') or {})
    discovery = _discover_runtime_capabilities(route, user_text)
    reply = f'Routed request to {target_module or "execution"} using virtual route {route_id}.'
    if route_id == 'drivers.device.control':
        device_type = str(entities.get('device_type') or 'device')
        requested_state = str(entities.get('requested_state') or entities.get('action') or 'requested')
        if discovery.get('driver_capabilities', {}).get('available'):
            matches = (discovery.get('driver_capabilities') or {}).get('matches') or []
            top_id = str(matches[0].get('driver_id') or 'driver') if matches else 'driver'
            reply = f'Routed request to driver control for {device_type} {requested_state}. Matching driver {top_id} is available for governed execution.'
        else:
            reply = f'Routed request to driver control for {device_type} {requested_state}. Runtime driver discovery is still required before execution.'
    execution_plan = _make_execution_plan(route, discovery, user_text)
    execution_result = _execute_ingress_plan(route, discovery, user_text, meta)
    details = dict(execution_result.get('details') or {})
    followup = details.get('followup') if isinstance(details.get('followup'), dict) else {}
    compass = details.get('compass') if isinstance(details.get('compass'), dict) else {}
    if isinstance(compass, dict) and compass:
        trace['compass'] = compass
    if execution_result.get('attempted'):
        if execution_result.get('executed'):
            task_kind = str((followup or {}).get('task_kind') or '')
            if task_kind == 'browser_search':
                reply = f"Opened {details.get('target_app') or 'the browser'} and searched for {followup.get('query') or entities.get('search_query') or 'the requested topic'}."
            elif task_kind == 'browser_open_url':
                reply = f"Opened {details.get('target_app') or 'the browser'} and navigated to {followup.get('target_url') or entities.get('target_url') or 'the requested page'}."
            elif task_kind == 'document_write':
                reply = f"Opened Word and wrote a starter document about {followup.get('topic') or entities.get('topic') or 'the requested topic'}."
            elif task_kind == 'open_named_document':
                reply = f"Opened Word and attempted to open the document named {followup.get('document_name') or entities.get('document_name') or 'the requested document'}."
            elif task_kind == 'spreadsheet_template':
                reply = 'Opened Excel and created a starter spreadsheet template.'
            elif task_kind == 'website_scaffold':
                reply = f"Opened {details.get('target_app') or 'the requested editor'} and created a starter website scaffold."
            elif task_kind == 'paint_draw':
                reply = str((followup or {}).get('result') or 'Opened Paint and completed the requested drawing step.')
            else:
                reply = reply.rstrip('.') + '. Direct internal execution completed.'
        elif execution_result.get('mode') == 'governed_draft':
            reason = str(details.get('reason') or 'explicit_user_consent_required')
            reply = f"The request was routed but not executed: {reason}."
        elif execution_result.get('mode') in {'direct_internal_dispatch', 'smget_dispatch'}:
            operator_packet = details.get('operator_packet') if isinstance(details.get('operator_packet'), dict) else {}
            operator_result = operator_packet.get('result') if isinstance(operator_packet.get('result'), dict) else {}
            operator_summary = str(operator_result.get('summary') or '').strip()
            reason = str((compass or {}).get('reason') or (followup or {}).get('reason') or operator_summary or details.get('error') or 'task incomplete')
            if details.get('ok') and route_id in {'system.application.control', 'documents.office.write'}:
                reply = f"Opened {details.get('target_app') or 'the requested application'}, but the follow-through step is not finished yet: {reason}."
            else:
                reply = reply.rstrip('.') + f'. SMGET execution was attempted but did not fully complete: {reason}.'
    artifacts = {'route_ticket': route, 'executor': target_module, 'entities': entities, 'runtime_discovery': discovery, 'execution_plan': execution_plan, 'execution_result': execution_result}
    actions = list(execution_plan.get('endpoint_calls') or []) if isinstance(execution_plan.get('endpoint_calls'), list) else []
    if execution_result.get('attempted'):
        actions.append({'type': 'execution_result', 'data': execution_result})
    return NeuronResult(ok=True, reply=reply, confidence=max(0.61, float(route.get('confidence') or 0.61)), intent=str(route.get('intent_hint') or route.get('domain') or 'action'), source='ingress_router', artifacts=artifacts, trace=trace, actions=actions)



def _agent_passport_neuron_gate(meta: Dict[str, Any], ingress_route: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    context = meta.get("agent_passport_context") if isinstance(meta.get("agent_passport_context"), dict) else meta.get("agent_return_packet") if isinstance(meta.get("agent_return_packet"), dict) else None
    if not isinstance(context, dict):
        return {"applicable": False, "ok": True, "decision": "NOT_APPLICABLE", "execution_authority": False}
    packet = dict(context)
    if not packet.get("requested_lane"):
        route_id = str(ingress_route.get("route_id") or "").lower()
        intent = str(ingress_route.get("intent_hint") or ingress_route.get("domain") or "answer").lower()
        if any(x in route_id for x in ("system.", "drivers.", "network.", "application.control", "office.write")):
            packet["requested_lane"] = "action"
        elif intent in {"research", "answer", "creative"}:
            packet["requested_lane"] = intent
        else:
            packet["requested_lane"] = "answer"
    try:
        import SarahMemoryCognitiveCompass as _PassportCompass  # type: ignore
        gate = _PassportCompass.validate_agent_passport_lane(packet)
    except Exception as exc:
        gate = {"ok": False, "applicable": True, "decision": "DENY_COMPASS_UNAVAILABLE", "reason": str(exc), "execution_authority": False}
    try:
        from SarahMemoryLedger import record_governance_receipt  # type: ignore
        passport = packet.get("passport") if isinstance(packet.get("passport"), dict) else {}
        record_governance_receipt(
            "neuron_agent_gate",
            "AGENT_RESULT_SUBMITTED_TO_NEURON" if gate.get("ok") else "AGENT_RESULT_HELD_BEFORE_NEURON",
            subject_id=str(passport.get("agent_id") or packet.get("agent_id") or "unknown_agent"),
            task_id=str(passport.get("task_id") or packet.get("task_id") or ""),
            lane=str(packet.get("requested_lane") or "agent_review"),
            verdict=str(gate.get("decision") or "UNKNOWN"),
            risk="high" if not gate.get("ok") else "medium",
            retention_class="agent_passport",
            payload_hash=str(packet.get("payload_hash") or ""),
            summary=str(gate.get("reason") or "agent passport Neuron gate"),
            metadata={"passport_id": str(passport.get("passport_id") or packet.get("passport_id") or ""), "sanitized": bool(packet.get("sanitized")), "execution_authority": False},
        )
    except Exception:
        pass
    return gate



def neuron_activate_legal_candidates(candidates, continuity_state=None, *, packet=None):
    """Activate/rank only candidates already declared legal by SMLProtocol.

    Route legality is not computed here.  LogicCalc owns the deterministic
    priority-vector math; Neuron owns only activation ordering.
    """
    legal = []
    rejected = []
    for raw in list(candidates or [])[:64]:
        c = dict(raw or {})
        gates = c.get("legal_gates") if isinstance(c.get("legal_gates"), dict) else {}
        required_gates = ("capability", "authority", "safety", "resource_feasible", "time_valid", "mission_compatible")
        gate_complete = all(k in gates for k in required_gates)
        derived_legal = bool(gate_complete and all(bool(gates.get(k, False)) for k in required_gates))
        sml_legal = bool(c.get("sml_legal", derived_legal)) and derived_legal
        if not sml_legal:
            rejected.append({**c, "activation_rejected": "not_sml_legal"})
            continue
        legal.append(c)
    ranked = []
    try:
        import SarahMemoryLogicCalc as _LC  # type: ignore
        score_fn = getattr(_LC, "sml_priority_vector", None)
    except Exception:
        score_fn = None
    for idx, c in enumerate(legal):
        if callable(score_fn):
            try:
                vector = tuple(score_fn(c, continuity_state or {}))
            except Exception:
                vector = (0.0, 0.0, 0.0, float(c.get("confidence") or 0.0), 0.0, 0.0)
        else:
            vector = (0.0, 0.0, 0.0, float(c.get("confidence") or 0.0), 0.0, 0.0)
        item = dict(c)
        item["activation_vector"] = list(vector)
        item["activation_owner"] = "SarahMemoryNeuron"
        item["route_definition_owner"] = "SarahMemorySMLProtocol"
        ranked.append(item)
    ranked.sort(key=lambda item: tuple(item.get("activation_vector") or []), reverse=True)
    selected = dict(ranked[0]) if ranked else {}
    return {
        "ok": bool(selected),
        "selected": selected,
        "ranked": ranked,
        "rejected": rejected,
        "activation_owner": "SarahMemoryNeuron",
        "route_definition_owner": "SarahMemorySMLProtocol",
        "execution_authority": False,
    }


def neuron_route(user_text: str, meta: Optional[Dict[str, Any]] = None, policy: Optional[Dict[str, Any]] = None) -> NeuronResult:
    _init_db()
    budget = _budget_limits()

    meta = meta or {}
    policy = policy or {}
    approved_modules = _approved_lane_modules()
    allowed_tiers = dict(policy.get("allowed_tiers") or {"tier0": True, "tier1": True, "tier2": True, "tier3": True})
    if not approved_modules.get("websym"):
        allowed_tiers["tier1"] = False
    if not approved_modules.get("research"):
        allowed_tiers["tier2"] = False
    if not approved_modules.get("api"):
        allowed_tiers["tier3"] = False
    # Enforce request-scoped local-only as a hard restriction.
    # Local-only disables Web/API, not local datasets or the local research organ.
    if bool(meta.get("local_only") or meta.get("offline")):
        allowed_tiers["tier3"] = False
        if approved_modules.get("research"):
            allowed_tiers["tier2"] = True
    inp = NeuronInput(text=user_text or "", meta=meta)
    terminal_directive = _normalize_terminal_ai_directive(inp.text)
    if terminal_directive.get("changed"):
        inp.meta.setdefault("terminal_directive", terminal_directive)
        inp.text = str(terminal_directive.get("text") or inp.text)
    ingress_route = _normalize_ingress_route(meta, inp.text or "")
    inp.meta["ingress_route"] = ingress_route
    trace: Dict[str, Any] = {"tiers": [], "agents": [], "budget": budget, "intent": None, "advcu": {}, "ingress": ingress_route}
    trace["policy"] = {"allowed_tiers": allowed_tiers, "approved_modules": approved_modules}
    trace["core_governance"] = _core_governance_trace()

    # Canonical SML structural route.  Neuron does not invent route legality.
    try:
        import SarahMemorySMLProtocol as _SML  # type: ignore
        _sml_packet = _SML.sml_build_ingress_packet(
            inp.text,
            caller="SarahMemoryNeuron.neuron_route",
            payload={"neuron_meta": {k: v for k, v in inp.meta.items() if k not in {"api_key", "token", "password", "secret"}}},
            api_context={"local_only": bool(inp.meta.get("local_only") or inp.meta.get("offline")), "surface": inp.meta.get("surface") or "neuron"},
            discover=False,
        )
        inp.meta["sml_packet"] = _sml_packet.to_dict() if hasattr(_sml_packet, "to_dict") else _sml_packet
        trace["sml"] = _SML.sml_packet_summary(_sml_packet)
        trace["sml_pipeline"] = list(getattr(_sml_packet, "pipeline", []) or [])
        trace["sml_mission"] = str((getattr(_sml_packet, "mission", {}) or {}).get("primary") or "Unknown")
        trace["route_definition_owner"] = "SarahMemorySMLProtocol"
        trace["route_activation_owner"] = "SarahMemoryNeuron"
    except Exception as _sml_exc:
        trace["sml"] = {"ok": False, "error": str(_sml_exc)[:300], "execution_authority": False}

    agent_gate = _agent_passport_neuron_gate(inp.meta, ingress_route, inp.text)
    if agent_gate.get("applicable"):
        trace["agent_passport_gate"] = agent_gate
        if not agent_gate.get("ok"):
            return NeuronResult(
                ok=False,
                reply="Returned AI-agent data is held outside Neuron because its passport, lane, sanitation, or user-review requirements are incomplete.",
                confidence=1.0,
                intent="agent_return_hold",
                source="neuron:agent_passport_gate",
                artifacts={"agent_passport_gate": agent_gate},
                trace=trace,
                actions=[],
            )
        inp.meta["agent_evidence_only"] = True
        inp.meta["execution_authority"] = False
        # Returned agent evidence may inform answer/research reasoning only.
        allowed_tiers["tier3"] = False if bool(meta.get("local_only") or meta.get("offline")) else allowed_tiers.get("tier3", False)

    # V10/V9C: preserve /api/chat classification packet and refuse to let a model
    # invent live body-map facts if the SelfAware route somehow failed upstream.
    classification_packet = meta.get("chat_classification_packet") if isinstance(meta.get("chat_classification_packet"), dict) else {}
    if not classification_packet:
        cp2 = (meta.get("context_packet") or {}).get("chat_classification_packet") if isinstance(meta.get("context_packet"), dict) else {}
        classification_packet = cp2 if isinstance(cp2, dict) else {}
    if classification_packet:
        trace["chat_classification_packet"] = classification_packet
        inp.meta["chat_classification_packet"] = classification_packet
        if str(classification_packet.get("domain") or "") == "selfaware_body":
            kind = str(classification_packet.get("fact_kind") or "hardware fact")
            reply = f"This {kind.replace('_', ' ')} question requires the SelfAware Evidence Court. I will not generate an unverified model answer for live body hardware."
            return NeuronResult(
                ok=False,
                reply=reply,
                confidence=0.0,
                intent="selfaware_body",
                source="neuron:selfaware_route_required",
                artifacts={"chat_classification_packet": classification_packet},
                trace=trace,
                actions=[],
            )

    # Tier-0: terminal/directive greeting guard. This intentionally precedes
    # AdvCU, Research, API, and multi-agent calibration so `/ai hello` can never
    # block behind model or database latency.
    if _is_greeting_text(inp.text):
        _trace_primary_lane(trace, "answer", "GreetingGuard")
        trace["intent"] = "greeting"
        trace["tiers"].append({"tier": 0, "engine": "GreetingGuard", "ok": True})
        return NeuronResult(
            ok=True,
            reply="Hello. SarahMemory AiOS is online and ready to route governed, local-first requests.",
            confidence=0.99,
            intent="greeting",
            source="greeting_guard",
            artifacts={"greeting": {"normalized_text": inp.text, "terminal_directive": inp.meta.get("terminal_directive")}},
            trace=trace,
            actions=[],
        )

    # Tier-0: Fast deterministic math (bypass heavy routing/QA gates)
    if allowed_tiers.get('tier0', True):
        det = _try_logiccalc(inp.text)
        if det and bool(det.get('ok')):
            _trace_primary_lane(trace, 'answer', str(det.get('engine') or 'LogicCalc'))
            trace['tiers'].append({'tier': 0, 'engine': str(det.get('engine') or 'LogicCalc'), 'ok': True})
            trace['intent'] = 'math'
            artifacts = {'math': {'ok': True, 'expr': det.get('expr'), 'value': det.get('value'), 'engine': det.get('engine')}, 'deterministic': det}
            v = det.get('value')
            reply = str(v if v is not None else (det.get('text') or ''))
            try:
                fv = float(v)
                if abs(fv - round(fv)) < 1e-9:
                    reply = str(int(round(fv)))
                else:
                    reply = str(fv)
            except Exception:
                reply = str(det.get('text') or reply)
            return NeuronResult(
                ok=True,
                reply=reply,
                intent='math',
                confidence=0.99,
                source='logiccalc',
                artifacts=artifacts,
                trace=trace,
            )

    # Tier-0.5: AdvCU delegation under a strict caller deadline.
    adv_call = _bounded_call(
        "advcu",
        lambda: _advcu_analyze(inp.text),
        timeout_s=_runtime_float("NEURON_ADVCU_TIMEOUT_SECONDS", 1.5),
        default={},
    )
    adv = adv_call.get("value") if adv_call.get("ok") and isinstance(adv_call.get("value"), dict) else {
        "intent": None, "confidence": None, "command": None, "semantic_packet": {}, "helper_payload": {}, "entities": {}, "raw": {}
    }
    trace["advcu"] = {
        "intent": adv.get("intent"),
        "confidence": adv.get("confidence"),
        "has_command": bool(adv.get("command")),
        "bounded_call": {k: v for k, v in adv_call.items() if k != "value"},
    }
    inp.meta["sm_helper_payload"] = _build_helper_payload(inp.text, str(adv.get("intent") or ""), adv)

    # Intent selection
    intent = _classify_intent(inp.text)
    if ingress_route.get("intent_hint") and float(ingress_route.get("confidence") or 0.0) >= 0.58:
        intent = str(ingress_route.get("intent_hint") or intent).lower()
    try:
        adv_intent = str(adv.get("intent") or "").strip()
        adv_conf = adv.get("confidence")
        if adv_intent and isinstance(adv_conf, (int, float)) and float(adv_conf) >= 0.55:
            adv_intent_l = adv_intent.lower()
            if adv_intent_l in {"identity", "identity_query"}:
                intent = "identity"
            elif intent == "identity" and adv_intent_l in {"question", "general", "chat"}:
                pass
            else:
                intent = adv_intent_l
    except Exception:
        pass

    inp.meta["intent"] = intent
    trace["intent"] = intent

    # WAVE7: deterministic lane-collapse audit. General answer lanes should
    # remain answer-only; higher-risk action/network/system lanes are tagged
    # before any helper execution path can proceed.
    requested_lane = "answer"
    ingress_domain = str(ingress_route.get("domain") or "chat").strip().lower()
    ingress_action = str(ingress_route.get("action") or "general_reply").strip().lower()
    ingress_target = str(ingress_route.get("target_module") or "").strip()
    default_chat_target = (
        ingress_domain == "chat"
        and ingress_action in {"general_reply", "chat", "reply"}
        and ingress_target in {"", "SarahMemoryReply"}
    )
    if _detect_system_kind(inp.text, intent):
        requested_lane = "system"
    elif (ingress_target and not default_chat_target) or intent in {"action", "command", "tool", "filesystem", "network"}:
        requested_lane = "action"
    lane_gate = _logiccalc_neuron_axis_guard(
        requested_lane,
        current_lane="answer",
        risk_hint=str(inp.meta.get("risk_hint") or "low"),
        user_present=bool(inp.meta.get("user_present", True)),
        user_consented=bool(inp.meta.get("user_consented", False)),
        route_confidence=float(ingress_route.get("confidence") or 0.0),
    )
    trace["logiccalc_neuron_axis_gate"] = lane_gate
    if requested_lane in {"action", "network", "system"} and int((lane_gate or {}).get("decision", 1) or 0) == 0:
        return NeuronResult(
            ok=False,
            reply="I held that request in the current lane because the Neuron Axis gate did not authorize the lane transition.",
            confidence=0.0,
            intent=intent,
            source="logiccalc_neuron_axis_gate",
            artifacts={"logiccalc_neuron_axis_gate": lane_gate},
            trace=trace,
            actions=[],
        )

    explicit_ingress_action = bool(
        ingress_domain != "chat"
        or (ingress_target and not default_chat_target)
        or str(ingress_route.get("route_id") or "").strip().lower() not in {"", "chat.general"}
    )
    if requested_lane in {"action", "network", "system"} and explicit_ingress_action and float(ingress_route.get("confidence") or 0.0) < 0.66:
        return NeuronResult(
            ok=False,
            reply="I held the structured action request because its ingress route confidence was below the execution threshold.",
            confidence=0.0,
            intent=intent,
            source="semantic_ingress_confidence_gate",
            artifacts={"ingress_route": ingress_route, "minimum_confidence": 0.66},
            trace=trace,
            actions=[],
        )

    greeting_reply = _try_greeting_reply(inp.text, intent, adv)
    if greeting_reply:
        _trace_primary_lane(trace, "answer", "GreetingGuard")
        trace["tiers"].append({"tier": 0, "engine": "GreetingGuard", "ok": True})
        artifacts = {"greeting": {"normalized_text": inp.text, "terminal_directive": inp.meta.get("terminal_directive")}}
        return NeuronResult(
            ok=True,
            reply=greeting_reply,
            confidence=0.98,
            intent="greeting",
            source="greeting_guard",
            artifacts=artifacts,
            trace=trace,
            actions=[],
        )

    # Tier-2L: General knowledge local-first answer lane.
    # Local-only/offline disables Web/API, not local datasets or local LLM.
    if requested_lane == "answer" and intent != "identity" and not _detect_system_kind(inp.text, intent) and bool(allowed_tiers.get("tier2", True)) and _looks_like_general_knowledge_query(inp.text, intent):
        local_research_data = _try_research(inp.text, local_only=True, intent=intent)
        if _research_has_usable_content(local_research_data, min_confidence=0.01):
            _trace_primary_lane(trace, "answer", "SarahMemoryResearch.Local")
            trace["tiers"].append({"tier": "2L", "engine": "LocalResearch", "ok": True, "local_only": True})
            merged, artifacts = _synthesize_evidence_reply("Here is what I found locally:", local_research_data or {})
            return NeuronResult(
                ok=True,
                reply=merged,
                confidence=max(0.58, min(0.92, float((local_research_data or {}).get("confidence") or 0.58))),
                intent="general_knowledge" if intent in {"question", "general", "chat", "unknown"} else intent,
                source="local_research",
                artifacts=artifacts,
                trace=trace,
            )
        trace["tiers"].append({"tier": "2L", "engine": "LocalResearch", "ok": False, "local_only": True})

    if intent == "identity":
        try:
            import SarahMemoryCognitiveSelf as _SMCognitiveSelf  # type: ignore
            fn = getattr(_SMCognitiveSelf, "answer_identity_question", None)
            packet = fn(inp.text) if callable(fn) else {}
        except Exception as exc:
            packet = {"ok": False, "error": str(exc), "execution_authority": False}
        ok = bool(isinstance(packet, dict) and packet.get("ok"))
        _trace_primary_lane(trace, "answer", "SarahMemoryCognitiveSelf")
        trace["tiers"].append({"tier": 0, "engine": "CognitiveSelf.Identity", "ok": ok})
        if not ok:
            return NeuronResult(ok=False, reply="Identity state is not currently verified.", confidence=0.25, intent="identity", source="cognitive_self", artifacts={"identity": packet}, trace=trace)
        return NeuronResult(
            ok=True,
            reply=str(packet.get("reply") or ""),
            confidence=float(packet.get("confidence") or 0.99),
            intent="identity",
            source=str(packet.get("source") or "SarahMemoryCognitiveSelf"),
            artifacts={"identity": packet},
            trace=trace,
        )


    if str(ingress_route.get("route_id") or "").startswith("research.weather") and bool(allowed_tiers.get("tier2", True)) and not inp.meta.get("offline"):
        research_data = _try_research(inp.text, local_only=False, intent=intent)
        if research_data:
            _trace_primary_lane(trace, 'answer', 'SarahMemoryResearch')
            trace["tiers"].append({"tier": "ingress", "engine": "SemanticIngress->Research", "ok": True})
            merged, artifacts = _synthesize_evidence_reply("Here is what I found:", research_data)
            res = NeuronResult(ok=True, reply=merged, confidence=max(0.72, float(ingress_route.get("confidence") or 0.72)), intent=intent, source="research", artifacts={"ingress_route": ingress_route, **(artifacts or {})}, trace=trace)
            _log_event("route", intent, res.confidence, res.source, {"input": inp.text, "trace": trace, "artifacts_keys": list(res.artifacts.keys())})
            return res
        trace["tiers"].append({"tier": "ingress", "engine": "SemanticIngress->Research", "ok": False})

    if intent not in {"device_query", "diagnostics", "identity"} and ingress_route.get("domain") in {"drivers", "system", "documents", "email", "network", "communication", "reminder", "avatar"} and float(ingress_route.get("confidence") or 0.0) >= 0.66:
        _trace_primary_lane(trace, 'action', str(ingress_route.get("target_module") or 'executor'))
        trace["tiers"].append({"tier": "ingress", "engine": "SemanticIngressExecutor", "ok": True})
        res = _ingress_execution_ticket(ingress_route, trace, inp.text, inp.meta)
        _log_event("route", intent, res.confidence, res.source, {"input": inp.text, "trace": trace, "artifacts_keys": list(res.artifacts.keys())})
        return res

    # Governance handshake: ask CognitiveServices for request-scoped policy and
    # optional CognitiveThinker co-review before routing high-impact work.
    gov_decision = None
    if _Cog and approved_modules.get("cognitive") and hasattr(_Cog, "govern_request"):
        try:
            gov_ctx = dict(inp.meta or {})
            gov_ctx.setdefault("local_only", bool(inp.meta.get("local_only") or inp.meta.get("offline")))
            gov_ctx.setdefault("safe_mode", _is_safe_mode())
            gov_ctx.setdefault("neuron_route", True)
            gov_ctx.setdefault("caller", "SarahMemoryNeuron.neuron_route")
            gov_ctx.setdefault("force_cognitive_thinker_consult", False)
            gov_call = _bounded_call(
                "cognitive_governor",
                lambda: _Cog.govern_request(  # type: ignore[attr-defined]
                    inp.text,
                    caller="SarahMemoryNeuron.neuron_route",
                    caller_context=gov_ctx,
                    user_present=bool(inp.meta.get("user_present", True)),
                    user_consented=bool(inp.meta.get("user_consented", False)),
                    proposed_action=dict(inp.meta.get("proposed_action") or {}),
                ),
                timeout_s=_runtime_float("NEURON_GOVERNOR_TIMEOUT_SECONDS", 2.0),
                default={"decision": "DEFER", "reasons": ["governor_timeout_or_unavailable"]},
            )
            gov_decision = gov_call.get("value") if isinstance(gov_call.get("value"), dict) else {"decision": "DEFER", "reasons": ["invalid_governor_packet"]}
            trace["governor_bounded_call"] = {k: v for k, v in gov_call.items() if k != "value"}
            if isinstance(gov_decision, dict):
                policy_from_governor = dict(gov_decision.get("routing_policy") or {})
                if isinstance(policy_from_governor.get("allowed_tiers"), dict):
                    for tier_name, tier_allowed in policy_from_governor.get("allowed_tiers", {}).items():
                        allowed_tiers[tier_name] = bool(allowed_tiers.get(tier_name, True) and tier_allowed)
                trace["governance"] = {
                    "decision": gov_decision.get("decision"),
                    "risk": gov_decision.get("risk"),
                    "risk_score": gov_decision.get("risk_score"),
                    "require_user": gov_decision.get("require_user"),
                    "recommended_next": gov_decision.get("recommended_next"),
                    "coequal_governance": gov_decision.get("coequal_governance") or {},
                }
                trace["policy"]["allowed_tiers"] = allowed_tiers

                decision_name = str(gov_decision.get("decision") or "").upper()
                lane_family = str(gov_decision.get("lane_family") or "").lower()
                if decision_name in ("DENY", "REQUIRE_USER", "DEFER") and lane_family in ("action", "network", "system"):
                    msg = str(gov_decision.get("recommended_next") or "Request blocked by governance.")
                    return NeuronResult(
                        ok=(decision_name == "DEFER"),
                        reply=msg,
                        confidence=0.90 if decision_name != "DEFER" else 0.70,
                        intent=intent,
                        source="governance",
                        artifacts={"governance": gov_decision},
                        trace=trace,
                    )
        except Exception as e:
            trace.setdefault("governance", {})["error"] = str(e)

    # System/Diagnostics lane. Neuron selects the legal route; CognitiveSelf and
    # Diagnostics own self/runtime interpretation and diagnostics functionality.
    system_kind = _detect_system_kind(inp.text, intent)
    if system_kind:
        if not bool(allowed_tiers.get("tier0", True)):
            trace["tiers"].append({"tier": 0, "engine": "SystemInfo", "ok": False, "reason": "policy_disallow"})
            return NeuronResult(ok=False, reply="System tools are disabled by policy in this runtime.", intent=intent, source="system", confidence=0.9, artifacts={}, trace=trace)
        if _is_public_device(inp.meta):
            trace["tiers"].append({"tier": 0, "engine": "SystemInfo", "ok": False, "reason": "public_web_restricted"})
            return NeuronResult(ok=False, reply="This request is not available in Public Web mode.", intent=intent, source="system", confidence=0.9, artifacts={}, trace=trace)

        if system_kind == "diagnostics":
            _trace_primary_lane(trace, "system", "SarahMemoryDiagnostics")
            try:
                import SarahMemoryDiagnostics as _D  # type: ignore
                fn = getattr(_D, "run_system_diagnostics", None)
                diag = fn() if callable(fn) else {"ok": False, "error": "diagnostics_callable_missing"}
            except Exception as exc:
                diag = {"ok": False, "error": str(exc)}
            ok = bool(isinstance(diag, dict) and diag.get("ok", False))
            trace["tiers"].append({"tier": 0, "engine": "SarahMemoryDiagnostics", "ok": ok})
            return NeuronResult(ok=ok, reply="Diagnostics completed." if ok else "Diagnostics did not complete successfully.", intent="diagnostics", source="diagnostics", confidence=0.9 if ok else 0.35, artifacts={"diagnostics": diag}, trace=trace)

        _trace_primary_lane(trace, "system", "SarahMemoryCognitiveSelf")
        try:
            import SarahMemoryCognitiveSelf as _SMCognitiveSelf  # type: ignore
            fn = getattr(_SMCognitiveSelf, "answer_runtime_system_question", None)
            packet = fn(system_kind, inp.text, context=inp.meta) if callable(fn) else {"ok": False, "error": "cognitive_self_runtime_answer_missing"}
        except Exception as exc:
            packet = {"ok": False, "error": str(exc), "execution_authority": False}
        ok = bool(isinstance(packet, dict) and packet.get("ok"))
        trace["tiers"].append({"tier": 0, "engine": "CognitiveSelf.RuntimeBody", "kind": system_kind, "ok": ok})
        return NeuronResult(
            ok=ok,
            reply=str(packet.get("reply") or "Runtime body evidence is not currently available."),
            intent="device_query",
            source=str(packet.get("source") or "SarahMemoryCognitiveSelf"),
            confidence=float(packet.get("confidence") or (0.9 if ok else 0.25)),
            artifacts={system_kind: packet.get("evidence") or packet.get("packet") or packet},
            trace=trace,
        )

    # Vision lane: classify once, then either answer visually or hand off to Action lane.
    vision_request = None
    vision_res = None
    if _is_visual_request(inp.meta.get("intent") or intent, inp.text, meta=inp.meta, adv=adv):
        try:
            vision_request = _build_vision_request(inp.text, inp.meta, adv)
            inp.meta["vision_request"] = vision_request
            trace["vision_request"] = {
                "query_type": vision_request.get("query_type"),
                "subject": vision_request.get("requested_subject") or vision_request.get("subject"),
                "action_expectation": vision_request.get("action_expectation"),
            }
        except Exception:
            vision_request = None

    if vision_request and str(vision_request.get("action_expectation") or "answer_only") == "answer_only":
        vision_res = _try_vision_lane(inp.text, meta=inp.meta, adv=adv)
    if vision_res is not None:
        trace.setdefault("tiers", []).extend(list((vision_res.trace or {}).get("tiers") or []))
        _trace_primary_lane(trace, "answer", "SarahMemorySOBJE")
        trace["vision"] = {
            "query_type": vision_res.intent,
            "frame_available": bool(((vision_res.artifacts or {}).get("vision") or {}).get("frame_available")),
        }
        res = vision_res
    else:
            # Action lane: emit local execution ticket (never execute here)
        if _is_action_intent(intent, inp.text, adv):
            if not approved_modules.get("filesystem"):
                trace["tiers"].append({"tier": "action", "engine": "ActionTicket", "ok": False, "reason": "filesystem_not_registered"})
                return NeuronResult(
                    ok=False,
                    reply="Action lane is not available because SarahMemoryFilesystem is not registered and approved.",
                    confidence=0.25,
                    intent="action",
                    source="action_ticket",
                    artifacts={"action_ticket_error": "filesystem_not_registered"},
                    trace=trace,
                )
            ticket = _make_action_ticket(inp.text, inp.meta, adv)
            _trace_primary_lane(trace, 'action', str(ticket.get('executor') or 'SarahMemoryFilesystem'))
            trace["tiers"].append({"tier": "action", "engine": "ActionTicket", "ok": True, "action": ticket.get("action")})
            res = NeuronResult(
                ok=True,
                reply=f"ACTION_TICKET::{ticket.get('action')}::{ticket.get('ticket_id')}",
                confidence=0.74,
                intent="action",
                source="action_ticket",
                artifacts={"action_ticket": ticket},
                trace=trace,
            )
            _log_event("route", "action", res.confidence, res.source, {"input": inp.text, "trace": trace, "artifacts_keys": list(res.artifacts.keys())})
            return res



        # Creative lane: emit job ticket
        if _is_creative_intent(intent, inp.text, adv):
            _trace_primary_lane(trace, 'creative', 'SarahMemoryCanvasStudio')
            if not approved_modules.get("canvas"):
                trace["tiers"].append({"tier": "creative", "engine": "CanvasStudioTicket", "ok": False, "reason": "canvas_not_registered"})
                return NeuronResult(
                    ok=False,
                    reply="Creative lane is not available because SarahMemoryCanvasStudio is not registered and approved.",
                    confidence=0.25,
                    intent="creative",
                    source="creative_ticket",
                    artifacts={"creative_ticket_error": "canvas_not_registered"},
                    trace=trace,
                )
            kind = _creative_kind(intent, inp.text, adv)
            ticket = _make_creative_job_ticket(inp.text, kind, inp.meta, adv)
            trace["tiers"].append({"tier": "creative", "engine": "CanvasStudioTicket", "ok": True, "kind": kind})
            res = NeuronResult(
                ok=True,
                reply=f"CREATIVE_JOB_TICKET::{kind}::{ticket.get('job_id')}",
                confidence=0.78,
                intent=kind,
                source="creative_ticket",
                artifacts={"job_ticket": ticket},
                trace=trace,
            )
            _log_event("route", kind, res.confidence, res.source, {"input": inp.text, "trace": trace, "artifacts_keys": list(res.artifacts.keys())})
            return res

        # Tier-0: LogicCalc
        det = _try_logiccalc(inp.text)
        if det:
            inp.meta["deterministic_hit"] = True
            _trace_primary_lane(trace, 'answer', str(det.get('engine') or 'LogicCalc'))
            trace["tiers"].append({"tier": 0, "engine": str(det.get('engine') or 'LogicCalc'), "ok": True})
            reply = det.get("reply") if isinstance(det, dict) else None
            if not reply and isinstance(det, dict):
                reply = det.get("text") or det.get("result")
            reply = str(reply) if reply else "Deterministic engine produced output (no text payload)."
            res = NeuronResult(ok=True, reply=reply, confidence=0.78, intent=intent, source="logiccalc", artifacts={"det": det}, trace=trace)

        else:
            trace["tiers"].append({"tier": 0, "engine": "LogicCalc", "ok": False})

            # Tier-1: WebSYM
            sym = _try_websym(inp.text) if bool(allowed_tiers.get("tier1", True)) else None
            if sym:
                _trace_primary_lane(trace, 'answer', 'SarahMemoryWebSYM')
                trace["tiers"].append({"tier": 1, "engine": "WebSYM", "ok": True})
                res = NeuronResult(ok=True, reply=sym, confidence=0.66, intent=intent, source="websym", artifacts={}, trace=trace)
            else:
                trace["tiers"].append({"tier": 1, "engine": "WebSYM", "ok": False})

                # Tier-2: Research lane (evidence-backed)
                research_data = None
                use_research_lane = bool(intent == "research" or _looks_like_general_knowledge_query(inp.text, intent))
                if bool(allowed_tiers.get("tier2", True)) and use_research_lane:
                    research_data = _try_research(
                        inp.text,
                        local_only=bool(inp.meta.get("local_only") or inp.meta.get("offline") or _is_local_only()),
                        intent=intent,
                    )
                if _research_has_usable_content(research_data, min_confidence=0.01):
                    _trace_primary_lane(trace, 'answer', 'SarahMemoryResearch')
                    trace["tiers"].append({"tier": 2, "engine": "Research", "ok": True, "local_only": bool(inp.meta.get("local_only") or inp.meta.get("offline") or _is_local_only())})
                    merged, artifacts = _synthesize_evidence_reply("Here is what I found:", research_data or {})
                    res = NeuronResult(ok=True, reply=merged, confidence=max(0.58, float((research_data or {}).get("confidence") or 0.70)), intent=intent, source="research", artifacts=artifacts, trace=trace)
                else:
                    trace["tiers"].append({"tier": 2, "engine": "Research", "ok": False, "local_only": bool(inp.meta.get("local_only") or inp.meta.get("offline") or _is_local_only())})

                    # Tier-3: API
                    api_reply = _try_api(inp.text, meta=inp.meta) if bool(allowed_tiers.get("tier3", True)) else None
                    if api_reply:
                        _trace_primary_lane(trace, 'answer', 'SarahMemoryAPI')
                        trace["tiers"].append({"tier": 3, "engine": "SarahMemoryAPI", "ok": True})
                        res = NeuronResult(ok=True, reply=api_reply, confidence=0.62, intent=intent, source="api", artifacts={}, trace=trace)
                    else:
                        trace["tiers"].append({"tier": 3, "engine": "SarahMemoryAPI", "ok": False})
                        res = NeuronResult(
                            ok=False,
                            reply="No engine produced an answer. Provide more constraints or enable an applicable tier.",
                            confidence=0.35,
                            intent=intent,
                            source="neuron",
                            artifacts={},
                            trace=trace,
                        )

    # Graph link
    try:
        if intent != "empty":
            _GRAPH.link("user_query", intent, "classified_as", 0.7, {"q": inp.text[:200]})
    except Exception:
        pass

    # Curiosity prompts
    try:
        curiosity = _curiosity_prompts(intent, inp.text, budget)
        if curiosity:
            res.artifacts["curiosity"] = curiosity
    except Exception:
        pass

    # Multi-agent calibration is advisory and strictly bounded. Each agent sees
    # the same immutable draft snapshot; deterministic merge order follows the
    # declared agent list, never thread completion order.
    agents: List[ThoughtAgent] = [AnalystAgent(), SkepticAgent(), OptimizerAgent(), EngineerAgent(), GovernorAgent()]
    agents = agents[: int(budget.get("max_parallel", 4))]

    draft = str(res.reply or "")
    conf = float(res.confidence)
    agent_timeout = _runtime_float("NEURON_AGENT_TIMEOUT_SECONDS", 0.25)
    for agent in agents:
        snapshot = draft
        call = _bounded_call(
            f"thought_agent:{agent.name}",
            lambda a=agent, d=snapshot: a.evaluate(inp, d),
            timeout_s=agent_timeout,
            default=(snapshot, 0.0, {"reason": "bounded_default"}),
        )
        value = call.get("value")
        if call.get("ok") and isinstance(value, tuple) and len(value) == 3:
            candidate, delta, notes = value
            draft = str(candidate if candidate is not None else snapshot)
            try:
                delta_f = float(delta)
            except Exception:
                delta_f = 0.0
            conf = float(max(0.0, min(0.99, conf + delta_f)))
            trace["agents"].append({"agent": agent.name, "delta": delta_f, "notes": notes, "bounded_call": {k: v for k, v in call.items() if k != "value"}})
        else:
            trace["agents"].append({"agent": agent.name, "delta": 0.0, "notes": {"error": call.get("error") or "agent_unavailable"}, "bounded_call": {k: v for k, v in call.items() if k != "value"}})

    # Compare-based QA gate
    try:
        cd, cart = _qa_compare_gate(inp.text, draft, intent)
        if cart:
            res.artifacts.update(cart)
        conf = float(max(0.0, min(0.99, conf + float(cd))))
        trace["agents"].append({"agent": "CompareQA", "delta": float(cd), "notes": {"enabled": bool(_Compare)}})
    except Exception:
        pass

    # Low-confidence evidence repair
    try:
        if conf < 0.55 and bool(allowed_tiers.get("tier2", True)):
            rdata = _try_research(
                inp.text,
                local_only=bool(inp.meta.get("local_only") or inp.meta.get("offline") or _is_local_only() or not bool(getattr(config, "WEB_RESEARCH_ENABLED", False))),
                intent=intent,
            )
            if _research_has_usable_content(rdata, min_confidence=0.01):
                merged, artifacts = _synthesize_evidence_reply(draft, rdata)
                if merged and str(merged).strip():
                    draft = str(merged).strip()
                    if not trace.get("primary_lane"):
                        _trace_primary_lane(trace, 'answer', 'SarahMemoryResearch')
                res.artifacts.update(artifacts)
                conf = float(max(0.0, min(0.99, conf + 0.06)))
                res.ok = True
                if getattr(res, "source", "neuron") in ("neuron", "", None):
                    res.source = "research"
                trace["tiers"].append({"tier": 2, "engine": "Research", "ok": True, "reason": "low_confidence_repair"})
    except Exception:
        pass

    _fallback_reply = "No engine produced an answer. Provide more constraints or enable an applicable tier."
    _draft_str = str(draft or "").strip()

    # If the route recovered content but draft still contains the default failure string,
    # promote the best available candidate from artifacts.
    try:
        if (not _draft_str) or (_draft_str == _fallback_reply):
            best_candidate = ""

            research_art = res.artifacts.get("research") if isinstance(res.artifacts, dict) else None
            if isinstance(research_art, dict):
                for key in ("data", "snippet"):
                    val = research_art.get(key)
                    if isinstance(val, str):
                        val = val.strip()
                        if val and not val.lower().startswith("sorry, i was unable to find any reliable information"):
                            best_candidate = val
                            break

            if not best_candidate:
                compare_art = res.artifacts.get("compare") if isinstance(res.artifacts, dict) else None
                if isinstance(compare_art, dict):
                    val = compare_art.get("api_response")
                    if isinstance(val, str):
                        val = val.strip()
                        if val and val != _fallback_reply and not val.lower().startswith("i couldn't find reliable information"):
                            best_candidate = val

            if best_candidate:
                draft = best_candidate
                _draft_str = best_candidate
                res.ok = True
                if getattr(res, "source", "neuron") in ("neuron", "", None):
                    res.source = "research"
                conf = float(max(conf, 0.58))
    except Exception:
        pass

    res.reply = _draft_str or _fallback_reply
    res.confidence = conf

    # Governance stamp
    res.artifacts.setdefault("governance", {})["neosky"] = "ARMED" if _neosky_armed() else "SAFE"

    # Log
    _log_event("route", intent, res.confidence, res.source, {"input": inp.text, "trace": trace, "artifacts_keys": list(res.artifacts.keys())})
    res.trace = trace
    return res


# -----------------------------------------------------------------------------
# Background neuron service (heartbeat-style)
# -----------------------------------------------------------------------------
_NEURON_THREAD: Optional[threading.Thread] = None
_NEURON_STOP = threading.Event()
_NEURON_Q: "queue.Queue[NeuronInput]" = queue.Queue(maxsize=100)

def neuron_submit(text: str, meta: Optional[Dict[str, Any]] = None) -> bool:
    try:
        _NEURON_Q.put_nowait(NeuronInput(text=text, meta=meta or {}))
        return True
    except Exception:
        return False

def neuron_tick() -> Optional[NeuronResult]:
    if _NEURON_Q.empty():
        return None
    try:
        inp = _NEURON_Q.get_nowait()
    except Exception:
        return None
    try:
        return neuron_route(inp.text, meta=inp.meta)
    except Exception as e:
        logger.error("neuron_tick error: %s", e)
        return NeuronResult(ok=False, reply=f"Neuron tick failed: {e}", confidence=0.1, intent="error", source="neuron")

def _neuron_loop(poll_s: float = 0.25) -> None:
    logger.info("[Neuron] background loop started")
    while not _NEURON_STOP.is_set():
        try:
            r = neuron_tick()
            if r:
                try:
                    if _Cog and _core_module_allowed("SarahMemoryCognitiveServices", "diagnostics", _Cog) and hasattr(_Cog, "notify_neuron_result"):
                        _Cog.notify_neuron_result(r.to_dict())  # type: ignore
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(poll_s)
    logger.info("[Neuron] background loop stopped")

def start_neuron_background() -> bool:
    global _NEURON_THREAD
    if _NEURON_THREAD and _NEURON_THREAD.is_alive():
        return True
    _NEURON_STOP.clear()
    _NEURON_THREAD = threading.Thread(target=_neuron_loop, daemon=True, name="SarahMemoryNeuronThread")
    _NEURON_THREAD.start()
    return True

def stop_neuron_background(timeout: float = 2.0) -> bool:
    _NEURON_STOP.set()
    try:
        thread = _NEURON_THREAD
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout)))
    except Exception:
        pass
    return not bool(_NEURON_THREAD and _NEURON_THREAD.is_alive())

def neuron_status() -> Dict[str, Any]:
    return {
        "running": bool(_NEURON_THREAD and _NEURON_THREAD.is_alive()),
        "queue": int(getattr(_NEURON_Q, "qsize", lambda: 0)()),
        "profile": _device_profile(),
        "db": _neuron_db_path(),
        "circuits": neuron_circuit_status(),
    }

# -----------------------------------------------------------------------------
# CLI quick test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    _init_db()
    print(json.dumps(neuron_status(), indent=2))
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        q = "Convert 12 ft to meters and explain units."
    out = neuron_route(q, {"cli": True})
    print(json.dumps(out.to_dict(), indent=2))
    
    
# --- SM V8.0 TRI-LAYER PATCH 2026-05-20 ---
def build_layered_cognition_packet(text: str, context_packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build Layer 2 language + Layer 3 emotion + identity packets for Neuron routing evidence."""
    try:
        import SarahMemoryCognitiveIdentityLayer as _CIL  # type: ignore
        tri = _CIL.build_tri_layer_input_packet(text, context_packet=context_packet)
    except Exception as e:
        tri = {"packet_type": "TriLayerInputPacket", "error": str(e), "raw_text": str(text or "")}
    try:
        if _AdvCU and hasattr(_AdvCU, "build_contextual_intent_packet"):
            tri["contextual_intent_packet"] = _AdvCU.build_contextual_intent_packet(
                text,
                language_context_packet=tri.get("language_context_packet"),
                context_packet=context_packet,
            )
    except Exception as e:
        tri.setdefault("errors", []).append({"advcu_contextual_intent": str(e)})
    return tri

# ====================================================================
# END OF SarahMemoryNeuron.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryNeuron',
    "version": "v9.0.0-alpha-sml-0.2",
    "category": 'Reasoning',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['activation_routing', 'reasoning'],
    "supported_missions": ['Conversation', 'Knowledge', 'Planning', 'Programming'],
    "supported_omega": ['Ω001', 'Ω002', 'Ω005', 'Ω010', 'Ω020', 'Ω030', 'Ω040'],
    "required_authority": ['Read'],
    "priority": 70,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryNeuron.py'},
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
        "component": 'SarahMemoryNeuron',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryNeuron', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

