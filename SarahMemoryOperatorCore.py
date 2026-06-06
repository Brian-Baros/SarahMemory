"""--==The SarahMemory Project==--
File: SarahMemoryOperatorCore.py
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
- Governed local operator control plane for SarahMemory AiOS / SMGET.
- Converts user intent into a normalized ActionContract before real execution.
- Owns the execution lifecycle state machine, verification discipline, rollback coordination,
  and auditable result packaging.
- Provides the hard execution choke-point between cognition/governance and domain executors.
- Keeps raw natural-language requests from directly touching OS / driver / service actions.

CORE DOCTRINE:
- No direct execution from raw AI output.
- All non-trivial actions require an ActionContract.
- Every state-changing action must be verified before success is reported.
- Rollback must exist wherever practical, and irreversibility must be explicit.
- Discovery is not activation. Registered capability and approval checks remain authoritative.
- Local-first, fail-soft, auditable, and safe-by-default.

RELATIONSHIP MODEL:
- SarahMemoryCognitiveCompass.py   -> orientation / anti-drift / goal re-anchor
- SarahMemoryCognitiveThinker.py   -> optional deeper planning / ethical possibility guidance
- SarahMemoryCognitiveServices.py  -> what may proceed right now
- SarahMemorySecurityGovernor.py   -> trust / sovereignty / integrity gate (when present)
- SarahMemoryOperatorCore.py       -> action contract + lifecycle + verify/rollback
- SarahMemoryNeuron.py             -> lane routing / dispatch selection
- Executors / Drivers              -> perform bounded domain work
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "operator_core"
# CATEGORY = "governed_execution"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "operator_core"
# HARDWARE_DOMAIN = "system_network_filesystem_devices"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "operator_core"
# FAMILY = "smget"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "SMGET execution choke-point. Builds ActionContracts, enforces lifecycle state transitions, runs verification and rollback, and emits auditable outcomes."
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
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Safe imports (never hard-fail the platform)
# ---------------------------------------------------------------------------
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None

try:
    import SarahMemoryAdvCU as _AdvCU  # type: ignore
except Exception:
    _AdvCU = None

try:
    import SarahMemoryCognitiveServices as _CogServices  # type: ignore
except Exception:
    _CogServices = None

try:
    import SarahMemoryNeuron as _Neuron  # type: ignore
except Exception:
    _Neuron = None

try:
    import SarahMemorySi as _Si  # type: ignore
except Exception:
    _Si = None

try:
    import SarahMemoryDiagnostics as _Diagnostics  # type: ignore
except Exception:
    _Diagnostics = None

try:
    import SarahMemoryCompare as _Compare  # type: ignore
except Exception:
    _Compare = None

try:
    import SarahMemorySecurityGovernor as _SecurityGovernor  # type: ignore
except Exception:
    _SecurityGovernor = None

try:
    import SarahMemoryAssuranceGate as _AssuranceGate  # type: ignore
except Exception:
    _AssuranceGate = None

try:
    import SarahMemoryTrustRegistry as _TrustRegistry  # type: ignore
except Exception:
    _TrustRegistry = None

try:
    import SarahMemorySafetyPolicies as _SafetyPolicies  # type: ignore
except Exception:
    _SafetyPolicies = None

try:
    import SarahMemoryMSDC as _MSDC  # type: ignore
except Exception:
    _MSDC = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("SarahMemoryOperatorCore")
logger.setLevel(logging.DEBUG if bool(getattr(config, "DEBUG_MODE", True)) else logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.propagate = False


# ---------------------------------------------------------------------------
# Constants / lifecycle
# ---------------------------------------------------------------------------
MODULE_NAME = "SarahMemoryOperatorCore"
MODULE_VERSION = "8.0.0"
_OPERATOR_DB_NAME = "operator_core.db"

STATE_RECEIVED = "RECEIVED"
STATE_CLASSIFIED = "CLASSIFIED"
STATE_AUTHORIZED = "AUTHORIZED"
STATE_PLANNED = "PLANNED"
STATE_DISPATCHED = "DISPATCHED"
STATE_EXECUTING = "EXECUTING"
STATE_VERIFIED = "VERIFIED"
STATE_SUCCEEDED = "SUCCEEDED"
STATE_FAILED = "FAILED"
STATE_ROLLED_BACK = "ROLLED_BACK"
STATE_REPORTED = "REPORTED"

RISK_TIER_0 = "TIER_0_INFO"
RISK_TIER_1 = "TIER_1_HARMLESS_LOCAL_UI"
RISK_TIER_2 = "TIER_2_BOUNDED_LOCAL_OPERATION"
RISK_TIER_3 = "TIER_3_PRIVILEGED_SYSTEM"
RISK_TIER_4 = "TIER_4_NETWORK_REMOTE_OR_DESTRUCTIVE"

MODE_SIMULATE = "simulate"
MODE_DRAFT = "draft"
MODE_APPLY = "apply"
MODE_ROLLBACK = "rollback"
_ALLOWED_EXECUTION_MODES = {MODE_SIMULATE, MODE_DRAFT, MODE_APPLY, MODE_ROLLBACK}


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


def _operator_db_path() -> str:
    return os.path.join(_datasets_dir(), _OPERATOR_DB_NAME)


def _ensure_dirs() -> None:
    try:
        os.makedirs(_data_dir(), exist_ok=True)
    except Exception:
        pass
    try:
        os.makedirs(_datasets_dir(), exist_ok=True)
    except Exception:
        pass
    try:
        os.makedirs(_settings_dir(), exist_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def _connect_db() -> sqlite3.Connection:
    _ensure_dirs()
    con = sqlite3.connect(_operator_db_path(), timeout=5.0, check_same_thread=False)
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
            CREATE TABLE IF NOT EXISTS operator_actions (
                contract_id TEXT PRIMARY KEY,
                ts TEXT,
                state TEXT,
                execution_mode TEXT,
                primary_lane TEXT,
                action_type TEXT,
                target TEXT,
                executor_name TEXT,
                risk_level TEXT,
                success INTEGER,
                audit_id TEXT,
                contract_json TEXT,
                result_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS operator_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                contract_id TEXT,
                state TEXT,
                event_kind TEXT,
                details TEXT,
                meta_json TEXT
            )
            """
        )
        con.commit()
    except Exception as exc:
        logger.debug("OperatorCore DB ensure failed: %s", exc)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def log_operator_event(
    contract_id: str,
    state: str,
    event_kind: str,
    details: str,
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    con = None
    try:
        _ensure_tables()
        con = _connect_db()
        cur = con.cursor()
        cur.execute(
            "INSERT INTO operator_events (ts, contract_id, state, event_kind, details, meta_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                str(contract_id or ""),
                str(state or ""),
                str(event_kind or ""),
                str(details or ""),
                json.dumps(meta or {}, ensure_ascii=False),
            ),
        )
        con.commit()
    except Exception as exc:
        logger.debug("OperatorCore event log failed: %s", exc)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dataclasses / contracts
# ---------------------------------------------------------------------------
@dataclass
class VerificationCheck:
    name: str
    description: str
    required: bool = True
    evidence_key: Optional[str] = None
    expected: Any = None


@dataclass
class RollbackStep:
    name: str
    description: str
    required: bool = False
    action_ref: Optional[str] = None


@dataclass
class ActionTransition:
    state: str
    ts: str
    reason: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionContract:
    contract_id: str
    intent_id: str
    session_id: str
    user_goal: str
    normalized_text: str
    primary_lane: str
    action_type: str
    target: str
    target_ref: str
    capability_name: str
    executor_name: str
    required_permissions: List[str] = field(default_factory=list)
    risk_level: str = RISK_TIER_0
    execution_mode: str = MODE_DRAFT
    requires_confirmation: bool = False
    preconditions: List[str] = field(default_factory=list)
    expected_result: str = ""
    verification_checks: List[VerificationCheck] = field(default_factory=list)
    rollback_plan: List[RollbackStep] = field(default_factory=list)
    origin: str = "unknown"
    source_surface: str = "unknown"
    trust_context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_ts: str = field(default_factory=lambda: datetime.now().isoformat())
    current_state: str = STATE_RECEIVED
    state_history: List[ActionTransition] = field(default_factory=list)

    def add_transition(self, state: str, reason: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
        self.current_state = str(state or self.current_state)
        self.state_history.append(
            ActionTransition(
                state=self.current_state,
                ts=datetime.now().isoformat(),
                reason=str(reason or ""),
                meta=dict(meta or {}),
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class ActionResult:
    contract_id: str
    audit_id: str
    state: str
    success: bool
    execution_mode: str
    executor_name: str
    summary: str
    decision: str = "UNKNOWN"
    execution_result: Dict[str, Any] = field(default_factory=dict)
    verification_result: Dict[str, Any] = field(default_factory=dict)
    rollback_result: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)
    started_ts: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_ts: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Executor base / registry
# ---------------------------------------------------------------------------
class BaseExecutor:
    name = "base_executor"
    capability = "generic"

    def prepare(self, contract: ActionContract) -> Dict[str, Any]:
        return {"ok": True, "prepared": True}

    def preflight_check(self, contract: ActionContract) -> Tuple[bool, Dict[str, Any]]:
        return True, {"ok": True, "preflight": "noop"}

    def execute(self, contract: ActionContract) -> Dict[str, Any]:
        return {"ok": False, "executed": False, "reason": "BaseExecutor cannot execute actions directly."}

    def verify(self, contract: ActionContract, execution_result: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        ok = bool(execution_result.get("ok"))
        return ok, {"ok": ok, "verification": "default_pass_through"}

    def rollback(self, contract: ActionContract, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": False, "rolled_back": False, "reason": "No rollback implemented."}

    def summarize_result(self, contract: ActionContract, result: ActionResult) -> str:
        if result.success:
            return f"Action '{contract.action_type}' completed successfully."
        return f"Action '{contract.action_type}' failed."


class NoOpExecutor(BaseExecutor):
    name = "noop_executor"
    capability = "noop"

    def execute(self, contract: ActionContract) -> Dict[str, Any]:
        return {
            "ok": True,
            "executed": False,
            "simulated": True,
            "reason": "NoOpExecutor handled a non-apply contract safely.",
        }

    def rollback(self, contract: ActionContract, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "rolled_back": False, "reason": "Nothing to rollback for noop execution."}


class SoftwareControlExecutor(BaseExecutor):
    name = "software_control_executor"
    capability = "software_interaction"

    def preflight_check(self, contract: ActionContract) -> Tuple[bool, Dict[str, Any]]:
        if _Si is None:
            return False, {"ok": False, "reason": "SarahMemorySi is unavailable."}
        if contract.action_type not in {"open_app", "close_app", "maximize_window", "minimize_window", "focus_window"}:
            return False, {"ok": False, "reason": f"Unsupported action type for software control: {contract.action_type}"}
        if not contract.target:
            return False, {"ok": False, "reason": "Missing target application."}
        return True, {"ok": True, "reason": "software_control_ready"}

    def execute(self, contract: ActionContract) -> Dict[str, Any]:
        if _Si is None:
            return {"ok": False, "executed": False, "reason": "SarahMemorySi is unavailable."}

        if contract.execution_mode != MODE_APPLY:
            return {
                "ok": True,
                "executed": False,
                "simulated": True,
                "command": f"{contract.metadata.get('surface_action', 'open')} {contract.target}",
                "reason": "Non-apply execution mode; command staged only.",
            }

        surface_action = str(contract.metadata.get("surface_action") or "open").strip().lower()
        command = f"{surface_action} {contract.target}".strip()
        try:
            fn_structured = getattr(_Si, 'execute_software_action', None)
            if callable(fn_structured):
                out = fn_structured(contract.action_type, contract.target, contract.execution_mode, metadata={**dict(contract.metadata or {}), 'surface_action': surface_action})
                if isinstance(out, dict):
                    out.setdefault('command', command)
                    return out
            fn = getattr(_Si, "manage_application_request", None)
            ok = bool(fn(command)) if callable(fn) else False
            return {"ok": ok, "executed": ok, "command": command, "reason": "software_control_executed" if ok else "software_control_failed"}
        except Exception as exc:
            return {"ok": False, "executed": False, "command": command, "reason": str(exc)}

    def verify(self, contract: ActionContract, execution_result: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        if _Si is None:
            return False, {"ok": False, "reason": "SarahMemorySi is unavailable."}

        if execution_result.get("simulated"):
            return True, {"ok": True, "verification": "simulated_only"}

        target = str(contract.target or "").strip()
        info: Dict[str, Any] = {}
        try:
            fn = getattr(_Si, 'get_application_info', None)
            if callable(fn):
                data = fn(target)
                if isinstance(data, dict):
                    info = data
        except Exception:
            info = {}

        running = bool(info.get('running') or info.get('is_running'))
        focused = bool(info.get('focused') or info.get('is_focused'))
        minimized = bool(info.get('minimized'))
        maximized = bool(info.get('maximized'))

        if contract.action_type == 'open_app':
            ok = bool(running)
            return ok, {'ok': ok, 'verification': 'application_running_check', 'reason': 'application_not_running_after_launch' if not ok else 'application_running_verified', 'info': info}
        if contract.action_type == 'close_app':
            ok = not bool(running)
            return ok, {'ok': ok, 'verification': 'application_closed_check', 'reason': 'application_still_running_after_close' if not ok else 'application_closed_verified', 'info': info}
        if contract.action_type == 'focus_window':
            ok = bool(focused)
            return ok, {'ok': ok, 'verification': 'window_focus_check', 'reason': 'window_not_focused' if not ok else 'window_focus_verified', 'info': info}
        if contract.action_type == 'minimize_window':
            ok = bool(minimized)
            return ok, {'ok': ok, 'verification': 'window_minimize_check', 'reason': 'window_not_minimized' if not ok else 'window_minimized_verified', 'info': info}
        if contract.action_type == 'maximize_window':
            ok = bool(maximized)
            return ok, {'ok': ok, 'verification': 'window_maximize_check', 'reason': 'window_not_maximized' if not ok else 'window_maximized_verified', 'info': info}

        return bool(execution_result.get('ok')), {'ok': bool(execution_result.get('ok')), 'verification': 'fallback_execution_status', 'info': info}

    def rollback(self, contract: ActionContract, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        if _Si is None:
            return {"ok": False, "rolled_back": False, "reason": "SarahMemorySi is unavailable."}

        target = str(contract.target or "").strip()
        if not target or contract.action_type != "open_app":
            return {"ok": True, "rolled_back": False, "reason": "No rollback needed for this action type."}

        try:
            fn = getattr(_Si, "manage_application_request", None)
            ok = bool(fn(f"close {target}")) if callable(fn) else False
            return {"ok": ok, "rolled_back": ok, "reason": "software_close_rollback" if ok else "software_close_rollback_failed"}
        except Exception as exc:
            return {"ok": False, "rolled_back": False, "reason": str(exc)}

    def summarize_result(self, contract: ActionContract, result: ActionResult) -> str:
        info = dict(result.verification_result.get('info') or {}) if isinstance(result.verification_result, dict) else {}
        simulated = bool((result.execution_result or {}).get('simulated'))
        if result.success and (simulated or str(result.execution_mode or '').strip().lower() != MODE_APPLY):
            return f"SMGET staged application action '{contract.action_type}' for target '{contract.target}' in {result.execution_mode} mode. No real application launch was performed."
        if result.success:
            if contract.action_type == 'open_app':
                return f"Verified application open for target '{contract.target}'."
            if contract.action_type == 'close_app':
                return f"Verified application close for target '{contract.target}'."
            if contract.action_type == 'focus_window':
                return f"Verified window focus for target '{contract.target}'."
            if contract.action_type == 'minimize_window':
                return f"Verified window minimize for target '{contract.target}'."
            if contract.action_type == 'maximize_window':
                return f"Verified window maximize for target '{contract.target}'."
            return f"Verified application action '{contract.action_type}' for target '{contract.target}'."
        if result.rollback_result.get("rolled_back"):
            return f"Application action '{contract.action_type}' failed verification and rollback was attempted for '{contract.target}'."
        reason = str((result.verification_result or {}).get('reason') or (result.execution_result or {}).get('reason') or 'verification_failed')
        if contract.action_type == 'open_app' and not bool(info.get('running') or info.get('is_running')):
            return f"Launch was requested for '{contract.target}', but SMGET could not verify that the application actually opened ({reason})."
        return f"Application action '{contract.action_type}' failed for target '{contract.target}' ({reason})."


class EmergencyInstinctExecutor(BaseExecutor):
    """Bounded executor for EmergencyInstinctActionContract packets.

    This executor makes the emergency path operational without granting raw physical
    authority. By default it stages/verifies the contract in simulate mode. Physical
    body/device dispatch only occurs when the contract explicitly carries
    allow_physical_dispatch=True and MSDC exposes a compatible dispatcher.
    """

    name = "emergency_instinct_executor"
    capability = "emergency_instinct"

    def preflight_check(self, contract: ActionContract) -> Tuple[bool, Dict[str, Any]]:
        meta = dict(contract.metadata or {})
        em_contract = meta.get("emergency_action_contract") if isinstance(meta.get("emergency_action_contract"), dict) else {}
        selected = em_contract.get("selected_action") if isinstance(em_contract.get("selected_action"), dict) else meta.get("selected_action")
        selected = selected if isinstance(selected, dict) else {}
        action_id = str(selected.get("action_id") or em_contract.get("selected_action_id") or contract.target or "").strip()
        if not action_id:
            return False, {"ok": False, "reason": "missing_emergency_selected_action"}
        if str(em_contract.get("schema") or "") != "SarahMemory.smget.emergency_action_contract.v1":
            return False, {"ok": False, "reason": "invalid_or_missing_emergency_contract_schema"}
        return True, {
            "ok": True,
            "reason": "emergency_contract_preflight_ready",
            "action_id": action_id,
            "allow_physical_dispatch": bool(em_contract.get("allow_physical_dispatch")),
        }

    def execute(self, contract: ActionContract) -> Dict[str, Any]:
        meta = dict(contract.metadata or {})
        em_contract = meta.get("emergency_action_contract") if isinstance(meta.get("emergency_action_contract"), dict) else {}
        selected = em_contract.get("selected_action") if isinstance(em_contract.get("selected_action"), dict) else {}
        action_id = str(selected.get("action_id") or em_contract.get("selected_action_id") or contract.target or "").strip()
        allow_physical = bool(em_contract.get("allow_physical_dispatch") or meta.get("allow_physical_dispatch"))

        # Default operational behavior: contract is accepted/staged, no physical actuation.
        if contract.execution_mode != MODE_APPLY or not allow_physical:
            return {
                "ok": True,
                "executed": False,
                "staged": True,
                "simulated": True,
                "action_id": action_id,
                "reason": "emergency_contract_staged; physical dispatch requires explicit allow_physical_dispatch and apply mode",
                "operator_core_active": True,
                "msdc_required_for_physical_action": True,
            }

        if _MSDC is None:
            return {
                "ok": True,
                "executed": False,
                "staged": True,
                "action_id": action_id,
                "reason": "MSDC unavailable; emergency contract retained as staged evidence",
            }

        try:
            fn = getattr(_MSDC, "msdc_dispatch_emergency_contract", None)
            if callable(fn):
                out = fn(em_contract, context={"operator_contract": contract.to_dict(), "caller": MODULE_NAME})
                return out if isinstance(out, dict) else {"ok": bool(out), "executed": bool(out), "raw_result": str(out)}
        except Exception as exc:
            return {"ok": False, "executed": False, "reason": "msdc_emergency_dispatch_exception", "error": str(exc)}

        return {
            "ok": True,
            "executed": False,
            "staged": True,
            "action_id": action_id,
            "reason": "MSDC emergency dispatcher not exposed; emergency contract retained as staged evidence",
        }

    def verify(self, contract: ActionContract, execution_result: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        ok = bool(execution_result.get("ok"))
        if execution_result.get("staged") or execution_result.get("simulated"):
            return ok, {"ok": ok, "verification": "emergency_contract_staged_and_auditable", "physical_action_verified": False}
        return ok, {"ok": ok, "verification": "emergency_dispatch_result_status", "physical_action_verified": bool(execution_result.get("executed"))}

    def rollback(self, contract: ActionContract, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        if execution_result.get("staged") or execution_result.get("simulated"):
            return {"ok": True, "rolled_back": False, "reason": "No physical action occurred; rollback not required."}
        return {"ok": False, "rolled_back": False, "reason": "Physical rollback must be handled by the owning MSDC driver."}

    def summarize_result(self, contract: ActionContract, result: ActionResult) -> str:
        action_id = str(contract.target or contract.action_type or "emergency_action")
        if result.success and (result.execution_result or {}).get("staged"):
            return f"Emergency instinct action '{action_id}' was accepted by OperatorCore and staged safely. No physical actuation was performed."
        if result.success and (result.execution_result or {}).get("executed"):
            return f"Emergency instinct action '{action_id}' was dispatched through MSDC and verified by OperatorCore result status."
        reason = str((result.execution_result or {}).get("reason") or (result.verification_result or {}).get("reason") or "verification_failed")
        return f"Emergency instinct action '{action_id}' did not complete ({reason})."


class RoboticBodyStagingExecutor(BaseExecutor):
    """Structured robotic body executor placeholder.

    This executor deliberately stages/simulates embodied actions only. Real
    actuation remains behind MSDC driver verification, SMGET authority,
    SecurityGovernor, AssuranceGate, and current perception evidence.
    """
    name = "robotic_body_staging_executor"
    capability = "robotic_body_control_staged"

    def preflight_check(self, contract: ActionContract) -> Tuple[bool, Dict[str, Any]]:
        meta = dict(contract.metadata or {})
        safety = meta.get("safety_envelope") if isinstance(meta.get("safety_envelope"), dict) else {}
        if contract.execution_mode == MODE_APPLY:
            return False, {
                "ok": False,
                "reason": "RoboticBodyStagingExecutor cannot perform apply-mode physical actuation.",
                "required_path": "verified MSDC robot driver + physical executor + SMGET apply authorization",
            }
        return True, {
            "ok": True,
            "reason": "robotic_body_staging_only",
            "safety_envelope": safety,
            "execution_authority": False,
        }

    def execute(self, contract: ActionContract) -> Dict[str, Any]:
        return {
            "ok": True,
            "executed": False,
            "simulated": True,
            "staged": True,
            "execution_authority": False,
            "reason": "Robotic body action staged only. No physical actuation occurred.",
            "body_target": contract.target,
            "safety_envelope": (contract.metadata or {}).get("safety_envelope") if isinstance((contract.metadata or {}).get("safety_envelope"), dict) else {},
            "required_before_apply": [
                "verified_body_part_driver",
                "fresh_perception_evidence",
                "safe_stop_available",
                "force_speed_torque_limits",
                "security_review",
                "assurance_review",
                "operator_audit",
            ],
        }

    def rollback(self, contract: ActionContract, execution_result: Dict[str, Any]) -> Dict[str, Any]:
        return {"ok": True, "rolled_back": False, "reason": "No physical actuation occurred; rollback not required."}


_EXECUTOR_LOCK = threading.RLock()
_EXECUTORS: Dict[str, BaseExecutor] = {
    NoOpExecutor.name: NoOpExecutor(),
    SoftwareControlExecutor.name: SoftwareControlExecutor(),
    EmergencyInstinctExecutor.name: EmergencyInstinctExecutor(),
    RoboticBodyStagingExecutor.name: RoboticBodyStagingExecutor(),
}


def register_executor(name: str, executor: BaseExecutor) -> bool:
    if not name or executor is None:
        return False
    with _EXECUTOR_LOCK:
        _EXECUTORS[str(name).strip()] = executor
    return True


def get_executor(name: str) -> Optional[BaseExecutor]:
    with _EXECUTOR_LOCK:
        return _EXECUTORS.get(str(name or "").strip())


def executor_registry_snapshot() -> Dict[str, Dict[str, Any]]:
    with _EXECUTOR_LOCK:
        return {
            k: {
                "name": getattr(v, "name", k),
                "capability": getattr(v, "capability", "unknown"),
                "type": v.__class__.__name__,
            }
            for k, v in _EXECUTORS.items()
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _flag(name: str, default: bool = False) -> bool:
    try:
        return bool(getattr(config, name, default))
    except Exception:
        return default


def _safe_mode() -> bool:
    return _flag("SAFE_MODE", False)


def _local_only() -> bool:
    return _flag("LOCAL_ONLY_MODE", False)


def _neosky_armed() -> bool:
    return _flag("NEOSKYMATRIX", False) and _flag("DEVELOPERSMODE", False)


def _registered_core_modules(capability: Optional[str] = None) -> Dict[str, Any]:
    try:
        fn = getattr(config, "sm_get_registered_core_modules", None)
        if callable(fn):
            data = fn(capability=capability) if capability is not None else fn()
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _core_module_approved(module_name: str, capability: Optional[str] = None, import_obj: Any = None) -> bool:
    key = os.path.splitext(os.path.basename(str(module_name or "").strip()))[0]
    try:
        fn = getattr(config, "sm_is_core_module_approved", None)
        if callable(fn):
            return bool(fn(key, capability=capability))
    except Exception:
        pass
    return bool(import_obj is not None)


def _governance_profile() -> Dict[str, Any]:
    try:
        fn = getattr(config, "sm_get_core_governance_profile", None)
        if callable(fn):
            data = fn()
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _session_id_from_meta(meta: Optional[Dict[str, Any]]) -> str:
    meta = dict(meta or {})
    for key in ("session_id", "sessionId", "sid", "chat_id", "chatId", "client_id", "clientId"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return uuid.uuid4().hex


def _infer_primary_lane(intent_label: str, output_type: str = "") -> str:
    t = str(intent_label or "").strip().lower()
    if t in {"creative", "creative_request", "build_request"}:
        return "creative"
    if t in {"diagnostics", "device_query", "system_control"}:
        return "system"
    if t in {"network_request", "network_access"}:
        return "network"
    if t in {"open_app", "close_quit", "window_mgmt", "play_media", "command"}:
        return "action"
    if output_type in {"action", "command"}:
        return "action"
    return "answer"


def _infer_action_type(intent_label: str, parsed_command: Optional[Dict[str, Any]], user_goal: str) -> str:
    t = str(intent_label or "").strip().lower()
    cmd = parsed_command or {}
    action = str(cmd.get("action") or "").strip().lower()
    if action in {"open", "launch", "start", "run"}:
        return "open_app"
    if action in {"close", "terminate", "quit", "exit", "kill"}:
        return "close_app"
    if action == "maximize":
        return "maximize_window"
    if action == "minimize":
        return "minimize_window"
    if action in {"focus", "bring"}:
        return "focus_window"
    if t in {"open_app"}:
        return "open_app"
    if t in {"window_mgmt"}:
        return "focus_window"
    if t in {"play_media"}:
        return "play_media"
    if t in {"diagnostics"}:
        return "run_diagnostics"
    if t in {"network_request", "network_access"}:
        return "network_operation"
    if t in {"creative", "creative_request", "build_request"}:
        return "create_artifact"
    if any(k in (user_goal or "").lower() for k in ("open ", "launch ", "start ")):
        return "open_app"
    return t or "general_action"


def _infer_target(parsed_command: Optional[Dict[str, Any]], proposed_action: Optional[Dict[str, Any]], user_goal: str) -> Tuple[str, str]:
    cmd = parsed_command or {}
    pa = proposed_action or {}
    entities = dict(pa.get('entities') or {}) if isinstance(pa.get('entities'), dict) else {}
    subject = str(pa.get('target') or entities.get('target_app_exec') or entities.get('target_app') or cmd.get("subject") or cmd.get("app_name") or cmd.get("target") or "").strip()
    if subject:
        return subject, subject
    text = str(user_goal or "").strip()
    return text[:120], text[:120]


def _looks_like_robotic_action(action_type: str, primary_lane: str, target: str) -> bool:
    joined = " ".join([str(action_type or "").lower(), str(primary_lane or "").lower(), str(target or "").lower()])
    return any(k in joined for k in ("robot", "servo", "gripper", "arm", "hand", "leg", "locomotion", "walk", "move", "posture", "balance", "torque", "force"))


def _infer_capability_and_executor(action_type: str, primary_lane: str, target: str, execution_mode: str) -> Tuple[str, str, List[str]]:
    permissions: List[str] = []
    if _looks_like_robotic_action(action_type, primary_lane, target):
        permissions = ["robotic.body.stage", "msdc.body_map.read", "safe_stop.required"]
        return "robotic_body_control_staged", RoboticBodyStagingExecutor.name, permissions
    if action_type in {"open_app", "close_app", "maximize_window", "minimize_window", "focus_window"}:
        permissions = ["app.control.local"]
        return "software_interaction", SoftwareControlExecutor.name, permissions
    if action_type == "run_diagnostics":
        permissions = ["system.diagnostics.read"]
        return "diagnostics", NoOpExecutor.name, permissions
    if primary_lane == "creative":
        permissions = ["creative.generate"]
        return "creative_lane", NoOpExecutor.name, permissions
    if primary_lane == "network":
        permissions = ["network.request.local" if _local_only() else "network.request.controlled"]
        return "network_lane", NoOpExecutor.name, permissions
    if execution_mode in {MODE_SIMULATE, MODE_DRAFT}:
        permissions = ["simulate.only"]
    return "generic_action", NoOpExecutor.name, permissions


def _infer_risk_level(action_type: str, primary_lane: str, execution_mode: str) -> str:
    if _looks_like_robotic_action(action_type, primary_lane, ""):
        return "TIER_ROBOT_OBSERVE_ONLY" if execution_mode in {MODE_SIMULATE, MODE_DRAFT} else "TIER_ROBOT_LOCOMOTION"
    if execution_mode in {MODE_SIMULATE, MODE_DRAFT}:
        return RISK_TIER_0
    if action_type in {"open_app", "close_app", "maximize_window", "minimize_window", "focus_window", "play_media"}:
        return RISK_TIER_1
    if action_type in {"run_diagnostics", "create_artifact"}:
        return RISK_TIER_2
    if primary_lane == "network":
        return RISK_TIER_4
    return RISK_TIER_2


def _default_preconditions(action_type: str, target: str, primary_lane: str) -> List[str]:
    checks: List[str] = []
    if target:
        checks.append("target_present")
    if action_type in {"open_app", "close_app", "maximize_window", "minimize_window", "focus_window"}:
        checks.append("executor_registered")
        checks.append("app_surface_allowed")
    if primary_lane == "network":
        checks.append("network_policy_allows")
    if _looks_like_robotic_action(action_type, primary_lane, target):
        checks.extend(["msdc_body_map_present", "smget_authority_required", "safe_stop_path_declared", "fresh_perception_required_before_apply"])
    if _safe_mode():
        checks.append("safe_mode_considered")
    return checks


def _default_verification_checks(action_type: str, target: str, execution_mode: str) -> List[VerificationCheck]:
    if execution_mode in {MODE_SIMULATE, MODE_DRAFT}:
        return [VerificationCheck(name="staged_only", description="Execution remained in a non-apply mode.", expected=True)]
    if _looks_like_robotic_action(action_type, "", target):
        return [
            VerificationCheck(name="safe_stop_available", description="Confirm a safe-stop path exists before physical robot execution.", evidence_key="safe_stop_available", expected=True),
            VerificationCheck(name="fresh_perception", description="Confirm current perception evidence is fresh before robot motion.", evidence_key="current_perception_fresh", expected=True),
            VerificationCheck(name="body_driver_verified", description="Confirm target robot body-part driver is verified.", evidence_key="body_part_driver_verified", expected=True),
        ]
    if action_type == "open_app":
        return [
            VerificationCheck(name="target_launched", description=f"Confirm that '{target}' is running or reachable.", evidence_key="running", expected=True),
        ]
    if action_type == "close_app":
        return [
            VerificationCheck(name="target_closed", description=f"Confirm that '{target}' is no longer running.", evidence_key="running", expected=False),
        ]
    return [VerificationCheck(name="executor_reported_success", description="Executor must report success.", evidence_key="ok", expected=True)]


def _default_rollback_plan(action_type: str, target: str, execution_mode: str) -> List[RollbackStep]:
    if execution_mode in {MODE_SIMULATE, MODE_DRAFT}:
        return [RollbackStep(name="noop", description="No rollback required for staged-only execution.")]
    if _looks_like_robotic_action(action_type, "", target):
        return [RollbackStep(name="safe_stop", description="Execute physical safe-stop through MSDC if robot actuation begins.", required=True, action_ref="safe_stop")]
    if action_type == "open_app":
        return [RollbackStep(name="close_target", description=f"Attempt to close '{target}' if launch succeeds but verification fails.", required=False, action_ref="close_app")]
    return [RollbackStep(name="report_only", description="No structured rollback available; report bounded failure.", required=False)]


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"


# ---------------------------------------------------------------------------
# Operator core
# ---------------------------------------------------------------------------
class SarahMemoryOperatorCore:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        _ensure_tables()

    def _transition(self, contract: ActionContract, state: str, reason: str, *, meta: Optional[Dict[str, Any]] = None) -> None:
        contract.add_transition(state, reason, meta=meta)
        log_operator_event(contract.contract_id, state, "state_transition", reason, meta=meta)

    def build_action_contract(
        self,
        user_goal: str,
        *,
        origin: str = "unknown",
        meta: Optional[Dict[str, Any]] = None,
        proposed_action: Optional[Dict[str, Any]] = None,
        execution_mode: str = MODE_DRAFT,
    ) -> ActionContract:
        meta = dict(meta or {})
        proposed_action = dict(proposed_action or {})
        mode = str(execution_mode or meta.get("execution_mode") or MODE_DRAFT).strip().lower()
        if mode not in _ALLOWED_EXECUTION_MODES:
            mode = MODE_DRAFT

        parsed_command: Dict[str, Any] = {}
        intent_label = "general"
        output_type = ""
        surface_action = "open"

        if _AdvCU is not None:
            try:
                fn_intent = getattr(_AdvCU, "classify_intent", None)
                if callable(fn_intent):
                    intent_label = str(fn_intent(user_goal) or intent_label)
            except Exception:
                pass
            try:
                fn_parse = getattr(_AdvCU, "parse_command", None)
                if callable(fn_parse):
                    parsed = fn_parse(user_goal)
                    if hasattr(parsed, "__dict__"):
                        parsed_command = {k: v for k, v in vars(parsed).items() if not k.startswith("_")}
                    elif isinstance(parsed, dict):
                        parsed_command = dict(parsed)
            except Exception:
                parsed_command = {}
            try:
                fn_packet = getattr(_AdvCU, "build_semantic_packet", None)
                if callable(fn_packet):
                    pkt = fn_packet(user_goal)
                    if isinstance(pkt, dict):
                        output_type = str(pkt.get("output_type") or output_type)
                        meta.setdefault("semantic_packet", pkt)
            except Exception:
                pass

        surface_action = str(parsed_command.get("action") or proposed_action.get("action") or surface_action).strip().lower() or "open"
        primary_lane = _infer_primary_lane(intent_label, output_type)
        action_type = str(proposed_action.get('action_type') or _infer_action_type(intent_label, parsed_command, user_goal) or 'general_action').strip().lower()
        target, target_ref = _infer_target(parsed_command, proposed_action, user_goal)
        if action_type in {'open_app', 'close_app', 'maximize_window', 'minimize_window', 'focus_window'} and _Si is not None:
            try:
                norm_fn = getattr(_Si, '_normalize_app_alias', None)
                if callable(norm_fn):
                    normalized_target = str(norm_fn(target) or target).strip()
                    if normalized_target:
                        target = normalized_target
                        target_ref = normalized_target
            except Exception:
                pass
        capability_name, executor_name, permissions = _infer_capability_and_executor(action_type, primary_lane, target, mode)
        risk_level = _infer_risk_level(action_type, primary_lane, mode)

        contract = ActionContract(
            contract_id=f"act_{uuid.uuid4().hex}",
            intent_id=f"intent_{uuid.uuid4().hex}",
            session_id=_session_id_from_meta(meta),
            user_goal=str(user_goal or "").strip(),
            normalized_text=str(user_goal or "").strip(),
            primary_lane=primary_lane,
            action_type=action_type,
            target=target,
            target_ref=target_ref,
            capability_name=capability_name,
            executor_name=executor_name,
            required_permissions=permissions,
            risk_level=risk_level,
            execution_mode=mode,
            requires_confirmation=bool(risk_level in {RISK_TIER_3, RISK_TIER_4}),
            preconditions=_default_preconditions(action_type, target, primary_lane),
            expected_result=f"Action '{action_type}' should complete for target '{target}'.",
            verification_checks=_default_verification_checks(action_type, target, mode),
            rollback_plan=_default_rollback_plan(action_type, target, mode),
            origin=str(origin or meta.get("source") or "unknown"),
            source_surface=str(meta.get("source_surface") or meta.get("surface") or origin or "unknown"),
            trust_context={
                "safe_mode": _safe_mode(),
                "local_only": _local_only(),
                "neosky_armed": _neosky_armed(),
                "governance_profile": _governance_profile(),
                "registered_modules": list(_registered_core_modules().keys())[:128],
            },
            metadata={
                "intent_label": intent_label,
                "parsed_command": parsed_command,
                "proposed_action": proposed_action,
                "surface_action": surface_action,
                "operator_family": "SMGET",
            },
        )
        self._transition(contract, STATE_CLASSIFIED, "ActionContract built from user goal.", meta={"intent": intent_label, "lane": primary_lane})
        return contract

    def _govern_contract(self, contract: ActionContract) -> Dict[str, Any]:
        if _CogServices is None:
            return {
                "decision": "ALLOW",
                "allow": True,
                "require_user": False,
                "risk_score": 0,
                "risk_factors": [],
                "reasons": ["CognitiveServices unavailable; fail-soft local allowance."],
                "recommended_next": "Proceed with bounded operator execution.",
            }

        try:
            fn = getattr(_CogServices, "govern_request", None)
            if callable(fn):
                return fn(
                    contract.user_goal,
                    caller=MODULE_NAME,
                    caller_context={
                        "run_mode": getattr(config, "RUN_MODE", "local") if config else "local",
                        "device_mode": getattr(config, "DEVICE_MODE", "headless") if config else "headless",
                        "safe_mode": _safe_mode(),
                        "local_only": _local_only(),
                        "session_id": contract.session_id,
                        "source_surface": contract.source_surface,
                        "mode_flags": {
                            "SAFE_MODE": _safe_mode(),
                            "LOCAL_ONLY_MODE": _local_only(),
                            "NEOSKYMATRIX": _flag("NEOSKYMATRIX", False),
                            "DEVELOPERSMODE": _flag("DEVELOPERSMODE", False),
                            "RUN_MODE": getattr(config, "RUN_MODE", "local") if config else "local",
                            "DEVICE_MODE": getattr(config, "DEVICE_MODE", "headless") if config else "headless",
                        },
                    },
                    user_present=True,
                    user_consented=not contract.requires_confirmation,
                    proposed_action={
                        "action_type": contract.action_type,
                        "executor_name": contract.executor_name,
                        "capability_name": contract.capability_name,
                        "risk_level": contract.risk_level,
                        "required_permissions": contract.required_permissions,
                        "target": contract.target,
                        "execution_mode": contract.execution_mode,
                    },
                )
        except Exception as exc:
            logger.debug("Govern request failed: %s", exc)

        return {
            "decision": "DEFER",
            "allow": False,
            "require_user": True,
            "risk_score": 50,
            "risk_factors": ["governance_unavailable"],
            "reasons": ["Governance response unavailable."],
            "recommended_next": "Request clarification or retry through governed path.",
        }

    def _security_gate(self, contract: ActionContract, governance: Dict[str, Any]) -> Dict[str, Any]:
        if _SecurityGovernor is None:
            return {
                "decision": "ALLOW",
                "allow": True,
                "reasons": ["SecurityGovernor not present yet; fail-soft to governance result."],
                "risk_score": governance.get("risk_score", 0),
            }
        for fn_name in ("evaluate_action", "govern_action", "review_action_contract"):
            try:
                fn = getattr(_SecurityGovernor, fn_name, None)
                if callable(fn):
                    out = fn(contract.to_dict(), governance)
                    if isinstance(out, dict):
                        return out
            except Exception as exc:
                logger.debug("SecurityGovernor.%s failed: %s", fn_name, exc)
        return {
            "decision": "ALLOW",
            "allow": True,
            "reasons": ["SecurityGovernor present but no compatible callable exposed yet."],
            "risk_score": governance.get("risk_score", 0),
        }

    def _assurance_gate(self, contract: ActionContract, governance: Dict[str, Any], security: Dict[str, Any]) -> Dict[str, Any]:
        if _AssuranceGate is None:
            return {
                "allow": True,
                "confidence": 1.0,
                "reasons": ["AssuranceGate not present yet; fallback to governance/security decisions."],
            }
        for fn_name in ("evaluate_action_assurance", "review_action_assurance", "assure_action"):
            try:
                fn = getattr(_AssuranceGate, fn_name, None)
                if callable(fn):
                    out = fn(contract.to_dict(), governance, security)
                    if isinstance(out, dict):
                        return out
            except Exception as exc:
                logger.debug("AssuranceGate.%s failed: %s", fn_name, exc)
        return {
            "allow": True,
            "confidence": 1.0,
            "reasons": ["AssuranceGate present but no compatible callable exposed yet."],
        }

    def _persist_action(self, contract: ActionContract, result: ActionResult) -> None:
        con = None
        try:
            _ensure_tables()
            con = _connect_db()
            cur = con.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO operator_actions (
                    contract_id, ts, state, execution_mode, primary_lane, action_type, target,
                    executor_name, risk_level, success, audit_id, contract_json, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    contract.contract_id,
                    datetime.now().isoformat(),
                    contract.current_state,
                    contract.execution_mode,
                    contract.primary_lane,
                    contract.action_type,
                    contract.target,
                    contract.executor_name,
                    contract.risk_level,
                    1 if result.success else 0,
                    result.audit_id,
                    _safe_json(contract.to_dict()),
                    _safe_json(result.to_dict()),
                ),
            )
            con.commit()
        except Exception as exc:
            logger.debug("OperatorCore persist failed: %s", exc)
        finally:
            try:
                if con:
                    con.close()
            except Exception:
                pass

    def process_contract(self, contract: ActionContract) -> ActionResult:
        audit_id = f"audit_{uuid.uuid4().hex}"
        started_ts = datetime.now().isoformat()
        errors: List[str] = []
        warnings: List[str] = []
        trace: Dict[str, Any] = {
            "governance_profile": _governance_profile(),
            "executor_registry": executor_registry_snapshot(),
        }

        governance = self._govern_contract(contract)
        trace["governance"] = governance
        if not bool(governance.get("allow", governance.get("decision") == "ALLOW")):
            self._transition(contract, STATE_FAILED, "Governance denied or deferred action.", meta=governance)
            result = ActionResult(
                contract_id=contract.contract_id,
                audit_id=audit_id,
                state=contract.current_state,
                success=False,
                execution_mode=contract.execution_mode,
                executor_name=contract.executor_name,
                summary="Governance blocked action execution.",
                decision=str(governance.get("decision") or "DENY"),
                errors=[str(x) for x in (governance.get("reasons") or [])],
                trace=trace,
                started_ts=started_ts,
                completed_ts=datetime.now().isoformat(),
            )
            self._persist_action(contract, result)
            return result
        self._transition(contract, STATE_AUTHORIZED, "Governance authorized action.", meta=governance)

        security = self._security_gate(contract, governance)
        trace["security"] = security
        if not bool(security.get("allow", security.get("decision") == "ALLOW")):
            self._transition(contract, STATE_FAILED, "SecurityGovernor blocked action.", meta=security)
            result = ActionResult(
                contract_id=contract.contract_id,
                audit_id=audit_id,
                state=contract.current_state,
                success=False,
                execution_mode=contract.execution_mode,
                executor_name=contract.executor_name,
                summary="Security gate blocked action execution.",
                decision=str(security.get("decision") or "DENY"),
                errors=[str(x) for x in (security.get("reasons") or [])],
                trace=trace,
                started_ts=started_ts,
                completed_ts=datetime.now().isoformat(),
            )
            self._persist_action(contract, result)
            return result

        assurance = self._assurance_gate(contract, governance, security)
        trace["assurance"] = assurance
        if not bool(assurance.get("allow", True)):
            self._transition(contract, STATE_FAILED, "Assurance gate blocked action.", meta=assurance)
            result = ActionResult(
                contract_id=contract.contract_id,
                audit_id=audit_id,
                state=contract.current_state,
                success=False,
                execution_mode=contract.execution_mode,
                executor_name=contract.executor_name,
                summary="Assurance gate blocked action execution.",
                decision="ASSURANCE_DENY",
                errors=[str(x) for x in (assurance.get("reasons") or [])],
                trace=trace,
                started_ts=started_ts,
                completed_ts=datetime.now().isoformat(),
            )
            self._persist_action(contract, result)
            return result

        executor = get_executor(contract.executor_name)
        if executor is None:
            self._transition(contract, STATE_FAILED, "No executor registered for contract.", meta={"executor_name": contract.executor_name})
            result = ActionResult(
                contract_id=contract.contract_id,
                audit_id=audit_id,
                state=contract.current_state,
                success=False,
                execution_mode=contract.execution_mode,
                executor_name=contract.executor_name,
                summary="No executor available for requested action.",
                decision="NO_EXECUTOR",
                errors=[f"Executor not registered: {contract.executor_name}"],
                trace=trace,
                started_ts=started_ts,
                completed_ts=datetime.now().isoformat(),
            )
            self._persist_action(contract, result)
            return result

        self._transition(contract, STATE_PLANNED, "Executor selected and plan prepared.", meta={"executor_name": contract.executor_name})
        try:
            prep = executor.prepare(contract)
        except Exception as exc:
            prep = {"ok": False, "reason": str(exc)}
        trace["prepare"] = prep

        ok_preflight, preflight = executor.preflight_check(contract)
        trace["preflight"] = preflight
        if not ok_preflight:
            self._transition(contract, STATE_FAILED, "Preflight check failed.", meta=preflight)
            result = ActionResult(
                contract_id=contract.contract_id,
                audit_id=audit_id,
                state=contract.current_state,
                success=False,
                execution_mode=contract.execution_mode,
                executor_name=contract.executor_name,
                summary="Preflight check failed.",
                decision="PREFLIGHT_FAIL",
                errors=[str(preflight.get("reason") or "preflight_failed")],
                trace=trace,
                started_ts=started_ts,
                completed_ts=datetime.now().isoformat(),
            )
            self._persist_action(contract, result)
            return result

        self._transition(contract, STATE_DISPATCHED, "Action dispatched to executor.", meta={"executor_name": contract.executor_name})
        self._transition(contract, STATE_EXECUTING, "Executor started bounded execution.")
        execution_result: Dict[str, Any]
        try:
            execution_result = executor.execute(contract)
        except Exception as exc:
            execution_result = {"ok": False, "executed": False, "reason": str(exc)}
        trace["execution"] = execution_result

        verified, verification_result = executor.verify(contract, execution_result)
        trace["verification"] = verification_result

        if verified:
            self._transition(contract, STATE_VERIFIED, "Verification passed.", meta=verification_result)
            self._transition(contract, STATE_SUCCEEDED, "Action completed successfully.")
            result = ActionResult(
                contract_id=contract.contract_id,
                audit_id=audit_id,
                state=contract.current_state,
                success=True,
                execution_mode=contract.execution_mode,
                executor_name=contract.executor_name,
                summary="",
                decision="ALLOW",
                execution_result=execution_result,
                verification_result=verification_result,
                warnings=warnings,
                trace=trace,
                started_ts=started_ts,
                completed_ts=datetime.now().isoformat(),
            )
            result.summary = executor.summarize_result(contract, result)
            self._persist_action(contract, result)
            return result

        errors.append(str(verification_result.get("reason") or verification_result.get("verification") or "verification_failed"))
        self._transition(contract, STATE_FAILED, "Verification failed.", meta=verification_result)

        rollback_result = {"ok": False, "rolled_back": False, "reason": "rollback_not_attempted"}
        if contract.rollback_plan:
            try:
                rollback_result = executor.rollback(contract, execution_result)
            except Exception as exc:
                rollback_result = {"ok": False, "rolled_back": False, "reason": str(exc)}
            trace["rollback"] = rollback_result
            if rollback_result.get("rolled_back"):
                self._transition(contract, STATE_ROLLED_BACK, "Rollback completed.", meta=rollback_result)
            else:
                warnings.append(str(rollback_result.get("reason") or "rollback_failed"))

        result = ActionResult(
            contract_id=contract.contract_id,
            audit_id=audit_id,
            state=contract.current_state,
            success=False,
            execution_mode=contract.execution_mode,
            executor_name=contract.executor_name,
            summary="",
            decision="VERIFY_FAIL",
            execution_result=execution_result,
            verification_result=verification_result,
            rollback_result=rollback_result,
            errors=errors,
            warnings=warnings,
            trace=trace,
            started_ts=started_ts,
            completed_ts=datetime.now().isoformat(),
        )
        result.summary = executor.summarize_result(contract, result)
        self._persist_action(contract, result)
        return result

    def process_action_request(
        self,
        user_goal: str,
        *,
        origin: str = "unknown",
        meta: Optional[Dict[str, Any]] = None,
        proposed_action: Optional[Dict[str, Any]] = None,
        execution_mode: str = MODE_DRAFT,
    ) -> Dict[str, Any]:
        contract = self.build_action_contract(
            user_goal,
            origin=origin,
            meta=meta,
            proposed_action=proposed_action,
            execution_mode=execution_mode,
        )
        result = self.process_contract(contract)
        self._transition(contract, STATE_REPORTED, "Result packaged for caller.", meta={"success": result.success, "audit_id": result.audit_id})
        self._persist_action(contract, result)
        return {
            "ok": result.success,
            "contract": contract.to_dict(),
            "result": result.to_dict(),
        }


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------
_CORE_SINGLETON: Optional[SarahMemoryOperatorCore] = None
_CORE_LOCK = threading.RLock()


def get_operator_core() -> SarahMemoryOperatorCore:
    global _CORE_SINGLETON
    with _CORE_LOCK:
        if _CORE_SINGLETON is None:
            _CORE_SINGLETON = SarahMemoryOperatorCore()
        return _CORE_SINGLETON


def build_action_contract(
    user_goal: str,
    *,
    origin: str = "unknown",
    meta: Optional[Dict[str, Any]] = None,
    proposed_action: Optional[Dict[str, Any]] = None,
    execution_mode: str = MODE_DRAFT,
) -> ActionContract:
    return get_operator_core().build_action_contract(
        user_goal,
        origin=origin,
        meta=meta,
        proposed_action=proposed_action,
        execution_mode=execution_mode,
    )


def process_action_request(
    user_goal: str,
    *,
    origin: str = "unknown",
    meta: Optional[Dict[str, Any]] = None,
    proposed_action: Optional[Dict[str, Any]] = None,
    execution_mode: str = MODE_DRAFT,
) -> Dict[str, Any]:
    return get_operator_core().process_action_request(
        user_goal,
        origin=origin,
        meta=meta,
        proposed_action=proposed_action,
        execution_mode=execution_mode,
    )


def _coerce_emergency_contract_to_action_contract(contract: Dict[str, Any]) -> ActionContract:
    c = dict(contract or {})
    selected = c.get("selected_action") if isinstance(c.get("selected_action"), dict) else {}
    action_id = str(selected.get("action_id") or c.get("selected_action_id") or c.get("target") or c.get("hazard_type") or "emergency_action").strip()
    mode = MODE_APPLY if bool(c.get("allow_physical_dispatch")) else MODE_SIMULATE
    risk = str(c.get("risk_level") or RISK_TIER_2)
    return ActionContract(
        contract_id=str(c.get("contract_id") or "emcontract-" + uuid.uuid4().hex[:12]),
        intent_id=str(c.get("incident_id") or uuid.uuid4().hex),
        session_id=str(c.get("incident_id") or uuid.uuid4().hex),
        user_goal=str(selected.get("title") or f"Emergency instinct action: {action_id}"),
        normalized_text=str(selected.get("title") or action_id).strip().lower(),
        primary_lane="action",
        action_type="emergency_instinct",
        target=action_id,
        target_ref=str(c.get("hazard_type") or action_id),
        capability_name="emergency_instinct",
        executor_name=EmergencyInstinctExecutor.name,
        required_permissions=["emergency.instinct.stage"] + (["msdc.physical.dispatch"] if mode == MODE_APPLY else []),
        risk_level=risk,
        execution_mode=mode,
        requires_confirmation=bool(c.get("requires_user") and not c.get("user_delay_override")),
        preconditions=["emergency_contract_schema_valid", "selected_action_present", "operator_core_choke_point"],
        expected_result="Emergency action contract is staged or dispatched through MSDC when explicitly allowed.",
        verification_checks=[VerificationCheck(name="emergency_contract_auditable", description="Emergency contract must remain auditable and bounded.", expected=True)],
        rollback_plan=[RollbackStep(name="emergency_safe_stop_or_report", description=str(c.get("rollback_plan") or "Stop/report/escalate; no silent rollback."), required=False)],
        origin="CognitiveServices.EmergencyInstinct",
        source_surface="living_loop",
        trust_context={"source": "CognitiveServices", "contract_schema": c.get("schema"), "decision": c.get("decision")},
        metadata={
            "emergency_action_contract": c,
            "selected_action": selected,
            "allow_physical_dispatch": bool(c.get("allow_physical_dispatch")),
            "bounded_action_allowed": bool(c.get("bounded_action_allowed")),
            "execution_authority": False,
        },
    )


def process_action_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Process a prebuilt governed action contract through OperatorCore.

    Currently supports EmergencyInstinctActionContract packets from the Cognitive
    Living Loop. This gives CognitiveServices a stable SMGET entry point without
    bypassing the OperatorCore lifecycle, verification, or audit database.
    """
    c = dict(contract or {}) if isinstance(contract, dict) else {}
    if str(c.get("schema") or "") == "SarahMemory.smget.emergency_action_contract.v1" or str(c.get("contract_type") or "") == "EmergencyInstinctActionContract":
        core = get_operator_core()
        action_contract = _coerce_emergency_contract_to_action_contract(c)
        result = core.process_contract(action_contract)
        core._transition(action_contract, STATE_REPORTED, "Emergency contract result packaged for caller.", meta={"success": result.success, "audit_id": result.audit_id})
        core._persist_action(action_contract, result)
        result_dict = result.to_dict()
        execution_result = result_dict.get("execution_result") if isinstance(result_dict.get("execution_result"), dict) else {}
        return {
            "ok": bool(result.success),
            "executed": bool(execution_result.get("executed")),
            "attempted": True,
            "function": "process_action_contract",
            "contract": action_contract.to_dict(),
            "result": result_dict,
            "reason": execution_result.get("reason") or result.summary,
        }
    return {
        "ok": False,
        "executed": False,
        "attempted": False,
        "reason": "unsupported_action_contract_schema",
        "schema": c.get("schema"),
        "contract_type": c.get("contract_type"),
    }


# Backward-compatible aliases for contract-oriented callers.
process_emergency_action_contract = process_action_contract
execute_action_contract = process_action_contract
operator_execute_contract = process_action_contract
dispatch_action_contract = process_action_contract


# ---------------------------------------------------------------------------
# Self-test / smoke check
# ---------------------------------------------------------------------------
def _run_self_test() -> bool:
    try:
        core = get_operator_core()
        out = core.process_action_request(
            "open notepad",
            origin="self_test",
            meta={"surface": "diagnostics", "session_id": "self_test_session"},
            execution_mode=MODE_SIMULATE,
        )
        return bool(out.get("ok") and isinstance(out.get("contract"), dict) and isinstance(out.get("result"), dict))
    except Exception as exc:
        logger.error("OperatorCore self-test failed: %s", exc)
        return False


def main() -> int:
    ok = _run_self_test()
    print(json.dumps({"ok": ok, "module": MODULE_NAME, "version": MODULE_VERSION}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())


# -----------------------------------------------------------------------------
# V10/V9F SMGET read-only evidence contract review
# -----------------------------------------------------------------------------
def review_read_only_evidence_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Validate read-only evidence/claim contracts without executing actions."""
    c = dict(contract or {})
    read_only = bool(c.get('read_only', True)); state_change = bool(c.get('state_change', False))
    mode = str(c.get('execution_mode') or MODE_SIMULATE).lower(); risk = str(c.get('risk_level') or RISK_TIER_0)
    reasons = []; allow = True; decision = 'ALLOW'
    if not read_only or state_change:
        allow = False; decision = 'DENY'; reasons.append('SMGET read-only evidence contract cannot change state.')
    if mode not in {MODE_SIMULATE, MODE_DRAFT}:
        allow = False; decision = 'DENY'; reasons.append('Read-only evidence contract must use simulate or draft mode.')
    if risk not in {RISK_TIER_0, RISK_TIER_1}:
        allow = False; decision = 'REQUIRE_USER'; reasons.append('Read-only evidence claim used non-trivial risk tier.')
    hard = c.get('hard_evidence_packet') if isinstance(c.get('hard_evidence_packet'), dict) else {}
    if not hard:
        allow = False; decision = 'SIMULATE_ONLY'; reasons.append('No hard evidence packet supplied.')
    if allow:
        reasons.append('SMGET validated read-only evidence contract; no actuator, file, network, driver, or OS mutation authorized.')
    review = {'ok': True, 'module': MODULE_NAME, 'decision': decision, 'allow': bool(allow), 'execution_mode': mode, 'risk_level': risk, 'read_only': read_only, 'state_change': state_change, 'contract': c, 'reasons': reasons}
    try:
        log_operator_event(str(c.get('contract_id') or 'readonly-evidence'), STATE_VERIFIED if allow else STATE_FAILED, 'ReadOnlyEvidenceReview', '; '.join(reasons), meta=review)
    except Exception:
        pass
    return review

# --- SM V8.0 TRI-LAYER PATCH 2026-05-20 ---
def attach_tri_layer_packets_to_meta(meta: Optional[Dict[str, Any]], tri_layer_packet: Optional[Dict[str, Any]] = None, six_question_packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Attach packet evidence to ActionContract metadata without granting execution authority."""
    out = dict(meta or {})
    if isinstance(tri_layer_packet, dict):
        out["tri_layer_input_packet"] = tri_layer_packet
    if isinstance(six_question_packet, dict):
        out["six_question_governance_packet"] = six_question_packet
    out["packet_metadata_is_evidence_only"] = True
    return out

# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 START ---
# Durable workflow/checkpoint helpers. These extend OperatorCore's lifecycle but
# do not start background workers or execute tasks.

WORKFLOW_PROPOSED = "PROPOSED"
WORKFLOW_PLANNED = "PLANNED"
WORKFLOW_WAITING_USER = "WAITING_USER"
WORKFLOW_APPROVED = "APPROVED"
WORKFLOW_EXECUTING = "EXECUTING"
WORKFLOW_VERIFYING = "VERIFYING"
WORKFLOW_COMPLETED = "COMPLETED"
WORKFLOW_FAILED = "FAILED"
WORKFLOW_ROLLED_BACK = "ROLLED_BACK"
WORKFLOW_CANCELLED = "CANCELLED"

_ALLOWED_WORKFLOW_TRANSITIONS = {
    WORKFLOW_PROPOSED: {WORKFLOW_PLANNED, WORKFLOW_WAITING_USER, WORKFLOW_CANCELLED},
    WORKFLOW_PLANNED: {WORKFLOW_WAITING_USER, WORKFLOW_APPROVED, WORKFLOW_CANCELLED},
    WORKFLOW_WAITING_USER: {WORKFLOW_APPROVED, WORKFLOW_CANCELLED},
    WORKFLOW_APPROVED: {WORKFLOW_EXECUTING, WORKFLOW_CANCELLED},
    WORKFLOW_EXECUTING: {WORKFLOW_VERIFYING, WORKFLOW_FAILED, WORKFLOW_ROLLED_BACK},
    WORKFLOW_VERIFYING: {WORKFLOW_COMPLETED, WORKFLOW_FAILED, WORKFLOW_ROLLED_BACK},
    WORKFLOW_FAILED: {WORKFLOW_ROLLED_BACK, WORKFLOW_CANCELLED},
    WORKFLOW_ROLLED_BACK: {WORKFLOW_REPORTED if 'WORKFLOW_REPORTED' in globals() else WORKFLOW_CANCELLED},
}

@dataclass
class DurableWorkflowCheckpoint:
    workflow_id: str
    state: str
    goal: str = ""
    task_id: str = ""
    checkpoint_ts: str = field(default_factory=lambda: datetime.now().isoformat())
    compact_state: Dict[str, Any] = field(default_factory=dict)
    write_policy: Dict[str, Any] = field(default_factory=lambda: {"ram_first": True, "write_on_state_boundary_only": True})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkflowStateMachine:
    def __init__(self, workflow_id: str, initial_state: str = WORKFLOW_PROPOSED) -> None:
        self.workflow_id = str(workflow_id or uuid.uuid4().hex)
        self.state = str(initial_state or WORKFLOW_PROPOSED)
        self.history: List[Dict[str, Any]] = []

    def can_transition(self, new_state: str) -> bool:
        if self.state == new_state:
            return True
        allowed = _ALLOWED_WORKFLOW_TRANSITIONS.get(self.state, set())
        return new_state in allowed

    def transition(self, new_state: str, reason: str = "") -> Dict[str, Any]:
        ns = str(new_state or "")
        ok = self.can_transition(ns)
        rec = {"ts": datetime.now().isoformat(), "from": self.state, "to": ns, "ok": ok, "reason": reason}
        if ok:
            self.state = ns
        self.history.append(rec)
        return {"ok": ok, "workflow_id": self.workflow_id, "state": self.state, "transition": rec, "history": list(self.history)}


def create_durable_workflow_checkpoint(workflow_id: str, state: str, *, goal: str = "", task_id: str = "", compact_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cp = DurableWorkflowCheckpoint(
        workflow_id=str(workflow_id or uuid.uuid4().hex),
        state=str(state or WORKFLOW_PROPOSED),
        goal=str(goal or "")[:500],
        task_id=str(task_id or ""),
        compact_state=dict(compact_state or {}),
    )
    return {"ok": True, "checkpoint": cp.to_dict(), "persisted": False, "note": "Checkpoint object built; persistence must be explicit/batched."}


def review_workflow_transition(workflow_id: str, current_state: str, requested_state: str, reason: str = "") -> Dict[str, Any]:
    sm = WorkflowStateMachine(workflow_id, initial_state=current_state)
    return sm.transition(requested_state, reason=reason)
# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 END ---
