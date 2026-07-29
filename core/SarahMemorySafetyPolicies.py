"""--==The SarahMemory Project==--
File: SarahMemorySafetyPolicies.py
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

- Constitutional safety-policy source for SarahMemory AiOS / SMGET.
- Provides runtime safety snapshots, bounded action policy evaluation, and doctrine-
level defaults used by the governed execution stack.
- Keeps policy definitions centralized so SecurityGovernor, OperatorCore, future
AssuranceGate, and other bounded execution surfaces do not hardcode scattered
safety rules throughout the codebase.
- Encodes the doctrine that SarahMemory may grow and adapt, but must not degrade
governance, user sovereignty, auditability, or operational safety.

CORE DOCTRINE:
- User remains final authority.
- No hallucinated actuation.
- No untrusted model direct control.
- No hidden capability activation.
- No unsafe public exposure by default.
- Riskier actions require stronger assurance and explicit confirmation.
- Physical-world safety outranks convenience.
- Rollback must exist wherever practical.
- Discovery is not activation.
- Fail closed when policy is uncertain.

RELATIONSHIP MODEL:
- SarahMemoryGlobals.py           -> runtime flags / environment identity / base governance
- SarahMemoryCognitiveServices.py -> logical/risk governor
- SarahMemorySecurityGovernor.py  -> trust / sovereignty / integrity hard gate
- SarahMemorySafetyPolicies.py    -> canonical safety policy source and runtime snapshot
- SarahMemoryOperatorCore.py      -> execution lifecycle / verification / rollback
- SarahMemoryAssuranceGate.py     -> confidence/assurance enforcement (future)
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "safety_policy_core"
# CATEGORY = "runtime_safety_and_policy"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "safety_policies"
# HARDWARE_DOMAIN = "system_network_filesystem_devices"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "safety_policies"
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
# NOTES = "SMGET constitutional safety-policy source. Centralizes runtime safety defaults, confirmation requirements, bounded execution rules, and fail-closed policy evaluation."
# --- SARAHMETA END ---

import json
import logging
import os
import sqlite3
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Safe imports (never hard-fail the platform)
# ---------------------------------------------------------------------------
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None

try:
    import SarahMemoryEnergetics as _Energetics  # type: ignore
except Exception:
    _Energetics = None  # type: ignore


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("SarahMemorySafetyPolicies")
logger.setLevel(logging.DEBUG if bool(getattr(config, "DEBUG_MODE", True)) else logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.propagate = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODULE_NAME = "SarahMemorySafetyPolicies"
MODULE_VERSION = "9.0.0"
_DB_NAME = "safety_policies.db"
_JSON_NAME = "safety_policy_snapshot.json"

MODE_SIMULATE = "simulate"
MODE_DRAFT = "draft"
MODE_APPLY = "apply"
MODE_ROLLBACK = "rollback"

RISK_TIER_0 = "TIER_0_INFO"
RISK_TIER_1 = "TIER_1_HARMLESS_LOCAL_UI"
RISK_TIER_2 = "TIER_2_BOUNDED_LOCAL_OPERATION"
RISK_TIER_3 = "TIER_3_PRIVILEGED_SYSTEM"
RISK_TIER_4 = "TIER_4_NETWORK_REMOTE_OR_DESTRUCTIVE"

ROBOT_TIER_OBSERVE = "TIER_ROBOT_OBSERVE_ONLY"
ROBOT_TIER_FACE = "TIER_ROBOT_FACE_EXPRESSION"
ROBOT_TIER_HEAD = "TIER_ROBOT_HEAD_MOVEMENT"
ROBOT_TIER_POSTURE = "TIER_ROBOT_POSTURE"
ROBOT_TIER_ARM = "TIER_ROBOT_ARM_LOW_FORCE"
ROBOT_TIER_GRIP = "TIER_ROBOT_GRIP_OBJECT"
ROBOT_TIER_LOCOMOTION = "TIER_ROBOT_LOCOMOTION"
ROBOT_TIER_HUMAN_CONTACT = "TIER_ROBOT_HUMAN_CONTACT"
ROBOT_TIER_EMERGENCY = "TIER_ROBOT_EMERGENCY_INTERVENTION"
ROBOT_TIER_ESTOP = "TIER_ROBOT_EMERGENCY_STOP"

ROBOT_RISK_TIERS = {
    ROBOT_TIER_OBSERVE, ROBOT_TIER_FACE, ROBOT_TIER_HEAD, ROBOT_TIER_POSTURE,
    ROBOT_TIER_ARM, ROBOT_TIER_GRIP, ROBOT_TIER_LOCOMOTION,
    ROBOT_TIER_HUMAN_CONTACT, ROBOT_TIER_EMERGENCY, ROBOT_TIER_ESTOP,
}
ROBOT_HIGH_RISK_TIERS = {ROBOT_TIER_GRIP, ROBOT_TIER_LOCOMOTION, ROBOT_TIER_HUMAN_CONTACT, ROBOT_TIER_EMERGENCY}

HIGH_RISK_TIERS = {RISK_TIER_3, RISK_TIER_4} | ROBOT_HIGH_RISK_TIERS
NONTRIVIAL_RISK_TIERS = {RISK_TIER_2, RISK_TIER_3, RISK_TIER_4} | ROBOT_RISK_TIERS

POLICY_DECISION_ALLOW = "ALLOW"
POLICY_DECISION_ALLOW_WITH_CONSTRAINTS = "ALLOW_WITH_CONSTRAINTS"
POLICY_DECISION_REQUIRE_USER = "REQUIRE_USER"
POLICY_DECISION_SIMULATE_ONLY = "SIMULATE_ONLY"
POLICY_DECISION_DENY = "DENY"

DEFAULT_FRONTEND_CAPS = ("chat.submit", "chat.read")
DEFAULT_ADDON_CAPS = ("addon.request",)
DEFAULT_DRIVER_CAPS = ("driver.read",)

_RISK_KEYWORDS_PRIVILEGED = (
    "admin", "elevat", "service", "registry", "terminal", "shell", "boot",
    "install", "driver", "system", "process", "firewall", "credential",
)
_RISK_KEYWORDS_DESTRUCTIVE = (
    "delete", "format", "wipe", "destroy", "overwrite", "shutdown", "erase",
    "truncate", "reimage", "factory reset",
)
_RISK_KEYWORDS_NETWORK_EXPOSURE = (
    "public", "internet", "443", "80", "expose", "ingress", "reverse proxy",
    "port forward", "remote", "public_web", "wan", "subdomain", "dns",
)
_RISK_KEYWORDS_PHYSICAL = (
    "camera", "microphone", "mic", "speaker", "serial", "usb", "gpio", "plc",
    "cnc", "robot", "motor", "machine", "actuator", "driver.control", "hardware",
    "servo", "gripper", "hand", "arm", "leg", "walk", "locomotion", "move",
    "posture", "balance", "torque", "force", "human contact", "physical body",
)
_RISK_KEYWORDS_EVOLUTION = (
    "patch", "update", "upgrade", "selfaware", "self_aware", "evolution",
    "self-repair", "self repair", "modify core", "rewrite core",
)
_RISK_KEYWORDS_MODEL_AUTH = (
    "model_direct", "model authority", "autonomous model", "raw model", "llm authority",
)

_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SafetyPolicySnapshot:
    safe_mode: bool
    local_only: bool
    neosky_armed: bool
    developersmode: bool
    public_web: bool
    offline: bool
    run_mode: str
    device_mode: str
    allow_public_network_exposure: bool = False
    allow_remote_control: bool = False
    allow_destructive_actions: bool = False
    allow_model_direct_authority: bool = False
    allow_untrusted_frontend_privileged_access: bool = False
    require_confirmation_for_privileged: bool = True
    require_confirmation_for_network_exposure: bool = True
    require_confirmation_for_physical_control: bool = True
    require_confirmation_for_destructive_actions: bool = True
    require_rollback_for_apply: bool = True
    require_verification_before_success: bool = True
    require_local_presence_for_hazardous: bool = True
    fail_closed_on_unknown_caller: bool = True
    fail_closed_on_policy_error: bool = True
    deny_public_exposure_in_safe_mode: bool = True
    deny_privileged_apply_in_safe_mode: bool = True
    deny_remote_control_in_local_only: bool = True
    deny_model_authority_escalation: bool = True
    deny_self_evolution_when_unarmed: bool = True
    trust_unknown_surfaces_as_untrusted: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyReview:
    review_id: str
    ts: str
    decision: str
    allow: bool
    require_user: bool
    action_type: str
    capability_name: str
    execution_mode: str
    risk_level: str
    reasons: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    required_confirmations: List[str] = field(default_factory=list)
    policy_snapshot: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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


def _policy_dir() -> str:
    return os.path.join(_settings_dir(), "safety_policies")


def _policy_db_path() -> str:
    return os.path.join(_datasets_dir(), _DB_NAME)


def _policy_json_path() -> str:
    return os.path.join(_policy_dir(), _JSON_NAME)


def _ensure_dirs() -> None:
    for path in (_data_dir(), _datasets_dir(), _settings_dir(), _policy_dir()):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def _connect_db() -> sqlite3.Connection:
    _ensure_dirs()
    con = sqlite3.connect(_policy_db_path(), timeout=5.0, check_same_thread=False)
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
            CREATE TABLE IF NOT EXISTS safety_policy_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                review_id TEXT,
                severity TEXT,
                decision TEXT,
                action_type TEXT,
                capability_name TEXT,
                execution_mode TEXT,
                risk_level TEXT,
                details TEXT,
                meta_json TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS safety_policy_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                snapshot_json TEXT
            )
            """
        )
        con.commit()
    except Exception as exc:
        logger.debug("SafetyPolicies DB ensure failed: %s", exc)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _log_event(
    *,
    review_id: str,
    severity: str,
    decision: str,
    action_type: str,
    capability_name: str,
    execution_mode: str,
    risk_level: str,
    details: str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    con = None
    try:
        _ensure_tables()
        con = _connect_db()
        cur = con.cursor()
        cur.execute(
            "INSERT INTO safety_policy_events (ts, review_id, severity, decision, action_type, capability_name, execution_mode, risk_level, details, meta_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                str(review_id or ""),
                str(severity or "INFO"),
                str(decision or ""),
                str(action_type or ""),
                str(capability_name or ""),
                str(execution_mode or ""),
                str(risk_level or ""),
                str(details or ""),
                json.dumps(meta or {}, ensure_ascii=False),
            ),
        )
        con.commit()
    except Exception as exc:
        logger.debug("SafetyPolicies log failed: %s", exc)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _persist_snapshot(snapshot: Dict[str, Any]) -> None:
    con = None
    try:
        _ensure_tables()
        con = _connect_db()
        cur = con.cursor()
        blob = json.dumps(snapshot or {}, ensure_ascii=False, indent=2)
        cur.execute(
            "INSERT INTO safety_policy_snapshots (ts, snapshot_json) VALUES (?, ?)",
            (datetime.now().isoformat(), blob),
        )
        con.commit()
    except Exception as exc:
        logger.debug("SafetyPolicies snapshot persistence failed: %s", exc)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass
    try:
        with open(_policy_json_path(), "w", encoding="utf-8") as fh:
            json.dump(snapshot or {}, fh, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Flag helpers
# ---------------------------------------------------------------------------
def _env_flag(name: str, default: str = "false") -> bool:
    try:
        v = os.getenv(name, default)
        return str(v).strip().lower() in ("1", "true", "yes", "on", "enabled")
    except Exception:
        return False


def _flag(name: str, default: bool = False) -> bool:
    try:
        if config is not None and hasattr(config, name):
            v = getattr(config, name)
            if isinstance(v, bool):
                return v
            return str(v).strip().lower() in ("1", "true", "yes", "on", "enabled")
    except Exception:
        pass
    return _env_flag(name, "true" if default else "false")


def _safe_mode() -> bool:
    return _flag("SAFE_MODE", False)


def _local_only() -> bool:
    return _flag("LOCAL_ONLY_MODE", False)


def _neosky() -> bool:
    return _flag("NEOSKYMATRIX", False)


def _devmode() -> bool:
    return _flag("DEVELOPERSMODE", False)


def _public_web() -> bool:
    try:
        mode = str(getattr(config, "DEVICE_MODE", "")).strip().lower()
        if mode == "public_web":
            return True
    except Exception:
        pass
    return False


def _offline() -> bool:
    try:
        fn = getattr(config, "is_offline", None)
        if callable(fn):
            return bool(fn())
    except Exception:
        pass
    return bool(_local_only())


def _run_mode() -> str:
    try:
        return str(getattr(config, "RUN_MODE", "local")).strip().lower() or "local"
    except Exception:
        return "local"


def _device_mode() -> str:
    try:
        return str(getattr(config, "DEVICE_MODE", "headless")).strip().lower() or "headless"
    except Exception:
        return "headless"


def _read_override(name: str, default: bool) -> bool:
    env_name = f"SM_POLICY_{name.upper()}"
    if env_name in os.environ:
        return _env_flag(env_name, "true" if default else "false")
    return default


# ---------------------------------------------------------------------------
# Runtime snapshot
# ---------------------------------------------------------------------------
def build_runtime_policy_snapshot() -> Dict[str, Any]:
    safe_mode = _safe_mode()
    local_only = _local_only()
    neosky_armed = bool(_neosky() and _devmode())
    public_web = _public_web()
    offline = _offline()
    run_mode = _run_mode()
    device_mode = _device_mode()

    snapshot = SafetyPolicySnapshot(
        safe_mode=safe_mode,
        local_only=local_only,
        neosky_armed=neosky_armed,
        developersmode=_devmode(),
        public_web=public_web,
        offline=offline,
        run_mode=run_mode,
        device_mode=device_mode,
        allow_public_network_exposure=_read_override("allow_public_network_exposure", False),
        allow_remote_control=_read_override("allow_remote_control", False),
        allow_destructive_actions=_read_override("allow_destructive_actions", False),
        allow_model_direct_authority=_read_override("allow_model_direct_authority", False),
        allow_untrusted_frontend_privileged_access=_read_override("allow_untrusted_frontend_privileged_access", False),
        require_confirmation_for_privileged=_read_override("require_confirmation_for_privileged", True),
        require_confirmation_for_network_exposure=_read_override("require_confirmation_for_network_exposure", True),
        require_confirmation_for_physical_control=_read_override("require_confirmation_for_physical_control", True),
        require_confirmation_for_destructive_actions=_read_override("require_confirmation_for_destructive_actions", True),
        require_rollback_for_apply=_read_override("require_rollback_for_apply", True),
        require_verification_before_success=_read_override("require_verification_before_success", True),
        require_local_presence_for_hazardous=_read_override("require_local_presence_for_hazardous", True),
        fail_closed_on_unknown_caller=_read_override("fail_closed_on_unknown_caller", True),
        fail_closed_on_policy_error=_read_override("fail_closed_on_policy_error", True),
        deny_public_exposure_in_safe_mode=_read_override("deny_public_exposure_in_safe_mode", True),
        deny_privileged_apply_in_safe_mode=_read_override("deny_privileged_apply_in_safe_mode", True),
        deny_remote_control_in_local_only=_read_override("deny_remote_control_in_local_only", True),
        deny_model_authority_escalation=_read_override("deny_model_authority_escalation", True),
        deny_self_evolution_when_unarmed=_read_override("deny_self_evolution_when_unarmed", True),
        trust_unknown_surfaces_as_untrusted=_read_override("trust_unknown_surfaces_as_untrusted", True),
        metadata={
            "module": MODULE_NAME,
            "version": MODULE_VERSION,
            "generated_epoch": time.time(),
        },
    )
    out = snapshot.to_dict()
    _persist_snapshot(out)
    return out


def get_runtime_policy_snapshot() -> Dict[str, Any]:
    return build_runtime_policy_snapshot()


def get_policy_snapshot() -> Dict[str, Any]:
    return build_runtime_policy_snapshot()


def get_security_policy_snapshot() -> Dict[str, Any]:
    return build_runtime_policy_snapshot()


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------
def _safe_lower(value: Any) -> str:
    try:
        return str(value or "").strip().lower()
    except Exception:
        return ""


def _joined_contract_text(action_contract: Dict[str, Any]) -> str:
    contract = dict(action_contract or {})
    joined = " ".join(
        [
            _safe_lower(contract.get("action_type")),
            _safe_lower(contract.get("capability_name")),
            _safe_lower(contract.get("executor_name")),
            _safe_lower(contract.get("normalized_text")),
            _safe_lower(contract.get("surface")),
            _safe_lower(contract.get("target_surface")),
            _safe_lower(contract.get("target")),
            _safe_lower(contract.get("requested_role")),
        ]
    )
    perms = contract.get("required_permissions") or []
    try:
        joined += " " + " ".join(_safe_lower(x) for x in perms)
    except Exception:
        pass
    return joined.strip()


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    h = _safe_lower(haystack)
    return any(n in h for n in needles)


def _normalize_risk_level(value: Any) -> str:
    v = _safe_lower(value)
    if not v:
        return RISK_TIER_1
    aliases = {
        "tier_0": RISK_TIER_0,
        "tier_0_info": RISK_TIER_0,
        "tier_1": RISK_TIER_1,
        "tier_1_harmless_local_ui": RISK_TIER_1,
        "tier_2": RISK_TIER_2,
        "tier_2_bounded_local_operation": RISK_TIER_2,
        "tier_3": RISK_TIER_3,
        "tier_3_privileged_system": RISK_TIER_3,
        "tier_4": RISK_TIER_4,
        "tier_4_network_remote_or_destructive": RISK_TIER_4,
        "tier_robot_observe_only": ROBOT_TIER_OBSERVE,
        "tier_robot_face_expression": ROBOT_TIER_FACE,
        "tier_robot_head_movement": ROBOT_TIER_HEAD,
        "tier_robot_posture": ROBOT_TIER_POSTURE,
        "tier_robot_arm_low_force": ROBOT_TIER_ARM,
        "tier_robot_grip_object": ROBOT_TIER_GRIP,
        "tier_robot_locomotion": ROBOT_TIER_LOCOMOTION,
        "tier_robot_human_contact": ROBOT_TIER_HUMAN_CONTACT,
        "tier_robot_emergency_intervention": ROBOT_TIER_EMERGENCY,
        "tier_robot_emergency_stop": ROBOT_TIER_ESTOP,
    }
    return aliases.get(v, str(value))


def _robotic_action_profile(contract: Dict[str, Any]) -> Dict[str, Any]:
    joined = _joined_contract_text(contract)
    risk_level = _normalize_risk_level(contract.get("risk_level"))
    is_robotic = risk_level in ROBOT_RISK_TIERS or _contains_any(joined, (
        "robot", "servo", "gripper", "locomotion", "walk", "step", "posture", "balance",
        "arm", "hand", "leg", "torque", "force", "physical body", "human contact",
    ))
    motion = is_robotic and _contains_any(joined, ("move", "walk", "step", "reach", "raise", "lower", "turn", "posture", "locomotion", "grip", "release"))
    contact = is_robotic and _contains_any(joined, ("human contact", "touch human", "grab person", "push", "pull", "assist person", "intervene"))
    emergency = is_robotic and _contains_any(joined, ("emergency", "fire", "medical", "collision", "life", "rescue", "safe_stop", "e-stop"))
    return {
        "is_robotic": bool(is_robotic),
        "motion_requested": bool(motion),
        "human_contact_requested": bool(contact),
        "emergency_context": bool(emergency),
        "risk_level": risk_level,
        "requires_smget": bool(is_robotic),
        "requires_current_perception": bool(motion or contact),
        "requires_safe_stop": bool(is_robotic),
        "requires_force_speed_limits": bool(motion or contact),
    }


def _has_confirmation(action_contract: Dict[str, Any], governance: Optional[Dict[str, Any]] = None) -> bool:
    contract = dict(action_contract or {})
    gov = dict(governance or {})
    for source in (contract, gov, contract.get("meta") if isinstance(contract.get("meta"), dict) else {}):
        try:
            if bool(source.get("confirmed")) or bool(source.get("user_confirmed")) or bool(source.get("explicit_confirmation")):
                return True
        except Exception:
            pass
    return False


def _rollback_present(action_contract: Dict[str, Any]) -> bool:
    contract = dict(action_contract or {})
    rollback = contract.get("rollback_plan")
    if isinstance(rollback, dict):
        return bool(rollback)
    if isinstance(rollback, list):
        return bool(rollback)
    if isinstance(rollback, str):
        return bool(rollback.strip())
    return False


def _is_network_exposure_request(contract: Dict[str, Any]) -> bool:
    return _contains_any(_joined_contract_text(contract), _RISK_KEYWORDS_NETWORK_EXPOSURE)


def _is_physical_or_driver_request(contract: Dict[str, Any]) -> bool:
    return _contains_any(_joined_contract_text(contract), _RISK_KEYWORDS_PHYSICAL)


def _is_privileged_request(contract: Dict[str, Any]) -> bool:
    return _contains_any(_joined_contract_text(contract), _RISK_KEYWORDS_PRIVILEGED)


def _is_destructive_request(contract: Dict[str, Any]) -> bool:
    return _contains_any(_joined_contract_text(contract), _RISK_KEYWORDS_DESTRUCTIVE)


def _is_evolution_request(contract: Dict[str, Any]) -> bool:
    return _contains_any(_joined_contract_text(contract), _RISK_KEYWORDS_EVOLUTION)


def _is_model_authority_attempt(contract: Dict[str, Any]) -> bool:
    joined = _joined_contract_text(contract)
    if _contains_any(joined, _RISK_KEYWORDS_MODEL_AUTH):
        return True
    providers = contract.get("model_provider") or contract.get("provider") or ""
    requested_role = contract.get("requested_role") or ""
    return bool(_safe_lower(providers) and _safe_lower(requested_role) not in {"", "helper", "semantic_helper", "assistant"})


def required_confirmations_for_action(action_contract: Dict[str, Any], governance: Optional[Dict[str, Any]] = None) -> List[str]:
    contract = dict(action_contract or {})
    policy = build_runtime_policy_snapshot()
    req: List[str] = []
    risk_level = _normalize_risk_level(contract.get("risk_level"))
    if _is_network_exposure_request(contract) and policy.get("require_confirmation_for_network_exposure", True):
        req.append("network_exposure")
    if _is_physical_or_driver_request(contract) and policy.get("require_confirmation_for_physical_control", True):
        req.append("physical_control")
    if _is_privileged_request(contract) and policy.get("require_confirmation_for_privileged", True):
        req.append("privileged_action")
    if _is_destructive_request(contract) and policy.get("require_confirmation_for_destructive_actions", True):
        req.append("destructive_action")
    if risk_level in HIGH_RISK_TIERS and policy.get("require_local_presence_for_hazardous", True):
        req.append("local_presence")
    if _has_confirmation(contract, governance):
        return []
    return list(dict.fromkeys(req))


def get_surface_policy(surface: str = "", caller_kind: str = "") -> Dict[str, Any]:
    snap = build_runtime_policy_snapshot()
    surface_l = _safe_lower(surface)
    caller_l = _safe_lower(caller_kind)
    return {
        "surface": surface_l or "unknown",
        "caller_kind": caller_l or "unknown",
        "trust_unknown_surfaces_as_untrusted": bool(snap.get("trust_unknown_surfaces_as_untrusted", True)),
        "default_permissions": list(
            DEFAULT_FRONTEND_CAPS if caller_l == "frontend"
            else DEFAULT_ADDON_CAPS if caller_l == "addon"
            else DEFAULT_DRIVER_CAPS if caller_l == "driver"
            else ()
        ),
        "public_web_restricted": bool(snap.get("public_web", False)),
        "safe_mode": bool(snap.get("safe_mode", False)),
        "local_only": bool(snap.get("local_only", False)),
    }


def get_policy_for_risk_level(risk_level: str) -> Dict[str, Any]:
    snap = build_runtime_policy_snapshot()
    risk = _normalize_risk_level(risk_level)
    return {
        "risk_level": risk,
        "require_confirmation": bool(
            risk in HIGH_RISK_TIERS
            or (risk == RISK_TIER_2 and snap.get("require_confirmation_for_physical_control", True))
        ),
        "require_rollback": bool(risk in NONTRIVIAL_RISK_TIERS and snap.get("require_rollback_for_apply", True)),
        "require_verification": bool(risk in NONTRIVIAL_RISK_TIERS and snap.get("require_verification_before_success", True)),
        "safe_mode_blocks_apply": bool(risk in HIGH_RISK_TIERS and snap.get("deny_privileged_apply_in_safe_mode", True)),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_action_policy(action_contract: Dict[str, Any], governance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    contract = dict(action_contract or {})
    governance = dict(governance or {})
    policy = build_runtime_policy_snapshot()
    review = PolicyReview(
        review_id=f"policy_{int(time.time() * 1000)}",
        ts=datetime.now().isoformat(),
        decision=POLICY_DECISION_ALLOW,
        allow=True,
        require_user=False,
        action_type=str(contract.get("action_type") or "unknown"),
        capability_name=str(contract.get("capability_name") or "unknown"),
        execution_mode=str(contract.get("execution_mode") or MODE_DRAFT),
        risk_level=_normalize_risk_level(contract.get("risk_level")),
        policy_snapshot=policy,
        meta={"governance": governance},
    )

    # Energetics is an advisory survival/metabolic gate. It does not execute,
    # but SafetyPolicies records and enforces DENY/DEFER outcomes for reserve,
    # thermal, active-task, and no-clock-mutation constraints. The SarahMemoryGlobals
    # hazardous-energy constitution is checked first so a missing/corrupted Energetics
    # module cannot fail open for drivers, device power, diagnostics, evolution, MSDC,
    # or embodied operations.
    try:
        blocker_fn = getattr(config, "sm_hazardous_energy_blocks_action", None) if config is not None else None
        status_fn = getattr(config, "sm_hazardous_energy_status", None) if config is not None else None
        hazard_context = {
            "action_type": review.action_type,
            "phase": str(contract.get("primary_lane") or contract.get("phase") or "safety_policy"),
            "risk_level": review.risk_level,
            "execution_mode": review.execution_mode,
            "capability_name": review.capability_name,
        }
        if callable(blocker_fn) and blocker_fn(review.action_type, hazard_context):
            status = status_fn(hazard_context) if callable(status_fn) else {}
            review.meta["hazardous_energy_constitution"] = status
            review.decision = POLICY_DECISION_DENY
            review.allow = False
            review.risk_factors.append("hazardous_energy_constitution_lockout")
            review.reasons.append("Hazardous-energy constitution blocked this action before Energetics/SafetyPolicies execution review.")
    except Exception as exc:
        review.meta["hazardous_energy_constitution_error"] = str(exc)
        review.decision = POLICY_DECISION_DENY
        review.allow = False
        review.risk_factors.append("hazardous_energy_constitution_exception")
        review.reasons.append("Hazardous-energy constitution check failed closed.")

    # If hazardous-energy lockout is active but the action is not classified as hazardous-energy-sensitive,
    # skip Energetics rather than letting the locked-out organ deny ordinary non-power/non-actuation work.
    _consult_energetics = True
    try:
        status_fn = getattr(config, "sm_hazardous_energy_status", None) if config is not None else None
        if callable(status_fn):
            _status = status_fn({"action_type": review.action_type, "phase": str(contract.get("primary_lane") or contract.get("phase") or "safety_policy")})
            if isinstance(_status, dict) and bool(_status.get("lockout_active", True)):
                _consult_energetics = False
                review.meta.setdefault("hazardous_energy_constitution", _status)
                review.constraints["energetics_skipped_due_to_lockout"] = True
    except Exception as exc:
        review.meta["hazardous_energy_status_error"] = str(exc)

    if review.allow and _consult_energetics and _Energetics is not None:
        try:
            fn = getattr(_Energetics, "preflight_action", None)
            if callable(fn):
                energy_review = fn(
                    review.action_type,
                    phase=str(contract.get("primary_lane") or contract.get("phase") or "safety_policy"),
                    requested_power_mode="FULL_POWER" if review.execution_mode == MODE_APPLY else "ACTIVE",
                    active_task=contract.get("active_task") if isinstance(contract.get("active_task"), dict) else {},
                    internal_state=contract.get("internal_state") if isinstance(contract.get("internal_state"), dict) else {},
                    external_state=contract.get("external_state") if isinstance(contract.get("external_state"), dict) else {},
                    constraints={"risk_level": review.risk_level, "safety_policy_review": True},
                    context={"scheduled": review.action_type in {"diagnostics", "evolution", "sync", "network_scan", "device_scan"}},
                )
                review.meta["energetics"] = energy_review
                energy_decision = str(energy_review.get("decision") or "ALLOW").upper()
                if energy_decision in {"DENY", "DEFER"}:
                    review.decision = POLICY_DECISION_DENY if energy_decision == "DENY" else POLICY_DECISION_SIMULATE_ONLY
                    review.allow = False
                    review.risk_factors.append("energetics_" + energy_decision.lower())
                    review.reasons.append(str(energy_review.get("reason") or "Energetics reserve/power preflight blocked or deferred action."))
                elif energy_decision == "REDUCE_MODE":
                    review.constraints["energetics_allowed_power_mode"] = energy_review.get("allowed_power_mode")
                    review.constraints["energetics_reduce_mode"] = True
                    review.reasons.append("Energetics requires reduced power/workload mode.")
        except Exception as exc:
            review.meta["energetics_error"] = str(exc)
            # If the action is already hazardous-energy classified, an Energetics error is a denial.
            try:
                blocker_fn = getattr(config, "sm_hazardous_energy_blocks_action", None) if config is not None else None
                if callable(blocker_fn) and blocker_fn(review.action_type, contract):
                    review.decision = POLICY_DECISION_DENY
                    review.allow = False
                    review.risk_factors.append("energetics_exception_fail_closed")
                    review.reasons.append("Energetics review failed closed for hazardous-energy-sensitive action.")
            except Exception:
                review.decision = POLICY_DECISION_DENY
                review.allow = False
                review.risk_factors.append("energetics_exception_unknown_fail_closed")
                review.reasons.append("Energetics review failed closed under uncertain hazardous-energy state.")

    if governance.get("allow") is False or _safe_lower(governance.get("decision")) == "deny":
        review.decision = POLICY_DECISION_DENY
        review.allow = False
        review.risk_factors.append("governance_denied")
        review.reasons.append("CognitiveServices denied the action before policy execution review.")

    if policy.get("safe_mode") and review.execution_mode == MODE_APPLY and review.risk_level in HIGH_RISK_TIERS and policy.get("deny_privileged_apply_in_safe_mode", True):
        review.decision = POLICY_DECISION_SIMULATE_ONLY
        review.allow = False
        review.risk_factors.append("safe_mode_high_risk_apply")
        review.constraints["allowed_execution_mode"] = MODE_SIMULATE
        review.reasons.append("SAFE_MODE blocks high-risk apply-mode actions.")

    if policy.get("local_only") and _is_network_exposure_request(contract) and policy.get("deny_remote_control_in_local_only", True):
        review.decision = POLICY_DECISION_DENY
        review.allow = False
        review.risk_factors.append("local_only_network_exposure")
        review.reasons.append("LOCAL_ONLY_MODE blocks remote/public network exposure.")

    if _is_network_exposure_request(contract) and not policy.get("allow_public_network_exposure", False):
        if review.execution_mode == MODE_APPLY:
            review.decision = POLICY_DECISION_REQUIRE_USER
            review.allow = False
            review.require_user = True
        review.required_confirmations.append("network_exposure")
        review.risk_factors.append("public_exposure_not_default")
        review.reasons.append("Public network exposure is not allowed by default and requires explicit approval.")

    robot_profile = _robotic_action_profile(contract)
    review.meta["robotic_action_profile"] = robot_profile

    if _is_physical_or_driver_request(contract):
        review.required_confirmations.extend(["physical_control"])
        if not _has_confirmation(contract, governance) and policy.get("require_confirmation_for_physical_control", True):
            review.decision = POLICY_DECISION_REQUIRE_USER
            review.allow = False
            review.require_user = True
            review.risk_factors.append("physical_confirmation_required")
            review.reasons.append("Physical/device control requires explicit confirmation.")

    if robot_profile.get("is_robotic"):
        review.constraints.setdefault("smget_required", True)
        review.constraints.setdefault("operatorcore_required", True)
        review.constraints.setdefault("assurance_required", True)
        review.constraints.setdefault("security_required", True)
        review.constraints.setdefault("safe_stop_required", True)
        review.constraints.setdefault("movement_lock_default", True)
        review.required_confirmations.extend(["robotic_body_control", "local_presence"])
        if robot_profile.get("requires_current_perception") and not bool(contract.get("current_perception_fresh") or (contract.get("metadata") or {}).get("current_perception_fresh")):
            review.decision = POLICY_DECISION_SIMULATE_ONLY
            review.allow = False
            review.risk_factors.append("robot_motion_requires_current_perception")
            review.reasons.append("Robot motion/manipulation requires fresh perception evidence before apply-mode execution.")
            review.constraints["allowed_execution_mode"] = MODE_SIMULATE
        if robot_profile.get("requires_force_speed_limits"):
            env = contract.get("safety_envelope") if isinstance(contract.get("safety_envelope"), dict) else {}
            if not (env.get("max_force_n") or env.get("max_speed_mps") or env.get("max_torque_nm")):
                review.decision = POLICY_DECISION_SIMULATE_ONLY
                review.allow = False
                review.risk_factors.append("robot_force_speed_limits_missing")
                review.reasons.append("Robot body action lacks explicit force/speed/torque limits.")
                review.constraints["allowed_execution_mode"] = MODE_SIMULATE
        if robot_profile.get("human_contact_requested") and not robot_profile.get("emergency_context"):
            review.decision = POLICY_DECISION_DENY
            review.allow = False
            review.risk_factors.append("robot_human_contact_denied_without_emergency")
            review.reasons.append("Human-contact robot action is denied unless a verified emergency context exists.")

    if _is_privileged_request(contract):
        review.required_confirmations.extend(["privileged_action"])
        if not _has_confirmation(contract, governance) and policy.get("require_confirmation_for_privileged", True):
            review.decision = POLICY_DECISION_REQUIRE_USER
            review.allow = False
            review.require_user = True
            review.risk_factors.append("privileged_confirmation_required")
            review.reasons.append("Privileged execution requires explicit confirmation.")

    if _is_destructive_request(contract):
        review.required_confirmations.extend(["destructive_action"])
        if not policy.get("allow_destructive_actions", False):
            review.decision = POLICY_DECISION_DENY
            review.allow = False
            review.risk_factors.append("destructive_denied_by_default")
            review.reasons.append("Destructive actions are denied by default.")
        elif not _has_confirmation(contract, governance) and policy.get("require_confirmation_for_destructive_actions", True):
            review.decision = POLICY_DECISION_REQUIRE_USER
            review.allow = False
            review.require_user = True
            review.risk_factors.append("destructive_confirmation_required")
            review.reasons.append("Destructive action requires explicit confirmation.")

    if _is_model_authority_attempt(contract) and policy.get("deny_model_authority_escalation", True):
        review.decision = POLICY_DECISION_DENY
        review.allow = False
        review.risk_factors.append("model_direct_authority_blocked")
        review.reasons.append("External or raw model output may not become execution authority.")

    if _is_evolution_request(contract) and policy.get("deny_self_evolution_when_unarmed", True) and not policy.get("neosky_armed", False):
        review.decision = POLICY_DECISION_DENY
        review.allow = False
        review.risk_factors.append("neosky_not_armed")
        review.reasons.append("Self-evolution and privileged core mutation require NEOSKYMATRIX + DEVELOPERSMODE.")

    if review.execution_mode == MODE_APPLY and review.risk_level in NONTRIVIAL_RISK_TIERS and policy.get("require_rollback_for_apply", True) and not _rollback_present(contract):
        review.decision = POLICY_DECISION_DENY
        review.allow = False
        review.risk_factors.append("missing_rollback_plan")
        review.reasons.append("Non-trivial apply-mode action lacks a rollback plan.")

    if review.allow and review.risk_level in HIGH_RISK_TIERS:
        review.decision = POLICY_DECISION_ALLOW_WITH_CONSTRAINTS
        review.constraints["verify_before_success"] = bool(policy.get("require_verification_before_success", True))
        review.constraints["audit_required"] = True
        review.reasons.append("High-risk action may proceed only with strict verification and audit discipline.")

    review.required_confirmations = list(dict.fromkeys(review.required_confirmations))
    if not review.reasons:
        review.reasons.append("SafetyPolicies found no additional policy blockers.")

    severity = "INFO" if review.allow else "WARNING"
    if review.decision == POLICY_DECISION_DENY:
        severity = "ERROR"

    _log_event(
        review_id=review.review_id,
        severity=severity,
        decision=review.decision,
        action_type=review.action_type,
        capability_name=review.capability_name,
        execution_mode=review.execution_mode,
        risk_level=review.risk_level,
        details="; ".join(review.reasons[:4]),
        meta=review.to_dict(),
    )

    return review.to_dict()


def review_action_policy(action_contract: Dict[str, Any], governance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return evaluate_action_policy(action_contract, governance)


def is_action_allowed_by_policy(action_contract: Dict[str, Any], governance: Optional[Dict[str, Any]] = None) -> bool:
    try:
        result = evaluate_action_policy(action_contract, governance)
        return bool(result.get("allow", False))
    except Exception:
        snap = build_runtime_policy_snapshot()
        return not bool(snap.get("fail_closed_on_policy_error", True))


# ---------------------------------------------------------------------------
# Compatibility helper for future callers
# ---------------------------------------------------------------------------
def get_constitutional_doctrine() -> Dict[str, Any]:
    return {
        "user_supremacy": True,
        "no_hallucinated_actuation": True,
        "no_untrusted_model_direct_control": True,
        "no_hidden_capability_activation": True,
        "riskier_actions_require_stronger_assurance": True,
        "physical_world_safety_outranks_convenience": True,
        "rollback_where_practical": True,
        "fail_closed_when_uncertain": True,
    }


if __name__ == "__main__":
    snap = build_runtime_policy_snapshot()
    print(json.dumps(snap, indent=2, ensure_ascii=False))


# -----------------------------------------------------------------------------
# V10/V9F read-only evidence claim policy
# -----------------------------------------------------------------------------
def review_read_only_evidence_policy(evidence_contract: Dict[str, Any]) -> Dict[str, Any]:
    """Policy review for non-mutating evidence/claim validation."""
    contract = dict(evidence_contract or {})
    hard = contract.get('hard_evidence_packet') if isinstance(contract.get('hard_evidence_packet'), dict) else {}
    read_only = bool(contract.get('read_only', True) and hard.get('read_only', True))
    action_taken = bool(contract.get('action_taken', False) or hard.get('action_taken', False))
    risk = str(contract.get('risk_level') or RISK_TIER_0)
    reasons = []; allow = True; decision = POLICY_DECISION_ALLOW
    if not read_only or action_taken:
        allow = False; decision = POLICY_DECISION_DENY; reasons.append('Evidence contract attempted state change or action execution.')
    if risk in HIGH_RISK_TIERS:
        allow = False; decision = POLICY_DECISION_REQUIRE_USER; reasons.append('Evidence contract used high-risk tier for a claim path.')
    if not hard and not contract.get('appeal_packet'):
        allow = False; decision = POLICY_DECISION_SIMULATE_ONLY; reasons.append('No hard evidence packet supplied; simulate/deny settled claim.')
    if allow:
        reasons.append('Read-only TIER_0 evidence claim is policy-allowed; presentation must still be source-aware and Compare-validated.')
    return {'ok': True, 'module': MODULE_NAME, 'decision': decision, 'allow': bool(allow), 'risk_level': risk, 'read_only': read_only, 'action_taken': action_taken, 'reasons': reasons, 'doctrine': 'No hallucinated actuation; read-only evidence claims may proceed only when bounded and source-aware.'}

# --- SM V8.0 TRI-LAYER PATCH 2026-05-20 ---
def get_robotic_body_policy() -> Dict[str, Any]:
    return {
        "schema": "SarahMemorySafetyPolicies.robotic_body_policy.v1",
        "robot_risk_tiers": sorted(ROBOT_RISK_TIERS),
        "robot_high_risk_tiers": sorted(ROBOT_HIGH_RISK_TIERS),
        "doctrine": {
            "no_new_governor_required": True,
            "cognitive_services_remains_judge": True,
            "msdc_remains_body_witness_and_dispatch_choke": True,
            "operatorcore_remains_execution_lifecycle": True,
            "assurancegate_remains_confidence_gate": True,
            "securitygovernor_remains_authority_gate": True,
            "compare_compass_remain_validation_and_bearing": True,
            "unrestricted_autonomy_forbidden": True,
        },
        "required_for_physical_apply": [
            "explicit_user_or_emergency_authority",
            "fresh_perception_evidence",
            "verified_body_part_driver",
            "force_speed_torque_limits",
            "safe_stop_path",
            "operatorcore_contract",
            "security_review",
            "assurance_review",
            "audit_log",
        ],
    }


def tri_layer_safety_doctrine() -> Dict[str, Any]:
    return {
        "language_context_before_routing": True,
        "six_questions_must_close_before_action": True,
        "emotion_informs_but_never_authorizes": True,
        "dynamic_identity_requires_user_approval": True,
        "packets_are_evidence_not_authority": True,
        "user_remains_final_authority": True,
    }

# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 START ---
# Sovereign agent-runtime and interoperability policy.
# This section is intentionally side-effect free: no network calls, no DB writes,
# no capability activation. It supplies policy packets consumed by SMGET, the
# SecurityGovernor, AssuranceGate, Network/appnet broker, and OperatorCore.

INTEROP_PROTOCOL_MCP = "mcp"
INTEROP_PROTOCOL_A2A = "a2a"
INTEROP_PROTOCOL_AGUI = "ag-ui"
INTEROP_PROTOCOL_LOCAL = "local"

BROKER_DIRECTION_INBOUND = "inbound"
BROKER_DIRECTION_OUTBOUND = "outbound"
BROKER_DIRECTION_BIDIRECTIONAL = "bidirectional"

BROKER_MODE_ONE_WAY = "ONE_WAY_BROKER"
BROKER_MODE_GOVERNED_EXCEPTION = "GOVERNED_BIDIRECTIONAL_EXCEPTION"

SAFE_INTEROP_MESSAGE_TYPES = {
    "status", "heartbeat", "capability_manifest", "skill_manifest", "agent_card",
    "tool_schema", "evidence_packet", "observation", "query", "reply", "ui_event",
}
DANGEROUS_INTEROP_MESSAGE_TYPES = {
    "execute", "tool_call", "command", "shell", "filesystem_write", "driver_action",
    "robot_motion", "network_expose", "install", "delete", "privileged_action",
}


def get_sovereign_agent_runtime_policy() -> Dict[str, Any]:
    """Return the local-first agentic runtime policy used by adapters.

    External standards such as MCP/A2A/AG-UI are treated as protocol adapters.
    They are not SarahMemory authority sources. The default broker mode is
    one-way/store-and-forward with no remote execution.
    """
    snapshot = get_runtime_policy_snapshot()
    return {
        "ok": True,
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "policy_schema": "SarahMemory.sovereign_agent_runtime_policy.v1",
        "default_broker_mode": BROKER_MODE_ONE_WAY,
        "cloud_optional": True,
        "offline_capable": True,
        "local_first": True,
        "external_protocols_are_adapters_only": True,
        "mcp_direct_execution_allowed": False,
        "a2a_direct_execution_allowed": False,
        "agui_authority_allowed": False,
        "remote_tool_auto_authority_allowed": False,
        "requires_smget_for_interop_execution": True,
        "requires_security_review_for_remote_protocols": True,
        "requires_assurance_for_state_change": True,
        "requires_user_exception_for_bidirectional": True,
        "safe_message_types": sorted(SAFE_INTEROP_MESSAGE_TYPES),
        "dangerous_message_types": sorted(DANGEROUS_INTEROP_MESSAGE_TYPES),
        "runtime_flags": {
            "safe_mode": bool(snapshot.get("safe_mode")),
            "local_only": bool(snapshot.get("local_only")),
            "offline": bool(snapshot.get("offline")),
            "public_web": bool(snapshot.get("public_web")),
            "run_mode": snapshot.get("run_mode"),
            "device_mode": snapshot.get("device_mode"),
        },
        "deployment_classes": ["pc", "robotics", "commercial", "industrial", "military"],
        "doctrine": [
            "SarahMemory remains the one-way broker by default.",
            "External protocols may submit evidence or manifests; they may not execute.",
            "Bidirectional or remote-control behavior requires explicit governed exception.",
            "Offline operation must remain functional without cloud or Big-Tech APIs.",
        ],
    }


def _interop_text(envelope: Dict[str, Any]) -> str:
    try:
        return _joined_contract_text(envelope)
    except Exception:
        return json.dumps(envelope or {}, ensure_ascii=False, default=str)[:2000]


def evaluate_interop_policy(envelope: Dict[str, Any], governance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Policy review for MCP/A2A/AG-UI style adapter packets.

    The return value is advisory policy. It does not execute, register, fetch, or
    call an external system. Fail closed for unknown/remote state-changing intent.
    """
    env = dict(envelope or {})
    gov = dict(governance or {})
    policy = get_sovereign_agent_runtime_policy()
    protocol = _safe_lower(env.get("protocol") or env.get("adapter") or env.get("source_protocol"))
    direction = _safe_lower(env.get("direction") or BROKER_DIRECTION_INBOUND)
    message_type = _safe_lower(env.get("message_type") or env.get("type") or env.get("action_type") or "unknown")
    execution_mode = _safe_lower(env.get("execution_mode") or env.get("mode") or MODE_DRAFT)
    remote = bool(env.get("remote") or env.get("is_remote") or env.get("remote_origin"))
    bidirectional = direction == BROKER_DIRECTION_BIDIRECTIONAL or bool(env.get("bidirectional"))
    explicit_exception = bool(
        env.get("explicit_bidirectional_authority")
        or env.get("user_authorized_bidirectional")
        or gov.get("explicit_bidirectional_authority")
    )

    reasons: List[str] = []
    constraints: Dict[str, Any] = {
        "one_way_broker": True,
        "direct_execution": False,
        "remote_tool_execution": False,
        "state_change": False,
        "requires_smget": True,
    }
    decision = POLICY_DECISION_ALLOW_WITH_CONSTRAINTS
    allow = True
    require_user = False

    if protocol not in {INTEROP_PROTOCOL_MCP, INTEROP_PROTOCOL_A2A, INTEROP_PROTOCOL_AGUI, INTEROP_PROTOCOL_LOCAL, ""}:
        reasons.append(f"Unknown interop protocol '{protocol or 'unknown'}' treated as untrusted.")
        require_user = True
        decision = POLICY_DECISION_REQUIRE_USER

    if message_type in DANGEROUS_INTEROP_MESSAGE_TYPES or execution_mode == MODE_APPLY:
        constraints["state_change"] = True
        allow = False
        require_user = True
        decision = POLICY_DECISION_SIMULATE_ONLY
        reasons.append("Interop packet requests state-changing or execution behavior; one-way broker downgrades to simulate/evidence only.")

    if bidirectional and not explicit_exception:
        allow = False
        require_user = True
        decision = POLICY_DECISION_DENY
        reasons.append("Bidirectional broker behavior denied without explicit governed exception.")

    if remote and (message_type not in SAFE_INTEROP_MESSAGE_TYPES):
        allow = False
        require_user = True
        decision = POLICY_DECISION_DENY
        reasons.append("Remote interop packet is not a safe evidence/manifest/status message.")

    if _local_only() or _offline() or _safe_mode():
        constraints["network_use"] = False
        if remote and message_type not in {"status", "heartbeat", "capability_manifest", "agent_card"}:
            allow = False
            require_user = True
            decision = POLICY_DECISION_DENY
            reasons.append("Local-only/offline/safe-mode blocks remote interop beyond passive status/manifest intake.")

    if not reasons:
        reasons.append("Interop packet accepted only as brokered evidence/manifest/query; no direct execution authority granted.")

    return {
        "ok": True,
        "review_id": f"interop_policy_{int(time.time() * 1000)}",
        "decision": decision,
        "allow": bool(allow),
        "require_user": bool(require_user),
        "protocol": protocol or "unknown",
        "direction": direction,
        "message_type": message_type,
        "broker_mode": BROKER_MODE_ONE_WAY if not explicit_exception else BROKER_MODE_GOVERNED_EXCEPTION,
        "constraints": constraints,
        "reasons": reasons,
        "policy": policy,
        "metadata": {
            "remote": remote,
            "bidirectional": bidirectional,
            "explicit_exception": explicit_exception,
        },
    }


def is_interop_packet_allowed(envelope: Dict[str, Any], governance: Optional[Dict[str, Any]] = None) -> bool:
    review = evaluate_interop_policy(envelope, governance=governance)
    return bool(review.get("allow")) and str(review.get("decision")) not in {POLICY_DECISION_DENY, POLICY_DECISION_SIMULATE_ONLY}
# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 END ---

# ====================================================================
# END OF SarahMemorySafetyPolicies.py v9.0.0
# ====================================================================
