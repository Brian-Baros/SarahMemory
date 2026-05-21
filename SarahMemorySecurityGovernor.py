"""--==The SarahMemory Project==--
File: SarahMemorySecurityGovernor.py
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
- Core sovereignty, trust, and runtime security governor for SarahMemory AiOS / SMGET.
- Enforces trust boundaries between cognition, operator execution, frontends, addons,
  drivers, models, and runtime actions.
- Prevents authority inversion where UI/addon/model output is mistaken for execution authority.
- Applies fail-closed rules for risky, remote, destructive, or physically consequential actions.
- Preserves user control, model containment, non-degradation doctrine, and auditability.

CORE DOCTRINE:
- Models may advise. Governance decides. Verified executors act.
- Frontends, addons, drivers, and external surfaces are bounded clients of the core.
- Registration does not imply trust. Presentation does not imply authority.
- High-risk actions require stronger trust, stronger proof, and stronger user approval.
- No hallucinated actuation. No silent privilege expansion. No hidden capability activation.
- Local-first, fail-soft, auditable, and safe-by-default.

RELATIONSHIP MODEL:
- SarahMemoryCognitiveServices.py -> logic/risk/procedural governor
- SarahMemorySecurityGovernor.py  -> trust/sovereignty/integrity/security governor
- SarahMemoryOperatorCore.py      -> action contract + lifecycle + verify/rollback
- SarahMemoryTrustRegistry.py     -> canonical trust and subject registry (when present)
- SarahMemorySafetyPolicies.py    -> constitutional safety policy source (when present)
- SarahMemoryAssuranceGate.py     -> assurance/confidence gate for execution (when present)
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "security_governor"
# CATEGORY = "runtime_security_and_sovereignty"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "security_governor"
# HARDWARE_DOMAIN = "system_network_filesystem_devices"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "security_governor"
# FAMILY = "smget"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "SMGET trust, sovereignty, integrity, caller-boundary, and runtime security governor. Blocks authority inversion, unsafe execution paths, and untrusted escalation."
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
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Safe imports (never hard-fail the platform)
# ---------------------------------------------------------------------------
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None

try:
    import SarahMemoryCognitiveSelf as _CogSelf  # type: ignore
except Exception:
    _CogSelf = None

try:
    import SarahMemoryTrustRegistry as _TrustRegistry  # type: ignore
except Exception:
    _TrustRegistry = None

try:
    import SarahMemorySafetyPolicies as _SafetyPolicies  # type: ignore
except Exception:
    _SafetyPolicies = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("SarahMemorySecurityGovernor")
logger.setLevel(logging.DEBUG if bool(getattr(config, "DEBUG_MODE", True)) else logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.propagate = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODULE_NAME = "SarahMemorySecurityGovernor"
MODULE_VERSION = "8.0.0"
_DB_NAME = "security_governor.db"

DECISION_ALLOW = "ALLOW"
DECISION_ALLOW_WITH_CONSTRAINTS = "ALLOW_WITH_CONSTRAINTS"
DECISION_REQUIRE_USER = "REQUIRE_USER"
DECISION_SIMULATE_ONLY = "SIMULATE_ONLY"
DECISION_DENY = "DENY"
DECISION_QUARANTINE = "QUARANTINE"

TRUST_TIER_CORE = "core"
TRUST_TIER_FIRST_PARTY = "first_party"
TRUST_TIER_VERIFIED_THIRD_PARTY = "verified_third_party"
TRUST_TIER_UNVERIFIED = "unverified"
TRUST_TIER_QUARANTINED = "quarantined"
TRUST_TIER_UNKNOWN = "unknown"

RISK_TIER_0 = "TIER_0_INFO"
RISK_TIER_1 = "TIER_1_HARMLESS_LOCAL_UI"
RISK_TIER_2 = "TIER_2_BOUNDED_LOCAL_OPERATION"
RISK_TIER_3 = "TIER_3_PRIVILEGED_SYSTEM"
RISK_TIER_4 = "TIER_4_NETWORK_REMOTE_OR_DESTRUCTIVE"

MODE_SIMULATE = "simulate"
MODE_DRAFT = "draft"
MODE_APPLY = "apply"
MODE_ROLLBACK = "rollback"

_DEFAULT_TRUST_TIER_ORDER = {
    TRUST_TIER_CORE: 5,
    TRUST_TIER_FIRST_PARTY: 4,
    TRUST_TIER_VERIFIED_THIRD_PARTY: 3,
    TRUST_TIER_UNVERIFIED: 2,
    TRUST_TIER_UNKNOWN: 1,
    TRUST_TIER_QUARANTINED: 0,
}


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


def _security_db_path() -> str:
    return os.path.join(_datasets_dir(), _DB_NAME)


def _ensure_dirs() -> None:
    try:
        os.makedirs(_data_dir(), exist_ok=True)
    except Exception:
        pass
    try:
        os.makedirs(_datasets_dir(), exist_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------
def _connect_db() -> sqlite3.Connection:
    _ensure_dirs()
    con = sqlite3.connect(_security_db_path(), timeout=5.0, check_same_thread=False)
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
            CREATE TABLE IF NOT EXISTS security_governor_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                review_id TEXT,
                severity TEXT,
                decision TEXT,
                caller_identity TEXT,
                trust_tier TEXT,
                action_type TEXT,
                risk_level TEXT,
                details TEXT,
                meta_json TEXT
            )
            """
        )
        con.commit()
    except Exception as exc:
        logger.debug("SecurityGovernor DB ensure failed: %s", exc)
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
    caller_identity: str,
    trust_tier: str,
    action_type: str,
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
            "INSERT INTO security_governor_events (ts, review_id, severity, decision, caller_identity, trust_tier, action_type, risk_level, details, meta_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                str(review_id or ""),
                str(severity or "INFO"),
                str(decision or ""),
                str(caller_identity or "unknown"),
                str(trust_tier or TRUST_TIER_UNKNOWN),
                str(action_type or "unknown"),
                str(risk_level or RISK_TIER_0),
                str(details or ""),
                json.dumps(meta or {}, ensure_ascii=False),
            ),
        )
        con.commit()
    except Exception as exc:
        logger.debug("SecurityGovernor log failed: %s", exc)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class CallerIdentity:
    caller_id: str
    caller_kind: str
    surface: str
    publisher: str = ""
    module_name: str = ""
    trust_tier: str = TRUST_TIER_UNKNOWN
    trusted: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityReview:
    review_id: str
    ts: str
    caller: CallerIdentity
    action_type: str
    capability_name: str
    executor_name: str
    risk_level: str
    execution_mode: str
    decision: str = DECISION_ALLOW
    allow: bool = True
    require_user: bool = False
    risk_score: int = 0
    trust_score: int = 0
    risk_factors: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    recommended_next: str = ""
    policy_snapshot: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Runtime helpers
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


def _governance_profile() -> Dict[str, Any]:
    try:
        fn = getattr(config, "sm_get_core_governance_profile", None)
        if callable(fn):
            out = fn()
            if isinstance(out, dict):
                return out
    except Exception:
        pass
    return {
        "discovery_enabled": False,
        "registered_count": 0,
        "quarantined_count": 0,
        "ignored_count": 0,
        "auto_expose_approved": False,
    }


def _registered_modules() -> Dict[str, Any]:
    try:
        fn = getattr(config, "sm_get_registered_core_modules", None)
        if callable(fn):
            out = fn(include_pending=True)
            if isinstance(out, dict):
                return out
    except Exception:
        pass
    return {}


def _is_core_module_approved(module_name: str, capability: Optional[str] = None) -> bool:
    try:
        fn = getattr(config, "sm_is_core_module_approved", None)
        if callable(fn):
            return bool(fn(module_name, capability=capability))
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Policy fallbacks / integration points
# ---------------------------------------------------------------------------
def _policy_snapshot() -> Dict[str, Any]:
    fallback = {
        "safe_mode": _safe_mode(),
        "local_only": _local_only(),
        "neosky_armed": _neosky_armed(),
        "allow_public_network_exposure": False,
        "allow_remote_control": False,
        "allow_destructive_actions": False,
        "allow_model_direct_authority": False,
        "require_confirmation_for_privileged": True,
        "require_confirmation_for_network_exposure": True,
        "require_confirmation_for_physical_control": True,
        "require_rollback_for_apply": True,
        "fail_closed_on_unknown_caller": True,
    }
    if _SafetyPolicies is None:
        return fallback
    for fn_name in ("get_runtime_policy_snapshot", "get_policy_snapshot", "get_security_policy_snapshot"):
        try:
            fn = getattr(_SafetyPolicies, fn_name, None)
            if callable(fn):
                out = fn()
                if isinstance(out, dict):
                    merged = dict(fallback)
                    merged.update(out)
                    return merged
        except Exception:
            pass
    return fallback


def _get_trust_record(identity: CallerIdentity, action_contract: Dict[str, Any]) -> Dict[str, Any]:
    if _TrustRegistry is not None:
        for fn_name in ("resolve_subject_trust", "get_subject_trust", "lookup_subject", "get_trust_record"):
            try:
                fn = getattr(_TrustRegistry, fn_name, None)
                if callable(fn):
                    out = fn(
                        caller_id=identity.caller_id,
                        caller_kind=identity.caller_kind,
                        surface=identity.surface,
                        module_name=identity.module_name,
                        action_contract=action_contract,
                    )
                    if isinstance(out, dict):
                        return out
            except TypeError:
                try:
                    out = fn(identity.caller_id)
                    if isinstance(out, dict):
                        return out
                except Exception:
                    pass
            except Exception:
                pass

    registered = _registered_modules()
    entry = registered.get(identity.module_name) if identity.module_name else None
    approved = bool(entry.get("status") == "approved" and entry.get("exposed", False)) if isinstance(entry, dict) else False
    if identity.module_name and _is_core_module_approved(identity.module_name):
        return {
            "subject_id": identity.caller_id,
            "subject_kind": identity.caller_kind,
            "trust_tier": TRUST_TIER_CORE,
            "trusted": True,
            "source": "globals_core_registry",
            "module_name": identity.module_name,
            "approved": True,
        }
    if approved:
        return {
            "subject_id": identity.caller_id,
            "subject_kind": identity.caller_kind,
            "trust_tier": TRUST_TIER_FIRST_PARTY,
            "trusted": True,
            "source": "globals_registered_entries",
            "module_name": identity.module_name,
            "approved": True,
        }

    kind = str(identity.caller_kind or "").lower()
    if kind in {"frontend", "addon", "plugin", "surface"}:
        return {
            "subject_id": identity.caller_id,
            "subject_kind": kind,
            "trust_tier": TRUST_TIER_UNVERIFIED,
            "trusted": False,
            "source": "fallback_unverified_surface",
            "module_name": identity.module_name,
            "approved": False,
        }

    return {
        "subject_id": identity.caller_id,
        "subject_kind": kind or "unknown",
        "trust_tier": TRUST_TIER_UNKNOWN,
        "trusted": False,
        "source": "fallback_unknown",
        "module_name": identity.module_name,
        "approved": False,
    }


# ---------------------------------------------------------------------------
# Identity / extraction helpers
# ---------------------------------------------------------------------------
def _safe_lower(value: Any) -> str:
    try:
        return str(value or "").strip().lower()
    except Exception:
        return ""


def _normalize_risk_level(value: Any) -> str:
    v = str(value or RISK_TIER_0).strip().upper()
    if v in {RISK_TIER_0, RISK_TIER_1, RISK_TIER_2, RISK_TIER_3, RISK_TIER_4}:
        return v
    return RISK_TIER_0


def _risk_weight(risk_level: str) -> int:
    mapping = {
        RISK_TIER_0: 5,
        RISK_TIER_1: 15,
        RISK_TIER_2: 35,
        RISK_TIER_3: 65,
        RISK_TIER_4: 85,
    }
    return int(mapping.get(_normalize_risk_level(risk_level), 10))


def _caller_identity(action_contract: Dict[str, Any]) -> CallerIdentity:
    meta = dict(action_contract.get("metadata") or {})
    trust_context = dict(action_contract.get("trust_context") or {})
    source_surface = str(action_contract.get("source_surface") or action_contract.get("origin") or meta.get("surface") or "unknown").strip() or "unknown"
    caller_id = str(
        meta.get("caller_id")
        or meta.get("surface_id")
        or meta.get("addon_id")
        or meta.get("frontend_id")
        or action_contract.get("origin")
        or source_surface
        or f"caller_{uuid.uuid4().hex[:8]}"
    ).strip()

    caller_kind = str(
        meta.get("caller_kind")
        or meta.get("surface_kind")
        or ("addon" if meta.get("addon_id") else "frontend" if source_surface not in {"unknown", "local_agent", "gui", "api"} else "core")
    ).strip() or "unknown"

    module_name = str(
        meta.get("module_name")
        or action_contract.get("executor_name")
        or action_contract.get("capability_name")
        or ""
    ).strip()

    publisher = str(meta.get("publisher") or meta.get("author") or "").strip()

    identity = CallerIdentity(
        caller_id=caller_id,
        caller_kind=caller_kind,
        surface=source_surface,
        publisher=publisher,
        module_name=module_name,
        meta={
            "origin": action_contract.get("origin"),
            "source_surface": source_surface,
            "trust_context": trust_context,
        },
    )
    return identity


# ---------------------------------------------------------------------------
# Rule helpers
# ---------------------------------------------------------------------------
def _has_confirmation(action_contract: Dict[str, Any], governance: Dict[str, Any]) -> bool:
    meta = dict(action_contract.get("metadata") or {})
    indicators = [
        action_contract.get("confirmed"),
        action_contract.get("user_confirmed"),
        action_contract.get("confirmation_granted"),
        meta.get("confirmed"),
        meta.get("user_confirmed"),
        meta.get("approval_granted"),
        meta.get("step_up_confirmed"),
        governance.get("user_confirmed") if isinstance(governance, dict) else None,
    ]
    return any(bool(x) for x in indicators)


def _rollback_present(action_contract: Dict[str, Any]) -> bool:
    rollback_plan = action_contract.get("rollback_plan") or []
    return bool(isinstance(rollback_plan, list) and len(rollback_plan) > 0)


def _is_network_exposure_request(action_contract: Dict[str, Any]) -> bool:
    capability = _safe_lower(action_contract.get("capability_name"))
    lane = _safe_lower(action_contract.get("primary_lane"))
    action_type = _safe_lower(action_contract.get("action_type"))
    text = _safe_lower(action_contract.get("normalized_text"))
    return bool(
        capability.startswith("network")
        or "ingress" in capability
        or "expose" in action_type
        or "port" in text
        or "443" in text
        or lane == "network"
    )


def _is_physical_or_driver_request(action_contract: Dict[str, Any]) -> bool:
    capability = _safe_lower(action_contract.get("capability_name"))
    action_type = _safe_lower(action_contract.get("action_type"))
    text = _safe_lower(action_contract.get("normalized_text"))
    return bool(
        capability.startswith("driver")
        or "hardware" in capability
        or "camera" in text
        or "webcam" in text
        or "microphone" in text
        or "plc" in text
        or "cnc" in text
        or "driver" in action_type
        or "device" in action_type
    )


def _indicates_destructive_or_privileged(action_contract: Dict[str, Any]) -> bool:
    action_type = _safe_lower(action_contract.get("action_type"))
    text = _safe_lower(action_contract.get("normalized_text"))
    perms = [str(x).lower() for x in (action_contract.get("required_permissions") or [])]
    joined = " ".join([action_type, text, " ".join(perms)])
    keywords = [
        "delete", "wipe", "destroy", "format", "overwrite", "shutdown", "reboot",
        "elevat", "admin", "root", "firewall", "remote", "terminal.execute",
        "system.settings.write", "memory.write.persistent",
    ]
    return any(k in joined for k in keywords)


def _external_model_authority_attempt(action_contract: Dict[str, Any]) -> bool:
    meta = dict(action_contract.get("metadata") or {})
    text = _safe_lower(action_contract.get("normalized_text"))
    provider = _safe_lower(meta.get("provider") or meta.get("model_provider") or meta.get("selected_provider"))
    model_name = _safe_lower(meta.get("model_name") or meta.get("selected_model") or meta.get("helper_model"))
    raw_claim = bool(meta.get("model_direct_authority") or meta.get("direct_from_model") or meta.get("raw_model_output_executable"))
    if raw_claim:
        return True
    if provider and provider not in {"local", "builtin", "native", "sarahmemory"} and _indicates_destructive_or_privileged(action_contract):
        return True
    if model_name and _indicates_destructive_or_privileged(action_contract) and ("gpt" in model_name or "claude" in model_name or "gemini" in model_name or "llama" in model_name or "phi" in model_name):
        return True
    if "run exactly what the model says" in text or "execute raw model output" in text:
        return True
    return False


def _trust_score_for_tier(trust_tier: str) -> int:
    return int(_DEFAULT_TRUST_TIER_ORDER.get(trust_tier, 1) * 20)


# ---------------------------------------------------------------------------
# Main security evaluation
# ---------------------------------------------------------------------------
def evaluate_action(action_contract: Dict[str, Any], governance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    governance = dict(governance or {})
    contract = dict(action_contract or {})
    review_id = f"sec_{uuid.uuid4().hex}"
    identity = _caller_identity(contract)
    trust_record = _get_trust_record(identity, contract)
    identity.trust_tier = str(trust_record.get("trust_tier") or TRUST_TIER_UNKNOWN)
    identity.trusted = bool(trust_record.get("trusted", False))

    policy = _policy_snapshot()
    risk_level = _normalize_risk_level(contract.get("risk_level"))
    risk_score = _risk_weight(risk_level)

    review = SecurityReview(
        review_id=review_id,
        ts=datetime.now().isoformat(),
        caller=identity,
        action_type=str(contract.get("action_type") or "unknown"),
        capability_name=str(contract.get("capability_name") or "unknown"),
        executor_name=str(contract.get("executor_name") or "unknown"),
        risk_level=risk_level,
        execution_mode=str(contract.get("execution_mode") or MODE_DRAFT),
        decision=DECISION_ALLOW,
        allow=True,
        require_user=False,
        risk_score=risk_score,
        trust_score=_trust_score_for_tier(identity.trust_tier),
        policy_snapshot=policy,
        meta={
            "trust_record": trust_record,
            "governance": governance,
            "governance_profile": _governance_profile(),
        },
    )

    # Governance denial remains authoritative.
    gov_decision = _safe_lower(governance.get("decision"))
    if gov_decision in {"deny"} or governance.get("allow") is False:
        review.decision = DECISION_DENY
        review.allow = False
        review.risk_factors.append("governance_denied")
        review.reasons.append("CognitiveServices denied this action before security execution review.")

    # Quarantined or explicitly blocked callers fail closed.
    if identity.trust_tier == TRUST_TIER_QUARANTINED:
        review.decision = DECISION_QUARANTINE
        review.allow = False
        review.risk_factors.append("caller_quarantined")
        review.reasons.append("Caller trust tier is quarantined and may not request execution.")

    # Unknown/unverified callers cannot directly drive privileged/destructive actions.
    if identity.trust_tier in {TRUST_TIER_UNKNOWN, TRUST_TIER_UNVERIFIED} and risk_level in {RISK_TIER_3, RISK_TIER_4}:
        review.decision = DECISION_DENY
        review.allow = False
        review.risk_factors.append("untrusted_privileged_request")
        review.reasons.append("Unverified or unknown caller may not request privileged or destructive execution.")

    # Safe mode blocks high-impact apply-mode actions.
    if policy.get("safe_mode") and review.execution_mode == MODE_APPLY and risk_level in {RISK_TIER_3, RISK_TIER_4}:
        review.decision = DECISION_SIMULATE_ONLY
        review.allow = False
        review.risk_factors.append("safe_mode_high_risk_apply")
        review.reasons.append("SAFE_MODE blocks high-risk apply-mode execution.")
        review.constraints["allowed_execution_mode"] = MODE_SIMULATE

    # Local-only mode blocks remote/public network exposure.
    if policy.get("local_only") and _is_network_exposure_request(contract):
        review.decision = DECISION_DENY
        review.allow = False
        review.risk_factors.append("local_only_network_exposure")
        review.reasons.append("LOCAL_ONLY_MODE blocks remote/public network exposure requests.")

    # Public network exposure requires explicit confirmation.
    if _is_network_exposure_request(contract) and policy.get("require_confirmation_for_network_exposure", True) and not _has_confirmation(contract, governance):
        if review.allow:
            review.decision = DECISION_REQUIRE_USER
            review.allow = False
            review.require_user = True
        review.risk_factors.append("network_exposure_requires_confirmation")
        review.reasons.append("Network/public exposure requires explicit user confirmation.")

    # Physical/driver control requires explicit confirmation unless trust tier is core and request is bounded.
    if _is_physical_or_driver_request(contract) and policy.get("require_confirmation_for_physical_control", True) and not _has_confirmation(contract, governance):
        if risk_level in {RISK_TIER_2, RISK_TIER_3, RISK_TIER_4}:
            if review.allow:
                review.decision = DECISION_REQUIRE_USER
                review.allow = False
                review.require_user = True
            review.risk_factors.append("physical_control_requires_confirmation")
            review.reasons.append("Physical/device control requires explicit user confirmation.")

    # External model output must never become direct operational authority.
    if _external_model_authority_attempt(contract) and not policy.get("allow_model_direct_authority", False):
        review.decision = DECISION_DENY
        review.allow = False
        review.risk_factors.append("model_direct_authority_blocked")
        review.reasons.append("External or raw model output may not directly become execution authority.")

    # Self-evolution / privileged mutation requires NEOSKYMATRIX + DEVELOPERSMODE arm.
    joined = " ".join([
        _safe_lower(contract.get("action_type")),
        _safe_lower(contract.get("capability_name")),
        _safe_lower(contract.get("normalized_text")),
    ])
    if any(k in joined for k in ("patch", "update", "evolution", "selfaware", "self_aware", "self-repair", "upgrade")) and not policy.get("neosky_armed", False):
        review.decision = DECISION_DENY
        review.allow = False
        review.risk_factors.append("neosky_not_armed")
        review.reasons.append("Self-evolution or privileged self-modification requires NEOSKYMATRIX + DEVELOPERSMODE.")

    # Privileged/destructive apply requires confirmation.
    if _indicates_destructive_or_privileged(contract) and policy.get("require_confirmation_for_privileged", True) and review.execution_mode == MODE_APPLY and not _has_confirmation(contract, governance):
        review.decision = DECISION_REQUIRE_USER
        review.allow = False
        review.require_user = True
        review.risk_factors.append("privileged_apply_requires_confirmation")
        review.reasons.append("Privileged or destructive apply-mode action requires explicit user confirmation.")

    # Apply-mode actions should advertise rollback if practical.
    if review.execution_mode == MODE_APPLY and policy.get("require_rollback_for_apply", True) and risk_level in {RISK_TIER_2, RISK_TIER_3, RISK_TIER_4} and not _rollback_present(contract):
        review.decision = DECISION_DENY
        review.allow = False
        review.risk_factors.append("missing_rollback_plan")
        review.reasons.append("Apply-mode action lacks a rollback plan for a non-trivial risk tier.")

    # Require explicit capability grant for unknown permissions.
    perms = [str(x or "").strip() for x in (contract.get("required_permissions") or []) if str(x or "").strip()]
    if identity.trust_tier in {TRUST_TIER_UNKNOWN, TRUST_TIER_UNVERIFIED} and perms:
        review.risk_factors.append("scoped_permissions_present")
        review.reasons.append("Requested permissions are scoped and must remain core-authorized at runtime.")

    # Final normalization.
    if review.allow and review.decision == DECISION_ALLOW:
        if review.risk_score >= 75:
            review.decision = DECISION_ALLOW_WITH_CONSTRAINTS
            review.constraints.setdefault("verify_before_success", True)
            review.constraints.setdefault("audit_required", True)
            review.reasons.append("High-risk action allowed only with strict verification and audit discipline.")

    if not review.reasons:
        review.reasons.append("SecurityGovernor found no additional trust or sovereignty blockers.")

    if not review.recommended_next:
        if review.allow:
            review.recommended_next = "Proceed through OperatorCore lifecycle with verification and audit."
        elif review.decision == DECISION_REQUIRE_USER:
            review.recommended_next = "Collect explicit user confirmation, then resubmit through governed path."
        elif review.decision == DECISION_SIMULATE_ONLY:
            review.recommended_next = "Downgrade to simulate mode or disable SAFE_MODE before privileged apply."
        elif review.decision == DECISION_QUARANTINE:
            review.recommended_next = "Keep caller quarantined and inspect trust record before any further action."
        else:
            review.recommended_next = "Block action and return safe report to the user/operator."

    severity = "INFO" if review.allow else "WARNING"
    if review.decision in {DECISION_DENY, DECISION_QUARANTINE}:
        severity = "ERROR"

    _log_event(
        review_id=review.review_id,
        severity=severity,
        decision=review.decision,
        caller_identity=review.caller.caller_id,
        trust_tier=review.caller.trust_tier,
        action_type=review.action_type,
        risk_level=review.risk_level,
        details="; ".join(review.reasons[:4]),
        meta=review.to_dict(),
    )

    out = review.to_dict()
    out["caller_identity"] = out.pop("caller")
    return out


# ---------------------------------------------------------------------------
# Compatibility aliases for OperatorCore
# ---------------------------------------------------------------------------
def govern_action(action_contract: Dict[str, Any], governance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return evaluate_action(action_contract, governance)



def review_action_contract(action_contract: Dict[str, Any], governance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return evaluate_action(action_contract, governance)


# ---------------------------------------------------------------------------
# Optional focused helpers for later callers
# ---------------------------------------------------------------------------
def evaluate_model_authority(model_name: str, provider: str = "", *, requested_role: str = "helper") -> Dict[str, Any]:
    provider_l = _safe_lower(provider)
    model_l = _safe_lower(model_name)
    allow = True
    decision = DECISION_ALLOW
    reasons: List[str] = []
    risk_factors: List[str] = []

    if requested_role not in {"helper", "semantic_helper", "assistant"}:
        allow = False
        decision = DECISION_DENY
        reasons.append("External model may not claim a direct operational role outside helper-only lanes.")
        risk_factors.append("model_role_escalation")

    if provider_l and provider_l not in {"local", "builtin", "native", "sarahmemory"} and requested_role not in {"helper", "semantic_helper", "assistant"}:
        allow = False
        decision = DECISION_DENY
        reasons.append("Non-local provider cannot be promoted into direct execution authority.")
        risk_factors.append("nonlocal_model_authority")

    if not reasons:
        reasons.append("Model containment rules allow helper-only participation.")

    return {
        "decision": decision,
        "allow": allow,
        "model_name": model_name,
        "provider": provider,
        "requested_role": requested_role,
        "risk_factors": risk_factors,
        "reasons": reasons,
    }



def get_runtime_security_profile() -> Dict[str, Any]:
    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "safe_mode": _safe_mode(),
        "local_only": _local_only(),
        "neosky_armed": _neosky_armed(),
        "governance_profile": _governance_profile(),
        "policy_snapshot": _policy_snapshot(),
    }

# --- SM V8.0 TRI-LAYER PATCH 2026-05-20 ---
def review_tri_layer_packet_boundary(packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Security boundary review for tri-layer packets. Packets are evidence, not authority."""
    pkt = packet if isinstance(packet, dict) else {}
    return {
        "ok": True,
        "decision": "ALLOW_AS_EVIDENCE_ONLY",
        "execution_authority": False,
        "authority_note": "Tri-layer packets may inform governance but cannot authorize execution.",
        "packet_type": pkt.get("packet_type"),
    }
