"""--==The SarahMemory Project==--
File: SarahMemoryAssuranceGate.py
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
- Assurance and confidence hard-gate for SarahMemory AiOS / SMGET.
- Converts "the system thinks this is okay" into "the system has enough assurance to proceed."
- Blocks low-confidence, weakly-specified, or under-verified action requests before real execution.
- Enforces the doctrine that risky execution requires stronger confidence, stronger verification planning,
  and stronger rollback readiness.
- Closes the "no hallucinated actuation" gap by refusing high-risk action flows that lack adequate
  evidence, structure, or confirmation.

CORE DOCTRINE:
- No hallucinated actuation.
- Riskier actions require stronger assurance.
- Apply-mode execution must not rely on vague or low-evidence intent.
- Verification readiness matters as much as confidence.
- Rollback readiness matters for non-trivial apply-mode operations.
- Fail closed when assurance is insufficient.

RELATIONSHIP MODEL:
- SarahMemoryCognitiveServices.py  -> logical/risk governance
- SarahMemorySecurityGovernor.py   -> trust / sovereignty / integrity hard gate
- SarahMemorySafetyPolicies.py     -> constitutional safety rules and runtime policy snapshot
- SarahMemoryAssuranceGate.py      -> assurance thresholding / confidence / readiness hard gate
- SarahMemoryOperatorCore.py       -> lifecycle / dispatch / verify / rollback orchestration
- Executors / Drivers              -> perform bounded work only after assurance passes
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "assurance_gate"
# CATEGORY = "runtime_assurance_and_confidence"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "assurance_gate"
# HARDWARE_DOMAIN = "system_network_filesystem_devices"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "assurance_gate"
# FAMILY = "smget"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "SMGET assurance gate. Evaluates confidence, verification readiness, rollback readiness, and execution risk before real action dispatch."
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


# ---------------------------------------------------------------------------
# Safe imports (never hard-fail the platform)
# ---------------------------------------------------------------------------
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None

try:
    import SarahMemorySafetyPolicies as _SafetyPolicies  # type: ignore
except Exception:
    _SafetyPolicies = None

try:
    import SarahMemoryCompare as _Compare  # type: ignore
except Exception:
    _Compare = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("SarahMemoryAssuranceGate")
logger.setLevel(logging.DEBUG if bool(getattr(config, "DEBUG_MODE", True)) else logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.propagate = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODULE_NAME = "SarahMemoryAssuranceGate"
MODULE_VERSION = "8.0.0"
_DB_NAME = "assurance_gate.db"
_JSON_NAME = "assurance_gate_snapshot.json"

MODE_SIMULATE = "simulate"
MODE_DRAFT = "draft"
MODE_APPLY = "apply"
MODE_ROLLBACK = "rollback"

RISK_TIER_0 = "TIER_0_INFO"
RISK_TIER_1 = "TIER_1_HARMLESS_LOCAL_UI"
RISK_TIER_2 = "TIER_2_BOUNDED_LOCAL_OPERATION"
RISK_TIER_3 = "TIER_3_PRIVILEGED_SYSTEM"
RISK_TIER_4 = "TIER_4_NETWORK_REMOTE_OR_DESTRUCTIVE"

DECISION_ALLOW = "ALLOW"
DECISION_REQUIRE_USER = "REQUIRE_USER"
DECISION_SIMULATE_ONLY = "SIMULATE_ONLY"
DECISION_DENY = "DENY"

TRUST_TIER_CORE = "core"
TRUST_TIER_FIRST_PARTY = "first_party"
TRUST_TIER_VERIFIED_THIRD_PARTY = "verified_third_party"
TRUST_TIER_UNVERIFIED = "unverified"
TRUST_TIER_UNKNOWN = "unknown"
TRUST_TIER_QUARANTINED = "quarantined"

_LOCK = threading.RLock()


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class AssuranceSignal:
    name: str
    value: float
    source: str
    weight: float = 1.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssuranceReview:
    review_id: str
    ts: str
    decision: str
    allow: bool
    require_user: bool
    action_type: str
    capability_name: str
    execution_mode: str
    risk_level: str
    confidence: float
    assurance_score: float
    threshold: float
    verification_ready: bool
    rollback_ready: bool
    reasons: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    signals: List[Dict[str, Any]] = field(default_factory=list)
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


def _assurance_db_path() -> str:
    return os.path.join(_datasets_dir(), _DB_NAME)


def _snapshot_path() -> str:
    return os.path.join(_settings_dir(), _JSON_NAME)


def _ensure_dirs() -> None:
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
def _connect_db() -> Optional[sqlite3.Connection]:
    try:
        _ensure_dirs()
        con = sqlite3.connect(_assurance_db_path(), timeout=5.0, check_same_thread=False)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
            con.execute("PRAGMA busy_timeout=5000;")
        except Exception:
            pass
        return con
    except Exception:
        return None


def _ensure_tables() -> None:
    con = None
    try:
        con = _connect_db()
        if con is None:
            return
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS assurance_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                review_id TEXT,
                decision TEXT,
                action_type TEXT,
                capability_name TEXT,
                execution_mode TEXT,
                risk_level TEXT,
                confidence REAL,
                assurance_score REAL,
                threshold REAL,
                details TEXT,
                meta_json TEXT
            )
            """
        )
        con.commit()
    except Exception as exc:
        logger.debug("AssuranceGate DB ensure failed: %s", exc)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _log_event(review: AssuranceReview) -> None:
    con = None
    try:
        _ensure_tables()
        con = _connect_db()
        if con is None:
            return
        cur = con.cursor()
        cur.execute(
            "INSERT INTO assurance_events (ts, review_id, decision, action_type, capability_name, execution_mode, risk_level, confidence, assurance_score, threshold, details, meta_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now().isoformat(),
                review.review_id,
                review.decision,
                review.action_type,
                review.capability_name,
                review.execution_mode,
                review.risk_level,
                float(review.confidence),
                float(review.assurance_score),
                float(review.threshold),
                "; ".join(review.reasons[:4]),
                json.dumps(review.to_dict(), ensure_ascii=False),
            ),
        )
        con.commit()
    except Exception as exc:
        logger.debug("AssuranceGate event log failed: %s", exc)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _write_snapshot(review: AssuranceReview) -> None:
    try:
        _ensure_dirs()
        with open(_snapshot_path(), "w", encoding="utf-8") as f:
            json.dump(review.to_dict(), f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _safe_lower(value: Any) -> str:
    try:
        return str(value or "").strip().lower()
    except Exception:
        return ""


def _safe_json(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return "{}"


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v
    except Exception:
        return float(default)


def _risk_weight(risk_level: str) -> float:
    mapping = {
        RISK_TIER_0: 0.10,
        RISK_TIER_1: 0.18,
        RISK_TIER_2: 0.35,
        RISK_TIER_3: 0.62,
        RISK_TIER_4: 0.82,
    }
    return float(mapping.get(str(risk_level or RISK_TIER_0), 0.35))


def _trust_bonus(trust_tier: str) -> float:
    mapping = {
        TRUST_TIER_CORE: 0.18,
        TRUST_TIER_FIRST_PARTY: 0.12,
        TRUST_TIER_VERIFIED_THIRD_PARTY: 0.06,
        TRUST_TIER_UNVERIFIED: -0.04,
        TRUST_TIER_UNKNOWN: -0.08,
        TRUST_TIER_QUARANTINED: -0.25,
    }
    return float(mapping.get(str(trust_tier or TRUST_TIER_UNKNOWN), -0.08))


def _mode_bonus(execution_mode: str) -> float:
    mapping = {
        MODE_SIMULATE: 0.10,
        MODE_DRAFT: 0.05,
        MODE_APPLY: -0.02,
        MODE_ROLLBACK: 0.02,
    }
    return float(mapping.get(str(execution_mode or MODE_DRAFT), 0.0))


def _threshold_for(execution_mode: str, risk_level: str) -> float:
    mode = str(execution_mode or MODE_DRAFT)
    risk = str(risk_level or RISK_TIER_0)
    table = {
        MODE_SIMULATE: {
            RISK_TIER_0: 0.10,
            RISK_TIER_1: 0.15,
            RISK_TIER_2: 0.25,
            RISK_TIER_3: 0.35,
            RISK_TIER_4: 0.45,
        },
        MODE_DRAFT: {
            RISK_TIER_0: 0.20,
            RISK_TIER_1: 0.28,
            RISK_TIER_2: 0.42,
            RISK_TIER_3: 0.58,
            RISK_TIER_4: 0.70,
        },
        MODE_APPLY: {
            RISK_TIER_0: 0.45,
            RISK_TIER_1: 0.55,
            RISK_TIER_2: 0.70,
            RISK_TIER_3: 0.84,
            RISK_TIER_4: 0.92,
        },
        MODE_ROLLBACK: {
            RISK_TIER_0: 0.30,
            RISK_TIER_1: 0.38,
            RISK_TIER_2: 0.52,
            RISK_TIER_3: 0.60,
            RISK_TIER_4: 0.70,
        },
    }
    return float(table.get(mode, table[MODE_DRAFT]).get(risk, 0.58))


def _now_iso() -> str:
    return datetime.now().isoformat()


def _policy_snapshot() -> Dict[str, Any]:
    default = {
        "safe_mode": bool(getattr(config, "SAFE_MODE", False)) if config else False,
        "local_only": bool(getattr(config, "LOCAL_ONLY_MODE", False)) if config else False,
        "neosky_armed": bool(getattr(config, "NEOSKYMATRIX", False) and getattr(config, "DEVELOPERSMODE", False)) if config else False,
        "require_rollback_for_apply": True,
        "require_verification_before_success": True,
        "require_confirmation_for_privileged": True,
        "require_confirmation_for_network_exposure": True,
        "require_confirmation_for_physical_control": True,
        "fail_closed_on_policy_error": True,
        "allow_model_direct_authority": False,
        "riskier_actions_require_stronger_assurance": True,
    }
    if _SafetyPolicies is None:
        return default
    for fn_name in ("get_runtime_policy_snapshot", "get_policy_snapshot", "get_security_policy_snapshot"):
        try:
            fn = getattr(_SafetyPolicies, fn_name, None)
            if callable(fn):
                out = fn()
                if isinstance(out, dict):
                    merged = dict(default)
                    merged.update(out)
                    return merged
        except Exception as exc:
            logger.debug("SafetyPolicies.%s failed: %s", fn_name, exc)
    return default


def _extract_confidence_signals(action_contract: Dict[str, Any], governance: Dict[str, Any], security: Dict[str, Any]) -> List[AssuranceSignal]:
    contract = dict(action_contract or {})
    meta = dict(contract.get("metadata") or {})
    signals: List[AssuranceSignal] = []

    def _maybe_add(name: str, value: Any, source: str, weight: float = 1.0, *, scale_from_pct: bool = False) -> None:
        if value is None or value == "":
            return
        try:
            fv = float(value)
            if scale_from_pct or fv > 1.0:
                fv = fv / 100.0
            fv = _clamp01(fv)
            signals.append(AssuranceSignal(name=name, value=fv, source=source, weight=weight))
        except Exception:
            return

    # Contract/meta-originated confidence.
    for key, weight in (
        ("confidence", 1.0),
        ("intent_confidence", 1.0),
        ("advcu_confidence", 1.1),
        ("neuron_confidence", 1.1),
        ("plan_confidence", 0.9),
        ("executor_confidence", 0.9),
        ("verification_confidence", 1.2),
    ):
        _maybe_add(key, meta.get(key), "contract.metadata", weight)
        _maybe_add(key, contract.get(key), "contract", weight)

    # Governance-derived signal.
    _maybe_add("governance_confidence", governance.get("confidence"), "governance", 1.1)
    _maybe_add("governance_risk_inverse", 100.0 - float(governance.get("risk_score", 0) or 0), "governance", 0.8, scale_from_pct=True)

    # Security-derived signal.
    _maybe_add("security_confidence", security.get("confidence"), "security", 1.0)
    risk_score = security.get("risk_score")
    if risk_score is not None:
        try:
            _maybe_add("security_risk_inverse", 100.0 - float(risk_score), "security", 0.8, scale_from_pct=True)
        except Exception:
            pass

    # If no explicit confidence is available, use a conservative fallback depending on execution mode.
    if not signals:
        mode = _safe_lower(contract.get("execution_mode")) or MODE_DRAFT
        fallback = {
            MODE_SIMULATE: 0.45,
            MODE_DRAFT: 0.40,
            MODE_APPLY: 0.32,
            MODE_ROLLBACK: 0.38,
        }.get(mode, 0.40)
        signals.append(AssuranceSignal(name="fallback_confidence", value=fallback, source="assurance_gate", weight=0.7))

    return signals


def _weighted_confidence(signals: List[AssuranceSignal]) -> float:
    if not signals:
        return 0.0
    total_weight = 0.0
    total = 0.0
    for sig in signals:
        w = float(sig.weight or 0.0)
        total_weight += w
        total += float(sig.value) * w
    if total_weight <= 0.0:
        return 0.0
    return _clamp01(total / total_weight)


def _verification_readiness(action_contract: Dict[str, Any]) -> Tuple[float, bool, List[str]]:
    contract = dict(action_contract or {})
    missing: List[str] = []
    score = 0.0

    verification_checks = contract.get("verification_checks") or []
    expected_result = str(contract.get("expected_result") or "").strip()
    preconditions = contract.get("preconditions") or []

    if isinstance(verification_checks, list) and verification_checks:
        score += 0.55
    else:
        missing.append("verification_checks")

    if expected_result:
        score += 0.25
    else:
        missing.append("expected_result")

    if isinstance(preconditions, list) and preconditions:
        score += 0.10
    else:
        missing.append("preconditions")

    # A small bonus if target + capability are explicit rather than vague.
    if str(contract.get("target") or "").strip() or str(contract.get("target_ref") or "").strip():
        score += 0.05
    else:
        missing.append("target")

    if str(contract.get("capability_name") or "").strip():
        score += 0.05
    else:
        missing.append("capability_name")

    ready = score >= 0.60
    return (_clamp01(score), ready, missing)


def _rollback_readiness(action_contract: Dict[str, Any]) -> Tuple[float, bool, List[str]]:
    contract = dict(action_contract or {})
    risk_level = str(contract.get("risk_level") or RISK_TIER_0)
    mode = str(contract.get("execution_mode") or MODE_DRAFT)
    missing: List[str] = []

    if mode != MODE_APPLY:
        return (1.0, True, missing)

    rollback_plan = contract.get("rollback_plan") or []
    if risk_level in {RISK_TIER_0, RISK_TIER_1}:
        if rollback_plan:
            return (1.0, True, missing)
        return (0.80, True, missing)

    if isinstance(rollback_plan, list) and rollback_plan:
        return (1.0, True, missing)

    missing.append("rollback_plan")
    return (0.0, False, missing)


def _has_explicit_confirmation(action_contract: Dict[str, Any], governance: Dict[str, Any], security: Dict[str, Any]) -> bool:
    contract = dict(action_contract or {})
    meta = dict(contract.get("metadata") or {})
    if bool(meta.get("user_confirmed") or meta.get("confirmed") or meta.get("approval_granted") or meta.get("operator_confirmed")):
        return True
    if bool(contract.get("user_confirmed") or contract.get("confirmed")):
        return True
    if bool(governance.get("user_confirmed") or governance.get("confirmed") or governance.get("approval_granted")):
        return True
    if bool(security.get("user_confirmed") or security.get("confirmed") or security.get("approval_granted")):
        return True
    return False


def _trust_tier(security: Dict[str, Any]) -> str:
    caller = security.get("caller_identity") or {}
    if isinstance(caller, dict):
        tier = str(caller.get("trust_tier") or "").strip()
        if tier:
            return tier
    return TRUST_TIER_UNKNOWN


def _seems_hazardous(action_contract: Dict[str, Any]) -> bool:
    contract = dict(action_contract or {})
    joined = " ".join([
        _safe_lower(contract.get("action_type")),
        _safe_lower(contract.get("capability_name")),
        _safe_lower(contract.get("target")),
        _safe_lower(contract.get("normalized_text")),
        " ".join([_safe_lower(x) for x in (contract.get("required_permissions") or [])]),
    ])
    keywords = (
        "delete", "wipe", "format", "firewall", "driver.control", "camera", "microphone", "plc",
        "cnc", "machine", "serial", "usb", "remote", "public", "ingress", "port", "terminal.execute",
        "admin", "service", "boot", "network",
    )
    return any(k in joined for k in keywords)


def _compare_hint(action_contract: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if _Compare is None:
        return None
    contract = dict(action_contract or {})
    text = str(contract.get("normalized_text") or "").strip()
    expected = str(contract.get("expected_result") or "").strip()
    if not text or not expected:
        return None
    try:
        fn = getattr(_Compare, "evaluate_similarity", None)
        if callable(fn):
            score = fn(text, expected)
            if score is not None:
                try:
                    score_f = float(score)
                    if score_f > 1.0:
                        score_f = score_f / 100.0
                    return {"name": "compare_similarity", "score": _clamp01(score_f)}
                except Exception:
                    return None
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# Main assurance evaluation
# ---------------------------------------------------------------------------
def evaluate_action_assurance(action_contract: Dict[str, Any], governance: Optional[Dict[str, Any]] = None, security: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    governance = dict(governance or {})
    security = dict(security or {})
    contract = dict(action_contract or {})
    review_id = f"asg_{uuid.uuid4().hex}"

    action_type = str(contract.get("action_type") or "unknown")
    capability_name = str(contract.get("capability_name") or "unknown")
    execution_mode = str(contract.get("execution_mode") or MODE_DRAFT)
    risk_level = str(contract.get("risk_level") or RISK_TIER_0)

    policy = _policy_snapshot()
    threshold = _threshold_for(execution_mode, risk_level)
    signals = _extract_confidence_signals(contract, governance, security)
    weighted_conf = _weighted_confidence(signals)

    compare_hint = _compare_hint(contract)
    if compare_hint and compare_hint.get("score") is not None:
        try:
            signals.append(AssuranceSignal(name="compare_similarity", value=_clamp01(compare_hint["score"]), source="compare", weight=0.6))
            weighted_conf = _weighted_confidence(signals)
        except Exception:
            pass

    verification_score, verification_ready, verification_missing = _verification_readiness(contract)
    rollback_score, rollback_ready, rollback_missing = _rollback_readiness(contract)
    trust_tier = _trust_tier(security)
    trust_bonus = _trust_bonus(trust_tier)
    mode_bonus = _mode_bonus(execution_mode)
    risk_penalty = _risk_weight(risk_level)

    # Final assurance score intentionally blends evidence + structure + trust.
    assurance_score = weighted_conf
    assurance_score += (verification_score * 0.30)
    assurance_score += (rollback_score * 0.10)
    assurance_score += trust_bonus
    assurance_score += mode_bonus
    assurance_score -= (risk_penalty * 0.12)
    assurance_score = _clamp01(assurance_score)

    review = AssuranceReview(
        review_id=review_id,
        ts=_now_iso(),
        decision=DECISION_ALLOW,
        allow=True,
        require_user=False,
        action_type=action_type,
        capability_name=capability_name,
        execution_mode=execution_mode,
        risk_level=risk_level,
        confidence=weighted_conf,
        assurance_score=assurance_score,
        threshold=threshold,
        verification_ready=verification_ready,
        rollback_ready=rollback_ready,
        policy_snapshot=policy,
        meta={
            "verification_score": verification_score,
            "rollback_score": rollback_score,
            "trust_tier": trust_tier,
            "risk_penalty": risk_penalty,
        },
    )

    # Governance and security remain upstream authoritative. Assurance never re-allows what they denied.
    if governance.get("allow") is False or _safe_lower(governance.get("decision")) in {"deny", "defer"}:
        review.decision = DECISION_DENY
        review.allow = False
        review.risk_factors.append("governance_block_present")
        review.reasons.append("Upstream governance already blocked this action.")

    sec_decision = _safe_lower(security.get("decision"))
    if security.get("allow") is False or sec_decision in {"deny", "quarantine"}:
        review.decision = DECISION_DENY
        review.allow = False
        review.risk_factors.append("security_block_present")
        review.reasons.append("SecurityGovernor already blocked this action.")

    if trust_tier == TRUST_TIER_QUARANTINED:
        review.decision = DECISION_DENY
        review.allow = False
        review.risk_factors.append("quarantined_trust_tier")
        review.reasons.append("Quarantined callers may not pass assurance review.")

    confirmed = _has_explicit_confirmation(contract, governance, security)
    requires_confirmation = bool(contract.get("requires_confirmation"))
    if requires_confirmation and not confirmed:
        review.decision = DECISION_REQUIRE_USER
        review.allow = False
        review.require_user = True
        review.risk_factors.append("confirmation_missing")
        review.reasons.append("ActionContract requires explicit confirmation and none was provided.")

    if execution_mode == MODE_APPLY and not verification_ready:
        review.decision = DECISION_DENY
        review.allow = False
        review.risk_factors.append("verification_plan_insufficient")
        review.missing_requirements.extend(verification_missing)
        review.reasons.append("Apply-mode action lacks a strong enough verification plan.")

    if execution_mode == MODE_APPLY and risk_level in {RISK_TIER_2, RISK_TIER_3, RISK_TIER_4} and not rollback_ready and policy.get("require_rollback_for_apply", True):
        review.decision = DECISION_DENY
        review.allow = False
        review.risk_factors.append("rollback_plan_insufficient")
        review.missing_requirements.extend(rollback_missing)
        review.reasons.append("Non-trivial apply-mode action lacks rollback readiness.")

    if _seems_hazardous(contract) and weighted_conf < 0.55:
        review.decision = DECISION_DENY
        review.allow = False
        review.risk_factors.append("hazardous_low_confidence")
        review.reasons.append("Hazardous-looking action request does not have enough confidence evidence.")

    if policy.get("riskier_actions_require_stronger_assurance", True) and assurance_score < threshold:
        if execution_mode == MODE_APPLY:
            review.decision = DECISION_DENY
            review.allow = False
            review.risk_factors.append("assurance_below_threshold")
            review.reasons.append(f"Assurance score {assurance_score:.2f} is below required threshold {threshold:.2f} for this risk tier.")
        else:
            review.decision = DECISION_SIMULATE_ONLY
            review.allow = False
            review.risk_factors.append("assurance_below_threshold")
            review.reasons.append(f"Assurance score {assurance_score:.2f} is below required threshold {threshold:.2f}; downgrade to simulate/draft path.")
            review.constraints["allowed_execution_mode"] = MODE_SIMULATE

    # Higher-risk unknown callers should not pass unless evidence and confirmation are strong.
    if trust_tier in {TRUST_TIER_UNKNOWN, TRUST_TIER_UNVERIFIED} and risk_level in {RISK_TIER_3, RISK_TIER_4}:
        if not confirmed or assurance_score < max(threshold, 0.88):
            review.decision = DECISION_DENY
            review.allow = False
            review.risk_factors.append("untrusted_high_risk_low_assurance")
            review.reasons.append("Untrusted high-risk action lacks sufficient assurance for execution.")

    if execution_mode == MODE_ROLLBACK and not rollback_ready:
        review.decision = DECISION_DENY
        review.allow = False
        review.risk_factors.append("rollback_requested_without_plan")
        review.reasons.append("Rollback was requested but no rollback plan is available.")

    if policy.get("safe_mode") and execution_mode == MODE_APPLY and risk_level in {RISK_TIER_3, RISK_TIER_4}:
        review.decision = DECISION_SIMULATE_ONLY
        review.allow = False
        review.risk_factors.append("safe_mode_high_risk_apply")
        review.reasons.append("SAFE_MODE requires simulation/draft rather than high-risk apply-mode execution.")
        review.constraints["allowed_execution_mode"] = MODE_SIMULATE

    if policy.get("local_only") and risk_level == RISK_TIER_4 and "network" in _safe_lower(capability_name):
        review.decision = DECISION_DENY
        review.allow = False
        review.risk_factors.append("local_only_remote_action")
        review.reasons.append("LOCAL_ONLY_MODE blocks remote/network-destructive execution in assurance review.")

    if not review.reasons:
        review.reasons.append("AssuranceGate found adequate confidence and readiness for the requested execution mode.")

    if review.allow:
        review.constraints.setdefault("verify_before_success", True)
        review.constraints.setdefault("audit_required", True)
        if execution_mode == MODE_APPLY and risk_level in {RISK_TIER_2, RISK_TIER_3, RISK_TIER_4}:
            review.constraints.setdefault("postcondition_verification_required", True)

    review.signals = [asdict(sig) for sig in signals]

    _log_event(review)
    _write_snapshot(review)
    return review.to_dict()


# ---------------------------------------------------------------------------
# Compatibility aliases for OperatorCore
# ---------------------------------------------------------------------------
def review_action_assurance(action_contract: Dict[str, Any], governance: Optional[Dict[str, Any]] = None, security: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return evaluate_action_assurance(action_contract, governance, security)



def assure_action(action_contract: Dict[str, Any], governance: Optional[Dict[str, Any]] = None, security: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return evaluate_action_assurance(action_contract, governance, security)


# ---------------------------------------------------------------------------
# Optional helpers for diagnostics / later wiring
# ---------------------------------------------------------------------------
def get_assurance_snapshot() -> Dict[str, Any]:
    try:
        path = _snapshot_path()
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}



def get_assurance_profile() -> Dict[str, Any]:
    return {
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "thresholds": {
            MODE_SIMULATE: {
                RISK_TIER_0: 0.10,
                RISK_TIER_1: 0.15,
                RISK_TIER_2: 0.25,
                RISK_TIER_3: 0.35,
                RISK_TIER_4: 0.45,
            },
            MODE_DRAFT: {
                RISK_TIER_0: 0.20,
                RISK_TIER_1: 0.28,
                RISK_TIER_2: 0.42,
                RISK_TIER_3: 0.58,
                RISK_TIER_4: 0.70,
            },
            MODE_APPLY: {
                RISK_TIER_0: 0.45,
                RISK_TIER_1: 0.55,
                RISK_TIER_2: 0.70,
                RISK_TIER_3: 0.84,
                RISK_TIER_4: 0.92,
            },
            MODE_ROLLBACK: {
                RISK_TIER_0: 0.30,
                RISK_TIER_1: 0.38,
                RISK_TIER_2: 0.52,
                RISK_TIER_3: 0.60,
                RISK_TIER_4: 0.70,
            },
        },
        "signals_considered": [
            "confidence",
            "intent_confidence",
            "advcu_confidence",
            "neuron_confidence",
            "plan_confidence",
            "executor_confidence",
            "verification_confidence",
            "governance_confidence",
            "governance_risk_inverse",
            "security_confidence",
            "security_risk_inverse",
        ],
        "doctrine": {
            "no_hallucinated_actuation": True,
            "verification_required_for_apply": True,
            "rollback_required_for_nontrivial_apply": True,
            "riskier_actions_require_stronger_assurance": True,
            "fail_closed_on_low_assurance": True,
        },
    }


if __name__ == "__main__":
    demo_contract = {
        "action_type": "open_app_and_write_summary",
        "capability_name": "software.word.write",
        "execution_mode": MODE_APPLY,
        "risk_level": RISK_TIER_2,
        "requires_confirmation": False,
        "expected_result": "Word is opened and a summary is written into a document.",
        "verification_checks": [
            {"name": "word_running", "required": True},
            {"name": "document_written", "required": True},
        ],
        "rollback_plan": [
            {"name": "close_unsaved_document", "required": False},
        ],
        "metadata": {
            "advcu_confidence": 0.86,
            "neuron_confidence": 0.82,
        },
    }
    demo_governance = {"decision": "ALLOW", "allow": True, "risk_score": 30}
    demo_security = {
        "decision": "ALLOW",
        "allow": True,
        "risk_score": 30,
        "caller_identity": {"caller_id": "core:demo", "trust_tier": TRUST_TIER_CORE},
    }
    print(json.dumps(evaluate_action_assurance(demo_contract, demo_governance, demo_security), indent=2, ensure_ascii=False))
