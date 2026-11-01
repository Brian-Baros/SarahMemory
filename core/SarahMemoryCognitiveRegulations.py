"""--==The SarahMemory Project==--
File: SarahMemoryCognitiveRegulations.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.1.0a (Enterprise)
Date: 2026-11-01
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
PURPOSE
-------
Cognitive self-regulation: decide whether more cognition, information gathering,
resource consumption, or agent activity remains justified by a CURRENT mission.

This module is a deliberative, evidence-only advisor. It does NOT authorize,
execute, terminate processes, query models, scan files, call networks, issue
passports, persist data, or override SafetyPolicies/AgentFirewall/SMGET/
AssuranceGate/OperatorCore/CognitiveServices. A "continue" recommendation
is never permission to execute. The responsible owner must verify evidence,
policy, consent, and mission state, and enforce any stops or pauses.

No side effects at import; Python 3.11; standard-library only.
"""
from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "cognitive_self_regulation_advisor"
# CATEGORY = "cognition"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "system"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "cognitive_regulations"
# FAMILY = "core_cognition"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-11-01"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime / GCAIOS"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Evidence-only judgment on necessity/sufficiency/cessation. Cannot authorize, execute, or self-modify."
# --- SARAHMETA END ---

import copy
import hashlib
import hmac
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

MODULE_NAME = "SarahMemoryCognitiveRegulations"
MODULE_VERSION = "0.1.0"
REGULATION_SCHEMA = "SarahMemory.CognitiveRegulation.v1"
SMUGCC_EXTENSION_SCHEMA = "SarahMemory.SMUGCC.cognitive_regulation.v1"
SMUGCC_SCHEMA = "SarahMemory.SMUGCC.v1"

# All results are ADVISORY. Enforcement lives in the existing governance stack.
CONTINUE_WARRANTED = "CONTINUE_WARRANTED"
STOP_SATISFIED = "STOP_SATISFIED"
STOP_BOUNDARY = "STOP_BOUNDARY"
REDUCE_SCOPE = "REDUCE_SCOPE"
DEFER = "DEFER"
REQUIRE_USER = "REQUIRE_USER"
PAUSE_REVIEW = "PAUSE_REVIEW"

VALID_RECOMMENDATIONS = frozenset({
    CONTINUE_WARRANTED, STOP_SATISFIED, STOP_BOUNDARY, REDUCE_SCOPE,
    DEFER, REQUIRE_USER, PAUSE_REVIEW,
})

@dataclass(frozen=True)
class RegulationPolicy:
    """Illustrative tuning parameters; must be calibrated by task class.

    These are NOT universal measures of intelligence, safety, or correctness.
    A verified-completion receipt, not these scores, governs STOP_SATISFIED.
    """
    min_relevance: float = 0.60
    min_evidence_quality: float = 0.82
    low_marginal_gain: float = 0.025
    plateau_observations: int = 3
    review_risk: float = 0.75
    review_cost: float = 0.85

    def __post_init__(self) -> None:
        for field in ("min_relevance", "min_evidence_quality", "low_marginal_gain",
                      "review_risk", "review_cost"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("invalid policy metric: " + field)
        if isinstance(self.plateau_observations, bool) or not isinstance(self.plateau_observations, int) or not 2 <= self.plateau_observations <= 12:
            raise ValueError("plateau_observations must be an integer between 2 and 12")


SML_ORGAN_METADATA = {
    "name": MODULE_NAME,
    "role": "cognitive_self_regulation_advisor",
    "version": MODULE_VERSION,
    "capabilities": ["regulation_assessment", "sufficiency_advice", "smugcc_regulation_extension"],
    "execution_authority": False,
    "internal_only": True,
    "metadata": {"advisory_only": True, "side_effect_free_core": True},
}


def _metric(raw: Any, key: str, errors: list[str]) -> Optional[float]:
    """Do not fabricate unknown scores or accept NaN, booleans, or strings."""
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or not 0.0 <= raw <= 1.0:
        errors.append("invalid_" + key)
        return None
    return float(raw)


def _count(raw: Any, key: str, errors: list[str]) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw <= 100_000:
        errors.append("invalid_" + key)
        return None
    return raw


def _literal_true(value: Any) -> bool:
    """Avoid truthy string values such as 'false' accidentally granting a state."""
    return value is True


def _digest(value: Mapping[str, Any]) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _decision(recommendation: str, reason_codes: Sequence[str], *, mission_id: str = "",
              task_id: str = "", evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return no prompt, raw file path, secret, or original mission content."""
    if recommendation not in VALID_RECOMMENDATIONS:
        raise ValueError("invalid recommendation")
    facts = dict(evidence or {})
    receipt = {
        "schema": REGULATION_SCHEMA,
        "organ": MODULE_NAME,
        "version": MODULE_VERSION,
        "mission_id": mission_id[:160],
        "task_id": task_id[:160],
        "recommendation": recommendation,
        "reason_codes": list(dict.fromkeys(reason_codes))[:24],
        "evidence_summary": facts,
        "advisory_only": True,
        "execution_authority": False,
        "action_authorized": False,
        "passport_issued": False,
    }
    receipt["receipt_hash"] = _digest(receipt)
    return receipt


def evaluate_regulation(context: Mapping[str, Any], *, policy: Optional[RegulationPolicy] = None) -> Dict[str, Any]:
    """Evaluate current mission sufficiency without side effects.

    Expected context (all inputs are evidence claims, NOT authority):
      mission_id, task_id, objective, governance_decision,
      user_stop_requested, consent_revoked, high_impact, consent_verified,
      mission_completed, compare_verified, remaining_steps, evidence_quality,
      relevance, expected_gain, risk, cost, gain_history, budget_exhausted,
      unresolved_critical_step, user_requested_continuation.

    Caller MUST obtain fresh authoritative governance/Compare/Compass state.
    This function does not authenticate callers or owner reports.
    """
    p = policy or RegulationPolicy()
    if not isinstance(p, RegulationPolicy):
        raise TypeError("policy must be RegulationPolicy")
    if not isinstance(context, Mapping):
        return _decision(DEFER, ["context_not_mapping"])
    errors: list[str] = []
    mid = context.get("mission_id")
    tid = context.get("task_id")
    mission_id = mid if isinstance(mid, str) else ""
    task_id = tid if isinstance(tid, str) else ""
    objective = context.get("objective")
    decision_raw = context.get("governance_decision")
    governance = str(decision_raw).upper() if isinstance(decision_raw, str) else "UNKNOWN"
    # Include only bounded structured observations in audit output.
    relevance = _metric(context.get("relevance"), "relevance", errors)
    quality = _metric(context.get("evidence_quality"), "evidence_quality", errors)
    risk = _metric(context.get("risk"), "risk", errors)
    cost = _metric(context.get("cost"), "cost", errors)
    gain = _metric(context.get("expected_gain"), "expected_gain", errors)
    remaining = _count(context.get("remaining_steps"), "remaining_steps", errors)
    history_raw = context.get("gain_history", [])
    history: list[float] = []
    if not isinstance(history_raw, (tuple, list)) or len(history_raw) > 100:
        errors.append("invalid_gain_history")
    else:
        for value in history_raw:
            parsed = _metric(value, "gain_history", errors)
            if parsed is not None:
                history.append(parsed)
    # Governance flags are evidence claims, not truthy/falsy user strings.
    # Preserve historical optional/missing fields, but fail closed on malformed
    # values supplied by callers (e.g. "false", 0, 1, [] or {}).
    boolean_controls = (
        "user_stop_requested", "consent_revoked", "high_impact",
        "consent_verified", "mission_completed", "compare_verified",
        "budget_exhausted", "unresolved_critical_step",
        "user_requested_continuation",
    )
    for control in boolean_controls:
        if control in context and not isinstance(context[control], bool):
            errors.append("invalid_" + control)
    proof = {
        "governance": governance if governance in {"ALLOW", "DENY", "DEFER", "REQUIRE_USER"} else "UNKNOWN",
        "mission_completed": _literal_true(context.get("mission_completed")),
        "compare_verified": _literal_true(context.get("compare_verified")),
        "remaining_steps": remaining,
        "evidence_quality": quality,
        "relevance": relevance,
        "risk": risk,
        "cost": cost,
        "expected_gain": gain,
        "history_samples": len(history),
        "budget_exhausted": _literal_true(context.get("budget_exhausted")),
        "unresolved_critical_step": _literal_true(context.get("unresolved_critical_step")),
    }

    def emit(state: str, *reasons: str) -> Dict[str, Any]:
        return _decision(state, reasons, mission_id=mission_id, task_id=task_id, evidence=proof)

    # A deny, revocation or user's stop instruction must be observed even if
    # nonessential telemetry is malformed or missing.
    if _literal_true(context.get("user_stop_requested")):
        return emit(STOP_BOUNDARY, "user_requested_stop")
    if _literal_true(context.get("consent_revoked")):
        return emit(STOP_BOUNDARY, "consent_revoked")
    if governance == "DENY":
        return emit(STOP_BOUNDARY, "governor_denied")
    if governance == "REQUIRE_USER":
        return emit(REQUIRE_USER, "governor_requires_user")
    if governance != "ALLOW":
        return emit(DEFER, "governance_not_confirmed")
    if _literal_true(context.get("budget_exhausted")):
        return emit(STOP_BOUNDARY, "authorized_budget_exhausted")
    if _literal_true(context.get("high_impact")) and not _literal_true(context.get("consent_verified")):
        return emit(REQUIRE_USER, "high_impact_requires_verified_consent")
    if not mission_id.strip() or not task_id.strip() or not isinstance(objective, str) or not objective.strip():
        return emit(DEFER, "missing_mission_identity_or_objective")
    if errors:
        return emit(DEFER, *errors[:12])
    if _literal_true(context.get("unresolved_critical_step")):
        return emit(PAUSE_REVIEW, "critical_step_not_resolved")

    completed = _literal_true(context.get("mission_completed"))
    verified = _literal_true(context.get("compare_verified"))
    if completed and verified and remaining == 0:
        if _literal_true(context.get("user_requested_continuation")):
            return emit(PAUSE_REVIEW, "new_scope_requires_new_mission_or_review")
        return emit(STOP_SATISFIED, "completed_and_compare_verified", "no_remaining_steps")
    if completed and not verified:
        return emit(PAUSE_REVIEW, "completion_claim_without_compare_proof")
    if completed and remaining != 0:
        return emit(PAUSE_REVIEW, "completion_claim_conflicts_with_remaining_steps")
    if remaining is None:
        return emit(DEFER, "remaining_steps_unknown")
    if quality is None or relevance is None or gain is None:
        return emit(DEFER, "missing_sufficiency_metrics")
    if risk is None or cost is None:
        return emit(DEFER, "missing_cost_or_risk_evidence")
    if risk >= p.review_risk:
        return emit(PAUSE_REVIEW, "risk_requires_governance_review")
    if cost >= p.review_cost:
        return emit(REDUCE_SCOPE, "cost_disproportionate_to_expected_gain")
    if relevance < p.min_relevance:
        return emit(REDUCE_SCOPE, "additional_information_not_mission_relevant")
    plateau = (len(history) >= p.plateau_observations and
               all(v <= p.low_marginal_gain for v in history[-p.plateau_observations:]))
    if plateau and gain <= p.low_marginal_gain and quality >= p.min_evidence_quality:
        # Plateau is NOT proof the mission was completed; NEVER claim success.
        return emit(REDUCE_SCOPE, "diminishing_information_returns", "completion_not_verified")
    if quality < p.min_evidence_quality:
        return emit(CONTINUE_WARRANTED, "more_verified_evidence_needed", "not_execution_authorization")
    if gain <= p.low_marginal_gain:
        return emit(REDUCE_SCOPE, "low_expected_marginal_information_value")
    return emit(CONTINUE_WARRANTED, "continued_work_appears_mission_relevant", "not_execution_authorization")


def build_smugcc_regulation_extension(regulation: Mapping[str, Any]) -> Dict[str, Any]:
    """Produce a bounded, inspectable SMUGCC v1 governance extension.

    An extension is still a CLAIM. TrustRegistry/AssuranceGate/OperatorCore must
    independently verify its provenance; no signature is fabricated here.
    """
    if not isinstance(regulation, Mapping) or regulation.get("schema") != REGULATION_SCHEMA:
        raise ValueError("regulation packet schema mismatch")
    recommendation = regulation.get("recommendation")
    if not isinstance(recommendation, str) or recommendation not in VALID_RECOMMENDATIONS:
        raise ValueError("invalid recommendation")
    if regulation.get("organ") != MODULE_NAME or regulation.get("version") != MODULE_VERSION:
        raise ValueError("regulation producer/version mismatch")
    if (regulation.get("advisory_only") is not True or
            regulation.get("execution_authority") is not False or
            regulation.get("action_authorized") is not False or
            regulation.get("passport_issued") is not False):
        raise ValueError("regulation may not assert execution authority")
    for key in ("mission_id", "task_id"):
        value = regulation.get(key)
        if not isinstance(value, str) or len(value) > 160:
            raise ValueError("invalid regulation " + key)
    reasons = regulation.get("reason_codes")
    if (not isinstance(reasons, list) or len(reasons) > 24 or
            any(not isinstance(reason, str) for reason in reasons)):
        raise ValueError("invalid regulation reason_codes")
    if not isinstance(regulation.get("evidence_summary"), Mapping):
        raise ValueError("invalid regulation evidence_summary")
    recorded_hash = regulation.get("receipt_hash")
    if (not isinstance(recorded_hash, str) or
            re.fullmatch(r"[0-9a-f]{64}", recorded_hash) is None):
        raise ValueError("invalid regulation receipt_hash")
    unsigned = {key: value for key, value in regulation.items() if key != "receipt_hash"}
    try:
        computed_hash = _digest(unsigned)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("regulation receipt not canonicalizable") from error
    if not hmac.compare_digest(recorded_hash, computed_hash):
        raise ValueError("regulation receipt integrity mismatch")
    # This is unkeyed content integrity only; signing/trust remains with
    # TrustRegistry / AssuranceGate / OperatorCore.
    return {
        "schema": SMUGCC_EXTENSION_SCHEMA,
        "producer": MODULE_NAME,
        "mission_id": str(regulation.get("mission_id") or "")[:160],
        "task_id": str(regulation.get("task_id") or "")[:160],
        "recommendation": regulation["recommendation"],
        "reason_codes": list(regulation.get("reason_codes") or [])[:24],
        "receipt_hash": str(regulation.get("receipt_hash") or "")[:64],
        "audit_owner": "SarahMemoryLedger",
        "verification_owner": "SarahMemoryCompare / SarahMemoryAssuranceGate",
        "final_execution_owner": "SarahMemoryOperatorCore",
        "advisory_only": True,
        "execution_authority": False,
        "action_authorized": False,
    }


def attach_regulation_to_smugcc_envelope(envelope: Mapping[str, Any], regulation: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a COPY of a SMUGCC envelope with an optional advice extension.

    Does not call SMUGCC's validator. The integrating caller must validate
    the resulting envelope and register the extension with its schema policy.
    """
    if not isinstance(envelope, Mapping) or envelope.get("schema") != SMUGCC_SCHEMA:
        raise ValueError("expected SMUGCC v1 envelope")
    if not isinstance(envelope.get("governance"), Mapping):
        raise ValueError("missing governance section")
    extension = build_smugcc_regulation_extension(regulation)
    mission = envelope.get("mission")
    if not isinstance(mission, Mapping):
        raise ValueError("missing mission section")
    if mission.get("mission_id") != extension["mission_id"] or mission.get("task_id") != extension["task_id"]:
        raise ValueError("mission/task mismatch: regulation advice cannot be replayed across missions")
    result = copy.deepcopy(dict(envelope))
    existing = result["governance"].get("cognitive_regulation")
    if "cognitive_regulation" in result["governance"] and existing != extension:
        raise ValueError("conflicting cognitive regulation evidence")
    result["governance"]["cognitive_regulation"] = extension
    # Existing grant/permission/passport fields are NOT altered or widened.
    return result


def sml_get_metadata() -> Dict[str, Any]:
    return copy.deepcopy(SML_ORGAN_METADATA)


def sml_health() -> Dict[str, Any]:
    return {
        "status": "Degraded", "availability": 1.0, "integrity": None,
        "performance": None, "reliability": None, "confidence": None,
        "latency_ms": None, "stability": None, "compatibility": None,
        "notes": ["Advisor importable; full SMUGCC/SMGET runtime wiring and calibration not yet validated"],
    }


def sml_diagnostics() -> Dict[str, Any]:
    return {"status": "OK", "component": MODULE_NAME, "sml_adapter": True,
            "metadata": sml_get_metadata(), "health": sml_health(), "execution_authority": False}


def sml_receive_packet(packet: Any, *, action: str = "observe", note: str = "", updates: Any = None) -> Any:
    """Compatibility adapter. No direct imports/registration at module import time."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ=MODULE_NAME, action=action,
                                note=note or "cognitive regulation observation", updates=updates)
    except (ImportError, AttributeError):
        return packet


def get_regulation_contract() -> Dict[str, Any]:
    """Human-readable integration metadata for tools and governance reviewers."""
    return {
        "organ": MODULE_NAME,
        "version": MODULE_VERSION,
        "input_contract": "caller-supplied mission/governance/Compare/Compass evidence; owner verification required",
        "output_schema": REGULATION_SCHEMA,
        "smugcc_extension_schema": SMUGCC_EXTENSION_SCHEMA,
        "recommendations": sorted(VALID_RECOMMENDATIONS),
        "execution_authority": False,
        "requires_existing_governance": True,
        "writes_memory": False,
        "calls_network": False,
        "changes_runtime": False,
    }


__all__ = [
    "MODULE_NAME", "MODULE_VERSION", "REGULATION_SCHEMA", "SMUGCC_EXTENSION_SCHEMA",
    "RegulationPolicy", "evaluate_regulation", "build_smugcc_regulation_extension",
    "attach_regulation_to_smugcc_envelope", "get_regulation_contract",
    "sml_get_metadata", "sml_health", "sml_diagnostics", "sml_receive_packet",
]
