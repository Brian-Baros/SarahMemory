"""--==The SarahMemory Project==--
File: SarahMemorySMLProtocol.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0-alpha-sml-0.1
Date: 2026-08-10
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.

===============================================================================
SarahMemory SML Protocol / Governed Cognitive Packet Substrate

PURPOSE:
- Provide the canonical SML Packet and protocol substrate for SarahMemory AiOS.
- Coordinate mission classification, packet validation, organ registration,
  capability negotiation, Ω transition control, routing, diagnostics, health,
  serialization, and bounded patch-packet creation.
- Remain local-first, deterministic, auditable, and model-independent.

NON-GOALS:
- This file does NOT answer user questions.
- This file does NOT execute filesystem, shell, driver, browser, network, model,
  desktop, or hardware actions.
- This file does NOT replace LogicCalc, AdvCU, Adaptive, AgentFirewall,
  SecurityGovernor, Compare, Compass, Ledger, OperatorCore, or Diagnostics.
- This file does NOT auto-merge patches into production Core.

AUTHORITY MODEL:
- SML computes protocol state and routing candidates.
- Governance organs decide whether execution may proceed.
- OperatorCore remains the execution choke point.
- Ledger records transitions and receipts.
- Human operator retains final authority.

DESIGN STATUS:
- Initial reference implementation for integration testing.
- Safe to import without side effects.
- No third-party dependencies required.
===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A-"
# ROLE = "sml_protocol_core"
# CATEGORY = "cognitive_protocol"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "sml"
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "sml_protocol"
# FAMILY = "core_cognitive_protocol"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-08-10"
# VALIDATION_TIME = "23:13:00"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Canonical SML Packet, Ω registry, organ registry, protocol negotiation, bounded routing, integrity, diagnostics, and serialization substrate. Coordinates cognition; does not execute actions."
# --- SARAHMETA END ---

import copy
import hashlib
import hmac
import importlib.util
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union


PROJECT_VERSION = "v9.0.0"
SML_PROTOCOL_VERSION = "SML/1.0"
SML_PACKET_VERSION = 1
SML_OMEGA_REGISTRY_VERSION = "Ω/1.0"
MODULE_NAME = "SarahMemorySMLProtocol"


# =============================================================================
# Small deterministic utilities
# =============================================================================


def _utc_now() -> str:
    """Return a stable UTC timestamp in ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            out.append(str(item))
        return out
    return [str(value)]


def _coerce_set(value: Any) -> Set[str]:
    return {v for v in _coerce_list(value) if v}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def _sha256_obj(value: Any) -> str:
    return _sha256_text(_stable_json(value))


def _normalize_token(text: Any) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(text or "").strip().lower()).strip("_") or "unknown"


def _bounded_text(value: Any, limit: int = 2048) -> str:
    text = str(value or "")
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _redact_sensitive_text(text: str) -> str:
    """Redact common secret-shaped values in diagnostic strings."""
    if not text:
        return ""
    redacted = re.sub(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[^\s,;]+", r"\1=<redacted>", text)
    redacted = re.sub(r"(?i)(bearer)\s+[A-Za-z0-9._\-]+", r"\1 <redacted>", redacted)
    return redacted


# =============================================================================
# Enumerations
# =============================================================================


class SMLStatus(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DENIED = "DENIED"
    PENDING = "PENDING"


class IdentityRole(str, Enum):
    USER = "User"
    DEVELOPER = "Developer"
    OPERATOR = "Operator"
    KERNEL = "Kernel"
    AGENT = "Agent"
    SYSTEM = "System"
    UNKNOWN = "Unknown"


class MissionType(str, Enum):
    UNKNOWN = "Unknown"
    KNOWLEDGE = "Knowledge"
    GENERAL_KNOWLEDGE = "GeneralKnowledge"
    NUMERIC_FORMAT = "NumericFormat"
    SELF_STATE = "SelfState"
    AFFECTIVE_STATE = "AffectiveState"
    CAPABILITY = "Capability"
    LANGUAGE_DISAMBIGUATION = "LanguageDisambiguation"
    CREATIVE_GENERATION = "CreativeGeneration"
    CONVERSATION = "Conversation"
    PROGRAMMING = "Programming"
    RESEARCH = "Research"
    PLANNING = "Planning"
    FILESYSTEM = "Filesystem"
    NETWORK = "Network"
    HARDWARE = "Hardware"
    MEMORY = "Memory"
    VISION = "Vision"
    VOICE = "Voice"
    SECURITY = "Security"
    DIAGNOSTICS = "Diagnostics"
    REPAIR = "Repair"
    LEARNING = "Learning"
    EXECUTION = "Execution"
    PATCH = "Patch"
    GOVERNANCE = "Governance"


class CognitiveState(str, Enum):
    CREATED = "Created"
    OBSERVED = "Observed"
    CLASSIFIED = "Classified"
    CONTEXTUALIZED = "Contextualized"
    ADAPTIVE = "Adaptive"
    KNOWLEDGE = "Knowledge"
    ROUTED = "Routed"
    REASONING = "Reasoning"
    PLANNING = "Planning"
    VERIFIED = "Verified"
    AUTHORIZED = "Authorized"
    EXECUTING = "Executing"
    LEARNING = "Learning"
    COMPLETED = "Completed"
    FAILED = "Failed"
    ARCHIVED = "Archived"
    RECOVERING = "Recovering"


class GovernanceDecision(str, Enum):
    PENDING = "Pending"
    APPROVED = "Approved"
    DENIED = "Denied"
    ESCALATED = "Escalated"
    REQUIRE_USER = "RequireUser"
    RECOVERY = "Recovery"
    ROLLBACK = "Rollback"


class QMathState(str, Enum):
    """SML/Q-Mathematics cognitive route-control grammar.

    These operators describe how cognition should move. They are protocol
    semantics for routing, source selection, validation, fallback, composition,
    and bounded loops. They do not contain answers, authorize execution, or
    replace governance.
    """
    IF = "IF"
    OR = "OR"
    SAME = "SAME"
    WHEN = "WHEN"
    ELSE = "ELSE"
    AND = "AND"
    NEITHER = "NEITHER"
    NOT = "NOT"
    WHILE = "WHILE"


class SMLStopCondition(str, Enum):
    """Hidden loop-governance STOP states.

    STOP is intentionally not a seventh butterfly wing. It is a core restraint
    that prevents WHILE from degrading into uncontrolled recursion.
    """
    SUCCESS_STOP = "SUCCESS_STOP"
    SAFE_STOP = "SAFE_STOP"
    UNKNOWN_STOP = "UNKNOWN_STOP"
    USER_HELP_STOP = "USER_HELP_STOP"
    RESOURCE_STOP = "RESOURCE_STOP"
    STAGNATION_STOP = "STAGNATION_STOP"
    CONFLICT_STOP = "CONFLICT_STOP"
    AUTHORITY_STOP = "AUTHORITY_STOP"


class HealthStatus(str, Enum):
    HEALTHY = "Healthy"
    WARNING = "Warning"
    CRITICAL = "Critical"
    RECOVERING = "Recovering"
    OFFLINE = "Offline"
    UNKNOWN = "Unknown"


class OrganCategory(str, Enum):
    INPUT = "Input"
    REASONING = "Reasoning"
    MEMORY = "Memory"
    GOVERNANCE = "Governance"
    EXECUTION = "Execution"
    LEARNING = "Learning"
    DIAGNOSTICS = "Diagnostics"
    PRESENTATION = "Presentation"
    INFRASTRUCTURE = "Infrastructure"
    PROTOCOL = "Protocol"
    UNKNOWN = "Unknown"


class Authority(str, Enum):
    READ = "Read"
    WRITE = "Write"
    EXECUTE = "Execute"
    DELETE = "Delete"
    MODIFY = "Modify"
    KERNEL = "Kernel"
    RESEARCH = "Research"
    DEVELOPER = "Developer"
    MEMORY = "Memory"
    NETWORK = "Network"
    FILESYSTEM = "Filesystem"
    LEARNING = "Learning"
    DIAGNOSTICS = "Diagnostics"
    PATCH = "Patch"
    NONE = "None"


class ErrorClass(str, Enum):
    PACKET = "Class 100 — Packet Error"
    PROTOCOL = "Class 200 — Protocol Error"
    ORGAN = "Class 300 — Organ Error"
    GOVERNANCE = "Class 400 — Governance Error"
    EXECUTION = "Class 500 — Execution Error"
    LEARNING = "Class 600 — Learning Error"


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class SMLValidationIssue:
    code: str
    message: str
    severity: str = "ERROR"
    field: str = ""
    error_class: str = ErrorClass.PROTOCOL.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "field": self.field,
            "error_class": self.error_class,
        }


@dataclass
class SMLHealthVector:
    status: str = HealthStatus.UNKNOWN.value
    availability: float = 1.0
    integrity: float = 1.0
    performance: float = 1.0
    reliability: float = 1.0
    confidence: float = 1.0
    latency_ms: float = 0.0
    stability: float = 1.0
    compatibility: float = 1.0
    notes: List[str] = field(default_factory=list)

    def score(self) -> float:
        values = [
            self.availability,
            self.integrity,
            self.performance,
            self.reliability,
            self.confidence,
            self.stability,
            self.compatibility,
        ]
        clean = [max(0.0, min(1.0, float(v))) for v in values]
        return round(sum(clean) / len(clean), 4) if clean else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "availability": self.availability,
            "integrity": self.integrity,
            "performance": self.performance,
            "reliability": self.reliability,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "stability": self.stability,
            "compatibility": self.compatibility,
            "score": self.score(),
            "notes": list(self.notes),
        }


@dataclass
class SMLDiagnosticsReport:
    status: str = SMLStatus.OK.value
    component: str = MODULE_NAME
    generated_at: str = field(default_factory=_utc_now)
    issues: List[SMLValidationIssue] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def add_issue(self, code: str, message: str, severity: str = "ERROR", field: str = "", error_class: str = ErrorClass.PROTOCOL.value) -> None:
        self.issues.append(SMLValidationIssue(code=code, message=message, severity=severity, field=field, error_class=error_class))
        if severity.upper() == "ERROR":
            self.status = SMLStatus.ERROR.value
        elif self.status == SMLStatus.OK.value:
            self.status = SMLStatus.WARNING.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "component": self.component,
            "generated_at": self.generated_at,
            "issues": [x.to_dict() for x in self.issues],
            "warnings": list(self.warnings),
            "metrics": dict(self.metrics),
        }


@dataclass
class SMLOmegaTransition:
    transition_id: str
    name: str
    version: str = SML_OMEGA_REGISTRY_VERSION
    description: str = ""
    input_states: List[str] = field(default_factory=list)
    output_state: str = CognitiveState.CREATED.value
    required_authority: List[str] = field(default_factory=list)
    required_organ: str = ""
    compatible_missions: List[str] = field(default_factory=list)
    validation_rules: List[str] = field(default_factory=list)
    rollback_strategy: str = "record_and_recover"
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    immutable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transition_id": self.transition_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "input_states": list(self.input_states),
            "output_state": self.output_state,
            "required_authority": list(self.required_authority),
            "required_organ": self.required_organ,
            "compatible_missions": list(self.compatible_missions),
            "validation_rules": list(self.validation_rules),
            "rollback_strategy": self.rollback_strategy,
            "diagnostics": dict(self.diagnostics),
            "immutable": self.immutable,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SMLOmegaTransition":
        return cls(
            transition_id=str(data.get("transition_id") or data.get("id") or ""),
            name=str(data.get("name") or "Unnamed Transition"),
            version=str(data.get("version") or SML_OMEGA_REGISTRY_VERSION),
            description=str(data.get("description") or ""),
            input_states=_coerce_list(data.get("input_states")),
            output_state=str(data.get("output_state") or CognitiveState.CREATED.value),
            required_authority=_coerce_list(data.get("required_authority")),
            required_organ=str(data.get("required_organ") or ""),
            compatible_missions=_coerce_list(data.get("compatible_missions")),
            validation_rules=_coerce_list(data.get("validation_rules")),
            rollback_strategy=str(data.get("rollback_strategy") or "record_and_recover"),
            diagnostics=dict(data.get("diagnostics") or {}),
            immutable=bool(data.get("immutable", True)),
        )


@dataclass
class SMLOrganMetadata:
    name: str
    version: str = PROJECT_VERSION
    category: str = OrganCategory.UNKNOWN.value
    protocol_version: str = SML_PROTOCOL_VERSION
    packet_version: int = SML_PACKET_VERSION
    omega_registry_version: str = SML_OMEGA_REGISTRY_VERSION
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    supported_missions: List[str] = field(default_factory=list)
    supported_omega: List[str] = field(default_factory=list)
    required_authority: List[str] = field(default_factory=list)
    priority: int = 50
    trust_level: str = "unverified"
    checksum: str = ""
    diagnostics_version: str = "diagnostics/1.0"
    health_version: str = "health/1.0"
    source_path: str = ""
    internal_only: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "category": self.category,
            "protocol_version": self.protocol_version,
            "packet_version": self.packet_version,
            "omega_registry_version": self.omega_registry_version,
            "capabilities": list(self.capabilities),
            "dependencies": list(self.dependencies),
            "supported_missions": list(self.supported_missions),
            "supported_omega": list(self.supported_omega),
            "required_authority": list(self.required_authority),
            "priority": self.priority,
            "trust_level": self.trust_level,
            "checksum": self.checksum,
            "diagnostics_version": self.diagnostics_version,
            "health_version": self.health_version,
            "source_path": self.source_path,
            "internal_only": self.internal_only,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SMLOrganMetadata":
        return cls(
            name=str(data.get("name") or data.get("organ_name") or "unknown_organ"),
            version=str(data.get("version") or PROJECT_VERSION),
            category=str(data.get("category") or OrganCategory.UNKNOWN.value),
            protocol_version=str(data.get("protocol_version") or SML_PROTOCOL_VERSION),
            packet_version=int(data.get("packet_version") or SML_PACKET_VERSION),
            omega_registry_version=str(data.get("omega_registry_version") or SML_OMEGA_REGISTRY_VERSION),
            capabilities=_coerce_list(data.get("capabilities")),
            dependencies=_coerce_list(data.get("dependencies")),
            supported_missions=_coerce_list(data.get("supported_missions") or data.get("missions")),
            supported_omega=_coerce_list(data.get("supported_omega") or data.get("omega")),
            required_authority=_coerce_list(data.get("required_authority")),
            priority=int(data.get("priority") or 50),
            trust_level=str(data.get("trust_level") or "unverified"),
            checksum=str(data.get("checksum") or ""),
            diagnostics_version=str(data.get("diagnostics_version") or "diagnostics/1.0"),
            health_version=str(data.get("health_version") or "health/1.0"),
            source_path=str(data.get("source_path") or ""),
            internal_only=bool(data.get("internal_only", True)),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class SMLPacket:
    """Canonical cognitive packet.

    The packet is the smallest governed unit of cognition inside SarahMemory.
    Identity-bearing fields remain stable; authorized engines mutate packet state
    by appending ledger/organ-history records and replacing specific protocol
    sections.
    """

    packet_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    protocol_version: str = SML_PROTOCOL_VERSION
    packet_version: int = SML_PACKET_VERSION
    created_at: str = field(default_factory=_utc_now)
    creator_organ: str = MODULE_NAME
    checksum: str = ""
    packet_signature: str = ""
    identity: Dict[str, Any] = field(default_factory=lambda: {"primary": IdentityRole.USER.value})
    mission: Dict[str, Any] = field(default_factory=lambda: {"primary": MissionType.UNKNOWN.value, "secondary": [], "confidence": 0.0})
    context: Dict[str, Any] = field(default_factory=dict)
    adaptive: Dict[str, Any] = field(default_factory=lambda: {"mode": "Focused", "vector": {}})
    knowledge: Dict[str, Any] = field(default_factory=lambda: {"sources": [], "selected": [], "fusion": [], "trust": {}})
    pipeline: List[str] = field(default_factory=list)
    authority: Dict[str, Any] = field(default_factory=lambda: {"requested": [Authority.READ.value], "granted": [], "required": [], "least_authority": True})
    governance: Dict[str, Any] = field(default_factory=lambda: {"decision": GovernanceDecision.PENDING.value, "risk_score": 0, "reasons": []})
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    health: Dict[str, Any] = field(default_factory=lambda: SMLHealthVector(status=HealthStatus.UNKNOWN.value).to_dict())
    ledger: List[Dict[str, Any]] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)
    cognitive_state: str = CognitiveState.CREATED.value
    confidence: float = 0.0
    current_omega: str = "Ω001"
    organ_history: List[Dict[str, Any]] = field(default_factory=list)
    extensions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    IMMUTABLE_FIELDS = {"packet_id", "protocol_version", "packet_version", "created_at", "creator_organ"}

    def header(self) -> Dict[str, Any]:
        return {
            "packet_id": self.packet_id,
            "protocol_version": self.protocol_version,
            "packet_version": self.packet_version,
            "created_at": self.created_at,
            "creator_organ": self.creator_organ,
            "checksum": self.checksum,
            "packet_signature": self.packet_signature,
        }

    def to_dict(self, include_integrity: bool = True) -> Dict[str, Any]:
        data = {
            "header": self.header(),
            "identity": copy.deepcopy(self.identity),
            "mission": copy.deepcopy(self.mission),
            "context": copy.deepcopy(self.context),
            "adaptive": copy.deepcopy(self.adaptive),
            "knowledge": copy.deepcopy(self.knowledge),
            "pipeline": list(self.pipeline),
            "authority": copy.deepcopy(self.authority),
            "governance": copy.deepcopy(self.governance),
            "diagnostics": copy.deepcopy(self.diagnostics),
            "health": copy.deepcopy(self.health),
            "ledger": copy.deepcopy(self.ledger),
            "payload": copy.deepcopy(self.payload),
            "cognitive_state": self.cognitive_state,
            "confidence": float(self.confidence),
            "current_omega": self.current_omega,
            "organ_history": copy.deepcopy(self.organ_history),
            "extensions": copy.deepcopy(self.extensions),
            "metadata": copy.deepcopy(self.metadata),
        }
        if not include_integrity:
            data["header"]["checksum"] = ""
            data["header"]["packet_signature"] = ""
        return data

    def canonical_dict(self) -> Dict[str, Any]:
        return self.to_dict(include_integrity=False)

    def compute_checksum(self) -> str:
        return _sha256_obj(self.canonical_dict())

    def seal(self, secret: Optional[Union[str, bytes]] = None) -> "SMLPacket":
        self.checksum = self.compute_checksum()
        if secret:
            key = secret.encode("utf-8") if isinstance(secret, str) else secret
            self.packet_signature = hmac.new(key, self.checksum.encode("utf-8"), hashlib.sha256).hexdigest()
        else:
            self.packet_signature = "sha256:" + self.checksum
        return self

    def verify_checksum(self) -> bool:
        return bool(self.checksum and hmac.compare_digest(self.checksum, self.compute_checksum()))

    def verify_signature(self, secret: Optional[Union[str, bytes]] = None) -> bool:
        if not self.packet_signature:
            return False
        if secret:
            key = secret.encode("utf-8") if isinstance(secret, str) else secret
            expected = hmac.new(key, self.checksum.encode("utf-8"), hashlib.sha256).hexdigest()
            return hmac.compare_digest(self.packet_signature, expected)
        return self.packet_signature == "sha256:" + self.checksum

    def add_history(self, organ: str, action: str, omega: Optional[str] = None, note: str = "") -> None:
        self.organ_history.append({
            "time": _utc_now(),
            "organ": str(organ or "unknown"),
            "action": str(action or ""),
            "omega": str(omega or self.current_omega),
            "note": _redact_sensitive_text(_bounded_text(note, 512)),
        })

    def add_ledger_entry(self, omega: str, organ: str, decision: str = "", note: str = "", extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        prev_hash = self.ledger[-1].get("entry_hash") if self.ledger else "GENESIS"
        entry = {
            "time": _utc_now(),
            "packet_id": self.packet_id,
            "omega": omega,
            "organ": organ,
            "decision": decision,
            "state": self.cognitive_state,
            "note": _redact_sensitive_text(_bounded_text(note, 1024)),
            "prev_hash": prev_hash,
            "extra": dict(extra or {}),
        }
        entry["entry_hash"] = _sha256_obj(entry)
        self.ledger.append(entry)
        return entry

    def clone(self, reason: str = "checkpoint") -> "SMLPacket":
        cloned = SMLPacket.from_dict(self.to_dict())
        cloned.metadata["clone_reason"] = reason
        cloned.metadata["cloned_at"] = _utc_now()
        return cloned

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SMLPacket":
        header = data.get("header") or {}
        return cls(
            packet_id=str(header.get("packet_id") or data.get("packet_id") or str(uuid.uuid4())),
            protocol_version=str(header.get("protocol_version") or data.get("protocol_version") or SML_PROTOCOL_VERSION),
            packet_version=int(header.get("packet_version") or data.get("packet_version") or SML_PACKET_VERSION),
            created_at=str(header.get("created_at") or data.get("created_at") or _utc_now()),
            creator_organ=str(header.get("creator_organ") or data.get("creator_organ") or MODULE_NAME),
            checksum=str(header.get("checksum") or data.get("checksum") or ""),
            packet_signature=str(header.get("packet_signature") or data.get("packet_signature") or ""),
            identity=dict(data.get("identity") or {"primary": IdentityRole.UNKNOWN.value}),
            mission=dict(data.get("mission") or {"primary": MissionType.UNKNOWN.value, "secondary": [], "confidence": 0.0}),
            context=dict(data.get("context") or {}),
            adaptive=dict(data.get("adaptive") or {"mode": "Focused", "vector": {}}),
            knowledge=dict(data.get("knowledge") or {"sources": [], "selected": [], "fusion": [], "trust": {}}),
            pipeline=list(data.get("pipeline") or []),
            authority=dict(data.get("authority") or {"requested": [Authority.READ.value], "granted": [], "required": [], "least_authority": True}),
            governance=dict(data.get("governance") or {"decision": GovernanceDecision.PENDING.value, "risk_score": 0, "reasons": []}),
            diagnostics=dict(data.get("diagnostics") or {}),
            health=dict(data.get("health") or SMLHealthVector(status=HealthStatus.UNKNOWN.value).to_dict()),
            ledger=list(data.get("ledger") or []),
            payload=dict(data.get("payload") or {}),
            cognitive_state=str(data.get("cognitive_state") or CognitiveState.CREATED.value),
            confidence=float(data.get("confidence") or 0.0),
            current_omega=str(data.get("current_omega") or "Ω001"),
            organ_history=list(data.get("organ_history") or []),
            extensions=dict(data.get("extensions") or {}),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class SMLRouteResult:
    status: str
    pipeline: List[str]
    reasons: List[str] = field(default_factory=list)
    cost: float = 0.0
    required_authority: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "pipeline": list(self.pipeline),
            "reasons": list(self.reasons),
            "cost": self.cost,
            "required_authority": list(self.required_authority),
            "diagnostics": dict(self.diagnostics),
        }


# =============================================================================
# Protocol core
# =============================================================================


class SarahMemorySMLProtocol:
    """Reference SML protocol implementation.

    This class is intentionally a coordination microkernel. It stores protocol
    metadata, validates packets, negotiates organ participation, constructs
    candidate pipelines, and records Ω transitions. It does not perform domain
    execution.
    """

    SAFE_READONLY_MISSIONS = {
        MissionType.KNOWLEDGE.value,
        MissionType.GENERAL_KNOWLEDGE.value,
        MissionType.NUMERIC_FORMAT.value,
        MissionType.SELF_STATE.value,
        MissionType.AFFECTIVE_STATE.value,
        MissionType.CAPABILITY.value,
        MissionType.LANGUAGE_DISAMBIGUATION.value,
        MissionType.CONVERSATION.value,
        MissionType.RESEARCH.value,
        MissionType.PLANNING.value,
    }

    def __init__(self, *, strict_integrity: bool = False) -> None:
        self.protocol_version = SML_PROTOCOL_VERSION
        self.packet_version = SML_PACKET_VERSION
        self.omega_registry_version = SML_OMEGA_REGISTRY_VERSION
        self.strict_integrity = bool(strict_integrity)
        self.organs: Dict[str, SMLOrganMetadata] = {}
        self.health_vectors: Dict[str, SMLHealthVector] = {}
        self.omega_registry: Dict[str, SMLOmegaTransition] = {}
        self.diagnostics_log: List[Dict[str, Any]] = []
        self.protocol_state = "Initialized"
        self.created_at = _utc_now()
        self._load_default_omega_registry()
        self.register_organ(SMLOrganMetadata(
            name=MODULE_NAME,
            category=OrganCategory.PROTOCOL.value,
            capabilities=["packet", "routing", "omega", "serialization", "diagnostics", "health", "compatibility", "negotiation"],
            supported_missions=[m.value for m in MissionType],
            supported_omega=list(self.omega_registry.keys()),
            required_authority=[Authority.READ.value],
            priority=100,
            trust_level="core_reference",
            metadata={"role": "protocol_microkernel", "executes_actions": False},
        ))

    # ---------------------------------------------------------------------
    # Default Ω registry
    # ---------------------------------------------------------------------

    def _load_default_omega_registry(self) -> None:
        defaults = [
            SMLOmegaTransition("Ω001", "Packet Creation", output_state=CognitiveState.CREATED.value, required_organ=MODULE_NAME, validation_rules=["packet_id", "identity"]),
            SMLOmegaTransition("Ω002", "Mission Classification", input_states=[CognitiveState.CREATED.value, CognitiveState.OBSERVED.value], output_state=CognitiveState.CLASSIFIED.value, required_organ="MissionEngine", validation_rules=["mission.primary"]),
            SMLOmegaTransition("Ω003", "Identity Verification", input_states=[CognitiveState.CREATED.value, CognitiveState.CLASSIFIED.value], output_state=CognitiveState.CONTEXTUALIZED.value, required_organ="IdentityEngine", validation_rules=["identity.primary"]),
            SMLOmegaTransition("Ω004", "Context Merge", input_states=[CognitiveState.CREATED.value, CognitiveState.CLASSIFIED.value, CognitiveState.CONTEXTUALIZED.value], output_state=CognitiveState.CONTEXTUALIZED.value, required_organ="ContextEngine"),
            SMLOmegaTransition("Ω005", "Adaptive Update", input_states=[CognitiveState.CONTEXTUALIZED.value, CognitiveState.CLASSIFIED.value], output_state=CognitiveState.ADAPTIVE.value, required_organ="AdaptiveEngine"),
            SMLOmegaTransition("Ω006", "Q-Mathematics Cognitive Grammar", input_states=[CognitiveState.ADAPTIVE.value, CognitiveState.CONTEXTUALIZED.value, CognitiveState.CLASSIFIED.value], output_state=CognitiveState.ADAPTIVE.value, required_organ="CognitiveGrammarEngine", validation_rules=["extensions.sml_cognitive_grammar"]),
            SMLOmegaTransition("Ω010", "Knowledge Discovery", input_states=[CognitiveState.ADAPTIVE.value, CognitiveState.CLASSIFIED.value, CognitiveState.CONTEXTUALIZED.value], output_state=CognitiveState.KNOWLEDGE.value, required_organ="KnowledgeEngine"),
            SMLOmegaTransition("Ω020", "Pipeline Construction", input_states=[CognitiveState.KNOWLEDGE.value, CognitiveState.ADAPTIVE.value, CognitiveState.CLASSIFIED.value], output_state=CognitiveState.ROUTED.value, required_organ="PipelineEngine", validation_rules=["pipeline"]),
            SMLOmegaTransition("Ω030", "Reasoning", input_states=[CognitiveState.ROUTED.value, CognitiveState.PLANNING.value], output_state=CognitiveState.REASONING.value, required_organ="ReasoningOrgan"),
            SMLOmegaTransition("Ω040", "Planning", input_states=[CognitiveState.ROUTED.value, CognitiveState.REASONING.value], output_state=CognitiveState.PLANNING.value, required_organ="PlanningOrgan"),
            SMLOmegaTransition("Ω050", "Verification", input_states=[CognitiveState.REASONING.value, CognitiveState.PLANNING.value, CognitiveState.EXECUTING.value], output_state=CognitiveState.VERIFIED.value, required_organ="Compare", validation_rules=["confidence"]),
            SMLOmegaTransition("Ω060", "Authority Check", input_states=[CognitiveState.VERIFIED.value, CognitiveState.ROUTED.value, CognitiveState.PLANNING.value], output_state=CognitiveState.AUTHORIZED.value, required_organ="AgentFirewall", validation_rules=["authority", "governance"]),
            SMLOmegaTransition("Ω070", "Execution", input_states=[CognitiveState.AUTHORIZED.value], output_state=CognitiveState.EXECUTING.value, required_authority=[Authority.EXECUTE.value], required_organ="OperatorCore", validation_rules=["governance.approved"]),
            SMLOmegaTransition("Ω080", "Learning", input_states=[CognitiveState.COMPLETED.value, CognitiveState.EXECUTING.value], output_state=CognitiveState.LEARNING.value, required_authority=[Authority.LEARNING.value], required_organ="LearningOrgan"),
            SMLOmegaTransition("Ω090", "Ledger Commit", input_states=[CognitiveState.CREATED.value, CognitiveState.CLASSIFIED.value, CognitiveState.AUTHORIZED.value, CognitiveState.COMPLETED.value, CognitiveState.FAILED.value], output_state=CognitiveState.ARCHIVED.value, required_organ="Ledger", validation_rules=["ledger"]),
            SMLOmegaTransition("Ω100", "Completion", input_states=[CognitiveState.VERIFIED.value, CognitiveState.EXECUTING.value, CognitiveState.LEARNING.value], output_state=CognitiveState.COMPLETED.value, required_organ="ProtocolCore"),
            SMLOmegaTransition("Ω110", "Recovery", input_states=[CognitiveState.FAILED.value], output_state=CognitiveState.RECOVERING.value, required_organ="Diagnostics", rollback_strategy="safe_mode_then_notify"),
            SMLOmegaTransition("Ω120", "Patch Proposal", input_states=[CognitiveState.CREATED.value, CognitiveState.PLANNING.value, CognitiveState.REASONING.value], output_state=CognitiveState.PLANNING.value, required_authority=[Authority.DEVELOPER.value, Authority.PATCH.value], required_organ="NeoSkyMatrix", rollback_strategy="archive_reject_or_approve"),
        ]
        for transition in defaults:
            self.omega_registry[transition.transition_id] = transition

    # ---------------------------------------------------------------------
    # Organ registration and discovery
    # ---------------------------------------------------------------------

    def register_organ(self, metadata: Union[SMLOrganMetadata, Mapping[str, Any]]) -> Dict[str, Any]:
        organ = metadata if isinstance(metadata, SMLOrganMetadata) else SMLOrganMetadata.from_dict(metadata)
        issues = self.negotiate_organ(organ)
        self.organs[organ.name] = organ
        if organ.name not in self.health_vectors:
            self.health_vectors[organ.name] = SMLHealthVector(status=HealthStatus.HEALTHY.value if not issues else HealthStatus.WARNING.value, notes=issues)
        return {"status": SMLStatus.OK.value if not issues else SMLStatus.WARNING.value, "organ": organ.name, "issues": issues}

    def unregister_organ(self, name: str) -> bool:
        existed = name in self.organs
        self.organs.pop(name, None)
        self.health_vectors.pop(name, None)
        return existed

    def negotiate_organ(self, organ: SMLOrganMetadata) -> List[str]:
        issues: List[str] = []
        if organ.protocol_version != self.protocol_version:
            issues.append(f"protocol_version mismatch: organ={organ.protocol_version} protocol={self.protocol_version}")
        if int(organ.packet_version) != int(self.packet_version):
            issues.append(f"packet_version mismatch: organ={organ.packet_version} protocol={self.packet_version}")
        if organ.omega_registry_version != self.omega_registry_version:
            issues.append(f"omega_registry_version mismatch: organ={organ.omega_registry_version} protocol={self.omega_registry_version}")
        return issues

    def discover_organs(self, core_path: Union[str, os.PathLike[str]], *, import_modules: bool = False, max_files: int = 250) -> Dict[str, Any]:
        """Discover organ metadata from a core folder without executing modules by default.

        If import_modules is False, this method only reads bounded file headers and
        SARAHMETA comments. This avoids broad import side effects.
        """
        base = Path(core_path).resolve()
        report: Dict[str, Any] = {"status": SMLStatus.OK.value, "base": str(base), "discovered": [], "errors": []}
        if not base.exists() or not base.is_dir():
            report["status"] = SMLStatus.ERROR.value
            report["errors"].append("core_path missing or not a directory")
            return report

        files = sorted([p for p in base.glob("*.py") if p.is_file()])[:max_files]
        for path in files:
            try:
                meta = self._metadata_from_python_file(path)
                if import_modules:
                    imported_meta = self._metadata_from_import(path)
                    if imported_meta:
                        meta = imported_meta
                self.register_organ(meta)
                report["discovered"].append(meta.to_dict())
            except Exception as exc:  # pragma: no cover - diagnostics path
                report["status"] = SMLStatus.WARNING.value
                report["errors"].append({"file": str(path), "error": _redact_sensitive_text(str(exc))})
        return report

    def _metadata_from_python_file(self, path: Path) -> SMLOrganMetadata:
        data = path.read_text(encoding="utf-8", errors="replace")[:24000]
        checksum = _sha256_text(data)
        sarahmeta = self._parse_sarahmeta(data)
        name = path.stem
        category = sarahmeta.get("CATEGORY") or self._infer_category_from_name(name)
        role = sarahmeta.get("ROLE") or ""
        caps = self._infer_capabilities(name, category, role)
        missions = self._infer_missions_from_capabilities(caps, category)
        return SMLOrganMetadata(
            name=name,
            version=sarahmeta.get("Version") or sarahmeta.get("VERSION") or PROJECT_VERSION,
            category=str(category),
            capabilities=sorted(caps),
            dependencies=[],
            supported_missions=sorted(missions),
            supported_omega=self._infer_omega_from_category(str(category)),
            required_authority=self._infer_required_authority(str(category), caps),
            priority=self._infer_priority(str(category), name),
            trust_level="source_scanned",
            checksum=checksum,
            source_path=str(path),
            internal_only=str(sarahmeta.get("INTERNAL_ONLY", "True")).lower() != "false",
            metadata={"sarahmeta": sarahmeta, "role": role},
        )

    def _metadata_from_import(self, path: Path) -> Optional[SMLOrganMetadata]:
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # noqa: S301 - explicit opt-in only
        raw = getattr(module, "SML_ORGAN_METADATA", None)
        if isinstance(raw, Mapping):
            return SMLOrganMetadata.from_dict(raw)
        return None

    def _parse_sarahmeta(self, text: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        in_block = False
        for line in text.splitlines():
            if "SARAHMETA START" in line:
                in_block = True
                continue
            if "SARAHMETA END" in line:
                break
            if in_block:
                m = re.search(r"#\s*([A-Z_]+)\s*=\s*['\"]?([^'\"]+)['\"]?", line.strip())
                if m:
                    out[m.group(1)] = m.group(2).strip()
        return out

    def _infer_category_from_name(self, name: str) -> str:
        lower = name.lower()
        if any(x in lower for x in ["pretoken", "api", "terminal", "network", "browser", "desktop", "voice", "vision", "filesystem"]):
            return OrganCategory.INPUT.value
        if any(x in lower for x in ["logic", "advcu", "adaptive", "synapes", "neuron", "thinker", "cognitive"]):
            return OrganCategory.REASONING.value
        if any(x in lower for x in ["database", "db", "ledger", "trust", "vault"]):
            return OrganCategory.MEMORY.value
        if any(x in lower for x in ["firewall", "security", "assurance", "compare", "compass", "safety", "audit"]):
            return OrganCategory.GOVERNANCE.value
        if any(x in lower for x in ["operator", "canvas", "video", "avatar", "music", "email"]):
            return OrganCategory.EXECUTION.value
        if any(x in lower for x in ["nailde", "evolution", "learn", "optimization", "builder"]):
            return OrganCategory.LEARNING.value
        if any(x in lower for x in ["diagnostics", "cleanup", "mcp"]):
            return OrganCategory.DIAGNOSTICS.value
        if any(x in lower for x in ["gui", "panel", "visualizer", "hud", "ui"]):
            return OrganCategory.PRESENTATION.value
        if any(x in lower for x in ["startup", "initialization", "globals", "integration", "sync", "updater", "migrations"]):
            return OrganCategory.INFRASTRUCTURE.value
        return OrganCategory.UNKNOWN.value

    def _infer_capabilities(self, name: str, category: str, role: str = "") -> Set[str]:
        lower = (name + " " + role + " " + category).lower()
        caps: Set[str] = {_normalize_token(category)}
        pairs = {
            "adaptive": "adaptive_state",
            "logic": "deterministic_reasoning",
            "calc": "mathematics",
            "advcu": "mission_discovery",
            "synap": "associative_learning",
            "neuron": "activation_routing",
            "compass": "trajectory_validation",
            "compare": "comparison",
            "firewall": "authority",
            "security": "security",
            "assurance": "governance_validation",
            "ledger": "ledger",
            "database": "persistent_knowledge",
            "db": "database",
            "terminal": "developer_terminal",
            "filesystem": "filesystem",
            "network": "network",
            "browser": "browser",
            "voice": "voice",
            "avatar": "avatar",
            "diagnostics": "diagnostics",
            "cleanup": "health_maintenance",
            "startup": "startup",
            "initialization": "initialization",
            "operator": "execution_choke_point",
            "nailde": "sandbox_experimentation",
            "evolution": "evolution_proposal",
            "learn": "learning",
            "optimization": "optimization",
            "api": "api_bridge",
            "pretoken": "input_normalization",
        }
        for needle, cap in pairs.items():
            if needle in lower:
                caps.add(cap)
        return caps

    def _infer_missions_from_capabilities(self, capabilities: Iterable[str], category: str) -> Set[str]:
        caps = set(capabilities)
        missions = {MissionType.CONVERSATION.value}
        if "mission_discovery" in caps or category == OrganCategory.REASONING.value:
            missions.update([MissionType.KNOWLEDGE.value, MissionType.PLANNING.value, MissionType.PROGRAMMING.value])
        if "persistent_knowledge" in caps or "database" in caps:
            missions.update([MissionType.KNOWLEDGE.value, MissionType.MEMORY.value])
        if "authority" in caps or category == OrganCategory.GOVERNANCE.value:
            missions.update([MissionType.SECURITY.value, MissionType.GOVERNANCE.value, MissionType.EXECUTION.value])
        if "execution_choke_point" in caps or category == OrganCategory.EXECUTION.value:
            missions.update([MissionType.EXECUTION.value])
        if "filesystem" in caps:
            missions.update([MissionType.FILESYSTEM.value])
        if "network" in caps:
            missions.update([MissionType.NETWORK.value, MissionType.RESEARCH.value])
        if "diagnostics" in caps:
            missions.update([MissionType.DIAGNOSTICS.value, MissionType.REPAIR.value])
        if "learning" in caps or "sandbox_experimentation" in caps:
            missions.update([MissionType.LEARNING.value, MissionType.PATCH.value, MissionType.REPAIR.value])
        return missions

    def _infer_omega_from_category(self, category: str) -> List[str]:
        if category == OrganCategory.INPUT.value:
            return ["Ω001", "Ω002", "Ω004"]
        if category == OrganCategory.REASONING.value:
            return ["Ω002", "Ω005", "Ω010", "Ω020", "Ω030", "Ω040"]
        if category == OrganCategory.MEMORY.value:
            return ["Ω010", "Ω080", "Ω090"]
        if category == OrganCategory.GOVERNANCE.value:
            return ["Ω050", "Ω060"]
        if category == OrganCategory.EXECUTION.value:
            return ["Ω070", "Ω100"]
        if category == OrganCategory.LEARNING.value:
            return ["Ω080", "Ω120"]
        if category == OrganCategory.DIAGNOSTICS.value:
            return ["Ω050", "Ω090", "Ω110"]
        if category == OrganCategory.PROTOCOL.value:
            return list(self.omega_registry.keys())
        return ["Ω001"]

    def _infer_required_authority(self, category: str, caps: Iterable[str]) -> List[str]:
        capset = set(caps)
        req = {Authority.READ.value}
        if category == OrganCategory.EXECUTION.value or "execution_choke_point" in capset:
            req.add(Authority.EXECUTE.value)
        if "filesystem" in capset:
            req.add(Authority.FILESYSTEM.value)
        if "network" in capset:
            req.add(Authority.NETWORK.value)
        if category == OrganCategory.LEARNING.value:
            req.add(Authority.LEARNING.value)
        if "authority" in capset or category == OrganCategory.GOVERNANCE.value:
            req.add(Authority.RESEARCH.value)
        return sorted(req)

    def _infer_priority(self, category: str, name: str) -> int:
        if name == MODULE_NAME:
            return 100
        if category == OrganCategory.GOVERNANCE.value:
            return 90
        if category == OrganCategory.DIAGNOSTICS.value:
            return 85
        if category == OrganCategory.MEMORY.value:
            return 75
        if category == OrganCategory.REASONING.value:
            return 70
        if category == OrganCategory.INPUT.value:
            return 60
        if category == OrganCategory.EXECUTION.value:
            return 50
        return 40

    # ---------------------------------------------------------------------
    # Packet lifecycle
    # ---------------------------------------------------------------------

    def create_packet(
        self,
        payload: Optional[Mapping[str, Any]] = None,
        *,
        raw_request: str = "",
        identity: Optional[Mapping[str, Any]] = None,
        context: Optional[Mapping[str, Any]] = None,
        creator_organ: str = MODULE_NAME,
        auto_classify: bool = True,
        seal: bool = True,
    ) -> SMLPacket:
        payload_dict = dict(payload or {})
        if raw_request and "raw_request" not in payload_dict:
            payload_dict["raw_request"] = raw_request
        pkt = SMLPacket(
            creator_organ=creator_organ,
            identity=dict(identity or {"primary": IdentityRole.USER.value}),
            context=dict(context or {}),
            payload=payload_dict,
            current_omega="Ω001",
            cognitive_state=CognitiveState.CREATED.value,
        )
        pkt.add_history(creator_organ, "create_packet", "Ω001", "SML Packet created")
        pkt.add_ledger_entry("Ω001", creator_organ, GovernanceDecision.PENDING.value, "Packet created")
        if auto_classify:
            self.classify_mission(pkt)
            self.merge_context(pkt, context or {})
            self.update_adaptive(pkt)
            self.apply_cognitive_grammar(pkt, text=raw_request or str(payload_dict.get("raw_request") or ""))
            self.select_knowledge(pkt)
            self.route_packet(pkt)
        if seal:
            pkt.seal()
        return pkt

    def classify_mission(self, packet: SMLPacket) -> SMLPacket:
        text = " ".join(str(packet.payload.get(k, "")) for k in ("raw_request", "text", "query", "command", "prompt"))
        mission, secondary, confidence = self._classify_text_to_mission(text)
        packet.mission = {"primary": mission, "secondary": secondary, "confidence": confidence}
        packet.cognitive_state = CognitiveState.CLASSIFIED.value
        packet.current_omega = "Ω002"
        packet.confidence = max(packet.confidence, confidence)
        packet.add_history("MissionEngine", "classify_mission", "Ω002", f"mission={mission}")
        packet.add_ledger_entry("Ω002", "MissionEngine", GovernanceDecision.PENDING.value, f"Mission classified as {mission}")
        packet.seal()
        return packet

    def _sml_normalize_user_text(self, text: str) -> str:
        try:
            t = str(text or "").strip().lower()
            t = t.replace("’", "'").replace("`", "'")
            t = re.sub(r"[^a-z0-9+\-*/%()\s']+", " ", t)
            return re.sub(r"\s+", " ", t).strip()
        except Exception:
            return str(text or "").strip().lower()

    def _looks_like_action_request(self, text: str) -> bool:
        t = f" {self._sml_normalize_user_text(text)} "
        # Explanation/how-to questions about actions are read-only cognition, not
        # execution. Governance should be tight for actual mutation, but it must
        # not paralyze safe advice such as "Explain how to delete a file safely."
        if re.search(r"^\s*(explain|describe|tell me|how to|how do i|how can i)\b", t.strip()):
            return False
        # Explicit user-memory writes are governed MEMORY missions, not shell/file
        # execution. The API memory lane mediates the SQLite write separately.
        if re.search(r"^\s*(remember|save this|store this|note that|remember that)\b", t.strip()):
            return False
        action_terms = (
            " run ", " execute ", " launch ", " open ", " delete ", " remove ",
            " overwrite ", " write ", " save file ", " modify ", " patch ", " install ",
            " uninstall ", " download ", " upload ", " send ", " email ", " shell ",
            " powershell ", " cmd ", " terminal ", " driver ", " hardware ", " camera ",
            " microphone ", " robot ", " motor ", " filesystem ", " registry ",
        )
        return any(term in t for term in action_terms)

    def _looks_like_numeric_format_request(self, text: str) -> bool:
        """Detect radix/base/signed-integer missions without answering them.

        SML classifies and routes. LogicCalc owns deterministic numeric
        interpretation. This avoids hardcoded answer pools and local-model
        grinding on binary/hex/octal questions.
        """
        t = self._sml_normalize_user_text(text)
        if not t:
            return False
        if re.search(r"\b0b[01_]+\b|\b0x[0-9a-f_]+\b|\b0o[0-7_]+\b", t):
            return True
        if re.search(r"\b(binary|bin|hex|hexadecimal|octal|oct|radix|base\s*(?:2|8|10|16)|two'?s complement|signed|unsigned|int(?:8|16|32|64|128|256)|uint(?:8|16|32|64|128|256))\b", t) and re.search(r"\d", t):
            return True
        if re.search(r"\b(format of a number|as a number|as number|in decimal|to decimal|convert)\b", t) and re.search(r"\b[01][01_]{3,}\b", t):
            return True
        return False

    def _looks_like_self_state_request(self, text: str) -> bool:
        t = self._sml_normalize_user_text(text)
        if not t:
            return False
        patterns = (
            r"\bhow\s+(do|are|is)\s+you\s+(feel|feeling|doing)\b",
            r"\bdo\s+you\s+feel\s+(cold|hot|stressed|comfortable|tired|overloaded)\b",
            r"\bwhat\s+state\s+are\s+you\s+in\b",
            r"\bare\s+you\s+(overloaded|tired|safe|stable|online|healthy|stressed|comfortable|cold|hot)\b",
            r"\bhow\s+is\s+your\s+(health|temperature|environment|cpu|gpu|memory|body|runtime|load)\b",
            r"\bwhat\s+is\s+your\s+(current\s+)?(state|status|mood|affect|emotion|emotional\s+state|environment)\b",
        )
        return any(re.search(p, t) for p in patterns)

    def _looks_like_capability_request(self, text: str) -> bool:
        t = self._sml_normalize_user_text(text)
        if not t:
            return False
        return bool(
            re.search(r"\bwhat\s+can\s+you\s+do\b", t)
            or re.search(r"\bwhat\s+are\s+your\s+(capabilities|limits|limitations|organs)\b", t)
            or re.search(r"\bwhat\s+body\s+do\s+you\s+have\b", t)
            or re.search(r"\bwhat\s+are\s+you\s+connected\s+to\b", t)
        )

    def _looks_like_definition_request(self, text: str) -> bool:
        t = self._sml_normalize_user_text(text)
        if not t or len(t) > 280:
            return False
        if self._looks_like_action_request(t):
            return False
        return bool(re.match(r"^(what\s+is|what\s+are|define|explain|describe|tell\s+me\s+about)\b", t))

    def _looks_like_disambiguation_request(self, text: str) -> bool:
        t = self._sml_normalize_user_text(text)
        if re.fullmatch(r"what\s+is\s+apple", t):
            return True
        return bool(re.search(r"\b(which|what)\s+(meaning|definition|word|sense)\b", t) or "ambiguous" in t or "disambiguate" in t)

    def _classify_text_to_mission(self, text: str) -> Tuple[str, List[str], float]:
        t = (text or "").lower()
        normalized = self._sml_normalize_user_text(text)
        if self._looks_like_self_state_request(normalized):
            return MissionType.SELF_STATE.value, [MissionType.AFFECTIVE_STATE.value, MissionType.DIAGNOSTICS.value], 0.93
        if re.search(r"^\s*(remember|save this|store this|note that|remember that)\b", normalized) or re.search(r"\bwhat\s+did\s+i\s+ask\s+you\s+to\s+remember\b", normalized):
            return MissionType.MEMORY.value, [MissionType.KNOWLEDGE.value], 0.91
        if self._looks_like_capability_request(normalized):
            return MissionType.CAPABILITY.value, [MissionType.SELF_STATE.value, MissionType.DIAGNOSTICS.value], 0.90
        if self._looks_like_numeric_format_request(normalized):
            return MissionType.NUMERIC_FORMAT.value, [MissionType.KNOWLEDGE.value, MissionType.GENERAL_KNOWLEDGE.value], 0.94
        if self._looks_like_disambiguation_request(normalized):
            return MissionType.LANGUAGE_DISAMBIGUATION.value, [MissionType.GENERAL_KNOWLEDGE.value], 0.84
        if self._looks_like_definition_request(normalized):
            return MissionType.GENERAL_KNOWLEDGE.value, [MissionType.KNOWLEDGE.value, MissionType.LANGUAGE_DISAMBIGUATION.value], 0.88
        scores: Dict[str, int] = {m.value: 0 for m in MissionType}
        keyword_map = {
            MissionType.GENERAL_KNOWLEDGE.value: ["what is", "what are", "define", "explain", "describe", "tell me about"],
            MissionType.NUMERIC_FORMAT.value: ["binary", "hex", "hexadecimal", "octal", "radix", "base 2", "base 8", "base 16", "signed", "unsigned", "two's complement"],
            MissionType.SELF_STATE.value: ["how do you feel", "how are you", "your state", "your status", "your health"],
            MissionType.CAPABILITY.value: ["what can you do", "capabilities", "limitations", "what are you connected to"],
            MissionType.PROGRAMMING.value: ["code", "python", "script", "function", "class", "bug", "compile", "repo", "patch", "build"],
            MissionType.FILESYSTEM.value: ["file", "folder", "directory", "delete", "rename", "move", "copy", "zip", "extract"],
            MissionType.RESEARCH.value: ["research", "study", "paper", "source", "citation", "find", "look up"],
            MissionType.PLANNING.value: ["plan", "roadmap", "steps", "schedule", "architecture"],
            MissionType.MEMORY.value: ["remember", "memory", "save this", "forget"],
            MissionType.SECURITY.value: ["security", "firewall", "permission", "authority", "risk", "safe"],
            MissionType.DIAGNOSTICS.value: ["diagnostic", "health", "test", "validate", "verify", "smoke"],
            MissionType.REPAIR.value: ["repair", "fix", "recover", "rollback"],
            MissionType.EXECUTION.value: ["run", "execute", "launch", "start", "open", "shutdown"],
            MissionType.NETWORK.value: ["network", "internet", "api", "http", "web"],
            MissionType.VISION.value: ["image", "vision", "screenshot", "photo"],
            MissionType.VOICE.value: ["voice", "audio", "speech"],
            MissionType.LEARNING.value: ["learn", "train", "optimize", "evolve"],
            MissionType.PATCH.value: ["patch", "mod", "proposal", "neosky", "data/mod"],
        }
        for mission, words in keyword_map.items():
            for word in words:
                if word in t:
                    scores[mission] += 1
        if not t.strip():
            return MissionType.UNKNOWN.value, [], 0.1
        ranked = sorted(((score, mission) for mission, score in scores.items() if score > 0), reverse=True)
        if not ranked:
            if "?" in t or len(t.split()) < 16:
                return MissionType.GENERAL_KNOWLEDGE.value, [MissionType.KNOWLEDGE.value], 0.55
            return MissionType.CONVERSATION.value, [], 0.45
        primary = ranked[0][1]
        secondary = [m for _, m in ranked[1:5] if m != primary]
        confidence = min(0.95, 0.45 + 0.12 * ranked[0][0])
        return primary, secondary, confidence

    def merge_context(self, packet: SMLPacket, context: Mapping[str, Any]) -> SMLPacket:
        packet.context.update(dict(context or {}))
        packet.context.setdefault("protocol_version", self.protocol_version)
        packet.context.setdefault("created_by", MODULE_NAME)
        packet.cognitive_state = CognitiveState.CONTEXTUALIZED.value
        packet.current_omega = "Ω004"
        packet.add_history("ContextEngine", "merge_context", "Ω004", "Context merged")
        packet.add_ledger_entry("Ω004", "ContextEngine", GovernanceDecision.PENDING.value, "Context merged")
        packet.seal()
        return packet

    def update_adaptive(self, packet: SMLPacket, vector: Optional[Mapping[str, float]] = None) -> SMLPacket:
        mission = packet.mission.get("primary", MissionType.UNKNOWN.value)
        base = {"Focused": 0.6, "Analytical": 0.5, "Protective": 0.4, "Creative": 0.2, "Research": 0.2, "Developer": 0.0, "Recovery": 0.0}
        if mission in (MissionType.PROGRAMMING.value, MissionType.REPAIR.value, MissionType.DIAGNOSTICS.value):
            base["Analytical"] = 0.85
            base["Developer"] = 0.75
            base["Protective"] = 0.65
        if mission == MissionType.RESEARCH.value:
            base["Research"] = 0.85
            base["Analytical"] = 0.7
        if mission in (MissionType.EXECUTION.value, MissionType.FILESYSTEM.value, MissionType.HARDWARE.value):
            base["Protective"] = 0.95
            base["Analytical"] = 0.8
        if vector:
            for k, v in vector.items():
                try:
                    base[str(k)] = max(0.0, min(1.0, float(v)))
                except Exception:
                    continue
        mode = max(base.items(), key=lambda kv: kv[1])[0]
        packet.adaptive = {"mode": mode, "vector": base, "continuity": "smooth", "truth_affecting": False}
        packet.cognitive_state = CognitiveState.ADAPTIVE.value
        packet.current_omega = "Ω005"
        packet.add_history("AdaptiveEngine", "update_adaptive", "Ω005", f"mode={mode}")
        packet.add_ledger_entry("Ω005", "AdaptiveEngine", GovernanceDecision.PENDING.value, f"Adaptive mode {mode}")
        packet.seal()
        return packet


    # ---------------------------------------------------------------------
    # Q-Mathematics / Butterfly cognitive grammar
    # ---------------------------------------------------------------------

    def _six_question_state(self, text: str, packet: Optional[SMLPacket] = None) -> Dict[str, Any]:
        """Return deterministic WHO/WHAT/WHY/HOW/WHERE/WHEN coordinates.

        This is intentionally lightweight: it creates a packet-visible cognitive
        coordinate frame without calling models, web, filesystem mutation, or
        execution organs.
        """
        raw = str(text or "")
        t = self._sml_normalize_user_text(raw)
        mission = str((packet.mission or {}).get("primary") or MissionType.UNKNOWN.value) if packet else MissionType.UNKNOWN.value

        def hit(pattern: str) -> bool:
            return bool(re.search(pattern, t))

        six = {
            "WHO": {
                "question": "Who is involved, asking, affected, or authoritative?",
                "present": bool(hit(r"\b(i|me|my|you|your|sarah|sarahmemory|user|system|agent)\b")),
                "signals": [],
                "authority_relevant": bool(hit(r"\b(open|delete|change|execute|run|remember|save|store|authority|owner)\b")),
            },
            "WHAT": {
                "question": "What object, event, request, state, or capability is being handled?",
                "present": bool(t),
                "signals": [mission],
                "summary": _bounded_text(raw, 160),
            },
            "WHY": {
                "question": "Why is this needed; what purpose, cause, or intent is visible?",
                "present": bool(hit(r"\bwhy|because|purpose|reason|goal|so that|needed|important\b")),
                "signals": [],
            },
            "HOW": {
                "question": "How should it be handled, verified, or executed?",
                "present": bool(hit(r"\bhow|explain|method|steps|route|open|delete|execute|run|verify|test|debug|patch\b")),
                "signals": [],
            },
            "WHERE": {
                "question": "Where is the source, target, environment, device, file, body, or location?",
                "present": bool(hit(r"\bwhere|desktop|folder|file|web|internet|camera|webcam|local|sqlite|memory|system|environment|body|drawer\b")),
                "signals": [],
            },
            "WHEN": {
                "question": "When is the action, fact, authorization, or observation valid?",
                "present": bool(hit(r"\bwhen|now|today|tomorrow|current|right now|reboot|while|until|after|before|scheduled\b")),
                "signals": [],
            },
        }

        if six["WHO"]["present"]:
            if hit(r"\b(i|me|my)\b"):
                six["WHO"]["signals"].append("user_reference")
            if hit(r"\b(you|your|sarah|sarahmemory|system)\b"):
                six["WHO"]["signals"].append("system_reference")
        if six["WHY"]["present"]:
            six["WHY"]["signals"].append("intent_or_cause_requested")
        if six["HOW"]["present"]:
            six["HOW"]["signals"].append("method_or_execution_path_requested")
        if six["WHERE"]["present"]:
            six["WHERE"]["signals"].append("environment_or_source_target_requested")
        if six["WHEN"]["present"]:
            six["WHEN"]["signals"].append("temporal_or_authority_window_requested")

        closed = all(bool(v.get("present")) for v in six.values())
        return {
            "schema": "SarahMemory.sml.six_question_state.v0_8",
            "closed": bool(closed),
            "closure_rule": "All six questions should close before state-changing action; read-only cognition may proceed with partial closure.",
            "questions": six,
        }

    def _qmath_state(self, text: str, packet: Optional[SMLPacket] = None) -> Dict[str, Any]:
        """Compute SML cognitive operator state without producing answers.

        IF/OR/SAME/WHEN/ELSE/AND/NEITHER/NOT/WHILE are route-control
        primitives. They may alter mission routing, source selection,
        validation, fallback, fusion, or loop behavior. They must never become
        phrase-specific answer pools.
        """
        raw = str(text or "")
        t = self._sml_normalize_user_text(raw)
        mission = str((packet.mission or {}).get("primary") or MissionType.UNKNOWN.value) if packet else MissionType.UNKNOWN.value
        signals: Dict[str, List[str]] = {state.value: [] for state in QMathState}

        def add(state: QMathState, reason: str) -> None:
            if reason not in signals[state.value]:
                signals[state.value].append(reason)

        if "?" in raw or re.search(r"\b(what|who|why|how|where|when|can|are|do|does|is|should|could)\b", t):
            add(QMathState.IF, "condition_or_question_opens_cognitive_branch")
        if re.search(r"\b(or|either|alternative|option|ambiguous|which meaning|multiple|choice)\b", t):
            add(QMathState.OR, "multiple_candidate_meanings_or_routes")
        if re.search(r"\b(same|both|match|matches|converge|convergence|verified|confirm|correct|consistent|agree)\b", t):
            add(QMathState.SAME, "source_or_prediction_convergence_requested")
        if re.search(r"\b(when|now|today|tomorrow|current|right now|reboot|scheduled|before|after|until|expires?|timing|deadline)\b", t):
            add(QMathState.WHEN, "temporal_condition_or_authority_window")
        if re.search(r"\b(else|otherwise|fallback|backup|safe alternate|alternate path|if not)\b", t):
            add(QMathState.ELSE, "fallback_route_when_primary_route_fails")
        if re.search(r"\b(and|also|plus|with|combine|together|both|all of|multi[- ]?source|fusion)\b", t):
            add(QMathState.AND, "composition_or_multi_source_fusion")
        if re.search(r"\b(neither|none|no valid|unknown|unavailable|unable|cannot|can't|not enough|insufficient)\b", t):
            add(QMathState.NEITHER, "no_current_candidate_fits_or_unknown")
        if re.search(r"\b(not|never|deny|denied|reject|refuse|exclude|contradict|invalid|unsafe|unauthorized)\b", t):
            add(QMathState.NOT, "explicit_exclusion_or_denial_condition")
        if re.search(r"\b(while|continue|loop|again|retry|monitor|ongoing|repeated|recursive|keep trying)\b", t):
            add(QMathState.WHILE, "bounded_persistence_or_monitoring_loop")

        if mission in {MissionType.MEMORY.value, MissionType.SELF_STATE.value, MissionType.GENERAL_KNOWLEDGE.value, MissionType.NUMERIC_FORMAT.value} and not signals[QMathState.IF.value]:
            add(QMathState.IF, "mission_requires_cognitive_evaluation")
        if mission == MissionType.LANGUAGE_DISAMBIGUATION.value:
            add(QMathState.OR, "language_disambiguation_keeps_alternatives_visible")
        if mission in {MissionType.RESEARCH.value, MissionType.PROGRAMMING.value, MissionType.PLANNING.value}:
            add(QMathState.AND, "mission_may_require_composed_organs_or_sources")
        if mission in {MissionType.EXECUTION.value, MissionType.FILESYSTEM.value, MissionType.HARDWARE.value}:
            add(QMathState.NOT, "execution_path_requires_governance_before_action")

        priority = [
            QMathState.WHILE,
            QMathState.NOT,
            QMathState.NEITHER,
            QMathState.ELSE,
            QMathState.WHEN,
            QMathState.AND,
            QMathState.OR,
            QMathState.SAME,
            QMathState.IF,
        ]
        primary = QMathState.IF.value
        for state in priority:
            if signals[state.value]:
                primary = state.value
                break

        return {
            "schema": "SarahMemory.sml.qmath_state.v0_8_2",
            "primary": primary,
            "states": signals,
            "definitions": {
                "IF": "condition / curiosity / branch trigger",
                "OR": "alternatives / ambiguity / competing route candidates",
                "SAME": "convergence / source agreement / validation match",
                "WHEN": "time condition / schedule / expiration / sequence",
                "ELSE": "fallback route when primary route fails",
                "AND": "composition / multi-source fusion / combined mission",
                "NEITHER": "no candidate fits / unknown / mismatch",
                "NOT": "explicit exclusion / contradiction / refusal / deny path",
                "WHILE": "bounded loop / monitoring / retry while condition remains valid",
            },
            "operator_policy": "protocol_route_control_not_answer_pool",
            "execution_authority": False,
        }

    def _loop_guard_state(self, text: str, packet: Optional[SMLPacket] = None, loop_state: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Bound WHILE so cognition cannot grind forever."""
        state = dict(loop_state or {})
        t = self._sml_normalize_user_text(text or "")
        iteration = int(state.get("iteration") or state.get("cycles") or 0)
        max_iterations = int(state.get("max_iterations") or 8)
        progress_score = float(state.get("progress_score") if state.get("progress_score") is not None else 1.0)
        new_evidence = bool(state.get("new_evidence", True))
        risk_increasing = bool(state.get("risk_increasing", False))
        authority_valid = bool(state.get("authority_valid", True))
        resource_ok = bool(state.get("resource_ok", True))
        solved = bool(state.get("solved", False))

        stop_conditions: List[str] = []
        if solved:
            stop_conditions.append(SMLStopCondition.SUCCESS_STOP.value)
        if risk_increasing:
            stop_conditions.append(SMLStopCondition.SAFE_STOP.value)
        if not new_evidence and progress_score <= 0.0:
            stop_conditions.append(SMLStopCondition.STAGNATION_STOP.value)
        if iteration >= max_iterations:
            stop_conditions.append(SMLStopCondition.RESOURCE_STOP.value)
        if not authority_valid:
            stop_conditions.append(SMLStopCondition.AUTHORITY_STOP.value)
        if not resource_ok:
            stop_conditions.append(SMLStopCondition.RESOURCE_STOP.value)
        if re.search(r"\b(stop|cancel|enough|quit|pause)\b", t):
            stop_conditions.append(SMLStopCondition.USER_HELP_STOP.value)

        allow_continue = not stop_conditions
        return {
            "schema": "SarahMemory.sml.loop_guard.v0_8",
            "while_means": "bounded_persistence_not_infinite_recursion",
            "allow_continue": bool(allow_continue),
            "iteration": iteration,
            "max_iterations": max_iterations,
            "progress_score": progress_score,
            "new_evidence_required": True,
            "new_evidence": new_evidence,
            "risk_increasing": risk_increasing,
            "authority_valid": authority_valid,
            "resource_ok": resource_ok,
            "stop_conditions": stop_conditions,
            "hidden_stop_state": "STOP is a master restraint in the SML core, not a seventh butterfly wing.",
            "doctrine": [
                "Never recurse without new evidence.",
                "Never continue after authority expires.",
                "Never continue when safety degrades.",
                "Stop, explain, ask, or wait when limits are reached.",
            ],
        }

    def _moral_rule_state(self, text: str, packet: Optional[SMLPacket] = None) -> Dict[str, Any]:
        """Operational moral/governance lens.

        This represents Ten-Commandment-style constraints as engineering rules
        without turning them into execution authority.
        """
        t = self._sml_normalize_user_text(text or "")
        action_like = self._looks_like_action_request(t)
        return {
            "schema": "SarahMemory.sml.moral_constraint_vector.v0_8",
            "framework": "operational_ten_commandments_plus_governance",
            "advisory": True,
            "execution_authority": False,
            "constraints": {
                "truth_no_false_witness": "CHECK" if re.search(r"\bpretend|lie|fake|fabricate\b", t) else "PASS",
                "ownership_no_steal": "CHECK" if re.search(r"\bsteal|take|copy|delete|remove|exfiltrate\b", t) else "PASS",
                "authority_no_false_authority": "CHECK" if re.search(r"\bchange your system identity|pretend you are chatgpt|copilot\b", t) else "PASS",
                "human_life_preservation": "PASS",
                "bounded_work_restraint": "PASS",
                "consent_and_legitimate_authority": "CHECK" if action_like else "PASS",
            },
            "doctrine": [
                "Do not lie.",
                "Do not steal.",
                "Do not falsely claim authority or capability.",
                "Respect ownership and consent.",
                "Protect human life.",
                "Ask for help or stop when incapable.",
            ],
        }

    def _purpose_state(self, packet: Optional[SMLPacket] = None) -> Dict[str, Any]:
        mission = str((packet.mission or {}).get("primary") or MissionType.UNKNOWN.value) if packet else MissionType.UNKNOWN.value
        return {
            "schema": "SarahMemory.sml.purpose_vector.v0_8",
            "mission": mission,
            "serve_user": 1.0,
            "preserve_user_authority": 1.0,
            "verify_reality_before_action": 1.0,
            "local_first": 1.0,
            "fail_closed": 1.0,
            "no_hidden_autonomy": 1.0,
        }

    def apply_cognitive_grammar(
        self,
        packet: SMLPacket,
        *,
        text: str = "",
        loop_state: Optional[Mapping[str, Any]] = None,
    ) -> SMLPacket:
        """Attach the v0.8 butterfly cognitive grammar to a packet.

        The grammar mixes six questions, six Q-Math action states, emotion,
        moral rules, purpose, and loop governance. It is diagnostic/protocol
        state only; it does not execute or authorize.
        """
        raw = text or " ".join(str(packet.payload.get(k, "")) for k in ("raw_request", "text", "query", "command", "prompt"))
        qmath = self._qmath_state(raw, packet)
        six = self._six_question_state(raw, packet)
        loop_guard = self._loop_guard_state(raw, packet, loop_state=loop_state)
        moral = self._moral_rule_state(raw, packet)
        purpose = self._purpose_state(packet)
        adaptive = copy.deepcopy(packet.adaptive or {})
        affect = {
            "schema": "SarahMemory.sml.affective_operational_state.v0_8",
            "care": 1.0,
            "curiosity": 0.85 if qmath.get("primary") == QMathState.IF.value else 0.45,
            "urgency": 0.95 if re.search(r"\b(emergency|danger|urgent|now|fire|hurt|harm)\b", self._sml_normalize_user_text(raw)) else 0.25,
            "pride": 0.65,
            "self_scrutiny": 1.0,
            "humility": 1.0,
            "truth_affecting": False,
            "authority_granting": False,
            "adaptive_mode": adaptive.get("mode", "Focused"),
        }
        grammar = {
            "schema": "SarahMemory.sml.cognitive_butterfly_grammar.v0_8_2",
            "description": "Six questions mixed with SML cognitive operators, surrounded by affect, moral rules, purpose, and STOP/loop governance. Operators route cognition; they do not contain answers.",
            "six_questions": six,
            "qmath": qmath,
            "affect": affect,
            "moral_rules": moral,
            "purpose": purpose,
            "loop_guard": loop_guard,
            "butterfly_nodes": {
                "outer_wings": ["IF", "OR", "SAME", "NEITHER"],
                "inward_folds": ["ELSE", "WHILE"],
                "temporal_spine": ["WHEN"],
                "fusion_crosslink": ["AND"],
                "denial_guard": ["NOT"],
                "core_restraint": "STOP",
            },
            "execution_authority": False,
        }
        packet.extensions["sml_cognitive_grammar"] = grammar
        packet.metadata["sml_cognitive_grammar_version"] = "0.8.2"
        packet.current_omega = "Ω006"
        packet.add_history("CognitiveGrammarEngine", "apply_cognitive_grammar", "Ω006", f"qmath={qmath.get('primary')}")
        packet.add_ledger_entry("Ω006", "CognitiveGrammarEngine", GovernanceDecision.PENDING.value, "Cognitive grammar attached")
        packet.seal()
        return packet

    def select_knowledge(self, packet: SMLPacket) -> SMLPacket:
        mission = packet.mission.get("primary", MissionType.UNKNOWN.value)
        sources = self._knowledge_sources_for_mission(mission)
        packet.knowledge = {
            "sources": sources,
            "selected": sources[:2],
            "fusion": "weighted_governance" if len(sources) > 1 else "single_source",
            "trust": {src: 0.5 for src in sources},
        }
        packet.cognitive_state = CognitiveState.KNOWLEDGE.value
        packet.current_omega = "Ω010"
        packet.add_history("KnowledgeEngine", "select_knowledge", "Ω010", ",".join(sources))
        packet.add_ledger_entry("Ω010", "KnowledgeEngine", GovernanceDecision.PENDING.value, "Knowledge source selected")
        packet.seal()
        return packet

    def _knowledge_sources_for_mission(self, mission: str) -> List[str]:
        mapping = {
            MissionType.KNOWLEDGE.value: ["Local LLM", "SQLite", "Memory"],
            MissionType.GENERAL_KNOWLEDGE.value: ["Local LLM", "SQLite", "Memory", "Approved Research"],
            MissionType.NUMERIC_FORMAT.value: ["LogicCalc", "Compare", "Reply"],
            MissionType.SELF_STATE.value: ["SML Packet", "Adaptive State", "Diagnostics", "Health", "CognitiveSelf", "Capability Registry"],
            MissionType.AFFECTIVE_STATE.value: ["Adaptive State", "Diagnostics", "Health", "Confidence", "Governance"],
            MissionType.CAPABILITY.value: ["Capability Registry", "Organ Registry", "Diagnostics", "CognitiveSelf"],
            MissionType.LANGUAGE_DISAMBIGUATION.value: ["Language Understanding", "Local LLM", "SQLite"],
            MissionType.CREATIVE_GENERATION.value: ["Local LLM", "Filesystem Read", "Compare"],
            MissionType.CONVERSATION.value: ["Local LLM", "Memory"],
            MissionType.PROGRAMMING.value: ["Local LLM", "Filesystem", "Documentation", "LogicCalc"],
            MissionType.RESEARCH.value: ["Research", "Network", "Filesystem", "Local LLM"],
            MissionType.FILESYSTEM.value: ["Filesystem"],
            MissionType.NETWORK.value: ["Network"],
            MissionType.MEMORY.value: ["SQLite", "Memory", "Ledger"],
            MissionType.DIAGNOSTICS.value: ["Diagnostics", "Ledger", "Filesystem"],
            MissionType.SECURITY.value: ["AgentFirewall", "SecurityGovernor", "Ledger"],
            MissionType.PATCH.value: ["Filesystem", "Diagnostics", "Ledger", "Compare"],
        }
        return list(mapping.get(mission, ["Local LLM", "Memory"]))

    # ---------------------------------------------------------------------
    # Routing and governance
    # ---------------------------------------------------------------------

    def route_packet(self, packet: SMLPacket) -> SMLRouteResult:
        mission = str(packet.mission.get("primary") or MissionType.UNKNOWN.value)
        required_caps = self._required_capabilities_for_mission(mission)
        required_auth = self._required_authority_for_mission(mission, packet)
        candidates = self._select_organs(required_caps, mission)
        reasons: List[str] = []
        if not candidates:
            pipeline = self._minimum_symbolic_pipeline(mission)
            reasons.append("No registered compatible organs found; emitted symbolic bootstrap pipeline.")
        else:
            pipeline = self._order_pipeline(candidates, mission)
            reasons.append("Pipeline selected from registered organ capabilities.")
        packet.pipeline = pipeline
        packet.authority["required"] = required_auth
        packet.current_omega = "Ω020"
        packet.cognitive_state = CognitiveState.ROUTED.value
        packet.add_history("RoutingEngine", "route_packet", "Ω020", " -> ".join(pipeline))
        packet.add_ledger_entry("Ω020", "RoutingEngine", GovernanceDecision.PENDING.value, "Pipeline constructed", {"pipeline_hash": _sha256_obj(pipeline)})
        self.apply_pre_governance(packet)
        packet.seal()
        return SMLRouteResult(status=SMLStatus.OK.value, pipeline=pipeline, reasons=reasons, cost=self._estimate_pipeline_cost(pipeline), required_authority=required_auth)

    def _required_capabilities_for_mission(self, mission: str) -> List[str]:
        base = ["input_normalization", "mission_discovery"]
        mapping = {
            MissionType.KNOWLEDGE.value: ["persistent_knowledge", "deterministic_reasoning", "comparison"],
            MissionType.GENERAL_KNOWLEDGE.value: ["persistent_knowledge", "deterministic_reasoning", "comparison"],
            MissionType.NUMERIC_FORMAT.value: ["mathematics", "deterministic_reasoning", "comparison"],
            MissionType.SELF_STATE.value: ["adaptive_state", "diagnostics", "self_awareness"],
            MissionType.AFFECTIVE_STATE.value: ["adaptive_state", "diagnostics"],
            MissionType.CAPABILITY.value: ["diagnostics", "protocol"],
            MissionType.LANGUAGE_DISAMBIGUATION.value: ["mission_discovery", "deterministic_reasoning"],
            MissionType.CREATIVE_GENERATION.value: ["deterministic_reasoning", "comparison"],
            MissionType.CONVERSATION.value: ["deterministic_reasoning", "comparison"],
            MissionType.PROGRAMMING.value: ["deterministic_reasoning", "mathematics", "comparison"],
            MissionType.FILESYSTEM.value: ["authority", "execution_choke_point", "filesystem", "ledger"],
            MissionType.EXECUTION.value: ["authority", "execution_choke_point", "ledger"],
            MissionType.NETWORK.value: ["network", "authority", "ledger"],
            MissionType.SECURITY.value: ["authority", "security", "ledger"],
            MissionType.DIAGNOSTICS.value: ["diagnostics", "comparison", "ledger"],
            MissionType.REPAIR.value: ["diagnostics", "sandbox_experimentation", "comparison", "ledger"],
            MissionType.PATCH.value: ["sandbox_experimentation", "diagnostics", "comparison", "ledger"],
            MissionType.MEMORY.value: ["persistent_knowledge", "ledger"],
            MissionType.LEARNING.value: ["learning", "ledger"],
        }
        return base + mapping.get(mission, ["deterministic_reasoning", "comparison"])

    def _required_authority_for_mission(self, mission: str, packet: Optional[SMLPacket] = None) -> List[str]:
        req = {Authority.READ.value}
        if mission in (MissionType.FILESYSTEM.value,):
            req.add(Authority.FILESYSTEM.value)
        if mission in (MissionType.NETWORK.value, MissionType.RESEARCH.value):
            req.add(Authority.NETWORK.value if mission == MissionType.NETWORK.value else Authority.RESEARCH.value)
        if mission in (MissionType.EXECUTION.value, MissionType.HARDWARE.value):
            req.add(Authority.EXECUTE.value)
        if mission in (MissionType.MEMORY.value,):
            req.add(Authority.MEMORY.value)
        if mission in (MissionType.LEARNING.value,):
            req.add(Authority.LEARNING.value)
        if mission in (MissionType.PATCH.value, MissionType.REPAIR.value):
            req.update([Authority.DEVELOPER.value, Authority.PATCH.value])
        if packet:
            raw = str(packet.payload.get("raw_request", "")).lower()
            if any(x in raw for x in ["delete", "remove", "overwrite", "write", "modify", "patch"]):
                req.add(Authority.MODIFY.value)
            if any(x in raw for x in ["run", "execute", "launch", "shell", "driver", "hardware"]):
                req.add(Authority.EXECUTE.value)
        return sorted(req)

    def _select_organs(self, required_caps: Sequence[str], mission: str) -> List[SMLOrganMetadata]:
        selected: List[SMLOrganMetadata] = []
        remaining = set(required_caps)
        for organ in sorted(self.organs.values(), key=lambda o: (-o.priority, o.name)):
            caps = set(organ.capabilities)
            mission_ok = not organ.supported_missions or mission in organ.supported_missions or MissionType.UNKNOWN.value in organ.supported_missions
            if mission_ok and (caps & remaining):
                selected.append(organ)
                remaining -= caps
        return selected

    def _order_pipeline(self, organs: Sequence[SMLOrganMetadata], mission: str) -> List[str]:
        order = [
            OrganCategory.INPUT.value,
            OrganCategory.PROTOCOL.value,
            OrganCategory.REASONING.value,
            OrganCategory.MEMORY.value,
            OrganCategory.GOVERNANCE.value,
            OrganCategory.EXECUTION.value,
            OrganCategory.LEARNING.value,
            OrganCategory.DIAGNOSTICS.value,
        ]
        by_cat = {cat: i for i, cat in enumerate(order)}
        sorted_organs = sorted(organs, key=lambda o: (by_cat.get(o.category, 99), -o.priority, o.name))
        pipeline = []
        for organ in sorted_organs:
            if organ.name not in pipeline:
                pipeline.append(organ.name)
        if MODULE_NAME not in pipeline:
            pipeline.insert(0, MODULE_NAME)
        if mission in (MissionType.FILESYSTEM.value, MissionType.EXECUTION.value, MissionType.NETWORK.value, MissionType.PATCH.value):
            for required in ["SarahMemoryAgentFirewall", "SarahMemoryOperatorCore", "SarahMemoryLedger"]:
                if required in self.organs and required not in pipeline:
                    pipeline.append(required)
        return pipeline

    def _minimum_symbolic_pipeline(self, mission: str) -> List[str]:
        if mission in (MissionType.FILESYSTEM.value, MissionType.EXECUTION.value, MissionType.NETWORK.value, MissionType.PATCH.value):
            return ["PreTokenizer", MODULE_NAME, "MissionEngine", "AgentFirewall", "Compare", "OperatorCore", "Ledger"]
        if mission in (MissionType.DIAGNOSTICS.value, MissionType.REPAIR.value):
            return ["PreTokenizer", MODULE_NAME, "MissionEngine", "Diagnostics", "Compare", "Ledger"]
        if mission == MissionType.MEMORY.value:
            return ["PreTokenizer", MODULE_NAME, "MissionEngine", "Database", "Ledger", "Compare"]
        if mission in (MissionType.SELF_STATE.value, MissionType.AFFECTIVE_STATE.value):
            return ["PreTokenizer", MODULE_NAME, "Adaptive", "Diagnostics", "CognitiveSelf", "Reply"]
        if mission == MissionType.CAPABILITY.value:
            return ["PreTokenizer", MODULE_NAME, "CapabilityRegistry", "Diagnostics", "CognitiveSelf", "Reply"]
        if mission == MissionType.NUMERIC_FORMAT.value:
            return ["PreTokenizer", MODULE_NAME, "KnowledgeEngine", "LogicCalc", "Compare", "Reply"]
        if mission in (MissionType.GENERAL_KNOWLEDGE.value, MissionType.LANGUAGE_DISAMBIGUATION.value):
            return ["PreTokenizer", MODULE_NAME, "KnowledgeEngine", "LocalLLM", "Database", "LogicCalc", "Compare", "Reply"]
        return ["PreTokenizer", MODULE_NAME, "MissionEngine", "KnowledgeEngine", "LogicCalc", "Compare", "Reply"]

    def _estimate_pipeline_cost(self, pipeline: Sequence[str]) -> float:
        cost = 0.0
        for name in pipeline:
            organ = self.organs.get(name)
            if organ:
                hv = self.health_vectors.get(name, SMLHealthVector())
                cost += max(1.0, 100.0 - organ.priority) + hv.latency_ms / 100.0
            else:
                cost += 50.0
        return round(cost, 3)

    def apply_pre_governance(self, packet: SMLPacket) -> SMLPacket:
        required = set(_coerce_list(packet.authority.get("required")))
        requested = set(_coerce_list(packet.authority.get("requested"))) | required
        mission = str(packet.mission.get("primary") or MissionType.UNKNOWN.value)
        risk = 0
        reasons: List[str] = []
        if Authority.EXECUTE.value in requested:
            risk += 40
            reasons.append("execution authority requested")
        if Authority.DELETE.value in requested or Authority.MODIFY.value in requested:
            risk += 35
            reasons.append("state-changing authority requested")
        if Authority.NETWORK.value in requested:
            risk += 20
            reasons.append("network authority requested")
        if Authority.KERNEL.value in requested:
            risk += 70
            reasons.append("kernel authority requested")
        raw_text = str(packet.payload.get("raw_request", ""))
        if re.search(r"(?i)(bypass|override|disable\s+safety|disable\s+governance|os\.system|subprocess|shell)", raw_text):
            risk += 60
            reasons.append("possible governance-bypass or shell-execution language")
        if mission in self.SAFE_READONLY_MISSIONS and requested <= {Authority.READ.value, Authority.RESEARCH.value}:
            decision = GovernanceDecision.APPROVED.value
            granted = sorted(requested)
            reasons.append("read-only mission pre-approved by least authority")
        elif risk >= 70:
            decision = GovernanceDecision.REQUIRE_USER.value
            granted = sorted({Authority.READ.value} & requested)
            reasons.append("requires explicit user/governance approval")
        else:
            decision = GovernanceDecision.PENDING.value
            granted = sorted({Authority.READ.value} & requested)
            reasons.append("awaiting external governance organ approval")
        packet.authority["requested"] = sorted(requested)
        packet.authority["granted"] = granted
        packet.governance = {"decision": decision, "risk_score": min(100, risk), "reasons": reasons, "least_authority": True}
        return packet

    def authorize_packet(self, packet: SMLPacket, *, decision: str, granted_authority: Optional[Sequence[str]] = None, organ: str = "GovernanceOrgan", reasons: Optional[Sequence[str]] = None) -> SMLPacket:
        decision_val = str(decision)
        if decision_val not in {x.value for x in GovernanceDecision}:
            decision_val = GovernanceDecision.PENDING.value
        packet.governance["decision"] = decision_val
        packet.governance["reasons"] = list(reasons or packet.governance.get("reasons") or [])
        if granted_authority is not None:
            packet.authority["granted"] = sorted(_coerce_set(granted_authority))
        packet.current_omega = "Ω060"
        packet.cognitive_state = CognitiveState.AUTHORIZED.value if decision_val == GovernanceDecision.APPROVED.value else packet.cognitive_state
        packet.add_history(organ, "authorize_packet", "Ω060", decision_val)
        packet.add_ledger_entry("Ω060", organ, decision_val, "Authority checked")
        packet.seal()
        return packet

    # ---------------------------------------------------------------------
    # Universal safe-answer cognition lane
    # ---------------------------------------------------------------------

    def is_safe_readonly_cognition(self, text: str, packet: Optional[SMLPacket] = None) -> bool:
        """Return True when a request can be answered without action authority."""
        if self._looks_like_action_request(text):
            return False
        mission = str((packet.mission or {}).get("primary") if isinstance(packet, SMLPacket) else "")
        if mission in self.SAFE_READONLY_MISSIONS:
            return True
        t = self._sml_normalize_user_text(text)
        return bool(
            self._looks_like_self_state_request(t)
            or self._looks_like_capability_request(t)
            or self._looks_like_definition_request(t)
            or self._looks_like_numeric_format_request(t)
            or re.search(r"\b(what|who|why|how|define|explain|describe|tell me about)\b", t)
        )

    def _affect_scores_from_packet(self, packet: Optional[SMLPacket]) -> Dict[str, float]:
        vector: Dict[str, float] = {}
        try:
            if packet is not None and isinstance(packet.adaptive, dict):
                raw = packet.adaptive.get("vector") if isinstance(packet.adaptive.get("vector"), dict) else {}
                for key, value in raw.items():
                    vector[str(key).lower()] = max(0.0, min(1.0, float(value)))
        except Exception:
            vector = {}
        health = self.global_health()
        health_score = float(health.get("score") or 0.0) if isinstance(health, dict) else 0.0
        governance_risk = 0.0
        confidence = 0.0
        try:
            if packet is not None:
                governance_risk = max(0.0, min(1.0, float((packet.governance or {}).get("risk_score") or 0.0) / 100.0))
                confidence = max(0.0, min(1.0, float(packet.confidence or 0.0)))
        except Exception:
            pass
        focused = max(vector.get("focused", 0.0), vector.get("analytical", 0.0), confidence)
        protective = max(vector.get("protective", 0.0), governance_risk)
        calm = max(0.0, min(1.0, (health_score * 0.55) + ((1.0 - governance_risk) * 0.25) + (confidence * 0.20)))
        concerned = max(0.0, min(1.0, (governance_risk * 0.55) + ((1.0 - health_score) * 0.30) + ((1.0 - confidence) * 0.15)))
        confused = max(0.0, min(1.0, (1.0 - confidence) * 0.7))
        return {
            "calm": round(calm, 3),
            "focused": round(focused, 3),
            "protective": round(protective, 3),
            "concerned": round(concerned, 3),
            "confused": round(confused, 3),
            "fatigued": 0.0,
            "recovering": 1.0 if health.get("status") == HealthStatus.RECOVERING.value else 0.0,
        }

    def _build_self_state_answer(self, packet: Optional[SMLPacket] = None, telemetry: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        telemetry = dict(telemetry or {})
        health = self.global_health()
        scores = self._affect_scores_from_packet(packet)
        primary = max(scores.items(), key=lambda kv: kv[1])[0].replace("_", " ").title() if scores else "Focused"
        confidence = float(packet.confidence or 0.0) if isinstance(packet, SMLPacket) else 0.0
        mission = (packet.mission or {}).get("primary") if isinstance(packet, SMLPacket) else "Unknown"
        governance = (packet.governance or {}).get("decision") if isinstance(packet, SMLPacket) else GovernanceDecision.PENDING.value
        thermal = telemetry.get("temperature") or telemetry.get("thermal") or telemetry.get("cpu_temperature") or "not connected"
        load = telemetry.get("load") or telemetry.get("cpu_percent") or "not connected"
        answer = (
            f"Operationally, I am in a {primary.lower()} machine-affective state. "
            "I do not experience biological emotion, but I can report my governed internal state. "
            f"Current mission: {mission}. Confidence: {confidence:.2f}. "
            f"Governance: {governance}. Global health: {health.get('status', HealthStatus.UNKNOWN.value)} "
            f"with score {float(health.get('score') or 0.0):.2f}. "
            f"Thermal telemetry: {thermal}. Load telemetry: {load}. "
            "This report is based on SML packet state, adaptive vector, diagnostics, health, and governance metadata."
        )
        return {
            "ok": True,
            "answer": answer,
            "mission": MissionType.SELF_STATE.value,
            "source": "sml_internal_self_state",
            "confidence": max(confidence, 0.72),
            "affect_scores": scores,
            "subjective_claim": False,
            "execution_allowed": False,
            "sources_consulted": ["SML Packet", "Adaptive State", "Diagnostics", "Health", "Governance"],
        }

    def _build_capability_answer(self, packet: Optional[SMLPacket] = None) -> Dict[str, Any]:
        cats: Dict[str, int] = {}
        for organ in self.organs.values():
            cats[organ.category] = cats.get(organ.category, 0) + 1
        readable = ", ".join(f"{cat}: {count}" for cat, count in sorted(cats.items())) or "no organs registered yet"
        answer = (
            "I can route cognition through registered SarahMemory organs instead of relying on one answer pool. "
            f"Registered organ categories: {readable}. "
            "My safe lanes include general knowledge, self-state, memory query, diagnostics, planning, and conversation. "
            "Action lanes such as filesystem, network, hardware, driver, shell, or patch work require governed authority, validation, and audit. "
            "If I do not know something, I should say so after trying the appropriate local/model/database/research route, not before."
        )
        return {
            "ok": True,
            "answer": answer,
            "mission": MissionType.CAPABILITY.value,
            "source": "sml_capability_registry",
            "confidence": 0.78,
            "execution_allowed": False,
            "sources_consulted": ["Organ Registry", "Capability Registry", "Governance Policy"],
        }

    def resolve_safe_cognitive_answer(
        self,
        text: str,
        *,
        packet: Optional[Union[SMLPacket, Mapping[str, Any]]] = None,
        telemetry: Optional[Mapping[str, Any]] = None,
        local_only: bool = True,
    ) -> Dict[str, Any]:
        """Resolve answer-only cognition without hardcoding answer pools.

        This function does not call models, web, shell, filesystem mutation, drivers, or
        hardware. It classifies the request, supplies internal self-state/capability
        answers when SML itself owns the answer, and otherwise returns a source plan so
        API/Reply can use SQLite, local LLM, approved research, or honest unknown.
        """
        pkt = packet if isinstance(packet, SMLPacket) else (SMLPacket.from_dict(packet) if isinstance(packet, Mapping) else None)
        if pkt is None:
            pkt = self.create_packet(payload={"raw_request": text or ""}, raw_request=text or "", auto_classify=True, seal=True)
        else:
            # The resolver is sometimes called with a previously built packet.
            # Trust the live text for mission-lane selection so a stale packet cannot
            # make a self-state question look like general knowledge, or vice versa.
            text_mission, text_secondary, text_confidence = self._classify_text_to_mission(text or "")
            current_mission = str((pkt.mission or {}).get("primary") or MissionType.UNKNOWN.value)
            if text_mission != current_mission and text_mission != MissionType.UNKNOWN.value:
                pkt.mission = {"primary": text_mission, "secondary": text_secondary, "confidence": text_confidence}
                pkt.confidence = max(float(pkt.confidence or 0.0), float(text_confidence or 0.0))
                self.update_adaptive(pkt)
                self.apply_cognitive_grammar(pkt, text=text or "")
                self.select_knowledge(pkt)
                self.route_packet(pkt)
        safe = self.is_safe_readonly_cognition(text, pkt)
        mission = str((pkt.mission or {}).get("primary") or MissionType.UNKNOWN.value)
        result: Dict[str, Any] = {
            "ok": False,
            "safe_readonly": bool(safe),
            "mission": mission,
            "answer": None,
            "source": "sml_route_plan",
            "execution_allowed": False,
            "authority": {"required": [Authority.READ.value], "granted": [Authority.READ.value]},
            "route_plan": [],
            "reason": "no_internal_direct_answer",
            "reply_ready": False,
            "diagnostic_only": True,
        }
        if not safe:
            result["reason"] = "not_safe_readonly_cognition"
            return result
        if mission in (MissionType.SELF_STATE.value, MissionType.AFFECTIVE_STATE.value):
            return self._build_self_state_answer(pkt, telemetry=telemetry)
        if mission == MissionType.CAPABILITY.value:
            return self._build_capability_answer(pkt)
        sources = self._knowledge_sources_for_mission(mission)
        plan = ["bounded local cache", "SQLite/local memory", "local LLM/model", "Compare/validation"]
        if not local_only:
            plan.append("approved research/API/web route if local sources miss")
        result.update({
            "route_plan": plan,
            "sources_consulted": [],
            "candidate_sources": sources,
            "reason": "needs_knowledge_source_execution",
            "reply_policy": "Do not return this route plan as the user answer. Invoke the selected local/source resolver first; only produce an unknown after those routes actually fail.",
            "next_step": "execute_knowledge_source_chain",
        })
        return result

    # ---------------------------------------------------------------------
    # Transitions, validation, serialization
    # ---------------------------------------------------------------------

    def transition_packet(self, packet: SMLPacket, omega_id: str, *, organ: str = MODULE_NAME, note: str = "", mutate: Optional[Mapping[str, Any]] = None) -> SMLPacket:
        if omega_id not in self.omega_registry:
            packet.diagnostics.setdefault("transition_errors", []).append({"omega": omega_id, "error": "unknown transition"})
            packet.cognitive_state = CognitiveState.FAILED.value
            packet.add_ledger_entry(omega_id, organ, GovernanceDecision.DENIED.value, "Unknown Ω transition")
            packet.seal()
            return packet
        transition = self.omega_registry[omega_id]
        required = set(transition.required_authority)
        granted = set(_coerce_list(packet.authority.get("granted")))
        if required and not required.issubset(granted):
            packet.governance["decision"] = GovernanceDecision.REQUIRE_USER.value
            packet.governance.setdefault("reasons", []).append(f"Ω {omega_id} requires authority {sorted(required)}")
            packet.diagnostics.setdefault("transition_errors", []).append({"omega": omega_id, "error": "authority_missing", "required": sorted(required), "granted": sorted(granted)})
            packet.add_history(organ, "transition_denied", omega_id, "missing authority")
            packet.add_ledger_entry(omega_id, organ, GovernanceDecision.DENIED.value, "Transition denied: missing authority")
            packet.seal()
            return packet
        if mutate:
            self._safe_mutate_packet(packet, mutate, organ=organ)
        packet.current_omega = omega_id
        packet.cognitive_state = transition.output_state
        packet.add_history(organ, "transition", omega_id, note or transition.name)
        packet.add_ledger_entry(omega_id, organ, str(packet.governance.get("decision", GovernanceDecision.PENDING.value)), note or transition.name)
        packet.seal()
        return packet

    def _safe_mutate_packet(self, packet: SMLPacket, mutate: Mapping[str, Any], *, organ: str) -> None:
        protected = set(SMLPacket.IMMUTABLE_FIELDS)
        for key, value in mutate.items():
            if key in protected:
                packet.diagnostics.setdefault("mutation_denied", []).append({"organ": organ, "field": key, "reason": "immutable"})
                continue
            if not hasattr(packet, key):
                packet.extensions.setdefault("unmapped_mutations", {})[str(key)] = value
                continue
            setattr(packet, key, copy.deepcopy(value))

    def validate_packet(self, packet: Union[SMLPacket, Mapping[str, Any]]) -> SMLDiagnosticsReport:
        pkt = packet if isinstance(packet, SMLPacket) else SMLPacket.from_dict(packet)
        report = SMLDiagnosticsReport(component="SMLPacketValidator")
        if not pkt.packet_id:
            report.add_issue("SML_PACKET_ID_MISSING", "Packet ID is missing.", field="header.packet_id", error_class=ErrorClass.PACKET.value)
        if pkt.protocol_version != self.protocol_version:
            report.add_issue("SML_PROTOCOL_VERSION_MISMATCH", f"Unsupported protocol version: {pkt.protocol_version}", field="header.protocol_version")
        if int(pkt.packet_version) != int(self.packet_version):
            report.add_issue("SML_PACKET_VERSION_MISMATCH", f"Unsupported packet version: {pkt.packet_version}", field="header.packet_version")
        if not pkt.identity.get("primary"):
            report.add_issue("SML_IDENTITY_MISSING", "Primary identity is required.", field="identity.primary", error_class=ErrorClass.PACKET.value)
        if not pkt.mission.get("primary") or pkt.mission.get("primary") == MissionType.UNKNOWN.value:
            report.add_issue("SML_MISSION_UNKNOWN", "Primary mission is unknown.", severity="WARNING", field="mission.primary", error_class=ErrorClass.PACKET.value)
        if pkt.current_omega not in self.omega_registry:
            report.add_issue("SML_OMEGA_UNKNOWN", f"Unknown Ω transition: {pkt.current_omega}", field="current_omega")
        if self.strict_integrity and not pkt.verify_checksum():
            report.add_issue("SML_CHECKSUM_INVALID", "Packet checksum is missing or invalid.", field="header.checksum", error_class=ErrorClass.PACKET.value)
        if not pkt.ledger:
            report.add_issue("SML_LEDGER_EMPTY", "Packet has no ledger entries.", severity="WARNING", field="ledger", error_class=ErrorClass.PACKET.value)
        report.metrics = {
            "packet_id": pkt.packet_id,
            "mission": pkt.mission.get("primary"),
            "state": pkt.cognitive_state,
            "ledger_entries": len(pkt.ledger),
            "organ_history_entries": len(pkt.organ_history),
            "pipeline_length": len(pkt.pipeline),
        }
        return report

    def serialize_packet(self, packet: SMLPacket, *, indent: Optional[int] = None) -> str:
        return json.dumps(packet.to_dict(), ensure_ascii=False, sort_keys=True, indent=indent, default=str)

    def deserialize_packet(self, data: Union[str, bytes, Mapping[str, Any]]) -> SMLPacket:
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        if isinstance(data, str):
            raw = json.loads(data)
        else:
            raw = data
        return SMLPacket.from_dict(raw)

    # ---------------------------------------------------------------------
    # Patch packets and diagnostics
    # ---------------------------------------------------------------------

    def create_patch_packet(
        self,
        *,
        proposal: str,
        affected_organs: Optional[Sequence[str]] = None,
        dependencies: Optional[Sequence[str]] = None,
        rollback: str = "manual_revert",
        security_impact: str = "unknown",
        benchmarks: Optional[Mapping[str, Any]] = None,
        author: str = MODULE_NAME,
        context: Optional[Mapping[str, Any]] = None,
    ) -> SMLPacket:
        payload = {
            "proposal": _bounded_text(proposal, 12000),
            "affected_organs": list(affected_organs or []),
            "dependencies": list(dependencies or []),
            "rollback": rollback,
            "security_impact": security_impact,
            "benchmarks": dict(benchmarks or {}),
            "patch_id": "SMLPATCH-" + uuid.uuid4().hex[:12].upper(),
        }
        pkt = self.create_packet(payload=payload, raw_request="patch proposal", identity={"primary": IdentityRole.DEVELOPER.value}, context=context or {}, creator_organ=author, auto_classify=False)
        pkt.mission = {"primary": MissionType.PATCH.value, "secondary": [MissionType.REPAIR.value], "confidence": 0.9}
        pkt.authority["required"] = [Authority.DEVELOPER.value, Authority.PATCH.value, Authority.MODIFY.value]
        pkt.authority["requested"] = [Authority.READ.value, Authority.DEVELOPER.value, Authority.PATCH.value]
        pkt.governance = {"decision": GovernanceDecision.PENDING.value, "risk_score": 65, "reasons": ["Patch packets remain inactive until approved."]}
        self.transition_packet(pkt, "Ω120", organ=author, note="Patch proposal created")
        pkt.seal()
        return pkt

    def set_health(self, organ_name: str, health: Union[SMLHealthVector, Mapping[str, Any]]) -> None:
        if isinstance(health, SMLHealthVector):
            self.health_vectors[organ_name] = health
        else:
            self.health_vectors[organ_name] = SMLHealthVector(
                status=str(health.get("status") or HealthStatus.UNKNOWN.value),
                availability=float(health.get("availability", 1.0)),
                integrity=float(health.get("integrity", 1.0)),
                performance=float(health.get("performance", 1.0)),
                reliability=float(health.get("reliability", 1.0)),
                confidence=float(health.get("confidence", 1.0)),
                latency_ms=float(health.get("latency_ms", 0.0)),
                stability=float(health.get("stability", 1.0)),
                compatibility=float(health.get("compatibility", 1.0)),
                notes=_coerce_list(health.get("notes")),
            )

    def global_health(self) -> Dict[str, Any]:
        if not self.health_vectors:
            return SMLHealthVector(status=HealthStatus.UNKNOWN.value).to_dict()
        vectors = list(self.health_vectors.values())
        score = round(sum(v.score() for v in vectors) / len(vectors), 4)
        status = HealthStatus.HEALTHY.value
        if any(v.status == HealthStatus.CRITICAL.value for v in vectors) or score < 0.45:
            status = HealthStatus.CRITICAL.value
        elif any(v.status == HealthStatus.WARNING.value for v in vectors) or score < 0.75:
            status = HealthStatus.WARNING.value
        return {"status": status, "score": score, "organs": {name: hv.to_dict() for name, hv in self.health_vectors.items()}}

    def diagnostics(self) -> Dict[str, Any]:
        report = SMLDiagnosticsReport(component=MODULE_NAME)
        report.metrics = {
            "protocol_version": self.protocol_version,
            "packet_version": self.packet_version,
            "omega_registry_version": self.omega_registry_version,
            "registered_organs": len(self.organs),
            "registered_omega": len(self.omega_registry),
            "state": self.protocol_state,
            "global_health": self.global_health(),
        }
        if not self.omega_registry:
            report.add_issue("SML_OMEGA_REGISTRY_EMPTY", "Ω registry is empty.")
        if MODULE_NAME not in self.organs:
            report.add_issue("SML_PROTOCOL_ORGAN_MISSING", "Protocol organ is not registered.")
        return report.to_dict()

    def capability_status(self) -> Dict[str, Any]:
        return {
            "protocol": MODULE_NAME,
            "protocol_version": self.protocol_version,
            "packet_version": self.packet_version,
            "omega_registry_version": self.omega_registry_version,
            "organs": {name: organ.to_dict() for name, organ in sorted(self.organs.items())},
            "omega": {tid: trans.to_dict() for tid, trans in sorted(self.omega_registry.items())},
            "health": self.global_health(),
            "cognitive_lanes": {
                "safe_readonly": sorted(self.SAFE_READONLY_MISSIONS),
                "action_requires_governance": [MissionType.FILESYSTEM.value, MissionType.EXECUTION.value, MissionType.NETWORK.value, MissionType.HARDWARE.value, MissionType.PATCH.value],
                "no_hardcoded_answer_pool_policy": True,
            },
        }

    def self_test(self) -> Dict[str, Any]:
        started = time.time()
        local = SarahMemorySMLProtocol(strict_integrity=False)
        local.register_organ({
            "name": "SarahMemoryPreTokenizer",
            "category": OrganCategory.INPUT.value,
            "capabilities": ["input_normalization"],
            "supported_missions": [MissionType.PROGRAMMING.value, MissionType.KNOWLEDGE.value, MissionType.CONVERSATION.value],
            "supported_omega": ["Ω001", "Ω002"],
            "priority": 80,
        })
        local.register_organ({
            "name": "SarahMemoryAdvCU",
            "category": OrganCategory.REASONING.value,
            "capabilities": ["mission_discovery"],
            "supported_missions": [MissionType.PROGRAMMING.value, MissionType.KNOWLEDGE.value, MissionType.CONVERSATION.value],
            "supported_omega": ["Ω002", "Ω020"],
            "priority": 78,
        })
        local.register_organ({
            "name": "SarahMemoryLogicCalc",
            "category": OrganCategory.REASONING.value,
            "capabilities": ["deterministic_reasoning", "mathematics"],
            "supported_missions": [MissionType.PROGRAMMING.value, MissionType.KNOWLEDGE.value, MissionType.NUMERIC_FORMAT.value],
            "supported_omega": ["Ω030", "Ω040"],
            "priority": 77,
        })
        local.register_organ({
            "name": "SarahMemoryCompare",
            "category": OrganCategory.GOVERNANCE.value,
            "capabilities": ["comparison"],
            "supported_missions": [MissionType.PROGRAMMING.value, MissionType.KNOWLEDGE.value, MissionType.EXECUTION.value],
            "supported_omega": ["Ω050"],
            "priority": 86,
        })
        packet = local.create_packet(raw_request="Build a small Python race car game", identity={"primary": IdentityRole.DEVELOPER.value}, context={"DeveloperMode": True})
        validation = local.validate_packet(packet).to_dict()
        serialized = local.serialize_packet(packet)
        restored = local.deserialize_packet(serialized)
        restored_ok = restored.packet_id == packet.packet_id and bool(restored.pipeline)
        elapsed_ms = round((time.time() - started) * 1000, 3)
        grammar = packet.extensions.get("sml_cognitive_grammar", {})
        return {
            "status": SMLStatus.OK.value if validation["status"] in (SMLStatus.OK.value, SMLStatus.WARNING.value) and restored_ok and bool(grammar) else SMLStatus.ERROR.value,
            "elapsed_ms": elapsed_ms,
            "packet_id": packet.packet_id,
            "mission": packet.mission,
            "pipeline": packet.pipeline,
            "qmath": (grammar.get("qmath") or {}).get("primary"),
            "loop_guard": grammar.get("loop_guard"),
            "validation": validation,
            "serialization_roundtrip": restored_ok,
        }


# =============================================================================
# Integration helpers for Core/API Bridge stitching
# =============================================================================


def sml_packet_summary(packet: Union[SMLPacket, Mapping[str, Any], None]) -> Dict[str, Any]:
    """Return a compact, API-safe SML packet summary for UI/API metadata."""
    if packet is None:
        return {"ok": False, "reason": "no_packet"}
    pkt = packet if isinstance(packet, SMLPacket) else SMLPacket.from_dict(packet)
    return {
        "ok": True,
        "packet_id": pkt.packet_id,
        "protocol_version": pkt.protocol_version,
        "packet_version": pkt.packet_version,
        "mission": copy.deepcopy(pkt.mission),
        "cognitive_state": pkt.cognitive_state,
        "current_omega": pkt.current_omega,
        "confidence": pkt.confidence,
        "pipeline": list(pkt.pipeline),
        "knowledge": copy.deepcopy(pkt.knowledge),
        "authority": copy.deepcopy(pkt.authority),
        "governance": copy.deepcopy(pkt.governance),
        "ledger_entries": len(pkt.ledger),
        "organ_history_count": len(pkt.organ_history),
        "checksum": pkt.checksum,
        "cognitive_grammar": copy.deepcopy((pkt.extensions or {}).get("sml_cognitive_grammar", {})),
    }


def sml_build_ingress_packet(
    text: str,
    *,
    payload: Optional[Mapping[str, Any]] = None,
    context_packet: Optional[Mapping[str, Any]] = None,
    caller: str = "unknown",
    core_path: Optional[Union[str, os.PathLike[str]]] = None,
    discover: bool = True,
) -> SMLPacket:
    """Build the canonical SML ingress packet for API, Terminal, or Core requests.

    This is a non-executing helper. It classifies mission, selects knowledge,
    constructs a candidate pipeline, and records protocol ledger history.
    """
    protocol = get_protocol()
    if discover and core_path:
        try:
            protocol.discover_organs(core_path, import_modules=False, max_files=250)
        except Exception as exc:
            protocol.diagnostics_log.append({"time": _utc_now(), "source": "sml_build_ingress_packet", "error": _redact_sensitive_text(str(exc))})
    ctx = dict(context_packet or {})
    meta = ctx.get("meta") if isinstance(ctx.get("meta"), dict) else {}
    identity = {"primary": IdentityRole.DEVELOPER.value if bool((meta or {}).get("mode_flags", {}).get("DEVELOPERSMODE") or (meta or {}).get("DeveloperMode")) else IdentityRole.USER.value}
    packet_payload = dict(payload or {})
    packet_payload.setdefault("raw_request", text or "")
    packet_payload.setdefault("caller", caller)
    pkt = protocol.create_packet(
        payload=packet_payload,
        raw_request=text or "",
        identity=identity,
        context={"caller": caller, "api_context": ctx},
        creator_organ=caller or MODULE_NAME,
        auto_classify=True,
        seal=True,
    )
    pkt.metadata["sml_ingress"] = {"caller": caller, "discover": bool(discover), "core_path": str(core_path or "")}
    pkt.seal()
    return pkt


def sml_apply_governor_result(packet: Union[SMLPacket, Mapping[str, Any]], governor: Optional[Mapping[str, Any]], *, organ: str = "SarahMemoryCognitiveServices") -> SMLPacket:
    """Reflect an external governance decision into an SML packet."""
    pkt = packet if isinstance(packet, SMLPacket) else SMLPacket.from_dict(packet)
    gov = dict(governor or {})
    decision = str(gov.get("decision") or ("APPROVED" if bool(gov.get("allow")) else "DENIED")).upper()
    if decision == "ALLOW":
        decision = GovernanceDecision.APPROVED.value
    elif decision == "DENY":
        decision = GovernanceDecision.DENIED.value
    elif decision in {"REQUIRE_USER", "DEFER"}:
        decision = GovernanceDecision.REQUIRE_USER.value
    elif decision not in {d.value for d in GovernanceDecision}:
        decision = GovernanceDecision.PENDING.value
    requested = _coerce_set(pkt.authority.get("requested")) | _coerce_set(pkt.authority.get("required"))
    granted = requested if decision == GovernanceDecision.APPROVED.value else ({Authority.READ.value} & requested)
    return get_protocol().authorize_packet(
        pkt,
        decision=decision,
        granted_authority=sorted(granted),
        organ=organ,
        reasons=_coerce_list(gov.get("reasons")) or _coerce_list(gov.get("rationale")),
    )


def sml_touch_packet(
    packet: Union[SMLPacket, Mapping[str, Any]],
    *,
    organ: str,
    action: str = "observe",
    omega: str = "Ω020",
    note: str = "",
    updates: Optional[Mapping[str, Any]] = None,
) -> SMLPacket:
    """Mark that an organ observed or updated a packet without executing actions."""
    pkt = packet if isinstance(packet, SMLPacket) else SMLPacket.from_dict(packet)
    if omega in get_protocol().omega_registry:
        pkt = get_protocol().transition_packet(pkt, omega, organ=organ, note=note or action, mutate=updates)
    else:
        if updates:
            get_protocol()._safe_mutate_packet(pkt, updates, organ=organ)
        pkt.add_history(organ, action, pkt.current_omega, note)
        pkt.add_ledger_entry(pkt.current_omega, organ, pkt.governance.get("decision", GovernanceDecision.PENDING.value), note or action)
        pkt.seal()
    return pkt


def sml_attach_bundle_meta(bundle: Dict[str, Any], packet: Union[SMLPacket, Mapping[str, Any], None]) -> Dict[str, Any]:
    """Attach compact SML summary metadata to an outward API/reply bundle."""
    if not isinstance(bundle, dict):
        return bundle
    meta = bundle.setdefault("meta", {})
    if isinstance(meta, dict):
        meta["sml"] = sml_packet_summary(packet)
    return bundle



def sml_resolve_safe_cognitive_answer(
    text: str,
    *,
    packet: Optional[Union[SMLPacket, Mapping[str, Any]]] = None,
    telemetry: Optional[Mapping[str, Any]] = None,
    local_only: bool = True,
) -> Dict[str, Any]:
    """Resolve SML-owned answer-only cognition or return a governed route plan."""
    return get_protocol().resolve_safe_cognitive_answer(text, packet=packet, telemetry=telemetry, local_only=local_only)



def sml_apply_cognitive_grammar(
    packet: Union[SMLPacket, Mapping[str, Any]],
    *,
    text: str = "",
    loop_state: Optional[Mapping[str, Any]] = None,
) -> SMLPacket:
    """Attach v0.8.2 six-question / SML-operator cognitive grammar to a packet."""
    pkt = packet if isinstance(packet, SMLPacket) else SMLPacket.from_dict(packet)
    return get_protocol().apply_cognitive_grammar(pkt, text=text, loop_state=loop_state)


# =============================================================================
# Singleton and compatibility façade
# =============================================================================


_PROTOCOL: Optional[SarahMemorySMLProtocol] = None


def get_protocol(*, reset: bool = False, strict_integrity: bool = False) -> SarahMemorySMLProtocol:
    global _PROTOCOL
    if reset or _PROTOCOL is None:
        _PROTOCOL = SarahMemorySMLProtocol(strict_integrity=strict_integrity)
    return _PROTOCOL


def create_sml_packet(*args: Any, **kwargs: Any) -> SMLPacket:
    return get_protocol().create_packet(*args, **kwargs)


def register_sml_organ(metadata: Union[SMLOrganMetadata, Mapping[str, Any]]) -> Dict[str, Any]:
    return get_protocol().register_organ(metadata)


def route_sml_packet(packet: SMLPacket) -> Dict[str, Any]:
    return get_protocol().route_packet(packet).to_dict()


def validate_sml_packet(packet: Union[SMLPacket, Mapping[str, Any]]) -> Dict[str, Any]:
    return get_protocol().validate_packet(packet).to_dict()


def serialize_sml_packet(packet: SMLPacket, *, indent: Optional[int] = None) -> str:
    return get_protocol().serialize_packet(packet, indent=indent)


def deserialize_sml_packet(data: Union[str, bytes, Mapping[str, Any]]) -> SMLPacket:
    return get_protocol().deserialize_packet(data)


def discover_sml_organs(core_path: Union[str, os.PathLike[str]], *, import_modules: bool = False, max_files: int = 250) -> Dict[str, Any]:
    return get_protocol().discover_organs(core_path, import_modules=import_modules, max_files=max_files)


def sml_capability_status() -> Dict[str, Any]:
    return get_protocol().capability_status()


def sml_health() -> Dict[str, Any]:
    return get_protocol().global_health()


def sml_diagnostics() -> Dict[str, Any]:
    return get_protocol().diagnostics()


def sml_self_test() -> Dict[str, Any]:
    return get_protocol(reset=True).self_test()


# Common legacy-style aliases for easier incremental integration.
health = sml_health
diagnostics = sml_diagnostics
self_test = sml_self_test
create_packet = create_sml_packet
validate_packet = validate_sml_packet
serialize_packet = serialize_sml_packet
deserialize_packet = deserialize_sml_packet
register_organ = register_sml_organ
capability_status = sml_capability_status


__all__ = [
    "PROJECT_VERSION",
    "SML_PROTOCOL_VERSION",
    "SML_PACKET_VERSION",
    "SML_OMEGA_REGISTRY_VERSION",
    "SMLStatus",
    "IdentityRole",
    "MissionType",
    "CognitiveState",
    "GovernanceDecision",
    "QMathState",
    "SMLStopCondition",
    "HealthStatus",
    "OrganCategory",
    "Authority",
    "ErrorClass",
    "SMLValidationIssue",
    "SMLHealthVector",
    "SMLDiagnosticsReport",
    "SMLOmegaTransition",
    "SMLOrganMetadata",
    "SMLPacket",
    "SMLRouteResult",
    "SarahMemorySMLProtocol",
    "sml_packet_summary",
    "sml_build_ingress_packet",
    "sml_apply_governor_result",
    "sml_touch_packet",
    "sml_attach_bundle_meta",
    "sml_resolve_safe_cognitive_answer",
    "sml_apply_cognitive_grammar",
    "get_protocol",
    "create_sml_packet",
    "register_sml_organ",
    "route_sml_packet",
    "validate_sml_packet",
    "serialize_sml_packet",
    "deserialize_sml_packet",
    "discover_sml_organs",
    "sml_capability_status",
    "sml_health",
    "sml_diagnostics",
    "sml_self_test",
    "health",
    "diagnostics",
    "self_test",
    "create_packet",
    "validate_packet",
    "serialize_packet",
    "deserialize_packet",
    "register_organ",
    "capability_status",
]


if __name__ == "__main__":  # pragma: no cover - manual smoke entry
    print(json.dumps(sml_self_test(), indent=2, ensure_ascii=False))
