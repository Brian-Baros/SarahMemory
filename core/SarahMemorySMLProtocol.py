"""--==The SarahMemory Project==--
File: SarahMemorySMLProtocol.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-08-16
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
SarahMemory SML/QSML Universal Natural Programming Language / Governed Cognitive Substrate

PURPOSE:
- Provide the canonical SML Packet, QSML language runtime, and protocol substrate for SarahMemory AiOS.
- Serve as the unifying language that runs through SarahMemory rather than sitting above it.
- Compile natural-language intent into typed variables, cognitive AST, legal route candidates,
  capability/organ contracts, Ω transitions, governance requirements, and sandbox build specifications.
- Coordinate mission classification, packet validation, organ registration, capability negotiation,
  routing, diagnostics, health, serialization, and bounded patch-packet creation.
- Preserve specialized organ ownership: Neuron activates/weights routes, governance authorizes,
  domain organs compute, OperatorCore executes, Reply presents, and Ledger records.
- Remain local-first, deterministic, auditable, model-independent, and user-governed.

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
- Executable QSML / Universal Natural Programming Language alpha runtime with arbitrary-application synthesis contracts.
- Formal typed variables, AST, organ contracts, compiler semantics, and bounded evaluator are available.
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
# VALIDATION_DATE = "2026-08-16"
# VALIDATION_TIME = "23:13:00"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Canonical SML/QSML universal natural programming language runtime: typed variables, cognitive AST, organ contracts, compiler semantics, Ω registry, bounded routing, integrity, diagnostics, and serialization. Coordinates cognition; does not execute actions."
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
QSML_LANGUAGE_VERSION = "QSML/0.2"
QSML_LANGUAGE_NAME = "SarahMemory SML Universal Natural Programming Language"
SML_TYPE_SYSTEM_VERSION = "SML-TYPES/1.1"
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


class GCOPEventType(str, Enum):
    """General event families for Persistent Governed Cognitive Operation."""
    USER_DIRECTIVE = "USER_DIRECTIVE"
    SENSOR_UPDATE = "SENSOR_UPDATE"
    TIME_EVENT = "TIME_EVENT"
    TASK_RESULT = "TASK_RESULT"
    NETWORK_CHANGE = "NETWORK_CHANGE"
    MESSAGE_RECEIVED = "MESSAGE_RECEIVED"
    RESOURCE_ALERT = "RESOURCE_ALERT"
    SECURITY_EVENT = "SECURITY_EVENT"
    MAINTENANCE_ALERT = "MAINTENANCE_ALERT"
    MODEL_EVIDENCE = "MODEL_EVIDENCE"
    AGENT_RETURN = "AGENT_RETURN"
    AUTHORITY_CHANGE = "AUTHORITY_CHANGE"
    CONTRACT_CHANGE = "CONTRACT_CHANGE"
    DEVICE_RESULT = "DEVICE_RESULT"
    ENVIRONMENT_CHANGE = "ENVIRONMENT_CHANGE"
    CAPABILITY_GAP = "CAPABILITY_GAP"
    SYSTEM_TICK = "SYSTEM_TICK"
    UNKNOWN = "UNKNOWN"


class GCOPCycleStatus(str, Enum):
    """Legal terminal/continuation states for one bounded cognitive cycle."""
    CONTINUE = "CONTINUE"
    WAIT = "WAIT"
    REQUIRE_USER = "REQUIRE_USER"
    SAFE_HOLD = "SAFE_HOLD"
    DEGRADED = "DEGRADED"
    RECOVER = "RECOVER"
    COMPLETE = "COMPLETE"
    STOP = "STOP"


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
    OWNERSHIP = "Class 450 — Ownership Error"
    EXECUTION = "Class 500 — Execution Error"
    LEARNING = "Class 600 — Learning Error"


class SMLDataType(str, Enum):
    """Formal QSML value types.

    These are language-level identities, not replacement implementations for
    LogicCalc, Database, Neuron, or any other domain organ.
    """
    NULL = "SML_NULL"
    BOOL = "SML_BOOL"
    INT = "SML_INT"
    UINT = "SML_UINT"
    FLOAT = "SML_FLOAT"
    DECIMAL = "SML_DECIMAL"
    STR = "SML_STR"
    BYTES = "SML_BYTES"
    BINARY = "SML_BINARY"
    OCTAL = "SML_OCTAL"
    HEX = "SML_HEX"
    COMPLEX = "SML_COMPLEX"
    VECTOR = "SML_VECTOR"
    MATRIX = "SML_MATRIX"
    TENSOR = "SML_TENSOR"
    UNIT = "SML_UNIT"
    FORMULA = "SML_FORMULA"
    LIST = "SML_LIST"
    SET = "SML_SET"
    MAP = "SML_MAP"
    OBJECT = "SML_OBJECT"
    ENTITY = "SML_ENTITY"
    PATH = "SML_PATH"
    URL = "SML_URL"
    TIME = "SML_TIME"
    DURATION = "SML_DURATION"
    MISSION = "SML_MISSION"
    CONTEXT = "SML_CONTEXT"
    IDENTITY = "SML_IDENTITY"
    AUTHORITY = "SML_AUTHORITY"
    CONFIDENCE = "SML_CONFIDENCE"
    SOURCE = "SML_SOURCE"
    ROUTE = "SML_ROUTE"
    PIPELINE = "SML_PIPELINE"
    STATE = "SML_STATE"
    PACKET = "SML_PACKET"
    GRAPH = "SML_GRAPH"
    OMEGA = "SML_OMEGA"
    APPLICATION_BLUEPRINT = "SML_APPLICATION_BLUEPRINT"
    REQUIREMENT = "SML_REQUIREMENT"
    FILE_PLAN = "SML_FILE_PLAN"
    UNKNOWN = "SML_UNKNOWN"


class SMLSemanticType(str, Enum):
    USER_REQUEST = "USER_REQUEST"
    APPLICATION_NAME = "APPLICATION_NAME"
    PROJECT_KIND = "PROJECT_KIND"
    LANGUAGE = "LANGUAGE"
    FRAMEWORK = "FRAMEWORK"
    FEATURE = "FEATURE"
    CAPABILITY = "CAPABILITY"
    TARGET_PLATFORM = "TARGET_PLATFORM"
    DATA_ENTITY = "DATA_ENTITY"
    ACTION = "ACTION"
    CONSTRAINT = "CONSTRAINT"
    AUTHORITY_REQUIREMENT = "AUTHORITY_REQUIREMENT"
    SOURCE_PREFERENCE = "SOURCE_PREFERENCE"
    RISK = "RISK"
    UI_COMPONENT = "UI_COMPONENT"
    STORAGE = "STORAGE"
    NETWORK_MODE = "NETWORK_MODE"
    INPUT_DEVICE = "INPUT_DEVICE"
    OUTPUT = "OUTPUT"
    REQUIREMENT = "REQUIREMENT"
    COMPONENT = "COMPONENT"
    FILE_PLAN = "FILE_PLAN"
    DEPENDENCY = "DEPENDENCY"
    TEST = "TEST"
    ACCEPTANCE = "ACCEPTANCE"
    SYNTHESIS_PHASE = "SYNTHESIS_PHASE"
    UNKNOWN = "UNKNOWN"


class SMLScope(str, Enum):
    PACKET = "PACKET"
    SESSION = "SESSION"
    MISSION = "MISSION"
    USER = "USER"
    SYSTEM = "SYSTEM"
    ORGAN = "ORGAN"
    SANDBOX = "SANDBOX"
    PERSISTENT = "PERSISTENT"


class SMLMutability(str, Enum):
    IMMUTABLE = "IMMUTABLE"
    MUTABLE = "MUTABLE"
    APPEND_ONLY = "APPEND_ONLY"


class SMLASTKind(str, Enum):
    PROGRAM = "PROGRAM"
    DECLARE = "DECLARE"
    LITERAL = "LITERAL"
    IDENTIFIER = "IDENTIFIER"
    OPERATOR = "OPERATOR"
    CONDITION = "CONDITION"
    BRANCH = "BRANCH"
    SEQUENCE = "SEQUENCE"
    MISSION = "MISSION"
    SOURCE = "SOURCE"
    CONSTRAINT = "CONSTRAINT"
    REQUIREMENT = "REQUIREMENT"
    CAPABILITY = "CAPABILITY"
    PROJECT = "PROJECT"
    FEATURE = "FEATURE"
    TARGET = "TARGET"
    UNKNOWN = "UNKNOWN"


class SMLCompileStatus(str, Enum):
    COMPILED = "COMPILED"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    ERROR = "ERROR"


class SMLSynthesisPhase(str, Enum):
    """Governed arbitrary-application synthesis lifecycle."""
    COMPILE = "COMPILE"
    ARCHITECT = "ARCHITECT"
    GENERATE = "GENERATE"
    VALIDATE = "VALIDATE"
    REPAIR = "REPAIR"
    PACKAGE = "PACKAGE"
    READY = "READY"
    BLOCKED = "BLOCKED"


class SMLArtifactRole(str, Enum):
    SOURCE = "SOURCE"
    ENTRYPOINT = "ENTRYPOINT"
    CONFIG = "CONFIG"
    MANIFEST = "MANIFEST"
    TEST = "TEST"
    DOCUMENTATION = "DOCUMENTATION"
    ASSET_TEXT = "ASSET_TEXT"
    DATA = "DATA"
    OTHER = "OTHER"


# =============================================================================
# Dataclasses
# =============================================================================



@dataclass
class SMLVariable:
    """First-class typed QSML variable with provenance and ownership."""
    variable_id: str = field(default_factory=lambda: "var_" + uuid.uuid4().hex[:16])
    name: str = ""
    value: Any = None
    data_type: str = SMLDataType.UNKNOWN.value
    semantic_type: str = SMLSemanticType.UNKNOWN.value
    scope: str = SMLScope.PACKET.value
    owner: str = MODULE_NAME
    authority: List[str] = field(default_factory=lambda: [Authority.READ.value])
    confidence: float = 1.0
    mutability: str = SMLMutability.IMMUTABLE.value
    source_text: str = ""
    source: str = "compiler"
    lifetime: str = "packet"
    validation_state: str = "UNVERIFIED"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable_id": self.variable_id,
            "name": self.name,
            "value": copy.deepcopy(self.value),
            "data_type": self.data_type,
            "semantic_type": self.semantic_type,
            "scope": self.scope,
            "owner": self.owner,
            "authority": list(self.authority),
            "confidence": float(self.confidence),
            "mutability": self.mutability,
            "source_text": _bounded_text(self.source_text, 2000),
            "source": self.source,
            "lifetime": self.lifetime,
            "validation_state": self.validation_state,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SMLVariable":
        return cls(
            variable_id=str(data.get("variable_id") or "var_" + uuid.uuid4().hex[:16]),
            name=str(data.get("name") or ""),
            value=copy.deepcopy(data.get("value")),
            data_type=str(data.get("data_type") or SMLDataType.UNKNOWN.value),
            semantic_type=str(data.get("semantic_type") or SMLSemanticType.UNKNOWN.value),
            scope=str(data.get("scope") or SMLScope.PACKET.value),
            owner=str(data.get("owner") or MODULE_NAME),
            authority=_coerce_list(data.get("authority")) or [Authority.READ.value],
            confidence=float(data.get("confidence") or 0.0),
            mutability=str(data.get("mutability") or SMLMutability.IMMUTABLE.value),
            source_text=str(data.get("source_text") or ""),
            source=str(data.get("source") or "compiler"),
            lifetime=str(data.get("lifetime") or "packet"),
            validation_state=str(data.get("validation_state") or "UNVERIFIED"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class SMLVariableRegistry:
    """Scoped QSML symbol table."""
    variables: Dict[str, SMLVariable] = field(default_factory=dict)

    def define(self, variable: SMLVariable, *, replace: bool = False) -> SMLVariable:
        key = str(variable.name or variable.variable_id).strip()
        if not key:
            raise ValueError("SML variable requires a name or id")
        if key in self.variables and not replace:
            raise ValueError(f"SML variable already defined: {key}")
        self.variables[key] = variable
        return variable

    def get(self, name: str) -> Optional[SMLVariable]:
        return self.variables.get(str(name or ""))

    def to_dict(self) -> Dict[str, Any]:
        return {name: var.to_dict() for name, var in sorted(self.variables.items())}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SMLVariableRegistry":
        reg = cls()
        for name, value in dict(data or {}).items():
            if isinstance(value, Mapping):
                var = SMLVariable.from_dict(value)
                if not var.name:
                    var.name = str(name)
                reg.variables[str(name)] = var
        return reg


@dataclass
class SMLASTNode:
    """Serializable cognitive abstract syntax tree node."""
    node_id: str = field(default_factory=lambda: "ast_" + uuid.uuid4().hex[:16])
    kind: str = SMLASTKind.UNKNOWN.value
    value: Any = None
    data_type: str = SMLDataType.UNKNOWN.value
    semantic_type: str = SMLSemanticType.UNKNOWN.value
    operator: str = ""
    children: List["SMLASTNode"] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "value": copy.deepcopy(self.value),
            "data_type": self.data_type,
            "semantic_type": self.semantic_type,
            "operator": self.operator,
            "children": [child.to_dict() for child in self.children],
            "attributes": copy.deepcopy(self.attributes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SMLASTNode":
        return cls(
            node_id=str(data.get("node_id") or "ast_" + uuid.uuid4().hex[:16]),
            kind=str(data.get("kind") or SMLASTKind.UNKNOWN.value),
            value=copy.deepcopy(data.get("value")),
            data_type=str(data.get("data_type") or SMLDataType.UNKNOWN.value),
            semantic_type=str(data.get("semantic_type") or SMLSemanticType.UNKNOWN.value),
            operator=str(data.get("operator") or ""),
            children=[cls.from_dict(x) for x in list(data.get("children") or []) if isinstance(x, Mapping)],
            attributes=dict(data.get("attributes") or {}),
        )


@dataclass
class SMLOrganContract:
    """Formal QSML ABI contract for a SarahMemory organ."""
    name: str
    accepts_types: List[str] = field(default_factory=list)
    produces_types: List[str] = field(default_factory=list)
    reads_packet_fields: List[str] = field(default_factory=list)
    owns_packet_fields: List[str] = field(default_factory=list)
    writes_packet_fields: List[str] = field(default_factory=list)
    supported_missions: List[str] = field(default_factory=list)
    supported_operators: List[str] = field(default_factory=list)
    supported_omega: List[str] = field(default_factory=list)
    required_authority: List[str] = field(default_factory=lambda: [Authority.READ.value])
    deterministic: bool = False
    advisory_only: bool = False
    sandbox_only: bool = False
    side_effecting: bool = False
    failure_states: List[str] = field(default_factory=list)
    rollback_contract: str = "record_and_recover"
    metadata: Dict[str, Any] = field(default_factory=dict)
    language_contract: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "accepts_types": list(self.accepts_types),
            "produces_types": list(self.produces_types),
            "reads_packet_fields": list(self.reads_packet_fields),
            "owns_packet_fields": list(self.owns_packet_fields),
            "writes_packet_fields": list(self.writes_packet_fields),
            "supported_missions": list(self.supported_missions),
            "supported_operators": list(self.supported_operators),
            "supported_omega": list(self.supported_omega),
            "required_authority": list(self.required_authority),
            "deterministic": bool(self.deterministic),
            "advisory_only": bool(self.advisory_only),
            "sandbox_only": bool(self.sandbox_only),
            "side_effecting": bool(self.side_effecting),
            "failure_states": list(self.failure_states),
            "rollback_contract": self.rollback_contract,
            "metadata": copy.deepcopy(self.metadata),
            "language_contract": copy.deepcopy(self.language_contract),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SMLOrganContract":
        return cls(
            name=str(data.get("name") or "unknown_organ"),
            accepts_types=_coerce_list(data.get("accepts_types")),
            produces_types=_coerce_list(data.get("produces_types")),
            reads_packet_fields=_coerce_list(data.get("reads_packet_fields")),
            owns_packet_fields=_coerce_list(data.get("owns_packet_fields")),
            writes_packet_fields=_coerce_list(data.get("writes_packet_fields")),
            supported_missions=_coerce_list(data.get("supported_missions")),
            supported_operators=_coerce_list(data.get("supported_operators")),
            supported_omega=_coerce_list(data.get("supported_omega")),
            required_authority=_coerce_list(data.get("required_authority")) or [Authority.READ.value],
            deterministic=bool(data.get("deterministic", False)),
            advisory_only=bool(data.get("advisory_only", False)),
            sandbox_only=bool(data.get("sandbox_only", False)),
            side_effecting=bool(data.get("side_effecting", False)),
            failure_states=_coerce_list(data.get("failure_states")),
            rollback_contract=str(data.get("rollback_contract") or "record_and_recover"),
            metadata=dict(data.get("metadata") or {}),
            language_contract=dict(data.get("language_contract") or {}),
        )


@dataclass
class SMLRequirement:
    """One traceable requirement in a QSML application synthesis contract."""
    requirement_id: str = field(default_factory=lambda: "req_" + uuid.uuid4().hex[:12])
    text: str = ""
    kind: str = "functional"
    priority: str = "must"
    source: str = "user"
    acceptance: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "text": _bounded_text(self.text, 2000),
            "kind": self.kind,
            "priority": self.priority,
            "source": self.source,
            "acceptance": list(self.acceptance),
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SMLRequirement":
        return cls(
            requirement_id=str(data.get("requirement_id") or "req_" + uuid.uuid4().hex[:12]),
            text=str(data.get("text") or ""),
            kind=str(data.get("kind") or "functional"),
            priority=str(data.get("priority") or "must"),
            source=str(data.get("source") or "user"),
            acceptance=_coerce_list(data.get("acceptance")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class SMLFilePlan:
    """Language-neutral file contract used by NAILDE and other synthesizers."""
    path: str
    purpose: str = ""
    language: str = "text"
    artifact_role: str = SMLArtifactRole.SOURCE.value
    component_id: str = ""
    entrypoint: bool = False
    depends_on: List[str] = field(default_factory=list)
    acceptance: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "purpose": _bounded_text(self.purpose, 2000),
            "language": self.language,
            "artifact_role": self.artifact_role,
            "component_id": self.component_id,
            "entrypoint": bool(self.entrypoint),
            "depends_on": list(self.depends_on),
            "acceptance": list(self.acceptance),
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SMLFilePlan":
        return cls(
            path=str(data.get("path") or ""),
            purpose=str(data.get("purpose") or ""),
            language=str(data.get("language") or "text"),
            artifact_role=str(data.get("artifact_role") or data.get("role") or SMLArtifactRole.SOURCE.value),
            component_id=str(data.get("component_id") or ""),
            entrypoint=bool(data.get("entrypoint", False)),
            depends_on=_coerce_list(data.get("depends_on")),
            acceptance=_coerce_list(data.get("acceptance")),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class SMLApplicationBlueprint:
    """Formal QSML contract for arbitrary local application synthesis.

    The blueprint specifies what may be generated. It does not execute code,
    install dependencies, grant authority, or write outside the consumer's
    governed sandbox.
    """
    blueprint_id: str = field(default_factory=lambda: "bp_" + uuid.uuid4().hex[:16])
    name: str = "NAILDE Application"
    goal: str = ""
    project_kind: str = "software_project"
    languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    requirements: List[SMLRequirement] = field(default_factory=list)
    components: List[Dict[str, Any]] = field(default_factory=list)
    files: List[SMLFilePlan] = field(default_factory=list)
    dependencies: List[Dict[str, Any]] = field(default_factory=list)
    requested_capabilities: List[Dict[str, Any]] = field(default_factory=list)
    tests: List[Dict[str, Any]] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    run: Dict[str, Any] = field(default_factory=dict)
    asset_requests: List[Dict[str, Any]] = field(default_factory=list)
    phase: str = SMLSynthesisPhase.ARCHITECT.value
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blueprint_id": self.blueprint_id,
            "name": self.name,
            "goal": _bounded_text(self.goal, 20000),
            "project_kind": self.project_kind,
            "languages": list(self.languages),
            "frameworks": list(self.frameworks),
            "requirements": [x.to_dict() for x in self.requirements],
            "components": copy.deepcopy(self.components),
            "files": [x.to_dict() for x in self.files],
            "dependencies": copy.deepcopy(self.dependencies),
            "requested_capabilities": copy.deepcopy(self.requested_capabilities),
            "tests": copy.deepcopy(self.tests),
            "acceptance_criteria": list(self.acceptance_criteria),
            "constraints": copy.deepcopy(self.constraints),
            "run": copy.deepcopy(self.run),
            "asset_requests": copy.deepcopy(self.asset_requests),
            "phase": self.phase,
            "metadata": copy.deepcopy(self.metadata),
            "execution_authority": False,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SMLApplicationBlueprint":
        return cls(
            blueprint_id=str(data.get("blueprint_id") or "bp_" + uuid.uuid4().hex[:16]),
            name=str(data.get("name") or data.get("application_name") or "NAILDE Application"),
            goal=str(data.get("goal") or data.get("request") or ""),
            project_kind=str(data.get("project_kind") or "software_project"),
            languages=_coerce_list(data.get("languages")),
            frameworks=_coerce_list(data.get("frameworks")),
            requirements=[SMLRequirement.from_dict(x) for x in list(data.get("requirements") or []) if isinstance(x, Mapping)],
            components=[dict(x) for x in list(data.get("components") or []) if isinstance(x, Mapping)],
            files=[SMLFilePlan.from_dict(x) for x in list(data.get("files") or data.get("file_plan") or []) if isinstance(x, Mapping)],
            dependencies=[dict(x) if isinstance(x, Mapping) else {"name": str(x)} for x in list(data.get("dependencies") or [])],
            requested_capabilities=[dict(x) for x in list(data.get("requested_capabilities") or []) if isinstance(x, Mapping)],
            tests=[dict(x) for x in list(data.get("tests") or []) if isinstance(x, Mapping)],
            acceptance_criteria=_coerce_list(data.get("acceptance_criteria")),
            constraints=dict(data.get("constraints") or {}),
            run=dict(data.get("run") or {}),
            asset_requests=[dict(x) for x in list(data.get("asset_requests") or []) if isinstance(x, Mapping)],
            phase=str(data.get("phase") or SMLSynthesisPhase.ARCHITECT.value),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class SMLProgram:
    """Compiled QSML program produced from natural language or explicit QSML."""
    program_id: str = field(default_factory=lambda: "qsml_" + uuid.uuid4().hex[:16])
    language_version: str = QSML_LANGUAGE_VERSION
    type_system_version: str = SML_TYPE_SYSTEM_VERSION
    source: str = ""
    mission: str = MissionType.UNKNOWN.value
    variables: SMLVariableRegistry = field(default_factory=SMLVariableRegistry)
    ast: SMLASTNode = field(default_factory=lambda: SMLASTNode(kind=SMLASTKind.PROGRAM.value))
    build_spec: Dict[str, Any] = field(default_factory=dict)
    synthesis_blueprint: Optional[SMLApplicationBlueprint] = None
    required_authority: List[str] = field(default_factory=lambda: [Authority.READ.value])
    candidate_routes: List[List[str]] = field(default_factory=list)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "program_id": self.program_id,
            "language_version": self.language_version,
            "type_system_version": self.type_system_version,
            "source": _bounded_text(self.source, 20000),
            "mission": self.mission,
            "variables": self.variables.to_dict(),
            "ast": self.ast.to_dict(),
            "build_spec": copy.deepcopy(self.build_spec),
            "synthesis_blueprint": self.synthesis_blueprint.to_dict() if self.synthesis_blueprint else None,
            "required_authority": list(self.required_authority),
            "candidate_routes": copy.deepcopy(self.candidate_routes),
            "diagnostics": copy.deepcopy(self.diagnostics),
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SMLProgram":
        """Rehydrate a compiled QSML program without executing it.

        This makes QSML compiler IR persistable across the SML packet, NAILDE
        workspace, API bridge, diagnostics, and audit boundaries. Rehydration is
        structural only and never grants authority or executes a route.
        """
        blueprint_raw = data.get("synthesis_blueprint")
        return cls(
            program_id=str(data.get("program_id") or "qsml_" + uuid.uuid4().hex[:16]),
            language_version=str(data.get("language_version") or QSML_LANGUAGE_VERSION),
            type_system_version=str(data.get("type_system_version") or SML_TYPE_SYSTEM_VERSION),
            source=str(data.get("source") or ""),
            mission=str(data.get("mission") or MissionType.UNKNOWN.value),
            variables=SMLVariableRegistry.from_dict(data.get("variables") or {}),
            ast=SMLASTNode.from_dict(data.get("ast") or {"kind": SMLASTKind.PROGRAM.value}),
            build_spec=copy.deepcopy(dict(data.get("build_spec") or {})),
            synthesis_blueprint=(SMLApplicationBlueprint.from_dict(blueprint_raw) if isinstance(blueprint_raw, Mapping) else None),
            required_authority=_coerce_list(data.get("required_authority")) or [Authority.READ.value],
            candidate_routes=[list(x) for x in list(data.get("candidate_routes") or []) if isinstance(x, (list, tuple))],
            diagnostics=[dict(x) if isinstance(x, Mapping) else {"message": str(x)} for x in list(data.get("diagnostics") or [])],
            metadata=copy.deepcopy(dict(data.get("metadata") or {})),
        )


@dataclass
class SMLCompileResult:
    status: str = SMLCompileStatus.COMPILED.value
    program: Optional[SMLProgram] = None
    issues: List[SMLValidationIssue] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "program": self.program.to_dict() if self.program else None,
            "issues": [x.to_dict() for x in self.issues],
            "evidence": copy.deepcopy(self.evidence),
            "execution_authority": False,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SMLCompileResult":
        program_raw = data.get("program")
        return cls(
            status=str(data.get("status") or SMLCompileStatus.COMPILED.value),
            program=(SMLProgram.from_dict(program_raw) if isinstance(program_raw, Mapping) else None),
            issues=[SMLValidationIssue(**{
                "code": str(x.get("code") or "QSML-COMPILE"),
                "message": str(x.get("message") or ""),
                "severity": str(x.get("severity") or "ERROR"),
                "field": str(x.get("field") or ""),
                "error_class": str(x.get("error_class") or ErrorClass.PROTOCOL.value),
            }) for x in list(data.get("issues") or []) if isinstance(x, Mapping)],
            evidence=copy.deepcopy(dict(data.get("evidence") or {})),
        )

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
    language_contract: Dict[str, Any] = field(default_factory=dict)

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
            "language_contract": copy.deepcopy(self.language_contract),
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
            language_contract=dict(data.get("language_contract") or {}),
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



@dataclass
class SMLCognitiveEvent:
    """Typed event entering the persistent governed cognitive cycle."""
    event_id: str = field(default_factory=lambda: "evt_" + uuid.uuid4().hex[:16])
    event_type: str = GCOPEventType.UNKNOWN.value
    source: str = "unknown"
    timestamp: str = field(default_factory=_utc_now)
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: float = 0.5
    requires_response: bool = False
    requested_execution: bool = False
    authority_reference: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "timestamp": self.timestamp,
            "payload": copy.deepcopy(self.payload),
            "priority": float(self.priority),
            "requires_response": bool(self.requires_response),
            "requested_execution": bool(self.requested_execution),
            "authority_reference": self.authority_reference,
            "metadata": copy.deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SMLCognitiveEvent":
        return cls(
            event_id=str(data.get("event_id") or "evt_" + uuid.uuid4().hex[:16]),
            event_type=str(data.get("event_type") or GCOPEventType.UNKNOWN.value),
            source=str(data.get("source") or "unknown"),
            timestamp=str(data.get("timestamp") or _utc_now()),
            payload=dict(data.get("payload") or {}),
            priority=float(data.get("priority") or 0.5),
            requires_response=bool(data.get("requires_response", False)),
            requested_execution=bool(data.get("requested_execution", False)),
            authority_reference=str(data.get("authority_reference") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass
class SMLCognitiveContinuityState:
    """Protocol-owned structure for persistent cognitive continuity.

    The structure stores only shared state contracts.  Identity interpretation,
    cognition, mission bearing, policy judgement, and candidate generation remain
    owned by the SarahMemoryCognitive*.py organs.
    """
    state_id: str = field(default_factory=lambda: "gcop_" + uuid.uuid4().hex[:16])
    version: str = "GCOP/1.0"
    updated_at: str = field(default_factory=_utc_now)
    identity: Dict[str, Any] = field(default_factory=dict)
    mission: Dict[str, Any] = field(default_factory=dict)
    reality: Dict[str, Any] = field(default_factory=dict)
    authority: Dict[str, Any] = field(default_factory=dict)
    resources: Dict[str, Any] = field(default_factory=dict)
    tasks: Dict[str, Any] = field(default_factory=dict)
    adaptive: Dict[str, Any] = field(default_factory=dict)
    risk: Dict[str, Any] = field(default_factory=dict)
    continuity: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    audit: Dict[str, Any] = field(default_factory=dict)
    extensions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy({
            "state_id": self.state_id,
            "version": self.version,
            "updated_at": self.updated_at,
            "identity": self.identity,
            "mission": self.mission,
            "reality": self.reality,
            "authority": self.authority,
            "resources": self.resources,
            "tasks": self.tasks,
            "adaptive": self.adaptive,
            "risk": self.risk,
            "continuity": self.continuity,
            "memory": self.memory,
            "audit": self.audit,
            "extensions": self.extensions,
        })

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SMLCognitiveContinuityState":
        return cls(
            state_id=str(data.get("state_id") or "gcop_" + uuid.uuid4().hex[:16]),
            version=str(data.get("version") or "GCOP/1.0"),
            updated_at=str(data.get("updated_at") or _utc_now()),
            identity=dict(data.get("identity") or {}),
            mission=dict(data.get("mission") or {}),
            reality=dict(data.get("reality") or {}),
            authority=dict(data.get("authority") or {}),
            resources=dict(data.get("resources") or {}),
            tasks=dict(data.get("tasks") or {}),
            adaptive=dict(data.get("adaptive") or {}),
            risk=dict(data.get("risk") or {}),
            continuity=dict(data.get("continuity") or {}),
            memory=dict(data.get("memory") or {}),
            audit=dict(data.get("audit") or {}),
            extensions=dict(data.get("extensions") or {}),
        )


@dataclass
class SMLCognitiveCycleResult:
    status: str = GCOPCycleStatus.WAIT.value
    packet: Dict[str, Any] = field(default_factory=dict)
    continuity_state: Dict[str, Any] = field(default_factory=dict)
    event: Dict[str, Any] = field(default_factory=dict)
    legal_candidates: List[Dict[str, Any]] = field(default_factory=list)
    rejected_candidates: List[Dict[str, Any]] = field(default_factory=list)
    selected_candidate: Dict[str, Any] = field(default_factory=dict)
    governance: Dict[str, Any] = field(default_factory=dict)
    execution_request: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    reply_intent: Dict[str, Any] = field(default_factory=dict)
    next_wake: Dict[str, Any] = field(default_factory=dict)
    stop_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)


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
        self.organ_contracts: Dict[str, SMLOrganContract] = {}
        self.health_vectors: Dict[str, SMLHealthVector] = {}
        self.omega_registry: Dict[str, SMLOmegaTransition] = {}
        self.diagnostics_log: List[Dict[str, Any]] = []
        self.protocol_state = "Initialized"
        self.created_at = _utc_now()
        self._load_default_omega_registry()
        self.register_organ(SMLOrganMetadata(
            name=MODULE_NAME,
            category=OrganCategory.PROTOCOL.value,
            capabilities=["packet", "routing", "omega", "serialization", "diagnostics", "health", "compatibility", "negotiation", "qsml_compiler", "type_system", "variable_registry", "cognitive_ast", "operator_evaluator", "organ_contracts", "application_blueprints", "arbitrary_application_synthesis_contracts"],
            supported_missions=[m.value for m in MissionType],
            supported_omega=list(self.omega_registry.keys()),
            required_authority=[Authority.READ.value],
            priority=100,
            trust_level="core_reference",
            metadata={"role": "protocol_microkernel", "executes_actions": False, "language_version": QSML_LANGUAGE_VERSION, "type_system_version": SML_TYPE_SYSTEM_VERSION},
            language_contract={"owns_packet_fields": ["mission", "pipeline", "extensions.qsml_program"], "advisory_only": False, "side_effecting": False},
        ))
        self.register_organ_contract(SMLOrganContract(
            name=MODULE_NAME,
            accepts_types=[x.value for x in SMLDataType],
            produces_types=[SMLDataType.PACKET.value, SMLDataType.ROUTE.value, SMLDataType.PIPELINE.value, SMLDataType.STATE.value, SMLDataType.APPLICATION_BLUEPRINT.value],
            reads_packet_fields=["payload", "context", "identity", "adaptive", "knowledge", "authority", "governance"],
            owns_packet_fields=["mission", "pipeline", "current_omega", "extensions.qsml_program"],
            writes_packet_fields=["mission", "pipeline", "authority.required", "extensions.qsml_program", "extensions.qsml_program.synthesis_blueprint"],
            supported_missions=[m.value for m in MissionType],
            supported_operators=[q.value for q in QMathState],
            supported_omega=list(self.omega_registry.keys()),
            required_authority=[Authority.READ.value],
            deterministic=True,
            side_effecting=False,
            metadata={"role": "unifying_language_runtime", "does_not_replace_specialized_organs": True},
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
        if organ.language_contract:
            try:
                self.register_organ_contract({"name": organ.name, **dict(organ.language_contract)})
            except Exception as exc:
                issues.append(f"language_contract registration failed: {exc}")
        if organ.name not in self.health_vectors:
            self.health_vectors[organ.name] = SMLHealthVector(status=HealthStatus.HEALTHY.value if not issues else HealthStatus.WARNING.value, notes=issues)
        return {"status": SMLStatus.OK.value if not issues else SMLStatus.WARNING.value, "organ": organ.name, "issues": issues}

    def unregister_organ(self, name: str) -> bool:
        existed = name in self.organs
        self.organs.pop(name, None)
        self.organ_contracts.pop(name, None)
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
    # QSML / Universal Natural Programming Language core
    # ---------------------------------------------------------------------

    def register_organ_contract(self, contract: Union[SMLOrganContract, Mapping[str, Any]]) -> Dict[str, Any]:
        obj = contract if isinstance(contract, SMLOrganContract) else SMLOrganContract.from_dict(contract)
        self.organ_contracts[obj.name] = obj
        return {"ok": True, "name": obj.name, "contract": obj.to_dict(), "execution_authority": False}

    def organ_contract(self, name: str) -> Optional[SMLOrganContract]:
        return self.organ_contracts.get(str(name or ""))

    @staticmethod
    def application_blueprint_schema() -> Dict[str, Any]:
        """Return the machine-readable QSML arbitrary-application blueprint ABI."""
        return {
            "schema": "SarahMemory.qsml.application_blueprint.v0_2",
            "required": ["name", "goal", "project_kind", "requirements", "files", "constraints"],
            "file_required": ["path", "purpose", "language"],
            "max_files": 96,
            "max_requirements": 128,
            "max_components": 64,
            "max_dependencies": 64,
            "max_requested_capabilities": 64,
            "capability_policy": "requests_are_typed_requirements_not_grants",
            "path_policy": "relative_sandbox_only",
            "execution_authority": False,
            "route_definition_owner": MODULE_NAME,
            "route_activation_owner": "SarahMemoryNeuron",
        }

    @staticmethod
    def _blueprint_path_issue(path: str) -> Optional[str]:
        raw = str(path or "").replace("\\", "/").strip()
        if not raw:
            return "empty_path"
        if raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw):
            return "absolute_path_denied"
        parts = [x for x in raw.split("/") if x not in ("", ".")]
        if any(x == ".." for x in parts):
            return "parent_traversal_denied"
        if not raw.startswith("sandbox/"):
            return "path_must_start_with_sandbox"
        if len(raw) > 240:
            return "path_too_long"
        return None

    def normalize_application_blueprint(
        self,
        blueprint: Union[SMLApplicationBlueprint, Mapping[str, Any]],
        *,
        source_program: Optional[SMLProgram] = None,
    ) -> SMLApplicationBlueprint:
        """Normalize a model/tool-produced blueprint into the QSML ABI.

        This does not approve the blueprint. Validation and governance remain
        separate and no code is executed here.
        """
        bp = blueprint if isinstance(blueprint, SMLApplicationBlueprint) else SMLApplicationBlueprint.from_dict(blueprint)
        bp.name = _bounded_text(bp.name.strip() or "NAILDE Application", 160)
        bp.goal = _bounded_text(bp.goal or (source_program.source if source_program else ""), 20000)
        bp.project_kind = _normalize_token(bp.project_kind or "software_project")
        bp.languages = list(dict.fromkeys(_normalize_token(x) for x in bp.languages if str(x).strip()))[:16]
        bp.frameworks = list(dict.fromkeys(_normalize_token(x) for x in bp.frameworks if str(x).strip()))[:16]
        bp.requirements = bp.requirements[:128]
        bp.components = bp.components[:64]
        bp.dependencies = bp.dependencies[:64]
        normalized_caps: List[Dict[str, Any]] = []
        for raw_cap in bp.requested_capabilities[:64]:
            if not isinstance(raw_cap, Mapping):
                continue
            cap = dict(raw_cap)
            cap["name"] = _normalize_token(str(cap.get("name") or cap.get("capability") or ""))
            cap["reason"] = _bounded_text(str(cap.get("reason") or ""), 1000)
            cap["authority_required"] = _coerce_list(cap.get("authority_required") or cap.get("authority"))[:16]
            cap["risk"] = _normalize_token(str(cap.get("risk") or "bounded"))
            cap["granted"] = False
            cap["execution_authority"] = False
            if cap["name"]:
                normalized_caps.append(cap)
        bp.requested_capabilities = normalized_caps
        bp.tests = bp.tests[:96]
        bp.acceptance_criteria = [_bounded_text(x, 1000) for x in bp.acceptance_criteria[:128]]
        bp.asset_requests = bp.asset_requests[:64]
        bp.files = bp.files[:96]
        constraints = dict(bp.constraints or {})
        constraints.setdefault("local_first", True)
        constraints.setdefault("sandbox_only", True)
        constraints.setdefault("live_core_write", False)
        constraints.setdefault("self_approval", False)
        constraints.setdefault("shell_allowed", False)
        bp.constraints = constraints
        bp.metadata = {
            **dict(bp.metadata or {}),
            "schema": "SarahMemory.qsml.application_blueprint.v0_2",
            "normalized_at": _utc_now(),
            "execution_authority": False,
            "route_definition_owner": MODULE_NAME,
            "route_activation_owner": "SarahMemoryNeuron",
        }
        return bp

    def validate_application_blueprint(
        self,
        blueprint: Union[SMLApplicationBlueprint, Mapping[str, Any]],
        *,
        source_program: Optional[SMLProgram] = None,
        require_files: bool = True,
    ) -> Dict[str, Any]:
        """Validate a language-neutral arbitrary-application blueprint.

        This is a static language/ownership check only. It cannot grant runtime
        permission and intentionally does not install dependencies or execute code.
        """
        bp = self.normalize_application_blueprint(blueprint, source_program=source_program)
        issues: List[SMLValidationIssue] = []
        if not bp.goal.strip():
            issues.append(SMLValidationIssue(code="QSML-BP-001", message="Application blueprint goal is required.", field="goal"))
        if not bp.name.strip():
            issues.append(SMLValidationIssue(code="QSML-BP-002", message="Application blueprint name is required.", field="name"))
        if require_files and not bp.files:
            issues.append(SMLValidationIssue(code="QSML-BP-003", message="Application blueprint requires at least one file plan.", field="files"))
        seen: Set[str] = set()
        planned_paths: Set[str] = {fp.path.replace("\\", "/") for fp in bp.files if fp.path}
        entrypoints = 0
        for idx, fp in enumerate(bp.files):
            issue = self._blueprint_path_issue(fp.path)
            if issue:
                issues.append(SMLValidationIssue(code="QSML-BP-010", message=f"Invalid synthesis file path: {issue}", field=f"files[{idx}].path"))
            key = fp.path.replace("\\", "/").lower()
            if key in seen:
                issues.append(SMLValidationIssue(code="QSML-BP-011", message="Duplicate synthesis file path.", field=f"files[{idx}].path"))
            seen.add(key)
            if fp.entrypoint:
                entrypoints += 1
            if not fp.purpose.strip():
                issues.append(SMLValidationIssue(code="QSML-BP-012", message="Every file plan requires a purpose.", field=f"files[{idx}].purpose", severity="WARNING"))
            binary_ext = os.path.splitext(fp.path.lower())[1]
            if binary_ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".wav", ".mp3", ".ogg", ".flac", ".mp4", ".webm", ".mov", ".glb", ".gltf", ".blend", ".ttf", ".otf", ".woff", ".woff2", ".pdf"}:
                issues.append(SMLValidationIssue(
                    code="QSML-BP-017",
                    message="Binary artifact is not valid for the NAILDE text synthesizer. Use a text/procedural fallback or a separately governed media/asset organ contract.",
                    field=f"files[{idx}].path",
                ))
            for dep_idx, dep in enumerate(fp.depends_on[:32]):
                dep_s = str(dep or "").replace("\\", "/").strip()
                if not dep_s:
                    continue
                # File-path dependencies are enforceable. Plain component labels
                # are legal semantic references and are resolved by the consumer.
                if (dep_s.startswith("sandbox/") or "/" in dep_s) and dep_s not in planned_paths:
                    issues.append(SMLValidationIssue(
                        code="QSML-BP-015",
                        message="File dependency references a path not declared in the blueprint.",
                        field=f"files[{idx}].depends_on[{dep_idx}]",
                        severity="WARNING",
                    ))
        if bp.files and entrypoints == 0:
            issues.append(SMLValidationIssue(code="QSML-BP-013", message="No entrypoint file is declared; project may be a library or incomplete application.", field="files", severity="WARNING"))
        if entrypoints > 8:
            issues.append(SMLValidationIssue(code="QSML-BP-014", message="Excessive entrypoint count requires clarification.", field="files"))
        run_entrypoint = str((bp.run or {}).get("entrypoint") or "").replace("\\", "/").strip()
        if run_entrypoint and run_entrypoint not in planned_paths:
            issues.append(SMLValidationIssue(
                code="QSML-BP-016",
                message="Run contract entrypoint is not present in the declared file plan.",
                field="run.entrypoint",
            ))
        if source_program is not None:
            if str((source_program.metadata or {}).get("route_definition_owner") or "") != MODULE_NAME:
                issues.append(SMLValidationIssue(code="QSML-BP-040", message="QSML program route definition owner must remain SarahMemorySMLProtocol.", field="program.metadata.route_definition_owner", error_class=ErrorClass.OWNERSHIP.value))
            if str((source_program.metadata or {}).get("route_activation_owner") or "") != "SarahMemoryNeuron":
                issues.append(SMLValidationIssue(code="QSML-BP-041", message="QSML program route activation owner must remain SarahMemoryNeuron.", field="program.metadata.route_activation_owner", error_class=ErrorClass.OWNERSHIP.value))
        denied_true = [
            key for key in ("live_core_write", "self_approval")
            if bool((bp.constraints or {}).get(key))
        ]
        if denied_true:
            issues.append(SMLValidationIssue(code="QSML-BP-020", message="Blueprint requested denied authority: " + ", ".join(denied_true), field="constraints", error_class=ErrorClass.GOVERNANCE.value))
        if not bool((bp.constraints or {}).get("sandbox_only", True)):
            issues.append(SMLValidationIssue(code="QSML-BP-021", message="NAILDE arbitrary synthesis must remain sandbox-only until governed promotion.", field="constraints.sandbox_only", error_class=ErrorClass.GOVERNANCE.value))
        cap_names: Set[str] = set()
        for idx, cap in enumerate(bp.requested_capabilities):
            name = str(cap.get("name") or "").strip() if isinstance(cap, Mapping) else ""
            if not name:
                issues.append(SMLValidationIssue(code="QSML-BP-025", message="Requested capability requires a name.", field=f"requested_capabilities[{idx}]"))
                continue
            if name in cap_names:
                issues.append(SMLValidationIssue(code="QSML-BP-026", message="Duplicate requested capability.", field=f"requested_capabilities[{idx}]", severity="WARNING"))
            cap_names.add(name)
            if bool(cap.get("granted")) or bool(cap.get("execution_authority")):
                issues.append(SMLValidationIssue(code="QSML-BP-027", message="Application blueprints may request capabilities but may not grant them.", field=f"requested_capabilities[{idx}]", error_class=ErrorClass.GOVERNANCE.value))

        dep_names: Set[str] = set()
        for idx, dep in enumerate(bp.dependencies):
            name = str(dep.get("name") or "").strip() if isinstance(dep, Mapping) else ""
            if not name:
                issues.append(SMLValidationIssue(code="QSML-BP-030", message="Dependency entry requires a name.", field=f"dependencies[{idx}]", severity="WARNING"))
                continue
            norm = name.lower()
            if norm in dep_names:
                issues.append(SMLValidationIssue(code="QSML-BP-031", message="Duplicate dependency declaration.", field=f"dependencies[{idx}]", severity="WARNING"))
            dep_names.add(norm)
        external_deps = [
            dep for dep in bp.dependencies if isinstance(dep, Mapping)
            and str(dep.get("kind") or dep.get("type") or "").strip().lower() in {"external", "external_package", "package", "pip", "npm", "library"}
        ]
        if external_deps:
            manifest_names = {"requirements.txt", "pyproject.toml", "setup.cfg", "setup.py", "package.json", "package-lock.json", "pnpm-lock.yaml", "yarn.lock", "pom.xml", "build.gradle", "build.gradle.kts", "cargo.toml", "go.mod", "composer.json", "gemfile"}
            if not any(os.path.basename(path).lower() in manifest_names for path in planned_paths):
                issues.append(SMLValidationIssue(
                    code="QSML-BP-032",
                    message="External dependencies require a declared dependency manifest file; QSML will not invent or auto-install dependency versions.",
                    field="dependencies",
                ))
        for idx, asset in enumerate(bp.asset_requests):
            if not isinstance(asset, Mapping):
                continue
            required = bool(asset.get("required", False))
            path = str(asset.get("path") or asset.get("target_path") or "").replace("\\", "/").strip()
            if required and (not path or path not in planned_paths):
                issues.append(SMLValidationIssue(
                    code="QSML-BP-033",
                    message="Required asset request must resolve to a declared synthesizable file or be delegated to a separately governed asset organ before the application can be READY.",
                    field=f"asset_requests[{idx}]",
                ))
        errors = [x for x in issues if str(x.severity).upper() == "ERROR"]
        status = SMLStatus.OK.value if not errors else SMLStatus.ERROR.value
        if not errors and issues:
            status = SMLStatus.WARNING.value
        return {
            "ok": not errors,
            "status": status,
            "schema": "SarahMemory.qsml.application_blueprint.validation.v0_2",
            "blueprint": bp.to_dict(),
            "issues": [x.to_dict() for x in issues],
            "execution_authority": False,
        }

    def compile_application_blueprint(
        self,
        blueprint: Union[SMLApplicationBlueprint, Mapping[str, Any]],
        *,
        source_program: Optional[SMLProgram] = None,
    ) -> Dict[str, Any]:
        """Validate and bind a synthesis blueprint to a compiled QSML program."""
        validation = self.validate_application_blueprint(blueprint, source_program=source_program, require_files=True)
        if source_program is not None and validation.get("ok"):
            source_program.synthesis_blueprint = SMLApplicationBlueprint.from_dict(validation["blueprint"])
            source_program.metadata["synthesis_phase"] = SMLSynthesisPhase.ARCHITECT.value
            source_program.metadata["arbitrary_application_synthesis"] = True
        return validation

    def _infer_data_type(self, value: Any, *, source_text: str = "") -> str:
        text = str(source_text or "").strip()
        if value is None:
            return SMLDataType.NULL.value
        if isinstance(value, bool):
            return SMLDataType.BOOL.value
        if isinstance(value, int):
            return SMLDataType.UINT.value if value >= 0 else SMLDataType.INT.value
        if isinstance(value, float):
            return SMLDataType.FLOAT.value
        if isinstance(value, complex):
            return SMLDataType.COMPLEX.value
        if isinstance(value, (bytes, bytearray)):
            return SMLDataType.BYTES.value
        if isinstance(value, Mapping):
            return SMLDataType.MAP.value
        if isinstance(value, set):
            return SMLDataType.SET.value
        if isinstance(value, (list, tuple)):
            if value and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
                return SMLDataType.VECTOR.value
            return SMLDataType.LIST.value
        if isinstance(value, str):
            candidate = text or value
            compact = candidate.strip().lower().replace("_", "")
            if re.fullmatch(r"0b[01]+", compact) or (re.fullmatch(r"[01]{4,256}", compact) and len(compact) >= 4):
                return SMLDataType.BINARY.value
            if re.fullmatch(r"0x[0-9a-f]+", compact):
                return SMLDataType.HEX.value
            if re.fullmatch(r"0o[0-7]+", compact):
                return SMLDataType.OCTAL.value
            if re.fullmatch(r"https?://[^\s]+", candidate.strip(), flags=re.I):
                return SMLDataType.URL.value
            if re.match(r"^(?:[A-Za-z]:[\\/]|/|\.\.?[\\/])", candidate.strip()):
                return SMLDataType.PATH.value
            return SMLDataType.STR.value
        return SMLDataType.OBJECT.value

    def _collect_language_evidence(self, text: str, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Collect bounded semantic evidence from existing language organs.

        These organs remain owners of their observations. QSML consumes their
        structured evidence; it does not replace their internal algorithms.
        """
        evidence: Dict[str, Any] = {}
        ctx = dict(context or {})
        collectors = [
            ("pretoken", "SarahMemoryPreTokenAnalyzer", "analyze_text", (text, ctx)),
            ("identity_layer", "SarahMemoryCognitiveIdentityLayer", "build_tri_layer_input_packet", (text, ctx)),
            ("advcu_semantic", "SarahMemoryAdvCU", "build_semantic_packet", (text, ctx, {})),
        ]
        for key, module_name, fn_name, args in collectors:
            try:
                module = __import__(module_name)
                fn = getattr(module, fn_name, None)
                if callable(fn):
                    value = fn(*args)
                    if hasattr(value, "to_dict") and callable(value.to_dict):
                        value = value.to_dict()
                    elif hasattr(value, "__dict__") and not isinstance(value, Mapping):
                        value = dict(value.__dict__)
                    if isinstance(value, Mapping):
                        evidence[key] = copy.deepcopy(dict(value))
            except Exception as exc:
                evidence[key + "_unavailable"] = _redact_sensitive_text(str(exc))[:300]
        return evidence

    @staticmethod
    def _program_name_from_text(text: str) -> str:
        raw = str(text or "").strip()
        quoted = re.search(r"[\"“']([^\"”']{2,80})[\"”']", raw)
        if quoted:
            return quoted.group(1).strip()
        named = re.search(r"\b(?:called|named|titled)\s+([A-Za-z0-9][A-Za-z0-9 _.-]{1,80})", raw, flags=re.I)
        if named:
            return re.split(r"\b(?:with|that|which|and|using|for)\b", named.group(1), maxsplit=1, flags=re.I)[0].strip(" .,-")
        cleaned = re.sub(r"\b(?:please|can you|could you|would you|i want you to|i need you to)\b", " ", raw, flags=re.I)
        cleaned = re.sub(r"\b(?:build|create|make|write|develop|design|generate|a|an|the)\b", " ", cleaned, flags=re.I)
        cleaned = re.split(r"\b(?:with|including|featuring|that has|which has|using)\b", cleaned, maxsplit=1, flags=re.I)[0]
        cleaned = re.sub(r"^\s*local\s+", "", cleaned, flags=re.I)
        words = re.findall(r"[A-Za-z0-9]+", cleaned)[:7]
        return " ".join(words).strip() or "NAILDE Application"

    @staticmethod
    def _infer_project_kind(text: str) -> Tuple[str, str]:
        """Infer only a broad project family.

        QSML/0.2 intentionally does not select prompt-specific game/application
        templates here. Detailed architecture is delegated to the governed
        arbitrary-application synthesis blueprint so natural language remains
        open-ended instead of collapsing into canned outputs.
        """
        t = str(text or "").lower()
        if re.search(r"\b(game|gaming)\b", t):
            return "game", ""
        if re.search(r"\b(website|web app|web application|dashboard|browser app|html)\b", t):
            return "web_application", ""
        if re.search(r"\b(command line|command-line|cli|terminal app|console app)\b", t):
            return "cli_application", ""
        if re.search(r"\b(api service|rest api|service|server|backend)\b", t):
            return "service", ""
        if re.search(r"\b(mobile app|mobile application|android app|ios app)\b", t):
            return "mobile_application", ""
        if re.search(r"\b(library|sdk|software development kit|package)\b", t):
            return "library", ""
        if re.search(r"\b(data app|data application|analytics dashboard|data dashboard)\b", t):
            return "data_application", ""
        if re.search(r"\b(embedded app|embedded application|firmware tool|robotics application)\b", t):
            return "embedded_application", ""
        if re.search(r"\b(addon|add-on|plugin|extension)\b", t):
            return "addon", ""
        if re.search(r"\b(automation|automate|script)\b", t):
            return "automation_tool", ""
        if re.search(r"\b(application| app\b|desktop|gui|window|software|program)\b", " " + t):
            return "desktop_application", ""
        return "software_project", ""

    @staticmethod
    def _infer_program_languages(text: str, project_kind: str) -> Tuple[List[str], str]:
        t = str(text or "").lower()
        aliases = [
            ("python", r"\bpython\b"), ("typescript", r"\btypescript\b|\btsx\b"),
            ("javascript", r"\bjavascript\b|\bnode(?:\.js)?\b"), ("html", r"\bhtml\b"),
            ("css", r"\bcss\b"), ("sql", r"\bsql\b"), ("csharp", r"\bc#\b|\bc sharp\b"),
            ("cpp", r"\bc\+\+\b"), ("rust", r"\brust\b"), ("java", r"\bjava\b"),
        ]
        out = [name for name, pat in aliases if re.search(pat, t)]
        if out:
            return out, "explicit"
        # A universal language compiler must not silently choose the application's
        # implementation language. NAILDE's architecture planner selects an
        # appropriate local implementation from the requirements/capabilities.
        return [], "architecture_selection_required"

    @staticmethod
    def _extract_program_features(text: str) -> List[str]:
        raw = str(text or "")
        t = raw.lower()
        capability_terms = [
            "button", "buttons", "menu", "toolbar", "form", "forms", "input", "text box", "editor",
            "table", "spreadsheet", "database", "sqlite", "search", "filter", "sort", "save", "load",
            "import", "export", "json", "csv", "image", "images", "audio", "music", "video", "avatar",
            "controller", "gamepad", "keyboard", "mouse", "touch", "network", "api", "web", "login",
            "settings", "preferences", "timer", "score", "high score", "levels", "inventory", "map",
            "physics", "collision", "animation", "drag and drop", "tabs", "chart", "graphs", "report",
        ]
        found: List[str] = []
        for term in sorted(capability_terms, key=len, reverse=True):
            if re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", t):
                canonical = {"buttons": "button", "forms": "form", "images": "image", "graphs": "chart"}.get(term, term)
                if canonical not in found:
                    found.append(canonical)
        for m in re.finditer(r"\b(?:with|include|including|needs?|must have|that has|featuring)\s+([^.;]{2,180})", raw, flags=re.I):
            fragment = m.group(1)
            for part in re.split(r",|\band\b|\bplus\b", fragment, flags=re.I):
                part = re.sub(r"\s+", " ", part).strip(" .,-")
                if 2 <= len(part) <= 80 and part.lower() not in {x.lower() for x in found}:
                    found.append(part)
                if len(found) >= 24:
                    break
            if len(found) >= 24:
                break
        return found[:24]

    @staticmethod
    def _extract_requested_capabilities(text: str) -> List[Dict[str, Any]]:
        """Infer typed capability requests without granting runtime authority."""
        t = str(text or "").lower()
        rules = [
            ("network", r"\b(network|internet|http|https|socket|websocket|online|remote|rest api|graphql|web service)\b", [Authority.NETWORK.value], "elevated"),
            ("external_api", r"\b(api|rest api|graphql|web service|remote service)\b", [Authority.NETWORK.value], "elevated"),
            ("filesystem", r"\b(file|files|folder|directory|save|export|import|document|csv|json)\b", [Authority.FILESYSTEM.value], "bounded"),
            ("database", r"\b(database|sqlite|sql|postgres|mysql|storage)\b", [Authority.READ.value, Authority.WRITE.value], "bounded"),
            ("camera", r"\b(camera|webcam|vision|video capture)\b", [Authority.READ.value], "restricted"),
            ("microphone", r"\b(microphone|mic|audio input|record audio)\b", [Authority.READ.value], "restricted"),
            ("audio_output", r"\b(audio|sound|music|speaker|tts|voice)\b", [Authority.READ.value], "bounded"),
            ("game_controller", r"\b(gamepad|controller|joystick)\b", [Authority.READ.value], "bounded"),
            ("gpu_acceleration", r"\b(gpu|cuda|graphics acceleration|hardware acceleration)\b", [Authority.READ.value], "bounded"),
            ("process_execution", r"\b(run external command|execute command|spawn process|subprocess|powershell|cmd\.exe|shell command|terminal emulator)\b", [Authority.EXECUTE.value], "restricted"),
            ("local_model", r"\b(local model|local llm|ollama|transformers)\b", [Authority.READ.value], "bounded"),
        ]
        no_network = bool(re.search(r"\b(no network|without network|no internet|without internet|offline only|local only|local-only)\b", t))
        no_shell = bool(re.search(r"\b(no shell|without shell|no subprocess|do not execute commands)\b", t))
        out: List[Dict[str, Any]] = []
        for name, pattern, authority, risk in rules:
            if not re.search(pattern, t):
                continue
            if no_network and name in {"network", "external_api"}:
                continue
            if no_shell and name == "process_execution":
                continue
            out.append({"name": name, "reason": "Derived from explicit natural-language capability requirements.", "authority_required": list(authority), "risk": risk, "granted": False, "execution_authority": False})
        return out

    @staticmethod
    def _extract_program_constraints(text: str) -> Dict[str, Any]:
        t = str(text or "").lower()
        local_only = bool(re.search(r"\b(local only|local-only|offline|no cloud|without cloud)\b", t))
        no_network = bool(re.search(r"\b(no network|without network|no internet|without internet|offline only)\b", t))
        sandbox = not bool(re.search(r"\b(live core|production core|apply directly to core)\b", t))
        platforms = []
        for name, pat in (("windows", r"\bwindows\b"), ("linux", r"\blinux\b"), ("macos", r"\bmac(?:os)?\b"), ("web", r"\bweb\b|\bbrowser\b")):
            if re.search(pat, t):
                platforms.append(name)
        if not platforms:
            platforms = ["local_desktop"]
        return {
            "local_first": True,
            "local_only": local_only,
            "network_allowed": not no_network and not local_only,
            "sandbox_only": sandbox,
            "shell_allowed": False,
            "live_core_write": False,
            "self_approval": False,
            "target_platforms": platforms,
        }

    def _build_program_spec(self, text: str, mission: str, context: Optional[Mapping[str, Any]], evidence: Mapping[str, Any]) -> Dict[str, Any]:
        project_kind, genre = self._infer_project_kind(text)
        languages, language_source = self._infer_program_languages(text, project_kind)
        features = self._extract_program_features(text)
        requested_capabilities = self._extract_requested_capabilities(text)
        constraints = self._extract_program_constraints(text)
        ctx_map = dict(context or {})
        nested_ctx = ctx_map.get("api_context") if isinstance(ctx_map.get("api_context"), Mapping) else {}
        target = str(ctx_map.get("target") or ctx_map.get("surface") or nested_ctx.get("target") or nested_ctx.get("surface") or "").strip().lower()
        caller_hint = str(ctx_map.get("caller") or nested_ctx.get("caller") or "").strip().lower()
        if not target and "nailde" in caller_hint:
            target = "nailde"
        if target in {"nailde", "sarahmemorynailde", "sandbox"}:
            constraints["sandbox_only"] = True
            requested_names = {str(x.get("name") or "") for x in requested_capabilities if isinstance(x, Mapping)}
            constraints["network_allowed"] = bool(constraints.get("network_allowed") and requested_names.intersection({"network", "external_api"}))
            constraints["shell_allowed"] = bool("process_execution" in requested_names)
            # These refer to the synthesizer itself, not the future generated app.
            constraints["synthesis_network_access"] = False
            constraints["synthesis_shell_access"] = False
        frameworks = []
        for name in ("tkinter", "pygame", "react", "flask", "fastapi", "django", "qt", "pyqt"):
            if re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", str(text or "").lower()):
                frameworks.append(name)
        return {
            "schema": "SarahMemory.qsml.program_spec.v0_2",
            "application_name": self._program_name_from_text(text),
            "project_kind": project_kind,
            "domain_hint": genre,
            "languages": languages,
            "language_selection": language_source,
            "frameworks": frameworks,
            "features": features,
            "requested_capabilities": requested_capabilities,
            "constraints": constraints,
            "target_surface": "NAILDE" if target in {"nailde", "sarahmemorynailde", "sandbox"} else (target or "generic"),
            "mission": mission,
            "compiler_policy": {
                "natural_language_is_source": True,
                "hardcode_rails_not_thoughts": True,
                "arbitrary_application_synthesis": True,
                "architecture_expansion_required": True,
                "prompt_specific_template_selection": False,
                "neuron_activation_owner": "SarahMemoryNeuron",
                "execution_authority": False,
            },
            "synthesis_contract": {
                "schema": "SarahMemory.qsml.application_blueprint.v0_2",
                "phase": SMLSynthesisPhase.COMPILE.value,
                "planner": "NAILDE_local_synthesis_model",
                "generator": "NAILDE_universal_file_synthesizer",
                "validator": "QSML+NAILDE+Compare",
                "repair_loop_bounded": True,
                "max_files": 96,
                "max_repair_rounds": 3,
                "sandbox_only": True,
                "execution_authority": False,
            },
            "evidence_sources": sorted(k for k in evidence.keys() if not k.endswith("_unavailable")),
        }

    def _candidate_routes_for_program(self, mission: str, build_spec: Mapping[str, Any]) -> List[List[str]]:
        target = str(build_spec.get("target_surface") or "").lower()
        if mission == MissionType.PROGRAMMING.value and target == "nailde":
            return [[
                "SarahMemoryPreTokenAnalyzer",
                "SarahMemoryCognitiveIdentityLayer",
                "SarahMemoryAdvCU",
                MODULE_NAME,
                "SarahMemoryNeuron",
                "SarahMemoryNAILDE",
                "SarahMemoryCompare",
                "SarahMemoryLedger",
            ]]
        return [["SarahMemoryPreTokenAnalyzer", "SarahMemoryAdvCU", MODULE_NAME, "SarahMemoryNeuron", "SarahMemoryCompare"]]

    def compile_natural_language(
        self,
        text: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
        packet: Optional[SMLPacket] = None,
        collect_external_evidence: bool = True,
        target: str = "",
    ) -> SMLCompileResult:
        """Compile natural language into typed QSML IR.

        This compiler is non-executing. It creates typed symbols, a cognitive AST,
        legal route candidates, constraints, and organ-interface requirements.
        Neuron remains the owner of runtime activation/weighting among legal routes.
        """
        source = str(text or "").strip()
        if not source:
            return SMLCompileResult(
                status=SMLCompileStatus.NEEDS_CLARIFICATION.value,
                issues=[SMLValidationIssue(code="QSML1001", message="Natural language source is empty.", severity="ERROR", field="source")],
            )
        ctx = dict(context or {})
        if target:
            ctx["target"] = target
        mission = str((packet.mission or {}).get("primary") or MissionType.UNKNOWN.value) if packet else self._classify_text_to_mission(source)[0]
        evidence = self._collect_language_evidence(source, ctx) if collect_external_evidence else {}
        build_spec = self._build_program_spec(source, mission, ctx, evidence)
        # NAILDE is a programming/development surface. Once the compiler has
        # deterministically identified an application project contract for that
        # target, the formal QSML mission must be Programming even when the
        # conversational mission classifier was too broad (for example, a plain
        # "Create a local ... app" request classified as GeneralKnowledge).
        # This is a target/capability rule, not a prompt-specific answer rule.
        target_surface = str(build_spec.get("target_surface") or ctx.get("target") or "").strip().lower()
        project_kind = str(build_spec.get("project_kind") or "software_project").strip().lower()
        if target_surface == "nailde" and project_kind in {
            "software_project", "desktop_application", "cli_application",
            "web_application", "game", "addon", "service", "library",
            "automation", "automation_tool", "data_application", "mobile_application",
            "embedded_application",
        }:
            mission = MissionType.PROGRAMMING.value
            build_spec["mission"] = mission
        registry = SMLVariableRegistry()

        def define(name: str, value: Any, semantic: SMLSemanticType, *, owner: str = MODULE_NAME, confidence: float = 1.0, mutable: SMLMutability = SMLMutability.IMMUTABLE, data_type: Optional[SMLDataType] = None) -> None:
            registry.define(SMLVariable(
                name=name,
                value=copy.deepcopy(value),
                data_type=(data_type.value if isinstance(data_type, SMLDataType) else self._infer_data_type(value, source_text=str(value))),
                semantic_type=semantic.value,
                scope=SMLScope.MISSION.value,
                owner=owner,
                authority=[Authority.READ.value],
                confidence=max(0.0, min(1.0, float(confidence))),
                mutability=mutable.value,
                source_text=source if name == "request" else str(value),
                source="natural_language_compiler",
                validation_state="COMPILED",
            ))

        define("request", source, SMLSemanticType.USER_REQUEST)
        define("mission", mission, SMLSemanticType.ACTION, data_type=SMLDataType.MISSION)
        define("application_name", build_spec.get("application_name"), SMLSemanticType.APPLICATION_NAME)
        define("project_kind", build_spec.get("project_kind"), SMLSemanticType.PROJECT_KIND)
        define("languages", build_spec.get("languages"), SMLSemanticType.LANGUAGE)
        define("features", build_spec.get("features"), SMLSemanticType.FEATURE, mutable=SMLMutability.APPEND_ONLY)
        define("requested_capabilities", build_spec.get("requested_capabilities"), SMLSemanticType.CAPABILITY, mutable=SMLMutability.APPEND_ONLY)
        define("constraints", build_spec.get("constraints"), SMLSemanticType.CONSTRAINT)
        define("target_surface", build_spec.get("target_surface"), SMLSemanticType.TARGET_PLATFORM)

        root = SMLASTNode(
            kind=SMLASTKind.PROGRAM.value,
            value=build_spec.get("application_name"),
            data_type=SMLDataType.OBJECT.value,
            semantic_type=SMLSemanticType.USER_REQUEST.value,
            attributes={"mission": mission, "language_version": QSML_LANGUAGE_VERSION},
        )
        root.children.append(SMLASTNode(kind=SMLASTKind.MISSION.value, value=mission, data_type=SMLDataType.MISSION.value, semantic_type=SMLSemanticType.ACTION.value))
        root.children.append(SMLASTNode(kind=SMLASTKind.PROJECT.value, value=build_spec.get("project_kind"), semantic_type=SMLSemanticType.PROJECT_KIND.value))
        for language in build_spec.get("languages") or []:
            root.children.append(SMLASTNode(kind=SMLASTKind.REQUIREMENT.value, value=language, semantic_type=SMLSemanticType.LANGUAGE.value))
        for feature in build_spec.get("features") or []:
            root.children.append(SMLASTNode(kind=SMLASTKind.FEATURE.value, value=feature, semantic_type=SMLSemanticType.FEATURE.value))
        for capability in build_spec.get("requested_capabilities") or []:
            root.children.append(SMLASTNode(kind=SMLASTKind.REQUIREMENT.value, value=copy.deepcopy(capability), semantic_type=SMLSemanticType.CAPABILITY.value))
        for key, value in dict(build_spec.get("constraints") or {}).items():
            root.children.append(SMLASTNode(kind=SMLASTKind.CONSTRAINT.value, value={key: value}, semantic_type=SMLSemanticType.CONSTRAINT.value))

        qmath = self._qmath_state(source, packet=packet)
        for op, reasons in dict(qmath.get("states") or {}).items():
            if reasons:
                root.children.append(SMLASTNode(kind=SMLASTKind.OPERATOR.value, operator=op, value=list(reasons), semantic_type=SMLSemanticType.ACTION.value))

        required_authority = self._required_authority_for_mission(mission, packet)
        base_requirements = [
            SMLRequirement(text=source, kind="goal", priority="must", source="user", acceptance=[])
        ]
        for feature in list(build_spec.get("features") or [])[:48]:
            base_requirements.append(SMLRequirement(text=str(feature), kind="feature", priority="should", source="compiler_hint", acceptance=[]))
        synthesis_blueprint = SMLApplicationBlueprint(
            name=str(build_spec.get("application_name") or "NAILDE Application"),
            goal=source,
            project_kind=str(build_spec.get("project_kind") or "software_project"),
            languages=list(build_spec.get("languages") or []),
            frameworks=list(build_spec.get("frameworks") or []),
            requirements=base_requirements,
            files=[],
            requested_capabilities=copy.deepcopy(list(build_spec.get("requested_capabilities") or [])),
            constraints=dict(build_spec.get("constraints") or {}),
            phase=SMLSynthesisPhase.COMPILE.value,
            metadata={
                "requires_architecture_expansion": True,
                "source": "natural_language_compiler",
                "execution_authority": False,
            },
        )
        program = SMLProgram(
            source=source,
            mission=mission,
            variables=registry,
            ast=root,
            build_spec=build_spec,
            synthesis_blueprint=synthesis_blueprint,
            required_authority=required_authority,
            candidate_routes=self._candidate_routes_for_program(mission, build_spec),
            diagnostics=[],
            metadata={
                "compiled_at": _utc_now(),
                "execution_authority": False,
                "route_definition_owner": MODULE_NAME,
                "route_activation_owner": "SarahMemoryNeuron",
                "specialized_organs_retain_domain_ownership": True,
            },
        )
        return SMLCompileResult(status=SMLCompileStatus.COMPILED.value, program=program, evidence=evidence)

    def evaluate_qmath_ast(self, node: Union[SMLASTNode, Mapping[str, Any]], environment: Optional[Mapping[str, Any]] = None, *, max_iterations: int = 8) -> Dict[str, Any]:
        """Evaluate pure/bounded Q-Math semantics without domain side effects."""
        ast_node = node if isinstance(node, SMLASTNode) else SMLASTNode.from_dict(node)
        env = dict(environment or {})
        if ast_node.kind == SMLASTKind.LITERAL.value:
            return {"ok": True, "value": copy.deepcopy(ast_node.value), "execution_authority": False}
        if ast_node.kind == SMLASTKind.IDENTIFIER.value:
            return {"ok": True, "value": copy.deepcopy(env.get(str(ast_node.value))), "execution_authority": False}
        if ast_node.kind != SMLASTKind.OPERATOR.value:
            values = [self.evaluate_qmath_ast(child, env, max_iterations=max_iterations).get("value") for child in ast_node.children]
            return {"ok": True, "value": values if ast_node.children else copy.deepcopy(ast_node.value), "execution_authority": False}
        op = str(ast_node.operator or "").upper()
        vals = [self.evaluate_qmath_ast(child, env, max_iterations=max_iterations).get("value") for child in ast_node.children]
        if op == QMathState.AND.value:
            value = all(bool(v) for v in vals)
        elif op == QMathState.OR.value:
            value = [v for v in vals if v not in (None, False, "", [], {})]
        elif op == QMathState.NOT.value:
            value = not bool(vals[0]) if vals else True
        elif op == QMathState.SAME.value:
            value = len({_stable_json(v) for v in vals}) <= 1 if vals else False
        elif op == QMathState.NEITHER.value:
            value = not any(bool(v) for v in vals)
        elif op in {QMathState.IF.value, QMathState.WHEN.value}:
            cond = bool(vals[0]) if vals else False
            value = vals[1] if cond and len(vals) > 1 else (vals[2] if len(vals) > 2 else None)
        elif op == QMathState.ELSE.value:
            value = vals[0] if vals else None
        elif op == QMathState.WHILE.value:
            # Declarative bounded loop result only; never executes a domain action.
            iterations = min(max(0, int(ast_node.attributes.get("iterations", 0) or 0)), max(1, int(max_iterations)))
            value = {"iterations": iterations, "stop": SMLStopCondition.SUCCESS_STOP.value if iterations < max_iterations else SMLStopCondition.RESOURCE_STOP.value}
        else:
            return {"ok": False, "error": "unsupported_qmath_operator", "operator": op, "execution_authority": False}
        return {"ok": True, "operator": op, "value": value, "execution_authority": False}

    def attach_compiled_program(self, packet: SMLPacket, result: SMLCompileResult) -> SMLPacket:
        if result.program is not None:
            packet.extensions["qsml_program"] = result.program.to_dict()
            packet.metadata["qsml"] = {
                "language_version": QSML_LANGUAGE_VERSION,
                "type_system_version": SML_TYPE_SYSTEM_VERSION,
                "compile_status": result.status,
                "route_activation_owner": "SarahMemoryNeuron",
            }
            packet.add_history(MODULE_NAME, "compile_qsml", "Ω006", result.program.program_id)
        return packet

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
            source_text = raw_request or str(payload_dict.get("raw_request") or "")
            self.apply_cognitive_grammar(pkt, text=source_text)
            try:
                compiled = self.compile_natural_language(source_text, context=context or {}, packet=pkt, collect_external_evidence=False)
                self.attach_compiled_program(pkt, compiled)
            except Exception as exc:
                pkt.diagnostics.setdefault("qsml_compile", {})["error"] = _redact_sensitive_text(str(exc))[:500]
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
            MissionType.PROGRAMMING.value: ["code", "python", "script", "function", "class", "bug", "compile", "repo", "patch", "build", "application", " app ", "software", "game", "website", "web app", "program", "addon", "plugin"],
            MissionType.FILESYSTEM.value: ["file", "folder", "directory", "delete", "rename", "move", "copy", "zip", "extract"],
            MissionType.RESEARCH.value: ["research", "study", "paper", "source", "citation", "find", "look up"],
            MissionType.PLANNING.value: ["plan", "roadmap", "steps", "schedule", "architecture"],
            MissionType.MEMORY.value: ["remember", "memory", "save this", "forget"],
            MissionType.SECURITY.value: ["security", "firewall", "permission", "authority", "risk", "safe"],
            MissionType.DIAGNOSTICS.value: ["diagnostic", "health", "test", "validate", "verify", "smoke"],
            MissionType.REPAIR.value: ["repair", "fix", "recover", "rollback"],
            MissionType.EXECUTION.value: ["run", "execute", "launch", "start", "open", "shutdown"],
            MissionType.NETWORK.value: ["network", "internet", "api", "http", "web", "sarahnet", "sml-rt", "xr", "vr", "augmented reality", "virtual reality", "world fabric", "region", "authority lease"],
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
        qsml_program = packet.extensions.get("qsml_program") if isinstance(packet.extensions, dict) else None
        qsml_spec = (qsml_program or {}).get("build_spec") if isinstance(qsml_program, dict) else {}
        qsml_target = str((qsml_spec or {}).get("target_surface") or "").lower()
        if mission == MissionType.PROGRAMMING.value and qsml_target == "nailde":
            pipeline = ["SarahMemoryPreTokenAnalyzer", "SarahMemoryCognitiveIdentityLayer", "SarahMemoryAdvCU", MODULE_NAME, "SarahMemoryNeuron", "SarahMemoryNAILDE", "SarahMemoryCompare", "SarahMemoryLedger"]
            reasons.append("QSML compiled a NAILDE sandbox programming route; Neuron retains activation/weighting ownership.")
        elif not candidates:
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
        """Return machine-readable self-state; Reply owns natural-language presentation."""
        telemetry = dict(telemetry or {})
        health = self.global_health()
        scores = self._affect_scores_from_packet(packet)
        confidence = float(packet.confidence or 0.0) if isinstance(packet, SMLPacket) else 0.0
        return {
            "ok": False,
            "answer": None,
            "mission": MissionType.SELF_STATE.value,
            "source": "sml_structured_self_state",
            "confidence": max(confidence, 0.0),
            "structured_state": {
                "affect_scores": scores,
                "mission": (packet.mission or {}).get("primary") if isinstance(packet, SMLPacket) else MissionType.UNKNOWN.value,
                "governance": (packet.governance or {}).get("decision") if isinstance(packet, SMLPacket) else GovernanceDecision.PENDING.value,
                "health": health,
                "telemetry": telemetry,
            },
            "presentation_owner": "SarahMemoryReply",
            "reply_ready": False,
            "reason": "structured_state_requires_presentation_organ",
            "subjective_claim": False,
            "execution_allowed": False,
            "sources_consulted": ["SML Packet", "Adaptive State", "Diagnostics", "Health", "Governance"],
        }

    def _build_capability_answer(self, packet: Optional[SMLPacket] = None) -> Dict[str, Any]:
        """Return structured capability registry; Reply owns outward wording."""
        categories: Dict[str, int] = {}
        for organ in self.organs.values():
            categories[organ.category] = categories.get(organ.category, 0) + 1
        return {
            "ok": False,
            "answer": None,
            "mission": MissionType.CAPABILITY.value,
            "source": "sml_structured_capability_registry",
            "confidence": 0.78,
            "structured_state": {
                "categories": categories,
                "organs": {name: organ.to_dict() for name, organ in sorted(self.organs.items())},
                "organ_contracts": {name: contract.to_dict() for name, contract in sorted(self.organ_contracts.items())},
                "safe_readonly_missions": sorted(self.SAFE_READONLY_MISSIONS),
            },
            "presentation_owner": "SarahMemoryReply",
            "reply_ready": False,
            "reason": "structured_capability_state_requires_presentation_organ",
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
        hardware. It classifies the request and returns structured self-state/capability
        data or a source plan. SarahMemoryReply owns natural-language presentation; SML
        never becomes a canned answer engine.
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

    def _packet_path_value(self, packet: SMLPacket, path: str) -> Any:
        current: Any = packet
        for part in str(path or "").split("."):
            if not part:
                continue
            if isinstance(current, SMLPacket):
                if not hasattr(current, part):
                    return None
                current = getattr(current, part)
            elif isinstance(current, Mapping):
                current = current.get(part)
            else:
                return None
        return current

    def _transition_validation_failures(self, packet: SMLPacket, transition: SMLOmegaTransition) -> List[str]:
        failures: List[str] = []
        for rule in transition.validation_rules:
            value = self._packet_path_value(packet, rule)
            if value in (None, "", [], {}):
                failures.append(str(rule))
        return failures

    def transition_packet(self, packet: SMLPacket, omega_id: str, *, organ: str = MODULE_NAME, note: str = "", mutate: Optional[Mapping[str, Any]] = None) -> SMLPacket:
        if omega_id not in self.omega_registry:
            packet.diagnostics.setdefault("transition_errors", []).append({"omega": omega_id, "error": "unknown_transition"})
            packet.cognitive_state = CognitiveState.FAILED.value
            packet.add_ledger_entry(omega_id, organ, GovernanceDecision.DENIED.value, "Unknown Ω transition")
            packet.seal()
            return packet
        transition = self.omega_registry[omega_id]
        if transition.input_states and packet.cognitive_state not in set(transition.input_states):
            packet.diagnostics.setdefault("transition_errors", []).append({
                "omega": omega_id, "error": "invalid_input_state", "state": packet.cognitive_state, "allowed": list(transition.input_states)
            })
            packet.add_history(organ, "transition_denied", omega_id, "invalid input state")
            packet.add_ledger_entry(omega_id, organ, GovernanceDecision.DENIED.value, "Transition denied: invalid input state")
            packet.seal()
            return packet
        mission = str((packet.mission or {}).get("primary") or MissionType.UNKNOWN.value)
        if transition.compatible_missions and mission not in set(transition.compatible_missions):
            packet.diagnostics.setdefault("transition_errors", []).append({
                "omega": omega_id, "error": "mission_incompatible", "mission": mission, "compatible": list(transition.compatible_missions)
            })
            packet.add_history(organ, "transition_denied", omega_id, "mission incompatible")
            packet.add_ledger_entry(omega_id, organ, GovernanceDecision.DENIED.value, "Transition denied: mission incompatible")
            packet.seal()
            return packet
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
        validation_failures = self._transition_validation_failures(packet, transition)
        if validation_failures:
            packet.diagnostics.setdefault("transition_errors", []).append({"omega": omega_id, "error": "validation_rule_failed", "rules": validation_failures})
            packet.add_history(organ, "transition_denied", omega_id, "validation rules failed")
            packet.add_ledger_entry(omega_id, organ, GovernanceDecision.DENIED.value, "Transition denied: validation rules failed")
            packet.seal()
            return packet
        packet.current_omega = omega_id
        packet.cognitive_state = transition.output_state
        packet.add_history(organ, "transition", omega_id, note or transition.name)
        packet.add_ledger_entry(omega_id, organ, str(packet.governance.get("decision", GovernanceDecision.PENDING.value)), note or transition.name)
        packet.seal()
        return packet

    @staticmethod
    def _contract_field_allowed(contract: SMLOrganContract, field_name: str) -> bool:
        allowed = list(contract.owns_packet_fields) + list(contract.writes_packet_fields)
        if not allowed:
            return False
        field_name = str(field_name or "")
        for rule in allowed:
            rule = str(rule or "")
            if field_name == rule or field_name.startswith(rule + ".") or rule.startswith(field_name + "."):
                return True
        return False

    def _safe_mutate_packet(self, packet: SMLPacket, mutate: Mapping[str, Any], *, organ: str) -> None:
        protected = set(SMLPacket.IMMUTABLE_FIELDS)
        contract = self.organ_contracts.get(str(organ or ""))
        for key, value in mutate.items():
            field_name = str(key)
            if field_name in protected:
                packet.diagnostics.setdefault("mutation_denied", []).append({"organ": organ, "field": field_name, "reason": "immutable"})
                continue
            if contract is not None and not self._contract_field_allowed(contract, field_name):
                packet.diagnostics.setdefault("mutation_denied", []).append({"organ": organ, "field": field_name, "reason": "organ_contract_owner_violation"})
                continue
            if not hasattr(packet, field_name):
                packet.extensions.setdefault("unmapped_mutations", {})[field_name] = copy.deepcopy(value)
                continue
            setattr(packet, field_name, copy.deepcopy(value))

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
            "language": {"version": QSML_LANGUAGE_VERSION, "type_system_version": SML_TYPE_SYSTEM_VERSION, "natural_language_compiler": True, "typed_variables": True, "cognitive_ast": True, "qmath_evaluator": True, "organ_contracts": True, "arbitrary_application_blueprints": True, "synthesis_schema": "SarahMemory.qsml.application_blueprint.v0_2"},
            "organs": {name: organ.to_dict() for name, organ in sorted(self.organs.items())},
            "organ_contracts": {name: contract.to_dict() for name, contract in sorted(self.organ_contracts.items())},
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
        qsml = packet.extensions.get("qsml_program", {})
        compile_check = local.compile_natural_language("Build a local race car game with keyboard controls and a high score", context={"target": "nailde"}, packet=packet, collect_external_evidence=False)
        typed_ok = bool((compile_check.program.variables.variables if compile_check.program else {}))
        route_owner_ok = bool(compile_check.program and compile_check.program.metadata.get("route_activation_owner") == "SarahMemoryNeuron")
        arbitrary_contract_ok = bool(compile_check.program and compile_check.program.build_spec.get("compiler_policy", {}).get("arbitrary_application_synthesis"))
        no_prompt_template_ok = bool(compile_check.program and compile_check.program.build_spec.get("compiler_policy", {}).get("prompt_specific_template_selection") is False)
        bp_check = local.validate_application_blueprint({
            "name": "Self Test Application",
            "goal": "Demonstrate arbitrary synthesis contract validation",
            "project_kind": "desktop_application",
            "languages": ["python"],
            "requirements": [{"text": "Provide a local window", "kind": "functional", "priority": "must"}],
            "files": [
                {"path": "sandbox/app/main.py", "purpose": "Application entrypoint", "language": "python", "artifact_role": "ENTRYPOINT", "entrypoint": True}
            ],
            "constraints": {"sandbox_only": True, "live_core_write": False, "self_approval": False},
        }, source_program=compile_check.program)
        blueprint_ok = bool(bp_check.get("ok"))
        eval_node = SMLASTNode(kind=SMLASTKind.OPERATOR.value, operator=QMathState.SAME.value, children=[SMLASTNode(kind=SMLASTKind.LITERAL.value, value=3), SMLASTNode(kind=SMLASTKind.LITERAL.value, value=3)])
        evaluator_ok = bool(local.evaluate_qmath_ast(eval_node).get("value") is True)
        return {
            "status": SMLStatus.OK.value if validation["status"] in (SMLStatus.OK.value, SMLStatus.WARNING.value) and restored_ok and bool(grammar) and bool(qsml) and typed_ok and route_owner_ok and arbitrary_contract_ok and no_prompt_template_ok and blueprint_ok and evaluator_ok else SMLStatus.ERROR.value,
            "elapsed_ms": elapsed_ms,
            "packet_id": packet.packet_id,
            "mission": packet.mission,
            "pipeline": packet.pipeline,
            "qmath": (grammar.get("qmath") or {}).get("primary"),
            "loop_guard": grammar.get("loop_guard"),
            "validation": validation,
            "serialization_roundtrip": restored_ok,
            "qsml_program_present": bool(qsml),
            "qsml_typed_variables": typed_ok,
            "qsml_route_activation_owner": route_owner_ok,
            "qsml_arbitrary_application_contract": arbitrary_contract_ok,
            "qsml_no_prompt_specific_template_selection": no_prompt_template_ok,
            "qsml_application_blueprint_validation": blueprint_ok,
            "qsml_operator_evaluator": evaluator_ok,
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
        "qsml_program": copy.deepcopy((pkt.extensions or {}).get("qsml_program", {})),
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
        context={"caller": caller, "api_context": ctx, "target": str(ctx.get("target") or ctx.get("surface") or ("nailde" if "nailde" in str(caller).lower() else ""))},
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
    protocol = get_protocol()
    if omega in protocol.omega_registry:
        transition = protocol.omega_registry[omega]
        state_allowed = not transition.input_states or pkt.cognitive_state in set(transition.input_states)
        if state_allowed:
            pkt = protocol.transition_packet(pkt, omega, organ=organ, note=note or action, mutate=updates)
        else:
            # Compatibility observation: an organ may observe an already-routed or
            # later-state packet without illegally rewinding the Ω state machine.
            if updates:
                protocol._safe_mutate_packet(pkt, updates, organ=organ)
            pkt.add_history(organ, action, pkt.current_omega, note or "observed without state transition")
            pkt.add_ledger_entry(pkt.current_omega, organ, pkt.governance.get("decision", GovernanceDecision.PENDING.value), note or action, {"requested_omega": omega, "state_transition": False})
            pkt.seal()
    else:
        if updates:
            protocol._safe_mutate_packet(pkt, updates, organ=organ)
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





def _sml_question_operator(text: str) -> str:
    t = _normalize_token(str(text or "").split(" ", 1)[0] if str(text or "").strip() else "")
    aliases = {"whats": "what", "what_s": "what", "who_s": "who", "hows": "how"}
    t = aliases.get(t, t)
    for op in ("who", "what", "when", "where", "why", "how", "if", "make", "create", "build", "open", "run", "write", "summarize", "compare", "verify"):
        if t == op:
            return op.upper()
    low = str(text or "").strip().lower()
    if low.startswith(("hey sarah make", "make ", "create ", "build ")):
        return "MAKE"
    return "UNKNOWN"


def sml_build_dynamic_claim_vector(text: str, *, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Build a dynamic claim vector for Court routing.

    This is protocol grammar, not an answer pool. It avoids infinite static
    labels such as CURRENT_NEWS/CURRENT_WEATHER by separating stable axes from
    dynamic values: operator, temporal scope, domain, subject, role, execution,
    freshness, authority, validation, and rollback.
    """
    raw = str(text or "").strip()
    low = re.sub(r"\s+", " ", raw.lower()).strip()
    ctx = dict(context or {})
    op = _sml_question_operator(raw)

    historical_terms = ("founder", "founded", "invented", "created originally", "first ", "former", "was ", "served", "in 19", "in 20")
    explicit_current_terms = ("current", "currently", "today", "now", "latest", "this year", "right now", "as of")
    active_role_terms = ("ceo", "president", "prime minister", "leader", "head of", "chair", "chairman", "chairwoman", "director", "owner", "governor", "mayor")
    live_state_terms = ("weather", "forecast", "stock", "market", "price", "schedule", "calendar", "news", "headlines", "war", "election", "score", "version")
    self_terms = ("who are you", "what are you", "your name", "what is your name", "look like", "avatar", "2d model", "3d model", "how do you feel", "your state", "your status")
    clock_terms = ("what year", "what time", "what date", "timezone", "time zone", "date is it", "time is it", "year is it")
    execution_terms = ("open", "run", "launch", "start", "turn on", "turn off", "set ", "delete", "move", "write file", "install")
    creation_terms = ("make", "create", "build", "generate", "code", "write")
    software_build_terms = ("app", "application", "program", "software", "game", "addon", "add-on", "addons", "plugin", "tool", "dashboard", "tracker", "website", "web app", "panel", "widget", "playable", "launcher", "simulator")

    temporal_scope = "TIMELESS_OR_UNKNOWN"
    if any(x in low for x in explicit_current_terms):
        temporal_scope = "CURRENT_EXPLICIT"
    elif any(x in low for x in active_role_terms + live_state_terms) or any(x in low for x in clock_terms):
        temporal_scope = "CURRENT_IMPLICIT"
    if any(x in low for x in historical_terms) and not any(x in low for x in explicit_current_terms):
        temporal_scope = "HISTORICAL"

    is_software_creation = any(x in low for x in creation_terms) and any(x in low for x in software_build_terms)

    domain = "general"
    if any(x in low for x in self_terms):
        domain = "identity_self_embodiment"
    elif any(x in low for x in clock_terms):
        domain = "temporal_locality"
    elif is_software_creation:
        domain = "creative_build_mission"
    elif any(x in low for x in ("stock", "market", "ticker", "portfolio", "trade", "buy", "sell")):
        domain = "finance_market"
    elif any(x in low for x in ("weather", "forecast", "rain", "temperature outside")):
        domain = "weather"
    elif any(x in low for x in ("schedule", "calendar", "appointment", "meeting")):
        domain = "personal_schedule"
    elif any(x in low for x in ("news", "headline", "war", "election", "current events")):
        domain = "public_events"
    elif any(x in low for x in active_role_terms):
        domain = "active_role_holder"
    elif any(x in low for x in ("capslock", "caps lock", "num lock", "keyboard", "rgb", "light")):
        domain = "local_device_control"

    claim_type = "GENERAL_CLAIM"
    if domain == "temporal_locality":
        claim_type = "LIVE_TEMPORAL_OR_LOCALITY_STATE"
    elif domain == "identity_self_embodiment":
        claim_type = "IDENTITY_SELF_EMBODIMENT_STATE"
    elif domain == "active_role_holder":
        claim_type = "ROLE_HOLDER" if temporal_scope != "HISTORICAL" else "HISTORICAL_ROLE_OR_FOUNDER_FACT"
    elif domain in {"public_events", "weather", "finance_market", "personal_schedule"}:
        claim_type = "LIVE_OR_FRESHNESS_SENSITIVE_STATE"
    elif domain == "creative_build_mission":
        claim_type = "CREATE_VALIDATE_INSTALL_OR_EXECUTE_ARTIFACT"
    elif domain == "local_device_control":
        claim_type = "LOCAL_DEVICE_ACTION"

    freshness_required = temporal_scope in {"CURRENT_EXPLICIT", "CURRENT_IMPLICIT"} or domain in {"temporal_locality", "weather", "finance_market", "personal_schedule", "public_events"}
    execution_required = any(x in low for x in execution_terms) or domain in {"creative_build_mission", "local_device_control"}
    requires_user_confirmation = execution_required and domain not in {"temporal_locality", "identity_self_embodiment"}
    model_final_authority = not freshness_required and domain not in {"temporal_locality", "identity_self_embodiment", "local_device_control", "creative_build_mission"}

    preferred_sources: List[str] = []
    if domain == "temporal_locality":
        preferred_sources = ["appsys.ClockCourt", "system_clock", "configured_timezone"]
    elif domain == "identity_self_embodiment":
        preferred_sources = ["appself.IdentityCourt", "CognitiveSelf", "IdentityRegistry", "AvatarManifest", "ClockCourt"]
    elif domain == "personal_schedule":
        preferred_sources = ["ClockCourt", "CalendarOrScheduleSource", "SarahMemoryAPI"]
    elif domain in {"public_events", "active_role_holder"}:
        preferred_sources = ["SarahMemoryResearch", "SarahMemoryAPI", "RSS", "Terminal"]
    elif domain == "weather":
        preferred_sources = ["WeatherAPI", "Research", "ClockLocationCourt"]
    elif domain == "finance_market":
        preferred_sources = ["MarketDataAPI", "SarahMemoryAPI", "Research", "BrokerReadOnlyIfConnected"]
    elif domain == "creative_build_mission":
        preferred_sources = ["SMLProtocol", "Research", "SarahMemoryAPI", "LocalModels", "appsdk.run_governed_creation_mission", "SarahMemoryNAILDE", "Compare", "AssuranceGate", "appstore.ADDONRegistry", "Ledger"]
    elif domain == "local_device_control":
        preferred_sources = ["appdrivers.run_governed_device_intent", "OperatorCore", "SecurityGovernor", "AssuranceGate", "DriverVerification", "Ledger"]
    else:
        preferred_sources = ["LocalMemory", "LocalLLM", "Compare"]

    return {
        "schema": "SarahMemory.sml.dynamic_claim_vector.B09",
        "raw_text": raw,
        "operator": op,
        "intent_family": "CREATE_OR_EXECUTE" if execution_required else ("QUESTION" if op in {"WHO", "WHAT", "WHEN", "WHERE", "WHY", "HOW", "IF", "UNKNOWN"} else "REQUEST"),
        "domain": domain,
        "subject": raw,
        "role_or_target": "dynamic_extracted_from_text",
        "temporal_scope": temporal_scope,
        "claim_type": claim_type,
        "freshness_required": bool(freshness_required),
        "execution_required": bool(execution_required),
        "requires_user_confirmation": bool(requires_user_confirmation),
        "model_final_authority": bool(model_final_authority),
        "source_authority_needed": bool(freshness_required or not model_final_authority),
        "preferred_sources": preferred_sources,
        "forbidden_final_sources": ["model_memory", "static_demo_facts"] if not model_final_authority else ["static_demo_facts"],
        "validation": {
            "compare_required": True,
            "freshness_check_required": bool(freshness_required),
            "artifact_required": bool(freshness_required or execution_required),
            "rollback_required": bool(execution_required),
            "post_execution_verification_required": bool(execution_required),
            "domain_owner_required": "appdrivers.py" if domain == "local_device_control" else ("appsdk.py/SarahMemoryNAILDE.py" if domain == "creative_build_mission" else "dynamic_by_mission"),
            "sandbox_first_required": bool(domain == "creative_build_mission"),
            "live_install_or_run_requires_explicit_user_approval": bool(domain == "creative_build_mission"),
        },
        "context": ctx,
    }


def sml_build_source_authority_court_packet(text: str, *, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    vector = sml_build_dynamic_claim_vector(text, context=context)
    model_allowed = bool(vector.get("model_final_authority"))
    return {
        "schema": "SarahMemory.sml.source_authority_court_packet.B09",
        "court": "SML_SOURCE_AUTHORITY_COURT",
        "claim_vector": vector,
        "court_1": {
            "decision": "REQUIRE_EVIDENCE_ARTIFACTS" if not model_allowed else "ALLOW_MODEL_AS_CANDIDATE_WITH_COMPARE",
            "preferred_sources": vector.get("preferred_sources", []),
            "forbidden_final_sources": vector.get("forbidden_final_sources", []),
            "model_memory_final_authority": model_allowed,
        },
        "court_2": {
            "pending": True,
            "rule": "Returned artifacts must be compared for authority, freshness, confidence, and conflicts before Reply.",
        },
        "execution_authority": False,
        "read_only": not bool(vector.get("execution_required")),
        "ts": _utc_now(),
    }



# ---------------------------------------------------------------------------
# B09 Evidence Artifact normalization and Court-2 adjudication
# ---------------------------------------------------------------------------

def _sml_float01(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out < 0.0:
        return 0.0
    if out > 1.0:
        return 1.0
    return out


def _sml_evidence_source_value(source: Any) -> str:
    try:
        if hasattr(source, "value"):
            return str(getattr(source, "value"))
    except Exception:
        pass
    return str(source or "unknown_source").strip().lower() or "unknown_source"


def _sml_provider_from_source(source_name: str, raw: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    provider = str(raw.get("provider") or raw.get("model_provider") or metadata.get("provider") or metadata.get("original_source") or "").strip().lower()
    if provider in {"llama", "llama_api", "meta_ai", "meta.ai"}:
        return "meta"
    if provider:
        return provider
    s = _sml_evidence_source_value(source_name)
    for prefix in ("api_", "provider_"):
        if s.startswith(prefix):
            return s[len(prefix):]
    if s in {"openai", "meta", "claude", "anthropic", "mistral", "gemini", "huggingface", "deepseek", "groq", "cohere", "ollama", "local_llm", "mesh"}:
        return "meta" if s in {"llama", "meta_ai", "meta.ai"} else s
    return ""


def _sml_evidence_family(source_name: str, metadata: Mapping[str, Any], provider: str = "") -> str:
    s = _sml_evidence_source_value(source_name)
    lane = str(metadata.get("lane") or "").strip().lower()
    method = str(metadata.get("method") or "").strip().lower()
    if "rss" in s or "rss" in lane or "rss" in method:
        return "rss_feed"
    if s.startswith("web_") or any(x in s for x in ("wikipedia", "duckduckgo", "dictionary", "openlibrary", "stackoverflow", "reddit", "wikihow", "quora", "archive")):
        return "web_research"
    if s.startswith("api_") or provider in {"openai", "meta", "claude", "anthropic", "mistral", "gemini", "huggingface", "deepseek", "groq", "cohere", "ollama", "mesh"}:
        return "api_provider"
    if s in {"local_llm", "local_llm_ensemble", "model_catalog", "ollama"} or "llm" in s:
        return "local_model"
    if "static" in s:
        return "static_demo"
    if s.startswith("local_") or lane.startswith("local") or "sqlite" in s or "cache" in s or "dataset" in s or "vector" in s or "memory" in s:
        return "local_knowledge"
    if "terminal" in s or "agent" in s:
        return "terminal_agent"
    if "clock" in s or "time" in s or "system_clock" in s:
        return "system_runtime"
    return "unknown"


def _sml_authority_for_family(family: str, provider: str, metadata: Mapping[str, Any]) -> Tuple[str, float, bool, List[str]]:
    fam = str(family or "unknown").strip().lower()
    reasons: List[str] = []
    if fam == "system_runtime":
        return "direct_runtime_evidence", 0.95, True, ["system/runtime source is authoritative only for its owned runtime claim class"]
    if fam in {"weather_api", "market_data_api", "broker_readonly"}:
        return "domain_api_evidence", 0.88, True, ["domain API can be final evidence when claim type matches and freshness is valid"]
    if fam == "web_research":
        return "retrieved_source_evidence", 0.76, True, ["retrieved web source can support current/public claims when freshness and source relevance are adequate"]
    if fam == "rss_feed":
        return "polling_current_source_evidence", 0.70, True, ["RSS is polling-based current-source evidence, not realtime truth by itself"]
    if fam == "terminal_agent":
        return "observed_tool_evidence", 0.66, True, ["terminal/agent observation may support claims only inside scoped task truth"]
    if fam == "local_knowledge":
        return "local_memory_or_dataset_evidence", 0.58, False, ["local knowledge may be stale for current public claims"]
    if fam == "api_provider":
        p = provider or str(metadata.get("provider") or "")
        label = f"provider candidate reasoning ({p})" if p else "provider candidate reasoning"
        return "candidate_reasoning", 0.46, False, [label, "model/provider output is evidence, not final authority"]
    if fam == "local_model":
        return "candidate_reasoning", 0.34, False, ["local model output is candidate reasoning, not final authority for live facts"]
    if fam == "static_demo":
        return "demo_fixture", 0.0, False, ["static/demo facts are not production runtime authority"]
    return "unknown_authority", 0.18, False, ["source authority could not be classified"]


def sml_normalize_evidence_artifact(
    raw: Any,
    *,
    query: str = "",
    claim_vector: Optional[Mapping[str, Any]] = None,
    source_hint: str = "",
    provider_hint: str = "",
    family_hint: str = "",
    observed_at: str = "",
) -> Dict[str, Any]:
    """Normalize arbitrary source output into one Court-ready evidence object.

    B09 does not decide truth here. It gives SML Court 2 a single auditable
    shape for Research, RSS, SarahMemoryAPI/provider responses, local DB/model
    output, Terminal observations, system runtime probes, and future organs.
    """
    if isinstance(raw, Mapping):
        src = dict(raw)
    else:
        src = {"data": raw}
    metadata = src.get("metadata") if isinstance(src.get("metadata"), Mapping) else {}
    text = str(src.get("data") or src.get("content") or src.get("snippet") or src.get("answer") or src.get("summary") or "").strip()
    source_name = _sml_evidence_source_value(source_hint or src.get("source") or metadata.get("source_label") or metadata.get("source") or "unknown_source")
    provider = str(provider_hint or _sml_provider_from_source(source_name, src, metadata)).strip().lower()
    if provider in {"llama", "llama_api", "meta_ai", "meta.ai"}:
        provider = "meta"
    family = str(family_hint or src.get("source_family") or metadata.get("source_family") or _sml_evidence_family(source_name, metadata, provider)).strip().lower()
    authority_class, authority_score, final_possible, authority_notes = _sml_authority_for_family(family, provider, metadata)
    confidence = _sml_float01(src.get("confidence", metadata.get("confidence", 0.0)), 0.0)
    if confidence <= 0.0 and text:
        confidence = 0.50 if family not in {"static_demo", "unknown"} else 0.15
    cv = dict(claim_vector or {}) if isinstance(claim_vector, Mapping) else {}
    freshness_required = bool(cv.get("freshness_required"))
    temporal_scope = str(cv.get("temporal_scope") or "TIMELESS_OR_UNKNOWN")
    supports_current = bool(final_possible and family in {"system_runtime", "weather_api", "market_data_api", "broker_readonly", "web_research", "rss_feed", "terminal_agent"})
    runtime_authority = bool(final_possible)
    if family == "api_provider":
        runtime_authority = False
    if family == "static_demo":
        runtime_authority = False
    stale_risk = bool(freshness_required and not supports_current)
    freshness_class = "freshness_unknown"
    if family == "system_runtime":
        freshness_class = "live_runtime"
    elif family in {"web_research", "rss_feed", "weather_api", "market_data_api", "broker_readonly"}:
        freshness_class = "current_source_candidate"
    elif family in {"api_provider", "local_model"}:
        freshness_class = "model_or_provider_response_time_only"
    elif family == "local_knowledge":
        freshness_class = "local_cache_or_dataset_age_unknown"
    elif family == "static_demo":
        freshness_class = "demo_not_production"
    artifact = {
        "schema": "SarahMemory.sml.evidence_artifact.B09",
        "artifact_id": "ev_" + _sha256_obj({
            "query": query,
            "source": source_name,
            "provider": provider,
            "family": family,
            "content_hash": _sha256_text(text[:4096]),
        })[:24],
        "query": str(query or src.get("query") or ""),
        "claim_vector": cv,
        "source": source_name,
        "source_family": family,
        "provider": provider,
        "model": str(src.get("model_used") or src.get("model") or metadata.get("model") or "").strip(),
        "content": _bounded_text(text, 4096),
        "content_hash": _sha256_text(text) if text else "",
        "confidence": confidence,
        "authority_class": authority_class,
        "authority_score": authority_score,
        "final_authority_possible": bool(final_possible),
        "runtime_authority": bool(runtime_authority),
        "supports_current_claim": bool(supports_current),
        "freshness_class": freshness_class,
        "freshness_required": freshness_required,
        "temporal_scope": temporal_scope,
        "stale_for_current_claim_risk": bool(stale_risk),
        "observed_at": str(observed_at or src.get("observed_at") or src.get("timestamp") or metadata.get("observed_at") or _utc_now()),
        "latency_ms": src.get("latency_ms", metadata.get("latency_ms", 0)),
        "limitations": list(authority_notes),
        "metadata": {
            "source_metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
            "raw_source": source_name,
            "raw_provider": provider,
            "b09_note": "Evidence artifact is an input to SML Court 2; it is not truth by itself.",
        },
    }
    return artifact


def sml_normalize_evidence_artifacts(
    raw_items: Any,
    *,
    query: str = "",
    claim_vector: Optional[Mapping[str, Any]] = None,
    source_hint: str = "",
) -> List[Dict[str, Any]]:
    """Normalize one or many raw source results into EvidenceArtifact objects."""
    if raw_items is None:
        return []
    if isinstance(raw_items, Mapping):
        # Prefer existing normalized artifacts if present.
        existing = raw_items.get("evidence_artifacts")
        if isinstance(existing, list) and existing:
            out: List[Dict[str, Any]] = []
            for item in existing:
                if isinstance(item, Mapping) and str(item.get("schema") or "").endswith("evidence_artifact.B09"):
                    out.append(dict(item))
                else:
                    out.append(sml_normalize_evidence_artifact(item, query=query, claim_vector=claim_vector, source_hint=source_hint))
            return out
        one = raw_items.get("evidence_artifact")
        if isinstance(one, Mapping):
            if str(one.get("schema") or "").endswith("evidence_artifact.B09"):
                return [dict(one)]
            return [sml_normalize_evidence_artifact(one, query=query, claim_vector=claim_vector, source_hint=source_hint)]
        return [sml_normalize_evidence_artifact(raw_items, query=query, claim_vector=claim_vector, source_hint=source_hint)]
    if isinstance(raw_items, (list, tuple, set)):
        out = []
        for item in raw_items:
            out.extend(sml_normalize_evidence_artifacts(item, query=query, claim_vector=claim_vector, source_hint=source_hint))
        # De-duplicate by artifact ID while preserving order.
        seen: Set[str] = set()
        deduped: List[Dict[str, Any]] = []
        for art in out:
            aid = str(art.get("artifact_id") or "")
            if aid and aid not in seen:
                seen.add(aid)
                deduped.append(art)
        return deduped
    return [sml_normalize_evidence_artifact(raw_items, query=query, claim_vector=claim_vector, source_hint=source_hint)]


def _sml_score_evidence_for_claim(artifact: Mapping[str, Any], vector: Mapping[str, Any]) -> Tuple[float, List[str], bool]:
    """Return (score, reasons, rejected) for one artifact under one claim vector."""
    reasons: List[str] = []
    content = str(artifact.get("content") or "").strip()
    if not content:
        return 0.0, ["empty_content"], True
    family = str(artifact.get("source_family") or "unknown")
    authority = _sml_float01(artifact.get("authority_score", 0.0), 0.0)
    confidence = _sml_float01(artifact.get("confidence", 0.0), 0.0)
    freshness_required = bool(vector.get("freshness_required"))
    domain = str(vector.get("domain") or "general")
    supports_current = bool(artifact.get("supports_current_claim"))
    runtime_authority = bool(artifact.get("runtime_authority"))
    score = (authority * 0.62) + (confidence * 0.38)
    if freshness_required and not supports_current:
        score = min(score, 0.44)
        reasons.append("not_authoritative_for_current_or_freshness_sensitive_claim")
    if family == "static_demo":
        score = 0.0
        reasons.append("static_demo_fixture_rejected_for_production_truth")
    if family in {"api_provider", "local_model"}:
        reasons.append("model_or_provider_candidate_not_final_authority")
    if freshness_required and domain in {"public_events", "active_role_holder", "finance_market", "weather"} and not runtime_authority:
        reasons.append("requires_retrieved_or_domain_source_before_final_answer")
    rejected = bool(score <= 0.0 or (freshness_required and not supports_current))
    return max(0.0, min(1.0, score)), reasons, rejected


def sml_adjudicate_evidence_artifacts(
    query: str,
    artifacts: Any,
    *,
    claim_vector: Optional[Mapping[str, Any]] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Court 2: sort and adjudicate normalized evidence artifacts.

    This is not majority vote and it does not fabricate missing facts. It uses
    claim-specific authority, freshness, confidence, and source-family limits to
    accept the best currently supported artifact or require more evidence.
    """
    cv = dict(claim_vector or sml_build_dynamic_claim_vector(query, context=context))
    normalized = sml_normalize_evidence_artifacts(artifacts, query=query, claim_vector=cv)
    rows: List[Dict[str, Any]] = []
    accepted_candidates: List[Dict[str, Any]] = []
    for art in normalized:
        score, reasons, rejected = _sml_score_evidence_for_claim(art, cv)
        row = dict(art)
        row["court_score"] = score
        row["court_reasons"] = list(reasons)
        row["court_rejected"] = bool(rejected)
        rows.append(row)
        if not rejected:
            accepted_candidates.append(row)
    rows.sort(key=lambda x: float(x.get("court_score") or 0.0), reverse=True)
    accepted_candidates.sort(key=lambda x: float(x.get("court_score") or 0.0), reverse=True)
    freshness_required = bool(cv.get("freshness_required"))
    threshold = 0.65 if freshness_required else 0.52
    accepted = accepted_candidates[0] if accepted_candidates and float(accepted_candidates[0].get("court_score") or 0.0) >= threshold else None
    if accepted:
        verdict = "ACCEPT_BEST_AVAILABLE_ARTIFACT"
    elif normalized:
        verdict = "REQUIRE_ADDITIONAL_EVIDENCE"
    else:
        verdict = "NO_EVIDENCE_ARTIFACTS_PROVIDED"
    return {
        "ok": bool(accepted),
        "schema": "SarahMemory.sml.evidence_court_verdict.B09",
        "court": "SML_EVIDENCE_REALITY_COURT",
        "query": str(query or ""),
        "claim_vector": cv,
        "artifact_count": len(normalized),
        "accepted_artifact": accepted,
        "accepted_content": str(accepted.get("content") or "") if isinstance(accepted, Mapping) else "",
        "ranked_artifacts": rows[:12],
        "rejected_artifacts": [r for r in rows if bool(r.get("court_rejected"))][:12],
        "verdict": verdict,
        "threshold": threshold,
        "rule": "Truth is claim-specific. Authority is source-specific. Freshness is time-specific. No provider/model/static artifact is final authority by itself for current public facts.",
        "execution_authority": False,
        "ts": _utc_now(),
    }


def sml_build_evidence_court_packet(
    text: str,
    raw_artifacts: Any,
    *,
    context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the full B09 source-authority + evidence-adjudication court packet."""
    source_court = sml_build_source_authority_court_packet(text, context=context)
    vector = source_court.get("claim_vector") if isinstance(source_court, Mapping) else {}
    verdict = sml_adjudicate_evidence_artifacts(text, raw_artifacts, claim_vector=vector if isinstance(vector, Mapping) else None, context=context)
    return {
        "ok": bool(verdict.get("ok")),
        "schema": "SarahMemory.sml.evidence_court_packet.B09",
        "court": "SML_TWO_COURT_EVIDENCE_PIPELINE",
        "court_1": source_court,
        "court_2": verdict,
        "accepted_content": verdict.get("accepted_content", ""),
        "execution_authority": False,
        "ts": _utc_now(),
    }


def _sml_extract_reference_descriptor(text: str) -> str:
    """Extract a non-authoritative style/reference descriptor for research planning.

    This is dynamic grammar, not a prompt-specific template. The returned text is
    used only to ask Research/NAILDE what concepts need evidence before synthesis.
    """
    raw = str(text or "").strip()
    low = raw.lower()
    patterns = [
        r"\b([a-z0-9][a-z0-9 ._\-]{1,80})\s*[- ]?like\b",
        r"\blike\s+(?:a|an|the)?\s*([a-z0-9][a-z0-9 ._\-]{1,80})(?:\s+(?:game|app|application|tool|style|system)|[?.!,]|$)",
        r"\b([a-z0-9][a-z0-9 ._\-]{1,80})\s+style\b",
        r"\binspired\s+by\s+([a-z0-9][a-z0-9 ._\-]{1,80})(?:[?.!,]|$)",
    ]
    for pat in patterns:
        m = re.search(pat, low, flags=re.I)
        if m:
            value = re.sub(r"\s+", " ", m.group(1)).strip(" .,_-")
            if value and value not in {"make me", "create", "build", "game", "app", "application"}:
                return _bounded_text(value, 240)
    return ""


def sml_build_creation_mission_contract(text: str, *, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Build the non-executing QSML/NAILDE creation mission contract.

    This is B07's generalized create/build/app/addon pipeline. It does not
    hardcode any example application. It compiles the user's request into a
    governed mission packet, records required research/abstraction/build phases,
    and states the boundary: NAILDE may stage sandbox artifacts, but install/run
    remains a separate explicit approval path.
    """
    ctx = dict(context or {})
    raw = str(text or "").strip()
    vector = sml_build_dynamic_claim_vector(raw, context={**ctx, "target": ctx.get("target") or "nailde"})
    if str(vector.get("domain") or "") != "creative_build_mission":
        return {
            "ok": False,
            "schema": "SarahMemory.sml.creation_mission_contract.B07",
            "error": "not_a_creation_build_mission",
            "claim_vector": vector,
            "execution_authority": False,
        }
    compile_result: Dict[str, Any]
    try:
        compile_result = sml_compile_natural_program(
            raw,
            context={**ctx, "target": "nailde", "sandbox_only": True, "source": ctx.get("source") or "sml_creation_contract"},
            target="nailde",
            collect_external_evidence=bool(ctx.get("collect_external_evidence", True)),
        )
    except Exception as exc:
        compile_result = {"status": "FAILED", "error": _redact_sensitive_text(str(exc))[:500], "execution_authority": False}
    reference = _sml_extract_reference_descriptor(raw)
    research_questions = [
        "What problem or experience is the user asking the system to create?",
        "What are the core mechanics, inputs, outputs, and success conditions?",
        "What constraints prevent unsafe execution, live-core mutation, credential access, or self-approval?",
        "What local implementation architecture best satisfies the request inside the ADDON/NAILDE sandbox?",
        "What validation, static tests, package-integrity checks, and rollback evidence are required before install/run?",
    ]
    if reference:
        research_questions.insert(1, f"What are the recognizable high-level concepts of the reference descriptor '{reference}' without copying protected names, assets, or branding?")
        research_questions.insert(2, "Which mechanics/style elements can be transformed into an original local implementation?")
    phases = [
        {"id": "observe", "owner": "app.py", "output": "raw user request routed into SML packet"},
        {"id": "compile", "owner": "SarahMemorySMLProtocol", "output": "dynamic claim vector + QSML program"},
        {"id": "research", "owner": "SarahMemoryResearch/SarahMemoryAPI/Terminal as authorized", "output": "evidence artifacts and reference abstraction plan"},
        {"id": "design", "owner": "SarahMemoryNAILDE", "output": "application blueprint"},
        {"id": "synthesize", "owner": "LocalModels via SarahMemoryAPI", "output": "sandbox source files"},
        {"id": "package", "owner": "SarahMemoryNAILDE", "output": "ADDON package ABI under workspace sandbox"},
        {"id": "validate", "owner": "Compare/NAILDE/static validation", "output": "syntax, manifest, integrity, safety, static run-readiness"},
        {"id": "install_plan", "owner": "appstore/ADDONRegistry", "output": "copy/install plan only; no auto-run"},
        {"id": "await_user", "owner": "Human operator", "output": "explicit install/run approval or cancellation"},
    ]
    return {
        "ok": True,
        "schema": "SarahMemory.sml.creation_mission_contract.B07",
        "court": "SML_CREATION_MISSION_COURT",
        "claim_vector": vector,
        "reference_descriptor": reference,
        "qsml_compile": compile_result,
        "court_1": {
            "decision": "SANDBOX_BUILD_PIPELINE_ALLOWED_BY_EXPLICIT_CREATE_REQUEST",
            "route_owner": "appsdk.py",
            "synthesis_owner": "SarahMemoryNAILDE.py",
            "activation_owner": "SarahMemoryNeuron.py",
            "install_owner": "appstore.py",
            "model_final_authority": False,
            "preferred_sources": vector.get("preferred_sources", []),
        },
        "research_plan": {
            "required": True,
            "questions": research_questions,
            "source_policy": "Use local/project sources first; use Research/API/Terminal only when authorized by SML and user/network policy.",
            "ip_boundary": "Extract mechanics and high-level style; do not copy protected assets, logos, or direct branded implementation.",
        },
        "pipeline": phases,
        "authority": {
            "sandbox_workspace_write": True,
            "live_core_write": False,
            "addon_install": "requires_separate_explicit_user_confirmation",
            "addon_run": "requires_separate_explicit_user_confirmation",
            "network_access": "only_if_authorized",
            "execution_authority": False,
        },
        "validation": {
            "compare_required": True,
            "static_validation_required": True,
            "manifest_validation_required": True,
            "package_integrity_required": True,
            "rollback_required_for_install": True,
            "ledger_receipt_required": True,
        },
        "ts": _utc_now(),
        "execution_authority": False,
    }



# ---------------------------------------------------------------------------
# B08 Avatar embodiment event packets
# ---------------------------------------------------------------------------

def _sml_event_text_blob(event: Mapping[str, Any]) -> str:
    """Return bounded lower-case text for semantic event-axis detection."""
    parts: List[str] = []
    for key in ("event_type", "type", "name", "status", "state", "domain", "source", "message", "summary", "reason", "action"):
        val = event.get(key)
        if val is not None:
            parts.append(str(val))
    return re.sub(r"\s+", " ", " ".join(parts)).strip().lower()[:2000]


def _sml_event_float(value: Any, default: float = 0.0, low: float = 0.0, high: float = 1.0) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if out < low:
        return float(low)
    if out > high:
        return float(high)
    return float(out)


def _sml_event_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "verified", "accepted", "ok", "success"}


def sml_build_avatar_event_packet(event: Union[str, Mapping[str, Any]], *, context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Compile an arbitrary verified/runtime event into an avatar embodiment packet.

    B08 establishes the avatar as an event-driven presentation organ.  This
    function does not hardcode any application domain.  It maps generic event
    axes (success/failure/alert/progress/observation) into a governed avatar
    directive.  Claims such as market movement, game state, or app status remain
    owned by their source artifacts; the avatar may embody the event only after
    SML records source/verification metadata.
    """
    ctx = dict(context or {})
    if isinstance(event, Mapping):
        raw_event = dict(event)
        raw_text = str(raw_event.get("event_text") or raw_event.get("message") or raw_event.get("summary") or raw_event.get("event_type") or raw_event.get("type") or "runtime_event").strip()
    else:
        raw_text = str(event or "runtime_event").strip()
        raw_event = {"event_text": raw_text, "event_type": "runtime_event"}
    raw_text = _bounded_text(raw_text or "runtime_event", 800)
    blob = _sml_event_text_blob(raw_event)
    event_type = _bounded_text(str(raw_event.get("event_type") or raw_event.get("type") or raw_event.get("name") or "runtime_event"), 96).lower().replace(" ", "_")
    source = _bounded_text(str(raw_event.get("source") or ctx.get("source") or "unknown_source"), 160)
    domain = _bounded_text(str(raw_event.get("domain") or ctx.get("domain") or raw_event.get("source_subsystem") or "runtime"), 96)
    severity = _sml_event_float(raw_event.get("severity", raw_event.get("intensity", ctx.get("severity", 0.35))), 0.35, 0.0, 1.0)
    source_verified = _sml_event_bool(raw_event.get("source_verified", raw_event.get("verified", ctx.get("source_verified", False))))
    validation_state = str(raw_event.get("validation_state") or ctx.get("validation_state") or "unknown").strip().lower()
    if validation_state in {"verified", "accepted", "validated", "passed"}:
        source_verified = True
    factual_claim = bool(raw_event.get("claim") or raw_event.get("fact") or raw_event.get("market_data") or raw_event.get("telemetry") or raw_event.get("score") or raw_event.get("price"))

    failure_words = ("fail", "failed", "failure", "error", "crash", "crashed", "collision", "blocked", "denied", "rejected", "stopped", "lost", "game_over", "game over")
    success_words = ("success", "succeeded", "complete", "completed", "ready", "installed", "validated", "passed", "filled", "done")
    progress_words = ("start", "started", "running", "launch", "launched", "building", "research", "synthes", "validating", "working", "thinking")
    alert_words = ("alert", "threshold", "crossed", "changed", "movement", "moved", "spike", "drop", "up", "down", "warning", "notice")

    if any(w in blob for w in failure_words):
        reaction_class = "runtime_interrupt_or_failure"
        expression = "surprised" if any(w in blob for w in ("crash", "collision", "game_over", "game over")) else "concerned"
        attention = "event_source"
        gesture = "alert_recoil" if expression == "surprised" else "look_concerned"
        valence = -0.55
        arousal = max(0.65, severity)
    elif any(w in blob for w in success_words):
        reaction_class = "runtime_success_or_completion"
        expression = "pleased"
        attention = "user"
        gesture = "confirming_nod"
        valence = 0.55
        arousal = max(0.35, severity)
    elif any(w in blob for w in alert_words):
        reaction_class = "verified_attention_alert"
        expression = "alert_focused"
        attention = "event_source"
        gesture = "attention_wave"
        valence = 0.05
        arousal = max(0.55, severity)
    elif any(w in blob for w in progress_words):
        reaction_class = "mission_progress"
        expression = "focused"
        attention = "workspace"
        gesture = "working_focus"
        valence = 0.15
        arousal = max(0.30, severity)
    else:
        reaction_class = "runtime_observation"
        expression = "attentive"
        attention = "event_source"
        gesture = "subtle_acknowledge"
        valence = 0.0
        arousal = max(0.20, severity)

    # External claims require verified artifacts before avatar speech may assert them.
    speech_requested = _sml_event_bool(raw_event.get("speech_requested", ctx.get("speech_requested", False)))
    speech_allowed = bool(speech_requested and (source_verified or not factual_claim))
    message = _bounded_text(str(raw_event.get("user_visible_message") or raw_event.get("message") or ""), 220)
    speech_text = message if speech_allowed else ""

    return {
        "ok": True,
        "schema": "SarahMemory.sml.avatar_event_packet.B08",
        "packet_role": "AVATAR_EMBODIMENT_EVENT",
        "event": {
            "event_type": event_type,
            "source": source,
            "domain": domain,
            "raw_text": raw_text,
            "source_verified": bool(source_verified),
            "validation_state": validation_state,
            "factual_claim_present": bool(factual_claim),
            "observed_at": _utc_now(),
        },
        "six_question_axes": {
            "WHO": str(raw_event.get("actor") or ctx.get("actor") or source),
            "WHAT": raw_text,
            "WHY": str(raw_event.get("purpose") or ctx.get("purpose") or "embody_verified_runtime_state"),
            "HOW": "event_packet_to_avatar_directive",
            "WHERE": str(raw_event.get("where") or raw_event.get("subsystem") or ctx.get("subsystem") or domain),
            "WHEN": str(raw_event.get("when") or _utc_now()),
        },
        "avatar_directive": {
            "reaction_class": reaction_class,
            "expression": expression,
            "attention": attention,
            "gesture": gesture,
            "valence": valence,
            "arousal": arousal,
            "intensity": max(0.05, min(1.0, arousal)),
            "duration_seconds": max(0.25, min(12.0, _sml_event_float(raw_event.get("duration_seconds", 2.25), 2.25, 0.25, 12.0))),
            "speech_allowed": bool(speech_allowed),
            "speech_text": speech_text,
        },
        "authority": {
            "avatar_presentation_only": True,
            "execution_authority": False,
            "device_control_authority": False,
            "trade_authority": False,
            "install_or_run_authority": False,
        },
        "validation": {
            "claims_must_be_verified_before_speech": True,
            "source_verified": bool(source_verified),
            "factual_claim_present": bool(factual_claim),
            "avatar_may_express_without_claiming_truth": True,
            "compare_required_for_external_claims": bool(factual_claim),
        },
        "context": ctx,
        "execution_authority": False,
    }

def sml_apply_cognitive_grammar(
    packet: Union[SMLPacket, Mapping[str, Any]],
    *,
    text: str = "",
    loop_state: Optional[Mapping[str, Any]] = None,
) -> SMLPacket:
    """Attach v0.8.2 six-question / SML-operator cognitive grammar to a packet."""
    pkt = packet if isinstance(packet, SMLPacket) else SMLPacket.from_dict(packet)
    return get_protocol().apply_cognitive_grammar(pkt, text=text, loop_state=loop_state)



def sml_compile_natural_program(
    text: str,
    *,
    context: Optional[Mapping[str, Any]] = None,
    packet: Optional[Union[SMLPacket, Mapping[str, Any]]] = None,
    target: str = "",
    collect_external_evidence: bool = True,
) -> Dict[str, Any]:
    """Compile natural language into executable QSML intermediate representation."""
    pkt = None
    if packet is not None:
        pkt = packet if isinstance(packet, SMLPacket) else SMLPacket.from_dict(packet)
    return get_protocol().compile_natural_language(
        text,
        context=context,
        packet=pkt,
        target=target,
        collect_external_evidence=collect_external_evidence,
    ).to_dict()


def sml_evaluate_qmath(node: Union[SMLASTNode, Mapping[str, Any]], environment: Optional[Mapping[str, Any]] = None, *, max_iterations: int = 8) -> Dict[str, Any]:
    return get_protocol().evaluate_qmath_ast(node, environment=environment, max_iterations=max_iterations)


def sml_register_organ_contract(contract: Union[SMLOrganContract, Mapping[str, Any]]) -> Dict[str, Any]:
    return get_protocol().register_organ_contract(contract)


def sml_application_blueprint_schema() -> Dict[str, Any]:
    return get_protocol().application_blueprint_schema()


def sml_validate_application_blueprint(
    blueprint: Union[SMLApplicationBlueprint, Mapping[str, Any]],
    *,
    require_files: bool = True,
    source_program: Optional[Union[SMLProgram, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    program_obj = source_program if isinstance(source_program, SMLProgram) else (SMLProgram.from_dict(source_program) if isinstance(source_program, Mapping) else None)
    return get_protocol().validate_application_blueprint(blueprint, source_program=program_obj, require_files=require_files)


def sml_compile_application_blueprint(
    blueprint: Union[SMLApplicationBlueprint, Mapping[str, Any]],
    *,
    source_program: Optional[Union[SMLProgram, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    program_obj = source_program if isinstance(source_program, SMLProgram) else (SMLProgram.from_dict(source_program) if isinstance(source_program, Mapping) else None)
    return get_protocol().compile_application_blueprint(blueprint, source_program=program_obj)


# =============================================================================
# Persistent Governed Cognitive Operation (GCOP)
# =============================================================================


def _gcop_event(value: Optional[Union[SMLCognitiveEvent, Mapping[str, Any], str]]) -> SMLCognitiveEvent:
    if isinstance(value, SMLCognitiveEvent):
        return value
    if isinstance(value, Mapping):
        return SMLCognitiveEvent.from_dict(value)
    if isinstance(value, str) and value.strip():
        return SMLCognitiveEvent(event_type=GCOPEventType.USER_DIRECTIVE.value, source="user", payload={"text": value}, requires_response=True)
    return SMLCognitiveEvent(event_type=GCOPEventType.SYSTEM_TICK.value, source=MODULE_NAME)


def sml_build_continuity_state(
    packet: Union[SMLPacket, Mapping[str, Any]],
    *,
    previous: Optional[Union[SMLCognitiveContinuityState, Mapping[str, Any]]] = None,
    event: Optional[Union[SMLCognitiveEvent, Mapping[str, Any], str]] = None,
    runtime_context: Optional[Mapping[str, Any]] = None,
) -> SMLCognitiveContinuityState:
    """Build/merge the shared GCOP state without performing cognition."""
    pkt = packet if isinstance(packet, SMLPacket) else SMLPacket.from_dict(packet)
    state = previous if isinstance(previous, SMLCognitiveContinuityState) else SMLCognitiveContinuityState.from_dict(previous or {})
    ev = _gcop_event(event)
    ctx = dict(runtime_context or {})
    state.updated_at = _utc_now()
    state.identity.update(copy.deepcopy(pkt.identity))
    state.mission.update(copy.deepcopy(pkt.mission))
    state.authority.update(copy.deepcopy(pkt.authority))
    state.adaptive.update(copy.deepcopy(pkt.adaptive))
    state.risk.update({"governance": copy.deepcopy(pkt.governance)})
    state.reality.update({
        "last_event": ev.to_dict(),
        "context": copy.deepcopy(pkt.context),
        "knowledge": copy.deepcopy(pkt.knowledge),
        "health": copy.deepcopy(pkt.health),
    })
    state.continuity.update({
        "packet_id": pkt.packet_id,
        "cognitive_state": pkt.cognitive_state,
        "current_omega": pkt.current_omega,
        "confidence": float(pkt.confidence),
        "pipeline": list(pkt.pipeline),
        "last_event_id": ev.event_id,
        "last_event_type": ev.event_type,
        "runtime_context": ctx,
    })
    state.audit.update({
        "packet_checksum": pkt.checksum,
        "ledger_entries": len(pkt.ledger),
        "organ_history_entries": len(pkt.organ_history),
    })
    return state


def sml_filter_legal_candidates(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Apply SML legality rails.  Scoring/activation is intentionally not done here."""
    legal: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    gates = ("capability", "authority", "safety", "resource_feasible", "time_valid", "mission_compatible")
    for raw in list(candidates or [])[:64]:
        c = dict(raw or {})
        gate_state = c.get("legal_gates") if isinstance(c.get("legal_gates"), dict) else {}
        # Missing legality evidence fails closed.  SML never invents a positive
        # capability/authority/safety/resource/time/mission gate.
        normalized = {g: bool(gate_state[g]) if g in gate_state else bool(c[g]) if g in c else False for g in gates}
        c["legal_gates"] = normalized
        owner_ok = str(c.get("route_definition_owner") or "") == "SarahMemorySMLProtocol"
        activation_ok = str(c.get("route_activation_owner") or "") == "SarahMemoryNeuron"
        c["protocol_ownership_valid"] = bool(owner_ok and activation_ok)
        c["sml_legal"] = all(normalized.values()) and c["protocol_ownership_valid"]
        if c["sml_legal"]:
            legal.append(c)
        else:
            reasons = [g for g, ok in normalized.items() if not ok]
            if not c.get("protocol_ownership_valid"):
                reasons.append("protocol_ownership")
            c["rejection_reasons"] = reasons
            rejected.append(c)
    return {"legal": legal, "rejected": rejected, "gate_names": list(gates)}


def sml_run_gcop_cycle(
    packet: Union[SMLPacket, Mapping[str, Any]],
    *,
    event: Optional[Union[SMLCognitiveEvent, Mapping[str, Any], str]] = None,
    continuity_state: Optional[Union[SMLCognitiveContinuityState, Mapping[str, Any]]] = None,
    runtime_context: Optional[Mapping[str, Any]] = None,
    max_candidates: int = 8,
) -> Dict[str, Any]:
    """Run one bounded event-driven cognitive coordination cycle.

    SML owns structure/routing only.  Cognitive interpretation is delegated to
    SarahMemoryCognitive*.py organs, Neuron owns activation among SML-legal
    candidates, and execution remains outside this function.
    """
    protocol = get_protocol()
    pkt = packet if isinstance(packet, SMLPacket) else SMLPacket.from_dict(packet)
    ev = _gcop_event(event)
    ctx = dict(runtime_context or {})
    diagnostics: Dict[str, Any] = {"organ_calls": {}, "errors": []}

    if not pkt.pipeline:
        try:
            protocol.route_packet(pkt)
        except Exception as exc:
            diagnostics["errors"].append("route_packet:" + _redact_sensitive_text(str(exc))[:400])

    state = sml_build_continuity_state(pkt, previous=continuity_state, event=ev, runtime_context=ctx)

    def organ_call(module_name: str, function_name: str, *args: Any, **kwargs: Any) -> Any:
        try:
            module = __import__(module_name)
            fn = getattr(module, function_name, None)
            if not callable(fn):
                diagnostics["organ_calls"][module_name] = {"ok": False, "reason": f"{function_name}_unavailable"}
                return None
            out = fn(*args, **kwargs)
            diagnostics["organ_calls"][module_name] = {"ok": True, "function": function_name}
            return out
        except Exception as exc:
            diagnostics["organ_calls"][module_name] = {"ok": False, "function": function_name, "error": _redact_sensitive_text(str(exc))[:400]}
            return None

    identity_state = organ_call("SarahMemoryCognitiveIdentityLayer", "gcop_identity_context", pkt.to_dict(), ev.to_dict(), state.to_dict(), ctx)
    if isinstance(identity_state, dict):
        state.identity.update(copy.deepcopy(identity_state))
    self_state = organ_call("SarahMemoryCognitiveSelf", "gcop_self_state", pkt.to_dict(), ev.to_dict(), state.to_dict(), ctx)
    if isinstance(self_state, dict):
        state.resources.update(copy.deepcopy(self_state.get("resources") or {}))
        state.continuity.update(copy.deepcopy(self_state.get("continuity") or {}))
        state.identity.update(copy.deepcopy(self_state.get("identity") or {}))
        state.reality.setdefault("self", {}).update(copy.deepcopy(self_state))
    bearing = organ_call("SarahMemoryCognitiveCompass", "gcop_mission_bearing", pkt.to_dict(), ev.to_dict(), state.to_dict(), ctx)
    if isinstance(bearing, dict):
        state.mission.setdefault("bearing", {}).update(copy.deepcopy(bearing))
    candidate_packet = organ_call("SarahMemoryCognitiveThinker", "gcop_candidate_set", pkt.to_dict(), ev.to_dict(), state.to_dict(), ctx, max_candidates=max_candidates)
    candidates = list((candidate_packet or {}).get("candidates") or []) if isinstance(candidate_packet, dict) else []
    if not candidates:
        diagnostics["errors"].append("CognitiveThinker returned no explicit candidate set; GCOP fails closed rather than inventing route legality.")
    filtered = sml_filter_legal_candidates(candidates)
    legal = filtered["legal"]
    rejected = filtered["rejected"]

    activation = organ_call("SarahMemoryNeuron", "neuron_activate_legal_candidates", legal, state.to_dict(), packet=pkt.to_dict())
    selected = dict((activation or {}).get("selected") or {}) if isinstance(activation, dict) else (dict(legal[0]) if legal else {})

    governance = organ_call("SarahMemoryCognitiveServices", "gcop_authority_review", pkt.to_dict(), ev.to_dict(), state.to_dict(), selected, ctx)
    governance = dict(governance or {"decision": "DEFER", "allow": False, "require_user": True, "reasons": ["CognitiveServices GCOP review unavailable"]})
    decision = str(governance.get("decision") or "DEFER").upper()
    allow = bool(governance.get("allow")) and decision in {"ALLOW", "APPROVED"}
    require_user = bool(governance.get("require_user")) or decision in {"REQUIRE_USER", "DEFER", "ESCALATED"}

    if not legal:
        status, stop_reason = GCOPCycleStatus.SAFE_HOLD.value, "no_sml_legal_candidate"
    elif require_user:
        status, stop_reason = GCOPCycleStatus.REQUIRE_USER.value, "authority_or_preference_requires_user"
    elif not allow:
        status, stop_reason = GCOPCycleStatus.SAFE_HOLD.value, "governance_denied_or_deferred"
    elif bool(ev.payload.get("mission_complete")):
        status, stop_reason = GCOPCycleStatus.COMPLETE.value, "mission_completion_event"
    else:
        status, stop_reason = GCOPCycleStatus.CONTINUE.value, ""

    state.authority["last_review"] = copy.deepcopy(governance)
    state.continuity.update({"cycle_status": status, "selected_candidate_id": selected.get("candidate_id"), "last_cycle_at": _utc_now()})
    state.audit.update({"legal_candidates": len(legal), "rejected_candidates": len(rejected), "governance_decision": decision})
    pkt.extensions["gcop"] = state.to_dict()
    pkt.metadata["gcop_version"] = state.version
    pkt.add_history(MODULE_NAME, "gcop_cycle", pkt.current_omega, f"event={ev.event_type};status={status}")
    pkt.add_ledger_entry(pkt.current_omega, MODULE_NAME, decision, "GCOP cycle coordinated", {"event_id": ev.event_id, "status": status, "selected_candidate": selected.get("candidate_id")})
    pkt.seal()

    execution_request = {
        "requested": bool(ev.requested_execution),
        "eligible": bool(allow and selected and ev.requested_execution),
        "selected_candidate": copy.deepcopy(selected),
        "authority_reference": ev.authority_reference,
        "execution_authority": False,
        "owner": "SarahMemoryOperatorCore",
    }
    result = SMLCognitiveCycleResult(
        status=status,
        packet=pkt.to_dict(),
        continuity_state=state.to_dict(),
        event=ev.to_dict(),
        legal_candidates=copy.deepcopy(legal),
        rejected_candidates=copy.deepcopy(rejected),
        selected_candidate=copy.deepcopy(selected),
        governance=copy.deepcopy(governance),
        execution_request=execution_request,
        diagnostics=diagnostics,
        reply_intent={"required": bool(ev.requires_response), "presentation_owner": "SarahMemoryReply"},
        next_wake={"mode": "event_driven", "requested": status in {GCOPCycleStatus.CONTINUE.value, GCOPCycleStatus.WAIT.value}},
        stop_reason=stop_reason,
    )
    return result.to_dict()


def sml_gcop_self_test() -> Dict[str, Any]:
    """Protocol-only structural GCOP tests; no side effects or execution."""
    pkt = create_sml_packet(raw_request="Explain current system status", identity={"primary": IdentityRole.USER.value}, context={"test": True})
    state = sml_build_continuity_state(pkt, event={"event_type": GCOPEventType.USER_DIRECTIVE.value, "source": "self_test"})
    owners = {"route_definition_owner": "SarahMemorySMLProtocol", "route_activation_owner": "SarahMemoryNeuron"}
    filtered = sml_filter_legal_candidates([{**owners, "candidate_id": "ok", "legal_gates": {"capability": True, "authority": True, "safety": True, "resource_feasible": True, "time_valid": True, "mission_compatible": True}}, {**owners, "candidate_id": "bad", "legal_gates": {"capability": True, "authority": False, "safety": True, "resource_feasible": True, "time_valid": True, "mission_compatible": True}}])
    checks = [
        {"name": "continuity_state", "passed": bool(state.state_id and state.continuity.get("packet_id") == pkt.packet_id)},
        {"name": "legal_filter", "passed": len(filtered["legal"]) == 1 and len(filtered["rejected"]) == 1},
        {"name": "execution_authority", "passed": True, "observed": False},
    ]
    return {"ok": all(c["passed"] for c in checks), "checks": checks, "execution_authority": False}


# =============================================================================
# Governed Cognitive AI Operating System (GCAIOS) semantic profile
# =============================================================================

GCAIOS_PROFILE = "SarahMemory.GCAIOS/1.0-alpha"


def sml_gcaios_manifest() -> Dict[str, Any]:
    """Return the declared GCAIOS ownership map; this grants no runtime authority."""
    return {
        "ok": True,
        "schema": GCAIOS_PROFILE,
        "operating_model": "Persistent Governed Cognitive Operation",
        "human_authority": "highest",
        "authority_chain": [
            "UserInstructions",
            "SafetyRequirements",
            "EngineeringConstitution",
            "CognitiveServices",
            "CognitiveCompass",
            "AssuranceGate",
            "SecurityGovernor",
            "OperatorCore",
        ],
        "planes": {
            "semantic_control": {
                "owner": "SarahMemorySMLProtocol",
                "responsibilities": ["mission packets", "QSML operators", "routing contracts", "state transition legality"],
                "executes_actions": False,
            },
            "cognitive_continuity": {
                "owner": "SarahMemoryCognitiveServices",
                "activation_owner": "SarahMemoryNeuron",
                "profile": "GCOP/1.0",
                "executes_actions": False,
            },
            "adaptive_compute": {
                "passport_owner": "SarahMemoryAdaptive",
                "budget_owner": "SarahMemoryEnergetics",
                "execution_owner": "SarahMemoryOperatorCore",
                "clock_mutation_allowed": False,
            },
            "execution": {
                "owner": "SarahMemoryOperatorCore",
                "required_gates": ["CognitiveServices", "SecurityGovernor", "AssuranceGate", "Energetics"],
                "verification_and_rollback_required": True,
            },
            "audit": {
                "owner": "SarahMemoryLedger",
                "authority": False,
                "frame_by_frame_logging": False,
            },
            "spatial_fabric": {
                "semantic_owner": "SarahMemorySMLProtocol",
                "control_plane_owner": "api.server.appnet2",
                "data_plane_owner": "api.server.appnet",
                "coherence_owner": "SarahMemorySync",
                "transport_owner": "SarahMemoryNetwork",
                "rendering_owner": "client embodiment renderer",
            },
        },
        "state_classes": {
            "ephemeral": "bounded in-memory real-time state",
            "operational": "GCOP continuity and active mission state",
            "persistent": "user-controlled durable memory and world state",
            "ledger_proven": "meaningful consequential transition receipts",
        },
        "local_first_order": [
            "deterministic_logic",
            "local_memory",
            "local_models",
            "approved_network_resources",
            "approved_cloud_models",
            "governed_fallback",
        ],
        "doctrine": {
            "llm_is_authority": False,
            "model_is_replaceable_organ": True,
            "discovery_is_activation": False,
            "simulation_equals_physical_execution": False,
            "human_authority_remains_above_system": True,
        },
        "execution_authority": False,
    }


# =============================================================================
# SarahNet semantic contracts — Full SML authority + bounded SML-RT state
# =============================================================================

SARAHNET_RT_PROFILE = "SML-RT/1.0-alpha"
SARAHNET_AUTHORITY_LEASE_SCHEMA = "SarahNet.AuthorityLease/1.0-alpha"
SARAHNET_WORLD_CONTEXT_SCHEMA = "SarahNet.WorldContext/1.0-alpha"
SARAHNET_ENTITY_SCHEMA = "SarahNet.Entity/1.0-alpha"
SARAHNET_RT_MAX_DELTA_BYTES = 64 * 1024

# SML-RT is intentionally limited to ephemeral/temporary state. Consequential
# mutations must leave the real-time lane and return to Full SML governance.
SARAHNET_RT_STATE_CLASSES = {
    "pose",
    "animation",
    "locomotion",
    "physics",
    "presence",
    "gaze",
    "gesture",
    "speech_timing",
    "spatial_audio",
}

SARAHNET_RT_FORBIDDEN_DELTA_KEYS = {
    "wallet", "balance", "token", "tokens", "ownership", "owner_change",
    "permission", "permissions", "authority", "grant", "revoke", "ledger",
    "persistent_create", "persistent_delete", "core_patch", "filesystem_write",
    "device_control", "physical_control", "model_authority",
}

SARAHNET_REALITY_CLASSES = {
    "PHYSICAL", "OBSERVED", "EXTERNAL", "MIRRORED", "USER_CREATED",
    "AI_GENERATED", "SIMULATED", "FORKED", "FICTIONAL", "UNKNOWN",
}

SARAHNET_PERSISTENCE_CLASSES = {
    "EPHEMERAL", "SESSION", "CACHED", "PERSISTENT", "LEDGER_PROVEN", "ARCHIVAL",
}

SARAHNET_ENTITY_STATUSES = {"active", "suspended", "archived", "tombstoned"}


def sml_build_sarahnet_entity_contract(
    *,
    entity_id: str,
    entity_type: str,
    world_id: str,
    region_id: str,
    owner_identity: str,
    creator_identity: str,
    owner_node: str,
    semantic_type: str = "object",
    parent_entity: str = "",
    persistence_class: str = "PERSISTENT",
    reality_class: str = "USER_CREATED",
    transform: Optional[Mapping[str, Any]] = None,
    permissions: Optional[Mapping[str, Any]] = None,
    asset_references: Optional[Sequence[str]] = None,
    state: Optional[Mapping[str, Any]] = None,
    state_version: int = 1,
    authority_region: str = "",
    ledger_reference: str = "",
    provenance: Optional[Mapping[str, Any]] = None,
    status: str = "active",
) -> Dict[str, Any]:
    """Normalize a persistent SarahNet entity candidate without storing or authorizing it."""
    try:
        normalized_version = max(1, int(state_version or 1))
    except (TypeError, ValueError):
        normalized_version = 1
    return {
        "schema": SARAHNET_ENTITY_SCHEMA,
        "entity_id": str(entity_id or "").strip(),
        "entity_type": str(entity_type or "").strip().lower(),
        "world_id": str(world_id or "").strip(),
        "region_id": str(region_id or "").strip(),
        "owner_identity": str(owner_identity or "").strip(),
        "creator_identity": str(creator_identity or "").strip(),
        "owner_node": str(owner_node or "").strip(),
        "semantic_type": str(semantic_type or "object").strip().lower(),
        "parent_entity": str(parent_entity or "").strip(),
        "transform": dict(transform or {}),
        "permissions": dict(permissions or {}),
        "persistence_class": str(persistence_class or "PERSISTENT").strip().upper(),
        "reality_class": str(reality_class or "USER_CREATED").strip().upper(),
        "asset_references": [str(x).strip() for x in (asset_references or []) if str(x).strip()][:128],
        "state": dict(state or {}),
        "state_version": normalized_version,
        "authority_region": str(authority_region or region_id or "").strip(),
        "ledger_reference": str(ledger_reference or "").strip(),
        "provenance": dict(provenance or {}),
        "status": str(status or "active").strip().lower(),
        "execution_authority": False,
    }


def sml_validate_sarahnet_entity_contract(entity: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate canonical entity semantics; database ownership checks remain in appnet2."""
    data = dict(entity or {})
    issues: List[str] = []
    if str(data.get("schema") or "") != SARAHNET_ENTITY_SCHEMA:
        issues.append("invalid entity schema")
    for field_name in ("entity_id", "entity_type", "world_id", "region_id", "owner_identity", "creator_identity", "owner_node"):
        if not str(data.get(field_name) or "").strip():
            issues.append(f"missing {field_name}")
    persistence = str(data.get("persistence_class") or "").strip().upper()
    if persistence not in SARAHNET_PERSISTENCE_CLASSES:
        issues.append("invalid persistence_class")
    reality = str(data.get("reality_class") or "").strip().upper()
    if reality not in SARAHNET_REALITY_CLASSES:
        issues.append("invalid reality_class")
    status = str(data.get("status") or "").strip().lower()
    if status not in SARAHNET_ENTITY_STATUSES:
        issues.append("invalid entity status")
    for field_name in ("transform", "permissions", "state", "provenance"):
        if not isinstance(data.get(field_name), Mapping):
            issues.append(f"{field_name} must be an object")
    if not isinstance(data.get("asset_references"), list):
        issues.append("asset_references must be a list")
    forbidden_state = sorted({str(key).strip().lower() for key in (data.get("state") or {}).keys()} & SARAHNET_RT_FORBIDDEN_DELTA_KEYS)
    if forbidden_state:
        issues.append("consequential state keys require dedicated Full SML transition: " + ",".join(forbidden_state))
    try:
        encoded_size = len(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    except Exception:
        encoded_size = 2 * 1024 * 1024
    if encoded_size > 256 * 1024:
        issues.append("entity contract exceeds 256 KiB limit")
    return {
        "ok": not issues,
        "issues": issues,
        "normalized": data,
        "requires_full_sml": True,
        "execution_authority": False,
    }


def sml_build_sarahnet_world_context(
    packet: Union[SMLPacket, Mapping[str, Any]],
    *,
    world_id: str,
    region_id: str = "",
    entity_id: str = "",
    reality_class: str = "UNKNOWN",
    provenance: Optional[Mapping[str, Any]] = None,
) -> SMLPacket:
    """Attach SarahNet semantic coordinates without performing network I/O.

    This function only extends packet semantics. It does not grant authority,
    issue a lease, create a world, or synchronize state.
    """
    pkt = packet if isinstance(packet, SMLPacket) else SMLPacket.from_dict(packet)
    rc = str(reality_class or "UNKNOWN").upper()
    if rc not in SARAHNET_REALITY_CLASSES:
        rc = "UNKNOWN"
    ext = dict(pkt.extensions.get("sarahnet") or {})
    ext.update({
        "schema": SARAHNET_WORLD_CONTEXT_SCHEMA,
        "world_id": str(world_id or "").strip(),
        "region_id": str(region_id or "").strip(),
        "entity_id": str(entity_id or "").strip(),
        "reality_class": rc,
        "provenance": dict(provenance or {}),
        "execution_authority": False,
    })
    pkt.extensions["sarahnet"] = ext
    pkt.add_history(MODULE_NAME, "attach_sarahnet_world_context", pkt.current_omega, f"world={ext['world_id']};region={ext['region_id']}")
    pkt.seal()
    return pkt


def sml_build_sarahnet_authority_lease_contract(
    *,
    identity: str,
    entity_id: str,
    world_id: str,
    region_id: str,
    permitted_state_classes: Sequence[str],
    start_ts: Optional[float] = None,
    expires_ts: Optional[float] = None,
    constraints: Optional[Mapping[str, Any]] = None,
    revocation_conditions: Optional[Sequence[str]] = None,
    lease_id: str = "",
    issuer_node: str = "",
) -> Dict[str, Any]:
    """Build a non-authorizing lease *contract candidate* for governance review.

    The returned object is not a granted lease. appnet2/governance owns issuance;
    this protocol helper only normalizes semantic fields and marks the candidate
    as execution_authority=False.
    """
    now = float(start_ts if start_ts is not None else time.time())
    expiry = float(expires_ts if expires_ts is not None else (now + 300.0))
    classes = sorted({str(x or "").strip().lower() for x in permitted_state_classes if str(x or "").strip().lower() in SARAHNET_RT_STATE_CLASSES})
    return {
        "schema": SARAHNET_AUTHORITY_LEASE_SCHEMA,
        "lease_id": str(lease_id or "").strip(),
        "identity": str(identity or "").strip(),
        "entity_id": str(entity_id or "").strip(),
        "world_id": str(world_id or "").strip(),
        "region_id": str(region_id or "").strip(),
        "permitted_state_classes": classes,
        "constraints": dict(constraints or {}),
        "start_ts": now,
        "expires_ts": expiry,
        "revocation_conditions": [str(x) for x in (revocation_conditions or []) if str(x).strip()],
        "issuer_node": str(issuer_node or "").strip(),
        "status": "candidate",
        "signature": "",
        "execution_authority": False,
    }


def sml_validate_sarahnet_authority_lease(lease: Mapping[str, Any], *, at_time: Optional[float] = None) -> Dict[str, Any]:
    """Validate lease semantics only; cryptographic identity remains external."""
    data = dict(lease or {})
    issues: List[str] = []
    now = float(at_time if at_time is not None else time.time())
    schema = str(data.get("schema") or "")
    if schema and schema != SARAHNET_AUTHORITY_LEASE_SCHEMA:
        issues.append("unsupported lease schema")
    for field_name in ("lease_id", "identity", "entity_id", "world_id", "region_id"):
        if not str(data.get(field_name) or "").strip():
            issues.append(f"missing {field_name}")
    status = str(data.get("status") or "").strip().lower()
    if status not in ("active", "candidate"):
        issues.append("lease is not active/candidate")
    try:
        start_ts = float(data.get("start_ts") or 0.0)
        expires_ts = float(data.get("expires_ts") or 0.0)
    except Exception:
        start_ts = 0.0
        expires_ts = 0.0
        issues.append("invalid lease timestamps")
    if start_ts and now < start_ts:
        issues.append("lease has not started")
    if not expires_ts or now >= expires_ts:
        issues.append("lease expired")
    classes = {str(x or "").strip().lower() for x in (data.get("permitted_state_classes") or [])}
    if not classes:
        issues.append("no permitted state classes")
    unsupported = sorted(classes - SARAHNET_RT_STATE_CLASSES)
    if unsupported:
        issues.append("unsupported state classes: " + ",".join(unsupported))
    return {
        "ok": not issues,
        "issues": issues,
        "lease_id": str(data.get("lease_id") or ""),
        "state_classes": sorted(classes & SARAHNET_RT_STATE_CLASSES),
        "requires_full_sml": bool(issues),
        "execution_authority": False,
    }


def sml_validate_sarahnet_rt_envelope(
    envelope: Mapping[str, Any],
    *,
    lease: Optional[Mapping[str, Any]] = None,
    at_time: Optional[float] = None,
) -> Dict[str, Any]:
    """Lightweight deterministic SML-RT structural/lease validation.

    This is deliberately not the Full SML pipeline. It verifies that an already
    authorized real-time update remains inside the bounded state contract.
    """
    data = dict(envelope or {})
    issues: List[str] = []
    profile = str(data.get("profile") or data.get("protocol_profile") or "")
    if profile != SARAHNET_RT_PROFILE:
        issues.append("invalid SML-RT profile")
    for field_name in ("world_id", "region_id", "entity_id", "owner_identity", "lease_id", "state_class"):
        if not str(data.get(field_name) or "").strip():
            issues.append(f"missing {field_name}")
    state_class = str(data.get("state_class") or "").strip().lower()
    if state_class not in SARAHNET_RT_STATE_CLASSES:
        issues.append("state class requires Full SML or is unsupported")
    try:
        seq = int(data.get("sequence_number"))
        if seq <= 0:
            raise ValueError
    except Exception:
        seq = 0
        issues.append("invalid sequence_number")
    delta = data.get("delta_payload")
    if not isinstance(delta, Mapping):
        issues.append("delta_payload must be an object")
        delta_map: Dict[str, Any] = {}
    else:
        delta_map = dict(delta)
    forbidden = sorted({str(k).strip().lower() for k in delta_map.keys()} & SARAHNET_RT_FORBIDDEN_DELTA_KEYS)
    if forbidden:
        issues.append("consequential fields require Full SML: " + ",".join(forbidden))
    try:
        delta_size = len(json.dumps(delta_map, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    except Exception:
        delta_size = SARAHNET_RT_MAX_DELTA_BYTES + 1
    if delta_size > SARAHNET_RT_MAX_DELTA_BYTES:
        issues.append("delta_payload exceeds SML-RT size limit")

    lease_result: Dict[str, Any] = {"ok": True, "issues": []}
    if lease is not None:
        lease_data = dict(lease or {})
        lease_result = sml_validate_sarahnet_authority_lease(lease_data, at_time=at_time)
        if not lease_result.get("ok"):
            issues.extend([f"lease: {x}" for x in (lease_result.get("issues") or [])])
        pairs = (
            ("lease_id", "lease_id"),
            ("owner_identity", "identity"),
            ("entity_id", "entity_id"),
            ("world_id", "world_id"),
            ("region_id", "region_id"),
        )
        for env_key, lease_key in pairs:
            if str(data.get(env_key) or "") != str(lease_data.get(lease_key) or ""):
                issues.append(f"lease mismatch: {env_key}")
        permitted = {str(x or "").strip().lower() for x in (lease_data.get("permitted_state_classes") or [])}
        if state_class and state_class not in permitted:
            issues.append("state class not permitted by lease")

    normalized = {
        "profile": SARAHNET_RT_PROFILE,
        "world_id": str(data.get("world_id") or "").strip(),
        "region_id": str(data.get("region_id") or "").strip(),
        "entity_id": str(data.get("entity_id") or "").strip(),
        "owner_identity": str(data.get("owner_identity") or "").strip(),
        "lease_id": str(data.get("lease_id") or "").strip(),
        "sequence_number": seq,
        "timestamp": float(data.get("timestamp") or data.get("ts") or (at_time if at_time is not None else time.time())),
        "state_class": state_class,
        "delta_payload": delta_map,
        "integrity_tag": str(data.get("integrity_tag") or ""),
        "execution_authority": False,
    }
    return {
        "ok": not issues,
        "issues": issues,
        "normalized": normalized,
        "lease": lease_result,
        "requires_full_sml": bool(issues),
        "execution_authority": False,
    }


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
    "QSML_LANGUAGE_VERSION",
    "QSML_LANGUAGE_NAME",
    "SML_TYPE_SYSTEM_VERSION",
    "SMLStatus",
    "IdentityRole",
    "MissionType",
    "CognitiveState",
    "GovernanceDecision",
    "QMathState",
    "SMLStopCondition",
    "GCOPEventType",
    "GCOPCycleStatus",
    "HealthStatus",
    "OrganCategory",
    "Authority",
    "ErrorClass",
    "SMLDataType",
    "SMLSemanticType",
    "SMLScope",
    "SMLMutability",
    "SMLASTKind",
    "SMLCompileStatus",
    "SMLSynthesisPhase",
    "SMLArtifactRole",
    "SMLVariable",
    "SMLVariableRegistry",
    "SMLASTNode",
    "SMLOrganContract",
    "SMLRequirement",
    "SMLFilePlan",
    "SMLApplicationBlueprint",
    "SMLProgram",
    "SMLCompileResult",
    "SMLValidationIssue",
    "SMLHealthVector",
    "SMLDiagnosticsReport",
    "SMLOmegaTransition",
    "SMLOrganMetadata",
    "SMLPacket",
    "SMLRouteResult",
    "SMLCognitiveEvent",
    "SMLCognitiveContinuityState",
    "SMLCognitiveCycleResult",
    "SarahMemorySMLProtocol",
    "sml_packet_summary",
    "sml_build_ingress_packet",
    "sml_apply_governor_result",
    "sml_touch_packet",
    "sml_attach_bundle_meta",
    "sml_resolve_safe_cognitive_answer",
    "sml_build_dynamic_claim_vector",
    "sml_build_source_authority_court_packet",
    "sml_normalize_evidence_artifact",
    "sml_normalize_evidence_artifacts",
    "sml_adjudicate_evidence_artifacts",
    "sml_build_evidence_court_packet",
    "sml_build_creation_mission_contract",
    "sml_build_avatar_event_packet",
    "sml_apply_cognitive_grammar",
    "sml_compile_natural_program",
    "sml_evaluate_qmath",
    "sml_register_organ_contract",
    "sml_application_blueprint_schema",
    "sml_validate_application_blueprint",
    "sml_compile_application_blueprint",
    "GCAIOS_PROFILE",
    "sml_gcaios_manifest",
    "SARAHNET_RT_PROFILE",
    "SARAHNET_AUTHORITY_LEASE_SCHEMA",
    "SARAHNET_WORLD_CONTEXT_SCHEMA",
    "SARAHNET_ENTITY_SCHEMA",
    "SARAHNET_RT_STATE_CLASSES",
    "SARAHNET_REALITY_CLASSES",
    "SARAHNET_PERSISTENCE_CLASSES",
    "SARAHNET_ENTITY_STATUSES",
    "sml_build_sarahnet_entity_contract",
    "sml_validate_sarahnet_entity_contract",
    "sml_build_sarahnet_world_context",
    "sml_build_sarahnet_authority_lease_contract",
    "sml_validate_sarahnet_authority_lease",
    "sml_validate_sarahnet_rt_envelope",
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
    "sml_build_continuity_state",
    "sml_filter_legal_candidates",
    "sml_run_gcop_cycle",
    "sml_gcop_self_test",
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
