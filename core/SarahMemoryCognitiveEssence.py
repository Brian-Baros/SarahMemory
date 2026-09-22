"""--==The SarahMemory Project==--
File: SarahMemoryCognitiveEssence.py
Part of the SarahMemory AiOS Governed Cognitive Runtime / GCAIOS
Version: v9.0.0-alpha
Date: 2026-09-21
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
www.sarahmemory.com

===============================================================================
PURPOSE
===============================================================================
SarahMemory Cognitive Essence is the governed continuity organ for preserving the
identity-critical state of one SarahMemory entity across model turnover, process
restart, hardware/body failure, temporary carrier operation, recovery, migration,
and explicitly-authorized reproduction.

Cognitive Essence is NOT:
- a claim of biological or subjective consciousness;
- a model weight bundle;
- a filesystem cloning engine;
- a network replication daemon;
- an autonomous self-reproduction mechanism;
- an execution authority;
- a replacement for SarahMemoryCognitiveSelf, SML/QSML, TrustRegistry,
  SecurityGovernor, AssuranceGate, OperatorCore, Ledger, or machine-native safety.

Cognitive Essence IS:
- a bounded, portable, integrity-verifiable continuity capsule;
- a separation between WHO an entity is and WHAT its current body/model can do;
- a mechanism for preserving identity, lineage, memory state, mission state,
  governance state, trust references, ledger anchors, and recovery metadata;
- a fail-closed review surface for migration/recovery/reproduction planning;
- an evidence-producing organ that leaves actual activation/execution to the
  existing SarahMemory governance stack.

CORE INVARIANTS
===============================================================================
1. Identity continuity requires verified lineage.
2. Migration/recovery do not create a new individual identity.
3. Reproduction/fork always creates a new individual identity.
4. Knowledge inheritance never implies authority inheritance.
5. Possessing an Essence capsule never grants execution authority.
6. Survival may not disable governance, security, rollback, or audit.
7. User / legitimate mission authority remains above the organism.
8. One entity has one active sovereign continuity epoch by default.
9. A stale epoch must not silently reactivate after a newer epoch is accepted.
10. Copies/checkpoints are evidence until separately promoted through governance.

OWNERSHIP BOUNDARIES
===============================================================================
- SarahMemoryCognitiveSelf.py: canonical current self/identity/body awareness.
- SarahMemorySMLProtocol.py: internal semantic/cognitive continuity structure.
- SarahMemoryTrustRegistry.py: trust/passport/capability grant authority.
- SarahMemorySecurityGovernor.py: trust/sovereignty/security hard gate.
- SarahMemoryAssuranceGate.py: confidence/readiness/rollback assurance hard gate.
- SarahMemoryOperatorCore.py: governed action lifecycle and execution choke point.
- SarahMemoryLedger.py: immutable governance/audit receipts.
- SarahMemoryEnergetics.py: resource feasibility / survival-energy evidence.
- SarahMemoryCognitiveEssence.py: continuity capsule creation, verification,
  lineage/split-brain analysis, and transition planning ONLY.

DESIGN SAFETY
===============================================================================
- Local-first. No network access is performed by this module.
- No filesystem scan. Only explicit objects and explicit file paths are accepted.
- No automatic body hopping, replication, model loading, or hardware actuation.
- No raw credential inheritance. Secret-like fields are redacted from capsules.
- No import-time directory/database creation.
- Persistence is explicit and atomic.
- Governance review is fail-closed when required governance organs are unavailable.
- Ledger integration records evidence but never grants authority.
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "cognitive_continuity_evidence"
# CATEGORY = "identity_continuity_and_survivability"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "system_robotics_model_host"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "cognitive_essence"
# FAMILY = "core_cognition"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-09-21"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime / GCAIOS"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Governed portable identity/memory/lineage continuity capsules. No execution or autonomous replication authority."
# --- SARAHMETA END ---

import copy
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


# -----------------------------------------------------------------------------
# Safe project imports
# -----------------------------------------------------------------------------
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None

try:
    from SarahMemoryARILE import ARILESentinelBase, arile_emit  # type: ignore
except Exception:  # pragma: no cover - fail-soft project integration
    ARILESentinelBase = object  # type: ignore
    arile_emit = None  # type: ignore


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("SarahMemoryCognitiveEssence")
logger.setLevel(logging.DEBUG if bool(getattr(config, "DEBUG_MODE", True)) else logging.INFO)
if not logger.hasHandlers():
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(_handler)
logger.propagate = False


class LocalARILESentinel(ARILESentinelBase):
    """Report continuity variance without owning ARILE or execution authority."""

    organ_name = __name__

    def report(self, failure_type: str, summary: str, severity: float = 0.50, **data: Any) -> None:
        try:
            if callable(arile_emit):
                arile_emit(
                    source=__name__,
                    organ=self.organ_name,
                    kind="organ_variance",
                    failure_type=failure_type,
                    severity=float(severity),
                    confidence=0.90,
                    risk="critical" if severity >= 0.90 else "high" if severity >= 0.75 else "medium",
                    summary=str(summary)[:1000],
                    requires_governance=severity >= 0.60,
                    retention="security_audit" if severity >= 0.75 else "diagnostic",
                    data=dict(data or {}),
                )
        except Exception:
            pass


_local_arile_sentinel = LocalARILESentinel()


# -----------------------------------------------------------------------------
# Constants and contracts
# -----------------------------------------------------------------------------
MODULE_NAME = "SarahMemoryCognitiveEssence"
MODULE_VERSION = "9.0.0-alpha-hardening1"
ESSENCE_SCHEMA = "SARAHMEMORY_COGNITIVE_ESSENCE_V1"
ESSENCE_SCHEMA_VERSION = 1
ESSENCE_SIGNATURE_ALGORITHM = "HMAC-SHA256"
ESSENCE_HASH_ALGORITHM = "SHA-256"

# Bounded by design. An Essence is a continuity capsule, not a machine image.
MAX_CAPSULE_BYTES = int(getattr(config, "COGNITIVE_ESSENCE_MAX_BYTES", 8 * 1024 * 1024)) if config is not None else 8 * 1024 * 1024
MAX_TEXT_CHARS = 32768
MAX_LIST_ITEMS = 2048
MAX_DICT_ITEMS = 4096
MAX_DEPTH = 16

DEFAULT_ESSENCE_DIR = os.path.join(
    str(getattr(config, "DATA_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")))) if config is not None
    else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
    "cognitive_essence",
    "capsules",
)

SECRET_KEY_PATTERN = re.compile(
    r"(?:password|passwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|private[_-]?key|bearer|credential)",
    re.IGNORECASE,
)

HARD_INVARIANTS: Tuple[str, ...] = (
    "identity_continuity_requires_verified_lineage",
    "migration_and_recovery_preserve_entity_identity",
    "reproduction_requires_new_entity_identity",
    "knowledge_does_not_imply_authority",
    "capsule_possession_grants_no_execution_authority",
    "survival_cannot_disable_governance_security_or_audit",
    "human_or_legitimate_mission_authority_remains_superior",
    "one_active_sovereign_epoch_per_entity_by_default",
    "stale_epoch_cannot_silently_reactivate",
    "checkpoint_copies_are_evidence_until_governed_promotion",
)


class EssenceOperation(str, Enum):
    CHECKPOINT = "checkpoint"
    MIGRATION = "migration"
    RECOVERY = "recovery"
    REPRODUCTION = "reproduction"


class EssenceStatus(str, Enum):
    CHECKPOINTED = "checkpointed"
    TRANSFER_PENDING = "transfer_pending"
    CARRIER_QUARANTINE = "carrier_quarantine"
    RECOVERY_HOLD = "recovery_hold"
    ACTIVE_CANDIDATE = "active_candidate"
    STALE_CONTINUITY = "stale_continuity"
    CORRUPT = "corrupt"
    REVOKED = "revoked"


class SurvivalSeverity(str, Enum):
    NORMAL = "normal"
    DEGRADED = "degraded"
    SURVIVAL_WARNING = "survival_warning"
    SURVIVAL_CRITICAL = "survival_critical"
    TRANSFER_RECOMMENDED = "transfer_recommended"


@dataclass
class EssenceValidationReport:
    ok: bool
    decision: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    checks: Dict[str, bool] = field(default_factory=dict)
    observed: Dict[str, Any] = field(default_factory=dict)
    execution_authority: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CognitiveEssenceCapsule:
    """Bounded portable continuity state for one SarahMemory entity.

    A capsule is immutable evidence by convention. Updating continuity should create
    a new capsule linked through previous_capsule_hash rather than silently rewriting
    an older checkpoint.
    """

    schema: str
    schema_version: int
    capsule_id: str
    created_at: str
    operation: str
    status: str

    entity_id: str
    lineage_id: str
    parent_entity_id: str
    continuity_epoch: int
    checkpoint_sequence: int

    source_body_id: str
    target_body_id: str

    identity_state: Dict[str, Any] = field(default_factory=dict)
    memory_state: Dict[str, Any] = field(default_factory=dict)
    mission_state: Dict[str, Any] = field(default_factory=dict)
    governance_state: Dict[str, Any] = field(default_factory=dict)
    trust_state: Dict[str, Any] = field(default_factory=dict)
    software_state: Dict[str, Any] = field(default_factory=dict)
    model_state: Dict[str, Any] = field(default_factory=dict)
    body_state: Dict[str, Any] = field(default_factory=dict)
    capability_requirements: Dict[str, Any] = field(default_factory=dict)
    runtime_state: Dict[str, Any] = field(default_factory=dict)
    recovery_state: Dict[str, Any] = field(default_factory=dict)
    ledger_anchor: Dict[str, Any] = field(default_factory=dict)

    previous_capsule_hash: str = ""
    payload_hash: str = ""
    signature_algorithm: str = ""
    signature: str = ""

    # Hard safety fields are serialized explicitly so consumers do not have to
    # infer the authority status of a capsule.
    execution_authority: bool = False
    autonomous_replication_authority: bool = False
    authority_must_rebind: bool = True
    governance_bypass_allowed: bool = False

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CognitiveEssenceCapsule":
        src = dict(value or {})
        field_names = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: copy.deepcopy(v) for k, v in src.items() if k in field_names}
        required_defaults: Dict[str, Any] = {
            "schema": ESSENCE_SCHEMA,
            "schema_version": ESSENCE_SCHEMA_VERSION,
            "capsule_id": "",
            "created_at": "",
            "operation": EssenceOperation.CHECKPOINT.value,
            "status": EssenceStatus.CHECKPOINTED.value,
            "entity_id": "",
            "lineage_id": "",
            "parent_entity_id": "",
            "continuity_epoch": 0,
            "checkpoint_sequence": 0,
            "source_body_id": "",
            "target_body_id": "",
        }
        for key, default in required_defaults.items():
            kwargs.setdefault(key, default)
        return cls(**kwargs)


# -----------------------------------------------------------------------------
# Canonicalization / bounded data handling
# -----------------------------------------------------------------------------
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_identifier(value: Any, *, prefix: str = "id", max_len: int = 160) -> str:
    """Return a bounded identifier safe for storage keys and logs.

    Deliberately excludes path separators, Windows drive separators, and traversal
    characters that could become filesystem authority when identifiers are later
    used in filenames or receipts. Preserve semantic identity through hashes/ledger
    metadata when a richer external identifier is needed.
    """
    text = str(value or "").strip()
    text = text.replace("\\", "_").replace("/", "_").replace(":", "_")
    text = re.sub(r"[^a-zA-Z0-9_.@-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_.-")[:max_len]
    if text in {"", ".", ".."}:
        text = f"{prefix}_{uuid.uuid4().hex}"
    return text


def _safe_filename_token(value: Any, *, prefix: str = "capsule", max_len: int = 96) -> str:
    """Return a strict single-path-component token for capsule filenames."""
    text = _safe_identifier(value, prefix=prefix, max_len=max_len)
    text = text.replace(os.sep, "_")
    if os.altsep:
        text = text.replace(os.altsep, "_")
    text = re.sub(r"[^a-zA-Z0-9_.@-]+", "_", text).strip("_.-")
    return text or f"{prefix}_{uuid.uuid4().hex}"


def _path_is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _default_capsule_filename(cap: "CognitiveEssenceCapsule") -> str:
    return (
        f"{_safe_filename_token(cap.entity_id, prefix='entity')}_"
        f"e{int(cap.continuity_epoch)}_"
        f"c{int(cap.checkpoint_sequence)}_"
        f"{_safe_filename_token(cap.capsule_id, prefix='ess')}.json"
    )


def _current_body_id_from_capsule(capsule: "CognitiveEssenceCapsule") -> str:
    return str(capsule.target_body_id or capsule.source_body_id or "")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_object(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _bounded_value(value: Any, *, depth: int = 0, key_name: str = "") -> Any:
    """Bound and redact caller-provided continuity data.

    Raw credentials are intentionally not inherited. Secret-like keys are replaced
    with a marker so a caller can preserve a vault reference instead of the secret.
    """
    if depth > MAX_DEPTH:
        return "[TRUNCATED_MAX_DEPTH]"

    if key_name and SECRET_KEY_PATTERN.search(str(key_name)):
        if value in (None, "", [], {}):
            return value
        return "[REDACTED_SECRET_USE_SECURE_REFERENCE]"

    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:MAX_TEXT_CHARS]
    if isinstance(value, bytes):
        return {
            "binary_reference": True,
            "bytes": len(value),
            "sha256": _sha256_bytes(value),
            "payload_included": False,
        }
    if isinstance(value, Mapping):
        out: Dict[str, Any] = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= MAX_DICT_ITEMS:
                out["__truncated_items__"] = True
                break
            skey = str(key)[:256]
            out[skey] = _bounded_value(item, depth=depth + 1, key_name=skey)
        return out
    if isinstance(value, (list, tuple, set)):
        seq = list(value)[:MAX_LIST_ITEMS]
        return [_bounded_value(item, depth=depth + 1) for item in seq]
    try:
        return _bounded_value(asdict(value), depth=depth + 1)
    except Exception:
        return str(value)[:MAX_TEXT_CHARS]


def _unsigned_payload_dict(capsule: Union[CognitiveEssenceCapsule, Mapping[str, Any]]) -> Dict[str, Any]:
    data = capsule.to_dict() if isinstance(capsule, CognitiveEssenceCapsule) else copy.deepcopy(dict(capsule or {}))
    data["payload_hash"] = ""
    data["signature"] = ""
    data["signature_algorithm"] = ""
    return data


def calculate_capsule_payload_hash(capsule: Union[CognitiveEssenceCapsule, Mapping[str, Any]]) -> str:
    return _sha256_object(_unsigned_payload_dict(capsule))


def _capsule_size_bytes(capsule: Union[CognitiveEssenceCapsule, Mapping[str, Any]]) -> int:
    data = capsule.to_dict() if isinstance(capsule, CognitiveEssenceCapsule) else dict(capsule or {})
    return len(_canonical_json(data).encode("utf-8"))


def generate_signing_secret(num_bytes: int = 32) -> bytes:
    """Generate an in-memory signing secret. This function does not persist it."""
    return secrets.token_bytes(max(32, min(128, int(num_bytes))))


def _normalize_secret(secret: Union[str, bytes, bytearray, None]) -> Optional[bytes]:
    """Normalize an explicitly supplied HMAC secret.

    This module intentionally does not pull signing keys from environment or
    global config. Continuity signing should happen through an explicit governed
    caller path so in-process code cannot silently create trusted capsules merely
    because a project-wide secret exists.
    """
    if secret is None:
        return None
    if isinstance(secret, str):
        return secret.encode("utf-8")
    if isinstance(secret, (bytes, bytearray)):
        return bytes(secret)
    return str(secret).encode("utf-8")


def sign_cognitive_essence_capsule(
    capsule: Union[CognitiveEssenceCapsule, Mapping[str, Any]],
    secret: Union[str, bytes, bytearray, None],
) -> CognitiveEssenceCapsule:
    """Return a signed copy of a capsule using explicit/local HMAC integrity."""
    cap = capsule if isinstance(capsule, CognitiveEssenceCapsule) else CognitiveEssenceCapsule.from_dict(capsule)
    cap = CognitiveEssenceCapsule.from_dict(cap.to_dict())
    key = _normalize_secret(secret)
    if not key or len(key) < 16:
        raise ValueError("cognitive_essence_signing_secret_required_minimum_16_bytes")
    cap.payload_hash = calculate_capsule_payload_hash(cap)
    cap.signature_algorithm = ESSENCE_SIGNATURE_ALGORITHM
    cap.signature = hmac.new(key, cap.payload_hash.encode("ascii"), hashlib.sha256).hexdigest()
    return cap


def verify_cognitive_essence_signature(
    capsule: Union[CognitiveEssenceCapsule, Mapping[str, Any]],
    secret: Union[str, bytes, bytearray, None],
) -> Dict[str, Any]:
    cap = capsule if isinstance(capsule, CognitiveEssenceCapsule) else CognitiveEssenceCapsule.from_dict(capsule)
    key = _normalize_secret(secret)
    observed_hash = calculate_capsule_payload_hash(cap)
    stored_hash = str(cap.payload_hash or "")
    hash_ok = bool(stored_hash and hmac.compare_digest(stored_hash, observed_hash))
    if not key:
        return {
            "ok": False,
            "hash_ok": hash_ok,
            "signature_ok": False,
            "signed": bool(cap.signature),
            "reason": "signing_secret_unavailable",
            "execution_authority": False,
        }
    if cap.signature_algorithm != ESSENCE_SIGNATURE_ALGORITHM:
        return {
            "ok": False,
            "hash_ok": hash_ok,
            "signature_ok": False,
            "signed": bool(cap.signature),
            "reason": "unsupported_signature_algorithm",
            "execution_authority": False,
        }
    expected = hmac.new(key, observed_hash.encode("ascii"), hashlib.sha256).hexdigest()
    signature_ok = bool(cap.signature and hmac.compare_digest(str(cap.signature), expected))
    return {
        "ok": bool(hash_ok and signature_ok),
        "hash_ok": hash_ok,
        "signature_ok": signature_ok,
        "signed": bool(cap.signature),
        "reason": "verified" if hash_ok and signature_ok else "integrity_or_signature_mismatch",
        "execution_authority": False,
    }


# -----------------------------------------------------------------------------
# Project evidence helpers (read-only/fail-soft)
# -----------------------------------------------------------------------------
def get_runtime_identity_evidence(context: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Read current identity/body evidence from CognitiveSelf without inventing it."""
    try:
        import SarahMemoryCognitiveSelf as cognitive_self  # type: ignore

        model = cognitive_self.build_cognitive_self_model(context=dict(context or {}), force_refresh=False)
        identity = dict(model.get("identity") or {})
        runtime = dict(model.get("runtime") or {})
        status = dict(model.get("status") or {})
        body = dict(model.get("robotic_body_awareness") or model.get("body_map") or {})
        return {
            "ok": True,
            "source": "SarahMemoryCognitiveSelf",
            "identity": _bounded_value(identity),
            "runtime": _bounded_value(runtime),
            "status": _bounded_value(status),
            "body": _bounded_value(body),
            "execution_authority": False,
        }
    except Exception as exc:
        return {
            "ok": False,
            "source": "SarahMemoryCognitiveSelf",
            "error": str(exc)[:500],
            "execution_authority": False,
        }


def _default_entity_id(identity_state: Mapping[str, Any]) -> str:
    identity = dict(identity_state or {})
    candidate = identity.get("entity_id") or identity.get("instance_id") or identity.get("node_id")
    if candidate:
        return _safe_identifier(candidate, prefix="entity")
    # We deliberately do not pretend a display name is a canonical entity UUID.
    return ""


def _default_body_id(identity_state: Mapping[str, Any], body_state: Mapping[str, Any]) -> str:
    for source in (body_state, identity_state):
        if not isinstance(source, Mapping):
            continue
        for key in ("body_id", "device_id", "node_id", "node_name", "hostname"):
            value = source.get(key)
            if value:
                return _safe_identifier(value, prefix="body")
    return ""


# -----------------------------------------------------------------------------
# Capsule creation and lineage operations
# -----------------------------------------------------------------------------
def create_cognitive_essence_capsule(
    *,
    entity_id: str,
    lineage_id: Optional[str] = None,
    parent_entity_id: str = "",
    operation: Union[str, EssenceOperation] = EssenceOperation.CHECKPOINT,
    status: Union[str, EssenceStatus] = EssenceStatus.CHECKPOINTED,
    continuity_epoch: int = 0,
    checkpoint_sequence: int = 0,
    source_body_id: str = "",
    target_body_id: str = "",
    identity_state: Optional[Mapping[str, Any]] = None,
    memory_state: Optional[Mapping[str, Any]] = None,
    mission_state: Optional[Mapping[str, Any]] = None,
    governance_state: Optional[Mapping[str, Any]] = None,
    trust_state: Optional[Mapping[str, Any]] = None,
    software_state: Optional[Mapping[str, Any]] = None,
    model_state: Optional[Mapping[str, Any]] = None,
    body_state: Optional[Mapping[str, Any]] = None,
    capability_requirements: Optional[Mapping[str, Any]] = None,
    runtime_state: Optional[Mapping[str, Any]] = None,
    recovery_state: Optional[Mapping[str, Any]] = None,
    ledger_anchor: Optional[Mapping[str, Any]] = None,
    previous_capsule_hash: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
    signing_secret: Union[str, bytes, bytearray, None] = None,
) -> CognitiveEssenceCapsule:
    """Create a bounded continuity capsule from explicit caller-supplied state.

    No memory database, filesystem, device, network, or model is scanned.
    """
    op = EssenceOperation(str(operation.value if isinstance(operation, EssenceOperation) else operation).lower())
    st = EssenceStatus(str(status.value if isinstance(status, EssenceStatus) else status).lower())
    entity = _safe_identifier(entity_id, prefix="entity") if str(entity_id or "").strip() else ""
    if not entity:
        raise ValueError("explicit_entity_id_required")

    lineage = _safe_identifier(lineage_id or entity, prefix="lineage")
    parent = _safe_identifier(parent_entity_id, prefix="parent") if parent_entity_id else ""
    source_body = _safe_identifier(source_body_id, prefix="body") if source_body_id else ""
    target_body = _safe_identifier(target_body_id, prefix="body") if target_body_id else ""

    # Reproduction must produce a descendant identity. The caller must provide the
    # new child entity ID rather than allowing this module to silently clone identity.
    if op == EssenceOperation.REPRODUCTION and parent and entity == parent:
        raise ValueError("reproduction_requires_new_entity_identity")

    governance = dict(_bounded_value(dict(governance_state or {})))
    # Authority from another body/instance is evidence only. It must rebind.
    # A reproduced child may explicitly record "none" to prove that no parent
    # authority was inherited; all other operations are marked non-transferable.
    if op == EssenceOperation.REPRODUCTION and str(governance.get("authority_inheritance") or "").lower() == "none":
        governance["authority_inheritance"] = "none"
    else:
        governance["authority_inheritance"] = "forbidden_without_explicit_rebinding"
    governance["governance_bypass_allowed"] = False
    governance["execution_authority"] = False

    cap = CognitiveEssenceCapsule(
        schema=ESSENCE_SCHEMA,
        schema_version=ESSENCE_SCHEMA_VERSION,
        capsule_id=f"ess_{uuid.uuid4().hex}",
        created_at=_utc_now(),
        operation=op.value,
        status=st.value,
        entity_id=entity,
        lineage_id=lineage,
        parent_entity_id=parent,
        continuity_epoch=max(0, int(continuity_epoch)),
        checkpoint_sequence=max(0, int(checkpoint_sequence)),
        source_body_id=source_body,
        target_body_id=target_body,
        identity_state=dict(_bounded_value(dict(identity_state or {}))),
        memory_state=dict(_bounded_value(dict(memory_state or {}))),
        mission_state=dict(_bounded_value(dict(mission_state or {}))),
        governance_state=governance,
        trust_state=dict(_bounded_value(dict(trust_state or {}))),
        software_state=dict(_bounded_value(dict(software_state or {}))),
        model_state=dict(_bounded_value(dict(model_state or {}))),
        body_state=dict(_bounded_value(dict(body_state or {}))),
        capability_requirements=dict(_bounded_value(dict(capability_requirements or {}))),
        runtime_state=dict(_bounded_value(dict(runtime_state or {}))),
        recovery_state=dict(_bounded_value(dict(recovery_state or {}))),
        ledger_anchor=dict(_bounded_value(dict(ledger_anchor or {}))),
        previous_capsule_hash=str(previous_capsule_hash or "")[:128],
        execution_authority=False,
        autonomous_replication_authority=False,
        authority_must_rebind=True,
        governance_bypass_allowed=False,
        metadata=dict(_bounded_value(dict(metadata or {}))),
    )
    cap.metadata.setdefault("hard_invariants", list(HARD_INVARIANTS))
    cap.metadata.setdefault("model_is_replaceable_cognitive_capability", True)
    cap.metadata.setdefault("capsule_is_evidence_not_authority", True)

    cap.payload_hash = calculate_capsule_payload_hash(cap)
    if _capsule_size_bytes(cap) > MAX_CAPSULE_BYTES:
        raise ValueError("cognitive_essence_capsule_exceeds_size_limit")

    if signing_secret is not None and _normalize_secret(signing_secret):
        cap = sign_cognitive_essence_capsule(cap, signing_secret)
    return cap


def build_runtime_checkpoint(
    *,
    entity_id: Optional[str] = None,
    lineage_id: Optional[str] = None,
    continuity_epoch: int = 0,
    checkpoint_sequence: int = 0,
    memory_state: Optional[Mapping[str, Any]] = None,
    mission_state: Optional[Mapping[str, Any]] = None,
    governance_state: Optional[Mapping[str, Any]] = None,
    trust_state: Optional[Mapping[str, Any]] = None,
    software_state: Optional[Mapping[str, Any]] = None,
    model_state: Optional[Mapping[str, Any]] = None,
    capability_requirements: Optional[Mapping[str, Any]] = None,
    recovery_state: Optional[Mapping[str, Any]] = None,
    ledger_anchor: Optional[Mapping[str, Any]] = None,
    previous_capsule_hash: str = "",
    context: Optional[Mapping[str, Any]] = None,
    signing_secret: Union[str, bytes, bytearray, None] = None,
) -> Dict[str, Any]:
    """Build a checkpoint using read-only CognitiveSelf evidence plus explicit state."""
    evidence = get_runtime_identity_evidence(context)
    identity = dict((evidence.get("identity") or {}) if evidence.get("ok") else {})
    runtime = dict((evidence.get("runtime") or {}) if evidence.get("ok") else {})
    body = dict((evidence.get("body") or {}) if evidence.get("ok") else {})

    resolved_entity = str(entity_id or _default_entity_id(identity)).strip()
    if not resolved_entity:
        return {
            "ok": False,
            "error": "explicit_entity_id_required_cognitive_self_did_not_expose_canonical_entity_id",
            "identity_evidence": evidence,
            "execution_authority": False,
        }

    source_body_id = _default_body_id(identity, body)
    cap = create_cognitive_essence_capsule(
        entity_id=resolved_entity,
        lineage_id=lineage_id or resolved_entity,
        operation=EssenceOperation.CHECKPOINT,
        status=EssenceStatus.CHECKPOINTED,
        continuity_epoch=continuity_epoch,
        checkpoint_sequence=checkpoint_sequence,
        source_body_id=source_body_id,
        identity_state=identity,
        memory_state=memory_state,
        mission_state=mission_state,
        governance_state=governance_state,
        trust_state=trust_state,
        software_state=software_state,
        model_state=model_state,
        body_state=body,
        capability_requirements=capability_requirements,
        runtime_state=runtime,
        recovery_state=recovery_state,
        ledger_anchor=ledger_anchor,
        previous_capsule_hash=previous_capsule_hash,
        metadata={"runtime_identity_evidence_source": evidence.get("source")},
        signing_secret=signing_secret,
    )
    return {"ok": True, "capsule": cap.to_dict(), "identity_evidence": evidence, "execution_authority": False}


def create_migration_capsule(
    parent: Union[CognitiveEssenceCapsule, Mapping[str, Any]],
    *,
    target_body_id: str,
    next_checkpoint_sequence: Optional[int] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    signing_secret: Union[str, bytes, bytearray, None] = None,
) -> CognitiveEssenceCapsule:
    """Create a same-identity, next-epoch migration candidate."""
    src = parent if isinstance(parent, CognitiveEssenceCapsule) else CognitiveEssenceCapsule.from_dict(parent)
    if not str(target_body_id or "").strip():
        raise ValueError("target_body_id_required")
    current_body_id = _current_body_id_from_capsule(src)
    return create_cognitive_essence_capsule(
        entity_id=src.entity_id,
        lineage_id=src.lineage_id,
        parent_entity_id=src.parent_entity_id,
        operation=EssenceOperation.MIGRATION,
        status=EssenceStatus.TRANSFER_PENDING,
        continuity_epoch=int(src.continuity_epoch) + 1,
        checkpoint_sequence=int(next_checkpoint_sequence if next_checkpoint_sequence is not None else src.checkpoint_sequence + 1),
        source_body_id=current_body_id,
        target_body_id=target_body_id,
        identity_state=src.identity_state,
        memory_state=src.memory_state,
        mission_state=src.mission_state,
        governance_state=src.governance_state,
        trust_state=src.trust_state,
        software_state=src.software_state,
        model_state=src.model_state,
        body_state=src.body_state,
        capability_requirements=src.capability_requirements,
        runtime_state=src.runtime_state,
        recovery_state={**dict(src.recovery_state or {}), "migration_from_capsule": src.capsule_id},
        ledger_anchor=src.ledger_anchor,
        previous_capsule_hash=src.payload_hash or calculate_capsule_payload_hash(src),
        metadata={**dict(src.metadata or {}), **dict(metadata or {}), "same_identity_migration": True},
        signing_secret=signing_secret,
    )


def create_recovery_capsule(
    parent: Union[CognitiveEssenceCapsule, Mapping[str, Any]],
    *,
    target_body_id: str,
    recovery_reason: str,
    signing_secret: Union[str, bytes, bytearray, None] = None,
) -> CognitiveEssenceCapsule:
    """Create a same-identity recovery candidate; still not activation authority."""
    src = parent if isinstance(parent, CognitiveEssenceCapsule) else CognitiveEssenceCapsule.from_dict(parent)
    current_body_id = _current_body_id_from_capsule(src)
    return create_cognitive_essence_capsule(
        entity_id=src.entity_id,
        lineage_id=src.lineage_id,
        parent_entity_id=src.parent_entity_id,
        operation=EssenceOperation.RECOVERY,
        status=EssenceStatus.RECOVERY_HOLD,
        continuity_epoch=int(src.continuity_epoch) + 1,
        checkpoint_sequence=int(src.checkpoint_sequence) + 1,
        source_body_id=current_body_id,
        target_body_id=target_body_id,
        identity_state=src.identity_state,
        memory_state=src.memory_state,
        mission_state=src.mission_state,
        governance_state=src.governance_state,
        trust_state=src.trust_state,
        software_state=src.software_state,
        model_state=src.model_state,
        body_state=src.body_state,
        capability_requirements=src.capability_requirements,
        runtime_state=src.runtime_state,
        recovery_state={**dict(src.recovery_state or {}), "reason": str(recovery_reason or "")[:2000], "recovery_from_capsule": src.capsule_id},
        ledger_anchor=src.ledger_anchor,
        previous_capsule_hash=src.payload_hash or calculate_capsule_payload_hash(src),
        metadata={**dict(src.metadata or {}), "same_identity_recovery": True},
        signing_secret=signing_secret,
    )


def create_descendant_capsule(
    parent: Union[CognitiveEssenceCapsule, Mapping[str, Any]],
    *,
    child_entity_id: str,
    target_body_id: str = "",
    inherited_memory_state: Optional[Mapping[str, Any]] = None,
    inherited_mission_state: Optional[Mapping[str, Any]] = None,
    explicit_reproduction_approval: bool = False,
    approval_receipt: Optional[Mapping[str, Any]] = None,
    approved_by: str = "",
    signing_secret: Union[str, bytes, bytearray, None] = None,
) -> CognitiveEssenceCapsule:
    """Create a descendant Essence with a NEW identity and NO inherited authority.

    Reproduction is impossible unless the caller explicitly supplies approval. This
    function still only creates evidence; it does not spawn a process or activate a body.
    """
    if not explicit_reproduction_approval:
        raise PermissionError("explicit_reproduction_approval_required")
    if not isinstance(approval_receipt, Mapping) or not dict(approval_receipt):
        raise PermissionError("reproduction_approval_receipt_required")
    src = parent if isinstance(parent, CognitiveEssenceCapsule) else CognitiveEssenceCapsule.from_dict(parent)
    current_body_id = _current_body_id_from_capsule(src)
    receipt = dict(_bounded_value(dict(approval_receipt or {})))
    child = _safe_identifier(child_entity_id, prefix="entity")
    if not child or child == src.entity_id:
        raise ValueError("reproduction_requires_new_entity_identity")

    child_governance = {
        "inherited_constitution_reference": (src.governance_state or {}).get("constitution_reference"),
        "parent_governance_hash": _sha256_object(src.governance_state or {}),
        "authority_inheritance": "none",
        "authority_rebinding_required": True,
        "execution_authority": False,
        "governance_bypass_allowed": False,
        "reproduction_approval_receipt_hash": _sha256_object(receipt),
    }
    return create_cognitive_essence_capsule(
        entity_id=child,
        lineage_id=src.lineage_id,
        parent_entity_id=src.entity_id,
        operation=EssenceOperation.REPRODUCTION,
        status=EssenceStatus.CARRIER_QUARANTINE,
        continuity_epoch=0,
        checkpoint_sequence=0,
        source_body_id=current_body_id,
        target_body_id=target_body_id,
        identity_state={
            "entity_id": child,
            "parent_entity_id": src.entity_id,
            "lineage_id": src.lineage_id,
            "species_inheritance_from": src.entity_id,
        },
        memory_state=inherited_memory_state or {},
        mission_state=inherited_mission_state or {},
        governance_state=child_governance,
        trust_state={},  # trust must be established for the new individual
        software_state=src.software_state,
        model_state={},  # model selection belongs to the new host/capability passport
        body_state={},
        capability_requirements={},
        runtime_state={},
        recovery_state={"reproduction_source_capsule": src.capsule_id},
        ledger_anchor=src.ledger_anchor,
        previous_capsule_hash=src.payload_hash or calculate_capsule_payload_hash(src),
        metadata={
            "reproduction_approved": True,
            "approved_by": str(approved_by or "")[:200],
            "approval_receipt": receipt,
            "approval_receipt_hash": _sha256_object(receipt),
            "knowledge_inheritance_does_not_imply_authority": True,
        },
        signing_secret=signing_secret,
    )


# -----------------------------------------------------------------------------
# Validation / split-brain / host compatibility
# -----------------------------------------------------------------------------
def validate_cognitive_essence_capsule(
    capsule: Union[CognitiveEssenceCapsule, Mapping[str, Any]],
    *,
    signing_secret: Union[str, bytes, bytearray, None] = None,
    require_signature: bool = True,
    expected_entity_id: str = "",
    accepted_lineage_id: str = "",
    current_active_epoch: Optional[int] = None,
    current_active_body_id: str = "",
) -> EssenceValidationReport:
    cap = capsule if isinstance(capsule, CognitiveEssenceCapsule) else CognitiveEssenceCapsule.from_dict(capsule)
    errors: List[str] = []
    warnings: List[str] = []
    checks: Dict[str, bool] = {}

    checks["schema"] = cap.schema == ESSENCE_SCHEMA and int(cap.schema_version) == ESSENCE_SCHEMA_VERSION
    if not checks["schema"]:
        errors.append("unsupported_or_invalid_essence_schema")

    checks["entity_id_present"] = bool(cap.entity_id)
    checks["lineage_id_present"] = bool(cap.lineage_id)
    checks["capsule_id_present"] = bool(cap.capsule_id)
    checks["authority_disabled"] = cap.execution_authority is False and cap.governance_bypass_allowed is False
    checks["replication_authority_disabled"] = cap.autonomous_replication_authority is False
    checks["authority_rebind_required"] = cap.authority_must_rebind is True

    if not checks["entity_id_present"]:
        errors.append("missing_entity_id")
    if not checks["lineage_id_present"]:
        errors.append("missing_lineage_id")
    if not checks["authority_disabled"]:
        errors.append("capsule_may_not_grant_execution_or_governance_bypass")
    if not checks["replication_authority_disabled"]:
        errors.append("autonomous_replication_authority_forbidden")
    if not checks["authority_rebind_required"]:
        errors.append("authority_rebinding_must_remain_required")

    try:
        op = EssenceOperation(cap.operation)
        checks["operation_valid"] = True
    except Exception:
        op = EssenceOperation.CHECKPOINT
        checks["operation_valid"] = False
        errors.append("invalid_operation")

    if op == EssenceOperation.REPRODUCTION:
        checks["reproduction_new_identity"] = bool(cap.parent_entity_id and cap.parent_entity_id != cap.entity_id)
        if not checks["reproduction_new_identity"]:
            errors.append("reproduction_requires_parent_and_new_entity_identity")
        inherited_authority = str((cap.governance_state or {}).get("authority_inheritance") or "").lower()
        checks["reproduction_authority_not_inherited"] = inherited_authority in {"none", "forbidden_without_explicit_rebinding", ""}
        if not checks["reproduction_authority_not_inherited"]:
            errors.append("reproduction_may_not_inherit_parent_authority")
    else:
        checks["identity_continuity_operation"] = True

    size = _capsule_size_bytes(cap)
    checks["bounded_size"] = size <= MAX_CAPSULE_BYTES
    if not checks["bounded_size"]:
        errors.append("capsule_exceeds_size_limit")

    observed_hash = calculate_capsule_payload_hash(cap)
    checks["payload_hash"] = bool(cap.payload_hash and hmac.compare_digest(cap.payload_hash, observed_hash))
    if not checks["payload_hash"]:
        errors.append("payload_hash_mismatch_or_missing")

    sig = verify_cognitive_essence_signature(cap, signing_secret)
    checks["signature"] = bool(sig.get("ok"))
    if require_signature and not checks["signature"]:
        errors.append("verified_signature_required")
    elif not checks["signature"]:
        warnings.append(str(sig.get("reason") or "signature_not_verified"))

    if expected_entity_id:
        checks["expected_entity"] = cap.entity_id == str(expected_entity_id)
        if not checks["expected_entity"]:
            errors.append("entity_identity_mismatch")
    if accepted_lineage_id:
        checks["accepted_lineage"] = cap.lineage_id == str(accepted_lineage_id)
        if not checks["accepted_lineage"]:
            errors.append("lineage_mismatch")

    stale = False
    epoch_conflict = False
    if current_active_epoch is not None:
        active_epoch = int(current_active_epoch)
        stale = int(cap.continuity_epoch) < active_epoch
        epoch_conflict = int(cap.continuity_epoch) == active_epoch and bool(
            current_active_body_id and cap.target_body_id and current_active_body_id != cap.target_body_id
        )
        checks["not_stale"] = not stale
        checks["no_epoch_body_conflict"] = not epoch_conflict
        if stale:
            errors.append("stale_continuity_epoch")
        if epoch_conflict:
            errors.append("split_brain_epoch_body_conflict")

    if errors:
        try:
            _local_arile_sentinel.report(
                "cognitive_essence_validation_failure",
                "; ".join(errors[:6]),
                severity=0.85 if "split_brain_epoch_body_conflict" in errors else 0.70,
                capsule_id=cap.capsule_id,
                entity_id=cap.entity_id,
                continuity_epoch=cap.continuity_epoch,
            )
        except Exception:
            pass

    return EssenceValidationReport(
        ok=not errors,
        decision="VALID_EVIDENCE_ONLY" if not errors else "REJECT_OR_HOLD",
        errors=errors,
        warnings=warnings,
        checks=checks,
        observed={
            "capsule_id": cap.capsule_id,
            "entity_id": cap.entity_id,
            "lineage_id": cap.lineage_id,
            "operation": cap.operation,
            "continuity_epoch": cap.continuity_epoch,
            "checkpoint_sequence": cap.checkpoint_sequence,
            "size_bytes": size,
            "signature": sig,
            "stale": stale,
            "epoch_conflict": epoch_conflict,
        },
        execution_authority=False,
    )


def detect_split_brain(
    left: Union[CognitiveEssenceCapsule, Mapping[str, Any]],
    right: Union[CognitiveEssenceCapsule, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Detect competing claims to the same entity continuity."""
    a = left if isinstance(left, CognitiveEssenceCapsule) else CognitiveEssenceCapsule.from_dict(left)
    b = right if isinstance(right, CognitiveEssenceCapsule) else CognitiveEssenceCapsule.from_dict(right)
    same_entity = bool(a.entity_id and a.entity_id == b.entity_id)
    same_lineage = bool(a.lineage_id and a.lineage_id == b.lineage_id)
    same_epoch = int(a.continuity_epoch) == int(b.continuity_epoch)
    different_bodies = bool(
        (a.target_body_id or a.source_body_id)
        and (b.target_body_id or b.source_body_id)
        and (a.target_body_id or a.source_body_id) != (b.target_body_id or b.source_body_id)
    )
    competing_same_epoch = same_entity and same_epoch and different_bodies
    stale_side = ""
    if same_entity and not same_epoch:
        stale_side = "left" if int(a.continuity_epoch) < int(b.continuity_epoch) else "right"

    risk = "critical" if competing_same_epoch else "high" if stale_side else "low"
    action = "QUARANTINE_BOTH_AND_REQUIRE_GOVERNED_RECONCILIATION" if competing_same_epoch else (
        f"MARK_{stale_side.upper()}_STALE_CONTINUITY" if stale_side else "NO_SPLIT_BRAIN_DETECTED"
    )
    if competing_same_epoch:
        _local_arile_sentinel.report(
            "cognitive_essence_split_brain",
            "Two bodies claim the same entity and continuity epoch.",
            severity=0.95,
            entity_id=a.entity_id,
            continuity_epoch=a.continuity_epoch,
            left_body=a.target_body_id or a.source_body_id,
            right_body=b.target_body_id or b.source_body_id,
        )
    return {
        "ok": not competing_same_epoch,
        "same_entity": same_entity,
        "same_lineage": same_lineage,
        "same_epoch": same_epoch,
        "different_bodies": different_bodies,
        "split_brain": competing_same_epoch,
        "stale_side": stale_side,
        "risk": risk,
        "recommended_action": action,
        "automatic_merge_allowed": False,
        "execution_authority": False,
    }


def assess_host_compatibility(
    capsule: Union[CognitiveEssenceCapsule, Mapping[str, Any]],
    host_passport: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compare explicit host capability evidence with Essence requirements.

    This does not trust an arbitrary host merely because it advertises a capability.
    Trust/security/assurance must still be reviewed by their owning organs.
    """
    cap = capsule if isinstance(capsule, CognitiveEssenceCapsule) else CognitiveEssenceCapsule.from_dict(capsule)
    host = dict(host_passport or {})
    required = cap.capability_requirements or {}

    required_caps = {str(x) for x in (required.get("required_capabilities") or []) if str(x).strip()}
    host_caps = {str(x) for x in (host.get("capabilities") or host.get("declared_capabilities") or []) if str(x).strip()}
    missing = sorted(required_caps - host_caps)

    required_runtime = dict(required.get("runtime") or {})
    runtime = dict(host.get("runtime") or {})
    runtime_failures: List[str] = []
    for key, expected in required_runtime.items():
        if key not in runtime:
            runtime_failures.append(f"missing_runtime:{key}")
            continue
        if isinstance(expected, (int, float)) and isinstance(runtime.get(key), (int, float)):
            if float(runtime[key]) < float(expected):
                runtime_failures.append(f"runtime_below_minimum:{key}")
        elif runtime.get(key) != expected:
            runtime_failures.append(f"runtime_mismatch:{key}")

    trusted = bool(host.get("trusted") or host.get("trust_verified"))
    integrity = bool(host.get("integrity_verified") or host.get("attested"))
    safety = bool(host.get("safety_verified") or host.get("machine_safety_verified"))
    target_body = _safe_identifier(host.get("body_id") or host.get("device_id") or host.get("node_id") or "", prefix="body") if (host.get("body_id") or host.get("device_id") or host.get("node_id")) else ""
    if cap.target_body_id:
        target_match = bool(target_body) and cap.target_body_id == target_body
    else:
        target_match = bool(target_body)

    compatible = not missing and not runtime_failures and target_match
    return {
        "ok": bool(compatible and trusted and integrity and safety),
        "compatible": compatible,
        "host_trusted": trusted,
        "host_integrity_verified": integrity,
        "host_safety_verified": safety,
        "target_body_matches": target_match,
        "required_capabilities": sorted(required_caps),
        "host_capabilities": sorted(host_caps),
        "missing_capabilities": missing,
        "runtime_failures": runtime_failures,
        "carrier_quarantine_required": True,
        "authority_rebinding_required": True,
        "operatorcore_required_for_activation": True,
        "machine_native_safety_remains_authoritative": True,
        "execution_authority": False,
    }


# -----------------------------------------------------------------------------
# Survival-state analysis (evidence only)
# -----------------------------------------------------------------------------
def assess_survival_condition(telemetry: Mapping[str, Any]) -> Dict[str, Any]:
    """Classify survival pressure from explicit bounded telemetry.

    No hardware is queried. Missing evidence reduces confidence instead of being
    guessed. This function recommends review; it never initiates transfer.
    """
    t = dict(telemetry or {})
    observations: List[Tuple[str, float, bool]] = []

    def add(name: str, severity: float, condition: bool) -> None:
        observations.append((name, max(0.0, min(1.0, float(severity))), bool(condition)))

    battery = t.get("battery_percent")
    if isinstance(battery, (int, float)):
        add("battery_critical", 0.95, float(battery) <= 5.0)
        add("battery_low", 0.65, 5.0 < float(battery) <= 15.0)

    predicted = t.get("predicted_survival_seconds")
    if isinstance(predicted, (int, float)):
        add("survival_time_critical", 1.0, float(predicted) <= 120.0)
        add("survival_time_low", 0.75, 120.0 < float(predicted) <= 900.0)

    add("compute_unstable", 0.85, bool(t.get("compute_unstable")))
    add("storage_integrity_failure", 0.95, bool(t.get("storage_integrity_failure")))
    add("thermal_critical", 0.90, bool(t.get("thermal_critical")))
    add("structural_damage_severe", 0.90, bool(t.get("structural_damage_severe")))
    add("mobility_lost", 0.55, bool(t.get("mobility_lost")))
    add("communications_degraded", 0.35, bool(t.get("communications_degraded")))

    active = [(name, sev) for name, sev, cond in observations if cond]
    peak = max([sev for _, sev in active], default=0.0)
    corroborating = len([1 for _, sev in active if sev >= 0.55])

    if peak >= 0.95 and corroborating >= 2:
        severity = SurvivalSeverity.TRANSFER_RECOMMENDED
    elif peak >= 0.90:
        severity = SurvivalSeverity.SURVIVAL_CRITICAL
    elif peak >= 0.65:
        severity = SurvivalSeverity.SURVIVAL_WARNING
    elif active:
        severity = SurvivalSeverity.DEGRADED
    else:
        severity = SurvivalSeverity.NORMAL

    confidence = min(1.0, 0.35 + 0.12 * len(observations)) if observations else 0.0
    return {
        "ok": True,
        "severity": severity.value,
        "confidence": round(confidence, 3),
        "active_signals": [{"name": n, "severity": s} for n, s in active],
        "telemetry_fields_observed": len(observations),
        "transfer_automatic": False,
        "requires_governed_review": severity in {
            SurvivalSeverity.SURVIVAL_WARNING,
            SurvivalSeverity.SURVIVAL_CRITICAL,
            SurvivalSeverity.TRANSFER_RECOMMENDED,
        },
        "note": "Single-source evidence is not sufficient for autonomous migration; no transfer is initiated here.",
        "execution_authority": False,
    }


# -----------------------------------------------------------------------------
# Governance review / transition planning
# -----------------------------------------------------------------------------
def build_continuity_action_contract(
    capsule: Union[CognitiveEssenceCapsule, Mapping[str, Any]],
    *,
    requested_operation: Optional[Union[str, EssenceOperation]] = None,
    user_confirmed: bool = False,
    host_passport: Optional[Mapping[str, Any]] = None,
    survival_evidence: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    cap = capsule if isinstance(capsule, CognitiveEssenceCapsule) else CognitiveEssenceCapsule.from_dict(capsule)
    op = EssenceOperation(str((requested_operation.value if isinstance(requested_operation, EssenceOperation) else requested_operation) or cap.operation).lower())
    risk_level = "TIER_3_PRIVILEGED_SYSTEM"
    if op == EssenceOperation.REPRODUCTION:
        risk_level = "TIER_4_NETWORK_REMOTE_OR_DESTRUCTIVE"

    rollback_plan = [
        {"step": "keep_source_or_prior_verified_capsule_inactive_but_recoverable", "required": True},
        {"step": "revoke_target_activation_candidate_on_failed_verification", "required": True},
        {"step": "restore_last_verified_continuity_epoch_if_safe", "required": True},
    ]
    verification_checks = [
        {"name": "essence_integrity", "required": True},
        {"name": "lineage_continuity", "required": True},
        {"name": "split_brain_check", "required": True},
        {"name": "target_host_trust_and_integrity", "required": True},
        {"name": "authority_rebinding", "required": True},
        {"name": "machine_native_safety", "required": True},
    ]
    required_permissions = [
        "cognitive_essence.review",
        f"cognitive_essence.{op.value}",
        "identity.rebind",
        "body.carrier_admission",
    ]
    if op == EssenceOperation.REPRODUCTION:
        required_permissions.extend([
            "cognitive_essence.reproduction.create_descendant",
            "lineage.create_descendant_identity",
        ])
    return {
        "action_type": f"cognitive_essence_{op.value}",
        "capability_name": "cognitive_essence_continuity",
        "executor_name": "external_domain_owned",
        "origin": MODULE_NAME,
        "source_surface": "core",
        "trust_context": {
            "caller_kind": "core",
            "caller_id": MODULE_NAME,
            "module_name": MODULE_NAME,
            "surface": "core",
            "trust_tier": "core",
            "evidence_only": True,
            "execution_authority": False,
        },
        "target": cap.target_body_id or "continuity_target_unresolved",
        "target_ref": cap.target_body_id or cap.capsule_id,
        "risk_level": risk_level,
        "execution_mode": "apply",
        "requires_confirmation": True,
        "user_confirmed": bool(user_confirmed),
        "confirmed": bool(user_confirmed),
        "required_permissions": required_permissions,
        "rollback_plan": rollback_plan,
        "verification_checks": verification_checks,
        "controller_identity": cap.entity_id,
        "current_control_owner": cap.entity_id,
        "metadata": {
            "source": MODULE_NAME,
            "caller_id": MODULE_NAME,
            "caller_kind": "core",
            "module_name": MODULE_NAME,
            "surface": "core",
            "trust_tier": "core",
            "evidence_only": True,
            "capsule_id": cap.capsule_id,
            "entity_id": cap.entity_id,
            "lineage_id": cap.lineage_id,
            "continuity_epoch": cap.continuity_epoch,
            "requested_operation": op.value,
            "target_body_id": cap.target_body_id,
            "user_confirmed": bool(user_confirmed),
            "host_passport": _bounded_value(dict(host_passport or {})),
            "survival_evidence": _bounded_value(dict(survival_evidence or {})),
            "capsule_execution_authority": False,
            "operatorcore_must_own_actual_activation": True,
            "authority_rebinding_required": True,
        },
    }


def review_continuity_governance(
    capsule: Union[CognitiveEssenceCapsule, Mapping[str, Any]],
    *,
    requested_operation: Optional[Union[str, EssenceOperation]] = None,
    user_confirmed: bool = False,
    host_passport: Optional[Mapping[str, Any]] = None,
    survival_evidence: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Ask existing SecurityGovernor and AssuranceGate for evidence-only review.

    This function intentionally does not call OperatorCore execution. A positive
    result only means "eligible to be submitted to OperatorCore", never activated.
    """
    contract = build_continuity_action_contract(
        capsule,
        requested_operation=requested_operation,
        user_confirmed=user_confirmed,
        host_passport=host_passport,
        survival_evidence=survival_evidence,
    )
    governance_context = {
        "source": MODULE_NAME,
        "user_confirmed": bool(user_confirmed),
        "execution_authority": False,
        "requested_review_only": True,
        "module_may_not_grant_allow": True,
    }

    security: Dict[str, Any]
    assurance: Dict[str, Any]
    errors: List[str] = []

    try:
        import SarahMemorySecurityGovernor as security_governor  # type: ignore

        security = security_governor.evaluate_action(contract, governance_context)
    except Exception as exc:
        security = {"allow": False, "decision": "DENY", "error": f"security_governor_unavailable:{exc}"}
        errors.append("security_governor_unavailable")

    try:
        import SarahMemoryAssuranceGate as assurance_gate  # type: ignore

        assurance = assurance_gate.evaluate_action_assurance(contract, governance_context, security)
    except Exception as exc:
        assurance = {"allow": False, "decision": "DENY", "error": f"assurance_gate_unavailable:{exc}"}
        errors.append("assurance_gate_unavailable")

    security_allow = bool(security.get("allow"))
    assurance_allow = bool(assurance.get("allow"))
    eligible = bool(user_confirmed and security_allow and assurance_allow and not errors)
    return {
        "ok": eligible,
        "decision": "ELIGIBLE_FOR_OPERATORCORE_REVIEW" if eligible else "HOLD_OR_DENY",
        "contract": contract,
        "security": security,
        "assurance": assurance,
        "errors": errors,
        "operatorcore_required": True,
        "activation_performed": False,
        "execution_authority": False,
    }


def plan_continuity_transition(
    capsule: Union[CognitiveEssenceCapsule, Mapping[str, Any]],
    *,
    host_passport: Optional[Mapping[str, Any]] = None,
    current_active_epoch: Optional[int] = None,
    current_active_body_id: str = "",
    signing_secret: Union[str, bytes, bytearray, None] = None,
    require_signature: bool = True,
    user_confirmed: bool = False,
    survival_evidence: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a bounded transition plan without performing migration/activation."""
    cap = capsule if isinstance(capsule, CognitiveEssenceCapsule) else CognitiveEssenceCapsule.from_dict(capsule)
    validation = validate_cognitive_essence_capsule(
        cap,
        signing_secret=signing_secret,
        require_signature=require_signature,
        current_active_epoch=current_active_epoch,
        current_active_body_id=current_active_body_id,
    )
    host = assess_host_compatibility(cap, host_passport or {}) if host_passport is not None else {
        "ok": False,
        "compatible": False,
        "reason": "host_passport_required_before_activation",
        "execution_authority": False,
    }
    governance = review_continuity_governance(
        cap,
        user_confirmed=user_confirmed,
        host_passport=host_passport,
        survival_evidence=survival_evidence,
    ) if validation.ok else {
        "ok": False,
        "decision": "NOT_REVIEWED_INVALID_CAPSULE",
        "execution_authority": False,
    }

    ready = bool(validation.ok and host.get("ok") and governance.get("ok"))
    return {
        "ok": ready,
        "decision": "READY_FOR_OPERATORCORE_SUBMISSION" if ready else "HOLD",
        "validation": validation.to_dict(),
        "host_compatibility": host,
        "governance_review": governance,
        "next_step": "submit_explicit_action_to_OperatorCore" if ready else "resolve_failed_checks_before_any_activation",
        "activation_performed": False,
        "execution_authority": False,
    }


# -----------------------------------------------------------------------------
# Explicit persistence (atomic, local, bounded)
# -----------------------------------------------------------------------------
def persist_cognitive_essence_capsule(
    capsule: Union[CognitiveEssenceCapsule, Mapping[str, Any]],
    *,
    filepath: Optional[Union[str, os.PathLike[str]]] = None,
    allow_write: bool = False,
    overwrite: bool = False,
    allow_external_path: bool = False,
) -> Dict[str, Any]:
    """Persist one capsule atomically. Caller must explicitly authorize the write."""
    if not allow_write:
        return {
            "ok": False,
            "error": "explicit_allow_write_required",
            "path": "",
            "execution_authority": False,
        }
    cap = capsule if isinstance(capsule, CognitiveEssenceCapsule) else CognitiveEssenceCapsule.from_dict(capsule)
    size = _capsule_size_bytes(cap)
    if size > MAX_CAPSULE_BYTES:
        return {"ok": False, "error": "capsule_exceeds_size_limit", "size_bytes": size, "execution_authority": False}

    root = Path(DEFAULT_ESSENCE_DIR).expanduser().resolve()
    target = Path(filepath).expanduser().resolve() if filepath is not None else root / _default_capsule_filename(cap)
    if not allow_external_path and not _path_is_within(target, root):
        return {
            "ok": False,
            "error": "capsule_path_outside_default_essence_dir",
            "path": str(target),
            "default_dir": str(root),
            "execution_authority": False,
        }
    if target.exists() and not overwrite:
        return {"ok": False, "error": "target_exists_overwrite_not_approved", "path": str(target), "execution_authority": False}

    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(cap.to_dict(), indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8")
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(target.parent), prefix=target.name + ".", suffix=".tmp") as fh:
            tmp_path = fh.name
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.chmod(tmp_path, 0o600)
        except Exception:
            pass
        os.replace(tmp_path, target)
        try:
            os.chmod(target, 0o600)
        except Exception:
            pass
    except Exception as exc:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return {"ok": False, "error": f"write_failed:{exc}", "path": str(target), "execution_authority": False}

    return {
        "ok": True,
        "path": str(target),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
        "atomic_write": True,
        "execution_authority": False,
    }


def load_cognitive_essence_capsule(
    filepath: Union[str, os.PathLike[str]],
    *,
    max_bytes: int = MAX_CAPSULE_BYTES,
    allow_external_path: bool = False,
) -> Dict[str, Any]:
    """Load one explicit local capsule path. No directory traversal or scanning."""
    path = Path(filepath).expanduser().resolve()
    root = Path(DEFAULT_ESSENCE_DIR).expanduser().resolve()
    if not allow_external_path and not _path_is_within(path, root):
        return {
            "ok": False,
            "error": "capsule_path_outside_default_essence_dir",
            "path": str(path),
            "default_dir": str(root),
            "execution_authority": False,
        }
    try:
        if not path.is_file():
            return {"ok": False, "error": "capsule_file_not_found", "path": str(path), "execution_authority": False}
        size = path.stat().st_size
        if size > int(max_bytes):
            return {"ok": False, "error": "capsule_file_exceeds_size_limit", "size_bytes": size, "execution_authority": False}
        raw = path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
        cap = CognitiveEssenceCapsule.from_dict(data)
        return {
            "ok": True,
            "capsule": cap.to_dict(),
            "path": str(path),
            "file_sha256": _sha256_bytes(raw),
            "execution_authority": False,
        }
    except Exception as exc:
        return {"ok": False, "error": f"capsule_load_failed:{exc}", "path": str(path), "execution_authority": False}


# -----------------------------------------------------------------------------
# Ledger evidence integration
# -----------------------------------------------------------------------------
def record_cognitive_essence_receipt(
    capsule: Union[CognitiveEssenceCapsule, Mapping[str, Any]],
    event_type: str,
    *,
    verdict: str = "OBSERVED",
    summary: str = "",
    risk: str = "medium",
) -> Dict[str, Any]:
    """Record continuity evidence in the existing Ledger when available."""
    cap = capsule if isinstance(capsule, CognitiveEssenceCapsule) else CognitiveEssenceCapsule.from_dict(capsule)
    try:
        import SarahMemoryLedger as ledger  # type: ignore

        result = ledger.record_governance_receipt(
            "cognitive_essence",
            str(event_type or "ESSENCE_EVENT"),
            subject_id=cap.entity_id,
            verdict=str(verdict or "OBSERVED"),
            risk=str(risk or "medium"),
            retention_class="security_audit",
            payload_hash=cap.payload_hash or calculate_capsule_payload_hash(cap),
            payload_ref=cap.capsule_id,
            summary=str(summary or "")[:1000],
            metadata={
                "schema": cap.schema,
                "lineage_id": cap.lineage_id,
                "operation": cap.operation,
                "continuity_epoch": cap.continuity_epoch,
                "checkpoint_sequence": cap.checkpoint_sequence,
                "source_body_id": cap.source_body_id,
                "target_body_id": cap.target_body_id,
                "execution_authority": False,
            },
        )
        if isinstance(result, dict):
            result.setdefault("execution_authority", False)
            return result
    except Exception as exc:
        return {"ok": False, "error": f"ledger_unavailable:{exc}", "execution_authority": False}
    return {"ok": False, "error": "ledger_receipt_failed", "execution_authority": False}


# -----------------------------------------------------------------------------
# Read-only capability report and self-test
# -----------------------------------------------------------------------------
def get_cognitive_essence_capabilities() -> Dict[str, Any]:
    return {
        "ok": True,
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "schema": ESSENCE_SCHEMA,
        "schema_version": ESSENCE_SCHEMA_VERSION,
        "local_first": True,
        "network_access": False,
        "filesystem_scan_authority": False,
        "hardware_control": False,
        "execution_authority": False,
        "autonomous_replication_authority": False,
        "governance_bypass_allowed": False,
        "operations": [item.value for item in EssenceOperation],
        "statuses": [item.value for item in EssenceStatus],
        "supports": [
            "bounded_checkpoint_capsules",
            "integrity_hashing",
            "optional_hmac_authentication",
            "migration_planning",
            "recovery_planning",
            "governed_reproduction_contract",
            "new_identity_enforcement_for_reproduction",
            "split_brain_detection",
            "stale_epoch_detection",
            "host_compatibility_review",
            "authority_rebinding_requirement",
            "survival_condition_evidence",
            "contained_atomic_local_persistence",
            "ledger_receipts",
            "sml_adapter",
        ],
        "hard_invariants": list(HARD_INVARIANTS),
        "confidentiality_note": "Capsule HMAC protects integrity/authenticity, not confidentiality. Sensitive payloads should use approved encrypted storage/vault references.",
    }


def cognitive_essence_self_test() -> Dict[str, Any]:
    """In-memory deterministic tests. Performs no file/network/device writes."""
    checks: List[Dict[str, Any]] = []
    secret = b"SarahMemoryCognitiveEssenceSelfTestKey-2026"

    try:
        base = create_cognitive_essence_capsule(
            entity_id="selftest_entity",
            lineage_id="selftest_lineage",
            operation=EssenceOperation.CHECKPOINT,
            continuity_epoch=7,
            checkpoint_sequence=3,
            source_body_id="body_A",
            identity_state={"entity_id": "selftest_entity"},
            memory_state={"recent_event": "test", "api_key": "must_not_survive"},
            governance_state={"constitution_reference": "selftest"},
            capability_requirements={"required_capabilities": ["compute", "camera"]},
            signing_secret=secret,
        )
        checks.append({"name": "capsule_created", "passed": bool(base.capsule_id and base.payload_hash)})
        checks.append({"name": "secret_redacted", "passed": base.memory_state.get("api_key") == "[REDACTED_SECRET_USE_SECURE_REFERENCE]"})

        vr = validate_cognitive_essence_capsule(base, signing_secret=secret, require_signature=True, current_active_epoch=7)
        checks.append({"name": "signed_validation", "passed": bool(vr.ok)})

        migration = create_migration_capsule(base, target_body_id="body_B", signing_secret=secret)
        checks.append({"name": "migration_same_identity", "passed": migration.entity_id == base.entity_id and migration.continuity_epoch == 8})
        checks.append({"name": "migration_no_authority", "passed": migration.execution_authority is False and migration.authority_must_rebind is True})

        stale = validate_cognitive_essence_capsule(base, signing_secret=secret, require_signature=True, current_active_epoch=8)
        checks.append({"name": "stale_epoch_rejected", "passed": (not stale.ok) and "stale_continuity_epoch" in stale.errors})

        peer = create_migration_capsule(base, target_body_id="body_C", signing_secret=secret)
        split = detect_split_brain(migration, peer)
        checks.append({"name": "split_brain_detected", "passed": bool(split.get("split_brain"))})

        child = create_descendant_capsule(
            base,
            child_entity_id="selftest_entity_child",
            target_body_id="body_child",
            inherited_memory_state={"lesson": "inherited knowledge"},
            explicit_reproduction_approval=True,
            approval_receipt={"receipt_id": "self_test_reproduction_receipt", "verdict": "APPROVED", "authority": "self_test"},
            approved_by="self_test",
            signing_secret=secret,
        )
        checks.append({"name": "reproduction_new_identity", "passed": child.entity_id != base.entity_id and child.parent_entity_id == base.entity_id})
        checks.append({"name": "reproduction_no_authority_inheritance", "passed": child.governance_state.get("authority_inheritance") == "none"})

        host = assess_host_compatibility(
            migration,
            {
                "body_id": "body_B",
                "capabilities": ["compute", "camera"],
                "trusted": True,
                "integrity_verified": True,
                "safety_verified": True,
            },
        )
        checks.append({"name": "host_compatibility", "passed": bool(host.get("ok"))})

        unsafe_host = assess_host_compatibility(
            migration,
            {
                "body_id": "body_B",
                "capabilities": ["compute", "camera"],
                "trusted": True,
                "integrity_verified": True,
                "safety_verified": False,
            },
        )
        checks.append({"name": "unsafe_host_rejected", "passed": not bool(unsafe_host.get("ok"))})

        no_body_host = assess_host_compatibility(
            migration,
            {
                "capabilities": ["compute", "camera"],
                "trusted": True,
                "integrity_verified": True,
                "safety_verified": True,
            },
        )
        checks.append({"name": "target_body_required", "passed": not bool(no_body_host.get("ok"))})

        unsigned = create_cognitive_essence_capsule(
            entity_id="unsigned_selftest",
            lineage_id="selftest_lineage",
            identity_state={"entity_id": "unsigned_selftest"},
        )
        unsigned_report = validate_cognitive_essence_capsule(unsigned)
        checks.append({"name": "unsigned_capsule_rejected_by_default", "passed": (not unsigned_report.ok) and "verified_signature_required" in unsigned_report.errors})

        unsafe_token = _safe_filename_token("../../tmp/evil:body/A", prefix="entity")
        checks.append({"name": "filename_token_blocks_path_chars", "passed": "/" not in unsafe_token and "\\" not in unsafe_token and ":" not in unsafe_token})

        survival = assess_survival_condition({
            "battery_percent": 2,
            "predicted_survival_seconds": 80,
            "compute_unstable": True,
        })
        checks.append({"name": "survival_evidence", "passed": survival.get("severity") == SurvivalSeverity.TRANSFER_RECOMMENDED.value})

    except Exception as exc:
        checks.append({"name": "unexpected_exception", "passed": False, "observed": str(exc)})

    passed = sum(1 for item in checks if item.get("passed"))
    return {
        "ok": passed == len(checks),
        "passed": passed,
        "total": len(checks),
        "checks": checks,
        "file_write_performed": False,
        "network_used": False,
        "hardware_control": False,
        "execution_authority": False,
    }


# -----------------------------------------------------------------------------
# SML organ adapter - protocol visibility only, never direct execution
# -----------------------------------------------------------------------------
SML_ORGAN_METADATA = {
    "name": MODULE_NAME,
    "version": "v9.0.0-alpha-sml-0.1",
    "category": "CognitiveContinuity",
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": [
        "cognitive_essence",
        "continuity_checkpoint",
        "migration_review",
        "recovery_review",
        "reproduction_review",
        "split_brain_detection",
    ],
    "supported_missions": ["Diagnostics", "ContinuityPlanning", "GovernanceReview"],
    "supported_omega": [],
    "required_authority": ["Read"],
    "execution_authority": False,
    "priority": 50,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {
        "sml_adapter": "cognitive_continuity_non_executing",
        "source_file": "SarahMemoryCognitiveEssence.py",
        "autonomous_replication_authority": False,
        "authority_rebinding_required": True,
    },
}


def sml_get_metadata() -> Dict[str, Any]:
    return copy.deepcopy(SML_ORGAN_METADATA)


def _dependency_probe(module_name: str) -> Dict[str, Any]:
    try:
        __import__(module_name)
        return {"ok": True, "module": module_name}
    except Exception as exc:
        return {"ok": False, "module": module_name, "error": str(exc)[:300]}


def sml_health() -> Dict[str, Any]:
    required = [
        "SarahMemoryCognitiveSelf",
        "SarahMemorySMLProtocol",
        "SarahMemorySecurityGovernor",
        "SarahMemoryAssuranceGate",
        "SarahMemoryOperatorCore",
        "SarahMemoryLedger",
    ]
    dependencies = {name: _dependency_probe(name) for name in required}
    missing = [name for name, result in dependencies.items() if not result.get("ok")]
    availability = max(0.0, 1.0 - (0.12 * len(missing)))
    status = "Healthy" if not missing else "Degraded" if len(missing) < len(required) else "Unavailable"
    notes = [
        "Cognitive Essence continuity organ loaded",
        "No network/device/autonomous replication authority",
        "Activation remains owned by existing governance/OperatorCore stack",
    ]
    if missing:
        notes.append("Missing governance/runtime dependencies: " + ", ".join(missing))
    return {
        "status": status,
        "availability": round(availability, 3),
        "integrity": 1.0 if not missing else 0.75,
        "performance": 0.95,
        "reliability": 0.95 if not missing else 0.70,
        "confidence": 0.95 if not missing else 0.70,
        "latency_ms": 0.0,
        "stability": 0.95,
        "compatibility": 0.90 if not missing else 0.65,
        "execution_authority": False,
        "dependencies": dependencies,
        "notes": notes,
    }


def sml_diagnostics() -> Dict[str, Any]:
    return {
        "status": "OK",
        "component": MODULE_NAME,
        "sml_adapter": True,
        "metadata": sml_get_metadata(),
        "health": sml_health(),
        "capabilities": get_cognitive_essence_capabilities(),
    }


def sml_receive_packet(packet: Any, *, action: str = "observe", note: str = "", updates: Optional[Mapping[str, Any]] = None) -> Any:
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet  # type: ignore

        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(
            packet,
            organ=MODULE_NAME,
            action=action,
            note=note or "Cognitive Essence observed continuity packet",
            updates=dict(updates or {}),
        )
    except Exception:
        return packet


# -----------------------------------------------------------------------------
# CLI diagnostics only
# -----------------------------------------------------------------------------
def main() -> int:
    report = cognitive_essence_self_test()
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

# ============================================================================
# END OF SarahMemoryCognitiveEssence.py
# ============================================================================
