"""--==The SarahMemory Project==--
File: SarahMemoryWorldMatrixFabric_DRAFT.py
Part of the SarahMemory AiOS / GCAIOS governed cognitive runtime draft layer
Version: v0.1.0-draft
Date: 2026-09-28
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.

Purpose
-------
Non-invasive draft scaffold for the next SarahMemory build-out:

    SarahNet Cognitive Fabric -> AgentFirewall / SMUGCC boundary
    -> SarahMemory World Matrix state -> AvatarPanel / VR HUD packets

This file is intentionally a draft integration layer. It does not replace the
existing SarahMemoryAgentFirewall, SarahMemoryAiFunctions, SarahMemoryAvatar,
SarahMemoryAvatarPanel, SarahMemoryVRHudRenderer, CanvasStudio, AvatarBuilder,
or Blender bootstrap files.

Safety doctrine
---------------
- No third-party model downloads.
- No authentication/subscription/API-key bypass.
- No hidden endpoint reverse engineering.
- No CAPTCHA / anti-bot bypass.
- No physical-world execution authority.
- No robot/device/driver control authority.
- World packets are display/state packets only.
- All external cognitive surfaces must be treated as untrusted until governed by
  AgentFirewall / SMUGCC / TrustRegistry policy.

Core idea
---------
The "world" is the persistent semantic substrate. VR/AR/XR and AvatarPanel are
rendering/perception surfaces. External models are temporary cognitive organs;
SarahMemory retains identity, memory, governance, provenance, and authority.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


MODULE_NAME = "SarahMemoryWorldMatrixFabric"
MODULE_VERSION = "0.1.0-draft"
WORLD_SCHEMA = "SARAHMEMORY_WORLD_MATRIX_STATE_V1"
WORLD_EVENT_SCHEMA = "SARAHMEMORY_WORLD_EVENT_V1"
WORLD_PACKET_SCHEMA = "SMWORLD_PACKET_V1"
AVATAR_PANEL_WORLD_PACKET_SCHEMA = "SARAHMEMORY_AVATAR_PANEL_WORLD_PACKET_V1"
HUD_PACKET_SCHEMA = "SMHUD_PACKET_V1"


# ---------------------------------------------------------------------------
# Enums and bounded classifications
# ---------------------------------------------------------------------------

class WorldObjectKind(str, Enum):
    REAL = "real"                  # Direct physical-world counterpart exists.
    DIGITAL_TWIN = "digital_twin"  # Represents a real object but is not authority over it.
    VIRTUAL = "virtual"            # Digital-native persistent object.
    SIMULATED = "simulated"        # Counterfactual/scenario object.


class EpistemicState(str, Enum):
    OBSERVED = "observed"
    VERIFIED = "verified"
    INFERRED = "inferred"
    SIMULATED = "simulated"
    UNKNOWN = "unknown"


class EntityType(str, Enum):
    WORLD = "world"
    REGION = "region"
    SITE = "site"
    STRUCTURE = "structure"
    ROOM = "room"
    MACHINE = "machine"
    DEVICE = "device"
    SENSOR = "sensor"
    HUMAN = "human"
    AI_ENTITY = "ai_entity"
    AVATAR = "avatar"
    MODEL_RESOURCE = "model_resource"
    DATA_CENTER = "data_center"
    STORAGE = "storage"
    COMPUTE_NODE = "compute_node"
    ASSET = "asset"
    DOCUMENT = "document"
    EVENT = "event"
    UNKNOWN = "unknown"


class AuthorityBoundary(str, Enum):
    DISPLAY_ONLY = "display_only"
    READ_RETURN_ONLY = "read_return_only"
    SIMULATION_ONLY = "simulation_only"
    USER_APPROVAL_REQUIRED = "user_approval_required"
    PHYSICAL_ACTION_FORBIDDEN = "physical_action_forbidden"


class WorldEventKind(str, Enum):
    OBSERVATION = "observation"
    COGNITIVE_RECEIPT = "cognitive_receipt"
    ENTITY_CREATED = "entity_created"
    ENTITY_UPDATED = "entity_updated"
    RELATIONSHIP_UPDATED = "relationship_updated"
    XR_PACKET_BUILT = "xr_packet_built"
    AVATAR_PACKET_BUILT = "avatar_packet_built"
    SCENARIO_FORK = "scenario_fork"
    POLICY_DENIED = "policy_denied"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Vec3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    frame: str = "local"


@dataclass
class GeoPose:
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt_m: Optional[float] = None
    heading_deg: Optional[float] = None
    frame: str = "wgs84"


@dataclass
class Provenance:
    source: str = "unknown"
    source_kind: str = "unknown"
    observed_at: float = field(default_factory=time.time)
    confidence: float = 0.0
    evidence_hash: str = ""
    notes: str = ""


@dataclass
class WorldRelationship:
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 0.50
    provenance: Provenance = field(default_factory=Provenance)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorldEntity:
    entity_id: str
    name: str
    entity_type: EntityType = EntityType.UNKNOWN
    object_kind: WorldObjectKind = WorldObjectKind.VIRTUAL
    epistemic_state: EpistemicState = EpistemicState.UNKNOWN
    local_position: Vec3 = field(default_factory=Vec3)
    geo_pose: Optional[GeoPose] = None
    authority_boundary: AuthorityBoundary = AuthorityBoundary.DISPLAY_ONLY
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[WorldRelationship] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    revision: int = 0

    def update(self, **fields: Any) -> None:
        for key, value in fields.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.revision += 1
        self.updated_at = time.time()


@dataclass
class WorldEvent:
    event_id: str
    kind: WorldEventKind
    subject_id: str = ""
    summary: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    authority_boundary: AuthorityBoundary = AuthorityBoundary.DISPLAY_ONLY
    timestamp: float = field(default_factory=time.time)
    event_hash: str = ""

    def seal(self) -> "WorldEvent":
        material = json.dumps(
            {
                "event_id": self.event_id,
                "kind": self.kind.value,
                "subject_id": self.subject_id,
                "summary": self.summary,
                "payload": self.payload,
                "authority_boundary": self.authority_boundary.value,
                "timestamp": self.timestamp,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        self.event_hash = hashlib.sha256(material).hexdigest()
        return self


@dataclass
class WorldSnapshot:
    schema: str = WORLD_SCHEMA
    world_id: str = "sarahmemory.world.root"
    generated_at: float = field(default_factory=time.time)
    entity_count: int = 0
    relationship_count: int = 0
    entities: List[Dict[str, Any]] = field(default_factory=list)
    execution_authority: bool = False
    physical_action_authority: bool = False


@dataclass
class WorldRequest:
    capability: str
    payload: Any
    requester_id: str = "sarah"
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: Dict[str, Any] = field(default_factory=dict)
    max_resources: int = 3
    require_consensus: bool = False


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _safe_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)


def _sha256_obj(obj: Any) -> str:
    return hashlib.sha256(_safe_json(obj).encode("utf-8")).hexdigest()


def _base_dir() -> Path:
    try:
        import SarahMemoryGlobals as config  # type: ignore
        base = getattr(config, "BASE_DIR", None)
        if base:
            return Path(str(base)).expanduser().resolve()
    except Exception:
        pass
    here = Path(__file__).resolve()
    if here.parent.name.lower() == "core":
        return here.parent.parent.resolve()
    for parent in here.parents:
        if (parent / "core").is_dir() and ((parent / "data").is_dir() or (parent / "resources").is_dir()):
            return parent.resolve()
    return Path.cwd().resolve()


def _data_dir() -> Path:
    try:
        import SarahMemoryGlobals as config  # type: ignore
        data = getattr(config, "DATA_DIR", None)
        if data:
            return Path(str(data)).expanduser().resolve()
    except Exception:
        pass
    return (_base_dir() / "data").resolve()


def _world_dir() -> Path:
    path = _data_dir() / "world_matrix"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Lightweight ledger/state store
# ---------------------------------------------------------------------------

class WorldLedger:
    """Append-only local draft ledger for world events.

    This is not a replacement for the authoritative SarahMemory ledger. It is a
    draft local event journal so the world layer can be tested without mutating
    other modules.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or (_world_dir() / "world_events.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: WorldEvent) -> WorldEvent:
        event.seal()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(_safe_json(asdict(event)) + "\n")
        return event


class WorldStateStore:
    """In-memory world state with optional JSON persistence."""

    def __init__(self, state_path: Optional[Path] = None, ledger: Optional[WorldLedger] = None) -> None:
        self.state_path = state_path or (_world_dir() / "world_state.json")
        self.ledger = ledger or WorldLedger()
        self.entities: Dict[str, WorldEntity] = {}
        self.relationships: List[WorldRelationship] = []

    def add_entity(self, entity: WorldEntity) -> WorldEntity:
        if not entity.entity_id:
            entity.entity_id = f"entity.{uuid.uuid4()}"
        entity.updated_at = time.time()
        entity.revision += 1
        self.entities[entity.entity_id] = entity
        self.ledger.append(WorldEvent(
            event_id=str(uuid.uuid4()),
            kind=WorldEventKind.ENTITY_CREATED,
            subject_id=entity.entity_id,
            summary=f"Entity registered: {entity.name}",
            payload={"entity": asdict(entity)},
            authority_boundary=entity.authority_boundary,
        ))
        return entity

    def upsert_entity(self, entity: WorldEntity) -> WorldEntity:
        if entity.entity_id in self.entities:
            current = self.entities[entity.entity_id]
            current.update(
                name=entity.name,
                entity_type=entity.entity_type,
                object_kind=entity.object_kind,
                epistemic_state=entity.epistemic_state,
                local_position=entity.local_position,
                geo_pose=entity.geo_pose,
                authority_boundary=entity.authority_boundary,
                properties=entity.properties,
                provenance=entity.provenance,
            )
            self.ledger.append(WorldEvent(
                event_id=str(uuid.uuid4()),
                kind=WorldEventKind.ENTITY_UPDATED,
                subject_id=current.entity_id,
                summary=f"Entity updated: {current.name}",
                payload={"entity": asdict(current)},
                authority_boundary=current.authority_boundary,
            ))
            return current
        return self.add_entity(entity)

    def get(self, entity_id: str) -> Optional[WorldEntity]:
        return self.entities.get(entity_id)

    def add_relationship(self, relationship: WorldRelationship) -> WorldRelationship:
        self.relationships.append(relationship)
        if relationship.subject_id in self.entities:
            self.entities[relationship.subject_id].relationships.append(relationship)
            self.entities[relationship.subject_id].updated_at = time.time()
        self.ledger.append(WorldEvent(
            event_id=str(uuid.uuid4()),
            kind=WorldEventKind.RELATIONSHIP_UPDATED,
            subject_id=relationship.subject_id,
            summary=f"Relationship: {relationship.subject_id} {relationship.predicate} {relationship.object_id}",
            payload={"relationship": asdict(relationship)},
            authority_boundary=AuthorityBoundary.DISPLAY_ONLY,
        ))
        return relationship

    def snapshot(self, limit: int = 250) -> WorldSnapshot:
        entities = [asdict(e) for e in list(self.entities.values())[: max(1, limit)]]
        return WorldSnapshot(
            entity_count=len(self.entities),
            relationship_count=len(self.relationships),
            entities=entities,
        )

    def save(self) -> None:
        material = asdict(self.snapshot(limit=1000000))
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(_safe_json(material), encoding="utf-8")
        os.replace(str(tmp), str(self.state_path))

    def load(self) -> None:
        if not self.state_path.exists():
            return
        obj = json.loads(self.state_path.read_text(encoding="utf-8"))
        entities = obj.get("entities") if isinstance(obj, dict) else []
        if not isinstance(entities, list):
            return
        self.entities.clear()
        self.relationships.clear()
        for raw in entities:
            try:
                entity = self._entity_from_dict(raw)
                self.entities[entity.entity_id] = entity
                self.relationships.extend(entity.relationships)
            except Exception:
                continue

    @staticmethod
    def _entity_from_dict(raw: Dict[str, Any]) -> WorldEntity:
        local = raw.get("local_position") or {}
        geo = raw.get("geo_pose")
        prov = raw.get("provenance") or {}
        rels = []
        for r in raw.get("relationships") or []:
            rp = r.get("provenance") or {}
            rels.append(WorldRelationship(
                subject_id=str(r.get("subject_id") or ""),
                predicate=str(r.get("predicate") or ""),
                object_id=str(r.get("object_id") or ""),
                confidence=float(r.get("confidence") or 0.0),
                provenance=Provenance(**{k: rp.get(k) for k in Provenance.__dataclass_fields__.keys() if k in rp}),
                metadata=dict(r.get("metadata") or {}),
            ))
        return WorldEntity(
            entity_id=str(raw.get("entity_id") or f"entity.{uuid.uuid4()}"),
            name=str(raw.get("name") or "Unnamed Entity"),
            entity_type=EntityType(str(raw.get("entity_type") or EntityType.UNKNOWN.value)),
            object_kind=WorldObjectKind(str(raw.get("object_kind") or WorldObjectKind.VIRTUAL.value)),
            epistemic_state=EpistemicState(str(raw.get("epistemic_state") or EpistemicState.UNKNOWN.value)),
            local_position=Vec3(**{k: local.get(k) for k in Vec3.__dataclass_fields__.keys() if k in local}),
            geo_pose=GeoPose(**{k: geo.get(k) for k in GeoPose.__dataclass_fields__.keys() if k in geo}) if isinstance(geo, dict) else None,
            authority_boundary=AuthorityBoundary(str(raw.get("authority_boundary") or AuthorityBoundary.DISPLAY_ONLY.value)),
            properties=dict(raw.get("properties") or {}),
            relationships=rels,
            provenance=Provenance(**{k: prov.get(k) for k in Provenance.__dataclass_fields__.keys() if k in prov}),
            created_at=float(raw.get("created_at") or time.time()),
            updated_at=float(raw.get("updated_at") or time.time()),
            revision=int(raw.get("revision") or 0),
        )


# ---------------------------------------------------------------------------
# AgentFirewall / SMUGCC bridge
# ---------------------------------------------------------------------------

class AgentFirewallBridge:
    """Optional bridge into SarahMemoryAgentFirewall.

    If the authoritative module is not importable in the current test folder, the
    bridge fails closed for external surfaces and only allows local/display-only
    internal packets.
    """

    def __init__(self) -> None:
        self.module = None
        try:
            self.module = importlib.import_module("SarahMemoryAgentFirewall")
        except Exception:
            self.module = None

    def build_smugcc_envelope(self, *, provider: str, source_protocol: str, capability: str, payload: Any) -> Dict[str, Any]:
        payload_hash = _sha256_obj(payload)
        return {
            "schema": "SarahMemory.SMUGCC.envelope.draft.v1",
            "identity": {
                "origin": "external" if provider not in {"local", "sarahmemory"} else "local",
                "provider": provider,
                "subject_id": f"capability::{capability}",
            },
            "protocol": {
                "source_protocol": source_protocol,
                "payload_hash": payload_hash,
            },
            "authority": {
                "requested": ["read_return_only"],
                "forbidden": ["shell", "filesystem", "device", "memory", "execute", "admin", "physical_action"],
            },
            "governance": {
                "safety_policy_required": True,
                "security_governor_required": True,
                "assurance_required": True,
                "operatorcore_required": True,
                "ledger_required": True,
            },
            "passport": {
                "required": provider not in {"local", "sarahmemory"},
                "scope": "cognitive.external.read_return_only",
            },
            "payload": {
                "capability": capability,
                "digest": payload_hash,
            },
            "execution_authority": False,
        }

    def inspect_external_cognitive_surface(self, envelope: Dict[str, Any]) -> Dict[str, Any]:
        if self.module is not None and hasattr(self.module, "guard_smugcc_external_boundary"):
            try:
                return self.module.guard_smugcc_external_boundary(envelope, source=MODULE_NAME, remote_addr="sarahnet-world")
            except Exception as exc:
                return {
                    "ok": False,
                    "verdict": "DENY",
                    "reason": f"agent_firewall_exception:{type(exc).__name__}",
                    "execution_authority": False,
                }
        # Safe fallback: local is allowed; external requires authoritative firewall.
        identity = envelope.get("identity") if isinstance(envelope.get("identity"), dict) else {}
        origin = str(identity.get("origin") or "").lower()
        if origin == "local":
            return {"ok": True, "verdict": "ALLOW", "reason": "fallback_local_display_only", "execution_authority": False}
        return {
            "ok": False,
            "verdict": "REQUIRE_AUTHORITATIVE_AGENT_FIREWALL",
            "reason": "SarahMemoryAgentFirewall_not_importable_in_draft_context",
            "execution_authority": False,
        }


# ---------------------------------------------------------------------------
# Cognitive fabric bridge
# ---------------------------------------------------------------------------

class CognitiveFabricBridge:
    """Adapter into SarahMemoryCognitiveSNFabric when present.

    This bridge is intentionally conservative. It can run local/deterministic
    fabric demos but treats public web resources as unavailable unless the real
    SarahMemory browser executor and policy path are wired separately.
    """

    def __init__(self, firewall: Optional[AgentFirewallBridge] = None) -> None:
        self.firewall = firewall or AgentFirewallBridge()
        self.module = None
        self.fabric = None
        try:
            self.module = importlib.import_module("SarahMemoryCognitiveSNFabric")
        except Exception:
            self.module = None
        if self.module is not None and hasattr(self.module, "build_demo_fabric"):
            try:
                self.fabric = self.module.build_demo_fabric()
            except Exception:
                self.fabric = None

    def execute(self, request: WorldRequest) -> Dict[str, Any]:
        envelope = self.firewall.build_smugcc_envelope(
            provider="sarahmemory",
            source_protocol="local_world_matrix",
            capability=request.capability,
            payload=request.payload,
        )
        verdict = self.firewall.inspect_external_cognitive_surface(envelope)
        if not verdict.get("ok") and str(verdict.get("verdict")) not in {"ALLOW", "REQUIRE_TRUSTREGISTRY_PASSPORT"}:
            return {
                "ok": False,
                "status": "denied",
                "reason": verdict.get("reason", "policy_denied"),
                "firewall": verdict,
                "execution_authority": False,
            }

        if self.fabric is None or self.module is None:
            return {
                "ok": False,
                "status": "unavailable",
                "reason": "SarahMemoryCognitiveSNFabric_not_available_or_not_wired",
                "firewall": verdict,
                "execution_authority": False,
            }

        try:
            cognitive_request = self.module.CognitiveRequest(
                capability=request.capability,
                payload=request.payload,
                requester_id=request.requester_id,
                request_id=request.request_id,
                context=request.context,
                require_consensus=request.require_consensus,
                max_resources=request.max_resources,
            )
            receipt = self.fabric.execute(cognitive_request)
            return {
                "ok": bool(getattr(receipt, "fused_output", None) is not None),
                "status": "ok",
                "receipt": asdict(receipt),
                "firewall": verdict,
                "execution_authority": False,
            }
        except Exception as exc:
            return {
                "ok": False,
                "status": "failed",
                "reason": f"cognitive_fabric_exception:{type(exc).__name__}:{exc}",
                "firewall": verdict,
                "execution_authority": False,
            }


# ---------------------------------------------------------------------------
# XR / Avatar display packet builder
# ---------------------------------------------------------------------------

class WorldDisplayBridge:
    """Converts semantic world state into display packets.

    These packets are read-only visualization contracts. They are suitable for
    AvatarPanel/VRHudRenderer consumption but carry no execution authority.
    """

    def __init__(self, store: WorldStateStore) -> None:
        self.store = store

    @staticmethod
    def _entity_to_target(entity: WorldEntity, idx: int = 0) -> Dict[str, Any]:
        # Screen-space bbox is optional. Default creates a stable diagnostic box
        # rather than pretending to know real camera coordinates.
        seed = int(hashlib.sha256(entity.entity_id.encode("utf-8")).hexdigest()[:8], 16)
        x = 0.10 + ((seed % 70) / 100.0)
        y = 0.15 + (((seed // 7) % 60) / 100.0)
        w = 0.12
        h = 0.10
        return {
            "id": entity.entity_id,
            "label": entity.name,
            "class": entity.entity_type.value,
            "confidence": float(entity.provenance.confidence or 0.50),
            "bbox": [max(0.0, x), max(0.0, y), min(1.0, x + w), min(1.0, y + h)],
            "vectors": {
                "dx": entity.local_position.x,
                "dy": entity.local_position.y,
                "dz_est": entity.local_position.z,
            },
            "world": {
                "object_kind": entity.object_kind.value,
                "epistemic_state": entity.epistemic_state.value,
                "authority_boundary": entity.authority_boundary.value,
            },
        }

    def build_world_packet(self, *, observer_id: str = "sarah", limit: int = 32) -> Dict[str, Any]:
        entities = list(self.store.entities.values())[: max(1, limit)]
        return {
            "ok": True,
            "schema": WORLD_PACKET_SCHEMA,
            "observer_id": observer_id,
            "generated_at": time.time(),
            "world_id": "sarahmemory.world.root",
            "entity_count": len(self.store.entities),
            "relationship_count": len(self.store.relationships),
            "entities": [asdict(e) for e in entities],
            "execution_authority": False,
            "physical_action_authority": False,
        }

    def build_hud_packet(self, *, observer_id: str = "sarah", limit: int = 16) -> Dict[str, Any]:
        entities = list(self.store.entities.values())[: max(1, limit)]
        packet = {
            "schema": HUD_PACKET_SCHEMA,
            "source": MODULE_NAME,
            "world_packet_schema": WORLD_PACKET_SCHEMA,
            "mode": "WORLD_MATRIX_OBSERVE_ONLY",
            "frame": {
                "frame_id": f"world-{int(time.time() * 1000)}",
                "source": "SarahMemoryWorldMatrixFabric",
                "width": 1920,
                "height": 1080,
            },
            "active_targets": [self._entity_to_target(e, i) for i, e in enumerate(entities)],
            "compute_integrity": {
                "thread_state": {"active_threads": 0},
                "memory_pool_mb": "draft",
                "world_entities": len(self.store.entities),
                "world_relationships": len(self.store.relationships),
            },
            "kinetic_integrity": {
                "body_state": "OBSERVE_ONLY",
                "movement_lock": True,
                "devices": [],
            },
            "smget_state": {
                "state": "NO_ACTIVE_ACTION_CONTRACT",
                "decision": "READ_ONLY_WITNESS",
                "rollback_ready": True,
            },
            "authority": {
                "execution_authority": False,
                "physical_action_authority": False,
                "movement_locked": True,
                "user_final_authority": True,
            },
            "generated_at": time.time(),
        }
        self.store.ledger.append(WorldEvent(
            event_id=str(uuid.uuid4()),
            kind=WorldEventKind.XR_PACKET_BUILT,
            subject_id=observer_id,
            summary="Built read-only SMHUD world packet",
            payload={"target_count": len(packet["active_targets"]), "schema": HUD_PACKET_SCHEMA},
            authority_boundary=AuthorityBoundary.DISPLAY_ONLY,
        ))
        return packet

    def build_avatar_panel_packet(self, *, observer_id: str = "sarah", mode: str = "WORLD_VIEW", limit: int = 64) -> Dict[str, Any]:
        entities = list(self.store.entities.values())[: max(1, limit)]
        packet = {
            "ok": True,
            "schema": AVATAR_PANEL_WORLD_PACKET_SCHEMA,
            "source": MODULE_NAME,
            "observer_id": observer_id,
            "suggested_panel_mode": mode,
            "world_id": "sarahmemory.world.root",
            "scene": {
                "title": "SarahMemory World Matrix",
                "entity_count": len(self.store.entities),
                "relationship_count": len(self.store.relationships),
                "entities": [asdict(e) for e in entities],
            },
            "avatar_state_overlay": {
                "attention": "world_matrix",
                "thinking": False,
                "listening": False,
                "speaking": False,
            },
            "controls": {
                "can_execute_actions": False,
                "can_authorize_movement": False,
                "can_control_devices": False,
                "read_only": True,
            },
            "authority": {
                "execution_authority": False,
                "physical_action_authority": False,
                "user_final_authority": True,
            },
            "generated_at": time.time(),
        }
        self.store.ledger.append(WorldEvent(
            event_id=str(uuid.uuid4()),
            kind=WorldEventKind.AVATAR_PACKET_BUILT,
            subject_id=observer_id,
            summary="Built AvatarPanel world packet",
            payload={"entity_count": len(entities), "schema": AVATAR_PANEL_WORLD_PACKET_SCHEMA},
            authority_boundary=AuthorityBoundary.DISPLAY_ONLY,
        ))
        return packet


# ---------------------------------------------------------------------------
# Main integration organ
# ---------------------------------------------------------------------------

class SarahMemoryWorldMatrixFabric:
    """Draft integration organ for SarahMemory World / SarahNet / XR."""

    def __init__(
        self,
        store: Optional[WorldStateStore] = None,
        firewall: Optional[AgentFirewallBridge] = None,
        cognitive: Optional[CognitiveFabricBridge] = None,
    ) -> None:
        self.store = store or WorldStateStore()
        self.firewall = firewall or AgentFirewallBridge()
        self.cognitive = cognitive or CognitiveFabricBridge(self.firewall)
        self.display = WorldDisplayBridge(self.store)

    def boot_seed_world(self) -> Dict[str, Any]:
        """Create minimal root entities for a useful first visualization."""
        root = self.store.upsert_entity(WorldEntity(
            entity_id="world.sarahmemory.root",
            name="SarahMemory World Root",
            entity_type=EntityType.WORLD,
            object_kind=WorldObjectKind.VIRTUAL,
            epistemic_state=EpistemicState.VERIFIED,
            authority_boundary=AuthorityBoundary.DISPLAY_ONLY,
            provenance=Provenance(source=MODULE_NAME, source_kind="system_seed", confidence=1.0),
            properties={
                "description": "Persistent governed semantic world substrate.",
                "rendering_role": "world_root",
            },
        ))
        sarah = self.store.upsert_entity(WorldEntity(
            entity_id="ai.sarah.primary",
            name="Sarah",
            entity_type=EntityType.AI_ENTITY,
            object_kind=WorldObjectKind.VIRTUAL,
            epistemic_state=EpistemicState.VERIFIED,
            authority_boundary=AuthorityBoundary.READ_RETURN_ONLY,
            local_position=Vec3(0.0, 0.0, 0.0, "world_root"),
            provenance=Provenance(source=MODULE_NAME, source_kind="system_seed", confidence=1.0),
            properties={
                "role": "governed_identity",
                "external_models_are_organs": True,
                "execution_authority": False,
            },
        ))
        panel = self.store.upsert_entity(WorldEntity(
            entity_id="surface.avatar_panel.primary",
            name="Avatar Panel Portal",
            entity_type=EntityType.DEVICE,
            object_kind=WorldObjectKind.DIGITAL_TWIN,
            epistemic_state=EpistemicState.VERIFIED,
            authority_boundary=AuthorityBoundary.DISPLAY_ONLY,
            local_position=Vec3(1.0, 0.0, 0.0, "world_root"),
            provenance=Provenance(source=MODULE_NAME, source_kind="system_seed", confidence=1.0),
            properties={
                "surface": "AvatarPanel",
                "modes": ["AVATAR_2D", "AVATAR_3D", "WORLD_VIEW", "XR_VIEW"],
                "read_only": True,
            },
        ))
        hud = self.store.upsert_entity(WorldEntity(
            entity_id="surface.vr_hud.primary",
            name="VR Operator HUD",
            entity_type=EntityType.DEVICE,
            object_kind=WorldObjectKind.DIGITAL_TWIN,
            epistemic_state=EpistemicState.VERIFIED,
            authority_boundary=AuthorityBoundary.DISPLAY_ONLY,
            local_position=Vec3(-1.0, 0.0, 0.0, "world_root"),
            provenance=Provenance(source=MODULE_NAME, source_kind="system_seed", confidence=1.0),
            properties={
                "surface": "VRHudRenderer",
                "mode": "OBSERVE_ONLY",
                "read_only": True,
            },
        ))
        self.store.add_relationship(WorldRelationship(root.entity_id, "contains", sarah.entity_id, 1.0))
        self.store.add_relationship(WorldRelationship(root.entity_id, "contains", panel.entity_id, 1.0))
        self.store.add_relationship(WorldRelationship(root.entity_id, "contains", hud.entity_id, 1.0))
        self.store.save()
        return asdict(self.store.snapshot())

    def submit_observation(
        self,
        *,
        entity_id: str,
        name: str,
        entity_type: EntityType = EntityType.UNKNOWN,
        object_kind: WorldObjectKind = WorldObjectKind.DIGITAL_TWIN,
        epistemic_state: EpistemicState = EpistemicState.OBSERVED,
        source: str = "unknown",
        confidence: float = 0.50,
        properties: Optional[Dict[str, Any]] = None,
    ) -> WorldEntity:
        entity = WorldEntity(
            entity_id=entity_id,
            name=name,
            entity_type=entity_type,
            object_kind=object_kind,
            epistemic_state=epistemic_state,
            authority_boundary=AuthorityBoundary.DISPLAY_ONLY,
            provenance=Provenance(source=source, source_kind="observation", confidence=max(0.0, min(1.0, confidence))),
            properties=properties or {},
        )
        out = self.store.upsert_entity(entity)
        self.store.ledger.append(WorldEvent(
            event_id=str(uuid.uuid4()),
            kind=WorldEventKind.OBSERVATION,
            subject_id=out.entity_id,
            summary=f"Observation submitted: {out.name}",
            payload={"entity": asdict(out)},
            authority_boundary=AuthorityBoundary.DISPLAY_ONLY,
        ))
        self.store.save()
        return out

    def run_cognitive_request(self, request: WorldRequest) -> Dict[str, Any]:
        result = self.cognitive.execute(request)
        self.store.ledger.append(WorldEvent(
            event_id=str(uuid.uuid4()),
            kind=WorldEventKind.COGNITIVE_RECEIPT if result.get("ok") else WorldEventKind.POLICY_DENIED,
            subject_id=request.request_id,
            summary=f"Cognitive request {result.get('status')}: {request.capability}",
            payload=result,
            authority_boundary=AuthorityBoundary.READ_RETURN_ONLY,
        ))
        return result

    def fork_scenario(self, scenario_name: str, assumption_delta: Dict[str, Any]) -> Dict[str, Any]:
        scenario_id = f"scenario.{uuid.uuid4()}"
        base_hash = _sha256_obj(asdict(self.store.snapshot(limit=1000000)))
        event = WorldEvent(
            event_id=str(uuid.uuid4()),
            kind=WorldEventKind.SCENARIO_FORK,
            subject_id=scenario_id,
            summary=f"Scenario fork created: {scenario_name}",
            payload={
                "scenario_id": scenario_id,
                "scenario_name": scenario_name,
                "base_world_hash": base_hash,
                "assumption_delta": assumption_delta,
                "simulation_only": True,
            },
            authority_boundary=AuthorityBoundary.SIMULATION_ONLY,
        )
        self.store.ledger.append(event)
        return {
            "ok": True,
            "schema": "SARAHMEMORY_WORLD_SCENARIO_FORK_V1",
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "base_world_hash": base_hash,
            "assumption_delta": assumption_delta,
            "execution_authority": False,
            "physical_action_authority": False,
        }

    def build_packets(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "world": self.display.build_world_packet(),
            "avatar_panel": self.display.build_avatar_panel_packet(),
            "vr_hud": self.display.build_hud_packet(),
            "execution_authority": False,
            "physical_action_authority": False,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def build_demo_world() -> SarahMemoryWorldMatrixFabric:
    world = SarahMemoryWorldMatrixFabric()
    world.boot_seed_world()
    world.submit_observation(
        entity_id="resource.local.reasoner",
        name="Local Reasoner Capability",
        entity_type=EntityType.MODEL_RESOURCE,
        object_kind=WorldObjectKind.VIRTUAL,
        epistemic_state=EpistemicState.VERIFIED,
        source=MODULE_NAME,
        confidence=0.95,
        properties={
            "capability": "reason.compare",
            "access_mode": "local",
            "authority": "read_return_only",
        },
    )
    world.submit_observation(
        entity_id="surface.public_web_model.placeholder",
        name="Public Web Model Placeholder",
        entity_type=EntityType.MODEL_RESOURCE,
        object_kind=WorldObjectKind.VIRTUAL,
        epistemic_state=EpistemicState.UNKNOWN,
        source=MODULE_NAME,
        confidence=0.25,
        properties={
            "capability": "unknown",
            "access_mode": "public_web_ui",
            "automation_permitted": None,
            "blocked_until_policy_verified": True,
        },
    )
    return world


def _smoke_test() -> None:
    world = build_demo_world()
    cognitive = world.run_cognitive_request(WorldRequest(
        capability="reason.compare",
        payload="SarahMemory World Matrix smoke test",
        requester_id="sarah",
        require_consensus=True,
        max_resources=2,
    ))
    packets = world.build_packets()
    scenario = world.fork_scenario("demo_counterfactual", {"assumption": "resource.local.reasoner unavailable"})
    print(json.dumps({
        "module": MODULE_NAME,
        "version": MODULE_VERSION,
        "cognitive": cognitive,
        "packets": {
            "world_schema": packets["world"]["schema"],
            "avatar_panel_schema": packets["avatar_panel"]["schema"],
            "vr_hud_schema": packets["vr_hud"]["schema"],
            "vr_target_count": len(packets["vr_hud"].get("active_targets", [])),
        },
        "scenario": scenario,
    }, indent=2, default=str))


if __name__ == "__main__":
    _smoke_test()
