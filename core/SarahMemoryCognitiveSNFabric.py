"""--==The SarahMemory Project==--
File: SarahMemoryCognitiveSNFabric.py
Part of the SarahMemory Companion AI-bot Platform
Version: v9.0.0
Date: 2026-09-28
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


Purpose
-------
Prototype a governed SarahNet "distributed cognitive fabric" that presents many
replaceable computational resources as one SarahMemory capability surface.

- merge or copy third-party model weights;
- bypass authentication, subscriptions, quotas, CAPTCHAs, robots controls,
  access controls, or provider policies;
- reverse engineer hidden/private endpoints;
- grant external models authority over SarahMemory;
- execute physical-world actions.

Public browser resources are represented through an adapter interface. A real
browser executor should only automate sites whose terms/policies permit that
use and should use the ordinary public UI.

Architecture
------------
Request -> Capability Planner -> Policy Gate -> Candidate Resources
        -> Task Assembly -> Execute -> Validate/Compare -> Fuse
        -> Provenance Receipt -> SarahMemory

External models are temporary cognitive organs. SarahMemory retains identity,
governance, memory, provenance, routing, and final authority.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple


DRAFT_VERSION = "0.1.0"


class AccessMode(str, Enum):
    LOCAL = "local"
    SARAHNET_PEER = "sarahnet_peer"
    PUBLIC_WEB_UI = "public_web_ui"
    DETERMINISTIC = "deterministic"


class TrustState(str, Enum):
    TRUSTED = "trusted"
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    BLOCKED = "blocked"


class ExecutionStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    DENIED = "denied"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class Capability:
    name: str
    input_types: Tuple[str, ...] = ("text",)
    output_types: Tuple[str, ...] = ("text",)
    description: str = ""


@dataclass
class ResourceDescriptor:
    resource_id: str
    display_name: str
    access_mode: AccessMode
    capabilities: List[Capability]
    endpoint_hint: Optional[str] = None
    requires_auth: bool = False
    requires_subscription: bool = False
    requires_download: bool = False
    public_ui: bool = False
    automation_permitted: Optional[bool] = None
    trust: TrustState = TrustState.UNVERIFIED
    availability: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def supports(self, capability_name: str) -> bool:
        return any(c.name == capability_name for c in self.capabilities)


@dataclass
class CognitiveRequest:
    capability: str
    payload: Any
    requester_id: str = "sarah"
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    context: Dict[str, Any] = field(default_factory=dict)
    require_consensus: bool = False
    max_resources: int = 3


@dataclass
class SMUGCCEnvelope:
    """Minimal draft envelope; replace with the project's authoritative SMUGCC schema."""

    envelope_id: str
    request_id: str
    requester_id: str
    capability: str
    authority_scope: str
    payload_digest: str
    created_at: float
    constraints: Dict[str, Any]


@dataclass
class ResourceResult:
    resource_id: str
    status: ExecutionStatus
    output: Any = None
    error: Optional[str] = None
    elapsed_ms: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FabricReceipt:
    request_id: str
    capability: str
    selected_resources: List[str]
    results: List[ResourceResult]
    fused_output: Any
    confidence: float
    disagreements: List[str]
    timestamp: float
    receipt_hash: str = ""

    def seal(self) -> "FabricReceipt":
        material = json.dumps(
            {
                "request_id": self.request_id,
                "capability": self.capability,
                "selected_resources": self.selected_resources,
                "results": [asdict(r) for r in self.results],
                "fused_output": self.fused_output,
                "confidence": self.confidence,
                "disagreements": self.disagreements,
                "timestamp": self.timestamp,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        self.receipt_hash = hashlib.sha256(material).hexdigest()
        return self


class ResourceAdapter(Protocol):
    def execute(self, resource: ResourceDescriptor, request: CognitiveRequest) -> ResourceResult:
        ...


class PolicyGate:
    """Draft stand-in for AgentFirewall + governance/assurance policy."""

    def authorize_resource(self, resource: ResourceDescriptor, request: CognitiveRequest) -> Tuple[bool, str]:
        if resource.trust == TrustState.BLOCKED:
            return False, "resource_blocked"
        if not resource.availability:
            return False, "resource_unavailable"
        if resource.requires_download:
            return False, "downloads_disallowed"
        if resource.requires_auth:
            return False, "authentication_disallowed"
        if resource.requires_subscription:
            return False, "subscription_disallowed"
        if resource.access_mode == AccessMode.PUBLIC_WEB_UI:
            if not resource.public_ui:
                return False, "not_public_ui"
            if resource.automation_permitted is not True:
                return False, "web_automation_not_explicitly_permitted"
        if not resource.supports(request.capability):
            return False, "capability_mismatch"
        return True, "authorized"


class CapabilityRegistry:
    def __init__(self) -> None:
        self._resources: Dict[str, ResourceDescriptor] = {}

    def register(self, resource: ResourceDescriptor) -> None:
        self._resources[resource.resource_id] = resource

    def unregister(self, resource_id: str) -> None:
        self._resources.pop(resource_id, None)

    def candidates(self, capability: str) -> List[ResourceDescriptor]:
        return [r for r in self._resources.values() if r.supports(capability)]

    def snapshot(self) -> Dict[str, Any]:
        return {rid: asdict(resource) for rid, resource in self._resources.items()}


class CallableAdapter:
    """Adapter for approved local/deterministic functions used during testing."""

    def __init__(self, functions: Dict[str, Callable[[Any, Dict[str, Any]], Any]]) -> None:
        self.functions = functions

    def execute(self, resource: ResourceDescriptor, request: CognitiveRequest) -> ResourceResult:
        started = time.perf_counter()
        fn = self.functions.get(resource.resource_id)
        if fn is None:
            return ResourceResult(resource.resource_id, ExecutionStatus.UNAVAILABLE, error="no_callable")
        try:
            output = fn(request.payload, request.context)
            return ResourceResult(
                resource_id=resource.resource_id,
                status=ExecutionStatus.OK,
                output=output,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                evidence={"access_mode": resource.access_mode.value},
            )
        except Exception as exc:  # draft boundary: adapters convert failures to receipts
            return ResourceResult(
                resource_id=resource.resource_id,
                status=ExecutionStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )


class PublicWebUIAdapter:
    """
    Safe placeholder for ordinary public-browser interaction.

    Wire this to SarahMemory's approved browser/UI agent only after the resource
    is confirmed public and automation is permitted. It intentionally contains
    no scraping, credential, CAPTCHA, anti-bot, hidden-endpoint, or bypass code.
    """

    def execute(self, resource: ResourceDescriptor, request: CognitiveRequest) -> ResourceResult:
        return ResourceResult(
            resource_id=resource.resource_id,
            status=ExecutionStatus.UNAVAILABLE,
            error="browser_executor_not_wired_in_draft",
            evidence={
                "public_ui": resource.public_ui,
                "automation_permitted": resource.automation_permitted,
                "endpoint_hint": resource.endpoint_hint,
            },
        )


class ResultFusion:
    """Conservative draft fusion: retain disagreements rather than invent consensus."""

    @staticmethod
    def fuse(results: Sequence[ResourceResult]) -> Tuple[Any, float, List[str]]:
        successful = [r for r in results if r.status == ExecutionStatus.OK]
        if not successful:
            return None, 0.0, ["no_successful_resources"]
        if len(successful) == 1:
            return successful[0].output, 0.50, []

        normalized = [json.dumps(r.output, sort_keys=True, default=str) for r in successful]
        counts: Dict[str, int] = {}
        for value in normalized:
            counts[value] = counts.get(value, 0) + 1
        winner, winner_count = max(counts.items(), key=lambda item: item[1])
        confidence = winner_count / len(successful)
        disagreements = []
        if len(counts) > 1:
            disagreements.append(f"{len(counts)} distinct outputs across {len(successful)} resources")

        # Return the original object matching the winning normalized representation.
        for result, norm in zip(successful, normalized):
            if norm == winner:
                return result.output, confidence, disagreements
        return successful[0].output, confidence, disagreements


class SarahNetCognitiveFabric:
    def __init__(self, registry: Optional[CapabilityRegistry] = None, gate: Optional[PolicyGate] = None) -> None:
        self.registry = registry or CapabilityRegistry()
        self.gate = gate or PolicyGate()
        self.adapters: Dict[AccessMode, ResourceAdapter] = {}
        self.receipts: List[FabricReceipt] = []

    def bind_adapter(self, mode: AccessMode, adapter: ResourceAdapter) -> None:
        self.adapters[mode] = adapter

    def _make_envelope(self, request: CognitiveRequest) -> SMUGCCEnvelope:
        digest = hashlib.sha256(json.dumps(request.payload, sort_keys=True, default=str).encode()).hexdigest()
        return SMUGCCEnvelope(
            envelope_id=str(uuid.uuid4()),
            request_id=request.request_id,
            requester_id=request.requester_id,
            capability=request.capability,
            authority_scope="cognitive.external.read_return_only",
            payload_digest=digest,
            created_at=time.time(),
            constraints={
                "no_download": True,
                "no_authentication": True,
                "no_subscription": True,
                "no_access_control_bypass": True,
                "external_authority": False,
            },
        )

    def plan(self, request: CognitiveRequest) -> Tuple[SMUGCCEnvelope, List[ResourceDescriptor], List[str]]:
        envelope = self._make_envelope(request)
        allowed: List[ResourceDescriptor] = []
        denied: List[str] = []

        # Prefer trusted/verified resources, then stable IDs for deterministic routing.
        trust_order = {TrustState.TRUSTED: 0, TrustState.VERIFIED: 1, TrustState.UNVERIFIED: 2, TrustState.BLOCKED: 3}
        candidates = sorted(
            self.registry.candidates(request.capability),
            key=lambda r: (trust_order[r.trust], r.resource_id),
        )
        for resource in candidates:
            ok, reason = self.gate.authorize_resource(resource, request)
            if ok:
                allowed.append(resource)
            else:
                denied.append(f"{resource.resource_id}:{reason}")
            if len(allowed) >= max(1, request.max_resources):
                break
        return envelope, allowed, denied

    def execute(self, request: CognitiveRequest) -> FabricReceipt:
        envelope, selected, denied = self.plan(request)
        results: List[ResourceResult] = []

        for resource in selected:
            adapter = self.adapters.get(resource.access_mode)
            if adapter is None:
                results.append(ResourceResult(resource.resource_id, ExecutionStatus.UNAVAILABLE, error="adapter_not_bound"))
                continue
            result = adapter.execute(resource, request)
            result.evidence.setdefault("smugcc_envelope_id", envelope.envelope_id)
            result.evidence.setdefault("payload_digest", envelope.payload_digest)
            results.append(result)

        fused, confidence, disagreements = ResultFusion.fuse(results)
        disagreements.extend(denied)

        receipt = FabricReceipt(
            request_id=request.request_id,
            capability=request.capability,
            selected_resources=[r.resource_id for r in selected],
            results=results,
            fused_output=fused,
            confidence=confidence,
            disagreements=disagreements,
            timestamp=time.time(),
        ).seal()
        self.receipts.append(receipt)
        return receipt


# ---------------------------------------------------------------------------
# Draft capability-discovery normalization
# ---------------------------------------------------------------------------

def normalize_discovered_resource(raw: Dict[str, Any]) -> ResourceDescriptor:
    """
    Convert discovery metadata into SarahNet's vendor-neutral descriptor.

    A future discovery organ can populate `raw` from public directories,
    provider-published capability documents, or an approved browser inspection.
    Discovery alone never grants execution authority.
    """
    capabilities = [
        Capability(
            name=c["name"],
            input_types=tuple(c.get("input_types", ["text"])),
            output_types=tuple(c.get("output_types", ["text"])),
            description=c.get("description", ""),
        )
        for c in raw.get("capabilities", [])
    ]
    return ResourceDescriptor(
        resource_id=raw["resource_id"],
        display_name=raw.get("display_name", raw["resource_id"]),
        access_mode=AccessMode(raw.get("access_mode", AccessMode.PUBLIC_WEB_UI.value)),
        capabilities=capabilities,
        endpoint_hint=raw.get("endpoint_hint"),
        requires_auth=bool(raw.get("requires_auth", False)),
        requires_subscription=bool(raw.get("requires_subscription", False)),
        requires_download=bool(raw.get("requires_download", False)),
        public_ui=bool(raw.get("public_ui", False)),
        automation_permitted=raw.get("automation_permitted"),
        trust=TrustState(raw.get("trust", TrustState.UNVERIFIED.value)),
        availability=bool(raw.get("availability", True)),
        metadata=dict(raw.get("metadata", {})),
    )


def build_demo_fabric() -> SarahNetCognitiveFabric:
    """Self-contained smoke-test fabric; no network access."""
    registry = CapabilityRegistry()
    cap = Capability("reason.compare", ("text",), ("text",), "Bounded reasoning candidate")

    registry.register(ResourceDescriptor(
        resource_id="local_reasoner_a",
        display_name="Local Reasoner A",
        access_mode=AccessMode.LOCAL,
        capabilities=[cap],
        trust=TrustState.TRUSTED,
    ))
    registry.register(ResourceDescriptor(
        resource_id="local_reasoner_b",
        display_name="Local Reasoner B",
        access_mode=AccessMode.LOCAL,
        capabilities=[cap],
        trust=TrustState.VERIFIED,
    ))

    # Demonstrates a web resource that is discoverable but NOT executable until
    # automation permission is explicitly known to be allowed.
    registry.register(ResourceDescriptor(
        resource_id="example_public_web_model",
        display_name="Example Public Web Model",
        access_mode=AccessMode.PUBLIC_WEB_UI,
        capabilities=[cap],
        public_ui=True,
        automation_permitted=None,
        trust=TrustState.UNVERIFIED,
        endpoint_hint="https://example.invalid/public-demo",
    ))

    functions = {
        "local_reasoner_a": lambda payload, ctx: {"answer": str(payload).strip(), "source": "A"},
        "local_reasoner_b": lambda payload, ctx: {"answer": str(payload).strip(), "source": "B"},
    }

    fabric = SarahNetCognitiveFabric(registry)
    fabric.bind_adapter(AccessMode.LOCAL, CallableAdapter(functions))
    fabric.bind_adapter(AccessMode.PUBLIC_WEB_UI, PublicWebUIAdapter())
    return fabric


def _smoke_test() -> None:
    fabric = build_demo_fabric()
    request = CognitiveRequest(
        capability="reason.compare",
        payload="SarahNet cognitive fabric smoke test",
        require_consensus=True,
        max_resources=3,
    )
    receipt = fabric.execute(request)
    print(json.dumps(asdict(receipt), indent=2, default=str))


if __name__ == "__main__":
    _smoke_test()
