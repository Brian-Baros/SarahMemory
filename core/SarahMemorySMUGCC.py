"""--==The SarahMemory Project==--
File: SarahMemorySMUGCC.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-09-16
Author: Brian Lee Baros / SOFTDEV0 LLC

===============================================================================
SarahMemory Universal Governed Cognitive Contract (SMUGCC)

SMUGCC is a public, vendor-neutral cognitive ABI for external systems that want
to become SarahMemory-compatible. It is contract-only and validator-only.

DOCTRINE:
- SMUGCC does not execute agents, tools, models, devices, shell commands, or IO.
- SMUGCC does not issue passports, grant authority, write memory, or route work.
- SMUGCC maps external vocabulary into existing SarahMemory owners.
- Existing organs remain authoritative for runtime behavior.
- Importing this module has no side effects.
===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A-"
# ROLE = "smugcc_contract_validator"
# CATEGORY = "external_cognitive_contract"
# USER_FACING = False
# UI_EXPOSURE = "api_read_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "smugcc"
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "smugcc_contract"
# FAMILY = "governed_cognitive_contract"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = True
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-09-16"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Vendor-neutral contract/schema/validator layer only. No execution authority, no provider calls, no passport issuance, no memory writes."
# --- SARAHMETA END ---

import copy
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence


MODULE_NAME = "SarahMemorySMUGCC"
MODULE_VERSION = "1.0.0"
SMUGCC_SCHEMA = "SarahMemory.SMUGCC.v1"
SMUGCC_CONTRACT_VERSION = "1.0.0"

NO_EXECUTION_AUTHORITY = {
    "execution_authority": False,
    "contract_only": True,
    "validator_only": True,
    "issues_passports": False,
    "calls_external_providers": False,
    "writes_memory": False,
    "controls_devices": False,
    "shell_execution": False,
}

SMUGCC_LIFECYCLE = [
    "DISCOVER",
    "IDENTIFY",
    "DECLARE",
    "NEGOTIATE",
    "PASSPORT",
    "EXECUTE",
    "RETURN",
    "VERIFY",
    "AUDIT",
    "DISCONNECT",
]

SMUGCC_PIPELINE = [
    "External Provider / Tool / Agent / Robot / Service",
    "Provider Adapter",
    "SMUGCC Contract Envelope",
    "SML/QSML Packet",
    "Neuron Routing",
    "Governance Spine",
    "OperatorCore / Execution Surface",
    "Evidence Capture",
    "Return Verification",
    "Ledger/Audit",
    "User/System Release",
]

SMUGCC_OWNERSHIP_MAP = {
    "smugcc": {
        "owner": "SarahMemorySMUGCC",
        "owns": [
            "external contract vocabulary",
            "canonical envelope schema",
            "envelope validation",
            "adapter declaration contracts",
            "compatibility reporting",
        ],
        "does_not_own": [
            "execution",
            "routing authority",
            "passport issuance",
            "policy enforcement",
            "ledger persistence",
            "memory writes",
            "device control",
        ],
    },
    "sml_qsml": {"owner": "SarahMemorySMLProtocol", "owns": ["internal packet normalization", "cognitive packet schema"]},
    "neuron": {"owner": "SarahMemoryNeuron", "owns": ["routing", "activation"]},
    "trust": {"owner": "SarahMemoryTrustRegistry", "owns": ["identity", "capability grants", "agent passports"]},
    "firewall": {"owner": "SarahMemoryAgentFirewall", "owns": ["external-agent containment", "inbound/outbound boundary inspection"]},
    "safety": {"owner": "SarahMemorySafetyPolicies", "owns": ["safety policy"]},
    "security": {"owner": "SarahMemorySecurityGovernor", "owns": ["security enforcement"]},
    "assurance": {"owner": "SarahMemoryAssuranceGate", "owns": ["proof requirements"]},
    "compare": {"owner": "SarahMemoryCompare", "owns": ["evidence", "diff", "claim verification"]},
    "compass": {"owner": "SarahMemoryCognitiveCompass", "owns": ["direction", "trajectory alignment"]},
    "operatorcore": {"owner": "SarahMemoryOperatorCore", "owns": ["final action gating"]},
    "ledger": {"owner": "SarahMemoryLedger", "owns": ["receipts", "audit"]},
    "api": {"owner": "SarahMemoryAPI", "owns": ["provider communications"]},
    "terminal": {"owner": "SarahMemoryTerminal", "owns": ["governed task and mission interface"]},
    "nailde": {"owner": "SarahMemoryNAILDE", "owns": ["sandbox-first adapter and app development"]},
    "network": {"owner": "SarahMemoryNetwork / SarahNet", "owns": ["transport", "protocol normalization"]},
}

CANONICAL_SMUGCC_SCHEMA: Dict[str, Any] = {
    "schema": SMUGCC_SCHEMA,
    "contract_version": SMUGCC_CONTRACT_VERSION,
    "identity": {
        "subject_id": "",
        "provider": "",
        "implementation": "",
        "version": "",
        "origin": "",
        "trust_level": "unknown",
    },
    "protocol": {
        "source_protocol": "",
        "source_version": "",
        "adapter_id": "",
        "adapter_version": "",
    },
    "mission": {
        "mission_id": "",
        "task_id": "",
        "objective": "",
        "intent": "",
        "requested_by": "",
        "created_at": "",
    },
    "capabilities": {
        "declared": [],
        "requested": [],
        "granted": [],
        "denied": [],
    },
    "authority": {
        "requested": [],
        "granted": [],
        "denied": [],
        "execution_authority": False,
        "requires_user_approval": True,
    },
    "resources": {
        "allowed_sources": [],
        "denied_sources": [],
        "allowed_methods": ["GET"],
        "filesystem_allowed": False,
        "network_allowed": False,
        "memory_allowed": False,
        "device_allowed": False,
        "shell_allowed": False,
    },
    "passport": {
        "required": True,
        "passport_id": "",
        "issued_by": "SarahMemoryTrustRegistry",
        "expires_at": "",
        "one_time_use": True,
        "return_nonce_required": True,
    },
    "governance": {
        "risk_level": "unknown",
        "safety_required": True,
        "security_required": True,
        "assurance_required": True,
        "compare_required": True,
        "compass_required": True,
        "operatorcore_required": True,
        "ledger_required": True,
    },
    "payload": {
        "input": None,
        "normalized": None,
        "vendor_payload_hash": "",
        "smugcc_payload_hash": "",
    },
    "return_contract": {
        "expected_output_type": "evidence_bound_result",
        "evidence_required": True,
        "signature_required": False,
        "hash_required": True,
        "receipt_required": True,
    },
    "audit": {
        "receipt_id": "",
        "receipt_hash": "",
        "retention_class": "contract",
        "trace_id": "",
    },
}

ADAPTER_DECLARATIONS: Dict[str, Dict[str, Any]] = {
    "openai_style_provider": {
        "adapter_id": "openai_style_provider",
        "source_protocol": "OpenAI-style",
        "translates_to": SMUGCC_SCHEMA,
        "capabilities_declared": ["chat.request", "tool.result.return", "evidence.return"],
        "credential_aliases_required": ["OPENAI_API_ALIAS"],
        "limits": {"execution_authority": False, "provider_calls": False},
        "evidence_support": ["response_hash", "tool_call_trace"],
        "error_normalization": True,
    },
    "anthropic_mcp_style_provider": {
        "adapter_id": "anthropic_mcp_style_provider",
        "source_protocol": "Anthropic/MCP-style",
        "translates_to": SMUGCC_SCHEMA,
        "capabilities_declared": ["message.request", "mcp.tool.declare", "evidence.return"],
        "credential_aliases_required": ["ANTHROPIC_API_ALIAS", "MCP_SERVER_ALIAS"],
        "limits": {"execution_authority": False, "provider_calls": False},
        "evidence_support": ["tool_result_hash", "server_trace_id"],
        "error_normalization": True,
    },
    "google_a2a_style_provider": {
        "adapter_id": "google_a2a_style_provider",
        "source_protocol": "Google/A2A-style",
        "translates_to": SMUGCC_SCHEMA,
        "capabilities_declared": ["agent.declare", "task.negotiate", "evidence.return"],
        "credential_aliases_required": ["GOOGLE_PROVIDER_ALIAS"],
        "limits": {"execution_authority": False, "provider_calls": False},
        "evidence_support": ["agent_trace", "payload_hash"],
        "error_normalization": True,
    },
    "nvidia_nim_agentiq_style_provider": {
        "adapter_id": "nvidia_nim_agentiq_style_provider",
        "source_protocol": "NVIDIA/Nemotron/NIM/AgentIQ-style",
        "translates_to": SMUGCC_SCHEMA,
        "capabilities_declared": ["model.endpoint.declare", "agent.workflow.declare", "evidence.return"],
        "credential_aliases_required": ["NVIDIA_PROVIDER_ALIAS"],
        "limits": {"execution_authority": False, "provider_calls": False},
        "evidence_support": ["model_result_hash", "workflow_trace"],
        "error_normalization": True,
    },
    "microsoft_mai_azure_style_provider": {
        "adapter_id": "microsoft_mai_azure_style_provider",
        "source_protocol": "Microsoft/MAI/Azure-style",
        "translates_to": SMUGCC_SCHEMA,
        "capabilities_declared": ["assistant.request", "azure.tool.declare", "evidence.return"],
        "credential_aliases_required": ["MICROSOFT_PROVIDER_ALIAS"],
        "limits": {"execution_authority": False, "provider_calls": False},
        "evidence_support": ["request_id", "payload_hash"],
        "error_normalization": True,
    },
    "local_ollama_style_provider": {
        "adapter_id": "local_ollama_style_provider",
        "source_protocol": "local-model/Ollama-style",
        "translates_to": SMUGCC_SCHEMA,
        "capabilities_declared": ["local_model.request", "local_model.metadata", "evidence.return"],
        "credential_aliases_required": [],
        "limits": {"execution_authority": False, "provider_calls": False},
        "evidence_support": ["model_name", "prompt_hash", "response_hash"],
        "error_normalization": True,
    },
    "generic_rest_tool": {
        "adapter_id": "generic_rest_tool",
        "source_protocol": "generic REST tool",
        "translates_to": SMUGCC_SCHEMA,
        "capabilities_declared": ["http.get.declare", "tool.metadata", "evidence.return"],
        "credential_aliases_required": ["REST_TOOL_CREDENTIAL_ALIAS"],
        "limits": {"execution_authority": False, "allowed_methods": ["GET"], "provider_calls": False},
        "evidence_support": ["status_code", "body_hash"],
        "error_normalization": True,
    },
    "generic_device_robot_plc_telemetry_node": {
        "adapter_id": "generic_device_robot_plc_telemetry_node",
        "source_protocol": "generic device/robot/PLC telemetry node",
        "translates_to": SMUGCC_SCHEMA,
        "capabilities_declared": ["telemetry.read", "device.metadata", "evidence.return"],
        "credential_aliases_required": ["DEVICE_TELEMETRY_ALIAS"],
        "limits": {"execution_authority": False, "device_control": False, "telemetry_only": True},
        "evidence_support": ["telemetry_hash", "source_timestamp"],
        "error_normalization": True,
    },
}

SML_ORGAN_METADATA = {
    "name": MODULE_NAME,
    "role": "external_cognitive_contract_validator",
    "version": MODULE_VERSION,
    "protocol": SMUGCC_SCHEMA,
    "capabilities": ["contract_schema", "contract_validation", "adapter_declaration", "compatibility_report"],
    "execution_authority": False,
    "internal_only": False,
    "metadata": {"contract_only": True, "no_execution_authority": True},
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _sha256_obj(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8", errors="replace")).hexdigest()


def _deepcopy_dict(value: Mapping[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(dict(value))


def _merge_section(defaults: Mapping[str, Any], value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    out = _deepcopy_dict(defaults)
    if isinstance(value, Mapping):
        for key, item in value.items():
            out[str(key)] = copy.deepcopy(item)
    return out


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, set):
        return sorted(value)
    return [value]


def _text(value: Any) -> str:
    return str(value or "").strip()


def _add_error(errors: List[Dict[str, Any]], code: str, path: str, message: str, owner: str) -> None:
    errors.append({"code": code, "path": path, "message": message, "owner": owner})


def build_smugcc_envelope(
    *,
    identity: Optional[Mapping[str, Any]] = None,
    protocol: Optional[Mapping[str, Any]] = None,
    mission: Optional[Mapping[str, Any]] = None,
    capabilities: Optional[Mapping[str, Any]] = None,
    authority: Optional[Mapping[str, Any]] = None,
    resources: Optional[Mapping[str, Any]] = None,
    passport: Optional[Mapping[str, Any]] = None,
    governance: Optional[Mapping[str, Any]] = None,
    payload: Optional[Mapping[str, Any]] = None,
    return_contract: Optional[Mapping[str, Any]] = None,
    audit: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a canonical SMUGCC envelope without executing or authorizing work."""
    defaults = CANONICAL_SMUGCC_SCHEMA
    envelope = {
        "schema": SMUGCC_SCHEMA,
        "contract_version": SMUGCC_CONTRACT_VERSION,
        "identity": _merge_section(defaults["identity"], identity),
        "protocol": _merge_section(defaults["protocol"], protocol),
        "mission": _merge_section(defaults["mission"], mission),
        "capabilities": _merge_section(defaults["capabilities"], capabilities),
        "authority": _merge_section(defaults["authority"], authority),
        "resources": _merge_section(defaults["resources"], resources),
        "passport": _merge_section(defaults["passport"], passport),
        "governance": _merge_section(defaults["governance"], governance),
        "payload": _merge_section(defaults["payload"], payload),
        "return_contract": _merge_section(defaults["return_contract"], return_contract),
        "audit": _merge_section(defaults["audit"], audit),
    }
    if not _text(envelope["mission"].get("created_at")):
        envelope["mission"]["created_at"] = _utc_now()
    if not _text(envelope["mission"].get("mission_id")) or not _text(envelope["mission"].get("task_id")):
        draft_seed = {
            "identity": envelope["identity"],
            "protocol": envelope["protocol"],
            "objective": envelope["mission"].get("objective"),
            "intent": envelope["mission"].get("intent"),
            "requested_by": envelope["mission"].get("requested_by"),
            "payload_hash": envelope["payload"].get("vendor_payload_hash"),
        }
        draft_id = "smugcc-" + _sha256_obj(draft_seed)[:16]
        envelope["mission"]["mission_id"] = _text(envelope["mission"].get("mission_id")) or draft_id
        envelope["mission"]["task_id"] = _text(envelope["mission"].get("task_id")) or draft_id
    envelope["authority"]["execution_authority"] = False
    envelope["authority"].setdefault("requires_user_approval", True)
    envelope["resources"].setdefault("allowed_methods", ["GET"])
    envelope["payload"]["vendor_payload_hash"] = envelope["payload"].get("vendor_payload_hash") or _sha256_obj(envelope["payload"].get("input"))
    payload_for_hash = copy.deepcopy(envelope["payload"])
    payload_for_hash["smugcc_payload_hash"] = ""
    envelope["payload"]["smugcc_payload_hash"] = _sha256_obj(payload_for_hash)
    return envelope


def validate_adapter_declaration(declaration: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate an adapter declaration only; never invoke the adapter."""
    errors: List[Dict[str, Any]] = []
    if not isinstance(declaration, Mapping):
        return {"ok": False, "errors": [{"code": "adapter_not_object", "path": "$", "message": "Adapter declaration must be an object.", "owner": MODULE_NAME}]}
    for key in ("adapter_id", "source_protocol", "translates_to"):
        if not _text(declaration.get(key)):
            _add_error(errors, f"missing_{key}", key, f"Adapter declaration requires {key}.", MODULE_NAME)
    if declaration.get("translates_to") != SMUGCC_SCHEMA:
        _add_error(errors, "adapter_wrong_target", "translates_to", "Adapter must target the SMUGCC schema.", MODULE_NAME)
    limits = declaration.get("limits") if isinstance(declaration.get("limits"), Mapping) else {}
    if bool(limits.get("execution_authority")):
        _add_error(errors, "adapter_execution_authority_denied", "limits.execution_authority", "Adapters cannot own execution authority.", "SarahMemoryOperatorCore")
    if bool(limits.get("device_control")):
        _add_error(errors, "adapter_device_control_denied", "limits.device_control", "Adapters cannot control devices.", "SarahMemorySecurityGovernor")
    return {"ok": not errors, "schema": "SarahMemory.SMUGCC.adapter_declaration.v1", "errors": errors, "execution_authority": False}


def validate_smugcc_envelope(envelope: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate a SMUGCC envelope deterministically with fail-closed defaults."""
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    if not isinstance(envelope, Mapping):
        _add_error(errors, "envelope_not_object", "$", "SMUGCC envelope must be an object.", MODULE_NAME)
        return {"ok": False, "schema": SMUGCC_SCHEMA, "errors": errors, "warnings": warnings, "execution_authority": False}

    if envelope.get("schema") != SMUGCC_SCHEMA:
        _add_error(errors, "schema_mismatch", "schema", "Envelope schema must be SarahMemory.SMUGCC.v1.", MODULE_NAME)
    if _text(envelope.get("contract_version")) != SMUGCC_CONTRACT_VERSION:
        _add_error(errors, "contract_version_mismatch", "contract_version", "Contract version must be 1.0.0.", MODULE_NAME)

    required_sections = [
        "identity",
        "protocol",
        "mission",
        "capabilities",
        "authority",
        "resources",
        "passport",
        "governance",
        "payload",
        "return_contract",
        "audit",
    ]
    for section in required_sections:
        if not isinstance(envelope.get(section), Mapping):
            _add_error(errors, f"missing_{section}", section, f"Envelope requires object section: {section}.", MODULE_NAME)

    identity = envelope.get("identity") if isinstance(envelope.get("identity"), Mapping) else {}
    protocol = envelope.get("protocol") if isinstance(envelope.get("protocol"), Mapping) else {}
    mission = envelope.get("mission") if isinstance(envelope.get("mission"), Mapping) else {}
    authority = envelope.get("authority") if isinstance(envelope.get("authority"), Mapping) else {}
    resources = envelope.get("resources") if isinstance(envelope.get("resources"), Mapping) else {}
    passport = envelope.get("passport") if isinstance(envelope.get("passport"), Mapping) else {}
    governance = envelope.get("governance") if isinstance(envelope.get("governance"), Mapping) else {}
    payload = envelope.get("payload") if isinstance(envelope.get("payload"), Mapping) else {}
    return_contract = envelope.get("return_contract") if isinstance(envelope.get("return_contract"), Mapping) else {}

    for field in ("subject_id", "provider", "origin"):
        if not _text(identity.get(field)):
            _add_error(errors, f"missing_identity_{field}", f"identity.{field}", f"Identity requires {field}.", "SarahMemoryTrustRegistry")
    for field in ("source_protocol", "adapter_id"):
        if not _text(protocol.get(field)):
            _add_error(errors, f"missing_protocol_{field}", f"protocol.{field}", f"Protocol requires {field}.", MODULE_NAME)
    for field in ("mission_id", "task_id", "objective", "intent", "requested_by", "created_at"):
        if not _text(mission.get(field)):
            _add_error(errors, f"missing_mission_{field}", f"mission.{field}", f"Mission requires {field}.", "SarahMemoryTerminal")

    if bool(authority.get("execution_authority")):
        _add_error(errors, "execution_authority_self_grant_denied", "authority.execution_authority", "SMUGCC cannot self-grant execution authority.", "SarahMemoryOperatorCore")
    if not bool(authority.get("requires_user_approval", True)):
        _add_error(errors, "user_approval_required", "authority.requires_user_approval", "External contracts require user approval by default.", "SarahMemoryOperatorCore")

    allowed_sources = [str(x).strip() for x in _as_list(resources.get("allowed_sources"))]
    allowed_methods = [str(x).strip().upper() for x in _as_list(resources.get("allowed_methods") or ["GET"])]
    wildcard_values = {"*", "/*", "ALL", "ANY"}
    for idx, source in enumerate(allowed_sources):
        if source.upper() in wildcard_values or re.search(r"(^|/)\*(/|$)", source):
            _add_error(errors, "wildcard_resource_denied", f"resources.allowed_sources[{idx}]", "Wildcard allowed resources are denied.", "SarahMemorySecurityGovernor")
    for idx, method in enumerate(allowed_methods):
        if method in wildcard_values:
            _add_error(errors, "wildcard_method_denied", f"resources.allowed_methods[{idx}]", "Wildcard methods are denied.", "SarahMemorySecurityGovernor")
        if method not in {"GET", "HEAD", "OPTIONS"}:
            _add_error(errors, "mutating_method_denied_by_default", f"resources.allowed_methods[{idx}]", "SMUGCC defaults to read-only methods.", "SarahMemorySecurityGovernor")

    if bool(resources.get("shell_allowed")):
        _add_error(errors, "shell_authority_denied_by_default", "resources.shell_allowed", "Shell access is denied by default.", "SarahMemorySecurityGovernor")
    if bool(resources.get("filesystem_allowed")):
        _add_error(errors, "filesystem_write_authority_denied_by_default", "resources.filesystem_allowed", "Filesystem authority is denied by default.", "SarahMemorySecurityGovernor")
    if bool(resources.get("memory_allowed")) and not bool(governance.get("memory_governed")):
        _add_error(errors, "memory_authority_requires_governance", "resources.memory_allowed", "Memory access requires explicit existing governance.", "SarahMemoryTrustRegistry")
    if bool(resources.get("device_allowed")) and not bool(governance.get("hard_governance_path")):
        _add_error(errors, "device_authority_requires_hard_governance", "resources.device_allowed", "Device access requires a hard governance path.", "SarahMemorySecurityGovernor")

    external_origin = _text(identity.get("origin")).lower() not in {"", "local", "localhost", "sarahmemory"}
    if external_origin and passport.get("required") is False:
        _add_error(errors, "passport_required_for_external_contract", "passport.required", "External contracts cannot bypass TrustRegistry passports.", "SarahMemoryTrustRegistry")
    if passport.get("required") is True and _text(passport.get("issued_by")) != "SarahMemoryTrustRegistry":
        _add_error(errors, "passport_issuer_must_be_trust_registry", "passport.issued_by", "Passports must be issued by SarahMemoryTrustRegistry.", "SarahMemoryTrustRegistry")
    if _text(passport.get("passport_id")) and not bool(passport.get("return_nonce_required", True)):
        _add_error(errors, "passport_return_nonce_required", "passport.return_nonce_required", "Passported returns require a return nonce.", "SarahMemoryTrustRegistry")
    if _text(passport.get("passport_id")) and not bool(return_contract.get("signature_required", False)):
        _add_error(errors, "passported_return_signature_required", "return_contract.signature_required", "Passported returns require a signature.", "SarahMemoryTrustRegistry")

    governance_required = {
        "safety_required": "SarahMemorySafetyPolicies",
        "security_required": "SarahMemorySecurityGovernor",
        "assurance_required": "SarahMemoryAssuranceGate",
        "compare_required": "SarahMemoryCompare",
        "compass_required": "SarahMemoryCognitiveCompass",
        "operatorcore_required": "SarahMemoryOperatorCore",
        "ledger_required": "SarahMemoryLedger",
    }
    for key, owner in governance_required.items():
        if not bool(governance.get(key, True)):
            _add_error(errors, f"governance_{key}_required", f"governance.{key}", f"{owner} cannot be bypassed.", owner)

    if not bool(return_contract.get("evidence_required", True)):
        _add_error(errors, "return_evidence_required", "return_contract.evidence_required", "Return contracts require evidence.", "SarahMemoryCompare")
    if not bool(return_contract.get("hash_required", True)):
        _add_error(errors, "return_hash_required", "return_contract.hash_required", "Return contracts require hashes.", "SarahMemoryCompare")
    if not bool(return_contract.get("receipt_required", True)):
        _add_error(errors, "return_receipt_required", "return_contract.receipt_required", "Return contracts require ledger receipts.", "SarahMemoryLedger")

    if payload.get("vendor_payload_hash") and not re.fullmatch(r"[A-Fa-f0-9]{64}", _text(payload.get("vendor_payload_hash"))):
        _add_error(errors, "invalid_vendor_payload_hash", "payload.vendor_payload_hash", "Vendor payload hash must be SHA-256 hex.", MODULE_NAME)
    if payload.get("smugcc_payload_hash") and not re.fullmatch(r"[A-Fa-f0-9]{64}", _text(payload.get("smugcc_payload_hash"))):
        _add_error(errors, "invalid_smugcc_payload_hash", "payload.smugcc_payload_hash", "SMUGCC payload hash must be SHA-256 hex.", MODULE_NAME)

    return {
        "ok": not errors,
        "schema": SMUGCC_SCHEMA,
        "contract_version": SMUGCC_CONTRACT_VERSION,
        "errors": errors,
        "warnings": warnings,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "execution_authority": False,
        "owners_checked": sorted({err["owner"] for err in errors}),
    }


def smugcc_to_sml_packet_dict(envelope: Mapping[str, Any]) -> Dict[str, Any]:
    """Map a SMUGCC envelope to an SML-shaped packet dictionary without routing."""
    env = copy.deepcopy(dict(envelope or {}))
    validation = validate_smugcc_envelope(env)
    identity = env.get("identity") if isinstance(env.get("identity"), Mapping) else {}
    mission = env.get("mission") if isinstance(env.get("mission"), Mapping) else {}
    authority = env.get("authority") if isinstance(env.get("authority"), Mapping) else {}
    governance = env.get("governance") if isinstance(env.get("governance"), Mapping) else {}
    payload = env.get("payload") if isinstance(env.get("payload"), Mapping) else {}
    return {
        "identity": {
            "primary": "Agent",
            "subject_id": identity.get("subject_id", ""),
            "provider": identity.get("provider", ""),
            "origin": identity.get("origin", ""),
            "trust_level": identity.get("trust_level", "unknown"),
        },
        "mission": {
            "primary": "Governance",
            "mission_id": mission.get("mission_id", ""),
            "task_id": mission.get("task_id", ""),
            "objective": mission.get("objective", ""),
            "intent": mission.get("intent", ""),
        },
        "pipeline": list(SMUGCC_PIPELINE),
        "authority": {
            "requested": _as_list(authority.get("requested")),
            "granted": [],
            "required": ["TrustRegistry", "AgentFirewall", "SafetyPolicies", "SecurityGovernor", "AssuranceGate", "Compare", "CognitiveCompass", "OperatorCore", "Ledger"],
            "least_authority": True,
            "execution_authority": False,
        },
        "governance": {
            "decision": "PENDING",
            "risk_level": governance.get("risk_level", "unknown"),
            "reasons": ["SMUGCC mapping is non-executing and requires existing governance organs."],
        },
        "payload": {"input": payload.get("input"), "normalized": payload.get("normalized")},
        "extensions": {"smugcc": env, "smugcc_validation": validation},
        "metadata": {"creator_organ": MODULE_NAME, "execution_authority": False},
    }


def smugcc_to_action_contract_dict(envelope: Mapping[str, Any]) -> Dict[str, Any]:
    """Map SMUGCC to an OperatorCore ActionContract-shaped dict without execution."""
    env = copy.deepcopy(dict(envelope or {}))
    validation = validate_smugcc_envelope(env)
    identity = env.get("identity") if isinstance(env.get("identity"), Mapping) else {}
    mission = env.get("mission") if isinstance(env.get("mission"), Mapping) else {}
    protocol = env.get("protocol") if isinstance(env.get("protocol"), Mapping) else {}
    capabilities = env.get("capabilities") if isinstance(env.get("capabilities"), Mapping) else {}
    governance = env.get("governance") if isinstance(env.get("governance"), Mapping) else {}
    resources = env.get("resources") if isinstance(env.get("resources"), Mapping) else {}
    authority = env.get("authority") if isinstance(env.get("authority"), Mapping) else {}
    requested = _as_list(capabilities.get("requested") or authority.get("requested"))
    methods = [str(x).upper() for x in _as_list(capabilities.get("allowed_methods"))]
    risk = _text(governance.get("risk_level") or "medium").lower()
    mode = _text(governance.get("execution_mode") or "draft").lower()
    if mode not in {"draft", "simulate", "apply", "rollback"}:
        mode = "draft"
    return {
        "schema": "SarahMemory.SMUGCC.action_contract_view.v1",
        "contract_type": "SMUGCCActionContractView",
        "contract_id": _text(mission.get("task_id") or mission.get("mission_id") or "smugcc-draft"),
        "user_goal": _text(mission.get("objective")),
        "normalized_text": _text(mission.get("objective")),
        "primary_lane": "external_cognitive_contract" if _text(identity.get("origin")).lower() == "external" else "local_contract",
        "action_type": "smugcc_contract_stage",
        "target": _text(protocol.get("adapter_id") or identity.get("provider") or "smugcc"),
        "target_ref": _text(identity.get("subject_id") or protocol.get("adapter_id")),
        "capability_name": "smugcc.contract",
        "executor_name": "none_contract_only",
        "required_permissions": requested,
        "risk_level": risk,
        "execution_mode": mode,
        "requires_confirmation": mode == "apply" or risk in {"high", "critical", "tier_3_privileged_system", "tier_4_network_remote_or_destructive"},
        "preconditions": ["valid_smugcc_envelope", "trustregistry_identity_check", "agentfirewall_boundary_check"],
        "verification_checks": ["smugcc_validation_ok", "owner_trace_complete", "ledger_receipt_prepared"],
        "rollback_plan": ["discard_staged_contract", "retain_ledger_receipt"],
        "metadata": {
            "source": MODULE_NAME,
            "smugcc_envelope": env,
            "smugcc_validation": validation,
            "source_protocol": protocol.get("source_protocol"),
            "allowed_methods": methods,
            "allowed_sources": _as_list(resources.get("allowed_sources")),
            "execution_authority": False,
        },
        "execution_authority": False,
    }


def smugcc_lifecycle_trace(envelope: Mapping[str, Any]) -> Dict[str, Any]:
    validation = validate_smugcc_envelope(envelope)
    return {
        "ok": True,
        "schema": "SarahMemory.SMUGCC.lifecycle_trace.v1",
        "lifecycle": [{"stage": stage, "status": "READY" if validation.get("ok") or stage == "validate" else "PENDING"} for stage in SMUGCC_LIFECYCLE],
        "pipeline": list(SMUGCC_PIPELINE),
        "validation": validation,
        "execution_authority": False,
    }


def smugcc_owner_trace(envelope: Mapping[str, Any]) -> Dict[str, Any]:
    validation = validate_smugcc_envelope(envelope)
    grouped: Dict[str, list] = {}
    for err in list(validation.get("errors") or []):
        owner = _text(err.get("owner") or MODULE_NAME)
        grouped.setdefault(owner, []).append(err)
    return {
        "ok": bool(validation.get("ok")),
        "schema": "SarahMemory.SMUGCC.owner_trace.v1",
        "owners": smugcc_ownership_map(),
        "errors_by_owner": grouped,
        "owner_sequence": list(SMUGCC_PIPELINE),
        "execution_authority": False,
    }


def smugcc_receipt_payload(envelope: Mapping[str, Any], event_type: str) -> Dict[str, Any]:
    env = copy.deepcopy(dict(envelope or {}))
    identity = env.get("identity") if isinstance(env.get("identity"), Mapping) else {}
    mission = env.get("mission") if isinstance(env.get("mission"), Mapping) else {}
    protocol = env.get("protocol") if isinstance(env.get("protocol"), Mapping) else {}
    compact = {
        "schema": "SarahMemory.SMUGCC.receipt_payload.v1",
        "event_type": _text(event_type or "SMUGCC_EVENT")[:96],
        "subject_id": _text(identity.get("subject_id"))[:180],
        "provider": _text(identity.get("provider"))[:96],
        "origin": _text(identity.get("origin"))[:64],
        "source_protocol": _text(protocol.get("source_protocol"))[:96],
        "adapter_id": _text(protocol.get("adapter_id"))[:120],
        "mission_id": _text(mission.get("mission_id"))[:180],
        "task_id": _text(mission.get("task_id"))[:180],
        "objective_hash": _sha256_obj({"objective": mission.get("objective")}),
        "validation_ok": bool(validate_smugcc_envelope(env).get("ok")),
        "execution_authority": False,
    }
    compact["payload_hash"] = _sha256_obj(compact)
    return compact


def smugcc_schema_view() -> Dict[str, Any]:
    return copy.deepcopy(CANONICAL_SMUGCC_SCHEMA)


def smugcc_adapter_declarations() -> Dict[str, Dict[str, Any]]:
    return copy.deepcopy(ADAPTER_DECLARATIONS)


def smugcc_ownership_map() -> Dict[str, Any]:
    return copy.deepcopy(SMUGCC_OWNERSHIP_MAP)


def smugcc_compatibility_report(envelope: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    validation = validate_smugcc_envelope(envelope) if isinstance(envelope, Mapping) else None
    adapters = []
    for adapter in ADAPTER_DECLARATIONS.values():
        adapters.append({
            "adapter_id": adapter["adapter_id"],
            "source_protocol": adapter["source_protocol"],
            "valid": validate_adapter_declaration(adapter)["ok"],
            "execution_authority": False,
        })
    return {
        "ok": True if validation is None else bool(validation.get("ok")),
        "schema": "SarahMemory.SMUGCC.compatibility_report.v1",
        "smugcc_schema": SMUGCC_SCHEMA,
        "contract_version": SMUGCC_CONTRACT_VERSION,
        "contract_only": True,
        "execution_authority": False,
        "lifecycle": list(SMUGCC_LIFECYCLE),
        "pipeline": list(SMUGCC_PIPELINE),
        "owners": smugcc_ownership_map(),
        "known_adapters": adapters,
        "validation": validation or {"ok": None, "message": "No envelope supplied."},
    }


def smugcc_status() -> Dict[str, Any]:
    return {
        "ok": True,
        "schema": "SarahMemory.SMUGCC.status.v1",
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "contract_schema": SMUGCC_SCHEMA,
        "contract_version": SMUGCC_CONTRACT_VERSION,
        "doctrine": copy.deepcopy(NO_EXECUTION_AUTHORITY),
        "lifecycle": list(SMUGCC_LIFECYCLE),
        "adapter_count": len(ADAPTER_DECLARATIONS),
        "known_adapters": sorted(ADAPTER_DECLARATIONS.keys()),
        "owners": smugcc_ownership_map(),
    }


def get_smugcc_status() -> Dict[str, Any]:
    return smugcc_status()


def get_smugcc_schema() -> Dict[str, Any]:
    return {
        "ok": True,
        "schema": "SarahMemory.SMUGCC.schema_view.v1",
        "contract_schema": SMUGCC_SCHEMA,
        "contract_version": SMUGCC_CONTRACT_VERSION,
        "envelope": smugcc_schema_view(),
        "adapters": smugcc_adapter_declarations(),
        "ownership": smugcc_ownership_map(),
        "execution_authority": False,
    }


def get_smugcc_compatibility_report(envelope: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    return smugcc_compatibility_report(envelope)


def sml_get_metadata() -> Dict[str, Any]:
    return copy.deepcopy(SML_ORGAN_METADATA)


def sml_health() -> Dict[str, Any]:
    return {
        "ok": True,
        "status": "OK",
        "component": MODULE_NAME,
        "schema": SMUGCC_SCHEMA,
        "contract_only": True,
        "execution_authority": False,
    }


def sml_diagnostics() -> Dict[str, Any]:
    return {
        "ok": True,
        "status": "OK",
        "component": MODULE_NAME,
        "metadata": sml_get_metadata(),
        "health": sml_health(),
        "adapter_validation": {key: validate_adapter_declaration(value) for key, value in ADAPTER_DECLARATIONS.items()},
    }


def sml_receive_packet(packet: Any, *, action: str = "observe", note: str = "", updates: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    return {
        "ok": True,
        "status": "OBSERVED",
        "component": MODULE_NAME,
        "action": str(action or "observe"),
        "note": str(note or "")[:512],
        "updates_accepted": bool(updates),
        "execution_authority": False,
        "message": "SMUGCC observes contract context only; it does not route or execute SML packets.",
    }


__all__ = [
    "ADAPTER_DECLARATIONS",
    "CANONICAL_SMUGCC_SCHEMA",
    "MODULE_NAME",
    "MODULE_VERSION",
    "NO_EXECUTION_AUTHORITY",
    "SML_ORGAN_METADATA",
    "SMUGCC_CONTRACT_VERSION",
    "SMUGCC_LIFECYCLE",
    "SMUGCC_OWNERSHIP_MAP",
    "SMUGCC_PIPELINE",
    "SMUGCC_SCHEMA",
    "build_smugcc_envelope",
    "get_smugcc_compatibility_report",
    "get_smugcc_schema",
    "get_smugcc_status",
    "sml_diagnostics",
    "sml_get_metadata",
    "sml_health",
    "sml_receive_packet",
    "smugcc_adapter_declarations",
    "smugcc_compatibility_report",
    "smugcc_ownership_map",
    "smugcc_schema_view",
    "smugcc_status",
    "smugcc_lifecycle_trace",
    "smugcc_owner_trace",
    "smugcc_receipt_payload",
    "smugcc_to_action_contract_dict",
    "smugcc_to_sml_packet_dict",
    "validate_adapter_declaration",
    "validate_smugcc_envelope",
]
