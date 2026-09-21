"""SarahMemory SMUGCC API bridge.

Thin Flask bridge for the contract-only SMUGCC organ. This module delegates to
CORE owners and never executes providers, shell commands, filesystem mutations,
device control, or passport issuance.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from flask import Blueprint, jsonify, request
except Exception:  # pragma: no cover
    Blueprint = None  # type: ignore
    jsonify = None  # type: ignore
    request = None  # type: ignore


SCHEMA = "SarahMemory.api.smugcc_bridge.v1"
BLUEPRINT_NAME = "sarahmemory_smugcc"
_LOGGER = None

appsmugcc_bp = Blueprint(BLUEPRINT_NAME, __name__) if Blueprint is not None else None


def _ensure_core_path() -> None:
    try:
        here = Path(__file__).resolve()
        for root in (here.parent, here.parent.parent, here.parent.parent.parent, Path("C:/SarahMemory")):
            core = root / "core"
            if core.is_dir() and str(core) not in sys.path:
                sys.path.insert(0, str(core))
    except Exception:
        pass


def _core(name: str):
    _ensure_core_path()
    return importlib.import_module(name)


def _json_payload() -> Dict[str, Any]:
    try:
        data = request.get_json(silent=True) if request is not None else {}
    except Exception:
        data = {}
    return data if isinstance(data, dict) else {}


def _envelope_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    env = payload.get("envelope") if isinstance(payload.get("envelope"), dict) else payload
    return env if isinstance(env, dict) else {}


def _safe_call(module_name: str, fn_name: str, *args: Any, default: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    try:
        mod = _core(module_name)
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            out = fn(*args, **kwargs)
            return out if isinstance(out, dict) else {"ok": False, "error": "non_dict_result", "source": f"{module_name}.{fn_name}", "execution_authority": False}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "source": f"{module_name}.{fn_name}", "execution_authority": False}
    return default or {"ok": False, "error": "callable_unavailable", "source": f"{module_name}.{fn_name}", "execution_authority": False}


def _record_receipt(event_type: str, envelope: Dict[str, Any], verdict: str, summary: str) -> Dict[str, Any]:
    try:
        smugcc = _core("SarahMemorySMUGCC")
        payload = smugcc.smugcc_receipt_payload(envelope, event_type)
        ledger = _core("SarahMemoryLedger")
        fn = getattr(ledger, "record_governance_receipt", None)
        if not callable(fn):
            return {"ok": False, "error": "ledger_receipt_unavailable", "execution_authority": False}
        identity = envelope.get("identity") if isinstance(envelope.get("identity"), dict) else {}
        mission = envelope.get("mission") if isinstance(envelope.get("mission"), dict) else {}
        return fn(
            "smugcc",
            event_type,
            subject_id=str(identity.get("subject_id") or ""),
            task_id=str(mission.get("task_id") or mission.get("mission_id") or ""),
            lane="smugcc",
            verdict=verdict,
            risk=str(((envelope.get("governance") or {}) if isinstance(envelope.get("governance"), dict) else {}).get("risk_level") or "medium"),
            retention_class="smugcc_contract",
            payload_hash=str(payload.get("payload_hash") or ""),
            summary=summary,
            metadata=payload,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "execution_authority": False}


@appsmugcc_bp.get("/api/smugcc/status")
def api_smugcc_status():
    return jsonify(_safe_call("SarahMemorySMUGCC", "get_smugcc_status")), 200


@appsmugcc_bp.get("/api/smugcc/schema")
def api_smugcc_schema():
    return jsonify(_safe_call("SarahMemorySMUGCC", "get_smugcc_schema")), 200


@appsmugcc_bp.get("/api/smugcc/compatibility")
def api_smugcc_compatibility():
    return jsonify(_safe_call("SarahMemorySMUGCC", "get_smugcc_compatibility_report")), 200


@appsmugcc_bp.route("/api/smugcc/validate", methods=["GET", "POST"])
def api_smugcc_validate():
    if request.method == "GET":
        smugcc = _core("SarahMemorySMUGCC")
        sample = smugcc.build_smugcc_envelope(
            identity={"subject_id": "example:external", "provider": "example", "origin": "external"},
            protocol={"source_protocol": "example", "adapter_id": "generic_rest_tool"},
            mission={"mission_id": "example-mission", "task_id": "example-task", "objective": "Validate contract envelope only.", "intent": "contract_validation", "requested_by": "api_smugcc_validate"},
        )
        return jsonify({"ok": True, "schema": SCHEMA, "sample": sample, "sample_validation": smugcc.validate_smugcc_envelope(sample), "execution_authority": False}), 200
    envelope = _envelope_from_payload(_json_payload())
    result = _safe_call("SarahMemorySMUGCC", "validate_smugcc_envelope", envelope)
    result["owner_trace"] = _safe_call("SarahMemorySMUGCC", "smugcc_owner_trace", envelope)
    if bool(result.get("ok")):
        result["receipt"] = _record_receipt("SMUGCC_VALIDATED", envelope, "ALLOW", "SMUGCC envelope validated without execution.")
    return jsonify(result), 200


@appsmugcc_bp.post("/api/smugcc/adapter/validate")
def api_smugcc_adapter_validate():
    payload = _json_payload()
    declaration = payload.get("declaration") if isinstance(payload.get("declaration"), dict) else payload
    result = _safe_call("SarahMemorySMUGCC", "validate_adapter_declaration", declaration if isinstance(declaration, dict) else {})
    return jsonify({"ok": bool(result.get("ok")), "schema": SCHEMA, "validation": result, "execution_authority": False}), 200


@appsmugcc_bp.post("/api/smugcc/build")
def api_smugcc_build():
    payload = _json_payload()
    smugcc = _core("SarahMemorySMUGCC")
    envelope = smugcc.build_smugcc_envelope(
        identity=payload.get("identity") if isinstance(payload.get("identity"), dict) else {
            "subject_id": payload.get("subject_id") or payload.get("adapter_id") or "smugcc:builder",
            "provider": payload.get("provider") or "unknown",
            "origin": payload.get("origin") or "external",
        },
        protocol=payload.get("protocol") if isinstance(payload.get("protocol"), dict) else {
            "source_protocol": payload.get("source_protocol") or payload.get("protocol_name") or "unknown",
            "adapter_id": payload.get("adapter_id") or "generic_rest_tool",
        },
        mission=payload.get("mission") if isinstance(payload.get("mission"), dict) else {
            "objective": payload.get("objective") or payload.get("mission") or "",
            "intent": payload.get("intent") or "contract_build",
            "requested_by": payload.get("requested_by") or "smugcc_panel",
            "expected_output_type": payload.get("expected_output_type") or "contract",
        },
        capabilities=payload.get("capabilities") if isinstance(payload.get("capabilities"), dict) else {
            "requested": payload.get("requested_capabilities") or [],
            "allowed_methods": payload.get("allowed_methods") or ["GET"],
        },
        resources=payload.get("resources") if isinstance(payload.get("resources"), dict) else {
            "allowed_sources": payload.get("allowed_sources") or [],
        },
        governance=payload.get("governance") if isinstance(payload.get("governance"), dict) else {
            "risk_level": payload.get("risk_level") or "medium",
        },
    )
    return jsonify({"ok": True, "schema": SCHEMA, "envelope": envelope, "validation": smugcc.validate_smugcc_envelope(envelope), "execution_authority": False}), 200


@appsmugcc_bp.post("/api/smugcc/trace")
def api_smugcc_trace():
    payload = _json_payload()
    envelope = _envelope_from_payload(payload)
    trace = {
        "ok": True,
        "schema": "SarahMemory.SMUGCC.trace.api.v1",
        "task_id": str(payload.get("task_id") or ((envelope.get("mission") or {}) if isinstance(envelope.get("mission"), dict) else {}).get("task_id") or ""),
        "SMUGCC": _safe_call("SarahMemorySMUGCC", "validate_smugcc_envelope", envelope),
        "SMLProtocol": _safe_call("SarahMemorySMLProtocol", "sml_smugcc_to_packet", envelope),
        "AgentFirewall": _safe_call("SarahMemoryAgentFirewall", "inspect_smugcc_envelope", envelope, source="api.smugcc.trace", remote_addr=getattr(request, "remote_addr", "")),
        "TrustRegistry": _safe_call("SarahMemoryTrustRegistry", "resolve_smugcc_subject", envelope),
        "SafetyPolicies": _safe_call("SarahMemorySafetyPolicies", "evaluate_smugcc_policy", envelope),
        "SecurityGovernor": _safe_call("SarahMemorySecurityGovernor", "review_smugcc_envelope", envelope),
        "AssuranceGate": _safe_call("SarahMemoryAssuranceGate", "review_smugcc_assurance", envelope),
        "OperatorCore": _safe_call("SarahMemoryOperatorCore", "review_smugcc_operator_gate", envelope),
        "Ledger": _safe_call("SarahMemorySMUGCC", "smugcc_receipt_payload", envelope, "SMUGCC_TRACE"),
        "execution_authority": False,
    }
    return jsonify(trace), 200


@appsmugcc_bp.post("/api/smugcc/mission/stage")
def api_smugcc_mission_stage():
    payload = _json_payload()
    envelope = _envelope_from_payload(payload) if isinstance(payload.get("envelope"), dict) else {}
    if not envelope:
        build_payload = dict(payload)
        build_payload.setdefault("requested_by", "smugcc_mission_stage")
        smugcc = _core("SarahMemorySMUGCC")
        envelope = smugcc.build_smugcc_envelope(
            identity=build_payload.get("identity") if isinstance(build_payload.get("identity"), dict) else {"subject_id": build_payload.get("subject_id") or build_payload.get("adapter_id") or "smugcc:mission", "provider": build_payload.get("provider") or "unknown", "origin": build_payload.get("origin") or "external"},
            protocol=build_payload.get("protocol") if isinstance(build_payload.get("protocol"), dict) else {"source_protocol": build_payload.get("source_protocol") or "unknown", "adapter_id": build_payload.get("adapter_id") or "generic_rest_tool"},
            mission=build_payload.get("mission") if isinstance(build_payload.get("mission"), dict) else {"objective": build_payload.get("objective") or build_payload.get("mission") or "", "intent": build_payload.get("intent") or "mission_stage", "requested_by": build_payload.get("requested_by")},
            governance=build_payload.get("governance") if isinstance(build_payload.get("governance"), dict) else {"risk_level": build_payload.get("risk_level") or "medium"},
        )
    operator = _safe_call("SarahMemoryOperatorCore", "stage_smugcc_action", envelope, mode=str(payload.get("mode") or "draft"))
    receipt = _record_receipt("SMUGCC_MISSION_STAGED", envelope, str(operator.get("decision") or "STAGED"), "SMUGCC mission staged for governed review; no execution performed.")
    return jsonify({"ok": bool(operator.get("ok", True)), "schema": SCHEMA, "envelope": envelope, "operator": operator, "receipt": receipt, "requires_approval": bool(operator.get("requires_approval", True)), "execution_authority": False}), 200


@appsmugcc_bp.get("/api/smugcc/receipts")
def api_smugcc_receipts():
    try:
        ledger = _core("SarahMemoryLedger")
        rows = ledger.get_governance_receipts(domain="smugcc", limit=int(request.args.get("limit") or 50))
        return jsonify({"ok": True, "schema": SCHEMA, "receipts": rows, "execution_authority": False}), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "schema": SCHEMA, "receipts": [], "execution_authority": False}), 200


@appsmugcc_bp.get("/api/smugcc/passports")
def api_smugcc_passports():
    try:
        trust = _core("SarahMemoryTrustRegistry")
        passports = trust.list_agent_passports(limit=int(request.args.get("limit") or 50))
    except Exception as exc:
        return jsonify({"ok": False, "schema": SCHEMA, "error": str(exc), "passports": [], "execution_authority": False}), 200
    return jsonify({"ok": True, "schema": SCHEMA, "passports": passports if isinstance(passports, list) else [], "execution_authority": False}), 200


@appsmugcc_bp.get("/api/smugcc/quarantine")
def api_smugcc_quarantine():
    firewall = _safe_call("SarahMemoryAgentFirewall", "collect_agent_visibility_snapshot", include_os_surface=False, max_process_rows=0, default={"ok": False, "error": "agent_visibility_snapshot_unavailable"})
    return jsonify({"ok": True, "schema": SCHEMA, "quarantine": firewall, "execution_authority": False}), 200


def init_app(flask_app: Any, logger: Optional[Any] = None) -> Dict[str, Any]:
    global _LOGGER
    _LOGGER = logger
    if appsmugcc_bp is None:
        return {"ok": False, "registered": False, "error": "flask_unavailable", "schema": SCHEMA}
    try:
        if getattr(flask_app, "blueprints", None) is not None and BLUEPRINT_NAME in flask_app.blueprints:
            return {"ok": True, "registered": False, "already_registered": True, "blueprint": BLUEPRINT_NAME, "schema": SCHEMA}
        flask_app.register_blueprint(appsmugcc_bp)
        return {"ok": True, "registered": True, "blueprint": BLUEPRINT_NAME, "schema": SCHEMA}
    except Exception as exc:
        try:
            if logger is not None:
                logger.warning("SMUGCC API bridge registration failed: %s", exc)
        except Exception:
            pass
        return {"ok": False, "registered": False, "error": str(exc), "schema": SCHEMA}


__all__ = ["appsmugcc_bp", "init_app"]
