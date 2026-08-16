"""--==The SarahMemory Project==--
File: appsdk.py
Part of the SarahMemory AiOS Governed API Bridge
Version: v9.0.0-alpha
Date: 2026-08-07
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.

NAILDE SDK/API bridge for SarahMemory AiOS.

Purpose:
- Keep NAILDE /api/nailde/* routes out of api/server/app.py.
- Register a small Flask Blueprint from app.py using init_app(app).
- Lazily call SarahMemoryNAILDE.py.
- Preserve local-first, sandbox-first, non-authoritative governance boundaries.

This file does not execute shell commands, modify live CORE/API/UI files, write
hardware/devices, alter production tensor weights, write DLScreen global weights,
or approve its own output.
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "api_sdk_bridge"
# CATEGORY = "nailde_api_bridge"
# USER_FACING = False
# UI_EXPOSURE = "api_only"
# DEPLOYMENT_TARGET = "api/server"
# API_DOMAIN = "nailde"
# HARDWARE_DOMAIN = "none"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "appsdk_nailde_bridge"
# FAMILY = "api_bridge"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-08-07"
# PROJECT_SECTION = "SarahMemory AiOS Governed API Bridge"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Blueprint-owned NAILDE SDK routes. app.py performs registration only. NAILDE remains sandbox-first with no execution authority."
# --- SARAHMETA END ---

import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Blueprint, jsonify, request

SDK_SCHEMA = "SarahMemory.api.appsdk.nailde_bridge.v1"
_BLUEPRINT_NAME = "sarahmemory_appsdk"
_RUNTIME_LOCK = threading.RLock()
_RUNTIME = None
_LOGGER = None

appsdk_bp = Blueprint(_BLUEPRINT_NAME, __name__)


def _log_warning(message: str, *args: Any) -> None:
    try:
        logger = _LOGGER
        if logger is not None and hasattr(logger, "warning"):
            logger.warning(message, *args)
    except Exception:
        pass


def _ensure_core_path() -> None:
    """Best-effort path support for both api/server/appsdk.py and flat api_bridge/appsdk.py layouts."""
    try:
        here = Path(__file__).resolve()
        candidate_roots = [here.parent, here.parent.parent, here.parent.parent.parent, Path.cwd()]
        for root in candidate_roots:
            for candidate in (root / "core", root, root.parent / "core"):
                try:
                    if candidate.is_dir() and str(candidate) not in sys.path:
                        if (candidate / "SarahMemoryNAILDE.py").is_file() or (candidate / "SarahMemoryGlobals.py").is_file():
                            sys.path.insert(0, str(candidate))
                except Exception:
                    continue
    except Exception:
        pass


def _request_allowed() -> bool:
    """NAILDE is local-first because it can expose code, desktop, devices, and development state."""
    try:
        if str(os.getenv("SARAH_NAILDE_REMOTE_ALLOWED", "0")).strip().lower() in ("1", "true", "yes", "on"):
            return True
        remote = str(request.remote_addr or "").strip().lower()
        if remote in ("127.0.0.1", "::1", "localhost", ""):
            return True
        if remote.endswith("127.0.0.1"):
            return True
    except Exception:
        pass
    return False


def _blocked_response():
    return jsonify({
        "ok": False,
        "schema": SDK_SCHEMA,
        "error": "nailde_remote_blocked",
        "message": "NAILDE is local-only by default because it can expose code, desktop, device, model, and development state. Set SARAH_NAILDE_REMOTE_ALLOWED=1 only after explicit operator approval.",
        "source": "api.appsdk.nailde.guard",
        "execution_authority": False,
    }), 403


def _runtime():
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is not None:
            return _RUNTIME
        _ensure_core_path()
        try:
            import SarahMemoryNAILDE as _SMNAILDE  # type: ignore
            _RUNTIME = _SMNAILDE.get_nailde_runtime()
            return _RUNTIME
        except Exception as exc:
            _log_warning("SarahMemoryNAILDE runtime unavailable: %s", exc)
            return None


def _payload() -> Dict[str, Any]:
    payload = request.get_json(silent=True) if request.method in ("POST", "PUT", "PATCH") else {}
    if not isinstance(payload, dict):
        payload = {}
    try:
        for key in ("workspace_id", "goal", "problem", "selected_object", "category", "model_id", "action", "command", "query", "path", "mode"):
            if request.args.get(key) is not None and key not in payload:
                payload[key] = request.args.get(key)
    except Exception:
        pass
    return payload


def _runtime_unavailable(source: str):
    return jsonify({
        "ok": False,
        "schema": SDK_SCHEMA,
        "error": "nailde_runtime_unavailable",
        "source": source,
        "execution_authority": False,
    }), 503


def _exception_response(source: str, exc: Exception):
    return jsonify({
        "ok": False,
        "schema": SDK_SCHEMA,
        "error": str(exc),
        "source": source,
        "execution_authority": False,
    }), 500


@appsdk_bp.route("/api/nailde/status", methods=["GET"])
def api_nailde_status():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.status")
    try:
        return jsonify(rt.status()), 200
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.status", exc)


@appsdk_bp.route("/api/nailde/sdk", methods=["GET"])
def api_nailde_sdk():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.sdk")
    try:
        return jsonify(rt.sdk_catalog()), 200
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.sdk", exc)



@appsdk_bp.route("/api/nailde/environment", methods=["GET"])
def api_nailde_environment():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.environment")
    try:
        return jsonify(rt.environment_blueprint()), 200
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.environment", exc)


@appsdk_bp.route("/api/nailde/files", methods=["GET", "POST"])
def api_nailde_files():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.files")
    try:
        payload = _payload()
        if request.method == "GET":
            mode = str(request.args.get("mode") or "list").strip().lower()
            if mode == "read":
                payload["workspace_id"] = request.args.get("workspace_id") or payload.get("workspace_id")
                payload["path"] = request.args.get("path") or payload.get("path")
                return jsonify(rt.read_workspace_file(payload)), 200
            payload["workspace_id"] = request.args.get("workspace_id") or payload.get("workspace_id")
            return jsonify(rt.workspace_files(payload)), 200
        action = str(payload.get("action") or "save").strip().lower()
        if action == "read":
            return jsonify(rt.read_workspace_file(payload)), 200
        if action == "list":
            return jsonify(rt.workspace_files(payload)), 200
        result = rt.save_workspace_file(payload)
        return jsonify(result), 200 if result.get("ok") else 400
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.files", exc)


@appsdk_bp.route("/api/nailde/code/draft", methods=["POST"])
def api_nailde_code_draft():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.code_draft")
    try:
        result = rt.natural_language_code_draft(_payload())
        return jsonify(result), 200 if result.get("ok") else 400
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.code_draft", exc)


@appsdk_bp.route("/api/nailde/agent/mission", methods=["POST"])
def api_nailde_agent_mission():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.agent_mission")
    try:
        return jsonify(rt.agent_mission(_payload())), 200
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.agent_mission", exc)


@appsdk_bp.route("/api/nailde/validate/text", methods=["POST"])
def api_nailde_validate_text():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.validate_text")
    try:
        return jsonify(rt.validate_text_artifacts(_payload())), 200
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.validate_text", exc)


@appsdk_bp.route("/api/nailde/reconcile", methods=["POST"])
def api_nailde_reconcile():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.reconcile")
    try:
        return jsonify(rt.reconcile_edits(_payload())), 200
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.reconcile", exc)


@appsdk_bp.route("/api/nailde/layout", methods=["GET", "POST"])
def api_nailde_layout():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.layout")
    try:
        payload = _payload()
        if request.method == "GET":
            payload["workspace_id"] = request.args.get("workspace_id") or payload.get("workspace_id") or "__global__"
            payload["action"] = request.args.get("action") or "load"
        return jsonify(rt.workbench_layout(payload)), 200
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.layout", exc)


@appsdk_bp.route("/api/nailde/toolbox", methods=["GET"])
def api_nailde_toolbox():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.toolbox")
    try:
        return jsonify(rt.toolbox_catalog()), 200
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.toolbox", exc)


@appsdk_bp.route("/api/nailde/search", methods=["POST"])
def api_nailde_search():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.search")
    try:
        return jsonify(rt.search_workspace(_payload())), 200
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.search", exc)


@appsdk_bp.route("/api/nailde/command", methods=["POST"])
def api_nailde_command():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.command")
    try:
        result = rt.command_dispatch(_payload())
        return jsonify(result), 200 if result.get("ok") else 409
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.command", exc)


@appsdk_bp.route("/api/nailde/scaffold", methods=["POST"])
def api_nailde_scaffold():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.scaffold")
    try:
        result = rt.scaffold_extreme_project(_payload())
        return jsonify(result), 200 if result.get("ok") else 400
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.scaffold", exc)


@appsdk_bp.route("/api/nailde/workspaces", methods=["POST"])
def api_nailde_create_workspace():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.workspaces")
    try:
        result = rt.create_workspace(_payload())
        return jsonify(result), 201 if result.get("ok") else 400
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.workspaces", exc)


@appsdk_bp.route("/api/nailde/awareness", methods=["GET", "POST"])
def api_nailde_awareness():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.awareness")
    payload = _payload()
    if request.method == "GET":
        payload["include_desktop"] = str(request.args.get("include_desktop") or request.args.get("desktop") or "0").strip().lower() in ("1", "true", "yes", "on")
    try:
        return jsonify(rt.workbench_awareness(payload)), 200
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.awareness", exc)


@appsdk_bp.route("/api/nailde/thought-loop", methods=["POST"])
def api_nailde_thought_loop():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.thought_loop")
    try:
        return jsonify(rt.thought_loop(_payload())), 200
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.thought_loop", exc)


@appsdk_bp.route("/api/nailde/weightlab/simulate", methods=["POST"])
def api_nailde_weightlab_simulate():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.weightlab")
    try:
        return jsonify(rt.weightlab_simulate(_payload())), 200
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.weightlab", exc)



@appsdk_bp.route("/api/nailde/filesystem/status", methods=["GET"])
def api_nailde_filesystem_status():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.filesystem_status")
    try:
        return jsonify(rt.filesystem_status()), 200
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.filesystem_status", exc)


@appsdk_bp.route("/api/nailde/filesystem/map", methods=["GET", "POST"])
def api_nailde_filesystem_map():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.filesystem_map")
    try:
        payload = _payload()
        if request.method == "GET":
            payload["workspace_id"] = request.args.get("workspace_id") or payload.get("workspace_id") or ""
            payload["include_checksums"] = str(request.args.get("include_checksums") or "0").strip().lower() in ("1", "true", "yes", "on")
        return jsonify(rt.filesystem_map(payload)), 200
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.filesystem_map", exc)


@appsdk_bp.route("/api/nailde/editor/validate", methods=["POST"])
def api_nailde_editor_validate():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.editor_validate")
    try:
        return jsonify(rt.editor_validate(_payload())), 200
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.editor_validate", exc)


@appsdk_bp.route("/api/nailde/editor/create-application", methods=["POST"])
def api_nailde_editor_create_application():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.editor_create_application")
    try:
        result = rt.create_application_from_editor(_payload())
        return jsonify(result), 200 if result.get("ok") else 400
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.editor_create_application", exc)


@appsdk_bp.route("/api/nailde/settings", methods=["GET", "POST"])
def api_nailde_settings():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.settings")
    try:
        payload = _payload()
        if request.method == "GET":
            payload["action"] = request.args.get("action") or "load"
        else:
            payload["action"] = payload.get("action") or "save"
        return jsonify(rt.settings_state(payload)), 200
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.settings", exc)


@appsdk_bp.route("/api/nailde/github/plan", methods=["POST"])
def api_nailde_github_plan():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.github_plan")
    try:
        result = rt.github_plan(_payload())
        return jsonify(result), 200 if result.get("ok") else 400
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.github_plan", exc)


@appsdk_bp.route("/api/nailde/addons/install-plan", methods=["POST"])
def api_nailde_addon_install_plan():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.addon_install_plan")
    try:
        result = rt.addon_install_plan(_payload())
        return jsonify(result), 200 if result.get("ok") else 400
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.addon_install_plan", exc)


@appsdk_bp.route("/api/nailde/addons/install-authorized", methods=["POST"])
def api_nailde_addon_install_authorized():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.addon_install_authorized")
    try:
        result = rt.addon_install_authorized(_payload())
        return jsonify(result), 200 if result.get("ok") else 409
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.addon_install_authorized", exc)

@appsdk_bp.route("/api/nailde/auto-build", methods=["POST"])
def api_nailde_auto_build():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.auto_build")
    try:
        result = rt.auto_build_from_prompt(_payload())
        status = 200
        if result.get("requires_workspace_decision"):
            status = 409
        elif not result.get("ok"):
            status = 400
        return jsonify(result), status
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.auto_build", exc)


@appsdk_bp.route("/api/nailde/workspace/autosave", methods=["POST"])
def api_nailde_workspace_autosave():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.workspace_autosave")
    try:
        result = rt.workspace_autosave(_payload())
        return jsonify(result), 200 if result.get("ok") else 400
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.workspace_autosave", exc)


@appsdk_bp.route("/api/nailde/workspace/recovery", methods=["GET", "POST"])
def api_nailde_workspace_recovery():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.workspace_recovery")
    try:
        payload = _payload()
        if request.method == "GET":
            payload["action"] = request.args.get("action") or "latest"
        result = rt.workspace_recovery(payload)
        return jsonify(result), 200 if result.get("ok") else 400
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.workspace_recovery", exc)


@appsdk_bp.route("/api/nailde/workspace/decision", methods=["POST"])
def api_nailde_workspace_decision():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.workspace_decision")
    try:
        result = rt.workspace_decision(_payload())
        return jsonify(result), 200 if result.get("ok") else 400
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.workspace_decision", exc)


@appsdk_bp.route("/api/nailde/avatar/message", methods=["POST"])
def api_nailde_avatar_message():
    if not _request_allowed():
        return _blocked_response()
    rt = _runtime()
    if rt is None:
        return _runtime_unavailable("api.appsdk.nailde.avatar")
    try:
        return jsonify(rt.avatar_message(_payload())), 200
    except Exception as exc:
        return _exception_response("api.appsdk.nailde.avatar", exc)


def init_app(flask_app: Any, logger: Optional[Any] = None) -> Dict[str, Any]:
    """Register the NAILDE SDK Blueprint on an existing Flask app.

    app.py should call this once after creating the Flask app object. The route
    implementation stays in appsdk.py; app.py remains the process ingress.
    """
    global _LOGGER
    _LOGGER = logger
    try:
        if getattr(flask_app, "blueprints", None) is not None and _BLUEPRINT_NAME in flask_app.blueprints:
            return {"ok": True, "registered": False, "already_registered": True, "blueprint": _BLUEPRINT_NAME, "schema": SDK_SCHEMA}
        flask_app.register_blueprint(appsdk_bp)
        return {"ok": True, "registered": True, "blueprint": _BLUEPRINT_NAME, "schema": SDK_SCHEMA}
    except Exception as exc:
        _log_warning("NAILDE appsdk blueprint registration failed: %s", exc)
        return {"ok": False, "registered": False, "error": str(exc), "blueprint": _BLUEPRINT_NAME, "schema": SDK_SCHEMA}


__all__ = ["appsdk_bp", "init_app"]

# ====================================================================
# END OF appsdk.py v9.0.0-alpha
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing API bridge adapter.
SML_ORGAN_METADATA = {
    "name": 'appsdk',
    "version": "v9.0.0-alpha-sml-0.2",
    "category": "Input",
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['api_bridge', 'transport', 'sml_bridge_candidate'],
    "supported_missions": ['Conversation', 'Execution', 'Knowledge', 'Diagnostics'],
    "supported_omega": ['Ω001', 'Ω002', 'Ω004', 'Ω020'],
    "required_authority": ['Read'],
    "priority": 58,
    "trust_level": "api_bridge_integrated",
    "internal_only": False,
    "metadata": {"sml_adapter": "api_bridge_non_executing", "source_file": 'appsdk.py'},
}

def sml_get_metadata():
    return dict(SML_ORGAN_METADATA)

def sml_health():
    return {"status": "Healthy", "availability": 1.0, "integrity": 1.0, "performance": 1.0, "reliability": 1.0, "confidence": 0.75, "latency_ms": 0.0, "stability": 1.0, "compatibility": 1.0, "notes": ["SML API adapter present"]}

def sml_diagnostics():
    return {"status": "OK", "component": 'appsdk', "sml_adapter": True, "metadata": dict(SML_ORGAN_METADATA), "health": sml_health()}
# --- SML ORGAN ADAPTER END ---

