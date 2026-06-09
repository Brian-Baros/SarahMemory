"""--==The SarahMemory Project==--
File: api/server/appdevbridge.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-06-06
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

Purpose: SarahMemory Developer Bridge endpoints for ChatGPT-assisted engineering packets
==============================================================================================
Design goals:
- NO endpoint collisions with app.py or app*.py files.
- Everything is namespaced under /api/devbridge/*.
- Temporary supervised bridge for ChatGPT Project Session collaboration.
- Packet export/import/staging/validation/audit without bloating app.py.
- No autonomous code application. Real file writes require explicit approval + developer gate.
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "api_bridge"
# CATEGORY = "developer_bridge"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "api_server"
# API_DOMAIN = "devbridge"
# HARDWARE_DOMAIN = "filesystem_optional"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "developer_bridge_api"
# FAMILY = "developer_tools"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-06-06"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Developer bridge API under /api/devbridge/* for ChatGPT packet exchange, response import, staging, validation, and explicitly approved patch application."
# --- SARAHMETA END ---

"""
SarahMemory DevBridge v9.0.0

This module creates a governed bridge lane for the SarahMemory ResearchPanel.
It lets the FrontEnd generate structured engineering packets, import ChatGPT
responses, stage patch candidates, validate staged files, and apply only after
explicit approval and local developer gating.

Important governance rule:
    observe -> packetize -> import response -> stage -> validate -> approve -> apply

The bridge does not call OpenAI APIs. It is a user-mediated bridge for the
ChatGPT web/project session and SarahMemory runtime.
"""


import ast
import hashlib
import json
import os
import py_compile
import re
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from flask import Blueprint, jsonify, request

bp = Blueprint("appdevbridge_v800", __name__)

_CONNECT_SQLITE: Optional[Callable[..., Any]] = None
_META_DB: Optional[str] = None
_API_KEY_AUTH_OK: Optional[Callable[[], bool]] = None
_SIGN_OK: Optional[Callable[[bytes, str], bool]] = None

PROJECT_VERSION = os.environ.get("PROJECT_VERSION", "9.0.0")
DEFAULT_CHATGPT_SESSION_URL = (
    "https://chatgpt.com/g/g-p-67e0408d2a048191ae36592e6e45ae89-"
    "the-sarahmemory-project-v8-0-0-aios/c/69fde7eb-0204-83e8-8b9e-65636d3ece39"
)

MAX_TEXT_CHARS = int(os.environ.get("SARAH_DEVBRIDGE_MAX_TEXT_CHARS", "2000000") or 2000000)
MAX_FILES_PER_STAGE = int(os.environ.get("SARAH_DEVBRIDGE_MAX_FILES_PER_STAGE", "25") or 25)
MAX_FILE_CHARS = int(os.environ.get("SARAH_DEVBRIDGE_MAX_FILE_CHARS", "2000000") or 2000000)
MAX_REPAIR_TICKETS_RETURN = int(os.environ.get("SARAH_DEVBRIDGE_MAX_REPAIR_TICKETS_RETURN", "100") or 100)
MAX_CMD_TICKETS_PER_PROCESS = int(os.environ.get("SARAH_DEVBRIDGE_MAX_CMD_TICKETS_PER_PROCESS", "50") or 50)
CMD_TICKET_RETENTION_DAYS = int(os.environ.get("SARAH_DEVBRIDGE_CMD_TICKET_RETENTION_DAYS", "3") or 3)

# Governance domains for generation/promotion lanes.
# Core repairs should target existing verified files. Generated apps/panels/drivers
# must start under data/devbridge/sandbox/* before they are promoted.
SANDBOX_DOMAINS = {"apps", "addons", "websites", "panels", "scada", "iot", "robotics", "science", "reports"}
EXPLICIT_CREATE_ZONES = (
    "data/devbridge/sandbox/",
    "data/addons/",
    "drivers/",
    "data/exports/",
)
CORE_EXISTING_ONLY_PREFIXES = (
    "api/",
    "app/",
    "ui/",
)
CORE_EXISTING_ONLY_FILES = (
    "app.py",
    "appdevbridge.py",
    "SarahMemoryGlobals.py",
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _now() -> float:
    return time.time()


def _iso(ts: Optional[float] = None) -> str:
    return datetime.utcfromtimestamp(ts or _now()).isoformat() + "Z"


def _j() -> Dict[str, Any]:
    data = request.get_json(silent=True)
    if isinstance(data, dict):
        return data
    # Flask can reject BOM-prefixed JSON bodies depending on the client and
    # content-type. Fall back to the same tolerant parser used for ticket files.
    try:
        parsed, diag = _json_loads_object_tolerant(_body_bytes())
        if parsed and diag.get("valid_json"):
            return parsed
    except Exception:
        pass
    return {}


def _ok(**kw: Any):
    return jsonify({"ok": True, **kw}), 200


def _err(msg: str, code: int = 400, **kw: Any):
    return jsonify({"ok": False, "error": msg, **kw}), code


def _body_bytes() -> bytes:
    try:
        return request.get_data(cache=True) or b""
    except Exception:
        return b""


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def _sha256_file(path: str) -> str:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _boolish(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "approved", "allow"}


def _sanitize_id(value: str, prefix: str = "item") -> str:
    raw = str(value or "").strip()
    if not raw:
        return _new_id(prefix)
    raw = raw.replace("\\", "/").split("/")[-1]
    raw = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).replace("..", "_")
    return (raw[:128] or _new_id(prefix))


def _sanitize_filename(value: str, default: str = "file.txt") -> str:
    raw = str(value or "").strip().replace("\\", "/").split("/")[-1]
    raw = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).replace("..", "_")
    return (raw[:255] or default)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj or {}, f, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(tmp, path)


def _decode_json_text(raw: Any) -> str:
    """Decode request/file JSON text with UTF-8 BOM tolerance.

    DevBridge tickets are often generated by PowerShell or external editors.
    Some of those writers emit UTF-8 BOMs or wrap JSON in markdown fences.
    This helper keeps the bridge strict on object shape while accepting that
    transport noise.
    """
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8-sig")
        except Exception:
            text = raw.decode("utf-8", errors="replace")
    else:
        text = str(raw)
    return text.replace("\x00", "").lstrip("\ufeff").strip()


def _json_loads_object_tolerant(raw: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse a JSON object with diagnostics and BOM/markdown tolerance."""
    text = _decode_json_text(raw)
    diag: Dict[str, Any] = {
        "valid_json": False,
        "json_type": "",
        "error": "",
        "used_tolerant_parse": False,
        "char_count": len(text),
    }
    if not text:
        diag["error"] = "empty_json_text"
        return {}, diag

    candidates: List[Tuple[str, bool]] = [(text, False)]
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidates.append((fence_match.group(1).strip(), True))
    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        candidates.append((text[first:last + 1], first != 0 or last != len(text) - 1))

    decoder = json.JSONDecoder()
    last_error = ""
    for candidate, tolerant in candidates:
        candidate = _decode_json_text(candidate)
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
            diag["json_type"] = type(obj).__name__
            if isinstance(obj, dict):
                diag["valid_json"] = True
                diag["used_tolerant_parse"] = bool(tolerant)
                return obj, diag
            diag["error"] = f"json_root_must_be_object:{type(obj).__name__}"
        except Exception as e:
            last_error = str(e)

        # Last-resort raw_decode scan: handles BOM + leading shell transcript noise.
        for offset, ch in enumerate(candidate):
            if ch != "{":
                continue
            try:
                obj, end = decoder.raw_decode(candidate[offset:])
                diag["json_type"] = type(obj).__name__
                if isinstance(obj, dict):
                    diag["valid_json"] = True
                    diag["used_tolerant_parse"] = True
                    diag["raw_decode_offset"] = offset
                    diag["raw_decode_end"] = offset + end
                    return obj, diag
                diag["error"] = f"json_root_must_be_object:{type(obj).__name__}"
            except Exception as e:
                last_error = str(e)
                continue

    if not diag.get("error"):
        diag["error"] = last_error or "json_parse_failed"
    return {}, diag


def _read_json(path: str) -> Dict[str, Any]:
    data, _diag = _read_json_with_diagnostics(path)
    return data


def _read_json_with_diagnostics(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    diag: Dict[str, Any] = {
        "path": path,
        "exists": os.path.exists(path),
        "is_file": os.path.isfile(path),
        "size_bytes": 0,
        "sha256": "",
        "valid_json": False,
        "json_type": "",
        "error": "",
        "used_tolerant_parse": False,
    }
    try:
        if not os.path.isfile(path):
            diag["error"] = "json_file_missing"
            return {}, diag
        with open(path, "rb") as f:
            raw = f.read()
        diag["size_bytes"] = len(raw)
        diag["sha256"] = hashlib.sha256(raw).hexdigest()
        data, parse_diag = _json_loads_object_tolerant(raw)
        diag.update(parse_diag)
        if not data and not diag.get("valid_json"):
            text = _decode_json_text(raw)
            diag["text_excerpt"] = text[:500]
        return data, diag
    except Exception as e:
        diag["error"] = str(e)
        return {}, diag


def _write_text(path: str, text: str) -> None:
    _ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text or "")
    os.replace(tmp, path)


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def _require_injected() -> bool:
    return bool(_CONNECT_SQLITE and _META_DB)


def _auth_ok(body: Optional[bytes] = None) -> bool:
    body = body if body is not None else _body_bytes()
    sig = (request.headers.get("X-Sarah-Signature") or "").strip()
    if sig and _SIGN_OK:
        try:
            return bool(_SIGN_OK(body or b"", sig))
        except Exception:
            return False
    if _API_KEY_AUTH_OK:
        try:
            return bool(_API_KEY_AUTH_OK())
        except Exception:
            return False
    # Dev fallback if app.py did not inject auth helpers.
    return True


def _is_loopback_request() -> bool:
    try:
        host = (request.remote_addr or "").strip().lower()
        return host in {"127.0.0.1", "::1", "localhost"}
    except Exception:
        return False


def _developer_mode_enabled() -> bool:
    env = os.environ.get("SARAH_DEVBRIDGE_ALLOW_APPLY", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    try:
        import SarahMemoryGlobals as SMG  # type: ignore
        return bool(getattr(SMG, "DEVELOPERSMODE", False))
    except Exception:
        return False


def _project_root() -> str:
    try:
        import SarahMemoryGlobals as SMG  # type: ignore
        for attr in ("ROOT_DIR", "BASE_DIR"):
            val = getattr(SMG, attr, None)
            if val:
                return os.path.abspath(os.fspath(val))
    except Exception:
        pass
    try:
        # appdevbridge.py normally lives in api/server; project root is two levels up.
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    except Exception:
        return os.path.abspath(os.getcwd())


def _data_dir() -> str:
    try:
        import SarahMemoryGlobals as SMG  # type: ignore
        val = getattr(SMG, "DATA_DIR", None)
        if val:
            return os.path.abspath(os.fspath(val))
    except Exception:
        pass
    if _META_DB:
        return os.path.abspath(os.path.dirname(_META_DB))
    return os.path.join(_project_root(), "data")


def _bridge_root() -> str:
    try:
        import SarahMemoryGlobals as SMG  # type: ignore
        val = getattr(SMG, "DEVBRIDGE_DIR", None)
        if val:
            root = os.path.abspath(os.fspath(val))
            _ensure_dir(root)
            return root
    except Exception:
        pass
    root = os.path.join(_data_dir(), "devbridge")
    _ensure_dir(root)
    return root


def _subdir(name: str) -> str:
    root = os.path.join(_bridge_root(), name)
    _ensure_dir(root)
    return root


def _state_path() -> str:
    return os.path.join(_bridge_root(), "devbridge_state.json")


def _session_target() -> str:
    state = _read_json(_state_path())
    configured = (
        os.environ.get("SARAHMEMORY_CHATGPT_PROJECT_URL")
        or os.environ.get("SARAH_DEVBRIDGE_CHATGPT_URL")
        or state.get("chatgpt_session_url")
        or DEFAULT_CHATGPT_SESSION_URL
    )
    return str(configured or "").strip()


def _safe_project_relpath(path_value: str) -> str:
    """Normalize a target path into a safe project-relative path.

    This function is intentionally strict for DevBridge.  It rejects absolute
    paths, Windows drive paths, UNC paths, URLs, null bytes, home expansion, and
    any path traversal.  The returned value is always POSIX-style relative text
    for manifest portability across Windows/Linux/PythonAnywhere.
    """
    raw = str(path_value or "").strip()
    if not raw:
        raise ValueError("missing_target_path")
    if "\x00" in raw:
        raise ValueError("target_path_contains_null_byte")
    raw = raw.replace("\\", "/")
    lowered = raw.lower()
    if re.match(r"^[a-z][a-z0-9+.-]*://", lowered):
        raise ValueError("target_path_must_not_be_url")
    if raw.startswith(("/", "~")):
        raise ValueError("target_path_must_be_project_relative")
    if re.match(r"^[a-zA-Z]:/", raw):
        raise ValueError("target_path_must_not_include_drive_letter")
    if raw.startswith("//"):
        raise ValueError("target_path_must_not_be_unc_path")

    parts = []
    for part in raw.split("/"):
        part = part.strip()
        if not part or part == ".":
            continue
        if part == "..":
            raise ValueError("target_path_traversal_detected")
        # Windows reserved separators are already normalized; keep filenames
        # otherwise intact so existing project naming is preserved.
        parts.append(part)
    rel = "/".join(parts)
    if not rel:
        raise ValueError("missing_target_path")
    if len(rel) > 500:
        raise ValueError("target_path_too_long")
    return rel


def _safe_project_path(path_value: str) -> Tuple[str, str]:
    rel = _safe_project_relpath(path_value)
    root = _project_root()
    final = os.path.abspath(os.path.join(root, rel))
    if not final.startswith(root + os.sep) and final != root:
        raise ValueError("target_path_outside_project")
    return rel, final


def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    raw = _decode_json_text(text)
    if not raw:
        return None
    if len(raw) > MAX_TEXT_CHARS:
        raw = raw[:MAX_TEXT_CHARS]
    data, diag = _json_loads_object_tolerant(raw)
    if data and diag.get("valid_json"):
        return data
    return None


def _normalize_candidate_files(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    raw_files: Any = None
    for key in ("files", "patched_files", "updated_files", "candidate_files"):
        if isinstance(payload.get(key), list):
            raw_files = payload.get(key)
            break
    if raw_files is None and isinstance(payload.get("patch"), dict):
        for key in ("files", "patched_files", "updated_files"):
            if isinstance(payload["patch"].get(key), list):
                raw_files = payload["patch"].get(key)
                break
    if not isinstance(raw_files, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in raw_files[:MAX_FILES_PER_STAGE]:
        if not isinstance(item, dict):
            continue
        target_path = str(
            item.get("path")
            or item.get("file")
            or item.get("filename")
            or item.get("target_path")
            or ""
        ).strip()
        content = item.get("content")
        if content is None:
            content = item.get("full_file")
        if content is None:
            content = item.get("code")
        if not target_path or content is None:
            continue
        content_text = str(content)
        if len(content_text) > MAX_FILE_CHARS:
            continue
        rel = _safe_project_relpath(target_path)
        out.append({
            "path": rel,
            "content": content_text,
            "allow_create": _boolish(item.get("allow_create") or item.get("new_file_approved"), False),
            "new_file_approved": _boolish(item.get("new_file_approved") or item.get("allow_create"), False),
            "expected_existing": _boolish(item.get("expected_existing"), False),
            "target_type": str(item.get("target_type") or item.get("artifact_type") or "").strip(),
            "reason": str(item.get("reason") or item.get("change_reason") or item.get("summary") or "").strip()[:1000],
            "ticket_ids": item.get("ticket_ids") if isinstance(item.get("ticket_ids"), list) else [],
            "batch_id": str(item.get("batch_id") or "").strip(),
        })
    return out


def _classify_target_path(rel: str) -> Dict[str, Any]:
    rel_norm = _safe_project_relpath(rel)
    low = rel_norm.lower()
    exists = False
    target_abs = ""
    try:
        _, target_abs = _safe_project_path(rel_norm)
        exists = os.path.exists(target_abs)
    except Exception:
        target_abs = ""

    zone = "core_project"
    if low.startswith("data/devbridge/sandbox/"):
        zone = "devbridge_sandbox"
    elif low.startswith("data/devbridge/"):
        zone = "devbridge_owned"
    elif low.startswith("data/addons/"):
        zone = "addon"
    elif low.startswith("data/drivers/"):
        zone = "driver"
    elif low.startswith("data/exports/"):
        zone = "export"
    elif low.startswith("data/ui/"):
        zone = "frontend_ui"
    elif low.startswith("api/") or low.startswith("app/") or low.endswith(".py"):
        zone = "core_runtime"

    return {
        "target_path": rel_norm,
        "target_abs": target_abs,
        "exists": exists,
        "zone": zone,
        "core_existing_only": low.startswith(CORE_EXISTING_ONLY_PREFIXES) or low in CORE_EXISTING_ONLY_FILES or zone in {"frontend_ui", "core_runtime"},
        "explicit_create_zone": low.startswith(EXPLICIT_CREATE_ZONES),
    }


def _path_policy_for_candidate(rel: str, *, allow_create: bool = False, expected_existing: bool = False) -> Dict[str, Any]:
    info = _classify_target_path(rel)
    warnings: List[str] = []
    errors: List[str] = []
    checks: List[str] = ["project_relative_path_verified"]

    if info["exists"]:
        checks.append("target_exists")
    else:
        if expected_existing:
            errors.append("target_expected_to_exist_but_missing")
        if info["core_existing_only"] and not allow_create:
            errors.append("new_core_file_requires_explicit_approval")
        elif not info["explicit_create_zone"] and not allow_create:
            errors.append("new_file_requires_explicit_approval")
        elif allow_create:
            checks.append("new_file_creation_explicitly_approved")
            warnings.append("new_file_creation_will_delete_on_rollback_if_applied")

    if info["zone"] in {"driver", "addon", "export"} and not allow_create and not info["exists"]:
        errors.append("promotion_target_requires_explicit_create_approval")
    if info["zone"] == "devbridge_sandbox":
        checks.append("sandbox_zone")
    if info["zone"] in {"driver", "addon"}:
        warnings.append("runtime_extension_zone_requires_manifest_validation_before_activation")
    if arile_is_protected_core_file(rel):
        errors.append("direct_patch_blocked_by_arile_policy")
        warnings.append("SarahMemoryARILE.py/SarahMemoryGlobals.py require protected-core manual promotion or ARILE overlay lane; DevBridge direct apply is blocked.")
        _arile_emit_devbridge("protected_core_stage_attempt", f"DevBridge staging blocked for protected core file: {rel}", target_path=rel)

    return {
        **info,
        "ok": not errors,
        "allow_create": bool(allow_create),
        "expected_existing": bool(expected_existing),
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
    }


def _path_check_report(data: Dict[str, Any], *, route: str = "POST /api/devbridge/path-check") -> Dict[str, Any]:
    raw_paths = data.get("paths") if isinstance(data.get("paths"), list) else []
    raw_paths = list(raw_paths)
    if data.get("path"):
        raw_paths.append(data.get("path"))
    allow_create = _boolish(data.get("allow_create") or data.get("new_file_approved"), False)
    expected_existing = _boolish(data.get("expected_existing"), False)

    results: List[Dict[str, Any]] = []
    for idx, path_value in enumerate(raw_paths[:MAX_FILES_PER_STAGE], start=1):
        input_path = str(path_value or "")
        try:
            rel = _safe_project_relpath(input_path)
            policy = _path_policy_for_candidate(rel, allow_create=allow_create, expected_existing=expected_existing)
            target_abs = str(policy.get("target_abs") or "")
            file_info: Dict[str, Any] = {}
            if target_abs and os.path.exists(target_abs):
                try:
                    st = os.stat(target_abs)
                    file_info = {
                        "is_file": os.path.isfile(target_abs),
                        "is_dir": os.path.isdir(target_abs),
                        "size_bytes": int(st.st_size),
                        "mtime": float(st.st_mtime),
                        "sha256": _sha256_file(target_abs) if os.path.isfile(target_abs) else "",
                    }
                except Exception as e:
                    file_info = {"stat_error": str(e)}
            results.append({
                **policy,
                "index": idx,
                "input_path": input_path,
                "normalized_path": rel,
                "file_info": file_info,
            })
        except Exception as e:
            results.append({
                "ok": False,
                "index": idx,
                "input_path": input_path,
                "target_path": input_path,
                "normalized_path": "",
                "exists": False,
                "zone": "invalid",
                "checks": [],
                "warnings": [],
                "errors": [str(e)],
                "file_info": {},
            })

    ok = bool(results) and all(bool(item.get("ok")) for item in results)
    existing_count = len([item for item in results if item.get("exists")])
    missing_count = len(results) - existing_count
    return {
        "ok": ok,
        "route": route,
        "project_root": _project_root(),
        "bridge_root": _bridge_root(),
        "allow_create": bool(allow_create),
        "expected_existing": bool(expected_existing),
        "requested_count": len(raw_paths),
        "checked_count": len(results),
        "truncated_count": max(0, len(raw_paths) - MAX_FILES_PER_STAGE),
        "existing_count": existing_count,
        "missing_count": missing_count,
        "all_existing": missing_count == 0 and bool(results),
        "paths": results,
        "generated_at": _iso(),
    }


def _stage_files(stage_id: str, files: List[Dict[str, Any]], source: str = "import-response") -> Dict[str, Any]:
    stage_id = _sanitize_id(stage_id, "stage")
    stage_dir = os.path.join(_subdir("staged"), stage_id)
    _ensure_dir(stage_dir)
    staged: List[Dict[str, Any]] = []
    for idx, item in enumerate(files[:MAX_FILES_PER_STAGE], start=1):
        rel = _safe_project_relpath(item.get("path") or "")
        content = str(item.get("content") or "")
        safe_name = f"{idx:03d}__{_sanitize_filename(rel.replace('/', '__'), 'file.txt')}"
        staged_path = os.path.join(stage_dir, safe_name)
        _write_text(staged_path, content)
        policy = _path_policy_for_candidate(
            rel,
            allow_create=_boolish(item.get("allow_create") or item.get("new_file_approved"), False),
            expected_existing=_boolish(item.get("expected_existing"), False),
        )
        staged.append({
            "target_path": rel,
            "staged_path": staged_path,
            "size_chars": len(content),
            "sha256": _sha256_text(content),
            "allow_create": bool(policy.get("allow_create")),
            "new_file_approved": bool(policy.get("allow_create")),
            "expected_existing": bool(policy.get("expected_existing")),
            "target_exists_at_stage": bool(policy.get("exists")),
            "pre_stage_sha256": _sha256_file(str(policy.get("target_abs") or "")) if policy.get("exists") else "",
            "target_zone": policy.get("zone"),
            "path_policy": policy,
            "target_type": str(item.get("target_type") or ""),
            "reason": str(item.get("reason") or ""),
            "ticket_ids": item.get("ticket_ids") if isinstance(item.get("ticket_ids"), list) else [],
            "batch_id": str(item.get("batch_id") or ""),
        })
    manifest = {
        "stage_id": stage_id,
        "source": source,
        "created_ts": _now(),
        "created_at": _iso(),
        "project_root": _project_root(),
        "files": staged,
        "status": "staged" if staged else "empty",
    }
    _write_json(os.path.join(stage_dir, "manifest.json"), manifest)
    return manifest


def _stage_manifest(stage_id: str) -> Dict[str, Any]:
    stage_id = _sanitize_id(stage_id, "stage")
    return _read_json(os.path.join(_subdir("staged"), stage_id, "manifest.json"))


def _path_inside(child: str, parent: str) -> bool:
    """Return True when child is equal to or inside parent after normalization."""
    try:
        child_abs = os.path.abspath(child)
        parent_abs = os.path.abspath(parent)
        common = os.path.commonpath([child_abs, parent_abs])
        return common == parent_abs
    except Exception:
        return False


def _stage_dir(stage_id: str) -> str:
    return os.path.abspath(os.path.join(_subdir("staged"), _sanitize_id(stage_id, "stage")))


def _validate_manifest_integrity(stage_id: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Validate staged manifest structure and filesystem boundaries.

    Checks performed:
      - manifest exists and has matching stage_id
      - staged files list is structurally valid
      - staged files live inside the stage directory
      - target paths are project-relative and resolve inside project root
      - recorded size/hash match staged content
    """
    sid = _sanitize_id(stage_id, "stage")
    stage_root = _stage_dir(sid)
    project_root = os.path.abspath(_project_root())
    result: Dict[str, Any] = {
        "ok": True,
        "stage_id": sid,
        "checks": [],
        "errors": [],
        "warnings": [],
        "files": [],
    }

    if not manifest:
        result["ok"] = False
        result["errors"].append("manifest_missing")
        return result

    manifest_stage_id = str(manifest.get("stage_id") or "").strip()
    if manifest_stage_id and _sanitize_id(manifest_stage_id, "stage") != sid:
        result["ok"] = False
        result["errors"].append("manifest_stage_id_mismatch")
    else:
        result["checks"].append("manifest_stage_id")

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        result["ok"] = False
        result["errors"].append("manifest_files_missing")
        return result
    if len(files) > MAX_FILES_PER_STAGE:
        result["ok"] = False
        result["errors"].append("manifest_too_many_files")

    for idx, item in enumerate(files, start=1):
        file_result: Dict[str, Any] = {
            "index": idx,
            "ok": True,
            "checks": [],
            "errors": [],
            "target_path": "",
            "staged_path": "",
        }
        if not isinstance(item, dict):
            file_result["ok"] = False
            file_result["errors"].append("manifest_file_entry_invalid")
            result["files"].append(file_result)
            continue

        target_raw = str(item.get("target_path") or "")
        staged_raw = str(item.get("staged_path") or "")
        try:
            rel, target_abs = _safe_project_path(target_raw)
            file_result["target_path"] = rel
            file_result["target_abs"] = target_abs
            if not _path_inside(target_abs, project_root):
                file_result["ok"] = False
                file_result["errors"].append("target_outside_project_root")
            else:
                file_result["checks"].append("target_inside_project_root")
        except Exception as e:
            file_result["ok"] = False
            file_result["errors"].append(f"target_path_invalid:{e}")

        staged_abs = os.path.abspath(staged_raw) if staged_raw else ""
        file_result["staged_path"] = staged_abs
        if not staged_abs or not os.path.isfile(staged_abs):
            file_result["ok"] = False
            file_result["errors"].append("staged_file_missing")
        elif not _path_inside(staged_abs, stage_root):
            file_result["ok"] = False
            file_result["errors"].append("staged_file_outside_stage_dir")
        else:
            file_result["checks"].append("staged_file_inside_stage_dir")

        if os.path.isfile(staged_abs):
            content = _read_text(staged_abs)
            actual_size = len(content)
            actual_sha = _sha256_text(content)
            file_result["actual_size_chars"] = actual_size
            file_result["actual_sha256"] = actual_sha
            try:
                expected_size = int(item.get("size_chars"))
            except Exception:
                expected_size = None
            expected_sha = str(item.get("sha256") or "").strip()
            if expected_size is not None and expected_size != actual_size:
                file_result["ok"] = False
                file_result["errors"].append("size_mismatch")
            else:
                file_result["checks"].append("size_match")
            if expected_sha and expected_sha != actual_sha:
                file_result["ok"] = False
                file_result["errors"].append("sha256_mismatch")
            else:
                file_result["checks"].append("sha256_match")

            if actual_size > MAX_FILE_CHARS:
                file_result["ok"] = False
                file_result["errors"].append("staged_file_too_large")

        result["files"].append(file_result)

    if not all(bool(f.get("ok")) for f in result["files"]):
        result["ok"] = False
    if result["ok"]:
        result["checks"].append("manifest_integrity")
    return result


def _validate_file_content(target_path: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rel = _safe_project_relpath(target_path)
    ext = os.path.splitext(rel)[1].lower()
    result: Dict[str, Any] = {
        "target_path": rel,
        "ok": True,
        "checks": [],
        "errors": [],
        "warnings": [],
        "extension": ext,
        "size_chars": len(content or ""),
    }

    metadata = metadata if isinstance(metadata, dict) else {}
    try:
        _, final_path = _safe_project_path(rel)
        result["target_abs"] = final_path
        result["checks"].append("target_project_relative")
        path_policy = _path_policy_for_candidate(
            rel,
            allow_create=_boolish(metadata.get("allow_create") or metadata.get("new_file_approved"), False),
            expected_existing=_boolish(metadata.get("expected_existing"), False),
        )
        result["path_policy"] = path_policy
        result["target_exists"] = bool(path_policy.get("exists"))
        result["target_zone"] = path_policy.get("zone")
        result["checks"].extend([c for c in path_policy.get("checks", []) if c not in result["checks"]])
        result["warnings"].extend(path_policy.get("warnings", []))
        if not path_policy.get("ok"):
            result["ok"] = False
            result["errors"].extend(path_policy.get("errors", []))
    except Exception as e:
        result["ok"] = False
        result["errors"].append(f"target_path_invalid:{e}")
        return result

    if len(content or "") > MAX_FILE_CHARS:
        result["ok"] = False
        result["errors"].append("content_too_large")
        return result
    if "\x00" in (content or ""):
        result["ok"] = False
        result["errors"].append("null_byte_detected")
        return result

    if ext == ".py":
        try:
            ast.parse(content or "", filename=rel)
            result["checks"].append("python_ast_parse")
        except SyntaxError as e:
            result["ok"] = False
            result["errors"].append(f"python_syntax:{e}")
        # py_compile catches a few more compile-time conditions.
        if result["ok"]:
            tmp_path = ""
            try:
                with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
                    tf.write(content or "")
                    tmp_path = tf.name
                py_compile.compile(tmp_path, doraise=True)
                result["checks"].append("py_compile")
            except Exception as e:
                result["ok"] = False
                result["errors"].append(f"py_compile:{e}")
            finally:
                try:
                    if tmp_path:
                        os.unlink(tmp_path)
                except Exception:
                    pass
    elif ext == ".json":
        try:
            json.loads(content or "{}")
            result["checks"].append("json_parse")
        except Exception as e:
            result["ok"] = False
            result["errors"].append(f"json_parse:{e}")
    elif ext in {".ts", ".tsx"}:
        # Full TS/TSX validation belongs to npm/tsc in the project environment.
        # This endpoint performs an existence/content integrity gate only.
        if not (content or "").strip():
            result["ok"] = False
            result["errors"].append("typescript_content_empty")
        else:
            result["checks"].append("typescript_text_integrity")
            result["warnings"].append("typescript_compile_not_run_by_devbridge")
    elif ext in {".js", ".jsx", ".css", ".html", ".md", ".txt", ".yml", ".yaml", ".toml"}:
        result["checks"].append("text_integrity")
    else:
        result["checks"].append("binary_or_unknown_skipped")
        result["warnings"].append("syntax_not_checked_for_extension")

    return result


def _validate_stage(stage_id: str) -> Dict[str, Any]:
    sid = _sanitize_id(stage_id, "stage")
    manifest = _stage_manifest(sid)
    if not manifest:
        return {"ok": False, "stage_id": sid, "error": "stage_not_found", "files": []}

    integrity = _validate_manifest_integrity(sid, manifest)
    results: List[Dict[str, Any]] = []

    for item in manifest.get("files") or []:
        if not isinstance(item, dict):
            continue
        target_path = str(item.get("target_path") or "")
        staged_path = str(item.get("staged_path") or "")
        content = _read_text(staged_path)
        try:
            content_result = _validate_file_content(target_path, content, item)
        except Exception as e:
            content_result = {
                "target_path": target_path,
                "ok": False,
                "checks": [],
                "errors": [f"content_validation_exception:{e}"],
            }
        results.append(content_result)

    ok = bool(integrity.get("ok")) and all(bool(r.get("ok")) for r in results) and bool(results)
    validation = {
        "ok": ok,
        "validated_ts": _now(),
        "validated_at": _iso(),
        "integrity": integrity,
        "files": results,
        "checks": [
            "manifest_integrity" if integrity.get("ok") else "manifest_integrity_failed",
            "content_validation" if all(bool(r.get("ok")) for r in results) and results else "content_validation_failed",
        ],
    }
    manifest["validation"] = validation
    manifest["status"] = "validated" if ok else "validation_failed"
    _write_json(os.path.join(_stage_dir(sid), "manifest.json"), manifest)
    return {"ok": ok, "stage_id": sid, "files": results, "integrity": integrity, "manifest": manifest}


def _ensure_tables() -> None:
    """Ensure DevBridge sqlite audit tables exist when META_DB is injected."""
    if not _require_injected():
        return
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS devbridge_records (
                record_id TEXT PRIMARY KEY,
                kind TEXT,
                created_ts REAL,
                updated_ts REAL,
                status TEXT,
                request_json TEXT,
                response_json TEXT,
                path TEXT,
                note TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_devbridge_records_updated ON devbridge_records(updated_ts)")
        con.commit()
    except Exception:
        try:
            if con:
                con.rollback()
        except Exception:
            pass
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _record(kind: str, record_id: str, status: str, request_obj: Dict[str, Any], response_obj: Dict[str, Any], path: str = "", note: str = "") -> None:
    if not _require_injected():
        return
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS devbridge_records (
                record_id TEXT PRIMARY KEY,
                kind TEXT,
                created_ts REAL,
                updated_ts REAL,
                status TEXT,
                request_json TEXT,
                response_json TEXT,
                path TEXT,
                note TEXT
            )
            """
        )
        now = _now()
        cur.execute(
            """
            INSERT INTO devbridge_records(record_id, kind, created_ts, updated_ts, status, request_json, response_json, path, note)
            VALUES(?,?,?,?,?,?,?,?,?)
            ON CONFLICT(record_id) DO UPDATE SET
                updated_ts=excluded.updated_ts,
                status=excluded.status,
                request_json=excluded.request_json,
                response_json=excluded.response_json,
                path=excluded.path,
                note=excluded.note
            """,
            (
                record_id,
                kind,
                now,
                now,
                status,
                json.dumps(request_obj or {}, ensure_ascii=False),
                json.dumps(response_obj or {}, ensure_ascii=False),
                path,
                note,
            ),
        )
        con.commit()
    except Exception:
        try:
            if con:
                con.rollback()
        except Exception:
            pass
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------------

def init_app(app, connect_sqlite=None, meta_db_path=None, api_key_auth_ok=None, sign_ok=None):
    """Called by app.py to inject shared helpers and mount the blueprint."""
    global _CONNECT_SQLITE, _META_DB, _API_KEY_AUTH_OK, _SIGN_OK
    _CONNECT_SQLITE = connect_sqlite
    _META_DB = meta_db_path
    _API_KEY_AUTH_OK = api_key_auth_ok
    _SIGN_OK = sign_ok

    try:
        _ensure_dir(_bridge_root())
        _ensure_dir(_subdir("packets"))
        _ensure_dir(_subdir("responses"))
        _ensure_dir(_subdir("staged"))
        _ensure_dir(_subdir("backups"))
        _ensure_dir(_subdir("audit"))
        _ensure_dir(_subdir("repair_tickets"))
        _ensure_dir(_subdir("repair_batches"))
        _ensure_dir(_subdir("sandbox"))
        _ensure_dir(_subdir("cmd_tickets"))
        _ensure_dir(_subdir("processed_tickets"))
        _ensure_dir(_subdir("failed_tickets"))
    except Exception:
        pass

    if "appdevbridge_v800" in getattr(app, "blueprints", {}):
        return True
    app.register_blueprint(bp)
    return True


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------

@bp.get("/api/devbridge/health")
def devbridge_health():
    return _ok(
        service="devbridge",
        version=PROJECT_VERSION,
        ts=_now(),
        enabled=True,
        project_root=_project_root(),
        bridge_root=_bridge_root(),
        session_target=_session_target(),
        apply_gate={
            "developer_mode": _developer_mode_enabled(),
            "loopback": _is_loopback_request(),
            "env_allow_apply": os.environ.get("SARAH_DEVBRIDGE_ALLOW_APPLY", "").strip().lower() in {"1", "true", "yes", "on"},
        },
        cmd_tickets=_cmd_ticket_counts(),
        cmd_ticket_inventory=_cmd_ticket_counts_extended(),
        routes={
            "cmd_tickets": "GET /api/devbridge/cmd-tickets",
            "cmd_tickets_process": "POST /api/devbridge/cmd-tickets/process",
            "path_check": "POST /api/devbridge/path-check",
        },
    )


@bp.get("/api/devbridge/session-target")
def devbridge_session_target_get():
    return _ok(
        url=_session_target(),
        session_target=_session_target(),
        mode="manual_chatgpt_project_session",
        note="Open this URL manually. DevBridge does not call OpenAI APIs.",
    )


@bp.post("/api/devbridge/session-target")
def devbridge_session_target_set():
    raw = _body_bytes()
    if not _auth_ok(raw):
        return _err("Unauthorized", 401)
    data = _j()
    url = str(data.get("url") or data.get("session_target") or "").strip()
    if not url.startswith("https://chatgpt.com/"):
        return _err("Only chatgpt.com project/session URLs are accepted for this bridge.", 400)
    state = _read_json(_state_path())
    state["chatgpt_session_url"] = url
    state["updated_ts"] = _now()
    state["updated_at"] = _iso()
    _write_json(_state_path(), state)
    return _ok(url=url, saved=True)


@bp.post("/api/devbridge/export-packet")
def devbridge_export_packet():
    raw = _body_bytes()
    if not _auth_ok(raw):
        return _err("Unauthorized", 401)
    data = _j()
    packet_id = _sanitize_id(str(data.get("packet_id") or ""), "packet")
    if not packet_id.startswith("packet_"):
        packet_id = _new_id("packet")

    packet = dict(data)
    packet.setdefault("packet_type", "sarahmemory.devbridge.request.v1")
    packet.setdefault("created_ts", _now())
    packet.setdefault("created_at", _iso())
    packet.setdefault("target_session_url", _session_target())
    packet.setdefault("source", "research_panel")
    packet.setdefault("version", PROJECT_VERSION)

    packet_path = os.path.join(_subdir("packets"), f"{packet_id}.json")
    _write_json(packet_path, packet)
    copy_text = json.dumps(packet, indent=2, sort_keys=True, ensure_ascii=False)

    response = {
        "packet_id": packet_id,
        "path": packet_path,
        "copy_text": copy_text,
        "size_chars": len(copy_text),
        "sha256": _sha256_text(copy_text),
    }
    _record("packet", packet_id, "exported", data, response, packet_path)
    return _ok(**response)


@bp.post("/api/devbridge/import-response")
def devbridge_import_response():
    raw = _body_bytes()
    if not _auth_ok(raw):
        return _err("Unauthorized", 401)
    data = _j()
    response_text = str(data.get("response_text") or data.get("text") or data.get("content") or "")
    if not response_text.strip():
        return _err("Missing response_text", 400)
    if len(response_text) > MAX_TEXT_CHARS:
        response_text = response_text[:MAX_TEXT_CHARS]

    response_id = _sanitize_id(str(data.get("response_id") or ""), "resp")
    if not response_id.startswith("resp_"):
        response_id = _new_id("resp")

    parsed = _extract_json_payload(response_text)
    candidate_files = _normalize_candidate_files(parsed or {})
    stage_manifest: Optional[Dict[str, Any]] = None
    if candidate_files:
        stage_manifest = _stage_files(response_id, candidate_files, source="import-response")

    response_payload = {
        "response_id": response_id,
        "packet_id": data.get("packet_id"),
        "imported_ts": _now(),
        "imported_at": _iso(),
        "source": str(data.get("source") or "research_panel"),
        "response_text": response_text,
        "parsed_json": parsed or {},
        "staged": bool(stage_manifest and stage_manifest.get("files")),
        "stage_id": response_id if stage_manifest else "",
        "stage_manifest": stage_manifest or {},
        "sha256": _sha256_text(response_text),
    }
    response_path = os.path.join(_subdir("responses"), f"{response_id}.json")
    _write_json(response_path, response_payload)

    _record("response", response_id, "imported", data, response_payload, response_path)
    return _ok(
        response_id=response_id,
        response_path=response_path,
        parsed=bool(parsed),
        staged=bool(response_payload["staged"]),
        stage_id=response_payload["stage_id"],
        staged_files=len((stage_manifest or {}).get("files") or []),
        message="ChatGPT response imported and staged." if response_payload["staged"] else "ChatGPT response imported. No full-file patch payload was detected yet.",
    )


@bp.post("/api/devbridge/stage-patch")
def devbridge_stage_patch():
    raw = _body_bytes()
    if not _auth_ok(raw):
        return _err("Unauthorized", 401)
    data = _j()
    files = _normalize_candidate_files(data)
    if not files and isinstance(data.get("response_text"), str):
        parsed = _extract_json_payload(str(data.get("response_text") or ""))
        files = _normalize_candidate_files(parsed or {})
    if not files:
        return _err("No stageable files found. Expected files/patched_files with path + content.", 400)
    stage_id = _sanitize_id(str(data.get("stage_id") or data.get("response_id") or ""), "stage")
    if not stage_id.startswith(("stage_", "resp_")):
        stage_id = _new_id("stage")
    manifest = _stage_files(stage_id, files, source="stage-patch")
    _record("stage", stage_id, "staged", data, manifest, os.path.join(_subdir("staged"), stage_id))
    return _ok(stage_id=stage_id, manifest=manifest, staged_files=len(manifest.get("files") or []))


@bp.post("/api/devbridge/validate")
def devbridge_validate():
    raw = _body_bytes()
    if not _auth_ok(raw):
        return _err("Unauthorized", 401)
    data = _j()
    stage_id = str(data.get("stage_id") or data.get("response_id") or "").strip()
    if not stage_id and bool(data.get("latest")):
        latest_stage = _latest_stage_manifest()
        stage_id = str(latest_stage.get("stage_id") or "").strip()
    if stage_id:
        result = _validate_stage(stage_id)
        _record("validate", _sanitize_id(stage_id, "stage"), "validated" if result.get("ok") else "validation_failed", data, result)
        return jsonify(result), 200 if result.get("ok") else 400

    files = _normalize_candidate_files(data)
    if not files:
        return _err("Missing stage_id or stageable files.", 400)
    results = []
    for f in files:
        try:
            results.append(_validate_file_content(f["path"], f["content"], f))
        except Exception as e:
            results.append({"target_path": f.get("path"), "ok": False, "errors": [str(e)], "checks": []})
    ok = all(bool(r.get("ok")) for r in results)
    return jsonify({"ok": ok, "files": results}), 200 if ok else 400


@bp.get("/api/devbridge/validate/<stage_id>")
def devbridge_validate_get(stage_id: str):
    if not _auth_ok(_body_bytes()):
        return _err("Unauthorized", 401)
    result = _validate_stage(stage_id)
    _record("validate", _sanitize_id(stage_id, "stage"), "validated" if result.get("ok") else "validation_failed", {"stage_id": stage_id, "method": "GET"}, result)
    return jsonify(result), 200 if result.get("ok") else 400


@bp.post("/api/devbridge/apply-approved")
def devbridge_apply_approved():
    raw = _body_bytes()
    if not _auth_ok(raw):
        return _err("Unauthorized", 401)
    data = _j()
    confirm = str(data.get("confirm") or data.get("confirm_phrase") or "").strip()
    if confirm != "APPLY APPROVED PATCH":
        return _err("Explicit confirm phrase required: APPLY APPROVED PATCH", 403)
    if not _developer_mode_enabled():
        return _err("Developer apply gate is closed. Enable DEVELOPERSMODE or SARAH_DEVBRIDGE_ALLOW_APPLY=true.", 403)
    # Cloud safety: unless env explicitly allows apply, require loopback.
    env_allows = os.environ.get("SARAH_DEVBRIDGE_ALLOW_APPLY", "").strip().lower() in {"1", "true", "yes", "on"}
    if not env_allows and not _is_loopback_request():
        return _err("Patch application is restricted to loopback unless SARAH_DEVBRIDGE_ALLOW_APPLY=true.", 403)

    stage_id = str(data.get("stage_id") or data.get("response_id") or "").strip()
    if not stage_id:
        return _err("Missing stage_id", 400)

    validation = _validate_stage(stage_id)
    if not validation.get("ok"):
        return _err("Validation failed; patch not applied.", 400, validation=validation)

    manifest = validation.get("manifest") if isinstance(validation.get("manifest"), dict) else _stage_manifest(stage_id)
    files = manifest.get("files") if isinstance(manifest.get("files"), list) else []
    if not files:
        return _err("No staged files to apply.", 400)

    backup_root = os.path.join(_subdir("backups"), _sanitize_id(stage_id, "stage") + "_" + str(int(_now())))
    applied: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for item in files:
        try:
            rel, target_path = _safe_project_path(str(item.get("target_path") or ""))
            staged_path = str(item.get("staged_path") or "")
            content = _read_text(staged_path)
            if not content and os.path.getsize(staged_path) == 0:
                content = ""
            allow_create = _boolish(item.get("allow_create") or item.get("new_file_approved"), False)
            policy = _path_policy_for_candidate(rel, allow_create=allow_create, expected_existing=_boolish(item.get("expected_existing"), False))
            if arile_is_protected_core_file(rel):
                _arile_emit_devbridge("protected_core_apply_attempt", f"DevBridge apply blocked for protected core file: {rel}", target_path=rel, stage_id=stage_id)
                raise PermissionError("direct_patch_blocked_by_arile_policy")
            if not policy.get("ok"):
                raise ValueError("path_policy_failed:" + ",".join(policy.get("errors", [])))
            existed_before = os.path.exists(target_path)
            pre_apply_sha = _sha256_file(target_path) if existed_before else ""
            backup_path = os.path.join(backup_root, rel)
            if existed_before:
                _ensure_dir(os.path.dirname(backup_path))
                shutil.copy2(target_path, backup_path)
            elif not allow_create:
                raise ValueError("target_missing_and_create_not_approved")
            _ensure_dir(os.path.dirname(target_path))
            _write_text(target_path, content)
            post_apply_sha = _sha256_file(target_path)
            applied.append({
                "target_path": rel,
                "target_abs": target_path,
                "backup_path": backup_path if os.path.exists(backup_path) else "",
                "existed_before": existed_before,
                "created_new_file": not existed_before,
                "pre_apply_sha256": pre_apply_sha,
                "sha256": _sha256_text(content),
                "post_apply_sha256": post_apply_sha,
                "rollback_status": "available" if existed_before or allow_create else "unavailable",
                "path_policy": policy,
            })
        except Exception as e:
            errors.append({"target_path": str(item.get("target_path") or ""), "error": str(e)})

    result = {
        "ok": not errors,
        "stage_id": stage_id,
        "applied": applied,
        "errors": errors,
        "backup_root": backup_root,
        "applied_ts": _now(),
        "applied_at": _iso(),
    }
    manifest["apply_result"] = result
    manifest["status"] = "applied" if result["ok"] else "apply_partial_or_failed"
    _write_json(os.path.join(_subdir("staged"), _sanitize_id(stage_id, "stage"), "manifest.json"), manifest)
    _record("apply", _sanitize_id(stage_id, "stage"), manifest["status"], data, result, backup_root)
    return jsonify(result), 200 if result["ok"] else 500



# -----------------------------------------------------------------------------
# Repair tickets, path verification, rollback, and sandbox governance
# -----------------------------------------------------------------------------

def _list_json_dir(folder: str, limit: int = 50) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        if not os.path.isdir(folder):
            return []
        candidates: List[Tuple[float, str]] = []
        for name in os.listdir(folder):
            if not name.lower().endswith(".json"):
                continue
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                candidates.append((float(os.path.getmtime(path)), path))
        candidates.sort(key=lambda x: x[0], reverse=True)
        for mtime, path in candidates[:max(1, limit)]:
            data = _read_json(path)
            if data:
                data.setdefault("_path", path)
                data.setdefault("_mtime", mtime)
                out.append(data)
    except Exception:
        return out
    return out


def _cmd_ticket_archive_root() -> str:
    root = os.path.join(_bridge_root(), "archive", "failed_tickets")
    _ensure_dir(root)
    return root


def _cmd_ticket_archive_count() -> int:
    root = _cmd_ticket_archive_root()
    total = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(root):
            total += len([name for name in filenames if name.lower().endswith(".json")])
    except Exception:
        return 0
    return total


def _cmd_ticket_counts() -> Dict[str, int]:
    """Return active queue/archive inventory counts as numeric values only."""
    return {
        "pending": int(_dir_count(_subdir("cmd_tickets"))),
        "processed": int(_dir_count(_subdir("processed_tickets"))),
        "failed": int(_dir_count(_subdir("failed_tickets"))),
    }


def _cmd_ticket_counts_extended() -> Dict[str, int]:
    counts = _cmd_ticket_counts()
    counts["archived_failed"] = int(_cmd_ticket_archive_count())
    counts["total_inventory"] = int(counts.get("pending", 0) + counts.get("processed", 0) + counts.get("failed", 0))
    return counts


def _repair_counts() -> Dict[str, Any]:
    cmd_counts = _cmd_ticket_counts()
    return {
        "tickets": int(_dir_count(_subdir("repair_tickets"))),
        "batches": int(_dir_count(_subdir("repair_batches"))),
        "sandboxes": int(len([name for name in os.listdir(_subdir("sandbox"))]) if os.path.isdir(_subdir("sandbox")) else 0),
        "cmd_tickets": cmd_counts,
        "cmd_tickets_pending": int(cmd_counts.get("pending", 0)),
        "cmd_tickets_processed": int(cmd_counts.get("processed", 0)),
        "cmd_tickets_failed": int(cmd_counts.get("failed", 0)),
        "cmd_tickets_archived_failed": int(_cmd_ticket_archive_count()),
    }


def _json_file_candidates(folder: str, limit: int = 50, *, newest_first: bool = True) -> List[str]:
    try:
        if not os.path.isdir(folder):
            return []
        candidates: List[Tuple[float, str]] = []
        for name in os.listdir(folder):
            if not name.lower().endswith(".json"):
                continue
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                candidates.append((float(os.path.getmtime(path)), path))
        candidates.sort(key=lambda item: item[0], reverse=bool(newest_first))
        return [path for _mtime, path in candidates[:max(1, limit)]]
    except Exception:
        return []


def _cmd_ticket_error_summary(record: Dict[str, Any]) -> str:
    if not isinstance(record, dict):
        return ""
    result = record.get("result") if isinstance(record.get("result"), dict) else {}
    original = record.get("original") if isinstance(record.get("original"), dict) else {}
    for value in (
        result.get("error"),
        result.get("detail"),
        record.get("error"),
        original.get("error"),
        original.get("issue"),
        original.get("summary"),
    ):
        text = str(value or "").strip()
        if text:
            return text[:500]
    path_errors: List[str] = []
    paths = result.get("paths") if isinstance(result.get("paths"), list) else []
    for item in paths:
        if not isinstance(item, dict):
            continue
        errs = item.get("errors") if isinstance(item.get("errors"), list) else []
        for err in errs:
            if err:
                path_errors.append(str(err))
    if path_errors:
        return "; ".join(path_errors)[:500]
    return ""


def _cmd_ticket_record_from_file(path: str, category: str) -> Dict[str, Any]:
    data, diag = _read_json_with_diagnostics(path)
    original = data.get("original") if isinstance(data.get("original"), dict) else data
    result = data.get("result") if isinstance(data.get("result"), dict) else {}
    preview = _cmd_ticket_preview(original if isinstance(original, dict) else {})
    mtime = os.path.getmtime(path) if os.path.exists(path) else 0.0
    out: Dict[str, Any] = {
        "file": os.path.basename(path),
        "path": path,
        "category": category,
        "mtime": mtime,
        "mtime_iso": _iso(mtime) if mtime else "",
        "processed_at": str(data.get("processed_at") or ""),
        "processed_ts": data.get("processed_ts") or 0,
        "status": str(data.get("status") or category),
        "diagnostics": diag,
        "error": _cmd_ticket_error_summary(data),
        "result_ok": bool(result.get("ok")),
        "result_error": str(result.get("error") or ""),
        **preview,
    }
    return out


def _cmd_ticket_records(folder: str, category: str, limit: int = 25, *, newest_first: bool = True) -> List[Dict[str, Any]]:
    return [_cmd_ticket_record_from_file(path, category) for path in _json_file_candidates(folder, limit, newest_first=newest_first)]


def _cmd_ticket_failed_archive_paths(limit: int = 25) -> List[Dict[str, Any]]:
    root = _cmd_ticket_archive_root()
    out: List[Tuple[float, str]] = []
    try:
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if not name.lower().endswith(".json"):
                    continue
                path = os.path.join(dirpath, name)
                if os.path.isfile(path):
                    out.append((float(os.path.getmtime(path)), path))
        out.sort(key=lambda item: item[0], reverse=True)
        return [_cmd_ticket_record_from_file(path, "archived_failed") for _mtime, path in out[:max(1, limit)]]
    except Exception:
        return []


def _find_active_failed_ticket(filename: str) -> str:
    safe = _sanitize_filename(filename, "ticket.json")
    folder = _subdir("failed_tickets")
    direct = os.path.join(folder, safe)
    if os.path.isfile(direct):
        return direct
    for path in _json_file_candidates(folder, 500, newest_first=True):
        if os.path.basename(path) == safe or os.path.basename(path).endswith("__" + safe):
            return path
    return ""


def _sandbox_domain_root(domain: str) -> str:
    d = _sanitize_id(domain, "domain").lower()
    if d not in SANDBOX_DOMAINS:
        d = "apps"
    root = os.path.join(_subdir("sandbox"), d)
    _ensure_dir(root)
    return root


def _atomic_move_ticket(src_path: str, dest_folder: str, status: str, result: Optional[Dict[str, Any]] = None) -> str:
    """Move a command ticket into processed/failed with a sidecar result block."""
    _ensure_dir(dest_folder)
    safe_name = _sanitize_filename(os.path.basename(src_path), "ticket.json")
    stamp = datetime.utcfromtimestamp(_now()).strftime("%Y%m%dT%H%M%SZ")
    dest_path = os.path.join(dest_folder, f"{stamp}__{status}__{safe_name}")
    try:
        original = _read_json(src_path)
        wrapped = {
            "ticket_file": safe_name,
            "processed_at": _iso(),
            "processed_ts": _now(),
            "status": status,
            "original": original,
            "result": result or {},
        }
        _write_json(dest_path, wrapped)
        try:
            os.remove(src_path)
        except Exception:
            pass
    except Exception:
        try:
            shutil.move(src_path, dest_path)
        except Exception:
            pass
    return dest_path


def _purge_old_ticket_files(folder: str, older_than_days: Optional[int] = None) -> Dict[str, Any]:
    days = int(older_than_days if older_than_days is not None else CMD_TICKET_RETENTION_DAYS)
    cutoff = _now() - max(1, days) * 86400
    removed: List[str] = []
    errors: List[Dict[str, str]] = []
    try:
        if not os.path.isdir(folder):
            return {"ok": True, "folder": folder, "removed": [], "errors": []}
        for name in os.listdir(folder):
            if not name.lower().endswith(".json"):
                continue
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    removed.append(path)
            except Exception as e:
                errors.append({"path": path, "error": str(e)})
    except Exception as e:
        errors.append({"folder": folder, "error": str(e)})
    return {"ok": not errors, "folder": folder, "removed": removed, "errors": errors}




def _record_governance_event(record: Dict[str, Any], event: Dict[str, Any]) -> Dict[str, Any]:
    """Append a bounded governance event list to a repair ticket/batch record."""
    if not isinstance(record, dict):
        record = {}
    events = record.get("governance_events")
    if not isinstance(events, list):
        events = []
    clean_event = {
        "ts": _now(),
        "at": _iso(),
        "source": str(event.get("source") or "cmd_ticket").strip()[:200],
        "action": str(event.get("action") or "update").strip()[:100],
        "status": str(event.get("status") or "noted").strip()[:100],
        "note": str(event.get("note") or "").strip()[:2000],
        "cmd_ticket_source": str(event.get("cmd_ticket_source") or "").strip(),
    }
    if isinstance(event.get("evidence"), list):
        clean_event["evidence"] = event.get("evidence")[:25]
    events.append(clean_event)
    record["governance_events"] = events[-80:]
    record["updated_ts"] = _now()
    record["updated_at"] = _iso()
    return record


def _status_counts_for_records(folder: str, id_key: str = "ticket_id") -> Dict[str, Any]:
    """Return lightweight status/severity/target-file counts for ticket dashboards."""
    status_counts: Dict[str, int] = {}
    severity_counts: Dict[str, int] = {}
    target_counts: Dict[str, int] = {}
    total = 0
    try:
        if not os.path.isdir(folder):
            return {"total": 0, "by_status": {}, "by_severity": {}, "by_target_file": {}}
        for name in os.listdir(folder):
            if not name.lower().endswith(".json"):
                continue
            path = os.path.join(folder, name)
            if not os.path.isfile(path):
                continue
            data = _read_json(path)
            if not isinstance(data, dict) or not data.get(id_key):
                continue
            total += 1
            status = str(data.get("status") or "unknown").strip() or "unknown"
            severity = str(data.get("severity") or "none").strip() or "none"
            target = str(data.get("target_file") or "none").strip() or "none"
            status_counts[status] = status_counts.get(status, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            target_counts[target] = target_counts.get(target, 0) + 1
    except Exception:
        pass
    return {
        "total": int(total),
        "by_status": status_counts,
        "by_severity": severity_counts,
        "by_target_file": target_counts,
    }


def _update_repair_record_from_cmd_ticket(
    data: Dict[str, Any],
    *,
    source_path: str = "",
    dry_run: bool = False,
    record_kind: str = "ticket",
) -> Dict[str, Any]:
    """Governed cmd-ticket lane for marking repair tickets/batches reviewed.

    This does not patch runtime files. It lets the operator feed disposition
    command tickets back through DevBridge so the backlog can be reduced without
    manual JSON editing.
    """
    is_batch = record_kind == "batch"
    id_key = "batch_id" if is_batch else "ticket_id"
    folder_name = "repair_batches" if is_batch else "repair_tickets"
    raw_id = str(data.get(id_key) or data.get("id") or data.get("record_id") or "").strip()
    record_id = _sanitize_id(raw_id, "batch" if is_batch else "ticket")
    if not raw_id or not record_id.startswith("batch_" if is_batch else "ticket_"):
        return {
            "ok": False,
            "error": f"missing_{id_key}",
            "ticket_type": "repair_batch_disposition" if is_batch else "repair_ticket_disposition",
            "source_path": source_path,
            "dry_run": bool(dry_run),
        }

    folder = _subdir(folder_name)
    path = os.path.join(folder, f"{record_id}.json")
    if not os.path.isfile(path):
        return {
            "ok": False,
            "error": "repair_record_not_found",
            id_key: record_id,
            "path": path,
            "source_path": source_path,
            "dry_run": bool(dry_run),
        }

    current = _read_json(path)
    if not isinstance(current, dict) or not current:
        return {
            "ok": False,
            "error": "repair_record_invalid_json",
            id_key: record_id,
            "path": path,
            "source_path": source_path,
            "dry_run": bool(dry_run),
        }

    allowed_statuses = {
        "detected",
        "buffered",
        "audit_ready",
        "preflight_passed",
        "preflight_failed",
        "satisfied",
        "deferred",
        "duplicate",
        "needs_patch",
        "needs_human_review",
        "rejected",
        "archived",
    }
    status = str(data.get("status") or data.get("disposition") or "audit_ready").strip().lower()
    if status not in allowed_statuses:
        return {
            "ok": False,
            "error": "unsupported_repair_record_status",
            "status": status,
            "allowed_statuses": sorted(allowed_statuses),
            id_key: record_id,
            "source_path": source_path,
            "dry_run": bool(dry_run),
        }

    note = str(data.get("note") or data.get("reason") or data.get("summary") or "").strip()[:4000]
    previous_status = str(current.get("status") or "")
    updated = dict(current)
    updated["status"] = status
    updated["disposition"] = status
    updated["disposition_source"] = str(data.get("source") or "cmd_ticket_disposition").strip()[:200]
    updated["disposition_note"] = note
    updated["disposition_updated_ts"] = _now()
    updated["disposition_updated_at"] = _iso()
    if data.get("superseded_by"):
        updated["superseded_by"] = str(data.get("superseded_by") or "").strip()[:300]
    if isinstance(data.get("evidence"), list):
        updated["disposition_evidence"] = data.get("evidence")[:50]
    if isinstance(data.get("tags"), list):
        prior_tags = updated.get("tags") if isinstance(updated.get("tags"), list) else []
        updated["tags"] = list(dict.fromkeys([str(x) for x in prior_tags + data.get("tags") if str(x).strip()]))[:80]
    _record_governance_event(updated, {
        "source": data.get("source") or "cmd_ticket_disposition",
        "action": "repair_batch_disposition" if is_batch else "repair_ticket_disposition",
        "status": status,
        "note": note,
        "evidence": data.get("evidence") if isinstance(data.get("evidence"), list) else [],
        "cmd_ticket_source": source_path,
    })

    if not dry_run:
        _write_json(path, updated)
        _record(
            "repair_batch_disposition" if is_batch else "repair_ticket_disposition",
            record_id,
            status,
            data,
            {"previous_status": previous_status, "new_status": status, id_key: record_id, "path": path},
            path,
        )

    return {
        "ok": True,
        "ticket_type": "repair_batch_disposition" if is_batch else "repair_ticket_disposition",
        id_key: record_id,
        "path": path,
        "previous_status": previous_status,
        "new_status": status,
        "dry_run": bool(dry_run),
        "note": note,
    }

def _cmd_ticket_type(data: Dict[str, Any]) -> str:
    command = str(data.get("command") or data.get("cmd") or data.get("ticket_type") or "").strip().lower()
    command = command.replace("-", "_").replace("/", "_").replace(".", "_")
    if command in {"path_check", "repair_ticket", "repair_batch", "sandbox_manifest"}:
        return command
    if command in {"repair_ticket_disposition", "ticket_disposition", "repair_ticket_update", "ticket_update"}:
        return "repair_ticket_disposition"
    if command in {"repair_batch_disposition", "batch_disposition", "repair_batch_update", "batch_update"}:
        return "repair_batch_disposition"
    if isinstance(data.get("paths"), list) or data.get("path"):
        # Raw path-check tickets use path/paths and do not carry patch content.
        if not any(k in data for k in ("content", "patched_files", "files", "response_text")):
            return "path_check"
    if data.get("batch_id") or isinstance(data.get("tickets"), list):
        return "repair_batch"
    if data.get("domain") and (data.get("name") or data.get("app_name") or data.get("project_name")):
        return "sandbox_manifest"
    if data.get("ticket_id") or (data.get("issue") and (data.get("target_file") or data.get("target_path") or data.get("file"))):
        return "repair_ticket"
    return "unknown"


def _cmd_ticket_execute(data: Dict[str, Any], *, source_path: str = "", dry_run: bool = False) -> Dict[str, Any]:
    ticket_type = _cmd_ticket_type(data)
    result: Dict[str, Any] = {
        "ok": False,
        "ticket_type": ticket_type,
        "dry_run": bool(dry_run),
        "source_path": source_path,
    }

    if ticket_type == "repair_ticket_disposition":
        return _update_repair_record_from_cmd_ticket(data, source_path=source_path, dry_run=dry_run, record_kind="ticket")

    if ticket_type == "repair_batch_disposition":
        return _update_repair_record_from_cmd_ticket(data, source_path=source_path, dry_run=dry_run, record_kind="batch")

    if ticket_type == "path_check":
        report = _path_check_report(data, route="cmd_ticket:path_check")
        if not report.get("paths"):
            result.update({"error": "Missing path or paths", **report})
            return result
        result.update(report)
        result["count"] = report.get("checked_count", 0)
        return result

    if ticket_type == "repair_ticket":
        title = str(data.get("title") or data.get("issue") or "Repair ticket").strip()[:300]
        target_file = str(data.get("target_file") or data.get("target_path") or data.get("file") or "").strip()
        target_policy: Dict[str, Any] = {}
        if target_file:
            try:
                target_policy = _path_policy_for_candidate(_safe_project_relpath(target_file), expected_existing=_boolish(data.get("expected_existing"), False))
            except Exception as e:
                target_policy = {"ok": False, "errors": [str(e)], "target_path": target_file}
        ticket_id = _sanitize_id(str(data.get("ticket_id") or ""), "ticket")
        if not ticket_id.startswith("ticket_"):
            ticket_id = _new_id("ticket")
        payload = {
            "ticket_id": ticket_id,
            "created_ts": _now(),
            "created_at": _iso(),
            "updated_ts": _now(),
            "updated_at": _iso(),
            "status": "detected",
            "title": title or "Repair ticket",
            "issue": str(data.get("issue") or data.get("summary") or title or "").strip()[:4000],
            "target_file": target_policy.get("target_path") or target_file,
            "target_policy": target_policy,
            "severity": str(data.get("severity") or "medium").strip().lower(),
            "source": str(data.get("source") or "cmd_ticket").strip(),
            "evidence": data.get("evidence") if isinstance(data.get("evidence"), list) else [],
            "related_files": data.get("related_files") if isinstance(data.get("related_files"), list) else [],
            "notes": str(data.get("notes") or "").strip()[:4000],
            "cmd_ticket_source": source_path,
        }
        path = os.path.join(_subdir("repair_tickets"), f"{ticket_id}.json")
        if not dry_run:
            _write_json(path, payload)
            _record("repair_ticket", ticket_id, "detected", data, payload, path)
        result.update({"ok": True, "ticket_id": ticket_id, "ticket": payload, "path": path})
        return result

    if ticket_type == "repair_batch":
        target_file = str(data.get("target_file") or data.get("target_path") or "").strip()
        tickets = data.get("tickets") if isinstance(data.get("tickets"), list) else []
        if not target_file and tickets:
            first = tickets[0] if isinstance(tickets[0], dict) else {}
            target_file = str(first.get("target_file") or first.get("target_path") or "")
        if not target_file:
            result.update({"error": "Missing target_file"})
            return result
        try:
            target_policy = _path_policy_for_candidate(_safe_project_relpath(target_file), expected_existing=True)
        except Exception as e:
            result.update({"error": "target_file_invalid", "detail": str(e)})
            return result
        if not target_policy.get("ok"):
            result.update({"error": "target_file_policy_failed", "target_policy": target_policy})
            return result
        batch_id = _sanitize_id(str(data.get("batch_id") or ""), "batch")
        if not batch_id.startswith("batch_"):
            batch_id = _new_id("batch")
        payload = {
            "batch_id": batch_id,
            "batch_type": "sarahmemory.repair_batch.v1",
            "created_ts": _now(),
            "created_at": _iso(),
            "updated_ts": _now(),
            "updated_at": _iso(),
            "status": "buffered",
            "target_file": target_policy.get("target_path"),
            "target_policy": target_policy,
            "tickets": tickets,
            "ticket_ids": data.get("ticket_ids") if isinstance(data.get("ticket_ids"), list) else [],
            "summary": str(data.get("summary") or "Grouped repair batch").strip()[:2000],
            "notes": str(data.get("notes") or "").strip()[:4000],
            "cmd_ticket_source": source_path,
        }
        path = os.path.join(_subdir("repair_batches"), f"{batch_id}.json")
        if not dry_run:
            _write_json(path, payload)
            _record("repair_batch", batch_id, "buffered", data, payload, path)
        result.update({"ok": True, "batch_id": batch_id, "batch": payload, "path": path})
        return result

    if ticket_type == "sandbox_manifest":
        domain = str(data.get("domain") or data.get("type") or "apps").strip().lower()
        if domain not in SANDBOX_DOMAINS:
            result.update({"error": "unsupported_sandbox_domain", "allowed": sorted(SANDBOX_DOMAINS)})
            return result
        name = _sanitize_id(str(data.get("name") or data.get("app_name") or data.get("project_name") or "sandbox"), "sandbox")
        sandbox_root = os.path.join(_sandbox_domain_root(domain), name)
        manifest = {
            "sandbox_id": f"sandbox_{domain}_{name}",
            "domain": domain,
            "name": name,
            "created_ts": _now(),
            "created_at": _iso(),
            "status": "sandboxed",
            "path": sandbox_root,
            "project_root": _project_root(),
            "purpose": str(data.get("purpose") or data.get("summary") or "").strip()[:2000],
            "promotion_target": str(data.get("promotion_target") or "").strip(),
            "permissions": data.get("permissions") if isinstance(data.get("permissions"), list) else [],
            "safety_class": str(data.get("safety_class") or "normal"),
            "requires_approval": True,
            "cmd_ticket_source": source_path,
        }
        manifest_path = os.path.join(sandbox_root, "manifest.json")
        if not dry_run:
            _ensure_dir(sandbox_root)
            _write_json(manifest_path, manifest)
            _record("sandbox", manifest["sandbox_id"], "sandboxed", data, manifest, manifest_path)
        result.update({"ok": True, "sandbox": manifest, "manifest_path": manifest_path})
        return result

    result.update({"error": "unknown_cmd_ticket_type", "hint": "Expected path/paths, ticket_id/issue+target_file, batch_id/tickets, or domain+name."})
    return result


def _cmd_ticket_files(limit: int = 50) -> List[str]:
    folder = _subdir("cmd_tickets")
    try:
        candidates: List[Tuple[float, str]] = []
        for name in os.listdir(folder):
            if not name.lower().endswith(".json"):
                continue
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                candidates.append((float(os.path.getmtime(path)), path))
        candidates.sort(key=lambda item: item[0])
        return [path for _, path in candidates[:max(1, limit)]]
    except Exception:
        return []


def _cmd_ticket_preview(data: Dict[str, Any]) -> Dict[str, Any]:
    ticket_type = _cmd_ticket_type(data) if isinstance(data, dict) else "invalid"
    paths: List[str] = []
    if isinstance(data, dict):
        if isinstance(data.get("paths"), list):
            paths.extend(str(x) for x in data.get("paths", [])[:MAX_FILES_PER_STAGE])
        for key in ("path", "target_file", "target_path", "file", "promotion_target"):
            value = data.get(key)
            if value:
                paths.append(str(value))
    return {
        "ticket_type": ticket_type,
        "command": str(data.get("command") or data.get("cmd") or data.get("ticket_type") or "") if isinstance(data, dict) else "",
        "ticket_id": str(data.get("ticket_id") or data.get("batch_id") or data.get("sandbox_id") or data.get("record_id") or "") if isinstance(data, dict) else "",
        "summary": str(data.get("summary") or data.get("issue") or data.get("title") or data.get("purpose") or data.get("note") or "")[:300] if isinstance(data, dict) else "",
        "disposition": str(data.get("disposition") or data.get("status") or "") if isinstance(data, dict) else "",
        "paths": paths[:MAX_FILES_PER_STAGE],
    }


@bp.get("/api/devbridge/cmd-tickets")
def devbridge_cmd_tickets_list():
    if not _auth_ok(_body_bytes()):
        return _err("Unauthorized", 401)
    requested_limit = max(1, min(200, int(request.args.get("limit") or 50)))
    detail_limit = max(1, min(100, int(request.args.get("detail_limit") or 25)))
    pending_paths = _cmd_ticket_files(limit=requested_limit)
    pending = []
    invalid_count = 0
    for path in pending_paths:
        data, diag = _read_json_with_diagnostics(path)
        if not diag.get("valid_json"):
            invalid_count += 1
        pending.append({
            "file": os.path.basename(path),
            "path": path,
            "category": "pending",
            "mtime": os.path.getmtime(path) if os.path.exists(path) else 0,
            "mtime_iso": _iso(os.path.getmtime(path)) if os.path.exists(path) else "",
            "diagnostics": diag,
            **_cmd_ticket_preview(data),
        })

    failed = _cmd_ticket_records(_subdir("failed_tickets"), "failed", detail_limit, newest_first=True)
    processed_recent = _cmd_ticket_records(_subdir("processed_tickets"), "processed", detail_limit, newest_first=True)
    archived_failed_recent = _cmd_ticket_failed_archive_paths(detail_limit)
    counts = _cmd_ticket_counts()
    generated_at = _iso()
    inventory_note = (
        "Cmd ticket counts are filesystem inventory counters. They change only when JSON files move between "
        "cmd_tickets, processed_tickets, failed_tickets, or archive. With pending=0, dry-run/process has nothing to process."
    )
    return _ok(
        root=_bridge_root(),
        pending_dir=_subdir("cmd_tickets"),
        processed_dir=_subdir("processed_tickets"),
        failed_dir=_subdir("failed_tickets"),
        failed_archive_dir=_cmd_ticket_archive_root(),
        counts=counts,
        extended_counts=_cmd_ticket_counts_extended(),
        pending=pending,
        pending_count=len(pending),
        failed=failed,
        failed_count=len(failed),
        processed_recent=processed_recent,
        processed_recent_count=len(processed_recent),
        archived_failed_recent=archived_failed_recent,
        archived_failed_count=_cmd_ticket_archive_count(),
        latest_failed=failed[0] if failed else None,
        latest_processed=processed_recent[0] if processed_recent else None,
        invalid_count=invalid_count,
        requested_limit=requested_limit,
        detail_limit=detail_limit,
        max_per_process=MAX_CMD_TICKETS_PER_PROCESS,
        retention_days=CMD_TICKET_RETENTION_DAYS,
        inventory_note=inventory_note,
        generated_at=generated_at,
        updated_at=generated_at,
    )


@bp.post("/api/devbridge/cmd-tickets/process")
def devbridge_cmd_tickets_process():
    raw = _body_bytes()
    if not _auth_ok(raw):
        return _err("Unauthorized", 401)
    data = _j()
    limit = max(1, min(MAX_CMD_TICKETS_PER_PROCESS, int(data.get("limit") or MAX_CMD_TICKETS_PER_PROCESS)))
    dry_run = _boolish(data.get("dry_run"), False)
    delete_after = _boolish(data.get("delete_after"), False)
    retention_days = int(data.get("retention_days") or CMD_TICKET_RETENTION_DAYS)

    processed_dir = _subdir("processed_tickets")
    failed_dir = _subdir("failed_tickets")
    purge = {
        "processed": _purge_old_ticket_files(processed_dir, retention_days),
        "failed": _purge_old_ticket_files(failed_dir, retention_days),
    }

    started_ts = _now()
    files = _cmd_ticket_files(limit=limit)
    results: List[Dict[str, Any]] = []
    for path in files:
        item_result: Dict[str, Any] = {
            "file": os.path.basename(path),
            "path": path,
            "ok": False,
        }
        try:
            payload, diagnostics = _read_json_with_diagnostics(path)
            item_result["diagnostics"] = diagnostics
            item_result.update(_cmd_ticket_preview(payload))
            if not payload or not diagnostics.get("valid_json"):
                item_result.update({"error": "invalid_or_empty_json", "detail": diagnostics.get("error") or "JSON root must be an object"})
                if not dry_run:
                    item_result["moved_to"] = _atomic_move_ticket(path, failed_dir, "failed", item_result)
                results.append(item_result)
                continue
            ticket_result = _cmd_ticket_execute(payload, source_path=path, dry_run=dry_run)
            item_result.update(ticket_result)
            if not dry_run:
                if ticket_result.get("ok"):
                    if delete_after:
                        try:
                            os.remove(path)
                            item_result["deleted"] = True
                        except Exception as e:
                            item_result["delete_error"] = str(e)
                    else:
                        item_result["moved_to"] = _atomic_move_ticket(path, processed_dir, "processed", ticket_result)
                else:
                    item_result["moved_to"] = _atomic_move_ticket(path, failed_dir, "failed", ticket_result)
            results.append(item_result)
        except Exception as e:
            item_result.update({"error": str(e)})
            if not dry_run:
                try:
                    item_result["moved_to"] = _atomic_move_ticket(path, failed_dir, "failed", item_result)
                except Exception as move_err:
                    item_result["move_error"] = str(move_err)
            results.append(item_result)

    ok = all(bool(item.get("ok")) for item in results)
    completed_ts = _now()
    response = {
        "ok": ok,
        "dry_run": dry_run,
        "requested_limit": limit,
        "discovered_count": len(files),
        "processed_count": len([item for item in results if item.get("ok")]),
        "failed_count": len([item for item in results if not item.get("ok")]),
        "invalid_json_count": len([item for item in results if item.get("error") == "invalid_or_empty_json"]),
        "started_at": _iso(started_ts),
        "completed_at": _iso(completed_ts),
        "duration_ms": int((completed_ts - started_ts) * 1000),
        "results": results,
        "counts": _cmd_ticket_counts(),
        "purge": purge,
        "pending_dir": _subdir("cmd_tickets"),
        "processed_dir": processed_dir,
        "failed_dir": failed_dir,
    }
    _record("cmd_ticket_batch", _new_id("cmdtickets"), "processed" if ok else "partial_or_failed", data, response, _subdir("cmd_tickets"))
    return jsonify(response), 200 if ok else 207


@bp.post("/api/devbridge/cmd-tickets/failed/archive")
def devbridge_cmd_tickets_failed_archive():
    raw = _body_bytes()
    if not _auth_ok(raw):
        return _err("Unauthorized", 401)
    if not _developer_mode_enabled():
        return _err("Developer mode is required to archive failed cmd tickets.", 403)
    data = _j()
    archive_all = _boolish(data.get("all"), False)
    raw_files = data.get("files") if isinstance(data.get("files"), list) else []
    single_file = str(data.get("file") or data.get("filename") or "").strip()
    if single_file:
        raw_files.append(single_file)

    candidates: List[str] = []
    if archive_all:
        candidates = _json_file_candidates(_subdir("failed_tickets"), 500, newest_first=False)
    else:
        for item in raw_files:
            found = _find_active_failed_ticket(str(item or ""))
            if found and found not in candidates:
                candidates.append(found)
    if not candidates:
        return _err("no_failed_cmd_tickets_selected", 400, counts=_cmd_ticket_counts())

    stamp = datetime.utcfromtimestamp(_now()).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = os.path.join(_cmd_ticket_archive_root(), stamp)
    _ensure_dir(archive_dir)
    moved: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for src in candidates:
        try:
            dest = os.path.join(archive_dir, _sanitize_filename(os.path.basename(src), "ticket.json"))
            shutil.move(src, dest)
            moved.append({"from": src, "to": dest, "file": os.path.basename(dest)})
        except Exception as e:
            errors.append({"path": src, "error": str(e)})
    result = {
        "ok": not errors,
        "archived_count": len(moved),
        "moved": moved,
        "errors": errors,
        "archive_dir": archive_dir,
        "counts": _cmd_ticket_counts(),
        "archived_failed_count": _cmd_ticket_archive_count(),
        "generated_at": _iso(),
    }
    _record("cmd_ticket_archive_failed", _new_id("cmdarchive"), "archived" if result["ok"] else "archive_partial_or_failed", data, result, archive_dir)
    return jsonify(result), 200 if result["ok"] else 207


@bp.post("/api/devbridge/cmd-tickets/failed/requeue")
def devbridge_cmd_tickets_failed_requeue():
    raw = _body_bytes()
    if not _auth_ok(raw):
        return _err("Unauthorized", 401)
    if not _developer_mode_enabled():
        return _err("Developer mode is required to requeue failed cmd tickets.", 403)
    data = _j()
    raw_files = data.get("files") if isinstance(data.get("files"), list) else []
    single_file = str(data.get("file") or data.get("filename") or "").strip()
    if single_file:
        raw_files.append(single_file)
    candidates: List[str] = []
    for item in raw_files:
        found = _find_active_failed_ticket(str(item or ""))
        if found and found not in candidates:
            candidates.append(found)
    if not candidates:
        return _err("no_failed_cmd_tickets_selected", 400, counts=_cmd_ticket_counts())

    pending_dir = _subdir("cmd_tickets")
    stamp = datetime.utcfromtimestamp(_now()).strftime("%Y%m%dT%H%M%SZ")
    evidence_dir = os.path.join(_cmd_ticket_archive_root(), "requeued", stamp)
    _ensure_dir(evidence_dir)
    requeued: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    for src in candidates:
        try:
            failed_record = _read_json(src)
            original = failed_record.get("original") if isinstance(failed_record.get("original"), dict) else {}
            if not original:
                raise ValueError("failed_ticket_missing_original_payload")
            original_name = _sanitize_filename(str(failed_record.get("ticket_file") or os.path.basename(src)), "ticket.json")
            dest_name = _sanitize_filename(f"requeued_{stamp}__{original_name}", "requeued_ticket.json")
            dest = os.path.join(pending_dir, dest_name)
            _write_json(dest, original)
            evidence = os.path.join(evidence_dir, _sanitize_filename(os.path.basename(src), "failed_ticket.json"))
            shutil.move(src, evidence)
            requeued.append({"from": src, "to": dest, "evidence": evidence, "file": dest_name})
        except Exception as e:
            errors.append({"path": src, "error": str(e)})
    result = {
        "ok": not errors,
        "requeued_count": len(requeued),
        "requeued": requeued,
        "errors": errors,
        "evidence_dir": evidence_dir,
        "counts": _cmd_ticket_counts(),
        "archived_failed_count": _cmd_ticket_archive_count(),
        "generated_at": _iso(),
    }
    _record("cmd_ticket_requeue_failed", _new_id("cmdrequeue"), "requeued" if result["ok"] else "requeue_partial_or_failed", data, result, pending_dir)
    return jsonify(result), 200 if result["ok"] else 207


@bp.post("/api/devbridge/cmd-tickets/purge")
def devbridge_cmd_tickets_purge():
    raw = _body_bytes()
    if not _auth_ok(raw):
        return _err("Unauthorized", 401)
    data = _j()
    days = int(data.get("retention_days") or CMD_TICKET_RETENTION_DAYS)
    result = {
        "processed": _purge_old_ticket_files(_subdir("processed_tickets"), days),
        "failed": _purge_old_ticket_files(_subdir("failed_tickets"), days),
        "counts": _cmd_ticket_counts(),
    }
    return _ok(**result)


@bp.post("/api/devbridge/path-check")
def devbridge_path_check():
    raw = _body_bytes()
    if not _auth_ok(raw):
        return _err("Unauthorized", 401)
    data = _j()
    report = _path_check_report(data)
    if not report.get("paths"):
        return _err("Missing path or paths", 400, **report)
    return jsonify(report), 200 if report.get("ok") else 400


@bp.post("/api/devbridge/repair-ticket")
def devbridge_repair_ticket_create():
    raw = _body_bytes()
    if not _auth_ok(raw):
        return _err("Unauthorized", 401)
    data = _j()
    title = str(data.get("title") or data.get("issue") or "Repair ticket").strip()[:300]
    target_file = str(data.get("target_file") or data.get("target_path") or data.get("file") or "").strip()
    target_policy: Dict[str, Any] = {}
    if target_file:
        try:
            target_policy = _path_policy_for_candidate(_safe_project_relpath(target_file), expected_existing=_boolish(data.get("expected_existing"), False))
        except Exception as e:
            target_policy = {"ok": False, "errors": [str(e)], "target_path": target_file}
    ticket_id = _sanitize_id(str(data.get("ticket_id") or ""), "ticket")
    if not ticket_id.startswith("ticket_"):
        ticket_id = _new_id("ticket")
    payload = {
        "ticket_id": ticket_id,
        "created_ts": _now(),
        "created_at": _iso(),
        "updated_ts": _now(),
        "updated_at": _iso(),
        "status": "detected",
        "title": title or "Repair ticket",
        "issue": str(data.get("issue") or data.get("summary") or title or "").strip()[:4000],
        "target_file": target_policy.get("target_path") or target_file,
        "target_policy": target_policy,
        "severity": str(data.get("severity") or "medium").strip().lower(),
        "source": str(data.get("source") or "devbridge").strip(),
        "evidence": data.get("evidence") if isinstance(data.get("evidence"), list) else [],
        "related_files": data.get("related_files") if isinstance(data.get("related_files"), list) else [],
        "notes": str(data.get("notes") or "").strip()[:4000],
    }
    path = os.path.join(_subdir("repair_tickets"), f"{ticket_id}.json")
    _write_json(path, payload)
    _record("repair_ticket", ticket_id, "detected", data, payload, path)
    return _ok(ticket_id=ticket_id, ticket=payload)


@bp.get("/api/devbridge/repair-tickets")
def devbridge_repair_tickets_list():
    if not _auth_ok(_body_bytes()):
        return _err("Unauthorized", 401)
    limit = max(1, min(MAX_REPAIR_TICKETS_RETURN, int(request.args.get("limit") or 50)))
    tickets = _list_json_dir(_subdir("repair_tickets"), limit)
    return _ok(tickets=tickets, count=len(tickets))


@bp.post("/api/devbridge/repair-batch")
def devbridge_repair_batch_create():
    raw = _body_bytes()
    if not _auth_ok(raw):
        return _err("Unauthorized", 401)
    data = _j()
    target_file = str(data.get("target_file") or data.get("target_path") or "").strip()
    tickets = data.get("tickets") if isinstance(data.get("tickets"), list) else []
    if not target_file and tickets:
        first = tickets[0] if isinstance(tickets[0], dict) else {}
        target_file = str(first.get("target_file") or first.get("target_path") or "")
    if not target_file:
        return _err("Missing target_file", 400)
    try:
        target_policy = _path_policy_for_candidate(_safe_project_relpath(target_file), expected_existing=True)
    except Exception as e:
        return _err("target_file_invalid", 400, detail=str(e))
    if not target_policy.get("ok"):
        return _err("target_file_policy_failed", 400, target_policy=target_policy)
    batch_id = _sanitize_id(str(data.get("batch_id") or ""), "batch")
    if not batch_id.startswith("batch_"):
        batch_id = _new_id("batch")
    payload = {
        "batch_id": batch_id,
        "batch_type": "sarahmemory.repair_batch.v1",
        "created_ts": _now(),
        "created_at": _iso(),
        "updated_ts": _now(),
        "updated_at": _iso(),
        "status": "buffered",
        "target_file": target_policy.get("target_path"),
        "target_policy": target_policy,
        "tickets": tickets,
        "ticket_ids": data.get("ticket_ids") if isinstance(data.get("ticket_ids"), list) else [],
        "summary": str(data.get("summary") or "Grouped repair batch").strip()[:2000],
        "notes": str(data.get("notes") or "").strip()[:4000],
    }
    path = os.path.join(_subdir("repair_batches"), f"{batch_id}.json")
    _write_json(path, payload)
    _record("repair_batch", batch_id, "buffered", data, payload, path)
    return _ok(batch_id=batch_id, batch=payload)


@bp.get("/api/devbridge/repair-batches")
def devbridge_repair_batches_list():
    if not _auth_ok(_body_bytes()):
        return _err("Unauthorized", 401)
    limit = max(1, min(MAX_REPAIR_TICKETS_RETURN, int(request.args.get("limit") or 50)))
    batches = _list_json_dir(_subdir("repair_batches"), limit)
    return _ok(batches=batches, count=len(batches))


@bp.post("/api/devbridge/sandbox/manifest")
def devbridge_sandbox_manifest_create():
    raw = _body_bytes()
    if not _auth_ok(raw):
        return _err("Unauthorized", 401)
    data = _j()
    domain = str(data.get("domain") or data.get("type") or "apps").strip().lower()
    if domain not in SANDBOX_DOMAINS:
        return _err("unsupported_sandbox_domain", 400, allowed=sorted(SANDBOX_DOMAINS))
    name = _sanitize_id(str(data.get("name") or data.get("app_name") or data.get("project_name") or "sandbox"), "sandbox")
    sandbox_root = os.path.join(_sandbox_domain_root(domain), name)
    _ensure_dir(sandbox_root)
    manifest = {
        "sandbox_id": f"sandbox_{domain}_{name}",
        "domain": domain,
        "name": name,
        "created_ts": _now(),
        "created_at": _iso(),
        "status": "sandboxed",
        "path": sandbox_root,
        "project_root": _project_root(),
        "purpose": str(data.get("purpose") or data.get("summary") or "").strip()[:2000],
        "promotion_target": str(data.get("promotion_target") or "").strip(),
        "permissions": data.get("permissions") if isinstance(data.get("permissions"), list) else [],
        "safety_class": str(data.get("safety_class") or "normal"),
        "requires_approval": True,
    }
    manifest_path = os.path.join(sandbox_root, "manifest.json")
    _write_json(manifest_path, manifest)
    _record("sandbox", manifest["sandbox_id"], "sandboxed", data, manifest, manifest_path)
    return _ok(sandbox=manifest, manifest_path=manifest_path)


@bp.post("/api/devbridge/rollback")
def devbridge_rollback():
    raw = _body_bytes()
    if not _auth_ok(raw):
        return _err("Unauthorized", 401)
    data = _j()
    confirm = str(data.get("confirm") or data.get("confirm_phrase") or "").strip()
    if confirm != "ROLLBACK APPROVED PATCH":
        return _err("Explicit confirm phrase required: ROLLBACK APPROVED PATCH", 403)
    if not _developer_mode_enabled():
        return _err("Developer rollback gate is closed. Enable DEVELOPERSMODE or SARAH_DEVBRIDGE_ALLOW_APPLY=true.", 403)
    env_allows = os.environ.get("SARAH_DEVBRIDGE_ALLOW_APPLY", "").strip().lower() in {"1", "true", "yes", "on"}
    if not env_allows and not _is_loopback_request():
        return _err("Rollback is restricted to loopback unless SARAH_DEVBRIDGE_ALLOW_APPLY=true.", 403)

    stage_id = str(data.get("stage_id") or data.get("response_id") or "").strip()
    if not stage_id and bool(data.get("latest")):
        stage_id = str(_latest_stage_manifest().get("stage_id") or "").strip()
    if not stage_id:
        return _err("Missing stage_id", 400)
    sid = _sanitize_id(stage_id, "stage")
    manifest = _stage_manifest(sid)
    if not manifest:
        return _err("stage_not_found", 404, stage_id=sid)
    apply_result = manifest.get("apply_result") if isinstance(manifest.get("apply_result"), dict) else {}
    applied_files = apply_result.get("applied") if isinstance(apply_result.get("applied"), list) else []
    if not applied_files:
        return _err("no_apply_result_to_rollback", 400, stage_id=sid)

    force = _boolish(data.get("force"), False)
    restored: List[Dict[str, Any]] = []
    deleted: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for item in applied_files:
        if not isinstance(item, dict):
            continue
        rel = str(item.get("target_path") or "")
        try:
            rel, target_abs = _safe_project_path(rel)
            backup_path = str(item.get("backup_path") or "")
            expected_current_sha = str(item.get("post_apply_sha256") or item.get("sha256") or "")
            current_sha = _sha256_file(target_abs) if os.path.exists(target_abs) else ""
            if expected_current_sha and current_sha and current_sha != expected_current_sha and not force:
                errors.append({"target_path": rel, "error": "rollback_conflict_current_file_changed", "current_sha256": current_sha, "expected_sha256": expected_current_sha})
                continue
            if backup_path and os.path.isfile(backup_path):
                _ensure_dir(os.path.dirname(target_abs))
                shutil.copy2(backup_path, target_abs)
                restored.append({"target_path": rel, "backup_path": backup_path, "restored_sha256": _sha256_file(target_abs)})
            elif _boolish(item.get("created_new_file"), False):
                if os.path.exists(target_abs):
                    os.remove(target_abs)
                    deleted.append({"target_path": rel, "deleted_created_file": True})
                else:
                    deleted.append({"target_path": rel, "already_absent": True})
            else:
                errors.append({"target_path": rel, "error": "backup_missing_and_not_created_file", "backup_path": backup_path})
        except Exception as e:
            errors.append({"target_path": rel, "error": str(e)})

    result = {
        "ok": not errors,
        "stage_id": sid,
        "restored": restored,
        "deleted": deleted,
        "errors": errors,
        "rollback_ts": _now(),
        "rollback_at": _iso(),
        "forced": force,
    }
    manifest["rollback_result"] = result
    manifest["status"] = "rolled_back" if result["ok"] else "rollback_partial_or_failed"
    _write_json(os.path.join(_stage_dir(sid), "manifest.json"), manifest)
    _record("rollback", sid, manifest["status"], data, result, os.path.join(_stage_dir(sid), "manifest.json"))
    return jsonify(result), 200 if result["ok"] else 409



@bp.get("/api/devbridge/repair-summary")
def devbridge_repair_summary():
    """Return repair backlog counts without requiring the UI to parse every file."""
    if not _auth_ok(_body_bytes()):
        return _err("Unauthorized", 401)
    return _ok(
        generated_at=_iso(),
        bridge_root=_bridge_root(),
        tickets=_status_counts_for_records(_subdir("repair_tickets"), "ticket_id"),
        batches=_status_counts_for_records(_subdir("repair_batches"), "batch_id"),
        cmd_tickets=_cmd_ticket_counts_extended(),
        note=(
            "repair-summary is a read-only backlog dashboard. Use repair_ticket_disposition or "
            "repair_batch_disposition cmd tickets to mark records audit_ready, satisfied, deferred, "
            "duplicate, or needs_patch without manually editing JSON."
        ),
    )

# -----------------------------------------------------------------------------
# Latest/status retrieval helpers
# -----------------------------------------------------------------------------

def _latest_json_from_dir(folder: str, suffix: str = ".json") -> Dict[str, Any]:
    """Return the newest JSON object from a bridge subdirectory.

    This is intentionally filesystem-first so the endpoint remains available
    even when META_DB injection or sqlite is unavailable.
    """
    try:
        if not os.path.isdir(folder):
            return {}
        candidates: List[Tuple[float, str]] = []
        for name in os.listdir(folder):
            if suffix and not name.lower().endswith(suffix.lower()):
                continue
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                try:
                    candidates.append((float(os.path.getmtime(path)), path))
                except Exception:
                    pass
        if not candidates:
            return {}
        candidates.sort(key=lambda item: item[0], reverse=True)
        data = _read_json(candidates[0][1])
        if isinstance(data, dict):
            data.setdefault("_path", candidates[0][1])
            data.setdefault("_mtime", candidates[0][0])
            return data
    except Exception:
        return {}
    return {}


def _latest_stage_manifest() -> Dict[str, Any]:
    """Return the newest staged manifest.json, if any."""
    try:
        staged_root = _subdir("staged")
        candidates: List[Tuple[float, str]] = []
        if not os.path.isdir(staged_root):
            return {}
        for stage_name in os.listdir(staged_root):
            manifest_path = os.path.join(staged_root, stage_name, "manifest.json")
            if os.path.isfile(manifest_path):
                try:
                    candidates.append((float(os.path.getmtime(manifest_path)), manifest_path))
                except Exception:
                    pass
        if not candidates:
            return {}
        candidates.sort(key=lambda item: item[0], reverse=True)
        data = _read_json(candidates[0][1])
        if isinstance(data, dict):
            data.setdefault("_path", candidates[0][1])
            data.setdefault("_mtime", candidates[0][0])
            return data
    except Exception:
        return {}
    return {}


def _latest_record_from_db() -> Dict[str, Any]:
    """Return newest sqlite audit record when META_DB is available."""
    if not _require_injected():
        return {}
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        _ensure_tables()
        cur.execute(
            """
            SELECT record_id, kind, created_ts, updated_ts, status, request_json, response_json, path, note
            FROM devbridge_records
            ORDER BY updated_ts DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if not row:
            return {}

        def _row_get(idx: int, key: str):
            try:
                return row[key]
            except Exception:
                return row[idx]

        request_json = _row_get(5, "request_json") or "{}"
        response_json = _row_get(6, "response_json") or "{}"
        try:
            request_obj = json.loads(request_json)
        except Exception:
            request_obj = {}
        try:
            response_obj = json.loads(response_json)
        except Exception:
            response_obj = {}
        return {
            "record_id": _row_get(0, "record_id"),
            "kind": _row_get(1, "kind"),
            "created_ts": _row_get(2, "created_ts"),
            "updated_ts": _row_get(3, "updated_ts"),
            "status": _row_get(4, "status"),
            "request": request_obj,
            "response": response_obj,
            "path": _row_get(7, "path"),
            "note": _row_get(8, "note"),
        }
    except Exception:
        return {}
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _dir_count(folder: str, suffix: str = ".json") -> int:
    try:
        if not os.path.isdir(folder):
            return 0
        return len([name for name in os.listdir(folder) if name.lower().endswith(suffix.lower())])
    except Exception:
        return 0


@bp.get("/api/devbridge/latest")
def devbridge_latest():
    """Return newest DevBridge packet/response/stage state.

    Frontend and PowerShell smoke tests use this endpoint after
    /api/devbridge/import-response. It is read-only and safe.
    """
    if not _auth_ok(_body_bytes()):
        return _err("Unauthorized", 401)

    packets_dir = _subdir("packets")
    responses_dir = _subdir("responses")
    latest_packet = _latest_json_from_dir(packets_dir)
    latest_response = _latest_json_from_dir(responses_dir)
    latest_stage = _latest_stage_manifest()
    latest_record = _latest_record_from_db()

    # Convenience aliases for older frontend/test code.
    latest_any = latest_response or latest_stage or latest_packet or latest_record or {}

    return _ok(
        service="devbridge",
        ts=_now(),
        version=PROJECT_VERSION,
        bridge_root=_bridge_root(),
        latest=latest_any,
        latest_packet=latest_packet,
        latest_response=latest_response,
        latest_stage=latest_stage,
        latest_record=latest_record,
        counts={
            "packets": _dir_count(packets_dir),
            "responses": _dir_count(responses_dir),
            "stages": len([name for name in os.listdir(_subdir("staged"))]) if os.path.isdir(_subdir("staged")) else 0,
        },
        repair_counts=_repair_counts(),
        cmd_tickets=_cmd_ticket_counts(),
        cmd_ticket_inventory=_cmd_ticket_counts_extended(),
        governance={
            "core_repairs_existing_only_by_default": True,
            "new_files_require_explicit_approval": True,
            "rollback_source": "data/devbridge/backups/<stage_id>_<apply_timestamp>",
            "sandbox_root": os.path.join(_bridge_root(), "sandbox"),
            "cmd_tickets_root": os.path.join(_bridge_root(), "cmd_tickets"),
            "cmd_tickets_endpoint": "POST /api/devbridge/cmd-tickets/process",
        },
    )


@bp.get("/api/devbridge/status")
def devbridge_status():
    """Read-only status alias for operational dashboards."""
    if not _auth_ok(_body_bytes()):
        return _err("Unauthorized", 401)
    return _ok(
        service="devbridge",
        enabled=True,
        ts=_now(),
        version=PROJECT_VERSION,
        bridge_root=_bridge_root(),
        session_target=_session_target(),
        apply_gate={
            "developer_mode": _developer_mode_enabled(),
            "env_allow_apply": os.environ.get("SARAH_DEVBRIDGE_ALLOW_APPLY", "").strip().lower() in {"1", "true", "yes", "on"},
            "loopback": _is_loopback_request(),
        },
        latest_response=bool(_latest_json_from_dir(_subdir("responses"))),
        latest_stage=bool(_latest_stage_manifest()),
        repair_counts=_repair_counts(),
        cmd_tickets=_cmd_ticket_counts(),
        cmd_ticket_inventory=_cmd_ticket_counts_extended(),
        governance={
            "core_repairs_existing_only_by_default": True,
            "new_files_require_explicit_approval": True,
            "rollback_endpoint": "POST /api/devbridge/rollback",
            "path_check_endpoint": "POST /api/devbridge/path-check",
            "sandbox_manifest_endpoint": "POST /api/devbridge/sandbox/manifest",
            "cmd_tickets_endpoint": "POST /api/devbridge/cmd-tickets/process",
            "cmd_tickets_pending_dir": os.path.join(_bridge_root(), "cmd_tickets"),
            "cmd_tickets_processed_dir": os.path.join(_bridge_root(), "processed_tickets"),
            "cmd_tickets_failed_dir": os.path.join(_bridge_root(), "failed_tickets"),
            "cmd_tickets_retention_days": CMD_TICKET_RETENTION_DAYS,
        },
    )


@bp.get("/api/devbridge/records")
def devbridge_records():
    if not _auth_ok(_body_bytes()):
        return _err("Unauthorized", 401)
    limit = max(1, min(200, int(request.args.get("limit") or 50)))
    rows: List[Dict[str, Any]] = []
    if _require_injected():
        con = None
        try:
            con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
            cur = con.cursor()
            cur.execute(
                "SELECT record_id, kind, created_ts, updated_ts, status, path, note FROM devbridge_records ORDER BY updated_ts DESC LIMIT ?",
                (limit,),
            )
            for r in cur.fetchall() or []:
                try:
                    rows.append({"record_id": r[0], "kind": r[1], "created_ts": r[2], "updated_ts": r[3], "status": r[4], "path": r[5], "note": r[6]})
                except Exception:
                    pass
        except Exception:
            rows = []
        finally:
            try:
                if con:
                    con.close()
            except Exception:
                pass
    return _ok(records=rows, count=len(rows))


@bp.get("/api/devbridge/packet/<packet_id>")
def devbridge_packet_get(packet_id: str):
    if not _auth_ok(_body_bytes()):
        return _err("Unauthorized", 401)
    packet_id = _sanitize_id(packet_id, "packet")
    data = _read_json(os.path.join(_subdir("packets"), f"{packet_id}.json"))
    if not data:
        return _err("Packet not found", 404)
    return _ok(packet=data, packet_id=packet_id)


@bp.get("/api/devbridge/response/<response_id>")
def devbridge_response_get(response_id: str):
    if not _auth_ok(_body_bytes()):
        return _err("Unauthorized", 401)
    response_id = _sanitize_id(response_id, "resp")
    data = _read_json(os.path.join(_subdir("responses"), f"{response_id}.json"))
    if not data:
        return _err("Response not found", 404)
    return _ok(response=data, response_id=response_id)

# ====================================================================
# END OF appdevbridge.py v9.0.0
# ====================================================================
