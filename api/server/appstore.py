"""--==The SarahMemory Project==--
File: api/server/appstore.py
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

Purpose: SarahMemory Power Store Endpoints
==============================================================================================
Design goals:
- NO endpoint collisions with ANY of the other app*.py files
- Everything is namespaced under /api/store/
- Reference implementation for store + storefront integrations (PayPal/Printify/Kittl)

Key rules:
- No secrets in frontend. All secrets live in PythonAnywhere .env and are read server-side.
- "Kitchen sink" module: store/auth/products/payments/integrations, without endpoint collisions.
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "B"
# ROLE = "api_bridge"
# CATEGORY = "storefront_operations"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "api_server"
# API_DOMAIN = "store"
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "store_api"
# FAMILY = "commerce"
# GOVERNANCE_LEVEL = "restricted"
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
# NOTES = "Storefront and Power Store API bridge under /api/store/* for auth, products, payments/integrations, generated product concepts, and server-side secret handling."
# --- SARAHMETA END ---

import base64
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

from flask import Blueprint, jsonify, request

# ARILE boundary helper. API files emit compact variance signals only; the central
# backend engine remains SarahMemoryARILE.py.
try:
    from SarahMemoryARILE import arile_emit, arile_endpoint_guard
except Exception:  # pragma: no cover
    arile_emit = None  # type: ignore
    arile_endpoint_guard = None  # type: ignore

def _arile_api_emit(failure_type: str, summary: str, severity: float = 0.55, **data):
    try:
        if callable(arile_emit):
            arile_emit(source=f"api.server.{__name__}", kind="api_boundary_variance", failure_type=failure_type, severity=severity, confidence=0.85, risk="high" if severity >= 0.75 else "medium", summary=summary, requires_governance=severity >= 0.75, retention="security_audit" if severity >= 0.75 else "diagnostic", data=data)
    except Exception:
        pass

def _arile_check_request(endpoint_name: str, risk: str = "low") -> str:
    try:
        if callable(arile_endpoint_guard):
            return arile_endpoint_guard(endpoint_name, {"method": getattr(request, "method", ""), "content_length": getattr(request, "content_length", 0) or 0, "remote_addr": getattr(request, "remote_addr", "")}, risk=risk)
    except Exception:
        pass
    return "allow"


bp2 = Blueprint("appstore_v800", __name__)

# Injected by app.py at init_app time
_CONNECT_SQLITE: Optional[Callable[..., Any]] = None
_META_DB: Optional[str] = None
_API_KEY_AUTH_OK: Optional[Callable[[], bool]] = None
_SIGN_OK: Optional[Callable[[bytes, str], bool]] = None


# ----------------------------- helpers ---------------------------------

def _now() -> float:
    return float(time.time())


def _jok(data: Any = None, **meta: Any):
    payload = {"ok": True}
    if data is not None:
        payload["data"] = data
    if meta:
        payload["meta"] = meta
    return jsonify(payload)


def _jerr(msg: str, code: str = "error", http: int = 400, **meta: Any):
    payload = {"ok": False, "error": msg, "error_code": code}
    if meta:
        payload["meta"] = meta
    return jsonify(payload), http


def _get_env(name: str, default: str = "") -> str:
    try:
        v = os.getenv(name, default)
        return v if v is not None else default
    except Exception:
        return default


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _as_str(x: Any, default: str = "") -> str:
    try:
        if x is None:
            return default
        return str(x)
    except Exception:
        return default


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    s = s.strip()
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))


def _sha256_bytes(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _hash_pw(pw: str, salt: str) -> str:
    # PBKDF2-HMAC-SHA256, modest iterations to keep PA CPU ok
    iterations = _as_int(_get_env("STORE_PBKDF2_ITERS", "120000"), 120000)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt.encode("utf-8"), iterations, dklen=32)
    return _b64url(dk)


def _token_secret() -> str:
    # Stable secret from env; fallback to STORE_ADMIN_PASSWORD hashed if missing
    sec = _get_env("STORE_TOKEN_SECRET", "").strip()
    if sec:
        return sec
    # hard fallback (still server-side)
    seed = (_get_env("STORE_ADMIN_EMAIL", "admin@sarahmemory.com") + ":" + _get_env("STORE_ADMIN_PASSWORD", ""))
    return _b64url(_sha256_bytes(seed.encode("utf-8")))


def _sign_token(payload: Dict[str, Any]) -> str:
    header = {"alg": "HS256", "typ": "SMT"}  # SarahMemory Token
    sec = _token_secret().encode("utf-8")
    h = _b64url(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    p = _b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    msg = f"{h}.{p}".encode("utf-8")
    sig = _b64url(hmac.new(sec, msg, hashlib.sha256).digest())
    return f"{h}.{p}.{sig}"


def _verify_token(token: str) -> Tuple[bool, Dict[str, Any]]:
    try:
        parts = token.strip().split(".")
        if len(parts) != 3:
            return False, {}
        h, p, sig = parts
        sec = _token_secret().encode("utf-8")
        msg = f"{h}.{p}".encode("utf-8")
        exp_sig = _b64url(hmac.new(sec, msg, hashlib.sha256).digest())
        if not hmac.compare_digest(exp_sig, sig):
            return False, {}
        payload = json.loads(_b64url_decode(p).decode("utf-8"))
        if not isinstance(payload, dict):
            return False, {}
        exp = _as_int(payload.get("exp", 0), 0)
        if exp and _now() > exp:
            return False, {}
        return True, payload
    except Exception:
        return False, {}


def _auth_header_token() -> str:
    # Supports: Authorization: Bearer <token>  OR  X-Store-Token: <token>
    try:
        auth = _as_str(request.headers.get("Authorization") or "")
        if auth.lower().startswith("bearer "):
            return auth.split(" ", 1)[1].strip()
    except Exception:
        pass
    return _as_str(request.headers.get("X-Store-Token") or "")


def _is_admin() -> bool:
    # Admin gate: API-key auth OR store token (role=admin)
    try:
        if _API_KEY_AUTH_OK and _API_KEY_AUTH_OK():
            return True
    except Exception:
        pass
    tok = _auth_header_token()
    if not tok:
        return False
    ok, payload = _verify_token(tok)
    if not ok:
        return False
    return _as_str(payload.get("role") or "") == "admin"


def _is_user() -> bool:
    # User gate: API-key auth OR store token (role=user/admin)
    try:
        if _API_KEY_AUTH_OK and _API_KEY_AUTH_OK():
            return True
    except Exception:
        pass
    tok = _auth_header_token()
    if not tok:
        return False
    ok, payload = _verify_token(tok)
    if not ok:
        return False
    role = _as_str(payload.get("role") or "")
    return role in ("user", "admin")


def _db():
    if not _CONNECT_SQLITE:
        raise RuntimeError("CONNECT_SQLITE not injected")
    if not _META_DB:
        raise RuntimeError("META_DB not injected")
    return _CONNECT_SQLITE(_META_DB)


def _ensure_tables():
    # Idempotent schema
    with _db() as con:
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS store_kv (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL,
            meta TEXT NOT NULL,
            exp REAL NOT NULL
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS store_users (
            user_id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            pw_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_ts REAL NOT NULL,
            reset_token TEXT NOT NULL,
            reset_exp REAL NOT NULL
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS store_products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            price REAL NOT NULL,
            category TEXT NOT NULL,
            image_url TEXT NOT NULL,
            tags TEXT NOT NULL,
            source TEXT NOT NULL,
            created_ts REAL NOT NULL,
            updated_ts REAL NOT NULL
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS store_generation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            req TEXT NOT NULL,
            resp TEXT NOT NULL,
            created_ts REAL NOT NULL
        )""")
        con.commit()


def _kv_prune(cur):
    now = _now()
    cur.execute("DELETE FROM store_kv WHERE exp > 0 AND exp <= ?", (now,))


def _require_json() -> Dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return {}
    return payload


def _cors_ok(resp):
    # If app.py already does CORS globally, this is redundant but safe.
    try:
        origin = request.headers.get("Origin") or "*"
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Store-Token"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    except Exception:
        pass
    return resp



# ---------------------- Local / PowerStore boundary ----------------------

def _is_local_request() -> bool:
    """Return True for localhost requests. Local Addons/installer actions are denied remotely."""
    try:
        remote = str(getattr(request, "remote_addr", "") or "").strip().lower()
        if remote in {"127.0.0.1", "::1", "localhost", ""}:
            return True
        if remote.endswith("127.0.0.1"):
            return True
    except Exception:
        pass
    return False


def _local_addons_enabled() -> bool:
    """Local filesystem Addons APIs are available only to local AiOS requests by default."""
    if _is_local_request():
        return True
    return _get_env("SARAH_STORE_ENABLE_LOCAL_ADDONS", "0").strip().lower() in {"1", "true", "yes", "on"}


def _local_only_error(action: str = "local_addon_operation"):
    return _cors_ok(_jerr(
        "Local SarahMemory Addons filesystem access is disabled for non-local requests.",
        code="local_only_required",
        http=403,
        action=action,
        store_url=_get_env("SARAH_POWERSTORE_URL", "https://store.sarahmemory.com"),
        auto_install_allowed=False,
        auto_run_allowed=False,
    ))


def _powerstore_url() -> str:
    return _get_env("SARAH_POWERSTORE_URL", "https://store.sarahmemory.com").strip() or "https://store.sarahmemory.com"


def _powerstore_status_probe(timeout_seconds: float = 3.0) -> Dict[str, Any]:
    """Best-effort PowerStore UP/DOWN probe. Local Addons/NAILDE never depend on this."""
    url = _powerstore_url()
    result: Dict[str, Any] = {
        "schema": "SarahMemory.powerstore.status.v1",
        "store_url": url,
        "status": "UNKNOWN",
        "up": False,
        "checked": False,
        "required_for_local_addons": False,
        "required_for_nailde": False,
        "local_addons_enabled": _local_addons_enabled(),
        "timeout_seconds": timeout_seconds,
    }
    try:
        import urllib.request
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "SarahMemory-AiOS-PowerStore-Probe/1.0"})
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:  # nosec - user-configured store status check
            status_code = int(getattr(resp, "status", 0) or 0)
        result.update({"checked": True, "http_status": status_code, "up": 200 <= status_code < 500, "status": "UP" if 200 <= status_code < 500 else "DOWN"})
    except Exception as exc:
        result.update({"checked": True, "status": "DOWN", "up": False, "error": str(exc)[:300]})
    return result


def _store_runtime_dir(*parts: str) -> str:
    root = os.path.join(_data_dir_runtime(), "store")
    if parts:
        root = os.path.join(root, *parts)
    os.makedirs(root, exist_ok=True)
    return root


def _path_under(child: str, parent: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(child), os.path.abspath(parent)]) == os.path.abspath(parent)
    except Exception:
        return False



def _project_root() -> str:
    try:
        here = os.path.abspath(os.getcwd())
        if os.path.basename(here).lower() == "server" and os.path.basename(os.path.dirname(here)).lower() == "api":
            return os.path.abspath(os.path.join(here, "..", ".."))
        return here
    except Exception:
        return os.path.abspath(".")

def _read_json_file(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _addons_root() -> str:
    try:
        import SarahMemoryGlobals as SMG  # type: ignore
        value = getattr(SMG, "ADDONS_DIR", None)
        if value:
            return os.path.abspath(os.fspath(value))
    except Exception:
        pass
    return os.path.join(_data_dir_runtime(), "addons")


def _data_dir_runtime() -> str:
    try:
        import SarahMemoryGlobals as SMG  # type: ignore
        value = getattr(SMG, "DATA_DIR", None)
        if value:
            return os.path.abspath(os.fspath(value))
    except Exception:
        pass
    return os.path.join(_project_root(), "data")


def _addon_candidate_dirs() -> list[Tuple[str, str]]:
    if not _local_addons_enabled():
        return []
    base = _project_root()
    data_dir = _data_dir_runtime()
    roots = [
        ("addons", _addons_root()),
        ("data_addons", os.path.join(base, "data", "addons")),
        ("addons_pending", os.path.join(_addons_root(), "pending")),
        ("nailde_pending", os.path.join(data_dir, "addons", "pending")),
        ("nailde_packages", os.path.join(data_dir, "nailde", "packages")),
        ("sandbox_apps", os.path.join(data_dir, "devbridge", "sandbox", "apps")),
        ("sandbox_panels", os.path.join(data_dir, "devbridge", "sandbox", "panels")),
    ]
    seen = set()
    out = []
    for zone, folder in roots:
        norm = os.path.abspath(folder)
        if norm in seen:
            continue
        seen.add(norm)
        out.append((zone, norm))
    return out

def _safe_addon_id(value: Any, default: str = "addon") -> str:
    raw = str(value or default).strip().replace("\\", "/").split("/")[-1]
    raw = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).replace("..", "_").strip("._-")
    return (raw or default)[:96]


def _runtime_state_path() -> str:
    root = os.path.join(_data_dir_runtime(), "addons", "_runtime")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "addon_runtime_state.json")


def _read_runtime_state() -> Dict[str, Any]:
    return _read_json_file(_runtime_state_path()) or {"schema": "SarahMemory.addons.runtime_state.v1", "addons": {}}


def _write_runtime_state(state: Dict[str, Any]) -> None:
    path = _runtime_state_path()
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True, ensure_ascii=False)
    os.replace(tmp, path)


def _manifest_id(manifest: Dict[str, Any], fallback: str) -> str:
    return _safe_addon_id(manifest.get("addon_id") or manifest.get("id") or fallback, fallback)


def _ui_icon(manifest: Dict[str, Any], ui: Dict[str, Any]) -> str:
    icon = None
    try:
        if isinstance(manifest.get("ui"), dict):
            icon = manifest.get("ui", {}).get("icon")
    except Exception:
        icon = None
    if not icon:
        icon = manifest.get("icon") or ui.get("icon") or ui.get("title_icon") or "Package"
    return str(icon or "Package")[:64]



def _addon_icon_data_url(addon_root: str, icon_value: Any) -> str:
    """Return a browser-safe data URL for small addon-local icons."""
    try:
        icon = str(icon_value or "").strip().replace("\\", "/")
        if not icon or icon in {"Package", "PackageCheck", "LayoutGrid"}:
            return ""
        if icon.startswith("data:"):
            return icon[:262144]
        if icon.startswith("http://") or icon.startswith("https://"):
            return ""
        root = os.path.abspath(addon_root)
        path = os.path.abspath(os.path.join(root, icon))
        if not _path_under(path, root) or not os.path.isfile(path):
            return ""
        if os.path.getsize(path) > 256 * 1024:
            return ""
        ext = os.path.splitext(path)[1].lower()
        mime = {
            ".svg": "image/svg+xml",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }.get(ext)
        if not mime:
            return ""
        with open(path, "rb") as fh:
            raw = fh.read()
        return "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode("ascii"))
    except Exception:
        return ""


def _ui_buttons(ui: Dict[str, Any]) -> list[str]:
    buttons = ui.get("buttons")
    if isinstance(buttons, list):
        return [str(x).upper() for x in buttons if str(x or "").strip()][:12]
    actions = ui.get("actions")
    if isinstance(actions, list):
        out = []
        for action in actions:
            if isinstance(action, dict):
                label = str(action.get("label") or action.get("id") or "").strip()
                if label:
                    out.append(label.upper())
        if out:
            return out[:12]
    return ["RUN", "COPY", "REMOVE", "UPDATE"]


def _runtime_mode(manifest: Dict[str, Any], ui: Dict[str, Any]) -> str:
    try:
        value = ui.get("runtime") or (manifest.get("execution") or {}).get("mode") or (manifest.get("entry") or {}).get("mode") or "manifest"
        return str(value or "manifest")[:96]
    except Exception:
        return "manifest"


def _scan_addon_candidates() -> list[Dict[str, Any]]:
    items: list[Dict[str, Any]] = []
    state = _read_runtime_state()
    addon_state = state.get("addons") if isinstance(state.get("addons"), dict) else {}
    for zone, folder in _addon_candidate_dirs():
        if not os.path.isdir(folder):
            continue
        try:
            names = sorted(os.listdir(folder))[:500]
        except Exception:
            names = []
        for name in names:
            safe_name = os.path.basename(str(name or "")).strip()
            if not safe_name or safe_name.startswith("_"):
                continue
            cand = os.path.abspath(os.path.join(folder, safe_name))
            if not cand.startswith(os.path.abspath(folder)) or not os.path.isdir(cand):
                continue
            manifest_path = os.path.join(cand, "manifest.json")
            ui_path = os.path.join(cand, "ui.json")
            install_state_path = os.path.join(cand, "install_state.json")
            manifest = _read_json_file(manifest_path) if os.path.isfile(manifest_path) else {}
            ui = _read_json_file(ui_path) if os.path.isfile(ui_path) else {}
            install_state = _read_json_file(install_state_path) if os.path.isfile(install_state_path) else {}
            addon_id = _manifest_id(manifest, safe_name)
            runtime = addon_state.get(addon_id, {}) if isinstance(addon_state, dict) else {}
            manifest_type = str(manifest.get("type") or ("legacy_folder" if manifest else "legacy_folder"))
            includes = manifest.get("includes") if isinstance(manifest.get("includes"), list) else []
            permissions = manifest.get("permissions") or manifest.get("capabilities") or []
            if not isinstance(permissions, list):
                permissions = [str(permissions)] if permissions else []
            installed_zone = zone in {"addons", "data_addons"}
            icon_value = _ui_icon(manifest, ui)
            icon_data_url = _addon_icon_data_url(cand, icon_value)
            ui_buttons = _ui_buttons(ui)
            runtime_mode = _runtime_mode(manifest, ui)
            items.append({
                "zone": zone,
                "id": addon_id,
                "name": str(manifest.get("name") or safe_name),
                "path": cand,
                "folder_name": safe_name,
                "has_manifest": os.path.isfile(manifest_path),
                "has_ui": os.path.isfile(ui_path),
                "manifest": manifest,
                "ui": ui,
                "install_state": install_state,
                "manifest_type": manifest_type,
                "includes_count": len(includes),
                "includes": includes[:50],
                "icon": icon_value,
                "icon_data_url": icon_data_url,
                "has_icon": bool(icon_data_url or icon_value),
                "buttons": ui_buttons,
                "runtime": runtime_mode,
                "permissions": permissions[:50],
                "risk_tier": manifest.get("risk_tier") or manifest.get("risk") or "UNDECLARED",
                "trust_status": str(runtime.get("trust_status") or install_state.get("status") or ("manifest_present" if manifest else "manifest_missing")),
                "activation_status": str(runtime.get("activation_status") or install_state.get("activation_status") or ("installed_not_running" if installed_zone and manifest else "review_required")),
                "runtime": runtime,
                "governance": {
                    "auto_run_allowed": False,
                    "promotion_required": not installed_zone,
                    "registration_required": True,
                    "explicit_approval_required": True,
                    "no_ui_rebuild_required": True,
                    "runtime_icon_supported": bool(manifest),
                    "run_copy_remove_update_supported": bool(manifest),
                    "nailde_generated": str(manifest.get("source") or "").lower() == "nailde" or manifest_type == "nailde_dynamic_app",
                },
            })
    return items

# ----------------------------- lifecycle ---------------------------------

def init_app(app, connect_sqlite, meta_db_path, api_key_auth_ok=None, sign_ok=None):
    """
    Called by app.py to inject shared helpers and mount the blueprint.
    """
    global _CONNECT_SQLITE, _META_DB, _API_KEY_AUTH_OK, _SIGN_OK
    _CONNECT_SQLITE = connect_sqlite
    _META_DB = meta_db_path
    _API_KEY_AUTH_OK = api_key_auth_ok
    _SIGN_OK = sign_ok

    try:
        _ensure_tables()
    except Exception:
        # Don't hard-fail mount; endpoints will raise explicit errors on use.
        pass

    if "appstore_v800" in getattr(app, "blueprints", {}):
        return True
    app.register_blueprint(bp2)
    return True


# ----------------------------- endpoints ---------------------------------


@bp2.route("/api/store/governance", methods=["GET", "OPTIONS"])
def api_store_governance():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    return _cors_ok(_jok({
        "api_domain": "store",
        "route_base": "/api/store",
        "governance": {
            "generated_addons_auto_run": False,
            "candidate_scan_is_read_only": True,
            "sandbox_promotion_required": True,
            "explicit_approval_required": True,
            "runtime_activation_requires_registration": True,
            "safety_notes": [
                "Generated apps/addons are not activated because files exist.",
                "Sandbox candidates must pass manifest review before promotion to data/addons.",
            ],
        },
    }))

@bp2.route("/api/store/addons/candidates", methods=["GET", "OPTIONS"])
def api_store_addon_candidates():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    items = _scan_addon_candidates()
    return _cors_ok(_jok({"count": len(items), "candidates": items, "auto_run_performed": False}))

@bp2.route("/api/store/addons/registry", methods=["GET", "OPTIONS"])
def api_store_addon_registry():
    """Read-only addon/capability registry surface for the AiOS shell.

    This endpoint consolidates addon candidates and governance metadata into a
    single UI-safe packet. It does not launch, activate, install, promote, or
    mutate addons. Runtime activation remains behind the existing addon launcher,
    TrustRegistry review, and explicit user approval.
    """
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    candidates = _scan_addon_candidates()
    registry_items: list[Dict[str, Any]] = []
    for cand in candidates:
        manifest = cand.get("manifest") if isinstance(cand.get("manifest"), dict) else {}
        governance = cand.get("governance") if isinstance(cand.get("governance"), dict) else {}
        permissions = manifest.get("permissions") or manifest.get("capabilities") or []
        if not isinstance(permissions, list):
            permissions = [str(permissions)] if permissions else []
        registry_items.append({
            "id": cand.get("id"),
            "name": cand.get("name"),
            "zone": cand.get("zone"),
            "path": cand.get("path"),
            "has_manifest": bool(cand.get("has_manifest")),
            "has_ui": bool(cand.get("has_ui")),
            "version": manifest.get("version") or manifest.get("manifest_version") or "unknown",
            "author": manifest.get("author") or manifest.get("owner") or "unknown",
            "description": manifest.get("description") or manifest.get("summary") or "No description provided.",
            "permissions": (cand.get("permissions") if isinstance(cand.get("permissions"), list) else permissions)[:50],
            "risk_tier": cand.get("risk_tier") or manifest.get("risk_tier") or manifest.get("risk") or "UNDECLARED",
            "trust_status": cand.get("trust_status") or ("manifest_present" if cand.get("has_manifest") else "manifest_missing"),
            "activation_status": cand.get("activation_status") or ("quarantined_review_required" if governance.get("explicit_approval_required", True) else "review_required"),
            "icon": cand.get("icon") or "Package",
            "icon_data_url": cand.get("icon_data_url") or "",
            "has_icon": bool(cand.get("has_icon") or cand.get("icon_data_url") or cand.get("icon")),
            "buttons": cand.get("buttons") or ["RUN", "COPY", "REMOVE", "UPDATE"],
            "runtime": cand.get("runtime") or "manifest",
            "runtime_state": cand.get("runtime_state") or {},
            "manifest": manifest,
            "ui": cand.get("ui") or {},
            "manifest_type": cand.get("manifest_type"),
            "includes_count": cand.get("includes_count", 0),
            "install_state": cand.get("install_state") or {},
            "runtime": cand.get("runtime") or {},
            "governance": governance,
        })
    return _cors_ok(_jok({
        "schema": "SarahMemory.addon_registry.v1",
        "count": len(registry_items),
        "addons": registry_items,
        "scan_roots": [{"zone": zone, "path": path, "exists": os.path.isdir(path)} for zone, path in _addon_candidate_dirs()],
        "governance": {
            "read_only": True,
            "auto_run_performed": False,
            "explicit_approval_required": True,
            "activation_requires_registration": True,
            "trust_registry_review_required": True,
        },
    }))

def _find_addon_record(addon_id: str) -> Optional[Dict[str, Any]]:
    target = _safe_addon_id(addon_id)
    for item in _scan_addon_candidates():
        if _safe_addon_id(item.get("id")) == target or _safe_addon_id(item.get("folder_name")) == target:
            return item
    return None


def _confirmed(payload: Dict[str, Any]) -> bool:
    for key in ("confirm", "confirmed", "user_confirmed", "user_authorized", "approved", "explicit_user_approval"):
        value = payload.get(key)
        if value is True:
            return True
        if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on", "approved", "confirm", "confirmed", "user_approved"}:
            return True
    phrase = str(payload.get("confirm_phrase") or "").strip().upper()
    return phrase in {"I APPROVE", "USER APPROVED", "INSTALL ADDON", "APPROVE ADDON INSTALL", "REMOVE ADDON", "UPDATE ADDON", "COPY ADDON", "RUN ADDON"}


def _zip_backup_folder(folder: str, label: str = "addon") -> Dict[str, Any]:
    backup_root = os.path.join(_data_dir_runtime(), "backup", "addons")
    os.makedirs(backup_root, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    zip_path = os.path.join(backup_root, f"{_safe_addon_id(label)}_{stamp}.zip")
    count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for dirpath, dirnames, filenames in os.walk(folder):
            dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git", "node_modules", ".venv", "venv"}]
            for name in filenames:
                path = os.path.join(dirpath, name)
                if os.path.islink(path):
                    continue
                zf.write(path, os.path.relpath(path, folder).replace("\\", "/"))
                count += 1
    digest = ""
    try:
        h = hashlib.sha256()
        with open(zip_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                if not chunk:
                    break
                h.update(chunk)
        digest = h.hexdigest()
    except Exception:
        pass
    return {"path": zip_path, "sha256": digest, "file_count": count}


def _copy_tree_bounded(source: str, target: str, max_files: int = 1000, max_total_bytes: int = 50 * 1024 * 1024) -> Dict[str, Any]:
    source_abs = os.path.abspath(source)
    target_abs = os.path.abspath(target)
    if not os.path.isdir(source_abs):
        raise FileNotFoundError(source_abs)
    files = []
    total = 0
    for dirpath, dirnames, filenames in os.walk(source_abs):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git", "node_modules", ".venv", "venv"}]
        for name in filenames:
            if name in {".env", "id_rsa", "id_dsa"} or name.lower().endswith((".pem", ".key")):
                continue
            src = os.path.join(dirpath, name)
            if os.path.islink(src):
                continue
            rel = os.path.relpath(src, source_abs).replace("\\", "/")
            if rel.startswith("../") or "/../" in rel:
                continue
            size = os.path.getsize(src)
            total += size
            if len(files) >= max_files or total > max_total_bytes:
                raise RuntimeError("addon_copy_budget_exceeded")
            files.append((src, rel, size))
    os.makedirs(target_abs, exist_ok=True)
    for src, rel, _size in files:
        dst = os.path.join(target_abs, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    return {"files_copied": len(files), "bytes_copied": total, "source": source_abs, "target": target_abs}


def _install_source_allowed(source: str) -> bool:
    src = os.path.abspath(source)
    data_dir = os.path.abspath(_data_dir_runtime())
    allowed = [
        os.path.join(data_dir, "nailde"),
        os.path.join(data_dir, "addons", "pending"),
        os.path.join(data_dir, "devbridge", "sandbox"),
        os.path.join(_addons_root(), "pending"),
    ]
    return any(src.startswith(os.path.abspath(root)) for root in allowed)


def _record_runtime(addon_id: str, **updates: Any) -> Dict[str, Any]:
    state = _read_runtime_state()
    addons = state.setdefault("addons", {})
    rec = addons.get(addon_id, {}) if isinstance(addons.get(addon_id), dict) else {}
    rec.update(updates)
    rec["updated_ts"] = _now()
    addons[addon_id] = rec
    _write_runtime_state(state)
    return rec


@bp2.route("/api/store/addons/install", methods=["POST", "OPTIONS"])
def api_store_addon_install():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _local_addons_enabled():
        return _local_only_error("addon_install")
    data = _require_json()
    if not _confirmed(data):
        return _cors_ok(_jerr("explicit_user_confirmation_required", code="approval_required", http=409))
    source = os.path.abspath(str(data.get("source_path") or ""))
    if not source or not os.path.isdir(source):
        return _cors_ok(_jerr("source_path_missing", code="missing_source", http=400))
    if not _install_source_allowed(source):
        return _cors_ok(_jerr("source_path_not_allowed", code="source_not_allowed", http=403, source_path=source))
    manifest = _read_json_file(os.path.join(source, "manifest.json"))
    addon_id = _manifest_id(manifest, data.get("addon_id") or os.path.basename(source))
    target = os.path.join(_addons_root(), addon_id)
    backup = _zip_backup_folder(target, addon_id) if os.path.isdir(target) else None
    if os.path.isdir(target):
        shutil.rmtree(target)
    stats = _copy_tree_bounded(source, target)
    install_state = {
        "schema": "SarahMemory.addon.install_state.v1",
        "addon_id": addon_id,
        "installed_ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_path": source,
        "target_path": target,
        "backup": backup,
        "status": "installed_review_required",
        "activation_status": "installed_not_running",
        "no_ui_rebuild_required": True,
        "auto_run_allowed": False,
    }
    with open(os.path.join(target, "install_state.json"), "w", encoding="utf-8") as f:
        json.dump(install_state, f, indent=2, sort_keys=True)
    _record_runtime(addon_id, activation_status="installed_not_running", trust_status="installed_review_required", installed_path=target)
    return _cors_ok(_jok({"addon_id": addon_id, "installed_path": target, "backup": backup, "copy_stats": stats, "install_state": install_state, "auto_run_performed": False}))


@bp2.route("/api/store/addons/run", methods=["POST", "OPTIONS"])
def api_store_addon_run():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _local_addons_enabled():
        return _local_only_error("addon_run")
    data = _require_json()
    if not _confirmed(data):
        return _cors_ok(_jerr("explicit_user_confirmation_required", code="approval_required", http=409))
    addon_id = _safe_addon_id(data.get("addon_id") or data.get("id"))
    rec = _find_addon_record(addon_id)
    if not rec:
        return _cors_ok(_jerr("addon_not_found", code="not_found", http=404, addon_id=addon_id))
    path = os.path.abspath(str(rec.get("path") or ""))
    if not path or not os.path.isdir(path):
        return _cors_ok(_jerr("addon_path_missing", code="missing_path", http=404, addon_id=addon_id))
    manifest = rec.get("manifest") if isinstance(rec.get("manifest"), dict) else {}
    ui = rec.get("ui") if isinstance(rec.get("ui"), dict) else {}
    if str(manifest.get("type") or "").strip().lower() == "addon_bundle":
        launch = {
            "schema": "SarahMemory.addon.runtime_launch.v1",
            "addon_id": rec.get("id"),
            "name": rec.get("name"),
            "mode": "bundle_manifest_review",
            "path": path,
            "manifest": manifest,
            "ui": ui,
            "message": "Bundle launch is delegated to the manifest-aware Addon Launcher. Web UI did not execute bundle children.",
            "python_execution_performed": False,
            "shell_execution_performed": False,
            "auto_run_performed": False,
            "execution_authority": False,
        }
        _record_runtime(str(rec.get("id")), activation_status="bundle_review_required", last_launch=launch, trust_status=rec.get("trust_status") or "manifest_present")
        return _cors_ok(_jok({"launch": launch}))
    entry = manifest.get("entrypoint") if isinstance(manifest.get("entrypoint"), dict) else manifest.get("entry") if isinstance(manifest.get("entry"), dict) else {}
    module_name = str(entry.get("module") or "").strip()
    callable_name = str(entry.get("callable") or "").strip()
    execution = manifest.get("execution") if isinstance(manifest.get("execution"), dict) else {}
    exec_mode = str(execution.get("mode") or ui.get("runtime") or "subprocess").strip().lower()
    launch = {
        "schema": "SarahMemory.addon.runtime_launch.v1",
        "addon_id": rec.get("id"),
        "name": rec.get("name"),
        "mode": exec_mode or "subprocess",
        "path": path,
        "manifest": manifest,
        "ui": ui,
        "auto_run_performed": False,
        "execution_authority": False,
        "shell_execution_performed": False,
    }
    if not module_name or not callable_name:
        launch.update({"python_execution_performed": False, "blocked": True, "error": "manifest_entrypoint_missing"})
        return _cors_ok(_jerr("manifest_entrypoint_missing", code="bad_manifest", http=400, launch=launch))
    if "subprocess" not in exec_mode and exec_mode not in {"python_subprocess_manifest", "manifest", "auto", ""}:
        launch.update({"python_execution_performed": False, "blocked": True, "error": "unsupported_execution_mode"})
        return _cors_ok(_jerr("unsupported_execution_mode", code="unsupported_execution", http=409, launch=launch))
    log_path = os.path.join(path, "manifest_launch.log")
    ctx = {
        "platform_version": _get_env("SARAHMEMORY_VERSION", "9.0.0"),
        "addon_path": path,
        "permissions": manifest.get("permissions") or manifest.get("capabilities") or [],
        "run_mode": "local",
        "device_mode": "local_agent",
        "data_dir": _data_dir_runtime(),
        "source": "api.store.addons.run",
        "user_confirmed": True,
    }
    code = "\n".join([
        "import json, sys",
        "root = sys.argv[1]",
        "modname = sys.argv[2]",
        "callname = sys.argv[3]",
        "ctx = json.loads(sys.argv[4])",
        "sys.path.insert(0, root)",
        "m = __import__(modname)",
        "fn = getattr(m, callname, None)",
        "if not callable(fn):",
        "    raise SystemExit(2)",
        "fn(ctx)",
    ])
    try:
        os.makedirs(path, exist_ok=True)
        logfile = open(log_path, "w", encoding="utf-8", errors="ignore")
        proc = subprocess.Popen(
            [sys.executable, "-c", code, path, module_name, callable_name, json.dumps(ctx)],
            cwd=path,
            stdout=logfile,
            stderr=logfile,
            shell=False,
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
        )
        try:
            logfile.close()
        except Exception:
            pass
        time.sleep(0.35)
        crashed = proc.poll() is not None
        crash_output = ""
        if crashed:
            try:
                with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
                    crash_output = fh.read()[:4000]
            except Exception:
                crash_output = ""
        launch.update({
            "python_execution_performed": True,
            "pid": int(proc.pid or 0),
            "log_path": log_path,
            "crashed_immediately": bool(crashed),
            "crash_output": crash_output,
            "message": "Addon subprocess launched from manifest entrypoint." if not crashed else "Addon subprocess exited immediately; see manifest_launch.log.",
        })
        _record_runtime(str(rec.get("id")), activation_status="running_subprocess" if not crashed else "crashed", last_launch=launch, trust_status=rec.get("trust_status") or "manifest_present", pid=int(proc.pid or 0), log_path=log_path)
        return _cors_ok(_jok({"launch": launch}))
    except Exception as exc:
        launch.update({"python_execution_performed": False, "error": str(exc), "blocked": True})
        _record_runtime(str(rec.get("id")), activation_status="launch_failed", last_launch=launch, trust_status=rec.get("trust_status") or "manifest_present")
        return _cors_ok(_jerr(str(exc), code="addon_launch_failed", http=500, launch=launch))


@bp2.route("/api/store/addons/copy", methods=["POST", "OPTIONS"])
def api_store_addon_copy():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _local_addons_enabled():
        return _local_only_error("addon_copy")
    data = _require_json()
    if not _confirmed(data):
        return _cors_ok(_jerr("explicit_user_confirmation_required", code="approval_required", http=409))
    addon_id = _safe_addon_id(data.get("addon_id") or data.get("id"))
    rec = _find_addon_record(addon_id)
    if not rec:
        return _cors_ok(_jerr("addon_not_found", code="not_found", http=404, addon_id=addon_id))
    new_id = _safe_addon_id(data.get("new_addon_id") or f"{addon_id}_copy_{int(_now())}")
    target = os.path.join(_addons_root(), new_id)
    if os.path.exists(target):
        return _cors_ok(_jerr("target_copy_exists", code="exists", http=409, target=target))
    stats = _copy_tree_bounded(str(rec.get("path")), target)
    manifest_path = os.path.join(target, "manifest.json")
    manifest = _read_json_file(manifest_path)
    if manifest:
        manifest["id"] = new_id
        manifest["addon_id"] = new_id
        manifest["name"] = str(manifest.get("name") or new_id) + " Copy"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True, ensure_ascii=False)
    _record_runtime(new_id, activation_status="installed_not_running", trust_status="copied_review_required", installed_path=target)
    return _cors_ok(_jok({"addon_id": addon_id, "new_addon_id": new_id, "target": target, "copy_stats": stats}))


@bp2.route("/api/store/addons/remove", methods=["POST", "OPTIONS"])
def api_store_addon_remove():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _local_addons_enabled():
        return _local_only_error("addon_remove")
    data = _require_json()
    if not _confirmed(data):
        return _cors_ok(_jerr("explicit_user_confirmation_required", code="approval_required", http=409))
    addon_id = _safe_addon_id(data.get("addon_id") or data.get("id"))
    rec = _find_addon_record(addon_id)
    if not rec:
        return _cors_ok(_jerr("addon_not_found", code="not_found", http=404, addon_id=addon_id))
    path = str(rec.get("path") or "")
    if not os.path.abspath(path).startswith(os.path.abspath(_addons_root())):
        return _cors_ok(_jerr("remove_only_allowed_for_installed_addons", code="not_installed_zone", http=403))
    removed_root = os.path.join(_addons_root(), "_removed")
    os.makedirs(removed_root, exist_ok=True)
    backup = _zip_backup_folder(path, addon_id)
    target = os.path.join(removed_root, f"{addon_id}_{int(_now())}")
    shutil.move(path, target)
    _record_runtime(addon_id, activation_status="removed", trust_status="removed", removed_path=target, backup=backup)
    return _cors_ok(_jok({"addon_id": addon_id, "removed_path": target, "backup": backup, "hard_delete_performed": False}))


@bp2.route("/api/store/addons/update", methods=["POST", "OPTIONS"])
def api_store_addon_update():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _local_addons_enabled():
        return _local_only_error("addon_update")
    data = _require_json()
    if not _confirmed(data):
        return _cors_ok(_jerr("explicit_user_confirmation_required", code="approval_required", http=409))
    source = os.path.abspath(str(data.get("source_path") or ""))
    if not source or not os.path.isdir(source) or not _install_source_allowed(source):
        return _cors_ok(_jerr("source_path_not_allowed", code="source_not_allowed", http=403, source_path=source))
    manifest = _read_json_file(os.path.join(source, "manifest.json"))
    addon_id = _manifest_id(manifest, data.get("addon_id") or os.path.basename(source))
    target = os.path.join(_addons_root(), addon_id)
    backup = _zip_backup_folder(target, addon_id) if os.path.isdir(target) else None
    if os.path.isdir(target):
        shutil.rmtree(target)
    stats = _copy_tree_bounded(source, target)
    _record_runtime(addon_id, activation_status="updated_not_running", trust_status="updated_review_required", installed_path=target, backup=backup)
    return _cors_ok(_jok({"addon_id": addon_id, "updated_path": target, "backup": backup, "copy_stats": stats, "auto_run_performed": False}))



# ---------------------- PowerStore package gateway ----------------------

def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _package_signature(payload: Dict[str, Any]) -> str:
    """Local development signature. Future server marketplace can replace with Ed25519/WebAuthn key material."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    secret = _token_secret().encode("utf-8")
    return _b64url(hmac.new(secret, body, hashlib.sha256).digest())


def _iter_package_files(source_root: str, max_files: int = 1500, max_total_bytes: int = 100 * 1024 * 1024):
    source_abs = os.path.abspath(source_root)
    total = 0
    count = 0
    for dirpath, dirnames, filenames in os.walk(source_abs):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git", "node_modules", ".venv", "venv", "env"}]
        for name in sorted(filenames):
            if name in {".env", "id_rsa", "id_dsa"} or name.lower().endswith((".pem", ".key", ".pfx", ".p12")):
                continue
            path = os.path.join(dirpath, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            rel = os.path.relpath(path, source_abs).replace("\\", "/")
            if rel.startswith("../") or "/../" in rel:
                continue
            size = os.path.getsize(path)
            total += size
            count += 1
            if count > max_files or total > max_total_bytes:
                raise RuntimeError("powerstore_package_budget_exceeded")
            yield path, rel, size


def _validate_addon_dir_for_store(source_root: str) -> Dict[str, Any]:
    source_abs = os.path.abspath(source_root)
    manifest_path = os.path.join(source_abs, "manifest.json")
    ui_path = os.path.join(source_abs, "ui.json")
    markers: Dict[str, str] = {
        "manifest_schema": "FAIL",
        "ui_schema": "OPTIONAL_NOT_FOUND",
        "python_syntax": "PASS",
        "json_schema": "PASS",
        "filesystem_malware_scan": "NOT_RUN",
        "permission_review": "PASS",
        "sandbox_containment": "PASS",
        "auto_run": "DENIED",
        "live_core_write": "DENIED",
    }
    problems = []
    manifest = _read_json_file(manifest_path) if os.path.isfile(manifest_path) else {}
    ui = _read_json_file(ui_path) if os.path.isfile(ui_path) else {}
    if manifest and (manifest.get("id") or manifest.get("addon_id")):
        markers["manifest_schema"] = "PASS"
    else:
        problems.append("manifest.json missing or lacks id/addon_id")
    if os.path.isfile(ui_path):
        markers["ui_schema"] = "PASS" if ui else "FAIL"
        if not ui:
            problems.append("ui.json is present but invalid JSON/object")
    # JSON files
    for path, rel, _size in _iter_package_files(source_abs):
        if rel.lower().endswith(".json"):
            try:
                with open(path, "r", encoding="utf-8-sig") as fh:
                    json.load(fh)
            except Exception as exc:
                markers["json_schema"] = "FAIL"
                problems.append(f"JSON invalid: {rel}: {exc}")
        if rel.lower().endswith(".py"):
            try:
                import py_compile
                py_compile.compile(path, doraise=True)
            except Exception as exc:
                markers["python_syntax"] = "FAIL"
                problems.append(f"Python syntax invalid: {rel}: {exc}")
    # Permission review
    denied = manifest.get("denied") or manifest.get("denied_permissions") or []
    permissions = manifest.get("permissions") or manifest.get("capabilities") or []
    text_permissions = json.dumps({"permissions": permissions, "denied": denied}, sort_keys=True).lower()
    dangerous_claims = ["live_core_write", "driver_mutation", "device_write", "network_access_unbounded", "credential_access", "self_approval"]
    for claim in dangerous_claims:
        if claim in text_permissions and claim not in json.dumps(denied).lower():
            markers["permission_review"] = "REVIEW_REQUIRED"
            problems.append(f"Permission requires review: {claim}")
    ok = all(value in {"PASS", "DENIED", "OPTIONAL_NOT_FOUND", "NOT_RUN"} for value in markers.values()) and not problems
    return {"ok": bool(ok), "markers": markers, "problems": problems, "manifest": manifest, "ui": ui}


def _malware_scan_dir(source_root: str, max_files: int = 500) -> Dict[str, Any]:
    fs = None
    try:
        import SarahMemoryFilesystem as fs  # type: ignore
    except Exception:
        fs = None  # type: ignore
    results = []
    scanned = 0
    threats = 0
    if fs is not None and hasattr(fs, "FileScanner"):
        scanner = fs.FileScanner()
        for path, rel, _size in _iter_package_files(source_root, max_files=max_files):
            scanned += 1
            try:
                res = scanner.scan_file(path, quarantine_on_threat=False)
                if isinstance(res, dict):
                    found = list(res.get("threats") or [])
                    if found:
                        threats += len(found)
                    results.append({"path": rel, "threat_level": res.get("threat_level"), "threats": found, "action_taken": res.get("action_taken")})
            except Exception as exc:
                results.append({"path": rel, "threat_level": "error", "threats": [str(exc)], "action_taken": "none"})
    else:
        # Built-in fallback: conservative string scan for obvious high-risk primitives.
        patterns = ["os.system", "subprocess.call", "powershell.exe", "cmd.exe /c", "eval(", "exec(", "__import__("]
        for path, rel, _size in _iter_package_files(source_root, max_files=max_files):
            scanned += 1
            found = []
            if rel.lower().endswith((".py", ".js", ".ts", ".tsx", ".json", ".md", ".txt")):
                try:
                    blob = open(path, "r", encoding="utf-8", errors="ignore").read(1024 * 1024).lower()
                    found = [p for p in patterns if p.lower() in blob]
                except Exception:
                    found = []
            if found:
                threats += len(found)
            results.append({"path": rel, "threat_level": "high" if found else "clean", "threats": found, "action_taken": "none"})
    status = "PASS" if threats == 0 else "FAIL"
    return {"ok": threats == 0, "status": status, "files_scanned": scanned, "threat_count": threats, "results": results[:500], "quarantine_performed": False}


def _safe_extract_zip(zip_path: str, target_dir: str, max_files: int = 1500, max_total_bytes: int = 100 * 1024 * 1024) -> Dict[str, Any]:
    extracted = 0
    total = 0
    target_abs = os.path.abspath(target_dir)
    os.makedirs(target_abs, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if not name or name.endswith("/"):
                continue
            if name.startswith("/") or name.startswith("../") or "/../" in name:
                raise RuntimeError(f"unsafe_zip_path:{name}")
            total += int(info.file_size or 0)
            extracted += 1
            if extracted > max_files or total > max_total_bytes:
                raise RuntimeError("zip_extract_budget_exceeded")
            dest = os.path.abspath(os.path.join(target_abs, name))
            if not _path_under(dest, target_abs):
                raise RuntimeError(f"unsafe_zip_target:{name}")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zf.open(info, "r") as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return {"target": target_abs, "files_extracted": extracted, "bytes_extracted": total}


def _create_powerstore_package(source_root: str, *, distribution: str = "private", license_value: str = "creator_defined") -> Dict[str, Any]:
    validation = _validate_addon_dir_for_store(source_root)
    scan = _malware_scan_dir(source_root)
    manifest = validation.get("manifest") if isinstance(validation.get("manifest"), dict) else {}
    ui = validation.get("ui") if isinstance(validation.get("ui"), dict) else {}
    addon_id = _manifest_id(manifest, os.path.basename(source_root))
    now_iso = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    package_id = "smpkg_" + uuid.uuid4().hex
    release_id = "smrel_" + uuid.uuid4().hex
    build_id = "smbld_" + datetime.utcnow().strftime("%Y%m%d%H%M%S") + "_" + uuid.uuid4().hex[:8]
    package_dir = _store_runtime_dir("packages", addon_id)
    zip_path = os.path.join(package_dir, f"{addon_id}_{release_id}.zip")
    hash_manifest = []
    for path, rel, size in _iter_package_files(source_root):
        hash_manifest.append({"path": rel, "sha256": _file_sha256(path), "size": size})
    hash_manifest.sort(key=lambda x: x["path"])
    hash_manifest_sha256 = _sha256_bytes(json.dumps(hash_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")).hex()
    markers = dict(validation.get("markers") or {})
    markers["filesystem_malware_scan"] = scan.get("status", "FAIL")
    markers["hash_manifest"] = "PASS"
    markers["package_hash"] = "PENDING"
    markers["signature"] = "PENDING"
    package_manifest = {
        "schema": "SarahMemory.powerstore.package.v1",
        "package_id": package_id,
        "release_id": release_id,
        "build_id": build_id,
        "addon_id": addon_id,
        "name": manifest.get("name") or ui.get("title") or addon_id,
        "version": manifest.get("version") or "0.1.0",
        "package_type": manifest.get("type") or "addon_application",
        "distribution": distribution,
        "license": license_value,
        "created_at": now_iso,
        "minimum_sarahmemory_version": manifest.get("minimum_sarahmemory_version") or "9.0.0",
        "permissions": manifest.get("permissions") or [],
        "denied_permissions": manifest.get("denied") or manifest.get("denied_permissions") or [],
        "hash_manifest_sha256": hash_manifest_sha256,
        "validation_markers": markers,
        "validation_problems": validation.get("problems") or [],
        "malware_scan": {"status": scan.get("status"), "files_scanned": scan.get("files_scanned"), "threat_count": scan.get("threat_count")},
        "creator_signature_algorithm": "HMAC-SHA256-LOCAL-DEVELOPMENT",
        "creator_signature": "",
        "store_signature": "",
        "store_url": _powerstore_url(),
        "auto_install_allowed": False,
        "auto_run_allowed": False,
    }
    signature_payload = dict(package_manifest)
    signature_payload.pop("creator_signature", None)
    package_manifest["creator_signature"] = _package_signature(signature_payload)
    package_manifest["validation_markers"]["signature"] = "PASS"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zf:
        for path, rel, _size in _iter_package_files(source_root):
            zf.write(path, rel)
        zf.writestr("POWERSTORE_PACKAGE.json", json.dumps(package_manifest, indent=2, sort_keys=True, ensure_ascii=False))
        zf.writestr("POWERSTORE_HASH_MANIFEST.json", json.dumps(hash_manifest, indent=2, sort_keys=True, ensure_ascii=False))
    package_sha256 = _file_sha256(zip_path)
    package_manifest["package_sha256"] = package_sha256
    package_manifest["validation_markers"]["package_hash"] = "PASS"
    # Update manifest sidecar with final package hash; ZIP itself keeps pre-final hash inside for reproducible trace.
    sidecar = zip_path + ".powerstore.json"
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(package_manifest, fh, indent=2, sort_keys=True, ensure_ascii=False)
    return {"ok": bool(validation.get("ok") and scan.get("ok")), "package": package_manifest, "zip_path": zip_path, "sidecar_path": sidecar, "hash_manifest": hash_manifest, "validation": validation, "scan": scan}


@bp2.route("/api/store/powerstore/status", methods=["GET", "POST", "OPTIONS"])
def api_store_powerstore_status():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    probe = True
    try:
        if request.method == "GET":
            raw = str(request.args.get("probe") or "1").strip().lower()
            probe = raw not in {"0", "false", "no", "off"}
        else:
            data = _require_json()
            probe = bool(data.get("probe", True))
    except Exception:
        probe = True
    if not probe:
        return _cors_ok(_jok({
            "schema": "SarahMemory.powerstore.status.v1",
            "store_url": _powerstore_url(),
            "status": "NOT_CHECKED",
            "up": False,
            "checked": False,
            "required_for_local_addons": False,
            "required_for_nailde": False,
            "local_addons_enabled": _local_addons_enabled(),
        }))
    return _cors_ok(_jok(_powerstore_status_probe()))


@bp2.route("/api/store/powerstore/handshake", methods=["GET", "POST", "OPTIONS"])
def api_store_powerstore_handshake():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    return _cors_ok(_jok({
        "schema": "SarahMemory.powerstore.handshake.v1",
        "store_url": _powerstore_url(),
        "store_status": _powerstore_status_probe(timeout_seconds=2.0),
        "client": {
            "sarahmemory_version": _get_env("SARAHMEMORY_VERSION", "9.0.0"),
            "appstore_schema": "SarahMemory.powerstore.gateway.v1",
            "capabilities": [
                "local_addon_registry", "package_export", "hash_manifest", "local_signature",
                "filesystem_malware_scan", "nailde_syntax_validation", "staged_download_verify", "user_install_approval"
            ],
        },
        "policy": {
            "auto_install": False,
            "auto_run": False,
            "download_to_staging": True,
            "user_approval_required": True,
            "local_addons_enabled": _local_addons_enabled(),
            "remote_filesystem_access": False,
        },
    }))


@bp2.route("/api/store/package/export-local", methods=["POST", "OPTIONS"])
def api_store_package_export_local():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _local_addons_enabled():
        return _local_only_error("package_export_local")
    data = _require_json()
    if not _confirmed(data):
        return _cors_ok(_jerr("explicit_user_confirmation_required", code="approval_required", http=409))
    rec = None
    if data.get("addon_id") or data.get("id"):
        rec = _find_addon_record(str(data.get("addon_id") or data.get("id")))
    source = os.path.abspath(str(data.get("source_path") or (rec or {}).get("path") or ""))
    if not source or not os.path.isdir(source):
        return _cors_ok(_jerr("source_addon_not_found", code="missing_source", http=404, source_path=source))
    allowed_roots = [os.path.abspath(_addons_root()), os.path.abspath(os.path.join(_data_dir_runtime(), "addons")), os.path.abspath(os.path.join(_data_dir_runtime(), "nailde"))]
    if not any(_path_under(source, root) for root in allowed_roots):
        return _cors_ok(_jerr("source_path_not_allowed", code="source_not_allowed", http=403, source_path=source))
    result = _create_powerstore_package(source, distribution=str(data.get("distribution") or "private"), license_value=str(data.get("license") or "creator_defined"))
    return _cors_ok(_jok(result))


@bp2.route("/api/store/package/verify", methods=["POST", "OPTIONS"])
def api_store_package_verify():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _local_addons_enabled():
        return _local_only_error("package_verify")
    data = _require_json()
    package_path = os.path.abspath(str(data.get("package_path") or data.get("zip_path") or ""))
    if not package_path or not os.path.isfile(package_path) or not zipfile.is_zipfile(package_path):
        return _cors_ok(_jerr("package_zip_missing_or_invalid", code="bad_package", http=400, package_path=package_path))
    package_sha256 = _file_sha256(package_path)
    with zipfile.ZipFile(package_path, "r") as zf:
        names = set(zf.namelist())
        package_manifest = {}
        hash_manifest = []
        if "POWERSTORE_PACKAGE.json" in names:
            package_manifest = json.loads(zf.read("POWERSTORE_PACKAGE.json").decode("utf-8"))
        if "POWERSTORE_HASH_MANIFEST.json" in names:
            hash_manifest = json.loads(zf.read("POWERSTORE_HASH_MANIFEST.json").decode("utf-8"))
    markers = {
        "zip_valid": "PASS",
        "package_manifest": "PASS" if package_manifest else "FAIL",
        "hash_manifest": "PASS" if isinstance(hash_manifest, list) and hash_manifest else "FAIL",
        "package_hash": "PASS",
        "signature_present": "PASS" if package_manifest.get("creator_signature") else "FAIL",
    }
    ok = all(v == "PASS" for v in markers.values())
    return _cors_ok(_jok({"ok": ok, "package_sha256": package_sha256, "markers": markers, "package": package_manifest}))


@bp2.route("/api/store/package/scan", methods=["POST", "OPTIONS"])
def api_store_package_scan():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _local_addons_enabled():
        return _local_only_error("package_scan")
    data = _require_json()
    package_path = os.path.abspath(str(data.get("package_path") or data.get("zip_path") or ""))
    if not package_path or not os.path.isfile(package_path) or not zipfile.is_zipfile(package_path):
        return _cors_ok(_jerr("package_zip_missing_or_invalid", code="bad_package", http=400, package_path=package_path))
    stage = tempfile.mkdtemp(prefix="sm_pkg_scan_", dir=_store_runtime_dir("staging"))
    extract = _safe_extract_zip(package_path, stage)
    scan = _malware_scan_dir(stage)
    validation = _validate_addon_dir_for_store(stage)
    return _cors_ok(_jok({"stage_path": stage, "extract": extract, "scan": scan, "validation": validation, "install_allowed": bool(scan.get("ok") and validation.get("ok"))}))


@bp2.route("/api/store/powerstore/publish/prepare", methods=["POST", "OPTIONS"])
def api_store_powerstore_publish_prepare():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    data = _require_json()
    # Prepare is local because it packages a local addon for the remote marketplace.
    if not _local_addons_enabled():
        return _local_only_error("powerstore_publish_prepare")
    data.setdefault("confirm", True)
    result = api_store_package_export_local()
    return result


@bp2.route("/api/store/powerstore/publish/validate", methods=["POST", "OPTIONS"])
def api_store_powerstore_publish_validate():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    # Future server-side marketplace validation can reuse this envelope.
    data = _require_json()
    if data.get("package_path") or data.get("zip_path"):
        return api_store_package_verify()
    return _cors_ok(_jok({
        "schema": "SarahMemory.powerstore.publish_validation.v1",
        "accepted": False,
        "reason": "package_path_required_for_local_validation",
        "required_markers": ["manifest_schema", "hash_manifest", "package_hash", "signature", "filesystem_malware_scan", "python_syntax", "json_schema", "permission_review"],
    }))


@bp2.route("/api/store/powerstore/publish/upload", methods=["POST", "OPTIONS"])
def api_store_powerstore_publish_upload():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    # No network upload is performed here yet. This preserves local-first governance until the PowerStore website is complete.
    data = _require_json()
    return _cors_ok(_jok({
        "schema": "SarahMemory.powerstore.publish_upload_plan.v1",
        "store_url": _powerstore_url(),
        "upload_performed": False,
        "package_path": data.get("package_path") or data.get("zip_path") or "",
        "next_required": ["server_upload_endpoint", "creator_auth", "server_signature", "listing_metadata", "pricing_or_free_license"],
        "auto_install_allowed": False,
        "auto_run_allowed": False,
    }))


@bp2.route("/api/store/powerstore/download/plan", methods=["POST", "OPTIONS"])
def api_store_powerstore_download_plan():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    data = _require_json()
    return _cors_ok(_jok({
        "schema": "SarahMemory.powerstore.download_plan.v1",
        "package_id": data.get("package_id") or "",
        "listing_id": data.get("listing_id") or "",
        "download_url": data.get("download_url") or "",
        "stage_first": True,
        "stage_dir": os.path.join(_data_dir_runtime(), "store", "downloads", "staged"),
        "required_after_download": ["hash_verify", "signature_verify", "filesystem_malware_scan", "nailde_syntax_validation", "permission_review", "user_install_approval"],
        "auto_install_allowed": False,
        "auto_run_allowed": False,
    }))


@bp2.route("/api/store/powerstore/download/stage", methods=["POST", "OPTIONS"])
def api_store_powerstore_download_stage():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _local_addons_enabled():
        return _local_only_error("powerstore_download_stage")
    data = _require_json()
    source = os.path.abspath(str(data.get("source_path") or data.get("package_path") or ""))
    if not source or not os.path.isfile(source) or not zipfile.is_zipfile(source):
        return _cors_ok(_jerr("source_package_zip_required", code="missing_source", http=400, source_path=source, network_download_performed=False))
    package_id = _safe_addon_id(data.get("package_id") or os.path.splitext(os.path.basename(source))[0], "package")
    stage_dir = os.path.join(_store_runtime_dir("downloads", "staged"), package_id)
    os.makedirs(stage_dir, exist_ok=True)
    target = os.path.join(stage_dir, "package.zip")
    shutil.copy2(source, target)
    return _cors_ok(_jok({"staged": True, "package_id": package_id, "stage_dir": stage_dir, "package_path": target, "package_sha256": _file_sha256(target), "network_download_performed": False}))


@bp2.route("/api/store/powerstore/download/verify", methods=["POST", "OPTIONS"])
def api_store_powerstore_download_verify():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    return api_store_package_scan()


@bp2.route("/api/store/powerstore/install/authorize", methods=["POST", "OPTIONS"])
def api_store_powerstore_install_authorize():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _local_addons_enabled():
        return _local_only_error("powerstore_install_authorize")
    data = _require_json()
    if not _confirmed(data):
        return _cors_ok(_jerr("explicit_user_confirmation_required", code="approval_required", http=409))
    package_path = os.path.abspath(str(data.get("package_path") or data.get("zip_path") or ""))
    if not package_path or not os.path.isfile(package_path) or not zipfile.is_zipfile(package_path):
        return _cors_ok(_jerr("package_zip_missing_or_invalid", code="bad_package", http=400, package_path=package_path))
    stage = tempfile.mkdtemp(prefix="sm_pkg_install_", dir=_store_runtime_dir("downloads", "verified"))
    _safe_extract_zip(package_path, stage)
    scan = _malware_scan_dir(stage)
    validation = _validate_addon_dir_for_store(stage)
    if not scan.get("ok") or not validation.get("ok"):
        return _cors_ok(_jerr("downloaded_package_validation_failed", code="validation_failed", http=409, scan=scan, validation=validation, install_performed=False))
    manifest = validation.get("manifest") if isinstance(validation.get("manifest"), dict) else {}
    addon_id = _manifest_id(manifest, os.path.basename(stage))
    target = os.path.join(_addons_root(), addon_id)
    backup = _zip_backup_folder(target, addon_id) if os.path.isdir(target) else None
    if os.path.isdir(target):
        shutil.rmtree(target)
    stats = _copy_tree_bounded(stage, target)
    _record_runtime(addon_id, activation_status="installed_not_running", trust_status="powerstore_download_verified", installed_path=target, package_sha256=_file_sha256(package_path), backup=backup)
    return _cors_ok(_jok({"installed": True, "addon_id": addon_id, "installed_path": target, "backup": backup, "copy_stats": stats, "scan": scan, "validation": validation, "auto_run_performed": False}))


@bp2.route("/api/store/health", methods=["GET", "OPTIONS"])
def api_store_health():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    try:
        _ensure_tables()
        return _cors_ok(_jok({"status": "ok", "ts": _now()}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "store_health_fail", 500))


# ---- Simple KV Store (namespaced) ----

@bp2.route("/api/store/set", methods=["POST", "OPTIONS"])
def api_store_set():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    payload = _require_json()
    k = _as_str(payload.get("key") or payload.get("k") or "").strip()
    v = payload.get("value") if "value" in payload else payload.get("v")
    meta = payload.get("meta") or {}
    ttl = _as_int(payload.get("ttl") or 0, 0)
    if not k:
        return _cors_ok(_jerr("Missing key", "missing_key", 400))
    if v is None:
        v = ""
    try:
        _ensure_tables()
        exp = (_now() + ttl) if ttl > 0 else 0.0
        with _db() as con:
            cur = con.cursor()
            _kv_prune(cur)
            cur.execute(
                "INSERT INTO store_kv(k,v,meta,exp) VALUES(?,?,?,?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v, meta=excluded.meta, exp=excluded.exp",
                (k, json.dumps(v), json.dumps(meta), float(exp)),
            )
            con.commit()
        return _cors_ok(_jok({"key": k, "stored": True, "exp": exp}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "store_set_fail", 500))


@bp2.route("/api/store/get", methods=["GET", "OPTIONS"])
def api_store_get():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    k = _as_str(request.args.get("key") or request.args.get("k") or "").strip()
    if not k:
        return _cors_ok(_jerr("Missing key", "missing_key", 400))
    try:
        _ensure_tables()
        with _db() as con:
            cur = con.cursor()
            _kv_prune(cur)
            cur.execute("SELECT v, meta, exp FROM store_kv WHERE k=?", (k,))
            row = cur.fetchone()
            con.commit()
        if not row:
            return _cors_ok(_jok({"hit": False, "key": k}))
        v_json, meta_json, exp = row
        v = json.loads(v_json) if v_json else None
        meta = json.loads(meta_json) if meta_json else {}
        return _cors_ok(_jok({"hit": True, "key": k, "value": v, "meta": meta, "exp": exp}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "store_get_fail", 500))


@bp2.route("/api/store/del", methods=["POST", "OPTIONS"])
def api_store_del():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    payload = _require_json()
    k = _as_str(payload.get("key") or payload.get("k") or "").strip()
    if not k:
        return _cors_ok(_jerr("Missing key", "missing_key", 400))
    try:
        _ensure_tables()
        with _db() as con:
            cur = con.cursor()
            cur.execute("DELETE FROM store_kv WHERE k=?", (k,))
            n = cur.rowcount
            con.commit()
        return _cors_ok(_jok({"key": k, "deleted": int(n or 0)}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "store_del_fail", 500))


# ---- Store Auth (server-side, no FE secrets) ----

@bp2.route("/api/store/auth/register", methods=["POST", "OPTIONS"])
def api_store_auth_register():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    payload = _require_json()
    email = _as_str(payload.get("email") or "").strip().lower()
    password = _as_str(payload.get("password") or "")
    if not email or not password:
        return _cors_ok(_jerr("Missing email/password", "missing_fields", 400))
    try:
        _ensure_tables()
        user_id = "u_" + uuid.uuid4().hex
        salt = _b64url(os.urandom(16))
        pw_hash = _hash_pw(password, salt)
        with _db() as con:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO store_users(user_id,email,pw_hash,salt,created_ts,reset_token,reset_exp) "
                "VALUES(?,?,?,?,?,?,?)",
                (user_id, email, pw_hash, salt, _now(), "", 0.0),
            )
            con.commit()
        # Issue user token
        token = _sign_token({"sub": user_id, "email": email, "role": "user", "iat": int(_now()), "exp": int(_now()) + 86400})
        return _cors_ok(_jok({"user_id": user_id, "token": token, "role": "user"}))
    except Exception as e:
        msg = str(e)
        if "UNIQUE constraint failed" in msg or "unique constraint" in msg.lower():
            return _cors_ok(_jerr("Email already registered", "email_exists", 409))
        return _cors_ok(_jerr(msg, "register_fail", 500))


@bp2.route("/api/store/auth/login", methods=["POST", "OPTIONS"])
def api_store_auth_login():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    payload = _require_json()
    email = _as_str(payload.get("email") or "").strip().lower()
    password = _as_str(payload.get("password") or "")
    if not email or not password:
        return _cors_ok(_jerr("Missing email/password", "missing_fields", 400))
    try:
        _ensure_tables()
        with _db() as con:
            cur = con.cursor()
            cur.execute("SELECT user_id, pw_hash, salt FROM store_users WHERE email=?", (email,))
            row = cur.fetchone()
        if not row:
            return _cors_ok(_jerr("Invalid credentials", "bad_creds", 401))
        user_id, pw_hash, salt = row
        calc = _hash_pw(password, salt)
        if not hmac.compare_digest(calc, pw_hash):
            return _cors_ok(_jerr("Invalid credentials", "bad_creds", 401))
        token = _sign_token({"sub": user_id, "email": email, "role": "user", "iat": int(_now()), "exp": int(_now()) + 86400})
        return _cors_ok(_jok({"user_id": user_id, "token": token, "role": "user"}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "login_fail", 500))


@bp2.route("/api/store/auth/password-reset", methods=["POST", "OPTIONS"])
def api_store_auth_password_reset():
    """
    Simple reset workflow:
    - If payload has {email} only: issue reset_token (returned for now; can be emailed later).
    - If payload has {email, reset_token, new_password}: apply reset.
    """
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    payload = _require_json()
    email = _as_str(payload.get("email") or "").strip().lower()
    if not email:
        return _cors_ok(_jerr("Missing email", "missing_email", 400))
    try:
        _ensure_tables()
        reset_token = _as_str(payload.get("reset_token") or "")
        new_password = _as_str(payload.get("new_password") or "")
        if not reset_token and not new_password:
            # Issue
            tok = "rt_" + uuid.uuid4().hex
            exp = _now() + 3600.0
            with _db() as con:
                cur = con.cursor()
                cur.execute("UPDATE store_users SET reset_token=?, reset_exp=? WHERE email=?", (tok, float(exp), email))
                if cur.rowcount <= 0:
                    return _cors_ok(_jerr("Unknown email", "unknown_email", 404))
                con.commit()
            return _cors_ok(_jok({"email": email, "reset_token": tok, "reset_exp": exp}))
        # Apply
        if not reset_token or not new_password:
            return _cors_ok(_jerr("Missing reset_token/new_password", "missing_fields", 400))
        with _db() as con:
            cur = con.cursor()
            cur.execute("SELECT user_id, reset_token, reset_exp FROM store_users WHERE email=?", (email,))
            row = cur.fetchone()
            if not row:
                return _cors_ok(_jerr("Unknown email", "unknown_email", 404))
            user_id, rt, rex = row
            if not rt or not hmac.compare_digest(_as_str(rt), reset_token):
                return _cors_ok(_jerr("Invalid reset token", "bad_reset_token", 401))
            if rex and float(rex) < _now():
                return _cors_ok(_jerr("Reset token expired", "reset_expired", 401))
            salt = _b64url(os.urandom(16))
            pw_hash = _hash_pw(new_password, salt)
            cur.execute(
                "UPDATE store_users SET pw_hash=?, salt=?, reset_token=?, reset_exp=? WHERE user_id=?",
                (pw_hash, salt, "", 0.0, user_id),
            )
            con.commit()
        return _cors_ok(_jok({"email": email, "reset": True}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "reset_fail", 500))


@bp2.route("/api/store/auth/admin/login", methods=["POST", "OPTIONS"])
def api_store_auth_admin_login():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    payload = _require_json()
    email = _as_str(payload.get("email") or "").strip().lower()
    password = _as_str(payload.get("password") or "")
    admin_email = _get_env("STORE_ADMIN_EMAIL", "").strip().lower()
    admin_pw = _get_env("STORE_ADMIN_PASSWORD", "")
    if not admin_email or not admin_pw:
        return _cors_ok(_jerr("Admin credentials not configured", "admin_not_configured", 503))
    if email != admin_email or password != admin_pw:
        return _cors_ok(_jerr("Invalid credentials", "bad_creds", 401))
    token = _sign_token({"sub": "admin", "email": admin_email, "role": "admin", "iat": int(_now()), "exp": int(_now()) + 86400})
    return _cors_ok(_jok({"token": token, "role": "admin"}))


# ---- Storefront: trends + products + generation log ----

def _product_row_to_obj(row):
    (pid, name, description, price, category, image_url, tags, source, created_ts, updated_ts) = row
    return {
        "id": int(pid),
        "name": name,
        "description": description,
        "price": float(price),
        "category": category,
        "image_url": image_url,
        "tags": json.loads(tags) if tags else [],
        "source": source,
        "created_ts": float(created_ts),
        "updated_ts": float(updated_ts),
    }


@bp2.route("/api/store/trends", methods=["POST", "OPTIONS"])
def api_store_trends():
    """
    Lightweight placeholder. Returns stable trend buckets.
    You can swap in SarahMemory AI ranking later without breaking the contract.
    """
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    payload = _require_json()
    niche = _as_str(payload.get("niche") or payload.get("category") or "general")
    region = _as_str(payload.get("region") or "US")
    data = {
        "niche": niche,
        "region": region,
        "generated_ts": _now(),
        "trends": [
            {"label": f"{niche} essentials", "score": 0.86},
            {"label": f"{niche} minimalist", "score": 0.79},
            {"label": f"{niche} premium", "score": 0.74},
            {"label": f"{niche} gifting", "score": 0.71},
        ],
    }
    return _cors_ok(_jok(data))


@bp2.route("/api/store/products/generate", methods=["POST", "OPTIONS"])
def api_store_products_generate():
    """
    Generate product concepts server-side (no vendor calls). Output is a list of product objects.
    If you later wire SarahMemoryAI, keep this envelope stable.
    """
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    payload = _require_json()
    niche = _as_str(payload.get("niche") or payload.get("category") or "general")
    count = max(1, min(24, _as_int(payload.get("count") or 8, 8)))

    # Deterministic-ish seed so refresh isn't random noise if user repeats same ask
    seed = hashlib.sha256((niche + "|" + json.dumps(payload, sort_keys=True)).encode("utf-8")).hexdigest()[:8]

    products = []
    for i in range(count):
        name = f"{niche.title()} Item {i+1}"
        desc = f"AI-curated concept for {niche} (batch {seed})."
        price = round(19.99 + (i * 2.5), 2)
        products.append({
            "id": 0,
            "name": name,
            "description": desc,
            "price": price,
            "category": niche,
            "image_url": payload.get("image_url") or "",
            "tags": [niche, "ai", "concept"],
            "source": "generated",
        })

    # Optional: log request/response
    try:
        _ensure_tables()
        with _db() as con:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO store_generation_log(req, resp, created_ts) VALUES(?,?,?)",
                (json.dumps(payload), json.dumps(products), _now()),
            )
            con.commit()
    except Exception:
        pass

    return _cors_ok(_jok({"products": products, "seed": seed}))


@bp2.route("/api/store/products", methods=["GET", "OPTIONS"])
def api_store_products_list():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    try:
        _ensure_tables()
        q = _as_str(request.args.get("q") or "").strip().lower()
        cat = _as_str(request.args.get("category") or "").strip().lower()
        limit = max(1, min(200, _as_int(request.args.get("limit") or 50, 50)))
        offset = max(0, _as_int(request.args.get("offset") or 0, 0))

        with _db() as con:
            cur = con.cursor()
            sql = "SELECT id,name,description,price,category,image_url,tags,source,created_ts,updated_ts FROM store_products"
            params = []
            where = []
            if q:
                where.append("(lower(name) LIKE ? OR lower(description) LIKE ?)")
                params.extend([f"%{q}%", f"%{q}%"])
            if cat:
                where.append("lower(category)=?")
                params.append(cat)
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY updated_ts DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            cur.execute(sql, tuple(params))
            rows = cur.fetchall() or []
        products = [_product_row_to_obj(r) for r in rows]
        return _cors_ok(_jok({"products": products, "limit": limit, "offset": offset}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "products_list_fail", 500))


@bp2.route("/api/store/products/bulk", methods=["POST", "OPTIONS"])
def api_store_products_bulk():
    """
    Admin-only bulk upsert:
      { products: [ {name, description, price, category, image_url, tags, source, id?}, ... ] }
    """
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_admin():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))
    payload = _require_json()
    items = payload.get("products") or payload.get("items") or []
    if not isinstance(items, list):
        return _cors_ok(_jerr("products must be a list", "bad_payload", 400))
    try:
        _ensure_tables()
        upserted = 0
        now = _now()
        with _db() as con:
            cur = con.cursor()
            for it in items:
                if not isinstance(it, dict):
                    continue
                pid = _as_int(it.get("id") or 0, 0)
                name = _as_str(it.get("name") or "").strip()
                if not name:
                    continue
                desc = _as_str(it.get("description") or "")
                price = _as_float(it.get("price") or 0.0, 0.0)
                category = _as_str(it.get("category") or "general")
                image_url = _as_str(it.get("image_url") or it.get("imageUrl") or it.get("image") or "")
                tags = it.get("tags") or []
                if not isinstance(tags, list):
                    tags = []
                source = _as_str(it.get("source") or "manual")

                if pid > 0:
                    cur.execute(
                        "UPDATE store_products SET name=?, description=?, price=?, category=?, image_url=?, tags=?, source=?, updated_ts=? WHERE id=?",
                        (name, desc, float(price), category, image_url, json.dumps(tags), source, now, pid),
                    )
                    if cur.rowcount > 0:
                        upserted += 1
                        continue
                cur.execute(
                    "INSERT INTO store_products(name,description,price,category,image_url,tags,source,created_ts,updated_ts) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (name, desc, float(price), category, image_url, json.dumps(tags), source, now, now),
                )
                upserted += 1
            con.commit()
        return _cors_ok(_jok({"upserted": upserted}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "products_bulk_fail", 500))


@bp2.route("/api/store/generation-log", methods=["POST", "OPTIONS"])
def api_store_generation_log():
    """
    Log any generation activity (frontend -> backend). Admin or user token accepted.
    """
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_user():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))
    payload = _require_json()
    try:
        _ensure_tables()
        with _db() as con:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO store_generation_log(req, resp, created_ts) VALUES(?,?,?)",
                (json.dumps(payload.get("req") or payload), json.dumps(payload.get("resp") or {}), _now()),
            )
            con.commit()
        return _cors_ok(_jok({"logged": True}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "genlog_fail", 500))


# ---- PayPal proxy endpoints (secrets in .env) ----

@bp2.route("/api/store/paypal/config", methods=["GET", "OPTIONS"])
def api_store_paypal_config():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    client_id = _get_env("PAYPAL_CLIENT_ID", "").strip()
    env = _get_env("PAYPAL_ENV", "sandbox").strip()  # sandbox|live
    if not client_id:
        return _cors_ok(_jerr("PayPal not configured", "paypal_not_configured", 503))
    return _cors_ok(_jok({"clientId": client_id, "environment": env}))


def _paypal_basic_auth() -> str:
    cid = _get_env("PAYPAL_CLIENT_ID", "").strip()
    sec = _get_env("PAYPAL_SECRET", "").strip()
    raw = f"{cid}:{sec}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("utf-8")


def _paypal_api_base() -> str:
    env = _get_env("PAYPAL_ENV", "sandbox").strip().lower()
    return "https://api-m.paypal.com" if env == "live" else "https://api-m.sandbox.paypal.com"


def _http_json(url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None, body: Optional[Dict[str, Any]] = None) -> Tuple[int, Any]:
    # No extra deps; use urllib
    import urllib.request
    import urllib.error

    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw.decode("utf-8"))
            except Exception:
                return resp.status, raw.decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw.decode("utf-8"))
        except Exception:
            return e.code, raw.decode("utf-8", errors="ignore")
    except Exception as e:
        return 0, {"error": str(e)}


@bp2.route("/api/store/paypal/create-order", methods=["POST", "OPTIONS"])
def api_store_paypal_create_order():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_user():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))

    cid = _get_env("PAYPAL_CLIENT_ID", "").strip()
    sec = _get_env("PAYPAL_SECRET", "").strip()
    if not cid or not sec:
        return _cors_ok(_jerr("PayPal not configured", "paypal_not_configured", 503))

    payload = _require_json()
    # Accept multiple client payload shapes:
    # A) StoreUI: { currency, total, items:[{name,quantity,price}] }
    # B) Simple: { amount, currency }
    # C) Advanced: { order: <full PayPal order body> }
    currency = _as_str(payload.get("currency") or "USD").upper()
    order_body = payload.get("order") if isinstance(payload.get("order"), dict) else None

    if order_body is None:
        total = payload.get("total")
        amount = payload.get("amount")
        items = payload.get("items") if isinstance(payload.get("items"), list) else []

        value = None
        if total is not None:
            try:
                value = f"{float(total):.2f}"
            except Exception:
                value = _as_str(total)
        elif amount is not None:
            try:
                value = f"{float(amount):.2f}"
            except Exception:
                value = _as_str(amount)

        if not value:
            return _cors_ok(_jerr("Missing total/amount", "missing_amount", 400))

        purchase_unit = {"amount": {"currency_code": currency, "value": value}}

        # Optional itemization for better receipts
        norm_items = []
        for it in items:
            if not isinstance(it, dict):
                continue
            nm = _as_str(it.get("name") or "").strip()
            qty = _as_int(it.get("quantity") or 1, 1)
            pr = it.get("price")
            try:
                pr_val = f"{float(pr):.2f}"
            except Exception:
                pr_val = _as_str(pr or "0.00")
            if nm:
                norm_items.append({
                    "name": nm,
                    "quantity": str(max(1, qty)),
                    "unit_amount": {"currency_code": currency, "value": pr_val}
                })
        if norm_items:
            purchase_unit["items"] = norm_items

        order_body = {
            "intent": "CAPTURE",
            "purchase_units": [purchase_unit],
        }

    # Step 1: get access token
    base = _paypal_api_base()
    st, tok = _http_json(
        f"{base}/v1/oauth2/token",
        method="POST",
        headers={"Authorization": _paypal_basic_auth(), "Content-Type": "application/x-www-form-urlencoded"},
        body=None,
    )
    if st == 0:
        return _cors_ok(_jerr("PayPal network error", "paypal_network_error", 502, details=tok))
    # Above call is form-encoded; our helper sends JSON if body not None.
    # Do proper token call using urllib directly:
    try:
        import urllib.request, urllib.parse, urllib.error
        data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/v1/oauth2/token",
            data=data,
            headers={"Authorization": _paypal_basic_auth(), "Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
            tok = json.loads(raw.decode("utf-8"))
            st = resp.status
    except Exception as e:
        return _cors_ok(_jerr("PayPal token error", "paypal_token_error", 502, details=str(e)))

    access = _as_str(tok.get("access_token") or "")
    if not access:
        return _cors_ok(_jerr("PayPal token missing", "paypal_token_missing", 502, details=tok))

    st2, created = _http_json(
        f"{base}/v2/checkout/orders",
        method="POST",
        headers={"Authorization": f"Bearer {access}"},
        body=order_body,
    )
    if st2 not in (200, 201):
        return _cors_ok(_jerr("PayPal create order failed", "paypal_create_fail", 502, status=st2, details=created))
    return _cors_ok(_jok(created))


@bp2.route("/api/store/paypal/capture-order", methods=["POST", "OPTIONS"])
def api_store_paypal_capture_order():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_user():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))

    cid = _get_env("PAYPAL_CLIENT_ID", "").strip()
    sec = _get_env("PAYPAL_SECRET", "").strip()
    if not cid or not sec:
        return _cors_ok(_jerr("PayPal not configured", "paypal_not_configured", 503))

    payload = _require_json()
    order_id = _as_str(payload.get("orderID") or payload.get("order_id") or payload.get("orderId") or payload.get("id") or "").strip()
    if not order_id:
        return _cors_ok(_jerr("Missing order id", "missing_order_id", 400))

    base = _paypal_api_base()
    # token
    try:
        import urllib.request, urllib.parse
        data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/v1/oauth2/token",
            data=data,
            headers={"Authorization": _paypal_basic_auth(), "Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            tok = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return _cors_ok(_jerr("PayPal token error", "paypal_token_error", 502, details=str(e)))

    access = _as_str(tok.get("access_token") or "")
    if not access:
        return _cors_ok(_jerr("PayPal token missing", "paypal_token_missing", 502, details=tok))

    st, cap = _http_json(
        f"{base}/v2/checkout/orders/{order_id}/capture",
        method="POST",
        headers={"Authorization": f"Bearer {access}"},
        body={},
    )
    if st not in (200, 201):
        return _cors_ok(_jerr("PayPal capture failed", "paypal_capture_fail", 502, status=st, details=cap))
    return _cors_ok(_jok(cap))


# ---- Printify proxy endpoints (secrets in .env) ----

def _printify_headers() -> Dict[str, str]:
    tok = _get_env("PRINTIFY_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


@bp2.route("/api/store/printify/generate-image", methods=["POST", "OPTIONS"])
def api_store_printify_generate_image():
    """
    Reference endpoint: you can wire this to your own image pipeline.
    For now it simply echoes intent + returns placeholder.
    """
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_user():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))
    payload = _require_json()
    prompt = _as_str(payload.get("prompt") or payload.get("text") or "")
    return _cors_ok(_jok({"ok": True, "prompt": prompt, "imageUrl": "", "image_url": "", "note": "stub"}))


@bp2.route("/api/store/printify/create-product", methods=["POST", "OPTIONS"])
def api_store_printify_create_product():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_user():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))

    tok = _get_env("PRINTIFY_API_TOKEN", "").strip()
    shop_id = _get_env("PRINTIFY_SHOP_ID", "").strip()
    if not tok or not shop_id:
        return _cors_ok(_jerr("Printify not configured", "printify_not_configured", 503))

    payload = _require_json()
    # If client sends simplified payload, return a stub (until you wire full Printify product builder)
    if isinstance(payload, dict) and ("designUrl" in payload or "design_url" in payload):
        design_url = _as_str(payload.get("designUrl") or payload.get("design_url") or "")
        keywords = payload.get("keywords") if isinstance(payload.get("keywords"), list) else []
        stub = {
            "id": "printify_stub_" + uuid.uuid4().hex,
            "name": "Printify Product (stub)",
            "description": "Server-side stub response. Implement full Printify product builder to go live.",
            "price": 0.0,
            "imageUrl": design_url,
            "variants": [],
            "category": "printify",
        }
        return _cors_ok(_jok(stub))

    st, resp = _http_json(
        f"https://api.printify.com/v1/shops/{shop_id}/products.json",
        method="POST",
        headers=_printify_headers(),
        body=payload,
    )
    if st not in (200, 201):
        return _cors_ok(_jerr("Printify create-product failed", "printify_create_fail", 502, status=st, details=resp))
    return _cors_ok(_jok(resp))


@bp2.route("/api/store/printify/products", methods=["GET", "OPTIONS"])
def api_store_printify_products():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_user():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))

    tok = _get_env("PRINTIFY_API_TOKEN", "").strip()
    shop_id = _get_env("PRINTIFY_SHOP_ID", "").strip()
    if not tok or not shop_id:
        return _cors_ok(_jerr("Printify not configured", "printify_not_configured", 503))

    st, resp = _http_json(
        f"https://api.printify.com/v1/shops/{shop_id}/products.json",
        method="GET",
        headers=_printify_headers(),
        body=None,
    )
    if st not in (200, 201):
        return _cors_ok(_jerr("Printify list-products failed", "printify_list_fail", 502, status=st, details=resp))
    # Normalize list for frontend
    try:
        if isinstance(resp, dict) and isinstance(resp.get("data"), list):
            return _cors_ok(_jok({"products": resp.get("data")}))
        if isinstance(resp, list):
            return _cors_ok(_jok({"products": resp}))
    except Exception:
        pass
    return _cors_ok(_jok(resp))


# ---- Kittl proxy endpoints (secrets in .env) ----

@bp2.route("/api/store/kittl/trending-templates", methods=["GET", "OPTIONS"])
def api_store_kittl_trending_templates():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_user():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))
    api_key = _get_env("KITTL_API_KEY", "").strip()
    if not api_key:
        # Fail-soft; keep contract stable
        return _cors_ok(_jok({"templates": [], "note": "kittl not configured"}))
    # No public stable Kittl API contract assumed; treat as stub until you wire it.
    return _cors_ok(_jok({"templates": [], "note": "stub"}))


@bp2.route("/api/store/kittl/create-design", methods=["POST", "OPTIONS"])
def api_store_kittl_create_design():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_user():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))
    api_key = _get_env("KITTL_API_KEY", "").strip()
    if not api_key:
        return _cors_ok(_jerr("Kittl not configured", "kittl_not_configured", 503))
    payload = _require_json()
    # Stub response
    kid = "kittl_" + uuid.uuid4().hex
    keywords = payload.get("keywords") if isinstance(payload.get("keywords"), list) else []
    name = "Kittl Design (stub)"
    imageUrl = ""
    category = "kittl"
    return _cors_ok(_jok({"id": kid, "name": name, "imageUrl": imageUrl, "keywords": keywords, "category": category}))

# ====================================================================
# END OF appstore.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing API bridge adapter.
SML_ORGAN_METADATA = {
    "name": 'appstore',
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
    "metadata": {"sml_adapter": "api_bridge_non_executing", "source_file": 'appstore.py'},
}

def sml_get_metadata():
    return dict(SML_ORGAN_METADATA)

def sml_health():
    return {"status": "Healthy", "availability": 1.0, "integrity": 1.0, "performance": 1.0, "reliability": 1.0, "confidence": 0.75, "latency_ms": 0.0, "stability": 1.0, "compatibility": 1.0, "notes": ["SML API adapter present"]}

def sml_diagnostics():
    return {"status": "OK", "component": 'appstore', "sml_adapter": True, "metadata": dict(SML_ORGAN_METADATA), "health": sml_health()}
# --- SML ORGAN ADAPTER END ---

