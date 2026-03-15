""" --==The SarahMemory Project==--
# File: /app/server/appdrivers.py
# Purpose: Universal Driver API Router (v8.0.0)
# Part of the SarahMemory Companion AI-bot Platform
# Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
# www.linkedin.com/in/brian-baros-29962a176
# https://www.facebook.com/bbaros
# brian.baros@sarahmemory.com
# 'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
# https://www.sarahmemory.com
# https://api.sarahmemory.com
# https://ai.sarahmemory.com
# https://store.sarahmemory.com
#
# Purpose:
# - Handles governed driver discovery, configuration, validation, connection,
#   status, and generic action routing for attached or currently unattached hardware.
# - Drivers live under ../data/drivers/<driver_id>/ with:
#     manifest.json, ui.json, defaults.json, config.json, driver.py
# - No driver code is imported at boot unless an endpoint needs it.
# - SAFE_MODE blocks live connection/session start while still allowing discovery,
#   schema reads, config management, and validation.
# - Supports both legacy patch-style apply(app) mounting and modern init_app(...)
#   mounting from app.py.
#==================================================================================================
"""
from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "api_bridge"
# CATEGORY = "external_driver_and_operations"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "api_server"
# API_DOMAIN = "drivers"
# HARDWARE_DOMAIN = "serial_usb_network_gpio_plc_midi_generic"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "driver_router"
# FAMILY = "core_drivers"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = True
# NOTES = "Universal governed driver surface for discovery, config, validation, session control, and action routing for hardware integrations under /api/drivers/*."
# --- SARAHMETA END ---

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from flask import jsonify, request

# Optional: SarahMemory global policy flags (SAFE_MODE, LOCAL_ONLY_MODE, etc.)
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None

# Optional auth/sign helpers injected by app.py
_CONNECT_SQLITE: Optional[Callable[..., Any]] = None
_META_DB: Optional[str] = None
_API_KEY_AUTH_OK: Optional[Callable[[], bool]] = None
_SIGN_OK: Optional[Callable[[bytes, str], bool]] = None
_ROUTES_REGISTERED = False


# ------------------------------ Paths & Helpers ------------------------------

def _cwd() -> Path:
    try:
        return Path(os.getcwd()).resolve()
    except Exception:
        return Path(".").resolve()


def _data_dir() -> Path:
    try:
        if config and hasattr(config, "DATA_DIR"):
            return Path(getattr(config, "DATA_DIR")).expanduser().resolve()
    except Exception:
        pass
    return (_cwd() / "data").resolve()


def _drivers_root() -> Path:
    return (_data_dir() / "drivers").resolve()


def _registry_root() -> Path:
    return (_data_dir() / "registry").resolve()


def _drivers_registry_path() -> Path:
    return (_registry_root() / "drivers.json").resolve()


def _safe_mode() -> bool:
    try:
        if config and hasattr(config, "SAFE_MODE"):
            return bool(getattr(config, "SAFE_MODE"))
    except Exception:
        pass
    return False


def _local_only_mode() -> bool:
    try:
        if config and hasattr(config, "LOCAL_ONLY_MODE"):
            return bool(getattr(config, "LOCAL_ONLY_MODE"))
    except Exception:
        pass
    return False


def _neoskymatrix() -> bool:
    try:
        if config and hasattr(config, "NEOSKYMATRIX"):
            return bool(getattr(config, "NEOSKYMATRIX"))
    except Exception:
        pass
    return False


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _request_body() -> bytes:
    try:
        return request.get_data(cache=True) or b""
    except Exception:
        return b""


def _verify_auth() -> bool:
    sig = str(request.headers.get("X-Sarah-Signature") or "").strip()
    if sig and callable(_SIGN_OK):
        try:
            return bool(_SIGN_OK(_request_body(), sig))
        except Exception:
            return False
    if callable(_API_KEY_AUTH_OK):
        try:
            return bool(_API_KEY_AUTH_OK())
        except Exception:
            return False
    return True


def _ok(**payload: Any):
    data = {"ok": True}
    data.update(payload)
    return jsonify(data), 200


def _err(msg: str, status: int = 400, details: Any = None, **extra: Any):
    payload = {"ok": False, "error": msg}
    if details is not None:
        payload["details"] = details
    if extra:
        payload.update(extra)
    return jsonify(payload), status


def _connect_sqlite(path: str):
    if callable(_CONNECT_SQLITE):
        return _CONNECT_SQLITE(path)
    import sqlite3

    con = sqlite3.connect(path, timeout=5.0)
    con.row_factory = sqlite3.Row
    return con


def _meta_db_path() -> str:
    if _META_DB:
        return _META_DB
    return str((_data_dir() / "meta.db").resolve())


def _ensure_meta_tables() -> None:
    con = None
    try:
        con = _connect_sqlite(_meta_db_path())
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS driver_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                driver_id TEXT,
                action TEXT,
                status TEXT,
                details_json TEXT
            )
            """
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


def _log_event(driver_id: str, action: str, status: str, details: Optional[Dict[str, Any]] = None) -> None:
    con = None
    try:
        _ensure_meta_tables()
        con = _connect_sqlite(_meta_db_path())
        cur = con.cursor()
        cur.execute(
            "INSERT INTO driver_events (ts, driver_id, action, status, details_json) VALUES (?, ?, ?, ?, ?)",
            (float(time.time()), str(driver_id or ""), str(action or ""), str(status or ""), json.dumps(details or {}, ensure_ascii=False)),
        )
        con.commit()
    except Exception:
        pass
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


# ------------------------------ Registry (enabled/autoload/trust) ------------------------------

def _load_registry() -> Dict[str, Any]:
    reg = _read_json(_drivers_registry_path(), default={})
    if not isinstance(reg, dict):
        reg = {}
    return reg


def _save_registry(reg: Dict[str, Any]) -> None:
    _write_json(_drivers_registry_path(), reg)


def _get_reg_entry(reg: Dict[str, Any], driver_id: str) -> Dict[str, Any]:
    entry = reg.get(driver_id)
    if not isinstance(entry, dict):
        entry = {}
    return entry


# ------------------------------ Driver Discovery ------------------------------

def _driver_dir(driver_id: str) -> Path:
    return (_drivers_root() / driver_id).resolve()


def _discover_driver_ids() -> list[str]:
    root = _drivers_root()
    if not root.exists():
        return []
    out: list[str] = []
    try:
        for p in root.iterdir():
            if p.is_dir() and (p / "manifest.json").exists():
                out.append(p.name)
    except Exception:
        pass
    return sorted(out)


def _load_manifest(driver_id: str) -> Dict[str, Any]:
    mpath = _driver_dir(driver_id) / "manifest.json"
    mf = _read_json(mpath, default={})
    if not isinstance(mf, dict):
        mf = {}
    if "id" not in mf:
        mf["id"] = driver_id
    return mf


def _load_ui_schema(driver_id: str) -> Dict[str, Any]:
    upath = _driver_dir(driver_id) / "ui.json"
    ui = _read_json(upath, default={})
    return ui if isinstance(ui, dict) else {}


def _load_defaults(driver_id: str) -> Dict[str, Any]:
    dpath = _driver_dir(driver_id) / "defaults.json"
    d = _read_json(dpath, default={})
    return d if isinstance(d, dict) else {}


def _load_config(driver_id: str) -> Dict[str, Any]:
    cpath = _driver_dir(driver_id) / "config.json"
    c = _read_json(cpath, default=None)
    if c is None:
        return _load_defaults(driver_id)
    return c if isinstance(c, dict) else _load_defaults(driver_id)


def _save_config(driver_id: str, cfg: Dict[str, Any]) -> None:
    cpath = _driver_dir(driver_id) / "config.json"
    _write_json(cpath, cfg)


def _reset_config(driver_id: str) -> Dict[str, Any]:
    defaults = _load_defaults(driver_id)
    _save_config(driver_id, defaults)
    return defaults


# ------------------------------ Lazy Import Driver Module ------------------------------

def _load_driver_module(driver_id: str) -> Tuple[Optional[Any], Optional[str]]:
    ddir = _driver_dir(driver_id)
    py = ddir / "driver.py"
    if not py.exists():
        return None, f"driver.py not found for {driver_id}"

    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(f"sm_driver_{driver_id}", str(py))
        if spec is None or spec.loader is None:
            return None, f"Unable to create import spec for {driver_id}"
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
        return mod, None
    except Exception as e:
        return None, f"Failed to import driver {driver_id}: {e}"


# ------------------------------ Sessions (runtime only) ------------------------------

_SESSIONS: Dict[str, Dict[str, Any]] = {}


def _new_instance_id(driver_id: str) -> str:
    return f"DRV-{driver_id}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}Z-{hex(int(time.time()*1000))[-4:].upper()}"


def _session_get(driver_id: str) -> Dict[str, Any]:
    s = _SESSIONS.get(driver_id)
    return s if isinstance(s, dict) else {}


def _session_set(driver_id: str, sess: Dict[str, Any]) -> None:
    _SESSIONS[driver_id] = sess


def _session_clear(driver_id: str) -> None:
    if driver_id in _SESSIONS:
        del _SESSIONS[driver_id]


def _build_driver_context(driver_id: str, instance_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = {
        "via": "api",
        "driver_id": driver_id,
        "instance_id": instance_id,
        "data_dir": str(_data_dir()),
        "drivers_root": str(_drivers_root()),
        "safe_mode": _safe_mode(),
        "local_only_mode": _local_only_mode(),
        "neoskymatrix": _neoskymatrix(),
        "request_ip": str(getattr(request, "remote_addr", "") or ""),
        "request_method": str(getattr(request, "method", "") or ""),
    }
    if isinstance(extra, dict) and extra:
        ctx.update(extra)
    return ctx


def _driver_connect(driver_id: str, cfg: Dict[str, Any], connect_payload: Optional[Dict[str, Any]] = None):
    if driver_id not in _discover_driver_ids():
        return _err("Unknown driver_id", 404)
    if _safe_mode():
        return _err("SAFE_MODE active: driver session start blocked", 403)

    reg = _load_registry()
    entry = _get_reg_entry(reg, driver_id)
    enabled = bool(entry.get("enabled", _load_manifest(driver_id).get("enabled", True)))
    if not enabled:
        return _err("Driver is disabled in registry", 403)

    mod, err = _load_driver_module(driver_id)
    if err:
        _log_event(driver_id, "connect", "error", {"error": err})
        return _err(err, 500)

    try:
        instance_id = _new_instance_id(driver_id)
        context = _build_driver_context(driver_id, instance_id=instance_id, extra={"connect_payload": connect_payload or {}})
        if hasattr(mod, "driver_connect"):
            out = mod.driver_connect(context=context, config=cfg, payload=connect_payload or {})  # type: ignore[attr-defined]
        elif hasattr(mod, "driver_init"):
            out = mod.driver_init(context=context, config=cfg)  # type: ignore[attr-defined]
        else:
            out = {"ok": True, "note": "driver_connect/driver_init not implemented"}

        sess = {
            "instance_id": instance_id,
            "started_ts": time.time(),
            "meta": out,
            "config": cfg,
            "connected": bool((out or {}).get("ok", True)) if isinstance(out, dict) else True,
        }
        _session_set(driver_id, sess)
        _log_event(driver_id, "connect", "ok", {"instance_id": instance_id})
        return _ok(driver_id=driver_id, instance_id=instance_id, session=sess, result=out)
    except Exception as e:
        _log_event(driver_id, "connect", "error", {"error": str(e)})
        return _err(f"session start failed: {e}", 500, details=traceback.format_exc())


def _driver_disconnect(driver_id: str):
    if driver_id not in _discover_driver_ids():
        return _err("Unknown driver_id", 404)

    mod, err = _load_driver_module(driver_id)
    if err:
        return _err(err, 500)

    sess = _session_get(driver_id)
    context = _build_driver_context(driver_id, instance_id=sess.get("instance_id"))

    try:
        ok = True
        if hasattr(mod, "driver_disconnect"):
            ok = bool(mod.driver_disconnect(context=context))  # type: ignore[attr-defined]
        elif hasattr(mod, "driver_shutdown"):
            ok = bool(mod.driver_shutdown(context=context))  # type: ignore[attr-defined]
        _session_clear(driver_id)
        _log_event(driver_id, "disconnect", "ok", {"instance_id": sess.get("instance_id")})
        return _ok(driver_id=driver_id, stopped=True, disconnected=True)
    except Exception as e:
        _log_event(driver_id, "disconnect", "error", {"error": str(e)})
        return _err(f"session stop failed: {e}", 500, details=traceback.format_exc())


def _driver_discover(driver_id: str, payload: Optional[Dict[str, Any]] = None):
    if driver_id not in _discover_driver_ids():
        return _err("Unknown driver_id", 404)

    manifest = _load_manifest(driver_id)
    config_data = _load_config(driver_id)
    reg = _load_registry()
    entry = _get_reg_entry(reg, driver_id)
    mod, err = _load_driver_module(driver_id)

    base = {
        "ok": True,
        "driver_id": driver_id,
        "manifest": manifest,
        "enabled": bool(entry.get("enabled", manifest.get("enabled", True))),
        "trusted": bool(entry.get("trusted", False)),
        "autoload": bool(entry.get("autoload", manifest.get("autoload", False))),
        "connected": bool(_session_get(driver_id)),
        "config": config_data,
    }

    if err:
        base["module_status"] = {"ok": False, "error": err}
        return jsonify(base), 200

    try:
        context = _build_driver_context(driver_id, instance_id=_session_get(driver_id).get("instance_id"), extra={"discover_payload": payload or {}})
        if hasattr(mod, "driver_discover"):
            out = mod.driver_discover(context=context, config=config_data, payload=payload or {})  # type: ignore[attr-defined]
        elif hasattr(mod, "driver_scan"):
            out = mod.driver_scan(context=context, config=config_data, payload=payload or {})  # type: ignore[attr-defined]
        else:
            out = {
                "ok": True,
                "note": "driver_discover not implemented",
                "expected_targets": manifest.get("targets", []),
                "transport": manifest.get("transport") or manifest.get("protocol") or "generic",
            }
        base["discovery"] = out
        return jsonify(base), 200
    except Exception as e:
        return _err(f"discover failed: {e}", 500, details=traceback.format_exc())


# ------------------------------ Flask Mount ------------------------------

def apply(app):
    global _ROUTES_REGISTERED

    if _ROUTES_REGISTERED:
        return app

    _drivers_root().mkdir(parents=True, exist_ok=True)
    _registry_root().mkdir(parents=True, exist_ok=True)
    _ensure_meta_tables()

    @app.route("/api/drivers/capabilities", methods=["GET"])
    def drivers_capabilities():
        return _ok(
            api_domain="drivers",
            routes_base="/api/drivers",
            safe_mode=_safe_mode(),
            local_only_mode=_local_only_mode(),
            neoskymatrix=_neoskymatrix(),
            supports=[
                "list",
                "schema",
                "config",
                "registry",
                "validate",
                "discover",
                "connect",
                "disconnect",
                "status",
                "actions",
            ],
        )

    @app.route("/api/drivers", methods=["GET"])
    def drivers_list():
        reg = _load_registry()
        ids = _discover_driver_ids()
        items = []
        for did in ids:
            mf = _load_manifest(did)
            r = _get_reg_entry(reg, did)
            sess = _session_get(did)
            items.append({
                "id": did,
                "manifest": mf,
                "enabled": bool(r.get("enabled", mf.get("enabled", True))),
                "autoload": bool(r.get("autoload", mf.get("autoload", False))),
                "trusted": bool(r.get("trusted", False)),
                "connected": bool(sess),
                "instance_id": sess.get("instance_id"),
            })
        return jsonify({"ok": True, "safe_mode": _safe_mode(), "drivers": items})

    @app.route("/api/drivers/<driver_id>/schema", methods=["GET"])
    def drivers_schema(driver_id: str):
        if driver_id not in _discover_driver_ids():
            return _err("Unknown driver_id", 404)
        mf = _load_manifest(driver_id)
        ui = _load_ui_schema(driver_id)
        return jsonify({"ok": True, "manifest": mf, "ui": ui})

    @app.route("/api/drivers/<driver_id>/config", methods=["GET", "POST", "DELETE"])
    def drivers_config(driver_id: str):
        if driver_id not in _discover_driver_ids():
            return _err("Unknown driver_id", 404)

        if request.method == "GET":
            defaults = _load_defaults(driver_id)
            cfg = _load_config(driver_id)
            return jsonify({"ok": True, "config": cfg, "defaults": defaults})

        if not _verify_auth():
            return _err("Unauthorized", 401)

        if request.method == "POST":
            body = request.get_json(force=True, silent=True) or {}
            cfg = body.get("config", body)
            if not isinstance(cfg, dict):
                return _err("config must be an object", 400)
            _save_config(driver_id, cfg)
            _log_event(driver_id, "config_update", "ok", {"keys": sorted(list(cfg.keys()))[:50]})
            return jsonify({"ok": True})

        defaults = _reset_config(driver_id)
        _log_event(driver_id, "config_reset", "ok", {})
        return jsonify({"ok": True, "reset": True, "config": defaults})

    @app.route("/api/drivers/registry", methods=["GET"])
    def drivers_registry_get():
        return jsonify({"ok": True, "registry": _load_registry()})

    @app.route("/api/drivers/<driver_id>/registry", methods=["POST"])
    def drivers_registry_set(driver_id: str):
        if driver_id not in _discover_driver_ids():
            return _err("Unknown driver_id", 404)
        if not _verify_auth():
            return _err("Unauthorized", 401)

        body = request.get_json(force=True, silent=True) or {}
        patch = body.get("registry", body)
        if not isinstance(patch, dict):
            return _err("registry patch must be an object", 400)

        reg = _load_registry()
        entry = _get_reg_entry(reg, driver_id)
        for k in ("enabled", "autoload", "trusted", "notes"):
            if k in patch:
                entry[k] = patch[k]
        reg[driver_id] = entry
        _save_registry(reg)
        _log_event(driver_id, "registry_update", "ok", entry)
        return jsonify({"ok": True, "driver_id": driver_id, "registry": entry})

    @app.route("/api/drivers/<driver_id>/validate", methods=["POST"])
    def drivers_validate(driver_id: str):
        if driver_id not in _discover_driver_ids():
            return _err("Unknown driver_id", 404)

        body = request.get_json(force=True, silent=True) or {}
        cfg = body.get("config") or _load_config(driver_id)
        if not isinstance(cfg, dict):
            return _err("config must be an object", 400)

        mod, err = _load_driver_module(driver_id)
        if err:
            return _err(err, 500)

        try:
            context = _build_driver_context(driver_id, instance_id=_session_get(driver_id).get("instance_id"), extra={"validate_payload": body})
            if hasattr(mod, "driver_validate"):
                res = mod.driver_validate(context=context, config=cfg, payload=body)  # type: ignore[attr-defined]
                return jsonify(res if isinstance(res, dict) else {"ok": True, "result": res})
            return jsonify({"ok": True, "warnings": ["driver_validate not implemented"]})
        except Exception as e:
            return _err(f"validate failed: {e}", 500, details=traceback.format_exc())

    @app.route("/api/drivers/<driver_id>/discover", methods=["GET", "POST"])
    def drivers_discover(driver_id: str):
        payload = {}
        if request.method == "POST":
            payload = request.get_json(force=True, silent=True) or {}
        return _driver_discover(driver_id, payload=payload)

    @app.route("/api/drivers/<driver_id>/connect", methods=["POST"])
    def drivers_connect(driver_id: str):
        body = request.get_json(force=True, silent=True) or {}
        cfg = body.get("config") or _load_config(driver_id)
        if not isinstance(cfg, dict):
            return _err("config must be an object", 400)
        if not _verify_auth():
            return _err("Unauthorized", 401)
        return _driver_connect(driver_id, cfg=cfg, connect_payload=body.get("payload") or body)

    @app.route("/api/drivers/<driver_id>/disconnect", methods=["POST"])
    def drivers_disconnect(driver_id: str):
        if not _verify_auth():
            return _err("Unauthorized", 401)
        return _driver_disconnect(driver_id)

    @app.route("/api/drivers/<driver_id>/session/start", methods=["POST"])
    def drivers_session_start(driver_id: str):
        body = request.get_json(force=True, silent=True) or {}
        cfg = body.get("config") or _load_config(driver_id)
        if not isinstance(cfg, dict):
            return _err("config must be an object", 400)
        if not _verify_auth():
            return _err("Unauthorized", 401)
        return _driver_connect(driver_id, cfg=cfg, connect_payload=body.get("payload") or body)

    @app.route("/api/drivers/<driver_id>/session/stop", methods=["POST"])
    def drivers_session_stop(driver_id: str):
        if not _verify_auth():
            return _err("Unauthorized", 401)
        return _driver_disconnect(driver_id)

    @app.route("/api/drivers/<driver_id>/status", methods=["GET"])
    def drivers_status(driver_id: str):
        if driver_id not in _discover_driver_ids():
            return _err("Unknown driver_id", 404)

        sess = _session_get(driver_id)
        mod, err = _load_driver_module(driver_id)
        if err:
            return jsonify({"ok": False, "error": err, "session": sess}), 500

        context = _build_driver_context(driver_id, instance_id=sess.get("instance_id"))

        try:
            if hasattr(mod, "driver_status"):
                st = mod.driver_status(context=context)  # type: ignore[attr-defined]
                return jsonify({"ok": True, "session": sess, "status": st})
            return jsonify({"ok": True, "session": sess, "status": {"ok": True, "note": "driver_status not implemented"}})
        except Exception as e:
            return _err(f"status failed: {e}", 500, details=traceback.format_exc())

    @app.route("/api/drivers/<driver_id>/actions/<action_id>", methods=["POST"])
    def drivers_action(driver_id: str, action_id: str):
        if driver_id not in _discover_driver_ids():
            return _err("Unknown driver_id", 404)

        body = request.get_json(force=True, silent=True) or {}
        payload = body.get("payload", body)

        sess = _session_get(driver_id)
        mod, err = _load_driver_module(driver_id)
        if err:
            return _err(err, 500)

        context = _build_driver_context(driver_id, instance_id=sess.get("instance_id"), extra={"action_id": action_id})

        try:
            if hasattr(mod, "driver_action"):
                out = mod.driver_action(action_id=action_id, context=context, payload=payload)  # type: ignore[attr-defined]
                return jsonify(out if isinstance(out, dict) else {"ok": True, "result": out})

            fn_name = f"action_{action_id}"
            if hasattr(mod, fn_name):
                fn = getattr(mod, fn_name)
                out = fn(context=context, payload=payload)  # type: ignore[misc]
                return jsonify(out if isinstance(out, dict) else {"ok": True, "result": out})

            return _err(f"Action '{action_id}' not implemented by driver", 404)
        except Exception as e:
            return _err(f"action failed: {e}", 500, details=traceback.format_exc())

    _ROUTES_REGISTERED = True
    return app


# ------------------------------ app.py Integration ------------------------------

def init_app(app, connect_sqlite=None, meta_db_path=None, api_key_auth_ok=None, sign_ok=None):
    global _CONNECT_SQLITE, _META_DB, _API_KEY_AUTH_OK, _SIGN_OK

    _CONNECT_SQLITE = connect_sqlite
    _META_DB = meta_db_path
    _API_KEY_AUTH_OK = api_key_auth_ok
    _SIGN_OK = sign_ok
    return apply(app)
