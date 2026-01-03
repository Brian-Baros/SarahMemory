# --==The SarahMemory Project==--
# File: ../data/mods/v800/app_v800_drivers_api_patch.py
# Patch: Universal Driver API Router (v8.0.0)
#
# Purpose:
# - Keep mods/v800 small: ONE patch handles ALL drivers (Arduino, MIDI, PLC, etc.)
# - Drivers live under ../data/drivers/<driver_id>/ with:
#     manifest.json, ui.json, defaults.json, config.json, driver.py
# - This patch ONLY:
#     - lists/manages driver packages and configs
#     - lazy-loads driver.py on demand
#     - starts/stops driver sessions (instance IDs)
#
# Notes:
# - No driver code is imported at boot; only when endpoints are called.
# - Uses SAFE_MODE gates (if present) to prevent autoload/session start if desired.

from __future__ import annotations

import os
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from flask import request, jsonify

# Optional: SarahMemory global policy flags (SAFE_MODE, LOCAL_ONLY_MODE, etc.)
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None


# ------------------------------ Paths & Helpers ------------------------------

def _cwd() -> Path:
    try:
        return Path(os.getcwd()).resolve()
    except Exception:
        return Path(".").resolve()

def _data_dir() -> Path:
    # Prefer config.DATA_DIR if available; else ./data
    try:
        if config and hasattr(config, "DATA_DIR"):
            return Path(getattr(config, "DATA_DIR")).expanduser().resolve()
    except Exception:
        pass
    return (_cwd() / "data").resolve()

def _drivers_root() -> Path:
    return (_data_dir() / "drivers").resolve()

def _registry_root() -> Path:
    # Keep registry separate from driver folders
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

def _read_json(path: Path, default: Any = None) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def _err(msg: str, status: int = 400, details: Any = None):
    payload = {"ok": False, "error": msg}
    if details is not None:
        payload["details"] = details
    return jsonify(payload), status


# ------------------------------ Registry (enabled/autoload/trust) ------------------------------

def _load_registry() -> Dict[str, Any]:
    reg = _read_json(_drivers_registry_path(), default={})
    if not isinstance(reg, dict):
        reg = {}
    # Format:
    # {
    #   "com.softdev0.arduino.usb": {"enabled": true, "autoload": false, "trusted": false, "notes": "..."},
    #   ...
    # }
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
    # driver_id is folder name under drivers root
    return (_drivers_root() / driver_id).resolve()

def _discover_driver_ids() -> list[str]:
    root = _drivers_root()
    if not root.exists():
        return []
    out: list[str] = []
    try:
        for p in root.iterdir():
            if p.is_dir():
                if (p / "manifest.json").exists():
                    out.append(p.name)
    except Exception:
        pass
    return sorted(out)

def _load_manifest(driver_id: str) -> Dict[str, Any]:
    mpath = _driver_dir(driver_id) / "manifest.json"
    mf = _read_json(mpath, default={})
    if not isinstance(mf, dict):
        mf = {}
    # Ensure id is present
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
        # If no config, fall back to defaults (but do NOT write unless asked)
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
        spec.loader.exec_module(mod)  # type: ignore
        return mod, None
    except Exception as e:
        return None, f"Failed to import driver {driver_id}: {e}"


# ------------------------------ Sessions (runtime only) ------------------------------

# In-memory sessions: cleared on process restart (reboot/exit)
# sessions[driver_id] = {"instance_id": "...", "started_ts": ..., "meta": {...}}
_SESSIONS: Dict[str, Dict[str, Any]] = {}

def _new_instance_id(driver_id: str) -> str:
    # stable, readable instance IDs
    return f"DRV-{driver_id}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}Z-{hex(int(time.time()*1000))[-4:].upper()}"

def _session_get(driver_id: str) -> Dict[str, Any]:
    s = _SESSIONS.get(driver_id)
    return s if isinstance(s, dict) else {}

def _session_set(driver_id: str, sess: Dict[str, Any]) -> None:
    _SESSIONS[driver_id] = sess

def _session_clear(driver_id: str) -> None:
    if driver_id in _SESSIONS:
        del _SESSIONS[driver_id]


# ------------------------------ Flask Patch Apply ------------------------------

def apply(app):
    """
    Called by the v800 auto-mod loader. Registers universal driver endpoints.
    """

    # Ensure folders exist
    _drivers_root().mkdir(parents=True, exist_ok=True)
    _registry_root().mkdir(parents=True, exist_ok=True)

    # -------------------------- List & Metadata --------------------------

    @app.route("/api/drivers", methods=["GET"])
    def drivers_list():
        reg = _load_registry()
        ids = _discover_driver_ids()
        items = []
        for did in ids:
            mf = _load_manifest(did)
            r = _get_reg_entry(reg, did)
            items.append({
                "id": did,
                "manifest": mf,
                "enabled": bool(r.get("enabled", mf.get("enabled", True))),
                "autoload": bool(r.get("autoload", mf.get("autoload", False))),
                "trusted": bool(r.get("trusted", False)),
            })
        return jsonify({"ok": True, "safe_mode": _safe_mode(), "drivers": items})

    @app.route("/api/drivers/<driver_id>/schema", methods=["GET"])
    def drivers_schema(driver_id: str):
        if driver_id not in _discover_driver_ids():
            return _err("Unknown driver_id", 404)
        mf = _load_manifest(driver_id)
        ui = _load_ui_schema(driver_id)
        return jsonify({"ok": True, "manifest": mf, "ui": ui})

    # -------------------------- Config --------------------------

    @app.route("/api/drivers/<driver_id>/config", methods=["GET", "POST", "DELETE"])
    def drivers_config(driver_id: str):
        if driver_id not in _discover_driver_ids():
            return _err("Unknown driver_id", 404)

        if request.method == "GET":
            defaults = _load_defaults(driver_id)
            cfg = _load_config(driver_id)
            return jsonify({"ok": True, "config": cfg, "defaults": defaults})

        if request.method == "POST":
            body = request.get_json(force=True, silent=True) or {}
            cfg = body.get("config", body)
            if not isinstance(cfg, dict):
                return _err("config must be an object", 400)
            _save_config(driver_id, cfg)
            return jsonify({"ok": True})

        # DELETE = reset
        defaults = _reset_config(driver_id)
        return jsonify({"ok": True, "reset": True, "config": defaults})

    # -------------------------- Registry (enabled/autoload/trust) --------------------------

    @app.route("/api/drivers/registry", methods=["GET"])
    def drivers_registry_get():
        return jsonify({"ok": True, "registry": _load_registry()})

    @app.route("/api/drivers/<driver_id>/registry", methods=["POST"])
    def drivers_registry_set(driver_id: str):
        if driver_id not in _discover_driver_ids():
            return _err("Unknown driver_id", 404)
        body = request.get_json(force=True, silent=True) or {}
        patch = body.get("registry", body)
        if not isinstance(patch, dict):
            return _err("registry patch must be an object", 400)

        reg = _load_registry()
        entry = _get_reg_entry(reg, driver_id)
        # allow setting only known keys (extend later)
        for k in ("enabled", "autoload", "trusted", "notes"):
            if k in patch:
                entry[k] = patch[k]
        reg[driver_id] = entry
        _save_registry(reg)
        return jsonify({"ok": True, "driver_id": driver_id, "registry": entry})

    # -------------------------- Validate (dry run) --------------------------

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

        # driver_validate(config) is recommended. If missing, return ok.
        try:
            if hasattr(mod, "driver_validate"):
                res = mod.driver_validate(cfg)
                return jsonify(res if isinstance(res, dict) else {"ok": True, "result": res})
            return jsonify({"ok": True, "warnings": ["driver_validate not implemented"]})
        except Exception as e:
            return _err(f"validate failed: {e}", 500, details=traceback.format_exc())

    # -------------------------- Sessions --------------------------

    @app.route("/api/drivers/<driver_id>/session/start", methods=["POST"])
    def drivers_session_start(driver_id: str):
        if driver_id not in _discover_driver_ids():
            return _err("Unknown driver_id", 404)

        # Safety: block autostarts in SAFE_MODE unless explicitly allowed later
        if _safe_mode():
            return _err("SAFE_MODE active: driver session start blocked", 403)

        reg = _load_registry()
        entry = _get_reg_entry(reg, driver_id)
        enabled = bool(entry.get("enabled", True))
        if not enabled:
            return _err("Driver is disabled in registry", 403)

        body = request.get_json(force=True, silent=True) or {}
        cfg = body.get("config") or _load_config(driver_id)
        if not isinstance(cfg, dict):
            return _err("config must be an object", 400)

        mod, err = _load_driver_module(driver_id)
        if err:
            return _err(err, 500)

        # Start session
        try:
            instance_id = _new_instance_id(driver_id)
            context = {
                "via": "api",
                "driver_id": driver_id,
                "instance_id": instance_id,
                "data_dir": str(_data_dir()),
                "drivers_root": str(_drivers_root()),
                "safe_mode": _safe_mode(),
            }

            # driver_init(context, config) preferred signature
            if hasattr(mod, "driver_init"):
                out = mod.driver_init(context=context, config=cfg)  # type: ignore
            else:
                # Legacy: init not defined
                out = {"ok": True, "note": "driver_init not implemented"}

            sess = {"instance_id": instance_id, "started_ts": time.time(), "meta": out}
            _session_set(driver_id, sess)

            resp = {"ok": True, "driver_id": driver_id, "instance_id": instance_id, "result": out}
            return jsonify(resp)
        except Exception as e:
            return _err(f"session start failed: {e}", 500, details=traceback.format_exc())

    @app.route("/api/drivers/<driver_id>/session/stop", methods=["POST"])
    def drivers_session_stop(driver_id: str):
        if driver_id not in _discover_driver_ids():
            return _err("Unknown driver_id", 404)

        mod, err = _load_driver_module(driver_id)
        if err:
            return _err(err, 500)

        sess = _session_get(driver_id)
        context = {
            "via": "api",
            "driver_id": driver_id,
            "instance_id": sess.get("instance_id"),
            "safe_mode": _safe_mode(),
        }

        try:
            ok = True
            if hasattr(mod, "driver_shutdown"):
                ok = bool(mod.driver_shutdown(context=context))  # type: ignore
            _session_clear(driver_id)  # release instance id on stop
            return jsonify({"ok": ok, "driver_id": driver_id, "stopped": True})
        except Exception as e:
            return _err(f"session stop failed: {e}", 500, details=traceback.format_exc())

    @app.route("/api/drivers/<driver_id>/status", methods=["GET"])
    def drivers_status(driver_id: str):
        if driver_id not in _discover_driver_ids():
            return _err("Unknown driver_id", 404)

        sess = _session_get(driver_id)
        mod, err = _load_driver_module(driver_id)
        if err:
            # still return session state if import fails
            return jsonify({"ok": False, "error": err, "session": sess}), 500

        context = {
            "via": "api",
            "driver_id": driver_id,
            "instance_id": sess.get("instance_id"),
            "safe_mode": _safe_mode(),
        }

        try:
            if hasattr(mod, "driver_status"):
                st = mod.driver_status(context=context)  # type: ignore
                return jsonify({"ok": True, "session": sess, "status": st})
            return jsonify({"ok": True, "session": sess, "status": {"ok": True, "note": "driver_status not implemented"}})
        except Exception as e:
            return _err(f"status failed: {e}", 500, details=traceback.format_exc())

    # -------------------------- Generic Action Bus --------------------------

    @app.route("/api/drivers/<driver_id>/actions/<action_id>", methods=["POST"])
    def drivers_action(driver_id: str, action_id: str):
        if driver_id not in _discover_driver_ids():
            return _err("Unknown driver_id", 404)

        # If SAFE_MODE, you can still allow harmless actions like scan/validate.
        # Drivers can self-enforce inside driver_action as well.
        body = request.get_json(force=True, silent=True) or {}
        payload = body.get("payload", body)

        sess = _session_get(driver_id)
        mod, err = _load_driver_module(driver_id)
        if err:
            return _err(err, 500)

        context = {
            "via": "api",
            "driver_id": driver_id,
            "instance_id": sess.get("instance_id"),
            "safe_mode": _safe_mode(),
        }

        try:
            # Preferred: driver_action(action_id, context, payload)
            if hasattr(mod, "driver_action"):
                out = mod.driver_action(action_id=action_id, context=context, payload=payload)  # type: ignore
                return jsonify(out if isinstance(out, dict) else {"ok": True, "result": out})

            # Fallback: function named action_<id>
            fn_name = f"action_{action_id}"
            if hasattr(mod, fn_name):
                fn = getattr(mod, fn_name)
                out = fn(context=context, payload=payload)  # type: ignore
                return jsonify(out if isinstance(out, dict) else {"ok": True, "result": out})

            return _err(f"Action '{action_id}' not implemented by driver", 404)

        except Exception as e:
            return _err(f"action failed: {e}", 500, details=traceback.format_exc())

    return app
