"""--==The SarahMemory Project==--
File: api/server/appdrivers.py
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

Purpose: Universal Driver API Router (v9.0.0)

- Handles governed driver discovery, configuration, validation, connection,
status, and generic action routing for attached or currently unattached hardware.
- Drivers live under ../data/drivers/<driver_id>/ with:
manifest.json, ui.json, defaults.json, config.json, driver.py
- No driver code is imported at boot unless an endpoint needs it.
- SAFE_MODE blocks live connection/session start while still allowing discovery,
schema reads, config management, and validation.
- Supports both legacy patch-style apply(app) mounting and modern init_app(...)
mounting from app.py.
==================================================================================================
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
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-06-06"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Universal governed driver surface for discovery, config, validation, session control, and action routing for hardware integrations under /api/drivers/*."
# --- SARAHMETA END ---

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from flask import jsonify, request

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
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "core" / "SarahMemoryGlobals.py").exists() or (parent / "SarahMemoryGlobals.py").exists():
                return parent.resolve()
    except Exception:
        pass
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


def _settings_root() -> Path:
    """Runtime driver registry state belongs under data/settings."""
    try:
        if config and hasattr(config, "SETTINGS_DIR"):
            return Path(getattr(config, "SETTINGS_DIR")).expanduser().resolve()
    except Exception:
        pass
    return (_data_dir() / "settings").resolve()


def _registry_root() -> Path:
    """Legacy registry directory retained only for one-time migration fallback."""
    return (_data_dir() / "registry").resolve()


def _migrate_legacy_json_once(primary: Path, legacy: Path) -> Path:
    """Copy legacy registry JSON into data/settings once; future writes stay primary."""
    try:
        if (not primary.exists()) and legacy.exists() and legacy.is_file():
            primary.parent.mkdir(parents=True, exist_ok=True)
            primary.write_text(legacy.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass
    return primary


def _drivers_registry_path() -> Path:
    return _migrate_legacy_json_once(
        (_settings_root() / "drivers.json").resolve(),
        (_registry_root() / "drivers.json").resolve(),
    )


def _boot_root() -> Path:
    return (_data_dir() / "boot").resolve()


def _boot_drivers_root() -> Path:
    return (_boot_root() / "drivers").resolve()


def _boot_registry_path() -> Path:
    return (_boot_root() / "boot_drivers.json").resolve()


def _is_boot_driver_id(driver_id: str) -> bool:
    return str(driver_id or "").startswith("com.softdev0.boot.")


def _boot_registry_blob() -> Dict[str, Any]:
    blob = _read_json(_boot_registry_path(), default={})
    return blob if isinstance(blob, dict) else {}


def _boot_registry_entries() -> Dict[str, Dict[str, Any]]:
    blob = _boot_registry_blob()
    entries = blob.get("drivers", []) if isinstance(blob, dict) else []
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(entries, list):
        for item in entries:
            if isinstance(item, dict) and item.get("id"):
                out[str(item["id"])] = item
    return out


def _boot_driver_order() -> list[str]:
    entries = _boot_registry_entries()
    if not entries:
        return []

    graph: Dict[str, set[str]] = {}
    indegree: Dict[str, int] = {}
    levels: Dict[str, int] = {}
    for did, meta in entries.items():
        deps = meta.get("dependencies", [])
        deps = deps if isinstance(deps, list) else []
        graph[did] = set(str(x) for x in deps if str(x) in entries and str(x) != did)
        indegree.setdefault(did, 0)
        levels[did] = int(meta.get("level", meta.get("load_priority", 999)) or 999)

    for did, deps in graph.items():
        for dep in deps:
            indegree[did] = indegree.get(did, 0) + 1
            indegree.setdefault(dep, 0)

    ready = sorted([did for did, deg in indegree.items() if deg == 0], key=lambda x: (levels.get(x, 999), x))
    ordered: list[str] = []
    while ready:
        did = ready.pop(0)
        ordered.append(did)
        for other, deps in graph.items():
            if did in deps:
                deps.remove(did)
                indegree[other] -= 1
                if indegree[other] == 0:
                    ready.append(other)
                    ready.sort(key=lambda x: (levels.get(x, 999), x))

    remaining = [did for did in entries.keys() if did not in ordered]
    remaining.sort(key=lambda x: (levels.get(x, 999), x))
    ordered.extend(remaining)
    return ordered


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
    # Developer fallback: allow only loopback when no auth verifier was injected.
    try:
        remote = str(getattr(request, "remote_addr", "") or "").strip().lower()
        return remote in ("", "127.0.0.1", "::1", "localhost")
    except Exception:
        return False

def _user_confirmed(payload: Optional[Dict[str, Any]] = None) -> bool:
    payload = payload if isinstance(payload, dict) else {}
    for key in ("confirm", "confirmed", "user_confirmed", "user_authorized", "approved", "explicit_user_approval"):
        value = payload.get(key)
        if value is True:
            return True
        if isinstance(value, str) and value.strip().lower() in ("1", "true", "yes", "on", "approved", "confirm", "confirmed", "user_approved"):
            return True
    return str(payload.get("confirm_phrase") or "").strip().upper() in {"I APPROVE", "USER APPROVED", "CONFIRM ACTION", "APPROVE GOVERNED ACTION"}


def _operator_authorize_driver_action(driver_id: str, action_id: str, payload: Any, *, user_consented: bool) -> Dict[str, Any]:
    """Route state-changing driver intent through OperatorCore before domain dispatch."""
    try:
        import SarahMemoryOperatorCore as _OperatorCore  # type: ignore
        fn = getattr(_OperatorCore, "authorize_external_domain_action", None)
        if not callable(fn):
            raise RuntimeError("OperatorCore authorization surface unavailable")
        pa = {
            "action_type": "driver_action",
            "capability_name": "device_control",
            "target": str(driver_id),
            "target_ref": str(driver_id),
            "driver_id": str(driver_id),
            "driver_action_id": str(action_id),
            "required_permissions": ["Execute", "DeviceControl"],
            "risk_level": "TIER_3_PHYSICAL_OR_DEVICE",
            "reason": "User-requested governed driver action through appdrivers domain owner.",
            "rollback_plan": "Driver/domain safe-stop or device-specific rollback when supported; otherwise verify failure and stop.",
            "tests": ["driver_manifest_present", "energetics_preflight", "post_dispatch_result_verification"],
            "dry_run": False,
            "touches_filesystem": False,
            "physical_actuation": True,
            "payload_summary": payload if isinstance(payload, dict) else {"value": str(payload)[:500]},
        }
        return fn(
            f"Execute driver action {action_id} on {driver_id}",
            origin="api.server.appdrivers",
            proposed_action=pa,
            meta={"api_domain": "drivers", "driver_id": str(driver_id), "action_id": str(action_id)},
            user_consented=bool(user_consented),
        )
    except Exception as exc:
        return {"ok": False, "allow": False, "decision": "DEFER", "domain_execution_authorized": False, "reason": f"operatorcore_unavailable:{exc}", "execution_authority": False}


def _operator_record_driver_result(auth: Dict[str, Any], result: Dict[str, Any], *, verified: bool) -> Dict[str, Any]:
    try:
        import SarahMemoryOperatorCore as _OperatorCore  # type: ignore
        fn = getattr(_OperatorCore, "record_external_domain_action_result", None)
        if callable(fn):
            return fn(auth, result, verified=bool(verified), origin="api.server.appdrivers")
    except Exception:
        pass
    return {"ok": False, "verified": False, "execution_authority": False}


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
    return str((_data_dir() / "memory" / "datasets" / "meta.db").resolve())


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
    if _is_boot_driver_id(driver_id):
        return (_boot_drivers_root() / driver_id).resolve()
    return (_drivers_root() / driver_id).resolve()


def _discover_driver_ids() -> list[str]:
    runtime_ids: list[str] = []
    boot_ids: list[str] = []

    root = _drivers_root()
    if root.exists():
        try:
            for p in root.iterdir():
                if p.is_dir() and (p / "manifest.json").exists():
                    runtime_ids.append(p.name)
        except Exception:
            pass

    boot_root = _boot_drivers_root()
    if boot_root.exists():
        try:
            for p in boot_root.iterdir():
                if p.is_dir() and (p / "manifest.json").exists():
                    boot_ids.append(p.name)
        except Exception:
            pass

    ordered_boot = [did for did in _boot_driver_order() if did in set(boot_ids)]
    unordered_boot = sorted([did for did in boot_ids if did not in set(ordered_boot)])
    return ordered_boot + unordered_boot + sorted(runtime_ids)


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



def _validate_driver_manifest_shape(driver_id: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Validate generated driver manifest contract without importing driver.py."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(manifest, dict):
        return {"ok": False, "errors": ["manifest_not_object"], "warnings": warnings}

    declared_id = str(manifest.get("id") or driver_id or "").strip()
    if not declared_id:
        errors.append("missing_id")
    elif declared_id != driver_id:
        warnings.append("manifest_id_differs_from_folder")

    dangerous_autoload = bool(manifest.get("autoload", False))
    if dangerous_autoload:
        warnings.append("autoload_requested_requires_registry_and_user_consent")

    for key in ("name", "version", "description"):
        if not str(manifest.get(key) or "").strip():
            warnings.append(f"missing_optional_{key}")

    if manifest.get("permissions") is not None and not isinstance(manifest.get("permissions"), (list, dict)):
        errors.append("permissions_must_be_list_or_object")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "driver_id": driver_id,
        "declared_id": declared_id,
        "autoload_requested": dangerous_autoload,
        "lazy_load_required": True,
        "safe_reset_supported": True,
        "direct_boot_activation_allowed": False,
    }

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


def _driver_energetics_preflight(driver_id: str, action_id: str = "connect", requested_power_mode: str = "ACTIVE", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Ask Energetics before driver connection/action escalation.

    Driver API does not directly mutate power states. This verdict only permits,
    defers, or mode-reduces the request before driver code is loaded/executed.

    Industrial safety rule: driver paths are hazardous-energy-sensitive because a
    driver may later reach real sensors, relays, serial buses, servos, motors, or
    vehicle/drone hardware. Therefore missing/corrupted Energetics never fails open.
    """
    ctx = {"action_type": action_id, "driver_id": driver_id, "payload": payload or {}, "scheduled": False, "source": "appdrivers"}
    try:
        blocker_fn = getattr(config, "sm_hazardous_energy_blocks_action", None)
        status_fn = getattr(config, "sm_hazardous_energy_status", None)
        if callable(blocker_fn) and blocker_fn("driver_action", ctx):
            status = status_fn(ctx) if callable(status_fn) else {}
            return {
                "ok": False,
                "decision": "DENY",
                "reason": "Hazardous-energy constitution blocks driver connection/action until preflight clears.",
                "allowed_power_mode": "LOW_POWER",
                "constitution": status,
                "source": "appdrivers._driver_energetics_preflight",
            }
    except Exception as exc:
        return {"ok": False, "decision": "DENY", "reason": f"Driver hazardous-energy constitution check failed closed: {exc}", "allowed_power_mode": "LOW_POWER"}
    try:
        import SarahMemoryEnergetics as _Energetics  # type: ignore
        fn = getattr(_Energetics, "recommend_device_power_mode", None)
        if callable(fn):
            return fn(
                device_type=f"driver:{driver_id}",
                requested_power_mode=requested_power_mode,
                context=ctx,
            )
    except Exception as exc:
        return {"ok": False, "decision": "DENY", "reason": f"Energetics driver bridge failed closed: {exc}", "allowed_power_mode": "LOW_POWER"}
    return {"ok": False, "decision": "DENY", "reason": "Energetics bridge unavailable; driver action fails closed.", "allowed_power_mode": "LOW_POWER"}


def _driver_connect(driver_id: str, cfg: Dict[str, Any], connect_payload: Optional[Dict[str, Any]] = None):
    if driver_id not in _discover_driver_ids():
        return _err("Unknown driver_id", 404)
    if _safe_mode() and not _is_boot_driver_id(driver_id):
        return _err("SAFE_MODE active: driver session start blocked", 403)

    _energy_verdict = _driver_energetics_preflight(driver_id, "connect", "ACTIVE", connect_payload or {})
    if str(_energy_verdict.get("decision") or "ALLOW").upper() in {"DENY", "DEFER", "REDUCE_MODE"} or not bool(_energy_verdict.get("ok", True)):
        return _err("Driver session start blocked by hazardous-energy / Energetics preflight", 409, details=_energy_verdict)

    reg = _load_registry()
    entry = _get_reg_entry(reg, driver_id)
    manifest = _load_manifest(driver_id)
    boot_meta = _boot_registry_entries().get(driver_id, {}) if _is_boot_driver_id(driver_id) else {}
    enabled = bool(entry.get("enabled", manifest.get("enabled", True)))
    if _is_boot_driver_id(driver_id):
        enabled = bool(boot_meta.get("enabled", enabled))
    if not enabled:
        return _err("Driver is disabled in registry", 403)

    dependencies = boot_meta.get("dependencies", manifest.get("dependencies", [])) if _is_boot_driver_id(driver_id) else manifest.get("dependencies", [])
    dependencies = dependencies if isinstance(dependencies, list) else []
    unmet = []
    for dep in dependencies:
        dep_id = str(dep)
        dep_sess = _session_get(dep_id)
        dep_ready = bool(dep_sess) and bool(dep_sess.get("connected"))
        dep_meta = dep_sess.get("meta") if isinstance(dep_sess, dict) else {}
        if isinstance(dep_meta, dict):
            dep_ready = dep_ready and bool(dep_meta.get("ready", dep_meta.get("ok", True)))
        if not dep_ready:
            unmet.append(dep_id)
    if unmet:
        return _err("Driver dependencies not ready", 409, details={"driver_id": driver_id, "unmet_dependencies": unmet})

    mod, err = _load_driver_module(driver_id)
    if err:
        _log_event(driver_id, "connect", "error", {"error": err})
        return _err(err, 500)

    try:
        instance_id = _new_instance_id(driver_id)
        context = _build_driver_context(driver_id, instance_id=instance_id, extra={"connect_payload": connect_payload or {}, "energetics": _energy_verdict})
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


# ------------------------------ Governed Chat Device Actions ------------------------------
# B06: appdrivers owns bounded local-device execution. app.py may detect a
# chat route, but appdrivers performs governance, dispatch, verification, and
# OperatorCore audit for local keyboard/RGB/device actions. This keeps driver
# authority out of the central chat bridge.

_KEY_LOCK_ACTION_ID = "system_keyboard_lock_state"
_KEY_LOCK_DRIVER_ID = "system.keyboard.local_os"


def _json_plain(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
        return value
    except Exception:
        try:
            if isinstance(value, dict):
                return {str(k): _json_plain(v) for k, v in value.items()}
            if isinstance(value, (list, tuple, set)):
                return [_json_plain(v) for v in list(value)]
            return str(value)
        except Exception:
            return "<unserializable>"


def _normalize_chat_device_route(text: str = "", route: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a bounded local-device route inferred from SML/app.py route data.

    This helper is intentionally small. It does not become an infinite device
    intent table. It recognizes only the stable low-level route families B06 is
    allowed to dispatch, then leaves all unknown device work to normal driver
    discovery/actions.
    """
    route = dict(route or {}) if isinstance(route, dict) else {}
    t = str(text or route.get("text") or "").strip().lower()
    t = t.replace("capslock", "caps lock").replace("numlock", "num lock").replace("scrolllock", "scroll lock")
    route_id = str(route.get("route_id") or "").strip()

    if route_id == "system.keyboard.key_state" or (
        any(k in t for k in ("caps lock", "num lock", "scroll lock"))
        and any(k in t for k in ("turn", "put", "set", "enable", "disable", "switch"))
    ):
        key_name = str(route.get("key_name") or "").strip().lower()
        if not key_name:
            key_name = "caps_lock" if "caps lock" in t else ("num_lock" if "num lock" in t else "scroll_lock")
        key_name = key_name.replace(" ", "_").replace("capslock", "caps_lock").replace("numlock", "num_lock").replace("scrolllock", "scroll_lock")
        requested_state = str(route.get("requested_state") or "").strip().lower()
        if requested_state not in {"on", "off"}:
            requested_state = "off" if any(k in t for k in ("turn off", "switch off", "disable", "off")) else "on"
        return {
            "ok": True,
            "route_id": "system.keyboard.key_state",
            "action_family": "keyboard_lock_state",
            "driver_id": _KEY_LOCK_DRIVER_ID,
            "action_id": _KEY_LOCK_ACTION_ID,
            "key_name": key_name,
            "requested_state": requested_state,
            "risk_level": "TIER_1_HARMLESS_LOCAL_UI",
            "requires_confirmation": True,
        }

    if route_id == "drivers.keyboard.lighting" or (
        "keyboard" in t and any(k in t for k in ("light", "lights", "led", "rgb", "backlight", "color", "colors", "colour", "colours"))
    ):
        color = str(route.get("value") or "").strip().lower()
        if not color or color == "requested":
            for c in ("red", "green", "blue", "purple", "yellow", "white", "orange", "pink"):
                if c in t:
                    color = c
                    break
        requested_state = str(route.get("requested_state") or "").strip().lower() or ("on" if any(k in t for k in ("turn on", "enable", "activate")) else "")
        return {
            "ok": True,
            "route_id": "drivers.keyboard.lighting",
            "action_family": "keyboard_lighting",
            "driver_id": "dynamic.driver.catalog",
            "action_id": "keyboard_lighting_set",
            "device_type": "keyboard",
            "value": color or "requested",
            "requested_state": requested_state or None,
            "risk_level": "TIER_2_BOUNDED_LOCAL_OPERATION",
            "requires_confirmation": True,
        }

    return {"ok": False, "error": "unsupported_or_unknown_device_intent", "route_id": route_id or "unknown"}


def _authorize_bounded_local_device_action(
    *,
    action_type: str,
    capability_name: str,
    target: str,
    action_id: str,
    payload: Dict[str, Any],
    user_goal: str,
    risk_level: str,
    user_consented: bool,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Authorize a bounded appdrivers-owned action through OperatorCore.

    Uses a non-hazardous action_type for PC keyboard lock-state changes so the
    hazardous-energy constitution does not falsely classify Caps Lock as robot/
    motor/driver actuation. Higher-risk driver actions still use the generic
    driver action path above.
    """
    try:
        import SarahMemoryOperatorCore as _OperatorCore  # type: ignore
        fn = getattr(_OperatorCore, "authorize_external_domain_action", None)
        if not callable(fn):
            raise RuntimeError("OperatorCore authorization surface unavailable")
        proposed_action = {
            "action_type": str(action_type),
            "capability_name": str(capability_name),
            "target": str(target),
            "target_ref": str(target),
            "driver_id": str(target),
            "driver_action_id": str(action_id),
            "required_permissions": ["Execute", "DeviceControl"],
            "risk_level": str(risk_level or "TIER_2_BOUNDED_LOCAL_OPERATION"),
            "reason": "User-requested bounded local-device action through appdrivers domain owner.",
            "rollback_plan": "Verify final device state when possible; if requested state is not reached, report failure and stop. Manual reversal remains available for keyboard lock state.",
            "tests": ["operatorcore_authorization", "bounded_dispatch", "post_dispatch_state_verification", "audit_record"],
            "dry_run": False,
            "touches_filesystem": False,
            "touches_network": False,
            "physical_actuation": False,
            "payload_summary": _json_plain(payload),
        }
        return fn(
            str(user_goal or f"Execute {action_id} on {target}"),
            origin="api.server.appdrivers.chat_device_action",
            proposed_action=proposed_action,
            meta={"api_domain": "drivers", "bounded_local_device_action": True, **dict(meta or {})},
            user_consented=bool(user_consented),
        )
    except Exception as exc:
        return {"ok": False, "allow": False, "decision": "DEFER", "domain_execution_authorized": False, "execution_authority": False, "reason": f"operatorcore_unavailable:{exc}"}


def _keyboard_lock_nice_name(key_name: str) -> str:
    return {"caps_lock": "Caps Lock", "num_lock": "Num Lock", "scroll_lock": "Scroll Lock"}.get(str(key_name), str(key_name))


def _set_keyboard_lock_state_native(key_name: str, requested_state: str) -> Dict[str, Any]:
    """Bounded OS adapter for Caps/Num/Scroll Lock. Windows-only for now.

    This is the execution primitive owned by appdrivers, not app.py. It verifies
    final state and fails closed on unsupported hosts instead of producing LLM
    advice.
    """
    key_name = str(key_name or "caps_lock").strip().lower().replace(" ", "_")
    requested_state = str(requested_state or "on").strip().lower()
    vk_map = {"caps_lock": 0x14, "num_lock": 0x90, "scroll_lock": 0x91}
    if key_name not in vk_map:
        return {"ok": False, "verified": False, "error": "unsupported_keyboard_lock_key", "key_name": key_name, "requested_state": requested_state}
    nice = _keyboard_lock_nice_name(key_name)
    desired_on = requested_state != "off"
    if os.name != "nt":
        return {
            "ok": False,
            "verified": False,
            "error": "keyboard_lock_state_not_supported_on_this_os",
            "presentation_text": f"{nice} control is governed, but this runtime does not currently expose a verified OS adapter for {os.name}.",
            "key_name": key_name,
            "requested_state": requested_state,
            "os": os.name,
            "action_taken": False,
        }
    try:
        import ctypes
        import time as _time
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        vk = vk_map[key_name]
        key_event_flags_extended = 0x0001
        key_event_flags_keyup = 0x0002
        changed = False
        for _ in range(4):
            current_on = bool(user32.GetKeyState(vk) & 1)
            if current_on == desired_on:
                break
            user32.keybd_event(vk, 0x45, key_event_flags_extended, 0)
            user32.keybd_event(vk, 0x45, key_event_flags_extended | key_event_flags_keyup, 0)
            changed = True
            _time.sleep(0.05)
        final_on = bool(user32.GetKeyState(vk) & 1)
        verified = final_on == desired_on
        final_state = "on" if final_on else "off"
        if verified:
            verb = "turned" if changed else "already"
            text = f"{nice} {verb} {final_state.upper()}."
        else:
            text = f"Unable to verify {nice} reached the requested {requested_state.upper()} state."
        return {
            "ok": bool(verified),
            "verified": bool(verified),
            "presentation_text": text,
            "key_name": key_name,
            "requested_state": requested_state,
            "final_state": final_state,
            "changed": bool(changed),
            "action_taken": bool(changed),
            "os": "nt",
        }
    except Exception as exc:
        return {"ok": False, "verified": False, "error": f"keyboard_lock_state_failed:{exc}", "key_name": key_name, "requested_state": requested_state, "action_taken": False}


def _find_keyboard_lighting_driver_candidates() -> list[Dict[str, Any]]:
    candidates: list[Dict[str, Any]] = []
    try:
        for driver_id in _discover_driver_ids():
            try:
                manifest = _load_manifest(driver_id)
            except Exception:
                manifest = {}
            raw = json.dumps(manifest or {}, ensure_ascii=False).lower()
            actions = manifest.get("actions_supported") if isinstance(manifest, dict) else []
            actions = actions if isinstance(actions, list) else []
            action_blob = " ".join(str(x).lower() for x in actions)
            if any(x in raw for x in ("keyboard", "rgb", "lighting", "backlight", "led")) and any(x in raw + " " + action_blob for x in ("rgb", "lighting", "backlight", "set_color", "led")):
                candidates.append({"driver_id": str(driver_id), "manifest": manifest, "actions_supported": actions})
    except Exception:
        return candidates
    return candidates


def _execute_keyboard_lighting_via_governed_driver(route: Dict[str, Any], *, user_consented: bool, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Try a governed runtime driver for keyboard lighting.

    This deliberately does not spawn arbitrary OpenRGB commands from app.py or
    appdrivers. Lighting succeeds only when a registered SarahMemory driver
    exposes an appropriate action.
    """
    candidates = _find_keyboard_lighting_driver_candidates()
    if not candidates:
        return {
            "ok": False,
            "verified": False,
            "decision": "DRIVER_UNAVAILABLE",
            "presentation_text": "Keyboard lighting control is governed, but no registered keyboard/RGB lighting driver is currently available.",
            "driver_candidates": [],
            "action_taken": False,
        }
    preferred_actions = ["keyboard_lighting_set", "keyboard_rgb_set", "set_color", "set_rgb", "set_led_color", "set_backlight", "lighting_set"]
    payload = {
        "source": "appdrivers.chat_device_action",
        "device_type": "keyboard",
        "value": route.get("value"),
        "requested_state": route.get("requested_state"),
        "meta": dict(meta or {}),
    }
    errors = []
    for candidate in candidates:
        driver_id = str(candidate.get("driver_id") or "")
        manifest = candidate.get("manifest") if isinstance(candidate.get("manifest"), dict) else {}
        actions_blob = json.dumps(manifest or {}, ensure_ascii=False).lower()
        selected_action = ""
        for action_id in preferred_actions:
            if action_id.lower() in actions_blob:
                selected_action = action_id
                break
        if not selected_action:
            continue
        auth = _operator_authorize_driver_action(driver_id, selected_action, payload, user_consented=user_consented)
        if not bool(auth.get("domain_execution_authorized")):
            errors.append({"driver_id": driver_id, "action_id": selected_action, "authorization": auth})
            continue
        try:
            cfg = _load_config(driver_id)
            _driver_connect(driver_id, cfg=cfg if isinstance(cfg, dict) else {}, connect_payload={"source": "appdrivers.chat_device_action", "autoconnect_for_action": True})
            mod, err = _load_driver_module(driver_id)
            if err or mod is None:
                failure = {"ok": False, "error": err or "driver_module_unavailable"}
                audit = _operator_record_driver_result(auth, failure, verified=False)
                errors.append({"driver_id": driver_id, "action_id": selected_action, "result": failure, "operator_audit": audit})
                continue
            context = _build_driver_context(driver_id, instance_id=_session_get(driver_id).get("instance_id"), extra={"action_id": selected_action, "operator_contract": auth.get("contract")})
            if hasattr(mod, "driver_action"):
                out = mod.driver_action(action_id=selected_action, context=context, payload=payload)  # type: ignore[attr-defined]
            else:
                fn_name = f"action_{selected_action}"
                if not hasattr(mod, fn_name):
                    continue
                out = getattr(mod, fn_name)(context=context, payload=payload)  # type: ignore[misc]
            result = out if isinstance(out, dict) else {"ok": True, "result": out}
            verified = bool(result.get("ok", True)) and not bool(result.get("error"))
            audit = _operator_record_driver_result(auth, result, verified=verified)
            return {
                "ok": bool(verified),
                "verified": bool(verified),
                "decision": "EXECUTED" if verified else "FAILED_VERIFICATION",
                "presentation_text": f"Keyboard lighting action {'completed' if verified else 'failed verification'} through governed driver {driver_id}.",
                "driver_id": driver_id,
                "action_id": selected_action,
                "driver_result": _json_plain(result),
                "operator_audit": audit,
                "action_taken": bool(verified),
            }
        except Exception as exc:
            errors.append({"driver_id": driver_id, "action_id": selected_action, "error": str(exc), "traceback": traceback.format_exc()})
    return {
        "ok": False,
        "verified": False,
        "decision": "NO_DRIVER_ACTION_SUCCEEDED",
        "presentation_text": "Keyboard lighting matched a governed device route, but no registered driver action completed and verified.",
        "driver_candidates": [{"driver_id": c.get("driver_id")} for c in candidates],
        "errors": errors[-5:],
        "action_taken": False,
    }


def run_governed_device_intent(
    text: str = "",
    *,
    route: Optional[Dict[str, Any]] = None,
    user_consented: bool = False,
    source: str = "api_chat",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Public in-process B06 bridge used by app.py chat routing.

    Returns a plain dict. It never asks a local model to explain how to perform
    device control and never lets app.py execute the action directly.
    """
    meta = dict(meta or {}) if isinstance(meta, dict) else {}
    normalized = _normalize_chat_device_route(text, route)
    if not normalized.get("ok"):
        return {"ok": False, "handled": False, "decision": "UNSUPPORTED_DEVICE_INTENT", "error": normalized.get("error"), "route": normalized}
    if not user_consented:
        return {
            "ok": False,
            "handled": True,
            "blocked": True,
            "decision": "REQUIRE_USER",
            "requires_confirmation": True,
            "presentation_text": "This local device action requires explicit user confirmation before appdrivers can execute it.",
            "route": normalized,
            "source": source,
        }

    if normalized.get("action_family") == "keyboard_lock_state":
        payload = {"route": normalized, "user_text": str(text or ""), "source": source, "meta": meta}
        auth = _authorize_bounded_local_device_action(
            action_type="local_os_keyboard_lock_state",
            capability_name="pc_keyboard_lock_state",
            target=_KEY_LOCK_DRIVER_ID,
            action_id=_KEY_LOCK_ACTION_ID,
            payload=payload,
            user_goal=str(text or f"Set {_keyboard_lock_nice_name(str(normalized.get('key_name')))} {normalized.get('requested_state')}"),
            risk_level=str(normalized.get("risk_level") or "TIER_1_HARMLESS_LOCAL_UI"),
            user_consented=True,
            meta={"route_id": normalized.get("route_id"), **meta},
        )
        if not bool(auth.get("domain_execution_authorized")):
            return {
                "ok": False,
                "handled": True,
                "blocked": True,
                "decision": str(auth.get("decision") or "DEFER"),
                "presentation_text": "Keyboard lock-state action was blocked by OperatorCore governance.",
                "authorization": auth,
                "route": normalized,
                "action_taken": False,
            }
        result = _set_keyboard_lock_state_native(str(normalized.get("key_name") or "caps_lock"), str(normalized.get("requested_state") or "on"))
        audit = _operator_record_driver_result(auth, result, verified=bool(result.get("verified")))
        result.update({
            "handled": True,
            "decision": "EXECUTED" if result.get("verified") else "FAILED_VERIFICATION",
            "authorization": auth,
            "operator_audit": audit,
            "route": normalized,
            "source": source,
        })
        result.setdefault("presentation_text", "Keyboard lock-state action completed." if result.get("ok") else "Keyboard lock-state action did not verify.")
        return _json_plain(result)

    if normalized.get("action_family") == "keyboard_lighting":
        result = _execute_keyboard_lighting_via_governed_driver(normalized, user_consented=True, meta={"source": source, **meta})
        result.update({"handled": True, "route": normalized, "source": source})
        return _json_plain(result)

    return {"ok": False, "handled": False, "decision": "UNSUPPORTED_DEVICE_ACTION_FAMILY", "route": normalized}


# ------------------------------ Flask Mount ------------------------------

def apply(app):
    global _ROUTES_REGISTERED

    if _ROUTES_REGISTERED:
        return app

    _drivers_root().mkdir(parents=True, exist_ok=True)
    _boot_drivers_root().mkdir(parents=True, exist_ok=True)
    _registry_root().mkdir(parents=True, exist_ok=True)
    _boot_root().mkdir(parents=True, exist_ok=True)
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
                "governed_chat_device_action",
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
            boot_meta = _boot_registry_entries().get(did, {}) if _is_boot_driver_id(did) else {}
            items.append({
                "id": did,
                "manifest": mf,
                "enabled": bool(r.get("enabled", mf.get("enabled", True))),
                "autoload": bool(r.get("autoload", mf.get("autoload", False))),
                "trusted": bool(r.get("trusted", False)),
                "connected": bool(sess),
                "instance_id": sess.get("instance_id"),
                "level": boot_meta.get("level", mf.get("level")),
                "dependencies": boot_meta.get("dependencies", mf.get("dependencies", [])),
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



    @app.route("/api/drivers/governance", methods=["GET"])
    def drivers_governance():
        return _ok(
            api_domain="drivers",
            route_base="/api/drivers",
            governance={
                "lazy_load_only": True,
                "direct_boot_activation_allowed": False,
                "safe_mode": _safe_mode(),
                "local_only_mode": _local_only_mode(),
                "driver_code_imported_on_list": False,
                "connect_requires_auth": True,
                "registry_required_for_autoload": True,
                "generated_driver_policy": {
                    "manifest_required": True,
                    "validate_before_connect": True,
                    "explicit_user_or_developer_approval_required": True,
                    "safe_stop_required": True,
                },
            },
            active_sessions=len(_SESSIONS),
        )

    @app.route("/api/drivers/manifest/audit", methods=["GET"])
    def drivers_manifest_audit():
        reg = _load_registry()
        items = []
        for did in _discover_driver_ids():
            mf = _load_manifest(did)
            entry = _get_reg_entry(reg, did)
            audit = _validate_driver_manifest_shape(did, mf)
            items.append({
                "driver_id": did,
                "manifest": mf,
                "registry": entry,
                "audit": audit,
                "connected": bool(_session_get(did)),
            })
        return jsonify({"ok": True, "count": len(items), "drivers": items})

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
        for k in ("enabled", "autoload", "trusted", "notes", "manufacturer", "driver_signature", "signature_type", "trust_level", "source", "hash", "level", "load_priority", "dependencies"):
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
        if not _verify_auth():
            return _err("Unauthorized", 401)

        body = request.get_json(force=True, silent=True) or {}
        cfg = body.get("config") or _load_config(driver_id)
        if not isinstance(cfg, dict):
            return _err("config must be an object", 400)

        manifest = _load_manifest(driver_id)
        manifest_audit = _validate_driver_manifest_shape(driver_id, manifest)
        dry_run = bool(body.get("dry_run", body.get("dryRun", False)))

        if dry_run:
            return jsonify({
                "ok": bool(manifest_audit.get("ok", False)),
                "dry_run": True,
                "driver_id": driver_id,
                "manifest_audit": manifest_audit,
                "config_keys": sorted(list(cfg.keys()))[:100],
                "would_import_driver_module": False,
                "would_connect": False,
            })

        mod, err = _load_driver_module(driver_id)
        if err:
            return _err(err, 500, details={"manifest_audit": manifest_audit})

        try:
            context = _build_driver_context(driver_id, instance_id=_session_get(driver_id).get("instance_id"), extra={"validate_payload": body, "manifest_audit": manifest_audit})
            if hasattr(mod, "driver_validate"):
                res = mod.driver_validate(context=context, config=cfg, payload=body)  # type: ignore[attr-defined]
                if isinstance(res, dict):
                    res.setdefault("manifest_audit", manifest_audit)
                    return jsonify(res)
                return jsonify({"ok": True, "result": res, "manifest_audit": manifest_audit})
            return jsonify({"ok": bool(manifest_audit.get("ok", False)), "warnings": ["driver_validate not implemented"], "manifest_audit": manifest_audit})
        except Exception as e:
            return _err(f"validate failed: {e}", 500, details={"traceback": traceback.format_exc(), "manifest_audit": manifest_audit})

    @app.route("/api/drivers/<driver_id>/discover", methods=["GET", "POST"])
    def drivers_discover(driver_id: str):
        payload = {}
        if request.method == "POST":
            if not _verify_auth():
                return _err("Unauthorized", 401)
            payload = request.get_json(force=True, silent=True) or {}
        return _driver_discover(driver_id, payload=payload)

    @app.route("/api/drivers/<driver_id>/connect", methods=["POST"])
    def drivers_connect(driver_id: str):
        body = request.get_json(force=True, silent=True) or {}
        if not _user_confirmed(body):
            return _err("Driver session start requires explicit user confirmation", 403, decision="REQUIRE_USER", source="appdrivers.governance")
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


    @app.route("/api/drivers/actions/governed", methods=["POST"])
    def drivers_governed_action():
        if not _verify_auth():
            return _err("Unauthorized", 401)
        body = request.get_json(force=True, silent=True) or {}
        text = str(body.get("text") or body.get("user_text") or "")
        route = body.get("route") if isinstance(body.get("route"), dict) else {}
        result = run_governed_device_intent(
            text,
            route=route,
            user_consented=_user_confirmed(body),
            source="api.drivers.actions.governed",
            meta={"http_route": "/api/drivers/actions/governed"},
        )
        status = 200 if bool(result.get("ok")) else (403 if str(result.get("decision") or "") == "REQUIRE_USER" else 409)
        return jsonify(result), status

    @app.route("/api/drivers/<driver_id>/actions/<action_id>", methods=["POST"])
    def drivers_action(driver_id: str, action_id: str):
        if driver_id not in _discover_driver_ids():
            return _err("Unknown driver_id", 404)
        if not _verify_auth():
            return _err("Unauthorized", 401)

        body = request.get_json(force=True, silent=True) or {}
        if not _user_confirmed(body):
            return _err("Driver action requires explicit user confirmation", 403, decision="REQUIRE_USER", source="appdrivers.governance")
        payload = body.get("payload", body)

        _energy_verdict = _driver_energetics_preflight(driver_id, action_id, "FULL_POWER" if str(action_id).lower() in {"scan", "start", "move", "drive", "fly", "lift", "connect"} else "ACTIVE", payload if isinstance(payload, dict) else {"payload": str(payload)[:500]})
        if str(_energy_verdict.get("decision") or "ALLOW").upper() in {"DENY", "DEFER", "REDUCE_MODE"} or not bool(_energy_verdict.get("ok", True)):
            return _err("Driver action blocked by hazardous-energy / Energetics preflight", 409, details=_energy_verdict)

        authorization = _operator_authorize_driver_action(driver_id, action_id, payload, user_consented=True)
        if not bool(authorization.get("domain_execution_authorized")):
            return _err(
                "Driver action blocked by OperatorCore governance/security/assurance gates",
                409,
                details=authorization,
                decision=str(authorization.get("decision") or "DEFER"),
                source="OperatorCore",
            )

        sess = _session_get(driver_id)
        mod, err = _load_driver_module(driver_id)
        if err:
            audit = _operator_record_driver_result(authorization, {"ok": False, "error": err}, verified=False)
            return _err(err, 500, operator_audit=audit)

        context = _build_driver_context(driver_id, instance_id=sess.get("instance_id"), extra={"action_id": action_id, "energetics": _energy_verdict, "operator_contract": authorization.get("contract")})

        try:
            if hasattr(mod, "driver_action"):
                out = mod.driver_action(action_id=action_id, context=context, payload=payload)  # type: ignore[attr-defined]
            else:
                fn_name = f"action_{action_id}"
                if not hasattr(mod, fn_name):
                    audit = _operator_record_driver_result(authorization, {"ok": False, "error": "action_not_implemented"}, verified=False)
                    return _err(f"Action '{action_id}' not implemented by driver", 404, operator_audit=audit)
                fn = getattr(mod, fn_name)
                out = fn(context=context, payload=payload)  # type: ignore[misc]

            result = out if isinstance(out, dict) else {"ok": True, "result": out}
            verified = bool(result.get("ok", True)) and not bool(result.get("error"))
            result["operator_audit"] = _operator_record_driver_result(authorization, result, verified=verified)
            result["operator_contract_id"] = str((authorization.get("contract") or {}).get("contract_id") or "")
            return jsonify(result), 200 if verified else 500
        except Exception as e:
            failure = {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
            audit = _operator_record_driver_result(authorization, failure, verified=False)
            return _err(f"action failed: {e}", 500, details=failure["traceback"], operator_audit=audit)


    # Register emergency contract validation through the same apply(app) bridge path.
    # This avoids a module-import failure from an undefined Blueprint while preserving
    # the existing endpoint contract and governance behavior.
    app.add_url_rule(
        "/api/drivers/emergency/contract/validate",
        "api_drivers_emergency_contract_validate",
        api_drivers_emergency_contract_validate,
        methods=["POST"],
    )

    _ROUTES_REGISTERED = True
    return app


# ------------------------------ app.py Integration ------------------------------



# =============================================================================
# SM V8.0 Cognitive Instinct Driver Contract Bridge
# =============================================================================
# Driver bridge validates emergency action contracts for future robot bodies.
# It does not execute hardware actions by itself.
# =============================================================================

def api_drivers_emergency_contract_validate():
    if not _verify_auth():
        return _err("unauthorized", 401)
    data = request.get_json(silent=True) or {}
    try:
        import SarahMemoryCognitiveServices as _CogServices  # type: ignore
        result = _CogServices.evaluate_emergency_instinct(data, caller="appdrivers.emergency_contract_validate")
        contract = result.get("action_contract") if isinstance(result.get("action_contract"), dict) else {}
        return _ok(
            validation={
                "contract_present": bool(contract),
                "contract_id": contract.get("contract_id"),
                "operator_core_dispatch_required": True,
                "msdc_body_dispatch_required_for_physical_action": True,
                "driver_bridge_executes_directly": False,
                "bounded_action_allowed": bool(result.get("bounded_action_allowed")),
                "decision": result.get("decision"),
            },
            instinct=result,
            source="appdrivers.emergency_contract_validate",
        )
    except Exception as exc:
        return _err(str(exc), 500, source="appdrivers.emergency_contract_validate")



def init_app(app, connect_sqlite=None, meta_db_path=None, api_key_auth_ok=None, sign_ok=None):
    global _CONNECT_SQLITE, _META_DB, _API_KEY_AUTH_OK, _SIGN_OK

    _CONNECT_SQLITE = connect_sqlite
    _META_DB = meta_db_path
    _API_KEY_AUTH_OK = api_key_auth_ok
    _SIGN_OK = sign_ok
    return apply(app)

# ====================================================================
# END OF appdrivers.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing API bridge adapter.
SML_ORGAN_METADATA = {
    "name": 'appdrivers',
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
    "metadata": {"sml_adapter": "api_bridge_non_executing", "source_file": 'appdrivers.py'},
}

def sml_get_metadata():
    return dict(SML_ORGAN_METADATA)

def sml_health():
    return {"status": "Healthy", "availability": 1.0, "integrity": 1.0, "performance": 1.0, "reliability": 1.0, "confidence": 0.75, "latency_ms": 0.0, "stability": 1.0, "compatibility": 1.0, "notes": ["SML API adapter present"]}

def sml_diagnostics():
    return {"status": "OK", "component": 'appdrivers', "sml_adapter": True, "metadata": dict(SML_ORGAN_METADATA), "health": sml_health()}
# --- SML ORGAN ADAPTER END ---

