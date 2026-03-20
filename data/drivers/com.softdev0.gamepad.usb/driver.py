"""
--==The SarahMemory Project==--
 File: ./data/drivers/com.softdev0.[DRIVERNAME].[TYPE]/driver.py
 Purpose: SarahMemory AiOS governed boot-layer bridge driver.
 Part of the SarahMemory Companion AI-bot Platform
 Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
 www.linkedin.com/in/brian-baros-29962a176
 https://www.facebook.com/bbaros
 brian.baros@sarahmemory.com
 'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
 https://www.sarahmemory.com
 https://api.sarahmemory.com
 https://ai.sarahmemory.com
 https://store.sarahmemory.com
 ==============================================================================================
"""
# SarahMemory Driver Package: com.softdev0.gamepad.usb
# Name: Game Controller USB Driver
#
# Governed USB gamepad driver implementation.
# - No privileged OS driver installs
# - No synthetic OS input injection
# - No auto-probing unless invoked by action
# - Degrades cleanly when optional input libraries are unavailable

from __future__ import annotations

import importlib
import threading
import time
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional

DRIVER_ID = "com.softdev0.gamepad.usb"
DRIVER_NAME = "Game Controller USB Driver"
VERSION = "1.0.0"

_SUPPORTED_BACKENDS = ("pygame", "inputs", "hid")

_SESSION: Dict[str, Any] = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "connected": False,
    "active_backend": None,
    "connected_device": None,
    "listener_running": False,
    "last_state": {},
    "last_result": None,
    "meta": {},
}

_RUNTIME: Dict[str, Any] = {
    "config": {},
    "events": [],
    "listener_thread": None,
    "listener_stop": None,
    "pygame": {
        "module": None,
        "initialized": False,
        "joystick": None,
        "device_index": None,
    },
}


_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "autoload": False,
    "backend": "",
    "backend_priority": ["pygame", "inputs", "hid"],
    "profile": "default",
    "device_index": 0,
    "device_name": "",
    "vendor_id": "",
    "product_id": "",
    "deadzone": 0.1,
    "poll_interval_ms": 20,
    "event_buffer_limit": 256,
    "auto_connect": False,
    "normalize_axes": True,
    "trigger_mode": "signed",
    "allow_rumble": True,
    "allow_light_control": True,
    "vendor_passthrough": False,
}


def _now_id() -> str:
    return "DRV-%s-%sZ-%s" % (
        DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _clamp(n: float, low: float, high: float) -> float:
    return max(low, min(high, n))


def _normalize_config(config: Optional[dict]) -> Dict[str, Any]:
    merged = deepcopy(_DEFAULTS)
    if isinstance(config, dict):
        merged.update(config)
    if not isinstance(merged.get("backend_priority"), list):
        merged["backend_priority"] = list(_DEFAULTS["backend_priority"])
    try:
        merged["deadzone"] = float(merged.get("deadzone", 0.1))
    except Exception:
        merged["deadzone"] = 0.1
    merged["deadzone"] = _clamp(merged["deadzone"], 0.0, 0.95)
    try:
        merged["poll_interval_ms"] = int(merged.get("poll_interval_ms", 20))
    except Exception:
        merged["poll_interval_ms"] = 20
    merged["poll_interval_ms"] = max(5, merged["poll_interval_ms"])
    try:
        merged["event_buffer_limit"] = int(merged.get("event_buffer_limit", 256))
    except Exception:
        merged["event_buffer_limit"] = 256
    merged["event_buffer_limit"] = max(10, merged["event_buffer_limit"])
    try:
        merged["device_index"] = int(merged.get("device_index", 0))
    except Exception:
        merged["device_index"] = 0
    merged["backend"] = str(merged.get("backend") or "").strip().lower()
    merged["profile"] = str(merged.get("profile") or "default")
    merged["device_name"] = str(merged.get("device_name") or "")
    merged["vendor_id"] = str(merged.get("vendor_id") or "")
    merged["product_id"] = str(merged.get("product_id") or "")
    merged["trigger_mode"] = str(merged.get("trigger_mode") or "signed")
    return merged


def _trim_events() -> None:
    limit = int(_RUNTIME["config"].get("event_buffer_limit", 256) or 256)
    if len(_RUNTIME["events"]) > limit:
        _RUNTIME["events"] = _RUNTIME["events"][-limit:]


def _push_event(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    item = {
        "ts": _utc_now(),
        "type": event_type,
        "payload": payload,
    }
    _RUNTIME["events"].append(item)
    _trim_events()
    return item


def _safe_repr(obj: Any) -> str:
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def _module_available(name: str) -> bool:
    try:
        return importlib.import_module(name) is not None
    except Exception:
        return False


def _backend_probe() -> Dict[str, Any]:
    result = {"available": {}, "preferred": None}
    cfg = _RUNTIME.get("config") or _DEFAULTS
    explicit = cfg.get("backend")
    priority = [explicit] if explicit else list(cfg.get("backend_priority") or _DEFAULTS["backend_priority"])
    priority = [b for b in priority if b]
    for backend in _SUPPORTED_BACKENDS:
        result["available"][backend] = _module_available(backend)
    for backend in priority:
        if result["available"].get(backend):
            result["preferred"] = backend
            break
    if not result["preferred"]:
        for backend in _SUPPORTED_BACKENDS:
            if result["available"].get(backend):
                result["preferred"] = backend
                break
    return result


def _to_hex_or_blank(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return hex(int(value))
    except Exception:
        return _safe_repr(value)


# ------------------------- pygame backend -------------------------
def _pygame_init() -> Dict[str, Any]:
    if _RUNTIME["pygame"]["initialized"]:
        return {"ok": True}
    try:
        pygame = importlib.import_module("pygame")
        pygame.init()
        pygame.joystick.init()
        _RUNTIME["pygame"]["module"] = pygame
        _RUNTIME["pygame"]["initialized"] = True
        return {"ok": True}
    except Exception as exc:
        return {"ok": False, "error": f"pygame_init_failed: {exc}"}


def _pygame_list_devices() -> List[Dict[str, Any]]:
    init = _pygame_init()
    if not init.get("ok"):
        return []
    pygame = _RUNTIME["pygame"]["module"]
    count = pygame.joystick.get_count()
    devices = []
    for idx in range(count):
        js = pygame.joystick.Joystick(idx)
        try:
            js.init()
        except Exception:
            pass
        info = {
            "backend": "pygame",
            "index": idx,
            "name": _safe_repr(getattr(js, "get_name", lambda: "")()),
            "guid": _safe_repr(getattr(js, "get_guid", lambda: "")()),
            "instance_id": _safe_repr(getattr(js, "get_instance_id", lambda: "")()),
            "axes": int(getattr(js, "get_numaxes", lambda: 0)()),
            "buttons": int(getattr(js, "get_numbuttons", lambda: 0)()),
            "hats": int(getattr(js, "get_numhats", lambda: 0)()),
            "balls": int(getattr(js, "get_numballs", lambda: 0)()),
            "power_level": _safe_repr(getattr(js, "get_power_level", lambda: "unknown")()),
        }
        devices.append(info)
    return devices


def _pygame_match_device(devices: List[Dict[str, Any]], cfg: Dict[str, Any], payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    device_index = payload.get("device_index", cfg.get("device_index", 0))
    device_name = str(payload.get("device_name") or cfg.get("device_name") or "").strip().lower()
    vendor_id = str(payload.get("vendor_id") or cfg.get("vendor_id") or "").strip().lower()
    product_id = str(payload.get("product_id") or cfg.get("product_id") or "").strip().lower()
    for dev in devices:
        if device_name and device_name in str(dev.get("name", "")).lower():
            return dev
        if vendor_id and vendor_id in str(dev.get("guid", "")).lower():
            return dev
        if product_id and product_id in str(dev.get("guid", "")).lower():
            return dev
    for dev in devices:
        if int(dev.get("index", -1)) == int(device_index):
            return dev
    return devices[0] if devices else None


def _pygame_connect(cfg: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    devices = _pygame_list_devices()
    if not devices:
        return {"ok": False, "error": "no_usb_gamepad_detected"}
    target = _pygame_match_device(devices, cfg, payload)
    if not target:
        return {"ok": False, "error": "no_matching_usb_gamepad"}
    pygame = _RUNTIME["pygame"]["module"]
    js = pygame.joystick.Joystick(int(target["index"]))
    js.init()
    _RUNTIME["pygame"]["joystick"] = js
    _RUNTIME["pygame"]["device_index"] = int(target["index"])
    _SESSION["connected"] = True
    _SESSION["active_backend"] = "pygame"
    _SESSION["connected_device"] = deepcopy(target)
    _SESSION["status"] = "connected"
    return {"ok": True, "device": deepcopy(target)}


def _apply_deadzone(v: float, dz: float) -> float:
    if abs(v) < dz:
        return 0.0
    return float(v)


def _normalize_trigger(v: float, mode: str) -> float:
    if mode == "unsigned":
        return (float(v) + 1.0) / 2.0
    return float(v)


def _pygame_read_state(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not _SESSION.get("connected") or _SESSION.get("active_backend") != "pygame":
        return {"ok": False, "error": "not_connected"}
    pygame = _RUNTIME["pygame"]["module"]
    js = _RUNTIME["pygame"]["joystick"]
    if pygame is None or js is None:
        return {"ok": False, "error": "backend_not_ready"}
    try:
        pygame.event.pump()
    except Exception:
        pass
    dz = float(cfg.get("deadzone", 0.1))
    trigger_mode = str(cfg.get("trigger_mode", "signed"))
    axes = []
    for i in range(js.get_numaxes()):
        value = js.get_axis(i)
        value = _apply_deadzone(value, dz)
        if i in (4, 5):
            value = _normalize_trigger(value, trigger_mode)
        axes.append(float(value))
    buttons = [bool(js.get_button(i)) for i in range(js.get_numbuttons())]
    hats = []
    for i in range(js.get_numhats()):
        hats.append(tuple(js.get_hat(i)))
    state = {
        "ts": _utc_now(),
        "backend": "pygame",
        "name": _safe_repr(js.get_name()),
        "axes": axes,
        "buttons": buttons,
        "hats": hats,
    }
    _SESSION["last_state"] = deepcopy(state)
    return {"ok": True, "state": state}


def _pygame_rumble(payload: Dict[str, Any]) -> Dict[str, Any]:
    js = _RUNTIME["pygame"].get("joystick")
    if js is None:
        return {"ok": False, "error": "not_connected"}
    if not hasattr(js, "rumble"):
        return {"ok": False, "error": "rumble_not_supported_by_backend"}
    low = float(payload.get("low_frequency", payload.get("low", 0.5) or 0.5))
    high = float(payload.get("high_frequency", payload.get("high", 0.5) or 0.5))
    duration_ms = int(payload.get("duration_ms", 250) or 250)
    try:
        ok = js.rumble(_clamp(low, 0.0, 1.0), _clamp(high, 0.0, 1.0), max(1, duration_ms))
        return {"ok": bool(ok), "accepted": bool(ok)}
    except Exception as exc:
        return {"ok": False, "error": f"rumble_failed: {exc}"}


def _pygame_close() -> None:
    js = _RUNTIME["pygame"].get("joystick")
    if js is not None:
        try:
            js.quit()
        except Exception:
            pass
    _RUNTIME["pygame"]["joystick"] = None
    _RUNTIME["pygame"]["device_index"] = None


# ------------------------- generic helpers -------------------------
def _clear_connection() -> None:
    _pygame_close()
    _SESSION["connected"] = False
    _SESSION["active_backend"] = None
    _SESSION["connected_device"] = None
    _SESSION["listener_running"] = False
    _SESSION["last_state"] = {}


def _listener_loop() -> None:
    stop = _RUNTIME.get("listener_stop")
    while stop and not stop.is_set():
        cfg = _RUNTIME.get("config") or _DEFAULTS
        state_result = _read_state()
        if state_result.get("ok"):
            _push_event("state", state_result.get("state", {}))
        time.sleep(max(0.005, float(cfg.get("poll_interval_ms", 20)) / 1000.0))
    _SESSION["listener_running"] = False


def _start_listener() -> Dict[str, Any]:
    if _SESSION.get("listener_running"):
        return {"ok": True, "already_running": True}
    if not _SESSION.get("connected"):
        return {"ok": False, "error": "not_connected"}
    stop = threading.Event()
    thread = threading.Thread(target=_listener_loop, daemon=True)
    _RUNTIME["listener_stop"] = stop
    _RUNTIME["listener_thread"] = thread
    _SESSION["listener_running"] = True
    thread.start()
    return {"ok": True, "listener_running": True}


def _stop_listener() -> Dict[str, Any]:
    stop = _RUNTIME.get("listener_stop")
    thread = _RUNTIME.get("listener_thread")
    if stop is not None:
        stop.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=1.0)
    _RUNTIME["listener_stop"] = None
    _RUNTIME["listener_thread"] = None
    _SESSION["listener_running"] = False
    return {"ok": True}


def _list_controllers() -> Dict[str, Any]:
    probe = _backend_probe()
    devices: List[Dict[str, Any]] = []
    if probe.get("available", {}).get("pygame"):
        devices.extend(_pygame_list_devices())
    return {"ok": True, "probe": probe, "controllers": devices}


def _connect(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    cfg = _RUNTIME.get("config") or _DEFAULTS
    probe = _backend_probe()
    backend = str(payload.get("backend") or cfg.get("backend") or probe.get("preferred") or "")
    if backend != "pygame":
        return {
            "ok": False,
            "error": "no_supported_backend_available",
            "probe": probe,
            "detail": "This package prefers pygame for live USB controller support in the current implementation.",
        }
    result = _pygame_connect(cfg, payload)
    result["probe"] = probe
    return result


def _disconnect() -> Dict[str, Any]:
    _stop_listener()
    _clear_connection()
    _SESSION["status"] = "ready"
    return {"ok": True}


def _read_state() -> Dict[str, Any]:
    cfg = _RUNTIME.get("config") or _DEFAULTS
    backend = _SESSION.get("active_backend")
    if backend == "pygame":
        return _pygame_read_state(cfg)
    return {"ok": False, "error": "not_connected"}


def _inject_event(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    event_type = str(payload.get("type") or "synthetic")
    event_payload = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
    item = _push_event(event_type, deepcopy(event_payload))
    return {"ok": True, "event": item}


def _get_events(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    limit = int(payload.get("limit", len(_RUNTIME["events"])) or len(_RUNTIME["events"]))
    limit = max(1, limit)
    return {"ok": True, "events": deepcopy(_RUNTIME["events"][-limit:])}


def _clear_events() -> Dict[str, Any]:
    _RUNTIME["events"] = []
    return {"ok": True}


def _set_config(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    current = deepcopy(_RUNTIME.get("config") or _DEFAULTS)
    current.update(payload)
    _RUNTIME["config"] = _normalize_config(current)
    return {"ok": True, "config": deepcopy(_RUNTIME["config"])}


def _get_config() -> Dict[str, Any]:
    return {"ok": True, "config": deepcopy(_RUNTIME.get("config") or _DEFAULTS)}


def _rumble(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    cfg = _RUNTIME.get("config") or _DEFAULTS
    if not cfg.get("allow_rumble", True):
        return {"ok": False, "error": "rumble_disabled_by_config"}
    backend = _SESSION.get("active_backend")
    if backend == "pygame":
        return _pygame_rumble(payload)
    return {"ok": False, "error": "rumble_not_supported_by_active_backend"}


def _set_light(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    cfg = _RUNTIME.get("config") or _DEFAULTS
    if not cfg.get("allow_light_control", True):
        return {"ok": False, "error": "light_control_disabled_by_config"}
    return {
        "ok": False,
        "error": "light_control_not_supported_by_current_backend",
        "payload": deepcopy(payload),
    }


def _vendor_passthrough(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = _RUNTIME.get("config") or _DEFAULTS
    if not cfg.get("vendor_passthrough", False):
        return {"ok": False, "error": "vendor_passthrough_disabled"}
    return {"ok": False, "error": "vendor_passthrough_not_implemented", "payload": deepcopy(payload or {})}


def driver_info() -> Dict[str, Any]:
    return {
        "id": DRIVER_ID,
        "name": DRIVER_NAME,
        "version": VERSION,
        "session": deepcopy(_SESSION),
        "defaults": deepcopy(_DEFAULTS),
    }


def driver_validate(config: dict) -> Dict[str, Any]:
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be an object"], "warnings": []}
    warnings: List[str] = []
    backend = str(config.get("backend") or "").strip().lower()
    if backend and backend not in _SUPPORTED_BACKENDS:
        warnings.append(f"unsupported backend requested: {backend}")
    return {"ok": True, "errors": [], "warnings": warnings}


def driver_init(context: Optional[dict] = None, config: Optional[dict] = None) -> Dict[str, Any]:
    config = config or {}
    validate = driver_validate(config)
    if not validate.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": validate}

    _RUNTIME["config"] = _normalize_config(config)
    _RUNTIME["events"] = []
    _RUNTIME["listener_thread"] = None
    _RUNTIME["listener_stop"] = None

    _SESSION["instance_id"] = _now_id()
    _SESSION["started_ts"] = time.time()
    _SESSION["status"] = "ready"
    _SESSION["error"] = None
    _SESSION["connected"] = False
    _SESSION["active_backend"] = None
    _SESSION["connected_device"] = None
    _SESSION["listener_running"] = False
    _SESSION["last_state"] = {}
    _SESSION["last_result"] = None
    _SESSION["meta"] = {"context": context or {}, "config": deepcopy(_RUNTIME["config"])}

    if _RUNTIME["config"].get("auto_connect"):
        _connect({})

    return {
        "ok": True,
        "instance_id": _SESSION["instance_id"],
        "info": driver_info(),
        "validate": validate,
        "probe": _backend_probe(),
    }


def driver_shutdown(context: Optional[dict] = None) -> bool:
    _stop_listener()
    _clear_connection()
    try:
        if _RUNTIME["pygame"].get("initialized") and _RUNTIME["pygame"].get("module"):
            pygame = _RUNTIME["pygame"]["module"]
            try:
                pygame.joystick.quit()
            except Exception:
                pass
            try:
                pygame.quit()
            except Exception:
                pass
    except Exception:
        pass
    _RUNTIME["pygame"] = {"module": None, "initialized": False, "joystick": None, "device_index": None}
    _RUNTIME["config"] = {}
    _RUNTIME["events"] = []
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _SESSION["meta"] = {}
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    _SESSION["last_result"] = None
    return True


def driver_status(context: Optional[dict] = None) -> Dict[str, Any]:
    return {
        "ok": True,
        "session": deepcopy(_SESSION),
        "probe": _backend_probe(),
        "event_count": len(_RUNTIME.get("events", [])),
    }


def driver_action(action_id: str, context: Optional[dict] = None, payload: Optional[dict] = None) -> Dict[str, Any]:
    payload = payload or {}
    action_id = str(action_id or "").strip()

    handlers = {
        "ping": lambda: {"ok": True, "driver": DRIVER_ID, "ts": _utc_now()},
        "backend_probe": _backend_probe,
        "list_controllers": _list_controllers,
        "connect": lambda: _connect(payload),
        "disconnect": _disconnect,
        "read_state": _read_state,
        "start_listener": _start_listener,
        "stop_listener": _stop_listener,
        "get_events": lambda: _get_events(payload),
        "clear_events": _clear_events,
        "inject_event": lambda: _inject_event(payload),
        "get_config": _get_config,
        "set_config": lambda: _set_config(payload),
        "rumble": lambda: _rumble(payload),
        "set_light": lambda: _set_light(payload),
        "vendor_passthrough": lambda: _vendor_passthrough(payload),
        "safe_stop": _disconnect,
    }

    handler = handlers.get(action_id)
    if handler is None:
        result = {"ok": False, "error": f"Action '{action_id}' not implemented by {DRIVER_ID}", "action_id": action_id}
    else:
        try:
            result = handler()
        except Exception as exc:
            result = {"ok": False, "error": f"action_failed: {exc}", "action_id": action_id}

    _SESSION["last_result"] = deepcopy(result)
    if not result.get("ok"):
        _SESSION["error"] = result.get("error")
    else:
        _SESSION["error"] = None
    return result
