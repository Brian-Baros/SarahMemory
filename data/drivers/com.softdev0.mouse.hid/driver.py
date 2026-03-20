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
# com.softdev0.mouse.hid/driver.py
# HID mouse monitor/decoder driver for SarahMemory AiOS
# Transport: hid

from __future__ import annotations

import os
import time
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

_DRIVER_ID = "com.softdev0.mouse.hid"

_DEFAULT_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "autoload": False,
    "backend_priority": ["hid"],
    "capture_mode": "listen",
    "device_filter": "",
    "vendor_id": None,
    "product_id": None,
    "usage_page": 1,
    "usage": 2,
    "interface_number": None,
    "path": "",
    "serial_number": "",
    "read_size": 8,
    "read_timeout_ms": 120,
    "poll_interval_ms": 20,
    "buffer_limit": 512,
    "emit_raw_reports": False,
    "emit_move_events": True,
    "emit_button_events": True,
    "emit_wheel_events": True,
    "absolute_mode": False,
    "normalize_axes": False,
    "x_min": 0,
    "x_max": 32767,
    "y_min": 0,
    "y_max": 32767,
    "dedupe_window_ms": 0,
    "allow_output_control": False,
    "allow_vendor_passthrough": False,
}

_SESSION: Dict[str, Any] = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "notes": [],
    "connected": False,
    "selected_device": None,
    "backend": None,
    "listener_running": False,
    "last_event_ts": None,
    "event_count": 0,
    "report_count": 0,
    "button_state": {"left": False, "right": False, "middle": False, "button4": False, "button5": False},
    "position": {"x": 0, "y": 0},
    "wheel": {"vertical": 0, "horizontal": 0},
}

_STATE: Dict[str, Any] = {
    "config": dict(_DEFAULT_CONFIG),
    "context": {},
    "device": None,
    "listener_thread": None,
    "stop_event": None,
    "buffer": [],
    "last_report": None,
    "last_emit_ts": 0.0,
    "hid_module": None,
}


def _new_instance_id() -> str:
    return "DRV-%s-%sZ-%s" % (
        _DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _now() -> float:
    return time.time()


def _append_note(msg: str) -> None:
    notes = _SESSION.setdefault("notes", [])
    notes.append(msg)
    if len(notes) > 50:
        del notes[:-50]


def _merge_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(_DEFAULT_CONFIG)
    if isinstance(config, dict):
        merged.update({k: v for k, v in config.items() if v is not None})
    return merged


def _mask_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return dict(config)


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, str) and value.lower().startswith("0x"):
            return int(value, 16)
        return int(value)
    except Exception:
        return None


def _import_hid():
    if _STATE.get("hid_module") is not None:
        return _STATE["hid_module"]
    try:
        import hid  # type: ignore
        _STATE["hid_module"] = hid
        return hid
    except Exception:
        return None


def _hid_available() -> bool:
    return _import_hid() is not None


def driver_info() -> Dict[str, Any]:
    return {
        "id": _DRIVER_ID,
        "name": "Mouse HID Driver",
        "version": "1.0.0",
        "transport": "hid",
        "session": dict(_SESSION),
    }


def driver_validate(config: Dict[str, Any]):
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be a JSON object"], "warnings": warnings}
    for key in ["enabled", "autoload", "emit_raw_reports", "emit_move_events", "emit_button_events", "emit_wheel_events", "absolute_mode", "normalize_axes", "allow_output_control", "allow_vendor_passthrough"]:
        if key in config and not isinstance(config.get(key), bool):
            errors.append(f"{key} must be a boolean")
    for key in ["capture_mode", "device_filter", "path", "serial_number"]:
        if key in config and config.get(key) is not None and not isinstance(config.get(key), str):
            errors.append(f"{key} must be a string")
    for key in ["read_size", "read_timeout_ms", "poll_interval_ms", "buffer_limit", "dedupe_window_ms", "x_min", "x_max", "y_min", "y_max"]:
        if key in config and _coerce_int(config.get(key)) is None:
            errors.append(f"{key} must be an integer")
    capture_mode = str(config.get("capture_mode", _DEFAULT_CONFIG["capture_mode"]))
    if capture_mode not in {"listen", "exclusive"}:
        errors.append("capture_mode must be 'listen' or 'exclusive'")
    read_size = _coerce_int(config.get("read_size", _DEFAULT_CONFIG["read_size"]))
    if read_size is not None and read_size < 3:
        errors.append("read_size must be >= 3")
    if not _hid_available():
        warnings.append("hid backend not installed; live HID capture unavailable until hidapi/python-hid is installed")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def _push_event(event: Dict[str, Any]) -> None:
    buf = _STATE.setdefault("buffer", [])
    dedupe_ms = int(_STATE["config"].get("dedupe_window_ms") or 0)
    now = _now()
    if dedupe_ms > 0 and buf:
        last = buf[-1]
        if last.get("kind") == event.get("kind") and last.get("data") == event.get("data"):
            if (now - float(last.get("ts") or 0)) * 1000.0 <= dedupe_ms:
                return
    event["ts"] = now
    buf.append(event)
    limit = int(_STATE["config"].get("buffer_limit") or 512)
    if len(buf) > limit:
        del buf[:-limit]
    _SESSION["last_event_ts"] = now
    _SESSION["event_count"] = int(_SESSION.get("event_count") or 0) + 1


def _normalize_axis(value: int, min_v: int, max_v: int) -> float:
    if max_v <= min_v:
        return float(value)
    return max(0.0, min(1.0, (float(value) - float(min_v)) / float(max_v - min_v)))


def _buttons_from_byte(b: int) -> Dict[str, bool]:
    return {
        "left": bool(b & 0x01),
        "right": bool(b & 0x02),
        "middle": bool(b & 0x04),
        "button4": bool(b & 0x08),
        "button5": bool(b & 0x10),
    }


def _to_signed8(v: int) -> int:
    return v - 256 if v > 127 else v


def _to_signed16(lo: int, hi: int) -> int:
    value = (hi << 8) | lo
    return value - 65536 if value > 32767 else value


def _decode_report(report: List[int]) -> Dict[str, Any]:
    if not report:
        return {"ok": False, "error": "empty_report"}
    absolute = bool(_STATE["config"].get("absolute_mode"))
    emit_raw = bool(_STATE["config"].get("emit_raw_reports"))
    buttons = _buttons_from_byte(int(report[0]) if len(report) > 0 else 0)

    if absolute and len(report) >= 5:
        x = _to_signed16(int(report[1]), int(report[2]))
        y = _to_signed16(int(report[3]), int(report[4]))
        wheel = _to_signed8(int(report[5])) if len(report) > 5 else 0
        hwheel = _to_signed8(int(report[6])) if len(report) > 6 else 0
    else:
        x = _to_signed8(int(report[1])) if len(report) > 1 else 0
        y = _to_signed8(int(report[2])) if len(report) > 2 else 0
        wheel = _to_signed8(int(report[3])) if len(report) > 3 else 0
        hwheel = _to_signed8(int(report[4])) if len(report) > 4 else 0

    prev_buttons = dict(_SESSION.get("button_state") or {})
    _SESSION["button_state"] = buttons

    if absolute:
        _SESSION["position"] = {"x": x, "y": y}
    else:
        pos = dict(_SESSION.get("position") or {"x": 0, "y": 0})
        pos["x"] = int(pos.get("x", 0)) + x
        pos["y"] = int(pos.get("y", 0)) + y
        _SESSION["position"] = pos
    wheel_state = dict(_SESSION.get("wheel") or {"vertical": 0, "horizontal": 0})
    wheel_state["vertical"] = int(wheel_state.get("vertical", 0)) + wheel
    wheel_state["horizontal"] = int(wheel_state.get("horizontal", 0)) + hwheel
    _SESSION["wheel"] = wheel_state
    _SESSION["report_count"] = int(_SESSION.get("report_count") or 0) + 1

    events: List[Dict[str, Any]] = []
    if bool(_STATE["config"].get("emit_move_events")) and (x != 0 or y != 0):
        move_data: Dict[str, Any] = {"dx": x, "dy": y, "absolute": absolute, "position": dict(_SESSION["position"])}
        if absolute and bool(_STATE["config"].get("normalize_axes")):
            move_data["normalized_x"] = _normalize_axis(x, int(_STATE["config"].get("x_min") or 0), int(_STATE["config"].get("x_max") or 32767))
            move_data["normalized_y"] = _normalize_axis(y, int(_STATE["config"].get("y_min") or 0), int(_STATE["config"].get("y_max") or 32767))
        events.append({"kind": "move", "data": move_data})
    if bool(_STATE["config"].get("emit_wheel_events")) and (wheel != 0 or hwheel != 0):
        events.append({"kind": "wheel", "data": {"vertical": wheel, "horizontal": hwheel, "state": dict(_SESSION["wheel"])}})
    if bool(_STATE["config"].get("emit_button_events")):
        for key, pressed in buttons.items():
            if prev_buttons.get(key) != pressed:
                events.append({"kind": "button", "data": {"button": key, "pressed": pressed}})
    if emit_raw:
        events.append({"kind": "raw_report", "data": {"bytes": list(report)}})

    for ev in events:
        _push_event(ev)
    _STATE["last_report"] = list(report)
    return {"ok": True, "report": list(report), "decoded": events, "position": dict(_SESSION["position"]), "buttons": dict(buttons), "wheel": dict(_SESSION["wheel"])}


def _device_matches(dev: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    filt = str(cfg.get("device_filter") or "").strip().lower()
    if filt:
        hay = " ".join([
            str(dev.get("product_string") or ""),
            str(dev.get("manufacturer_string") or ""),
            str(dev.get("serial_number") or ""),
            str(dev.get("path") or ""),
            f"{dev.get('vendor_id', '')}:{dev.get('product_id', '')}",
        ]).lower()
        if filt not in hay:
            return False
    checks = {
        "vendor_id": _coerce_int(cfg.get("vendor_id")),
        "product_id": _coerce_int(cfg.get("product_id")),
        "usage_page": _coerce_int(cfg.get("usage_page")),
        "usage": _coerce_int(cfg.get("usage")),
        "interface_number": _coerce_int(cfg.get("interface_number")),
    }
    for key, expected in checks.items():
        if expected is not None and _coerce_int(dev.get(key)) != expected:
            return False
    path = str(cfg.get("path") or "").strip()
    if path and str(dev.get("path") or "") != path:
        return False
    serial_number = str(cfg.get("serial_number") or "").strip()
    if serial_number and str(dev.get("serial_number") or "") != serial_number:
        return False
    return True


def _sanitize_device_info(dev: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key in ["vendor_id", "product_id", "usage_page", "usage", "interface_number", "serial_number", "product_string", "manufacturer_string", "release_number"]:
        if key in dev:
            out[key] = dev.get(key)
    path = dev.get("path")
    if isinstance(path, (bytes, bytearray)):
        try:
            path = path.decode("utf-8", "ignore")
        except Exception:
            path = repr(path)
    out["path"] = path
    return out


def _enumerate_devices() -> Dict[str, Any]:
    hid = _import_hid()
    if hid is None:
        return {"ok": False, "error": "hid_unavailable", "devices": []}
    try:
        devices = hid.enumerate()
        sanitized = [_sanitize_device_info(d) for d in devices if _device_matches(d, _STATE["config"])]
        return {"ok": True, "devices": sanitized, "count": len(sanitized)}
    except Exception as exc:
        return {"ok": False, "error": "enumerate_failed", "details": str(exc), "devices": []}


def _select_first_device() -> Optional[Dict[str, Any]]:
    res = _enumerate_devices()
    if not res.get("ok"):
        return None
    devices = res.get("devices") or []
    return devices[0] if devices else None


def _open_selected_device(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = dict(_STATE["config"])
    if isinstance(payload, dict):
        cfg.update({k: v for k, v in payload.items() if k in cfg and v not in (None, "")})
    hid = _import_hid()
    if hid is None:
        return {"ok": False, "error": "hid_unavailable"}
    selected = _select_first_device() if not payload or not payload.get("path") else None
    if isinstance(payload, dict) and payload.get("path"):
        selected = {
            "path": payload.get("path"),
            "vendor_id": _coerce_int(payload.get("vendor_id")),
            "product_id": _coerce_int(payload.get("product_id")),
            "usage_page": _coerce_int(payload.get("usage_page")),
            "usage": _coerce_int(payload.get("usage")),
            "interface_number": _coerce_int(payload.get("interface_number")),
            "serial_number": payload.get("serial_number"),
            "product_string": payload.get("product_string"),
            "manufacturer_string": payload.get("manufacturer_string"),
        }
    if not selected:
        return {"ok": False, "error": "device_not_found"}
    try:
        dev = hid.device()
        path = selected.get("path")
        if path:
            if isinstance(path, str):
                dev.open_path(path.encode() if os.name != "nt" else path)
            else:
                dev.open_path(path)
        else:
            dev.open(int(selected["vendor_id"]), int(selected["product_id"]))
        timeout = int(cfg.get("read_timeout_ms") or 120)
        if hasattr(dev, "set_nonblocking"):
            dev.set_nonblocking(0 if timeout > 0 else 1)
        _STATE["device"] = dev
        _SESSION["connected"] = True
        _SESSION["selected_device"] = dict(selected)
        _SESSION["backend"] = "hid"
        _append_note("Mouse HID device opened")
        return {"ok": True, "device": dict(selected)}
    except Exception as exc:
        return {"ok": False, "error": "open_failed", "details": str(exc), "device": selected}


def _close_device() -> Dict[str, Any]:
    dev = _STATE.get("device")
    if dev is not None:
        try:
            dev.close()
        except Exception:
            pass
    _STATE["device"] = None
    _SESSION["connected"] = False
    _SESSION["selected_device"] = None
    return {"ok": True}


def _read_report_once() -> Dict[str, Any]:
    dev = _STATE.get("device")
    if dev is None:
        return {"ok": False, "error": "not_connected"}
    size = int(_STATE["config"].get("read_size") or 8)
    timeout = int(_STATE["config"].get("read_timeout_ms") or 120)
    try:
        data = dev.read(size, timeout)
        if not data:
            return {"ok": True, "report": [], "decoded": [], "empty": True}
        return _decode_report(list(data))
    except TypeError:
        try:
            data = dev.read(size)
            if not data:
                return {"ok": True, "report": [], "decoded": [], "empty": True}
            return _decode_report(list(data))
        except Exception as exc:
            return {"ok": False, "error": "read_failed", "details": str(exc)}
    except Exception as exc:
        return {"ok": False, "error": "read_failed", "details": str(exc)}


def _listener_loop() -> None:
    stop_event = _STATE.get("stop_event")
    _SESSION["listener_running"] = True
    _SESSION["status"] = "listening"
    poll_s = max(0.001, int(_STATE["config"].get("poll_interval_ms") or 20) / 1000.0)
    while stop_event is not None and not stop_event.is_set():
        res = _read_report_once()
        if not res.get("ok"):
            err = res.get("error")
            if err not in {"not_connected"}:
                _SESSION["error"] = err
                _append_note(f"listener_read_error:{err}")
            time.sleep(poll_s)
            continue
        time.sleep(poll_s)
    _SESSION["listener_running"] = False
    if _SESSION.get("status") != "stopped":
        _SESSION["status"] = "ready" if _SESSION.get("connected") else "idle"


def _start_listener() -> Dict[str, Any]:
    if _STATE.get("listener_thread") and _STATE["listener_thread"].is_alive():
        return {"ok": True, "already_running": True}
    if _STATE.get("device") is None:
        opened = _open_selected_device()
        if not opened.get("ok"):
            return opened
    stop_event = threading.Event()
    _STATE["stop_event"] = stop_event
    th = threading.Thread(target=_listener_loop, name="mouse-hid-listener", daemon=True)
    _STATE["listener_thread"] = th
    th.start()
    return {"ok": True, "listener_running": True}


def _stop_listener() -> Dict[str, Any]:
    stop_event = _STATE.get("stop_event")
    th = _STATE.get("listener_thread")
    if stop_event is not None:
        stop_event.set()
    if th is not None and th.is_alive():
        th.join(timeout=1.0)
    _STATE["stop_event"] = None
    _STATE["listener_thread"] = None
    _SESSION["listener_running"] = False
    if _SESSION.get("status") != "stopped":
        _SESSION["status"] = "ready" if _SESSION.get("connected") else "idle"
    return {"ok": True}


def _probe_backend() -> Dict[str, Any]:
    return {
        "ok": True,
        "backends": [{"name": "hid", "available": _hid_available()}],
        "preferred": list(_STATE["config"].get("backend_priority") or ["hid"]),
    }


def _get_config() -> Dict[str, Any]:
    return {"ok": True, "config": _mask_config(dict(_STATE["config"]))}


def _set_config(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    merged = dict(_STATE["config"])
    merged.update(payload)
    v = driver_validate(merged)
    if not v.get("ok"):
        return {"ok": False, "error": "validation_failed", "details": v}
    _STATE["config"] = _merge_config(merged)
    return {"ok": True, "config": _mask_config(dict(_STATE["config"])), "validate": v}


def _inject_report(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    report = payload.get("report")
    if not isinstance(report, list) or not all(isinstance(x, int) for x in report):
        return {"ok": False, "error": "report must be a list[int]"}
    return _decode_report(report)


def _feed_events(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    items = payload.get("events")
    if not isinstance(items, list):
        return {"ok": False, "error": "events must be a list"}
    pushed = 0
    for item in items:
        if isinstance(item, dict) and item.get("kind"):
            _push_event({"kind": str(item["kind"]), "data": dict(item.get("data") or {})})
            pushed += 1
    return {"ok": True, "pushed": pushed, "buffer_size": len(_STATE.get("buffer") or [])}


def _get_buffer(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    limit = _coerce_int(payload.get("limit")) or len(_STATE.get("buffer") or [])
    data = list(_STATE.get("buffer") or [])
    if limit > 0:
        data = data[-limit:]
    return {"ok": True, "events": data, "count": len(data)}


def _clear_buffer() -> Dict[str, Any]:
    _STATE["buffer"] = []
    return {"ok": True, "cleared": True}


def _safe_stop() -> Dict[str, Any]:
    _stop_listener()
    _close_device()
    _SESSION["status"] = "ready"
    return {"ok": True, "stopped": True}


def _output_control(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not bool(_STATE["config"].get("allow_output_control")):
        return {"ok": False, "error": "output_control_disabled"}
    return {"ok": False, "error": "not_implemented", "details": "OS-level mouse injection is intentionally gated and not implemented in this package."}


def driver_init(context=None, config=None):
    merged = _merge_config(config)
    v = driver_validate(merged)
    if not v.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": v}
    _STATE["context"] = context or {}
    _STATE["config"] = merged
    _STATE["buffer"] = []
    _STATE["last_report"] = None
    _STATE["device"] = None
    _STATE["listener_thread"] = None
    _STATE["stop_event"] = None
    _SESSION.update({
        "instance_id": _new_instance_id(),
        "started_ts": _now(),
        "status": "ready",
        "error": None,
        "notes": [],
        "connected": False,
        "selected_device": None,
        "backend": None,
        "listener_running": False,
        "last_event_ts": None,
        "event_count": 0,
        "report_count": 0,
        "button_state": {"left": False, "right": False, "middle": False, "button4": False, "button5": False},
        "position": {"x": 0, "y": 0},
        "wheel": {"vertical": 0, "horizontal": 0},
    })
    for warn in v.get("warnings") or []:
        _append_note(f"warning:{warn}")
    return {"ok": True, "instance_id": _SESSION["instance_id"], "validate": v, "info": driver_info()}


def driver_shutdown(context=None):
    try:
        _stop_listener()
    finally:
        _close_device()
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _append_note("Shutdown completed.")
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    return True


def driver_status(context=None):
    return {
        "ok": True,
        "session": dict(_SESSION),
        "config": _mask_config(dict(_STATE.get("config") or {})),
        "buffer_size": len(_STATE.get("buffer") or []),
        "last_report": list(_STATE.get("last_report") or []) if _STATE.get("last_report") is not None else None,
    }


def driver_action(action_id: str, context=None, payload=None):
    action_id = str(action_id or "").strip().lower()
    payload = payload or {}
    if action_id == "ping":
        return {"ok": True, "pong": True, "ts": _now()}
    if action_id == "probe_backend":
        return _probe_backend()
    if action_id in {"list_hid", "discover_devices", "list_devices"}:
        return _enumerate_devices()
    if action_id in {"open", "connect"}:
        return _open_selected_device(payload)
    if action_id in {"close", "disconnect"}:
        return _close_device()
    if action_id in {"read_once", "poll_once"}:
        return _read_report_once()
    if action_id == "start_listen":
        return _start_listener()
    if action_id == "stop_listen":
        return _stop_listener()
    if action_id == "get_buffer":
        return _get_buffer(payload)
    if action_id == "clear_buffer":
        return _clear_buffer()
    if action_id == "get_config":
        return _get_config()
    if action_id == "set_config":
        return _set_config(payload)
    if action_id == "inject_report":
        return _inject_report(payload)
    if action_id == "feed_events":
        return _feed_events(payload)
    if action_id in {"safe_stop", "stop"}:
        return _safe_stop()
    if action_id in {"output_control", "move_pointer", "click"}:
        return _output_control(payload)
    if action_id == "vendor_passthrough":
        if not bool(_STATE["config"].get("allow_vendor_passthrough")):
            return {"ok": False, "error": "vendor_passthrough_disabled"}
        return {"ok": False, "error": "not_implemented", "details": "Vendor passthrough is gated and not implemented in this package."}
    return {"ok": False, "error": f"Action '{action_id}' not implemented"}
