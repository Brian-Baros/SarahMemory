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
# com.softdev0.ir.sensor.usb/driver.py
# USB IR Sensor driver package for SarahMemory AiOS
# Supports generic USB/HID/serial IR sensors and Wii-style IR point tracking adapters
# Degrades cleanly when optional backends are unavailable.

import json
import math
import shutil
import subprocess
import threading
import time
from datetime import datetime

_DRIVER_ID = "com.softdev0.ir.sensor.usb"

try:
    import hid  # type: ignore
except Exception:
    hid = None

try:
    import serial  # type: ignore
    from serial.tools import list_ports  # type: ignore
except Exception:
    serial = None
    list_ports = None

_SESSION = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "notes": [],
    "listener_running": False,
    "active_backend": None,
    "device": None,
    "last_sample": None,
    "last_event_ts": None,
    "buffer_size": 0,
}

_RUNTIME = {
    "config": {},
    "device_handle": None,
    "serial_handle": None,
    "listener_thread": None,
    "listener_stop": None,
    "buffer": [],
    "stats": {
        "samples": 0,
        "events": 0,
        "errors": 0,
        "dropped": 0,
    },
}


def _utc_now():
    return time.time()


def _new_instance_id():
    return "DRV-%s-%sZ-%s" % (
        _DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper()
    )


def _mask(value):
    if value is None:
        return None
    text = str(value)
    if len(text) <= 4:
        return "*" * len(text)
    return text[:2] + ("*" * (len(text) - 4)) + text[-2:]


def _append_note(message):
    _SESSION["notes"] = (_SESSION.get("notes") or [])[-24:] + [str(message)]


def _set_error(message):
    _SESSION["error"] = str(message)
    _append_note("ERROR: %s" % message)
    _RUNTIME["stats"]["errors"] += 1


def _safe_int(value, default=None):
    try:
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        text = str(value).strip()
        if text.lower().startswith("0x"):
            return int(text, 16)
        return int(float(text))
    except Exception:
        return default


def _safe_float(value, default=None):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _normalize_config(config):
    cfg = dict(config or {})
    normalized = {
        "enabled": bool(cfg.get("enabled", True)),
        "autoload": bool(cfg.get("autoload", False)),
        "backend_priority": list(cfg.get("backend_priority") or ["hid", "serial"]),
        "device_filter": str(cfg.get("device_filter") or "").strip(),
        "vendor_id": _safe_int(cfg.get("vendor_id")),
        "product_id": _safe_int(cfg.get("product_id")),
        "usage_page": _safe_int(cfg.get("usage_page")),
        "usage": _safe_int(cfg.get("usage")),
        "interface_number": _safe_int(cfg.get("interface_number")),
        "serial_number": str(cfg.get("serial_number") or "").strip() or None,
        "path": str(cfg.get("path") or "").strip() or None,
        "mode": str(cfg.get("mode") or "tracking").strip().lower(),
        "point_count": max(1, min(8, _safe_int(cfg.get("point_count"), 4) or 4)),
        "normalize_points": bool(cfg.get("normalize_points", True)),
        "frame_width": max(1, _safe_int(cfg.get("frame_width"), 1024) or 1024),
        "frame_height": max(1, _safe_int(cfg.get("frame_height"), 768) or 768),
        "sample_timeout_ms": max(1, _safe_int(cfg.get("sample_timeout_ms"), 100) or 100),
        "line_timeout_ms": max(1, _safe_int(cfg.get("line_timeout_ms"), 300) or 300),
        "poll_interval_ms": max(1, _safe_int(cfg.get("poll_interval_ms"), 25) or 25),
        "report_size": max(1, _safe_int(cfg.get("report_size"), 64) or 64),
        "buffer_limit": max(1, _safe_int(cfg.get("buffer_limit"), 256) or 256),
        "emit_raw_reports": bool(cfg.get("emit_raw_reports", False)),
        "emit_samples": bool(cfg.get("emit_samples", True)),
        "event_mode": str(cfg.get("event_mode") or "sample").strip().lower(),
        "serial_port": str(cfg.get("serial_port") or "").strip() or None,
        "baud_rate": max(1200, _safe_int(cfg.get("baud_rate"), 115200) or 115200),
        "serial_bytesize": max(5, min(8, _safe_int(cfg.get("serial_bytesize"), 8) or 8)),
        "serial_parity": str(cfg.get("serial_parity") or "N").strip().upper()[:1] or "N",
        "serial_stopbits": _safe_float(cfg.get("serial_stopbits"), 1) or 1,
        "serial_encoding": str(cfg.get("serial_encoding") or "utf-8").strip() or "utf-8",
        "minimum_confidence": max(0.0, min(1.0, _safe_float(cfg.get("minimum_confidence"), 0.0) or 0.0)),
        "dedupe_window_ms": max(0, _safe_int(cfg.get("dedupe_window_ms"), 0) or 0),
        "allow_shell_probe": bool(cfg.get("allow_shell_probe", False)),
    }
    return normalized


def driver_info():
    return {
        "id": _DRIVER_ID,
        "name": "IR Sensor (USB) Driver",
        "version": "1.0.0",
        "transport": "usb",
        "session": dict(_SESSION),
        "runtime": {
            "active_backend": _SESSION.get("active_backend"),
            "device": _SESSION.get("device"),
            "stats": dict(_RUNTIME["stats"]),
        }
    }


def driver_validate(config: dict):
    errors = []
    warnings = []
    if not isinstance(config, dict):
        errors.append("config must be a JSON object")
        return {"ok": False, "errors": errors, "warnings": warnings}
    cfg = _normalize_config(config)
    if cfg["mode"] not in {"tracking", "raw", "events"}:
        errors.append("mode must be one of: tracking, raw, events")
    if cfg["serial_parity"] not in {"N", "E", "O", "M", "S"}:
        errors.append("serial_parity must be one of N,E,O,M,S")
    if "hid" in cfg["backend_priority"] and hid is None:
        warnings.append("hid backend unavailable")
    if "serial" in cfg["backend_priority"] and serial is None:
        warnings.append("serial backend unavailable")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings, "normalized": cfg}


def _reset_runtime():
    _RUNTIME["config"] = {}
    _RUNTIME["device_handle"] = None
    _RUNTIME["serial_handle"] = None
    _RUNTIME["listener_thread"] = None
    _RUNTIME["listener_stop"] = None
    _RUNTIME["buffer"] = []
    _RUNTIME["stats"] = {"samples": 0, "events": 0, "errors": 0, "dropped": 0}


def _clear_handles():
    dev = _RUNTIME.get("device_handle")
    if dev is not None:
        try:
            dev.close()
        except Exception:
            pass
    ser = _RUNTIME.get("serial_handle")
    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass
    _RUNTIME["device_handle"] = None
    _RUNTIME["serial_handle"] = None


def driver_init(context=None, config=None):
    config = config or {}
    v = driver_validate(config)
    if not v.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": v}

    _reset_runtime()
    _RUNTIME["config"] = v["normalized"]
    _SESSION["instance_id"] = _new_instance_id()
    _SESSION["started_ts"] = time.time()
    _SESSION["status"] = "ready"
    _SESSION["error"] = None
    _SESSION["notes"] = ["IR sensor driver initialized."]
    _SESSION["listener_running"] = False
    _SESSION["active_backend"] = None
    _SESSION["device"] = None
    _SESSION["last_sample"] = None
    _SESSION["last_event_ts"] = None
    _SESSION["buffer_size"] = 0
    return {"ok": True, "instance_id": _SESSION["instance_id"], "validate": v, "info": driver_info()}


def driver_shutdown(context=None):
    _stop_listener()
    _clear_handles()
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _SESSION["notes"] = ["Shutdown completed."]
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    _SESSION["listener_running"] = False
    _SESSION["active_backend"] = None
    _SESSION["device"] = None
    _SESSION["last_sample"] = None
    _SESSION["last_event_ts"] = None
    _SESSION["buffer_size"] = 0
    _reset_runtime()
    return True


def driver_status(context=None):
    return {
        "ok": True,
        "session": dict(_SESSION),
        "runtime": {
            "config": _public_config(),
            "stats": dict(_RUNTIME["stats"]),
            "buffer_size": len(_RUNTIME["buffer"]),
        }
    }


def _public_config():
    cfg = dict(_RUNTIME.get("config") or {})
    return cfg


def _probe_backends():
    return {
        "hid": {
            "available": hid is not None,
            "module": "hid" if hid is not None else None,
        },
        "serial": {
            "available": serial is not None and list_ports is not None,
            "module": "pyserial" if serial is not None else None,
        },
        "shell": {
            "available": any(shutil.which(x) for x in ["lsusb", "udevadm"]),
            "tools": [x for x in ["lsusb", "udevadm"] if shutil.which(x)],
        }
    }


def _match_text(value, needle):
    if not needle:
        return True
    return needle.lower() in str(value or "").lower()


def _list_hid_devices():
    if hid is None:
        return []
    results = []
    try:
        for item in hid.enumerate():
            results.append({
                "backend": "hid",
                "path": item.get("path").decode(errors="ignore") if isinstance(item.get("path"), (bytes, bytearray)) else item.get("path"),
                "vendor_id": item.get("vendor_id"),
                "product_id": item.get("product_id"),
                "usage_page": item.get("usage_page"),
                "usage": item.get("usage"),
                "interface_number": item.get("interface_number"),
                "product_string": item.get("product_string"),
                "manufacturer_string": item.get("manufacturer_string"),
                "serial_number": item.get("serial_number"),
                "release_number": item.get("release_number"),
            })
    except Exception as exc:
        _set_error("hid_enumerate_failed: %s" % exc)
    return results


def _list_serial_devices():
    if list_ports is None:
        return []
    rows = []
    try:
        for port in list_ports.comports():
            rows.append({
                "backend": "serial",
                "device": port.device,
                "name": port.name,
                "description": port.description,
                "hwid": port.hwid,
                "vid": getattr(port, "vid", None),
                "pid": getattr(port, "pid", None),
                "serial_number": getattr(port, "serial_number", None),
                "manufacturer": getattr(port, "manufacturer", None),
                "product": getattr(port, "product", None),
                "interface": getattr(port, "interface", None),
            })
    except Exception as exc:
        _set_error("serial_enumerate_failed: %s" % exc)
    return rows


def _shell_probe():
    if not _RUNTIME["config"].get("allow_shell_probe"):
        return {"ok": False, "error": "shell_probe_disabled"}
    result = {"lsusb": None, "udevadm": []}
    if shutil.which("lsusb"):
        try:
            cp = subprocess.run(["lsusb"], capture_output=True, text=True, timeout=5)
            result["lsusb"] = cp.stdout.strip().splitlines()
        except Exception as exc:
            result["lsusb"] = ["ERROR: %s" % exc]
    return {"ok": True, "result": result}


def _device_matches_hid(item, cfg):
    if cfg["vendor_id"] is not None and item.get("vendor_id") != cfg["vendor_id"]:
        return False
    if cfg["product_id"] is not None and item.get("product_id") != cfg["product_id"]:
        return False
    if cfg["usage_page"] is not None and item.get("usage_page") != cfg["usage_page"]:
        return False
    if cfg["usage"] is not None and item.get("usage") != cfg["usage"]:
        return False
    if cfg["interface_number"] is not None and item.get("interface_number") != cfg["interface_number"]:
        return False
    if cfg["serial_number"] and str(item.get("serial_number") or "") != cfg["serial_number"]:
        return False
    if cfg["path"] and str(item.get("path") or "") != cfg["path"]:
        return False
    if cfg["device_filter"]:
        filter_text = cfg["device_filter"]
        hay = " ".join([
            str(item.get("product_string") or ""),
            str(item.get("manufacturer_string") or ""),
            str(item.get("serial_number") or ""),
            str(item.get("path") or ""),
            "%04x:%04x" % (item.get("vendor_id") or 0, item.get("product_id") or 0),
        ])
        if filter_text.lower() not in hay.lower():
            return False
    return True


def _device_matches_serial(item, cfg):
    if cfg["vendor_id"] is not None and item.get("vid") != cfg["vendor_id"]:
        return False
    if cfg["product_id"] is not None and item.get("pid") != cfg["product_id"]:
        return False
    if cfg["serial_number"] and str(item.get("serial_number") or "") != cfg["serial_number"]:
        return False
    if cfg["serial_port"] and str(item.get("device") or "") != cfg["serial_port"]:
        return False
    if cfg["device_filter"]:
        filter_text = cfg["device_filter"]
        hay = " ".join([
            str(item.get("device") or ""),
            str(item.get("description") or ""),
            str(item.get("product") or ""),
            str(item.get("manufacturer") or ""),
            "%04x:%04x" % (item.get("vid") or 0, item.get("pid") or 0),
        ])
        if filter_text.lower() not in hay.lower():
            return False
    return True


def _select_device():
    cfg = _RUNTIME["config"]
    devices = discover_devices()["devices"]
    for backend in cfg["backend_priority"]:
        if backend == "hid":
            for item in devices:
                if item.get("backend") == "hid" and _device_matches_hid(item, cfg):
                    return item
        if backend == "serial":
            for item in devices:
                if item.get("backend") == "serial" and _device_matches_serial(item, cfg):
                    return item
    return None


def discover_devices():
    devices = _list_hid_devices() + _list_serial_devices()
    return {"ok": True, "devices": devices, "count": len(devices)}


def _open_selected():
    selected = _select_device()
    if not selected:
        return {"ok": False, "error": "device_not_found"}
    _clear_handles()
    cfg = _RUNTIME["config"]
    if selected["backend"] == "hid":
        if hid is None:
            return {"ok": False, "error": "hid_backend_unavailable"}
        try:
            dev = hid.device()
            path = selected.get("path")
            if path is not None:
                dev.open_path(path.encode() if isinstance(path, str) else path)
            else:
                dev.open(selected.get("vendor_id"), selected.get("product_id"))
            dev.set_nonblocking(1)
            _RUNTIME["device_handle"] = dev
            _SESSION["active_backend"] = "hid"
            _SESSION["device"] = selected
            _append_note("Connected HID IR sensor.")
            return {"ok": True, "backend": "hid", "device": selected}
        except Exception as exc:
            _set_error("hid_open_failed: %s" % exc)
            return {"ok": False, "error": "hid_open_failed", "details": str(exc)}
    if selected["backend"] == "serial":
        if serial is None:
            return {"ok": False, "error": "serial_backend_unavailable"}
        try:
            ser = serial.Serial(
                port=selected.get("device"),
                baudrate=cfg["baud_rate"],
                bytesize=cfg["serial_bytesize"],
                parity=cfg["serial_parity"],
                stopbits=cfg["serial_stopbits"],
                timeout=cfg["sample_timeout_ms"] / 1000.0,
                write_timeout=cfg["line_timeout_ms"] / 1000.0,
            )
            _RUNTIME["serial_handle"] = ser
            _SESSION["active_backend"] = "serial"
            _SESSION["device"] = selected
            _append_note("Connected serial IR sensor.")
            return {"ok": True, "backend": "serial", "device": selected}
        except Exception as exc:
            _set_error("serial_open_failed: %s" % exc)
            return {"ok": False, "error": "serial_open_failed", "details": str(exc)}
    return {"ok": False, "error": "unsupported_backend"}


def _close_device():
    _clear_handles()
    _SESSION["active_backend"] = None
    _SESSION["device"] = None
    return {"ok": True}


def _normalize_points(points):
    cfg = _RUNTIME["config"]
    if not cfg["normalize_points"]:
        return points
    fw = float(cfg["frame_width"] or 1)
    fh = float(cfg["frame_height"] or 1)
    out = []
    for p in points:
        x = p.get("x")
        y = p.get("y")
        if x is None or y is None:
            out.append(dict(p))
            continue
        q = dict(p)
        q["x_norm"] = max(0.0, min(1.0, float(x) / fw))
        q["y_norm"] = max(0.0, min(1.0, float(y) / fh))
        out.append(q)
    return out


def _decode_ir_points_from_report(report):
    cfg = _RUNTIME["config"]
    data = [int(x) & 0xFF for x in (report or [])]
    points = []
    # Heuristic: 3-byte point tuples [x_lo, y_lo, c] repeated.
    # This supports generic adapters and test mode, not a vendor-locked protocol.
    chunks = cfg["point_count"]
    for i in range(chunks):
        base = i * 3
        if base + 2 >= len(data):
            break
        x = data[base]
        y = data[base + 1]
        c = data[base + 2] / 255.0
        if x == 0 and y == 0 and c <= 0:
            continue
        if c < cfg["minimum_confidence"]:
            continue
        points.append({"id": i, "x": x, "y": y, "confidence": round(c, 4)})
    return _normalize_points(points)


def _parse_serial_line(line):
    # Accept either JSON lines or CSV-like x,y[,confidence]
    text = (line or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            if "points" in payload and isinstance(payload["points"], list):
                payload["points"] = _normalize_points(payload["points"])
            return payload
    except Exception:
        pass
    if "," in text:
        parts = [x.strip() for x in text.split(",")]
        if len(parts) >= 2:
            x = _safe_int(parts[0])
            y = _safe_int(parts[1])
            conf = _safe_float(parts[2], 1.0) if len(parts) >= 3 else 1.0
            if x is not None and y is not None:
                return {"mode": "tracking", "points": _normalize_points([{"id": 0, "x": x, "y": y, "confidence": conf}])}
    return {"mode": "raw", "text": text}


def _emit_event(event):
    if event is None:
        return None
    now = _utc_now()
    cfg = _RUNTIME["config"]
    dedupe_ms = cfg.get("dedupe_window_ms", 0)
    if dedupe_ms > 0 and _RUNTIME["buffer"]:
        prev = _RUNTIME["buffer"][-1]
        if prev.get("payload") == event and (now - prev.get("ts", 0)) * 1000.0 <= dedupe_ms:
            return None
    item = {"ts": now, "payload": event}
    _RUNTIME["buffer"].append(item)
    if len(_RUNTIME["buffer"]) > cfg["buffer_limit"]:
        _RUNTIME["buffer"].pop(0)
        _RUNTIME["stats"]["dropped"] += 1
    _RUNTIME["stats"]["events"] += 1
    _SESSION["buffer_size"] = len(_RUNTIME["buffer"])
    _SESSION["last_event_ts"] = now
    _SESSION["last_sample"] = event
    return item


def _read_once():
    cfg = _RUNTIME["config"]
    if _SESSION["active_backend"] == "hid" and _RUNTIME["device_handle"] is not None:
        try:
            report = _RUNTIME["device_handle"].read(cfg["report_size"])
            if not report:
                return {"ok": True, "sample": None}
            _RUNTIME["stats"]["samples"] += 1
            payload = {"mode": cfg["mode"], "points": _decode_ir_points_from_report(report)}
            if cfg["emit_raw_reports"]:
                payload["raw_report"] = list(report)
            _emit_event(payload)
            return {"ok": True, "sample": payload}
        except Exception as exc:
            _set_error("hid_read_failed: %s" % exc)
            return {"ok": False, "error": "hid_read_failed", "details": str(exc)}
    if _SESSION["active_backend"] == "serial" and _RUNTIME["serial_handle"] is not None:
        try:
            line = _RUNTIME["serial_handle"].readline()
            if not line:
                return {"ok": True, "sample": None}
            text = line.decode(_RUNTIME["config"]["serial_encoding"], errors="replace")
            payload = _parse_serial_line(text)
            _RUNTIME["stats"]["samples"] += 1
            _emit_event(payload)
            return {"ok": True, "sample": payload}
        except Exception as exc:
            _set_error("serial_read_failed: %s" % exc)
            return {"ok": False, "error": "serial_read_failed", "details": str(exc)}
    return {"ok": False, "error": "not_connected"}


def _listener_loop():
    stop_event = _RUNTIME["listener_stop"]
    interval = (_RUNTIME["config"].get("poll_interval_ms", 25) or 25) / 1000.0
    while stop_event and not stop_event.is_set():
        _read_once()
        time.sleep(interval)
    _SESSION["listener_running"] = False


def _start_listener():
    if _SESSION["listener_running"]:
        return {"ok": True, "already_running": True}
    if _SESSION["active_backend"] is None:
        opened = _open_selected()
        if not opened.get("ok"):
            return opened
    stop_event = threading.Event()
    thread = threading.Thread(target=_listener_loop, name="ir-sensor-listener", daemon=True)
    _RUNTIME["listener_stop"] = stop_event
    _RUNTIME["listener_thread"] = thread
    _SESSION["listener_running"] = True
    thread.start()
    return {"ok": True, "listener_running": True}


def _stop_listener():
    ev = _RUNTIME.get("listener_stop")
    th = _RUNTIME.get("listener_thread")
    if ev is not None:
        try:
            ev.set()
        except Exception:
            pass
    if th is not None and th.is_alive():
        th.join(timeout=1.5)
    _RUNTIME["listener_stop"] = None
    _RUNTIME["listener_thread"] = None
    _SESSION["listener_running"] = False
    return {"ok": True}


def _get_buffer(limit=None):
    if limit is None:
        return {"ok": True, "events": list(_RUNTIME["buffer"])}
    return {"ok": True, "events": list(_RUNTIME["buffer"][-max(1, int(limit)):])}


def _clear_buffer():
    _RUNTIME["buffer"] = []
    _SESSION["buffer_size"] = 0
    return {"ok": True}


def _inject_report(report):
    if not isinstance(report, (list, tuple)):
        return {"ok": False, "error": "report must be a list"}
    payload = {"mode": _RUNTIME["config"].get("mode", "tracking"), "points": _decode_ir_points_from_report(report), "raw_report": list(report)}
    _emit_event(payload)
    return {"ok": True, "sample": payload}


def _feed_points(points):
    if not isinstance(points, list):
        return {"ok": False, "error": "points must be a list"}
    normalized = []
    for idx, p in enumerate(points):
        if not isinstance(p, dict):
            continue
        normalized.append({
            "id": _safe_int(p.get("id"), idx),
            "x": _safe_float(p.get("x"), 0),
            "y": _safe_float(p.get("y"), 0),
            "confidence": _safe_float(p.get("confidence"), 1.0) or 1.0,
        })
    payload = {"mode": "tracking", "points": _normalize_points(normalized)}
    _emit_event(payload)
    return {"ok": True, "sample": payload}


def driver_action(action_id: str, context=None, payload=None):
    payload = payload or {}
    try:
        if action_id == "ping":
            return {"ok": True, "pong": True, "ts": time.time()}
        if action_id == "probe_backends":
            return {"ok": True, "backends": _probe_backends()}
        if action_id == "discover_devices":
            return discover_devices()
        if action_id == "shell_probe":
            return _shell_probe()
        if action_id == "connect":
            return _open_selected()
        if action_id == "disconnect":
            _stop_listener()
            return _close_device()
        if action_id == "read_once":
            if _SESSION["active_backend"] is None:
                opened = _open_selected()
                if not opened.get("ok"):
                    return opened
            return _read_once()
        if action_id == "start_listen":
            return _start_listener()
        if action_id == "stop_listen":
            return _stop_listener()
        if action_id == "get_buffer":
            return _get_buffer(payload.get("limit"))
        if action_id == "clear_buffer":
            return _clear_buffer()
        if action_id == "inject_report":
            return _inject_report(payload.get("report"))
        if action_id == "feed_points":
            return _feed_points(payload.get("points"))
        if action_id == "get_config":
            return {"ok": True, "config": _public_config()}
        if action_id == "set_config":
            new_cfg = dict(_RUNTIME.get("config") or {})
            new_cfg.update(payload if isinstance(payload, dict) else {})
            v = driver_validate(new_cfg)
            if not v.get("ok"):
                return {"ok": False, "error": "validation_failed", "details": v}
            _RUNTIME["config"] = v["normalized"]
            return {"ok": True, "config": _public_config(), "warnings": v.get("warnings", [])}
        if action_id == "safe_stop":
            _stop_listener()
            _close_device()
            _SESSION["status"] = "ready"
            return {"ok": True, "status": "stopped"}
        return {"ok": False, "error": f"Action '{action_id}' not implemented"}
    except Exception as exc:
        _set_error(str(exc))
        return {"ok": False, "error": "driver_action_failed", "details": str(exc)}
