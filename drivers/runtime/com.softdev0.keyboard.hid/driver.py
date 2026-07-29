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
# --== The SarahMemory Project ==--
# Driver: com.softdev0.keyboard.hid
# Purpose: Governed HID keyboard monitoring / mapping driver

import json
import os
import threading
import time
from collections import deque
from datetime import datetime

_DRIVER_ID = "com.softdev0.keyboard.hid"

_SESSION = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "notes": [],
    "backend": None,
    "listener_running": False,
    "device": None,
    "last_event": None,
    "last_scan": [],
}

_STATE = {
    "config": {},
    "buffer": deque(maxlen=512),
    "pressed": set(),
    "listener_thread": None,
    "stop_event": None,
    "backend_mod": None,
    "device_handle": None,
}

_KEYBOARD_USAGE_PAGE = 0x01
_KEYBOARD_USAGE = 0x06

_MOD_BITS = {
    0x01: "LEFT_CTRL",
    0x02: "LEFT_SHIFT",
    0x04: "LEFT_ALT",
    0x08: "LEFT_META",
    0x10: "RIGHT_CTRL",
    0x20: "RIGHT_SHIFT",
    0x40: "RIGHT_ALT",
    0x80: "RIGHT_META",
}

_KEYCODE_MAP = {
    0x04: ("a", "A"), 0x05: ("b", "B"), 0x06: ("c", "C"), 0x07: ("d", "D"),
    0x08: ("e", "E"), 0x09: ("f", "F"), 0x0A: ("g", "G"), 0x0B: ("h", "H"),
    0x0C: ("i", "I"), 0x0D: ("j", "J"), 0x0E: ("k", "K"), 0x0F: ("l", "L"),
    0x10: ("m", "M"), 0x11: ("n", "N"), 0x12: ("o", "O"), 0x13: ("p", "P"),
    0x14: ("q", "Q"), 0x15: ("r", "R"), 0x16: ("s", "S"), 0x17: ("t", "T"),
    0x18: ("u", "U"), 0x19: ("v", "V"), 0x1A: ("w", "W"), 0x1B: ("x", "X"),
    0x1C: ("y", "Y"), 0x1D: ("z", "Z"),
    0x1E: ("1", "!"), 0x1F: ("2", "@"), 0x20: ("3", "#"), 0x21: ("4", "$"),
    0x22: ("5", "%"), 0x23: ("6", "^"), 0x24: ("7", "&"), 0x25: ("8", "*"),
    0x26: ("9", "("), 0x27: ("0", ")"),
    0x28: ("ENTER", "ENTER"), 0x29: ("ESC", "ESC"), 0x2A: ("BACKSPACE", "BACKSPACE"),
    0x2B: ("TAB", "TAB"), 0x2C: (" ", " "), 0x2D: ("-", "_"), 0x2E: ("=", "+"),
    0x2F: ("[", "{"), 0x30: ("]", "}"), 0x31: ("\\", "|"), 0x33: (";", ":"),
    0x34: ("'", '"'), 0x35: ("`", "~"), 0x36: (",", "<"), 0x37: (".", ">"),
    0x38: ("/", "?"), 0x39: ("CAPSLOCK", "CAPSLOCK"),
    0x3A: ("F1", "F1"), 0x3B: ("F2", "F2"), 0x3C: ("F3", "F3"), 0x3D: ("F4", "F4"),
    0x3E: ("F5", "F5"), 0x3F: ("F6", "F6"), 0x40: ("F7", "F7"), 0x41: ("F8", "F8"),
    0x42: ("F9", "F9"), 0x43: ("F10", "F10"), 0x44: ("F11", "F11"), 0x45: ("F12", "F12"),
    0x46: ("PRINTSCREEN", "PRINTSCREEN"), 0x47: ("SCROLLLOCK", "SCROLLLOCK"),
    0x48: ("PAUSE", "PAUSE"), 0x49: ("INSERT", "INSERT"), 0x4A: ("HOME", "HOME"),
    0x4B: ("PAGEUP", "PAGEUP"), 0x4C: ("DELETE", "DELETE"), 0x4D: ("END", "END"),
    0x4E: ("PAGEDOWN", "PAGEDOWN"), 0x4F: ("RIGHT", "RIGHT"), 0x50: ("LEFT", "LEFT"),
    0x51: ("DOWN", "DOWN"), 0x52: ("UP", "UP"),
}


def _utc_now():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_instance_id():
    return "DRV-%s-%sZ-%s" % (
        _DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _append_note(msg):
    notes = _SESSION.setdefault("notes", [])
    notes.append(f"[{_utc_now()}] {msg}")
    if len(notes) > 50:
        del notes[:-50]


def _mask_device(dev):
    if not isinstance(dev, dict):
        return dev
    data = dict(dev)
    path = data.get("path")
    if isinstance(path, bytes):
        try:
            path = path.decode("utf-8", "ignore")
        except Exception:
            path = repr(path)
    if isinstance(path, str) and len(path) > 12:
        data["path"] = path[:4] + "..." + path[-4:]
    serial = data.get("serial_number")
    if isinstance(serial, str) and len(serial) > 4:
        data["serial_number"] = serial[:2] + "***" + serial[-2:]
    return data


def _default_config():
    return {
        "enabled": True,
        "autoload": False,
        "backend_priority": ["hid"],
        "capture_mode": "listen",
        "device_filter": "",
        "vendor_id": None,
        "product_id": None,
        "usage_page": _KEYBOARD_USAGE_PAGE,
        "usage": _KEYBOARD_USAGE,
        "interface_number": None,
        "path": "",
        "serial_number": "",
        "read_size": 8,
        "read_timeout_ms": 100,
        "poll_interval_ms": 25,
        "line_timeout_ms": 200,
        "dedupe_window_ms": 50,
        "buffer_limit": 512,
        "emit_raw": False,
        "emit_keyup": False,
        "emit_modifiers": True,
        "case_mode": "native",
        "newline_on_enter": True,
        "append_enter_token": False,
        "allow_output_control": False,
        "allow_vendor_passthrough": False,
    }


def _normalize_config(config):
    merged = _default_config()
    if isinstance(config, dict):
        merged.update(config)
    try:
        merged["read_size"] = max(8, int(merged.get("read_size", 8)))
    except Exception:
        merged["read_size"] = 8
    for k, default in (("read_timeout_ms", 100), ("poll_interval_ms", 25), ("line_timeout_ms", 200), ("dedupe_window_ms", 50), ("buffer_limit", 512)):
        try:
            merged[k] = max(0, int(merged.get(k, default)))
        except Exception:
            merged[k] = default
    merged["capture_mode"] = str(merged.get("capture_mode", "listen") or "listen").lower()
    merged["case_mode"] = str(merged.get("case_mode", "native") or "native").lower()
    return merged


def driver_info():
    return {
        "id": _DRIVER_ID,
        "name": "Keyboard HID Driver",
        "version": "1.0.0",
        "transport": "hid",
        "session": dict(_SESSION),
    }


def driver_validate(config: dict):
    errors = []
    warnings = []
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be a JSON object"], "warnings": warnings}
    if "enabled" in config and not isinstance(config.get("enabled"), bool):
        errors.append("enabled must be a boolean")
    if "autoload" in config and not isinstance(config.get("autoload"), bool):
        errors.append("autoload must be a boolean")
    if "capture_mode" in config and str(config.get("capture_mode")).lower() not in {"listen", "exclusive"}:
        errors.append("capture_mode must be 'listen' or 'exclusive'")
    if "case_mode" in config and str(config.get("case_mode")).lower() not in {"native", "upper", "lower"}:
        errors.append("case_mode must be native|upper|lower")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def _probe_backend():
    result = {"hid": False, "errors": []}
    try:
        import hid  # type: ignore
        _STATE["backend_mod"] = hid
        _SESSION["backend"] = "hid"
        result["hid"] = True
    except Exception as exc:
        _STATE["backend_mod"] = None
        _SESSION["backend"] = None
        result["errors"].append(f"hid unavailable: {exc}")
    return {"ok": result["hid"], "backend": "hid" if result["hid"] else None, "details": result}


def _enumerate_devices():
    if _STATE.get("backend_mod") is None:
        _probe_backend()
    hid = _STATE.get("backend_mod")
    if hid is None:
        return []
    out = []
    try:
        for dev in hid.enumerate():
            item = dict(dev)
            path = item.get("path")
            if isinstance(path, bytes):
                try:
                    item["path"] = path.decode("utf-8", "ignore")
                except Exception:
                    item["path"] = repr(path)
            out.append(item)
    except Exception as exc:
        _append_note(f"enumerate failed: {exc}")
    return out


def _match_device(dev, cfg):
    def _eq(name):
        want = cfg.get(name)
        return want in (None, "", []) or dev.get(name) == want
    if not _eq("vendor_id"):
        return False
    if not _eq("product_id"):
        return False
    if not _eq("usage_page"):
        return False
    if not _eq("usage"):
        return False
    if cfg.get("interface_number") not in (None, "") and dev.get("interface_number") != cfg.get("interface_number"):
        return False
    if cfg.get("path"):
        p = dev.get("path")
        if isinstance(p, bytes):
            p = p.decode("utf-8", "ignore")
        if p != cfg.get("path"):
            return False
    if cfg.get("serial_number") and dev.get("serial_number") != cfg.get("serial_number"):
        return False
    filt = str(cfg.get("device_filter") or "").strip().lower()
    if filt:
        blob = " ".join(str(dev.get(k, "")) for k in ("manufacturer_string", "product_string", "serial_number", "path")).lower()
        if filt not in blob:
            return False
    return True


def _select_device(cfg):
    devices = _enumerate_devices()
    preferred = [d for d in devices if _match_device(d, cfg)]
    if preferred:
        return preferred[0], devices
    fallback = [d for d in devices if d.get("usage_page") == _KEYBOARD_USAGE_PAGE and d.get("usage") == _KEYBOARD_USAGE]
    if fallback:
        return fallback[0], devices
    return (devices[0] if devices else None), devices


def _open_device(cfg):
    if _STATE.get("backend_mod") is None:
        probe = _probe_backend()
        if not probe.get("ok"):
            return {"ok": False, "error": "backend_unavailable", "details": probe}
    hid = _STATE["backend_mod"]
    dev_meta, devices = _select_device(cfg)
    if not dev_meta:
        return {"ok": False, "error": "device_not_found", "devices": [_mask_device(d) for d in devices]}
    try:
        path = dev_meta.get("path")
        handle = hid.device()
        if isinstance(path, str):
            handle.open_path(path.encode("utf-8"))
        else:
            handle.open_path(path)
        handle.set_nonblocking(1)
        _STATE["device_handle"] = handle
        _SESSION["device"] = _mask_device(dev_meta)
        _append_note("device opened")
        return {"ok": True, "device": _SESSION["device"], "devices": [_mask_device(d) for d in devices]}
    except Exception as exc:
        _append_note(f"open failed: {exc}")
        return {"ok": False, "error": "open_failed", "details": str(exc), "device": _mask_device(dev_meta)}


def _close_device():
    handle = _STATE.get("device_handle")
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass
    _STATE["device_handle"] = None
    _SESSION["device"] = None


def _translate_key(code, shift=False, case_mode="native"):
    pair = _KEYCODE_MAP.get(code)
    if not pair:
        return f"KEY_{code:02X}"
    token = pair[1] if shift else pair[0]
    if len(token) == 1:
        if case_mode == "upper":
            return token.upper()
        if case_mode == "lower":
            return token.lower()
    return token


def _decode_report(report, cfg):
    if not report:
        return {"ok": False, "error": "empty_report"}
    data = list(report)
    if len(data) < 8:
        data.extend([0] * (8 - len(data)))
    modifiers = data[0]
    shift = bool(modifiers & (0x02 | 0x20))
    pressed_codes = [c for c in data[2:8] if c]
    new_codes = [c for c in pressed_codes if c not in _STATE["pressed"]]
    released_codes = [c for c in list(_STATE["pressed"]) if c not in pressed_codes]
    _STATE["pressed"] = set(pressed_codes)
    keys_down = [_translate_key(c, shift, cfg.get("case_mode", "native")) for c in new_codes]
    keys_up = [_translate_key(c, shift, cfg.get("case_mode", "native")) for c in released_codes]
    mods = [name for bit, name in _MOD_BITS.items() if modifiers & bit]
    text = ""
    for key in keys_down:
        if key == "ENTER":
            if cfg.get("append_enter_token"):
                text += "<ENTER>"
            if cfg.get("newline_on_enter"):
                text += "\n"
        elif len(key) == 1:
            text += key
        elif key == "TAB":
            text += "\t"
        elif key == "SPACE":
            text += " "
    event = {
        "ts": time.time(),
        "iso_ts": _utc_now(),
        "modifiers": mods,
        "keys_down": keys_down,
        "keys_up": keys_up,
        "text": text,
        "pressed_codes": pressed_codes,
    }
    if cfg.get("emit_raw"):
        event["raw_report"] = data
    return {"ok": True, "event": event}


def _push_event(event):
    limit = int(_STATE["config"].get("buffer_limit", 512)) or 512
    if _STATE["buffer"].maxlen != limit:
        _STATE["buffer"] = deque(list(_STATE["buffer"]), maxlen=limit)
    _STATE["buffer"].append(event)
    _SESSION["last_event"] = event


def _read_once(cfg=None):
    cfg = cfg or _STATE["config"]
    handle = _STATE.get("device_handle")
    if handle is None:
        opened = _open_device(cfg)
        if not opened.get("ok"):
            return opened
        handle = _STATE.get("device_handle")
    try:
        report = handle.read(int(cfg.get("read_size", 8)), int(cfg.get("read_timeout_ms", 100)))
    except TypeError:
        report = handle.read(int(cfg.get("read_size", 8)))
    except Exception as exc:
        return {"ok": False, "error": "read_failed", "details": str(exc)}
    if not report:
        return {"ok": True, "idle": True}
    decoded = _decode_report(report, cfg)
    if decoded.get("ok"):
        event = decoded["event"]
        if event.get("keys_down") or (cfg.get("emit_keyup") and event.get("keys_up")) or event.get("text") or (cfg.get("emit_modifiers") and event.get("modifiers")):
            _push_event(event)
    return decoded


def _listener_loop():
    cfg = _STATE["config"]
    sleep_s = max(0.005, float(cfg.get("poll_interval_ms", 25)) / 1000.0)
    while _STATE["stop_event"] and not _STATE["stop_event"].is_set():
        _read_once(cfg)
        time.sleep(sleep_s)
    _SESSION["listener_running"] = False


def _start_listener():
    if _SESSION.get("listener_running"):
        return {"ok": True, "already_running": True}
    if _STATE.get("device_handle") is None:
        opened = _open_device(_STATE["config"])
        if not opened.get("ok"):
            return opened
    _STATE["stop_event"] = threading.Event()
    t = threading.Thread(target=_listener_loop, name="KeyboardHIDListener", daemon=True)
    _STATE["listener_thread"] = t
    _SESSION["listener_running"] = True
    t.start()
    return {"ok": True, "listener_running": True}


def _stop_listener():
    ev = _STATE.get("stop_event")
    if ev is not None:
        ev.set()
    t = _STATE.get("listener_thread")
    if t and t.is_alive():
        t.join(timeout=1.0)
    _STATE["listener_thread"] = None
    _STATE["stop_event"] = None
    _SESSION["listener_running"] = False
    return {"ok": True, "listener_running": False}


def driver_init(context=None, config=None):
    config = _normalize_config(config or {})
    v = driver_validate(config)
    if not v.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": v}
    _STATE["config"] = config
    _STATE["buffer"] = deque(maxlen=int(config.get("buffer_limit", 512)))
    _STATE["pressed"] = set()
    probe = _probe_backend()
    _SESSION["instance_id"] = _new_instance_id()
    _SESSION["started_ts"] = time.time()
    _SESSION["status"] = "ready" if probe.get("ok") else "degraded"
    _SESSION["error"] = None if probe.get("ok") else "backend_unavailable"
    _SESSION["notes"] = []
    _append_note("driver initialized")
    return {"ok": True, "instance_id": _SESSION["instance_id"], "validate": v, "probe": probe, "info": driver_info()}


def driver_shutdown(context=None):
    _stop_listener()
    _close_device()
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _append_note("shutdown completed")
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    _SESSION["listener_running"] = False
    return True


def driver_status(context=None):
    return {
        "ok": True,
        "session": dict(_SESSION),
        "buffer_size": len(_STATE["buffer"]),
        "config": dict(_STATE.get("config") or {}),
    }


def driver_action(action_id: str, context=None, payload=None):
    payload = payload or {}
    if action_id == "ping":
        return {"ok": True, "pong": True, "ts": time.time()}
    if action_id == "probe_backend":
        return _probe_backend()
    if action_id == "list_hid":
        return {"ok": True, "devices": [_mask_device(d) for d in _enumerate_devices()]}
    if action_id == "open_device":
        cfg = dict(_STATE.get("config") or {})
        cfg.update(payload if isinstance(payload, dict) else {})
        return _open_device(cfg)
    if action_id == "close_device":
        _close_device()
        return {"ok": True}
    if action_id == "read_once":
        return _read_once(dict(_STATE.get("config") or {}))
    if action_id == "start_listen":
        return _start_listener()
    if action_id == "stop_listen":
        return _stop_listener()
    if action_id == "get_buffer":
        return {"ok": True, "events": list(_STATE["buffer"])}
    if action_id == "clear_buffer":
        _STATE["buffer"].clear()
        return {"ok": True, "cleared": True}
    if action_id == "get_config":
        return {"ok": True, "config": dict(_STATE.get("config") or {})}
    if action_id == "set_config":
        if not isinstance(payload, dict):
            return {"ok": False, "error": "payload must be object"}
        merged = dict(_STATE.get("config") or {})
        merged.update(payload)
        merged = _normalize_config(merged)
        v = driver_validate(merged)
        if not v.get("ok"):
            return {"ok": False, "error": "validation_failed", "details": v}
        _STATE["config"] = merged
        return {"ok": True, "config": merged}
    if action_id == "inject_report":
        report = payload.get("report") if isinstance(payload, dict) else None
        if not isinstance(report, list):
            return {"ok": False, "error": "report must be list[int]"}
        return _decode_report(report, dict(_STATE.get("config") or {}))
    if action_id == "feed_keystrokes":
        text = str(payload.get("text", "")) if isinstance(payload, dict) else ""
        event = {"ts": time.time(), "iso_ts": _utc_now(), "modifiers": [], "keys_down": [], "keys_up": [], "text": text, "synthetic": True}
        _push_event(event)
        return {"ok": True, "event": event}
    if action_id == "safe_stop":
        _stop_listener()
        _close_device()
        return {"ok": True, "stopped": True}
    return {"ok": False, "error": f"Action '{action_id}' not implemented"}
