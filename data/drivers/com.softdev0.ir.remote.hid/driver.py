# SarahMemory Driver Package: com.softdev0.ir.remote.hid
# Name: IR Remote Receiver HID Driver
# Purpose:
#   Cross-platform HID IR receiver driver for receivers/remotes that present as
#   keyboard-like HID devices or generic HID endpoints.
#
# Design notes:
#   - No privileged OS driver installs.
#   - No auto-probing unless invoked.
#   - Uses hidapi when available for raw HID reads.
#   - Provides keyboard-wedge style parsing and raw report injection for testing.
#   - Keeps governance gates for shell/vendor passthrough disabled by default.

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

DRIVER_ID = "com.softdev0.ir.remote.hid"
DRIVER_NAME = "IR Remote Receiver HID Driver"
VERSION = "1.0.0"

try:
    import hid  # type: ignore
except Exception:
    hid = None

KEYBOARD_USAGE_PAGE = 0x01
KEYBOARD_USAGE = 0x06

# Minimal HID keyboard usage translation for common IR keyboard-emulating remotes.
KEYCODE_MAP = {
    0x04: "a", 0x05: "b", 0x06: "c", 0x07: "d", 0x08: "e", 0x09: "f",
    0x0A: "g", 0x0B: "h", 0x0C: "i", 0x0D: "j", 0x0E: "k", 0x0F: "l",
    0x10: "m", 0x11: "n", 0x12: "o", 0x13: "p", 0x14: "q", 0x15: "r",
    0x16: "s", 0x17: "t", 0x18: "u", 0x19: "v", 0x1A: "w", 0x1B: "x",
    0x1C: "y", 0x1D: "z",
    0x1E: "1", 0x1F: "2", 0x20: "3", 0x21: "4", 0x22: "5",
    0x23: "6", 0x24: "7", 0x25: "8", 0x26: "9", 0x27: "0",
    0x28: "ENTER", 0x29: "ESC", 0x2A: "BACKSPACE", 0x2B: "TAB", 0x2C: "SPACE",
    0x2D: "-", 0x2E: "=", 0x2F: "[", 0x30: "]", 0x31: "\\", 0x33: ";",
    0x34: "'", 0x35: "`", 0x36: ",", 0x37: ".", 0x38: "/",
    0x4F: "RIGHT", 0x50: "LEFT", 0x51: "DOWN", 0x52: "UP",
    0x58: "ENTER",
    0x59: "KP1", 0x5A: "KP2", 0x5B: "KP3", 0x5C: "KP4", 0x5D: "KP5",
    0x5E: "KP6", 0x5F: "KP7", 0x60: "KP8", 0x61: "KP9", 0x62: "KP0",
    0x63: "KPDOT",
    0x66: "POWER",
    0x80: "VOLUME_UP", 0x81: "VOLUME_DOWN",
}

SHIFTED_KEYCODE_MAP = {
    0x1E: "!", 0x1F: "@", 0x20: "#", 0x21: "$", 0x22: "%",
    0x23: "^", 0x24: "&", 0x25: "*", 0x26: "(", 0x27: ")",
    0x2D: "_", 0x2E: "+", 0x2F: "{", 0x30: "}", 0x31: "|",
    0x33: ":", 0x34: '"', 0x35: "~", 0x36: "<", 0x37: ">", 0x38: "?",
}

DEFAULTS = {
    "enabled": True,
    "autoload": False,
    "vendor_id": "",
    "product_id": "",
    "usage_page": "0x0001",
    "usage": "0x0006",
    "interface_number": "",
    "path": "",
    "product_name_contains": "",
    "serial_number": "",
    "report_size": 8,
    "read_timeout_ms": 250,
    "line_timeout_ms": 1500,
    "buffer_limit": 200,
    "event_buffer_limit": 500,
    "dedupe_window_ms": 175,
    "repeat_suppression": True,
    "decode_mode": "keyboard_wedge",
    "case_mode": "preserve",
    "append_enter": True,
    "auto_append_newline": False,
    "map_file": "",
    "emit_raw_reports": False,
    "vendor_passthrough_enabled": False,
}

_SESSION: Dict[str, Any] = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "meta": {},
    "config": dict(DEFAULTS),
    "listener_running": False,
    "listener_backend": None,
    "connected_device": None,
    "last_scan": None,
    "last_scan_ts": None,
    "last_report": None,
    "events": deque(maxlen=DEFAULTS["event_buffer_limit"]),
    "scans": deque(maxlen=DEFAULTS["buffer_limit"]),
    "partial_line": "",
    "device_handle": None,
    "listener_thread": None,
    "stop_event": None,
    "keymap": {},
}


def _now_id() -> str:
    return "DRV-%s-%sZ-%s" % (
        DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _utc_ts() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value in (None, ""):
        return default
    if isinstance(value, int):
        return value
    try:
        s = str(value).strip().lower()
        if s.startswith("0x"):
            return int(s, 16)
        return int(s)
    except Exception:
        return default


def _sanitize_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(DEFAULTS)
    if isinstance(config, dict):
        merged.update(config)
    merged["report_size"] = max(1, int(merged.get("report_size", 8)))
    merged["read_timeout_ms"] = max(1, int(merged.get("read_timeout_ms", 250)))
    merged["line_timeout_ms"] = max(10, int(merged.get("line_timeout_ms", 1500)))
    merged["buffer_limit"] = max(1, int(merged.get("buffer_limit", 200)))
    merged["event_buffer_limit"] = max(1, int(merged.get("event_buffer_limit", 500)))
    merged["dedupe_window_ms"] = max(0, int(merged.get("dedupe_window_ms", 175)))
    merged["repeat_suppression"] = bool(merged.get("repeat_suppression", True))
    merged["append_enter"] = bool(merged.get("append_enter", True))
    merged["auto_append_newline"] = bool(merged.get("auto_append_newline", False))
    merged["emit_raw_reports"] = bool(merged.get("emit_raw_reports", False))
    merged["vendor_passthrough_enabled"] = bool(merged.get("vendor_passthrough_enabled", False))
    merged["decode_mode"] = str(merged.get("decode_mode", "keyboard_wedge") or "keyboard_wedge")
    merged["case_mode"] = str(merged.get("case_mode", "preserve") or "preserve")
    return merged


def _reset_buffers(cfg: Dict[str, Any]) -> None:
    _SESSION["events"] = deque(list(_SESSION.get("events", [])), maxlen=cfg["event_buffer_limit"])
    _SESSION["scans"] = deque(list(_SESSION.get("scans", [])), maxlen=cfg["buffer_limit"])


def _load_keymap(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _event(kind: str, **kwargs: Any) -> Dict[str, Any]:
    evt = {"ts": _utc_ts(), "kind": kind}
    evt.update(kwargs)
    _SESSION["events"].append(evt)
    return evt


def _get_scan_history() -> List[Dict[str, Any]]:
    return list(_SESSION["scans"])


def _append_scan(text: str, source: str = "decoded", code: Optional[str] = None, raw: Optional[List[int]] = None) -> Dict[str, Any]:
    now_ms = int(time.time() * 1000)
    dedupe = int(_SESSION["config"].get("dedupe_window_ms", 175))
    if _SESSION["config"].get("repeat_suppression", True):
        last = _SESSION.get("last_scan")
        last_ts = _SESSION.get("last_scan_ts") or 0
        if last == text and (now_ms - last_ts) <= dedupe:
            return {"ok": True, "suppressed": True, "text": text}

    item = {
        "ts": _utc_ts(),
        "text": text,
        "source": source,
        "code": code,
        "raw": raw,
    }
    _SESSION["scans"].append(item)
    _SESSION["last_scan"] = text
    _SESSION["last_scan_ts"] = now_ms
    _event("scan", text=text, source=source, code=code, raw=raw)
    return {"ok": True, "scan": item}


def _apply_case_mode(text: str) -> str:
    mode = _SESSION["config"].get("case_mode", "preserve")
    if mode == "upper":
        return text.upper()
    if mode == "lower":
        return text.lower()
    return text


def _decode_keycode(keycode: int, modifiers: int = 0) -> Optional[str]:
    shifted = bool(modifiers & 0x22) or bool(modifiers & 0x02)
    if shifted:
        if keycode in SHIFTED_KEYCODE_MAP:
            return SHIFTED_KEYCODE_MAP[keycode]
        base = KEYCODE_MAP.get(keycode)
        return base.upper() if isinstance(base, str) and len(base) == 1 and base.isalpha() else base
    return KEYCODE_MAP.get(keycode)


def _parse_keyboard_report(report: List[int]) -> Dict[str, Any]:
    if not report:
        return {"ok": False, "error": "empty_report"}
    modifiers = report[0] if len(report) > 0 else 0
    keycodes = [k for k in report[2:8] if k]
    if not keycodes:
        return {"ok": True, "keys": [], "tokens": [], "line": None}

    tokens: List[str] = []
    for keycode in keycodes:
        token = _decode_keycode(keycode, modifiers)
        if token is None:
            token = f"KEY_{keycode:02X}"
        tokens.append(token)

    completed_lines = []
    partial = _SESSION.get("partial_line", "")
    for token in tokens:
        if token in ("ENTER",):
            if partial or _SESSION["config"].get("append_enter", True):
                completed_lines.append(partial)
            partial = ""
        elif token == "SPACE":
            partial += " "
        elif token == "TAB":
            partial += "\t"
        elif token in ("BACKSPACE",):
            partial = partial[:-1]
        elif token.startswith("KEY_"):
            partial += f"[{token}]"
        elif len(token) == 1 or token.startswith("KP") or token in (
            "UP", "DOWN", "LEFT", "RIGHT", "POWER", "VOLUME_UP", "VOLUME_DOWN", "ESC"
        ):
            partial += token if len(token) == 1 else f"[{token}]"
        else:
            partial += f"[{token}]"

    _SESSION["partial_line"] = partial
    finalized = []
    for line in completed_lines:
        line = _apply_case_mode(line)
        if _SESSION["config"].get("auto_append_newline"):
            line += "\n"
        finalized.append(line)
        _append_scan(line, source="hid_keyboard_report", raw=report)

    return {"ok": True, "keys": keycodes, "tokens": tokens, "lines": finalized, "partial_line": partial}


def _parse_text_tokens(text: str) -> Dict[str, Any]:
    if text is None:
        return {"ok": False, "error": "text_required"}
    line = _apply_case_mode(str(text))
    return _append_scan(line, source="text")


def _enumerate_hid() -> List[Dict[str, Any]]:
    devices: List[Dict[str, Any]] = []
    if hid is None:
        return devices
    try:
        for item in hid.enumerate():
            dev = {
                "path": item.get("path").decode() if isinstance(item.get("path"), (bytes, bytearray)) else item.get("path"),
                "vendor_id": item.get("vendor_id"),
                "product_id": item.get("product_id"),
                "serial_number": item.get("serial_number"),
                "release_number": item.get("release_number"),
                "manufacturer_string": item.get("manufacturer_string"),
                "product_string": item.get("product_string"),
                "usage_page": item.get("usage_page"),
                "usage": item.get("usage"),
                "interface_number": item.get("interface_number"),
            }
            devices.append(dev)
    except Exception as exc:
        _SESSION["error"] = f"hid_enumerate_failed: {exc}"
    return devices


def _matches_config(dev: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    vid = _safe_int(cfg.get("vendor_id"))
    pid = _safe_int(cfg.get("product_id"))
    usage_page = _safe_int(cfg.get("usage_page"))
    usage = _safe_int(cfg.get("usage"))
    iface = _safe_int(cfg.get("interface_number"))
    path = str(cfg.get("path") or "").strip()
    product_contains = str(cfg.get("product_name_contains") or "").strip().lower()
    serial_no = str(cfg.get("serial_number") or "").strip()

    if vid is not None and dev.get("vendor_id") != vid:
        return False
    if pid is not None and dev.get("product_id") != pid:
        return False
    if usage_page is not None and dev.get("usage_page") not in (usage_page, None):
        return False
    if usage is not None and dev.get("usage") not in (usage, None):
        return False
    if iface is not None and dev.get("interface_number") not in (iface, None):
        return False
    if path and str(dev.get("path") or "") != path:
        return False
    if product_contains:
        name = f"{dev.get('manufacturer_string') or ''} {dev.get('product_string') or ''}".lower()
        if product_contains not in name:
            return False
    if serial_no and str(dev.get("serial_number") or "") != serial_no:
        return False
    return True


def _find_device(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    devices = _enumerate_hid()
    for dev in devices:
        if _matches_config(dev, cfg):
            return dev
    if cfg.get("decode_mode") == "keyboard_wedge":
        for dev in devices:
            if dev.get("usage_page") == KEYBOARD_USAGE_PAGE and dev.get("usage") == KEYBOARD_USAGE:
                return dev
    return devices[0] if devices else None


def _open_device(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if hid is None:
        return {"ok": False, "error": "hid_backend_missing"}
    dev_meta = _find_device(cfg)
    if not dev_meta:
        return {"ok": False, "error": "device_not_found"}
    try:
        handle = hid.device()
        path = dev_meta.get("path")
        if path:
            handle.open_path(path.encode() if isinstance(path, str) else path)
        else:
            handle.open(dev_meta["vendor_id"], dev_meta["product_id"])
        try:
            handle.set_nonblocking(True)
        except Exception:
            pass
        _SESSION["device_handle"] = handle
        _SESSION["connected_device"] = dev_meta
        return {"ok": True, "device": dev_meta}
    except Exception as exc:
        return {"ok": False, "error": f"open_failed: {exc}", "device": dev_meta}


def _close_device() -> None:
    handle = _SESSION.get("device_handle")
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass
    _SESSION["device_handle"] = None
    _SESSION["connected_device"] = None


def _read_report_once(report_size: Optional[int] = None, timeout_ms: Optional[int] = None) -> Dict[str, Any]:
    handle = _SESSION.get("device_handle")
    if handle is None:
        return {"ok": False, "error": "not_connected"}
    size = int(report_size or _SESSION["config"].get("report_size", 8))
    timeout = int(timeout_ms or _SESSION["config"].get("read_timeout_ms", 250))
    try:
        data = handle.read(size, timeout)
        if not data:
            return {"ok": True, "report": [], "empty": True}
        report = [int(x) & 0xFF for x in data]
        _SESSION["last_report"] = report
        evt = _event("raw_report", report=report)
        result = {"ok": True, "report": report, "event": evt}
        if _SESSION["config"].get("decode_mode") == "keyboard_wedge":
            result["decoded"] = _parse_keyboard_report(report)
        return result
    except Exception as exc:
        return {"ok": False, "error": f"read_failed: {exc}"}


def _listener_loop() -> None:
    stop_event = _SESSION.get("stop_event")
    _SESSION["listener_running"] = True
    _SESSION["status"] = "listening"
    try:
        while stop_event is not None and not stop_event.is_set():
            res = _read_report_once()
            if not res.get("ok"):
                _event("listener_error", error=res.get("error"))
                time.sleep(0.1)
                continue
            if res.get("empty"):
                time.sleep(0.01)
                continue
            if _SESSION["config"].get("emit_raw_reports"):
                _event("report_emitted", report=res.get("report"))
    finally:
        _SESSION["listener_running"] = False
        _SESSION["status"] = "ready" if _SESSION.get("instance_id") else "idle"


def _start_listener() -> Dict[str, Any]:
    if _SESSION.get("listener_running"):
        return {"ok": True, "already_running": True}
    if _SESSION.get("device_handle") is None:
        opened = _open_device(_SESSION["config"])
        if not opened.get("ok"):
            return opened
    stop_event = threading.Event()
    thread = threading.Thread(target=_listener_loop, name="ir-remote-hid-listener", daemon=True)
    _SESSION["stop_event"] = stop_event
    _SESSION["listener_thread"] = thread
    thread.start()
    return {"ok": True, "listener_running": True, "device": _SESSION.get("connected_device")}


def _stop_listener() -> Dict[str, Any]:
    stop_event = _SESSION.get("stop_event")
    thread = _SESSION.get("listener_thread")
    if stop_event is not None:
        stop_event.set()
    if thread is not None and getattr(thread, "is_alive", lambda: False)():
        thread.join(timeout=1.5)
    _SESSION["stop_event"] = None
    _SESSION["listener_thread"] = None
    _SESSION["listener_running"] = False
    return {"ok": True}


def driver_info() -> Dict[str, Any]:
    return {
        "id": DRIVER_ID,
        "name": DRIVER_NAME,
        "version": VERSION,
        "session": driver_status().get("session"),
    }


def driver_validate(config: Dict[str, Any]):
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be an object"], "warnings": []}
    cfg = _sanitize_config(config)
    warnings = []
    if hid is None:
        warnings.append("hid backend not installed; live HID reads unavailable")
    if cfg.get("map_file") and not os.path.isfile(cfg.get("map_file")):
        warnings.append("map_file not found")
    return {"ok": True, "errors": [], "warnings": warnings, "config": cfg}


def driver_init(context=None, config=None):
    cfg = _sanitize_config(config or {})
    v = driver_validate(cfg)
    if not v.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": v}

    _SESSION["instance_id"] = _now_id()
    _SESSION["started_ts"] = time.time()
    _SESSION["status"] = "ready"
    _SESSION["error"] = None
    _SESSION["meta"] = {"context": context or {}}
    _SESSION["config"] = cfg
    _SESSION["keymap"] = _load_keymap(cfg.get("map_file", ""))
    _reset_buffers(cfg)
    _SESSION["partial_line"] = ""
    return {"ok": True, "instance_id": _SESSION["instance_id"], "info": driver_info(), "validate": v}


def driver_shutdown(context=None):
    _stop_listener()
    _close_device()
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _SESSION["meta"] = {}
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    return True


def driver_status(context=None):
    session = {
        "instance_id": _SESSION.get("instance_id"),
        "started_ts": _SESSION.get("started_ts"),
        "status": _SESSION.get("status"),
        "error": _SESSION.get("error"),
        "listener_running": _SESSION.get("listener_running"),
        "listener_backend": "hidapi" if hid is not None else None,
        "connected_device": _SESSION.get("connected_device"),
        "last_scan": _SESSION.get("last_scan"),
        "last_report": _SESSION.get("last_report"),
        "scan_count": len(_SESSION.get("scans", [])),
        "event_count": len(_SESSION.get("events", [])),
        "hid_backend_available": hid is not None,
    }
    return {"ok": True, "session": session}


def driver_action(action_id: str, context=None, payload=None):
    payload = payload or {}
    action_id = str(action_id or "").strip()

    if action_id == "ping":
        return {"ok": True, "driver": DRIVER_ID, "version": VERSION, "ts": _utc_ts()}

    if action_id == "probe_backend":
        return {"ok": True, "hid_available": hid is not None}

    if action_id == "list_hid":
        devices = _enumerate_hid()
        return {"ok": True, "devices": devices, "count": len(devices), "hid_available": hid is not None}

    if action_id == "open":
        cfg = _sanitize_config({**_SESSION.get("config", {}), **payload})
        _SESSION["config"] = cfg
        opened = _open_device(cfg)
        if not opened.get("ok"):
            _SESSION["error"] = opened.get("error")
            return opened
        _SESSION["status"] = "ready"
        return opened

    if action_id == "close":
        _stop_listener()
        _close_device()
        return {"ok": True}

    if action_id == "start_listen":
        return _start_listener()

    if action_id == "stop_listen":
        return _stop_listener()

    if action_id == "read_once":
        if _SESSION.get("device_handle") is None:
            opened = _open_device(_SESSION["config"])
            if not opened.get("ok"):
                return opened
        return _read_report_once(payload.get("report_size"), payload.get("timeout_ms"))

    if action_id == "inject_report":
        report = payload.get("report")
        if not isinstance(report, list):
            return {"ok": False, "error": "report_list_required"}
        report = [int(x) & 0xFF for x in report]
        _SESSION["last_report"] = report
        _event("raw_report", report=report, synthetic=True)
        decoded = _parse_keyboard_report(report) if _SESSION["config"].get("decode_mode") == "keyboard_wedge" else None
        return {"ok": True, "report": report, "decoded": decoded}

    if action_id == "feed_keystrokes":
        text = payload.get("text", "")
        if not isinstance(text, str):
            return {"ok": False, "error": "text_required"}
        return _parse_text_tokens(text)

    if action_id == "parse_text":
        return _parse_text_tokens(payload.get("text"))

    if action_id == "get_buffer":
        return {"ok": True, "scans": _get_scan_history()}

    if action_id == "clear_buffer":
        _SESSION["scans"].clear()
        _SESSION["events"].clear()
        _SESSION["partial_line"] = ""
        return {"ok": True}

    if action_id == "get_events":
        return {"ok": True, "events": list(_SESSION.get("events", []))}

    if action_id == "get_config":
        cfg = dict(_SESSION.get("config", {}))
        return {"ok": True, "config": cfg}

    if action_id == "set_config":
        updates = payload.get("config") if isinstance(payload.get("config"), dict) else payload
        cfg = _sanitize_config({**_SESSION.get("config", {}), **(updates or {})})
        _SESSION["config"] = cfg
        _SESSION["keymap"] = _load_keymap(cfg.get("map_file", ""))
        _reset_buffers(cfg)
        return {"ok": True, "config": cfg}

    if action_id == "safe_stop":
        _stop_listener()
        _close_device()
        _SESSION["status"] = "ready" if _SESSION.get("instance_id") else "idle"
        return {"ok": True, "status": _SESSION.get("status")}

    if action_id == "vendor_passthrough":
        if not _SESSION["config"].get("vendor_passthrough_enabled"):
            return {"ok": False, "error": "vendor_passthrough_disabled"}
        return {"ok": False, "error": "vendor_passthrough_not_implemented"}

    return {
        "ok": False,
        "error": f"Action '{action_id}' not implemented by {DRIVER_ID}",
        "action_id": action_id,
    }
