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
from __future__ import annotations

import copy
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Deque, Dict, Iterable, List, Optional

DRIVER_ID = "com.softdev0.barcode.hid"
DRIVER_NAME = "Barcode Scanner HID Driver"
VERSION = "1.0.0"

try:
    import hid  # type: ignore
except Exception:  # pragma: no cover
    hid = None

DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "autoload": False,
    "vendor_id": "",
    "product_id": "",
    "usage_page": "",
    "usage": "",
    "interface_number": "",
    "path": "",
    "read_size": 64,
    "read_timeout_ms": 120,
    "line_timeout_ms": 80,
    "suffix": "\n",
    "prefix": "",
    "charset": "utf-8",
    "strict_decode": False,
    "trim_whitespace": True,
    "dedupe_window_ms": 400,
    "max_buffer": 200,
    "keyboard_wedge": True,
    "case_mode": "preserve",
}

KEYCODE_MAP: Dict[int, tuple[str, str]] = {
    4: ("a", "A"), 5: ("b", "B"), 6: ("c", "C"), 7: ("d", "D"), 8: ("e", "E"),
    9: ("f", "F"), 10: ("g", "G"), 11: ("h", "H"), 12: ("i", "I"), 13: ("j", "J"),
    14: ("k", "K"), 15: ("l", "L"), 16: ("m", "M"), 17: ("n", "N"), 18: ("o", "O"),
    19: ("p", "P"), 20: ("q", "Q"), 21: ("r", "R"), 22: ("s", "S"), 23: ("t", "T"),
    24: ("u", "U"), 25: ("v", "V"), 26: ("w", "W"), 27: ("x", "X"), 28: ("y", "Y"),
    29: ("z", "Z"),
    30: ("1", "!"), 31: ("2", "@"), 32: ("3", "#"), 33: ("4", "$"), 34: ("5", "%"),
    35: ("6", "^"), 36: ("7", "&"), 37: ("8", "*"), 38: ("9", "("), 39: ("0", ")"),
    40: ("\n", "\n"), 41: ("\x1b", "\x1b"), 42: ("\b", "\b"), 43: ("\t", "\t"), 44: (" ", " "),
    45: ("-", "_"), 46: ("=", "+"), 47: ("[", "{"), 48: ("]", "}"), 49: ("\\", "|"),
    51: (";", ":"), 52: ("'", '"'), 53: ("`", "~"), 54: (",", "<"), 55: (".", ">"), 56: ("/", "?"),
}

_SESSION: Dict[str, Any] = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "meta": {},
    "listener": {
        "active": False,
        "device_open": False,
        "thread": None,
        "selected_device": None,
        "last_scan": None,
        "last_scan_ts": None,
        "last_error": None,
        "reports_seen": 0,
        "scan_count": 0,
    },
    "config": copy.deepcopy(DEFAULTS),
    "buffer": deque(maxlen=DEFAULTS["max_buffer"]),
    "notes": [],
}

_LOCK = threading.RLock()
_STOP_EVENT = threading.Event()
_LISTENER_THREAD: Optional[threading.Thread] = None
_DEVICE_HANDLE = None
_PARTIAL_TEXT = ""
_PARTIAL_TS: Optional[float] = None
_LAST_SCAN_TEXT = ""
_LAST_SCAN_TS = 0.0


def _now_id() -> str:
    return "DRV-%s-%sZ-%s" % (
        DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _utc_iso_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _masked_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return dict(cfg)


def _session_snapshot() -> Dict[str, Any]:
    snap = copy.deepcopy(_SESSION)
    snap["buffer"] = list(_SESSION["buffer"])
    return snap


def _add_note(message: str) -> None:
    with _LOCK:
        _SESSION["notes"].append({"ts": _utc_iso_now(), "message": str(message)})
        _SESSION["notes"] = _SESSION["notes"][-50:]


def _normalize_intish(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return ""
    try:
        if text.lower().startswith("0x"):
            return int(text, 16)
        return int(text)
    except Exception:
        return value


def _normalize_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = copy.deepcopy(DEFAULTS)
    if isinstance(config, dict):
        merged.update(config)
    for key in ("vendor_id", "product_id", "usage_page", "usage", "interface_number"):
        merged[key] = _normalize_intish(merged.get(key))
    for key in ("read_size", "read_timeout_ms", "line_timeout_ms", "dedupe_window_ms", "max_buffer"):
        try:
            merged[key] = int(merged.get(key))
        except Exception:
            merged[key] = DEFAULTS[key]
    merged["enabled"] = bool(merged.get("enabled", True))
    merged["autoload"] = bool(merged.get("autoload", False))
    merged["keyboard_wedge"] = bool(merged.get("keyboard_wedge", True))
    merged["strict_decode"] = bool(merged.get("strict_decode", False))
    merged["trim_whitespace"] = bool(merged.get("trim_whitespace", True))
    merged["charset"] = str(merged.get("charset") or "utf-8")
    merged["suffix"] = str(merged.get("suffix") or "\n")
    merged["prefix"] = str(merged.get("prefix") or "")
    merged["path"] = str(merged.get("path") or "")
    merged["case_mode"] = str(merged.get("case_mode") or "preserve").lower()
    if merged["case_mode"] not in {"preserve", "upper", "lower"}:
        merged["case_mode"] = "preserve"
    merged["read_size"] = max(8, min(2048, merged["read_size"]))
    merged["read_timeout_ms"] = max(10, min(10000, merged["read_timeout_ms"]))
    merged["line_timeout_ms"] = max(5, min(30000, merged["line_timeout_ms"]))
    merged["dedupe_window_ms"] = max(0, min(600000, merged["dedupe_window_ms"]))
    merged["max_buffer"] = max(1, min(10000, merged["max_buffer"]))
    return merged


def driver_info() -> Dict[str, Any]:
    return {
        "id": DRIVER_ID,
        "name": DRIVER_NAME,
        "version": VERSION,
        "session": _session_snapshot(),
        "hid_available": hid is not None,
    }


def driver_validate(config: dict) -> Dict[str, Any]:
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be an object"], "warnings": []}
    cfg = _normalize_config(config)
    warnings: List[str] = []
    if hid is None:
        warnings.append("hid library not available; live HID enumeration/listening disabled")
    if not cfg.get("enabled", True):
        warnings.append("driver is disabled")
    if not cfg.get("path") and not any(cfg.get(k) not in (None, "") for k in ("vendor_id", "product_id")):
        warnings.append("no device selector configured; auto-selection may match first keyboard-wedge HID")
    return {"ok": True, "errors": [], "warnings": warnings, "normalized": cfg}


def _enumerate_devices() -> List[Dict[str, Any]]:
    if hid is None:
        return []
    items = hid.enumerate() or []
    devices: List[Dict[str, Any]] = []
    for item in items:
        devices.append({
            "vendor_id": item.get("vendor_id"),
            "product_id": item.get("product_id"),
            "serial_number": item.get("serial_number"),
            "release_number": item.get("release_number"),
            "manufacturer_string": item.get("manufacturer_string"),
            "product_string": item.get("product_string"),
            "usage_page": item.get("usage_page"),
            "usage": item.get("usage"),
            "interface_number": item.get("interface_number"),
            "path": item.get("path").decode(errors="ignore") if isinstance(item.get("path"), (bytes, bytearray)) else item.get("path"),
        })
    return devices


def _device_matches(device: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    for key in ("vendor_id", "product_id", "usage_page", "usage", "interface_number"):
        wanted = cfg.get(key)
        if wanted in (None, ""):
            continue
        if device.get(key) != wanted:
            return False
    path = cfg.get("path")
    if path and str(device.get("path") or "") != path:
        return False
    return True


def _select_device(cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    devices = _enumerate_devices()
    matched = [d for d in devices if _device_matches(d, cfg)]
    if matched:
        return matched[0]
    if cfg.get("keyboard_wedge"):
        for d in devices:
            if d.get("usage_page") == 0x01 and d.get("usage") == 0x06:
                return d
    return devices[0] if devices and not cfg.get("path") and not cfg.get("vendor_id") and not cfg.get("product_id") else None


def _apply_case_mode(text: str, cfg: Dict[str, Any]) -> str:
    mode = cfg.get("case_mode", "preserve")
    if mode == "upper":
        return text.upper()
    if mode == "lower":
        return text.lower()
    return text


def _finalize_scan(text: str, source: str = "hid") -> Optional[Dict[str, Any]]:
    global _LAST_SCAN_TEXT, _LAST_SCAN_TS
    cfg = _SESSION["config"]
    value = text
    if cfg.get("trim_whitespace", True):
        value = value.strip()
    value = _apply_case_mode(value, cfg)
    if cfg.get("prefix") and value.startswith(cfg["prefix"]):
        value = value[len(cfg["prefix"]):]
    suffix = cfg.get("suffix") or ""
    if suffix and value.endswith(suffix):
        value = value[: -len(suffix)]
    if not value:
        return None
    now = time.time()
    if value == _LAST_SCAN_TEXT and (now - _LAST_SCAN_TS) * 1000 <= int(cfg.get("dedupe_window_ms", 0)):
        return {"ok": True, "duplicate": True, "text": value, "source": source}
    item = {"ts": _utc_iso_now(), "text": value, "source": source, "duplicate": False}
    with _LOCK:
        _SESSION["buffer"].append(item)
        _SESSION["listener"]["last_scan"] = value
        _SESSION["listener"]["last_scan_ts"] = item["ts"]
        _SESSION["listener"]["scan_count"] += 1
    _LAST_SCAN_TEXT = value
    _LAST_SCAN_TS = now
    return {"ok": True, "item": item}


def _hid_report_to_text(report: Iterable[int]) -> str:
    values = list(report)
    if not values or len(values) < 3:
        return ""
    modifier = values[0]
    keycodes = values[2:]
    shift = bool(modifier & 0x22)
    chars: List[str] = []
    for code in keycodes:
        if not code:
            continue
        pair = KEYCODE_MAP.get(int(code))
        if pair:
            chars.append(pair[1] if shift else pair[0])
    return "".join(chars)


def _ingest_text(text: str, source: str = "hid") -> Dict[str, Any]:
    global _PARTIAL_TEXT, _PARTIAL_TS
    cfg = _SESSION["config"]
    suffix = cfg.get("suffix") or "\n"
    now = time.time()
    if _PARTIAL_TS is not None and (now - _PARTIAL_TS) * 1000 > cfg.get("line_timeout_ms", 80) and _PARTIAL_TEXT:
        _finalize_scan(_PARTIAL_TEXT, source=source)
        _PARTIAL_TEXT = ""
    _PARTIAL_TS = now
    _PARTIAL_TEXT += text
    emitted: List[Dict[str, Any]] = []
    while suffix and suffix in _PARTIAL_TEXT:
        idx = _PARTIAL_TEXT.find(suffix)
        chunk = _PARTIAL_TEXT[:idx]
        _PARTIAL_TEXT = _PARTIAL_TEXT[idx + len(suffix):]
        out = _finalize_scan(chunk, source=source)
        if out and not out.get("duplicate"):
            emitted.append(out["item"])
    return {"ok": True, "emitted": emitted, "partial": _PARTIAL_TEXT}


def _open_selected_device(cfg: Dict[str, Any]):
    if hid is None:
        raise RuntimeError("hid library not available")
    selected = _select_device(cfg)
    if not selected:
        raise RuntimeError("no matching HID device found")
    device = hid.device()
    path = selected.get("path")
    if path:
        encoded = path.encode() if isinstance(path, str) else path
        device.open_path(encoded)
    else:
        device.open(int(selected["vendor_id"]), int(selected["product_id"]))
    try:
        device.set_nonblocking(True)
    except Exception:
        pass
    with _LOCK:
        _SESSION["listener"]["selected_device"] = selected
        _SESSION["listener"]["device_open"] = True
    return device


def _listener_loop() -> None:
    global _DEVICE_HANDLE
    cfg = _SESSION["config"]
    with _LOCK:
        _SESSION["listener"]["active"] = True
        _SESSION["status"] = "listening"
        _SESSION["listener"]["last_error"] = None
    try:
        _DEVICE_HANDLE = _open_selected_device(cfg)
        _add_note("barcode HID listener started")
        while not _STOP_EVENT.is_set():
            try:
                report = _DEVICE_HANDLE.read(cfg["read_size"], cfg["read_timeout_ms"])
            except TypeError:
                report = _DEVICE_HANDLE.read(cfg["read_size"])
            if not report:
                continue
            with _LOCK:
                _SESSION["listener"]["reports_seen"] += 1
            text = _hid_report_to_text(report)
            if text:
                _ingest_text(text, source="hid")
    except Exception as exc:
        with _LOCK:
            _SESSION["error"] = str(exc)
            _SESSION["listener"]["last_error"] = str(exc)
            _SESSION["status"] = "error"
        _add_note(f"listener error: {exc}")
    finally:
        if _DEVICE_HANDLE is not None:
            try:
                _DEVICE_HANDLE.close()
            except Exception:
                pass
        with _LOCK:
            _SESSION["listener"]["device_open"] = False
            _SESSION["listener"]["active"] = False
            if _SESSION["status"] != "error":
                _SESSION["status"] = "ready"
        _DEVICE_HANDLE = None
        _add_note("barcode HID listener stopped")


def _start_listener() -> Dict[str, Any]:
    global _LISTENER_THREAD
    if hid is None:
        return {"ok": False, "error": "hid library not available"}
    if _LISTENER_THREAD and _LISTENER_THREAD.is_alive():
        return {"ok": True, "already_running": True, "status": driver_status()}
    _STOP_EVENT.clear()
    _LISTENER_THREAD = threading.Thread(target=_listener_loop, name="barcode-hid-listener", daemon=True)
    _LISTENER_THREAD.start()
    return {"ok": True, "started": True}


def _stop_listener() -> Dict[str, Any]:
    global _LISTENER_THREAD
    _STOP_EVENT.set()
    if _LISTENER_THREAD and _LISTENER_THREAD.is_alive():
        _LISTENER_THREAD.join(timeout=2.5)
    _LISTENER_THREAD = None
    return {"ok": True, "stopped": True}


def driver_init(context=None, config=None) -> Dict[str, Any]:
    config = config or {}
    validation = driver_validate(config)
    if not validation.get("ok"):
        with _LOCK:
            _SESSION["status"] = "error"
            _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": validation}
    cfg = validation["normalized"]
    with _LOCK:
        _SESSION["instance_id"] = _now_id()
        _SESSION["started_ts"] = time.time()
        _SESSION["status"] = "ready"
        _SESSION["error"] = None
        _SESSION["meta"] = {"context": context or {}, "config": config}
        _SESSION["config"] = cfg
        _SESSION["buffer"] = deque(list(_SESSION["buffer"]), maxlen=cfg["max_buffer"])
        _SESSION["listener"]["selected_device"] = None
        _SESSION["listener"]["last_error"] = None
    _add_note("driver initialized")
    return {"ok": True, "instance_id": _SESSION["instance_id"], "info": driver_info(), "validate": {k: v for k, v in validation.items() if k != "normalized"}}


def driver_shutdown(context=None) -> bool:
    _stop_listener()
    with _LOCK:
        _SESSION["status"] = "stopped"
        _SESSION["error"] = None
        _SESSION["meta"] = {}
        _SESSION["instance_id"] = None
        _SESSION["started_ts"] = None
        _SESSION["listener"]["selected_device"] = None
    _add_note("driver shutdown")
    return True


def driver_status(context=None) -> Dict[str, Any]:
    return {"ok": True, "session": _session_snapshot(), "hid_available": hid is not None, "devices": _enumerate_devices() if hid is not None else [], "config": _masked_config(_SESSION["config"])}


def _payload_config(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return _SESSION["config"]
    merged = copy.deepcopy(_SESSION["config"])
    if isinstance(payload.get("config"), dict):
        merged.update(payload["config"])
    for key in DEFAULTS:
        if key in payload:
            merged[key] = payload[key]
    return _normalize_config(merged)


def driver_action(action_id: str, context=None, payload=None) -> Dict[str, Any]:
    payload = payload or {}
    if action_id == "ping":
        return {"ok": True, "driver": DRIVER_ID, "version": VERSION, "ts": _utc_iso_now()}
    if action_id == "list_hid":
        return {"ok": True, "devices": _enumerate_devices(), "hid_available": hid is not None}
    if action_id == "select_device":
        cfg = _payload_config(payload)
        selected = _select_device(cfg)
        return {"ok": selected is not None, "selected_device": selected, "config": _masked_config(cfg)}
    if action_id == "start_listen":
        cfg = _payload_config(payload)
        with _LOCK:
            _SESSION["config"] = cfg
            _SESSION["buffer"] = deque(list(_SESSION["buffer"]), maxlen=cfg["max_buffer"])
        return _start_listener()
    if action_id == "stop_listen":
        return _stop_listener()
    if action_id == "read_once":
        if hid is None:
            return {"ok": False, "error": "hid library not available"}
        cfg = _payload_config(payload)
        try:
            device = _open_selected_device(cfg)
            try:
                report = device.read(cfg["read_size"], max(cfg["read_timeout_ms"], 300))
            except TypeError:
                report = device.read(cfg["read_size"])
            finally:
                try:
                    device.close()
                except Exception:
                    pass
            text = _hid_report_to_text(report or [])
            result = _ingest_text(text, source="hid-read-once") if text else {"ok": True, "emitted": [], "partial": _PARTIAL_TEXT}
            return {"ok": True, "report": list(report or []), "text": text, "result": result}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
    if action_id == "inject_report":
        report = payload.get("report") if isinstance(payload, dict) else None
        if not isinstance(report, list):
            return {"ok": False, "error": "payload.report must be a list of integers"}
        text = _hid_report_to_text(report)
        return {"ok": True, "text": text, "result": _ingest_text(text, source="inject_report")}
    if action_id == "feed_keystrokes":
        text = str(payload.get("text") or "")
        if not text:
            return {"ok": False, "error": "payload.text required"}
        return {"ok": True, "result": _ingest_text(text, source="feed_keystrokes")}
    if action_id == "parse_text":
        text = str(payload.get("text") or "")
        return {"ok": True, "result": _ingest_text(text, source="parse_text")}
    if action_id == "get_buffer":
        return {"ok": True, "items": list(_SESSION["buffer"]), "count": len(_SESSION["buffer"])}
    if action_id == "clear_buffer":
        with _LOCK:
            _SESSION["buffer"].clear()
        return {"ok": True, "cleared": True}
    if action_id == "get_config":
        return {"ok": True, "config": _masked_config(_SESSION["config"])}
    if action_id == "set_config":
        cfg = _payload_config(payload)
        with _LOCK:
            _SESSION["config"] = cfg
            _SESSION["buffer"] = deque(list(_SESSION["buffer"]), maxlen=cfg["max_buffer"])
        return {"ok": True, "config": _masked_config(cfg)}
    if action_id == "safe_stop":
        stop = _stop_listener()
        return {"ok": True, "stopped": stop.get("ok", False)}
    return {"ok": False, "error": f"Action '{action_id}' not implemented by {DRIVER_ID}", "action_id": action_id}
