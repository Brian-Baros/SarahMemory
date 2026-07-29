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
import json
import queue
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

DRIVER_ID = "com.softdev0.rfid.serial"
DRIVER_NAME = "RFID Reader/Writer Serial Driver"
VERSION = "1.0.0"

try:
    import serial  # type: ignore
    import serial.tools.list_ports  # type: ignore
    SERIAL_AVAILABLE = True
except Exception:
    serial = None
    SERIAL_AVAILABLE = False

_SESSION: Dict[str, Any] = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "meta": {},
    "connected": False,
    "port": None,
    "history": [],
    "last_result": None,
    "listener_running": False,
    "bytes_rx": 0,
    "bytes_tx": 0,
}

_STATE: Dict[str, Any] = {
    "ser": None,
    "listener_thread": None,
    "stop_event": None,
    "buffer": [],
    "partial": b"",
    "last_tag": None,
    "last_tag_ts": 0.0,
}

DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "autoload": False,
    "port": "",
    "baud": 115200,
    "bytesize": 8,
    "parity": "N",
    "stopbits": 1,
    "timeout_s": 1.0,
    "write_timeout_s": 1.0,
    "protocol_profile": "generic_ascii",
    "line_terminator": "\\n",
    "command_terminator": "\\r\\n",
    "tag_prefix": "",
    "tag_suffix": "",
    "hex_input": False,
    "read_only": True,
    "require_write_confirmation": True,
    "auto_reconnect": False,
    "poll_interval_s": 0.05,
    "buffer_limit": 500,
    "dedupe_window_ms": 750,
    "emit_raw": False,
    "allow_shell_probe": False,
}

PROFILE_COMMANDS: Dict[str, Dict[str, str]] = {
    "generic_ascii": {
        "inventory": "READ",
        "read": "READ",
        "write": "WRITE {payload}",
        "halt": "STOP",
    },
    "newline_ascii": {
        "inventory": "SCAN",
        "read": "SCAN",
        "write": "WRITE {payload}",
        "halt": "STOP",
    },
    "xbee_like_hex": {
        "inventory": "AA01",
        "read": "AA01",
        "write": "AA02{payload}",
        "halt": "AAFF",
    },
}


def _now_id() -> str:
    return "DRV-%s-%sZ-%s" % (
        DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _expand_escapes(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return bytes(text, "utf-8").decode("unicode_escape")


def _normalize_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(DEFAULTS)
    if isinstance(config, dict):
        merged.update(config)
    merged["baud"] = int(merged.get("baud", 115200) or 115200)
    merged["bytesize"] = int(merged.get("bytesize", 8) or 8)
    merged["stopbits"] = int(merged.get("stopbits", 1) or 1)
    merged["timeout_s"] = float(merged.get("timeout_s", 1.0) or 1.0)
    merged["write_timeout_s"] = float(merged.get("write_timeout_s", 1.0) or 1.0)
    merged["poll_interval_s"] = float(merged.get("poll_interval_s", 0.05) or 0.05)
    merged["buffer_limit"] = int(merged.get("buffer_limit", 500) or 500)
    merged["dedupe_window_ms"] = int(merged.get("dedupe_window_ms", 750) or 750)
    merged["line_terminator"] = _expand_escapes(merged.get("line_terminator", "\\n"))
    merged["command_terminator"] = _expand_escapes(merged.get("command_terminator", "\\r\\n"))
    merged["tag_prefix"] = str(merged.get("tag_prefix", "") or "")
    merged["tag_suffix"] = str(merged.get("tag_suffix", "") or "")
    merged["protocol_profile"] = str(merged.get("protocol_profile", "generic_ascii") or "generic_ascii")
    merged["parity"] = str(merged.get("parity", "N") or "N").upper()
    return merged


def _mask_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return dict(config)


def _record_history(kind: str, payload: Dict[str, Any]) -> None:
    hist = _SESSION.setdefault("history", [])
    hist.append({"ts": time.time(), "kind": kind, "payload": payload})
    limit = int(_SESSION.get("meta", {}).get("config", {}).get("buffer_limit", 500) or 500)
    if len(hist) > limit:
        del hist[:-limit]


def driver_info() -> Dict[str, Any]:
    return {
        "id": DRIVER_ID,
        "name": DRIVER_NAME,
        "version": VERSION,
        "session": dict(_SESSION),
        "serial_available": SERIAL_AVAILABLE,
    }


def driver_validate(config: Dict[str, Any]):
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be an object"], "warnings": []}
    cfg = _normalize_config(config)
    if cfg["protocol_profile"] not in PROFILE_COMMANDS:
        warnings.append(f"unknown protocol_profile '{cfg['protocol_profile']}', generic formatting will be used")
    if not SERIAL_AVAILABLE:
        warnings.append("pyserial not installed; serial operations unavailable")
    if cfg["baud"] <= 0:
        errors.append("baud must be > 0")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def driver_init(context=None, config=None):
    cfg = _normalize_config(config or {})
    v = driver_validate(cfg)
    if not v.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": v}
    _SESSION["instance_id"] = _now_id()
    _SESSION["started_ts"] = time.time()
    _SESSION["status"] = "ready"
    _SESSION["error"] = None
    _SESSION["meta"] = {"context": context or {}, "config": cfg}
    _SESSION["connected"] = False
    _SESSION["port"] = cfg.get("port") or None
    _SESSION["history"] = []
    _SESSION["bytes_rx"] = 0
    _SESSION["bytes_tx"] = 0
    _STATE["buffer"] = []
    _STATE["partial"] = b""
    _STATE["last_tag"] = None
    _STATE["last_tag_ts"] = 0.0
    return {"ok": True, "instance_id": _SESSION["instance_id"], "info": driver_info(), "validate": v}


def _serial_kwargs(cfg: Dict[str, Any]) -> Dict[str, Any]:
    parity = {
        "N": serial.PARITY_NONE if SERIAL_AVAILABLE else "N",
        "E": serial.PARITY_EVEN if SERIAL_AVAILABLE else "E",
        "O": serial.PARITY_ODD if SERIAL_AVAILABLE else "O",
        "M": serial.PARITY_MARK if SERIAL_AVAILABLE else "M",
        "S": serial.PARITY_SPACE if SERIAL_AVAILABLE else "S",
    }.get(cfg.get("parity", "N"), serial.PARITY_NONE if SERIAL_AVAILABLE else "N")
    stopbits = {1: serial.STOPBITS_ONE if SERIAL_AVAILABLE else 1, 2: serial.STOPBITS_TWO if SERIAL_AVAILABLE else 2}.get(
        int(cfg.get("stopbits", 1)), serial.STOPBITS_ONE if SERIAL_AVAILABLE else 1
    )
    bytesize = {
        5: serial.FIVEBITS if SERIAL_AVAILABLE else 5,
        6: serial.SIXBITS if SERIAL_AVAILABLE else 6,
        7: serial.SEVENBITS if SERIAL_AVAILABLE else 7,
        8: serial.EIGHTBITS if SERIAL_AVAILABLE else 8,
    }.get(int(cfg.get("bytesize", 8)), serial.EIGHTBITS if SERIAL_AVAILABLE else 8)
    return {
        "port": cfg.get("port"),
        "baudrate": cfg.get("baud", 115200),
        "timeout": cfg.get("timeout_s", 1.0),
        "write_timeout": cfg.get("write_timeout_s", 1.0),
        "bytesize": bytesize,
        "parity": parity,
        "stopbits": stopbits,
    }


def _ensure_serial_open(cfg: Dict[str, Any]):
    if not SERIAL_AVAILABLE:
        return {"ok": False, "error": "pyserial_unavailable"}
    if _STATE.get("ser") and getattr(_STATE["ser"], "is_open", False):
        return {"ok": True, "connected": True, "port": _SESSION.get("port")}
    port = cfg.get("port")
    if not port:
        return {"ok": False, "error": "port_required"}
    try:
        _STATE["ser"] = serial.Serial(**_serial_kwargs(cfg))
        _SESSION["connected"] = True
        _SESSION["port"] = port
        _SESSION["status"] = "connected"
        return {"ok": True, "connected": True, "port": port}
    except Exception as exc:
        _SESSION["error"] = str(exc)
        _SESSION["status"] = "error"
        return {"ok": False, "error": "connect_failed", "details": str(exc)}


def _close_serial():
    ser = _STATE.get("ser")
    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass
    _STATE["ser"] = None
    _SESSION["connected"] = False
    _SESSION["listener_running"] = False
    if _SESSION.get("status") != "stopped":
        _SESSION["status"] = "ready"


def _format_command(action: str, payload: Optional[Dict[str, Any]], cfg: Dict[str, Any]) -> bytes:
    profile = PROFILE_COMMANDS.get(cfg.get("protocol_profile"), PROFILE_COMMANDS["generic_ascii"])
    template = profile.get(action, "")
    payload = payload or {}
    data = template.format(payload=payload.get("tag_data", ""), uid=payload.get("uid", ""))
    if cfg.get("hex_input"):
        data = re.sub(r"[^0-9A-Fa-f]", "", data)
        return bytes.fromhex(data)
    return (data + cfg.get("command_terminator", "\r\n")).encode("utf-8", errors="ignore")


def _parse_tag_text(text: str, cfg: Dict[str, Any]) -> Optional[str]:
    line = text.strip()
    if not line:
        return None
    prefix = cfg.get("tag_prefix", "")
    suffix = cfg.get("tag_suffix", "")
    if prefix and line.startswith(prefix):
        line = line[len(prefix):]
    if suffix and line.endswith(suffix):
        line = line[: -len(suffix)]
    line = line.strip()
    m = re.search(r"([A-Fa-f0-9]{8,64})", line)
    return m.group(1).upper() if m else line


def _append_event(event: Dict[str, Any], cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tag = event.get("tag")
    now = time.time()
    dedupe_window_s = max(0.0, float(cfg.get("dedupe_window_ms", 750)) / 1000.0)
    if tag and tag == _STATE.get("last_tag") and (now - float(_STATE.get("last_tag_ts") or 0.0)) < dedupe_window_s:
        return None
    if tag:
        _STATE["last_tag"] = tag
        _STATE["last_tag_ts"] = now
    event["ts"] = now
    _STATE.setdefault("buffer", []).append(event)
    limit = int(cfg.get("buffer_limit", 500) or 500)
    if len(_STATE["buffer"]) > limit:
        del _STATE["buffer"][:-limit]
    _SESSION["last_result"] = event
    return event


def _drain_lines(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    ser = _STATE.get("ser")
    if not ser:
        return []
    out: List[Dict[str, Any]] = []
    term = cfg.get("line_terminator", "\n").encode("utf-8")
    try:
        waiting = int(getattr(ser, "in_waiting", 0) or 0)
        data = ser.read(waiting or 1)
    except Exception:
        return out
    if not data:
        return out
    _SESSION["bytes_rx"] += len(data)
    partial = (_STATE.get("partial") or b"") + data
    while True:
        idx = partial.find(term)
        if idx < 0:
            break
        raw_line = partial[:idx]
        partial = partial[idx + len(term):]
        try:
            text = raw_line.decode("utf-8", errors="ignore")
        except Exception:
            text = raw_line.hex().upper()
        tag = _parse_tag_text(text, cfg)
        event = {"kind": "tag", "raw": text if cfg.get("emit_raw") else None, "tag": tag}
        appended = _append_event(event, cfg)
        if appended:
            out.append(appended)
    _STATE["partial"] = partial
    return out


def _listener_loop(cfg: Dict[str, Any]):
    while _STATE.get("stop_event") and not _STATE["stop_event"].is_set():
        _drain_lines(cfg)
        time.sleep(float(cfg.get("poll_interval_s", 0.05) or 0.05))
    _SESSION["listener_running"] = False


def _start_listener(cfg: Dict[str, Any]):
    if _SESSION.get("listener_running"):
        return {"ok": True, "listener_running": True}
    conn = _ensure_serial_open(cfg)
    if not conn.get("ok"):
        return conn
    _STATE["stop_event"] = threading.Event()
    t = threading.Thread(target=_listener_loop, args=(cfg,), daemon=True, name="rfid-serial-listener")
    _STATE["listener_thread"] = t
    t.start()
    _SESSION["listener_running"] = True
    return {"ok": True, "listener_running": True}


def _stop_listener():
    ev = _STATE.get("stop_event")
    if ev is not None:
        ev.set()
    t = _STATE.get("listener_thread")
    if t is not None and t.is_alive():
        t.join(timeout=1.5)
    _STATE["listener_thread"] = None
    _STATE["stop_event"] = None
    _SESSION["listener_running"] = False
    return {"ok": True}


def driver_shutdown(context=None):
    _stop_listener()
    _close_serial()
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _SESSION["meta"] = {}
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    return True


def driver_status(context=None):
    return {
        "ok": True,
        "session": dict(_SESSION),
        "buffer_size": len(_STATE.get("buffer", [])),
        "serial_available": SERIAL_AVAILABLE,
    }


def _list_ports():
    if not SERIAL_AVAILABLE:
        return {"ok": False, "error": "pyserial_unavailable", "ports": []}
    ports = []
    for p in serial.tools.list_ports.comports():
        ports.append(
            {
                "device": p.device,
                "name": p.name,
                "description": p.description,
                "hwid": p.hwid,
                "manufacturer": getattr(p, "manufacturer", None),
                "product": getattr(p, "product", None),
                "serial_number": getattr(p, "serial_number", None),
                "vid": getattr(p, "vid", None),
                "pid": getattr(p, "pid", None),
            }
        )
    return {"ok": True, "ports": ports}


def _write_serial(data: bytes):
    ser = _STATE.get("ser")
    if ser is None or not getattr(ser, "is_open", False):
        return {"ok": False, "error": "not_connected"}
    try:
        sent = ser.write(data)
        _SESSION["bytes_tx"] += int(sent or 0)
        return {"ok": True, "bytes_sent": int(sent or 0)}
    except Exception as exc:
        return {"ok": False, "error": "write_failed", "details": str(exc)}


def driver_action(action_id: str, context=None, payload=None):
    payload = payload or {}
    cfg = _normalize_config((_SESSION.get("meta") or {}).get("config") or {})
    if action_id == "scan_ports":
        return _list_ports()
    if action_id == "get_config":
        return {"ok": True, "config": _mask_config(cfg)}
    if action_id == "set_config":
        cfg.update(payload or {})
        cfg = _normalize_config(cfg)
        _SESSION.setdefault("meta", {})["config"] = cfg
        return {"ok": True, "config": _mask_config(cfg)}
    if action_id == "connect":
        if payload:
            cfg.update(payload)
            cfg = _normalize_config(cfg)
            _SESSION.setdefault("meta", {})["config"] = cfg
        return _ensure_serial_open(cfg)
    if action_id == "disconnect":
        _stop_listener()
        _close_serial()
        return {"ok": True, "connected": False}
    if action_id == "start_listen":
        return _start_listener(cfg)
    if action_id == "stop_listen":
        return _stop_listener()
    if action_id == "read_tag":
        conn = _ensure_serial_open(cfg)
        if not conn.get("ok"):
            return conn
        cmd = _format_command("read", payload, cfg)
        write_res = _write_serial(cmd)
        if not write_res.get("ok"):
            return write_res
        deadline = time.time() + float(payload.get("wait_s", cfg.get("timeout_s", 1.0)))
        events: List[Dict[str, Any]] = []
        while time.time() < deadline:
            events.extend(_drain_lines(cfg))
            if events:
                break
            time.sleep(float(cfg.get("poll_interval_s", 0.05) or 0.05))
        return {"ok": bool(events), "events": events, "write": write_res}
    if action_id == "inventory":
        conn = _ensure_serial_open(cfg)
        if not conn.get("ok"):
            return conn
        cmd = _format_command("inventory", payload, cfg)
        write_res = _write_serial(cmd)
        if not write_res.get("ok"):
            return write_res
        time.sleep(float(payload.get("settle_s", 0.2) or 0.2))
        events = _drain_lines(cfg)
        return {"ok": True, "events": events, "write": write_res}
    if action_id == "write_tag":
        if bool(cfg.get("read_only", True)):
            return {"ok": False, "error": "read_only"}
        if bool(cfg.get("require_write_confirmation", True)) and not bool(payload.get("confirm_write")):
            return {"ok": False, "error": "write_confirmation_required"}
        conn = _ensure_serial_open(cfg)
        if not conn.get("ok"):
            return conn
        cmd = _format_command("write", payload, cfg)
        return _write_serial(cmd)
    if action_id == "send_raw":
        if bool(cfg.get("read_only", True)) and not bool(payload.get("allow_in_read_only")):
            return {"ok": False, "error": "read_only"}
        raw = payload.get("raw")
        if raw is None:
            return {"ok": False, "error": "raw_required"}
        conn = _ensure_serial_open(cfg)
        if not conn.get("ok"):
            return conn
        if isinstance(raw, str):
            if bool(payload.get("hex", False)):
                raw_bytes = bytes.fromhex(re.sub(r"[^0-9A-Fa-f]", "", raw))
            else:
                raw_bytes = raw.encode("utf-8", errors="ignore")
        elif isinstance(raw, (bytes, bytearray)):
            raw_bytes = bytes(raw)
        else:
            return {"ok": False, "error": "invalid_raw_type"}
        return _write_serial(raw_bytes)
    if action_id == "get_buffer":
        return {"ok": True, "buffer": list(_STATE.get("buffer", []))}
    if action_id == "clear_buffer":
        _STATE["buffer"] = []
        return {"ok": True}
    if action_id == "feed_text":
        text = str(payload.get("text", "") or "")
        events = []
        for line in re.split(r"\r?\n", text):
            tag = _parse_tag_text(line, cfg)
            if tag:
                ev = _append_event({"kind": "tag", "raw": line if cfg.get("emit_raw") else None, "tag": tag}, cfg)
                if ev:
                    events.append(ev)
        return {"ok": True, "events": events}
    if action_id == "safe_stop":
        _stop_listener()
        try:
            if _STATE.get("ser") and getattr(_STATE["ser"], "is_open", False):
                _write_serial(_format_command("halt", payload, cfg))
            return {"ok": True}
        finally:
            _close_serial()
    return {"ok": False, "error": f"Action '{action_id}' not implemented by {DRIVER_ID}", "action_id": action_id}
