import json
import os
import queue
import re
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import serial  # type: ignore
    from serial.tools import list_ports  # type: ignore
except Exception:
    serial = None
    list_ports = None

_DRIVER_ID = "com.softdev0.cnc.grbl.serial"
_DRIVER_NAME = "CNC GRBL Controller Driver"
_DRIVER_VERSION = "1.0.0"
_MOTION_RE = re.compile(r"^\s*(G0|G1|G2|G3|\$J=|\$H|!|~|\x18)", re.IGNORECASE)
_AXES = {"x", "y", "z", "a", "b", "c"}

_SESSION: Dict[str, Any] = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "notes": [],
    "config": {},
    "connected": False,
    "port": None,
    "baud": None,
    "controller": {},
    "last_status": None,
    "last_response": None,
    "last_alarm": None,
    "queue_depth": 0,
    "history": [],
    "last_command_ts": None,
}

_SERIAL = None
_READER_THREAD = None
_STOP_READER = None
_RX_QUEUE: "queue.Queue[str]" = queue.Queue()
_LINE_BUFFER: List[str] = []
_LOCK = threading.RLock()

_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "autoload": False,
    "port": "",
    "baud": 115200,
    "startup_delay_ms": 1800,
    "read_timeout": 0.25,
    "write_timeout": 2.0,
    "status_timeout": 1.5,
    "response_timeout": 4.0,
    "stream_delay_ms": 25,
    "soft_reset_on_connect": False,
    "auto_unlock_on_connect": False,
    "require_confirm_motion": True,
    "require_confirm_home": True,
    "safe_z_mm": 5.0,
    "jog_feed_mm_min": 250.0,
    "spindle_off_on_safe_stop": True,
    "coolant_off_on_safe_stop": True,
    "keep_history": 200,
}


def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def _new_instance_id() -> str:
    return f"DRV-{_DRIVER_ID}-{_utc_stamp()}-{hex(int(time.time()*1000))[-4:].upper()}"


def _note(message: str) -> None:
    with _LOCK:
        _SESSION.setdefault("notes", []).append(message)
        _SESSION["notes"] = _SESSION["notes"][-50:]


def _history(entry: Dict[str, Any]) -> None:
    with _LOCK:
        hist = _SESSION.setdefault("history", [])
        hist.append(entry)
        keep = int(_SESSION.get("config", {}).get("keep_history", _DEFAULTS["keep_history"]))
        _SESSION["history"] = hist[-max(10, keep):]


def _mask_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return dict(config or {})


def _merged_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(_DEFAULTS)
    if isinstance(config, dict):
        merged.update(config)
    return merged


def driver_info() -> Dict[str, Any]:
    return {
        "id": _DRIVER_ID,
        "name": _DRIVER_NAME,
        "version": _DRIVER_VERSION,
        "transport": "serial",
        "session": dict(_SESSION),
        "actions": [
            "ping", "list_ports", "connect", "disconnect", "get_status", "query_status",
            "unlock", "home", "safe_stop", "feed_hold", "resume", "soft_reset",
            "send_gcode", "send_realtime", "run_macro", "jog", "set_zero", "goto_zero",
            "get_settings", "get_parameters", "get_parser_state", "drain_buffer", "get_config", "set_config"
        ],
    }


def driver_validate(config: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be a JSON object"], "warnings": warnings}
    for key in ("enabled", "autoload", "soft_reset_on_connect", "auto_unlock_on_connect", "require_confirm_motion", "require_confirm_home", "spindle_off_on_safe_stop", "coolant_off_on_safe_stop"):
        if key in config and not isinstance(config.get(key), bool):
            errors.append(f"{key} must be a boolean")
    for key in ("baud",):
        if key in config and not isinstance(config.get(key), int):
            errors.append(f"{key} must be an integer")
    for key in ("startup_delay_ms", "stream_delay_ms"):
        if key in config and not isinstance(config.get(key), (int, float)):
            errors.append(f"{key} must be numeric")
    for key in ("read_timeout", "write_timeout", "status_timeout", "response_timeout", "safe_z_mm", "jog_feed_mm_min"):
        if key in config and not isinstance(config.get(key), (int, float)):
            errors.append(f"{key} must be numeric")
    if config.get("port") is not None and not isinstance(config.get("port"), str):
        errors.append("port must be a string")
    if serial is None:
        warnings.append("pyserial is not installed; live GRBL serial control is unavailable")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def _serial_available() -> bool:
    return serial is not None


def _reader_loop() -> None:
    global _SERIAL
    while _STOP_READER and not _STOP_READER.is_set():
        ser = _SERIAL
        if ser is None:
            break
        try:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            _RX_QUEUE.put(line)
            with _LOCK:
                _SESSION["last_response"] = line
                if line.startswith("<") and line.endswith(">"):
                    _SESSION["last_status"] = _parse_status(line)
                if line.lower().startswith("alarm"):
                    _SESSION["last_alarm"] = line
            _history({"ts": time.time(), "dir": "rx", "line": line})
        except Exception as exc:
            _note(f"reader_error: {exc}")
            with _LOCK:
                _SESSION["error"] = f"reader_error: {exc}"
            time.sleep(0.1)


def _clear_rx_queue() -> None:
    while True:
        try:
            _RX_QUEUE.get_nowait()
        except queue.Empty:
            return


def _read_until(predicate, timeout: float) -> List[str]:
    deadline = time.time() + max(0.05, timeout)
    lines: List[str] = []
    while time.time() < deadline:
        remaining = max(0.01, deadline - time.time())
        try:
            line = _RX_QUEUE.get(timeout=min(0.2, remaining))
            lines.append(line)
            if predicate(line, lines):
                break
        except queue.Empty:
            continue
    return lines


def _send_raw(data: bytes) -> None:
    global _SERIAL
    if _SERIAL is None:
        raise RuntimeError("serial_not_connected")
    _SERIAL.write(data)
    _SERIAL.flush()


def _send_line(line: str) -> None:
    clean = line.strip()
    if not clean:
        return
    _send_raw((clean + "\n").encode("utf-8"))
    with _LOCK:
        _SESSION["last_command_ts"] = time.time()
    _history({"ts": time.time(), "dir": "tx", "line": clean})


def _parse_status(line: str) -> Dict[str, Any]:
    body = line.strip()[1:-1]
    parts = body.split("|")
    out: Dict[str, Any] = {"raw": line, "state": parts[0] if parts else None}
    for part in parts[1:]:
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        out[k] = v
    return out


def _list_serial_ports() -> List[Dict[str, Any]]:
    if list_ports is None:
        return []
    results = []
    for port in list_ports.comports():
        results.append({
            "device": port.device,
            "name": getattr(port, "name", None),
            "description": getattr(port, "description", None),
            "hwid": getattr(port, "hwid", None),
            "vid": getattr(port, "vid", None),
            "pid": getattr(port, "pid", None),
            "serial_number": getattr(port, "serial_number", None),
            "manufacturer": getattr(port, "manufacturer", None),
            "product": getattr(port, "product", None),
            "location": getattr(port, "location", None),
            "interface": getattr(port, "interface", None),
        })
    return results


def _connect(config: Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global _SERIAL, _READER_THREAD, _STOP_READER
    payload = payload or {}
    if not _serial_available():
        return {"ok": False, "error": "pyserial_not_installed"}
    if _SERIAL is not None:
        return {"ok": True, "connected": True, "port": _SESSION.get("port"), "baud": _SESSION.get("baud")}

    cfg = dict(config)
    port = str(payload.get("port") or cfg.get("port") or "").strip()
    baud = int(payload.get("baud") or cfg.get("baud") or 115200)
    if not port:
        return {"ok": False, "error": "missing_port"}

    try:
        _SERIAL = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=float(cfg.get("read_timeout", 0.25)),
            write_timeout=float(cfg.get("write_timeout", 2.0)),
        )
        startup_delay_ms = float(cfg.get("startup_delay_ms", 1800))
        time.sleep(max(0.0, startup_delay_ms) / 1000.0)
        _clear_rx_queue()
        _STOP_READER = threading.Event()
        _READER_THREAD = threading.Thread(target=_reader_loop, name="grbl-reader", daemon=True)
        _READER_THREAD.start()
        with _LOCK:
            _SESSION["connected"] = True
            _SESSION["port"] = port
            _SESSION["baud"] = baud
            _SESSION["status"] = "connected"
        _note(f"connected to {port} @ {baud}")
        if cfg.get("soft_reset_on_connect"):
            _soft_reset(config=cfg)
        hello = _read_until(lambda line, _: line.lower().startswith("grbl") or line == "ok", float(cfg.get("response_timeout", 4.0)))
        if cfg.get("auto_unlock_on_connect"):
            _unlock(config=cfg)
        return {"ok": True, "connected": True, "port": port, "baud": baud, "hello": hello, "status": driver_status()}
    except Exception as exc:
        _SERIAL = None
        with _LOCK:
            _SESSION["connected"] = False
            _SESSION["status"] = "error"
            _SESSION["error"] = f"connect_failed: {exc}"
        return {"ok": False, "error": "connect_failed", "detail": str(exc)}


def _disconnect() -> Dict[str, Any]:
    global _SERIAL, _READER_THREAD, _STOP_READER
    try:
        if _STOP_READER is not None:
            _STOP_READER.set()
        if _SERIAL is not None:
            _SERIAL.close()
        if _READER_THREAD is not None and _READER_THREAD.is_alive():
            _READER_THREAD.join(timeout=0.75)
    finally:
        _SERIAL = None
        _READER_THREAD = None
        _STOP_READER = None
        with _LOCK:
            _SESSION["connected"] = False
            _SESSION["status"] = "ready"
        _note("disconnected")
    return {"ok": True, "connected": False}


def _require_connected() -> Optional[Dict[str, Any]]:
    if _SERIAL is None:
        return {"ok": False, "error": "serial_not_connected"}
    return None


def _is_motion_command(text: str) -> bool:
    return bool(_MOTION_RE.match(text or ""))


def _send_gcode_lines(lines: List[str], config: Dict[str, Any], expect_ok: bool = True) -> Dict[str, Any]:
    err = _require_connected()
    if err:
        return err
    timeout = float(config.get("response_timeout", 4.0))
    stream_delay = max(0.0, float(config.get("stream_delay_ms", 25.0)) / 1000.0)
    responses = []
    sent = []
    for line in lines:
        if not line or not line.strip():
            continue
        _send_line(line)
        sent.append(line.strip())
        if expect_ok:
            recv = _read_until(lambda item, _: item == "ok" or item.lower().startswith("error:") or item.lower().startswith("alarm:"), timeout)
            responses.append(recv)
            if recv and (recv[-1].lower().startswith("error:") or recv[-1].lower().startswith("alarm:")):
                return {"ok": False, "sent": sent, "responses": responses, "error": recv[-1]}
        time.sleep(stream_delay)
    return {"ok": True, "sent": sent, "responses": responses}


def _query_status(config: Dict[str, Any]) -> Dict[str, Any]:
    err = _require_connected()
    if err:
        return err
    _send_raw(b"?")
    lines = _read_until(lambda line, _: line.startswith("<") and line.endswith(">"), float(config.get("status_timeout", 1.5)))
    status = None
    for line in reversed(lines):
        if line.startswith("<") and line.endswith(">"):
            status = _parse_status(line)
            break
    with _LOCK:
        _SESSION["last_status"] = status or _SESSION.get("last_status")
    return {"ok": status is not None, "status": status, "lines": lines}


def _soft_reset(config: Dict[str, Any]) -> Dict[str, Any]:
    err = _require_connected()
    if err:
        return err
    _send_raw(b"\x18")
    lines = _read_until(lambda line, _: line.lower().startswith("grbl"), float(config.get("response_timeout", 4.0)))
    return {"ok": True, "lines": lines}


def _unlock(config: Dict[str, Any]) -> Dict[str, Any]:
    return _send_gcode_lines(["$X"], config)


def _home(config: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    if config.get("require_confirm_home", True) and not bool(payload.get("confirm")):
        return {"ok": False, "error": "confirmation_required", "detail": "Set payload.confirm=true to home axes."}
    return _send_gcode_lines(["$H"], config, expect_ok=False)


def _safe_stop(config: Dict[str, Any]) -> Dict[str, Any]:
    err = _require_connected()
    if err:
        return err
    out = {"steps": []}
    _send_raw(b"!")
    out["steps"].append("feed_hold")
    time.sleep(0.1)
    if bool(config.get("spindle_off_on_safe_stop", True)):
        _send_line("M5")
        out["steps"].append("spindle_off")
    if bool(config.get("coolant_off_on_safe_stop", True)):
        _send_line("M9")
        out["steps"].append("coolant_off")
    return {"ok": True, **out}


def _jog(config: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    if config.get("require_confirm_motion", True) and not bool(payload.get("confirm")):
        return {"ok": False, "error": "confirmation_required", "detail": "Set payload.confirm=true to jog."}
    axis = str(payload.get("axis") or "").lower()
    distance = payload.get("distance_mm")
    feed = float(payload.get("feed_mm_min") or config.get("jog_feed_mm_min", 250.0))
    if axis not in _AXES:
        return {"ok": False, "error": "invalid_axis"}
    if not isinstance(distance, (int, float)):
        return {"ok": False, "error": "invalid_distance_mm"}
    line = f"$J=G91 {axis.upper()}{float(distance):.3f} F{feed:.3f}"
    return _send_gcode_lines([line], config, expect_ok=True)


def _set_zero(config: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    axes = payload.get("axes") or ["X", "Y", "Z"]
    parts = []
    for axis in axes:
        a = str(axis).upper()
        if a.lower() in _AXES:
            parts.append(f"{a}0")
    if not parts:
        return {"ok": False, "error": "no_axes_selected"}
    return _send_gcode_lines(["G10 L20 P1 " + " ".join(parts)], config)


def _goto_zero(config: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    if config.get("require_confirm_motion", True) and not bool(payload.get("confirm")):
        return {"ok": False, "error": "confirmation_required", "detail": "Set payload.confirm=true to move to zero."}
    axes = payload.get("axes") or ["X", "Y", "Z"]
    feed = payload.get("feed_mm_min")
    words = []
    for axis in axes:
        a = str(axis).upper()
        if a.lower() in _AXES:
            words.append(f"{a}0")
    if not words:
        return {"ok": False, "error": "no_axes_selected"}
    line = "G90 G0 " + " ".join(words)
    if isinstance(feed, (int, float)):
        line += f" F{float(feed):.3f}"
    return _send_gcode_lines([line], config)


def _run_macro(config: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    lines = payload.get("lines") or payload.get("gcode") or []
    if isinstance(lines, str):
        lines = [x for x in lines.splitlines() if x.strip()]
    if not isinstance(lines, list) or not lines:
        return {"ok": False, "error": "missing_lines"}
    if config.get("require_confirm_motion", True):
        motion_present = any(_is_motion_command(str(line)) for line in lines)
        if motion_present and not bool(payload.get("confirm")):
            return {"ok": False, "error": "confirmation_required", "detail": "Set payload.confirm=true to execute motion macro."}
    return _send_gcode_lines([str(x) for x in lines], config)


def _send_gcode(config: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    line = payload.get("line")
    lines = payload.get("lines")
    if line is not None:
        lines = [line]
    if isinstance(lines, str):
        lines = [x for x in lines.splitlines() if x.strip()]
    if not isinstance(lines, list) or not lines:
        return {"ok": False, "error": "missing_line_or_lines"}
    if config.get("require_confirm_motion", True):
        motion_present = any(_is_motion_command(str(item)) for item in lines)
        if motion_present and not bool(payload.get("confirm")):
            return {"ok": False, "error": "confirmation_required", "detail": "Set payload.confirm=true to send motion commands."}
    return _send_gcode_lines([str(x) for x in lines], config)


def _get_settings(config: Dict[str, Any]) -> Dict[str, Any]:
    return _send_gcode_lines(["$$"], config, expect_ok=False)


def _get_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    return _send_gcode_lines(["$#"], config, expect_ok=False)


def _get_parser_state(config: Dict[str, Any]) -> Dict[str, Any]:
    return _send_gcode_lines(["$G"], config, expect_ok=False)


def _send_realtime(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = payload or {}
    err = _require_connected()
    if err:
        return err
    code = payload.get("code")
    if code is None:
        return {"ok": False, "error": "missing_code"}
    if isinstance(code, int):
        data = bytes([code])
    elif isinstance(code, str):
        text = code.strip().lower()
        mapping = {
            "status": b"?",
            "feed_hold": b"!",
            "resume": b"~",
            "soft_reset": b"\x18",
            "jog_cancel": b"\x85",
        }
        if text in mapping:
            data = mapping[text]
        elif text.startswith("0x"):
            data = bytes([int(text, 16)])
        else:
            return {"ok": False, "error": "invalid_code_string"}
    else:
        return {"ok": False, "error": "invalid_code"}
    _send_raw(data)
    _history({"ts": time.time(), "dir": "tx", "line": f"<realtime:{list(data)}>"})
    return {"ok": True, "sent": list(data)}


def driver_init(context=None, config=None):
    merged = _merged_config(config)
    v = driver_validate(merged)
    if not v.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": v}

    with _LOCK:
        _SESSION["instance_id"] = _new_instance_id()
        _SESSION["started_ts"] = time.time()
        _SESSION["status"] = "ready"
        _SESSION["error"] = None
        _SESSION["notes"] = ["GRBL driver initialized."]
        _SESSION["config"] = merged
        _SESSION["controller"] = {}
        _SESSION["last_status"] = None
        _SESSION["last_response"] = None
        _SESSION["last_alarm"] = None
        _SESSION["history"] = []
    return {"ok": True, "instance_id": _SESSION["instance_id"], "validate": v, "info": driver_info()}


def driver_shutdown(context=None):
    _disconnect()
    with _LOCK:
        _SESSION["status"] = "stopped"
        _SESSION["error"] = None
        _SESSION["notes"] = ["Shutdown completed."]
        _SESSION["instance_id"] = None
        _SESSION["started_ts"] = None
    return True


def driver_status(context=None):
    ports = _list_serial_ports()
    return {
        "ok": True,
        "session": dict(_SESSION),
        "serial_available": _serial_available(),
        "ports": ports,
    }


def driver_action(action_id: str, context=None, payload=None):
    payload = payload or {}
    config = _merged_config(_SESSION.get("config", {}))

    if action_id == "ping":
        return {"ok": True, "pong": True, "ts": time.time(), "connected": _SESSION.get("connected", False)}
    if action_id == "list_ports":
        return {"ok": True, "ports": _list_serial_ports()}
    if action_id == "connect":
        return _connect(config, payload)
    if action_id == "disconnect":
        return _disconnect()
    if action_id in ("get_status", "query_status"):
        return _query_status(config)
    if action_id == "unlock":
        return _unlock(config)
    if action_id == "home":
        return _home(config, payload)
    if action_id == "safe_stop":
        return _safe_stop(config)
    if action_id == "feed_hold":
        return _send_realtime({"code": "feed_hold"})
    if action_id == "resume":
        return _send_realtime({"code": "resume"})
    if action_id == "soft_reset":
        return _soft_reset(config)
    if action_id == "send_gcode":
        return _send_gcode(config, payload)
    if action_id == "send_realtime":
        return _send_realtime(payload)
    if action_id == "run_macro":
        return _run_macro(config, payload)
    if action_id == "jog":
        return _jog(config, payload)
    if action_id == "set_zero":
        return _set_zero(config, payload)
    if action_id == "goto_zero":
        return _goto_zero(config, payload)
    if action_id == "get_settings":
        return _get_settings(config)
    if action_id == "get_parameters":
        return _get_parameters(config)
    if action_id == "get_parser_state":
        return _get_parser_state(config)
    if action_id == "drain_buffer":
        lines = _read_until(lambda _line, _acc: False, float(payload.get("timeout", 0.2)))
        return {"ok": True, "lines": lines}
    if action_id == "get_config":
        return {"ok": True, "config": _mask_config(config)}
    if action_id == "set_config":
        updates = payload.get("config") if isinstance(payload.get("config"), dict) else payload
        if not isinstance(updates, dict):
            return {"ok": False, "error": "invalid_config_payload"}
        new_cfg = dict(config)
        new_cfg.update(updates)
        v = driver_validate(new_cfg)
        if not v.get("ok"):
            return {"ok": False, "error": "validation_failed", "details": v}
        with _LOCK:
            _SESSION["config"] = new_cfg
        return {"ok": True, "config": _mask_config(new_cfg), "validate": v}
    return {"ok": False, "error": f"Action '{action_id}' not implemented"}
