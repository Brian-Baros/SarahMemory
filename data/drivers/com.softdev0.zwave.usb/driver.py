import json
import os
import queue
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

_DRIVER_ID = "com.softdev0.zwave.usb"

try:
    import serial  # type: ignore
    import serial.tools.list_ports  # type: ignore
except Exception:
    serial = None

_SESSION: Dict[str, Any] = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "notes": [],
    "connected": False,
    "port": None,
    "stack_mode": "zwave_js_server",
    "history": [],
    "devices": {},
    "network": {
        "home_id": None,
        "controller_node_id": None,
        "state": "unknown",
        "last_update_ts": None,
    },
}

_RUNTIME: Dict[str, Any] = {
    "config": {},
    "serial": None,
    "listener": None,
    "stop_event": None,
    "queue": None,
    "buffer": [],
    "raw_rx_buffer": [],
    "process": None,
}


def _new_instance_id() -> str:
    return "DRV-%s-%sZ-%s" % (
        _DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _now() -> float:
    return time.time()


def _append_note(note: str) -> None:
    notes = _SESSION.setdefault("notes", [])
    notes.append(note)
    if len(notes) > 50:
        del notes[:-50]


def _add_history(entry: Dict[str, Any]) -> None:
    hist = _SESSION.setdefault("history", [])
    hist.append(entry)
    limit = int(_RUNTIME.get("config", {}).get("history_limit", 200) or 200)
    if len(hist) > limit:
        del hist[:-limit]


def _push_buffer(event: Dict[str, Any]) -> None:
    buf = _RUNTIME.setdefault("buffer", [])
    buf.append(event)
    limit = int(_RUNTIME.get("config", {}).get("event_buffer_limit", 500) or 500)
    if len(buf) > limit:
        del buf[:-limit]


def _push_raw_line(line: str) -> None:
    raw = _RUNTIME.setdefault("raw_rx_buffer", [])
    raw.append({"ts": _now(), "line": line})
    limit = int(_RUNTIME.get("config", {}).get("raw_buffer_limit", 500) or 500)
    if len(raw) > limit:
        del raw[:-limit]


def _default_config() -> Dict[str, Any]:
    return {
        "enabled": True,
        "autoload": False,
        "stack_mode": "zwave_js_server",
        "port": "",
        "baud": 115200,
        "timeout": 2.0,
        "event_buffer_limit": 500,
        "raw_buffer_limit": 500,
        "history_limit": 200,
        "allow_raw_send": False,
        "allow_stack_launch": False,
        "require_send_confirmation": True,
        "zwave_js_server_url": "ws://127.0.0.1:3000",
        "zwave_js_ui_url": "http://127.0.0.1:8091",
        "openzwave_url": "",
        "device_hints": ["zwave", "z-stick", "aeotec", "500", "700", "800", "silabs"],
        "line_mode": "ascii",
    }


def _merged_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = _default_config()
    if isinstance(config, dict):
        merged.update(config)
    return merged


def driver_info() -> Dict[str, Any]:
    return {
        "id": _DRIVER_ID,
        "name": "Z-Wave USB Controller Driver",
        "version": "2.0.0",
        "transport": "usb",
        "session": dict(_SESSION),
    }


def driver_validate(config: dict):
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(config, dict):
        errors.append("config must be a JSON object")
        return {"ok": False, "errors": errors, "warnings": warnings}
    for key in ("enabled", "autoload", "allow_raw_send", "allow_stack_launch", "require_send_confirmation"):
        if key in config and not isinstance(config.get(key), bool):
            errors.append(f"{key} must be a boolean")
    for key in ("baud", "event_buffer_limit", "raw_buffer_limit", "history_limit"):
        if key in config and not isinstance(config.get(key), int):
            errors.append(f"{key} must be an integer")
    if "timeout" in config and not isinstance(config.get("timeout"), (int, float)):
        errors.append("timeout must be a number")
    allowed_modes = {"zwave_js_server", "zwave_js_ui", "openzwave", "raw_serial", "vendor"}
    if "stack_mode" in config and config.get("stack_mode") not in allowed_modes:
        errors.append(f"stack_mode must be one of {sorted(allowed_modes)}")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def _serial_available() -> bool:
    return serial is not None


def _backend_probe() -> Dict[str, Any]:
    return {
        "ok": True,
        "serial": _serial_available(),
        "node": shutil.which("node") is not None,
        "zwave_js_ui": shutil.which("zwave-js-ui") is not None,
        "zwave_js_server": shutil.which("zwave-js-server") is not None,
        "ozwd": shutil.which("ozwd") is not None,
    }


def _list_ports() -> List[Dict[str, Any]]:
    ports: List[Dict[str, Any]] = []
    if serial is None:
        return ports
    hints = [str(x).lower() for x in _RUNTIME.get("config", {}).get("device_hints", [])]
    for p in serial.tools.list_ports.comports():
        hay = " ".join([str(getattr(p, "device", "")), str(getattr(p, "description", "")), str(getattr(p, "manufacturer", "")), str(getattr(p, "hwid", ""))]).lower()
        ports.append({
            "device": getattr(p, "device", None),
            "name": getattr(p, "name", None),
            "description": getattr(p, "description", None),
            "manufacturer": getattr(p, "manufacturer", None),
            "serial_number": getattr(p, "serial_number", None),
            "vid": getattr(p, "vid", None),
            "pid": getattr(p, "pid", None),
            "hwid": getattr(p, "hwid", None),
            "zwave_hint": any(h in hay for h in hints) if hints else False,
        })
    return ports


def _auto_port() -> Optional[str]:
    config_port = str(_RUNTIME.get("config", {}).get("port") or "").strip()
    if config_port:
        return config_port
    ports = _list_ports()
    for p in ports:
        if p.get("zwave_hint") and p.get("device"):
            return str(p["device"])
    if ports:
        return str(ports[0].get("device") or "") or None
    return None


def _close_serial() -> None:
    ser = _RUNTIME.get("serial")
    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass
    _RUNTIME["serial"] = None
    _SESSION["connected"] = False


def _decode_line(line: str) -> Dict[str, Any]:
    event: Dict[str, Any] = {"ts": _now(), "raw": line, "kind": "raw"}
    stripped = line.strip()
    if not stripped:
        return event
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            payload = json.loads(stripped)
            event["kind"] = "json"
            event["payload"] = payload
            if isinstance(payload, dict):
                node_id = payload.get("nodeId") or payload.get("node_id")
                if node_id is not None:
                    _SESSION.setdefault("devices", {})[str(node_id)] = payload
                if payload.get("homeId"):
                    _SESSION["network"]["home_id"] = payload.get("homeId")
                if payload.get("controllerNodeId"):
                    _SESSION["network"]["controller_node_id"] = payload.get("controllerNodeId")
                _SESSION["network"]["last_update_ts"] = _now()
        except Exception:
            pass
    elif stripped.startswith("0x"):
        event["kind"] = "hex"
    else:
        event["kind"] = "text"
    return event


def _listener_loop() -> None:
    stop_event: threading.Event = _RUNTIME["stop_event"]
    ser = _RUNTIME.get("serial")
    while not stop_event.is_set():
        if ser is None:
            break
        try:
            data = ser.readline()
            if not data:
                continue
            line = data.decode("utf-8", errors="replace")
            _push_raw_line(line)
            event = _decode_line(line)
            _push_buffer(event)
        except Exception as exc:
            _append_note(f"listener_error: {exc}")
            _SESSION["error"] = str(exc)
            time.sleep(0.2)


def _start_listener() -> Dict[str, Any]:
    if _RUNTIME.get("listener") and _RUNTIME["listener"].is_alive():
        return {"ok": True, "already_running": True}
    stop_event = threading.Event()
    _RUNTIME["stop_event"] = stop_event
    thread = threading.Thread(target=_listener_loop, daemon=True)
    _RUNTIME["listener"] = thread
    thread.start()
    return {"ok": True, "started": True}


def _stop_listener() -> Dict[str, Any]:
    stop_event = _RUNTIME.get("stop_event")
    if stop_event is not None:
        stop_event.set()
    thread = _RUNTIME.get("listener")
    if thread is not None and thread.is_alive():
        thread.join(timeout=1.0)
    _RUNTIME["listener"] = None
    _RUNTIME["stop_event"] = None
    return {"ok": True}


def _connect_serial() -> Dict[str, Any]:
    if serial is None:
        return {"ok": False, "error": "pyserial_not_installed"}
    if _RUNTIME.get("serial") is not None:
        return {"ok": True, "already_connected": True, "port": _SESSION.get("port")}
    port = _auto_port()
    if not port:
        return {"ok": False, "error": "no_serial_port_found", "ports": _list_ports()}
    cfg = _RUNTIME.get("config", {})
    try:
        ser = serial.Serial(port=port, baudrate=int(cfg.get("baud", 115200)), timeout=float(cfg.get("timeout", 2.0)))
        _RUNTIME["serial"] = ser
        _SESSION["connected"] = True
        _SESSION["port"] = port
        _append_note(f"connected_serial:{port}")
        return {"ok": True, "port": port}
    except Exception as exc:
        _SESSION["error"] = str(exc)
        return {"ok": False, "error": "serial_connect_failed", "details": str(exc), "port": port}


def _ensure_confirmation(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    payload = payload or {}
    if _RUNTIME.get("config", {}).get("require_send_confirmation", True) and not bool(payload.get("confirm")):
        return {"ok": False, "error": "confirmation_required"}
    return None


def _send_raw(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not bool(_RUNTIME.get("config", {}).get("allow_raw_send", False)):
        return {"ok": False, "error": "raw_send_disabled"}
    gate = _ensure_confirmation(payload)
    if gate:
        return gate
    ser = _RUNTIME.get("serial")
    if ser is None:
        return {"ok": False, "error": "not_connected"}
    data = payload.get("data") if isinstance(payload, dict) else None
    if data is None:
        return {"ok": False, "error": "missing_data"}
    if isinstance(data, str):
        if payload.get("hex"):
            try:
                raw = bytes.fromhex(data.replace(" ", ""))
            except Exception as exc:
                return {"ok": False, "error": "invalid_hex", "details": str(exc)}
        else:
            raw = data.encode("utf-8")
    elif isinstance(data, list):
        raw = bytes(int(x) & 0xFF for x in data)
    else:
        return {"ok": False, "error": "unsupported_data_type"}
    try:
        ser.write(raw)
        ser.flush()
        _add_history({"ts": _now(), "action": "send_raw", "bytes": len(raw)})
        return {"ok": True, "bytes_sent": len(raw)}
    except Exception as exc:
        return {"ok": False, "error": "send_failed", "details": str(exc)}


def _launch_stack(payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not bool(_RUNTIME.get("config", {}).get("allow_stack_launch", False)):
        return {"ok": False, "error": "stack_launch_disabled"}
    gate = _ensure_confirmation(payload)
    if gate:
        return gate
    mode = str((payload or {}).get("stack_mode") or _RUNTIME.get("config", {}).get("stack_mode") or "zwave_js_server")
    commands = {
        "zwave_js_server": ["zwave-js-server"],
        "zwave_js_ui": ["zwave-js-ui"],
        "openzwave": ["ozwd"],
    }
    cmd = commands.get(mode)
    if not cmd:
        return {"ok": False, "error": "unsupported_stack_mode", "stack_mode": mode}
    if shutil.which(cmd[0]) is None:
        return {"ok": False, "error": "stack_binary_not_found", "binary": cmd[0]}
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _RUNTIME["process"] = proc
        _add_history({"ts": _now(), "action": "launch_stack", "stack_mode": mode, "pid": proc.pid})
        return {"ok": True, "stack_mode": mode, "pid": proc.pid}
    except Exception as exc:
        return {"ok": False, "error": "stack_launch_failed", "details": str(exc)}


def driver_init(context=None, config=None):
    merged = _merged_config(config)
    v = driver_validate(merged)
    if not v.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": v}
    _RUNTIME["config"] = merged
    _RUNTIME["buffer"] = []
    _RUNTIME["raw_rx_buffer"] = []
    _RUNTIME["queue"] = queue.Queue()
    _SESSION["instance_id"] = _new_instance_id()
    _SESSION["started_ts"] = _now()
    _SESSION["status"] = "ready"
    _SESSION["error"] = None
    _SESSION["notes"] = ["Z-Wave USB bridge initialized."]
    _SESSION["stack_mode"] = merged.get("stack_mode", "zwave_js_server")
    return {"ok": True, "instance_id": _SESSION["instance_id"], "validate": v, "info": driver_info()}


def driver_shutdown(context=None):
    _stop_listener()
    _close_serial()
    proc = _RUNTIME.get("process")
    if proc is not None:
        try:
            proc.terminate()
        except Exception:
            pass
    _RUNTIME["process"] = None
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _SESSION["notes"] = ["Shutdown completed."]
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    _SESSION["connected"] = False
    _SESSION["port"] = None
    return True


def driver_status(context=None):
    return {
        "ok": True,
        "session": dict(_SESSION),
        "buffer_len": len(_RUNTIME.get("buffer", [])),
        "raw_buffer_len": len(_RUNTIME.get("raw_rx_buffer", [])),
        "backend_probe": _backend_probe(),
    }


def driver_action(action_id: str, context=None, payload=None):
    payload = payload or {}
    if action_id == "ping":
        return {"ok": True, "pong": True, "ts": _now()}
    if action_id == "probe_backends":
        return {"ok": True, "backends": _backend_probe()}
    if action_id in ("list_ports", "discover_devices", "list_devices"):
        return {"ok": True, "ports": _list_ports(), "devices": list(_SESSION.get("devices", {}).values())}
    if action_id == "connect":
        return _connect_serial()
    if action_id == "disconnect":
        _stop_listener()
        _close_serial()
        return {"ok": True}
    if action_id == "start_listen":
        if _RUNTIME.get("serial") is None:
            conn = _connect_serial()
            if not conn.get("ok"):
                return conn
        return _start_listener()
    if action_id == "stop_listen":
        return _stop_listener()
    if action_id == "read_once":
        ser = _RUNTIME.get("serial")
        if ser is None:
            conn = _connect_serial()
            if not conn.get("ok"):
                return conn
            ser = _RUNTIME.get("serial")
        try:
            data = ser.readline()
            line = data.decode("utf-8", errors="replace") if data else ""
            if line:
                _push_raw_line(line)
                event = _decode_line(line)
                _push_buffer(event)
                return {"ok": True, "event": event}
            return {"ok": True, "event": None}
        except Exception as exc:
            return {"ok": False, "error": "read_failed", "details": str(exc)}
    if action_id == "get_buffer":
        return {"ok": True, "items": list(_RUNTIME.get("buffer", []))}
    if action_id == "clear_buffer":
        _RUNTIME["buffer"] = []
        _RUNTIME["raw_rx_buffer"] = []
        return {"ok": True}
    if action_id == "send_raw":
        return _send_raw(payload)
    if action_id == "launch_stack":
        return _launch_stack(payload)
    if action_id == "feed_text":
        text = str(payload.get("text") or "")
        for line in text.splitlines():
            _push_raw_line(line)
            _push_buffer(_decode_line(line))
        return {"ok": True, "added": len(text.splitlines())}
    if action_id == "describe_capabilities":
        return {
            "ok": True,
            "capabilities": {
                "serial_bridge": True,
                "device_registry": True,
                "raw_send": bool(_RUNTIME.get("config", {}).get("allow_raw_send", False)),
                "external_stacks": ["zwave_js_server", "zwave_js_ui", "openzwave"],
                "preferred_paths": ["/dev/serial/by-id", "COMx"],
            },
        }
    if action_id == "get_config":
        safe = dict(_RUNTIME.get("config", {}))
        return {"ok": True, "config": safe}
    if action_id == "set_config":
        updates = payload.get("config") if isinstance(payload, dict) else None
        if not isinstance(updates, dict):
            return {"ok": False, "error": "config_object_required"}
        merged = dict(_RUNTIME.get("config", {}))
        merged.update(updates)
        v = driver_validate(merged)
        if not v.get("ok"):
            return {"ok": False, "error": "validation_failed", "details": v}
        _RUNTIME["config"] = merged
        _SESSION["stack_mode"] = merged.get("stack_mode", _SESSION.get("stack_mode"))
        return {"ok": True, "config": merged, "validate": v}
    if action_id == "safe_stop":
        _stop_listener()
        return {"ok": True, "stopped": True}
    return {"ok": False, "error": f"Action '{action_id}' not implemented"}
