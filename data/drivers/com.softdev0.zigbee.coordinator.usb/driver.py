import json
import os
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

_DRIVER_ID = "com.softdev0.zigbee.coordinator.usb"
_DRIVER_NAME = "Zigbee Coordinator Driver"
_DRIVER_VERSION = "2.0.0"

try:
    import serial  # type: ignore
    from serial.tools import list_ports  # type: ignore
except Exception:
    serial = None
    list_ports = None

_SESSION: Dict[str, Any] = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "notes": [],
    "port_open": False,
    "connected_port": None,
    "last_result": None,
    "listener_running": False,
    "frames_rx": 0,
    "frames_tx": 0,
    "network_state": "unknown",
    "pan_id": None,
    "extended_pan_id": None,
    "channel": None,
    "permit_join": False,
}

_STATE: Dict[str, Any] = {
    "serial_handle": None,
    "listener_thread": None,
    "listener_stop": None,
    "config": {},
    "frame_buffer": [],
    "event_buffer": [],
    "device_registry": {},
}

_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "autoload": False,
    "port": "",
    "baud": 115200,
    "bytesize": 8,
    "parity": "N",
    "stopbits": 1,
    "timeout": 1.0,
    "write_timeout": 2.0,
    "stack": "raw_serial",
    "auto_discover": True,
    "port_hint": "zigbee|conbee|cc2652|sonoff|ezsp|ember|skyconnect",
    "frame_encoding": "hex",
    "line_terminator": "\\r\\n",
    "poll_interval": 0.25,
    "frame_buffer_limit": 500,
    "event_buffer_limit": 500,
    "allow_raw_send": False,
    "require_send_confirmation": True,
    "allow_stack_launch": False,
    "stack_command": "",
    "stack_workdir": "",
    "mqtt_host": "127.0.0.1",
    "mqtt_port": 1883,
    "mqtt_base_topic": "zigbee2mqtt",
    "deconz_api_base": "http://127.0.0.1:8080/api",
    "deconz_api_key": "",
    "vendor_passthrough": False,
}


def _new_instance_id() -> str:
    return "DRV-%s-%sZ-%s" % (
        _DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _utc() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _push_note(msg: str) -> None:
    notes = _SESSION.setdefault("notes", [])
    notes.append(f"{_utc()} {msg}")
    if len(notes) > 100:
        del notes[:-100]


def _push_buffer(key: str, item: Dict[str, Any], limit_key: str) -> None:
    buf = _STATE.setdefault(key, [])
    buf.append(item)
    limit = int(_STATE["config"].get(limit_key, _DEFAULTS[limit_key]))
    if len(buf) > limit:
        del buf[:-limit]


def _mask_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg)
    for k in ("deconz_api_key",):
        if out.get(k):
            out[k] = "***"
    return out


def _normalize_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(_DEFAULTS)
    if isinstance(config, dict):
        cfg.update(config)
    cfg["stack"] = str(cfg.get("stack", "raw_serial") or "raw_serial").strip()
    cfg["port"] = str(cfg.get("port", "") or "").strip()
    cfg["port_hint"] = str(cfg.get("port_hint", "") or "").strip().lower()
    cfg["frame_encoding"] = str(cfg.get("frame_encoding", "hex") or "hex").strip().lower()
    cfg["line_terminator"] = str(cfg.get("line_terminator", "\\r\\n") or "")
    cfg["deconz_api_base"] = str(cfg.get("deconz_api_base", "") or "").strip()
    cfg["mqtt_base_topic"] = str(cfg.get("mqtt_base_topic", "zigbee2mqtt") or "zigbee2mqtt").strip().strip("/")
    cfg["bytesize"] = int(cfg.get("bytesize", 8))
    cfg["stopbits"] = int(cfg.get("stopbits", 1))
    cfg["baud"] = int(cfg.get("baud", 115200))
    cfg["timeout"] = float(cfg.get("timeout", 1.0))
    cfg["write_timeout"] = float(cfg.get("write_timeout", 2.0))
    cfg["poll_interval"] = max(0.05, float(cfg.get("poll_interval", 0.25)))
    cfg["frame_buffer_limit"] = max(10, int(cfg.get("frame_buffer_limit", 500)))
    cfg["event_buffer_limit"] = max(10, int(cfg.get("event_buffer_limit", 500)))
    return cfg


def driver_info() -> Dict[str, Any]:
    return {
        "id": _DRIVER_ID,
        "name": _DRIVER_NAME,
        "version": _DRIVER_VERSION,
        "transport": "usb_serial",
        "session": dict(_SESSION),
        "config": _mask_config(_STATE.get("config", {})),
    }


def driver_validate(config: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be a JSON object"], "warnings": []}
    cfg = _normalize_config(config)
    if cfg["baud"] <= 0:
        errors.append("baud must be > 0")
    if cfg["bytesize"] not in (5, 6, 7, 8):
        errors.append("bytesize must be 5, 6, 7, or 8")
    if cfg["stopbits"] not in (1, 2):
        errors.append("stopbits must be 1 or 2")
    if cfg["frame_encoding"] not in ("hex", "ascii", "json"):
        errors.append("frame_encoding must be hex, ascii, or json")
    if cfg["stack"] not in ("raw_serial", "zigbee2mqtt", "deconz", "zha_bridge", "vendor"):
        errors.append("stack must be raw_serial, zigbee2mqtt, deconz, zha_bridge, or vendor")
    if serial is None:
        warnings.append("pyserial not installed; serial operations unavailable")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings, "normalized": cfg}


def _serial_parity(name: str):
    if serial is None:
        return None
    mapping = {
        "N": serial.PARITY_NONE,
        "E": serial.PARITY_EVEN,
        "O": serial.PARITY_ODD,
        "M": serial.PARITY_MARK,
        "S": serial.PARITY_SPACE,
    }
    return mapping.get(str(name).upper(), serial.PARITY_NONE)


def _open_serial(cfg: Dict[str, Any], port: Optional[str] = None):
    if serial is None:
        raise RuntimeError("pyserial not installed")
    target = str(port or cfg.get("port") or "").strip()
    if not target:
        raise RuntimeError("no serial port configured")
    handle = serial.Serial(
        port=target,
        baudrate=cfg["baud"],
        bytesize=cfg["bytesize"],
        parity=_serial_parity(cfg.get("parity", "N")),
        stopbits=cfg["stopbits"],
        timeout=cfg["timeout"],
        write_timeout=cfg["write_timeout"],
    )
    return handle


def _close_serial() -> None:
    handle = _STATE.get("serial_handle")
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass
    _STATE["serial_handle"] = None
    _SESSION["port_open"] = False
    _SESSION["connected_port"] = None


def _decode_line(raw: bytes) -> Dict[str, Any]:
    cfg = _STATE["config"]
    enc = cfg.get("frame_encoding", "hex")
    text = raw.decode("utf-8", errors="replace").strip()
    item: Dict[str, Any] = {
        "ts": time.time(),
        "utc": _utc(),
        "raw_hex": raw.hex().upper(),
        "text": text,
        "kind": "frame",
    }
    if enc == "json":
        try:
            parsed = json.loads(text)
            item["parsed"] = parsed
            item["kind"] = "json"
            if isinstance(parsed, dict):
                _merge_network_state(parsed)
                _maybe_register_device(parsed)
        except Exception:
            item["parse_error"] = "invalid_json"
    elif enc == "ascii":
        lower = text.lower()
        if "permit join" in lower:
            _SESSION["permit_join"] = "on" in lower or "true" in lower or "1" in lower
        if "channel" in lower and any(ch.isdigit() for ch in lower):
            digits = "".join(ch for ch in lower if ch.isdigit())
            if digits:
                _SESSION["channel"] = int(digits[:2])
    _push_buffer("frame_buffer", item, "frame_buffer_limit")
    _push_buffer("event_buffer", item, "event_buffer_limit")
    _SESSION["frames_rx"] += 1
    return item


def _merge_network_state(data: Dict[str, Any]) -> None:
    for src_key, dst_key in (
        ("pan_id", "pan_id"),
        ("extended_pan_id", "extended_pan_id"),
        ("ext_pan_id", "extended_pan_id"),
        ("channel", "channel"),
        ("permit_join", "permit_join"),
        ("network_state", "network_state"),
        ("state", "network_state"),
    ):
        if src_key in data:
            _SESSION[dst_key] = data[src_key]


def _maybe_register_device(data: Dict[str, Any]) -> None:
    dev_id = data.get("ieee_address") or data.get("ieee") or data.get("id") or data.get("friendly_name") or data.get("device")
    if not dev_id:
        return
    entry = _STATE.setdefault("device_registry", {}).get(str(dev_id), {})
    entry.update(data)
    entry["last_seen_ts"] = time.time()
    _STATE["device_registry"][str(dev_id)] = entry


def _listener_loop() -> None:
    stop_event = _STATE.get("listener_stop")
    handle = _STATE.get("serial_handle")
    _SESSION["listener_running"] = True
    _push_note("listener started")
    try:
        while stop_event is not None and not stop_event.is_set():
            if handle is None:
                break
            try:
                raw = handle.readline()
            except Exception as exc:
                _push_note(f"listener read failed: {exc}")
                break
            if raw:
                try:
                    _decode_line(raw)
                except Exception as exc:
                    _push_buffer("event_buffer", {"ts": time.time(), "utc": _utc(), "kind": "error", "error": str(exc)}, "event_buffer_limit")
            else:
                time.sleep(_STATE["config"].get("poll_interval", 0.25))
    finally:
        _SESSION["listener_running"] = False
        _push_note("listener stopped")


def _stop_listener() -> None:
    stop_event = _STATE.get("listener_stop")
    th = _STATE.get("listener_thread")
    if stop_event is not None:
        stop_event.set()
    if th is not None and th.is_alive():
        th.join(timeout=2.0)
    _STATE["listener_thread"] = None
    _STATE["listener_stop"] = None
    _SESSION["listener_running"] = False


def _probe_backends() -> Dict[str, Any]:
    return {
        "ok": True,
        "serial": serial is not None,
        "serial_tools": list_ports is not None,
        "zigbee2mqtt_cli": shutil.which("zigbee2mqtt") is not None,
        "deconz": shutil.which("deCONZ") is not None or shutil.which("deconz") is not None,
        "mosquitto_pub": shutil.which("mosquitto_pub") is not None,
        "mosquitto_sub": shutil.which("mosquitto_sub") is not None,
    }


def _list_ports() -> Dict[str, Any]:
    ports: List[Dict[str, Any]] = []
    if list_ports is None:
        return {"ok": False, "error": "serial_tools_unavailable", "ports": ports}
    hint = _STATE["config"].get("port_hint", "")
    for p in list_ports.comports():
        text = " ".join(filter(None, [p.device, p.description, p.manufacturer or "", p.hwid]))
        ports.append({
            "device": p.device,
            "description": p.description,
            "manufacturer": getattr(p, "manufacturer", None),
            "vid": getattr(p, "vid", None),
            "pid": getattr(p, "pid", None),
            "serial_number": getattr(p, "serial_number", None),
            "hwid": p.hwid,
            "hint_match": bool(hint and hint in text.lower()),
        })
    return {"ok": True, "ports": ports}


def _autoselect_port() -> Optional[str]:
    res = _list_ports()
    if not res.get("ok"):
        return None
    ports = res.get("ports", [])
    for p in ports:
        if p.get("hint_match"):
            return p["device"]
    return ports[0]["device"] if ports else None


def _serial_write(data: bytes) -> Dict[str, Any]:
    handle = _STATE.get("serial_handle")
    if handle is None:
        return {"ok": False, "error": "not_connected"}
    handle.write(data)
    handle.flush()
    _SESSION["frames_tx"] += 1
    result = {"ok": True, "bytes": len(data), "hex": data.hex().upper(), "ts": time.time()}
    _SESSION["last_result"] = result
    return result


def _send_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _STATE["config"]
    if not cfg.get("allow_raw_send"):
        return {"ok": False, "error": "raw_send_disabled"}
    if cfg.get("require_send_confirmation", True) and not bool(payload.get("confirm_send")):
        return {"ok": False, "error": "confirmation_required"}
    if "hex" in payload:
        try:
            data = bytes.fromhex(str(payload.get("hex", "")).replace(" ", ""))
        except Exception:
            return {"ok": False, "error": "invalid_hex"}
    else:
        text = str(payload.get("text", ""))
        data = text.encode("utf-8")
    term = str(cfg.get("line_terminator", "") or "")
    if term:
        data += term.encode("utf-8")
    return _serial_write(data)


def _launch_stack(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _STATE["config"]
    if not cfg.get("allow_stack_launch"):
        return {"ok": False, "error": "stack_launch_disabled"}
    cmd = str(payload.get("command") or cfg.get("stack_command") or "").strip()
    if not cmd:
        return {"ok": False, "error": "missing_command"}
    workdir = str(payload.get("workdir") or cfg.get("stack_workdir") or "").strip() or None
    try:
        proc = subprocess.Popen(cmd, cwd=workdir, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        return {"ok": False, "error": f"launch_failed: {exc}"}
    return {"ok": True, "pid": proc.pid, "command": cmd, "workdir": workdir}


def driver_init(context=None, config=None):
    v = driver_validate(config or {})
    if not v.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": v}
    _STATE["config"] = v["normalized"]
    _STATE["frame_buffer"] = []
    _STATE["event_buffer"] = []
    _STATE["device_registry"] = {}
    _SESSION.update({
        "instance_id": _new_instance_id(),
        "started_ts": time.time(),
        "status": "ready",
        "error": None,
        "notes": [],
        "port_open": False,
        "connected_port": None,
        "last_result": None,
        "listener_running": False,
        "frames_rx": 0,
        "frames_tx": 0,
        "network_state": "unknown",
        "pan_id": None,
        "extended_pan_id": None,
        "channel": None,
        "permit_join": False,
    })
    _push_note("driver initialized")
    if _STATE["config"].get("auto_discover") and not _STATE["config"].get("port"):
        auto = _autoselect_port()
        if auto:
            _STATE["config"]["port"] = auto
            _push_note(f"auto-selected port {auto}")
    return {"ok": True, "instance_id": _SESSION["instance_id"], "validate": v, "info": driver_info()}


def driver_shutdown(context=None):
    _stop_listener()
    _close_serial()
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _push_note("shutdown completed")
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    return True


def driver_status(context=None):
    return {
        "ok": True,
        "session": dict(_SESSION),
        "config": _mask_config(_STATE.get("config", {})),
        "buffer_sizes": {
            "frames": len(_STATE.get("frame_buffer", [])),
            "events": len(_STATE.get("event_buffer", [])),
            "devices": len(_STATE.get("device_registry", {})),
        },
    }


def driver_action(action_id: str, context=None, payload=None):
    payload = payload or {}
    try:
        if action_id == "ping":
            return {"ok": True, "pong": True, "ts": time.time()}
        if action_id == "probe_backends":
            return _probe_backends()
        if action_id in ("list_ports", "discover_ports"):
            return _list_ports()
        if action_id == "get_config":
            return {"ok": True, "config": _mask_config(_STATE.get("config", {}))}
        if action_id == "set_config":
            merged = dict(_STATE.get("config", {}))
            merged.update(payload or {})
            v = driver_validate(merged)
            if not v.get("ok"):
                return {"ok": False, "error": "validation_failed", "details": v}
            _STATE["config"] = v["normalized"]
            return {"ok": True, "config": _mask_config(_STATE["config"]), "warnings": v.get("warnings", [])}
        if action_id == "connect":
            if _STATE.get("serial_handle") is not None:
                return {"ok": True, "already_connected": True, "port": _SESSION.get("connected_port")}
            cfg = _STATE["config"]
            port = str(payload.get("port") or cfg.get("port") or "").strip() or (_autoselect_port() if cfg.get("auto_discover") else "")
            if not port:
                return {"ok": False, "error": "no_port"}
            handle = _open_serial(cfg, port)
            _STATE["serial_handle"] = handle
            _SESSION["port_open"] = True
            _SESSION["connected_port"] = port
            _SESSION["status"] = "connected"
            _push_note(f"connected to {port}")
            return {"ok": True, "port": port, "baud": cfg.get("baud")}
        if action_id == "disconnect":
            _stop_listener()
            _close_serial()
            _SESSION["status"] = "ready"
            _push_note("disconnected")
            return {"ok": True}
        if action_id == "start_listener":
            if _STATE.get("serial_handle") is None:
                return {"ok": False, "error": "not_connected"}
            if _STATE.get("listener_thread") is not None and _STATE["listener_thread"].is_alive():
                return {"ok": True, "already_running": True}
            _STATE["listener_stop"] = threading.Event()
            th = threading.Thread(target=_listener_loop, name="zigbee-coordinator-listener", daemon=True)
            _STATE["listener_thread"] = th
            th.start()
            return {"ok": True, "started": True}
        if action_id == "stop_listener":
            _stop_listener()
            return {"ok": True, "stopped": True}
        if action_id == "read_once":
            handle = _STATE.get("serial_handle")
            if handle is None:
                return {"ok": False, "error": "not_connected"}
            raw = handle.readline()
            if not raw:
                return {"ok": True, "data": None}
            return {"ok": True, "data": _decode_line(raw)}
        if action_id == "send_raw":
            return _send_payload(payload)
        if action_id == "feed_text":
            text = str(payload.get("text", ""))
            data = text.encode("utf-8")
            item = _decode_line(data)
            return {"ok": True, "data": item}
        if action_id == "get_frames":
            return {"ok": True, "frames": list(_STATE.get("frame_buffer", []))}
        if action_id == "get_events":
            return {"ok": True, "events": list(_STATE.get("event_buffer", []))}
        if action_id == "clear_buffers":
            _STATE["frame_buffer"] = []
            _STATE["event_buffer"] = []
            return {"ok": True}
        if action_id == "list_devices":
            return {"ok": True, "devices": list(_STATE.get("device_registry", {}).values())}
        if action_id == "set_network_state":
            _merge_network_state(payload)
            return {"ok": True, "session": dict(_SESSION)}
        if action_id == "launch_stack":
            return _launch_stack(payload)
        if action_id == "safe_stop":
            _stop_listener()
            return {"ok": True, "stopped": True}
        return {"ok": False, "error": f"Action '{action_id}' not implemented"}
    except Exception as exc:
        _SESSION["error"] = str(exc)
        _SESSION["last_result"] = {"ok": False, "error": str(exc), "action": action_id}
        return {"ok": False, "error": str(exc), "action": action_id}
