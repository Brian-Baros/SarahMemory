# com.softdev0.midi.bluetooth/driver.py
# BLE MIDI driver package for SarahMemory AiOS
# Transport: bluetooth

from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

_DRIVER_ID = "com.softdev0.midi.bluetooth"
_BLE_MIDI_SERVICE_UUID = "03b80e5a-ede8-4b33-a751-6ce34ec4c700"
_BLE_MIDI_IO_CHAR_UUID = "7772e5db-3868-4112-a1a9-f2669d106bf3"

try:
    from bleak import BleakScanner, BleakClient  # type: ignore
    _BLEAK_OK = True
except Exception:
    BleakScanner = None
    BleakClient = None
    _BLEAK_OK = False

try:
    import mido  # type: ignore
    _MIDO_OK = True
except Exception:
    mido = None
    _MIDO_OK = False

_SESSION: Dict[str, Any] = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "notes": [],
    "connected": False,
    "device": None,
    "backend": None,
    "listener_running": False,
    "last_event_ts": None,
    "events_seen": 0,
}

_CONFIG: Dict[str, Any] = {}
_EVENT_BUFFER: List[Dict[str, Any]] = []
_EVENT_LOCK = threading.Lock()
_LISTENER_THREAD = None
_STOP_EVENT = threading.Event()
_CLIENT = None
_VIRTUAL_OUT = None


def _new_instance_id() -> str:
    return "DRV-%s-%sZ-%s" % (
        _DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _note(msg: str) -> None:
    _SESSION.setdefault("notes", []).append(msg)
    _SESSION["notes"] = _SESSION["notes"][-50:]


def _merge_config(config: Optional[dict]) -> Dict[str, Any]:
    merged = {
        "enabled": True,
        "autoload": False,
        "device_name": "",
        "device_address": "",
        "service_uuid": _BLE_MIDI_SERVICE_UUID,
        "characteristic_uuid": _BLE_MIDI_IO_CHAR_UUID,
        "backend_priority": ["bleak", "mido_virtual"],
        "auto_pair": False,
        "auto_connect": False,
        "scan_timeout": 8.0,
        "connect_timeout": 15.0,
        "buffer_limit": 1000,
        "dedupe_window_ms": 0,
        "emit_raw_packets": False,
        "emit_clock": True,
        "emit_active_sensing": False,
        "virtual_port_name": "SarahMemory BLE MIDI",
        "allow_send": True,
        "allow_vendor_sysex": False,
    }
    if isinstance(config, dict):
        merged.update(config)
    return merged


def driver_info() -> Dict[str, Any]:
    return {
        "id": _DRIVER_ID,
        "name": "MIDI Bluetooth Driver",
        "version": "1.0.0",
        "transport": "bluetooth",
        "backend_support": {"bleak": _BLEAK_OK, "mido": _MIDO_OK},
        "session": dict(_SESSION),
    }


def driver_validate(config: dict):
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be a JSON object"], "warnings": []}
    bool_fields = ["enabled", "autoload", "auto_pair", "auto_connect", "emit_raw_packets", "emit_clock", "emit_active_sensing", "allow_send", "allow_vendor_sysex"]
    str_fields = ["device_name", "device_address", "service_uuid", "characteristic_uuid", "virtual_port_name"]
    num_fields = ["scan_timeout", "connect_timeout", "buffer_limit", "dedupe_window_ms"]
    for k in bool_fields:
        if k in config and not isinstance(config.get(k), bool):
            errors.append(f"{k} must be a boolean")
    for k in str_fields:
        if k in config and not isinstance(config.get(k), str):
            errors.append(f"{k} must be a string")
    for k in num_fields:
        if k in config and not isinstance(config.get(k), (int, float)):
            errors.append(f"{k} must be numeric")
    if "backend_priority" in config and not isinstance(config.get("backend_priority"), list):
        errors.append("backend_priority must be a list")
    if not _BLEAK_OK:
        warnings.append("bleak not installed; BLE scan/connect unavailable in this environment")
    if not _MIDO_OK:
        warnings.append("mido not installed; MIDI message object support unavailable in this environment")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def _push_event(event: Dict[str, Any]) -> Dict[str, Any]:
    event.setdefault("ts", time.time())
    dedupe_ms = int(_CONFIG.get("dedupe_window_ms") or 0)
    with _EVENT_LOCK:
        if dedupe_ms and _EVENT_BUFFER:
            prev = _EVENT_BUFFER[-1]
            if prev.get("type") == event.get("type") and prev.get("data") == event.get("data") and (event["ts"] - prev.get("ts", 0)) * 1000.0 < dedupe_ms:
                return event
        _EVENT_BUFFER.append(event)
        limit = max(1, int(_CONFIG.get("buffer_limit") or 1000))
        if len(_EVENT_BUFFER) > limit:
            del _EVENT_BUFFER[:-limit]
    _SESSION["events_seen"] = int(_SESSION.get("events_seen") or 0) + 1
    _SESSION["last_event_ts"] = event["ts"]
    return event


def _clear_events() -> Dict[str, Any]:
    with _EVENT_LOCK:
        _EVENT_BUFFER.clear()
    return {"ok": True, "cleared": True}


def _get_events(limit: Optional[int] = None) -> Dict[str, Any]:
    with _EVENT_LOCK:
        data = list(_EVENT_BUFFER)
    if isinstance(limit, int) and limit > 0:
        data = data[-limit:]
    return {"ok": True, "events": data, "count": len(data)}


def _clear_runtime() -> None:
    global _CLIENT, _LISTENER_THREAD, _VIRTUAL_OUT
    _STOP_EVENT.set()
    try:
        if _LISTENER_THREAD and _LISTENER_THREAD.is_alive():
            _LISTENER_THREAD.join(timeout=1.5)
    except Exception:
        pass
    _LISTENER_THREAD = None
    _CLIENT = None
    try:
        if _VIRTUAL_OUT is not None:
            _VIRTUAL_OUT.close()
    except Exception:
        pass
    _VIRTUAL_OUT = None
    _SESSION["connected"] = False
    _SESSION["listener_running"] = False
    _SESSION["device"] = None
    _SESSION["backend"] = None


def _masked_config() -> Dict[str, Any]:
    return dict(_CONFIG)


def driver_init(context=None, config=None):
    global _CONFIG
    _CONFIG = _merge_config(config)
    v = driver_validate(_CONFIG)
    if not v.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": v}
    _SESSION["instance_id"] = _new_instance_id()
    _SESSION["started_ts"] = time.time()
    _SESSION["status"] = "ready"
    _SESSION["error"] = None
    _SESSION["notes"] = []
    _SESSION["events_seen"] = 0
    _SESSION["last_event_ts"] = None
    _clear_events()
    _note("Driver initialized.")
    return {"ok": True, "instance_id": _SESSION["instance_id"], "validate": v, "info": driver_info()}


def driver_shutdown(context=None):
    _clear_runtime()
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _SESSION["notes"] = ["Shutdown completed."]
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    return True


def driver_status(context=None):
    return {
        "ok": True,
        "session": dict(_SESSION),
        "config": _masked_config(),
        "buffered_events": len(_EVENT_BUFFER),
        "backend_support": {"bleak": _BLEAK_OK, "mido": _MIDO_OK},
    }


def _status_to_type(status: int) -> str:
    if 0x80 <= status <= 0x8F:
        return "note_off"
    if 0x90 <= status <= 0x9F:
        return "note_on"
    if 0xA0 <= status <= 0xAF:
        return "poly_aftertouch"
    if 0xB0 <= status <= 0xBF:
        return "control_change"
    if 0xC0 <= status <= 0xCF:
        return "program_change"
    if 0xD0 <= status <= 0xDF:
        return "channel_aftertouch"
    if 0xE0 <= status <= 0xEF:
        return "pitchwheel"
    system = {0xF0: "sysex", 0xF1: "quarter_frame", 0xF2: "songpos", 0xF3: "song_select", 0xF6: "tune_request", 0xF8: "clock", 0xFA: "start", 0xFB: "continue", 0xFC: "stop", 0xFE: "active_sensing", 0xFF: "reset"}
    return system.get(status, "midi")


def _parse_ble_midi_packet(packet: bytes) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not packet:
        return out
    cleaned: List[int] = []
    for b in packet:
        if b & 0x80:
            if b >= 0xF8 or 0x80 <= b <= 0xEF or b in (0xF0, 0xF7):
                cleaned.append(b)
        else:
            cleaned.append(b)
    i = 0
    running_status = None
    while i < len(cleaned):
        b = cleaned[i]
        if b >= 0x80:
            status = b
            running_status = status
            i += 1
        else:
            if running_status is None:
                i += 1
                continue
            status = running_status
        if 0x80 <= status <= 0xBF or 0xE0 <= status <= 0xEF:
            data = cleaned[i:i+2]
            if len(data) < 2:
                break
            i += 2
        elif 0xC0 <= status <= 0xDF:
            data = cleaned[i:i+1]
            if len(data) < 1:
                break
            i += 1
        elif status in (0xF1, 0xF3):
            data = cleaned[i:i+1]
            i += len(data)
        elif status == 0xF2:
            data = cleaned[i:i+2]
            i += len(data)
        elif status == 0xF0:
            data = []
            while i < len(cleaned):
                data.append(cleaned[i])
                if cleaned[i] == 0xF7:
                    i += 1
                    break
                i += 1
        else:
            data = []
        raw = [status] + data
        event = {"type": _status_to_type(status), "status": status, "data": data, "raw": raw}
        if _MIDO_OK:
            try:
                event["mido"] = str(mido.Message.from_bytes(raw))
            except Exception:
                pass
        out.append(event)
    return out


def _event_from_text(text: str) -> Dict[str, Any]:
    e = {"type": "text", "text": text, "data": text}
    if _MIDO_OK:
        try:
            e["mido"] = str(mido.Message.from_str(text))
        except Exception:
            pass
    return e


async def _scan_ble_devices_async(timeout: float) -> List[Dict[str, Any]]:
    if not _BLEAK_OK:
        return []
    found = await BleakScanner.discover(timeout=timeout)
    out = []
    for d in found:
        uuids = []
        try:
            uuids = list(getattr(d, "metadata", {}).get("uuids") or [])
        except Exception:
            uuids = []
        out.append({
            "name": getattr(d, "name", None),
            "address": getattr(d, "address", None),
            "rssi": getattr(d, "rssi", None),
            "uuids": uuids,
            "is_ble_midi": _BLE_MIDI_SERVICE_UUID.lower() in [str(u).lower() for u in uuids],
        })
    return out


def _run_async(coro):
    return asyncio.run(coro)


def _scan_ble_devices(timeout: Optional[float] = None):
    if not _BLEAK_OK:
        return {"ok": False, "error": "bleak_unavailable"}
    timeout = float(timeout or _CONFIG.get("scan_timeout") or 8.0)
    try:
        devices = _run_async(_scan_ble_devices_async(timeout))
        return {"ok": True, "devices": devices, "count": len(devices)}
    except Exception as exc:
        return {"ok": False, "error": "scan_failed", "details": str(exc)}


def _select_scan_result(scan: List[Dict[str, Any]], payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    name = str(payload.get("device_name") or _CONFIG.get("device_name") or "").strip().lower()
    address = str(payload.get("device_address") or _CONFIG.get("device_address") or "").strip().lower()
    candidates = [d for d in scan if d.get("is_ble_midi")] or scan
    for d in candidates:
        if address and str(d.get("address") or "").lower() == address:
            return d
    for d in candidates:
        if name and name in str(d.get("name") or "").lower():
            return d
    return candidates[0] if candidates else None


async def _ble_listen_session(device: Dict[str, Any], stop_event: threading.Event):
    global _CLIENT
    address = device.get("address")
    char_uuid = str(_CONFIG.get("characteristic_uuid") or _BLE_MIDI_IO_CHAR_UUID)
    timeout = float(_CONFIG.get("connect_timeout") or 15.0)

    def handle_notification(sender, data: bytearray):
        raw = bytes(data)
        if _CONFIG.get("emit_raw_packets"):
            _push_event({"type": "raw_packet", "data": list(raw), "sender": str(sender)})
        for event in _parse_ble_midi_packet(raw):
            if event["type"] == "clock" and not _CONFIG.get("emit_clock", True):
                continue
            if event["type"] == "active_sensing" and not _CONFIG.get("emit_active_sensing", False):
                continue
            _push_event(event)

    async with BleakClient(address, timeout=timeout, pair=bool(_CONFIG.get("auto_pair", False))) as client:
        _CLIENT = client
        _SESSION["connected"] = True
        _SESSION["backend"] = "bleak"
        _SESSION["device"] = device
        _SESSION["listener_running"] = True
        await client.start_notify(char_uuid, handle_notification)
        while not stop_event.is_set():
            await asyncio.sleep(0.1)
        try:
            await client.stop_notify(char_uuid)
        except Exception:
            pass
    _SESSION["connected"] = False
    _SESSION["listener_running"] = False
    _CLIENT = None


def _listener_main(device: Dict[str, Any]) -> None:
    try:
        asyncio.run(_ble_listen_session(device, _STOP_EVENT))
    except Exception as exc:
        _SESSION["error"] = str(exc)
        _SESSION["status"] = "error"
        _note(f"Listener failed: {exc}")
    finally:
        _SESSION["listener_running"] = False
        _SESSION["connected"] = False


def _connect(payload: Optional[Dict[str, Any]] = None):
    payload = payload or {}
    if not _BLEAK_OK:
        return {"ok": False, "error": "bleak_unavailable"}
    if _SESSION.get("listener_running"):
        return {"ok": True, "connected": True, "device": _SESSION.get("device"), "already_running": True}
    scan = _scan_ble_devices(payload.get("scan_timeout"))
    if not scan.get("ok"):
        return scan
    device = _select_scan_result(scan.get("devices", []), payload)
    if not device:
        return {"ok": False, "error": "device_not_found"}
    _STOP_EVENT.clear()
    global _LISTENER_THREAD
    _LISTENER_THREAD = threading.Thread(target=_listener_main, args=(device,), daemon=True, name="sm-midi-bt-listener")
    _LISTENER_THREAD.start()
    _SESSION["status"] = "ready"
    _note(f"Connecting to {device.get('name') or device.get('address')}")
    return {"ok": True, "connecting": True, "device": device}


def _disconnect():
    _STOP_EVENT.set()
    try:
        if _LISTENER_THREAD and _LISTENER_THREAD.is_alive():
            _LISTENER_THREAD.join(timeout=2.0)
    except Exception:
        pass
    _SESSION["connected"] = False
    _SESSION["listener_running"] = False
    return {"ok": True, "disconnected": True}


def _send_midi_bytes(data: List[int]):
    if not _CONFIG.get("allow_send", True):
        return {"ok": False, "error": "send_disabled"}
    if data and data[0] == 0xF0 and not _CONFIG.get("allow_vendor_sysex", False):
        return {"ok": False, "error": "sysex_disabled"}
    sent = False
    global _VIRTUAL_OUT
    if _MIDO_OK:
        try:
            if _VIRTUAL_OUT is None:
                try:
                    _VIRTUAL_OUT = mido.open_output(_CONFIG.get("virtual_port_name") or "SarahMemory BLE MIDI", virtual=True)
                except Exception:
                    _VIRTUAL_OUT = None
            if _VIRTUAL_OUT is not None:
                _VIRTUAL_OUT.send(mido.Message.from_bytes(data))
                sent = True
        except Exception:
            pass
    event = {"type": "tx", "data": list(data), "sent": sent}
    _push_event(event)
    return {"ok": sent, "sent": sent, "data": list(data)}


def _virtual_ports_info():
    if not _MIDO_OK:
        return {"ok": False, "error": "mido_unavailable"}
    try:
        return {"ok": True, "inputs": list(mido.get_input_names()), "outputs": list(mido.get_output_names())}
    except Exception as exc:
        return {"ok": False, "error": "midi_ports_failed", "details": str(exc)}


def driver_action(action_id: str, context=None, payload=None):
    global _CONFIG
    payload = payload or {}
    if action_id == "ping":
        return {"ok": True, "pong": True, "ts": time.time()}
    if action_id == "probe_backends":
        return {"ok": True, "bleak": _BLEAK_OK, "mido": _MIDO_OK}
    if action_id == "scan":
        return _scan_ble_devices(payload.get("scan_timeout"))
    if action_id == "connect":
        return _connect(payload)
    if action_id == "disconnect":
        return _disconnect()
    if action_id == "get_status":
        return driver_status(context)
    if action_id == "get_config":
        return {"ok": True, "config": _masked_config()}
    if action_id == "set_config":
        merged = dict(_CONFIG)
        if isinstance(payload, dict):
            merged.update(payload)
        v = driver_validate(merged)
        if not v.get("ok"):
            return {"ok": False, "error": "validation_failed", "details": v}
        _CONFIG = merged
        return {"ok": True, "config": _masked_config(), "validate": v}
    if action_id == "get_events":
        return _get_events(payload.get("limit"))
    if action_id == "clear_events":
        return _clear_events()
    if action_id == "parse_packet":
        data = payload.get("data") or []
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except Exception:
                data = []
        packet = bytes(int(x) & 0xFF for x in data)
        events = [_push_event(e) for e in _parse_ble_midi_packet(packet)]
        return {"ok": True, "events": events, "count": len(events)}
    if action_id == "send_bytes":
        data = payload.get("data") or []
        return _send_midi_bytes([int(x) & 0xFF for x in data])
    if action_id == "send_message":
        if not _MIDO_OK:
            return {"ok": False, "error": "mido_unavailable"}
        try:
            msg = mido.Message.from_str(str(payload.get("message") or ""))
            return _send_midi_bytes(msg.bytes())
        except Exception as exc:
            return {"ok": False, "error": "message_parse_failed", "details": str(exc)}
    if action_id == "feed_text":
        return {"ok": True, "event": _push_event(_event_from_text(str(payload.get("text") or "")))}
    if action_id == "list_midi_ports":
        return _virtual_ports_info()
    if action_id == "safe_stop":
        return _disconnect()
    return {"ok": False, "error": f"Action '{action_id}' not implemented"}
