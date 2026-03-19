# SarahMemory Driver Package: com.softdev0.midi.usb
# Name: MIDI USB Driver
# Purpose: Governed USB MIDI input/output driver using optional Python MIDI backends.
# Notes:
# - No OS driver installation.
# - No auto-probing unless explicitly requested.
# - Uses optional libraries when available and degrades cleanly when missing.

import time
import threading
from datetime import datetime
from copy import deepcopy

DRIVER_ID = "com.softdev0.midi.usb"
DRIVER_NAME = "MIDI USB Driver"
VERSION = "1.0.0"

DEFAULT_CONFIG = {
    "enabled": True,
    "autoload": False,
    "backend_priority": ["mido", "pygame.midi"],
    "auto_detect": True,
    "input_name": "",
    "output_name": "",
    "device_filter": "",
    "throttle_ms": 0,
    "poll_interval_ms": 10,
    "event_buffer_limit": 512,
    "emit_raw": False,
    "ignore_active_sensing": True,
    "ignore_clock": False,
    "ignore_sysex": False,
    "allow_output": True,
    "allow_sysex_output": False,
    "virtual_input_name": "SarahMemory MIDI In",
    "virtual_output_name": "SarahMemory MIDI Out",
    "default_channel": 0,
    "velocity_default": 100,
}

_SESSION = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "meta": {},
    "config": deepcopy(DEFAULT_CONFIG),
    "backend": None,
    "connected": False,
    "input_name": None,
    "output_name": None,
    "last_poll_ts": 0.0,
    "events": [],
    "listener_thread": None,
    "stop_event": None,
    "input_port": None,
    "output_port": None,
    "notes": [],
    "last_result": None,
}


def _now_id():
    return "DRV-%s-%sZ-%s" % (
        DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _note(msg):
    _SESSION["notes"].append({"ts": time.time(), "msg": str(msg)})
    _SESSION["notes"] = _SESSION["notes"][-50:]


def _mask_config(cfg):
    return dict(cfg or {})


def _normalize_config(config):
    cfg = deepcopy(DEFAULT_CONFIG)
    if isinstance(config, dict):
        cfg.update(config)
    if not isinstance(cfg.get("backend_priority"), list):
        cfg["backend_priority"] = list(DEFAULT_CONFIG["backend_priority"])
    cfg["event_buffer_limit"] = max(1, int(cfg.get("event_buffer_limit", 512)))
    cfg["throttle_ms"] = max(0, int(cfg.get("throttle_ms", 0)))
    cfg["poll_interval_ms"] = max(1, int(cfg.get("poll_interval_ms", 10)))
    cfg["default_channel"] = max(0, min(15, int(cfg.get("default_channel", 0))))
    cfg["velocity_default"] = max(0, min(127, int(cfg.get("velocity_default", 100))))
    return cfg


def _truncate_events():
    limit = int(_SESSION["config"].get("event_buffer_limit", 512))
    _SESSION["events"] = _SESSION["events"][-limit:]


def _append_event(evt):
    evt = dict(evt or {})
    evt.setdefault("ts", time.time())
    _SESSION["events"].append(evt)
    _truncate_events()


def _safe_json_value(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_safe_json_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe_json_value(val) for k, val in v.items()}
    return str(v)


def _import_mido():
    try:
        import mido  # type: ignore
        return mido
    except Exception:
        return None


def _import_pygame_midi():
    try:
        import pygame.midi as pgm  # type: ignore
        return pgm
    except Exception:
        return None


def _probe_backends():
    mido = _import_mido()
    pgm = _import_pygame_midi()
    result = {
        "mido": {
            "available": bool(mido),
            "version": getattr(mido, "__version__", None) if mido else None,
            "backend": None,
        },
        "pygame.midi": {
            "available": bool(pgm),
            "version": getattr(pgm, "ver", None) if pgm else None,
        },
    }
    if mido:
        try:
            result["mido"]["backend"] = str(mido.backend)
        except Exception:
            pass
    return result


def _choose_backend():
    probe = _probe_backends()
    for name in _SESSION["config"].get("backend_priority", []):
        if probe.get(name, {}).get("available"):
            return name
    if probe["mido"]["available"]:
        return "mido"
    if probe["pygame.midi"]["available"]:
        return "pygame.midi"
    return None


def _list_devices_mido():
    mido = _import_mido()
    if not mido:
        return {"ok": False, "error": "mido_unavailable", "devices": []}
    try:
        ins = list(mido.get_input_names())
        outs = list(mido.get_output_names())
        return {
            "ok": True,
            "backend": "mido",
            "inputs": ins,
            "outputs": outs,
            "devices": [{"name": n, "direction": "input"} for n in ins] + [{"name": n, "direction": "output"} for n in outs],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "devices": []}


def _list_devices_pygame():
    pgm = _import_pygame_midi()
    if not pgm:
        return {"ok": False, "error": "pygame_midi_unavailable", "devices": []}
    try:
        if not pgm.get_init():
            pgm.init()
        count = pgm.get_count()
        devices = []
        for idx in range(count):
            info = pgm.get_device_info(idx)
            iface, name, is_input, is_output, opened = info
            devices.append({
                "id": idx,
                "interface": iface.decode(errors="ignore") if isinstance(iface, bytes) else str(iface),
                "name": name.decode(errors="ignore") if isinstance(name, bytes) else str(name),
                "is_input": bool(is_input),
                "is_output": bool(is_output),
                "opened": bool(opened),
            })
        return {"ok": True, "backend": "pygame.midi", "devices": devices}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "devices": []}


def _list_devices():
    backend = _SESSION.get("backend") or _choose_backend()
    if backend == "mido":
        return _list_devices_mido()
    if backend == "pygame.midi":
        return _list_devices_pygame()
    return {"ok": False, "error": "no_midi_backend", "devices": []}


def _match_name(candidates, requested, filter_text=""):
    requested = (requested or "").strip().lower()
    filter_text = (filter_text or "").strip().lower()
    if requested:
        for item in candidates:
            name = str(item).lower()
            if requested == name or requested in name:
                return item
    if filter_text:
        for item in candidates:
            name = str(item).lower()
            if filter_text in name:
                return item
    return candidates[0] if candidates else None


def _connect_mido(payload=None):
    payload = payload or {}
    mido = _import_mido()
    if not mido:
        return {"ok": False, "error": "mido_unavailable"}
    cfg = _SESSION["config"]
    try:
        input_names = list(mido.get_input_names())
        output_names = list(mido.get_output_names())
        input_name = payload.get("input_name") or cfg.get("input_name")
        output_name = payload.get("output_name") or cfg.get("output_name")
        filt = payload.get("device_filter") or cfg.get("device_filter")
        if cfg.get("auto_detect", True):
            input_name = _match_name(input_names, input_name, filt)
            output_name = _match_name(output_names, output_name, filt)
        if input_name:
            _SESSION["input_port"] = mido.open_input(input_name)
            _SESSION["input_name"] = input_name
        if output_name:
            _SESSION["output_port"] = mido.open_output(output_name)
            _SESSION["output_name"] = output_name
        _SESSION["connected"] = bool(_SESSION["input_port"] or _SESSION["output_port"])
        return {
            "ok": _SESSION["connected"],
            "backend": "mido",
            "input_name": _SESSION["input_name"],
            "output_name": _SESSION["output_name"],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "backend": "mido"}


def _find_pygame_device(devices, want_input, requested, filter_text=""):
    cands = []
    for d in devices:
        if want_input and d.get("is_input"):
            cands.append(d)
        if (not want_input) and d.get("is_output"):
            cands.append(d)
    if requested:
        low = str(requested).lower()
        for d in cands:
            if low == str(d.get("id")).lower() or low in str(d.get("name","")).lower():
                return d
    if filter_text:
        low = str(filter_text).lower()
        for d in cands:
            if low in str(d.get("name","")).lower():
                return d
    return cands[0] if cands else None


def _connect_pygame(payload=None):
    payload = payload or {}
    pgm = _import_pygame_midi()
    if not pgm:
        return {"ok": False, "error": "pygame_midi_unavailable"}
    cfg = _SESSION["config"]
    try:
        if not pgm.get_init():
            pgm.init()
        devices = _list_devices_pygame().get("devices", [])
        filt = payload.get("device_filter") or cfg.get("device_filter")
        in_req = payload.get("input_name") or cfg.get("input_name")
        out_req = payload.get("output_name") or cfg.get("output_name")
        in_dev = _find_pygame_device(devices, True, in_req, filt)
        out_dev = _find_pygame_device(devices, False, out_req, filt)
        if in_dev:
            _SESSION["input_port"] = pgm.Input(in_dev["id"])
            _SESSION["input_name"] = in_dev.get("name")
        if out_dev:
            _SESSION["output_port"] = pgm.Output(out_dev["id"])
            _SESSION["output_name"] = out_dev.get("name")
        _SESSION["connected"] = bool(_SESSION["input_port"] or _SESSION["output_port"])
        return {
            "ok": _SESSION["connected"],
            "backend": "pygame.midi",
            "input_name": _SESSION["input_name"],
            "output_name": _SESSION["output_name"],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "backend": "pygame.midi"}


def _disconnect_ports():
    for key in ("input_port", "output_port"):
        port = _SESSION.get(key)
        if port is not None:
            try:
                close = getattr(port, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            _SESSION[key] = None
    _SESSION["input_name"] = None
    _SESSION["output_name"] = None
    _SESSION["connected"] = False
    return {"ok": True}


def _message_to_event(msg):
    evt = {"type": getattr(msg, "type", "unknown")}
    for attr in ("channel", "note", "velocity", "control", "value", "program", "pitch", "time"):
        if hasattr(msg, attr):
            evt[attr] = _safe_json_value(getattr(msg, attr))
    if hasattr(msg, "bytes"):
        try:
            evt["bytes"] = list(msg.bytes())
        except Exception:
            pass
    evt["repr"] = str(msg)
    return evt


def _filter_event(evt):
    cfg = _SESSION["config"]
    t = evt.get("type")
    if cfg.get("ignore_active_sensing") and t == "active_sensing":
        return False
    if cfg.get("ignore_clock") and t == "clock":
        return False
    if cfg.get("ignore_sysex") and t == "sysex":
        return False
    return True


def _poll_mido_once():
    port = _SESSION.get("input_port")
    if port is None:
        return {"ok": False, "error": "not_connected"}
    events = []
    try:
        pending = getattr(port, "iter_pending", None)
        msgs = list(pending()) if callable(pending) else []
        for msg in msgs:
            evt = _message_to_event(msg)
            if _filter_event(evt):
                _append_event(evt)
                events.append(evt)
        return {"ok": True, "count": len(events), "events": events}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _poll_pygame_once():
    port = _SESSION.get("input_port")
    if port is None:
        return {"ok": False, "error": "not_connected"}
    events = []
    try:
        while port.poll():
            data = port.read(32)
            for raw, timestamp in data:
                evt = {
                    "type": "raw_midi",
                    "bytes": list(raw),
                    "timestamp": timestamp,
                }
                if _filter_event(evt):
                    _append_event(evt)
                    events.append(evt)
        return {"ok": True, "count": len(events), "events": events}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _listener_loop():
    stop_event = _SESSION.get("stop_event")
    while stop_event and not stop_event.is_set():
        now = time.time() * 1000.0
        throttle = _SESSION["config"].get("throttle_ms", 0)
        if now - (_SESSION.get("last_poll_ts") or 0) >= throttle:
            if _SESSION.get("backend") == "mido":
                _poll_mido_once()
            elif _SESSION.get("backend") == "pygame.midi":
                _poll_pygame_once()
            _SESSION["last_poll_ts"] = now
        time.sleep(_SESSION["config"].get("poll_interval_ms", 10) / 1000.0)


def _start_listener():
    if _SESSION.get("listener_thread") and _SESSION["listener_thread"].is_alive():
        return {"ok": True, "already_running": True}
    _SESSION["stop_event"] = threading.Event()
    t = threading.Thread(target=_listener_loop, name="MidiUsbListener", daemon=True)
    _SESSION["listener_thread"] = t
    t.start()
    return {"ok": True, "started": True}


def _stop_listener():
    ev = _SESSION.get("stop_event")
    if ev:
        ev.set()
    t = _SESSION.get("listener_thread")
    if t and t.is_alive():
        t.join(timeout=1.0)
    _SESSION["listener_thread"] = None
    _SESSION["stop_event"] = None
    return {"ok": True}


def _send_mido_message(payload):
    if not _SESSION["config"].get("allow_output", True):
        return {"ok": False, "error": "output_disabled"}
    port = _SESSION.get("output_port")
    if port is None:
        return {"ok": False, "error": "output_not_connected"}
    mido = _import_mido()
    if not mido:
        return {"ok": False, "error": "mido_unavailable"}
    msg_type = payload.get("type", "note_on")
    if msg_type == "sysex" and not _SESSION["config"].get("allow_sysex_output", False):
        return {"ok": False, "error": "sysex_disabled"}
    try:
        kwargs = {k: v for k, v in payload.items() if k != "type"}
        if "channel" not in kwargs:
            kwargs["channel"] = _SESSION["config"].get("default_channel", 0)
        if msg_type in ("note_on", "note_off") and "velocity" not in kwargs:
            kwargs["velocity"] = _SESSION["config"].get("velocity_default", 100)
        msg = mido.Message(msg_type, **kwargs)
        port.send(msg)
        evt = {"type": "sent", "message": _message_to_event(msg)}
        _append_event(evt)
        return {"ok": True, "sent": evt}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _send_pygame_raw(payload):
    if not _SESSION["config"].get("allow_output", True):
        return {"ok": False, "error": "output_disabled"}
    port = _SESSION.get("output_port")
    if port is None:
        return {"ok": False, "error": "output_not_connected"}
    data = payload.get("bytes") or []
    if not isinstance(data, list) or not data:
        return {"ok": False, "error": "bytes_required"}
    try:
        port.write_short(*([int(x) & 0xFF for x in data] + [0, 0, 0])[:4])
        evt = {"type": "sent_raw", "bytes": [int(x) & 0xFF for x in data]}
        _append_event(evt)
        return {"ok": True, "sent": evt}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def driver_info():
    return {
        "id": DRIVER_ID,
        "name": DRIVER_NAME,
        "version": VERSION,
        "session": driver_status().get("session"),
    }


def driver_validate(config: dict):
    errors = []
    warnings = []
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be an object"], "warnings": []}
    try:
        _normalize_config(config)
    except Exception as exc:
        errors.append(str(exc))
    if config.get("allow_sysex_output") and not config.get("allow_output", True):
        warnings.append("allow_sysex_output ignored while allow_output is false")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def driver_init(context=None, config=None):
    config = config or {}
    v = driver_validate(config)
    if not v.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": v}
    _SESSION["instance_id"] = _now_id()
    _SESSION["started_ts"] = time.time()
    _SESSION["status"] = "ready"
    _SESSION["error"] = None
    _SESSION["meta"] = {"context": context or {}}
    _SESSION["config"] = _normalize_config(config)
    _SESSION["backend"] = _choose_backend()
    _SESSION["events"] = []
    _SESSION["notes"] = []
    _SESSION["last_result"] = None
    _note(f"initialized backend={_SESSION['backend']}")
    return {"ok": True, "instance_id": _SESSION["instance_id"], "validate": v, "info": driver_info()}


def driver_shutdown(context=None):
    _stop_listener()
    _disconnect_ports()
    try:
        pgm = _import_pygame_midi()
        if pgm and pgm.get_init():
            pgm.quit()
    except Exception:
        pass
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _SESSION["meta"] = {}
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    _SESSION["backend"] = None
    return True


def driver_status(context=None):
    return {
        "ok": True,
        "session": {
            "instance_id": _SESSION.get("instance_id"),
            "started_ts": _SESSION.get("started_ts"),
            "status": _SESSION.get("status"),
            "error": _SESSION.get("error"),
            "backend": _SESSION.get("backend"),
            "connected": _SESSION.get("connected"),
            "input_name": _SESSION.get("input_name"),
            "output_name": _SESSION.get("output_name"),
            "config": _mask_config(_SESSION.get("config")),
            "event_count": len(_SESSION.get("events", [])),
            "notes": list(_SESSION.get("notes", [])),
            "last_result": _safe_json_value(_SESSION.get("last_result")),
        },
    }


def driver_action(action_id: str, context=None, payload=None):
    payload = payload or {}
    try:
        if action_id == "ping":
            res = {"ok": True, "driver": DRIVER_ID, "ts": time.time()}
        elif action_id == "probe_backends":
            res = {"ok": True, "backends": _probe_backends()}
        elif action_id == "list_devices":
            res = _list_devices()
        elif action_id == "connect":
            if _SESSION.get("backend") is None:
                _SESSION["backend"] = _choose_backend()
            if _SESSION.get("backend") == "mido":
                res = _connect_mido(payload)
            elif _SESSION.get("backend") == "pygame.midi":
                res = _connect_pygame(payload)
            else:
                res = {"ok": False, "error": "no_midi_backend"}
        elif action_id == "disconnect":
            _stop_listener()
            res = _disconnect_ports()
        elif action_id == "start_listener":
            res = _start_listener()
        elif action_id == "stop_listener":
            res = _stop_listener()
        elif action_id == "poll_once":
            if _SESSION.get("backend") == "mido":
                res = _poll_mido_once()
            elif _SESSION.get("backend") == "pygame.midi":
                res = _poll_pygame_once()
            else:
                res = {"ok": False, "error": "no_midi_backend"}
        elif action_id == "get_events":
            res = {"ok": True, "events": list(_SESSION.get("events", []))}
        elif action_id == "clear_events":
            _SESSION["events"] = []
            res = {"ok": True, "cleared": True}
        elif action_id == "send_message":
            if _SESSION.get("backend") == "mido":
                res = _send_mido_message(payload)
            elif _SESSION.get("backend") == "pygame.midi":
                res = _send_pygame_raw(payload)
            else:
                res = {"ok": False, "error": "no_midi_backend"}
        elif action_id == "send_raw":
            res = _send_pygame_raw(payload)
        elif action_id == "feed_text":
            text = str(payload.get("text", ""))
            events = []
            for token in text.split():
                evt = {"type": "text_token", "text": token}
                _append_event(evt)
                events.append(evt)
            res = {"ok": True, "count": len(events), "events": events}
        elif action_id == "parse_packet":
            data = payload.get("bytes") or []
            evt = {"type": "raw_packet", "bytes": [int(x) & 0xFF for x in data]}
            _append_event(evt)
            res = {"ok": True, "event": evt}
        elif action_id == "get_config":
            res = {"ok": True, "config": _mask_config(_SESSION.get("config"))}
        elif action_id == "set_config":
            candidate = _normalize_config({**_SESSION.get("config", {}), **(payload.get("config") or {})})
            val = driver_validate(candidate)
            if not val.get("ok"):
                res = {"ok": False, "error": "validation_failed", "details": val}
            else:
                _SESSION["config"] = candidate
                res = {"ok": True, "config": _mask_config(candidate), "validate": val}
        elif action_id == "safe_stop":
            _stop_listener()
            res = _disconnect_ports()
        else:
            res = {"ok": False, "error": f"Action '{action_id}' not implemented by {DRIVER_ID}", "action_id": action_id}
    except Exception as exc:
        _SESSION["error"] = str(exc)
        res = {"ok": False, "error": str(exc), "action_id": action_id}
    _SESSION["last_result"] = _safe_json_value(res)
    return res
