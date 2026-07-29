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

import json
import os
import platform
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

DRIVER_ID = "com.softdev0.gamepad.bluetooth"
DRIVER_NAME = "Game Controller Bluetooth Driver"
VERSION = "1.0.0"

_SESSION: Dict[str, Any] = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "meta": {},
    "connected": False,
    "active_device": None,
    "last_scan": [],
    "last_state": {},
    "last_event": None,
    "event_buffer": [],
    "notes": [],
    "backend": None,
    "listener_running": False,
}

_LOCK = threading.RLock()
_LISTENER_STOP = threading.Event()
_LISTENER_THREAD: Optional[threading.Thread] = None
_CAPTURE_QUEUE: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=1024)

_GAMEPAD = None
_PYGAME = None
_PYGAME_JOYSTICK = None
_PYGAME_HAPTIC = None

DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "autoload": False,
    "device_name": "",
    "device_mac": "",
    "backend_priority": ["pygame", "inputs", "hid"],
    "auto_pair": False,
    "auto_connect": False,
    "allow_os_pairing_commands": True,
    "scan_seconds": 6,
    "poll_hz": 60,
    "deadzone": 0.12,
    "buffer_limit": 256,
    "normalize_axes": True,
    "button_map": {},
    "axis_map": {},
    "hat_map": {},
    "rumble_supported": True,
    "lightbar_supported": True,
    "vendor_passthrough_enabled": False,
    "pairing_mode": "standard",
}


def _now_id() -> str:
    return "DRV-%s-%sZ-%s" % (
        DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _utc_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _note(msg: str) -> None:
    with _LOCK:
        _SESSION.setdefault("notes", []).append({"ts": _utc_iso(), "msg": msg})
        _SESSION["notes"] = _SESSION["notes"][-50:]


def _mask_mac(value: str) -> str:
    if not value or len(value) < 5:
        return value
    return "**:**:**:" + value.replace("-", ":")[-8:]


def _safe_json(data: Any) -> Any:
    try:
        json.dumps(data)
        return data
    except Exception:
        if isinstance(data, dict):
            return {str(k): _safe_json(v) for k, v in data.items()}
        if isinstance(data, list):
            return [_safe_json(v) for v in data]
        return str(data)


def _merged_config(config: Optional[dict]) -> dict:
    merged = dict(DEFAULTS)
    if isinstance(config, dict):
        for k, v in config.items():
            merged[k] = v
    merged["scan_seconds"] = max(1, int(merged.get("scan_seconds", 6) or 6))
    merged["poll_hz"] = max(1, min(500, int(merged.get("poll_hz", 60) or 60)))
    merged["deadzone"] = max(0.0, min(0.99, float(merged.get("deadzone", 0.12) or 0.12)))
    merged["buffer_limit"] = max(10, min(5000, int(merged.get("buffer_limit", 256) or 256)))
    if not isinstance(merged.get("backend_priority"), list):
        merged["backend_priority"] = ["pygame", "inputs", "hid"]
    return merged


def _public_config(cfg: dict) -> dict:
    public = dict(cfg)
    if public.get("device_mac"):
        public["device_mac"] = _mask_mac(public["device_mac"])
    return public


def driver_info() -> dict:
    with _LOCK:
        return {
            "id": DRIVER_ID,
            "name": DRIVER_NAME,
            "version": VERSION,
            "session": _safe_json(dict(_SESSION)),
        }


def driver_validate(config: dict):
    errors: List[str] = []
    warnings: List[str] = []
    if config is not None and not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be an object"], "warnings": []}
    cfg = _merged_config(config or {})
    if cfg.get("device_mac") and len(str(cfg.get("device_mac"))) < 11:
        errors.append("device_mac looks too short")
    if not cfg.get("enabled", True):
        warnings.append("driver is disabled")
    if not shutil.which("bluetoothctl") and platform.system().lower() == "linux":
        warnings.append("bluetoothctl not found; Linux scan/pair helpers limited")
    if platform.system().lower() == "windows":
        warnings.append("Windows Bluetooth pairing is best-effort only; controller input depends on installed gamepad stack")
    return {"ok": len(errors) == 0, "errors": errors, "warnings": warnings}


def _detect_backend(preferred: List[str]) -> dict:
    available: Dict[str, Any] = {"selected": None, "available": [], "details": {}}
    for name in preferred:
        if name == "pygame":
            try:
                import pygame  # type: ignore
                available["available"].append("pygame")
                available["details"]["pygame"] = getattr(pygame, "version", None) and pygame.version.ver
                if not available["selected"]:
                    available["selected"] = "pygame"
            except Exception as exc:
                available["details"]["pygame"] = f"unavailable: {exc}"
        elif name == "inputs":
            try:
                import inputs  # type: ignore
                available["available"].append("inputs")
                available["details"]["inputs"] = "ok"
                if not available["selected"]:
                    available["selected"] = "inputs"
            except Exception as exc:
                available["details"]["inputs"] = f"unavailable: {exc}"
        elif name == "hid":
            try:
                import hid  # type: ignore
                available["available"].append("hid")
                available["details"]["hid"] = "ok"
                if not available["selected"]:
                    available["selected"] = "hid"
            except Exception as exc:
                available["details"]["hid"] = f"unavailable: {exc}"
    return available


def driver_init(context=None, config=None):
    config = config or {}
    v = driver_validate(config)
    if not v.get("ok"):
        with _LOCK:
            _SESSION["status"] = "error"
            _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": v}

    cfg = _merged_config(config)
    backend = _detect_backend(cfg.get("backend_priority", []))
    with _LOCK:
        _SESSION["instance_id"] = _now_id()
        _SESSION["started_ts"] = time.time()
        _SESSION["status"] = "ready" if cfg.get("enabled", True) else "disabled"
        _SESSION["error"] = None
        _SESSION["meta"] = {"context": context or {}, "config": cfg}
        _SESSION["backend"] = backend
        _SESSION["connected"] = False
        _SESSION["active_device"] = None
        _SESSION["event_buffer"] = []
        _SESSION["last_state"] = {}
    _note("driver initialized")
    return {
        "ok": True,
        "instance_id": _SESSION["instance_id"],
        "validate": v,
        "backend": backend,
        "config": _public_config(cfg),
    }


def _close_backend_objects() -> None:
    global _GAMEPAD, _PYGAME, _PYGAME_JOYSTICK, _PYGAME_HAPTIC
    try:
        if _PYGAME_HAPTIC is not None:
            try:
                _PYGAME_HAPTIC.stop()
            except Exception:
                pass
        _PYGAME_HAPTIC = None
        if _PYGAME_JOYSTICK is not None:
            try:
                _PYGAME_JOYSTICK.quit()
            except Exception:
                pass
        _PYGAME_JOYSTICK = None
        if _PYGAME is not None:
            try:
                _PYGAME.joystick.quit()
            except Exception:
                pass
            try:
                _PYGAME.quit()
            except Exception:
                pass
        _PYGAME = None
        _GAMEPAD = None
    except Exception:
        pass


def driver_shutdown(context=None):
    _stop_listener()
    _close_backend_objects()
    with _LOCK:
        _SESSION["status"] = "stopped"
        _SESSION["error"] = None
        _SESSION["meta"] = {}
        _SESSION["instance_id"] = None
        _SESSION["started_ts"] = None
        _SESSION["connected"] = False
        _SESSION["active_device"] = None
        _SESSION["listener_running"] = False
    _note("driver shutdown")
    return True


def driver_status(context=None):
    with _LOCK:
        cfg = _merged_config((_SESSION.get("meta") or {}).get("config") or {})
        return {
            "ok": True,
            "session": _safe_json(dict(_SESSION)),
            "config": _public_config(cfg),
        }


def _run_cmd(cmd: List[str], timeout: int = 15) -> dict:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "ok": p.returncode == 0,
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "cmd": cmd,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "cmd": cmd}


def _scan_linux(seconds: int) -> List[dict]:
    if not shutil.which("bluetoothctl"):
        return []
    # passive paired/trusted list first
    devices: Dict[str, dict] = {}
    for base_cmd in (["bluetoothctl", "devices"], ["bluetoothctl", "paired-devices"]):
        out = _run_cmd(base_cmd, timeout=8)
        for line in str(out.get("stdout", "")).splitlines():
            line = line.strip()
            if line.startswith("Device "):
                parts = line.split(" ", 2)
                if len(parts) >= 3:
                    mac = parts[1].strip()
                    name = parts[2].strip()
                    devices[mac] = {"name": name, "mac": mac, "paired": base_cmd[1] == "paired-devices", "source": "bluetoothctl"}
    # optional short scan session
    subprocess.run(["bluetoothctl", "scan", "on"], capture_output=True, text=True, timeout=8)
    time.sleep(min(max(seconds, 1), 8))
    subprocess.run(["bluetoothctl", "scan", "off"], capture_output=True, text=True, timeout=8)
    out = _run_cmd(["bluetoothctl", "devices"], timeout=8)
    for line in str(out.get("stdout", "")).splitlines():
        line = line.strip()
        if line.startswith("Device "):
            parts = line.split(" ", 2)
            if len(parts) >= 3:
                mac = parts[1].strip()
                name = parts[2].strip()
                record = devices.get(mac, {"mac": mac})
                record.update({"name": name, "source": "bluetoothctl"})
                devices[mac] = record
    return list(devices.values())


def _scan_windows(seconds: int) -> List[dict]:
    cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-PnpDevice -Class Bluetooth | Select-Object FriendlyName,InstanceId,Status | ConvertTo-Json -Depth 3",
    ]
    out = _run_cmd(cmd, timeout=12)
    items: List[dict] = []
    if out.get("ok") and out.get("stdout"):
        try:
            data = json.loads(out["stdout"])
            if isinstance(data, dict):
                data = [data]
            for item in data:
                name = item.get("FriendlyName") or item.get("InstanceId") or "Bluetooth Device"
                items.append({"name": name, "instance_id": item.get("InstanceId"), "status": item.get("Status"), "source": "powershell"})
        except Exception:
            pass
    return items


def _scan_macos(seconds: int) -> List[dict]:
    # system_profiler is slow but broadly available
    out = _run_cmd(["system_profiler", "SPBluetoothDataType", "-json"], timeout=20)
    items: List[dict] = []
    if out.get("ok") and out.get("stdout"):
        try:
            data = json.loads(out["stdout"])
            blob = json.dumps(data)
            items.append({"name": "Bluetooth inventory available", "details": "Use host pairing UI for exact connect flow", "source": "system_profiler", "summary_len": len(blob)})
        except Exception:
            pass
    return items


def _scan_bt(seconds: int) -> dict:
    system = platform.system().lower()
    if system == "linux":
        devices = _scan_linux(seconds)
    elif system == "windows":
        devices = _scan_windows(seconds)
    elif system == "darwin":
        devices = _scan_macos(seconds)
    else:
        devices = []
    with _LOCK:
        _SESSION["last_scan"] = devices
    return {"ok": True, "devices": devices, "count": len(devices), "platform": system}


def _pair_linux(mac: str, trust: bool = True, connect: bool = True) -> dict:
    result = {"ok": True, "steps": []}
    if not shutil.which("bluetoothctl"):
        return {"ok": False, "error": "bluetoothctl not found"}
    for cmd in (["bluetoothctl", "pair", mac], ["bluetoothctl", "trust", mac] if trust else None, ["bluetoothctl", "connect", mac] if connect else None):
        if not cmd:
            continue
        step = _run_cmd(cmd, timeout=25)
        result["steps"].append(step)
        if not step.get("ok"):
            result["ok"] = False
    return result


def _pair_best_effort(name: str, mac: str, cfg: dict) -> dict:
    system = platform.system().lower()
    if not cfg.get("allow_os_pairing_commands", True):
        return {"ok": False, "error": "os pairing commands disabled"}
    if system == "linux" and mac:
        return _pair_linux(mac, trust=True, connect=True)
    return {
        "ok": False,
        "error": "pairing must be completed by host OS for this platform/device",
        "platform": system,
        "device_name": name,
        "device_mac": _mask_mac(mac),
    }


def _pygame_list_devices() -> List[dict]:
    global _PYGAME
    import pygame  # type: ignore
    _PYGAME = pygame
    if not pygame.get_init():
        pygame.init()
    pygame.joystick.init()
    devices = []
    count = pygame.joystick.get_count()
    for idx in range(count):
        js = pygame.joystick.Joystick(idx)
        js.init()
        devices.append({
            "index": idx,
            "name": js.get_name(),
            "guid": getattr(js, "get_guid", lambda: None)(),
            "num_axes": js.get_numaxes(),
            "num_buttons": js.get_numbuttons(),
            "num_hats": js.get_numhats(),
            "backend": "pygame",
        })
    return devices


def _pygame_connect(selector: dict) -> dict:
    global _PYGAME, _PYGAME_JOYSTICK
    devices = _pygame_list_devices()
    target = None
    wanted_name = str(selector.get("device_name") or "").strip().lower()
    wanted_index = selector.get("index")
    for item in devices:
        if wanted_index is not None and item.get("index") == wanted_index:
            target = item
            break
        if wanted_name and wanted_name in str(item.get("name", "")).lower():
            target = item
            break
    if target is None and devices:
        target = devices[0]
    if target is None:
        return {"ok": False, "error": "no gamepad detected by pygame", "devices": devices}
    js = _PYGAME.joystick.Joystick(target["index"])
    js.init()
    _PYGAME_JOYSTICK = js
    with _LOCK:
        _SESSION["connected"] = True
        _SESSION["active_device"] = target
        _SESSION["backend"]["selected"] = "pygame"
    return {"ok": True, "device": target, "devices": devices}


def _pygame_state(cfg: dict) -> dict:
    if _PYGAME is None or _PYGAME_JOYSTICK is None:
        return {"ok": False, "error": "pygame joystick not connected"}
    try:
        _PYGAME.event.pump()
    except Exception:
        pass
    dz = float(cfg.get("deadzone", 0.12) or 0.12)
    axes = {}
    for i in range(_PYGAME_JOYSTICK.get_numaxes()):
        val = float(_PYGAME_JOYSTICK.get_axis(i))
        if abs(val) < dz:
            val = 0.0
        axes[str(i)] = round(val, 4)
    buttons = {str(i): bool(_PYGAME_JOYSTICK.get_button(i)) for i in range(_PYGAME_JOYSTICK.get_numbuttons())}
    hats = {str(i): _PYGAME_JOYSTICK.get_hat(i) for i in range(_PYGAME_JOYSTICK.get_numhats())}
    state = {"axes": axes, "buttons": buttons, "hats": hats, "ts": _utc_iso()}
    with _LOCK:
        _SESSION["last_state"] = state
    return {"ok": True, "state": state}


def _push_event(evt: dict, cfg: dict) -> None:
    evt = dict(evt)
    evt.setdefault("ts", _utc_iso())
    with _LOCK:
        _SESSION["last_event"] = evt
        buf = _SESSION.setdefault("event_buffer", [])
        buf.append(evt)
        limit = int(cfg.get("buffer_limit", 256) or 256)
        _SESSION["event_buffer"] = buf[-limit:]
    try:
        _CAPTURE_QUEUE.put_nowait(evt)
    except Exception:
        pass


def _listener_loop(cfg: dict) -> None:
    global _PYGAME
    hz = max(1, int(cfg.get("poll_hz", 60) or 60))
    delay = 1.0 / hz
    _note("listener started")
    while not _LISTENER_STOP.is_set():
        backend = ((_SESSION.get("backend") or {}).get("selected"))
        if backend == "pygame" and _PYGAME is not None and _PYGAME_JOYSTICK is not None:
            try:
                for event in _PYGAME.event.get():
                    out = {"type": str(getattr(event, "type", "unknown"))}
                    for attr in ("button", "axis", "value", "hat", "instance_id"):
                        if hasattr(event, attr):
                            out[attr] = getattr(event, attr)
                    _push_event(out, cfg)
                _pygame_state(cfg)
            except Exception as exc:
                _note(f"listener error: {exc}")
        time.sleep(delay)
    _note("listener stopped")


def _start_listener(cfg: dict) -> dict:
    global _LISTENER_THREAD
    if _LISTENER_THREAD and _LISTENER_THREAD.is_alive():
        return {"ok": True, "already_running": True}
    _LISTENER_STOP.clear()
    _LISTENER_THREAD = threading.Thread(target=_listener_loop, args=(cfg,), daemon=True, name="gamepad-bt-listener")
    _LISTENER_THREAD.start()
    with _LOCK:
        _SESSION["listener_running"] = True
    return {"ok": True, "listener_running": True}


def _stop_listener() -> dict:
    global _LISTENER_THREAD
    _LISTENER_STOP.set()
    if _LISTENER_THREAD and _LISTENER_THREAD.is_alive():
        _LISTENER_THREAD.join(timeout=1.5)
    with _LOCK:
        _SESSION["listener_running"] = False
    return {"ok": True, "listener_running": False}


def _disconnect() -> dict:
    _stop_listener()
    _close_backend_objects()
    with _LOCK:
        _SESSION["connected"] = False
        _SESSION["active_device"] = None
    return {"ok": True}


def _set_light(payload: dict) -> dict:
    # Generic contract only. Actual per-controller LED/lightbar control depends on backend/vendor support.
    color = payload.get("color") or payload.get("rgb") or {"r": 0, "g": 0, "b": 255}
    enabled = bool(payload.get("enabled", True))
    _note(f"light request accepted: enabled={enabled} color={color}")
    return {
        "ok": False,
        "error": "generic bluetooth gamepad light control requires vendor-specific backend support",
        "requested": {"enabled": enabled, "color": color},
    }


def _rumble(payload: dict) -> dict:
    duration_ms = int(payload.get("duration_ms", 300) or 300)
    strong = float(payload.get("strong", 1.0) or 1.0)
    weak = float(payload.get("weak", 1.0) or 1.0)
    # pygame haptics support varies heavily
    _note(f"rumble request duration={duration_ms} strong={strong} weak={weak}")
    return {
        "ok": False,
        "error": "generic rumble requires controller/backend support not universally exposed",
        "requested": {"duration_ms": duration_ms, "strong": strong, "weak": weak},
    }


def _vendor_passthrough(payload: dict, cfg: dict) -> dict:
    if not cfg.get("vendor_passthrough_enabled", False):
        return {"ok": False, "error": "vendor_passthrough_disabled"}
    return {"ok": False, "error": "vendor passthrough not available in generic package", "payload": _safe_json(payload)}


def driver_action(action_id: str, context=None, payload=None):
    payload = payload or {}
    cfg = _merged_config((_SESSION.get("meta") or {}).get("config") or {})

    try:
        if action_id == "ping":
            return {"ok": True, "driver": DRIVER_ID, "ts": _utc_iso()}

        if action_id == "get_config":
            return {"ok": True, "config": _public_config(cfg)}

        if action_id == "set_config":
            cfg.update(payload or {})
            cfg = _merged_config(cfg)
            with _LOCK:
                _SESSION["meta"]["config"] = cfg
            return {"ok": True, "config": _public_config(cfg)}

        if action_id == "backend_probe":
            info = _detect_backend(cfg.get("backend_priority", []))
            with _LOCK:
                _SESSION["backend"] = info
            return {"ok": True, "backend": info}

        if action_id in ("scan_bt", "scan", "discover_devices"):
            return _scan_bt(int(payload.get("scan_seconds", cfg.get("scan_seconds", 6)) or 6))

        if action_id in ("pair_connect", "pair", "connect"):
            selector = {
                "device_name": payload.get("device_name", cfg.get("device_name", "")),
                "device_mac": payload.get("device_mac", cfg.get("device_mac", "")),
                "index": payload.get("index"),
            }
            if payload.get("pair") or action_id in ("pair_connect", "pair"):
                pair_res = _pair_best_effort(selector.get("device_name", ""), selector.get("device_mac", ""), cfg)
                if not pair_res.get("ok") and action_id == "pair":
                    return pair_res
            backend = ((_SESSION.get("backend") or {}).get("selected")) or _detect_backend(cfg.get("backend_priority", [])) .get("selected")
            if backend == "pygame":
                res = _pygame_connect(selector)
            else:
                res = {"ok": False, "error": "no supported input backend available", "backend": _SESSION.get("backend")}
            if res.get("ok") and bool(payload.get("start_listen", True)):
                _start_listener(cfg)
            return res

        if action_id == "disconnect":
            return _disconnect()

        if action_id == "list_gamepads":
            devices = []
            try:
                devices.extend(_pygame_list_devices())
            except Exception as exc:
                devices.append({"backend": "pygame", "error": str(exc)})
            return {"ok": True, "devices": devices}

        if action_id == "start_listen":
            return _start_listener(cfg)

        if action_id == "stop_listen":
            return _stop_listener()

        if action_id in ("get_state", "poll_state"):
            backend = ((_SESSION.get("backend") or {}).get("selected"))
            if backend == "pygame":
                return _pygame_state(cfg)
            return {"ok": False, "error": "no active backend"}

        if action_id == "get_events":
            with _LOCK:
                return {"ok": True, "events": list(_SESSION.get("event_buffer", []))}

        if action_id == "clear_events":
            with _LOCK:
                _SESSION["event_buffer"] = []
            return {"ok": True}

        if action_id == "inject_event":
            _push_event(payload or {"type": "synthetic"}, cfg)
            return {"ok": True, "injected": _safe_json(payload)}

        if action_id == "set_light":
            return _set_light(payload)

        if action_id == "rumble":
            return _rumble(payload)

        if action_id == "vendor_passthrough":
            return _vendor_passthrough(payload, cfg)

        return {"ok": False, "error": f"Action '{action_id}' not implemented by {DRIVER_ID}", "action_id": action_id}
    except Exception as exc:
        with _LOCK:
            _SESSION["error"] = str(exc)
        return {"ok": False, "error": str(exc), "action_id": action_id}
