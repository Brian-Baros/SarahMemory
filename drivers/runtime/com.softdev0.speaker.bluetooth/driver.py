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
import os
import platform
import shutil
import socket
import subprocess
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

DRIVER_ID = "com.softdev0.speaker.bluetooth"
DRIVER_NAME = "Bluetooth Speaker/Soundbar Driver"
VERSION = "1.0.0"

_SESSION = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "connected": False,
    "device": {},
    "backend": None,
    "notes": [],
    "history": [],
    "listener_running": False,
}
_LOCK = threading.RLock()
_STOP = threading.Event()
_LISTENER = None


def _now_id() -> str:
    return "DRV-%s-%sZ-%s" % (
        DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _add_note(msg: str) -> None:
    with _LOCK:
        _SESSION.setdefault("notes", []).append(msg)
        _SESSION["notes"] = _SESSION["notes"][-50:]


def _add_history(action: str, ok: bool, result: Optional[Dict[str, Any]] = None) -> None:
    with _LOCK:
        _SESSION.setdefault("history", []).append(
            {
                "ts": time.time(),
                "action": action,
                "ok": bool(ok),
                "result": result or {},
            }
        )
        _SESSION["history"] = _SESSION["history"][-100:]


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _mask_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cfg or {})
    if out.get("pairing_pin"):
        out["pairing_pin"] = "***"
    return out


def driver_info():
    return {
        "id": DRIVER_ID,
        "name": DRIVER_NAME,
        "version": VERSION,
        "session": dict(_SESSION),
    }


def driver_validate(config: dict):
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be an object"], "warnings": []}
    for k in ("enabled", "autoload", "auto_pair", "auto_connect", "allow_pair", "allow_disconnect", "allow_volume_control", "allow_media_control", "allow_device_commands"):
        if k in config and not isinstance(config.get(k), bool):
            errors.append(f"{k} must be boolean")
    if "port" in config and not isinstance(config.get("port"), int):
        errors.append("port must be int")
    if "volume_step" in config and not isinstance(config.get("volume_step"), int):
        errors.append("volume_step must be int")
    if "connection_timeout_sec" in config and not isinstance(config.get("connection_timeout_sec"), (int, float)):
        errors.append("connection_timeout_sec must be number")
    if not config.get("device_name") and not config.get("device_mac"):
        warnings.append("device_name or device_mac should be set for deterministic connect")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def _run(cmd: List[str], timeout: float = 15.0) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "ok": proc.returncode == 0,
            "code": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "cmd": cmd,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "cmd": cmd}


def _probe_backends() -> Dict[str, Any]:
    system = platform.system().lower()
    return {
        "ok": True,
        "system": system,
        "bluetoothctl": _which("bluetoothctl"),
        "pactl": _which("pactl"),
        "wpctl": _which("wpctl"),
        "playerctl": _which("playerctl"),
        "blueutil": _which("blueutil"),
        "osascript": _which("osascript"),
        "powershell": _which("powershell") or _which("pwsh"),
        "netsh": _which("netsh") if system == "windows" else None,
    }


def _scan_linux() -> Dict[str, Any]:
    if not _which("bluetoothctl"):
        return {"ok": False, "error": "bluetoothctl_not_found", "devices": []}
    devices = []
    res = _run(["bluetoothctl", "devices"], timeout=10)
    if res.get("ok"):
        for line in (res.get("stdout") or "").splitlines():
            parts = line.strip().split(" ", 2)
            if len(parts) >= 3 and parts[0] == "Device":
                devices.append({"mac": parts[1], "name": parts[2]})
    return {"ok": True, "devices": devices, "backend": "bluetoothctl"}


def _scan_macos() -> Dict[str, Any]:
    if not _which("system_profiler"):
        return {"ok": False, "error": "system_profiler_not_found", "devices": []}
    res = _run(["system_profiler", "SPBluetoothDataType"], timeout=20)
    return {"ok": res.get("ok", False), "devices": [], "raw": res.get("stdout", ""), "backend": "system_profiler"}


def _scan_windows() -> Dict[str, Any]:
    pwsh = _which("powershell") or _which("pwsh")
    if not pwsh:
        return {"ok": False, "error": "powershell_not_found", "devices": []}
    script = "Get-PnpDevice -Class Bluetooth | Select-Object Status,FriendlyName,InstanceId | ConvertTo-Json -Depth 3"
    res = _run([pwsh, "-NoProfile", "-Command", script], timeout=20)
    data = []
    if res.get("ok") and res.get("stdout"):
        try:
            parsed = json.loads(res["stdout"])
            if isinstance(parsed, dict):
                parsed = [parsed]
            for item in parsed or []:
                data.append({"name": item.get("FriendlyName", ""), "status": item.get("Status"), "instance_id": item.get("InstanceId")})
        except Exception:
            pass
    return {"ok": res.get("ok", False), "devices": data, "backend": "powershell", "raw": res.get("stdout", "")}


def _scan_devices() -> Dict[str, Any]:
    system = platform.system().lower()
    if system == "linux":
        return _scan_linux()
    if system == "darwin":
        return _scan_macos()
    if system == "windows":
        return _scan_windows()
    return {"ok": False, "error": f"unsupported_platform:{system}", "devices": []}


def _matches(dev: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    mac = (cfg.get("device_mac") or "").strip().lower()
    name = (cfg.get("device_name") or "").strip().lower()
    if mac and mac == (dev.get("mac") or "").strip().lower():
        return True
    if name and name in (dev.get("name") or "").strip().lower():
        return True
    return not mac and not name


def _connect_linux(cfg: Dict[str, Any]) -> Dict[str, Any]:
    scan = _scan_linux()
    target = next((d for d in scan.get("devices", []) if _matches(d, cfg)), None)
    if not target:
        return {"ok": False, "error": "device_not_found", "scan": scan}
    mac = target["mac"]
    out = {"pair": None, "trust": None, "connect": None}
    if cfg.get("allow_pair", True):
        out["pair"] = _run(["bluetoothctl", "pair", mac], timeout=30)
        out["trust"] = _run(["bluetoothctl", "trust", mac], timeout=10)
    out["connect"] = _run(["bluetoothctl", "connect", mac], timeout=float(cfg.get("connection_timeout_sec", 20)))
    ok = bool(out["connect"] and out["connect"].get("ok"))
    return {"ok": ok, "device": target, "steps": out, "backend": "bluetoothctl"}


def _connect_macos(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if _which("blueutil") and cfg.get("device_mac"):
        res = _run(["blueutil", "--connect", cfg["device_mac"]], timeout=float(cfg.get("connection_timeout_sec", 20)))
        return {"ok": res.get("ok", False), "device": {"mac": cfg.get("device_mac"), "name": cfg.get("device_name", "")}, "steps": {"connect": res}, "backend": "blueutil"}
    return {"ok": False, "error": "no_supported_backend_for_connect"}


def _connect_windows(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": False, "error": "windows_pair_connect_not_safely_automated_in_generic_driver"}


def _connect_device(cfg: Dict[str, Any]) -> Dict[str, Any]:
    system = platform.system().lower()
    if system == "linux":
        return _connect_linux(cfg)
    if system == "darwin":
        return _connect_macos(cfg)
    if system == "windows":
        return _connect_windows(cfg)
    return {"ok": False, "error": f"unsupported_platform:{system}"}


def _disconnect_device(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if not cfg.get("allow_disconnect", True):
        return {"ok": False, "error": "disconnect_not_allowed"}
    system = platform.system().lower()
    mac = cfg.get("device_mac") or _SESSION.get("device", {}).get("mac")
    if system == "linux" and _which("bluetoothctl") and mac:
        res = _run(["bluetoothctl", "disconnect", mac], timeout=15)
        return {"ok": res.get("ok", False), "backend": "bluetoothctl", "result": res}
    if system == "darwin" and _which("blueutil") and mac:
        res = _run(["blueutil", "--disconnect", mac], timeout=15)
        return {"ok": res.get("ok", False), "backend": "blueutil", "result": res}
    return {"ok": False, "error": "disconnect_backend_unavailable"}


def _set_volume(cfg: Dict[str, Any], value: Optional[int] = None, delta: Optional[int] = None) -> Dict[str, Any]:
    if not cfg.get("allow_volume_control", True):
        return {"ok": False, "error": "volume_control_not_allowed"}
    system = platform.system().lower()
    step = int(cfg.get("volume_step", 5))
    if value is None and delta is None:
        delta = step
    if system == "linux":
        if _which("wpctl"):
            if value is not None:
                res = _run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{max(0, min(150, value))}%"], timeout=10)
            else:
                sign = "+" if (delta or 0) >= 0 else "-"
                res = _run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{abs(int(delta))}%{sign}"], timeout=10)
            return {"ok": res.get("ok", False), "backend": "wpctl", "result": res}
        if _which("pactl"):
            if value is not None:
                res = _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{max(0, min(150, value))}%"], timeout=10)
            else:
                sign = "+" if (delta or 0) >= 0 else "-"
                res = _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{sign}{abs(int(delta))}%"], timeout=10)
            return {"ok": res.get("ok", False), "backend": "pactl", "result": res}
    if system == "darwin" and _which("osascript"):
        if value is None:
            value = max(0, min(100, 50 + int(delta or step)))
        res = _run(["osascript", "-e", f"set volume output volume {max(0, min(100, int(value)))}"], timeout=10)
        return {"ok": res.get("ok", False), "backend": "osascript", "result": res}
    if system == "windows":
        return {"ok": False, "error": "generic_windows_volume_backend_not_implemented"}
    return {"ok": False, "error": "volume_backend_unavailable"}


def _toggle_mute(cfg: Dict[str, Any], state: Optional[bool] = None) -> Dict[str, Any]:
    if not cfg.get("allow_volume_control", True):
        return {"ok": False, "error": "volume_control_not_allowed"}
    system = platform.system().lower()
    if system == "linux":
        if _which("wpctl"):
            arg = "1" if state is True else "0" if state is False else "toggle"
            res = _run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", arg], timeout=10)
            return {"ok": res.get("ok", False), "backend": "wpctl", "result": res}
        if _which("pactl"):
            arg = "1" if state is True else "0" if state is False else "toggle"
            res = _run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", arg], timeout=10)
            return {"ok": res.get("ok", False), "backend": "pactl", "result": res}
    if system == "darwin" and _which("osascript"):
        target = "true" if state is not False else "false"
        res = _run(["osascript", "-e", f"set volume output muted {target}"], timeout=10)
        return {"ok": res.get("ok", False), "backend": "osascript", "result": res}
    return {"ok": False, "error": "mute_backend_unavailable"}


def _media_control(cfg: Dict[str, Any], command: str) -> Dict[str, Any]:
    if not cfg.get("allow_media_control", True):
        return {"ok": False, "error": "media_control_not_allowed"}
    if _which("playerctl"):
        res = _run(["playerctl", command], timeout=10)
        return {"ok": res.get("ok", False), "backend": "playerctl", "result": res}
    return {"ok": False, "error": "media_backend_unavailable"}


def _listener_loop(interval: float) -> None:
    while not _STOP.wait(interval):
        with _LOCK:
            if not _SESSION.get("listener_running"):
                break
        _add_note("listener_tick")


def _start_listener(cfg: Dict[str, Any]) -> Dict[str, Any]:
    global _LISTENER
    with _LOCK:
        if _SESSION.get("listener_running"):
            return {"ok": True, "already_running": True}
        _STOP.clear()
        _SESSION["listener_running"] = True
    interval = float(cfg.get("poll_interval_sec", 2.0))
    _LISTENER = threading.Thread(target=_listener_loop, args=(interval,), daemon=True)
    _LISTENER.start()
    return {"ok": True, "listener_running": True, "interval": interval}


def _stop_listener() -> Dict[str, Any]:
    _STOP.set()
    with _LOCK:
        _SESSION["listener_running"] = False
    return {"ok": True, "listener_running": False}


def driver_init(context=None, config=None):
    config = config or {}
    v = driver_validate(config)
    if not v.get("ok"):
        with _LOCK:
            _SESSION["status"] = "error"
            _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": v}
    with _LOCK:
        _SESSION["instance_id"] = _now_id()
        _SESSION["started_ts"] = time.time()
        _SESSION["status"] = "ready"
        _SESSION["error"] = None
        _SESSION["backend"] = _probe_backends()
        _SESSION["notes"] = ["initialized"]
        _SESSION["history"] = []
        _SESSION["connected"] = False
        _SESSION["device"] = {}
        _SESSION["listener_running"] = False
    return {"ok": True, "instance_id": _SESSION["instance_id"], "info": driver_info(), "validate": v}


def driver_shutdown(context=None):
    _stop_listener()
    with _LOCK:
        _SESSION["status"] = "stopped"
        _SESSION["error"] = None
        _SESSION["connected"] = False
        _SESSION["device"] = {}
        _SESSION["instance_id"] = None
        _SESSION["started_ts"] = None
    return True


def driver_status(context=None):
    return {"ok": True, "session": dict(_SESSION), "backend_probe": _probe_backends()}


def driver_action(action_id: str, context=None, payload=None):
    payload = payload or {}
    cfg = dict((_SESSION.get("meta") or {}).get("config") or {})
    cfg.update(payload.get("config") or {})
    result: Dict[str, Any]
    if action_id == "ping":
        result = {"ok": True, "pong": True, "ts": time.time()}
    elif action_id == "probe_backends":
        result = _probe_backends()
    elif action_id in ("scan_bt", "scan_devices"):
        result = _scan_devices()
    elif action_id == "connect":
        result = _connect_device(cfg)
        if result.get("ok"):
            with _LOCK:
                _SESSION["connected"] = True
                _SESSION["device"] = result.get("device") or {}
    elif action_id == "disconnect":
        result = _disconnect_device(cfg)
        if result.get("ok"):
            with _LOCK:
                _SESSION["connected"] = False
                _SESSION["device"] = {}
    elif action_id == "set_volume":
        result = _set_volume(cfg, value=payload.get("value"), delta=payload.get("delta"))
    elif action_id == "mute":
        result = _toggle_mute(cfg, True)
    elif action_id == "unmute":
        result = _toggle_mute(cfg, False)
    elif action_id == "toggle_mute":
        result = _toggle_mute(cfg, None)
    elif action_id in ("play", "pause", "next", "previous", "stop"):
        cmd = {"play": "play-pause", "pause": "pause", "next": "next", "previous": "previous", "stop": "stop"}[action_id]
        result = _media_control(cfg, cmd)
    elif action_id == "start_listener":
        result = _start_listener(cfg)
    elif action_id == "stop_listener":
        result = _stop_listener()
    elif action_id == "get_config":
        result = {"ok": True, "config": _mask_config(cfg)}
    elif action_id == "set_config":
        result = {"ok": True, "config": _mask_config(cfg), "warning": "runtime_only_update"}
    elif action_id == "safe_stop":
        result = _stop_listener()
    else:
        result = {"ok": False, "error": f"Action '{action_id}' not implemented by {DRIVER_ID}", "action_id": action_id}
    _add_history(action_id, bool(result.get("ok")), result)
    return result
