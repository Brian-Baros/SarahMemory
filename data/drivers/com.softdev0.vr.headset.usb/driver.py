# SarahMemory Driver Package: com.softdev0.vr.headset.usb
# Name: VR Headset XR Bridge Driver
#
# Governed VR/XR hardware/runtime bridge for SarahMemory AiOS.
# This package is intentionally a runtime bridge, not a full rendering engine.
# It detects XR-capable runtimes and devices, negotiates launch/configuration,
# routes scene/avatar requests into the wider AiOS pipeline, and degrades cleanly
# when optional OS or XR backends are unavailable.

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DRIVER_ID = "com.softdev0.vr.headset.usb"
DRIVER_NAME = "VR Headset XR Bridge Driver"
VERSION = "2.0.0"

_DEFAULTS = {
    "enabled": True,
    "autoload": False,
    "runtime_preference": ["openxr", "steamvr", "monado", "openvr"],
    "allow_pose_tracking": True,
    "allow_eye_tracking": False,
    "allow_hand_tracking": True,
    "allow_passthrough": False,
    "allow_microphone": True,
    "allow_secondary_display": True,
    "allow_scene_requests": True,
    "allow_runtime_launch": True,
    "allow_resolution_changes": True,
    "require_confirm_for_scene_launch": True,
    "require_confirm_for_runtime_launch": False,
    "allow_vendor_tools": False,
    "allow_bluetooth_glasses": True,
    "display_mode": "auto",
    "mirror_mode": "auto",
    "render_scale": 1.0,
    "target_width": 0,
    "target_height": 0,
    "target_refresh_hz": 0,
    "target_runtime": "OpenXR",
    "target_headset_name": "",
    "target_headset_serial": "",
    "preferred_audio_output": "",
    "preferred_audio_input": "",
    "preferred_controller_mode": "auto",
    "preferred_scene_transport": "file",
    "scene_request_dir": "",
    "runtime_command": "",
    "runtime_stop_command": "",
    "session_name": "SarahMemory XR Session",
    "poll_interval_ms": 1000,
    "history_limit": 200,
}

_SESSION: Dict[str, Any] = {
    "instance_id": None,
    "started_ts": None,
    "status": "idle",
    "error": None,
    "meta": {},
    "config": deepcopy(_DEFAULTS),
    "history": [],
    "last_detect": {},
    "runtime_active": False,
    "runtime_process": None,
    "current_scene_request": None,
}


def _now_id() -> str:
    return "DRV-%s-%sZ-%s" % (
        DRIVER_ID,
        datetime.utcnow().strftime("%Y%m%dT%H%M%S"),
        hex(int(time.time() * 1000))[-4:].upper(),
    )


def _ts() -> float:
    return time.time()


def _safe_jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _push_history(kind: str, detail: Any) -> None:
    item = {"ts": _ts(), "kind": kind, "detail": _safe_jsonable(detail)}
    _SESSION["history"].append(item)
    limit = int(_SESSION.get("config", {}).get("history_limit", 200) or 200)
    if len(_SESSION["history"]) > limit:
        _SESSION["history"] = _SESSION["history"][-limit:]


def _merge_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = deepcopy(_DEFAULTS)
    if isinstance(config, dict):
        merged.update(config)
    if not merged.get("scene_request_dir"):
        merged["scene_request_dir"] = os.path.join(tempfile.gettempdir(), "SarahMemoryXR")
    return merged


def _run(cmd: List[str], timeout: int = 10) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "cmd": cmd,
        }
    except FileNotFoundError:
        return {"ok": False, "error": "tool_not_found", "cmd": cmd}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "cmd": cmd}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "cmd": cmd}


def _tool_exists(name: str) -> bool:
    return shutil.which(name) is not None


def _platform() -> str:
    return platform.system().lower()


def _probe_backends() -> Dict[str, Any]:
    sysname = _platform()
    tools = {
        "steam": _tool_exists("steam"),
        "xrmonitors": _tool_exists("xrmonitors"),
        "xrandr": _tool_exists("xrandr"),
        "bluetoothctl": _tool_exists("bluetoothctl"),
        "pactl": _tool_exists("pactl"),
        "pw-cli": _tool_exists("pw-cli"),
        "wpctl": _tool_exists("wpctl"),
        "wmic": _tool_exists("wmic"),
        "powershell": _tool_exists("powershell") or _tool_exists("pwsh"),
        "system_profiler": _tool_exists("system_profiler"),
        "ioreg": _tool_exists("ioreg"),
        "adb": _tool_exists("adb"),
    }
    env = {
        "XR_RUNTIME_JSON": os.environ.get("XR_RUNTIME_JSON", ""),
        "OPENXR_RUNTIME_JSON": os.environ.get("OPENXR_RUNTIME_JSON", ""),
        "STEAMVR_ACTIVE": bool(os.environ.get("VR_OVERRIDE") or os.environ.get("STEAMVR_TOOLSDIR")),
    }
    runtimes = []
    if env["XR_RUNTIME_JSON"] or env["OPENXR_RUNTIME_JSON"]:
        runtimes.append({"name": "OpenXR", "active": True, "source": "env"})
    if tools["steam"]:
        runtimes.append({"name": "SteamVR", "available": True})
    if tools["xrmonitors"]:
        runtimes.append({"name": "Monado", "available": True})
    if sysname == "windows":
        runtimes.append({"name": "Windows Mixed Reality", "available": True})
    return {"ok": True, "platform": sysname, "tools": tools, "env": env, "runtimes": runtimes}


def _socket_ping(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except Exception:
        return False


def _linux_detect() -> Dict[str, Any]:
    headsets: List[Dict[str, Any]] = []
    displays: List[Dict[str, Any]] = []
    audio: Dict[str, List[Dict[str, Any]]] = {"inputs": [], "outputs": []}
    bt: List[Dict[str, Any]] = []

    xrj = os.environ.get("XR_RUNTIME_JSON") or os.environ.get("OPENXR_RUNTIME_JSON")
    if xrj:
        headsets.append({"runtime_hint": "OpenXR", "runtime_json": xrj})

    if _tool_exists("xrandr"):
        res = _run(["xrandr", "--query"], timeout=8)
        if res.get("ok"):
            for line in res.get("stdout", "").splitlines():
                if " connected" in line:
                    name = line.split()[0]
                    displays.append({"name": name, "raw": line})
                    lower = line.lower()
                    if any(k in lower for k in ["vr", "vive", "index", "rift", "quest", "hmd", "xreal"]):
                        headsets.append({"display": name, "raw": line, "source": "xrandr"})

    if _tool_exists("bluetoothctl"):
        res = _run(["bluetoothctl", "devices"], timeout=8)
        if res.get("ok"):
            for line in res.get("stdout", "").splitlines():
                parts = line.split(maxsplit=2)
                if len(parts) == 3:
                    item = {"mac": parts[1], "name": parts[2]}
                    bt.append(item)
                    if any(k in parts[2].lower() for k in ["xreal", "nreal", "glasses", "quest", "vive", "vr", "meta"]):
                        headsets.append({"bluetooth": item, "source": "bluetoothctl"})

    if _tool_exists("pactl"):
        res = _run(["pactl", "list", "short", "sources"], timeout=8)
        if res.get("ok"):
            for line in res.get("stdout", "").splitlines():
                cols = line.split("\t")
                if len(cols) >= 2:
                    audio["inputs"].append({"id": cols[0], "name": cols[1], "raw": line})
        res = _run(["pactl", "list", "short", "sinks"], timeout=8)
        if res.get("ok"):
            for line in res.get("stdout", "").splitlines():
                cols = line.split("\t")
                if len(cols) >= 2:
                    audio["outputs"].append({"id": cols[0], "name": cols[1], "raw": line})

    return {"headsets": headsets, "displays": displays, "audio": audio, "bluetooth": bt}


def _windows_detect() -> Dict[str, Any]:
    headsets: List[Dict[str, Any]] = []
    displays: List[Dict[str, Any]] = []
    audio: Dict[str, List[Dict[str, Any]]] = {"inputs": [], "outputs": []}
    bt: List[Dict[str, Any]] = []

    ps = shutil.which("powershell") or shutil.which("pwsh")
    if ps:
        script = "Get-PnpDevice | Select-Object -ExpandProperty FriendlyName"
        res = _run([ps, "-NoProfile", "-Command", script], timeout=15)
        if res.get("ok"):
            for line in res.get("stdout", "").splitlines():
                name = line.strip()
                if not name:
                    continue
                if any(k in name.lower() for k in ["vr", "vive", "rift", "quest", "index", "mixed reality", "xreal", "hololens"]):
                    headsets.append({"name": name, "source": "Get-PnpDevice"})
                if any(k in name.lower() for k in ["monitor", "display"]):
                    displays.append({"name": name})
        # audio endpoints
        script = "Get-CimInstance Win32_SoundDevice | Select-Object -ExpandProperty Name"
        res = _run([ps, "-NoProfile", "-Command", script], timeout=15)
        if res.get("ok"):
            for line in res.get("stdout", "").splitlines():
                name = line.strip()
                if name:
                    audio["outputs"].append({"name": name})

    if _tool_exists("wmic"):
        res = _run(["wmic", "desktopmonitor", "get", "name"], timeout=10)
        if res.get("ok"):
            for line in res.get("stdout", "").splitlines()[1:]:
                name = line.strip()
                if name:
                    displays.append({"name": name, "source": "wmic"})
    return {"headsets": headsets, "displays": displays, "audio": audio, "bluetooth": bt}


def _mac_detect() -> Dict[str, Any]:
    headsets: List[Dict[str, Any]] = []
    displays: List[Dict[str, Any]] = []
    audio: Dict[str, List[Dict[str, Any]]] = {"inputs": [], "outputs": []}
    bt: List[Dict[str, Any]] = []

    if _tool_exists("system_profiler"):
        res = _run(["system_profiler", "SPDisplaysDataType"], timeout=20)
        if res.get("ok"):
            for line in res.get("stdout", "").splitlines():
                name = line.strip()
                if any(k in name.lower() for k in ["vr", "xreal", "display", "monitor"]):
                    displays.append({"name": name})
        res = _run(["system_profiler", "SPBluetoothDataType"], timeout=20)
        if res.get("ok"):
            for line in res.get("stdout", "").splitlines():
                name = line.strip()
                if any(k in name.lower() for k in ["quest", "vive", "vr", "xreal", "glasses"]):
                    bt.append({"name": name})
                    headsets.append({"name": name, "source": "SPBluetoothDataType"})
    return {"headsets": headsets, "displays": displays, "audio": audio, "bluetooth": bt}


def _detect_headsets() -> Dict[str, Any]:
    sysname = _platform()
    backend = _probe_backends()
    if sysname == "linux":
        data = _linux_detect()
    elif sysname == "windows":
        data = _windows_detect()
    elif sysname == "darwin":
        data = _mac_detect()
    else:
        data = {"headsets": [], "displays": [], "audio": {"inputs": [], "outputs": []}, "bluetooth": []}

    # Deduplicate by JSON string
    def dedupe(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out = []
        for item in items:
            key = json.dumps(item, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out

    data["headsets"] = dedupe(data.get("headsets", []))
    data["displays"] = dedupe(data.get("displays", []))
    data["bluetooth"] = dedupe(data.get("bluetooth", []))
    data["backend_probe"] = backend
    data["ok"] = True
    data["ts"] = _ts()
    _SESSION["last_detect"] = deepcopy(data)
    _push_history("detect_headsets", {"headsets": len(data["headsets"]), "displays": len(data["displays"])})
    return data


def _ensure_confirmation(cfg: Dict[str, Any], payload: Dict[str, Any], cfg_key: str, error: str) -> Optional[Dict[str, Any]]:
    if cfg.get(cfg_key) and not bool(payload.get("confirm") or payload.get("confirmed")):
        return {"ok": False, "error": error, "requires_confirmation": True}
    return None


def _make_scene_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _SESSION["config"]
    if not cfg.get("allow_scene_requests", True):
        return {"ok": False, "error": "scene_requests_disabled"}
    denial = _ensure_confirmation(cfg, payload, "require_confirm_for_scene_launch", "scene_launch_confirmation_required")
    if denial:
        return denial

    scene_request = {
        "request_id": f"XRSCENE-{int(_ts()*1000)}",
        "ts": _ts(),
        "session_name": cfg.get("session_name"),
        "title": payload.get("title") or payload.get("prompt") or "SarahMemory XR Scene",
        "prompt": payload.get("prompt", ""),
        "style": payload.get("style", ""),
        "mode": payload.get("mode", "immersive"),
        "environment_type": payload.get("environment_type", "generated"),
        "avatar_enabled": bool(payload.get("avatar_enabled", True)),
        "secondary_display": bool(payload.get("secondary_display", cfg.get("allow_secondary_display", True))),
        "input_preferences": payload.get("input_preferences", ["vr_controllers", "voice", "keyboard_mouse", "gamepad"]),
        "safety_profile": payload.get("safety_profile", "standard"),
        "resolution_hint": {
            "width": payload.get("width", cfg.get("target_width", 0)),
            "height": payload.get("height", cfg.get("target_height", 0)),
            "refresh_hz": payload.get("refresh_hz", cfg.get("target_refresh_hz", 0)),
            "render_scale": payload.get("render_scale", cfg.get("render_scale", 1.0)),
        },
        "runtime_target": cfg.get("target_runtime"),
        "headset_target": cfg.get("target_headset_name"),
    }
    req_dir = Path(cfg.get("scene_request_dir") or tempfile.gettempdir())
    req_dir.mkdir(parents=True, exist_ok=True)
    path = req_dir / f"{scene_request['request_id']}.json"
    path.write_text(json.dumps(scene_request, indent=2), encoding="utf-8")
    _SESSION["current_scene_request"] = str(path)
    _push_history("scene_request", {"path": str(path), "title": scene_request["title"]})
    return {"ok": True, "scene_request": scene_request, "path": str(path)}


def _start_runtime(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _SESSION["config"]
    if not cfg.get("allow_runtime_launch", True):
        return {"ok": False, "error": "runtime_launch_disabled"}
    denial = _ensure_confirmation(cfg, payload, "require_confirm_for_runtime_launch", "runtime_launch_confirmation_required")
    if denial:
        return denial

    cmd = payload.get("command") or cfg.get("runtime_command")
    if not cmd:
        target = str(payload.get("runtime") or cfg.get("target_runtime") or "OpenXR").lower()
        if target == "steamvr":
            if _platform() == "windows":
                cmd = 'cmd /c start steam://rungameid/250820'
            else:
                cmd = 'steam steam://rungameid/250820'
        elif target == "monado":
            cmd = 'monado-service'
        else:
            return {"ok": False, "error": "runtime_command_not_configured", "target_runtime": target}

    if isinstance(cmd, str):
        popen_kwargs = {"shell": True}
        args = cmd
    else:
        popen_kwargs = {"shell": False}
        args = cmd

    try:
        proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **popen_kwargs)
        _SESSION["runtime_process"] = getattr(proc, "pid", None)
        _SESSION["runtime_active"] = True
        _SESSION["status"] = "runtime_active"
        _push_history("start_runtime", {"command": cmd, "pid": _SESSION["runtime_process"]})
        return {"ok": True, "pid": _SESSION["runtime_process"], "command": cmd}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "command": cmd}


def _stop_runtime(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _SESSION["config"]
    cmd = payload.get("command") or cfg.get("runtime_stop_command")
    ran = None
    if cmd:
        if isinstance(cmd, str):
            ran = _run(["bash", "-lc", cmd], timeout=15) if _platform() != "windows" else _run(["cmd", "/c", cmd], timeout=15)
        elif isinstance(cmd, list):
            ran = _run(cmd, timeout=15)
    _SESSION["runtime_active"] = False
    _SESSION["runtime_process"] = None
    if _SESSION["status"] != "stopped":
        _SESSION["status"] = "ready"
    _push_history("stop_runtime", {"result": ran})
    return {"ok": True, "stop_result": ran}


def _set_resolution(payload: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _SESSION["config"]
    if not cfg.get("allow_resolution_changes", True):
        return {"ok": False, "error": "resolution_change_disabled"}
    width = int(payload.get("width", cfg.get("target_width", 0)) or 0)
    height = int(payload.get("height", cfg.get("target_height", 0)) or 0)
    hz = int(payload.get("refresh_hz", cfg.get("target_refresh_hz", 0)) or 0)
    display_name = str(payload.get("display_name") or payload.get("display") or "")
    if width <= 0 or height <= 0:
        return {"ok": False, "error": "invalid_resolution"}

    applied = {"requested": {"width": width, "height": height, "refresh_hz": hz, "display_name": display_name}, "applied": False}
    sysname = _platform()
    if sysname == "linux" and _tool_exists("xrandr") and display_name:
        cmd = ["xrandr", "--output", display_name, "--mode", f"{width}x{height}"]
        if hz > 0:
            cmd.extend(["--rate", str(hz)])
        applied["command_result"] = _run(cmd, timeout=10)
        applied["applied"] = bool(applied["command_result"].get("ok"))
    else:
        # config-level commit only; actual runtime/display application delegated to runtime/OS layer
        applied["applied"] = False
        applied["note"] = "resolution stored in config for runtime/display layer to consume"

    cfg["target_width"] = width
    cfg["target_height"] = height
    cfg["target_refresh_hz"] = hz
    _push_history("set_resolution", applied)
    return {"ok": True, **applied}


def _list_inputs() -> Dict[str, Any]:
    detect = _SESSION.get("last_detect") or _detect_headsets()
    controllers = []
    names = [
        "vr_controllers",
        "ps_move",
        "gamepad",
        "keyboard_mouse",
        "voice",
        "bluetooth_glasses",
    ]
    for name in names:
        controllers.append({"id": name, "available": True, "mode": "generic"})
    return {"ok": True, "inputs": controllers, "audio": detect.get("audio", {}), "bluetooth": detect.get("bluetooth", [])}


def driver_info() -> Dict[str, Any]:
    return {
        "id": DRIVER_ID,
        "name": DRIVER_NAME,
        "version": VERSION,
        "session": {
            "instance_id": _SESSION.get("instance_id"),
            "started_ts": _SESSION.get("started_ts"),
            "status": _SESSION.get("status"),
            "error": _SESSION.get("error"),
            "runtime_active": _SESSION.get("runtime_active"),
            "runtime_process": _SESSION.get("runtime_process"),
            "current_scene_request": _SESSION.get("current_scene_request"),
        },
    }


def driver_validate(config: Dict[str, Any]) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be an object"], "warnings": []}
    if config.get("render_scale") is not None:
        try:
            rs = float(config.get("render_scale"))
            if rs <= 0:
                errors.append("render_scale must be > 0")
        except Exception:
            errors.append("render_scale must be numeric")
    if config.get("target_width") and int(config.get("target_width", 0)) < 0:
        errors.append("target_width must be >= 0")
    if config.get("target_height") and int(config.get("target_height", 0)) < 0:
        errors.append("target_height must be >= 0")
    if config.get("allow_vendor_tools"):
        warnings.append("vendor tools enabled; behavior becomes vendor/runtime dependent")
    if config.get("allow_scene_requests") and not config.get("scene_request_dir"):
        warnings.append("scene_request_dir not set; temporary directory will be used")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def driver_init(context: Optional[Dict[str, Any]] = None, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    merged = _merge_config(config)
    v = driver_validate(merged)
    if not v.get("ok"):
        _SESSION["status"] = "error"
        _SESSION["error"] = "validation_failed"
        return {"ok": False, "error": "validation_failed", "details": v}

    _SESSION["instance_id"] = _now_id()
    _SESSION["started_ts"] = _ts()
    _SESSION["status"] = "ready"
    _SESSION["error"] = None
    _SESSION["meta"] = {"context": context or {}}
    _SESSION["config"] = merged
    _SESSION["history"] = []
    _SESSION["last_detect"] = {}
    _SESSION["runtime_active"] = False
    _SESSION["runtime_process"] = None
    _SESSION["current_scene_request"] = None
    _push_history("init", {"config": merged})
    return {"ok": True, "instance_id": _SESSION["instance_id"], "info": driver_info(), "validate": v}


def driver_shutdown(context: Optional[Dict[str, Any]] = None) -> bool:
    try:
        _stop_runtime({})
    except Exception:
        pass
    _SESSION["status"] = "stopped"
    _SESSION["error"] = None
    _SESSION["meta"] = {}
    _SESSION["instance_id"] = None
    _SESSION["started_ts"] = None
    _SESSION["runtime_active"] = False
    _SESSION["runtime_process"] = None
    _SESSION["current_scene_request"] = None
    return True


def driver_status(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    uptime = None
    if _SESSION.get("started_ts"):
        uptime = _ts() - float(_SESSION["started_ts"])
    return {
        "ok": True,
        "session": {
            "instance_id": _SESSION.get("instance_id"),
            "started_ts": _SESSION.get("started_ts"),
            "status": _SESSION.get("status"),
            "error": _SESSION.get("error"),
            "runtime_active": _SESSION.get("runtime_active"),
            "runtime_process": _SESSION.get("runtime_process"),
            "current_scene_request": _SESSION.get("current_scene_request"),
            "uptime_s": uptime,
        },
        "last_detect": deepcopy(_SESSION.get("last_detect") or {}),
        "config": deepcopy(_SESSION.get("config") or {}),
        "history_tail": deepcopy((_SESSION.get("history") or [])[-20:]),
    }


def driver_action(action_id: str, context: Optional[Dict[str, Any]] = None, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload or {}
    try:
        if action_id == "ping":
            return {"ok": True, "driver": DRIVER_ID, "ts": _ts()}
        if action_id == "probe_backends":
            return _probe_backends()
        if action_id in {"detect", "detect_headsets", "discover"}:
            return _detect_headsets()
        if action_id == "list_inputs":
            return _list_inputs()
        if action_id == "get_config":
            return {"ok": True, "config": deepcopy(_SESSION.get("config") or {})}
        if action_id == "set_config":
            cfg = deepcopy(_SESSION.get("config") or {})
            patch = payload.get("config") if isinstance(payload.get("config"), dict) else payload
            cfg.update(patch)
            v = driver_validate(cfg)
            if not v.get("ok"):
                return {"ok": False, "error": "validation_failed", "details": v}
            _SESSION["config"] = cfg
            _push_history("set_config", patch)
            return {"ok": True, "config": deepcopy(cfg), "validate": v}
        if action_id == "start_runtime":
            return _start_runtime(payload)
        if action_id == "stop_runtime":
            return _stop_runtime(payload)
        if action_id == "set_resolution":
            return _set_resolution(payload)
        if action_id == "launch_scene_request":
            return _make_scene_request(payload)
        if action_id == "start_avatar_mode":
            _push_history("start_avatar_mode", payload)
            return {"ok": True, "avatar_mode": True, "note": "Avatar mode requested; rendering handled by runtime/application layer"}
        if action_id == "stop_avatar_mode":
            _push_history("stop_avatar_mode", payload)
            return {"ok": True, "avatar_mode": False}
        if action_id == "describe_capabilities":
            return {
                "ok": True,
                "capabilities": {
                    "runtime_bridge": True,
                    "openxr_targeting": True,
                    "runtime_launch": True,
                    "headset_detection": True,
                    "display_mode_negotiation": True,
                    "scene_request_export": True,
                    "avatar_mode_signaling": True,
                    "secondary_display_support": True,
                    "voice_input_preference": True,
                    "controller_routing": ["vr_controllers", "gamepad", "keyboard_mouse", "voice", "ps_move"],
                    "vendor_specific_controls": False,
                    "universal_world_generation_inside_driver": False,
                },
            }
        if action_id in {"safe_stop", "stop"}:
            _stop_runtime({})
            _SESSION["status"] = "ready"
            return {"ok": True, "stopped": True}
    except Exception as exc:
        _SESSION["error"] = str(exc)
        _push_history("exception", {"action_id": action_id, "error": str(exc)})
        return {"ok": False, "error": str(exc), "action_id": action_id}

    return {
        "ok": False,
        "error": f"Action '{action_id}' not implemented by {DRIVER_ID}",
        "action_id": action_id,
    }
