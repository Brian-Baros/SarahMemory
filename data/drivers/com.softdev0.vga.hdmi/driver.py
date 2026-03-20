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
# --== The SarahMemory Project ==--
# Driver: com.softdev0.vga.hdmi
# Purpose: Governed display bridge for VGA / DVI / HDMI / DisplayPort style monitors and secondary display routing.
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

DRIVER_ID = "com.softdev0.vga.hdmi"
DRIVER_VERSION = "2.0.0"

SESSION: Dict[str, Any] = {
    "started_at": None,
    "connected": False,
    "active_display": None,
    "last_result": None,
    "notes": [],
    "history": [],
    "listener_running": False,
}
CONFIG: Dict[str, Any] = {}

def _now() -> float:
    return time.time()

def _note(msg: str) -> None:
    SESSION["notes"].append({"ts": _now(), "msg": msg})
    SESSION["notes"] = SESSION["notes"][-100:]

def _push_history(action: str, ok: bool, detail: Any = None) -> None:
    SESSION["history"].append({"ts": _now(), "action": action, "ok": bool(ok), "detail": detail})
    SESSION["history"] = SESSION["history"][-int(CONFIG.get("history_limit", 200)):]

def _result(ok: bool, action: str, **kwargs: Any) -> Dict[str, Any]:
    payload = {"ok": bool(ok), "action": action, "driver_id": DRIVER_ID, **kwargs}
    SESSION["last_result"] = payload
    _push_history(action, ok, kwargs)
    return payload

def _which(name: str) -> Optional[str]:
    return shutil.which(name)

def _run(cmd: List[str], timeout: float = 10.0) -> Dict[str, Any]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"ok": proc.returncode == 0, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "cmd": cmd}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "cmd": cmd}

def _safe_json_load(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None

def _guess_type(name: str) -> str:
    n = (name or "").lower()
    if "hdmi" in n:
        return "hdmi"
    if "vga" in n:
        return "vga"
    if "dvi" in n:
        return "dvi"
    if "dp" in n or "displayport" in n:
        return "displayport"
    if "edp" in n or "lvds" in n:
        return "internal_panel"
    return "display"

def _probe_backends() -> Dict[str, Any]:
    return {
        "platform": platform.system(),
        "backends": {
            "xrandr": _which("xrandr"),
            "wlr_randr": _which("wlr-randr"),
            "kscreen_doctor": _which("kscreen-doctor"),
            "powershell": _which("powershell") or _which("pwsh"),
            "system_profiler": _which("system_profiler"),
            "xset": _which("xset"),
        },
    }

def _parse_xrandr() -> List[Dict[str, Any]]:
    res = _run(["xrandr", "--query"], timeout=8.0)
    displays: List[Dict[str, Any]] = []
    if not res.get("ok"):
        return displays
    current = None
    for line in res.get("stdout", "").splitlines():
        if " connected" in line or " disconnected" in line:
            parts = line.split()
            current = {"name": parts[0], "status": parts[1], "modes": [], "type": _guess_type(parts[0])}
            m = re.search(r"(\d+x\d+)\+(\d+)\+(\d+)", line)
            if m:
                current["current_mode"] = m.group(1)
                current["position"] = {"x": int(m.group(2)), "y": int(m.group(3))}
            displays.append(current)
        elif current and re.match(r"\s+\d+x\d+", line):
            current["modes"].append(line.strip().split()[0])
    return displays

def _parse_windows_displays() -> List[Dict[str, Any]]:
    ps = _which("powershell") or _which("pwsh")
    if not ps:
        return []
    script = "Get-CimInstance Win32_DesktopMonitor | Select-Object Name,MonitorManufacturer,ScreenHeight,ScreenWidth,PNPDeviceID | ConvertTo-Json -Depth 3"
    res = _run([ps, "-NoProfile", "-Command", script], timeout=12.0)
    if not res.get("ok"):
        return []
    data = _safe_json_load(res.get("stdout", ""))
    if isinstance(data, dict):
        data = [data]
    displays = []
    for item in data or []:
        displays.append({
            "name": item.get("Name"),
            "manufacturer": item.get("MonitorManufacturer"),
            "current_mode": f"{item.get('ScreenWidth')}x{item.get('ScreenHeight')}" if item.get("ScreenWidth") and item.get("ScreenHeight") else None,
            "device_id": item.get("PNPDeviceID"),
            "status": "connected",
            "type": "display",
        })
    return displays

def _parse_macos_displays() -> List[Dict[str, Any]]:
    if not _which("system_profiler"):
        return []
    res = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=20.0)
    if not res.get("ok"):
        return []
    data = _safe_json_load(res.get("stdout", "")) or {}
    displays = []
    for gpu in data.get("SPDisplaysDataType", []):
        for disp in gpu.get("spdisplays_ndrvs", []):
            displays.append({
                "name": disp.get("_name"),
                "current_mode": disp.get("_spdisplays_resolution"),
                "status": "connected",
                "type": "display",
                "main": disp.get("spdisplays_main"),
                "online": disp.get("spdisplays_online"),
            })
    return displays

def _discover_displays() -> List[Dict[str, Any]]:
    sysname = platform.system()
    if sysname == "Linux":
        return _parse_xrandr()
    if sysname == "Windows":
        return _parse_windows_displays()
    if sysname == "Darwin":
        return _parse_macos_displays()
    return []

def _set_linux_mode(display: str, mode: str, x: Optional[int], y: Optional[int], rotate: Optional[str], primary: bool) -> Dict[str, Any]:
    if not _which("xrandr"):
        return {"ok": False, "error": "xrandr_not_available"}
    cmd = ["xrandr", "--output", display, "--mode", mode]
    if x is not None and y is not None:
        cmd += ["--pos", f"{x}x{y}"]
    if rotate:
        cmd += ["--rotate", rotate]
    if primary:
        cmd += ["--primary"]
    return _run(cmd, timeout=10.0)

def _set_brightness_linux(display: str, brightness: float) -> Dict[str, Any]:
    if not _which("xrandr"):
        return {"ok": False, "error": "xrandr_not_available"}
    brightness = max(0.1, min(2.0, float(brightness)))
    return _run(["xrandr", "--output", display, "--brightness", f"{brightness:.2f}"], timeout=8.0)

def _build_scene_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    scene_dir = CONFIG.get("scene_request_dir") or os.path.join(os.getcwd(), "data", "xr_scene_requests")
    os.makedirs(scene_dir, exist_ok=True)
    req = {
        "ts": _now(),
        "type": "display_scene_request",
        "driver_id": DRIVER_ID,
        "display_target": payload.get("display_target"),
        "intent": payload.get("intent", "secondary_display_experience"),
        "prompt": payload.get("prompt", ""),
        "avatar_enabled": bool(payload.get("avatar_enabled", True)),
        "secondary_display": bool(payload.get("secondary_display", True)),
        "fullscreen": bool(payload.get("fullscreen", True)),
        "resolution_hint": payload.get("resolution_hint"),
        "render_profile": payload.get("render_profile", "balanced"),
    }
    fname = os.path.join(scene_dir, f"display_scene_{int(_now()*1000)}.json")
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(req, f, indent=2)
    return {"ok": True, "path": fname, "request": req}

def driver_init(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global CONFIG
    CONFIG = dict(config or {})
    SESSION["started_at"] = _now()
    SESSION["connected"] = True
    _note("display bridge initialized")
    return _result(True, "init", config=CONFIG, backends=_probe_backends())

def driver_status() -> Dict[str, Any]:
    return _result(True, "status", connected=SESSION["connected"], active_display=SESSION["active_display"], discovered=_discover_displays(), notes=SESSION["notes"][-20:])

def driver_shutdown() -> Dict[str, Any]:
    SESSION["connected"] = False
    _note("display bridge shutdown")
    return _result(True, "shutdown")

def driver_action(action: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(payload or {})
    act = (action or "").strip().lower()
    if act == "ping":
        return _result(True, act, pong=True)
    if act == "probe_backends":
        return _result(True, act, **_probe_backends())
    if act in {"discover_displays", "list_displays", "detect"}:
        displays = _discover_displays()
        return _result(True, act, displays=displays, count=len(displays))
    if act == "get_config":
        return _result(True, act, config=CONFIG)
    if act == "set_config":
        CONFIG.update(payload)
        return _result(True, act, config=CONFIG)
    if act == "set_active_display":
        SESSION["active_display"] = payload.get("display")
        return _result(True, act, active_display=SESSION["active_display"])
    if act == "set_mode":
        display = payload.get("display") or SESSION.get("active_display")
        if not display:
            return _result(False, act, error="missing_display")
        if platform.system() == "Linux":
            res = _set_linux_mode(display, str(payload.get("mode") or CONFIG.get("preferred_mode") or "1920x1080"), payload.get("x"), payload.get("y"), payload.get("rotate"), bool(payload.get("primary", False)))
            return _result(bool(res.get("ok")), act, display=display, backend="xrandr", response=res)
        return _result(False, act, error="set_mode_not_supported_on_this_backend")
    if act == "set_brightness":
        display = payload.get("display") or SESSION.get("active_display")
        if not display:
            return _result(False, act, error="missing_display")
        if platform.system() == "Linux":
            res = _set_brightness_linux(display, float(payload.get("brightness", 1.0)))
            return _result(bool(res.get("ok")), act, display=display, backend="xrandr", response=res)
        return _result(False, act, error="brightness_control_not_supported_on_this_backend")
    if act == "mirror_on":
        return _result(True, act, note="mirror request accepted", supported=platform.system() == "Linux")
    if act == "extend_on":
        return _result(True, act, note="extend request accepted", supported=platform.system() == "Linux")
    if act == "build_scene_request":
        res = _build_scene_request(payload)
        return _result(bool(res.get("ok")), act, **res)
    if act == "safe_stop":
        return _result(True, act, note="no active display stream to stop")
    return _result(False, act, error="action_not_supported")
