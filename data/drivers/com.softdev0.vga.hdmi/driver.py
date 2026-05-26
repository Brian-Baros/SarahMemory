"""
--==The SarahMemory Project==--
Driver: com.softdev0.vga.hdmi
Purpose: SarahMemory-native governed display bridge for HDMI / DisplayPort / VGA / DVI surfaces.

The driver witnesses display topology and builds operator HUD surface packets.
It does not take ownership of the GPU and does not bypass the host display stack.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

DRIVER_ID = "com.softdev0.vga.hdmi"
DRIVER_VERSION = "3.0.0"

_DEFAULTS: Dict[str, Any] = {'enabled': True, 'autoload': False, 'preferred_mode': '1920x1080', 'history_limit': 200, 'scene_request_dir': './data/xr_scene_requests', 'vr_surface': {'enabled': True, 'target_role': 'operator_vr_surface', 'selection_mode': 'auto_or_manual', 'preferred_transport': ['hdmi', 'displayport'], 'x': 1920, 'y': 0, 'width': 1920, 'height': 1080, 'fullscreen': True, 'mirror_preview': True, 'headset_surface': True, 'window_title': 'SM_A_HUD_DIRECT'}}
CONFIG: Dict[str, Any] = deepcopy(_DEFAULTS)
SESSION: Dict[str, Any] = {"started_at": None, "connected": False, "active_display": None, "last_result": None, "notes": [], "history": []}


def _now() -> float:
    return time.time()


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


def _push(action: str, ok: bool, detail: Any = None) -> None:
    SESSION.setdefault("history", []).append({"ts": _now(), "action": action, "ok": bool(ok), "detail": detail})
    SESSION["history"] = SESSION["history"][-int(CONFIG.get("history_limit", 200) or 200):]


def _result(ok: bool, action: str, **kwargs: Any) -> Dict[str, Any]:
    payload = {"ok": bool(ok), "action": action, "driver_id": DRIVER_ID, **kwargs}
    SESSION["last_result"] = payload
    _push(action, ok, kwargs)
    return payload


def _guess_type(name: str) -> str:
    n = (name or "").lower()
    if "hdmi" in n: return "hdmi"
    if "dp" in n or "displayport" in n: return "displayport"
    if "dvi" in n: return "dvi"
    if "vga" in n: return "vga"
    if "edp" in n or "lvds" in n: return "internal_panel"
    return "display"


def _probe_backends() -> Dict[str, Any]:
    return {"platform": platform.system(), "backends": {"xrandr": _which("xrandr"), "powershell": _which("powershell") or _which("pwsh"), "system_profiler": _which("system_profiler")}}


def _parse_xrandr() -> List[Dict[str, Any]]:
    if not _which("xrandr"):
        return []
    res = _run(["xrandr", "--query"], timeout=8.0)
    displays: List[Dict[str, Any]] = []
    if not res.get("ok"):
        return displays
    current = None
    for line in res.get("stdout", "").splitlines():
        if " connected" in line or " disconnected" in line:
            parts = line.split()
            current = {"name": parts[0], "status": parts[1], "modes": [], "type": _guess_type(parts[0]), "raw": line}
            m = re.search(r"(\d+)x(\d+)\+(\-?\d+)\+(\-?\d+)", line)
            if m:
                current["current_mode"] = f"{m.group(1)}x{m.group(2)}"
                current["width"] = int(m.group(1)); current["height"] = int(m.group(2))
                current["position"] = {"x": int(m.group(3)), "y": int(m.group(4))}
            displays.append(current)
        elif current and re.match(r"\s+\d+x\d+", line):
            current["modes"].append(line.strip().split()[0])
    return displays


def _parse_windows_displays() -> List[Dict[str, Any]]:
    ps = _which("powershell") or _which("pwsh")
    if not ps:
        return []
    script = "Get-CimInstance Win32_DesktopMonitor | Select-Object Name,MonitorManufacturer,ScreenHeight,ScreenWidth,PNPDeviceID | ConvertTo-Json -Depth 4"
    res = _run([ps, "-NoProfile", "-Command", script], timeout=12.0)
    if not res.get("ok"):
        return []
    data = _safe_json_load(res.get("stdout", ""))
    if isinstance(data, dict): data = [data]
    displays = []
    for idx, item in enumerate(data or []):
        if not isinstance(item, dict):
            continue
        w = item.get("ScreenWidth"); h = item.get("ScreenHeight")
        displays.append({
            "name": item.get("Name") or f"Display {idx+1}",
            "manufacturer": item.get("MonitorManufacturer"),
            "current_mode": f"{w}x{h}" if w and h else None,
            "width": w, "height": h,
            "device_id": item.get("PNPDeviceID"),
            "status": "connected",
            "type": "display",
            "index": idx,
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
            displays.append({"name": disp.get("_name"), "current_mode": disp.get("_spdisplays_resolution"), "status": "connected", "type": "display", "main": disp.get("spdisplays_main"), "online": disp.get("spdisplays_online")})
    return displays


def _discover_displays() -> List[Dict[str, Any]]:
    sysname = platform.system()
    if sysname == "Linux": return _parse_xrandr()
    if sysname == "Windows": return _parse_windows_displays()
    if sysname == "Darwin": return _parse_macos_displays()
    return []


def _score_display(display: Dict[str, Any]) -> int:
    score = 0
    t = str(display.get("type") or "").lower()
    raw = (str(display.get("name") or "") + " " + str(display.get("device_id") or "") + " " + str(display.get("raw") or "")).lower()
    if t in {"hdmi", "displayport"}: score += 40
    if any(k in raw for k in ("hdmi", "displayport", "psvr", "vr", "hmd", "sony")): score += 30
    if display.get("status") == "connected": score += 20
    if display.get("main") in (True, "Yes", "spdisplays_yes"): score -= 30
    return score


def _surface_packet(payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    displays = _discover_displays()
    surface_cfg = deepcopy(CONFIG.get("vr_surface") or {})
    ranked = sorted(displays, key=_score_display, reverse=True)
    chosen = None
    manual_name = str(payload.get("display") or surface_cfg.get("display") or "").strip()
    if manual_name:
        chosen = next((d for d in displays if str(d.get("name")) == manual_name or str(d.get("device_id")) == manual_name), None)
    if chosen is None and ranked:
        chosen = ranked[0]
    width = int(payload.get("width") or surface_cfg.get("width") or (chosen or {}).get("width") or 1920)
    height = int(payload.get("height") or surface_cfg.get("height") or (chosen or {}).get("height") or 1080)
    x = int(payload.get("x") if payload.get("x") is not None else surface_cfg.get("x", 1920))
    y = int(payload.get("y") if payload.get("y") is not None else surface_cfg.get("y", 0))
    surface = {
        "schema": "SarahMemory.display.operator_hud_surface.v1",
        "target_role": "operator_vr_surface",
        "selected_display": chosen or {},
        "candidate_displays": ranked,
        "bounds": {"x": x, "y": y, "width": width, "height": height},
        "fullscreen": bool(payload.get("fullscreen", surface_cfg.get("fullscreen", True))),
        "mirror_preview": bool(payload.get("mirror_preview", surface_cfg.get("mirror_preview", True))),
        "headset_surface": bool(payload.get("headset_surface", surface_cfg.get("headset_surface", True))),
        "window_title": str(surface_cfg.get("window_title") or "SM_A_HUD_DIRECT"),
        "movement_lock": True,
        "activation": "renderer_process_only",
    }
    return surface


def driver_init(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global CONFIG
    CONFIG = deepcopy(_DEFAULTS)
    if isinstance(config, dict):
        for k, v in config.items():
            if isinstance(v, dict) and isinstance(CONFIG.get(k), dict):
                merged = dict(CONFIG[k]); merged.update(v); CONFIG[k] = merged
            else:
                CONFIG[k] = v
    SESSION["started_at"] = _now(); SESSION["connected"] = True
    return _result(True, "init", config=CONFIG, backends=_probe_backends())


def driver_status() -> Dict[str, Any]:
    return _result(True, "status", connected=SESSION["connected"], active_display=SESSION.get("active_display"), discovered=_discover_displays(), surface=_surface_packet({}))


def driver_shutdown() -> Dict[str, Any]:
    SESSION["connected"] = False
    return _result(True, "shutdown")


def driver_action(action: str, context: Optional[Dict[str, Any]] = None, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Accept both appdrivers-style driver_action(action, payload) and MSDC-style driver_action(action, context, payload).
    if payload is None and isinstance(context, dict) and not any(k in context for k in ("caller", "body_part", "read_only_probe")):
        payload = context
    payload = dict(payload or {})
    act = (action or "").strip().lower()
    if act == "ping": return _result(True, act, pong=True)
    if act in {"probe", "probe_backends"}: return _result(True, act, **_probe_backends())
    if act in {"discover_displays", "list_displays", "detect"}:
        displays = _discover_displays(); return _result(True, act, displays=displays, count=len(displays))
    if act in {"operator_hud_surface", "build_hud_surface_request"}:
        surface = _surface_packet(payload)
        return _result(True, act, surface=surface, hud_surface=surface, displays=surface.get("candidate_displays", []))
    if act == "get_config": return _result(True, act, config=CONFIG)
    if act == "set_config":
        for k, v in payload.items():
            if isinstance(v, dict) and isinstance(CONFIG.get(k), dict):
                merged = dict(CONFIG[k]); merged.update(v); CONFIG[k] = merged
            else:
                CONFIG[k] = v
        return _result(True, act, config=CONFIG)
    if act == "set_active_display":
        SESSION["active_display"] = payload.get("display")
        return _result(True, act, active_display=SESSION["active_display"])
    if act == "safe_stop": return _result(True, act, note="display bridge has no owned stream to stop")
    return _result(False, act, error="action_not_supported")
