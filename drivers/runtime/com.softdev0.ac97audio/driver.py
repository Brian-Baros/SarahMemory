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
# Driver: com.softdev0.ac97audio
# Purpose: Governed legacy audio bridge for AC'97 / HDA-adjacent playback-capture routing and mixer control.
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

DRIVER_ID = "com.softdev0.ac97audio"
DRIVER_VERSION = "2.0.0"

SESSION: Dict[str, Any] = {
    "started_at": None,
    "connected": False,
    "active_output": None,
    "active_input": None,
    "muted": False,
    "last_result": None,
    "notes": [],
    "history": [],
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

def _probe_backends() -> Dict[str, Any]:
    return {
        "platform": platform.system(),
        "backends": {
            "amixer": _which("amixer"),
            "aplay": _which("aplay"),
            "arecord": _which("arecord"),
            "pactl": _which("pactl"),
            "wpctl": _which("wpctl"),
            "pw_record": _which("pw-record"),
            "pw_play": _which("pw-play"),
            "powershell": _which("powershell") or _which("pwsh"),
            "ffplay": _which("ffplay"),
        }
    }

def _discover_linux() -> Dict[str, Any]:
    outputs = []
    inputs = []
    if _which("aplay"):
        res = _run(["aplay", "-l"], timeout=8.0)
        outputs = [ln.strip() for ln in res.get("stdout", "").splitlines() if ln.strip()]
    if _which("arecord"):
        res = _run(["arecord", "-l"], timeout=8.0)
        inputs = [ln.strip() for ln in res.get("stdout", "").splitlines() if ln.strip()]
    mixers = []
    if _which("amixer"):
        res = _run(["amixer", "scontrols"], timeout=8.0)
        mixers = [ln.strip() for ln in res.get("stdout", "").splitlines() if ln.strip()]
    return {"outputs": outputs, "inputs": inputs, "mixers": mixers}

def _discover_windows() -> Dict[str, Any]:
    ps = _which("powershell") or _which("pwsh")
    if not ps:
        return {"outputs": [], "inputs": [], "mixers": []}
    script = "Get-CimInstance Win32_SoundDevice | Select-Object Name,Manufacturer,Status,PNPDeviceID | ConvertTo-Json -Depth 3"
    res = _run([ps, "-NoProfile", "-Command", script], timeout=12.0)
    data = []
    try:
        data = json.loads(res.get("stdout", "") or "[]")
    except Exception:
        data = []
    if isinstance(data, dict):
        data = [data]
    return {"outputs": data, "inputs": data, "mixers": []}

def _discover_audio() -> Dict[str, Any]:
    sysname = platform.system()
    if sysname == "Linux":
        return _discover_linux()
    if sysname == "Windows":
        return _discover_windows()
    return {"outputs": [], "inputs": [], "mixers": []}

def _set_volume_linux(volume: int) -> Dict[str, Any]:
    volume = max(0, min(100, int(volume)))
    if _which("amixer"):
        return _run(["amixer", "set", "Master", f"{volume}%"], timeout=8.0)
    if _which("wpctl"):
        return _run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{volume/100.0:.2f}"], timeout=8.0)
    if _which("pactl"):
        return _run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{volume}%"], timeout=8.0)
    return {"ok": False, "error": "no_volume_backend"}

def _set_mute_linux(mute: bool) -> Dict[str, Any]:
    if _which("amixer"):
        return _run(["amixer", "set", "Master", "mute" if mute else "unmute"], timeout=8.0)
    if _which("wpctl"):
        return _run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "1" if mute else "0"], timeout=8.0)
    if _which("pactl"):
        return _run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "1" if mute else "0"], timeout=8.0)
    return {"ok": False, "error": "no_mute_backend"}

def driver_init(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global CONFIG
    CONFIG = dict(config or {})
    SESSION["started_at"] = _now()
    SESSION["connected"] = True
    _note("ac97 audio bridge initialized")
    return _result(True, "init", config=CONFIG, backends=_probe_backends())

def driver_status() -> Dict[str, Any]:
    return _result(True, "status", connected=SESSION["connected"], active_output=SESSION["active_output"], active_input=SESSION["active_input"], muted=SESSION["muted"], discovered=_discover_audio())

def driver_shutdown() -> Dict[str, Any]:
    SESSION["connected"] = False
    _note("ac97 audio bridge shutdown")
    return _result(True, "shutdown")

def driver_action(action: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = dict(payload or {})
    act = (action or "").strip().lower()
    if act == "ping":
        return _result(True, act, pong=True)
    if act == "probe_backends":
        return _result(True, act, **_probe_backends())
    if act in {"discover_audio", "list_devices", "detect"}:
        info = _discover_audio()
        return _result(True, act, **info)
    if act == "get_config":
        return _result(True, act, config=CONFIG)
    if act == "set_config":
        CONFIG.update(payload)
        return _result(True, act, config=CONFIG)
    if act == "set_active_output":
        SESSION["active_output"] = payload.get("output")
        return _result(True, act, active_output=SESSION["active_output"])
    if act == "set_active_input":
        SESSION["active_input"] = payload.get("input")
        return _result(True, act, active_input=SESSION["active_input"])
    if act == "set_volume":
        volume = int(payload.get("volume", CONFIG.get("default_volume", 50)))
        if platform.system() == "Linux":
            res = _set_volume_linux(volume)
            return _result(bool(res.get("ok")), act, volume=volume, response=res)
        return _result(False, act, error="set_volume_not_supported_on_this_backend")
    if act == "mute":
        if platform.system() == "Linux":
            res = _set_mute_linux(True)
            SESSION["muted"] = bool(res.get("ok", False))
            return _result(bool(res.get("ok")), act, muted=True, response=res)
        return _result(False, act, error="mute_not_supported_on_this_backend")
    if act == "unmute":
        if platform.system() == "Linux":
            res = _set_mute_linux(False)
            SESSION["muted"] = False if res.get("ok") else SESSION["muted"]
            return _result(bool(res.get("ok")), act, muted=False, response=res)
        return _result(False, act, error="unmute_not_supported_on_this_backend")
    if act == "play_test_tone":
        freq = str(payload.get("frequency_hz", 440))
        dur = str(payload.get("duration_s", 2))
        if _which("speaker-test"):
            res = _run(["speaker-test", "-t", "sine", "-f", freq, "-l", "1"], timeout=float(dur) + 4.0)
            return _result(bool(res.get("ok")), act, response=res)
        return _result(False, act, error="speaker_test_backend_not_available")
    if act == "safe_stop":
        return _result(True, act, note="no active capture stream to stop")
    return _result(False, act, error="action_not_supported")
