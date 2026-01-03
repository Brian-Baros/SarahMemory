# scanners/software_scanner_light/addon.py
# SarahMemory AiOS Default SoftPack - Light Software Scanner

import os
import sys
import time
import json
import subprocess
from pathlib import Path

_ADDON_ID = "softpack.scanner.software_light"
_SESSION = {"instance_id": None, "started_ts": None, "status": "idle", "error": None}

def _platform():
    p = sys.platform.lower()
    if p.startswith("win"):
        return "windows"
    if p == "darwin":
        return "mac"
    return "linux"

def _run(cmd):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout.strip()
    except Exception:
        pass
    return ""

def _detect_windows(exe_names):
    found = {}
    for name in exe_names:
        out = _run(["where", name])
        if out:
            found[name] = out.splitlines()[0].strip()
    return found

def _detect_posix(bin_names):
    found = {}
    for name in bin_names:
        out = _run(["which", name])
        if out:
            found[name] = out.splitlines()[0].strip()
    return found

def _data_dir(context):
    try:
        if context and isinstance(context, dict) and context.get("data_dir"):
            return Path(context["data_dir"]).expanduser().resolve()
    except Exception:
        pass
    return Path(os.getcwd()).resolve() / "data"

def _write_cache(context, obj):
    dd = _data_dir(context)
    regdir = dd / "registry"
    regdir.mkdir(parents=True, exist_ok=True)
    path = regdir / "software_index.json"
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    return str(path)

def addon_info():
    return {"id": _ADDON_ID, "name": "Light Software Scanner", "session": dict(_SESSION), "platform": _platform()}

def addon_validate(config: dict):
    if config is None:
        config = {}
    if not isinstance(config, dict):
        return {"ok": False, "errors": ["config must be an object"], "warnings": []}
    return {"ok": True, "errors": [], "warnings": []}

def addon_init(context=None, config=None):
    v = addon_validate(config or {})
    if not v.get("ok"):
        _SESSION.update({"status":"error","error":"validation_failed"})
        return {"ok": False, "error":"validation_failed", "details": v}
    _SESSION.update({"instance_id": f"SP-SCAN-{int(time.time())}", "started_ts": time.time(), "status":"ready", "error": None})
    return {"ok": True, "instance_id": _SESSION["instance_id"], "info": addon_info()}

def addon_shutdown(context=None):
    _SESSION.update({"instance_id": None, "started_ts": None, "status":"stopped", "error": None})
    return True

def addon_action(action_id: str, context=None, payload=None):
    if action_id not in ("software.scan","scan"):
        return {"ok": False, "error": f"Action '{action_id}' not implemented"}

    plat = _platform()
    if plat == "windows":
        browsers = _detect_windows(["chrome.exe","msedge.exe","firefox.exe","opera.exe","brave.exe"])
        media = _detect_windows(["vlc.exe","wmplayer.exe"])
        office = _detect_windows(["WINWORD.EXE","EXCEL.EXE","POWERPNT.EXE","OUTLOOK.EXE","MSACCESS.EXE","soffice.exe"])
    else:
        browsers = _detect_posix(["google-chrome","chromium","chromium-browser","firefox","opera","brave-browser"])
        media = _detect_posix(["vlc","mpv","totem"])
        office = _detect_posix(["soffice","libreoffice"])

    index = {"ok": True, "ts": time.time(), "platform": plat, "browsers": browsers, "media": media, "office": office}
    cache_path = _write_cache(context, index)
    return {"ok": True, "index": index, "cache_path": cache_path}
