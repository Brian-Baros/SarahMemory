# providers/os_default/addon.py
# SarahMemory AiOS Default SoftPack - OS Default Provider
#
# Uses OS-default handlers to open URLs/files/folders.
# Cross-platform: Windows, Linux, macOS.

import os
import sys
import time
import subprocess
from pathlib import Path

_ADDON_ID = "softpack.provider.os_default"
_SESSION = {"instance_id": None, "started_ts": None, "status": "idle", "error": None}

def _platform():
    p = sys.platform.lower()
    if p.startswith("win"):
        return "windows"
    if p == "darwin":
        return "mac"
    return "linux"

def _open_target(target: str) -> None:
    plat = _platform()
    if plat == "windows":
        os.startfile(target)  # type: ignore[attr-defined]
        return
    if plat == "mac":
        subprocess.Popen(["open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    subprocess.Popen(["xdg-open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def _reveal_file(path: str) -> None:
    plat = _platform()
    p = str(Path(path).resolve())
    if plat == "windows":
        subprocess.Popen(["explorer", "/select,", p])
        return
    if plat == "mac":
        subprocess.Popen(["open", "-R", p])
        return
    subprocess.Popen(["xdg-open", str(Path(p).parent)])

def addon_info():
    return {"id": _ADDON_ID, "name": "OS Default Provider", "session": dict(_SESSION), "platform": _platform()}

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
        return {"ok": False, "error": "validation_failed", "details": v}
    _SESSION.update({"instance_id": f"SP-OS-{int(time.time())}", "started_ts": time.time(), "status":"ready", "error": None})
    return {"ok": True, "instance_id": _SESSION["instance_id"], "info": addon_info()}

def addon_shutdown(context=None):
    _SESSION.update({"instance_id": None, "started_ts": None, "status":"stopped", "error": None})
    return True

def addon_status(context=None):
    return {"ok": True, "session": dict(_SESSION)}

def addon_action(action_id: str, context=None, payload=None):
    payload = payload or {}
    if action_id == "web.open_url":
        url = payload.get("url") or payload.get("target")
        if not url or not isinstance(url, str):
            return {"ok": False, "error": "missing url"}
        _open_target(url)
        return {"ok": True}
    if action_id in ("file.open", "file.open_folder"):
        target = payload.get("path") or payload.get("target")
        if not target or not isinstance(target, str):
            return {"ok": False, "error": "missing path"}
        _open_target(target)
        return {"ok": True}
    if action_id == "file.reveal":
        target = payload.get("path") or payload.get("target")
        if not target or not isinstance(target, str):
            return {"ok": False, "error": "missing path"}
        _reveal_file(target)
        return {"ok": True}
    if action_id == "mail.open":
        to = payload.get("to","")
        subject = payload.get("subject","")
        body = payload.get("body","")
        mailto = f"mailto:{to}?subject={subject}&body={body}"
        _open_target(mailto)
        return {"ok": True}
    if action_id == "ping":
        return {"ok": True, "pong": True, "ts": time.time()}
    return {"ok": False, "error": f"Action '{action_id}' not implemented"}
