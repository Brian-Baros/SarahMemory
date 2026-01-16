# --==The SarahMemory Project==--
# File: /api/server/appsys.py
# ULTIMATE merged Flask server for SarahMemory (v8.0.0)
# Part of the SarahMemory Companion AI-bot Platform
# Author: © 2025 Brian Lee Baros. All Rights Reserved.
# www.linkedin.com/in/brian-baros-29962a176
# https://www.facebook.com/bbaros
# brian.baros@sarahmemory.com
# 'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
# https://www.sarahmemory.com
# https://api.sarahmemory.com
# https://ai.sarahmemory.com
# Purpose: System endpoints for local-only features (Files / OS utilities)
# Notes:
#  - MUST NOT expose PythonAnywhere server filesystem on ai.sarahmemory.com
#  - Local browsing is enabled ONLY for localhost requests by default
#  - app.py mounts this via appsys.init_app(app)

from __future__ import annotations

import os
import time
import json
import uuid
import shutil
import platform
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from flask import Blueprint, jsonify, request, send_file

bp = Blueprint("appsys_v800", __name__)

# In-memory download tokens (simple + effective for local mode)
# token -> (abs_path, expires_ts)
_DOWNLOAD_TOKENS: Dict[str, Tuple[str, float]] = {}

# ---------------------------------------------------------------------
# Local-only gate
# ---------------------------------------------------------------------

def _is_local_request() -> bool:
    """
    Returns True if request appears to originate from localhost.
    NOTE: In production behind proxies, remote_addr may be proxy IP.
    That is intentional: we do NOT want to expose server filesystem.
    """
    try:
        ra = (request.remote_addr or "").strip()
        if ra in ("127.0.0.1", "::1"):
            return True
    except Exception:
        pass
    return False

def _files_enabled() -> bool:
    """
    Enable browsing when:
      - Request is localhost, OR
      - Env SARAHMEMORY_ALLOW_SERVER_FILES=1 (explicit override)
    """
    if _is_local_request():
        return True
    return os.environ.get("SARAHMEMORY_ALLOW_SERVER_FILES", "0").strip().lower() in ("1", "true", "yes", "on")

def _ok(**payload):
    out = {"ok": True}
    out.update(payload)
    return jsonify(out), 200

def _err(msg: str, code: int = 400, **payload):
    out = {"ok": False, "error": msg}
    out.update(payload)
    return jsonify(out), code

def _norm_path(p: str) -> str:
    """
    Normalize path for the host OS.
    Accepts:
      - Windows: "C:\\", "C:/", etc.
      - Unix: "/home/user"
    """
    p = (p or "").strip()
    if not p:
        return ""
    # Allow UI to pass "/" as "root"; on Windows we interpret as "This PC"
    return p

def _safe_stat(path: str) -> Dict[str, Any]:
    try:
        st = os.stat(path)
        return {
            "size": int(st.st_size),
            "mtime": float(st.st_mtime),
        }
    except Exception:
        return {"size": 0, "mtime": 0}

def _list_dir(path: str) -> Tuple[bool, Any]:
    try:
        p = Path(path)
        if not p.exists():
            return False, f"Path not found: {path}"
        if not p.is_dir():
            return False, f"Not a directory: {path}"

        items = []
        for child in p.iterdir():
            name = child.name
            full = str(child)
            is_dir = child.is_dir()
            st = _safe_stat(full)
            items.append({
                "name": name,
                "path": full,
                "type": "folder" if is_dir else "file",
                "size": 0 if is_dir else int(st.get("size", 0) or 0),
                "modified": float(st.get("mtime", 0) or 0),
            })

        # folders first, then name
        items.sort(key=lambda x: (0 if x["type"] == "folder" else 1, (x["name"] or "").lower()))
        return True, items
    except Exception as e:
        return False, str(e)

def _get_drives() -> list:
    system = platform.system().lower()
    drives = []

    if system.startswith("win"):
        # Windows drive letters
        import string
        from ctypes import windll

        bitmask = windll.kernel32.GetLogicalDrives()
        for i, letter in enumerate(string.ascii_uppercase):
            if bitmask & (1 << i):
                root = f"{letter}:\\"
                drives.append({
                    "name": f"Local Disk ({letter}:)",
                    "path": root,
                    "kind": "drive",
                })
        return drives

    # Linux/mac: show root + mounted volumes if present
    drives.append({"name": "Root (/)", "path": "/", "kind": "drive"})
    for base in ("/mnt", "/media", "/Volumes"):
        try:
            b = Path(base)
            if b.exists() and b.is_dir():
                for child in b.iterdir():
                    if child.is_dir():
                        drives.append({"name": child.name, "path": str(child), "kind": "mount"})
        except Exception:
            pass
    return drives

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@bp.get("/api/files/capabilities")
def files_capabilities():
    enabled = _files_enabled()
    os_name = platform.system().lower()
    provider = "local_agent" if enabled else "browser_sandbox"

    caps = {
        "provider": provider,
        "os": os_name,
        "canBrowse": bool(enabled),
        "canListDrives": bool(enabled),
        "canMkdir": bool(enabled),
        "canRename": bool(enabled),
        "canDelete": bool(enabled),
        "canMove": bool(enabled),
        "canCopy": bool(enabled),
        "canDownload": bool(enabled),
        # destructive/system-level are disabled by default here
        "canTrash": False,
        "canUnmount": False,
        "canFormat": False,
    }

    # Helpful message for UI
    note = ""
    if not enabled:
        note = "File browsing disabled (cloud-safe mode). Run locally or enable SARAHMEMORY_ALLOW_SERVER_FILES=1 explicitly."

    return _ok(capabilities=caps, note=note)

@bp.get("/api/files/drives")
def files_drives():
    if not _files_enabled():
        return _err("Drive listing not available (cloud-safe mode).", 403)
    return _ok(drives=_get_drives())

@bp.post("/api/files/list")
def files_list():
    if not _files_enabled():
        return _err("File browsing not available (cloud-safe mode).", 403)

    data = request.get_json(silent=True) or {}
    path = _norm_path(data.get("path") or "")

    # Special case: UI might pass "/" as a starting point.
    # On Windows, treat "/" as "This PC" and return drives as folders.
    if platform.system().lower().startswith("win") and path in ("", "/"):
        drives = _get_drives()
        items = [{
            "name": d["name"],
            "path": d["path"],
            "type": "folder",
            "size": 0,
            "modified": 0,
        } for d in drives]
        return _ok(path="This PC", items=items)

    ok, items_or_err = _list_dir(path)
    if not ok:
        return _err(str(items_or_err), 404)
    return _ok(path=path, items=items_or_err)

@bp.post("/api/files/mkdir")
def files_mkdir():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)

    data = request.get_json(silent=True) or {}
    parent = _norm_path(data.get("path") or "")
    name = (data.get("name") or "").strip()
    if not parent or not name:
        return _err("Missing path or name")

    try:
        target = str(Path(parent) / name)
        os.makedirs(target, exist_ok=False)
        return _ok(created=True, path=target)
    except FileExistsError:
        return _err("Folder already exists", 409)
    except Exception as e:
        return _err("Failed to create folder", 500, detail=str(e))

@bp.post("/api/files/rename")
def files_rename():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)

    data = request.get_json(silent=True) or {}
    src = _norm_path(data.get("path") or "")
    new_name = (data.get("new_name") or "").strip()
    if not src or not new_name:
        return _err("Missing path or new_name")

    try:
        sp = Path(src)
        if not sp.exists():
            return _err("Source not found", 404)
        dst = str(sp.parent / new_name)
        os.rename(str(sp), dst)
        return _ok(renamed=True, path=dst)
    except Exception as e:
        return _err("Rename failed", 500, detail=str(e))

@bp.post("/api/files/delete")
def files_delete():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)

    data = request.get_json(silent=True) or {}
    src = _norm_path(data.get("path") or "")
    mode = (data.get("mode") or "permanent").strip().lower()
    if not src:
        return _err("Missing path")

    # NOTE: Trash/Recycle not implemented here yet (capability says canTrash=False)
    if mode != "permanent":
        return _err("Trash mode not supported yet", 400)

    try:
        sp = Path(src)
        if not sp.exists():
            return _err("Not found", 404)
        if sp.is_dir():
            shutil.rmtree(str(sp))
        else:
            os.remove(str(sp))
        return _ok(deleted=True)
    except Exception as e:
        return _err("Delete failed", 500, detail=str(e))

@bp.post("/api/files/move")
def files_move():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)

    data = request.get_json(silent=True) or {}
    src = _norm_path(data.get("src") or "")
    dst = _norm_path(data.get("dst") or "")
    if not src or not dst:
        return _err("Missing src or dst")

    try:
        shutil.move(src, dst)
        return _ok(moved=True)
    except Exception as e:
        return _err("Move failed", 500, detail=str(e))

@bp.post("/api/files/copy")
def files_copy():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)

    data = request.get_json(silent=True) or {}
    src = _norm_path(data.get("src") or "")
    dst = _norm_path(data.get("dst") or "")
    if not src or not dst:
        return _err("Missing src or dst")

    try:
        sp = Path(src)
        if not sp.exists():
            return _err("Source not found", 404)
        if sp.is_dir():
            shutil.copytree(src, dst)
        else:
            # ensure parent exists
            os.makedirs(str(Path(dst).parent), exist_ok=True)
            shutil.copy2(src, dst)
        return _ok(copied=True)
    except FileExistsError:
        return _err("Destination already exists", 409)
    except Exception as e:
        return _err("Copy failed", 500, detail=str(e))

@bp.post("/api/files/download")
def files_download():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)

    data = request.get_json(silent=True) or {}
    src = _norm_path(data.get("path") or "")
    if not src:
        return _err("Missing path")

    try:
        sp = Path(src)
        if not sp.exists() or not sp.is_file():
            return _err("File not found", 404)

        token = uuid.uuid4().hex
        expires = time.time() + 60.0  # 60 seconds
        _DOWNLOAD_TOKENS[token] = (str(sp), expires)

        return _ok(url=f"/api/files/raw/{token}", expires_in=60)
    except Exception as e:
        return _err("Download prep failed", 500, detail=str(e))

@bp.get("/api/files/raw/<token>")
def files_raw(token: str):
    try:
        rec = _DOWNLOAD_TOKENS.get(token)
        if not rec:
            return _err("Invalid token", 404)
        path, expires = rec
        if time.time() > expires:
            _DOWNLOAD_TOKENS.pop(token, None)
            return _err("Token expired", 410)
        p = Path(path)
        if not p.exists() or not p.is_file():
            return _err("File not found", 404)
        return send_file(str(p), as_attachment=True)
    except Exception as e:
        return _err("Download failed", 500, detail=str(e))

# ---------------------------------------------------------------------
# init_app (called by app.py ONCE)
# ---------------------------------------------------------------------

def init_app(app) -> None:
    # Prevent double-register
    if "appsys_v800" in getattr(app, "blueprints", {}):
        return
    app.register_blueprint(bp)
