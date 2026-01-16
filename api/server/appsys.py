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
from typing import Any, Dict, Optional, Tuple, List

from flask import Blueprint, jsonify, request, send_file

bp = Blueprint("appsys_v800", __name__)

# In-memory download tokens (simple + effective for local mode)
# token -> (abs_path, expires_ts)
_DOWNLOAD_TOKENS: Dict[str, Tuple[str, float]] = {}

# ---------------------------------------------------------------------
# SarahMemoryGlobals BASE_DIR (authoritative)
# ---------------------------------------------------------------------

try:
    import SarahMemoryGlobals as SMG  # type: ignore
except Exception:
    SMG = None  # fallback

def _get_base_dir() -> Path:
    """
    Returns BASE_DIR from SarahMemoryGlobals.py (authoritative),
    fallback to CWD if missing/unavailable.
    """
    try:
        if SMG is not None:
            bd = getattr(SMG, "BASE_DIR", None)
            if bd:
                return Path(str(bd)).expanduser()
    except Exception:
        pass
    return Path.cwd()

def _downloads_dir() -> Path:
    return _get_base_dir() / "downloads"

def _dumpster_dir() -> Path:
    return _get_base_dir() / "dumpster"

def _dumpster_items_dir() -> Path:
    # Keep dumpster contents organized but still inside BASE_DIR/dumpster
    return _dumpster_dir() / "files"

def _dumpster_index_path() -> Path:
    # metadata index lives inside dumpster
    return _dumpster_dir() / "index.json"

def _ensure_core_dirs() -> None:
    """
    Ensure BASE_DIR/downloads and BASE_DIR/dumpster exist.
    Only called when file features are enabled (local / explicit override).
    """
    try:
        _downloads_dir().mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        _dumpster_dir().mkdir(parents=True, exist_ok=True)
        _dumpster_items_dir().mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    # Ensure index exists
    try:
        ip = _dumpster_index_path()
        if not ip.exists():
            ip.write_text("{}", encoding="utf-8")
    except Exception:
        pass

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

def _sanitize_filename(name: str) -> str:
    name = (name or "").replace("\\", "/").split("/")[-1].strip()
    name = name.replace("\x00", "")
    name = name.replace("..", "_")
    if not name:
        return "file.bin"
    # Keep it simple, allow most characters, just bound length
    return name[:255]

def _unique_path_in_dir(dir_path: Path, filename: str) -> Path:
    """
    Create a unique path in dir_path. If filename exists, append " (n)".
    """
    base = _sanitize_filename(filename)
    candidate = dir_path / base
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    for i in range(1, 10000):
        alt = dir_path / f"{stem} ({i}){suffix}"
        if not alt.exists():
            return alt
    # fallback hard unique
    return dir_path / f"{stem}__{uuid.uuid4().hex}{suffix}"

# ---------------------------------------------------------------------
# Dumpster index helpers
# ---------------------------------------------------------------------

def _load_dumpster_index() -> Dict[str, Any]:
    """
    index.json is a dict of:
      id -> { id, orig_path, name, kind, trashed_ts, stored_path }
    """
    try:
        ip = _dumpster_index_path()
        if not ip.exists():
            return {}
        raw = ip.read_text(encoding="utf-8") or "{}"
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}

def _save_dumpster_index(data: Dict[str, Any]) -> None:
    """
    Atomic write best-effort: write temp then replace.
    """
    try:
        ip = _dumpster_index_path()
        tmp = ip.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(ip)
    except Exception:
        # best-effort; do not crash API
        pass

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
        # Trash + upload supported (still gated by _files_enabled)
        "canTrash": bool(enabled),
        "canUpload": bool(enabled),
        "canUnmount": False,
        "canFormat": False,
    }

    # Helpful message for UI
    note = ""
    if not enabled:
        note = "File browsing disabled (cloud-safe mode). Run locally or enable SARAHMEMORY_ALLOW_SERVER_FILES=1 explicitly."
    else:
        # Ensure directories exist once file features are enabled
        _ensure_core_dirs()
        note = f"Local files enabled. Downloads: {_downloads_dir()} • Dumpster: {_dumpster_dir()}"

    return _ok(capabilities=caps, note=note)

@bp.get("/api/files/drives")
def files_drives():
    if not _files_enabled():
        return _err("Drive listing not available (cloud-safe mode).", 403)
    _ensure_core_dirs()
    return _ok(drives=_get_drives())

@bp.post("/api/files/list")
def files_list():
    if not _files_enabled():
        return _err("File browsing not available (cloud-safe mode).", 403)

    _ensure_core_dirs()
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

    _ensure_core_dirs()
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

    _ensure_core_dirs()
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

@bp.post("/api/files/upload")
def files_upload():
    """
    Multipart upload endpoint.
    Saves incoming files into BASE_DIR/downloads.
    Accepts:
      - <input name="file"> single
      - <input name="files" multiple>
    Returns:
      { ok:true, saved:[{name,path,size,modified}] }
    """
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)

    _ensure_core_dirs()

    # Collect files from common keys
    incoming = []
    try:
        if "files" in request.files:
            incoming.extend(request.files.getlist("files"))
        if "file" in request.files:
            incoming.append(request.files.get("file"))
    except Exception:
        incoming = []

    # Filter None
    incoming = [f for f in incoming if f is not None]

    if not incoming:
        return _err("No files uploaded (expected multipart form-data with field 'file' or 'files')", 400)

    saved: List[Dict[str, Any]] = []
    dl_dir = _downloads_dir()

    # Optional upload cap (per-file). Default 200MB in local mode.
    max_bytes = int(os.environ.get("SARAHMEMORY_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))

    for f in incoming:
        try:
            orig_name = _sanitize_filename(getattr(f, "filename", "") or "file.bin")
            dst_path = _unique_path_in_dir(dl_dir, orig_name)

            # Stream save with size check
            total = 0
            with open(dst_path, "wb") as out:
                while True:
                    chunk = f.stream.read(1024 * 1024)  # 1MB
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        try:
                            out.close()
                        except Exception:
                            pass
                        try:
                            dst_path.unlink(missing_ok=True)  # py3.8+: missing_ok not always; handle below
                        except Exception:
                            try:
                                if dst_path.exists():
                                    dst_path.unlink()
                            except Exception:
                                pass
                        return _err(f"Upload too large (max {max_bytes} bytes per file)", 413, filename=orig_name)
                    out.write(chunk)

            st = _safe_stat(str(dst_path))
            saved.append({
                "name": dst_path.name,
                "path": str(dst_path),
                "type": "file",
                "size": int(st.get("size", 0) or 0),
                "modified": float(st.get("mtime", 0) or 0),
            })
        except Exception as e:
            return _err("Upload failed", 500, detail=str(e))

    return _ok(saved=saved, count=len(saved), downloads_dir=str(dl_dir))

@bp.post("/api/files/delete")
def files_delete():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)

    _ensure_core_dirs()
    data = request.get_json(silent=True) or {}
    src = _norm_path(data.get("path") or "")
    mode = (data.get("mode") or "permanent").strip().lower()
    if not src:
        return _err("Missing path")

    try:
        sp = Path(src)
        if not sp.exists():
            return _err("Not found", 404)

        if mode == "trash":
            # Move into BASE_DIR/dumpster
            did = uuid.uuid4().hex
            items_dir = _dumpster_items_dir()
            items_dir.mkdir(parents=True, exist_ok=True)

            # Store under a stable unique name
            safe_name = _sanitize_filename(sp.name)
            stored = items_dir / f"{did}__{safe_name}"

            # If already exists (very unlikely), make unique
            if stored.exists():
                stored = items_dir / f"{did}__{safe_name}__{uuid.uuid4().hex}"

            # Move
            shutil.move(str(sp), str(stored))

            # Index record
            idx = _load_dumpster_index()
            idx[did] = {
                "id": did,
                "orig_path": str(sp),
                "name": safe_name,
                "kind": "folder" if stored.is_dir() else "file",
                "trashed_ts": float(time.time()),
                "stored_path": str(stored),
            }
            _save_dumpster_index(idx)

            return _ok(trashed=True, id=did, stored_path=str(stored), dumpster_dir=str(_dumpster_dir()))

        # Permanent delete
        if sp.is_dir():
            shutil.rmtree(str(sp))
        else:
            os.remove(str(sp))
        return _ok(deleted=True)

    except Exception as e:
        return _err("Delete failed", 500, detail=str(e))

@bp.get("/api/files/trash/list")
def files_trash_list():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)

    _ensure_core_dirs()
    idx = _load_dumpster_index()

    out = []
    for k, v in (idx or {}).items():
        try:
            stored_path = str((v or {}).get("stored_path") or "")
            stored_exists = bool(stored_path and Path(stored_path).exists())
            out.append({
                "id": (v or {}).get("id") or k,
                "name": (v or {}).get("name") or "",
                "orig_path": (v or {}).get("orig_path") or "",
                "kind": (v or {}).get("kind") or "",
                "trashed_ts": (v or {}).get("trashed_ts") or 0,
                "stored_path": stored_path,
                "stored_exists": stored_exists,
            })
        except Exception:
            pass

    # newest first
    out.sort(key=lambda x: float(x.get("trashed_ts") or 0), reverse=True)
    return _ok(items=out, count=len(out), dumpster_dir=str(_dumpster_dir()))

@bp.post("/api/files/trash/restore")
def files_trash_restore():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)

    _ensure_core_dirs()
    data = request.get_json(silent=True) or {}
    did = (data.get("id") or "").strip()
    restore_to = (data.get("restore_to") or "").strip()  # optional override path

    if not did:
        return _err("Missing id", 400)

    idx = _load_dumpster_index()
    rec = (idx or {}).get(did)
    if not rec:
        return _err("Trash item not found", 404)

    stored_path = str(rec.get("stored_path") or "")
    orig_path = str(rec.get("orig_path") or "")
    if not stored_path:
        return _err("Corrupt trash record (missing stored_path)", 500)

    sp = Path(stored_path)
    if not sp.exists():
        return _err("Stored trash item missing on disk", 404)

    # Determine destination
    dst = Path(_norm_path(restore_to)) if restore_to else Path(_norm_path(orig_path))
    if not str(dst):
        # fallback: restore into downloads
        dst = _downloads_dir() / (_sanitize_filename(rec.get("name") or sp.name))

    # Ensure parent exists
    try:
        dst_parent = dst.parent
        dst_parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    # If destination exists, create a unique variant
    if dst.exists():
        dst = _unique_path_in_dir(dst.parent, dst.name)

    try:
        shutil.move(str(sp), str(dst))
    except Exception as e:
        return _err("Restore failed", 500, detail=str(e))

    # Remove from index
    try:
        idx.pop(did, None)
        _save_dumpster_index(idx)
    except Exception:
        pass

    return _ok(restored=True, id=did, path=str(dst))

@bp.post("/api/files/trash/empty")
def files_trash_empty():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)

    _ensure_core_dirs()

    # Remove all stored files/dirs in dumpster/files
    items_dir = _dumpster_items_dir()
    removed = 0
    try:
        if items_dir.exists() and items_dir.is_dir():
            for child in items_dir.iterdir():
                try:
                    if child.is_dir():
                        shutil.rmtree(str(child))
                    else:
                        child.unlink()
                    removed += 1
                except Exception:
                    pass
    except Exception:
        pass

    # Clear index
    try:
        _save_dumpster_index({})
    except Exception:
        pass

    return _ok(emptied=True, removed=removed)

@bp.post("/api/files/move")
def files_move():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)

    _ensure_core_dirs()
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

    _ensure_core_dirs()
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

    _ensure_core_dirs()
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