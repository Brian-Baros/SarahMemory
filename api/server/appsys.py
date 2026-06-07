"""--==The SarahMemory Project==--
File: api/server/appsys.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-06-06
Time: 10:11:54
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, SarahMemory AiOS, and all Parts of the SarahMemory Project are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
https://store.sarahmemory.com

System endpoints module for SarahMemory (v9.0.0)
==============================================================================================
Purpose: System endpoints for local-only features (Files / OS utilities)
Notes:
- MUST NOT expose PythonAnywhere server filesystem on ai.sarahmemory.com
- Local browsing is enabled ONLY for localhost requests by default
- app.py mounts this via appsys.init_app(app)

v9.0.0 hardening:
- Path traversal protection (canonicalize + enforce under BASE_DIR)
- /api/files/upload (multipart + base64 JSON fallback) -> BASE_DIR/downloads
- Trash workflow (BASE_DIR/dumpster/items + index.json)
- Append-only activity log (DATA_DIR/logs/api_events.log)
- Browser proxy fetch endpoints (Reader Mode) + native open hooks
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "B"
# ROLE = "api_bridge"
# CATEGORY = "system_local_operations"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "api_server"
# API_DOMAIN = "system"
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "system_api"
# FAMILY = "system_ops"
# GOVERNANCE_LEVEL = "restricted"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-06-06"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "System/local-operations API bridge for files, local OS utilities, guarded ingestion, browser reader-mode helpers, and cloud-safe local-only controls."
# --- SARAHMETA END ---

import base64
import hashlib
import json
import logging
import os
import platform
import shutil
import time
import uuid
import smtplib
import imaplib
from email import message_from_bytes
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import ipaddress
import re as _re
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
import bleach

from flask import Blueprint, jsonify, request, send_file

bp = Blueprint("appsys_v800", __name__)
logger = logging.getLogger(__name__)

# In-memory download tokens (simple + effective for local mode)
# token -> (abs_path, expires_ts)
_DOWNLOAD_TOKENS: Dict[str, Tuple[str, float]] = {}

# ---------------------------------------------------------------------
# Paths (prefer SarahMemoryGlobals)
# ---------------------------------------------------------------------

def _get_base_dir() -> Path:
    """Return SarahMemoryGlobals.BASE_DIR if available; otherwise fallback to cwd."""
    try:
        import SarahMemoryGlobals as config  # type: ignore
        base = getattr(config, "BASE_DIR", None)
        if base:
            return Path(str(base)).expanduser().resolve()
    except Exception:
        pass
    return Path(os.getcwd()).expanduser().resolve()


def _get_data_dir() -> Path:
    """Return SarahMemoryGlobals.DATA_DIR if available; else BASE_DIR/data."""
    try:
        import SarahMemoryGlobals as config  # type: ignore
        dd = getattr(config, "DATA_DIR", None)
        if dd:
            return Path(str(dd)).expanduser().resolve()
    except Exception:
        pass
    return (_get_base_dir() / "data").resolve()


def _downloads_dir() -> Path:
    return (_get_base_dir() / "downloads").resolve()


def _dumpster_dir() -> Path:
    return (_get_base_dir() / "dumpster").resolve()


def _dumpster_items_dir() -> Path:
    return (_dumpster_dir() / "items").resolve()


def _dumpster_index_path() -> Path:
    return (_dumpster_dir() / "index.json").resolve()


def _ensure_core_dirs() -> None:
    try:
        _downloads_dir().mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        _dumpster_items_dir().mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    try:
        (_get_data_dir() / "logs").mkdir(parents=True, exist_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------
# Local-only gate
# ---------------------------------------------------------------------

def _is_local_request() -> bool:
    """True if request appears to originate from localhost."""
    try:
        ra = (request.remote_addr or "").strip()
        if ra in ("127.0.0.1", "::1"):
            return True
    except Exception:
        pass
    return False


def _files_enabled() -> bool:
    """Enable file ops when localhost OR explicit override."""
    if _is_local_request():
        return True
    return os.environ.get("SARAHMEMORY_ALLOW_SERVER_FILES", "0").strip().lower() in ("1", "true", "yes", "on")


def _local_ingest_enabled(*, require_local_only_mode: bool = False) -> bool:
    """Enable local ingestion endpoints under governance.

    Governance contract:
      - MUST be a localhost request
      - MUST have SarahMemoryGlobals.NEOSKYMATRIX=True (master kill switch)

    Optional stricter requirement (for high-risk ops like manual file ingestion):
      - require_local_only_mode=True -> requires SarahMemoryGlobals.LOCAL_ONLY_MODE=True
    """
    if not _is_local_request():
        return False
    try:
        import SarahMemoryGlobals as _G  # type: ignore
        if not bool(getattr(_G, "NEOSKYMATRIX", False)):
            return False
        if require_local_only_mode and (not bool(getattr(_G, "LOCAL_ONLY_MODE", False))):
            return False
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------

def _ok(**payload):
    out = {"ok": True}
    out.update(payload)
    resp = jsonify(out)
    # CORS-friendly defaults (safe; app.py may also set these)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp, 200


def _err(msg: str, code: int = 400, **payload):
    out = {"ok": False, "error": msg}
    out.update(payload)
    resp = jsonify(out)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return resp, code


# ---------------------------------------------------------------------
# Activity log helper
# ---------------------------------------------------------------------

def log_file_event(action: str, path: str, user: str = "system", details: Optional[dict] = None) -> None:
    """Append-only event log for file operations."""
    try:
        log_dir = _get_data_dir() / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "api_events.log"

        event = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "action": action,
            "path": path,
            "user": user,
            "details": details or {},
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Failed to log event: {e}")


# ---------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------

def _sanitize_filename(name: str) -> str:
    name = (name or "").replace("\\", "/").split("/")[-1]
    name = name.strip().replace("..", "_")
    if not name:
        return "file.bin"
    name = "".join(ch for ch in name if ch.isprintable())
    return name[:255]




# Sensitive directories/files that must NEVER be reachable via API file operations
# (prevents policy/mod tampering and server-side leakage).
_PROTECTED_REL_PREFIXES = (
    "data/mods",          # includes ./data/mods/v800 policies
    "data/memory",        # memory datasets/logs
    "data/vault",         # any future secure vault
)
_PROTECTED_REL_FILES = (
    ".env",
    "SarahMemoryGlobals.py",
)

def _is_protected_path(candidate: Path, base: Path) -> bool:
    """Return True if candidate is within a protected subtree or is a protected file."""
    try:
        rel = candidate.relative_to(base)
    except Exception:
        return True
    rel_norm = str(rel).replace("\\", "/").lstrip("/").lower()
    if rel_norm in (f.lower() for f in _PROTECTED_REL_FILES):
        return True
    for pref in _PROTECTED_REL_PREFIXES:
        pref_norm = str(pref).replace("\\", "/").lstrip("/").lower().rstrip("/")
        if rel_norm == pref_norm or rel_norm.startswith(pref_norm + "/"):
            return True
    return False

def _norm_path(p: str) -> str:
    """Normalize and validate path (prevent traversal). Returns ABS path under BASE_DIR or empty."""
    p = (p or "").strip()
    if not p:
        return ""

    base = _get_base_dir()

    try:
        candidate = (base / p).resolve() if not Path(p).is_absolute() else Path(p).expanduser().resolve()
        try:
            candidate.relative_to(base)
            # Block protected areas (policies/mods/memory) from API access by default.
            if _is_protected_path(candidate, base):
                if _is_local_request() and os.environ.get("SARAHMEMORY_ALLOW_PROTECTED_PATHS", "0").strip().lower() in ("1", "true", "yes", "on"):
                    return str(candidate)
                logger.warning(f"Protected path blocked: {p}")
                return ""
            return str(candidate)
        except ValueError:
            logger.warning(f"Path traversal attempt blocked: {p}")
            return ""
    except Exception as e:
        logger.error(f"Path normalization error: {e}")
        return ""


def _safe_stat(path: str) -> Dict[str, Any]:
    try:
        st = os.stat(path)
        return {"size": int(st.st_size), "mtime": float(st.st_mtime)}
    except Exception:
        return {"size": 0, "mtime": 0}


def _list_dir(abs_path: str) -> Tuple[bool, Any]:
    try:
        p = Path(abs_path)
        if not p.exists():
            return False, f"Path not found: {abs_path}"
        if not p.is_dir():
            return False, f"Not a directory: {abs_path}"

        items = []
        for child in p.iterdir():
            st = _safe_stat(str(child))
            items.append({
                "name": child.name,
                "path": str(child.relative_to(_get_base_dir())),
                "type": "folder" if child.is_dir() else "file",
                "size": 0 if child.is_dir() else int(st.get("size", 0) or 0),
                "modified": float(st.get("mtime", 0) or 0),
            })

        items.sort(key=lambda x: (0 if x["type"] == "folder" else 1, (x["name"] or "").lower()))
        return True, items
    except Exception as e:
        return False, str(e)


def _virtual_drives() -> list:
    base = _get_base_dir()
    return [
        {"name": "SarahMemory Root", "path": ".", "kind": "drive"},
        {"name": "Downloads", "path": "downloads", "kind": "folder"},
        {"name": "Dumpster", "path": "dumpster", "kind": "folder"},
        {"name": "Data", "path": "data", "kind": "folder"},
    ]


def _unique_path_in_dir(dir_path: Path, filename: str) -> Path:
    filename = _sanitize_filename(filename)
    dst = dir_path / filename
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    for i in range(1, 10_000):
        cand = dir_path / f"{stem} ({i}){suffix}"
        if not cand.exists():
            return cand
    return dir_path / f"{stem} ({uuid.uuid4().hex}){suffix}"


def _load_dumpster_index() -> Dict[str, Any]:
    p = _dumpster_index_path()
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8") or "{}")
    except Exception:
        pass
    return {}


def _save_dumpster_index(idx: Dict[str, Any]) -> None:
    p = _dumpster_index_path()
    try:
        _dumpster_dir().mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(idx, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to save dumpster index: {e}")


# ---------------------------------------------------------------------
# Routes: Files
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
        "canTrash": bool(enabled),
        "canUpload": bool(enabled),
        "canUnmount": False,
        "canFormat": False,
    }

    note = ""
    if not enabled:
        note = "File browsing disabled (cloud-safe mode). Run locally or enable SARAHMEMORY_ALLOW_SERVER_FILES=1 explicitly."

    return _ok(capabilities=caps, note=note, base=(str(_get_base_dir()) if enabled else ""))


@bp.get("/api/files/drives")
def files_drives():
    if not _files_enabled():
        return _err("Drive listing not available (cloud-safe mode).", 403)
    _ensure_core_dirs()
    return _ok(drives=_virtual_drives())


@bp.post("/api/files/list")
def files_list():
    if not _files_enabled():
        return _err("File browsing not available (cloud-safe mode).", 403)

    _ensure_core_dirs()
    data = request.get_json(silent=True) or {}
    rel = (data.get("path") or "").strip() or "."

    if rel in ("/", "\\", ""):
        rel = "."

    abs_path = _norm_path(rel)
    if not abs_path:
        return _err("Invalid path", 400)

    if Path(abs_path).resolve() == _get_base_dir().resolve():
        drives = _virtual_drives()
        items = [{
            "name": d["name"],
            "path": d["path"],
            "type": "folder",
            "size": 0,
            "modified": 0,
        } for d in drives]
        return _ok(path=".", items=items)

    ok, items_or_err = _list_dir(abs_path)
    if not ok:
        return _err(str(items_or_err), 404)
    return _ok(path=str(Path(abs_path).relative_to(_get_base_dir())), items=items_or_err)


@bp.post("/api/files/mkdir")
def files_mkdir():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)

    _ensure_core_dirs()
    data = request.get_json(silent=True) or {}
    parent_rel = (data.get("path") or "").strip()
    name = (data.get("name") or "").strip()
    if not parent_rel or not name:
        return _err("Missing path or name")

    parent_abs = _norm_path(parent_rel)
    if not parent_abs:
        return _err("Invalid path")

    try:
        target = Path(parent_abs) / _sanitize_filename(name)
        target.mkdir(parents=True, exist_ok=False)
        log_file_event("mkdir", str(target), details={"rel": str(target.relative_to(_get_base_dir()))})
        return _ok(created=True, path=str(target.relative_to(_get_base_dir())))
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
    src_rel = (data.get("path") or "").strip()
    new_name = (data.get("new_name") or "").strip()
    if not src_rel or not new_name:
        return _err("Missing path or new_name")

    src_abs = _norm_path(src_rel)
    if not src_abs:
        return _err("Invalid path")

    try:
        sp = Path(src_abs)
        if not sp.exists():
            return _err("Source not found", 404)
        dst = sp.parent / _sanitize_filename(new_name)
        sp.rename(dst)
        log_file_event("rename", str(dst), details={"from": src_rel, "to": str(dst.relative_to(_get_base_dir()))})
        return _ok(renamed=True, path=str(dst.relative_to(_get_base_dir())))
    except Exception as e:
        return _err("Rename failed", 500, detail=str(e))


@bp.post("/api/files/upload")
def files_upload():
    """Multipart + JSON base64 fallback upload. Saves to BASE_DIR/downloads with SHA256."""
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)

    _ensure_core_dirs()
    dl_dir = _downloads_dir()

    files = []
    try:
        if request.files:
            files = request.files.getlist("files") or [request.files.get("file")]
            files = [f for f in files if f and getattr(f, "filename", "")]
    except Exception:
        files = []

    if not files:
        body = request.get_json(silent=True) or {}
        b64_data = body.get("data") or body.get("data_b64")
        filename = body.get("filename") or "upload.bin"
        if b64_data:
            try:
                raw = base64.b64decode(str(b64_data).encode("utf-8"))
            except Exception as e:
                return _err(f"Invalid base64: {e}", 400)

            from io import BytesIO

            class FakeFile:
                def __init__(self, data: bytes, name: str):
                    self.stream = BytesIO(data)
                    self.filename = name

            files = [FakeFile(raw, str(filename))]

    if not files:
        return _err("No files uploaded", 400)

    max_bytes = int(os.getenv("SARAHMEMORY_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
    saved = []

    for f in files:
        try:
            orig_name = _sanitize_filename(getattr(f, "filename", "") or "file.bin")
            dst_path = _unique_path_in_dir(dl_dir, orig_name)

            sha256 = hashlib.sha256()
            total = 0
            with open(dst_path, "wb") as out_f:
                while True:
                    chunk = f.stream.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        out_f.close()
                        try:
                            dst_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        return _err(f"Upload too large (max {max_bytes} bytes)", 413)
                    sha256.update(chunk)
                    out_f.write(chunk)

            st = _safe_stat(str(dst_path))
            result = {
                "name": dst_path.name,
                "path": str(dst_path.relative_to(_get_base_dir())),
                "size": int(st.get("size", 0)),
                "sha256": sha256.hexdigest(),
                "modified": float(st.get("mtime", 0)),
            }
            saved.append(result)
            log_file_event("upload", str(dst_path), details={"size": total, "sha256": result["sha256"]})
        except Exception as e:
            logger.exception("Upload failed")
            return _err(f"Upload failed: {e}", 500)

    return _ok(saved=saved, count=len(saved))


@bp.post("/api/ingest/eat_this")
def ingest_eat_this():
    """
    Local-only manual ingestion endpoint.

    Trigger contract:
      - Only active for localhost requests AND SarahMemoryGlobals.LOCAL_ONLY_MODE=True AND NEOSKYMATRIX=True
      - Only runs when command text equals 'EAT THIS' (case-insensitive)
      - Accepts multipart form-data with one or more files under 'files' (or 'file')
      - NEVER forwards data to any external API; this is a local learn op.
    """
    if request.method == "OPTIONS":
        return _ok()

    if not _local_ingest_enabled(require_local_only_mode=True):
        return _err("Not available (local-only ingestion disabled).", 403)

    _ensure_core_dirs()

    # command gating
    cmd = ""
    try:
        cmd = (request.form.get("text") or request.form.get("command") or request.form.get("cmd") or "").strip()
    except Exception:
        cmd = ""
    if cmd.lower() != "eat this":
        return _err("Ingestion requires explicit 'EAT THIS' command.", 400)

    # gather files
    files = []
    try:
        if request.files:
            files = request.files.getlist("files") or [request.files.get("file")]
            files = [f for f in files if f and getattr(f, "filename", "")]
    except Exception:
        files = []
    if not files:
        return _err("No files uploaded", 400)

    # save into downloads/ingest for traceability
    ingest_dir = (_downloads_dir() / "ingest").resolve()
    ingest_dir.mkdir(parents=True, exist_ok=True)

    max_bytes = int(os.getenv("SARAHMEMORY_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))
    saved_paths = []
    saved_meta = []

    for f in files:
        try:
            orig_name = _sanitize_filename(getattr(f, "filename", "") or "file.bin")
            dst_path = _unique_path_in_dir(ingest_dir, orig_name)

            sha256 = hashlib.sha256()
            total = 0
            with open(dst_path, "wb") as out_f:
                while True:
                    chunk = f.stream.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        out_f.close()
                        try:
                            dst_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                        return _err(f"Upload too large (max {max_bytes} bytes)", 413)
                    sha256.update(chunk)
                    out_f.write(chunk)

            st = _safe_stat(str(dst_path))
            rel_path = str(dst_path.relative_to(_get_base_dir()))
            saved_paths.append(str(dst_path))
            saved_meta.append({
                "name": dst_path.name,
                "path": rel_path,
                "size": int(st.get("size", 0)),
                "sha256": sha256.hexdigest(),
                "modified": float(st.get("mtime", 0)),
            })

            log_file_event("eat_this_upload", str(dst_path), details={"size": total, "sha256": sha256.hexdigest()})
        except Exception as e:
            logger.exception("EatThis upload failed")
            return _err(f"Upload failed: {e}", 500)

    # Learn from files immediately (no indexing required)
    learned = 0
    errors = []
    try:
        import SarahMemorySystemLearn as SML  # type: ignore
        tuples = []
        for p in saved_paths:
            ext = os.path.splitext(p)[1].lower()
            tuples.append((p, ext))
        # process_files expects list[(path, type)]
        SML.process_files(tuples, learn_registry=False)
        learned = len(tuples)
    except Exception as e:
        errors.append(f"{type(e).__name__}: {e}")
        logger.exception("EatThis learning failed")

    return _ok(
        ingested=True,
        learned=learned,
        files=saved_meta,
        errors=errors,
    )



@bp.post("/api/index/push")
def index_push():
    """
    Local-only "push-to-index" endpoint for Frontend automation.

    Purpose:
      - Accepts lightweight hints and payloads from UI (ResearchScreen / FilesScreen)
      - Converts them into indexed artifacts inside system_index.db WITHOUT editing SarahMemorySystemIndexer.py
      - Enables SarahMemorySelfAware auto-learn loop to detect new rows in system_index.db and trigger learning.

    Security / Governance:
      - localhost only
      - requires NEOSKYMATRIX=True
      - NO extra kill-switch required (NEOSKYMATRIX is the master gate)

    Payload (JSON):
      { "kind": "web_text", "url": "...", "title": "...", "text": "...", "content_type": "...", "source": "research_screen" }
      { "kind": "file_event", "op": "upload|rename|move|copy|mkdir|delete", "path": "relative/or/abs", "paths": ["..."], "mime": "...", "size": 123 }
      { "items": [ ... any of the above ... ] }

    Notes:
      - For web_text: stores a local .txt artifact under downloads/index_push/web/ then indexes that file.
      - For file_event: indexes any existing files referenced (dirs are ignored).
    """
    if request.method == "OPTIONS":
        return _ok()

    # Master governance gate: localhost + NEOSKYMATRIX (LOCAL_ONLY_MODE is NOT required for index push)
    if not _local_ingest_enabled(require_local_only_mode=False):
        return _err("Not available (local-only indexing disabled).", 403)

    payload = request.get_json(silent=True) or {}

    items = payload.get("items")
    if not items:
        items = [payload]
    if not isinstance(items, list):
        items = [payload]

    try:
        import SarahMemorySystemIndexer as SI  # type: ignore
        SI.init_db()
    except Exception as e:
        logger.exception("Index push failed (indexer init)")
        return _err(f"Indexer init failed: {type(e).__name__}: {e}", 500)

    stored = []
    indexed = 0
    errors = []

    # Where we store "web_text" artifacts
    base_dir = _downloads_dir().resolve()
    push_dir = (base_dir / "index_push").resolve()
    web_dir = (push_dir / "web").resolve()
    try:
        web_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    def _index_file(abs_path: str, file_type: str = "") -> bool:
        nonlocal indexed
        try:
            if not abs_path:
                return False
            if not os.path.isfile(abs_path):
                return False
            ext = os.path.splitext(abs_path)[1].lower()
            ft = file_type or ext or "file"
            st = os.stat(abs_path)
            modified_iso = datetime.utcfromtimestamp(st.st_mtime).isoformat()
            sha = SI.compute_sha256(abs_path)
            SI.insert_file_record(abs_path, ft, int(st.st_size), modified_iso, sha)
            indexed += 1
            return True
        except Exception as e:
            errors.append(f"index_file {abs_path}: {type(e).__name__}: {e}")
            return False

    for it in items:
        try:
            if not isinstance(it, dict):
                continue
            kind = str(it.get("kind") or "").strip().lower()

            # -----------------------------
            # Research screen web text
            # -----------------------------
            if kind == "web_text":
                url = str(it.get("url") or "").strip()
                title = str(it.get("title") or "").strip()
                text = str(it.get("text") or "").strip()
                content_type = str(it.get("content_type") or "text/plain").strip()
                source = str(it.get("source") or "ui").strip()

                if not text:
                    continue

                # create stable-ish filename from url+title hash
                h = hashlib.sha256()
                h.update((url + "\n" + title + "\n" + source).encode("utf-8", errors="ignore"))
                digest = h.hexdigest()[:16]
                fname = f"web_{digest}_{int(time.time())}.txt"
                dst = (web_dir / fname).resolve()

                # write artifact
                header = []
                if title:
                    header.append(f"TITLE: {title}")
                if url:
                    header.append(f"URL: {url}")
                if content_type:
                    header.append(f"CONTENT_TYPE: {content_type}")
                if source:
                    header.append(f"SOURCE: {source}")
                header.append("----")
                blob = "\n".join(header) + "\n" + text + "\n"

                with open(dst, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(blob)

                stored.append({"kind": "web_text", "path": str(dst), "url": url, "title": title})
                _index_file(str(dst), file_type="web_text")
                continue

            # -----------------------------
            # Files screen events (path hints)
            # -----------------------------
            if kind == "file_event":
                paths = []
                if it.get("paths") and isinstance(it.get("paths"), list):
                    paths = [str(p) for p in it.get("paths") if p]
                elif it.get("path"):
                    paths = [str(it.get("path"))]

                for p in paths:
                    abs_p = _norm_path(p)
                    if not abs_p:
                        continue
                    if os.path.isfile(abs_p):
                        stored.append({"kind": "file_event", "path": abs_p, "op": it.get("op")})
                        _index_file(abs_p)
                continue

            # -----------------------------
            # Unknown kinds are ignored (forward compatible)
            # -----------------------------
        except Exception as e:
            errors.append(f"item: {type(e).__name__}: {e}")

    return _ok(pushed=True, stored=stored, indexed=indexed, errors=errors)

@bp.post("/api/files/delete")
def files_delete():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)

    _ensure_core_dirs()
    data = request.get_json(silent=True) or {}
    src_rel = (data.get("path") or "").strip()
    mode = (data.get("mode") or "permanent").strip().lower()

    src_abs = _norm_path(src_rel)
    if not src_abs:
        return _err("Missing or invalid path")

    sp = Path(src_abs)
    if not sp.exists():
        return _err("Not found", 404)

    try:
        if mode == "trash":
            did = uuid.uuid4().hex
            items_dir = _dumpster_items_dir()
            items_dir.mkdir(parents=True, exist_ok=True)

            safe_name = _sanitize_filename(sp.name)
            stored = items_dir / f"{did}__{safe_name}"
            if stored.exists():
                stored = items_dir / f"{did}__{safe_name}__{uuid.uuid4().hex}"

            shutil.move(str(sp), str(stored))

            idx = _load_dumpster_index()
            idx[did] = {
                "id": did,
                "orig_rel": src_rel,
                "orig_abs": str(sp),
                "name": safe_name,
                "kind": "folder" if stored.is_dir() else "file",
                "trashed_ts": float(time.time()),
                "stored_rel": str(stored.relative_to(_get_base_dir())),
                "stored_abs": str(stored),
            }
            _save_dumpster_index(idx)

            log_file_event("trash", str(sp), details={"dumpster_id": did, "stored": str(stored)})
            return _ok(trashed=True, id=did, stored_path=str(stored.relative_to(_get_base_dir())))

        if sp.is_dir():
            shutil.rmtree(str(sp))
        else:
            sp.unlink()
        log_file_event("delete_permanent", str(sp), details={"rel": src_rel})
        return _ok(deleted=True)
    except Exception as e:
        logger.exception("Delete failed")
        return _err(f"Delete failed: {e}", 500)


@bp.get("/api/files/trash/list")
def trash_list():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)
    _ensure_core_dirs()

    idx = _load_dumpster_index()
    items = list(idx.values())
    try:
        items.sort(key=lambda x: float(x.get("trashed_ts", 0)), reverse=True)
    except Exception:
        pass
    return _ok(items=items, count=len(items))


@bp.post("/api/files/trash/restore")
def trash_restore():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)
    _ensure_core_dirs()

    data = request.get_json(silent=True) or {}
    did = (data.get("id") or "").strip()
    if not did:
        return _err("Missing id")

    idx = _load_dumpster_index()
    rec = idx.get(did)
    if not rec:
        return _err("Not found", 404)

    stored_abs = Path(str(rec.get("stored_abs") or ""))
    orig_rel = str(rec.get("orig_rel") or "")
    orig_abs = _norm_path(orig_rel)
    if not stored_abs.exists():
        idx.pop(did, None)
        _save_dumpster_index(idx)
        return _err("Stored item missing", 410)
    if not orig_abs:
        return _err("Invalid original path", 400)

    try:
        dest = Path(orig_abs)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest = _unique_path_in_dir(dest.parent, dest.name)
        shutil.move(str(stored_abs), str(dest))

        idx.pop(did, None)
        _save_dumpster_index(idx)

        log_file_event("restore", str(dest), details={"dumpster_id": did})
        return _ok(restored=True, path=str(dest.relative_to(_get_base_dir())))
    except Exception as e:
        logger.exception("Restore failed")
        return _err(f"Restore failed: {e}", 500)


@bp.post("/api/files/trash/empty")
def trash_empty():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)
    _ensure_core_dirs()

    idx = _load_dumpster_index()
    removed = 0
    errors = 0
    for did, rec in list(idx.items()):
        try:
            stored_abs = Path(str(rec.get("stored_abs") or ""))
            if stored_abs.exists():
                if stored_abs.is_dir():
                    shutil.rmtree(str(stored_abs))
                else:
                    stored_abs.unlink()
            idx.pop(did, None)
            removed += 1
        except Exception:
            errors += 1

    _save_dumpster_index(idx)
    log_file_event("trash_empty", str(_dumpster_dir()), details={"removed": removed, "errors": errors})
    return _ok(emptied=True, removed=removed, errors=errors)


@bp.post("/api/files/move")
def files_move():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)

    _ensure_core_dirs()
    data = request.get_json(silent=True) or {}
    src_rel = (data.get("src") or "").strip()
    dst_rel = (data.get("dst") or "").strip()
    if not src_rel or not dst_rel:
        return _err("Missing src or dst")

    src_abs = _norm_path(src_rel)
    dst_abs = _norm_path(dst_rel)
    if not src_abs or not dst_abs:
        return _err("Invalid src or dst")

    try:
        Path(dst_abs).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src_abs, dst_abs)
        log_file_event("move", dst_abs, details={"from": src_rel, "to": dst_rel})
        return _ok(moved=True)
    except Exception as e:
        return _err("Move failed", 500, detail=str(e))


@bp.post("/api/files/copy")
def files_copy():
    if not _files_enabled():
        return _err("Not available (cloud-safe mode).", 403)

    _ensure_core_dirs()
    data = request.get_json(silent=True) or {}
    src_rel = (data.get("src") or "").strip()
    dst_rel = (data.get("dst") or "").strip()
    if not src_rel or not dst_rel:
        return _err("Missing src or dst")

    src_abs = _norm_path(src_rel)
    dst_abs = _norm_path(dst_rel)
    if not src_abs or not dst_abs:
        return _err("Invalid src or dst")

    try:
        sp = Path(src_abs)
        if not sp.exists():
            return _err("Source not found", 404)
        dp = Path(dst_abs)
        dp.parent.mkdir(parents=True, exist_ok=True)
        if sp.is_dir():
            shutil.copytree(src_abs, dst_abs)
        else:
            shutil.copy2(src_abs, dst_abs)
        log_file_event("copy", dst_abs, details={"from": src_rel, "to": dst_rel})
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
    src_rel = (data.get("path") or "").strip()
    if not src_rel:
        return _err("Missing path")

    src_abs = _norm_path(src_rel)
    if not src_abs:
        return _err("Invalid path")

    try:
        sp = Path(src_abs)
        if not sp.exists() or not sp.is_file():
            return _err("File not found", 404)

        token = uuid.uuid4().hex
        expires = time.time() + 60.0
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
# Communications domain ownership
# ---------------------------------------------------------------------
# Email routes were moved to appcomm.py and now live under:
#   /api/comm/email/*
# appsys.py remains the system/files/browser lane only.

# ---------------------------------------------------------------------
# Browser fetch/open endpoints (AI Reader + Native Browser hooks)
# ---------------------------------------------------------------------

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

def _is_private_or_local_host(hostname: str) -> bool:
    h = (hostname or "").strip().lower()
    if not h:
        return True
    if h in ("localhost",):
        return True
    try:
        ip = ipaddress.ip_address(h)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
    except Exception:
        if h.endswith(".local") or h.endswith(".lan"):
            return True
        return False

def _validate_fetch_url(raw_url: str) -> Tuple[bool, str, str]:
    """Return (ok, normalized_url, error_message)."""
    u = (raw_url or "").strip()
    if not u:
        return False, "", "Missing url"
    if not (u.startswith("http://") or u.startswith("https://")):
        if _re.match(r"^[a-z0-9.-]+\.[a-z]{2,}([/:].*)?$", u, _re.I):
            u = "https://" + u
        else:
            return False, "", "Invalid url"
    try:
        parsed = urlparse(u)
    except Exception:
        return False, "", "Invalid url"

    if parsed.scheme not in ("http", "https"):
        return False, "", "Unsupported scheme"
    if not parsed.netloc:
        return False, "", "Invalid host"

    host = parsed.hostname or ""
    if not _is_local_request() and _is_private_or_local_host(host):
        return False, "", "Blocked host"

    return True, u, ""

def _extract_readable_html_and_text(html_doc: str, base_url: str) -> Tuple[str, str, str, list]:
    """Return (title, clean_html, plain_text, links)."""
    soup = BeautifulSoup(html_doc or "", "html.parser")

    for tag in soup(["script", "style", "noscript", "iframe", "object", "embed", "link", "meta"]):
        try:
            tag.decompose()
        except Exception:
            pass

    title = ""
    try:
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
    except Exception:
        title = ""

    links = []
    for a in soup.find_all("a"):
        try:
            href = (a.get("href") or "").strip()
            if not href:
                continue
            abs_href = urljoin(base_url, href)
            a["href"] = abs_href
            if len(links) < 200:
                text = (a.get_text(" ", strip=True) or "")[:200]
                links.append({"text": text, "url": abs_href})
        except Exception:
            continue

    main = soup.find("main") or soup.find("article") or soup.body or soup
    raw_html = str(main)

    allowed_tags = [
        "a", "p", "br", "hr",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "strong", "em", "b", "i", "u",
        "ul", "ol", "li",
        "blockquote", "code", "pre",
        "table", "thead", "tbody", "tr", "th", "td",
        "img", "span", "div",
    ]
    allowed_attrs = {
        "a": ["href", "title", "target", "rel"],
        "img": ["src", "alt", "title"],
        "*": ["class"],
    }

    clean_html = bleach.clean(
        raw_html,
        tags=allowed_tags,
        attributes=allowed_attrs,
        protocols=["http", "https", "mailto"],
        strip=True,
    )

    text_soup = BeautifulSoup(clean_html, "html.parser")
    plain_text = text_soup.get_text("\n", strip=True)

    return title, clean_html, plain_text, links


def _browser_state_path() -> Path:
    return (_get_data_dir() / "browser_state.json").resolve()

def _read_browser_state() -> Dict[str, Any]:
    try:
        path = _browser_state_path()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def _write_browser_state(data: Dict[str, Any]) -> None:
    try:
        path = _browser_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        os.replace(str(tmp), str(path))
    except Exception as exc:
        logger.warning(f"Failed to write browser state: {exc}")


@bp.route("/api/browser/governance", methods=["GET", "OPTIONS"])
def browser_governance():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    return _ok(
        api_domain="browser",
        route_base="/api/browser",
        governance={
            "research_browser_state_file": str(_get_data_dir() / "browser_state.json"),
            "fetch_url_validation_required": True,
            "local_request_required_for_sensitive_ops": True,
            "rem_dl_observation_contract": "browser_state + reader packet + explicit frontend sync",
            "manual_app_route_additions_required": False,
            "safety_notes": [
                "Browser state synchronization is data capture, not autonomous OS command execution.",
                "REM/DL can consume browser observations only through governed state packets/routes.",
            ],
        },
    )

@bp.route("/api/browser/state", methods=["GET", "POST", "OPTIONS"])
def browser_state():
    """Store/read the Research Browser page state for Chat, REM, and DL lanes."""
    if request.method == "OPTIONS":
        return _ok()

    if request.method == "GET":
        state = _read_browser_state()
        return _ok(state=state, has_page=bool(state.get("url")))

    data = request.get_json(silent=True) or {}
    now = time.time()
    links = data.get("links") if isinstance(data.get("links"), list) else []
    text = str(data.get("text") or "")
    clean_html = str(data.get("clean_html") or "")

    state = {
        "ok": bool(data.get("ok", True)),
        "url": str(data.get("url") or "").strip(),
        "title": str(data.get("title") or data.get("url") or "").strip(),
        "text": text[:250000],
        "clean_html": clean_html[:250000],
        "links": links[:100],
        "content_type": str(data.get("content_type") or ""),
        "surface": str(data.get("surface") or "research_panel"),
        "source": str(data.get("source") or "research_panel"),
        "updated_ts": now,
        "updated_iso": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    _write_browser_state(state)
    return _ok(saved=True, state={k: v for k, v in state.items() if k not in ("clean_html", "text")}, text_chars=len(state.get("text") or ""))

@bp.route("/api/browser/fetch", methods=["GET", "POST", "OPTIONS"])
def browser_fetch():
    """
    Fetch a URL server-side and return a sanitized HTML + plaintext bundle.

    Accepts:
      - GET  /api/browser/fetch?url=https://example.com
      - POST /api/browser/fetch { "url": "https://example.com" }

    Cloud-safe:
      - Blocks private/loopback hosts unless request is local
      - Limits payload size and timeouts
    """
    if request.method == "OPTIONS":
        # Preflight-safe
        resp = jsonify({"ok": True})
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        return resp, 204

    raw_url = ""
    if request.method == "GET":
        raw_url = (request.args.get("url") or request.args.get("href") or "").strip()
    else:
        data = request.get_json(silent=True) or {}
        raw_url = (data.get("url") or data.get("href") or "").strip()

    ok, url, err = _validate_fetch_url(raw_url)
    if not ok:
        return _err(err or "Invalid url", 400)

    timeout = 12
    max_bytes = 2_000_000  # 2MB cap

    try:
        headers = {
            "User-Agent": _BROWSER_UA,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

        resp = requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True)
        ctype = (resp.headers.get("content-type") or "").lower()

        if "text/html" not in ctype and "application/xhtml+xml" not in ctype:
            raw = resp.raw.read(max_bytes, decode_content=True)
            snippet = raw[:4000].decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)[:4000]
            return _ok(
                url=resp.url,
                title=resp.url,
                clean_html=f"<pre>{bleach.clean(snippet)}</pre>",
                text=snippet,
                links=[],
                content_type=ctype,
            )

        chunks = []
        total = 0
        for chunk in resp.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                break
            chunks.append(chunk)

        raw_html = b"".join(chunks).decode(resp.encoding or "utf-8", errors="ignore")

        title, clean_html, plain_text, links = _extract_readable_html_and_text(raw_html, resp.url)

        if len(plain_text) > 200_000:
            plain_text = plain_text[:200_000] + "\n\n[Truncated]"

        return _ok(
            url=resp.url,
            title=title or resp.url,
            clean_html=clean_html,
            text=plain_text,
            links=links,
            content_type=ctype,
        )
    except requests.exceptions.Timeout:
        return _err("Fetch timeout", 504)
    except Exception as e:
        return _err("Fetch failed", 500, detail=str(e))


@bp.post("/api/browser/open")
@bp.post("/api/browser/open_native")
def browser_open_external():
    """Local-only: open URL in the system default browser."""
    if not _is_local_request():
        return _err("Not available (cloud-safe mode).", 403)

    data = request.get_json(silent=True) or {}
    raw_url = (data.get("url") or "").strip()
    ok, url, err = _validate_fetch_url(raw_url)
    if not ok:
        return _err(err or "Invalid url", 400)

    try:
        import webbrowser as _wb
        _wb.open(url)
        return _ok(opened=True, url=url)
    except Exception as e:
        return _err("Open failed", 500, detail=str(e))


# ---------------------------------------------------------------------
# init_app (called by app.py ONCE)
# ---------------------------------------------------------------------

def init_app(app) -> None:
    if "appsys_v800" in getattr(app, "blueprints", {}):
        return
    app.register_blueprint(bp)

# ====================================================================
# END OF appsys.py v9.0.0
# ====================================================================
