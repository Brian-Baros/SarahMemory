# --==The SarahMemory Project==--
# File: /api/server/appmedia.py
# Purpose: Creative Studio "media broker" endpoints (images/music/video + hybrid outputs)
# Part of the SarahMemory Companion AI-bot Platform
# Part of the SarahMemory Companion AI-bot Platform
# Author: © 2025 Brian Lee Baros. All Rights Reserved.
# www.linkedin.com/in/brian-baros-29962a176
# https://www.facebook.com/bbaros
# brian.baros@sarahmemory.com
# 'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
# https://www.sarahmemory.com
# https://api.sarahmemory.com
# https://ai.sarahmemory.com

# Design goals:
# - NO duplicate endpoints with app.py/appnet.py (avoid Flask route collisions)
# - Everything is namespaced under /api/media/*
# - Standardize output packaging (exports + manifest sidecar)
# - Provide a clean contract for Creative Studio UI: submit -> render -> download
# - Fail-soft: missing optional engines should not break the API surface

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, List

from flask import Blueprint, jsonify, request, send_file

# Stable blueprint name (prevents register collisions across reloads)
bp = Blueprint("appmedia_v800", __name__)

# ---------------------------
# Injected by app.py at boot
# ---------------------------
_CONNECT_SQLITE: Optional[Callable[..., Any]] = None
_META_DB: Optional[str] = None
_API_KEY_AUTH_OK: Optional[Callable[[], bool]] = None
_SIGN_OK: Optional[Callable[[bytes, str], bool]] = None

# ---------------------------
# Config / Limits
# ---------------------------
PROJECT_VERSION = os.environ.get("PROJECT_VERSION", "8.0.0")

MAX_PROMPT_CHARS = int(os.environ.get("MEDIA_MAX_PROMPT_CHARS", "12000"))
MAX_INLINE_B64_BYTES = int(os.environ.get("MEDIA_MAX_INLINE_B64_BYTES", str(8 * 1024 * 1024)))  # 8 MB
MAX_EXPORT_BYTES_HINT = int(os.environ.get("MEDIA_MAX_EXPORT_BYTES_HINT", str(256 * 1024 * 1024)))  # 256 MB (hint)

# Export root preference:
# 1) SarahMemoryGlobals.CANVAS_EXPORTS_DIR
# 2) <DATA_DIR>/canvas/exports if SarahMemoryGlobals available
# 3) alongside meta.db as fallback
DEFAULT_EXPORT_SUBDIR = os.environ.get("MEDIA_EXPORT_SUBDIR", "canvas/exports").strip()

# Optional “download token” mechanism (lightweight). If empty, download is open (dev).
DOWNLOAD_TOKEN = (os.environ.get("MEDIA_DOWNLOAD_TOKEN") or "").strip()

# ---------------------------
# Utilities
# ---------------------------

def _now() -> float:
    return time.time()

def _iso(ts: Optional[float] = None) -> str:
    return datetime.utcfromtimestamp(ts or _now()).isoformat() + "Z"

def _j() -> Dict[str, Any]:
    return request.get_json(silent=True) or {}

def _ok(**kw) -> Tuple[Any, int]:
    return jsonify({"ok": True, **kw}), 200

def _err(msg: str, code: int = 400, **kw) -> Tuple[Any, int]:
    return jsonify({"ok": False, "error": msg, **kw}), code

def _body_bytes() -> bytes:
    try:
        return request.get_data(cache=True) or b""
    except Exception:
        return b""

def _require_injected() -> bool:
    return bool(_CONNECT_SQLITE and _META_DB)

def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _safe_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"

def _sanitize_filename(name: str, default: str) -> str:
    n = (name or "").strip()
    if not n:
        n = default
    n = n.replace("\\", "/").split("/")[-1]
    n = re.sub(r"[^a-zA-Z0-9._-]+", "_", n)
    n = n.replace("..", "_")
    if not n:
        n = default
    return n[:255]

def _ensure_dir(p: str) -> None:
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass

def _get_export_root() -> str:
    # Best-effort: prefer SarahMemoryGlobals if present
    try:
        import SarahMemoryGlobals as SMG  # type: ignore
        d = getattr(SMG, "CANVAS_EXPORTS_DIR", None)
        if d:
            d = os.fspath(d)
            _ensure_dir(d)
            return d
        # fallback to DATA_DIR/canvas/exports
        base_dir = getattr(SMG, "DATA_DIR", None)
        if base_dir:
            root = os.path.join(os.fspath(base_dir), "canvas", "exports")
            _ensure_dir(root)
            return root
    except Exception:
        pass

    # fallback to meta.db directory
    if _META_DB:
        root = os.path.join(os.path.dirname(_META_DB), DEFAULT_EXPORT_SUBDIR)
        _ensure_dir(root)
        return root

    # last-resort
    root = os.path.join(os.getcwd(), "data", DEFAULT_EXPORT_SUBDIR)
    _ensure_dir(root)
    return root

def _job_dir(job_id: str) -> str:
    root = _get_export_root()
    # Keep per-job directories for clean packaging
    d = os.path.join(root, "jobs", job_id)
    _ensure_dir(d)
    return d

def _manifest_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), "manifest.json")

def _write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _b64d(s: str) -> bytes:
    return base64.b64decode((s or "").encode("ascii"), validate=False)

def _verify_auth(body: bytes) -> bool:
    """
    Accept either:
      - X-Sarah-Signature verified by injected _SIGN_OK(body, sig)
      - API key verified by injected _API_KEY_AUTH_OK()
    If nothing injected, allow (dev mode).
    """
    sig = (request.headers.get("X-Sarah-Signature") or "").strip()
    if sig and _SIGN_OK:
        try:
            return bool(_SIGN_OK(body or b"", sig))
        except Exception:
            return False

    if _API_KEY_AUTH_OK:
        try:
            return bool(_API_KEY_AUTH_OK())
        except Exception:
            return False

    return True

def _download_allowed() -> bool:
    # Optional lightweight download token
    if not DOWNLOAD_TOKEN:
        return True
    tok = (request.args.get("token") or request.headers.get("X-Sarah-Download-Token") or "").strip()
    return tok == DOWNLOAD_TOKEN

# ---------------------------
# DB: media_jobs
# ---------------------------

def _ensure_tables() -> None:
    if not _require_injected():
        return
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS media_jobs (
                job_id TEXT PRIMARY KEY,
                created_ts REAL,
                updated_ts REAL,
                status TEXT,
                kind TEXT,
                request_json TEXT,
                result_json TEXT,
                error_text TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_media_jobs_ts ON media_jobs(created_ts)")
        con.commit()
    except Exception:
        try:
            if con:
                con.rollback()
        except Exception:
            pass
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

def _db_upsert_job(job_id: str, status: str, kind: str, req: Dict[str, Any], result: Optional[Dict[str, Any]] = None, error: str = "") -> None:
    if not _require_injected():
        return
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        now = _now()
        cur.execute(
            """
            INSERT INTO media_jobs(job_id, created_ts, updated_ts, status, kind, request_json, result_json, error_text)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(job_id) DO UPDATE SET
                updated_ts=excluded.updated_ts,
                status=excluded.status,
                kind=excluded.kind,
                request_json=excluded.request_json,
                result_json=excluded.result_json,
                error_text=excluded.error_text
            """,
            (
                job_id,
                now,
                now,
                status,
                kind,
                json.dumps(req or {}, ensure_ascii=False),
                json.dumps(result or {}, ensure_ascii=False),
                (error or "")[:20000],
            ),
        )
        con.commit()
    except Exception:
        try:
            if con:
                con.rollback()
        except Exception:
            pass
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

def _db_get_job(job_id: str) -> Dict[str, Any]:
    if not _require_injected():
        # fall back to manifest only
        m = _read_json(_manifest_path(job_id))
        if m:
            return {"job_id": job_id, "status": m.get("status", "unknown"), "kind": m.get("kind"), "manifest": m}
        return {}
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute("SELECT job_id, created_ts, updated_ts, status, kind, request_json, result_json, error_text FROM media_jobs WHERE job_id=? LIMIT 1", (job_id,))
        row = cur.fetchone()
        if not row:
            return {}
        # sqlite3.Row compatible and tuple compatible
        def _get(i, k):
            try:
                return row[k]
            except Exception:
                return row[i]
        out = {
            "job_id": _get(0, "job_id"),
            "created_ts": _get(1, "created_ts"),
            "updated_ts": _get(2, "updated_ts"),
            "status": _get(3, "status"),
            "kind": _get(4, "kind"),
            "request": json.loads(_get(5, "request_json") or "{}"),
            "result": json.loads(_get(6, "result_json") or "{}"),
            "error": _get(7, "error_text") or "",
        }
        return out
    except Exception:
        return {}
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

# ---------------------------
# Engine adapters (fail-soft)
# ---------------------------

@dataclass
class RenderArtifact:
    role: str               # "image" | "audio" | "video" | "project" | "manifest"
    filename: str
    path: str
    mime: str
    size_bytes: int
    sha256: str

def _probe_engines() -> Dict[str, bool]:
    caps = {}
    try:
        import SarahMemoryCanvasStudio as _cs  # noqa
        caps["canvas_studio"] = True
    except Exception:
        caps["canvas_studio"] = False

    try:
        import SarahMemoryMusicGenerator as _mg  # noqa
        caps["music_generator"] = True
    except Exception:
        caps["music_generator"] = False

    try:
        import SarahMemoryLyricsToSong as _lts  # noqa
        caps["lyrics_to_song"] = True
    except Exception:
        caps["lyrics_to_song"] = False

    try:
        import SarahMemoryVideoEditorCore as _ve  # noqa
        caps["video_editor_core"] = True
    except Exception:
        caps["video_editor_core"] = False

    return caps

def _write_bytes(path: str, blob: bytes) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(blob)

def _artifact_from_path(role: str, path: str, mime: str) -> RenderArtifact:
    p = os.path.abspath(path)
    try:
        st = os.stat(p)
        size = int(st.st_size)
    except Exception:
        size = 0
    sha = ""
    try:
        with open(p, "rb") as f:
            sha = _sha256_hex(f.read())
    except Exception:
        sha = ""
    return RenderArtifact(
        role=role,
        filename=os.path.basename(p),
        path=p,
        mime=mime,
        size_bytes=size,
        sha256=sha,
    )

def _mime_from_ext(filename: str) -> str:
    ext = (filename.rsplit(".", 1)[-1] or "").lower()
    if ext in ("png",):
        return "image/png"
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    if ext in ("webp",):
        return "image/webp"
    if ext in ("wav",):
        return "audio/wav"
    if ext in ("mp3",):
        return "audio/mpeg"
    if ext in ("flac",):
        return "audio/flac"
    if ext in ("mp4",):
        return "video/mp4"
    if ext in ("webm",):
        return "video/webm"
    if ext in ("json",):
        return "application/json"
    return "application/octet-stream"

# ---------------------------
# Routes
# ---------------------------

@bp.get("/api/media/ping")
def media_ping():
    return _ok(pong=True, ts=_now(), version=PROJECT_VERSION, service="media-broker")

@bp.get("/api/media/capabilities")
def media_capabilities():
    return _ok(
        ts=_now(),
        version=PROJECT_VERSION,
        export_root=_get_export_root(),
        limits={
            "max_prompt_chars": MAX_PROMPT_CHARS,
            "max_inline_b64_bytes": MAX_INLINE_B64_BYTES,
            "max_export_bytes_hint": MAX_EXPORT_BYTES_HINT,
        },
        engines=_probe_engines(),
    )

@bp.post("/api/media/job/create")
def media_job_create():
    body = _body_bytes()
    if not _verify_auth(body):
        return _err("unauthorized", 401)

    payload = _j()
    kind = (payload.get("kind") or "generic").strip().lower()
    job_id = _safe_id("mediajob")

    # Establish job folder + baseline manifest
    jd = _job_dir(job_id)
    manifest = {
        "job_id": job_id,
        "kind": kind,
        "status": "created",
        "created_at": _iso(),
        "updated_at": _iso(),
        "export_root": _get_export_root(),
        "job_dir": jd,
        "request": payload,
        "artifacts": [],
        "errors": [],
        "notes": [],
        "version": PROJECT_VERSION,
    }
    _write_json(_manifest_path(job_id), manifest)
    _ensure_tables()
    _db_upsert_job(job_id, "created", kind, payload, result={"manifest": "manifest.json"})

    return _ok(job_id=job_id, manifest=manifest)

@bp.get("/api/media/job/status")
def media_job_status():
    job_id = (request.args.get("job_id") or "").strip()
    if not job_id:
        return _err("missing_job_id", 400)

    job = _db_get_job(job_id)
    if job:
        # Attach manifest if present
        m = _read_json(_manifest_path(job_id))
        if m:
            job["manifest"] = m
        return _ok(job=job)

    # fallback manifest-only
    m = _read_json(_manifest_path(job_id))
    if not m:
        return _err("job_not_found", 404, job_id=job_id)
    return _ok(job={"job_id": job_id, "status": m.get("status", "unknown"), "kind": m.get("kind"), "manifest": m})

@bp.post("/api/media/job/render")
def media_job_render():
    """
    Render entrypoint for Creative Studio.
    Expected payload:
    {
      "job_id": "...",            # optional; if missing, server creates
      "kind": "image|music|video|hybrid",
      "prompt": "...",            # for image generation (or guidance)
      "lyrics": "...",            # for LyricsToSong
      "music": {...},             # for MusicGenerator
      "video": {...},             # for VideoEditorCore
      "output": {
         "filename": "optional.ext",
         "format": "png|wav|mp4|...",
      }
    }
    """
    body = _body_bytes()
    if not _verify_auth(body):
        return _err("unauthorized", 401)

    payload = _j()
    kind = (payload.get("kind") or "generic").strip().lower()

    # Validate + clamp user text fields
    for k in ("prompt", "lyrics"):
        if k in payload and isinstance(payload[k], str) and len(payload[k]) > MAX_PROMPT_CHARS:
            payload[k] = payload[k][:MAX_PROMPT_CHARS]

    job_id = (payload.get("job_id") or "").strip() or _safe_id("mediajob")
    jd = _job_dir(job_id)
    mp = _manifest_path(job_id)

    _ensure_tables()
    _db_upsert_job(job_id, "running", kind, payload)

    manifest = _read_json(mp) or {
        "job_id": job_id,
        "kind": kind,
        "status": "running",
        "created_at": _iso(),
        "updated_at": _iso(),
        "export_root": _get_export_root(),
        "job_dir": jd,
        "request": payload,
        "artifacts": [],
        "errors": [],
        "notes": [],
        "version": PROJECT_VERSION,
    }

    manifest["kind"] = kind
    manifest["status"] = "running"
    manifest["updated_at"] = _iso()
    manifest["request"] = payload

    artifacts: List[RenderArtifact] = []

    try:
        # -------------------------
        # IMAGE path (CanvasStudio)
        # -------------------------
        if kind in ("image", "img", "canvas"):
            # Two supported modes:
            #  1) inline_b64: client already generated bytes (server packages it)
            #  2) prompt: server attempts to call CanvasStudio if it exposes a compatible API
            out = payload.get("output") or {}
            fmt = (out.get("format") or "png").strip().lower()
            if fmt not in ("png", "jpg", "jpeg", "webp"):
                fmt = "png"

            filename = _sanitize_filename(out.get("filename") or f"image.{fmt}", f"image.{fmt}")
            out_path = os.path.join(jd, filename)

            inline_b64 = (payload.get("inline_b64") or "").strip()
            if inline_b64:
                blob = _b64d(inline_b64)
                if len(blob) > MAX_INLINE_B64_BYTES:
                    raise ValueError("inline_b64_too_large")
                _write_bytes(out_path, blob)
            else:
                prompt = (payload.get("prompt") or "").strip()
                if not prompt:
                    raise ValueError("missing_prompt_or_inline_b64")

                # Best-effort adapter: we do NOT hard-require a specific method signature.
                # If CanvasStudio has generate_from_prompt/export helpers, we use them.
                try:
                    import SarahMemoryCanvasStudio as CS  # type: ignore
                    studio = getattr(CS, "CanvasStudio", None)
                    if studio is None:
                        raise RuntimeError("CanvasStudio_class_missing")
                    studio_obj = studio()

                    # Common naming patterns (fail-soft)
                    gen = getattr(studio_obj, "generate_from_prompt", None) or getattr(studio_obj, "generate", None)
                    if not callable(gen):
                        raise RuntimeError("CanvasStudio_generate_missing")

                    # Attempt generate
                    generated = gen(prompt, style=payload.get("style"), quality=payload.get("quality"))  # type: ignore[arg-type]

                    # Export strategy:
                    # - If returned bytes, write bytes
                    # - If returned path-like, copy/point to that output
                    # - If returned PIL/numpy, try studio.export_canvas or PIL save
                    if isinstance(generated, (bytes, bytearray)):
                        _write_bytes(out_path, bytes(generated))
                    elif isinstance(generated, str) and os.path.isfile(generated):
                        # copy into job dir
                        with open(generated, "rb") as f:
                            _write_bytes(out_path, f.read())
                    else:
                        # Try a generic export hook
                        exp = getattr(studio_obj, "export_canvas", None) or getattr(studio_obj, "export", None)
                        if callable(exp):
                            exp(generated, out_path, format=fmt.upper())  # type: ignore[misc]
                        else:
                            raise RuntimeError("CanvasStudio_export_missing")

                except Exception as e:
                    raise RuntimeError(f"canvas_render_failed:{e}")

            artifacts.append(_artifact_from_path("image", out_path, _mime_from_ext(out_path)))

        # -------------------------
        # MUSIC path (MusicGenerator)
        # -------------------------
        elif kind in ("music", "audio"):
            out = payload.get("output") or {}
            fmt = (out.get("format") or "wav").strip().lower()
            if fmt not in ("wav", "mp3", "flac", "ogg", "m4a"):
                fmt = "wav"

            filename = _sanitize_filename(out.get("filename") or f"music.{fmt}", f"music.{fmt}")
            out_path = os.path.join(jd, filename)

            inline_b64 = (payload.get("inline_b64") or "").strip()
            if inline_b64:
                blob = _b64d(inline_b64)
                if len(blob) > MAX_INLINE_B64_BYTES:
                    raise ValueError("inline_b64_too_large")
                _write_bytes(out_path, blob)
            else:
                # Best-effort: use MusicStudio export hooks if available
                try:
                    import SarahMemoryMusicGenerator as MG  # type: ignore
                    studio_cls = getattr(MG, "MusicStudio", None)
                    if studio_cls is None:
                        raise RuntimeError("MusicStudio_class_missing")
                    studio = studio_cls()

                    # Allow either a "music" dict or lightweight params at top-level
                    m = payload.get("music") or {}
                    duration = float(m.get("duration") or payload.get("duration") or 15.0)
                    style = (m.get("style") or payload.get("style") or "default").strip()
                    tempo = int(m.get("tempo") or payload.get("tempo") or 120)

                    # Try common entrypoints
                    gen_song = getattr(MG, "generate_song", None) or getattr(studio, "generate_song", None)
                    export = getattr(studio, "export_project", None) or getattr(studio, "export", None)

                    if callable(gen_song) and callable(export):
                        proj = getattr(studio, "create_project", None)
                        if callable(proj):
                            p = proj(f"Job {job_id}", tempo=tempo, key=(m.get("key") or "C"), time_signature=(4, 4))
                        else:
                            p = None

                        # Generate content
                        song = gen_song(duration=duration, style=style, tempo=tempo)  # type: ignore[misc]
                        # Export strategy:
                        # If song is bytes -> write
                        if isinstance(song, (bytes, bytearray)):
                            _write_bytes(out_path, bytes(song))
                        else:
                            # If project exists, attempt to export it
                            try:
                                export(p, out_path, format=fmt.upper(), quality=(m.get("quality") or "high"))  # type: ignore[misc]
                            except Exception:
                                # last resort: if MG can export directly
                                exporter = getattr(MG, "export_audio", None)
                                if callable(exporter):
                                    exporter(song, out_path, fmt=fmt)  # type: ignore[misc]
                                else:
                                    raise RuntimeError("music_export_failed")
                    else:
                        # Minimal fallback: if generate_tone exists, write a tone
                        gen_tone = getattr(MG, "generate_tone", None)
                        if callable(gen_tone):
                            tone = gen_tone(440.0, duration=duration)  # type: ignore[misc]
                            if isinstance(tone, (bytes, bytearray)):
                                _write_bytes(out_path, bytes(tone))
                            else:
                                raise RuntimeError("tone_not_bytes")
                        else:
                            raise RuntimeError("music_generator_missing_entrypoints")

                except Exception as e:
                    raise RuntimeError(f"music_render_failed:{e}")

            artifacts.append(_artifact_from_path("audio", out_path, _mime_from_ext(out_path)))

        # -------------------------
        # VIDEO path (VideoEditorCore)
        # -------------------------
        elif kind in ("video", "vid"):
            out = payload.get("output") or {}
            fmt = (out.get("format") or "mp4").strip().lower()
            if fmt not in ("mp4", "webm", "avi"):
                fmt = "mp4"

            filename = _sanitize_filename(out.get("filename") or f"video.{fmt}", f"video.{fmt}")
            out_path = os.path.join(jd, filename)

            inline_b64 = (payload.get("inline_b64") or "").strip()
            if inline_b64:
                blob = _b64d(inline_b64)
                if len(blob) > MAX_INLINE_B64_BYTES:
                    raise ValueError("inline_b64_too_large")
                _write_bytes(out_path, blob)
            else:
                try:
                    import SarahMemoryVideoEditorCore as VE  # type: ignore
                    editor_cls = getattr(VE, "VideoEditorCore", None)
                    if editor_cls is None:
                        raise RuntimeError("VideoEditorCore_class_missing")
                    editor = editor_cls()

                    v = payload.get("video") or {}
                    # Inputs can be:
                    # - "inputs": list of file paths already on server
                    # - "image_b64" + "audio_b64" for quick composition
                    inputs = v.get("inputs") or []
                    image_b64 = (v.get("image_b64") or "").strip()
                    audio_b64 = (v.get("audio_b64") or "").strip()

                    if image_b64:
                        img_path = os.path.join(jd, "input_image.png")
                        _write_bytes(img_path, _b64d(image_b64))
                        inputs = list(inputs) + [img_path]

                    if audio_b64:
                        aud_path = os.path.join(jd, "input_audio.wav")
                        _write_bytes(aud_path, _b64d(audio_b64))

                    # Preferred: create project + add clips (best-effort)
                    create_project = getattr(editor, "create_project", None)
                    export_project = getattr(editor, "export_project", None)

                    if callable(create_project) and callable(export_project):
                        res = v.get("resolution") or [1920, 1080]
                        fps = int(v.get("fps") or 30)
                        proj = create_project(f"Job {job_id}", resolution=(int(res[0]), int(res[1])), fps=fps)  # type: ignore[misc]

                        # Add inputs as clips if possible
                        add_clip = getattr(proj, "add_clip", None)
                        if callable(add_clip):
                            t = 0.0
                            for p in inputs:
                                if isinstance(p, str) and os.path.isfile(p):
                                    add_clip(p, start_time=t)  # type: ignore[misc]
                                    t += float(v.get("clip_spacing") or 0.0)

                        # Add audio if we staged it
                        if audio_b64 and os.path.isfile(aud_path):
                            add_audio = getattr(proj, "add_audio_track", None)
                            if callable(add_audio):
                                add_audio(aud_path, volume=float(v.get("audio_volume") or 0.8))  # type: ignore[misc]

                        export_project(proj, out_path, quality=(v.get("quality") or "high"), resolution=(v.get("export_resolution") or "1080p"))  # type: ignore[misc]
                    else:
                        raise RuntimeError("video_editor_missing_entrypoints")

                except Exception as e:
                    raise RuntimeError(f"video_render_failed:{e}")

            artifacts.append(_artifact_from_path("video", out_path, _mime_from_ext(out_path)))

        # -------------------------
        # HYBRID path (compose)
        # -------------------------
        elif kind in ("hybrid", "compose", "multimodal"):
            # Strategy:
            # - allow the client to provide multiple inline assets (image/audio/video)
            # - server packages them + writes a manifest suitable for downstream tools
            comp = payload.get("compose") or {}
            parts = comp.get("parts") or []

            if not isinstance(parts, list) or not parts:
                raise ValueError("missing_compose_parts")

            for i, part in enumerate(parts):
                if not isinstance(part, dict):
                    continue
                role = (part.get("role") or f"part_{i}").strip().lower()
                b64 = (part.get("b64") or "").strip()
                filename = _sanitize_filename(part.get("filename") or f"{role}_{i}", f"{role}_{i}")
                if not b64:
                    continue
                blob = _b64d(b64)
                if len(blob) > MAX_INLINE_B64_BYTES:
                    raise ValueError("inline_b64_too_large")
                path = os.path.join(jd, filename)
                _write_bytes(path, blob)
                artifacts.append(_artifact_from_path(role, path, _mime_from_ext(path)))

        else:
            raise ValueError(f"unsupported_kind:{kind}")

        # Finalize manifest
        artifact_list = []
        for a in artifacts:
            artifact_list.append(
                {
                    "role": a.role,
                    "filename": a.filename,
                    "path": a.path,
                    "mime": a.mime,
                    "size_bytes": a.size_bytes,
                    "sha256": a.sha256,
                }
            )

        manifest["artifacts"] = artifact_list
        manifest["status"] = "complete"
        manifest["updated_at"] = _iso()
        _write_json(mp, manifest)

        result = {
            "job_id": job_id,
            "status": "complete",
            "kind": kind,
            "job_dir": jd,
            "manifest": "manifest.json",
            "artifacts": artifact_list,
        }
        _db_upsert_job(job_id, "complete", kind, payload, result=result)
        return _ok(result=result)

    except Exception as e:
        msg = str(e)
        manifest["status"] = "failed"
        manifest["updated_at"] = _iso()
        manifest.setdefault("errors", []).append({"ts": _iso(), "error": msg})
        _write_json(mp, manifest)
        _db_upsert_job(job_id, "failed", kind, payload, result={"manifest": "manifest.json"}, error=msg)
        return _err("render_failed", 500, job_id=job_id, detail=msg)

@bp.get("/api/media/job/download")
def media_job_download():
    """
    Download an artifact from a job directory.
    Query:
      - job_id=...
      - filename=...
      - token=... (optional if MEDIA_DOWNLOAD_TOKEN set)
    """
    if not _download_allowed():
        return _err("download_unauthorized", 401)

    job_id = (request.args.get("job_id") or "").strip()
    filename = (request.args.get("filename") or "").strip()
    if not job_id or not filename:
        return _err("missing_job_id_or_filename", 400)

    filename = _sanitize_filename(filename, "file.bin")
    path = os.path.join(_job_dir(job_id), filename)
    if not os.path.isfile(path):
        return _err("file_not_found", 404, path=path)

    return send_file(path, as_attachment=True, download_name=filename, mimetype=_mime_from_ext(filename))

@bp.get("/api/media/job/manifest")
def media_job_manifest():
    job_id = (request.args.get("job_id") or "").strip()
    if not job_id:
        return _err("missing_job_id", 400)

    mp = _manifest_path(job_id)
    if not os.path.isfile(mp):
        return _err("manifest_not_found", 404, job_id=job_id)

    if not _download_allowed():
        return _err("manifest_unauthorized", 401)

    return send_file(mp, as_attachment=False, download_name="manifest.json", mimetype="application/json")

# ---------------------------
# Initialization / Injection
# ---------------------------

def init_appmedia(
    connect_sqlite: Callable[..., Any],
    meta_db_path: str,
    api_key_auth_ok: Optional[Callable[[], bool]] = None,
    sign_ok: Optional[Callable[[bytes, str], bool]] = None,
) -> Blueprint:
    """
    Called by app.py to inject storage/auth helpers, then mount blueprint.
    """
    global _CONNECT_SQLITE, _META_DB, _API_KEY_AUTH_OK, _SIGN_OK
    _CONNECT_SQLITE = connect_sqlite
    _META_DB = meta_db_path
    _API_KEY_AUTH_OK = api_key_auth_ok
    _SIGN_OK = sign_ok

    # Ensure tables exist at init time (fail-soft)
    _ensure_tables()
    return bp
