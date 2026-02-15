# --==The SarahMemory Project==--
# File: /api/server/appmedia.py
# Purpose: Creative Studio "media broker" endpoints (images/music/video + hybrid outputs)
# Part of the SarahMemory Companion AI-bot Platform
# Author: © 2025 Brian Lee Baros. All Rights Reserved.
# www.linkedin.com/in/brian-baros-29962a176
# https://www.facebook.com/bbaros
# brian.baros@sarahmemory.com
# 'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
# https://www.sarahmemory.com
# https://api.sarahmemory.com
# https://ai.sarahmemory.com

"""appmedia.py

Enterprise intent
- Stand up a stable, cloud-safe API surface for Creative Studio outputs.
- Standardize packaging: job folder + manifest.json + downloadable artifacts.
- Fail-soft: if an optional engine is missing, endpoints remain online.

API contract (high level)
- POST /api/media/job/create
- POST /api/media/job/render
- GET  /api/media/job/status?job_id=...
- GET  /api/media/job/download?job_id=...&filename=...
- GET  /api/media/job/manifest?job_id=...
- GET  /api/media/capabilities

Notes
- This module does NOT expose any filesystem outside the export root.
- All job IDs are sanitized to prevent path traversal.
"""

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

def _sanitize_job_id(job_id: str) -> str:
    """Hard containment: force job_id into a safe, filename-like token."""
    j = (job_id or "").strip()
    if not j:
        return _safe_id("mediajob")
    j = j.replace("\\", "/")
    j = j.split("/")[-1]
    j = re.sub(r"[^a-zA-Z0-9._-]+", "_", j)
    j = j.replace("..", "_")
    if not j:
        return _safe_id("mediajob")
    return j[:128]

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
    jid = _sanitize_job_id(job_id)
    d = os.path.join(root, "jobs", jid)
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
    """Authentication model (fail-soft for dev)

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
    if not DOWNLOAD_TOKEN:
        return True
    tok = (request.args.get("token") or request.headers.get("X-Sarah-Download-Token") or "").strip()
    return tok == DOWNLOAD_TOKEN

def _send_file_compat(path: str, *, as_attachment: bool, download_name: str, mimetype: str):
    """Compatibility shim for Flask < 2.0 (download_name not supported)."""
    try:
        return send_file(path, as_attachment=as_attachment, download_name=download_name, mimetype=mimetype)
    except TypeError:
        return send_file(path, as_attachment=as_attachment, mimetype=mimetype)

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
        cur.execute(
            """
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
            """
        )
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

def _db_upsert_job(
    job_id: str,
    status: str,
    kind: str,
    req: Dict[str, Any],
    result: Optional[Dict[str, Any]] = None,
    error: str = "",
) -> None:
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
    job_id = _sanitize_job_id(job_id)

    if not _require_injected():
        m = _read_json(_manifest_path(job_id))
        if m:
            return {"job_id": job_id, "status": m.get("status", "unknown"), "kind": m.get("kind"), "manifest": m}
        return {}

    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute(
            "SELECT job_id, created_ts, updated_ts, status, kind, request_json, result_json, error_text FROM media_jobs WHERE job_id=? LIMIT 1",
            (job_id,),
        )
        row = cur.fetchone()
        if not row:
            return {}

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
    role: str
    filename: str
    path: str
    mime: str
    size_bytes: int
    sha256: str

def _probe_engines() -> Dict[str, bool]:
    caps: Dict[str, bool] = {}

    try:
        import SarahMemoryCanvasStudio as _cs  # noqa: F401
        caps["canvas_studio"] = True
    except Exception:
        caps["canvas_studio"] = False

    try:
        import SarahMemoryMusicGenerator as _mg  # noqa: F401
        caps["music_generator"] = True
    except Exception:
        caps["music_generator"] = False

    try:
        import SarahMemoryLyricsToSong as _lts  # noqa: F401
        caps["lyrics_to_song"] = True
    except Exception:
        caps["lyrics_to_song"] = False

    try:
        import SarahMemoryVideoEditorCore as _ve  # noqa: F401
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

    return RenderArtifact(role=role, filename=os.path.basename(p), path=p, mime=mime, size_bytes=size, sha256=sha)

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
    job_id = _sanitize_job_id(payload.get("job_id") or _safe_id("mediajob"))

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
    job_id = _sanitize_job_id(request.args.get("job_id") or "")
    if not job_id:
        return _err("missing_job_id", 400)

    job = _db_get_job(job_id)
    if job:
        m = _read_json(_manifest_path(job_id))
        if m:
            job["manifest"] = m
        return _ok(job=job)

    m = _read_json(_manifest_path(job_id))
    if not m:
        return _err("job_not_found", 404, job_id=job_id)
    return _ok(job={"job_id": job_id, "status": m.get("status", "unknown"), "kind": m.get("kind"), "manifest": m})

@bp.post("/api/media/job/render")
def media_job_render():
    """Render entrypoint for Creative Studio.

    Engine modes
    - If inline_b64 is provided, the server packages it.
    - If inline_b64 is missing, the server attempts a best-effort call into the relevant engine.
    """
    body = _body_bytes()
    if not _verify_auth(body):
        return _err("unauthorized", 401)

    payload = _j()
    kind = (payload.get("kind") or "generic").strip().lower()

    for k in ("prompt", "lyrics"):
        if k in payload and isinstance(payload[k], str) and len(payload[k]) > MAX_PROMPT_CHARS:
            payload[k] = payload[k][:MAX_PROMPT_CHARS]

    job_id = _sanitize_job_id(payload.get("job_id") or _safe_id("mediajob"))
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
        if kind in ("image", "img", "canvas"):
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

                try:
                    import SarahMemoryCanvasStudio as CS  # type: ignore

                    studio_cls = getattr(CS, "CanvasStudio", None)
                    if studio_cls is None:
                        raise RuntimeError("CanvasStudio_class_missing")
                    studio = studio_cls()

                    gen = getattr(studio, "generate_from_prompt", None)
                    exp = getattr(studio, "export_canvas", None)
                    if not callable(gen) or not callable(exp):
                        raise RuntimeError("CanvasStudio_missing_entrypoints")

                    canvas = gen(
                        prompt,
                        width=int((payload.get("width") or 1024)),
                        height=int((payload.get("height") or 1024)),
                        style=(payload.get("style") or "default"),
                        quality=(payload.get("quality") or "standard"),
                    )

                    if canvas is None:
                        raise RuntimeError("canvas_generation_returned_none")

                    ok = exp(canvas, out_path, format=fmt.upper(), quality=int(payload.get("image_quality") or 90), flatten=True)
                    if not ok:
                        raise RuntimeError("canvas_export_failed")

                    if not os.path.isfile(out_path):
                        guess = f"{out_path}.{fmt}"
                        if os.path.isfile(guess):
                            out_path = guess
                        else:
                            newest = None
                            newest_ts = 0.0
                            for fn in os.listdir(jd):
                                fp = os.path.join(jd, fn)
                                if os.path.isfile(fp):
                                    ts = os.path.getmtime(fp)
                                    if ts >= newest_ts:
                                        newest_ts = ts
                                        newest = fp
                            if newest:
                                out_path = newest

                except Exception as e:
                    raise RuntimeError(f"canvas_render_failed:{e}")

            artifacts.append(_artifact_from_path("image", out_path, _mime_from_ext(out_path)))

        elif kind in ("music", "audio"):
            out = payload.get("output") or {}
            fmt = (out.get("format") or "wav").strip().lower()
            if fmt not in ("wav", "mp3", "flac"):
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
                try:
                    import SarahMemoryMusicGenerator as MG  # type: ignore

                    studio_cls = getattr(MG, "MusicStudio", None)
                    if studio_cls is None:
                        raise RuntimeError("MusicStudio_class_missing")
                    studio = studio_cls()

                    m = payload.get("music") or {}
                    duration = float(m.get("duration") or payload.get("duration") or 15.0)
                    style = (m.get("style") or payload.get("style") or "default").strip()
                    tempo = int(m.get("tempo") or payload.get("tempo") or 120)

                    create_project = getattr(studio, "create_project", None)
                    generate_song = getattr(studio, "generate_song", None) or getattr(MG, "generate_song", None)
                    export_project = getattr(studio, "export_project", None) or getattr(studio, "export", None)

                    project = None
                    if callable(create_project):
                        project = create_project(f"Job {job_id}", tempo=tempo, key=(m.get("key") or "C"), time_signature=(4, 4))

                    if callable(generate_song):
                        song = generate_song(duration=duration, style=style, tempo=tempo)  # type: ignore[misc]
                    else:
                        song = None

                    if isinstance(song, (bytes, bytearray)):
                        _write_bytes(out_path, bytes(song))
                    elif callable(export_project):
                        export_project(project, out_path, format=fmt.upper(), quality=(m.get("quality") or "high"))  # type: ignore[misc]
                        if not os.path.isfile(out_path):
                            guess = f"{out_path}.{fmt}"
                            if os.path.isfile(guess):
                                out_path = guess
                    else:
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

        elif kind in ("video", "vid"):
            out = payload.get("output") or {}
            fmt = (out.get("format") or "mp4").strip().lower()
            if fmt not in ("mp4", "webm"):
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
                    inputs = v.get("inputs") or []
                    image_b64 = (v.get("image_b64") or "").strip()
                    audio_b64 = (v.get("audio_b64") or "").strip()

                    if image_b64:
                        img_path = os.path.join(jd, "input_image.png")
                        _write_bytes(img_path, _b64d(image_b64))
                        inputs = list(inputs) + [img_path]

                    aud_path = os.path.join(jd, "input_audio.wav")
                    if audio_b64:
                        _write_bytes(aud_path, _b64d(audio_b64))

                    create_project = getattr(editor, "create_project", None)
                    export_project = getattr(editor, "export_project", None)

                    if callable(create_project) and callable(export_project):
                        res = v.get("resolution") or [1920, 1080]
                        fps = int(v.get("fps") or 30)
                        proj = create_project(f"Job {job_id}", resolution=(int(res[0]), int(res[1])), fps=fps)

                        add_clip = getattr(proj, "add_clip", None)
                        if callable(add_clip):
                            t = 0.0
                            for p in inputs:
                                if isinstance(p, str) and os.path.isfile(p):
                                    add_clip(p, start_time=t)
                                    t += float(v.get("clip_spacing") or 0.0)

                        if audio_b64 and os.path.isfile(aud_path):
                            add_audio = getattr(proj, "add_audio_track", None)
                            if callable(add_audio):
                                add_audio(aud_path, volume=float(v.get("audio_volume") or 0.8))

                        export_project(proj, out_path, quality=(v.get("quality") or "high"), resolution=(v.get("export_resolution") or "1080p"))

                        if not os.path.isfile(out_path):
                            guess = f"{out_path}.{fmt}"
                            if os.path.isfile(guess):
                                out_path = guess
                    else:
                        raise RuntimeError("video_editor_missing_entrypoints")

                except Exception as e:
                    raise RuntimeError(f"video_render_failed:{e}")

            artifacts.append(_artifact_from_path("video", out_path, _mime_from_ext(out_path)))

        elif kind in ("hybrid", "compose", "multimodal"):
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

        artifact_list = [
            {
                "role": a.role,
                "filename": a.filename,
                "path": a.path,
                "mime": a.mime,
                "size_bytes": a.size_bytes,
                "sha256": a.sha256,
            }
            for a in artifacts
        ]

        manifest["artifacts"] = artifact_list
        manifest["status"] = "complete"
        manifest["updated_at"] = _iso()
        _write_json(mp, manifest)

        result = {"job_id": job_id, "status": "complete", "kind": kind, "job_dir": jd, "manifest": "manifest.json", "artifacts": artifact_list}
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
    if not _download_allowed():
        return _err("download_unauthorized", 401)

    job_id = _sanitize_job_id(request.args.get("job_id") or "")
    filename = (request.args.get("filename") or "").strip()
    if not job_id or not filename:
        return _err("missing_job_id_or_filename", 400)

    filename = _sanitize_filename(filename, "file.bin")
    path = os.path.join(_job_dir(job_id), filename)
    if not os.path.isfile(path):
        return _err("file_not_found", 404, path=path)

    return _send_file_compat(path, as_attachment=True, download_name=filename, mimetype=_mime_from_ext(filename))

@bp.get("/api/media/job/manifest")
def media_job_manifest():
    job_id = _sanitize_job_id(request.args.get("job_id") or "")
    if not job_id:
        return _err("missing_job_id", 400)

    mp = _manifest_path(job_id)
    if not os.path.isfile(mp):
        return _err("manifest_not_found", 404, job_id=job_id)

    if not _download_allowed():
        return _err("manifest_unauthorized", 401)

    return _send_file_compat(mp, as_attachment=False, download_name="manifest.json", mimetype="application/json")

# ---------------------------
# Initialization / Injection
# ---------------------------

def init_appmedia(
    connect_sqlite: Callable[..., Any],
    meta_db_path: str,
    api_key_auth_ok: Optional[Callable[[], bool]] = None,
    sign_ok: Optional[Callable[[bytes, str], bool]] = None,
) -> Blueprint:
    """Inject storage/auth helpers and return blueprint."""
    global _CONNECT_SQLITE, _META_DB, _API_KEY_AUTH_OK, _SIGN_OK
    _CONNECT_SQLITE = connect_sqlite
    _META_DB = meta_db_path
    _API_KEY_AUTH_OK = api_key_auth_ok
    _SIGN_OK = sign_ok

    _ensure_tables()
    return bp

def init_app(
    app,
    connect_sqlite: Callable[..., Any],
    meta_db_path: str,
    api_key_auth_ok: Optional[Callable[[], bool]] = None,
    sign_ok: Optional[Callable[[bytes, str], bool]] = None,
) -> None:
    """Preferred: inject + register blueprint into Flask app (single-call)."""
    if "appmedia_v800" in getattr(app, "blueprints", {}):
        return

    init_appmedia(connect_sqlite, meta_db_path, api_key_auth_ok=api_key_auth_ok, sign_ok=sign_ok)
    _ensure_tables()
    app.register_blueprint(bp)
