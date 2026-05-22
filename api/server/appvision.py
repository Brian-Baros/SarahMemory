"""--==The SarahMemory Project==--
File: /api/server/appvision.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.

Governed Vision API Bridge
==========================
Purpose:
- Backend-owned vision policy, camera/device witness, and frame analysis bridge.
- Keeps frontend authority limited to ON/OFF and presentation buttons.
- Uses SarahMemoryMSDC.py for body/device/driver mapping.
- Uses SOBJE and FacialRecognition as frame interpreters, not hardware owners.
- Does not mutate SarahMemoryGlobals.py.
"""
from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "api_bridge"
# CATEGORY = "governed_vision_bridge"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "api_server"
# API_DOMAIN = "vision"
# HARDWARE_DOMAIN = "camera_usb_webcam"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "vision_bridge"
# FAMILY = "vision"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Backend-owned vision policy and bridge. MSDC maps device/body. SOBJE/FacialRecognition interpret frames. Frontend remains ON/OFF only."
# --- SARAHMETA END ---

import base64
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from flask import Blueprint, Response, request

bp = Blueprint("appvision_v800", __name__)

_CONNECT_SQLITE: Optional[Callable[..., Any]] = None
_META_DB: Optional[str] = None
_API_KEY_AUTH_OK: Optional[Callable[[], bool]] = None
_SIGN_OK: Optional[Callable[[bytes, str], bool]] = None
_ROUTES_REGISTERED = False

try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None

try:
    import cv2 as _cv2  # type: ignore
except Exception:
    _cv2 = None

try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None  # type: ignore

try:
    import SarahMemoryMSDC as _MSDC  # type: ignore
except Exception:
    _MSDC = None

try:
    import SarahMemorySOBJE as _SOBJE  # type: ignore
except Exception:
    _SOBJE = None

try:
    import SarahMemoryFacialRecognition as _FaceRec  # type: ignore
except Exception:
    _FaceRec = None


DEFAULT_POLICY: Dict[str, Any] = {
    "enabled": True,
    "accept_frontend_frames": True,
    "backend_controls_fps": True,
    "max_fps": 2,
    "max_width": 640,
    "max_height": 360,
    "jpeg_quality": 0.7,
    "frame_ttl_seconds": 10,
    "max_frame_chars": 1800000,
    "learning_default": "off",
    "identity_learning_requires_user_approval": True,
    "frontend_authority": ["camera_on_off", "preview_show_hide", "submit"],
    "backend_authority": ["frame_acceptance", "max_fps", "max_resolution", "analysis", "learning_gate", "driver_use"],
}


def _data_dir() -> Path:
    try:
        return Path(str(getattr(config, "DATA_DIR"))).expanduser().resolve()  # type: ignore[arg-type]
    except Exception:
        return (Path.cwd() / "data").resolve()


def _settings_dir() -> Path:
    """Return the runtime settings directory for generated policy JSON."""
    try:
        return Path(str(getattr(config, "SETTINGS_DIR"))).expanduser().resolve()  # type: ignore[arg-type]
    except Exception:
        return (_data_dir() / "settings").resolve()


def _legacy_registry_dir() -> Path:
    return (_data_dir() / "registry").resolve()


def _migrate_legacy_json_once(primary: Path, legacy: Path) -> Path:
    """Copy legacy registry JSON into data/settings once; future writes stay primary."""
    try:
        if (not primary.exists()) and legacy.exists() and legacy.is_file():
            primary.parent.mkdir(parents=True, exist_ok=True)
            primary.write_text(legacy.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass
    return primary


def _policy_path() -> Path:
    return _migrate_legacy_json_once(
        _settings_dir() / "vision_policy.json",
        _legacy_registry_dir() / "vision_policy.json",
    )


def _json_safe(value: Any, depth: int = 0) -> Any:
    if depth > 12:
        return str(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v, depth + 1) for k, v in value.items() if not callable(v)}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v, depth + 1) for v in list(value)]
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _response(payload: Dict[str, Any], status: int = 200):
    try:
        body = json.dumps(_json_safe(payload), ensure_ascii=False, allow_nan=False)
    except Exception as exc:
        body = json.dumps({"ok": False, "error": "json_serialization_failed", "detail": str(exc)})
        status = 500
    resp = Response(body, status=status, mimetype="application/json")
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Sarah-Signature, X-Session-Id"
    return resp


def _body_bytes() -> bytes:
    try:
        return request.get_data(cache=True) or b""
    except Exception:
        return b""


def _verify_auth() -> bool:
    sig = (request.headers.get("X-Sarah-Signature") or "").strip()
    if sig and _SIGN_OK:
        try:
            return bool(_SIGN_OK(_body_bytes(), sig))
        except Exception:
            return False
    if _API_KEY_AUTH_OK:
        try:
            return bool(_API_KEY_AUTH_OK())
        except Exception:
            return False
    return True


def _payload() -> Dict[str, Any]:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _read_policy() -> Dict[str, Any]:
    policy = dict(DEFAULT_POLICY)
    path = _policy_path()
    try:
        if path.exists():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                policy.update(loaded)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass
    return policy


def _write_policy(policy: Dict[str, Any]) -> bool:
    try:
        path = _policy_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(policy, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception:
        return False


def _extract_image_value(data: Dict[str, Any]) -> Optional[str]:
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    for key in ("frame", "image", "image_data", "imageData", "image_base64", "imageBase64", "data_url", "dataUrl", "latest_frame", "vision_frame"):
        val = data.get(key) if key in data else meta.get(key)
        if isinstance(val, dict):
            nested = _extract_image_value(val)
            if nested:
                return nested
        elif isinstance(val, str) and val.strip():
            return val.strip()
    images = data.get("images") if isinstance(data.get("images"), list) else meta.get("images") if isinstance(meta.get("images"), list) else []
    if images:
        first = images[0]
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            return _extract_image_value(first)
    return None


def _decode_frame(data: Dict[str, Any]):
    if _np is None or _cv2 is None:
        return None, "cv2_or_numpy_unavailable"
    raw = _extract_image_value(data)
    if not raw:
        return None, "no_frame_supplied"
    try:
        s = raw.strip()
        if s.startswith("data:image") and "," in s:
            s = s.split(",", 1)[1]
        blob = base64.b64decode(s, validate=False)
        arr = _np.frombuffer(blob, dtype=_np.uint8)
        frame = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
        if frame is None:
            return None, "decode_failed"
        return frame, "ok"
    except Exception as exc:
        return None, f"decode_exception:{exc}"


def _encode_frame_jpeg(frame: Any, quality: float = 0.7) -> Dict[str, Any]:
    if _cv2 is None or frame is None:
        return {"ok": False, "error": "cv2_or_frame_unavailable"}
    try:
        q = int(max(1, min(95, float(quality) * 100.0)))
        ok, buf = _cv2.imencode(".jpg", frame, [int(_cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return {"ok": False, "error": "jpeg_encode_failed"}
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        return {"ok": True, "image_b64": b64, "data_url": "data:image/jpeg;base64," + b64, "mime": "image/jpeg"}
    except Exception as exc:
        return {"ok": False, "error": "jpeg_encode_exception", "detail": str(exc)}


def _frame_shape(frame: Any) -> Optional[list]:
    try:
        return [int(x) for x in list(frame.shape)]
    except Exception:
        return None


def _analyze_frame(question: str, frame: Any, learning_allowed: bool = False) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": True,
        "question": question,
        "learning_allowed": bool(learning_allowed),
        "frame_shape": _frame_shape(frame),
        "sobje": None,
        "facial": None,
        "errors": [],
    }
    if frame is None:
        out["ok"] = False
        out["errors"].append("frame_missing")
        return out
    if _SOBJE is not None and hasattr(_SOBJE, "answer_visual_question"):
        try:
            payload = {
                "question": question,
                "query": question,
                "learning_allowed": bool(learning_allowed),
                "helper_payload": {"vision": {"learning_allowed": bool(learning_allowed)}},
            }
            out["sobje"] = _SOBJE.answer_visual_question(payload, frame)  # type: ignore[attr-defined]
        except Exception as exc:
            out["errors"].append(f"sobje_error:{exc}")
    else:
        out["errors"].append("sobje_unavailable")
    if _FaceRec is not None:
        try:
            if hasattr(_FaceRec, "analyze_face_frame"):
                out["facial"] = _FaceRec.analyze_face_frame(frame, allow_identity=False, allow_learning=bool(learning_allowed))  # type: ignore[attr-defined]
            elif hasattr(_FaceRec, "detect_faces_dnn"):
                faces = _FaceRec.detect_faces_dnn(frame)  # type: ignore[attr-defined]
                out["facial"] = {"ok": True, "face_count": len(faces) if isinstance(faces, (list, tuple)) else 0, "faces": faces}
        except Exception as exc:
            out["errors"].append(f"facial_error:{exc}")
    else:
        out["errors"].append("facial_recognition_unavailable")
    return out


@bp.get("/api/vision/policy")
def api_vision_policy():
    return _response({"ok": True, "policy": _read_policy(), "source": "appvision"})


@bp.post("/api/vision/policy")
def api_vision_policy_update():
    if not _verify_auth():
        return _response({"ok": False, "error": "auth_failed"}, 401)
    data = _payload()
    current = _read_policy()
    allowed = {"enabled", "accept_frontend_frames", "max_fps", "max_width", "max_height", "jpeg_quality", "frame_ttl_seconds", "max_frame_chars", "learning_default"}
    for key, value in data.items():
        if key in allowed:
            current[key] = value
    _write_policy(current)
    return _response({"ok": True, "policy": current, "source": "appvision"})


@bp.get("/api/vision/devices")
def api_vision_devices():
    """Return camera/body-map device status without blocking on live probes.

    Default behavior is intentionally read-only and non-invasive:
    - Build/persist the MSDC body map.
    - Report whether the camera driver package is present/manifest-valid.
    - Do NOT call driver discovery/probe by default because some camera
      backends can block while enumerating or probing hardware.

    Optional explicit probes:
      /api/vision/devices?discover=1
      /api/vision/devices?probe=1

    This keeps normal status checks fast while preserving a manual diagnostic path.
    """
    if _MSDC is None:
        return _response({"ok": False, "error": "SarahMemoryMSDC_unavailable"}, 503)
    try:
        body_map = _MSDC.msdc_map_body(persist=True)  # type: ignore[attr-defined]
        include_discover = str(request.args.get("discover") or "").strip().lower() in ("1", "true", "yes", "on")
        include_probe = str(request.args.get("probe") or "").strip().lower() in ("1", "true", "yes", "on")
        payload: Dict[str, Any] = {
            "ok": True,
            "body_map": body_map,
            "discover": {"ok": False, "skipped": True, "reason": "not_requested", "hint": "use ?discover=1 for explicit hardware enumeration"},
            "probe": {"ok": False, "skipped": True, "reason": "not_requested", "hint": "use ?probe=1 for explicit capability probe"},
            "source": "appvision.msdc",
            "non_blocking_default": True,
        }
        if include_discover:
            payload["discover"] = _MSDC.msdc_camera_discover()  # type: ignore[attr-defined]
        if include_probe:
            payload["probe"] = _MSDC.msdc_camera_probe()  # type: ignore[attr-defined]
        return _response(payload)
    except Exception as exc:
        return _response({"ok": False, "error": "vision_devices_failed", "detail": str(exc)}, 500)


@bp.get("/api/vision/court-witness")
def api_vision_court_witness():
    if _MSDC is None:
        return _response({"ok": False, "error": "SarahMemoryMSDC_unavailable"}, 503)
    include_probe = str(request.args.get("probe") or "").strip().lower() in ("1", "true", "yes")
    try:
        witness = _MSDC.msdc_court_witness(body_part="eyes", include_probe=include_probe)  # type: ignore[attr-defined]
        return _response({"ok": True, "witness": witness, "source": "appvision.msdc"})
    except Exception as exc:
        return _response({"ok": False, "error": "court_witness_failed", "detail": str(exc)}, 500)


@bp.post("/api/vision/local/open")
def api_vision_local_open():
    if _MSDC is None:
        return _response({"ok": False, "error": "SarahMemoryMSDC_unavailable"}, 503)
    data = _payload()
    user_authorized = bool(data.get("user_authorized") or data.get("user_confirmed") or data.get("confirm"))
    result = _MSDC.msdc_camera_open(user_authorized=user_authorized, payload=data)  # type: ignore[attr-defined]
    return _response(result, 200 if result.get("ok") else 403)


@bp.post("/api/vision/local/close")
def api_vision_local_close():
    if _MSDC is None:
        return _response({"ok": False, "error": "SarahMemoryMSDC_unavailable"}, 503)
    data = _payload()
    result = _MSDC.msdc_camera_close(payload=data)  # type: ignore[attr-defined]
    return _response(result, 200 if result.get("ok") else 400)


@bp.post("/api/vision/local/capture")
def api_vision_local_capture():
    if _MSDC is None:
        return _response({"ok": False, "error": "SarahMemoryMSDC_unavailable"}, 503)
    data = _payload()
    user_authorized = bool(data.get("user_authorized") or data.get("user_confirmed") or data.get("confirm"))
    result = _MSDC.msdc_camera_capture_b64(user_authorized=user_authorized, payload=data)  # type: ignore[attr-defined]
    return _response(result, 200 if result.get("ok") else 403)


@bp.post("/api/vision/analyze")
def api_vision_analyze():
    policy = _read_policy()
    if not bool(policy.get("enabled", True)):
        return _response({"ok": False, "error": "vision_disabled_by_backend_policy", "policy": policy}, 403)
    data = _payload()
    question = str(data.get("question") or data.get("text") or "What do you see?").strip() or "What do you see?"
    learning_allowed = bool(data.get("learning_allowed")) or str(policy.get("learning_default") or "off").lower() in ("on", "true", "1", "yes")
    frame, status = _decode_frame(data)
    if frame is None and bool(data.get("use_backend_capture")):
        if _MSDC is None:
            return _response({"ok": False, "error": "no_frontend_frame_and_msdc_unavailable", "decode_status": status}, 400)
        cap = _MSDC.msdc_camera_capture_b64(user_authorized=bool(data.get("user_authorized") or data.get("user_confirmed") or data.get("confirm")), payload=data)  # type: ignore[attr-defined]
        if not cap.get("ok"):
            return _response({"ok": False, "error": "backend_capture_failed", "capture": cap, "decode_status": status}, 403)
        frame, status = _decode_frame({"imageBase64": cap.get("data_url") or cap.get("image_b64")})
    if frame is None:
        return _response({"ok": False, "error": "no_decodable_frame", "decode_status": status}, 400)
    analysis = _analyze_frame(question, frame, learning_allowed=learning_allowed)
    return _response({"ok": bool(analysis.get("ok")), "analysis": analysis, "policy": policy, "source": "appvision.analysis"})


@bp.post("/api/vision/learning/approve")
def api_vision_learning_approve():
    # Placeholder approval surface; actual identity/object persistence stays in the governed core.
    data = _payload()
    return _response({
        "ok": True,
        "status": "approval_recorded_for_next_governed_learning_pass",
        "learning_allowed": bool(data.get("learning_allowed") or data.get("approve")),
        "identity_learning_requires_user_approval": True,
        "source": "appvision.learning_gate",
    })


def init_app(app, connect_sqlite=None, meta_db=None, api_key_auth_ok=None, sign_ok=None):
    global _CONNECT_SQLITE, _META_DB, _API_KEY_AUTH_OK, _SIGN_OK, _ROUTES_REGISTERED
    _CONNECT_SQLITE = connect_sqlite
    _META_DB = meta_db
    _API_KEY_AUTH_OK = api_key_auth_ok
    _SIGN_OK = sign_ok
    if _ROUTES_REGISTERED:
        return app
    try:
        app.register_blueprint(bp)
        _ROUTES_REGISTERED = True
    except ValueError:
        # Blueprint already mounted during dev reload.
        _ROUTES_REGISTERED = True
    return app


def apply(app):
    return init_app(app)
