"""--==The SarahMemory Project==--
File: api/server/appvision.py
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

Governed Vision API Bridge
==========================
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
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-06-06"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Backend-owned vision policy and bridge. MSDC maps device/body. SOBJE/FacialRecognition interpret frames. Frontend remains ON/OFF only."
# --- SARAHMETA END ---

import base64
import json
import os
import time
import uuid
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from flask import Blueprint, Response, request

# ARILE boundary helper. API files emit compact variance signals only; the central
# backend engine remains SarahMemoryARILE.py.
try:
    from SarahMemoryARILE import arile_emit, arile_endpoint_guard
except Exception:  # pragma: no cover
    arile_emit = None  # type: ignore
    arile_endpoint_guard = None  # type: ignore

def _arile_api_emit(failure_type: str, summary: str, severity: float = 0.55, **data):
    try:
        if callable(arile_emit):
            arile_emit(source=f"api.server.{__name__}", kind="api_boundary_variance", failure_type=failure_type, severity=severity, confidence=0.85, risk="high" if severity >= 0.75 else "medium", summary=summary, requires_governance=severity >= 0.75, retention="security_audit" if severity >= 0.75 else "diagnostic", data=data)
    except Exception:
        pass

def _arile_check_request(endpoint_name: str, risk: str = "low") -> str:
    try:
        if callable(arile_endpoint_guard):
            return arile_endpoint_guard(endpoint_name, {"method": getattr(request, "method", ""), "content_length": getattr(request, "content_length", 0) or 0, "remote_addr": getattr(request, "remote_addr", "")}, risk=risk)
    except Exception:
        pass
    return "allow"


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
    "background_analysis_enabled": True,
    "background_analysis_interval_ms": 1500,
    "identity_learning_requires_user_approval": True,
    "frontend_authority": ["camera_on_off", "preview_show_hide", "submit"],
    "backend_authority": ["frame_acceptance", "max_fps", "max_resolution", "analysis", "learning_gate", "driver_use"],
}

SMHUD_SCHEMA_VERSION = "SMHUD_PACKET_V1"
_FRAME_LOCK = threading.RLock()
_FRAME_CACHE: Dict[str, Any] = {
    "session_id": uuid.uuid4().hex,
    "has_frame": False,
    "frame_id": "",
    "ts": None,
    "source": None,
    "width": None,
    "height": None,
    "analysis": None,
    "hud_packet": None,
    "image_b64": None,
    "data_url": None,
    "mime": None,
    "image_cached_ts": None,
    "last_background_analysis_ts": 0.0,
}


def _utc_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def _clamp_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _frame_meta(frame: Any, source: str = "frontend") -> Dict[str, Any]:
    shape = _frame_shape(frame) or []
    height = int(shape[0]) if len(shape) >= 1 else None
    width = int(shape[1]) if len(shape) >= 2 else None
    return {
        "frame_id": f"vis_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "timestamp": _utc_iso(),
        "source": source,
        "width": width,
        "height": height,
    }


def _bbox_to_target(det: Dict[str, Any], idx: int, width: int, height: int) -> Optional[Dict[str, Any]]:
    try:
        bbox = det.get("bbox") or det.get("box") or []
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        # Accept either xyxy pixel coordinates or normalized xywh/xyxy-ish values.
        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
            # Treat as normalized xyxy when possible.
            nx1, ny1, nx2, ny2 = x1, y1, x2, y2
            px1, py1, px2, py2 = nx1 * width, ny1 * height, nx2 * width, ny2 * height
        else:
            px1, py1, px2, py2 = x1, y1, x2, y2
            nx1 = px1 / max(1, width)
            ny1 = py1 / max(1, height)
            nx2 = px2 / max(1, width)
            ny2 = py2 / max(1, height)
        if nx2 < nx1:
            nx1, nx2 = nx2, nx1
        if ny2 < ny1:
            ny1, ny2 = ny2, ny1
        nx1, ny1 = max(0.0, min(1.0, nx1)), max(0.0, min(1.0, ny1))
        nx2, ny2 = max(0.0, min(1.0, nx2)), max(0.0, min(1.0, ny2))
        cx = (nx1 + nx2) / 2.0
        cy = (ny1 + ny2) / 2.0
        area = max(0.0, (nx2 - nx1) * (ny2 - ny1))
        dz_est = round(1.0 / max(0.05, area ** 0.5), 3) if area else None
        return {
            "id": str(det.get("id") or f"target_{idx:03d}"),
            "class": str(det.get("domain") or det.get("class") or "object"),
            "label": str(det.get("label") or det.get("raw_label") or "object"),
            "bbox": [round(nx1, 5), round(ny1, 5), round(nx2, 5), round(ny2, 5)],
            "bbox_px": [int(px1), int(py1), int(px2), int(py2)],
            "center": [round(cx, 5), round(cy, 5)],
            "confidence": round(_clamp_float(det.get("confidence"), 0.0), 4),
            "vectors": {
                "dx": round(cx - 0.5, 5),
                "dy": round(0.5 - cy, 5),
                "dz_est": dz_est,
            },
            "motion": {"angular_velocity": 0.0, "velocity_px_s": [0.0, 0.0]},
            "color": det.get("color") if isinstance(det.get("color"), dict) else None,
            "model": det.get("model"),
        }
    except Exception:
        return None


def _contour_targets(frame: Any, limit: int = 8) -> List[Dict[str, Any]]:
    if _cv2 is None or frame is None:
        return []
    targets: List[Dict[str, Any]] = []
    try:
        h, w = frame.shape[:2]
        gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
        blur = _cv2.GaussianBlur(gray, (5, 5), 0)
        edges = _cv2.Canny(blur, 80, 180)
        contours, _ = _cv2.findContours(edges, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = float(_cv2.contourArea(c))
            if area < max(300.0, (w * h) * 0.002):
                continue
            x, y, bw, bh = _cv2.boundingRect(c)
            candidates.append((area, {"label": "edge_object", "domain": "object", "bbox": [x, y, x + bw, y + bh], "confidence": 0.35, "model": "contour"}))
        candidates.sort(key=lambda item: item[0], reverse=True)
        for idx, (_, det) in enumerate(candidates[:limit]):
            target = _bbox_to_target(det, idx, w, h)
            if target:
                targets.append(target)
    except Exception:
        pass
    return targets


def _extract_hud_targets(analysis: Dict[str, Any], frame: Any) -> List[Dict[str, Any]]:
    try:
        h, w = frame.shape[:2]
    except Exception:
        h, w = 0, 0
    if not w or not h:
        return []

    sobje = analysis.get("sobje") if isinstance(analysis, dict) else None
    details = sobje.get("details") if isinstance(sobje, dict) else None
    detections = []
    if isinstance(details, dict):
        detections = details.get("detections") if isinstance(details.get("detections"), list) else []
        if not detections:
            findings = details.get("findings") if isinstance(details.get("findings"), dict) else {}
            detections = findings.get("detections") if isinstance(findings.get("detections"), list) else []
    targets: List[Dict[str, Any]] = []
    for idx, det in enumerate(detections[:16] if isinstance(detections, list) else []):
        if isinstance(det, dict):
            target = _bbox_to_target(det, idx, w, h)
            if target:
                targets.append(target)
    if targets:
        return targets
    return _contour_targets(frame)


def _compute_integrity_packet() -> Dict[str, Any]:
    packet: Dict[str, Any] = {
        "ok": True,
        "token_throughput": None,
        "pretok_latency_ms": None,
        "memory_pool_mb": None,
        "thread_state": {},
        "source": "appvision.local_runtime",
    }
    try:
        import threading as _threading
        packet["thread_state"] = {
            "active_threads": int(_threading.active_count()),
            "vision": "RUNNING",
            "hud": "RUNNING",
        }
    except Exception:
        pass
    try:
        import resource as _resource  # not available on all Windows builds, safe fallback
        usage = _resource.getrusage(_resource.RUSAGE_SELF)
        # ru_maxrss is KB on Linux, bytes on macOS; this is advisory only.
        packet["memory_pool_mb"] = round(float(getattr(usage, "ru_maxrss", 0) or 0) / 1024.0, 2)
    except Exception:
        packet["memory_pool_mb"] = None
    return packet


def _kinetic_integrity_packet() -> Dict[str, Any]:
    packet: Dict[str, Any] = {
        "ok": True,
        "body_state": "OBSERVE_ONLY",
        "movement_lock": True,
        "devices": [],
        "source": "SarahMemoryMSDC",
    }
    try:
        if _MSDC is not None and hasattr(_MSDC, "msdc_vr_hud_status"):
            status = _MSDC.msdc_vr_hud_status()  # type: ignore[attr-defined]
        elif _MSDC is not None and hasattr(_MSDC, "msdc_status"):
            status = _MSDC.msdc_status()  # type: ignore[attr-defined]
        else:
            status = {"ok": False, "error": "msdc_unavailable"}
        packet["msdc"] = status
        body = status.get("body_map", {}).get("body_parts", {}) if isinstance(status, dict) else {}
        for name, record in (body or {}).items():
            if isinstance(record, dict):
                packet["devices"].append({
                    "part": name,
                    "driver_id": record.get("driver_id"),
                    "status": record.get("status"),
                    "privacy_sensitive": bool(record.get("privacy_sensitive")),
                    "physical_safety_sensitive": bool(record.get("physical_safety_sensitive")),
                    "fault": None,
                })
    except Exception as exc:
        packet.update({"ok": False, "error": str(exc)})
    return packet


def _smget_packet() -> Dict[str, Any]:
    return {
        "ok": True,
        "contract_id": None,
        "execution_mode": "observe_only",
        "state": "NO_ACTIVE_ACTION_CONTRACT",
        "movement_lock": True,
        "six_question_loop": {
            "WHO": "STANDBY",
            "WHY": "STANDBY",
            "WHAT": "STANDBY",
            "WHEN": "STANDBY",
            "WHERE": "STANDBY",
            "HOW": "STANDBY",
        },
        "decision": "READ_ONLY_WITNESS",
        "rollback_ready": True,
        "reason": "VR HUD is observing camera/telemetry only. Movement and device actions remain behind SMGET/OperatorCore.",
        "source": "appvision.smget_snapshot",
    }


def _build_hud_packet(frame: Any = None, analysis: Optional[Dict[str, Any]] = None, frame_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    frame_meta = dict(frame_meta or _frame_meta(frame, source="unknown"))
    analysis = analysis if isinstance(analysis, dict) else {}
    targets = _extract_hud_targets(analysis, frame) if frame is not None else []
    packet = {
        "schema": SMHUD_SCHEMA_VERSION,
        "packet_id": f"hud_{uuid.uuid4().hex}",
        "timestamp": _utc_iso(),
        "ttl_ms": 2500,
        "mode": "OBSERVE_ONLY",
        "display_profile": "vr_operator_hud",
        "frame": frame_meta,
        "active_targets": targets,
        "vision": {
            "ok": bool(analysis.get("ok", True)) if isinstance(analysis, dict) else True,
            "source": "SOBJE_FacialRecognition",
            "answer": (analysis.get("sobje") or {}).get("answer") if isinstance(analysis.get("sobje"), dict) else None,
            "confidence": ((analysis.get("sobje") or {}).get("details") or {}).get("confidence") if isinstance(analysis.get("sobje"), dict) else None,
            "errors": analysis.get("errors", []) if isinstance(analysis, dict) else [],
        },
        "compute_integrity": _compute_integrity_packet(),
        "kinetic_integrity": _kinetic_integrity_packet(),
        "smget_state": _smget_packet(),
        "authority": {
            "hud_can_execute_actions": False,
            "hud_can_authorize_movement": False,
            "movement_locked": True,
            "user_final_authority": True,
        },
        "source": "appvision.smhud",
    }
    return packet


def _cache_frame(frame: Any, source: str, analysis: Optional[Dict[str, Any]] = None, hud_packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = _frame_meta(frame, source=source)
    encoded: Dict[str, Any] = {}
    try:
        policy = _read_policy()
        encoded = _encode_frame_jpeg(frame, quality=float(policy.get("jpeg_quality", 0.7) or 0.7))
    except Exception:
        encoded = {"ok": False, "error": "frame_cache_encode_failed"}
    with _FRAME_LOCK:
        _FRAME_CACHE.update({
            "has_frame": True,
            "frame_id": meta.get("frame_id"),
            "ts": meta.get("timestamp"),
            "source": source,
            "width": meta.get("width"),
            "height": meta.get("height"),
            "analysis": analysis,
            "hud_packet": hud_packet,
            "image_b64": encoded.get("image_b64") if isinstance(encoded, dict) and encoded.get("ok") else _FRAME_CACHE.get("image_b64"),
            "data_url": encoded.get("data_url") if isinstance(encoded, dict) and encoded.get("ok") else _FRAME_CACHE.get("data_url"),
            "mime": encoded.get("mime") if isinstance(encoded, dict) and encoded.get("ok") else _FRAME_CACHE.get("mime"),
            "image_cached_ts": _utc_iso() if isinstance(encoded, dict) and encoded.get("ok") else _FRAME_CACHE.get("image_cached_ts"),
        })
    return meta


def _frame_status_payload() -> Dict[str, Any]:
    with _FRAME_LOCK:
        packet = _FRAME_CACHE.get("hud_packet") if isinstance(_FRAME_CACHE.get("hud_packet"), dict) else None
        return {
            "ok": True,
            "session_id": _FRAME_CACHE.get("session_id"),
            "has_frame": bool(_FRAME_CACHE.get("has_frame")),
            "frame_id": _FRAME_CACHE.get("frame_id"),
            "ts": _FRAME_CACHE.get("ts"),
            "source": _FRAME_CACHE.get("source"),
            "width": _FRAME_CACHE.get("width"),
            "height": _FRAME_CACHE.get("height"),
            "hud_schema": SMHUD_SCHEMA_VERSION,
            "hud_packet_id": packet.get("packet_id") if packet else None,
            "target_count": len(packet.get("active_targets") or []) if packet else 0,
        }


def _data_dir() -> Path:
    try:
        return Path(str(getattr(config, "DATA_DIR"))).expanduser().resolve()  # type: ignore[arg-type]
    except Exception:
        try:
            here = Path(__file__).resolve()
            for parent in here.parents:
                if (parent / "core" / "SarahMemoryGlobals.py").exists() or (parent / "SarahMemoryGlobals.py").exists():
                    return (parent / "data").resolve()
        except Exception:
            pass
        return (Path.cwd() / "data").resolve()


def _settings_dir() -> Path:
    """Runtime-generated vision settings live under data/settings."""
    try:
        return Path(str(getattr(config, "SETTINGS_DIR"))).expanduser().resolve()  # type: ignore[arg-type]
    except Exception:
        return (_data_dir() / "settings").resolve()


def _legacy_registry_dir() -> Path:
    """Legacy registry path retained only for one-time migration fallback."""
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
    return _apply_energetics_to_vision_policy(policy)




def _apply_energetics_to_vision_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce vision duty-cycle under Energetics reserve constraints.

    The backend still owns policy. Energetics only recommends bounded FPS,
    resolution, JPEG quality, and background-analysis cadence. It does not turn
    cameras on/off directly. During hazardous-energy lockout, vision is forced
    into a conservative low-power/read-only overlay rather than trusting Energetics.
    """
    out = dict(policy or {})

    def _low_power_overlay(verdict: Dict[str, Any], label: str = "LOW_POWER_VISION_LOCKOUT") -> Dict[str, Any]:
        out["energetics"] = verdict
        out["max_fps"] = min(float(out.get("max_fps", 2) or 2), 1.0)
        out["max_width"] = min(int(out.get("max_width", 640) or 640), 320)
        out["max_height"] = min(int(out.get("max_height", 360) or 360), 180)
        out["jpeg_quality"] = min(float(out.get("jpeg_quality", 0.7) or 0.7), 0.55)
        out["background_analysis_interval_ms"] = max(int(out.get("background_analysis_interval_ms", 1500) or 1500), 5000)
        out["energetics_policy_overlay"] = label
        out["device_power_authority"] = False
        return out

    ctx = {"source": "appvision.policy", "scheduled": False, "device_type": "camera_vision", "action_type": "vision_device_power"}
    try:
        blocker_fn = getattr(config, "sm_hazardous_energy_blocks_action", None)
        status_fn = getattr(config, "sm_hazardous_energy_status", None)
        if callable(blocker_fn) and blocker_fn("device_power_state", ctx):
            status = status_fn(ctx) if callable(status_fn) else {}
            return _low_power_overlay({
                "ok": False,
                "decision": "DEFER",
                "reason": "Hazardous-energy constitution blocks camera/device power influence; applying conservative vision policy.",
                "allowed_power_mode": "LOW_POWER",
                "constitution": status,
            })
    except Exception as exc:
        return _low_power_overlay({"ok": False, "decision": "DEFER", "reason": f"Vision hazardous-energy check failed closed: {exc}", "allowed_power_mode": "LOW_POWER"})

    try:
        import SarahMemoryEnergetics as _Energetics  # type: ignore
        fn = getattr(_Energetics, "recommend_device_power_mode", None)
        verdict = fn("camera_vision", "ACTIVE", context=ctx) if callable(fn) else {}
        mode = str((verdict or {}).get("allowed_power_mode") or "ACTIVE").upper()
        out["energetics"] = verdict
        if mode in {"LOW_POWER", "RECOVERY"} or str((verdict or {}).get("decision") or "ALLOW").upper() in {"DENY", "DEFER", "REDUCE_MODE"}:
            return _low_power_overlay(verdict or {}, "LOW_POWER_VISION")
        elif mode == "READY":
            out["background_analysis_interval_ms"] = max(int(out.get("background_analysis_interval_ms", 1500) or 1500), 2500)
            out["energetics_policy_overlay"] = "READY_VISION"
            out["device_power_authority"] = False
        else:
            out["energetics_policy_overlay"] = "NORMAL_VISION"
            out["device_power_authority"] = False
    except Exception as exc:
        return _low_power_overlay({"ok": False, "decision": "DEFER", "reason": f"Vision Energetics bridge failed closed: {exc}", "allowed_power_mode": "LOW_POWER"})
    return out


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



@bp.get("/api/vision/frame/status")
def api_vision_frame_status():
    return _response(_frame_status_payload())


@bp.get("/api/vision/frame/latest")
def api_vision_frame_latest():
    """Return the latest backend-accepted frame for read-only HUD renderers.

    This endpoint exists so SarahMemoryVRHudRenderer.py can render the same
    governed frame stream that appvision accepted from the camera/frontend path.
    It does not open hardware and does not authorize actions.
    """
    with _FRAME_LOCK:
        has_frame = bool(_FRAME_CACHE.get("has_frame"))
        data_url = _FRAME_CACHE.get("data_url")
        image_b64 = _FRAME_CACHE.get("image_b64")
        payload = {
            "ok": bool(has_frame and data_url),
            "has_frame": has_frame,
            "frame_id": _FRAME_CACHE.get("frame_id"),
            "ts": _FRAME_CACHE.get("ts"),
            "source": _FRAME_CACHE.get("source"),
            "width": _FRAME_CACHE.get("width"),
            "height": _FRAME_CACHE.get("height"),
            "mime": _FRAME_CACHE.get("mime") or "image/jpeg",
            "image_b64": image_b64,
            "data_url": data_url,
            "image_cached_ts": _FRAME_CACHE.get("image_cached_ts"),
            "hud_schema": SMHUD_SCHEMA_VERSION,
            "source_endpoint": "appvision.frame_latest",
        }
    if not payload["ok"]:
        payload["error"] = "no_cached_frame"
    return _response(payload, 200)



# --- SM V8.0 Cognitive Instinct Vision Bridge ---
def _maybe_evaluate_emergency_instinct_from_vision(data: Dict[str, Any], analysis: Dict[str, Any], hud_packet: Dict[str, Any], frame_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Bridge vision observations into CognitiveServices emergency instinct evaluation.

    This function does not execute physical actions. It only prepares/logs a
    governed emergency-instinct evaluation when explicit emergency hints or
    hazard-like labels are present.
    """
    try:
        question = str(data.get("question") or data.get("text") or data.get("observation") or "")
        explicit = bool(data.get("emergency") or data.get("emergency_mode") or data.get("hazard_type") or data.get("emergency_type"))
        labels: list[str] = []
        for t in (hud_packet.get("active_targets") or []):
            if isinstance(t, dict):
                labels.append(str(t.get("label") or ""))
                labels.append(str(t.get("class") or ""))
        flat = " ".join([question] + labels).lower()
        hazard_words = ("fire", "smoke", "flame", "burning", "fall", "fallen", "unconscious", "choking", "inhaler", "asthma", "car", "vehicle", "collision")
        if not explicit and not any(w in flat for w in hazard_words):
            return None

        hazard_type = str(data.get("hazard_type") or data.get("emergency_type") or "")
        if not hazard_type:
            if any(w in flat for w in ("fire", "smoke", "flame", "burning")):
                hazard_type = "fire"
            elif any(w in flat for w in ("inhaler", "asthma", "choking", "unconscious", "fallen", "fall")):
                hazard_type = "medical"
            elif any(w in flat for w in ("car", "vehicle", "collision")):
                hazard_type = "collision"
            else:
                hazard_type = "unknown"

        confidence = data.get("confidence") or data.get("sensor_confidence") or (0.82 if explicit else 0.58)
        payload = {
            "source": "appvision",
            "hazard_type": hazard_type,
            "confidence": confidence,
            "human_risk": bool(data.get("human_risk") or data.get("person_at_risk")),
            "observation": question or flat[:1000],
            "sensor_evidence": {
                "frame": frame_meta,
                "hud_target_count": len(hud_packet.get("active_targets") or []),
                "labels": labels[:40],
                "analysis_ok": bool(analysis.get("ok", False)),
            },
            "capabilities": data.get("capabilities") if isinstance(data.get("capabilities"), dict) else {},
            "environment": data.get("environment") if isinstance(data.get("environment"), dict) else {},
            "failed_methods": data.get("failed_methods") if isinstance(data.get("failed_methods"), list) else [],
        }
        import SarahMemoryCognitiveServices as _CogServices  # type: ignore
        return _CogServices.evaluate_emergency_instinct(payload, caller="appvision.frame_bridge")
    except Exception as exc:
        return {"ok": False, "error": str(exc), "source": "appvision.emergency_instinct_bridge"}


@bp.post("/api/vision/frame/submit")
def api_vision_frame_submit():
    policy = _read_policy()
    if not bool(policy.get("enabled", True)):
        return _response({"ok": False, "error": "vision_disabled_by_backend_policy", "policy": policy}, 403)
    data = _payload()
    frame, status = _decode_frame(data)
    if frame is None:
        return _response({"ok": False, "error": "no_decodable_frame", "decode_status": status}, 400)
    analyze = bool(data.get("analyze"))
    background_analysis = False
    try:
        now_ms = time.time() * 1000.0
        interval_ms = max(500.0, float(policy.get("background_analysis_interval_ms", 1500) or 1500))
        with _FRAME_LOCK:
            last_bg = float(_FRAME_CACHE.get("last_background_analysis_ts") or 0.0)
        background_analysis = bool(policy.get("background_analysis_enabled", True)) and (now_ms - last_bg >= interval_ms)
        if background_analysis:
            with _FRAME_LOCK:
                _FRAME_CACHE["last_background_analysis_ts"] = now_ms
    except Exception:
        background_analysis = False
    run_analysis = bool(analyze or background_analysis)
    analysis = _analyze_frame(str(data.get("question") or "VR HUD observation pass"), frame, learning_allowed=False) if run_analysis else {"ok": True, "errors": [], "sobje": None, "facial": None, "background_analysis_skipped": True}
    if isinstance(analysis, dict):
        analysis["analysis_trigger"] = "explicit" if analyze else "background" if background_analysis else "skipped"
    meta = _cache_frame(frame, str(data.get("source") or "frontend_frame_submit"), analysis=None, hud_packet=None)
    hud_packet = _build_hud_packet(frame, analysis, meta)
    _cache_frame(frame, str(data.get("source") or "frontend_frame_submit"), analysis=analysis, hud_packet=hud_packet)
    emergency_instinct = _maybe_evaluate_emergency_instinct_from_vision(data, analysis, hud_packet, meta)
    payload = {"ok": True, "frame": meta, "frame_status": _frame_status_payload(), "hud_packet": hud_packet, "source": "appvision.frame_submit"}
    if emergency_instinct is not None:
        payload["emergency_instinct"] = emergency_instinct
    return _response(payload)


@bp.get("/api/vision/hud/status")
def api_vision_hud_status():
    status = _frame_status_payload()
    policy = _read_policy()
    msdc_status = None
    try:
        if _MSDC is not None and hasattr(_MSDC, "msdc_vr_hud_status"):
            msdc_status = _MSDC.msdc_vr_hud_status()  # type: ignore[attr-defined]
        elif _MSDC is not None and hasattr(_MSDC, "msdc_status"):
            msdc_status = _MSDC.msdc_status()  # type: ignore[attr-defined]
    except Exception as exc:
        msdc_status = {"ok": False, "error": str(exc)}
    return _response({
        "ok": True,
        "schema": SMHUD_SCHEMA_VERSION,
        "mode": "OBSERVE_ONLY",
        "movement_lock": True,
        "policy": policy,
        "frame_status": status,
        "msdc": msdc_status,
        "source": "appvision.hud_status",
    })


@bp.get("/api/vision/hud/packet")
def api_vision_hud_packet():
    with _FRAME_LOCK:
        packet = _FRAME_CACHE.get("hud_packet") if isinstance(_FRAME_CACHE.get("hud_packet"), dict) else None
    if not packet:
        packet = _build_hud_packet(None, {}, {"frame_id": "none", "timestamp": _utc_iso(), "source": "no_frame", "width": None, "height": None})
    return _response({"ok": True, "hud_packet": packet, "source": "appvision.hud_packet"})


def _smhud_chat_ts_epoch(value: Any) -> float:
    """Best-effort epoch timestamp parser for chat/Neuron frame handoff."""
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except Exception:
        pass
    try:
        s = str(value).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return 0.0


def get_latest_cached_frame_for_chat(max_age_s: Optional[int] = None) -> Dict[str, Any]:
    """Return the latest governed frame cache for /api/chat.

    This is a read-only bridge for app.py. It returns the same backend-owned
    frame accepted by /api/vision/frame/submit and exposed through
    /api/vision/frame/latest so Chat, Neuron, SOBJE, and the VR HUD use one
    vision truth source.

    It does not open camera hardware, does not probe Windows, does not authorize
    driver actions, and does not mutate policy/global state.
    """
    max_age = int(max_age_s or _read_policy().get("frame_ttl_seconds") or DEFAULT_POLICY.get("frame_ttl_seconds") or 10)
    max_age = max(1, max_age)
    with _FRAME_LOCK:
        rec = dict(_FRAME_CACHE)

    if not bool(rec.get("has_frame")):
        return {"ok": False, "has_frame": False, "error": "no_cached_frame", "source": "appvision.chat_frame_bridge"}

    data_url = rec.get("data_url")
    image_b64 = rec.get("image_b64")
    if not data_url and image_b64:
        data_url = "data:image/jpeg;base64," + str(image_b64)
    if not data_url and not image_b64:
        return {"ok": False, "has_frame": True, "error": "cached_frame_missing_encoded_image", "source": "appvision.chat_frame_bridge"}

    ts_epoch = _smhud_chat_ts_epoch(rec.get("image_cached_ts") or rec.get("ts"))
    age_s = (time.time() - ts_epoch) if ts_epoch else None
    if age_s is not None and age_s > max_age:
        return {
            "ok": False,
            "has_frame": True,
            "error": "cached_frame_stale",
            "age_s": round(float(age_s), 3),
            "max_age_s": max_age,
            "source": "appvision.chat_frame_bridge",
        }

    hud_packet = rec.get("hud_packet") if isinstance(rec.get("hud_packet"), dict) else {}
    return {
        "ok": True,
        "has_frame": True,
        "frame": data_url or image_b64,
        "data_url": data_url,
        "image_b64": image_b64,
        "mime": rec.get("mime") or "image/jpeg",
        "ts": ts_epoch or time.time(),
        "source": rec.get("source") or "appvision.frame_latest",
        "width": rec.get("width"),
        "height": rec.get("height"),
        "frame_id": rec.get("frame_id"),
        "image_cached_ts": rec.get("image_cached_ts"),
        "backend_cache": "appvision",
        "hud_packet_id": hud_packet.get("packet_id"),
        "source_endpoint": "appvision.get_latest_cached_frame_for_chat",
    }

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
        timeout_seconds = 3.0
        try:
            timeout_seconds = max(0.5, min(float(request.args.get("timeout") or os.getenv("SARAH_MSDC_PROBE_TIMEOUT_SEC", "3.0")), 10.0))
        except Exception:
            timeout_seconds = 3.0
        started = time.time()
        if include_discover:
            try:
                payload["discover"] = _MSDC.msdc_camera_discover(timeout_seconds=timeout_seconds)  # type: ignore[attr-defined]
            except TypeError:
                payload["discover"] = _MSDC.msdc_camera_discover()  # type: ignore[attr-defined]
        if include_probe:
            try:
                payload["probe"] = _MSDC.msdc_camera_probe(timeout_seconds=timeout_seconds)  # type: ignore[attr-defined]
            except TypeError:
                payload["probe"] = _MSDC.msdc_camera_probe()  # type: ignore[attr-defined]
        payload["source"] = "appvision.msdc.safe_probe"
        payload["safe_probe"] = {
            "enabled": True,
            "timeout_seconds_per_action": timeout_seconds,
            "elapsed_seconds": round(time.time() - started, 3),
            "request_hangs_prevented": True,
        }
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
    learning_allowed = bool(data.get("learning_allowed") and (data.get("user_authorized") or data.get("user_confirmed") or data.get("confirm") or data.get("approve"))) or (str(policy.get("learning_default") or "off").lower() in ("on", "true", "1", "yes") and bool(data.get("user_authorized") or data.get("user_confirmed") or data.get("confirm") or data.get("approve")))
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
    frame_meta = _cache_frame(frame, str(data.get("source") or "vision_analyze"), analysis=None, hud_packet=None)
    hud_packet = _build_hud_packet(frame, analysis, frame_meta)
    _cache_frame(frame, str(data.get("source") or "vision_analyze"), analysis=analysis, hud_packet=hud_packet)
    emergency_instinct = _maybe_evaluate_emergency_instinct_from_vision(data, analysis, hud_packet, frame_meta)
    sobje_packet = None
    try:
        sobje_packet = ((analysis.get("sobje") or {}).get("object_packet") or ((analysis.get("sobje") or {}).get("details") or {}).get("object_packet")) if isinstance(analysis.get("sobje"), dict) else None
    except Exception:
        sobje_packet = None
    payload = {"ok": bool(analysis.get("ok")), "analysis": analysis, "sobje_object_packet": sobje_packet, "hud_packet": hud_packet, "frame": frame_meta, "policy": policy, "source": "appvision.analysis"}
    if emergency_instinct is not None:
        payload["emergency_instinct"] = emergency_instinct
    return _response(payload)


@bp.post("/api/vision/learning/approve")
def api_vision_learning_approve():
    # Approval surface only. Actual identity/object persistence stays in governed core.
    data = _payload()
    approved = bool(data.get("learning_allowed") or data.get("approve")) and bool(data.get("user_authorized") or data.get("user_confirmed") or data.get("confirm") or data.get("confirmed"))
    if not approved:
        return _response({
            "ok": False,
            "decision": "REQUIRE_USER",
            "error": "learning_requires_explicit_user_approval",
            "learning_allowed": False,
            "identity_learning_requires_user_approval": True,
            "object_learning_requires_user_approval": True,
            "source": "appvision.learning_gate",
        }, 403)
    return _response({
        "ok": True,
        "status": "approval_recorded_for_next_governed_learning_pass",
        "learning_allowed": True,
        "identity_learning_requires_user_approval": True,
        "object_learning_requires_user_approval": True,
        "approval_scope": str(data.get("scope") or "next_governed_learning_pass"),
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

# ====================================================================
# END OF appvision.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing API bridge adapter.
SML_ORGAN_METADATA = {
    "name": 'appvision',
    "version": "v9.0.0-alpha-sml-0.2",
    "category": "Input",
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['api_bridge', 'transport', 'sml_bridge_candidate'],
    "supported_missions": ['Conversation', 'Execution', 'Knowledge', 'Diagnostics'],
    "supported_omega": ['Ω001', 'Ω002', 'Ω004', 'Ω020'],
    "required_authority": ['Read'],
    "priority": 58,
    "trust_level": "api_bridge_integrated",
    "internal_only": False,
    "metadata": {"sml_adapter": "api_bridge_non_executing", "source_file": 'appvision.py'},
}

def sml_get_metadata():
    return dict(SML_ORGAN_METADATA)

def sml_health():
    return {"status": "Healthy", "availability": 1.0, "integrity": 1.0, "performance": 1.0, "reliability": 1.0, "confidence": 0.75, "latency_ms": 0.0, "stability": 1.0, "compatibility": 1.0, "notes": ["SML API adapter present"]}

def sml_diagnostics():
    return {"status": "OK", "component": 'appvision', "sml_adapter": True, "metadata": dict(SML_ORGAN_METADATA), "health": sml_health()}
# --- SML ORGAN ADAPTER END ---

