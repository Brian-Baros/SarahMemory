"""--==The SarahMemory Project==--
File: SarahMemoryVRHudRenderer.py
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

SarahMemory VR Operator HUD Renderer
====================================
- Dedicated read-only tactical/telepresence HUD surface for a secondary display
such as a PSVR HDMI processor-box output.
- Renders backend-governed camera frames and SMHUD_PACKET_V1 telemetry from
appvision.py.
- Does not discover hardware through Windows/PnP/WMI.
- Does not authorize actions.
- Does not control robot movement, servos, drivers, or the OS.
- Display placement is explicit user/driver configuration only.

Doctrine:
- Backend owns truth.
- MSDC owns body/device mapping.
- SMGET/OperatorCore own authority.
- HUD renderer is a bounded visual surface only.
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "vr_hud_renderer"
# CATEGORY = "operator_visual_surface"
# USER_FACING = True
# UI_EXPOSURE = "runtime_surface"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "vision"
# HARDWARE_DOMAIN = "display_camera_vr"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "vr_operator_hud_renderer"
# FAMILY = "vision"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-06-06"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Read-only VR Operator HUD renderer. Pulls appvision SMHUD_PACKET_V1 + latest governed frame; renders to configured display coordinates. No hardware discovery or action authority."
# --- SARAHMETA END ---

import argparse
import base64
import json
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:  # pragma: no cover
    config = None  # type: ignore

MODULE_NAME = "SarahMemoryVRHudRenderer"
MODULE_VERSION = "9.0.0"
SCHEMA = "SMHUD_RENDERER_CONFIG_V1"
HUD_PACKET_SCHEMA = "SMHUD_PACKET_V1"

DEFAULT_CONFIG: Dict[str, Any] = {
    "schema": SCHEMA,
    "api_base": "http://127.0.0.1:8000",
    "endpoints": {
        "frame_latest": "/api/vision/frame/latest",
        "hud_packet": "/api/vision/hud/packet",
        "hud_status": "/api/vision/hud/status",
    },
    "display": {
        "window_title": "SM_A_HUD_DIRECT",
        "x": 0,
        "y": 0,
        "width": 1920,
        "height": 1080,
        "fullscreen": True,
        "borderless": False,
        "move_window": True,
        "target_role": "operator_vr_surface",
        "mirror_x": True,
    },
    "mirror": {
        "enabled": True,
        "window_title": "SM_A_HUD_MIRROR",
        "x": 60,
        "y": 60,
        "width": 960,
        "height": 540,
        "fullscreen": False,
        "move_window": True
    },
    "headset": {
        "enabled": True,
        "profile_id": "psvr_v1_processor_box",
        "render_mode": "mono_mirror",
        "lens_correction": False,
        "stereo_split": False,
        "auto_start_on_headset_connected": True,
        "auto_stop_on_headset_disconnected": True
    },
    "compositor": {
        "enabled": True,
        "mode": "mirror_plus_headset",
        "fit": "cover",
        "safe_border_px": 0,
        "hud_overlay": True
    },
    "render": {
        "fps": 15,
        "frame_poll_hz": 8,
        "packet_poll_hz": 3,
        "status_poll_hz": 0.5,
        "filter": "mono_crimson",
        "grid": True,
        "target_brackets": True,
        "telemetry_tapes": True,
        "center_crosshair": True,
        "stale_packet_ms": 2500,
        "no_frame_background": "black",
        "safe_exit_keys": [27, 113],
    },
    "safety": {
        "observe_only": True,
        "movement_locked": True,
        "hud_can_execute_actions": False,
        "hud_can_authorize_movement": False,
        "require_backend_packet_schema": True,
    },
}


def _base_dir() -> Path:
    try:
        if config is not None and hasattr(config, "BASE_DIR"):
            return Path(str(getattr(config, "BASE_DIR"))).expanduser().resolve()
    except Exception:
        pass
    return Path.cwd().resolve()


def _data_dir() -> Path:
    try:
        if config is not None and hasattr(config, "DATA_DIR"):
            return Path(str(getattr(config, "DATA_DIR"))).expanduser().resolve()
    except Exception:
        pass
    return (_base_dir() / "data").resolve()


def _settings_dir() -> Path:
    try:
        if config is not None and hasattr(config, "SETTINGS_DIR"):
            return Path(str(getattr(config, "SETTINGS_DIR"))).expanduser().resolve()
    except Exception:
        pass
    return (_data_dir() / "settings").resolve()


def _default_config_path() -> Path:
    return (_settings_dir() / "vr_hud_renderer.json").resolve()


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists() and path.is_file():
            obj = json.loads(path.read_text(encoding="utf-8"))
            return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    return {}


def _write_json(path: Path, obj: Dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        os.replace(str(tmp), str(path))
        return True
    except Exception:
        return False


def load_config(path: Optional[str] = None) -> Tuple[Dict[str, Any], Path]:
    cfg_path = Path(path).expanduser().resolve() if path else _default_config_path()
    existing = _load_json(cfg_path)
    cfg = _deep_merge(DEFAULT_CONFIG, existing)
    cfg["schema"] = SCHEMA
    if not cfg_path.exists():
        _write_json(cfg_path, cfg)
    return cfg, cfg_path


def _endpoint(api_base: str, endpoint: str) -> str:
    return str(api_base or "").rstrip("/") + "/" + str(endpoint or "").lstrip("/")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if v != v:
            return default
        return v
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _utc_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class SharedState:
    frame: Any = None
    frame_meta: Dict[str, Any] = field(default_factory=dict)
    packet: Dict[str, Any] = field(default_factory=dict)
    status: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    last_frame_ms: int = 0
    last_packet_ms: int = 0
    last_status_ms: int = 0
    frame_count: int = 0
    packet_count: int = 0
    running: bool = True
    lock: threading.RLock = field(default_factory=threading.RLock)

    def add_error(self, message: str) -> None:
        with self.lock:
            self.errors.append(str(message)[-240:])
            self.errors = self.errors[-8:]


class TelemetryClient:
    def __init__(self, cfg: Dict[str, Any], state: SharedState):
        self.cfg = cfg
        self.state = state
        self.api_base = str(cfg.get("api_base") or "http://127.0.0.1:8000")
        endpoints = cfg.get("endpoints") if isinstance(cfg.get("endpoints"), dict) else {}
        self.frame_url = _endpoint(self.api_base, str(endpoints.get("frame_latest") or "/api/vision/frame/latest"))
        self.packet_url = _endpoint(self.api_base, str(endpoints.get("hud_packet") or "/api/vision/hud/packet"))
        self.status_url = _endpoint(self.api_base, str(endpoints.get("hud_status") or "/api/vision/hud/status"))
        render = cfg.get("render") if isinstance(cfg.get("render"), dict) else {}
        self.frame_interval = 1.0 / max(1.0, _safe_float(render.get("frame_poll_hz"), 24.0))
        self.packet_interval = 1.0 / max(1.0, _safe_float(render.get("packet_poll_hz"), 10.0))
        self.status_interval = 1.0 / max(0.2, _safe_float(render.get("status_poll_hz"), 1.0))
        self.timeout_sec = 0.45

    def _get_json(self, url: str) -> Dict[str, Any]:
        req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": f"{MODULE_NAME}/{MODULE_VERSION}"})
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:  # nosec - local endpoint by default
            raw = resp.read(8_000_000)
        obj = json.loads(raw.decode("utf-8", errors="replace"))
        return obj if isinstance(obj, dict) else {}

    def _decode_frame(self, payload: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        if cv2 is None or np is None:
            return None, {}
        b64 = payload.get("image_b64")
        if not isinstance(b64, str) or not b64.strip():
            data_url = payload.get("data_url")
            if isinstance(data_url, str) and "," in data_url:
                b64 = data_url.split(",", 1)[1]
        if not isinstance(b64, str) or not b64.strip():
            return None, {}
        blob = base64.b64decode(b64, validate=False)
        arr = np.frombuffer(blob, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        meta = {
            "frame_id": payload.get("frame_id"),
            "ts": payload.get("ts"),
            "source": payload.get("source"),
            "width": payload.get("width"),
            "height": payload.get("height"),
        }
        return img, meta

    def frame_loop(self) -> None:
        while self.state.running:
            started = time.time()
            try:
                payload = self._get_json(self.frame_url)
                if payload.get("ok"):
                    img, meta = self._decode_frame(payload)
                    if img is not None:
                        with self.state.lock:
                            self.state.frame = img
                            self.state.frame_meta = meta
                            self.state.last_frame_ms = _utc_ms()
                            self.state.frame_count += 1
                elif payload.get("error"):
                    self.state.add_error(f"frame:{payload.get('error')}")
            except urllib.error.URLError as exc:
                self.state.add_error(f"frame_url:{exc}")
            except Exception as exc:
                self.state.add_error(f"frame_exception:{exc}")
            elapsed = time.time() - started
            time.sleep(max(0.01, self.frame_interval - elapsed))

    def packet_loop(self) -> None:
        require_schema = bool(((self.cfg.get("safety") or {}) if isinstance(self.cfg.get("safety"), dict) else {}).get("require_backend_packet_schema", True))
        while self.state.running:
            started = time.time()
            try:
                payload = self._get_json(self.packet_url)
                packet = payload.get("hud_packet") if isinstance(payload.get("hud_packet"), dict) else payload
                if isinstance(packet, dict):
                    if require_schema and str(packet.get("schema") or "") != HUD_PACKET_SCHEMA:
                        self.state.add_error("packet:schema_mismatch")
                    else:
                        with self.state.lock:
                            self.state.packet = packet
                            self.state.last_packet_ms = _utc_ms()
                            self.state.packet_count += 1
            except urllib.error.URLError as exc:
                self.state.add_error(f"packet_url:{exc}")
            except Exception as exc:
                self.state.add_error(f"packet_exception:{exc}")
            elapsed = time.time() - started
            time.sleep(max(0.02, self.packet_interval - elapsed))

    def status_loop(self) -> None:
        while self.state.running:
            started = time.time()
            try:
                payload = self._get_json(self.status_url)
                if isinstance(payload, dict):
                    with self.state.lock:
                        self.state.status = payload
                        self.state.last_status_ms = _utc_ms()
            except Exception as exc:
                self.state.add_error(f"status:{exc}")
            elapsed = time.time() - started
            time.sleep(max(0.1, self.status_interval - elapsed))

    def start(self) -> List[threading.Thread]:
        threads = [
            threading.Thread(target=self.frame_loop, name="SMHUDFramePoll", daemon=True),
            threading.Thread(target=self.packet_loop, name="SMHUDPacketPoll", daemon=True),
            threading.Thread(target=self.status_loop, name="SMHUDStatusPoll", daemon=True),
        ]
        for t in threads:
            t.start()
        return threads



class VRCompositor:
    """In-file SarahMemory VR compositor.

    The compositor transforms one governed HUD frame into two visual surfaces:
    a desktop mirror and the configured headset output. It does not discover
    hardware, open cameras, send USB packets, or authorize actions.
    """

    def __init__(self, cfg: Dict[str, Any], state: SharedState):
        self.cfg = cfg
        self.state = state
        self.compositor_cfg = cfg.get("compositor") if isinstance(cfg.get("compositor"), dict) else {}
        self.headset_cfg = cfg.get("headset") if isinstance(cfg.get("headset"), dict) else {}
        self.mirror_cfg = cfg.get("mirror") if isinstance(cfg.get("mirror"), dict) else {}

    @staticmethod
    def _resize(frame: Any, width: int, height: int) -> Any:
        if cv2 is None or np is None:
            return frame
        try:
            return cv2.resize(frame, (max(1, int(width)), max(1, int(height))), interpolation=cv2.INTER_AREA)
        except Exception:
            return frame

    def compose_headset(self, hud_frame: Any, width: int, height: int) -> Any:
        if hud_frame is None:
            return hud_frame
        render_mode = str(self.headset_cfg.get("render_mode") or "mono_mirror").strip().lower()
        stereo = bool(self.headset_cfg.get("stereo_split", False)) or render_mode in {"side_by_side", "sbs", "stereo_sbs"}
        base = self._resize(hud_frame, width, height)
        if not stereo or cv2 is None or np is None:
            return base
        try:
            half_w = max(1, width // 2)
            eye = self._resize(hud_frame, half_w, height)
            return np.concatenate([eye, eye.copy()], axis=1)
        except Exception:
            return base

    def compose_mirror(self, hud_frame: Any, width: int, height: int) -> Any:
        return self._resize(hud_frame, width, height)

    def status(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "schema": "SarahMemoryVRCompositor.status.v1",
            "mode": str(self.compositor_cfg.get("mode") or "mirror_plus_headset"),
            "headset_enabled": bool(self.headset_cfg.get("enabled", True)),
            "mirror_enabled": bool(self.mirror_cfg.get("enabled", True)),
            "profile_id": str(self.headset_cfg.get("profile_id") or "psvr_v1_processor_box"),
            "render_mode": str(self.headset_cfg.get("render_mode") or "mono_mirror"),
            "movement_lock": True,
            "authority": "visual_only",
        }

class HudRenderer:
    def __init__(self, cfg: Dict[str, Any], state: SharedState):
        if cv2 is None or np is None:
            raise RuntimeError("SarahMemoryVRHudRenderer requires cv2 and numpy for the runtime HUD surface.")
        self.cfg = cfg
        self.state = state
        display = cfg.get("display") if isinstance(cfg.get("display"), dict) else {}
        render = cfg.get("render") if isinstance(cfg.get("render"), dict) else {}
        mirror = cfg.get("mirror") if isinstance(cfg.get("mirror"), dict) else {}
        headset = cfg.get("headset") if isinstance(cfg.get("headset"), dict) else {}
        self.title = str(display.get("window_title") or "SM_A_HUD_DIRECT")
        self.x = _safe_int(display.get("x"), 0)
        self.y = _safe_int(display.get("y"), 0)
        self.w = max(320, _safe_int(display.get("width"), 1920))
        self.h = max(240, _safe_int(display.get("height"), 1080))
        self.fullscreen = bool(display.get("fullscreen", True))
        self.move_window = bool(display.get("move_window", True))
        # Presentation-only horizontal correction. Raw appvision/SOBJE/FaceRec
        # coordinates remain canonical; only the displayed camera background and
        # the drawn target boxes are mirrored. HUD text is drawn after this step.
        self.mirror_camera_x = bool(display.get("mirror_x", True))
        self.headset_enabled = bool(headset.get("enabled", True))
        self.mirror_enabled = bool(mirror.get("enabled", True))
        self.mirror_title = str(mirror.get("window_title") or "SM_A_HUD_MIRROR")
        self.mirror_x = _safe_int(mirror.get("x"), 60)
        self.mirror_y = _safe_int(mirror.get("y"), 60)
        self.mirror_w = max(320, _safe_int(mirror.get("width"), 960))
        self.mirror_h = max(240, _safe_int(mirror.get("height"), 540))
        self.mirror_fullscreen = bool(mirror.get("fullscreen", False))
        self.mirror_move_window = bool(mirror.get("move_window", True))
        self.compositor = VRCompositor(cfg, state)
        self.fps = max(5.0, min(60.0, _safe_float(render.get("fps"), 15.0)))
        self.filter_name = str(render.get("filter") or "mono_crimson").strip().lower()
        self.grid = bool(render.get("grid", True))
        self.brackets = bool(render.get("target_brackets", True))
        self.tapes = bool(render.get("telemetry_tapes", True))
        self.crosshair = bool(render.get("center_crosshair", True))
        self.stale_packet_ms = max(250, _safe_int(render.get("stale_packet_ms"), 2500))
        self.safe_exit_keys = set(_safe_int(k, 0) for k in (render.get("safe_exit_keys") or [27, 113]))
        self._fps_accum = 0
        self._fps_last = time.time()
        self._fps_value = 0.0

    def _color(self, name: str) -> Tuple[int, int, int]:
        palette = {
            "red": (32, 32, 255),
            "red_dim": (10, 10, 110),
            "green": (32, 255, 80),
            "green_dim": (20, 90, 30),
            "white": (220, 220, 220),
            "gray": (120, 120, 120),
            "yellow": (30, 220, 230),
            "black": (0, 0, 0),
        }
        return palette.get(name, (32, 32, 255))

    def _apply_filter(self, frame: Any) -> Any:
        try:
            frame = cv2.resize(frame, (self.w, self.h), interpolation=cv2.INTER_AREA)
            if self.mirror_camera_x:
                frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.filter_name in {"mono_green", "green"}:
                out = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                out[:, :, 1] = gray
                out[:, :, 0] = (gray * 0.18).astype(np.uint8)
                return out
            if self.filter_name in {"gray", "mono", "monochrome"}:
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # default: mono_crimson. BGR channel: R is index 2.
            out = np.zeros((self.h, self.w, 3), dtype=np.uint8)
            out[:, :, 2] = gray
            out[:, :, 1] = (gray * 0.10).astype(np.uint8)
            out[:, :, 0] = (gray * 0.10).astype(np.uint8)
            return out
        except Exception:
            return self._blank_frame("FILTER ERROR")

    def _blank_frame(self, text: str = "NO BACKEND FRAME") -> Any:
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self._put_text(img, text, (max(20, self.w // 2 - 220), self.h // 2), 0.8, "red", 2)
        self._put_text(img, "WAITING FOR /api/vision/frame/latest", (max(20, self.w // 2 - 260), self.h // 2 + 36), 0.48, "gray", 1)
        return img

    def _put_text(self, img: Any, text: Any, xy: Tuple[int, int], scale: float = 0.42, color: str = "red", thick: int = 1) -> None:
        try:
            cv2.putText(img, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, self._color(color), thick, cv2.LINE_AA)
        except Exception:
            pass

    def _line(self, img: Any, p1: Tuple[int, int], p2: Tuple[int, int], color: str = "red_dim", thick: int = 1) -> None:
        try:
            cv2.line(img, p1, p2, self._color(color), thick, cv2.LINE_AA)
        except Exception:
            pass

    def _rect(self, img: Any, p1: Tuple[int, int], p2: Tuple[int, int], color: str = "red_dim", thick: int = 1) -> None:
        try:
            cv2.rectangle(img, p1, p2, self._color(color), thick, cv2.LINE_AA)
        except Exception:
            pass

    def _draw_grid(self, img: Any) -> None:
        if not self.grid:
            return
        step_x = max(64, self.w // 16)
        step_y = max(48, self.h // 12)
        for x in range(0, self.w, step_x):
            self._line(img, (x, 0), (x, self.h), "red_dim", 1)
        for y in range(0, self.h, step_y):
            self._line(img, (0, y), (self.w, y), "red_dim", 1)
        self._line(img, (self.w // 2, 0), (self.w // 2, self.h), "red_dim", 1)
        self._line(img, (0, self.h // 2), (self.w, self.h // 2), "red_dim", 1)

    def _draw_crosshair(self, img: Any) -> None:
        if not self.crosshair:
            return
        cx, cy = self.w // 2, self.h // 2
        self._line(img, (cx - 42, cy), (cx - 14, cy), "red", 1)
        self._line(img, (cx + 14, cy), (cx + 42, cy), "red", 1)
        self._line(img, (cx, cy - 42), (cx, cy - 14), "red", 1)
        self._line(img, (cx, cy + 14), (cx, cy + 42), "red", 1)
        try:
            cv2.circle(img, (cx, cy), 22, self._color("red_dim"), 1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 4, self._color("red"), 1, cv2.LINE_AA)
        except Exception:
            pass

    def _draw_panel(self, img: Any, x: int, y: int, w: int, h: int, title: str) -> None:
        overlay = img.copy()
        try:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.58, img, 0.42, 0, img)
        except Exception:
            pass
        self._rect(img, (x, y), (x + w, y + h), "red_dim", 1)
        self._put_text(img, title, (x + 10, y + 18), 0.36, "red", 1)

    def _draw_targets(self, img: Any, packet: Dict[str, Any]) -> None:
        if not self.brackets:
            return
        targets = packet.get("active_targets") if isinstance(packet.get("active_targets"), list) else []
        for idx, obj in enumerate(targets[:16]):
            if not isinstance(obj, dict):
                continue
            bbox = obj.get("bbox") if isinstance(obj.get("bbox"), list) else []
            if len(bbox) < 4:
                continue
            nx1 = max(0.0, min(1.0, _safe_float(bbox[0], 0.0)))
            ny1 = max(0.0, min(1.0, _safe_float(bbox[1], 0.0)))
            nx2 = max(0.0, min(1.0, _safe_float(bbox[2], 0.0)))
            ny2 = max(0.0, min(1.0, _safe_float(bbox[3], 0.0)))
            left, right = min(nx1, nx2), max(nx1, nx2)
            top, bottom = min(ny1, ny2), max(ny1, ny2)
            if self.mirror_camera_x:
                left, right = 1.0 - right, 1.0 - left
            x1 = int(left * self.w)
            y1 = int(top * self.h)
            x2 = int(right * self.w)
            y2 = int(bottom * self.h)
            if x2 <= x1 or y2 <= y1:
                continue
            bw, bh = x2 - x1, y2 - y1
            arm = max(12, min(48, int(min(bw, bh) * 0.24)))
            color = "red"
            # top-left
            self._line(img, (x1, y1), (x1 + arm, y1), color, 2)
            self._line(img, (x1, y1), (x1, y1 + arm), color, 2)
            # top-right
            self._line(img, (x2, y1), (x2 - arm, y1), color, 2)
            self._line(img, (x2, y1), (x2, y1 + arm), color, 2)
            # bottom-left
            self._line(img, (x1, y2), (x1 + arm, y2), color, 2)
            self._line(img, (x1, y2), (x1, y2 - arm), color, 2)
            # bottom-right
            self._line(img, (x2, y2), (x2 - arm, y2), color, 2)
            self._line(img, (x2, y2), (x2, y2 - arm), color, 2)
            label = str(obj.get("label") or obj.get("class") or obj.get("id") or f"TGT_{idx}").upper()[:32]
            conf = _safe_float(obj.get("confidence"), 0.0)
            vec = obj.get("vectors") if isinstance(obj.get("vectors"), dict) else {}
            dx = _safe_float(vec.get('dx'), 0.0)
            if self.mirror_camera_x:
                dx = -dx
            data = f"{label} CONF={conf:.2f} DX={dx:+.2f} DY={_safe_float(vec.get('dy'),0):+.2f} DZ={vec.get('dz_est', '--')}"
            self._put_text(img, data, (x1, max(18, y1 - 8)), 0.34, "red", 1)

    def _draw_tapes(self, img: Any, packet: Dict[str, Any], status: Dict[str, Any], errors: List[str]) -> None:
        if not self.tapes:
            return
        pad = 18
        pw = max(220, self.w // 5)
        ph = max(130, self.h // 6)
        self._draw_panel(img, pad, 54, pw, ph, "COMPUTE INTEGRITY")
        compute = packet.get("compute_integrity") if isinstance(packet.get("compute_integrity"), dict) else {}
        lines = [
            f"THREADS {((compute.get('thread_state') or {}) if isinstance(compute.get('thread_state'), dict) else {}).get('active_threads', '--')}",
            f"MEM_MB  {compute.get('memory_pool_mb', '--')}",
            f"FRAME   {self.state.frame_count}",
            f"PACKET  {self.state.packet_count}",
            "AUTH    VISUAL_ONLY",
        ]
        y = 84
        for line in lines:
            self._put_text(img, line, (pad + 12, y), 0.34, "white", 1)
            y += 20

        self._draw_panel(img, pad, 54 + ph + 16, pw, ph, "VISION FEED")
        frame = packet.get("frame") if isinstance(packet.get("frame"), dict) else {}
        lines = [
            f"FRAME {frame.get('frame_id') or self.state.frame_meta.get('frame_id') or '--'}"[:42],
            f"SRC   {frame.get('source') or self.state.frame_meta.get('source') or '--'}"[:42],
            f"SIZE  {frame.get('width') or self.state.frame_meta.get('width') or '--'}x{frame.get('height') or self.state.frame_meta.get('height') or '--'}",
            f"TGT   {len(packet.get('active_targets') or [])}",
        ]
        y = 84 + ph + 16
        for line in lines:
            self._put_text(img, line, (pad + 12, y), 0.34, "white", 1)
            y += 20

        rx = self.w - pw - pad
        self._draw_panel(img, rx, 54, pw, ph, "KINETIC INTEGRITY")
        kinetic = packet.get("kinetic_integrity") if isinstance(packet.get("kinetic_integrity"), dict) else {}
        lines = [
            f"BODY {kinetic.get('body_state', 'OBSERVE_ONLY')}",
            f"MOVE {'LOCKED' if kinetic.get('movement_lock', True) else 'UNLOCKED'}",
            f"DEV  {len(kinetic.get('devices') or [])}",
            "MODE READ_ONLY",
            "FAULT NONE",
        ]
        y = 84
        for line in lines:
            self._put_text(img, line, (rx + 12, y), 0.34, "white", 1)
            y += 20

        self._draw_panel(img, rx, 54 + ph + 16, pw, ph, "SMGET GATE")
        smget = packet.get("smget_state") if isinstance(packet.get("smget_state"), dict) else {}
        lines = [
            f"STATE {smget.get('state', 'NO_ACTIVE_ACTION_CONTRACT')}"[:42],
            f"DECIS {smget.get('decision', 'READ_ONLY_WITNESS')}"[:42],
            f"ROLL  {str(smget.get('rollback_ready', True)).upper()}",
            "AUTH  USER_FINAL",
            "MOVE  LOCKED",
        ]
        y = 84 + ph + 16
        for line in lines:
            self._put_text(img, line, (rx + 12, y), 0.34, "white", 1)
            y += 20

        if errors:
            ey = self.h - 150
            self._draw_panel(img, pad, ey, min(self.w - 2 * pad, 720), 104, "RENDERER TRACE")
            yy = ey + 30
            for err in errors[-3:]:
                self._put_text(img, str(err)[-96:], (pad + 12, yy), 0.32, "yellow", 1)
                yy += 22

    def _draw_bars(self, img: Any, packet: Dict[str, Any]) -> None:
        now = _utc_ms()
        packet_age = now - self.state.last_packet_ms if self.state.last_packet_ms else 999999
        frame_age = now - self.state.last_frame_ms if self.state.last_frame_ms else 999999
        stale = packet_age > self.stale_packet_ms
        top = f"SARAHMEMORY VR OPERATOR HUD   MODE OBSERVE_ONLY   FRAME_AGE {frame_age}ms   PACKET_AGE {packet_age}ms   FPS {self._fps_value:.1f}"
        self._put_text(img, top, (16, 24), 0.42, "red" if not stale else "yellow", 1)
        bottom = "OBSERVE_ONLY / MOVEMENT LOCKED / HUD CANNOT AUTHORIZE ACTIONS / BACKEND OWNS TRUTH / USER FINAL AUTHORITY"
        self._put_text(img, bottom, (16, self.h - 18), 0.42, "red", 1)
        if stale:
            self._put_text(img, "SMHUD PACKET STALE", (max(20, self.w // 2 - 150), self.h - 52), 0.6, "yellow", 2)

    def _update_fps(self) -> None:
        self._fps_accum += 1
        now = time.time()
        if now - self._fps_last >= 1.0:
            self._fps_value = self._fps_accum / max(0.001, now - self._fps_last)
            self._fps_accum = 0
            self._fps_last = now

    def _render_frame(self) -> Any:
        with self.state.lock:
            raw = None if self.state.frame is None else self.state.frame.copy()
            packet = dict(self.state.packet or {})
            status = dict(self.state.status or {})
            errors = list(self.state.errors or [])
        img = self._apply_filter(raw) if raw is not None else self._blank_frame()
        self._draw_grid(img)
        self._draw_crosshair(img)
        self._draw_targets(img, packet)
        self._draw_tapes(img, packet, status, errors)
        self._draw_bars(img, packet)
        self._update_fps()
        return img

    def _open_window(self, title: str, width: int, height: int, x: int, y: int, fullscreen: bool, move_window: bool) -> None:
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(title, width, height)
        except Exception:
            pass
        if move_window:
            try:
                cv2.moveWindow(title, x, y)
            except Exception:
                pass
        if fullscreen:
            try:
                cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except Exception:
                pass

    def run(self) -> int:
        if self.headset_enabled:
            self._open_window(self.title, self.w, self.h, self.x, self.y, self.fullscreen, self.move_window)
        if self.mirror_enabled:
            self._open_window(self.mirror_title, self.mirror_w, self.mirror_h, self.mirror_x, self.mirror_y, self.mirror_fullscreen, self.mirror_move_window)
        frame_interval = 1.0 / self.fps
        while self.state.running:
            started = time.time()
            hud = self._render_frame()
            if self.headset_enabled:
                cv2.imshow(self.title, self.compositor.compose_headset(hud, self.w, self.h))
            if self.mirror_enabled:
                cv2.imshow(self.mirror_title, self.compositor.compose_mirror(hud, self.mirror_w, self.mirror_h))
            key = cv2.waitKey(1) & 0xFF
            if key in self.safe_exit_keys:
                self.state.running = False
                break
            elapsed = time.time() - started
            time.sleep(max(0.001, frame_interval - elapsed))
        try:
            if self.headset_enabled:
                cv2.destroyWindow(self.title)
            if self.mirror_enabled:
                cv2.destroyWindow(self.mirror_title)
        except Exception:
            cv2.destroyAllWindows()
        return 0


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = dict(cfg)
    out.setdefault("display", {})
    out.setdefault("render", {})
    if args.api_base:
        out["api_base"] = args.api_base
    for key in ("x", "y", "width", "height"):
        val = getattr(args, key, None)
        if val is not None:
            out["display"][key] = int(val)
    if args.windowed:
        out["display"]["fullscreen"] = False
    if args.fullscreen:
        out["display"]["fullscreen"] = True
    if args.filter:
        out["render"]["filter"] = args.filter
    if args.fps:
        out["render"]["fps"] = float(args.fps)
    out.setdefault("mirror", {})
    out.setdefault("headset", {})
    if args.no_mirror:
        out["mirror"]["enabled"] = False
    if args.no_headset:
        out["headset"]["enabled"] = False
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SarahMemory read-only VR Operator HUD renderer.")
    p.add_argument("--config", default="", help="Path to vr_hud_renderer.json. Defaults to data/settings/vr_hud_renderer.json.")
    p.add_argument("--api-base", default="", help="SarahMemory local API base. Default: http://127.0.0.1:8000")
    p.add_argument("--x", type=int, default=None, help="Configured display X coordinate. Example for second display to the right: --x 1920")
    p.add_argument("--y", type=int, default=None, help="Configured display Y coordinate.")
    p.add_argument("--width", type=int, default=None, help="HUD window width. Default 1920.")
    p.add_argument("--height", type=int, default=None, help="HUD window height. Default 1080.")
    p.add_argument("--fullscreen", action="store_true", help="Force fullscreen window.")
    p.add_argument("--windowed", action="store_true", help="Force windowed mode.")
    p.add_argument("--filter", default="", choices=["mono_crimson", "mono_green", "gray", "mono", "green"], help="Video contrast filter.")
    p.add_argument("--fps", type=float, default=None, help="Renderer target FPS.")
    p.add_argument("--no-mirror", action="store_true", help="Disable desktop mirror popup.")
    p.add_argument("--no-headset", action="store_true", help="Disable headset output surface.")
    p.add_argument("--write-config", action="store_true", help="Write merged config with CLI overrides before launch.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg, cfg_path = load_config(args.config or None)
    cfg = apply_cli_overrides(cfg, args)
    if args.write_config:
        _write_json(cfg_path, cfg)
    print(f"[{MODULE_NAME}] config={cfg_path}")
    print(f"[{MODULE_NAME}] api_base={cfg.get('api_base')}")
    print(f"[{MODULE_NAME}] display=({cfg.get('display', {}).get('x')},{cfg.get('display', {}).get('y')}) {cfg.get('display', {}).get('width')}x{cfg.get('display', {}).get('height')} fullscreen={cfg.get('display', {}).get('fullscreen')}")
    print(f"[{MODULE_NAME}] mirror={cfg.get('mirror', {}).get('enabled')} headset={cfg.get('headset', {}).get('enabled')} compositor={cfg.get('compositor', {}).get('mode')} mirror_x={cfg.get('display', {}).get('mirror_x')}")
    print(f"[{MODULE_NAME}] OBSERVE_ONLY / MOVEMENT_LOCKED / VISUAL_SURFACE_ONLY")

    state = SharedState()

    def _stop(_signum=None, _frame=None):
        state.running = False

    try:
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)
    except Exception:
        pass

    client = TelemetryClient(cfg, state)
    client.start()
    try:
        renderer = HudRenderer(cfg, state)
        return renderer.run()
    except Exception as exc:
        state.running = False
        print(f"[{MODULE_NAME}] fatal={exc}", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

# ====================================================================
# END OF SarahMemoryVRHudRenderer.py v9.0.0
# ====================================================================
