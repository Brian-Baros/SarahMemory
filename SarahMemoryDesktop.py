# --==The SarahMemory Project==--
# File: SarahMemoryDesktop.py
# Part of the SarahMemory Companion AI-bot Platform / SarahMemory AiOS
# Version: v8.0.0
# Date: 2026-05-26
# Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
# www.sarahmemory.com
#==============================================================================================
"""
SarahMemoryDesktop.py

Single-file desktop mirror + desktop-task foundation for SarahMemory AiOS.

This module intentionally keeps all desktop-related responsibilities in one file,
implemented as separate classes instead of separate SarahMemoryDesktop*.py files:

- DesktopContracts: stable packet builders and risk labels.
- DesktopMirrorService: read-only screen capture, latest-frame cache, MJPEG stream.
- DesktopVisionService: observation packet over the latest desktop frame.
- DesktopOperatorService: governed action-request placeholder; no raw execution by default.
- DesktopAutonomyService: bounded task-request placeholder; no autonomous task loop by default.
- SarahMemoryDesktopRuntime: facade used by api/server/app.py routes.

Governance doctrine:
- Screen capture is read-only and local-first.
- Desktop control is disabled unless explicitly wired through OperatorCore/governance later.
- This file does not self-authorize mouse, keyboard, file, network, or hardware actions.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional

try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:  # pragma: no cover - SarahMemory can still report unavailable state
    config = None  # type: ignore

logger = logging.getLogger("SarahMemoryDesktop")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _flag_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _int_env(name: str, default: int, minimum: int = 1, maximum: int = 120) -> int:
    try:
        value = int(os.getenv(name, str(default)) or default)
        return max(minimum, min(maximum, value))
    except Exception:
        return int(default)


def _base_dir() -> str:
    try:
        return str(getattr(config, "BASE_DIR", os.getcwd())) if config else os.getcwd()
    except Exception:
        return os.getcwd()


def _data_dir() -> str:
    try:
        return str(getattr(config, "DATA_DIR", os.path.join(_base_dir(), "data"))) if config else os.path.join(_base_dir(), "data")
    except Exception:
        return os.path.join(_base_dir(), "data")


@dataclass
class DesktopFrame:
    frame_id: str
    ts: float
    width: int
    height: int
    mime: str
    image_bytes: bytes
    source: str = "desktop_mirror"
    monitor_index: int = 1
    meta: Dict[str, Any] = field(default_factory=dict)

    def b64(self) -> str:
        return base64.b64encode(self.image_bytes).decode("ascii")

    def data_url(self) -> str:
        return f"data:{self.mime};base64,{self.b64()}"

    def packet(self, include_image: bool = False) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "ok": True,
            "has_frame": True,
            "frame_id": self.frame_id,
            "ts": self.ts,
            "source": self.source,
            "width": self.width,
            "height": self.height,
            "mime": self.mime,
            "monitor_index": self.monitor_index,
            "meta": dict(self.meta or {}),
        }
        if include_image:
            out["image_b64"] = self.b64()
            out["data_url"] = self.data_url()
            out["frame"] = self.data_url()
        return out


class DesktopContracts:
    """Contract helper for desktop packets and risk labels."""

    LOW_RISK_ACTIONS = {"observe", "capture", "move_mouse_preview"}
    MEDIUM_RISK_ACTIONS = {"click", "double_click", "right_click", "hotkey", "type_text", "scroll"}
    HIGH_RISK_TERMS = {
        "delete", "format", "purchase", "buy", "send", "submit", "password", "login",
        "install", "uninstall", "admin", "administrator", "registry", "security", "payment",
    }

    @staticmethod
    def risk_for_action(action: str, payload: Optional[Dict[str, Any]] = None) -> str:
        action_l = str(action or "").strip().lower()
        payload_text = json.dumps(payload or {}, default=str).lower()
        if any(term in action_l or term in payload_text for term in DesktopContracts.HIGH_RISK_TERMS):
            return "high"
        if action_l in DesktopContracts.MEDIUM_RISK_ACTIONS:
            return "medium"
        if action_l in DesktopContracts.LOW_RISK_ACTIONS:
            return "low"
        return "medium"

    @staticmethod
    def status_packet(**kwargs: Any) -> Dict[str, Any]:
        packet = {
            "ok": True,
            "schema": "SarahMemory.desktop.status.v1",
            "source": "SarahMemoryDesktop",
            "ts": time.time(),
            "observe_only": True,
            "operator_execution_enabled": False,
            "autonomy_execution_enabled": False,
        }
        packet.update(kwargs)
        return packet

    @staticmethod
    def action_ticket(action: str, payload: Optional[Dict[str, Any]] = None, source: str = "api.desktop") -> Dict[str, Any]:
        payload = payload or {}
        risk = DesktopContracts.risk_for_action(action, payload)
        return {
            "ok": True,
            "schema": "SarahMemory.desktop.action_request.v1",
            "ticket_id": f"desktop_act_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
            "action": str(action or "unknown"),
            "args": payload,
            "risk": risk,
            "safety_level": risk,
            "requires_confirm": True,
            "executor": "SarahMemoryDesktopOperator",
            "source": source,
            "ts": time.time(),
            "status": "queued_for_governance",
            "executed": False,
            "note": "Desktop action was accepted as a governed request only. No mouse/keyboard action was executed by SarahMemoryDesktop.py.",
        }


class DesktopMirrorService:
    """Read-only desktop capture and MJPEG streaming service."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[DesktopFrame] = None
        self._frame_seq = 0
        self._last_error = ""
        self._running = False
        self.default_fps = _int_env("SARAH_DESKTOP_MIRROR_FPS", 6, minimum=1, maximum=30)
        self.default_monitor = _int_env("SARAH_DESKTOP_MONITOR_INDEX", 1, minimum=0, maximum=16)
        self.max_width = _int_env("SARAH_DESKTOP_MAX_WIDTH", 1280, minimum=320, maximum=7680)
        self.jpeg_quality = _int_env("SARAH_DESKTOP_JPEG_QUALITY", 70, minimum=20, maximum=95)
        self.enabled = _flag_env("SARAH_DESKTOP_MIRROR_ENABLED", True)

    def dependencies(self) -> Dict[str, Any]:
        deps = {"mss": False, "PIL": False}
        try:
            import mss  # noqa: F401
            deps["mss"] = True
        except Exception as exc:
            deps["mss_error"] = str(exc)
        try:
            from PIL import Image  # noqa: F401
            deps["PIL"] = True
        except Exception as exc:
            deps["PIL_error"] = str(exc)
        return deps

    def available(self) -> bool:
        return bool(self.enabled and self.dependencies().get("mss"))

    def status(self) -> Dict[str, Any]:
        with self._lock:
            latest = self._latest.packet(include_image=False) if self._latest else {"has_frame": False}
            running = self._running and self._thread is not None and self._thread.is_alive()
        deps = self.dependencies()
        return DesktopContracts.status_packet(
            enabled=bool(self.enabled),
            available=bool(self.available()),
            running=bool(running),
            default_fps=self.default_fps,
            monitor_index=self.default_monitor,
            max_width=self.max_width,
            jpeg_quality=self.jpeg_quality,
            dependencies=deps,
            latest=latest,
            last_error=self._last_error,
        )

    def _select_monitor(self, sct: Any, monitor_index: Optional[int] = None) -> Dict[str, int]:
        idx = self.default_monitor if monitor_index is None else int(monitor_index)
        monitors = getattr(sct, "monitors", []) or []
        if not monitors:
            raise RuntimeError("mss returned no monitors")
        if idx < 0 or idx >= len(monitors):
            idx = 1 if len(monitors) > 1 else 0
        return dict(monitors[idx])

    def _encode_shot(self, shot: Any) -> tuple[bytes, str, int, int]:
        width = int(getattr(shot, "width", 0) or 0)
        height = int(getattr(shot, "height", 0) or 0)
        raw = getattr(shot, "raw", None)
        if not raw:
            raise RuntimeError("mss frame has no raw bytes")

        try:
            from PIL import Image
            image = Image.frombytes("RGB", (width, height), raw, "raw", "BGRX")
            if self.max_width and width > self.max_width:
                ratio = self.max_width / float(width)
                new_h = max(1, int(height * ratio))
                image = image.resize((self.max_width, new_h))
                width, height = image.size
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=self.jpeg_quality, optimize=True)
            return buf.getvalue(), "image/jpeg", width, height
        except Exception as pil_exc:
            # PNG fallback keeps the feature alive if Pillow is missing.
            try:
                import mss.tools
                png = mss.tools.to_png(raw, shot.size)
                return png, "image/png", width, height
            except Exception as png_exc:
                raise RuntimeError(f"desktop frame encode failed: PIL={pil_exc}; PNG={png_exc}") from png_exc

    def capture_once(self, monitor_index: Optional[int] = None, source: str = "desktop_capture_once") -> Dict[str, Any]:
        if not self.enabled:
            self._last_error = "desktop mirror disabled by SARAH_DESKTOP_MIRROR_ENABLED"
            return {"ok": False, "error": self._last_error, "available": False}
        try:
            import mss
            with mss.mss() as sct:
                monitor = self._select_monitor(sct, monitor_index)
                shot = sct.grab(monitor)
                image_bytes, mime, width, height = self._encode_shot(shot)
            with self._lock:
                self._frame_seq += 1
                frame = DesktopFrame(
                    frame_id=f"desktop_{int(time.time() * 1000)}_{self._frame_seq}",
                    ts=time.time(),
                    width=width,
                    height=height,
                    mime=mime,
                    image_bytes=image_bytes,
                    source=source,
                    monitor_index=self.default_monitor if monitor_index is None else int(monitor_index),
                    meta={"observe_only": True, "capture_backend": "mss"},
                )
                self._latest = frame
                self._last_error = ""
            return frame.packet(include_image=False)
        except Exception as exc:
            self._last_error = str(exc)
            logger.warning("Desktop capture failed: %s", exc)
            return {"ok": False, "error": str(exc), "available": False, "source": source}

    def start(self, fps: Optional[int] = None, monitor_index: Optional[int] = None, reason: str = "api_start") -> Dict[str, Any]:
        if not self.available():
            return {"ok": False, "error": "desktop_mirror_unavailable", "status": self.status()}
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return {"ok": True, "already_running": True, "status": self.status()}
            self._stop_event.clear()
            fps_value = max(1, min(30, int(fps or self.default_fps)))
            mon_value = self.default_monitor if monitor_index is None else int(monitor_index)
            self._thread = threading.Thread(
                target=self._capture_loop,
                args=(fps_value, mon_value),
                name="SM_DesktopMirror",
                daemon=True,
            )
            self._running = True
            self._thread.start()
        # Prime one frame quickly so the UI has something to render.
        self.capture_once(monitor_index=monitor_index, source="desktop_start_prime")
        return {"ok": True, "started": True, "reason": reason, "status": self.status()}

    def stop(self, reason: str = "api_stop") -> Dict[str, Any]:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=2.0)
            except Exception:
                pass
        with self._lock:
            self._running = False
            self._thread = None
        return {"ok": True, "stopped": True, "reason": reason, "status": self.status()}

    def _capture_loop(self, fps: int, monitor_index: int) -> None:
        interval = 1.0 / max(1, fps)
        logger.info("Desktop mirror capture loop started: fps=%s monitor=%s", fps, monitor_index)
        try:
            while not self._stop_event.is_set():
                self.capture_once(monitor_index=monitor_index, source="desktop_capture_loop")
                time.sleep(interval)
        finally:
            with self._lock:
                self._running = False
            logger.info("Desktop mirror capture loop stopped")

    def latest(self, include_image: bool = True, auto_capture: bool = False) -> Dict[str, Any]:
        if auto_capture:
            self.capture_once(source="desktop_latest_auto_capture")
        with self._lock:
            frame = self._latest
        if frame is None:
            return {"ok": True, "has_frame": False, "available": self.available(), "last_error": self._last_error}
        return frame.packet(include_image=include_image)

    def mjpeg_stream(self, fps: Optional[int] = None) -> Generator[bytes, None, None]:
        fps_value = max(1, min(30, int(fps or self.default_fps)))
        interval = 1.0 / fps_value
        if not self.available():
            message = json.dumps({"ok": False, "error": "desktop_mirror_unavailable", "dependencies": self.dependencies()}).encode("utf-8")
            yield b"--frame\r\nContent-Type: application/json\r\n\r\n" + message + b"\r\n"
            return
        if not (self._thread is not None and self._thread.is_alive()):
            self.start(fps=fps_value, reason="mjpeg_stream_autostart")
        while True:
            with self._lock:
                frame = self._latest
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    + f"Content-Type: {frame.mime}\r\n".encode("ascii")
                    + f"X-Frame-Id: {frame.frame_id}\r\n".encode("ascii")
                    + b"Cache-Control: no-store\r\n\r\n"
                    + frame.image_bytes
                    + b"\r\n"
                )
            time.sleep(interval)


class DesktopVisionService:
    """Frame observation surface for future OCR/SOBJE desktop understanding."""

    def __init__(self, mirror: DesktopMirrorService) -> None:
        self.mirror = mirror

    def observe(self, include_image: bool = False) -> Dict[str, Any]:
        frame = self.mirror.latest(include_image=include_image, auto_capture=True)
        if not frame.get("has_frame"):
            return {
                "ok": False,
                "schema": "SarahMemory.desktop.observation.v1",
                "source": "SarahMemoryDesktopVision",
                "reason": "no_desktop_frame",
                "mirror_status": self.mirror.status(),
            }
        return {
            "ok": True,
            "schema": "SarahMemory.desktop.observation.v1",
            "source": "SarahMemoryDesktopVision",
            "observe_only": True,
            "frame": frame,
            "visible_text": [],
            "ui_regions": [],
            "active_window": None,
            "note": "Desktop frame captured. OCR/window/object interpretation is reserved for the next governed patch layer.",
        }


class DesktopOperatorService:
    """Governed desktop action request intake. Execution is intentionally disabled."""

    def request_action(self, payload: Optional[Dict[str, Any]] = None, source: str = "api.desktop.action") -> Dict[str, Any]:
        payload = payload or {}
        action = str(payload.get("action") or payload.get("type") or "unknown").strip().lower()
        ticket = DesktopContracts.action_ticket(action=action, payload=payload, source=source)
        ticket["operator_execution_enabled"] = False
        ticket["operator_note"] = "Wire this ticket through OperatorCore/SafetyPolicies/AssuranceGate before enabling real desktop control."
        return ticket


class DesktopAutonomyService:
    """Bounded desktop task request intake. No task loop executes in this first patch."""

    def __init__(self, vision: DesktopVisionService, operator: DesktopOperatorService) -> None:
        self.vision = vision
        self.operator = operator

    def request_task(self, payload: Optional[Dict[str, Any]] = None, source: str = "api.desktop.task") -> Dict[str, Any]:
        payload = payload or {}
        goal = str(payload.get("goal") or payload.get("task") or payload.get("text") or "").strip()
        task_id = f"desktop_task_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        return {
            "ok": True,
            "schema": "SarahMemory.desktop.task_request.v1",
            "task_id": task_id,
            "goal": goal,
            "source": source,
            "status": "received_not_started",
            "autonomy_execution_enabled": False,
            "requires_user_approval": True,
            "observe_only": True,
            "plan": [],
            "note": "Desktop autonomy request captured. Autonomous observe-act-verify loop is intentionally not enabled in this patch.",
            "ts": time.time(),
        }


class SarahMemoryDesktopRuntime:
    """Single facade for Flask/API integration."""

    def __init__(self) -> None:
        self.contracts = DesktopContracts()
        self.mirror = DesktopMirrorService()
        self.vision = DesktopVisionService(self.mirror)
        self.operator = DesktopOperatorService()
        self.autonomy = DesktopAutonomyService(self.vision, self.operator)

    def status(self) -> Dict[str, Any]:
        return self.mirror.status()

    def start(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload or {}
        fps = payload.get("fps")
        monitor = payload.get("monitor_index", payload.get("monitor"))
        return self.mirror.start(
            fps=int(fps) if fps not in (None, "") else None,
            monitor_index=int(monitor) if monitor not in (None, "") else None,
            reason=str(payload.get("reason") or "api_start"),
        )

    def stop(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload or {}
        return self.mirror.stop(reason=str(payload.get("reason") or "api_stop"))

    def capture(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = payload or {}
        monitor = payload.get("monitor_index", payload.get("monitor"))
        include_image = bool(payload.get("include_image", True))
        result = self.mirror.capture_once(
            monitor_index=int(monitor) if monitor not in (None, "") else None,
            source=str(payload.get("source") or "api_desktop_capture"),
        )
        if include_image and result.get("ok"):
            return self.mirror.latest(include_image=True)
        return result

    def latest(self, include_image: bool = True, auto_capture: bool = False) -> Dict[str, Any]:
        return self.mirror.latest(include_image=include_image, auto_capture=auto_capture)

    def mjpeg_stream(self, fps: Optional[int] = None) -> Generator[bytes, None, None]:
        return self.mirror.mjpeg_stream(fps=fps)

    def observe(self, include_image: bool = False) -> Dict[str, Any]:
        return self.vision.observe(include_image=include_image)

    def request_action(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.operator.request_action(payload or {})

    def request_task(self, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.autonomy.request_task(payload or {})


_RUNTIME: Optional[SarahMemoryDesktopRuntime] = None
_RUNTIME_LOCK = threading.RLock()


def get_desktop_runtime() -> SarahMemoryDesktopRuntime:
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = SarahMemoryDesktopRuntime()
        return _RUNTIME


# Compatibility aliases for future imports.
SarahMemoryDesktop = SarahMemoryDesktopRuntime
DesktopRuntime = SarahMemoryDesktopRuntime


if __name__ == "__main__":
    rt = get_desktop_runtime()
    print(json.dumps(rt.status(), indent=2, sort_keys=True, default=str))
