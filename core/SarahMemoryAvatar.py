"""--==The SarahMemory Project==--
File: SarahMemoryAvatar.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.1.0a (Enterprise)
Date: 2026-11-01
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

===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "B"
# ROLE = "avatar_engine"
# CATEGORY = "avatar_system"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = "display_audio"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "avatar_control"
# FAMILY = "avatar"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-11-01"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Avatar state/render/control engine for emotion sync, 2D rendering, animation hooks, lip sync helpers, and avatar state persistence."
# --- SARAHMETA END ---

import logging
import math
import json
import copy
import sys
from pathlib import Path
from contextlib import closing
from collections import deque
import random
import time
import os
import sqlite3
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional
import threading
try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk, ImageSequence
except ImportError:
    Image = ImageDraw = ImageFont = ImageTk = ImageSequence = None  # type: ignore
from SarahMemoryAdaptive import load_emotional_state
import SarahMemoryGlobals as config
from SarahMemoryGlobals import DATASETS_DIR  # for consistent pathing
import subprocess

DB_FILENAME = "avatar.db"
MODULE_VERSION = "9.1.0a"
_AVATAR_MAX_EVENTS = 48
_AVATAR_MAX_EVENT_BYTES = 4096
_AVATAR_MAX_JSON_DEPTH = 5
_AVATAR_MAX_TEXT = 240
_AVATAR_RENDER_TIMEOUT_S = 30
_AVATAR_RUNTIME_METRICS = {"db_reads": 0, "db_writes": 0, "db_failures": 0, "event_rejections": 0,
                           "render_denials": 0, "render_launches": 0}
_AVATAR_RENDER_LOCK = threading.BoundedSemaphore(1)
_AVATAR_LIPSYNC_LOCK = threading.BoundedSemaphore(2)
_AVATAR_LOG_LOCK = threading.RLock()
_AVATAR_DB_INIT_LOCK = threading.RLock()
_AVATAR_DB_READY = False
_AVATAR_REM_DB_READY = False
_AVATAR_RECENT_ERRORS = deque(maxlen=12)

def _avatar_record_error(kind: str, exc: Any) -> None:
    _AVATAR_RECENT_ERRORS.append({"component": str(kind)[:60], "error": str(exc)[:160]})

def _avatar_flag(name: str, default: bool = False) -> bool:
    value = getattr(config, name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}

def _avatar_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "on", "1"}
    return False

def _avatar_finite(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
        if not math.isfinite(number):
            return default
        return max(low, min(high, number))
    except (ValueError, TypeError, OverflowError):
        return default

def _avatar_text(value: Any, limit: int = _AVATAR_MAX_TEXT) -> str:
    try:
        return str(value if value is not None else "").replace("\x00", "")[:limit]
    except Exception:
        return ""

def _avatar_json_safe(value: Any, depth: int = 0) -> Any:
    """Bound JSON material; never carry objects, giant blobs, NaN or recursive graphs."""
    if depth >= _AVATAR_MAX_JSON_DEPTH:
        return "[truncated_depth]"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _avatar_text(value, 512)
    if isinstance(value, (int, float)):
        try:
            return value if math.isfinite(float(value)) else None
        except (OverflowError, TypeError, ValueError):
            return None
    if isinstance(value, dict):
        return {_avatar_text(k, 64): _avatar_json_safe(v, depth + 1) for k,v in list(value.items())[:24]}
    if isinstance(value, (tuple, list)):
        return [_avatar_json_safe(v, depth + 1) for v in value[:24]]
    return "[unsupported_type]"

def _avatar_event_metadata(value: Any) -> Dict[str, Any]:
    clean = _avatar_json_safe(value if isinstance(value, dict) else {})
    if not isinstance(clean, dict):
        return {}
    while len(json.dumps(clean, ensure_ascii=False).encode('utf8')) > _AVATAR_MAX_EVENT_BYTES and clean:
        clean.pop(next(reversed(clean)))
    return clean

def _avatar_approved_render_context() -> bool:
    """Operator provisioned flags are prerequisites, never an execution approval receipt."""
    return (not _avatar_flag("SAFE_MODE") and
            _avatar_flag("AVATAR_RENDER_PROCESS_APPROVED") and
            _avatar_flag("AVATAR_RENDER_GOVERNANCE_VERIFIED"))

def _avatar_contained_file(path: str, root: str, extensions: tuple[str, ...]) -> bool:
    try:
        candidate = Path(path).expanduser().resolve(strict=True)
        base = Path(root).expanduser().resolve(strict=True)
        return candidate.is_file() and candidate.suffix.lower() in extensions and candidate.is_relative_to(base)
    except (OSError, ValueError, RuntimeError):
        return False



@dataclass
class AvatarState:
    """Persistent live-avatar physical state.

    This is presentation/embodiment state only.  It does not own cognition,
    governance, network authority, or physical robot actuation.
    """
    expression: str = "neutral"
    emotion: str = "neutral"
    life_state: str = "ready"
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    blink: float = 0.0
    eyelid_left: float = 0.0
    eyelid_right: float = 0.0
    brow_left: float = 0.0
    brow_right: float = 0.0
    mouth_shape: str = "rest"
    jaw_open: float = 0.0
    head_pitch: float = 0.0
    head_yaw: float = 0.0
    head_roll: float = 0.0
    shoulder_left: float = 0.0
    shoulder_right: float = 0.0
    chest_expansion: float = 0.0
    breathing_phase: float = 0.0
    speaking_energy: float = 0.0
    emotional_energy: float = 0.0
    fatigue: float = 0.0
    attention: str = "user"
    gesture: str = "none"
    neon_phase: float = 0.0
    neon_intensity: float = 0.25
    speaking: bool = False
    listening: bool = False
    thinking: bool = False
    active_events: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    revision: int = 0
    updated_monotonic: float = field(default_factory=time.monotonic)

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["schema"] = "SarahMemory.avatar.state.live.v2"
        out["execution_authority"] = False
        return out


_AVATAR_STATE_LOCK = threading.RLock()
_LIVE_AVATAR_STATE = AvatarState()
_AVATAR_STATE_NUMERIC_FIELDS = {
    "gaze_x": (-1.0, 1.0), "gaze_y": (-1.0, 1.0), "blink": (0.0, 1.0),
    "eyelid_left": (-1.0, 1.0), "eyelid_right": (-1.0, 1.0),
    "brow_left": (-1.0, 1.0), "brow_right": (-1.0, 1.0),
    "jaw_open": (0.0, 1.0), "head_pitch": (-30.0, 30.0),
    "head_yaw": (-45.0, 45.0), "head_roll": (-20.0, 20.0),
    "shoulder_left": (-1.0, 1.0), "shoulder_right": (-1.0, 1.0),
    "chest_expansion": (-1.0, 1.0), "breathing_phase": (-1000000.0, 1000000.0),
    "speaking_energy": (0.0, 1.0), "emotional_energy": (0.0, 1.0),
    "fatigue": (0.0, 1.0), "neon_phase": (-1000000.0, 1000000.0),
    "neon_intensity": (0.0, 1.0),
}


def _avatar_clamp(value: Any, low: float, high: float) -> float:
    # Bounded deterministic local fallback; never substitute a lower bound for a valid value.
    safe = _avatar_finite(value, low, low, high)
    try:
        from SarahMemoryLogicCalc import sml_clamp  # type: ignore
        result = float(sml_clamp(safe, low, high))
        return _avatar_finite(result, safe, low, high)
    except Exception:
        return safe


def get_live_avatar_state() -> Dict[str, Any]:
    """Return a thread-safe snapshot of the persistent AvatarState."""
    with _AVATAR_STATE_LOCK:
        return _LIVE_AVATAR_STATE.to_dict()


_AVATAR_LITE_STATE_KEYS = (
    "mode",
    "expression",
    "emotion",
    "speaking",
    "listening",
    "thinking",
    "busy",
    "diagnostics",
    "current_action",
    "life_state",
    "life_enabled",
    "sequence",
    "heartbeat_count",
    "booted_at",
    "updated_at",
    "last_interaction_at",
    "last_life_tick",
    "last_random_at",
    "locked_until",
    "last_success_at",
    "last_error_at",
    "idle_seconds",
    "night_mode",
    "current_image",
    "avatar_image",
    "avatar_image_url",
)


def build_avatar_lite_state(
    state: Optional[Dict[str, Any]] = None,
    *,
    current_image: str = "",
    avatar_image_url: str = "",
    source: str = "SarahMemoryAvatar",
) -> Dict[str, Any]:
    """Build the hot-path AvatarPanel state without manifest/spec payloads.

    The full avatar route remains the owner of heavy manifest, 3D spec, and
    asset-discovery contracts. This helper keeps poll/heartbeat responses small
    and contract-only: no shell, no network, no file writes, no authority grant.
    """
    src = dict(state or get_live_avatar_state())
    now = time.time()
    out: Dict[str, Any] = {
        "ok": True,
        "success": True,
        "schema": "SarahMemory.avatar.state.lite.v1",
        "detail": "lite",
        "source": str(source or "SarahMemoryAvatar"),
        "execution_authority": False,
    }
    for key in _AVATAR_LITE_STATE_KEYS:
        if key in src:
            out[key] = src.get(key)
    if current_image:
        out["current_image"] = current_image
        out["avatar_image"] = current_image
    if avatar_image_url:
        out["avatar_image_url"] = avatar_image_url
    out.setdefault("mode", "avatar_2d")
    out.setdefault("expression", str(out.get("emotion") or "neutral"))
    out.setdefault("emotion", str(out.get("expression") or "neutral"))
    out.setdefault("speaking", False)
    out.setdefault("listening", False)
    out.setdefault("thinking", False)
    out.setdefault("busy", False)
    out.setdefault("diagnostics", False)
    out.setdefault("current_action", "idle")
    out.setdefault("life_state", "ready")
    try:
        out["idle_seconds"] = max(0.0, now - float(out.get("last_interaction_at") or now))
    except Exception:
        out["idle_seconds"] = 0.0
    for key, value in list(out.items()):
        if isinstance(value, str):
            out[key] = _avatar_text(value, 512)
        elif isinstance(value, (dict, list, tuple)):
            out[key] = _avatar_json_safe(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                if not math.isfinite(float(value)):
                    out[key] = 0.0
            except (TypeError, OverflowError):
                out[key] = 0.0
    return out


def update_live_avatar_state(updates: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    """Update presentation-only state. No physical, shell, or approval semantics."""
    change = dict(updates or {}) if isinstance(updates, dict) else {}
    change.update(kwargs)
    with _AVATAR_STATE_LOCK:
        changed = False
        for key, value in change.items():
            if key in {"schema", "execution_authority", "revision", "updated_monotonic"} or not hasattr(_LIVE_AVATAR_STATE, key):
                continue
            if key in _AVATAR_STATE_NUMERIC_FIELDS:
                low, high = _AVATAR_STATE_NUMERIC_FIELDS[key]
                prev = getattr(_LIVE_AVATAR_STATE, key)
                value = _avatar_finite(value, float(prev), low, high)
            elif key in {"speaking", "listening", "thinking"}:
                value = _avatar_bool(value)
            elif key == "active_events":
                if not isinstance(value, dict):
                    continue
                clean_events = {}
                for event_key, event_value in list(value.items())[-_AVATAR_MAX_EVENTS:]:
                    clean_events[_avatar_text(event_key, 80)] = _avatar_event_metadata(event_value)
                value = clean_events
            else:
                value = _avatar_text(value)
            if getattr(_LIVE_AVATAR_STATE, key) != value:
                setattr(_LIVE_AVATAR_STATE, key, value)
                changed = True
        if changed:
            _LIVE_AVATAR_STATE.revision += 1
            _LIVE_AVATAR_STATE.updated_monotonic = time.monotonic()
        return _LIVE_AVATAR_STATE.to_dict()


def resolve_live_avatar_event(event: str, *, current_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve a high-level UI/API avatar event into CORE-owned state updates.

    This function owns embodiment semantics only. It does not grant cognition,
    execution, device, network, or filesystem authority. API/UI bridges should
    call this resolver instead of inventing their own event mappings.
    """
    evt = str(event or "").strip().lower()
    now = time.time()
    current = dict(current_state or {})
    updates: Dict[str, Any] = {}
    lock_seconds = 0.0
    mark_interaction = False

    if evt in {"boot", "startup", "hello", "greeting"}:
        updates.update({
            "busy": False,
            "thinking": False,
            "diagnostics": False,
            "current_action": "boot_greeting",
            "expression": "hello",
            "emotion": "hello",
            "life_state": "boot_greeting",
        })
        lock_seconds = 5.0
    elif evt in {"success", "correct", "complete", "completed", "done", "ok", "approved"}:
        updates.update({
            "busy": False,
            "thinking": False,
            "diagnostics": False,
            "current_action": "success",
            "expression": "success",
            "emotion": "success",
            "life_state": "success",
            "last_success_at": now,
        })
        lock_seconds = 4.0
    elif evt in {"thumbs_up", "approval", "confirmed", "good"}:
        updates.update({
            "current_action": "thumbs_up",
            "expression": "thumbs_up",
            "emotion": "thumbs_up",
            "life_state": "success",
            "last_success_at": now,
        })
        lock_seconds = 4.0
    elif evt in {"error", "failed", "failure", "confused"}:
        updates.update({
            "current_action": "error",
            "expression": "concerned",
            "emotion": "concerned",
            "life_state": "error",
            "last_error_at": now,
        })
        lock_seconds = 4.0
    elif evt in {"diagnostics", "diagnostic", "self_check", "self_diagnostics"}:
        updates.update({
            "diagnostics": True,
            "current_action": "diagnostics",
            "expression": "serious_focus",
            "emotion": "serious_focus",
            "life_state": "diagnostics",
        })
        lock_seconds = 3.0
    elif evt in {"busy", "working", "processing"}:
        updates.update({
            "busy": True,
            "current_action": "busy",
            "expression": "pondering",
            "emotion": "pondering",
            "life_state": "busy",
        })
        lock_seconds = 3.0
    elif evt in {"idle", "ready", "reset"}:
        updates.update({
            "busy": False,
            "diagnostics": False,
            "thinking": False,
            "current_action": "idle",
            "expression": "ready",
            "emotion": "ready",
            "life_state": "ready",
        })
        lock_seconds = 1.0

    if updates:
        mark_interaction = True
    if updates.get("speaking") is False and current.get("current_action") == "speaking":
        updates.setdefault("current_action", "ready")
        updates.setdefault("expression", "ready")
        updates.setdefault("emotion", "ready")
        updates.setdefault("life_state", "ready")
        lock_seconds = max(lock_seconds, 1.5)
    if updates.get("listening") is False and current.get("current_action") == "listening":
        updates.setdefault("current_action", "ready")
        updates.setdefault("expression", "ready")
        updates.setdefault("emotion", "ready")
        updates.setdefault("life_state", "ready")
        lock_seconds = max(lock_seconds, 1.5)

    return {
        "ok": bool(updates),
        "schema": "SarahMemory.avatar.event_resolution.v1",
        "event": evt,
        "updates": updates,
        "lock_seconds": lock_seconds,
        "mark_interaction": mark_interaction,
        "execution_authority": False,
        "owner": "SarahMemoryAvatar",
    }


def queue_live_avatar_event(event_type: str, *, intensity: float = 1.0, duration_seconds: float = 1.0, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Bounded transient visual event; no device action or approval change."""
    event_key = _avatar_text(event_type or "event", 80).strip().lower() or "event"
    now = time.monotonic()
    meta = _avatar_event_metadata(metadata)
    with _AVATAR_STATE_LOCK:
        expired = [k for k,v in _LIVE_AVATAR_STATE.active_events.items()
                   if not isinstance(v, dict) or now - _avatar_finite(v.get("started_monotonic"), now, 0, now) > _avatar_finite(v.get("duration_seconds"), 1, 0.05, 30)]
        for key in expired:
            _LIVE_AVATAR_STATE.active_events.pop(key, None)
        if event_key not in _LIVE_AVATAR_STATE.active_events and len(_LIVE_AVATAR_STATE.active_events) >= _AVATAR_MAX_EVENTS:
            oldest = next(iter(_LIVE_AVATAR_STATE.active_events))
            _LIVE_AVATAR_STATE.active_events.pop(oldest, None)
            _AVATAR_RUNTIME_METRICS["event_rejections"] += 1
        _LIVE_AVATAR_STATE.active_events[event_key] = {
            "event_type": event_key, "started_monotonic": now,
            "intensity": _avatar_finite(intensity, 1.0, 0.0, 1.0),
            "duration_seconds": _avatar_finite(duration_seconds, 1.0, 0.05, 30.0),
            "metadata": meta,
        }
        _LIVE_AVATAR_STATE.revision += 1
        _LIVE_AVATAR_STATE.updated_monotonic = now
        return _LIVE_AVATAR_STATE.to_dict()



def _avatar_event_float(value: Any, default: float = 0.0, low: float = 0.0, high: float = 1.0) -> float:
    return _avatar_finite(value, default, low, high)


def _avatar_event_clean(value: Any, default: str = "") -> str:
    return _avatar_text(value if value is not None else default).strip()


def apply_sml_avatar_event_packet(packet: Dict[str, Any], *, source: str = "SarahMemoryAvatar.apply_sml_avatar_event_packet") -> Dict[str, Any]:
    """Interpret a SML directive solely as a presentation event.

    Source claims and confirmation flags are not authenticated by this module.
    An unverified completion signal cannot portray a verified success.
    """
    if not isinstance(packet, dict):
        return {"ok": False, "schema": "SarahMemory.avatar.event_apply.B08", "error": "packet_not_dict", "execution_authority": False}
    directive = packet.get("avatar_directive") if isinstance(packet.get("avatar_directive"), dict) else {}
    event = packet.get("event") if isinstance(packet.get("event"), dict) else {}
    if not directive:
        return {"ok": False, "schema": "SarahMemory.avatar.event_apply.B08", "error": "missing_avatar_directive", "execution_authority": False}
    expression = _avatar_event_clean(directive.get("expression"), "attentive").lower() or "attentive"
    attention = _avatar_event_clean(directive.get("attention"), "event_source").lower() or "event_source"
    gesture = _avatar_event_clean(directive.get("gesture"), "subtle_acknowledge").lower() or "subtle_acknowledge"
    reaction_class = _avatar_event_clean(directive.get("reaction_class"), "runtime_observation").lower()
    intensity = _avatar_event_float(directive.get("intensity", directive.get("arousal", 0.35)), 0.35, 0.0, 1.0)
    duration = _avatar_event_float(directive.get("duration_seconds", 2.25), 2.25, 0.25, 12.0)
    # Not even source_verified=True is proof of an independent verification receipt.
    unverified_completion = expression in {"success", "approved", "complete", "confirmed"} or reaction_class in {"mission_complete", "approval_granted"}
    if unverified_completion:
        expression, reaction_class = "attentive", "runtime_observation"
    updates: Dict[str, Any] = {
        "expression": expression, "emotion": expression, "attention": attention, "gesture": gesture,
        "emotional_energy": intensity,
        "neon_intensity": max(0.20, min(1.0, 0.25 + intensity * 0.55)),
        "thinking": reaction_class in {"mission_progress", "runtime_observation"},
        "listening": attention in {"user", "event_source"},
    }
    if expression in {"surprised", "alert_focused"}:
        updates.update({"brow_left": 0.45, "brow_right": 0.45, "eyelid_left": 0.25, "eyelid_right": 0.25, "head_pitch": -2.0})
    elif expression == "concerned":
        updates.update({"brow_left": -0.25, "brow_right": -0.25, "head_pitch": 2.0})
    elif expression == "pleased":
        updates.update({"brow_left": 0.15, "brow_right": 0.15, "head_pitch": -1.0})
    state = update_live_avatar_state(updates)
    event_type = _avatar_event_clean(event.get("event_type"), "runtime_event").lower().replace(" ", "_") or "runtime_event"
    queued = queue_live_avatar_event(event_type, intensity=intensity, duration_seconds=duration,
              metadata={"source": _avatar_event_clean(source, "unknown"),
                        "packet_schema": _avatar_event_clean(packet.get("schema")),
                        "reaction_class": reaction_class, "source_verified": False,
                        "speech_allowed": False, "speech_text": ""})
    log_avatar_event("SML Avatar Event B08", f"event={event_type}; expression={expression}; source={_avatar_event_clean(source)}; verified=False")
    return {"ok": True, "schema": "SarahMemory.avatar.event_apply.B08", "event_type": event_type,
            "reaction_class": reaction_class, "applied_expression": expression,
            "state": state, "queued_state": queued, "source_verified": False,
            "unverified_completion_downgraded": bool(unverified_completion), "execution_authority": False}

def prune_live_avatar_events(now_monotonic: Optional[float] = None) -> Dict[str, Any]:
    now = _avatar_finite(now_monotonic if now_monotonic is not None else time.monotonic(), time.monotonic(), 0.0, 1e20)
    with _AVATAR_STATE_LOCK:
        expired = []
        for key, evt in list(_LIVE_AVATAR_STATE.active_events.items()):
            start = _avatar_finite(evt.get("started_monotonic") if isinstance(evt, dict) else None, now, 0.0, 1e20)
            duration = _avatar_finite(evt.get("duration_seconds") if isinstance(evt, dict) else None, 0.0, 0.0, 30.0)
            if now - start > duration:
                expired.append(key)
        for key in expired:
            _LIVE_AVATAR_STATE.active_events.pop(key, None)
        if expired:
            _LIVE_AVATAR_STATE.revision += 1
            _LIVE_AVATAR_STATE.updated_monotonic = now
        return _LIVE_AVATAR_STATE.to_dict()


def _sm_avatar_base_dir() -> str:
    """Resolve SarahMemory install root for avatar resources without cwd dependency."""
    try:
        base = getattr(config, "BASE_DIR", None)
        if base:
            return os.path.abspath(os.path.expanduser(str(base)))
    except Exception:
        pass
    try:
        here = os.path.abspath(__file__)
        parent = os.path.dirname(here)
        if os.path.basename(parent).lower() == "core":
            return os.path.abspath(os.path.dirname(parent))
    except Exception:
        pass
    return os.path.abspath(os.getcwd())


def _sm_avatar_resources_dir() -> str:
    try:
        path = getattr(config, "RESOURCES_DIR", None)
        if path:
            return os.path.abspath(os.path.expanduser(str(path)))
    except Exception:
        pass
    return os.path.join(_sm_avatar_base_dir(), "resources")


# Setup logger (must exist before any init functions use it)
logger = logging.getLogger("SarahMemoryAvatar")
logger.setLevel(logging.DEBUG)
_handler = logging.NullHandler()
_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not logger.hasHandlers():
    logger.addHandler(_handler)

def _ensure_avatar_state_table():
    """Initialize legacy SQLite schema on first use only, never during import."""
    global _AVATAR_DB_READY
    if _AVATAR_DB_READY and os.path.isfile(os.path.join(DATASETS_DIR, DB_FILENAME)):
        return
    with _AVATAR_DB_INIT_LOCK:
        if _AVATAR_DB_READY and os.path.isfile(os.path.join(DATASETS_DIR, DB_FILENAME)):
            return
        try:
            db_path = os.path.join(DATASETS_DIR, DB_FILENAME)
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            with closing(sqlite3.connect(db_path, timeout=5)) as conn, conn:
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute("CREATE TABLE IF NOT EXISTS avatar_state (id INTEGER PRIMARY KEY, emotion TEXT, expression TEXT, state TEXT)")
                conn.execute("INSERT OR IGNORE INTO avatar_state (id, emotion, expression, state) VALUES (1, 'neutral', 'neutral', 'neutral')")
            _AVATAR_DB_READY = True
        except Exception as exc:
            _AVATAR_RUNTIME_METRICS["db_failures"] += 1
            _avatar_record_error("state_table", exc)
            logger.warning("avatar_state table unavailable: %s", exc)

# Lifecycle: DB initialization is lazy; imports never create/write SQLite files.




# ─────────────────────────────────────────────
# 3D AVATAR ORGAN SOFTWARE BODY PROFILE
# ─────────────────────────────────────────────
# This profile is for visual embodiment and future mapping only.  It does not
# authorize physical robot actuation.  A later Avatar-to-MSDC bridge must route
# through OperatorCore / MSDC / Safety / Assurance / Compare / user approval.
AVATAR_3D_DIR = getattr(config, "AVATAR_3D_DIR", os.path.join(_sm_avatar_resources_dir(), "avatars", "3D"))
AVATAR_3D_DEFAULT_MODEL = "SarahMemoryAvatar_RigBootstrap.glb"
AVATAR_BODY_PROFILE = {
    "profile_name": "SarahMemory_default_humanoid",
    "height_m": 1.68,
    "weight_kg_visual_target": 58.0,
    "head_height_ratio": 0.132,
    "shoulder_width_ratio": 0.245,
    "hip_width_ratio": 0.190,
    "leg_length_ratio": 0.515,
    "arm_span_ratio": 1.0,
    "center_of_mass_y_ratio": 0.55,
    "rig_units": "meters",
    "biological_plausibility": "visual_anatomy_reference_only",
    "robot_mapping_status": "future_bridge_required_no_physical_actuation_here",
}


def get_avatar_3d_body_profile():
    return copy.deepcopy(AVATAR_BODY_PROFILE)


def get_avatar_3d_model_path():
    """Resolve only GLB assets physically under the configured avatar asset root."""
    base = os.path.abspath(AVATAR_3D_DIR)
    for rel in (AVATAR_3D_DEFAULT_MODEL, os.path.join("default", AVATAR_3D_DEFAULT_MODEL),
                "sarahmemory_3d_avatar.glb", "sarahmemory_happy_face_ball.glb"):
        candidate = os.path.abspath(os.path.join(base, rel))
        if _avatar_contained_file(candidate, base, (".glb",)):
            return candidate
    # Preserve the legacy absent-model candidate contract, but never claim it exists.
    return os.path.join(base, AVATAR_3D_DEFAULT_MODEL)


def get_avatar_3d_runtime_spec():
    """Return AvatarPanel-compatible 3D runtime spec."""
    model_path = get_avatar_3d_model_path()
    exists = _avatar_contained_file(model_path, AVATAR_3D_DIR, (".glb",))
    model_file = os.path.basename(model_path)
    return {
        "renderMode": "gltf_model" if exists else "gold_standard_avatar",
        "modelUrl": f"/api/avatar/3d/{model_file}" if exists else "",
        "modelFile": model_file if exists else "",
        "loaderState": "3D_READY_GOLDSTANDARD_EMBODIED_ENTITY" if exists else "3D_FAILED_FALLBACK_GOLD_REFERENCE",
        "fallbackReason": "" if exists else "avatar_3d_model_missing_or_unavailable",
        "governance": {
            "visual_only": True,
            "physical_actuation_allowed": False,
            "msdc_dispatch_required_for_robot_body": True,
            "user_authority_required_for_embodied_action": True,
        },
        "pose": "stand",
        "gesture": "none",
        "lookAt": {"x": 0, "y": 1.45, "z": 0},
        "expression": get_avatar_emotion() if "get_avatar_emotion" in globals() else "neutral",
        "speaking": False,
        "listening": False,
        "quality": "goldstandard_entity",
        "fpsCap": 60,
        "lightingProfile": "high_end",
        "shadowQuality": "high",
                "animationAuthority": "visual_only_no_msdc_no_operator_action",
                "avatarEyeCameraAnchor": "Sarah_AvatarEye_Center",
        "stageOffsetY": -1.18,
        "avatarOffsetY": -0.52,
        "useRuntimeStage": True,
        "materialMode": "high_end",
        "runtimeVisualPriority": "gltf_model",
        "meshFallbackUrl": f"/api/avatar/3d/{model_file}" if exists else "",
        "forceMeshRuntime": bool(exists),
        "goldReferenceOnly": True,
        "authoringPolyTarget": 12000000,
        "runtimeLod": "goldstandard_entity",
        "productionAssetContract": "SARAHMEMORY_GOLDSTANDARD_EMBODIED_ENTITY_GLB_V1",
        "blueprintConstructReady": True,
        "visualEffectsMode": "secondary",
        "goldStandardScale": 1.0,
        "goldStandardYOffset": 0,
        "goldStandardPanelBottomPx": 58,
        "goldStandardPanelHeightPct": 92,
        "bodyProfile": get_avatar_3d_body_profile(),
        "source": "resources/avatars/3D",
        "robotBridge": {
            "enabled": False,
            "future_owner": "AvatarToMSDCBridge",
            "physical_actuation_allowed_here": False,
        },
    }


# ─────────────────────────────────────────────
# 2D AVATAR CONTROL
# ─────────────────────────────────────────────
EMOTION_EXPRESSIONS = {
    "joy": "😊",
    "anger": "😠",
    "fear": "😨",
    "trust": "🤝",
    "surprise": "😲",
    "neutral": "😐",
    "thinking": "🤔"
}
MOOD_MAP = {
    'joy': '😊',
    'fear': '😨',
    'trust': '😌',
    'anger': '😠',
    'surprise': '😲',
    'neutral': '😐',
    "thinking": "🤔"
    # Add more mappings as needed
}


def _hardware_tier_rating() -> str:
    """Return current tier_rating from SarahMemoryGlobals.hardware_score() (best-effort)."""
    try:
        hs_fn = getattr(config, "hardware_score", None)
        if callable(hs_fn):
            hs = hs_fn() or {}
            tr = str(hs.get("tier_rating") or "").strip()
            return tr if tr else "Unknown"
    except Exception:
        pass
    return "Unknown"

def log_avatar_event(event, details):
    """Opt-in bounded visual diagnostics, never a governance audit receipt."""
    if not _avatar_flag("AVATAR_EVENT_LOGGING_ENABLED", False):
        return
    with _AVATAR_LOG_LOCK:
        try:
            db_path = os.path.join(DATASETS_DIR, DB_FILENAME)
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            with closing(sqlite3.connect(db_path, timeout=5)) as conn, conn:
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute("CREATE TABLE IF NOT EXISTS avatar_events (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, event TEXT, details TEXT)")
                conn.execute("INSERT INTO avatar_events (timestamp, event, details) VALUES (?, ?, ?)",
                             (datetime.now().isoformat(), _avatar_text(event, 96), _avatar_text(details, 512)))
                # Bounded retention, not a security ledger; no repeated full-table scan.
                cap = 2048
                conn.execute("DELETE FROM avatar_events WHERE id IN (SELECT id FROM avatar_events ORDER BY id DESC LIMIT -1 OFFSET ?)", (cap,))
            _AVATAR_RUNTIME_METRICS["db_writes"] += 1
        except Exception as exc:
            _AVATAR_RUNTIME_METRICS["db_failures"] += 1
            _avatar_record_error("event_log", exc)
            logger.warning("Avatar event logging unavailable: %s", exc)

def _avatar_db_value(column: str, default: str = "neutral") -> str:
    if column not in {"state", "emotion", "expression"}:
        return default
    _ensure_avatar_state_table()
    try:
        with closing(sqlite3.connect(os.path.join(DATASETS_DIR, DB_FILENAME), timeout=5)) as conn, conn:
            row = conn.execute(f"SELECT {column} FROM avatar_state WHERE id=1").fetchone()
        _AVATAR_RUNTIME_METRICS["db_reads"] += 1
        return _avatar_text(row[0]) if row and row[0] else default
    except Exception as exc:
        _AVATAR_RUNTIME_METRICS["db_failures"] += 1
        _avatar_record_error("state_read", exc)
        return default

def _avatar_db_set(column: str, value: Any) -> bool:
    if column not in {"state", "emotion", "expression"}:
        return False
    _ensure_avatar_state_table()
    try:
        with closing(sqlite3.connect(os.path.join(DATASETS_DIR, DB_FILENAME), timeout=5)) as conn, conn:
            conn.execute(f"UPDATE avatar_state SET {column}=? WHERE id=1", (_avatar_text(value),))
        _AVATAR_RUNTIME_METRICS["db_writes"] += 1
        return True
    except Exception as exc:
        _AVATAR_RUNTIME_METRICS["db_failures"] += 1
        _avatar_record_error("state_write", exc)
        return False

def get_dominant_emotion():
    emotions = load_emotional_state()
    if not emotions:
        logger.warning("No emotional state data available. Defaulting to neutral.")
        log_avatar_event("Get Dominant Emotion", "No emotion data; defaulting to neutral.")
        return "neutral"
    dominant = max(emotions, key=emotions.get)
    logger.info(f"Dominant emotion: {dominant} ({emotions[dominant]})")
    log_avatar_event("Get Dominant Emotion", f"Dominant: {dominant} with value {emotions[dominant]}")
    return dominant

def draw_2d_avatar(expression, extra_overlay=None):
    """Render an image without launching an external viewer or a GUI event loop."""
    if Image is None or ImageDraw is None or ImageFont is None:
        logger.warning("Pillow missing; 2D avatar render unavailable")
        return None
    face = EMOTION_EXPRESSIONS.get(str(expression or "neutral"), EMOTION_EXPRESSIONS["neutral"])
    img = Image.new("RGB", (300, 300), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font_file = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", "arial.ttf")
        font = ImageFont.truetype(font_file, 100) if os.path.isfile(font_file) else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    try:
        draw.text((90, 100), face, font=font, fill="black")
    except Exception:
        draw.text((90, 100), ":-)", font=font, fill="black")
    if extra_overlay:
        draw.text((10, 10), _avatar_text(extra_overlay, 60), font=font, fill="red")
    return img
#---------------Tkinter GUI Integration (if needed)------------------
def load_sprite_frames(sprite_sheet_path, frame_width, frame_height, tk_root=None):
    """Bound sprite slices; only create PhotoImages with an explicitly supplied Tk root."""
    if Image is None:
        return []
    try:
        width, height = int(frame_width), int(frame_height)
        if not 1 <= width <= 2048 or not 1 <= height <= 2048:
            return []
        path = Path(sprite_sheet_path).expanduser().resolve(strict=True)
        if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
            return []
        with Image.open(path) as source:
            if source.width * source.height > 16_000_000:
                return []
            count = 0
            frames = []
            for y in range(0, source.height - height + 1, height):
                for x in range(0, source.width - width + 1, width):
                    if count >= 256:
                        return frames
                    frame = source.crop((x, y, x + width, y + height))
                    if tk_root is not None and ImageTk is not None:
                        try:
                            frame = ImageTk.PhotoImage(frame, master=tk_root)
                        except Exception:
                            pass
                    frames.append(frame)
                    count += 1
            return frames
    except Exception as exc:
        _avatar_record_error("sprites", exc)
        return []

def animate_walk(self):
    frames = getattr(self, "walk_frames", None) or []
    if not frames or not getattr(self, "avatar_label", None):
        return
    self.current_frame = (int(getattr(self, "current_frame", -1)) + 1) % len(frames)
    self.avatar_label.configure(image=frames[self.current_frame])
    if getattr(self, "root", None) is not None:
        self.root.after(100, self.animate_walk)
def animate_flames(self):
    if Image is None or ImageSequence is None or ImageTk is None or not getattr(self, "root", None):
        return
    try:
        resource = os.path.join(_sm_avatar_resources_dir(), "flames.gif")
        if not _avatar_contained_file(resource, _sm_avatar_resources_dir(), (".gif",)):
            return
        with Image.open(resource) as im:
            self.flame_frames = [ImageTk.PhotoImage(frame.copy(), master=self.root)
                                 for i, frame in enumerate(ImageSequence.Iterator(im)) if i < 90]
        if self.flame_frames:
            self.current_flame_frame = 0
            self.update_flames()
    except Exception as exc:
        _avatar_record_error("flames", exc)

def update_flames(self):
    frames = getattr(self, "flame_frames", None) or []
    if not frames or not getattr(self, "background_label", None):
        return
    idx = int(getattr(self, "current_flame_frame", 0)) % len(frames)
    self.background_label.configure(image=frames[idx])
    self.current_flame_frame = (idx + 1) % len(frames)
    if getattr(self, "root", None) is not None:
        self.root.after(100, self.update_flames)
def animate_lip_sync(self, duration):
    """Nonblocking Tk frame animation; never sleep on the UI thread."""
    frames = getattr(self, "lip_sync_frames", None) or []
    root = getattr(self, "root", None)
    if not frames or root is None or not getattr(self, "avatar_label", None):
        return
    until = time.monotonic() + _avatar_finite(duration, 0.0, 0, 30)
    def tick(index=0):
        if time.monotonic() >= until:
            return
        frame = frames[index % len(frames)]
        self.avatar_label.configure(image=frame)
        self.avatar_label.image = frame
        root.after(100, tick, index + 1)
    tick()
def animate_body(self):
    frames = getattr(self, "body_frames", None) or []
    if not frames or not getattr(self, "body_label", None):
        return
    idx = (int(getattr(self, "current_body_frame", -1)) + 1) % len(frames)
    self.current_body_frame = idx
    self.body_label.config(image=frames[idx])
    if getattr(self, "root", None) is not None:
        self.root.after(100, self.animate_body)
def update_head_movement(self):
    frames = getattr(self, "head_frames", None) or []
    if not frames or not getattr(self, "head_label", None):
        return
    idx = max(0, min(len(frames) - 1, int(self.ai_decide_head_pose())))
    self.head_label.config(image=frames[idx])
    if getattr(self, "root", None) is not None:
        self.root.after(100, self.update_head_movement)

def ai_decide_head_pose(self):
    frames = getattr(self, "head_frames", None) or []
    if not frames:
        return 0
    # No fake perception: deterministic neutral pose until a verified visual pose exists.
    return len(frames) // 2
# ─────────────────────────────────────────────

def update_gaze_direction(audio_input, current_gaze):
    """Accept only explicit verified gaze coordinates; never fabricate an audio model."""
    try:
        x, y = current_gaze
        if isinstance(audio_input, dict) and audio_input.get("gaze_verified") is True:
            target = (float(audio_input["gaze_x"]), float(audio_input["gaze_y"]))
            return interpolate((float(x), float(y)), target, 0.1)
        return (float(x), float(y))
    except Exception:
        return current_gaze
def sprite_main_loop():
    """Legacy compatibility entry: GUI owner must supply its own frame lifecycle."""
    logger.info("Sprite loop controlled by AvatarPanel/UnifiedAvatarController; no standalone capture loop started.")
    return None
def update_avatar_head(gaze_direction):
    """Update CORE visual gaze only; GUI widgets remain owned by their panel."""
    try:
        x, y = gaze_direction
        return update_live_avatar_state(gaze_x=_avatar_finite(x, 0.0, -1, 1),
                                        gaze_y=_avatar_finite(y, 0.0, -1, 1))
    except Exception:
        return get_live_avatar_state()
def extract_features(audio_input):
    # Do not fabricate sensed audio/vision features.
    return []
def interpolate(current, target, factor):
    t = _avatar_finite(factor, 0.0, 0.0, 1.0)
    if isinstance(current, (tuple, list)) and isinstance(target, (tuple, list)) and len(current) == len(target):
        try:
            return tuple(float(a) + (float(b) - float(a)) * t for a,b in zip(current, target))
        except Exception:
            return current
    try:
        return float(current) + (float(target) - float(current)) * t
    except Exception:
        return current
def get_audio_input():
    """The voice subsystem, not Avatar, owns microphone acquisition."""
    return None
def perform_scan_capture():
    """Compatibility adapter. Camera/desktop capture requires a user-approved vision owner.

    The original Avatar organ never implemented a capture pipeline. Do not take a
    hidden screenshot, open a microphone, or invent an authorization path here.
    """
    raise RuntimeError("Avatar cannot initiate camera/desktop capture: configure the authorized Vision capture owner")

def ultra_detect_objects(frame):
    """Delegate already-supplied image frames to CORE's existing SOBJE detector."""
    if frame is None:
        return []
    try:
        from SarahMemorySOBJE import ultra_detect_objects as _detect
        result = _detect(frame)
        return result if isinstance(result, list) else []
    except Exception as exc:
        _avatar_record_error("object_detection", exc)
        return []

def set_avatar_expression(expression):
    """Persist and mirror the existing legacy expression setter."""
    ok = _avatar_db_set("expression", expression)
    if ok:
        update_live_avatar_state(expression=_avatar_text(expression or "neutral"))
    return ok
def get_avatar_state():
    return _avatar_db_value("state")
def set_avatar_state(state):
    if _avatar_db_set("state", state):
        update_live_avatar_state(life_state=_avatar_text(state or "neutral"))
def get_avatar_emotion():
    return _avatar_db_value("emotion")
def set_avatar_emotion(emotion):
    if _avatar_db_set("emotion", emotion):
        update_live_avatar_state(emotion=_avatar_text(emotion or "neutral"))
def get_avatar_expression():
    return _avatar_db_value("expression")
def update_avatar_expression(expression=None):
    """Synchronize emotion and return the existing emoji presentation contract."""
    if not expression:
        expression = get_dominant_emotion()
    expression = _avatar_text(expression or "neutral", 80)
    set_avatar_expression(expression)
    set_avatar_emotion(expression)
    log_avatar_event("Update Avatar Expression", f"Expression set to {expression}")
    # Drawing is explicitly opt-in to the caller; never launch viewer or renderer here.
    interact_with_gui(expression, EMOTION_EXPRESSIONS.get(expression, ""))
    trigger_3d_animation(expression)
    emotions = load_emotional_state()
    if not isinstance(emotions, dict) or not emotions:
        return "neutral"
    try:
        valid = [(str(k), float(v)) for k, v in emotions.items() if math.isfinite(float(v))]
        top_mood = max(valid, key=lambda x: x[1])[0] if valid else "neutral"
    except Exception:
        top_mood = "neutral"
    return MOOD_MAP.get(top_mood, MOOD_MAP["neutral"])

def simulate_lip_sync_async(duration=2.0):
    """Bounded background visual mouth simulator, not an audio transcription engine."""
    if not _AVATAR_LIPSYNC_LOCK.acquire(blocking=False):
        return
    def run():
        try:
            simulate_lip_sync(duration)
        finally:
            _AVATAR_LIPSYNC_LOCK.release()
    threading.Thread(target=run, name="AvatarLipSync", daemon=True).start()

def simulate_lip_sync(duration=2.0):
    """Visual-only bounded mouth motion; speech timing must come from Voice organ."""
    seconds = _avatar_finite(duration, 2.0, 0.0, 15.0)
    deadline = time.monotonic() + seconds
    count = 0
    update_live_avatar_state(speaking=True)
    try:
        while time.monotonic() < deadline and count < 150:
            update_live_avatar_state(jaw_open=0.45 if count % 2 == 0 else 0.05,
                                     mouth_shape="open" if count % 2 == 0 else "rest")
            count += 1
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    finally:
        update_live_avatar_state(speaking=False, jaw_open=0.0, mouth_shape="rest")

def interact_with_gui(mood, face_repr):
    logger.info(f"[GUI] Mood: {mood}, Face: {face_repr}")
    log_avatar_event("GUI Interaction", f"Mood: {mood} | Face: {face_repr}")

def trigger_3d_animation(emotion):
    """Legacy animation notification; never directly spawn an arbitrary script."""
    if _hardware_tier_rating().lower() == "poor":
        return None
    # AvatarPanel/BlenderAvatarBootstrap own actual visual processing and GLB export.
    # An expression update does not convey independent execution approval.
    log_avatar_event("3D Animation Requested", _avatar_text(emotion, 80))
    return None
    #---------------------------------- End of subprocess tinkering
 # SarahMemoryAvatar.py (add this below trigger_3d_animation)

def display_interactive_3d_avatar(filepath, engine="Blender"):
    """Bounded opt-in local Blender rendering, requiring externally verified approval.

    Configuration flags are prerequisites only; operator/SMGET must independently
    control who is permitted to set them, before enabling this legacy path.
    """
    if (not _avatar_approved_render_context() or
        not _avatar_flag("AVATAR_RENDER_USER_CONSENTED") or
        _hardware_tier_rating().lower() == "poor"):
        _AVATAR_RUNTIME_METRICS["render_denials"] += 1
        return False
    if str(engine).lower() != "blender":
        _AVATAR_RUNTIME_METRICS["render_denials"] += 1
        return False
    if not _avatar_contained_file(str(filepath), AVATAR_3D_DIR, (".glb", ".blend")):
        _AVATAR_RUNTIME_METRICS["render_denials"] += 1
        return False
    tools_dir = os.fspath(getattr(config, "TOOLS_DIR", "") or "")
    script = os.path.join(tools_dir, "render_avatar.py") if tools_dir else ""
    if not script or not _avatar_contained_file(script, tools_dir, (".py",)):
        _AVATAR_RUNTIME_METRICS["render_denials"] += 1
        return False
    executable = str(getattr(config, "BLENDER_EXECUTABLE", "") or "").strip()
    if not executable or not os.path.isabs(executable) or not os.path.isfile(executable):
        _AVATAR_RUNTIME_METRICS["render_denials"] += 1
        return False
    if not _AVATAR_RENDER_LOCK.acquire(blocking=False):
        return False
    try:
        # Do not treat Globals flags as an execution authorization receipt.
        # Invoke the existing OperatorCore -> CognitiveServices -> SecurityGovernor
        # -> AssuranceGate chain for this explicit domain-owned render operation.
        try:
            from SarahMemoryOperatorCore import authorize_external_domain_action  # type: ignore
            approval = authorize_external_domain_action(
                "Render the user-selected local avatar asset in Blender",
                origin="SarahMemoryAvatar.display_interactive_3d_avatar",
                proposed_action={
                    "action_type": "avatar.render.blender", "target": str(Path(filepath).resolve()),
                    "capability_name": "avatar.render.local", "risk_level": "TIER_2_BOUNDED_LOCAL_OPERATION",
                    "required_permissions": ["local.render.launch"],
                },
                meta={"expected_result": "Blender exits successfully after rendering the selected avatar asset",
                      "verification_checks": [{"name": "render_process_exit_zero", "required": True}],
                      "preconditions": ["asset_exists", "renderer_exists"],
                      "rollback_plan": [{"name": "terminate_render_process", "required": True}]},
                user_consented=True,
            )
        except Exception as exc:
            _avatar_record_error("render_authorization", exc)
            return False
        if not (isinstance(approval, dict) and approval.get("allow") is True and
                approval.get("domain_execution_authorized") is True and approval.get("authorization_owner")):
            _AVATAR_RUNTIME_METRICS["render_denials"] += 1
            return False
        environment = {k: v for k, v in os.environ.items() if k.upper() in
                       {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "LANG", "DISPLAY", "XAUTHORITY"}}
        selected = str(Path(filepath).resolve())
        script_path = str(Path(script).resolve())
        if Path(selected).suffix.lower() == ".blend":
            command = [executable, "--background", selected, "--python", script_path]
        else:
            # GLB is not a Blender project; pass the asset explicitly to the import script.
            command = [executable, "--background", "--python", script_path, "--", selected]
        subprocess.run(command, timeout=_AVATAR_RENDER_TIMEOUT_S, check=True, env=environment)
        _AVATAR_RUNTIME_METRICS["render_launches"] += 1
        return True
    except Exception as exc:
        _avatar_record_error("blender_render", exc)
        return False
    finally:
        _AVATAR_RENDER_LOCK.release()

if __name__ == '__main__':

    import sys
    logger.info("Testing Unified Avatar Engine...")
    args = sys.argv[1:]

    if '--test2d' in args:
        dominant = get_dominant_emotion()
        update_avatar_expression(dominant)

    elif '--test3d' in args:
        dominant = get_dominant_emotion()
        trigger_3d_animation(dominant)

    elif '--lipsync' in args:
        simulate_lip_sync_async(3)

    else:
        print("Usage: python SarahMemoryAvatar.py [--test2d | --test3d | --lipsync]")

# [PATCH v7.7.2] Avatar render instruction hook
def apply_render_instructions(instr: dict):
    """
    Map high-level avatar state (mood/personality) to internal state.
    instr example: {"mood":"happy","energy":0.7,"style":"neon"}
    """
    try:
        mood = (instr or {}).get("mood") or "neutral"
        set_avatar_state(mood)
    except Exception:
        pass


# -----------------------------------------------------------------------------
# SARAH_AVATAR_REM_STATE_V1
# Backend REM Sleep state persistence helpers. Additive and legacy-safe.
# -----------------------------------------------------------------------------
def _ensure_avatar_rem_table():
    """Create REM state schema once on first REM operation."""
    global _AVATAR_REM_DB_READY
    if _AVATAR_REM_DB_READY and os.path.isfile(os.path.join(DATASETS_DIR, DB_FILENAME)):
        return
    with _AVATAR_DB_INIT_LOCK:
        if _AVATAR_REM_DB_READY and os.path.isfile(os.path.join(DATASETS_DIR, DB_FILENAME)):
            return
        try:
            db_path = os.path.join(DATASETS_DIR, DB_FILENAME)
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            with closing(sqlite3.connect(db_path, timeout=5)) as conn, conn:
                conn.execute("PRAGMA busy_timeout=5000")
                conn.execute("CREATE TABLE IF NOT EXISTS avatar_rem_state (id INTEGER PRIMARY KEY, ts TEXT, phase TEXT, expression TEXT, reason TEXT, metadata_json TEXT)")
                conn.execute("INSERT OR IGNORE INTO avatar_rem_state (id, ts, phase, expression, reason, metadata_json) VALUES (1, ?, ?, ?, ?, ?)",
                             (datetime.now().isoformat(), "awake", "ready", "initialized", "{}"))
            _AVATAR_REM_DB_READY = True
        except Exception as exc:
            _AVATAR_RUNTIME_METRICS["db_failures"] += 1
            _avatar_record_error("rem_table", exc)
            logger.warning("avatar_rem_state table unavailable: %s", exc)

def set_avatar_rem_state(phase: str, expression: str = "sleepy", reason: str = "", metadata: dict | None = None) -> bool:
    """Preserve legacy REM schema; commit and mirror state only on DB success."""
    try:
        _ensure_avatar_rem_table()
        clean_phase = _avatar_text(phase or "awake", 80)
        clean_expression = _avatar_text(expression or "ready", 80)
        clean_meta = _avatar_event_metadata(metadata)
        with closing(sqlite3.connect(os.path.join(DATASETS_DIR, DB_FILENAME), timeout=5)) as conn, conn:
            conn.execute("UPDATE avatar_rem_state SET ts=?, phase=?, expression=?, reason=?, metadata_json=? WHERE id=1",
                         (datetime.now().isoformat(), clean_phase, clean_expression, _avatar_text(reason, 256), json.dumps(clean_meta, ensure_ascii=False)))
        _AVATAR_RUNTIME_METRICS["db_writes"] += 1
        update_live_avatar_state(life_state=clean_phase, expression=clean_expression, emotion=clean_expression)
        log_avatar_event("REM State", f"phase={clean_phase}; expression={clean_expression}")
        return True
    except Exception as exc:
        _AVATAR_RUNTIME_METRICS["db_failures"] += 1
        _avatar_record_error("rem_write", exc)
        return False

def get_avatar_rem_state() -> dict:
    try:
        _ensure_avatar_rem_table()
        with closing(sqlite3.connect(os.path.join(DATASETS_DIR, DB_FILENAME), timeout=5)) as conn, conn:
            row = conn.execute("SELECT ts, phase, expression, reason, metadata_json FROM avatar_rem_state WHERE id=1").fetchone()
        _AVATAR_RUNTIME_METRICS["db_reads"] += 1
        if not row:
            return {"phase": "awake", "expression": "ready", "reason": "missing_row"}
        try:
            meta = _avatar_event_metadata(json.loads(row[4] or "{}"))
        except (ValueError, TypeError, json.JSONDecodeError):
            meta = {}
        return {"ts": row[0], "phase": row[1], "expression": row[2], "reason": row[3], "metadata": meta}
    except Exception as exc:
        _AVATAR_RUNTIME_METRICS["db_failures"] += 1
        _avatar_record_error("rem_read", exc)
        return {"phase": "awake", "expression": "ready", "reason": "persistence_unavailable"}

# REM table is created only when a REM state API is used.

# ====================================================================
# END OF SarahMemoryAvatar.py v9.1.0a
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryAvatar',
    "version": "v9.1.0a-alpha-sml-0.2",
    "category": 'Execution',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['avatar', 'avatar_event_stream', 'execution'],
    "supported_missions": ['Conversation', 'Execution'],
    "supported_omega": ['Ω001', 'Ω070', 'Ω100'],
    "required_authority": ['Execute', 'Read'],
    "priority": 50,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryAvatar.py'},
}



def live_avatar_state_self_test() -> Dict[str, Any]:
    """No side effects on live UI or user state: exercise a disposable state copy."""
    with _AVATAR_STATE_LOCK:
        sample = copy.deepcopy(_LIVE_AVATAR_STATE)
    sample.jaw_open = _avatar_finite(0.25, 0.0, 0.0, 1.0)
    sample.gaze_x = _avatar_finite(0.10, 0.0, -1.0, 1.0)
    sample.speaking = True
    sample.active_events["blink"] = {"event_type": "blink", "intensity": 0.8, "duration_seconds": 0.2,
                                      "started_monotonic": time.monotonic(), "metadata": {}}
    after = sample.to_dict()
    checks = [
        {"name": "schema", "passed": after.get("schema") == "SarahMemory.avatar.state.live.v2"},
        {"name": "state_mutation", "passed": abs(float(after.get("jaw_open") or 0.0) - 0.25) < 1e-9},
        {"name": "event_queue", "passed": "blink" in (after.get("active_events") or {})},
        {"name": "execution_authority", "passed": after.get("execution_authority") is False},
    ]
    return {"ok": all(c["passed"] for c in checks), "checks": checks, "execution_authority": False}


def sml_get_metadata():
    return copy.deepcopy(SML_ORGAN_METADATA)


def sml_health():
    """Observational health only: never assert live rendering or device availability."""
    failures = int(_AVATAR_RUNTIME_METRICS["db_failures"])
    status = "Degraded" if failures else "Standby"
    return {"status": status, "availability": 0.0, "integrity": 0.0, "performance": 0.0,
            "reliability": 0.0, "confidence": 0.0, "latency_ms": 0.0,
            "stability": 0.0, "compatibility": 1.0,
            "notes": ["Unmeasured operational scores are 0.0 (unknown, not verified); no GPU or GUI probe", 
                      f"db_failures={failures}", f"render_denials={_AVATAR_RUNTIME_METRICS['render_denials']}"]}


def sml_diagnostics():
    health = sml_health()
    return {"status": health["status"], "component": "SarahMemoryAvatar", "sml_adapter": True,
            "metadata": sml_get_metadata(), "health": health,
            "runtime": {**_AVATAR_RUNTIME_METRICS, "active_events": len(_LIVE_AVATAR_STATE.active_events)}}


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryAvatar', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---
