"""--==The SarahMemory Project==--
File: SarahMemoryRhythmCognition.py
Part of the SarahMemory Companion AI-bot Platform
Version: v9.0.0
Date: 2026-06-08
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

RHYTHM COGNITION ORGAN v9.0.0
=============================

PURPOSE:
- Core cadence / rhythm / tempo organ for SarahMemory AiOS.
- Converts adaptive emotional state, urgency, speech pressure, music/rhythm metadata,
  task context, safety mode, and system resource pressure into bounded cadence packets.
- Controls cognitive pacing for Living Loop, CognitiveThinker, AiFunctions, personality
  energy, and embodied motion suggestions without becoming an execution authority.
- Prevents cognitive thrashing by providing cooldowns, token budgets, backoff, and
  bounded loop recommendations.
- Supports humanoid robotics by producing rhythm-informed MotionIntent suggestions for
  SarahMemoryMSDC.py while preserving OperatorCore / SMGET / SafetyPolicies authority.

CORE DOCTRINE:
- Rhythm may modulate cadence.
- Rhythm may reduce thrashing.
- Rhythm may shape personality pace and embodied expression.
- Rhythm may NOT authorize actions.
- Rhythm may NOT bypass SMGET, SafetyPolicies, AssuranceGate, OperatorCore, or MSDC.
- Emotion and urgency may increase priority/cadence, but safety determines max pace.
- Music can influence timing/expression, never torque/collision rules/human-contact authority.
- Default behavior is RAM-first with throttled, explicit persistence only.

INTEGRATION POINTS:
- SarahMemoryAdaptive.py        -> emotional/resource basis
- SarahMemoryPersonality.py     -> personality energy/formality/verbosity profile
- SarahMemoryCognitiveThinker.py-> possibility/reflection cadence consumer
- SarahMemoryAiFunctions.py     -> agent/task-loop cadence consumer
- SarahMemoryOptimization.py    -> anti-thrash/resource doctrine alignment
- SarahMemoryMSDC.py            -> future body/device manager consumer of MotionIntentPacket
- OperatorCore / SMGET          -> hard authority gates for real-world execution

===============================================================================
"""
from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "rhythm_cognition_organ"
# CATEGORY = "cognition"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "rhythm_cognition"
# HARDWARE_DOMAIN = "audio_robotics_optional"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "rhythm_cognition"
# FAMILY = "core_cognition"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Core cadence organ. Produces governed rhythm/cadence/personality/motion packets. No action authority; prevents thrashing and supports embodied rhythm suggestions."
# --- SARAHMETA END ---

import json
import logging
import math
import os
import sqlite3
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None  # type: ignore

try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:
    psutil = None  # type: ignore
    _HAS_PSUTIL = False

logger = logging.getLogger("SarahMemoryRhythmCognition")
logger.setLevel(logging.DEBUG if bool(getattr(config, "DEBUG_MODE", False)) else logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.propagate = False

MODULE_NAME = "SarahMemoryRhythmCognition"
MODULE_VERSION = "9.0.0"
_DB_NAME = "rhythm_cognition.db"
_JSON_NAME = "rhythm_cognition_snapshot.json"

RHYTHM_STILL = "STILL"
RHYTHM_CALM = "CALM"
RHYTHM_FOCUSED = "FOCUSED"
RHYTHM_BUILD = "BUILD"
RHYTHM_DEBUG = "DEBUG"
RHYTHM_CREATIVE = "CREATIVE"
RHYTHM_REM = "REM"
RHYTHM_URGENT_ASSIST = "URGENT_ASSIST"
RHYTHM_EMERGENCY = "EMERGENCY"
RHYTHM_SAFE = "SAFE"

MOTION_STILL = "still"
MOTION_IDLE_SWAY = "idle_sway"
MOTION_HEAD_BOB = "head_bob"
MOTION_HAND_TAP = "hand_tap"
MOTION_SLOW_DANCE = "slow_dance"
MOTION_DANCE = "dance"
MOTION_WALK_PACE_SYNC = "walk_pace_sync"
MOTION_AVATAR_ONLY = "avatar_only"
MOTION_SAFE_STOP = "safe_stop"

URGENT_WORDS = frozenset({
    "hurry", "quick", "quickly", "fast", "faster", "urgent", "urgently", "asap",
    "immediately", "now", "rush", "run", "go", "move", "get there", "help", "emergency",
    "danger", "fire", "medical", "collision", "stop", "abort", "critical", "life",
})

CALM_WORDS = frozenset({"slow", "slowly", "calm", "careful", "carefully", "gentle", "soft", "quiet"})
DEBUG_WORDS = frozenset({"debug", "diagnose", "verify", "audit", "inspect", "trace", "test", "failure", "error"})
CREATIVE_WORDS = frozenset({"music", "song", "art", "lyrics", "creative", "imagine", "story", "dance", "avatar"})
BUILD_WORDS = frozenset({"build", "patch", "code", "write", "implement", "create", "update", "fix"})

_LOCK = threading.RLock()
_LAST_PACKET: Dict[str, Any] = {}
_LAST_PACKET_TS = 0.0
_PACKET_CACHE_TTL = 0.75
_THROTTLE_STATE: Dict[str, Dict[str, Any]] = {}
_EVENT_LAST_TS: Dict[str, float] = {}


def _now_iso() -> str:
    return datetime.now().isoformat()


def _base_dir() -> str:
    try:
        candidate = str(getattr(config, "BASE_DIR", "") or "").strip()
        return candidate or os.getcwd()
    except Exception:
        return os.getcwd()


def _data_dir() -> str:
    try:
        return str(getattr(config, "DATA_DIR", os.path.join(_base_dir(), "data")))
    except Exception:
        return os.path.join(_base_dir(), "data")


def _datasets_dir() -> str:
    try:
        return str(getattr(config, "DATASETS_DIR", os.path.join(_data_dir(), "memory", "datasets")))
    except Exception:
        return os.path.join(_data_dir(), "memory", "datasets")


def _settings_dir() -> str:
    try:
        return str(getattr(config, "SETTINGS_DIR", os.path.join(_data_dir(), "settings")))
    except Exception:
        return os.path.join(_data_dir(), "settings")


def _db_path() -> str:
    return os.path.join(_datasets_dir(), _DB_NAME)


def _snapshot_path() -> str:
    return os.path.join(_settings_dir(), _JSON_NAME)


def _ensure_dirs() -> None:
    for p in (_datasets_dir(), _settings_dir()):
        try:
            os.makedirs(p, exist_ok=True)
        except Exception:
            pass


def _safe_float(value: Any, default: float = 0.0, lo: Optional[float] = None, hi: Optional[float] = None) -> float:
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return default
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        return v
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    try:
        v = int(float(value))
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        return v
    except Exception:
        return default


def _flag(name: str, default: bool = False) -> bool:
    try:
        return bool(getattr(config, name, default))
    except Exception:
        return default


def _words(text: Any) -> List[str]:
    s = str(text or "").lower()
    out: List[str] = []
    token = ""
    for ch in s:
        if ch.isalnum() or ch in "_-":
            token += ch
        else:
            if token:
                out.append(token)
                token = ""
    if token:
        out.append(token)
    return out


def _contains_phrase(text: str, phrases: frozenset) -> bool:
    low = str(text or "").lower()
    toks = set(_words(low))
    for p in phrases:
        if " " in p:
            if p in low:
                return True
        elif p in toks:
            return True
    return False


def _system_pressure() -> Dict[str, Any]:
    cpu = 0.0
    memory = 0.0
    disk = 0.0
    tier = "unknown"
    try:
        if _HAS_PSUTIL and psutil is not None:
            cpu = float(psutil.cpu_percent(interval=None))
            memory = float(psutil.virtual_memory().percent)
            try:
                disk = float(psutil.disk_usage(_base_dir()).percent)
            except Exception:
                disk = 0.0
    except Exception:
        pass

    load = max(cpu, memory)
    if load >= 88.0:
        tier = "critical"
    elif load >= 75.0:
        tier = "high"
    elif load >= 55.0:
        tier = "moderate"
    else:
        tier = "normal"
    return {"cpu": round(cpu, 2), "memory": round(memory, 2), "disk": round(disk, 2), "pressure_tier": tier}


def _adaptive_basis(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = dict(context or {})
    basis: Dict[str, Any] = {
        "dominant_emotion": "neutral",
        "emotional_balance": 0.0,
        "engagement": 0.4,
        "openness": 0.6,
        "intensity": 0.2,
        "adaptive_mode": "balanced",
        "source": "fallback",
    }

    supplied = ctx.get("adaptive_basis") or ctx.get("emotion_basis") or ctx.get("emotional_metrics")
    if isinstance(supplied, dict):
        basis.update(supplied)
        basis["source"] = str(supplied.get("source") or "context")
        return basis

    try:
        import SarahMemoryAdaptive as _Adaptive  # type: ignore
        fn = getattr(_Adaptive, "get_adaptive_rhythm_basis", None)
        if callable(fn):
            data = fn(ctx)
            if isinstance(data, dict):
                basis.update(data)
                basis["source"] = str(data.get("source") or "SarahMemoryAdaptive.get_adaptive_rhythm_basis")
                return basis
        fn2 = getattr(_Adaptive, "get_emotional_metrics", None)
        if callable(fn2):
            data = fn2()
            if isinstance(data, dict):
                basis.update(data)
                basis["dominant_emotion"] = data.get("dominant_emotion") or data.get("label") or basis["dominant_emotion"]
                basis["adaptive_mode"] = data.get("mode") or basis["adaptive_mode"]
                basis["source"] = "SarahMemoryAdaptive.get_emotional_metrics"
    except Exception as exc:
        basis["error"] = str(exc)

    return basis


def _personality_basis(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = dict(context or {})
    supplied = ctx.get("personality_profile") or ctx.get("personality_basis")
    if isinstance(supplied, dict):
        out = dict(supplied)
        out["source"] = str(out.get("source") or "context")
        return out
    out: Dict[str, Any] = {"energy": 0.5, "formality": 0.5, "verbosity": 0.5, "mood": "neutral", "source": "fallback"}
    try:
        import SarahMemoryPersonality as _Personality  # type: ignore
        fn = getattr(_Personality, "get_personality_rhythm_profile", None)
        if callable(fn):
            data = fn(ctx)
            if isinstance(data, dict):
                out.update(data)
                out["source"] = str(data.get("source") or "SarahMemoryPersonality.get_personality_rhythm_profile")
                return out
        fn2 = getattr(_Personality, "get_time_based_personality", None)
        if callable(fn2):
            data = fn2()
            if isinstance(data, dict):
                out["energy"] = _safe_float(data.get("energy"), 0.5, 0.0, 1.0)
                out["formality"] = _safe_float(data.get("formality"), 0.5, 0.0, 1.0)
                out["verbosity"] = _safe_float(data.get("verbosity"), 0.5, 0.0, 1.0)
                out["source"] = "SarahMemoryPersonality.get_time_based_personality"
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _music_basis(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = dict(context or {})
    meta = ctx.get("music") or ctx.get("audio") or ctx.get("rhythm") or {}
    if not isinstance(meta, dict):
        meta = {}
    bpm = _safe_float(meta.get("bpm") or meta.get("tempo_bpm"), 0.0, 0.0, 260.0)
    intensity = _safe_float(meta.get("intensity") or meta.get("beat_density") or meta.get("energy"), 0.0, 0.0, 1.0)
    mood = str(meta.get("mood") or meta.get("genre") or "").strip().lower()
    return {"bpm": bpm, "intensity": intensity, "mood": mood, "source": "context" if meta else "none"}


@dataclass
class RhythmCadencePacket:
    packet_id: str
    ts: str
    rhythm_mode: str
    tempo_bpm: int
    heartbeat_interval_sec: float
    thinker_interval_sec: float
    agent_step_interval_sec: float
    living_loop_interval_sec: float
    memory_write_budget_per_min: int
    max_inner_cycles: int
    backoff_multiplier: float
    personality_energy: float
    verification_bias: float
    creative_bias: float
    urgency_score: float
    importance_score: float
    speech_pressure: float
    thrash_guard: bool = True
    execution_authority: bool = False
    smget_required: bool = True
    safety_authority: str = "SMGET/SafetyPolicies/AssuranceGate/OperatorCore"
    source: str = "adaptive_state"
    adaptive_basis: Dict[str, Any] = field(default_factory=dict)
    personality_basis: Dict[str, Any] = field(default_factory=dict)
    system_pressure: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MotionIntentPacket:
    packet_id: str
    ts: str
    rhythm_mode: str
    motion_profile: str
    tempo_bpm: int
    requested_pace: str
    safe_pace: str
    gesture_interval_sec: float
    step_interval_sec: float
    movement_speed_scale: float
    force_scale: float
    max_motion_radius_cm: int
    locomotion_requested: bool
    locomotion_allowed: bool
    upper_body_allowed: bool
    face_expression_allowed: bool
    requires_msdc_validation: bool = True
    requires_operator_contract: bool = True
    execution_authority: bool = False
    smget_required: bool = True
    safety_notes: List[str] = field(default_factory=list)
    cadence_packet: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _score_urgency(text: str, context: Dict[str, Any], adaptive: Dict[str, Any], music: Dict[str, Any]) -> Tuple[float, List[str]]:
    notes: List[str] = []
    score = _safe_float(context.get("urgency") or context.get("urgency_score"), 0.0, 0.0, 1.0)
    if _contains_phrase(text, URGENT_WORDS):
        score = max(score, 0.68)
        notes.append("Urgency words detected in command/context.")
    if str(context.get("hazard_type") or context.get("emergency_type") or "").strip():
        score = max(score, 0.85)
        notes.append("Hazard/emergency context detected.")
    if bool(context.get("emergency") or context.get("human_risk") or context.get("person_at_risk")):
        score = max(score, 0.92)
        notes.append("Emergency or human-risk flag detected.")

    dominant = str(adaptive.get("dominant_emotion") or adaptive.get("label") or "neutral").lower()
    intensity = _safe_float(adaptive.get("intensity"), 0.2, 0.0, 1.0)
    if dominant in ("fear", "anger", "surprise", "anticipation") and intensity >= 0.55:
        score = max(score, min(1.0, 0.45 + intensity * 0.45))
        notes.append(f"Adaptive emotion '{dominant}' indicates elevated cadence pressure.")

    if _safe_float(music.get("bpm"), 0.0) >= 135.0 and _safe_float(music.get("intensity"), 0.0) >= 0.55:
        score = max(score, 0.52)
        notes.append("High-tempo/high-intensity rhythm metadata detected.")

    return round(score, 4), notes


def _score_speech_pressure(text: str, context: Dict[str, Any]) -> float:
    explicit = context.get("speech_pressure")
    if explicit is not None:
        return round(_safe_float(explicit, 0.0, 0.0, 1.0), 4)
    s = str(text or "")
    if not s:
        return 0.0
    bang = min(0.30, s.count("!") * 0.08)
    caps_words = [w for w in s.split() if len(w) > 2 and w.isupper()]
    caps = min(0.30, len(caps_words) * 0.06)
    urgency = 0.35 if _contains_phrase(s, URGENT_WORDS) else 0.0
    return round(min(1.0, bang + caps + urgency), 4)


def _select_mode(text: str, context: Dict[str, Any], adaptive: Dict[str, Any], personality: Dict[str, Any], music: Dict[str, Any], system: Dict[str, Any], urgency: float) -> Tuple[str, List[str]]:
    notes: List[str] = []
    requested = str(context.get("rhythm_mode") or context.get("mode") or "").strip().upper()
    if requested in {RHYTHM_STILL, RHYTHM_CALM, RHYTHM_FOCUSED, RHYTHM_BUILD, RHYTHM_DEBUG, RHYTHM_CREATIVE, RHYTHM_REM, RHYTHM_URGENT_ASSIST, RHYTHM_EMERGENCY, RHYTHM_SAFE}:
        notes.append(f"Rhythm mode requested by context: {requested}.")
        return requested, notes

    if _flag("SAFE_MODE", False) or str(system.get("pressure_tier")) == "critical":
        notes.append("Safe/resource critical mode selected cadence SAFE.")
        return RHYTHM_SAFE, notes
    if bool(context.get("emergency") or context.get("human_risk") or context.get("person_at_risk")) or urgency >= 0.90:
        notes.append("Emergency cadence selected from verified/high urgency signals.")
        return RHYTHM_EMERGENCY, notes
    if urgency >= 0.62:
        notes.append("Urgent assist cadence selected.")
        return RHYTHM_URGENT_ASSIST, notes
    if bool(context.get("rem") or context.get("dream") or context.get("idle_reflection")):
        notes.append("REM/reflection cadence selected by context.")
        return RHYTHM_REM, notes
    if _contains_phrase(text, DEBUG_WORDS) or str(context.get("task_type") or "").lower() in ("debug", "diagnostics", "verification"):
        notes.append("Debug/diagnostic cadence selected.")
        return RHYTHM_DEBUG, notes
    if _contains_phrase(text, CREATIVE_WORDS) or str(context.get("task_type") or "").lower() in ("creative", "music", "avatar", "dance"):
        notes.append("Creative cadence selected.")
        return RHYTHM_CREATIVE, notes
    if _contains_phrase(text, BUILD_WORDS) or str(context.get("task_type") or "").lower() in ("build", "patch", "code"):
        notes.append("Build cadence selected.")
        return RHYTHM_BUILD, notes
    if _contains_phrase(text, CALM_WORDS):
        notes.append("Calm/safety cadence selected from wording.")
        return RHYTHM_CALM, notes
    if str(system.get("pressure_tier")) == "high":
        notes.append("Resource pressure high; calm cadence selected to prevent thrashing.")
        return RHYTHM_CALM, notes

    energy = _safe_float(personality.get("energy"), 0.5, 0.0, 1.0)
    engagement = _safe_float(adaptive.get("engagement"), 0.4, 0.0, 1.0)
    if energy < 0.25 and engagement < 0.25:
        notes.append("Low engagement/energy cadence selected STILL.")
        return RHYTHM_STILL, notes
    return RHYTHM_FOCUSED, notes


def _mode_parameters(mode: str, urgency: float, speech_pressure: float, personality: Dict[str, Any], music: Dict[str, Any], system: Dict[str, Any]) -> Dict[str, Any]:
    base: Dict[str, Dict[str, Any]] = {
        RHYTHM_STILL: {"bpm": 48, "heartbeat": 8.0, "thinker": 30.0, "agent": 3.5, "living": 10.0, "writes": 2, "cycles": 1, "backoff": 1.60, "verify": 0.80, "creative": 0.10},
        RHYTHM_CALM: {"bpm": 62, "heartbeat": 5.0, "thinker": 18.0, "agent": 2.4, "living": 7.5, "writes": 3, "cycles": 2, "backoff": 1.45, "verify": 0.82, "creative": 0.18},
        RHYTHM_FOCUSED: {"bpm": 92, "heartbeat": 3.0, "thinker": 9.0, "agent": 1.2, "living": 4.0, "writes": 6, "cycles": 4, "backoff": 1.25, "verify": 0.60, "creative": 0.35},
        RHYTHM_BUILD: {"bpm": 124, "heartbeat": 2.0, "thinker": 6.0, "agent": 0.75, "living": 3.0, "writes": 8, "cycles": 5, "backoff": 1.20, "verify": 0.64, "creative": 0.42},
        RHYTHM_DEBUG: {"bpm": 78, "heartbeat": 3.8, "thinker": 8.5, "agent": 1.6, "living": 4.8, "writes": 5, "cycles": 3, "backoff": 1.35, "verify": 0.92, "creative": 0.12},
        RHYTHM_CREATIVE: {"bpm": 110, "heartbeat": 2.4, "thinker": 5.5, "agent": 1.0, "living": 3.5, "writes": 7, "cycles": 5, "backoff": 1.20, "verify": 0.58, "creative": 0.85},
        RHYTHM_REM: {"bpm": 54, "heartbeat": 9.0, "thinker": 45.0, "agent": 4.0, "living": 15.0, "writes": 2, "cycles": 1, "backoff": 1.75, "verify": 0.86, "creative": 0.65},
        RHYTHM_URGENT_ASSIST: {"bpm": 138, "heartbeat": 1.0, "thinker": 2.5, "agent": 0.35, "living": 1.5, "writes": 6, "cycles": 3, "backoff": 1.15, "verify": 0.90, "creative": 0.08},
        RHYTHM_EMERGENCY: {"bpm": 164, "heartbeat": 0.5, "thinker": 0.9, "agent": 0.18, "living": 0.75, "writes": 4, "cycles": 2, "backoff": 1.05, "verify": 0.98, "creative": 0.00},
        RHYTHM_SAFE: {"bpm": 50, "heartbeat": 10.0, "thinker": 30.0, "agent": 5.0, "living": 15.0, "writes": 1, "cycles": 1, "backoff": 2.00, "verify": 0.98, "creative": 0.00},
    }
    p = dict(base.get(mode, base[RHYTHM_FOCUSED]))

    music_bpm = _safe_float(music.get("bpm"), 0.0, 0.0, 260.0)
    if music_bpm > 0 and mode not in (RHYTHM_EMERGENCY, RHYTHM_SAFE):
        # Nudge toward music BPM without allowing music to dominate safety cadence.
        p["bpm"] = int(round((float(p["bpm"]) * 0.70) + (music_bpm * 0.30)))

    energy = _safe_float(personality.get("energy"), 0.5, 0.0, 1.0)
    speed_nudge = 1.0 + ((energy - 0.5) * 0.20) + (urgency * 0.12) + (speech_pressure * 0.08)
    if str(system.get("pressure_tier")) in ("high", "critical"):
        speed_nudge *= 0.82

    for key in ("heartbeat", "thinker", "agent", "living"):
        p[key] = round(max(0.12, float(p[key]) / max(0.25, speed_nudge)), 3)

    p["bpm"] = _safe_int(p["bpm"], 92, 32, 190)
    p["writes"] = _safe_int(p["writes"], 4, 1, 12)
    p["cycles"] = _safe_int(p["cycles"], 3, 1, 8)
    p["backoff"] = round(_safe_float(p["backoff"], 1.25, 1.0, 2.5), 3)
    p["verify"] = round(_safe_float(p["verify"], 0.65, 0.0, 1.0), 3)
    p["creative"] = round(_safe_float(p["creative"], 0.3, 0.0, 1.0), 3)
    return p


def build_rhythm_cadence_packet(context: Optional[Dict[str, Any]] = None, *, force_refresh: bool = False) -> Dict[str, Any]:
    """Build the governed cadence packet. This function has no action authority."""
    global _LAST_PACKET_TS, _LAST_PACKET
    ctx = dict(context or {})
    now = time.time()
    with _LOCK:
        if not force_refresh and _LAST_PACKET and (now - _LAST_PACKET_TS) <= _PACKET_CACHE_TTL:
            # Return a copy so consumers cannot mutate shared state.
            out = dict(_LAST_PACKET)
            out["cache_hit"] = True
            return out

    text = str(ctx.get("text") or ctx.get("user_text") or ctx.get("command_text") or ctx.get("goal") or "")
    adaptive = _adaptive_basis(ctx)
    personality = _personality_basis(ctx)
    music = _music_basis(ctx)
    system = _system_pressure()
    urgency, urgency_notes = _score_urgency(text, ctx, adaptive, music)
    speech_pressure = _score_speech_pressure(text, ctx)
    importance_score = round(max(urgency, _safe_float(ctx.get("importance") or ctx.get("importance_score"), 0.0, 0.0, 1.0)), 4)
    mode, mode_notes = _select_mode(text, ctx, adaptive, personality, music, system, urgency)
    params = _mode_parameters(mode, urgency, speech_pressure, personality, music, system)

    packet = RhythmCadencePacket(
        packet_id="rhythm-" + uuid.uuid4().hex[:12],
        ts=_now_iso(),
        rhythm_mode=mode,
        tempo_bpm=int(params["bpm"]),
        heartbeat_interval_sec=float(params["heartbeat"]),
        thinker_interval_sec=float(params["thinker"]),
        agent_step_interval_sec=float(params["agent"]),
        living_loop_interval_sec=float(params["living"]),
        memory_write_budget_per_min=int(params["writes"]),
        max_inner_cycles=int(params["cycles"]),
        backoff_multiplier=float(params["backoff"]),
        personality_energy=round(_safe_float(personality.get("energy"), 0.5, 0.0, 1.0), 4),
        verification_bias=float(params["verify"]),
        creative_bias=float(params["creative"]),
        urgency_score=urgency,
        importance_score=importance_score,
        speech_pressure=speech_pressure,
        source="adaptive+personality+context+music+resources",
        adaptive_basis=adaptive,
        personality_basis=personality,
        system_pressure=system,
        notes=urgency_notes + mode_notes,
        constraints={
            "rhythm_is_not_authority": True,
            "music_may_not_control_torque": True,
            "emotion_may_not_override_safety": True,
            "operator_core_required_for_actions": True,
            "msdc_required_for_body_validation": True,
            "ram_first": True,
            "throttled_persistence_only": True,
        },
    ).to_dict()

    with _LOCK:
        _LAST_PACKET = dict(packet)
        _LAST_PACKET_TS = now
    return packet


def get_rhythm_cognition_packet(context: Optional[Dict[str, Any]] = None, *, force_refresh: bool = False) -> Dict[str, Any]:
    return build_rhythm_cadence_packet(context, force_refresh=force_refresh)


def get_current_cadence_packet(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return build_rhythm_cadence_packet(context)


def build_vibe_state_packet(context: Optional[Dict[str, Any]] = None, *, force_refresh: bool = False) -> Dict[str, Any]:
    pkt = build_rhythm_cadence_packet(context, force_refresh=force_refresh)
    pkt["packet_alias"] = "VibeStatePacket"
    pkt["truthful_vibecoding"] = {
        "definition": "rhythm-mediated cognitive scheduling under governance",
        "truth_before_vibe": True,
        "governance_before_motion": True,
    }
    return pkt


def _interval(packet_key: str, default: float, context: Optional[Dict[str, Any]] = None) -> float:
    try:
        pkt = build_rhythm_cadence_packet(context)
        return max(0.05, float(pkt.get(packet_key, default)))
    except Exception:
        return float(default)


def get_heartbeat_interval(context: Optional[Dict[str, Any]] = None, default: float = 3.0) -> float:
    return _interval("heartbeat_interval_sec", default, context)


def get_living_loop_interval(context: Optional[Dict[str, Any]] = None, default: float = 4.0) -> float:
    return _interval("living_loop_interval_sec", default, context)


def get_thinker_interval(context: Optional[Dict[str, Any]] = None, default: float = 9.0) -> float:
    return _interval("thinker_interval_sec", default, context)


def get_agent_step_interval(context: Optional[Dict[str, Any]] = None, default: float = 1.2) -> float:
    return _interval("agent_step_interval_sec", default, context)


def get_agent_watch_interval(context: Optional[Dict[str, Any]] = None, default: float = 0.10) -> float:
    try:
        pkt = build_rhythm_cadence_packet(context)
        # Watchers must stay lightweight: never below 0.08s, never above 0.50s.
        return max(0.08, min(0.50, float(pkt.get("agent_step_interval_sec", default)) * 0.25))
    except Exception:
        return float(default)


def throttle_key(key: str, *, min_interval_sec: Optional[float] = None, budget_per_min: Optional[int] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """RAM-first anti-thrash guard. Returns allow/defer without writing to disk."""
    k = str(key or "default")[:160]
    pkt = build_rhythm_cadence_packet(context)
    now = time.time()
    interval = float(min_interval_sec) if min_interval_sec is not None else max(0.15, float(pkt.get("agent_step_interval_sec", 1.0)))
    budget = int(budget_per_min) if budget_per_min is not None else int(pkt.get("memory_write_budget_per_min", 4) or 4)
    budget = max(1, min(60, budget))

    with _LOCK:
        rec = _THROTTLE_STATE.get(k) or {"last_ts": 0.0, "window_start": now, "count": 0, "defer_count": 0}
        if now - float(rec.get("window_start", now)) >= 60.0:
            rec["window_start"] = now
            rec["count"] = 0
            rec["defer_count"] = 0
        elapsed = now - float(rec.get("last_ts", 0.0) or 0.0)
        allowed = elapsed >= interval and int(rec.get("count", 0) or 0) < budget
        if allowed:
            rec["last_ts"] = now
            rec["count"] = int(rec.get("count", 0) or 0) + 1
        else:
            rec["defer_count"] = int(rec.get("defer_count", 0) or 0) + 1
        _THROTTLE_STATE[k] = rec

    return {
        "ok": True,
        "key": k,
        "allow": bool(allowed),
        "decision": "ALLOW" if allowed else "DEFER",
        "wait_remaining_sec": round(max(0.0, interval - elapsed), 3),
        "budget_per_min": budget,
        "count_this_window": int(rec.get("count", 0) or 0),
        "defer_count_this_window": int(rec.get("defer_count", 0) or 0),
        "rhythm_mode": pkt.get("rhythm_mode"),
        "execution_authority": False,
    }


def should_pulse(key: str, *, context: Optional[Dict[str, Any]] = None, min_interval_sec: Optional[float] = None) -> bool:
    return bool(throttle_key(key, context=context, min_interval_sec=min_interval_sec).get("allow"))


def apply_rhythm_to_personality_profile(profile: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a decorated personality profile; does not mutate Personality state."""
    base = dict(profile or {})
    pkt = build_rhythm_cadence_packet(context)
    mode = str(pkt.get("rhythm_mode") or RHYTHM_FOCUSED)
    base.setdefault("energy", pkt.get("personality_energy", 0.5))
    base["rhythm_mode"] = mode
    base["tempo_bpm"] = pkt.get("tempo_bpm")
    base["response_cadence"] = "fast" if mode in (RHYTHM_BUILD, RHYTHM_URGENT_ASSIST, RHYTHM_EMERGENCY) else "slow" if mode in (RHYTHM_STILL, RHYTHM_CALM, RHYTHM_SAFE, RHYTHM_REM) else "normal"
    base["verification_bias"] = pkt.get("verification_bias")
    base["creative_bias"] = pkt.get("creative_bias")
    base["execution_authority"] = False
    return base


def build_embodied_motion_packet(command_text: str = "", context: Optional[Dict[str, Any]] = None, body_packet: Optional[Dict[str, Any]] = None, hazard_packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a rhythm-informed MotionIntentPacket for MSDC/OperatorCore review. No actuation authority."""
    ctx = dict(context or {})
    if command_text:
        ctx["command_text"] = command_text
    if body_packet:
        ctx["body_packet"] = dict(body_packet)
    if hazard_packet:
        ctx.update({"hazard_packet": dict(hazard_packet), "hazard_type": hazard_packet.get("hazard_type") or hazard_packet.get("emergency_type")})
        if hazard_packet.get("human_risk") or hazard_packet.get("person_at_risk"):
            ctx["human_risk"] = True
    pkt = build_rhythm_cadence_packet(ctx, force_refresh=True)
    mode = str(pkt.get("rhythm_mode") or RHYTHM_FOCUSED)
    text = str(command_text or ctx.get("user_text") or ctx.get("text") or "")
    body = dict(body_packet or ctx.get("body_packet") or {}) if isinstance(body_packet or ctx.get("body_packet"), dict) else {}
    hazard = dict(hazard_packet or ctx.get("hazard_packet") or {}) if isinstance(hazard_packet or ctx.get("hazard_packet"), dict) else {}

    caps = body.get("capabilities") if isinstance(body.get("capabilities"), dict) else {}
    proximity_clear = bool(body.get("proximity_clear") or body.get("path_clear") or body.get("floor_verified"))
    floor_verified = bool(body.get("floor_verified") or body.get("stance_stable"))
    human_nearby = bool(body.get("human_nearby") or body.get("human_in_motion_radius"))
    can_locomote = bool(caps.get("locomotion") or body.get("can_move") or body.get("locomotion_available"))
    can_upper = bool(caps.get("upper_body") or caps.get("arms") or body.get("upper_body_available") or True)
    can_face = bool(caps.get("face_expression") or body.get("face_expression_available") or True)

    wants_dance = _contains_phrase(text, frozenset({"dance", "slow dance", "move to the music", "music"})) or str(ctx.get("task_type") or "") == "dance"
    wants_run = _contains_phrase(text, frozenset({"run", "hurry", "faster", "rush", "get there", "urgent"}))
    emergency = mode == RHYTHM_EMERGENCY or bool(ctx.get("emergency") or hazard.get("human_risk") or hazard.get("person_at_risk"))

    motion = MOTION_IDLE_SWAY
    requested_pace = "normal"
    safe_pace = "normal_walk"
    locomotion_requested = False
    locomotion_allowed = False
    speed = 0.30
    force = 0.08
    radius = 20
    gesture_interval = max(0.25, float(pkt.get("agent_step_interval_sec", 1.0)))
    step_interval = 0.0

    if mode in (RHYTHM_SAFE, RHYTHM_STILL):
        motion = MOTION_SAFE_STOP if mode == RHYTHM_SAFE else MOTION_STILL
        requested_pace = "stop"
        safe_pace = "stop"
        speed = 0.0
        force = 0.0
        radius = 0
    elif emergency:
        motion = MOTION_WALK_PACE_SYNC
        requested_pace = "fastest_safe_response"
        locomotion_requested = True
        locomotion_allowed = bool(can_locomote and floor_verified and proximity_clear)
        safe_pace = "fast_walk" if locomotion_allowed else "upper_body_alert_only"
        speed = 0.75 if locomotion_allowed else 0.0
        force = 0.18
        radius = 35 if locomotion_allowed else 10
        step_interval = 0.25 if locomotion_allowed else 0.0
    elif wants_run or mode == RHYTHM_URGENT_ASSIST:
        motion = MOTION_WALK_PACE_SYNC
        requested_pace = "fast"
        locomotion_requested = True
        locomotion_allowed = bool(can_locomote and floor_verified and proximity_clear and not human_nearby)
        safe_pace = "fast_walk" if locomotion_allowed else "slow_walk_or_scan"
        speed = 0.62 if locomotion_allowed else 0.18
        force = 0.12
        radius = 30 if locomotion_allowed else 12
        step_interval = 0.40 if locomotion_allowed else 0.0
    elif wants_dance and mode in (RHYTHM_CREATIVE, RHYTHM_FOCUSED, RHYTHM_BUILD, RHYTHM_CALM):
        if str(ctx.get("dance_style") or "").lower() == "slow" or mode == RHYTHM_CALM or _contains_phrase(text, frozenset({"slow dance"})):
            motion = MOTION_SLOW_DANCE
            requested_pace = "slow_dance"
            safe_pace = "upper_body_slow_dance"
            speed = 0.22
            force = 0.08
            radius = 20
            gesture_interval = max(0.8, float(pkt.get("agent_step_interval_sec", 1.0)))
        else:
            motion = MOTION_DANCE
            requested_pace = "dance"
            safe_pace = "upper_body_dance"
            speed = 0.35
            force = 0.10
            radius = 25
        locomotion_requested = bool(ctx.get("locomotion_requested", False))
        locomotion_allowed = bool(locomotion_requested and can_locomote and floor_verified and proximity_clear and not human_nearby)
    elif mode == RHYTHM_CALM:
        motion = MOTION_IDLE_SWAY
        requested_pace = "slow"
        safe_pace = "slow"
        speed = 0.18
        force = 0.06
        radius = 15
    else:
        motion = MOTION_HEAD_BOB if _safe_float(pkt.get("tempo_bpm"), 92) >= 100 else MOTION_IDLE_SWAY
        requested_pace = "normal"
        safe_pace = "normal"
        speed = 0.25
        force = 0.08
        radius = 20

    if human_nearby and not emergency:
        speed = min(speed, 0.18)
        force = min(force, 0.06)
        radius = min(radius, 12)
        if locomotion_requested:
            locomotion_allowed = False
            safe_pace = "stop_scan_or_upper_body_only"

    notes = [
        "MotionIntentPacket is suggestion-only; no actuation authority.",
        "MSDC must validate body state, joints, floor, path, proximity, and capability before motion.",
        "OperatorCore ActionContract and SMGET/SafetyPolicies/AssuranceGate remain required for real execution.",
    ]
    if human_nearby:
        notes.append("Human nearby: speed/radius/force capped and locomotion disabled unless emergency policy explicitly allows it.")
    if locomotion_requested and not locomotion_allowed:
        notes.append("Locomotion requested but not allowed from current body/environment evidence.")

    return MotionIntentPacket(
        packet_id="motion-" + uuid.uuid4().hex[:12],
        ts=_now_iso(),
        rhythm_mode=mode,
        motion_profile=motion,
        tempo_bpm=int(pkt.get("tempo_bpm", 92)),
        requested_pace=requested_pace,
        safe_pace=safe_pace,
        gesture_interval_sec=round(float(gesture_interval), 3),
        step_interval_sec=round(float(step_interval), 3),
        movement_speed_scale=round(max(0.0, min(1.0, speed)), 3),
        force_scale=round(max(0.0, min(1.0, force)), 3),
        max_motion_radius_cm=int(max(0, min(100, radius))),
        locomotion_requested=bool(locomotion_requested),
        locomotion_allowed=bool(locomotion_allowed),
        upper_body_allowed=bool(can_upper),
        face_expression_allowed=bool(can_face),
        safety_notes=notes,
        cadence_packet=pkt,
        context={
            "floor_verified": floor_verified,
            "proximity_clear": proximity_clear,
            "human_nearby": human_nearby,
            "can_locomote": can_locomote,
            "emergency": emergency,
        },
    ).to_dict()


def recommend_robot_motion(command_text: str = "", context: Optional[Dict[str, Any]] = None, body_packet: Optional[Dict[str, Any]] = None, hazard_packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return build_embodied_motion_packet(command_text, context=context, body_packet=body_packet, hazard_packet=hazard_packet)


def persist_rhythm_snapshot(packet: Optional[Dict[str, Any]] = None, *, reason: str = "manual") -> Dict[str, Any]:
    """Explicit, throttled persistence for diagnostics only. Not called automatically by loops."""
    pkt = dict(packet or build_rhythm_cadence_packet(force_refresh=True))
    now = time.time()
    key = "persist_snapshot"
    with _LOCK:
        last = float(_EVENT_LAST_TS.get(key, 0.0) or 0.0)
        if now - last < 30.0:
            return {"ok": False, "reason": "throttled", "wait_remaining_sec": round(30.0 - (now - last), 3)}
        _EVENT_LAST_TS[key] = now
    try:
        _ensure_dirs()
        with open(_snapshot_path(), "w", encoding="utf-8") as f:
            json.dump({"reason": reason, "packet": pkt}, f, ensure_ascii=False, indent=2)
        con = sqlite3.connect(_db_path(), timeout=5.0, check_same_thread=False)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS rhythm_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                event TEXT,
                rhythm_mode TEXT,
                tempo_bpm INTEGER,
                details_json TEXT
            )
            """
        )
        cur.execute(
            "INSERT INTO rhythm_events (ts, event, rhythm_mode, tempo_bpm, details_json) VALUES (?, ?, ?, ?, ?)",
            (_now_iso(), "snapshot", str(pkt.get("rhythm_mode") or ""), int(pkt.get("tempo_bpm") or 0), json.dumps({"reason": reason, "packet": pkt}, ensure_ascii=False)),
        )
        con.commit()
        con.close()
        return {"ok": True, "snapshot_path": _snapshot_path(), "db_path": _db_path()}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def get_rhythm_diagnostics(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    pkt = build_rhythm_cadence_packet(context, force_refresh=True)
    with _LOCK:
        throttle_keys = len(_THROTTLE_STATE)
    return {
        "ok": True,
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "packet": pkt,
        "throttle_keys": throttle_keys,
        "has_psutil": _HAS_PSUTIL,
        "doctrine": {
            "execution_authority": False,
            "rhythm_may_modulate_cadence": True,
            "rhythm_may_not_authorize_action": True,
            "emotion_may_not_override_safety": True,
            "music_may_not_directly_control_motors": True,
            "msdc_operator_smget_required_for_motion": True,
        },
    }


__all__ = [
    "RHYTHM_STILL", "RHYTHM_CALM", "RHYTHM_FOCUSED", "RHYTHM_BUILD", "RHYTHM_DEBUG",
    "RHYTHM_CREATIVE", "RHYTHM_REM", "RHYTHM_URGENT_ASSIST", "RHYTHM_EMERGENCY", "RHYTHM_SAFE",
    "MOTION_STILL", "MOTION_IDLE_SWAY", "MOTION_HEAD_BOB", "MOTION_HAND_TAP", "MOTION_SLOW_DANCE",
    "MOTION_DANCE", "MOTION_WALK_PACE_SYNC", "MOTION_AVATAR_ONLY", "MOTION_SAFE_STOP",
    "build_rhythm_cadence_packet", "get_rhythm_cognition_packet", "get_current_cadence_packet",
    "build_vibe_state_packet", "get_heartbeat_interval", "get_living_loop_interval", "get_thinker_interval",
    "get_agent_step_interval", "get_agent_watch_interval", "throttle_key", "should_pulse",
    "apply_rhythm_to_personality_profile", "build_embodied_motion_packet", "recommend_robot_motion",
    "persist_rhythm_snapshot", "get_rhythm_diagnostics",
]

# ====================================================================
# END OF SarahMemoryRhythmCognition.py v9.0.0
# ====================================================================
