"""
=== SarahMemory Project ===
File: SarahMemoryNeuron.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-03-01
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
==============================================================================================================================================================

PURPOSE:
--------
SarahMemoryNeuron is a cognitive axis module.
It consolidates:
- Meta-cognition (confidence, self-check, contradiction detection)
- Cross-domain synthesis (math/chem/physics/code/system constraints)
- Curiosity engine (gap detection + safe experiment proposals)
- Cognitive graph core (MeaningGraph-like lightweight memory links)
- Hybrid routing (deterministic first, API second, sandbox optional)
- Parallel thought architecture (Analyst/Skeptic/Optimizer/Engineer/Governor)
- AdvCU delegation for intent + command parsing (better routing immediately).
- Research lane insertion (Tier-2 evidence-backed answers).
- Creative job ticket output (standardizes creative requests for Studio).
- Compare-based QA gate (confidence calibration + consensus).
- SarahMemoryGlobals.py: identity + mode flags + paths + safety envelope
- SarahMemoryLogicCalc.py: deterministic scientific reasoning (Tier-0)
- SarahMemoryWebSYM.py: symbolic router (Tier-1)
- SarahMemoryResearch.py: evidence lane (Tier-2)
- SarahMemoryCanvasStudio.py: creative job ticketing (Creative lane)
- SarahMemoryCompare.py: QA gate (post-check)
- SarahMemoryAPI.py: multi-provider LLM routing (Tier-3)
- SarahMemoryCognitiveServices.py: orchestration/awareness bridge (optional)
"""

from __future__ import annotations
# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "router"
# CATEGORY = "cognition"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "neuron_axis"
# FAMILY = "core_cognition"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Central cognitive axis and lane selector. Owns primary routing, deterministic-first reasoning, helper delegation, research insertion, creative tickets, and QA gating."
# --- SARAHMETA END ---
import os
import re
import shutil
import subprocess
import platform
import sys
import time
import json
import queue
import sqlite3
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Safe imports (never hard-fail the platform)
# -----------------------------------------------------------------------------
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None

# Deterministic core (Tier-0)
try:
    from SarahMemoryLogicCalc import LogicCalc as _LogicCalc  # type: ignore
except Exception:
    _LogicCalc = None

# Symbolic router (Tier-1)
try:
    import SarahMemoryWebSYM as _WebSYM  # type: ignore
except Exception:
    _WebSYM = None

# Synapses governance + sandbox (Tier-2 / optional)
try:
    import SarahMemorySynapes as _Syn  # type: ignore
except Exception:
    _Syn = None

# Multi-provider LLM API (Tier-3)
try:
    import SarahMemoryAPI as _SMAPI  # type: ignore
except Exception:
    _SMAPI = None

# Optional orchestrator/awareness layer
try:
    import SarahMemoryCognitiveServices as _Cog  # type: ignore
except Exception:
    _Cog = None

# Tier-0.5: Advanced command understanding
try:
    import SarahMemoryAdvCU as _AdvCU  # type: ignore
except Exception:
    _AdvCU = None

# Tier-2: Evidence-backed research
try:
    import SarahMemoryResearch as _Research  # type: ignore
except Exception:
    _Research = None

# Creative lane: ticketing / directories / ids
try:
    import SarahMemoryCanvasStudio as _CanvasStudio  # type: ignore
except Exception:
    _CanvasStudio = None

# QA gate: compare / consensus
try:
    import SarahMemoryCompare as _Compare  # type: ignore
except Exception:
    _Compare = None

# Optional Vision workers (helper-only; never hard-fail the platform)
try:
    import SarahMemorySOBJE as _SOBJE  # type: ignore
except Exception:
    _SOBJE = None

try:
    import SarahMemoryFacialRecognition as _FaceRec  # type: ignore
except Exception:
    _FaceRec = None

try:
    import base64 as _sm_base64  # type: ignore
except Exception:
    _sm_base64 = None

try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None

try:
    import cv2 as _cv2  # type: ignore
except Exception:
    _cv2 = None


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("SarahMemoryNeuron")
logger.setLevel(logging.DEBUG if bool(getattr(config, "DEBUG_MODE", True)) else logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(_h)
logger.propagate = False


# -----------------------------------------------------------------------------
# Paths / DB (best-effort, non-fatal)
# -----------------------------------------------------------------------------
def _base_dir() -> str:
    try:
        return str(getattr(config, "BASE_DIR", os.getcwd()))
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

def _neuron_db_path() -> str:
    return os.path.join(_datasets_dir(), "neuron_axis.db")

def _ensure_dirs() -> None:
    try:
        os.makedirs(_datasets_dir(), exist_ok=True)
    except Exception:
        pass

def _connect_db() -> Optional[sqlite3.Connection]:
    try:
        _ensure_dirs()
        con = sqlite3.connect(_neuron_db_path(), check_same_thread=False, timeout=3.0)
        return con
    except Exception:
        return None

_DB: Optional[sqlite3.Connection] = None

def _init_db() -> None:
    global _DB
    if _DB is not None:
        return
    _DB = _connect_db()
    if _DB is None:
        return
    try:
        cur = _DB.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS neuron_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                kind TEXT,
                intent TEXT,
                confidence REAL,
                source TEXT,
                payload TEXT
            )
            """
        )
        _DB.commit()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Safety envelope
# -----------------------------------------------------------------------------
def _flag(name: str, default: bool = False) -> bool:
    try:
        return bool(getattr(config, name, default))
    except Exception:
        return default

def _is_safe_mode() -> bool:
    # SAFE_MODE is the master "no autonomous execution" gate
    return _flag("SAFE_MODE", True)

def _is_local_only() -> bool:
    return _flag("LOCAL_ONLY_MODE", False)

def _neosky_armed() -> bool:
    # Dual-key arm: NEOSKYMATRIX + DEVELOPERSMODE
    return _flag("NEOSKYMATRIX", False) and _flag("DEVELOPERSMODE", False)

def _device_profile() -> Dict[str, Any]:
    return {
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "base_dir": _base_dir(),
        "local_only": _is_local_only(),
        "safe_mode": _is_safe_mode(),
        "neosky_armed": _neosky_armed(),
    }

def _budget_limits() -> Dict[str, Any]:
    return {
        "max_parallel": int(getattr(config, "NEURON_MAX_PARALLEL", 4) if config else 4),
        "max_curiosity": int(getattr(config, "NEURON_MAX_CURIOSITY", 2) if config else 2),
        "max_trace_kb": int(getattr(config, "NEURON_MAX_TRACE_KB", 64) if config else 64),
        "max_links": int(getattr(config, "NEURON_MAX_LINKS", 32) if config else 32),
    }


def _core_registry_snapshot(force: bool = False) -> Dict[str, Any]:
    try:
        if config and hasattr(config, "sm_refresh_core_registry"):
            data = config.sm_refresh_core_registry(force=force)  # type: ignore[attr-defined]
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}
    return {}


def _core_module_allowed(module_name: str, capability: Optional[str] = None, import_obj: Any = None) -> bool:
    key = os.path.splitext(os.path.basename(str(module_name or "").strip()))[0]
    try:
        if config and hasattr(config, "sm_is_core_module_approved"):
            return bool(config.sm_is_core_module_approved(key, capability=capability))  # type: ignore[attr-defined]
    except Exception:
        pass
    return bool(import_obj is not None)


def _vision_helper_allowed(module_name: str, import_obj: Any) -> bool:
    """Allow read-only local vision helpers even if registry approval is incomplete.

    This is intentionally narrower than general core-module bypass:
    - only applies to vision helper modules
    - only when the module is already imported locally
    - used by the Answer/Vision helper lane, not action lanes
    """
    if _core_module_allowed(module_name, "vision", import_obj):
        return True
    try:
        return bool(import_obj is not None and str(module_name or "") in ("SarahMemorySOBJE", "SarahMemoryFacialRecognition"))
    except Exception:
        return False


def _core_governance_trace() -> Dict[str, Any]:
    try:
        if config and hasattr(config, "sm_get_core_governance_profile"):
            profile = config.sm_get_core_governance_profile()  # type: ignore[attr-defined]
            if isinstance(profile, dict):
                return {
                    "discovery_enabled": bool(profile.get("discovery_enabled", False)),
                    "registered_count": int(profile.get("registered_count", 0) or 0),
                    "quarantined_count": int(profile.get("quarantined_count", 0) or 0),
                    "ignored_count": int(profile.get("ignored_count", 0) or 0),
                }
    except Exception:
        pass
    return {
        "discovery_enabled": False,
        "registered_count": 0,
        "quarantined_count": 0,
        "ignored_count": 0,
    }


def _approved_lane_modules() -> Dict[str, bool]:
    return {
        "advcu": _core_module_allowed("SarahMemoryAdvCU", "helper", _AdvCU),
        "logiccalc": _core_module_allowed("SarahMemoryLogicCalc", "reasoning", _LogicCalc),
        "websym": _core_module_allowed("SarahMemoryWebSYM", "reasoning", _WebSYM),
        "research": _core_module_allowed("SarahMemoryResearch", "reasoning", _Research),
        "api": _core_module_allowed("SarahMemoryAPI", "helper", _SMAPI),
        "compare": _core_module_allowed("SarahMemoryCompare", "utility", _Compare),
        "canvas": _core_module_allowed("SarahMemoryCanvasStudio", "creative", _CanvasStudio),
        "filesystem": _core_module_allowed("SarahMemoryFilesystem", "action", True),
        "network": _core_module_allowed("SarahMemoryNetwork", "network", True),
        "cognitive": _core_module_allowed("SarahMemoryCognitiveServices", "diagnostics", _Cog),
    }


def _trace_primary_lane(trace: Dict[str, Any], lane: str, owner: str) -> None:
    try:
        trace["primary_lane"] = str(lane or "general")
        trace["primary_owner"] = str(owner or "neuron")
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Cognitive graph core (lightweight, local)
# -----------------------------------------------------------------------------
@dataclass
class GraphEdge:
    src: str
    dst: str
    rel: str
    w: float = 0.5
    meta: Dict[str, Any] = field(default_factory=dict)

class MeaningGraph:
    def __init__(self) -> None:
        self.edges: List[GraphEdge] = []

    def link(self, src: str, dst: str, rel: str, w: float = 0.5, meta: Optional[Dict[str, Any]] = None) -> None:
        self.edges.append(GraphEdge(src=src, dst=dst, rel=rel, w=float(w), meta=meta or {}))

_GRAPH = MeaningGraph()


# -----------------------------------------------------------------------------
# Structured I/O
# -----------------------------------------------------------------------------
@dataclass
class NeuronInput:
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NeuronResult:
    ok: bool
    reply: str
    confidence: float = 0.5
    intent: str = "general"
    source: str = "neuron"
    artifacts: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "reply": str(self.reply),
            "confidence": float(self.confidence),
            "intent": str(self.intent),
            "source": str(self.source),
            "artifacts": self.artifacts,
            "trace": self.trace,
            "actions": list(self.actions or []),
        }


# -----------------------------------------------------------------------------
# Thought agents (parallel-calibration layer)
# -----------------------------------------------------------------------------
class ThoughtAgent:
    name = "agent"

    def evaluate(self, inp: NeuronInput, draft: str) -> Tuple[str, float, Dict[str, Any]]:
        return draft, 0.0, {}

class AnalystAgent(ThoughtAgent):
    name = "Analyst"

    def evaluate(self, inp: NeuronInput, draft: str) -> Tuple[str, float, Dict[str, Any]]:
        if not draft or len(draft.strip()) < 10:
            return "I need a bit more detail to answer precisely. What constraints matter most?", -0.05, {"reason": "too_short"}
        return draft, 0.01, {"reason": "ok"}

class SkepticAgent(ThoughtAgent):
    name = "Skeptic"

    def evaluate(self, inp: NeuronInput, draft: str) -> Tuple[str, float, Dict[str, Any]]:
        t = (draft or "").lower()
        if any(k in t for k in ("not sure", "can't", "unknown", "maybe", "might be")):
            return draft, -0.08, {"reason": "uncertainty_markers"}
        return draft, 0.0, {"reason": "ok"}

class OptimizerAgent(ThoughtAgent):
    name = "Optimizer"

    def evaluate(self, inp: NeuronInput, draft: str) -> Tuple[str, float, Dict[str, Any]]:
        if len(draft) > 2200:
            return draft[:2200].rstrip() + "\n\n[Truncated for performance.]", -0.03, {"reason": "truncated"}
        return draft, 0.0, {"reason": "ok"}

class EngineerAgent(ThoughtAgent):
    name = "Engineer"

    def evaluate(self, inp: NeuronInput, draft: str) -> Tuple[str, float, Dict[str, Any]]:
        if _is_safe_mode() and ("autonomous" in (draft or "").lower() or "self-evolve" in (draft or "").lower()):
            return "SAFE_MODE: Autonomous operations are gated. Provide explicit user authorization.", -0.25, {"policy": "safe_mode_gate"}
        return draft, 0.0, {"policy": "ok"}

class GovernorAgent(ThoughtAgent):
    name = "Governor"

    def evaluate(self, inp: NeuronInput, draft: str) -> Tuple[str, float, Dict[str, Any]]:
        return draft, 0.0, {"governance": "ok"}


# -----------------------------------------------------------------------------
# Intent heuristics (fallback if AdvCU not available / not confident)
# -----------------------------------------------------------------------------
def _classify_intent(text: str) -> str:
    t = (text or "").strip().lower()
    if not t:
        return "empty"

    def _has_phrase(*phrases: str) -> bool:
        try:
            for phrase in phrases:
                p = str(phrase or "").strip().lower()
                if not p:
                    continue
                if re.search(r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])", t):
                    return True
        except Exception:
            pass
        return False

    def _looks_like_story_math() -> bool:
        has_count_question = _has_phrase("how many", "what is the total", "what's the total", "how much in total", "altogether", "combined", "in all", "total")
        has_quantities = bool(re.search(r"\b\d+\b", t)) or _has_phrase(
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
            "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand",
        )
        has_countable_noun = _has_phrase(
            "apple", "apples", "orange", "oranges", "banana", "bananas", "item", "items",
            "bottle", "bottles", "book", "books", "key", "keys", "coin", "coins",
        )
        return bool(has_count_question and has_quantities and has_countable_noun)

    if _has_phrase("calculate", "solve", "convert", "unit", "derivative", "integral", "matrix", "vector", "sqrt", "square root", "plus", "minus", "times", "multiplied by", "divide", "divided by", "sum of"):
        return "math"
    if re.search(r"\d", t) and any(op in t for op in ("+", "-", "*", "/")):
        return "math"
    if re.search(r"\b[a-z]\s*=\s*[-+]?\d+(?:\.\d+)?", t):
        return "math"
    if _looks_like_story_math():
        return "math"

    if _has_phrase("diagnos", "self-test", "self test", "health check"):
        return "diagnostics"
    if _has_phrase("gpu", "vram", "cuda", "disk space", "free disk", "free space", "storage", "drive space", "cpu", "ram", "memory usage", "system stats", "system status", "hardware stats"):
        return "device_query"
    if _has_phrase("chem", "molar", "stoichi", "compound", "element", "reaction", "acid", "base") or re.search(r"\bph\b", t):
        return "chemistry"
    if _has_phrase("optimize", "speed up", "performance", "refactor", "bug", "error", "traceback"):
        return "engineering"
    if _has_phrase("who are you", "version", "creator", "brian", "softdev0"):
        return "identity"
    if _has_phrase("research", "look up", "browse", "latest", "verify", "sources", "citation"):
        return "research"
    if _has_phrase("generate", "create", "make", "draw", "song", "music", "video", "avatar", "image"):
        return "creative"
    return "general"


# -----------------------------------------------------------------------------
# Tier-0.5: AdvCU delegation (intent + command parsing)
# -----------------------------------------------------------------------------

def _advcu_analyze(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "intent": None,
        "confidence": None,
        "command": None,
        "semantic_packet": {},
        "helper_payload": {},
        "entities": {},
        "raw": {},
    }
    if not _AdvCU:
        return out
    if not _core_module_allowed("SarahMemoryAdvCU", "helper", _AdvCU):
        out["raw"]["governance"] = "advcu_not_registered"
        return out

    try:
        parse_fn = getattr(_AdvCU, "parse_command", None)
        if callable(parse_fn):
            cmd = parse_fn(text)  # type: ignore
            cmd_dict: Dict[str, Any] = {}
            semantic_packet: Dict[str, Any] = {}
            if isinstance(cmd, dict):
                cmd_dict = dict(cmd)
                semantic_packet = dict(cmd_dict.get("semantic_packet") or {})
            else:
                try:
                    if hasattr(cmd, "to_dict") and callable(getattr(cmd, "to_dict")):
                        d = cmd.to_dict()  # type: ignore[attr-defined]
                        if isinstance(d, dict):
                            cmd_dict = dict(d)
                except Exception:
                    cmd_dict = {}
                try:
                    if hasattr(cmd, "to_semantic_packet") and callable(getattr(cmd, "to_semantic_packet")):
                        sp = cmd.to_semantic_packet()  # type: ignore[attr-defined]
                        if isinstance(sp, dict):
                            semantic_packet = dict(sp)
                except Exception:
                    semantic_packet = {}

            if cmd_dict or semantic_packet:
                out["command"] = cmd_dict or semantic_packet
                out["semantic_packet"] = semantic_packet or dict((cmd_dict.get("extra") or {}).get("semantic_packet") or {})
                helper_payload = dict(
                    (cmd_dict.get("helper_payload") or {})
                    or (out["semantic_packet"].get("helper_payload") if isinstance(out["semantic_packet"], dict) else {})
                    or {}
                )
                out["helper_payload"] = helper_payload
                out["intent"] = (
                    cmd_dict.get("intent")
                    or out["semantic_packet"].get("intent")
                    or cmd_dict.get("action")
                    or cmd_dict.get("type")
                )
                out["entities"] = {
                    "subject": cmd_dict.get("subject") or out["semantic_packet"].get("subject"),
                    "query_type": cmd_dict.get("query_type") or out["semantic_packet"].get("query_type"),
                    "attributes": cmd_dict.get("attributes") or out["semantic_packet"].get("attributes") or [],
                    "keywords": cmd_dict.get("keywords") or out["semantic_packet"].get("keywords") or [],
                    "module_hints": cmd_dict.get("module_hints") or out["semantic_packet"].get("module_hints") or [],
                    "action_expectation": cmd_dict.get("action_expectation") or out["semantic_packet"].get("action_expectation"),
                    "helper_payload": helper_payload,
                }
                out["raw"]["parse_command"] = cmd_dict or out["semantic_packet"]
    except Exception:
        pass

    try:
        clf_fn = getattr(_AdvCU, "classify_intent_with_confidence", None)
        if callable(clf_fn):
            intent, conf = clf_fn(text)  # type: ignore
            if intent:
                out["raw"]["classify_intent_with_confidence"] = {"intent": intent, "confidence": conf}
                if not out["intent"]:
                    out["intent"] = intent
                try:
                    out["confidence"] = float(conf) if conf is not None else out["confidence"]
                except Exception:
                    pass
    except Exception:
        pass

    return out


# -----------------------------------------------------------------------------
# Creative lane: job ticket output (unifies Studio contract)
# -----------------------------------------------------------------------------
def _is_creative_intent(intent: str, text: str, adv: Optional[Dict[str, Any]] = None) -> bool:
    t = (text or "").lower()
    i = (intent or "").lower()
    if i in ("image", "music", "song", "lyrics_to_song", "video", "avatar", "creative"):
        return True
    try:
        cmd = (adv or {}).get("command") or {}
        act = str(cmd.get("action") or cmd.get("intent") or "").lower()
        if any(k in act for k in ("image", "music", "song", "video", "avatar", "render", "generate")):
            return True
    except Exception:
        pass
    return any(k in t for k in (
        "generate an image", "make an image", "create an image", "draw ",
        "generate a song", "make a song", "lyrics to song", "compose music",
        "generate a video", "make a video", "create a video",
        "make an avatar", "create an avatar"
    ))

def _creative_kind(intent: str, text: str, adv: Optional[Dict[str, Any]] = None) -> str:
    t = (text or "").lower()
    i = (intent or "").lower()
    if i in ("image",):
        return "image"
    if i in ("music", "song", "lyrics_to_song"):
        return "music"
    if i in ("video",):
        return "video"
    if i in ("avatar",):
        return "avatar"
    cmd = (adv or {}).get("command") or {}
    act = str(cmd.get("action") or cmd.get("intent") or "").lower()
    for k in ("image", "music", "song", "video", "avatar"):
        if k in act:
            return "music" if k == "song" else k
    if "lyrics to song" in t or "song" in t or "music" in t:
        return "music"
    if "video" in t:
        return "video"
    if "avatar" in t:
        return "avatar"
    return "image"

def _make_creative_job_ticket(prompt: str, kind: str, meta: Dict[str, Any], adv: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        if _CanvasStudio and hasattr(_CanvasStudio, "ensure_canvas_directories"):
            _CanvasStudio.ensure_canvas_directories()  # type: ignore
    except Exception:
        pass

    job_id = None
    try:
        if _CanvasStudio and hasattr(_CanvasStudio, "generate_unique_id"):
            job_id = _CanvasStudio.generate_unique_id()  # type: ignore
    except Exception:
        job_id = None

    job_id = job_id or f"job_{int(time.time()*1000)}"
    cmd = (adv or {}).get("command") or {}
    entities = (adv or {}).get("entities") or {}
    style = entities.get("style") or cmd.get("style") or meta.get("style") or ""
    size = entities.get("size") or cmd.get("size") or meta.get("size") or ""
    duration = entities.get("duration") or cmd.get("duration") or meta.get("duration") or ""
    format_hint = entities.get("format") or cmd.get("format") or meta.get("format") or ""

    return {
        "job_id": job_id,
        "kind": kind,  # image|music|video|avatar
        "prompt": prompt,
        "params": {"style": style, "size": size, "duration": duration, "format": format_hint},
        "routing": {
            "requested_by": "neuron",
            "intent": meta.get("intent"),
            "offline": bool(meta.get("offline")),
            "local_only": bool(_is_local_only()),
        },
        "ts": time.time(),
    }


# -----------------------------------------------------------------------------
# Action lane: strict local executor ticketing (Filesystem/System operations)
# Neuron ONLY emits tickets; Kernel/Integration performs execution.
# -----------------------------------------------------------------------------
def _is_action_intent(intent: str, text: str, adv: Optional[Dict[str, Any]] = None) -> bool:
    cmd = (adv or {}).get("command") or {}
    act = str(cmd.get("action") or cmd.get("intent") or cmd.get("type") or "").lower()
    t = (text or "").lower()

    if act and any(k in act for k in (
        "copy", "move", "rename", "delete",
        "backup", "restore", "scan", "quarantine",
        "file_", "filesystem", "snapshot", "capture", "zoom"
    )):
        return True

    if any(k in t for k in (
        "take a snapshot", "capture this", "capture it", "take a picture", "take a photo",
        "save it", "save this", "zoom in", "zoom on", "magnify", "focus on"
    )) and (_contains_any_phrase_token(t, _VISUAL_SUBJECT_HINTS) or _has_deictic_visual_reference(t)):
        return True

    return any(k in t for k in (
        "copy file", "move file", "rename file", "delete file",
        "create backup", "restore backup", "scan file", "scan directory",
        "quarantine", "restore from quarantine"
    ))


def _map_action_to_executor_action(adv: Optional[Dict[str, Any]] = None, text: str = "") -> str:
    cmd = (adv or {}).get("command") or {}
    act = str(cmd.get("action") or cmd.get("intent") or cmd.get("type") or "").lower()
    t = (text or "").lower()

    if any(k in act for k in ("snapshot", "capture")) or any(k in t for k in ("take a snapshot", "capture this", "capture it", "take a picture", "take a photo")):
        if any(k in t for k in ("save it", "save this", "store it", "store this", "download this")):
            return "vision_snapshot_save"
        return "vision_snapshot"
    if "zoom" in act or any(k in t for k in ("zoom in", "zoom on", "magnify", "closer look", "focus on")):
        return "vision_zoom_focus"
    if "copy" in act or "copy" in t:
        return "file_copy"
    if "move" in act or "move" in t:
        return "file_move"
    if "rename" in act or "rename" in t:
        return "file_rename"
    if "delete" in act or "delete" in t:
        return "file_delete"
    if "attribute" in act or "attributes" in act:
        return "file_attrs"
    if "incremental" in act:
        return "backup_incremental"
    if "backup" in act or "create backup" in t:
        return "backup_full"
    if "restore" in act or "restore backup" in t:
        return "backup_restore"
    if "rotate" in act:
        return "backup_rotate"
    if "scan directory" in act or "scan_dir" in act or "scan directory" in t:
        return "scan_dir"
    if "scan" in act or "scan file" in t:
        return "scan_file"
    if "quarantine" in act and "restore" in act:
        return "quarantine_restore"
    if "restore from quarantine" in t:
        return "quarantine_restore"
    return act or "unknown"


def _infer_safety_level(executor_action: str) -> str:
    a = (executor_action or "").lower()
    if a in ("scan_file", "scan_dir"):
        return "low"
    if a in ("file_copy", "file_move", "file_rename", "backup_full", "backup_incremental", "backup_rotate"):
        return "medium"
    if a in ("file_delete", "backup_restore", "file_attrs", "quarantine_restore"):
        return "high"
    return "medium"

def _make_action_ticket(text: str, meta: Dict[str, Any], adv: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cmd = (adv or {}).get("command") or {}
    entities = (adv or {}).get("entities") or {}

    executor_action = _map_action_to_executor_action(adv=adv, text=text)
    safety_level = _infer_safety_level(executor_action)

    args: Dict[str, Any] = {}
    if isinstance(cmd, dict):
        args.update(cmd.get("args") or {})
    if isinstance(entities, dict):
        for k, v in (entities or {}).items():
            if k not in args:
                args[k] = v

    if "src" in args and "source" not in args:
        args["source"] = args["src"]
    if "dst" in args and "destination" not in args:
        args["destination"] = args["dst"]
    if "path" in args and "file_path" not in args and executor_action in ("file_delete", "scan_file", "file_attrs"):
        args["file_path"] = args["path"]
    if "dir" in args and "directory" not in args and executor_action == "scan_dir":
        args["directory"] = args["dir"]

    vision_request = meta.get("vision_request") if isinstance(meta.get("vision_request"), dict) else None
    if vision_request:
        args.setdefault("vision_request", vision_request)

    requires_confirm = True
    ticket_id = f"act_{int(time.time()*1000)}"
    executor_name = "SarahMemoryVision" if str(executor_action).startswith("vision_") else "SarahMemoryFilesystem"
    return {
        "ticket_id": ticket_id,
        "action": executor_action,
        "args": args,
        "safety_level": safety_level,
        "requires_confirm": requires_confirm,
        "executor": executor_name,
        "meta": {
            "intent": meta.get("intent"),
            "session_id": meta.get("session_id") or (meta.get("meta", {}) or {}).get("session_id") if isinstance(meta.get("meta"), dict) else None,
            "user_id": meta.get("user_id") or (meta.get("meta", {}) or {}).get("user_id") if isinstance(meta.get("meta"), dict) else None,
            "offline": bool(meta.get("offline") or meta.get("LOCAL_ONLY_MODE") or meta.get("local_only")),
        },
        "ts": time.time(),
    }

# -----------------------------------------------------------------------------
# Tier-2: Evidence-backed research lane
# -----------------------------------------------------------------------------
def _try_research(text: str) -> Optional[Dict[str, Any]]:
    if not _Research:
        return None
    if not _core_module_allowed("SarahMemoryResearch", "reasoning", _Research):
        return None
    try:
        fn = getattr(_Research, "get_research_data", None)
        if callable(fn):
            data = fn(text)  # type: ignore
            return data if isinstance(data, dict) else {"raw": data}
    except Exception:
        return None
    return None

def _synthesize_evidence_reply(base_reply: str, research_data: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    summ = research_data.get("summary") or research_data.get("answer") or research_data.get("result") or ""
    links = research_data.get("links") or research_data.get("sources") or []
    if isinstance(links, str):
        links = [links]

    appended: List[str] = []
    if isinstance(summ, str) and summ.strip():
        appended.append(summ.strip())
    if isinstance(links, list):
        cleaned = [str(x) for x in links if str(x).strip()][:6]
        if cleaned:
            appended.append("Sources: " + " | ".join(cleaned))

    if not appended:
        return base_reply, {"research": research_data}

    merged = (base_reply.rstrip() + "\n\n" + "\n".join(appended)).strip()
    return merged, {"research": research_data, "research_summary": appended[0] if appended else ""}


# -----------------------------------------------------------------------------
# Compare-based QA gate
# -----------------------------------------------------------------------------
def _qa_compare_gate(user_text: str, draft: str, intent: str) -> Tuple[float, Dict[str, Any]]:
    if not _Compare:
        return 0.0, {}
    if not _core_module_allowed("SarahMemoryCompare", "utility", _Compare):
        return 0.0, {"compare": {"status": "BYPASS", "reason": "compare_not_registered"}}
    try:
        fn = getattr(_Compare, "compare_reply", None)
        if callable(fn):
            # If signature doesn't match, exception will be caught.
            result = fn(user_text, draft, intent=intent)  # type: ignore
            if isinstance(result, dict):
                status = str(result.get("status") or "").upper()
                score = float(result.get("score") or result.get("similarity") or 0.0)
                if status in ("HIT", "PASS", "OK") and score >= 0.55:
                    return +0.08, {"compare": result}
                if status in ("MISS", "FAIL") or score < 0.35:
                    return -0.14, {"compare": result}
                return -0.05, {"compare": result}
    except Exception:
        return 0.0, {}
    return 0.0, {}


# -----------------------------------------------------------------------------
# Deterministic tiers
# -----------------------------------------------------------------------------
def _try_logiccalc(text: str) -> Optional[Dict[str, Any]]:
    """Attempt deterministic evaluation.

    Priority:
      1) SarahMemoryLogicCalc (if available and registry-approved)
      2) Safe fallback parser for basic arithmetic + sqrt when LogicCalc is unavailable
    """
    raw = (text or "").strip()
    if not raw:
        return None

    # Primary: project engine
    if _LogicCalc and _core_module_allowed("SarahMemoryLogicCalc", "reasoning", _LogicCalc):
        try:
            engine = _LogicCalc()
            if hasattr(engine, "answer"):
                out = engine.answer(raw)  # type: ignore
                if out:
                    return out
            if hasattr(engine, "solve"):
                out = engine.solve(raw)  # type: ignore
                if out:
                    return out
        except Exception:
            pass

    # Fallback: minimal safe evaluator (keeps math functionality alive even without 3rd-party libs)
    try:
        import ast
        import math

        t = raw.lower()

        # Normalize common phrasing into an expression
        t = t.replace("what is", "").replace("whats", "").replace("what\'s", "").strip()
        t = re.sub(r"^\s*the\s+", "", t)
        t = t.replace("the sum of", "").replace("sum of", "")
        t = t.replace("square root of", "sqrt(")
        # handle "sqrt(25)" already and close paren if we opened one
        if "sqrt(" in t and ")" not in t.split("sqrt(", 1)[1]:
            # append closing paren if missing
            t = t + ")"

        # Words -> numbers (very small, deterministic mapping)
        words = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
            "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
            "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
            "seventy": 70, "eighty": 80, "ninety": 90,
            "hundred": 100, "thousand": 1000,
        }

        def words_to_number(s: str) -> Optional[int]:
            toks = [w for w in re.split(r"[^a-z]+", s.lower()) if w]
            if not toks:
                return None
            total = 0
            current = 0
            seen = False
            for w in toks:
                if w not in words:
                    return None
                seen = True
                v = words[w]
                if v == 100:
                    current = max(1, current) * 100
                elif v == 1000:
                    total += max(1, current) * 1000
                    current = 0
                else:
                    current += v
            if not seen:
                return None
            return total + current

        # Replace plus/minus/times/divided
        t = t.replace(" plus ", " + ").replace(" minus ", " - ").replace(" times ", " * ")
        t = t.replace(" multiplied by ", " * ").replace(" x ", " * ")
        t = t.replace(" divided by ", " / ").replace(" over ", " / ")

        # Normalize digit scale phrases (e.g., '1 thousand' -> '(1*1000)')
        t = re.sub(r"\b(\d+(?:\.\d+)?)\s+(thousand|k)\b", r"(\1*1000)", t)
        t = re.sub(r"\b(\d+(?:\.\d+)?)\s+million\b", r"(\1*1000000)", t)
        t = re.sub(r"\b(\d+(?:\.\d+)?)\s+billion\b", r"(\1*1000000000)", t)

        # If still no digits but contains number words, attempt conversion on entire string
        if not re.search(r"\d", t):
            n = words_to_number(t)
            if n is not None:
                t = str(n)

        # Very small story-math fallback for additive word problems.
        # Example: "If Sam has five apples and Johnny has 3 apples, how many apples do they have total"
        if re.search(r"\b(how many|total|altogether|combined|in all)\b", t):
            qty_words = {
                "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
                "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
                "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
                "seventy": 70, "eighty": 80, "ninety": 90,
            }
            vals = [float(x) for x in re.findall(r"\b\d+(?:\.\d+)?\b", t)]
            for word, num in qty_words.items():
                vals.extend([float(num)] * len(re.findall(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", t)))
            if len(vals) >= 2 and re.search(r"\b(apple|apples|orange|oranges|banana|bananas|item|items|book|books|coin|coins|key|keys|bottle|bottles)\b", t):
                total = sum(vals)
                return {
                    "ok": True,
                    "engine": "neuron_story_math",
                    "value": total,
                    "text": f"Computed result using story-math evaluation = {total}",
                    "meta": {"normalized_from": raw, "intent": "calc_story"},
                    "meaning": {"props": [{"predicate": "evaluates", "args": [{"kind": "concept", "name": "story_math_total", "value": total}]}], "meta": {"intent": "calc_story"}},
                }

        # Convert patterns like "one thousand + two thousand"
        # Split on operators and convert each side when possible.
        parts = re.split(r"(\+|\-|\*|/|\(|\)|,)", t)
        rebuilt = []
        for p in parts:
            ps = p.strip()
            if not ps:
                rebuilt.append(p)
                continue
            if ps in {"+", "-", "*", "/", "(", ")", ","}:
                rebuilt.append(ps)
                continue
            if re.search(r"\d", ps):
                rebuilt.append(ps)
                continue
            n = words_to_number(ps)
            rebuilt.append(str(n) if n is not None else ps)
        expr = " ".join([x for x in rebuilt if x != ","]).strip()

        # Allow only a strict subset of AST nodes
        allowed_nodes = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
            ast.USub, ast.UAdd,
            ast.Constant, ast.Load, ast.Name,
        )

        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, allowed_nodes):
                return None
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id != "sqrt":
                    return None

        def _eval(node):
            if isinstance(node, ast.Expression):
                return _eval(node.body)
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float)):
                    return float(node.value)
                return None
            if isinstance(node, ast.UnaryOp):
                v = _eval(node.operand)
                if v is None:
                    return None
                if isinstance(node.op, ast.USub):
                    return -v
                if isinstance(node.op, ast.UAdd):
                    return v
                return None
            if isinstance(node, ast.BinOp):
                a = _eval(node.left)
                b = _eval(node.right)
                if a is None or b is None:
                    return None
                if isinstance(node.op, ast.Add):
                    return a + b
                if isinstance(node.op, ast.Sub):
                    return a - b
                if isinstance(node.op, ast.Mult):
                    return a * b
                if isinstance(node.op, ast.Div):
                    return a / b
                if isinstance(node.op, ast.Pow):
                    return a ** b
                return None
            if isinstance(node, ast.Call):
                arg0 = _eval(node.args[0]) if node.args else None
                if arg0 is None:
                    return None
                return math.sqrt(arg0)
            if isinstance(node, ast.Name) and node.id == "sqrt":
                return None
            return None

        val = _eval(tree)
        if val is None:
            return None

        return {
            "ok": True,
            "engine": "neuron_fallback",
            "value": val,
            "text": f"Computed result using deterministic evaluation: {expr} = {val}",
            "meta": {"expression": expr, "normalized_from": raw, "intent": "calc"},
            "meaning": {"props": [{"predicate": "evaluates", "args": [{"kind": "concept", "name": "expression", "value": expr}]}], "meta": {"intent": "calc"}},
        }
    except Exception:
        return None
# -----------------------------------------------------------------------------
# Tier-0: System / Diagnostics lane (safe reads)
# -----------------------------------------------------------------------------
def _is_public_device(meta: Dict[str, Any]) -> bool:
    dm = str((meta or {}).get("device_mode") or "").strip().lower()
    return dm.startswith("public")


def _detect_system_kind(text: str, intent: str = "") -> Optional[str]:
    t = (text or "").strip().lower()
    i = (intent or "").strip().lower()
    if i in ("diagnostics", "diagnostic"):
        return "diagnostics"
    if "diagnos" in t or "self-test" in t or "self test" in t or "health check" in t:
        return "diagnostics"
    # Hardware/system stats
    if any(k in t for k in ("gpu", "vram", "cuda", "graphics", "nvidia-smi")):
        return "gpu"
    if any(k in t for k in ("disk space", "free disk", "free space", "storage", "drive space")):
        return "disk"
    if any(k in t for k in ("cpu", "ram", "memory usage", "system stats", "system status")):
        return "system_stats"
    return None


def _disk_usage_summary(path: str) -> Dict[str, Any]:
    try:
        du = shutil.disk_usage(path)
        total, used, free = int(du.total), int(du.used), int(du.free)
        gb = 1024 ** 3
        return {
            "ok": True,
            "path": path,
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "total_gb": round(total / gb, 2),
            "used_gb": round(used / gb, 2),
            "free_gb": round(free / gb, 2),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "path": path}


def _gpu_stats_summary() -> Dict[str, Any]:
    # Try torch first (if available)
    try:
        import torch  # type: ignore
        out: Dict[str, Any] = {"ok": True, "backend": "torch", "cuda_available": bool(torch.cuda.is_available())}
        if torch.cuda.is_available():
            idx = 0
            try:
                idx = int(torch.cuda.current_device())
            except Exception:
                idx = 0
            out["device_index"] = idx
            try:
                out["name"] = torch.cuda.get_device_name(idx)
            except Exception:
                out["name"] = None
            try:
                props = torch.cuda.get_device_properties(idx)
                out["total_vram_bytes"] = int(getattr(props, "total_memory", 0) or 0)
                out["total_vram_gb"] = round(out["total_vram_bytes"] / (1024 ** 3), 2) if out["total_vram_bytes"] else None
            except Exception:
                pass
            try:
                free_b, total_b = torch.cuda.mem_get_info(idx)
                out["free_vram_bytes"] = int(free_b)
                out["free_vram_gb"] = round(int(free_b) / (1024 ** 3), 2)
                out["used_vram_gb"] = round((int(total_b) - int(free_b)) / (1024 ** 3), 2)
            except Exception:
                pass
        return out
    except Exception:
        pass

    # Fallback: nvidia-smi if installed
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        raw = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=3).strip()
        if not raw:
            return {"ok": False, "backend": "nvidia-smi", "error": "No output"}
        parts = [p.strip() for p in raw.split(",")]
        out = {"ok": True, "backend": "nvidia-smi", "raw": raw}
        if len(parts) >= 5:
            out.update(
                {
                    "name": parts[0],
                    "driver_version": parts[1],
                    "total_vram_mb": float(parts[2]),
                    "free_vram_mb": float(parts[3]),
                    "gpu_util_pct": float(parts[4]),
                }
            )
        return out
    except Exception as e:
        return {"ok": False, "backend": "nvidia-smi", "error": str(e)}


def _quick_system_stats() -> Dict[str, Any]:
    out: Dict[str, Any] = {"ok": True, "platform": platform.platform(), "python": platform.python_version()}
    # psutil is optional
    try:
        import psutil  # type: ignore

        out["cpu_percent"] = float(psutil.cpu_percent(interval=0.2))
        vm = psutil.virtual_memory()
        out["ram_total_gb"] = round(float(vm.total) / (1024 ** 3), 2)
        out["ram_available_gb"] = round(float(vm.available) / (1024 ** 3), 2)
        out["ram_percent"] = float(vm.percent)
    except Exception:
        pass
    return out


def _run_quick_diagnostics() -> Dict[str, Any]:
    # Always provide a safe, fast snapshot for the chat endpoint.
    base_dir = os.getcwd()
    try:
        import SarahMemoryGlobals as _G  # type: ignore
        base_dir = str(getattr(_G, "BASE_DIR", base_dir) or base_dir)
    except Exception:
        pass

    out: Dict[str, Any] = {
        "ok": True,
        "engine": "quick",
        "system": _quick_system_stats(),
        "gpu": _gpu_stats_summary(),
        "disk": _disk_usage_summary(base_dir),
    }

    # Optional: include slower full diagnostics if explicitly enabled.
    if str(os.environ.get("SM_NEURON_FULL_DIAGNOSTICS", "")).strip().lower() in {"1", "true", "yes"}:
        try:
            import SarahMemoryDiagnostics as _D  # type: ignore

            sysr = getattr(_D, "run_system_diagnostics", None)
            hwr = getattr(_D, "run_hardware_diagnostics", None)
            out["engine"] = "quick+SarahMemoryDiagnostics"
            if callable(sysr):
                out["full_system"] = sysr()
            if callable(hwr):
                out["full_hardware"] = hwr()
        except Exception as e:
            out["full_error"] = str(e)

    return out

def _try_websym(text: str) -> Optional[str]:
    if not _WebSYM:
        return None
    if not _core_module_allowed("SarahMemoryWebSYM", "reasoning", _WebSYM):
        return None
    try:
        for fn_name in ("route_query", "handle_query", "process_query", "websym_query"):
            fn = getattr(_WebSYM, fn_name, None)
            if callable(fn):
                out = fn(text)
                if isinstance(out, str) and out.strip():
                    return out
    except Exception:
        return None
    return None

def _try_api(text: str, meta: Optional[Dict[str, Any]] = None) -> Optional[str]:
    if _is_local_only():
        return None
    if not _SMAPI:
        return None
    if not _core_module_allowed("SarahMemoryAPI", "helper", _SMAPI):
        return None

    try:
        meta2 = dict(meta or {})
        helper_payload = meta2.get("sm_helper_payload")
        user_input = str(helper_payload or text or "").strip()
        if not user_input:
            return None

        try:
            if config and hasattr(config, "resolve_model"):
                rm = config.resolve_model("reasoning", text=text, meta=meta2)  # type: ignore[attr-defined]
                meta2.setdefault("sm_model_resolve", {})["reasoning"] = rm
                if isinstance(rm, dict) and rm.get("selected"):
                    meta2.setdefault("preferred_model_repo", rm.get("selected"))
        except Exception:
            pass

        fn = getattr(_SMAPI, "send_to_api", None)
        if callable(fn):
            resp = fn(user_input, **meta2)
            if isinstance(resp, str) and resp.strip():
                return resp
            if isinstance(resp, dict):
                if resp.get("reply"):
                    return str(resp["reply"])
                if resp.get("data"):
                    return str(resp["data"])
    except Exception:
        return None
    return None



# -----------------------------------------------------------------------------
# Curiosity engine (safe proposals; never executes without gate)
# -----------------------------------------------------------------------------
def _curiosity_prompts(intent: str, text: str, budget: Dict[str, Any]) -> List[str]:
    max_c = int(budget.get("max_curiosity", 2))
    t = (text or "").lower()
    prompts: List[str] = []

    if intent in ("chemistry", "math", "engineering"):
        prompts.append("If you want, I can propose 2-3 safe sandbox experiments (no execution) to explore variations.")
    if "mix" in t and "element" in t:
        prompts.append("Do you want a stoichiometry sandbox plan: balance reaction → compute yields → propose constraints?")
    if "optimize" in t or "speed" in t:
        prompts.append("Do you want a performance audit plan: profile → bottleneck map → safe patch proposal?")
    if "?" in t and "constraints" not in t:
        prompts.append("What constraints matter most (accuracy, speed, offline, local-only, safety gating)?")

    return [p for p in prompts if p][:max_c]


# -----------------------------------------------------------------------------
# Event logging
# -----------------------------------------------------------------------------
def _log_event(kind: str, intent: str, confidence: float, source: str, payload: Dict[str, Any]) -> None:
    if _DB is None:
        return
    try:
        s = json.dumps(payload, ensure_ascii=False)
        max_kb = int(getattr(config, "NEURON_EVENT_MAX_KB", 96) if config else 96)
        if len(s) > max_kb * 1024:
            s = s[: max_kb * 1024] + "…"
        cur = _DB.cursor()
        cur.execute(
            "INSERT INTO neuron_events (ts, kind, intent, confidence, source, payload) VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), kind, intent, float(confidence), source, s),
        )
        _DB.commit()
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Helper semantic-gap payloads (compressed helper-only LLM method)
# -----------------------------------------------------------------------------
def _build_helper_payload(user_text: str, intent: str, adv: Optional[Dict[str, Any]] = None) -> str:
    try:
        adv = adv or {}
        cmd = adv.get("command") or {}
        entities = adv.get("entities") or {}
        sem = adv.get("semantic_packet") or {}
        parts: List[str] = []
        if intent:
            parts.append(f"intent={intent}")
        query_type = cmd.get("query_type") or sem.get("query_type") or entities.get("query_type") or ""
        if query_type:
            parts.append(f"query_type={query_type}")
        action = cmd.get("action") or cmd.get("intent") or sem.get("action") or ""
        if action:
            parts.append(f"action={action}")
        for key in ("subject", "topic", "object", "target"):
            val = entities.get(key) or cmd.get(key) or sem.get(key)
            if val:
                parts.append(f"{key}={val}")
        attrs = entities.get("attributes") or sem.get("attributes") or entities.get("mods") or cmd.get("attributes") or []
        if isinstance(attrs, (list, tuple)) and attrs:
            cleaned = [str(x).strip() for x in attrs if str(x).strip()]
            if cleaned:
                parts.append("attributes=" + ",".join(cleaned[:8]))
        elif isinstance(attrs, str) and attrs.strip():
            parts.append(f"attributes={attrs.strip()}")
        keywords = entities.get("keywords") or sem.get("keywords") or cmd.get("keywords") or []
        if isinstance(keywords, (list, tuple)) and keywords:
            cleaned = [str(x).strip() for x in keywords if str(x).strip()]
            if cleaned:
                parts.append("keywords=" + ",".join(cleaned[:8]))
        if not parts:
            return str(user_text or "").strip()
        return " | ".join(parts)
    except Exception:
        return str(user_text or "").strip()



# -----------------------------------------------------------------------------
# Vision helper routing (category-driven, helper-only)
# -----------------------------------------------------------------------------
_VISUAL_QUERY_TYPES = {
    "identify_color",
    "read_text",
    "detect_objects",
    "detect_faces",
    "identify_person",
    "locate_subject",
    "track_motion",
    "estimate_distance",
    "inspect_safety_zone",
    "compare_before_after",
    "assess_style_or_fit",
    "scene_summary",
}

_VISUAL_STRONG_CUE_PHRASES = (
    "what color", "what colour", "color of", "colour of",
    "what does", "read this", "read the text", "text on", "say on", "ocr",
    "what do you see", "what can you see", "describe what you see", "scene summary", "summarize the scene",
    "can you see me", "do you see me", "look at me", "look at this", "look at that",
    "analyze this image", "analyse this image", "show me what you see",
    "track", "follow", "moving", "motion",
    "danger zone", "hazard zone", "safe zone",
    "how far", "how close", "too close", "distance",
    "before and after", "compare these", "what changed",
    "look fat", "look good", "look okay", "does this shirt make me",
    "who is this person", "is this", "do you recognize", "who is this", "where is my",
    "behind me", "in front of me", "left of me", "right of me", "next to me",
    "take a snapshot", "capture this", "take a picture", "take a photo",
    "zoom in", "zoom on", "magnify", "closer look", "focus on",
)

_VISUAL_SUBJECT_HINTS = (
    "shirt", "pants", "trousers", "jeans", "shorts", "dress", "skirt", "jacket", "coat",
    "hat", "cap", "glasses", "eyes", "eye", "face", "beard", "mouth", "nose", "hair",
    "employee", "worker", "person", "people", "woman", "man", "girl", "boy",
    "wife", "daughter", "son", "child", "mother", "father", "sister", "brother",
    "machine", "forklift", "car", "road", "controller", "keys", "key",
    "desk", "screen", "monitor", "door", "animal", "dog", "cat",
    "photo", "image", "picture", "camera", "frame", "webcam", "pc", "computer",
    "waterbottle", "water bottle", "bottle", "logo", "paper", "document", "page", "note",
)

_VISUAL_ATTRIBUTE_HINTS = (
    "color", "colour", "text", "distance", "motion", "fit", "style", "objects", "faces",
    "safety", "identity", "position", "location", "logo", "label",
)

_VISUAL_ACTION_CUE_PHRASES = (
    "take a snapshot", "capture this", "capture it", "take a picture", "take a photo",
    "save it", "save this", "store it", "store this", "download this",
    "zoom in", "zoom on", "magnify", "closer look", "focus on",
)

_VISUAL_DEICTIC_REFERENCES = (
    "this", "that", "these", "those", "here", "holding up", "in my hand", "on this",
)


def _contains_phrase_token(text: str, phrase: str) -> bool:
    t = str(text or "").strip().lower()
    p = str(phrase or "").strip().lower()
    if not t or not p:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])"
    try:
        return re.search(pattern, t) is not None
    except Exception:
        return p in t


def _contains_any_phrase_token(text: str, phrases: Tuple[str, ...]) -> bool:
    try:
        return any(_contains_phrase_token(text, p) for p in (phrases or ()))
    except Exception:
        return False


def _has_visual_media(meta: Optional[Dict[str, Any]]) -> bool:
    meta = meta or {}
    payload_meta = _meta_payload_block(meta)
    try:
        for key in ("frame", "current_frame", "latest_frame", "image", "image_data"):
            if meta.get(key) is not None or payload_meta.get(key) is not None:
                return True
        for key in ("images", "video"):
            if isinstance(meta.get(key), list) and meta.get(key):
                return True
            if isinstance(payload_meta.get(key), list) and payload_meta.get(key):
                return True
        files: List[Any] = []
        if isinstance(meta.get("files"), list):
            files.extend(meta.get("files") or [])
        if isinstance(payload_meta.get("files"), list):
            files.extend(payload_meta.get("files") or [])
        return any(_looks_like_image_path(x) for x in files)
    except Exception:
        return False


def _has_deictic_visual_reference(text: str) -> bool:
    return _contains_any_phrase_token(text, _VISUAL_DEICTIC_REFERENCES)


def _looks_like_person_identity_prompt(text: str) -> bool:
    t = str(text or "").strip().lower()
    if re.search(r"\bis this\s+[a-z][a-z'\-]+\s+or\s+[a-z][a-z'\-]+\b", t):
        return True
    return _contains_any_phrase_token(t, (
        "who is this person", "who is this", "do you recognize", "who am i",
        "this is my wife", "this is my daughter", "this is my son", "this is my child",
    ))
def _looks_like_spatial_subject_query(text: str) -> bool:
    t = str(text or "").strip().lower()
    if not t:
        return False
    has_relation = _contains_any_phrase_token(t, ("behind me", "in front of me", "left of me", "right of me", "next to me", "near me"))
    has_subject = _contains_any_phrase_token(t, ("wife", "daughter", "son", "child", "person", "worker", "employee", "keys", "key"))
    return bool(has_relation and has_subject)


def _looks_like_knowledge_query(text: str) -> bool:
    t = str(text or "").strip().lower()
    if not t:
        return False
    if _contains_any_phrase_token(t, _VISUAL_STRONG_CUE_PHRASES) or _contains_any_phrase_token(t, _VISUAL_SUBJECT_HINTS) or _has_deictic_visual_reference(t):
        return False
    return bool(
        re.search(r"^\s*(who|what|when|where|why|how)\s+is\b", t)
        or re.search(r"^\s*tell me about\b", t)
        or re.search(r"^\s*explain\b", t)
        or re.search(r"^\s*define\b", t)
    )


def _meta_context_packet(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        cp = (meta or {}).get("context_packet") or {}
        return cp if isinstance(cp, dict) else {}
    except Exception:
        return {}


def _meta_payload_block(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cp = _meta_context_packet(meta)
    block = cp.get("meta") if isinstance(cp, dict) else {}
    return block if isinstance(block, dict) else {}


def _looks_like_image_path(value: Any) -> bool:
    try:
        s = str(value or "").strip().lower()
        return bool(s) and any(s.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    except Exception:
        return False


def _decode_visual_frame(candidate: Any):
    if candidate is None or _np is None or _cv2 is None:
        return None
    try:
        if isinstance(candidate, _np.ndarray):
            return candidate.copy()
    except Exception:
        pass
    try:
        if isinstance(candidate, (bytes, bytearray)):
            arr = _np.frombuffer(candidate, dtype=_np.uint8)
            if arr.size:
                frame = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
    except Exception:
        pass
    try:
        if isinstance(candidate, dict):
            for key in ("frame", "bgr", "ndarray", "image", "data", "base64", "bytes", "path"):
                if key in candidate:
                    frame = _decode_visual_frame(candidate.get(key))
                    if frame is not None:
                        return frame
            return None
    except Exception:
        pass
    try:
        s = str(candidate or "").strip()
        if not s:
            return None
        if _looks_like_image_path(s) and os.path.exists(s):
            frame = _cv2.imread(s)
            if frame is not None:
                return frame
        if s.startswith("data:image") and "," in s and _sm_base64 is not None:
            s = s.split(",", 1)[1]
        if _sm_base64 is not None:
            blob = _sm_base64.b64decode(s, validate=False)
            arr = _np.frombuffer(blob, dtype=_np.uint8)
            if arr.size:
                frame = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
    except Exception:
        return None
    return None


def _extract_visual_frame(meta: Optional[Dict[str, Any]]):
    meta = meta or {}
    payload_meta = _meta_payload_block(meta)
    candidates: List[Any] = []
    for key in ("frame", "current_frame", "latest_frame", "image", "image_data"):
        if key in meta:
            candidates.append(meta.get(key))
    for key in ("frame", "current_frame", "latest_frame", "image", "image_data"):
        if key in payload_meta:
            candidates.append(payload_meta.get(key))
    for key in ("images", "video", "files"):
        val = meta.get(key)
        if isinstance(val, list) and val:
            candidates.extend(val[:2])
        val2 = payload_meta.get(key)
        if isinstance(val2, list) and val2:
            candidates.extend(val2[:2])
    for cand in candidates:
        frame = _decode_visual_frame(cand)
        if frame is not None:
            return frame
    return None


def _infer_visual_query_type(text: str) -> str:
    t = str(text or "").strip().lower()
    if not t:
        return ""

    if _contains_any_phrase_token(t, ("what color", "what colour", "color of", "colour of")):
        return "identify_color"
    if _contains_any_phrase_token(t, ("what does", "read this", "read the text", "text on", "say on", "ocr", "read what is on")):
        return "read_text"
    if _looks_like_person_identity_prompt(t):
        return "identify_person"
    if _looks_like_spatial_subject_query(t):
        return "locate_subject"
    if _contains_any_phrase_token(t, ("track", "follow", "moving", "motion")):
        return "track_motion"
    if _contains_any_phrase_token(t, ("danger zone", "hazard zone", "safe zone")):
        return "inspect_safety_zone"
    if _contains_any_phrase_token(t, ("distance", "how far", "how close", "too close", "far away")):
        return "estimate_distance"
    if _contains_any_phrase_token(t, ("look fat", "look good", "look okay", "outfit", "style", "fit", "does this shirt make me")):
        return "assess_style_or_fit"
    if _contains_any_phrase_token(t, ("what do you see", "what can you see", "describe what you see", "scene summary", "summarize the scene", "can you see me", "do you see me", "am i visible", "am i in frame", "look at me", "look at this", "look at that", "around me", "environment")):
        return "scene_summary"
    if _contains_any_phrase_token(t, ("color", "colour")) and _contains_any_phrase_token(t, _VISUAL_SUBJECT_HINTS):
        return "identify_color"
    if _contains_any_phrase_token(t, ("face", "eyes", "eye", "nose", "mouth", "beard", "glasses", "hair")):
        return "detect_faces"
    if _contains_any_phrase_token(t, ("detect", "identify", "what is in", "what objects", "what's in", "do you see my", "where are my")):
        return "detect_objects"

    has_visual_subject = _contains_any_phrase_token(t, _VISUAL_SUBJECT_HINTS)
    has_visual_attr = _contains_any_phrase_token(t, _VISUAL_ATTRIBUTE_HINTS)
    has_visual_action = _contains_any_phrase_token(t, _VISUAL_ACTION_CUE_PHRASES)
    if has_visual_subject and has_visual_attr:
        if _contains_any_phrase_token(t, ("color", "colour")):
            return "identify_color"
        if _contains_any_phrase_token(t, ("text", "read", "logo", "label")):
            return "read_text"
        if _contains_any_phrase_token(t, ("distance", "close", "far", "near")):
            return "estimate_distance"
        if _contains_any_phrase_token(t, ("fit", "style")):
            return "assess_style_or_fit"
        if _contains_any_phrase_token(t, ("identity", "recognize")):
            return "identify_person"
        return "scene_summary"
    if has_visual_action and (has_visual_subject or _has_deictic_visual_reference(t)):
        return "scene_summary"

    return ""


def _infer_visual_subject(text: str, query_type: str = "") -> str:
    t = f" {str(text or '').strip().lower()} "
    candidates = (
        "water bottle", "waterbottle", "bottle", "logo", "paper", "document", "page", "note",
        "shirt", "pants", "trousers", "jeans", "shorts", "dress", "skirt", "jacket", "coat",
        "hat", "cap", "glasses", "eyes", "eye", "face", "beard", "mouth", "nose", "hair",
        "employee", "worker", "person", "wife", "daughter", "son", "child", "mother", "father",
        "machine", "forklift", "car", "road", "controller", "keys", "key",
        "desk", "screen", "monitor", "door", "animal", "dog", "cat", "webcam", "camera", "pc", "computer",
    )
    for c in candidates:
        if f" {c} " in t:
            return c.replace(" ", "_") if " " in c else c
    if query_type in ("detect_faces", "identify_person"):
        return "person"
    if query_type == "locate_subject":
        return "subject"
    if query_type in ("scene_summary", "detect_objects"):
        return "scene"
    return ""


def _infer_visual_attributes(text: str, query_type: str = "") -> List[str]:
    t = str(text or "").strip().lower()
    attrs: List[str] = []
    if "color" in t or "colour" in t:
        attrs.append("color")
    if any(s in t for s in ("text", "read", "say on", "what does", "logo", "label")):
        attrs.append("text")
    if any(s in t for s in ("distance", "close", "far", "near", "behind", "front of")):
        attrs.append("distance")
    if "motion" in t or "moving" in t or "track" in t:
        attrs.append("motion")
    if query_type == "assess_style_or_fit":
        attrs.extend(["fit", "style"])
    if query_type == "identify_person":
        attrs.append("identity")
    if query_type == "locate_subject":
        attrs.append("position")
    out: List[str] = []
    for a in attrs:
        if a not in out:
            out.append(a)
    return out


def _infer_visual_action_expectation(text: str, query_type: str = "") -> str:
    t = str(text or "").strip().lower()
    if _contains_any_phrase_token(t, ("take a snapshot", "capture this", "capture it", "take a picture", "take a photo")):
        if _contains_any_phrase_token(t, ("save it", "save this", "store it", "store this", "download this")):
            return "capture_and_save"
        return "capture_only"
    if _contains_any_phrase_token(t, ("zoom in", "zoom on", "magnify", "closer look", "focus on")):
        return "zoom_focus"
    return "answer_only"


def _build_vision_request(text: str, meta: Dict[str, Any], adv: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    adv = adv or {}
    cmd = adv.get("command") or {}
    sem = adv.get("semantic_packet") or {}
    helper_payload = adv.get("helper_payload") or {}
    vision = {}
    if isinstance(helper_payload, dict):
        vision = dict(helper_payload.get("vision") or {})
    inferred_query_type = _infer_visual_query_type(text)
    semantic_query_type = str(vision.get("query_type") or sem.get("query_type") or cmd.get("query_type") or "").strip().lower()
    query_type = semantic_query_type if semantic_query_type in _VISUAL_QUERY_TYPES else inferred_query_type
    if not query_type and (_has_visual_media(meta) and _has_deictic_visual_reference(text)):
        query_type = "scene_summary"
    generic_subject = str(sem.get("subject") or cmd.get("subject") or "").strip().lower()
    requested_subject = vision.get("requested_subject") or (
        generic_subject if generic_subject and generic_subject not in {"color", "text", "question", "statement", "scene", "object", "read", "detect", "track"} else ""
    ) or _infer_visual_subject(text, query_type) or ""
    requested_attributes = vision.get("requested_attributes") or sem.get("attributes") or cmd.get("attributes") or []
    if isinstance(requested_attributes, str):
        requested_attributes = [requested_attributes]
    requested_attributes = [str(x).strip().lower() for x in list(requested_attributes or []) if str(x).strip()]
    valid_attribute_names = {"color", "text", "distance", "motion", "fit", "style", "objects", "faces", "count", "safety", "identity", "position"}
    if (not requested_attributes) or all(a not in valid_attribute_names for a in requested_attributes):
        requested_attributes = _infer_visual_attributes(text, query_type) or requested_attributes
    module_hints = vision.get("module_hints") or sem.get("module_hints") or cmd.get("module_hints") or []
    if isinstance(module_hints, str):
        module_hints = [module_hints]
    action_expectation = vision.get("action_expectation") or sem.get("action_expectation") or cmd.get("action_expectation") or _infer_visual_action_expectation(text, query_type)
    return {
        "text": str(text or "").strip(),
        "query_type": query_type,
        "requested_subject": requested_subject,
        "subject": requested_subject,
        "requested_attributes": [str(x).strip() for x in list(requested_attributes or []) if str(x).strip()],
        "attributes": [str(x).strip() for x in list(requested_attributes or []) if str(x).strip()],
        "action_expectation": action_expectation or "answer_only",
        "module_hints": [str(x).strip() for x in list(module_hints or []) if str(x).strip()] or ["SarahMemorySOBJE", "SarahMemoryFacialRecognition"],
        "intent": adv.get("intent") or sem.get("intent") or cmd.get("intent") or meta.get("intent") or "vision",
        "helper_payload": {
            "vision": vision,
            "semantic_packet": sem,
        },
        "sensor_context": {
            "ui": meta.get("ui") or _meta_context_packet(meta).get("ui"),
            "has_images": bool((meta.get("images") or []) or (_meta_payload_block(meta).get("images") or [])),
            "has_video": bool((meta.get("video") or []) or (_meta_payload_block(meta).get("video") or [])),
        },
    }


def _is_visual_request(intent: str, text: str, meta: Optional[Dict[str, Any]] = None, adv: Optional[Dict[str, Any]] = None) -> bool:
    adv = adv or {}
    sem = adv.get("semantic_packet") or {}
    entities = adv.get("entities") or {}
    helper_payload = adv.get("helper_payload") or {}
    vision_helper = {}
    if isinstance(helper_payload, dict) and isinstance(helper_payload.get("vision"), dict):
        vision_helper = dict(helper_payload.get("vision") or {})

    t = str(text or "").strip().lower()
    helper_query_type = str(vision_helper.get("query_type") or "").strip().lower()
    helper_subject = str(vision_helper.get("requested_subject") or vision_helper.get("subject") or "").strip().lower()
    helper_attrs = vision_helper.get("requested_attributes") or vision_helper.get("attributes") or []
    if isinstance(helper_attrs, str):
        helper_attrs = [helper_attrs]
    helper_attrs = [str(x).strip().lower() for x in list(helper_attrs or []) if str(x).strip()]

    sem_query_type = str(sem.get("query_type") or entities.get("query_type") or "").strip().lower()
    inferred_query_type = _infer_visual_query_type(t)
    candidate_query_type = ""
    for candidate in (helper_query_type, sem_query_type, inferred_query_type):
        if candidate in _VISUAL_QUERY_TYPES:
            candidate_query_type = candidate
            break
    if not candidate_query_type:
        return False

    has_visual_media = _has_visual_media(meta)
    has_visual_subject = _contains_any_phrase_token(t, _VISUAL_SUBJECT_HINTS)
    has_visual_phrase = _contains_any_phrase_token(t, _VISUAL_STRONG_CUE_PHRASES)
    has_deictic_reference = _has_deictic_visual_reference(t)
    helper_has_signal = bool(helper_subject or helper_attrs)

    if _looks_like_knowledge_query(t) and not (has_visual_media or has_visual_phrase or has_visual_subject or has_deictic_reference or helper_has_signal):
        return False

    if candidate_query_type in {"identify_person", "locate_subject"}:
        return bool(
            _looks_like_person_identity_prompt(t)
            or _looks_like_spatial_subject_query(t)
            or has_visual_phrase
            or (has_visual_media and (has_deictic_reference or has_visual_subject or helper_has_signal))
        )

    if candidate_query_type in {"scene_summary", "detect_objects"}:
        return bool(
            has_visual_phrase
            or has_visual_subject
            or helper_has_signal
            or (has_visual_media and (has_deictic_reference or helper_has_signal))
        )

    if candidate_query_type in {"identify_color", "read_text", "detect_faces", "estimate_distance", "track_motion", "inspect_safety_zone", "compare_before_after", "assess_style_or_fit"}:
        return bool(has_visual_phrase or has_visual_subject or helper_has_signal or (has_visual_media and has_deictic_reference))

    return False
def _augment_face_findings(frame: Any, findings: Dict[str, Any]) -> Dict[str, Any]:
    if frame is None or not _FaceRec or not _vision_helper_allowed("SarahMemoryFacialRecognition", _FaceRec):
        return findings
    try:
        if hasattr(_FaceRec, "detect_faces_dnn"):
            faces = _FaceRec.detect_faces_dnn(frame)  # type: ignore[attr-defined]
            count = len(faces) if isinstance(faces, (list, tuple)) else 0
            face_block = dict(findings.get("faces") or {})
            if count and int(face_block.get("count") or 0) <= 0:
                face_block["count"] = count
                findings["faces"] = face_block
        if hasattr(_FaceRec, "get_user_fer_state"):
            fer = _FaceRec.get_user_fer_state(frame=frame)  # type: ignore[attr-defined]
            if isinstance(fer, dict):
                findings.setdefault("faces", {}).setdefault("fer", fer)
    except Exception:
        pass
    return findings


def _legacy_visual_text(text: str, request: Dict[str, Any]) -> str:
    qtype = str(request.get("query_type") or "").strip().lower()
    subj = str(request.get("requested_subject") or request.get("subject") or "").strip().lower()
    if qtype == "read_text" and subj == "shirt":
        return "what does my shirt say"
    if qtype == "identify_color" and subj:
        return f"what color is my {subj}"
    if qtype == "detect_faces":
        return "what's on my face"
    return str(text or "")


def _try_vision_lane(text: str, meta: Optional[Dict[str, Any]] = None, adv: Optional[Dict[str, Any]] = None) -> Optional[NeuronResult]:
    if not _SOBJE:
        return None
    if not _vision_helper_allowed("SarahMemorySOBJE", _SOBJE):
        return None
    if not _is_visual_request(str((meta or {}).get("intent") or ""), text, meta=meta, adv=adv):
        return None

    request = _build_vision_request(text, meta or {}, adv)
    frame = _extract_visual_frame(meta)
    worker_payload = dict(request)
    worker_payload["subject"] = request.get("requested_subject") or request.get("subject")
    worker_payload["attributes"] = list(request.get("requested_attributes") or request.get("attributes") or [])
    helper_block = dict((worker_payload.get("helper_payload") or {}).get("vision") or {})
    helper_block.setdefault("query_type", worker_payload.get("query_type"))
    helper_block.setdefault("requested_subject", worker_payload.get("subject"))
    helper_block.setdefault("requested_attributes", worker_payload.get("attributes") or [])
    helper_block.setdefault("action_expectation", worker_payload.get("action_expectation"))
    helper_block.setdefault("module_hints", worker_payload.get("module_hints") or [])
    worker_payload["helper_payload"] = {"vision": helper_block}
    legacy_mode = False
    first_error = None
    try:
        out = _SOBJE.answer_visual_question(worker_payload, frame)  # type: ignore[attr-defined]
    except Exception as e:
        first_error = str(e)
        try:
            out = _SOBJE.answer_visual_question(_legacy_visual_text(text, request), frame)  # type: ignore[attr-defined]
            legacy_mode = True
        except Exception as e2:
            return NeuronResult(
                ok=False,
                reply=f"Vision worker error: {e2}",
                confidence=0.2,
                intent=str(request.get("query_type") or "vision"),
                source="vision",
                artifacts={"vision_error": str(e2), "vision_request": request, "vision_worker_error": first_error},
                trace={"tiers": [{"tier": "vision", "engine": "SOBJE", "ok": False, "error": str(e2)}]},
            )

    reply = str(out.get("answer") or "").strip() if isinstance(out, dict) else str(out or "").strip()
    if not reply:
        reply = "I processed the visual request."
    details = dict(out.get("details") or {}) if isinstance(out, dict) else {}
    findings = dict(details.get("findings") or {})
    if legacy_mode and not findings:
        findings = {
            "vision_request": request,
            "resolved_subject": request.get("requested_subject") or request.get("subject"),
            "resolved_attributes": {a: details.get(a) for a in (request.get("requested_attributes") or []) if details.get(a) is not None},
            "confidence": float(details.get("confidence") or 0.45),
            "legacy_mode": True,
        }
    findings = _augment_face_findings(frame, findings)
    if findings:
        details["findings"] = findings

    conf = 0.55
    try:
        conf = float((findings or {}).get("confidence") or details.get("confidence") or 0.55)
    except Exception:
        conf = 0.55
    if frame is None:
        conf = min(conf, 0.35)

    artifacts = {
        "vision": {
            "request": request,
            "details": details,
            "frame_available": bool(frame is not None),
            "legacy_mode": bool(legacy_mode),
        }
    }
    if findings:
        artifacts["vision_findings"] = findings

    return NeuronResult(
        ok=True,
        reply=reply,
        confidence=max(0.25, min(0.95, conf)),
        intent=str(request.get("query_type") or "vision"),
        source="vision",
        artifacts=artifacts,
        trace={"tiers": [{"tier": "vision", "engine": "SOBJE", "ok": True, "frame": bool(frame is not None)}]},
    )


# -----------------------------------------------------------------------------
# Public router surface
# -----------------------------------------------------------------------------


def _normalize_ingress_route(meta: Optional[Dict[str, Any]], user_text: str) -> Dict[str, Any]:
    meta = dict(meta or {})
    route = dict(meta.get("ingress_route") or {})
    if not isinstance(route, dict):
        route = {}
    route_id = str(route.get("route_id") or "chat.general").strip() or "chat.general"
    domain = str(route.get("domain") or "chat").strip() or "chat"
    action = str(route.get("action") or "general_reply").strip() or "general_reply"
    target_module = str(route.get("target_module") or "SarahMemoryReply").strip() or "SarahMemoryReply"
    transport_target = str(route.get("transport_target") or "/api/chat").strip() or "/api/chat"
    confidence = route.get("confidence")
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.0
    entities = route.get("entities") if isinstance(route.get("entities"), dict) else {}
    intent_hint = str(route.get("intent_hint") or "").strip().lower()
    if not intent_hint:
        if route_id.startswith("research.weather") or domain == "research":
            intent_hint = "research"
        elif route_id.startswith("avatar."):
            intent_hint = "creative"
        elif route_id.startswith("reminder."):
            intent_hint = "time"
        elif domain in {"drivers", "system", "documents", "email", "network", "communication"}:
            intent_hint = "action"
        else:
            intent_hint = domain.lower()
    return {"route_id": route_id, "domain": domain, "action": action, "target_module": target_module, "transport_target": transport_target, "confidence": max(0.0, min(0.99, confidence)), "entities": entities, "intent_hint": intent_hint, "normalized_text": str(route.get("normalized_text") or user_text or ""), "needs_discovery": bool(route.get("needs_discovery")), "source": str(route.get("source") or "semantic_ingress_router")}


def _datasets_db_path(name: str) -> str:
    candidates = [os.path.join(os.path.dirname(__file__), name), os.path.join(os.getcwd(), name), os.path.join(_datasets_dir(), name), os.path.join(_data_dir(), 'memory', 'datasets', name), os.path.join(_base_dir(), 'data', 'memory', 'datasets', name)]
    best = None
    best_size = -1
    for cand in candidates:
        try:
            if os.path.exists(cand):
                sz = os.path.getsize(cand)
                if sz > best_size:
                    best = cand
                    best_size = sz
        except Exception:
            pass
    return best or candidates[0]


def _parse_version_from_text(value: str) -> str:
    text = str(value or "")
    m = re.search(r"(?:version|ue[_\- ]?)(\d+(?:\.\d+){0,2})", text, re.I)
    if m:
        return m.group(1)
    m = re.search(r"(\d+(?:\.\d+){1,2})", text)
    return m.group(1) if m else ""


def _system_index_lookup(terms: List[str], limit: int = 12) -> Dict[str, Any]:
    out = {"files": [], "registry": [], "db": _datasets_db_path("system_index.db")}
    db_path = out["db"]
    if not os.path.exists(db_path):
        return out
    clean_terms = [str(t).strip().lower() for t in (terms or []) if str(t).strip()]
    if not clean_terms:
        return out
    con = None
    try:
        con = sqlite3.connect(db_path, timeout=2.0)
        cur = con.cursor()
        file_hits = []
        reg_hits = []
        for term in clean_terms:
            like = f"%{term}%"
            try:
                cur.execute("SELECT file_path, file_type FROM file_index WHERE lower(coalesce(file_path,'')) LIKE ? LIMIT ?", (like, int(limit)))
                for row in cur.fetchall():
                    item = {"file_path": row[0], "file_type": row[1], "term": term}
                    if item not in file_hits:
                        file_hits.append(item)
            except Exception:
                pass
            try:
                cur.execute("SELECT root_key, key_path, value_name, value_data FROM registry_index WHERE lower(coalesce(value_data,'')) LIKE ? OR lower(coalesce(value_name,'')) LIKE ? OR lower(coalesce(key_path,'')) LIKE ? LIMIT ?", (like, like, like, int(limit)))
                for row in cur.fetchall():
                    item = {"root_key": row[0], "key_path": row[1], "value_name": row[2], "value_data": row[3], "term": term}
                    if item not in reg_hits:
                        reg_hits.append(item)
            except Exception:
                pass
        out["files"] = file_hits[:limit]
        out["registry"] = reg_hits[:limit]
    except Exception:
        pass
    finally:
        try:
            if con: con.close()
        except Exception: pass
    return out


def _software_db_lookup(terms: List[str], limit: int = 8) -> List[Dict[str, Any]]:
    db_path = _datasets_db_path("software.db")
    if not os.path.exists(db_path):
        return []
    clean_terms = [str(t).strip().lower() for t in (terms or []) if str(t).strip()]
    if not clean_terms:
        return []
    hits = []
    con = None
    try:
        con = sqlite3.connect(db_path, timeout=2.0)
        cur = con.cursor()
        cur.execute("PRAGMA table_info(software_apps)")
        cols = {str(r[1]).lower() for r in cur.fetchall()}
        has_name = "name" in cols
        has_app_name = "app_name" in cols
        has_path = "path" in cols
        has_platform = "platform" in cols
        has_last_used = "last_used" in cols
        has_usage_count = "usage_count" in cols
        has_version = "version" in cols
        has_category = "category" in cols
        has_is_installed = "is_installed" in cols
        if not (has_name or has_app_name or has_path):
            return []
        select_name = "coalesce(name, app_name)" if (has_name and has_app_name) else ("name" if has_name else ("app_name" if has_app_name else "''"))
        select_path = "path" if has_path else "''"
        select_platform = "platform" if has_platform else "''"
        select_last_used = "last_used" if has_last_used else "''"
        select_usage_count = "usage_count" if has_usage_count else "0"
        select_version = "version" if has_version else "''"
        select_category = "category" if has_category else "''"
        select_installed = "is_installed" if has_is_installed else "NULL"
        for term in clean_terms:
            like = f"%{term}%"
            where_parts = []
            params = []
            if has_name:
                where_parts.append("lower(coalesce(name,'')) LIKE ?")
                params.append(like)
            if has_app_name:
                where_parts.append("lower(coalesce(app_name,'')) LIKE ?")
                params.append(like)
            if has_path:
                where_parts.append("lower(coalesce(path,'')) LIKE ?")
                params.append(like)
            if has_category:
                where_parts.append("lower(coalesce(category,'')) LIKE ?")
                params.append(like)
            if not where_parts:
                continue
            sql = f"SELECT {select_name} AS resolved_name, {select_path} AS path, {select_platform} AS platform, {select_last_used} AS last_used, {select_usage_count} AS usage_count, {select_version} AS version, {select_category} AS category, {select_installed} AS is_installed FROM software_apps WHERE ({' OR '.join(where_parts)}) ORDER BY coalesce(usage_count,0) DESC, coalesce(last_used,'') DESC LIMIT ?"
            params.append(int(limit))
            try:
                cur.execute(sql, tuple(params))
                for row in cur.fetchall():
                    item = {"name": row[0], "path": row[1], "platform": row[2], "last_used": row[3], "usage_count": row[4], "version": row[5], "category": row[6], "is_installed": row[7], "term": term}
                    if item not in hits:
                        hits.append(item)
            except Exception:
                pass
    except Exception:
        return []
    finally:
        try:
            if con: con.close()
        except Exception: pass
    return hits[:limit]


def _si_lookup(candidates: List[str]) -> List[Dict[str, Any]]:
    out = []
    try:
        import SarahMemorySi as _Si  # type: ignore
    except Exception:
        return out
    for name in candidates or []:
        try:
            path = _Si.get_app_path(name)
        except Exception:
            path = None
        if path:
            out.append({"name": name, "path": path, "source": "SarahMemorySi", "version": _parse_version_from_text(path)})
    return out


def _software_research_lookup(candidates: List[str], limit: int = 12) -> List[str]:
    try:
        import SarahMemorySoftwareResearch as _SSR  # type: ignore
        items = _SSR.list_installed_software()
    except Exception:
        items = []
    if not isinstance(items, list):
        return []
    out = []
    for item in [str(x) for x in items if isinstance(x, str)]:
        if any(str(c).lower() in item.lower() for c in (candidates or [])):
            out.append(item)
    return out[:limit]


def _discover_driver_capabilities(route: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    entities = dict(route.get("entities") or {})
    requested_device = str(entities.get("device_type") or "").strip().lower()
    requested_vendor = str(entities.get("vendor") or "").strip().lower()
    out = {"requested_device": requested_device, "requested_vendor": requested_vendor, "driver_ids": [], "matches": [], "available": False}
    try:
        import appdrivers as _AppDrivers  # type: ignore
        ids = list(_AppDrivers._discover_driver_ids())
    except Exception:
        ids = []
        return out
    out["driver_ids"] = ids
    for did in ids:
        try:
            manifest = _AppDrivers._load_manifest(did) or {}
        except Exception:
            manifest = {}
        blob = json.dumps(manifest, ensure_ascii=False).lower()
        if requested_device and requested_device not in blob and requested_device not in str(did).lower():
            if not (requested_device == 'webcam' and ('camera' in blob or 'cam' in blob)):
                continue
        if requested_vendor and requested_vendor not in blob and requested_vendor not in str(did).lower():
            continue
        actions = manifest.get('actions') if isinstance(manifest.get('actions'), list) else []
        out['matches'].append({'driver_id': did, 'manifest': manifest, 'actions': actions})
    out['available'] = bool(out['matches'])
    return out


def _derive_schedule_spec(user_text: str, route: Dict[str, Any]) -> Dict[str, Any]:
    text = str(user_text or '').lower()
    if any(p in text for p in ['daily','every day','everyday']):
        return {'requested': True, 'pattern': 'daily', 'time_hint': '09:00', 'summary': 'Recurring daily task requested'}
    if 'weekly' in text or 'every week' in text:
        return {'requested': True, 'pattern': 'weekly', 'time_hint': '09:00', 'summary': 'Recurring weekly task requested'}
    if 'monthly' in text or 'every month' in text:
        return {'requested': True, 'pattern': 'monthly', 'time_hint': '09:00', 'summary': 'Recurring monthly task requested'}
    if str(route.get('route_id') or '').startswith('reminder.'):
        return {'requested': True, 'pattern': 'one_shot_or_parse', 'summary': 'Reminder or calendar action requested'}
    return {'requested': False}


def _driver_action_hint(route: Dict[str, Any], user_text: str) -> str:
    text = str(user_text or '').lower()
    entities = dict(route.get('entities') or {})
    control_name = str(entities.get('control_name') or entities.get('key_name') or '').lower()
    state = str(entities.get('requested_state') or '').lower()
    device_type = str(entities.get('device_type') or '').lower()
    if control_name in {'caps_lock', 'num_lock', 'scroll_lock'} or any(k in text for k in ('caps lock', 'num lock', 'scroll lock')):
        return 'keyboard_lock_set'
    if device_type == 'keyboard' and (entities.get('value') or any(k in text for k in ('color','light','lights','led','rgb','backlight'))):
        return 'keyboard_rgb_set'
    if state in {'on','open','enable','activate','start'}:
        return 'power_on'
    if state in {'off','disable','stop','close'}:
        return 'power_off'
    if entities.get('value') or any(k in text for k in ('color','light','lights','led','rgb')):
        return 'set_value'
    return 'generic_action'


def _unwrap_flaskish_response(obj: Any) -> Dict[str, Any]:
    try:
        if isinstance(obj, tuple) and len(obj) >= 1:
            resp = obj[0]
            status = obj[1] if len(obj) > 1 else None
            if hasattr(resp, 'get_json'):
                data = resp.get_json(silent=True) or {}
            else:
                data = resp if isinstance(resp, dict) else {'result': str(resp)}
            if status is not None and isinstance(data, dict):
                data.setdefault('status', status)
            return data if isinstance(data, dict) else {'result': data}
        if hasattr(obj, 'get_json'):
            data = obj.get_json(silent=True) or {}
            return data if isinstance(data, dict) else {'result': data}
        if isinstance(obj, dict):
            return obj
        return {'result': obj}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def _discover_runtime_capabilities(route: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    route_id = str(route.get('route_id') or '')
    entities = dict(route.get('entities') or {})
    software_terms = []
    capability_type = 'generic'
    if route_id == 'avatar.create.activate':
        capability_type = 'avatar_engine'
        engine_pref = str(entities.get('engine_preference') or '').lower()
        software_terms = [engine_pref] if engine_pref else ['unreal engine','unreal','blender']
    elif route_id == 'documents.office.write':
        capability_type = 'office_document'
        software_terms = ['microsoft word','word','winword','libreoffice writer']
    elif route_id == 'system.application.control':
        capability_type = 'application_control'
        target = str(entities.get('target_app') or '').strip().lower()
        software_terms = [target] if target else []
    elif route_id == 'email.mail.automation':
        capability_type = 'mail_automation'
        software_terms = ['outlook','thunderbird']
    elif route_id == 'drivers.device.control':
        capability_type = 'driver_control'
    clean_terms = [t for t in software_terms if t]
    system_index = _system_index_lookup(clean_terms)
    software_db = _software_db_lookup(clean_terms)
    si_hits = _si_lookup(clean_terms)
    software_research = _software_research_lookup(clean_terms)
    driver_caps = _discover_driver_capabilities(route, user_text) if route_id == 'drivers.device.control' else {}
    schedule_spec = _derive_schedule_spec(user_text, route) if route_id in {'email.mail.automation','reminder.schedule.task'} else {'requested': False}
    discovered_versions = []
    for coll in (si_hits, software_db, system_index.get('files', []), system_index.get('registry', [])):
        for item in coll:
            raw = json.dumps(item, ensure_ascii=False) if isinstance(item, dict) else str(item)
            ver = _parse_version_from_text(raw)
            if ver and ver not in discovered_versions:
                discovered_versions.append(ver)
    return {'capability_type': capability_type, 'software_terms': clean_terms, 'system_index': system_index, 'software_db': software_db, 'si_hits': si_hits, 'software_research': software_research, 'driver_capabilities': driver_caps, 'schedule_spec': schedule_spec, 'discovered_versions': discovered_versions[:8], 'has_discovery_hits': bool(si_hits or software_db or system_index.get('files') or system_index.get('registry') or software_research or driver_caps.get('matches'))}


def _make_execution_plan(route: Dict[str, Any], discovery: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    route_id = str(route.get('route_id') or 'chat.general')
    entities = dict(route.get('entities') or {})
    plan = {'route_id': route_id, 'transport_target': str(route.get('transport_target') or ''), 'target_module': str(route.get('target_module') or ''), 'requires_confirmation': True, 'steps': [], 'endpoint_calls': [], 'adapter_gap': None}
    if route_id == 'drivers.device.control':
        matches = ((discovery.get('driver_capabilities') or {}).get('matches') or [])
        if matches:
            driver_id = str(matches[0].get('driver_id') or '')
            action_id = _driver_action_hint(route, user_text)
            payload = {'payload': {'requested_action': action_id, 'entities': entities, 'user_text': user_text}}
            plan['steps'] = [f'Discover governed driver capabilities for {driver_id}.', f'Connect driver session for {driver_id} if not already active.', f'Send action {action_id} with extracted entities.']
            plan['endpoint_calls'] = [{'method': 'POST', 'path': f'/api/drivers/{driver_id}/discover', 'payload': payload['payload']}, {'method': 'POST', 'path': f'/api/drivers/{driver_id}/connect', 'payload': payload}, {'method': 'POST', 'path': f'/api/drivers/{driver_id}/actions/{action_id}', 'payload': payload}]
            plan['requires_confirmation'] = False
        else:
            plan['steps'] = ['Run governed driver discovery and manifest scan before attempting hardware control.']
            plan['adapter_gap'] = 'driver_or_mapping_missing'
    elif route_id == 'email.mail.automation':
        sched = discovery.get('schedule_spec') or {}
        plan['steps'] = ['List inbox/spam messages through the communications lane.', 'Prepare unsubscribe actions only for messages positively classified as spam or promotional.']
        plan['endpoint_calls'] = [{'method': 'POST', 'path': '/api/comm/email/list', 'payload': {'folder': entities.get('target_folder') or 'spam', 'limit': 50, 'source': 'neuron_ingress'}}]
        if sched.get('requested'):
            plan['steps'].append('Create recurring reminder/scheduler entry for ongoing cleanup.')
            plan['endpoint_calls'].append({'method': 'POST', 'path': '/api/comm/reminders/save', 'payload': {'title': 'Empty spam trash', 'body': 'Governed recurring spam-trash cleanup task requested from chat ingress.', 'status': 'active', 'source': 'neuron_ingress', 'extra': {'pattern': sched.get('pattern'), 'time_hint': sched.get('time_hint')}}})
    elif route_id == 'documents.office.write':
        plan['steps'] = ['Discover a word-processing runtime from indexed/system software data.', 'Generate requested document content through the writing lane.', 'Open the document in the discovered software under user authority.']
        if discovery.get('si_hits') or discovery.get('software_db') or discovery.get('system_index', {}).get('registry'):
            plan['requires_confirmation'] = False
        else:
            plan['adapter_gap'] = 'software_runtime_not_confirmed'
    elif route_id == 'avatar.create.activate':
        plan['steps'] = ['Discover the preferred 3D engine/runtime from indexed system knowledge.', 'Create or extend the governed avatar adapter if the engine capability is incomplete.', 'Publish the avatar into the Avatar Panel without requiring a frontend rebuild.']
        if not discovery.get('has_discovery_hits'):
            plan['adapter_gap'] = 'avatar_engine_adapter_missing_or_unconfirmed'
    elif route_id == 'system.application.control':
        target_app = str(entities.get('target_app') or 'requested application')
        requested_state = str(entities.get('requested_state') or 'open')
        plan['steps'] = [f'Resolve the runtime path for {target_app} from indexed software sources.', f'Perform governed application state action: {requested_state}.']
        if discovery.get('si_hits') or discovery.get('software_db') or discovery.get('system_index', {}).get('registry'):
            plan['requires_confirmation'] = False
        else:
            plan['adapter_gap'] = 'app_runtime_not_confirmed'
    return plan




def _generic_set_lock_key_state(key_name: str, requested_state: str) -> Dict[str, Any]:
    key_name = str(key_name or '').strip().lower()
    requested_state = str(requested_state or 'on').strip().lower()
    vk_map = {'caps_lock': 0x14, 'num_lock': 0x90, 'scroll_lock': 0x91}
    if key_name not in vk_map:
        return {'ok': False, 'error': 'unsupported_lock_key', 'key_name': key_name}
    if os.name != 'nt':
        return {'ok': False, 'error': 'unsupported_os', 'os': os.name, 'key_name': key_name}
    try:
        import ctypes, time as _time
        user32 = ctypes.WinDLL('user32', use_last_error=True)
        desired_on = requested_state != 'off'
        vk = vk_map[key_name]
        KEYEVENTF_EXTENDEDKEY = 0x0001
        KEYEVENTF_KEYUP = 0x0002
        changed = False
        for _ in range(4):
            current_on = bool(user32.GetKeyState(vk) & 1)
            if current_on == desired_on:
                return {'ok': True, 'key_name': key_name, 'requested_state': requested_state, 'final_state': 'on' if current_on else 'off', 'changed': changed}
            user32.keybd_event(vk, 0x45, KEYEVENTF_EXTENDEDKEY, 0)
            user32.keybd_event(vk, 0x45, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
            changed = True
            _time.sleep(0.05)
        final_on = bool(user32.GetKeyState(vk) & 1)
        return {'ok': final_on == desired_on, 'key_name': key_name, 'requested_state': requested_state, 'final_state': 'on' if final_on else 'off', 'changed': changed}
    except Exception as e:
        return {'ok': False, 'error': str(e), 'key_name': key_name, 'requested_state': requested_state}


def _generic_keyboard_rgb_set(color_value: str) -> Dict[str, Any]:
    try:
        import shutil, subprocess
        op = shutil.which('openrgb') or shutil.which('OpenRGB')
        if not op:
            return {'ok': False, 'error': 'openrgb_not_found'}
        color = str(color_value or '').strip().lower() or 'white'
        color_map = {'red':'FF0000','green':'00FF00','blue':'0000FF','purple':'800080','yellow':'FFFF00','white':'FFFFFF','orange':'FFA500','pink':'FFC0CB'}
        cmd = [op, '--mode', 'static']
        if color in color_map:
            cmd.extend(['--color', color_map[color]])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        return {'ok': proc.returncode == 0, 'color': color, 'stdout': proc.stdout[-500:], 'stderr': proc.stderr[-500:], 'returncode': proc.returncode}
    except Exception as e:
        return {'ok': False, 'error': str(e), 'color': str(color_value or '')}

def _execute_ingress_plan(route: Dict[str, Any], discovery: Dict[str, Any], user_text: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    meta = meta or {}
    route_id = str(route.get('route_id') or '')
    entities = dict(route.get('entities') or {})
    result = {'attempted': False, 'executed': False, 'mode': 'plan_only', 'details': {}}
    if _is_safe_mode() and route_id in {'drivers.device.control', 'avatar.create.activate'}:
        result['details'] = {'ok': False, 'reason': 'safe_mode_gate'}
        return result
    try:
        if route_id == 'system.application.control':
            target_app = str(entities.get('target_app') or '').strip()
            requested_state = str(entities.get('requested_state') or 'open').strip().lower()
            if target_app:
                import SarahMemorySi as _Si  # type: ignore
                cmd = f"{requested_state} {target_app}"
                ok = bool(_Si.manage_application_request(cmd)) if hasattr(_Si, 'manage_application_request') else False
                return {'attempted': True, 'executed': ok, 'mode': 'direct_internal_dispatch', 'details': {'ok': ok, 'command': cmd, 'target_app': target_app, 'requested_state': requested_state}}
        if route_id == 'email.mail.automation':
            import appcomm as _AppComm  # type: ignore
            folder = str(entities.get('target_folder') or 'spam')
            listed = _unwrap_flaskish_response(_AppComm._email_poll({'folder': folder, 'limit': 50, 'source': 'neuron_ingress'})) if hasattr(_AppComm, '_email_poll') else {'ok': False, 'error': 'email_poll_unavailable'}
            details = {'listed': listed}
            executed = bool(listed.get('ok'))
            sched = discovery.get('schedule_spec') or {}
            if sched.get('requested') and hasattr(_AppComm, '_reminder_upsert'):
                reminder = _AppComm._reminder_upsert({'title': 'Empty spam trash', 'body': 'Governed recurring spam-trash cleanup task requested from chat ingress.', 'status': 'active', 'source': 'neuron_ingress', 'extra': {'pattern': sched.get('pattern'), 'time_hint': sched.get('time_hint')}})
                details['scheduler'] = reminder if isinstance(reminder, dict) else {'result': reminder}
                executed = executed or bool(details['scheduler'].get('ok'))
            return {'attempted': True, 'executed': executed, 'mode': 'direct_internal_dispatch', 'details': details}
        if route_id == 'drivers.device.control':
            action_id = _driver_action_hint(route, user_text)
            control_name = str(entities.get('control_name') or entities.get('key_name') or '').lower()
            if action_id == 'keyboard_lock_set' and control_name in {'caps_lock', 'num_lock', 'scroll_lock'}:
                action_res = _generic_set_lock_key_state(control_name, str(entities.get('requested_state') or 'on'))
                return {'attempted': True, 'executed': bool(action_res.get('ok')), 'mode': 'generic_local_control', 'details': {'action': action_res, 'action_id': action_id}}
            if action_id == 'keyboard_rgb_set':
                action_res = _generic_keyboard_rgb_set(str(entities.get('value') or 'white'))
                if bool(action_res.get('ok')):
                    return {'attempted': True, 'executed': True, 'mode': 'generic_local_control', 'details': {'action': action_res, 'action_id': action_id}}
            import appdrivers as _AppDrivers  # type: ignore
            matches = ((discovery.get('driver_capabilities') or {}).get('matches') or [])
            if matches and hasattr(_AppDrivers, '_driver_discover') and hasattr(_AppDrivers, '_driver_connect'):
                driver_id = str(matches[0].get('driver_id') or '')
                discovered = _unwrap_flaskish_response(_AppDrivers._driver_discover(driver_id, payload={'source': 'neuron_ingress', 'entities': entities}))
                connected = _unwrap_flaskish_response(_AppDrivers._driver_connect(driver_id, cfg={}, connect_payload={'source': 'neuron_ingress', 'entities': entities}))
                action_res = {'ok': False, 'skipped': True}
                try:
                    mod, err = _AppDrivers._load_driver_module(driver_id)
                    if not err and mod is not None:
                        context = _AppDrivers._build_driver_context(driver_id, instance_id=(_AppDrivers._session_get(driver_id) or {}).get('instance_id'), extra={'action_id': action_id})
                        payload = {'requested_action': action_id, 'entities': entities, 'user_text': user_text}
                        if hasattr(mod, 'driver_action'):
                            action_res = _unwrap_flaskish_response(mod.driver_action(action_id=action_id, context=context, payload=payload))
                        elif hasattr(mod, f'action_{action_id}'):
                            action_res = _unwrap_flaskish_response(getattr(mod, f'action_{action_id}')(context=context, payload=payload))
                except Exception as e:
                    action_res = {'ok': False, 'error': str(e)}
                executed = bool(discovered.get('ok')) or bool(connected.get('ok')) or bool(action_res.get('ok'))
                return {'attempted': True, 'executed': executed, 'mode': 'direct_internal_dispatch', 'details': {'driver_id': driver_id, 'discover': discovered, 'connect': connected, 'action': action_res, 'action_id': action_id}}
            return {'attempted': True, 'executed': False, 'mode': 'direct_internal_dispatch', 'details': {'ok': False, 'error': 'driver_unavailable', 'action_id': action_id, 'entities': entities}}
    except Exception as e:
        return {'attempted': True, 'executed': False, 'mode': 'direct_internal_dispatch', 'details': {'ok': False, 'error': str(e)}}
    return result


def _ingress_execution_ticket(route: Dict[str, Any], trace: Dict[str, Any], user_text: str = '') -> NeuronResult:
    route_id = str(route.get('route_id') or 'chat.general')
    target_module = str(route.get('target_module') or '')
    entities = dict(route.get('entities') or {})
    discovery = _discover_runtime_capabilities(route, user_text)
    reply = f'Routed request to {target_module or "execution"} using virtual route {route_id}.'
    if route_id == 'drivers.device.control':
        device_type = str(entities.get('device_type') or 'device')
        requested_state = str(entities.get('requested_state') or entities.get('action') or 'requested')
        if discovery.get('driver_capabilities', {}).get('available'):
            matches = (discovery.get('driver_capabilities') or {}).get('matches') or []
            top_id = str(matches[0].get('driver_id') or 'driver') if matches else 'driver'
            reply = f'Routed request to driver control for {device_type} {requested_state}. Matching driver {top_id} is available for governed execution.'
        else:
            reply = f'Routed request to driver control for {device_type} {requested_state}. Runtime driver discovery is still required before execution.'
    elif route_id == 'documents.office.write':
        topic = str(entities.get('topic') or 'the requested topic')
        if discovery.get('si_hits') or discovery.get('software_db') or discovery.get('system_index', {}).get('registry'):
            reply = f'Routed request to document automation for content generation on {topic}. A word-processing capability was discovered and is ready for governed execution planning.'
        else:
            reply = f'Routed request to document automation for content generation on {topic}. Runtime software discovery is required before execution.'
    elif route_id == 'avatar.create.activate':
        engine_pref = str(entities.get('engine_preference') or 'preferred engine')
        if discovery.get('has_discovery_hits'):
            reply = f'Routed request to avatar creation using {engine_pref}. Engine discovery located candidate runtime assets for governed adapter planning.'
        else:
            reply = f'Routed request to avatar creation using {engine_pref}. Runtime engine discovery is required before adapter planning.'
    elif route_id == 'email.mail.automation':
        sched = discovery.get('schedule_spec') or {}
        if sched.get('requested'):
            reply = 'Routed request to communications automation. Email handling and recurring scheduler planning were detected for governed execution.'
        else:
            reply = 'Routed request to communications automation. Mail capability discovery completed for governed execution planning.'
    execution_plan = _make_execution_plan(route, discovery, user_text)
    execution_result = _execute_ingress_plan(route, discovery, user_text, trace)
    if execution_result.get('attempted'):
        if execution_result.get('executed'):
            reply = reply.rstrip('.') + '. Direct internal execution completed.'
        elif execution_result.get('mode') == 'direct_internal_dispatch':
            reply = reply.rstrip('.') + '. Direct execution was attempted but did not fully complete; keeping governed execution plan ready.'
    artifacts = {'route_ticket': route, 'executor': target_module, 'entities': entities, 'runtime_discovery': discovery, 'execution_plan': execution_plan, 'execution_result': execution_result}
    actions = list(execution_plan.get('endpoint_calls') or []) if isinstance(execution_plan.get('endpoint_calls'), list) else []
    if execution_result.get('attempted'):
        actions.append({'type': 'execution_result', 'data': execution_result})
    return NeuronResult(ok=True, reply=reply, confidence=max(0.61, float(route.get('confidence') or 0.61)), intent=str(route.get('intent_hint') or route.get('domain') or 'action'), source='ingress_router', artifacts=artifacts, trace=trace, actions=actions)


def neuron_route(user_text: str, meta: Optional[Dict[str, Any]] = None, policy: Optional[Dict[str, Any]] = None) -> NeuronResult:
    _init_db()
    budget = _budget_limits()

    meta = meta or {}
    policy = policy or {}
    approved_modules = _approved_lane_modules()
    allowed_tiers = dict(policy.get("allowed_tiers") or {"tier0": True, "tier1": True, "tier2": True, "tier3": True})
    if not approved_modules.get("websym"):
        allowed_tiers["tier1"] = False
    if not approved_modules.get("research"):
        allowed_tiers["tier2"] = False
    if not approved_modules.get("api"):
        allowed_tiers["tier3"] = False
    # Enforce request-scoped local-only as a hard restriction.
    if bool(meta.get("local_only") or meta.get("offline")):
        allowed_tiers["tier3"] = False
    inp = NeuronInput(text=user_text or "", meta=meta)
    ingress_route = _normalize_ingress_route(meta, user_text or "")
    inp.meta["ingress_route"] = ingress_route
    trace: Dict[str, Any] = {"tiers": [], "agents": [], "budget": budget, "intent": None, "advcu": {}, "ingress": ingress_route}
    trace["policy"] = {"allowed_tiers": allowed_tiers, "approved_modules": approved_modules}
    trace["core_governance"] = _core_governance_trace()

    # Tier-0: Fast deterministic math (bypass heavy routing/QA gates)
    if allowed_tiers.get('tier0', True):
        det = _try_logiccalc(inp.text)
        if det and bool(det.get('ok')):
            _trace_primary_lane(trace, 'answer', str(det.get('engine') or 'LogicCalc'))
            trace['tiers'].append({'tier': 0, 'engine': str(det.get('engine') or 'LogicCalc'), 'ok': True})
            trace['intent'] = 'math'
            artifacts = {'math': {'ok': True, 'expr': det.get('expr'), 'value': det.get('value'), 'engine': det.get('engine')}, 'deterministic': det}
            v = det.get('value')
            reply = str(v if v is not None else (det.get('text') or ''))
            try:
                fv = float(v)
                if abs(fv - round(fv)) < 1e-9:
                    reply = str(int(round(fv)))
                else:
                    reply = str(fv)
            except Exception:
                reply = str(det.get('text') or reply)
            return NeuronResult(
                ok=True,
                reply=reply,
                intent='math',
                confidence=0.99,
                source='logiccalc',
                artifacts=artifacts,
                trace=trace,
            )

    # Tier-0.5: AdvCU delegation
    adv = _advcu_analyze(inp.text)
    trace["advcu"] = {"intent": adv.get("intent"), "confidence": adv.get("confidence"), "has_command": bool(adv.get("command"))}
    inp.meta["sm_helper_payload"] = _build_helper_payload(inp.text, str(adv.get("intent") or ""), adv)

    # Intent selection
    intent = _classify_intent(inp.text)
    if ingress_route.get("intent_hint") and float(ingress_route.get("confidence") or 0.0) >= 0.58:
        intent = str(ingress_route.get("intent_hint") or intent).lower()
    try:
        adv_intent = str(adv.get("intent") or "").strip()
        adv_conf = adv.get("confidence")
        if adv_intent and isinstance(adv_conf, (int, float)) and float(adv_conf) >= 0.55:
            intent = adv_intent.lower()
    except Exception:
        pass

    inp.meta["intent"] = intent
    trace["intent"] = intent

    if str(ingress_route.get("route_id") or "").startswith("research.weather") and bool(allowed_tiers.get("tier2", True)) and not inp.meta.get("offline"):
        research_data = _try_research(inp.text)
        if research_data:
            _trace_primary_lane(trace, 'answer', 'SarahMemoryResearch')
            trace["tiers"].append({"tier": "ingress", "engine": "SemanticIngress->Research", "ok": True})
            merged, artifacts = _synthesize_evidence_reply("Here is what I found:", research_data)
            res = NeuronResult(ok=True, reply=merged, confidence=max(0.72, float(ingress_route.get("confidence") or 0.72)), intent=intent, source="research", artifacts={"ingress_route": ingress_route, **(artifacts or {})}, trace=trace)
            _log_event("route", intent, res.confidence, res.source, {"input": inp.text, "trace": trace, "artifacts_keys": list(res.artifacts.keys())})
            return res
        trace["tiers"].append({"tier": "ingress", "engine": "SemanticIngress->Research", "ok": False})

    if ingress_route.get("domain") in {"drivers", "system", "documents", "email", "network", "communication", "reminder", "avatar"} and float(ingress_route.get("confidence") or 0.0) >= 0.66:
        _trace_primary_lane(trace, 'action', str(ingress_route.get("target_module") or 'executor'))
        trace["tiers"].append({"tier": "ingress", "engine": "SemanticIngressExecutor", "ok": True})
        res = _ingress_execution_ticket(ingress_route, trace, inp.text)
        _log_event("route", intent, res.confidence, res.source, {"input": inp.text, "trace": trace, "artifacts_keys": list(res.artifacts.keys())})
        return res

    # Governance handshake: ask CognitiveServices for request-scoped policy and
    # optional CognitiveThinker co-review before routing high-impact work.
    gov_decision = None
    if _Cog and approved_modules.get("cognitive") and hasattr(_Cog, "govern_request"):
        try:
            gov_ctx = dict(inp.meta or {})
            gov_ctx.setdefault("local_only", bool(inp.meta.get("local_only") or inp.meta.get("offline")))
            gov_ctx.setdefault("safe_mode", _is_safe_mode())
            gov_ctx.setdefault("neuron_route", True)
            gov_ctx.setdefault("caller", "SarahMemoryNeuron.neuron_route")
            gov_ctx.setdefault("force_cognitive_thinker_consult", False)
            gov_decision = _Cog.govern_request(  # type: ignore[attr-defined]
                inp.text,
                caller="SarahMemoryNeuron.neuron_route",
                caller_context=gov_ctx,
                user_present=bool(inp.meta.get("user_present", True)),
                user_consented=bool(inp.meta.get("user_consented", False)),
                proposed_action=dict(inp.meta.get("proposed_action") or {}),
            )
            if isinstance(gov_decision, dict):
                policy_from_governor = dict(gov_decision.get("routing_policy") or {})
                if isinstance(policy_from_governor.get("allowed_tiers"), dict):
                    for tier_name, tier_allowed in policy_from_governor.get("allowed_tiers", {}).items():
                        allowed_tiers[tier_name] = bool(allowed_tiers.get(tier_name, True) and tier_allowed)
                trace["governance"] = {
                    "decision": gov_decision.get("decision"),
                    "risk": gov_decision.get("risk"),
                    "risk_score": gov_decision.get("risk_score"),
                    "require_user": gov_decision.get("require_user"),
                    "recommended_next": gov_decision.get("recommended_next"),
                    "coequal_governance": gov_decision.get("coequal_governance") or {},
                }
                trace["policy"]["allowed_tiers"] = allowed_tiers

                decision_name = str(gov_decision.get("decision") or "").upper()
                lane_family = str(gov_decision.get("lane_family") or "").lower()
                if decision_name in ("DENY", "REQUIRE_USER", "DEFER") and lane_family in ("action", "network", "system"):
                    msg = str(gov_decision.get("recommended_next") or "Request blocked by governance.")
                    return NeuronResult(
                        ok=(decision_name == "DEFER"),
                        reply=msg,
                        confidence=0.90 if decision_name != "DEFER" else 0.70,
                        intent=intent,
                        source="governance",
                        artifacts={"governance": gov_decision},
                        trace=trace,
                    )
        except Exception as e:
            trace.setdefault("governance", {})["error"] = str(e)

    # System/Diagnostics lane (Tier-0, safe reads; blocked in Public Web mode)
    system_kind = _detect_system_kind(inp.text, intent)
    if system_kind:
        if not bool(allowed_tiers.get("tier0", True)):
            trace["tiers"].append({"tier": 0, "engine": "SystemInfo", "ok": False, "reason": "policy_disallow"})
            return NeuronResult(
                ok=False,
                reply="System tools are disabled by policy in this runtime.",
                intent=intent,
                source="system",
                confidence=0.9,
                artifacts={},
                trace=trace,
            )

        if _is_public_device(inp.meta):
            trace["tiers"].append({"tier": 0, "engine": "SystemInfo", "ok": False, "reason": "public_web_restricted"})
            return NeuronResult(
                ok=False,
                reply="This request is not available in Public Web mode.",
                intent=intent,
                source="system",
                confidence=0.9,
                artifacts={},
                trace=trace,
            )

        if system_kind == "diagnostics":
            _trace_primary_lane(trace, 'system', 'SarahMemoryDiagnostics')
            diag = _run_quick_diagnostics()
            ok = bool(diag.get("ok", False))
            trace["tiers"].append({"tier": 0, "engine": "Diagnostics", "ok": ok})
            return NeuronResult(
                ok=ok,
                reply="Diagnostics completed." if ok else "Diagnostics failed (see artifacts for details).",
                intent="diagnostics",
                source="diagnostics",
                confidence=0.9 if ok else 0.6,
                artifacts={"diagnostics": diag},
                trace=trace,
            )

        if system_kind == "gpu":
            _trace_primary_lane(trace, 'system', 'GPUStats')
            g = _gpu_stats_summary()
            ok = bool(g.get("ok", False))
            trace["tiers"].append({"tier": 0, "engine": "GPUStats", "ok": ok})
            if ok:
                # Prefer normalized fields when available
                name = g.get("name") or "GPU"
                if "free_vram_gb" in g and "total_vram_gb" in g:
                    reply = f"{name}: {g.get('free_vram_gb')} GB free / {g.get('total_vram_gb')} GB total VRAM."
                elif "free_vram_mb" in g and "total_vram_mb" in g:
                    reply = f"{name}: {g.get('free_vram_mb')} MB free / {g.get('total_vram_mb')} MB total VRAM."
                else:
                    reply = f"{name}: GPU stats captured."
            else:
                reply = "GPU stats are not available on this system."
            return NeuronResult(
                ok=ok,
                reply=reply,
                intent="device_query",
                source="gpu_stats",
                confidence=0.9 if ok else 0.6,
                artifacts={"gpu": g},
                trace=trace,
            )

        if system_kind == "disk":
            _trace_primary_lane(trace, 'system', 'DiskUsage')
            base_dir = os.getcwd()
            try:
                import SarahMemoryGlobals as _G  # type: ignore

                base_dir = str(getattr(_G, "BASE_DIR", base_dir) or base_dir)
            except Exception:
                pass
            du = _disk_usage_summary(base_dir)
            ok = bool(du.get("ok", False))
            trace["tiers"].append({"tier": 0, "engine": "DiskUsage", "ok": ok, "path": base_dir})
            if ok:
                reply = f"Disk free space for {du.get('path')}: {du.get('free_gb')} GB free / {du.get('total_gb')} GB total."
            else:
                reply = "Disk usage is not available on this system."
            return NeuronResult(
                ok=ok,
                reply=reply,
                intent="device_query",
                source="disk_usage",
                confidence=0.9 if ok else 0.6,
                artifacts={"disk": du},
                trace=trace,
            )

        if system_kind == "system_stats":
            _trace_primary_lane(trace, 'system', 'SystemStats')
            s = _quick_system_stats()
            ok = bool(s.get("ok", False))
            trace["tiers"].append({"tier": 0, "engine": "SystemStats", "ok": ok})
            parts = []
            if "cpu_percent" in s:
                parts.append(f"CPU: {s.get('cpu_percent')}%")
            if "ram_available_gb" in s and "ram_total_gb" in s:
                parts.append(f"RAM: {s.get('ram_available_gb')} GB free / {s.get('ram_total_gb')} GB total")
            reply = " | ".join(parts) if parts else "System stats captured."
            return NeuronResult(
                ok=ok,
                reply=reply,
                intent="device_query",
                source="system_stats",
                confidence=0.85 if ok else 0.6,
                artifacts={"system": s},
                trace=trace,
            )

    # Vision lane: classify once, then either answer visually or hand off to Action lane.
    vision_request = None
    vision_res = None
    if _is_visual_request(inp.meta.get("intent") or intent, inp.text, meta=inp.meta, adv=adv):
        try:
            vision_request = _build_vision_request(inp.text, inp.meta, adv)
            inp.meta["vision_request"] = vision_request
            trace["vision_request"] = {
                "query_type": vision_request.get("query_type"),
                "subject": vision_request.get("requested_subject") or vision_request.get("subject"),
                "action_expectation": vision_request.get("action_expectation"),
            }
        except Exception:
            vision_request = None

    if vision_request and str(vision_request.get("action_expectation") or "answer_only") == "answer_only":
        vision_res = _try_vision_lane(inp.text, meta=inp.meta, adv=adv)
    if vision_res is not None:
        trace.setdefault("tiers", []).extend(list((vision_res.trace or {}).get("tiers") or []))
        _trace_primary_lane(trace, "answer", "SarahMemorySOBJE")
        trace["vision"] = {
            "query_type": vision_res.intent,
            "frame_available": bool(((vision_res.artifacts or {}).get("vision") or {}).get("frame_available")),
        }
        res = vision_res
    else:
            # Action lane: emit local execution ticket (never execute here)
        if _is_action_intent(intent, inp.text, adv):
            if not approved_modules.get("filesystem"):
                trace["tiers"].append({"tier": "action", "engine": "ActionTicket", "ok": False, "reason": "filesystem_not_registered"})
                return NeuronResult(
                    ok=False,
                    reply="Action lane is not available because SarahMemoryFilesystem is not registered and approved.",
                    confidence=0.25,
                    intent="action",
                    source="action_ticket",
                    artifacts={"action_ticket_error": "filesystem_not_registered"},
                    trace=trace,
                )
            ticket = _make_action_ticket(inp.text, inp.meta, adv)
            _trace_primary_lane(trace, 'action', str(ticket.get('executor') or 'SarahMemoryFilesystem'))
            trace["tiers"].append({"tier": "action", "engine": "ActionTicket", "ok": True, "action": ticket.get("action")})
            res = NeuronResult(
                ok=True,
                reply=f"ACTION_TICKET::{ticket.get('action')}::{ticket.get('ticket_id')}",
                confidence=0.74,
                intent="action",
                source="action_ticket",
                artifacts={"action_ticket": ticket},
                trace=trace,
            )
            _log_event("route", "action", res.confidence, res.source, {"input": inp.text, "trace": trace, "artifacts_keys": list(res.artifacts.keys())})
            return res



        # Creative lane: emit job ticket
        if _is_creative_intent(intent, inp.text, adv):
            _trace_primary_lane(trace, 'creative', 'SarahMemoryCanvasStudio')
            if not approved_modules.get("canvas"):
                trace["tiers"].append({"tier": "creative", "engine": "CanvasStudioTicket", "ok": False, "reason": "canvas_not_registered"})
                return NeuronResult(
                    ok=False,
                    reply="Creative lane is not available because SarahMemoryCanvasStudio is not registered and approved.",
                    confidence=0.25,
                    intent="creative",
                    source="creative_ticket",
                    artifacts={"creative_ticket_error": "canvas_not_registered"},
                    trace=trace,
                )
            kind = _creative_kind(intent, inp.text, adv)
            ticket = _make_creative_job_ticket(inp.text, kind, inp.meta, adv)
            trace["tiers"].append({"tier": "creative", "engine": "CanvasStudioTicket", "ok": True, "kind": kind})
            res = NeuronResult(
                ok=True,
                reply=f"CREATIVE_JOB_TICKET::{kind}::{ticket.get('job_id')}",
                confidence=0.78,
                intent=kind,
                source="creative_ticket",
                artifacts={"job_ticket": ticket},
                trace=trace,
            )
            _log_event("route", kind, res.confidence, res.source, {"input": inp.text, "trace": trace, "artifacts_keys": list(res.artifacts.keys())})
            return res

        # Tier-0: LogicCalc
        det = _try_logiccalc(inp.text)
        if det:
            inp.meta["deterministic_hit"] = True
            _trace_primary_lane(trace, 'answer', str(det.get('engine') or 'LogicCalc'))
            trace["tiers"].append({"tier": 0, "engine": str(det.get('engine') or 'LogicCalc'), "ok": True})
            reply = det.get("reply") if isinstance(det, dict) else None
            if not reply and isinstance(det, dict):
                reply = det.get("text") or det.get("result")
            reply = str(reply) if reply else "Deterministic engine produced output (no text payload)."
            res = NeuronResult(ok=True, reply=reply, confidence=0.78, intent=intent, source="logiccalc", artifacts={"det": det}, trace=trace)

        else:
            trace["tiers"].append({"tier": 0, "engine": "LogicCalc", "ok": False})

            # Tier-1: WebSYM
            sym = _try_websym(inp.text) if bool(allowed_tiers.get("tier1", True)) else None
            if sym:
                _trace_primary_lane(trace, 'answer', 'SarahMemoryWebSYM')
                trace["tiers"].append({"tier": 1, "engine": "WebSYM", "ok": True})
                res = NeuronResult(ok=True, reply=sym, confidence=0.66, intent=intent, source="websym", artifacts={}, trace=trace)
            else:
                trace["tiers"].append({"tier": 1, "engine": "WebSYM", "ok": False})

                # Tier-2: Research lane (evidence-backed)
                research_data = None
                if bool(allowed_tiers.get("tier2", True)) and intent == "research" and not inp.meta.get("offline"):
                    research_data = _try_research(inp.text)
                if research_data:
                    _trace_primary_lane(trace, 'answer', 'SarahMemoryResearch')
                    trace["tiers"].append({"tier": 2, "engine": "Research", "ok": True})
                    merged, artifacts = _synthesize_evidence_reply("Here is what I found:", research_data)
                    res = NeuronResult(ok=True, reply=merged, confidence=0.70, intent=intent, source="research", artifacts=artifacts, trace=trace)
                else:
                    trace["tiers"].append({"tier": 2, "engine": "Research", "ok": False})

                    # Tier-3: API
                    api_reply = _try_api(inp.text, meta=inp.meta) if bool(allowed_tiers.get("tier3", True)) else None
                    if api_reply:
                        _trace_primary_lane(trace, 'answer', 'SarahMemoryAPI')
                        trace["tiers"].append({"tier": 3, "engine": "SarahMemoryAPI", "ok": True})
                        res = NeuronResult(ok=True, reply=api_reply, confidence=0.62, intent=intent, source="api", artifacts={}, trace=trace)
                    else:
                        trace["tiers"].append({"tier": 3, "engine": "SarahMemoryAPI", "ok": False})
                        res = NeuronResult(
                            ok=False,
                            reply="No engine produced an answer. Provide more constraints or enable an applicable tier.",
                            confidence=0.35,
                            intent=intent,
                            source="neuron",
                            artifacts={},
                            trace=trace,
                        )

    # Graph link
    try:
        if intent != "empty":
            _GRAPH.link("user_query", intent, "classified_as", 0.7, {"q": inp.text[:200]})
    except Exception:
        pass

    # Curiosity prompts
    try:
        curiosity = _curiosity_prompts(intent, inp.text, budget)
        if curiosity:
            res.artifacts["curiosity"] = curiosity
    except Exception:
        pass

    # Parallel thought calibration
    agents: List[ThoughtAgent] = [AnalystAgent(), SkepticAgent(), OptimizerAgent(), EngineerAgent(), GovernorAgent()]
    agents = agents[: int(budget.get("max_parallel", 4))]

    draft = res.reply
    conf = float(res.confidence)
    for a in agents:
        try:
            draft, delta, notes = a.evaluate(inp, draft)
            conf = float(max(0.0, min(0.99, conf + float(delta))))
            trace["agents"].append({"agent": a.name, "delta": float(delta), "notes": notes})
        except Exception as e:
            trace["agents"].append({"agent": getattr(a, "name", "Agent"), "delta": 0.0, "notes": {"error": str(e)}})

    # Compare-based QA gate
    try:
        cd, cart = _qa_compare_gate(inp.text, draft, intent)
        if cart:
            res.artifacts.update(cart)
        conf = float(max(0.0, min(0.99, conf + float(cd))))
        trace["agents"].append({"agent": "CompareQA", "delta": float(cd), "notes": {"enabled": bool(_Compare)}})
    except Exception:
        pass

    # Low-confidence evidence repair
    try:
        if conf < 0.55 and not inp.meta.get("offline") and bool(getattr(config, "WEB_RESEARCH_ENABLED", True)):
            rdata = _try_research(inp.text)
            if rdata:
                merged, artifacts = _synthesize_evidence_reply(draft, rdata)
                if merged and str(merged).strip():
                    draft = str(merged).strip()
                    if not trace.get("primary_lane"):
                        _trace_primary_lane(trace, 'answer', 'SarahMemoryResearch')
                res.artifacts.update(artifacts)
                conf = float(max(0.0, min(0.99, conf + 0.06)))
                res.ok = True
                if getattr(res, "source", "neuron") in ("neuron", "", None):
                    res.source = "research"
                trace["tiers"].append({"tier": 2, "engine": "Research", "ok": True, "reason": "low_confidence_repair"})
    except Exception:
        pass

    _fallback_reply = "No engine produced an answer. Provide more constraints or enable an applicable tier."
    _draft_str = str(draft or "").strip()

    # If the route recovered content but draft still contains the default failure string,
    # promote the best available candidate from artifacts.
    try:
        if (not _draft_str) or (_draft_str == _fallback_reply):
            best_candidate = ""

            research_art = res.artifacts.get("research") if isinstance(res.artifacts, dict) else None
            if isinstance(research_art, dict):
                for key in ("data", "snippet"):
                    val = research_art.get(key)
                    if isinstance(val, str):
                        val = val.strip()
                        if val and not val.lower().startswith("sorry, i was unable to find any reliable information"):
                            best_candidate = val
                            break

            if not best_candidate:
                compare_art = res.artifacts.get("compare") if isinstance(res.artifacts, dict) else None
                if isinstance(compare_art, dict):
                    val = compare_art.get("api_response")
                    if isinstance(val, str):
                        val = val.strip()
                        if val and val != _fallback_reply and not val.lower().startswith("i couldn't find reliable information"):
                            best_candidate = val

            if best_candidate:
                draft = best_candidate
                _draft_str = best_candidate
                res.ok = True
                if getattr(res, "source", "neuron") in ("neuron", "", None):
                    res.source = "research"
                conf = float(max(conf, 0.58))
    except Exception:
        pass

    res.reply = _draft_str or _fallback_reply
    res.confidence = conf

    # Governance stamp
    res.artifacts.setdefault("governance", {})["neosky"] = "ARMED" if _neosky_armed() else "SAFE"

    # Log
    _log_event("route", intent, res.confidence, res.source, {"input": inp.text, "trace": trace, "artifacts_keys": list(res.artifacts.keys())})
    res.trace = trace
    return res


# -----------------------------------------------------------------------------
# Background neuron service (heartbeat-style)
# -----------------------------------------------------------------------------
_NEURON_THREAD: Optional[threading.Thread] = None
_NEURON_STOP = threading.Event()
_NEURON_Q: "queue.Queue[NeuronInput]" = queue.Queue(maxsize=100)

def neuron_submit(text: str, meta: Optional[Dict[str, Any]] = None) -> bool:
    try:
        _NEURON_Q.put_nowait(NeuronInput(text=text, meta=meta or {}))
        return True
    except Exception:
        return False

def neuron_tick() -> Optional[NeuronResult]:
    if _NEURON_Q.empty():
        return None
    try:
        inp = _NEURON_Q.get_nowait()
    except Exception:
        return None
    try:
        return neuron_route(inp.text, meta=inp.meta)
    except Exception as e:
        logger.error("neuron_tick error: %s", e)
        return NeuronResult(ok=False, reply=f"Neuron tick failed: {e}", confidence=0.1, intent="error", source="neuron")

def _neuron_loop(poll_s: float = 0.25) -> None:
    logger.info("[Neuron] background loop started")
    while not _NEURON_STOP.is_set():
        try:
            r = neuron_tick()
            if r:
                try:
                    if _Cog and _core_module_allowed("SarahMemoryCognitiveServices", "diagnostics", _Cog) and hasattr(_Cog, "notify_neuron_result"):
                        _Cog.notify_neuron_result(r.to_dict())  # type: ignore
                except Exception:
                    pass
        except Exception:
            pass
        time.sleep(poll_s)
    logger.info("[Neuron] background loop stopped")

def start_neuron_background() -> bool:
    global _NEURON_THREAD
    if _NEURON_THREAD and _NEURON_THREAD.is_alive():
        return True
    _NEURON_STOP.clear()
    _NEURON_THREAD = threading.Thread(target=_neuron_loop, daemon=True, name="SarahMemoryNeuronThread")
    _NEURON_THREAD.start()
    return True

def stop_neuron_background() -> bool:
    _NEURON_STOP.set()
    return True

def neuron_status() -> Dict[str, Any]:
    return {
        "running": bool(_NEURON_THREAD and _NEURON_THREAD.is_alive()),
        "queue": int(getattr(_NEURON_Q, "qsize", lambda: 0)()),
        "profile": _device_profile(),
        "db": _neuron_db_path(),
    }

# -----------------------------------------------------------------------------
# CLI quick test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    _init_db()
    print(json.dumps(neuron_status(), indent=2))
    q = " ".join(sys.argv[1:]).strip()
    if not q:
        q = "Convert 12 ft to meters and explain units."
    out = neuron_route(q, {"cli": True})
    print(json.dumps(out.to_dict(), indent=2))
    
    
# ====================================================================
# END OF SarahMemoryNeuron.py v8.0.0
# ====================================================================
