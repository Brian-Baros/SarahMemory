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
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
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

    # ✅ FIX: required by __main__ (and useful everywhere)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "reply": str(self.reply),
            "confidence": float(self.confidence),
            "intent": str(self.intent),
            "source": str(self.source),
            "artifacts": self.artifacts,
            "trace": self.trace,
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
    if any(k in t for k in ("calculate", "solve", "convert", "unit", "derivative", "integral", "matrix", "vector", "sqrt", "square root", "plus", "minus", "times", "multipl", "divide", "divided", "+", "-", "*", "/")):
        return "math"
    # quick numeric expression heuristic
    if re.search(r"\d", t) and any(op in t for op in ("+", "-", "*", "/", "sqrt", "square root", "plus", "minus", "times", "divided")):
        return "math"
    if any(k in t for k in ("diagnos", "self-test", "self test", "health check")):
        return "diagnostics"
    if any(k in t for k in ("gpu", "vram", "cuda", "disk space", "free disk", "free space", "storage", "drive space", "cpu", "ram", "memory usage", "system stats", "system status", "hardware stats")):
        return "device_query"
    if any(k in t for k in ("chem", "molar", "stoichi", "compound", "element", "reaction", "ph", "acid", "base")):
        return "chemistry"
    if any(k in t for k in ("optimize", "speed up", "performance", "refactor", "bug", "error", "traceback")):
        return "engineering"
    if any(k in t for k in ("who are you", "version", "creator", "brian", "softdev0")):
        return "identity"
    if any(k in t for k in ("research", "look up", "browse", "latest", "verify", "sources", "citation")):
        return "research"
    if any(k in t for k in ("generate", "create", "make", "draw", "song", "music", "video", "avatar", "image")):
        return "creative"
    return "general"


# -----------------------------------------------------------------------------
# Tier-0.5: AdvCU delegation (intent + command parsing)
# -----------------------------------------------------------------------------
def _advcu_analyze(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"intent": None, "confidence": None, "command": None, "entities": {}, "raw": {}}
    if not _AdvCU:
        return out

    try:
        parse_fn = getattr(_AdvCU, "parse_command", None)
        if callable(parse_fn):
            cmd = parse_fn(text)  # type: ignore
            if isinstance(cmd, dict):
                out["command"] = cmd
                out["intent"] = cmd.get("intent") or cmd.get("action") or cmd.get("type")
                out["entities"] = cmd.get("entities") or cmd.get("args") or {}
                out["raw"]["parse_command"] = cmd
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
        "file_", "filesystem"
    )):
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

    requires_confirm = True
    ticket_id = f"act_{int(time.time()*1000)}"
    return {
        "ticket_id": ticket_id,
        "action": executor_action,
        "args": args,
        "safety_level": safety_level,
        "requires_confirm": requires_confirm,
        "executor": "SarahMemoryFilesystem",
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
      1) SarahMemoryLogicCalc (if available)
      2) Safe fallback parser for basic arithmetic + sqrt when LogicCalc is unavailable
    """
    raw = (text or "").strip()
    if not raw:
        return None

    # Primary: project engine
    if _LogicCalc:
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
    try:
        # Inject resolver hints (no hardcoded repos here; all canonical names live in SarahMemoryGlobals.py).
        meta2 = dict(meta or {})
        try:
            if config and hasattr(config, "resolve_model"):
                rm = config.resolve_model("reasoning", text=text, meta=meta2)  # type: ignore[attr-defined]
                # Lightweight, non-breaking hint packet for downstream routing/logging.
                meta2.setdefault("sm_model_resolve", {})["reasoning"] = rm
                if isinstance(rm, dict) and rm.get("selected"):
                    meta2.setdefault("preferred_model_repo", rm.get("selected"))
        except Exception:
            pass

        fn = getattr(_SMAPI, "send_to_api", None)
        if callable(fn):
            resp = fn(text, **meta2)
            if isinstance(resp, str) and resp.strip():
                return resp
            if isinstance(resp, dict) and resp.get("reply"):
                return str(resp["reply"])
    except Exception:
        return None
    return None
    if not _SMAPI:
        return None
    try:
        fn = getattr(_SMAPI, "send_to_api", None)
        if callable(fn):
            resp = fn(text, **(meta or {}))
            if isinstance(resp, str) and resp.strip():
                return resp
            if isinstance(resp, dict) and resp.get("reply"):
                return str(resp["reply"])
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
# Public router surface
# -----------------------------------------------------------------------------
def neuron_route(user_text: str, meta: Optional[Dict[str, Any]] = None, policy: Optional[Dict[str, Any]] = None) -> NeuronResult:
    _init_db()
    budget = _budget_limits()

    meta = meta or {}
    policy = policy or {}
    allowed_tiers = dict(policy.get("allowed_tiers") or {"tier0": True, "tier1": True, "tier2": True, "tier3": True})
    # Enforce request-scoped local-only as a hard restriction.
    if bool(meta.get("local_only") or meta.get("offline")):
        allowed_tiers["tier3"] = False
    inp = NeuronInput(text=user_text or "", meta=meta)
    trace: Dict[str, Any] = {"tiers": [], "agents": [], "budget": budget, "intent": None, "advcu": {}}
    trace["policy"] = {"allowed_tiers": allowed_tiers}

    # Tier-0: Fast deterministic math (bypass heavy routing/QA gates)
    if allowed_tiers.get('tier0', True):
        det = _try_logiccalc(inp.text)
        if det and bool(det.get('ok')):
            trace['tiers'].append({'tier': 0, 'engine': 'LogicCalc', 'ok': True})
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

    # Intent selection
    intent = _classify_intent(inp.text)
    try:
        adv_intent = str(adv.get("intent") or "").strip()
        adv_conf = adv.get("confidence")
        if adv_intent and isinstance(adv_conf, (int, float)) and float(adv_conf) >= 0.55:
            intent = adv_intent.lower()
    except Exception:
        pass

    inp.meta["intent"] = intent
    trace["intent"] = intent

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

    # Action lane: emit local execution ticket (never execute here)
    if _is_action_intent(intent, inp.text, adv):
        ticket = _make_action_ticket(inp.text, inp.meta, adv)
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
        trace["tiers"].append({"tier": 0, "engine": "LogicCalc", "ok": True})
        reply = det.get("reply") if isinstance(det, dict) else None
        if not reply and isinstance(det, dict):
            reply = det.get("text") or det.get("result")
        reply = str(reply) if reply else "Deterministic engine produced output (no text payload)."
        res = NeuronResult(ok=True, reply=reply, confidence=0.78, intent=intent, source="logiccalc", artifacts={"det": det}, trace=trace)

    else:
        trace["tiers"].append({"tier": 0, "engine": "LogicCalc", "ok": False})

        # Tier-1: WebSYM
        sym = _try_websym(inp.text)
        if sym:
            trace["tiers"].append({"tier": 1, "engine": "WebSYM", "ok": True})
            res = NeuronResult(ok=True, reply=sym, confidence=0.66, intent=intent, source="websym", artifacts={}, trace=trace)
        else:
            trace["tiers"].append({"tier": 1, "engine": "WebSYM", "ok": False})

            # Tier-2: Research lane (evidence-backed)
            research_data = None
            if intent == "research" and not inp.meta.get("offline"):
                research_data = _try_research(inp.text)
            if research_data:
                trace["tiers"].append({"tier": 2, "engine": "Research", "ok": True})
                merged, artifacts = _synthesize_evidence_reply("Here is what I found:", research_data)
                res = NeuronResult(ok=True, reply=merged, confidence=0.70, intent=intent, source="research", artifacts=artifacts, trace=trace)
            else:
                trace["tiers"].append({"tier": 2, "engine": "Research", "ok": False})

                # Tier-3: API
                api_reply = _try_api(inp.text, meta=inp.meta)
                if api_reply:
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
                draft = merged
                res.artifacts.update(artifacts)
                conf = float(max(0.0, min(0.99, conf + 0.06)))
                trace["tiers"].append({"tier": 2, "engine": "Research", "ok": True, "reason": "low_confidence_repair"})
    except Exception:
        pass

    res.reply = draft
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
                    if _Cog and hasattr(_Cog, "notify_neuron_result"):
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
# END OF SarahMemoryNetwork.py v8.0.0
# ====================================================================
