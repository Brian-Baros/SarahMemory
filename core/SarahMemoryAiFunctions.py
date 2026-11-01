"""--==The SarahMemory Project==--
File: SarahMemoryAiFunctions.py
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

========================
✓ ALL legacy functions from v7.7.5 RESTORED
✓ ALL v8.0 advanced features INCLUDED
✓ Public callable signatures preserved; stricter security semantics apply
===============================================================================
"""
from __future__ import annotations
# --- SARAHMETA START ---
# GRADE = "B"
# ROLE = "agent_orchestrator"
# CATEGORY = "ai_operations"
# USER_FACING = False
# UI_EXPOSURE = "internal_tool"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "ai_functions"
# FAMILY = "agent_runtime"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Advanced agent runtime/orchestration layer with task planning, tool orchestration, knowledge graph, predictive modeling, and governed execution helpers."
# --- SARAHMETA END ---
from SarahMemoryAdvCU import classify_intent
import re
import logging
import time
import sqlite3
import os
import json
import numpy as np
import random
import threading
import queue
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import pickle  # retained for old imports only; never deserialize untrusted state
import atexit
import uuid
import html
import contextlib
import math
import urllib.parse

# === Optional embedding shim (offline-safe) ===
try:
    from SarahMemoryAdvCU import embed_text as _advcu_embed
except Exception:
    _advcu_embed = None
try:
    from SarahMemoryCanvasStudio import CanvasStudio
except Exception:
    CanvasStudio = None

try:
    from SarahMemoryGlobals import (
        sm_is_core_module_approved,
        sm_get_registered_core_modules,
        sm_get_core_governance_profile,
    )
except Exception:
    def sm_is_core_module_approved(_module_name: str) -> bool:
        return False
    def sm_get_registered_core_modules() -> dict:
        return {}
    def sm_get_core_governance_profile() -> dict:
        return {"dynamic_registration": False, "auto_expose_approved": True}

def _fallback_embed(text: str, dim: int = 64):
    """Deterministic token-feature projection, not a trained semantic model."""
    dim = max(8, min(int(dim or 64), 4096))
    vector = [0.0] * dim
    terms = re.findall(r"[a-z0-9_]+", str(text or "").casefold())[:2048]
    for term in terms:
        digest = hashlib.sha256(term.encode("utf-8")).digest()
        pos = int.from_bytes(digest[:4], "big") % dim
        vector[pos] += 1 if digest[4] & 1 else -1
    norm = math.sqrt(sum(x*x for x in vector))
    return [x/norm for x in vector] if norm else vector

def lite_embed(texts):
    """Return one finite vector per input; no model download or cloud inference."""
    rows = [texts] if isinstance(texts, str) else list(texts or [])[:256]
    if not rows:
        return []
    if _advcu_embed is not None:
        try:
            vecs = _advcu_embed(rows)
            if isinstance(vecs, np.ndarray):
                vecs = vecs.tolist()
            if len(vecs) == len(rows) and all(isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0 for v in vecs):
                clean = [[float(x) for x in v] for v in vecs]
                if len({len(v) for v in clean}) == 1 and all(math.isfinite(x) for v in clean for x in v):
                    return clean
        except Exception:
            pass
    return [_fallback_embed(str(t)) for t in rows]

# === Remote hub bridge (non-blocking) ===
_REMOTE_JOB_SLOTS = threading.BoundedSemaphore(2)
_HUB = None
def _get_hub():
    global _HUB
    if _HUB is None:
        if (not _sm_network_allowed() or not _sm_flag("AI_FUNCTIONS_CONTEXT_SYNC_ENABLED", False)
                or not _sm_flag("ALLOW_HUB_SYNC", False)):
            return None
        try:
            from SarahMemoryNetwork import ServerConnection
            _HUB = ServerConnection()
            try:
                _HUB.start_heartbeat()
            except Exception:
                pass
        except Exception:
            _HUB = None
    return _HUB

def _fire_and_forget(coro_func, *args, **kwargs):
    try:
        import threading, asyncio
        def run():
            try:
                asyncio.run(coro_func(*args, **kwargs))
            except Exception:
                pass
        if (not _sm_network_allowed() or not _sm_flag("AI_FUNCTIONS_CONTEXT_SYNC_ENABLED", False)
            or not _sm_flag("ALLOW_HUB_SYNC", False)):
            return
        if not _REMOTE_JOB_SLOTS.acquire(blocking=False):
            return
        def bounded_run():
            try:
                run()
            finally:
                _REMOTE_JOB_SLOTS.release()
        threading.Thread(target=bounded_run, name="SM-Remote-Call", daemon=True).start()
    except Exception:
        pass

# === Core imports ===
try:
    import speech_recognition as sr
except Exception:
    sr = None

from SarahMemoryVoice import synthesize_voice
import SarahMemoryGlobals as config

# === Agent Consent Flag ===
try:
    AI_AGENT_REQUIRE_CONSENT = bool(getattr(config, "AI_AGENT_REQUIRE_CONSENT", False))
except Exception:
    AI_AGENT_REQUIRE_CONSENT = False

# === Agent Runtime Controls (safe defaults) ===
try:
    AI_AGENT_ENABLED = bool(getattr(config, 'AI_AGENT_ENABLED', False))
except Exception:
    AI_AGENT_ENABLED = False

# AI_AGENT_REQUIRE_CONSENT is treated as a list/tuple of risky keywords (not a bool).
try:
    _consent_val = getattr(config, 'AI_AGENT_REQUIRE_CONSENT', [])
    if isinstance(_consent_val, (list, tuple, set)):
        AI_AGENT_REQUIRE_CONSENT = list(_consent_val)
    elif bool(_consent_val):
        AI_AGENT_REQUIRE_CONSENT = ['delete', 'format', 'shutdown', 'wipe', 'encrypt']
    else:
        AI_AGENT_REQUIRE_CONSENT = []
except Exception:
    AI_AGENT_REQUIRE_CONSENT = []

try:
    AI_AGENT_USER_ACTIVITY_TIMEOUT_MS = int(getattr(config, 'AI_AGENT_USER_ACTIVITY_TIMEOUT_MS', 2500))
except Exception:
    AI_AGENT_USER_ACTIVITY_TIMEOUT_MS = 2500
try:
    AI_AGENT_RESUME_DELAY = int(getattr(config, 'AI_AGENT_RESUME_DELAY', 9000))
except Exception:
    AI_AGENT_RESUME_DELAY = 9000

AI_AGENT_STOP_PHRASES = tuple(getattr(config, 'AI_AGENT_STOP_PHRASES', (
    'stop sarah', 'emergency stop', 'halt now', 'stop', 'abort'
))) if hasattr(config, 'AI_AGENT_STOP_PHRASES') else (
    'stop sarah', 'emergency stop', 'halt now', 'stop', 'abort'
)
AI_AGENT_HALT_PHRASES = tuple(getattr(config, 'AI_AGENT_HALT_PHRASES', (
    'pause', 'hold on', 'wait', 'halt'
))) if hasattr(config, 'AI_AGENT_HALT_PHRASES') else (
    'pause', 'hold on', 'wait', 'halt'
)
AI_AGENT_RESUME_PHRASES = tuple(getattr(config, 'AI_AGENT_RESUME_PHRASES', (
    'resume', 'continue', 'go ahead'
))) if hasattr(config, 'AI_AGENT_RESUME_PHRASES') else (
    'resume', 'continue', 'go ahead'
)
AI_AGENT_CONFIRM_YES = tuple(getattr(config, 'AI_AGENT_CONFIRM_YES', (
    'yes', 'yeah', 'yep', 'confirm', 'do it', 'ok', 'okay'
))) if hasattr(config, 'AI_AGENT_CONFIRM_YES') else (
    'yes', 'yeah', 'yep', 'confirm', 'do it', 'ok', 'okay'
)

# Internal agent state (legacy compatibility)
_watch_thread_started = False
_watch_thread = None
_watch_stop_event = threading.Event()
_last_mouse = None
_agent_state = {
    'mode': 'IDLE',
    'last_human_ts': 0,
    'resume_eta_ms': 0,
}

from SarahMemoryGlobals import DATASETS_DIR

# === Safe/offline and local-only support ===
try:
    from SarahMemoryGlobals import SAFE_MODE, LOCAL_ONLY_MODE, is_offline
except Exception:
    SAFE_MODE = False
    LOCAL_ONLY_MODE = False
    def is_offline():
        return False

# === Logging setup ===
logger = logging.getLogger('SarahMemoryAiFunctions')
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(_h)
logger.propagate = False


# ---------------------------------------------------------------------------
# RHYTHM COGNITION BRIDGE - v9.0.0
# ---------------------------------------------------------------------------
def _sm_rhythm_packet(context: Optional[Dict[str, Any]] = None, *, force_refresh: bool = False) -> Dict[str, Any]:
    """Best-effort cadence packet. Rhythm is pacing only, never action authority."""
    try:
        import SarahMemoryRhythmCognition as _Rhythm  # type: ignore
        fn = getattr(_Rhythm, "get_rhythm_cognition_packet", None)
        if callable(fn):
            pkt = fn(context or {}, force_refresh=force_refresh)
            if isinstance(pkt, dict):
                return pkt
    except Exception as exc:
        return {"ok": False, "rhythm_mode": "FOCUSED", "agent_step_interval_sec": 1.2, "error": str(exc), "execution_authority": False}
    return {"ok": False, "rhythm_mode": "FOCUSED", "agent_step_interval_sec": 1.2, "execution_authority": False}


def _sm_rhythm_interval(kind: str, default: float, context: Optional[Dict[str, Any]] = None) -> float:
    try:
        import SarahMemoryRhythmCognition as _Rhythm  # type: ignore
        if kind == "meta_cognitive":
            return max(1.0, float(getattr(_Rhythm, "get_thinker_interval")(context or {}, default=default)))
        if kind == "learning":
            pkt = _sm_rhythm_packet(context or {})
            return max(5.0, float(pkt.get("living_loop_interval_sec", default)) * 6.0)
        if kind == "diagnostic":
            pkt = _sm_rhythm_packet(context or {})
            return max(3.0, float(pkt.get("heartbeat_interval_sec", default)) * 6.0)
        if kind == "predictive":
            pkt = _sm_rhythm_packet(context or {})
            return max(2.0, float(pkt.get("agent_step_interval_sec", default)) * 12.0)
        if kind == "agent_watch":
            fn = getattr(_Rhythm, "get_agent_watch_interval", None)
            if callable(fn):
                return max(0.08, min(0.50, float(fn(context or {}, default=default))))
        if kind == "agent_step":
            fn = getattr(_Rhythm, "get_agent_step_interval", None)
            if callable(fn):
                return max(0.12, float(fn(context or {}, default=default)))
    except Exception:
        pass
    return float(default)


def _sm_rhythm_throttle(key: str, *, context: Optional[Dict[str, Any]] = None, min_interval_sec: Optional[float] = None) -> Dict[str, Any]:
    try:
        import SarahMemoryRhythmCognition as _Rhythm  # type: ignore
        fn = getattr(_Rhythm, "throttle_key", None)
        if callable(fn):
            out = fn(key, context=context or {}, min_interval_sec=min_interval_sec)
            if isinstance(out, dict):
                return out
    except Exception as exc:
        return {"ok": False, "allow": False, "decision": "DEFER", "error": str(exc), "execution_authority": False}
    return {"ok": False, "allow": False, "decision": "DEFER", "reason": "rhythm_unavailable", "execution_authority": False}


def _sm_module_approved(module_name: str) -> bool:
    try:
        return bool(sm_is_core_module_approved(module_name))
    except Exception:
        return False

def _sm_registered_modules() -> dict:
    try:
        reg = sm_get_registered_core_modules() or {}
        return reg if isinstance(reg, dict) else {}
    except Exception:
        return {}

def _sm_creative_studio_registry() -> Dict[str, Dict[str, Any]]:
    reg = _sm_registered_modules()
    out: Dict[str, Dict[str, Any]] = {}
    for module_name, row in reg.items():
        if not isinstance(row, dict):
            continue
        cap = str(row.get("capability") or "").lower()
        if cap == "creative" or module_name in {
            "SarahMemoryCanvasStudio",
            "SarahMemoryVideoEditorCore",
            "SarahMemoryMusicGenerator",
            "SarahMemoryLyricsToSong",
        }:
            out[module_name] = row
    return out

def _sm_tool_allowed(tool_name: str) -> bool:
    mapping = {
        "memory_search": "SarahMemoryCompare",
        "web_search": "SarahMemoryResearch",
        "calc": "SarahMemoryWebSYM",
        "open_url": "SarahMemoryAiFunctions",
        "agent": "SarahMemoryAiFunctions",
        "creative": "SarahMemoryCanvasStudio",
    }
    module_name = mapping.get(tool_name)
    if not module_name:
        return False
    return _sm_module_approved(module_name)

def _sm_tool_error(tool_name: str) -> str:
    return f"Tool '{tool_name}' is not currently registered/approved in the SarahMemory core registry."

# v9.1.0a enterprise action/sovereignty boundary. Caller-supplied text and
# config flags do not authenticate user approval. Direct side-effect routes
# stay disabled unless the existing governed caller explicitly enables them.
def _sm_flag(name: str, default: bool = False) -> bool:
    try:
        value = os.environ.get(name, getattr(config, name, default))
    except Exception:
        value = default
    return value if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes", "on"}


def _sm_network_allowed() -> bool:
    try:
        offline = bool(is_offline())
    except Exception:
        offline = True
    return bool(not (SAFE_MODE or LOCAL_ONLY_MODE or offline)
                and not _sm_flag("SAFE_MODE") and not _sm_flag("LOCAL_ONLY_MODE")
                and _sm_flag("AI_AGENT_ALLOW_NETWORK_TASKS", False))


def _sm_action_permission(kind: str) -> Tuple[bool, str]:
    """Non-authorizing preflight. Privileged calls require verified upstream authority.

    As the legacy string-only entrypoints cannot carry signed approval receipts,
    their side effects remain disabled by default, even if a runtime flag enables
    the capability. Governed executors can instead consume a staged task proposal.
    """
    if SAFE_MODE or _sm_flag("SAFE_MODE"):
        return False, "safe_mode_denies_actions"
    if kind in {"network", "url"} and not _sm_network_allowed():
        return False, "network_forbidden_by_runtime_mode"
    if kind in {"desktop", "application", "network", "url", "surface"}:
        if not _sm_flag("AI_FUNCTIONS_ALLOW_DIRECT_ACTIONS", False):
            return False, "governed_operatorcore_execution_required"
        # Direct execution is still not equivalent to origin/approval verification.
        if not _sm_flag("AI_FUNCTIONS_VERIFIED_EXECUTION_CONTEXT", False):
            return False, "trusted_execution_context_required"
    if kind in {"desktop", "application", "surface"} and not _sm_flag("AI_AGENT_ENABLED", False):
        return False, "agent_disabled"
    if kind == "desktop" and not _sm_flag("AI_AGENT_ALLOW_REMOTE_CONTROL", False):
        return False, "desktop_control_not_permitted"
    if kind in {"application", "surface"} and not _sm_flag("AI_AGENT_ALLOW_APP_LAUNCH", False):
        return False, "application_launch_not_permitted"
    return True, "preflight_passed_not_authorization"


def _sm_phrase(text: str, phrase: str) -> bool:
    return bool(re.search(r"(?<!\w)" + re.escape(phrase.lower()) + r"(?!\w)", (text or "").lower()))


def _sm_user_command(text: str) -> bool:
    raw = (text or "").strip().lower()
    if re.match(r"^(?:what|why|how|when|where|who|explain|describe|define|can you explain|tell me about)\b", raw):
        return False
    return bool(re.match(r"^(?:please\s+)?(?:open|launch|start|close|move|click|press|type|scroll|draw|write|create|send|run|install|delete|focus|maximize|minimize|search|research|calculate|compute|recall|remember|browse|navigate)\b", raw))

_TOOL_WORKER_SLOTS = threading.BoundedSemaphore(8)

# === Context Management ===
context_buffer = []
context_embeddings = []
CONTEXT_DB_PATH = os.path.join(config.DATASETS_DIR, 'context_history.db')
_CONTEXT_LOCK = threading.RLock()
_CONTEXT_INITIALIZED = False
_CONTEXT_DB_SCHEMA_READY = False


def _context_db_connect(timeout: float = 5.0) -> sqlite3.Connection:
    """Open context DB only for explicit persistence calls."""
    os.makedirs(os.path.dirname(CONTEXT_DB_PATH),exist_ok=True)
    con=sqlite3.connect(CONTEXT_DB_PATH,timeout=max(1.0,min(float(timeout),30.0)),check_same_thread=False)
    try:
        allowed={"cache_size","temp_store","mmap_size","busy_timeout","wal_autocheckpoint"}
        pragmas=getattr(config,"SQLITE_CONNECTION_PRAGMAS",{}) or {}
        for key,value in pragmas.items():
            if key in allowed and re.fullmatch(r"-?\d{1,12}",str(value)):
                con.execute("PRAGMA "+key+"="+str(value))
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.execute("PRAGMA foreign_keys=ON")
        con.execute("PRAGMA busy_timeout=5000")
        con.execute("PRAGMA wal_autocheckpoint=1000")
    except sqlite3.DatabaseError:
        pass
    return con


def _ensure_context_schema(con: sqlite3.Connection) -> None:
    global _CONTEXT_DB_SCHEMA_READY
    with _CONTEXT_LOCK:
        if _CONTEXT_DB_SCHEMA_READY:
            return
        con.execute(
            """CREATE TABLE IF NOT EXISTS context_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                input TEXT,
                embedding TEXT,
                final_response TEXT,
                source TEXT,
                intent TEXT
            )"""
        )
        con.execute("CREATE INDEX IF NOT EXISTS idx_context_history_timestamp ON context_history(timestamp)")
        con.commit()
        _CONTEXT_DB_SCHEMA_READY = True


def checkpoint_context_history(mode: str = "PASSIVE") -> Dict[str, Any]:
    """Checkpoint context WAL state without scanning unrelated storage."""
    requested = str(mode or "PASSIVE").strip().upper()
    if not os.path.isfile(CONTEXT_DB_PATH):
        return {"ok": False, "mode": requested, "reason": "context_database_not_initialized"}
    if requested not in {"PASSIVE", "FULL", "RESTART", "TRUNCATE"}:
        requested = "PASSIVE"
    try:
        with _CONTEXT_LOCK:
            con = _context_db_connect(timeout=3.0)
            try:
                row = con.execute(f"PRAGMA wal_checkpoint({requested})").fetchone()
                return {"ok": True, "mode": requested, "result": list(row or [])}
            finally:
                con.close()
    except Exception as exc:
        return {"ok": False, "mode": requested, "error": str(exc)}

try:
    from SarahMemoryGlobals import get_context_config, agent_permissions_summary, get_mesh_sync_config
except Exception:
    def get_context_config():
        return {
            "enabled":           bool(getattr(config, "ENABLE_CONTEXT_BUFFER", True)),
            "buffer_size":       int(getattr(config, "CONTEXT_BUFFER_SIZE", 20)),
            "max_turns":         int(getattr(config, "CONTEXT_BUFFER_SIZE", 20)),
            "max_age_sec":       259200,
            "persist_to_db":     True,
            "db_path":           CONTEXT_DB_PATH,
            "enrichment_enabled":True,
        }

    def agent_permissions_summary():
        return {
            "run_mode":               getattr(config, "RUN_MODE", "local"),
            "device_mode":            getattr(config, "DEVICE_MODE", "headless"),
            "safe_mode":              bool(getattr(config, "SAFE_MODE", False)),
            "local_only":             bool(getattr(config, "LOCAL_ONLY_MODE", False)),
            "agent_enabled":          bool(getattr(config, "AI_AGENT_ENABLED", False)),
            "allow_app_launch":       bool(getattr(config, "AI_AGENT_ALLOW_APP_LAUNCH", False)),
            "allow_file_write":       bool(getattr(config, "AI_AGENT_ALLOW_FILE_WRITE", False)),
            "allow_remote_control":   bool(getattr(config, "AI_AGENT_ALLOW_REMOTE_CONTROL", False)),
            "allow_network_tasks":    bool(getattr(config, "AI_AGENT_ALLOW_NETWORK_TASKS", False)),
            "user_activity_timeout":  int(getattr(config, "AI_AGENT_USER_ACTIVITY_TIMEOUT_MS", 2500)),
            "resume_delay_ms":        int(getattr(config, "AI_AGENT_RESUME_DELAY", 9000)),
        }

    def get_mesh_sync_config():
        return {
            "node_name":            getattr(config, "NODE_NAME", "SarahMemoryNode"),
            "mesh_enabled":         bool(getattr(config, "MESH_SYNC_ENABLED", True)),
            "hub_allowed":          bool(getattr(config, "ALLOW_HUB_SYNC", True)),
            "safe_mode":            bool(getattr(config, "SAFE_MODE", False)),
            "safe_mode_only":       bool(getattr(config, "MESH_SYNC_SAFE_MODE_ONLY", False)),
            "sarahnet_enabled":     bool(getattr(config, "SARAHNET_ENABLED", True)),
            "web_base":             getattr(config, "SARAH_WEB_BASE", "https://www.sarahmemory.com"),
            "remote_sync_enabled":  bool(getattr(config, "REMOTE_SYNC_ENABLED", True)),
            "heartbeat_sec":        float(getattr(config, "REMOTE_HEARTBEAT_SEC", 30)),
            "http_timeout":         float(getattr(config, "REMOTE_HTTP_TIMEOUT", 6.0)),
        }


# ================================================================================
# NEW v8.0 ADVANCED STRUCTURES
# ================================================================================

class TaskPriority(Enum):
    """Task priority levels for the hierarchical planner"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """Represents a single task in the hierarchical planning system"""
    id: str
    description: str
    intent: str
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    subtasks: List['Task'] = field(default_factory=list)
    tools_required: List[str] = field(default_factory=list)
    estimated_duration: float = 0.0
    actual_duration: float = 0.0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert task to dictionary for serialization"""
        return {
            'id': self.id,
            'description': self.description,
            'intent': self.intent,
            'priority': self.priority.name,
            'status': self.status.value,
            'dependencies': self.dependencies,
            'subtasks': [st.to_dict() for st in self.subtasks],
            'tools_required': self.tools_required,
            'estimated_duration': self.estimated_duration,
            'actual_duration': self.actual_duration,
            'result': str(self.result) if self.result else None,
            'error': self.error,
            'retry_count': self.retry_count,
            'metadata': self.metadata
        }

@dataclass
class DecisionPoint:
    """Represents a decision made by the AI agent with reasoning"""
    timestamp: float
    context: str
    decision: str
    reasoning: List[str]
    confidence: float
    alternatives_considered: List[Tuple[str, float]]
    outcome: Optional[str] = None
    was_correct: Optional[bool] = None
    learned_lesson: Optional[str] = None

@dataclass
class KnowledgeNode:
    """Node in the knowledge graph for contextual understanding"""
    id: str
    content: str
    embedding: List[float]
    node_type: str  # 'fact', 'concept', 'procedure', 'preference', 'experience'
    timestamp: float
    connections: Dict[str, float] = field(default_factory=dict)  # id -> strength
    access_count: int = 0
    last_accessed: float = 0.0
    reliability: float = 1.0
    source: str = "user_interaction"

class PerformanceMetrics:
    """Tracks agent performance for self-improvement"""
    def __init__(self):
        self.task_success_rate: Dict[str, List[bool]] = defaultdict(list)
        self.tool_performance: Dict[str, List[float]] = defaultdict(list)
        self.intent_accuracy: Dict[str, List[bool]] = defaultdict(list)
        self.response_times: List[float] = []
        self.user_satisfaction: List[float] = []
        self.learning_rate: float = 0.01
        self.adaptation_history: List[Dict] = []

    def record_task_outcome(self, task_type: str, success: bool):
        """Record task execution outcome"""
        self.task_success_rate[task_type].append(success)
        if len(self.task_success_rate[task_type]) > 100:
            self.task_success_rate[task_type].pop(0)

    def record_tool_execution(self, tool_name: str, duration: float, success: bool):
        """Record tool execution metrics"""
        score = duration if success else float('inf')
        self.tool_performance[tool_name].append(score)
        if len(self.tool_performance[tool_name]) > 100:
            self.tool_performance[tool_name].pop(0)

    def get_tool_reliability(self, tool_name: str) -> float:
        """Calculate tool reliability score"""
        if tool_name not in self.tool_performance:
            return 0.5
        scores = self.tool_performance[tool_name]
        successful = [s for s in scores if s != float('inf')]
        return len(successful) / len(scores) if scores else 0.5

    def get_success_rate(self, task_type: str) -> float:
        """Get success rate for specific task type"""
        if task_type not in self.task_success_rate:
            return 0.5
        results = self.task_success_rate[task_type]
        return sum(results) / len(results) if results else 0.5

# ================================================================================
# GLOBAL STATE FOR ADVANCED AGENT
# ================================================================================

class AdvancedAgentState:
    """Centralized state management for the advanced AI agent"""
    def __init__(self):
        # Hierarchical planning
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=128)
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        self.task_graph: Dict[str, Set[str]] = defaultdict(set)

        # Meta-cognitive reasoning
        self.decision_history: deque = deque(maxlen=500)
        self.reasoning_chains: List[List[str]] = []
        self.confidence_threshold: float = 0.7
        self.self_reflection_interval: int = 10  # Reflect every N interactions
        self.interaction_count: int = 0

        # Knowledge graph
        self.knowledge_graph: Dict[str, KnowledgeNode] = {}
        self.concept_embeddings: Dict[str, np.ndarray] = {}
        self.relationship_types = ['causes', 'enables', 'requires', 'contradicts', 'supports', 'similar_to']

        # Performance tracking
        self.metrics = PerformanceMetrics()
        self.bottlenecks: List[Dict] = []
        self.optimization_suggestions: List[str] = []

        # Predictive modeling
        self.user_patterns: Dict[str, List[Dict]] = defaultdict(list)
        self.predicted_intents: queue.Queue = queue.Queue(maxsize=5)
        self.prediction_confidence: Dict[str, float] = {}

        # Multi-agent coordination
        self.peer_agents: Dict[str, Dict] = {}
        self.shared_knowledge: Dict[str, Any] = {}
        self.collaboration_requests: queue.Queue = queue.Queue(maxsize=64)

        # Autonomous learning
        self.learning_buffer: List[Dict] = []
        self.skill_improvements: Dict[str, float] = defaultdict(float)
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.success_patterns: Dict[str, int] = defaultdict(int)

        # Emotional intelligence
        self.emotional_context: Dict[str, Any] = {
            'user_mood': 'neutral',
            'conversation_tone': 'neutral',
            'stress_level': 0.0,
            'engagement_level': 0.5
        }

        # Tool orchestration
        self.tool_registry: Dict[str, Dict] = {}
        self.tool_dependencies: Dict[str, List[str]] = defaultdict(list)
        self.parallel_execution_enabled: bool = True
        self.max_parallel_tools: int = 5

        # Causal reasoning
        self.causal_model: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.intervention_history: List[Dict] = []
        self.outcome_predictions: Dict[str, List[float]] = defaultdict(list)

        # Self-diagnostic
        self.health_metrics: Dict[str, float] = {
            'response_latency': 0.0,
            'memory_usage': 0.0,
            'error_rate': 0.0,
            'learning_effectiveness': 1.0
        }
        self.diagnostic_queue: queue.Queue = queue.Queue(maxsize=64)

        # Thread management
        self.executor_threads: List[threading.Thread] = []
        self.shutdown_flag: threading.Event = threading.Event()
        self.lock = threading.RLock()
        self._initialized = False

    def initialize(self):
        """Initialize once; repeated imports never duplicate background workers."""
        with self.lock:
            if self._initialized and any(t.is_alive() for t in self.executor_threads):
                return
            logger.info("[AdvancedAgent] Initializing v8.0 Advanced Agent Systems...")
            self.shutdown_flag.clear()
            self._load_persistent_state()
            self._start_background_processes()
            self._initialized = True
            logger.info("[AdvancedAgent] Initialization complete.")

    def _load_persistent_state(self):
        """Load bounded JSON v9.1 state; never unpickle legacy files."""
        state_file = os.path.join(DATASETS_DIR, "advanced_agent_state.json")
        try:
            if not os.path.isfile(state_file) or os.path.getsize(state_file) > 4 * 1024 * 1024:
                return
            with open(state_file, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if not isinstance(saved, dict) or saved.get("schema") != "SarahMemory.AiFunctions.State.v9.1":
                logger.warning("[AdvancedAgent] Invalid state schema; skipped")
                return
            nodes = saved.get("knowledge_graph") or []
            for row in nodes[:512] if isinstance(nodes, list) else []:
                if not isinstance(row, dict) or not isinstance(row.get("content"), str):
                    continue
                try:
                    node = KnowledgeNode(id=str(row.get("id") or "")[:120],
                        content=row["content"][:2048],
                        embedding=[float(v) for v in (row.get("embedding") or [])[:4096]],
                        node_type=str(row.get("node_type") or "experience")[:40],
                        timestamp=float(row.get("timestamp") or 0),
                        source=str(row.get("source") or "restored")[:96],
                        reliability=max(0.0, min(1.0, float(row.get("reliability", 0.5)))))
                    if node.id and node.embedding and all(math.isfinite(x) for x in node.embedding):
                        self.knowledge_graph[node.id] = node
                except (ValueError, TypeError, OverflowError):
                    continue
            metrics = saved.get("task_success_rate") or {}
            if isinstance(metrics, dict):
                for k,v in list(metrics.items())[:64]:
                    if isinstance(v,list):
                        self.metrics.task_success_rate[str(k)[:80]] = [bool(x) for x in v[-100:]]
            patterns = saved.get("user_patterns") or {}
            if isinstance(patterns,dict):
                for k, v in list(patterns.items())[:128]:
                    if isinstance(v,list):
                        self.user_patterns[str(k)[:96]] = [x for x in v[-32:] if isinstance(x,dict)]
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("[AdvancedAgent] JSON state read failed: %s", exc)

    def _start_background_processes(self):
        """Start exactly one daemon worker per lifecycle loop."""
        alive_by_name = {t.name for t in self.executor_threads if t.is_alive()}
        specs = (
            ("MetaCognitive", self._meta_cognitive_loop),
            ("LearningLoop", self._learning_loop),
            ("SelfDiagnostic", self._self_diagnostic_loop),
            ("PredictiveModel", self._predictive_modeling_loop),
        )
        started = 0
        for name, target in specs:
            if name in alive_by_name:
                continue
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self.executor_threads.append(thread)
            started += 1
        self.executor_threads = [t for t in self.executor_threads if t.is_alive()]
        logger.info("[AdvancedAgent] Started %d background processes", started)

    def _meta_cognitive_loop(self):
        """Continuous meta-cognitive reasoning and self-reflection"""
        while not self.shutdown_flag.is_set():
            try:
                if self.shutdown_flag.wait(_sm_rhythm_interval("meta_cognitive", 30.0, {"loop": "meta_cognitive"})):
                    break
                with self.lock:
                    self._perform_self_reflection()
            except Exception as e:
                logger.error(f"[MetaCognitive] Error in reflection loop: {e}")

    def _learning_loop(self):
        """Continuous learning from interactions"""
        while not self.shutdown_flag.is_set():
            try:
                if self.shutdown_flag.wait(_sm_rhythm_interval("learning", 60.0, {"loop": "learning"})):
                    break
                with self.lock:
                    self._process_learning_buffer()
            except Exception as e:
                logger.error(f"[LearningLoop] Error: {e}")

    def _self_diagnostic_loop(self):
        """Monitor and diagnose performance issues"""
        while not self.shutdown_flag.is_set():
            try:
                if self.shutdown_flag.wait(_sm_rhythm_interval("diagnostic", 45.0, {"loop": "diagnostic"})):
                    break
                with self.lock:
                    self._run_diagnostics()
            except Exception as e:
                logger.error(f"[SelfDiagnostic] Error: {e}")

    def _predictive_modeling_loop(self):
        """Predict future user needs"""
        while not self.shutdown_flag.is_set():
            try:
                if self.shutdown_flag.wait(_sm_rhythm_interval("predictive", 20.0, {"loop": "predictive"})):
                    break
                with self.lock:
                    self._generate_predictions()
            except Exception as e:
                logger.error(f"[PredictiveModel] Error: {e}")

    def _perform_self_reflection(self):
        """Analyze recent decisions and performance"""
        if len(self.decision_history) < 5:
            return

        recent_decisions = list(self.decision_history)[-10:]
        observed = [d for d in recent_decisions if d.was_correct is not None]
        if len(observed) < 5:
            return
        successful = sum(1 for d in observed if d.was_correct is True)
        accuracy = successful / len(observed)

        if accuracy < 0.6:
            logger.warning(f"[MetaCognitive] Low decision accuracy: {accuracy:.2%}")
            self._adjust_confidence_threshold(decrease=True)
        elif accuracy > 0.85:
            logger.info(f"[MetaCognitive] High decision accuracy: {accuracy:.2%}")
            self._adjust_confidence_threshold(decrease=False)

    def _adjust_confidence_threshold(self, decrease: bool):
        """Adjust confidence threshold based on performance"""
        if decrease:
            self.confidence_threshold = min(0.9, self.confidence_threshold + 0.05)
        else:
            self.confidence_threshold = max(0.5, self.confidence_threshold - 0.02)
        logger.info(f"[MetaCognitive] Adjusted confidence threshold to {self.confidence_threshold:.2f}")

    def _process_learning_buffer(self):
        """Process only explicitly enabled bounded learning observations."""
        if not _sm_flag("AI_FUNCTIONS_LEARNING_ENABLED", False):
            self.learning_buffer.clear()
            return
        for experience in self.learning_buffer[:10]:
            if isinstance(experience, dict):
                try:
                    self._incorporate_experience(experience)
                except Exception as exc:
                    logger.debug("[Learning] Skipped malformed observation: %s", exc)
        del self.learning_buffer[:10]

    def _incorporate_experience(self, experience: Dict):
        """Incorporate a learning experience into the knowledge base"""
        exp_type = experience.get('type', 'unknown')

        if exp_type == 'task_success':
            pattern = experience.get('pattern')
            self.success_patterns[pattern] += 1
            self.skill_improvements[pattern] = self.skill_improvements[pattern] * 0.9 + 0.1
        elif exp_type == 'task_failure':
            pattern = experience.get('pattern')
            self.error_patterns[pattern] += 1
            if self.error_patterns[pattern] > 5:
                logger.warning(f"[Learning] Frequent failures in pattern: {pattern}")

    def _run_diagnostics(self):
        """Run self-diagnostics on agent performance"""
        try:
            # Check response latency
            if self.metrics.response_times:
                avg_latency = sum(self.metrics.response_times[-20:]) / len(self.metrics.response_times[-20:])
                self.health_metrics['response_latency'] = avg_latency
                if avg_latency > 2.0:
                    self.bottlenecks.append({
                        'type': 'high_latency',
                        'value': avg_latency,
                        'timestamp': time.time()
                    })

            # Check error patterns
            total_errors = sum(self.error_patterns.values())
            total_attempts = total_errors + sum(self.success_patterns.values())
            if total_attempts > 0:
                error_rate = total_errors / total_attempts
                self.health_metrics['error_rate'] = error_rate
                if error_rate > 0.2:
                    logger.warning(f"[SelfDiagnostic] High error rate detected: {error_rate:.2%}")
        except Exception as e:
            logger.error(f"[SelfDiagnostic] Diagnostic error: {e}")

    def _generate_predictions(self):
        """Make bounded, observational predictions without autonomous activation."""
        try:
            now = datetime.now()
            for key, rows in list(self.user_patterns.items())[:128]:
                if not isinstance(rows, list) or len(rows) < 3:
                    continue
                recent = [x for x in rows[-32:] if isinstance(x,dict) and
                    x.get("day") == now.weekday() and abs(int(x.get("hour", -100)) - now.hour) < 2]
                if len(recent) < 2:
                    continue
                score = len(recent) / min(32, len(rows))
                if score > 0.3 and not self.predicted_intents.full():
                    self.predicted_intents.put_nowait({"intent": recent[-1].get("intent"), "confidence": score, "context": key, "execution_authority": False})
        except Exception as exc:
            logger.debug("[PredictiveModel] Prediction unavailable: %s", exc)

    def shutdown(self, timeout: float = 3.0) -> Dict[str, Any]:
        """Stop lifecycle workers within one shared deadline and persist state."""
        if not self._initialized and not any(t.is_alive() for t in self.executor_threads):
            return {"ok": True, "joined": [], "alive": [], "standby": True}
        self.shutdown_flag.set()
        deadline = time.monotonic() + max(0.1, float(timeout))
        joined = []
        alive = []
        for thread in list(self.executor_threads):
            remaining = max(0.0, deadline - time.monotonic())
            if remaining > 0.0 and thread.is_alive() and thread is not threading.current_thread():
                thread.join(timeout=remaining)
            (alive if thread.is_alive() else joined).append(thread.name)
        try:
            self.save_state()
        except Exception:
            pass
        self.executor_threads = [t for t in self.executor_threads if t.is_alive()]
        self._initialized = bool(self.executor_threads)
        return {"ok": not alive, "joined": joined, "alive": alive}

    def save_state(self):
        """Persist bounded, versioned JSON only when explicitly enabled."""
        if not _sm_flag("AI_FUNCTIONS_PERSISTENT_LEARNING_ENABLED", False):
            return
        state_file = os.path.join(DATASETS_DIR, "advanced_agent_state.json")
        try:
            nodes = list(self.knowledge_graph.values())[-512:]
            state = {
                "schema": "SarahMemory.AiFunctions.State.v9.1", "timestamp": time.time(),
                "knowledge_graph": [
                    {"id": str(n.id)[:120], "content": str(n.content)[:2048],
                     "embedding": [float(x) for x in list(n.embedding)[:4096]],
                     "node_type": str(n.node_type)[:40], "timestamp": n.timestamp,
                     "source": str(n.source)[:96], "reliability": n.reliability}
                    for n in nodes],
                "task_success_rate": {str(k)[:80]: list(v)[-100:] for k,v in list(self.metrics.task_success_rate.items())[:64]},
                "user_patterns": {str(k)[:96]: list(v)[-32:] for k,v in list(self.user_patterns.items())[:128]},
            }
            encoded = json.dumps(state, allow_nan=False, ensure_ascii=False).encode("utf-8")
            if len(encoded) > 4 * 1024 * 1024:
                logger.warning("[AdvancedAgent] State exceeds persistence capacity")
                return
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            tmp_file = state_file + f".{os.getpid()}.{threading.get_ident()}.tmp"
            try:
                with open(tmp_file, "wb") as f:
                    f.write(encoded)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_file, state_file)
            finally:
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)
        except Exception as exc:
            logger.error("[AdvancedAgent] JSON state save failed: %s", exc)

# Global advanced agent state
ADVANCED_AGENT = AdvancedAgentState()

# ================================================================================
# HIERARCHICAL PLANNING ENGINE
# ================================================================================

class HierarchicalPlanner:
    """Advanced hierarchical task planner with goal decomposition"""

    def __init__(self, agent_state: AdvancedAgentState):
        self.agent = agent_state
        self.task_templates: Dict[str, Dict] = self._initialize_task_templates()
        self.decomposition_rules: Dict[str, List] = self._initialize_decomposition_rules()

    def _initialize_task_templates(self) -> Dict[str, Dict]:
        """Initialize templates for common task types"""
        return {
            'web_research': {
                'tools': ['memory_search', 'web_search', 'web_fetch'],
                'estimated_duration': 5.0,
                'can_parallelize': True
            },
            'file_operation': {
                'tools': ['file_read', 'file_write', 'file_search'],
                'estimated_duration': 2.0,
                'can_parallelize': False
            },
            'calculation': {
                'tools': ['calc', 'symbolic_calc'],
                'estimated_duration': 0.5,
                'can_parallelize': True
            },
            'system_control': {
                'tools': ['agent', 'open_app', 'system_command'],
                'estimated_duration': 3.0,
                'can_parallelize': False
            },
            'communication': {
                'tools': ['send_message', 'email', 'notification'],
                'estimated_duration': 2.0,
                'can_parallelize': True
            }
        }

    def _initialize_decomposition_rules(self) -> Dict[str, List]:
        """Initialize rules for task decomposition"""
        return {
            'research_and_summarize': [
                ('research', 'web_research'),
                ('analyze', 'analysis'),
                ('summarize', 'summarization')
            ],
            'compare_options': [
                ('gather_option_1', 'web_research'),
                ('gather_option_2', 'web_research'),
                ('compare', 'comparison'),
                ('recommend', 'decision_making')
            ],
            'automate_workflow': [
                ('analyze_workflow', 'analysis'),
                ('plan_automation', 'planning'),
                ('implement', 'system_control'),
                ('test', 'testing'),
                ('deploy', 'deployment')
            ]
        }

    def create_task(self, description: str, intent: str, priority: TaskPriority = TaskPriority.NORMAL) -> Task:
        """Create a new task with automatic decomposition"""
        description = str(description or "")[:2000]
        task_id = uuid.uuid4().hex[:12]

        task = Task(
            id=task_id,
            description=description,
            intent=intent,
            priority=priority,
            metadata={'creation_time': time.time()}
        )

        # Check if task needs decomposition
        if self._should_decompose(task):
            subtasks = self._decompose_task(task)
            task.subtasks = subtasks
            logger.info(f"[HierarchicalPlanner] Decomposed task '{description}' into {len(subtasks)} subtasks")

        # Identify required tools
        task.tools_required = self._identify_tools(task)

        # Estimate duration
        task.estimated_duration = self._estimate_duration(task)

        return task

    def _should_decompose(self, task: Task) -> bool:
        """Determine if task should be decomposed into subtasks"""
        complex_keywords = ['research and', 'compare', 'analyze', 'create report',
                           'automate', 'optimize', 'design', 'implement']
        return any(kw in task.description.lower() for kw in complex_keywords)

    def _decompose_task(self, task: Task) -> List[Task]:
        """Decompose complex task into subtasks"""
        subtasks = []
        description_lower = task.description.lower()

        # Check decomposition rules
        for pattern, steps in self.decomposition_rules.items():
            if pattern.replace('_', ' ') in description_lower:
                for step_name, step_type in steps[:16]:
                    subtask = Task(
                        id=f"{task.id}_{len(subtasks)}",
                        description=f"{step_name} for: {task.description}",
                        intent=step_type,
                        priority=task.priority,
                        metadata={'parent_task': task.id}
                    )
                    if subtasks:
                        subtask.dependencies.append(subtasks[-1].id)
                    subtasks.append(subtask)
                return subtasks

        # Default decomposition for complex tasks
        if 'and' in description_lower:
            parts = description_lower.split(' and ')[:16]
            for i, part in enumerate(parts):
                subtask = Task(
                    id=f"{task.id}_{i}",
                    description=part.strip(),
                    intent=task.intent,
                    priority=task.priority,
                    metadata={'parent_task': task.id}
                )
                if i > 0:
                    subtask.dependencies.append(f"{task.id}_{i-1}")
                subtasks.append(subtask)

        return subtasks

    def _identify_tools(self, task: Task) -> List[str]:
        """Plan only explicit command-like tool intents; questions remain questions."""
        desc = str(task.description or "").strip().lower()
        if not desc or not _sm_user_command(desc):
            if re.fullmatch(r"[\d\s+*/().%-]{1,180}", desc):
                return ["calc"]
            return []
        rule_map = {
            "web_search": ("search", "research", "look up"),
            "calc": ("calculate", "compute"),
            "memory_search": ("recall", "remember"),
            "open_url": ("open url", "browse", "navigate"),
            "agent": ("move mouse", "click", "press", "type", "scroll"),
        }
        return [tool for tool,phrases in rule_map.items() if any(_sm_phrase(desc, p) for p in phrases)]

    def _estimate_duration(self, task: Task) -> float:
        """Estimate task execution duration"""
        if task.subtasks:
            # For tasks with subtasks, sum subtask durations
            return sum(self._estimate_duration(st) for st in task.subtasks)

        # Use template if available
        if task.intent in self.task_templates:
            return self.task_templates[task.intent]['estimated_duration']

        # Default estimation based on tool count
        base_duration = 1.0
        tool_overhead = len(task.tools_required) * 0.5
        return base_duration + tool_overhead

    def plan_execution(self, task: Task) -> List[Tuple[str, Any]]:
        """Create execution plan for task"""
        execution_plan = []

        if task.subtasks:
            # Handle subtasks
            for subtask in task.subtasks:
                subtask_plan = self.plan_execution(subtask)
                execution_plan.extend(subtask_plan)
        else:
            # Create execution steps from tools
            for tool in task.tools_required:
                execution_plan.append((tool, task.description))

        return execution_plan

    def optimize_parallel_execution(self, tasks: List[Task]) -> List[List[Task]]:
        """Topological scheduling: dependencies never share a parallel group."""
        pending = {t.id:t for t in list(tasks)[:128]}
        groups = []
        done = set()
        while pending:
            ready = [t for t in pending.values() if all(d in done for d in t.dependencies)]
            if not ready:
                # Unknown dependency or cycle: fail closed without scheduling.
                break
            ready.sort(key=lambda x: x.priority.value)
            if not self.agent.parallel_execution_enabled:
                ready = ready[:1]
            ready = ready[:max(1, min(self.agent.max_parallel_tools, 8))]
            groups.append(ready)
            for t in ready:
                done.add(t.id)
                pending.pop(t.id, None)
        return groups

# ================================================================================
# META-COGNITIVE REASONING ENGINE
# ================================================================================

class MetaCognitiveReasoner:
    """Implements meta-cognitive reasoning and self-reflection"""

    def __init__(self, agent_state: AdvancedAgentState):
        self.agent = agent_state
        self.reasoning_strategies = [
            'forward_chaining',
            'backward_chaining',
            'analogy',
            'case_based',
            'abductive'
        ]

    def reason_about_decision(self, context: str, options: List[Tuple[str, float]]) -> DecisionPoint:
        """Make a decision with explicit reasoning"""
        reasoning_chain = []

        # Step 1: Analyze context
        context_analysis = self._analyze_context(context)
        reasoning_chain.append(f"Context analysis: {context_analysis}")

        # Step 2: Evaluate options
        evaluated_options = []
        for option, initial_score in options:
            adjusted_score = self._evaluate_option(option, context, initial_score)
            evaluated_options.append((option, adjusted_score))
            reasoning_chain.append(f"Option '{option}' scored {adjusted_score:.3f}")

        # Step 3: Select best option
        evaluated_options.sort(key=lambda x: x[1], reverse=True)
        best_option, best_score = evaluated_options[0]
        reasoning_chain.append(f"Selected '{best_option}' with confidence {best_score:.3f}")

        # Step 4: Consider alternatives
        alternatives = evaluated_options[1:4]
        reasoning_chain.append(f"Considered {len(alternatives)} alternatives")

        # Create decision point
        decision = DecisionPoint(
            timestamp=time.time(),
            context=context,
            decision=best_option,
            reasoning=reasoning_chain,
            confidence=best_score,
            alternatives_considered=alternatives
        )

        # Store in decision history
        self.agent.decision_history.append(decision)

        return decision

    def _analyze_context(self, context: str) -> str:
        """Analyze the context of a decision"""
        context_lower = context.lower()

        if 'urgent' in context_lower or 'quickly' in context_lower:
            return "Time-sensitive task requiring fast execution"
        elif 'carefully' in context_lower or 'important' in context_lower:
            return "High-stakes task requiring careful consideration"
        elif 'creative' in context_lower or 'innovative' in context_lower:
            return "Creative task requiring novel approaches"
        else:
            return "Standard task with normal execution requirements"

    def _evaluate_option(self, option: str, context: str, initial_score: float) -> float:
        """Evaluate an option considering context and history"""
        score = initial_score

        # Adjust based on past performance
        if option in self.agent.metrics.task_success_rate:
            success_rate = self.agent.metrics.get_success_rate(option)
            score *= (0.7 + 0.3 * success_rate)

        # Adjust based on context
        context_lower = context.lower()
        if 'safe' in context_lower and 'risky' in option.lower():
            score *= 0.5
        elif 'fast' in context_lower and 'quick' in option.lower():
            score *= 1.3

        return max(0.0, min(1.0, score))

    def reflect_on_outcome(self, decision: DecisionPoint, actual_outcome: str, success: bool):
        """Reflect on a decision outcome and learn"""
        decision.outcome = actual_outcome
        decision.was_correct = success

        if success:
            lesson = f"Strategy '{decision.decision}' works well for context: {decision.context[:50]}"
            self.agent.success_patterns[decision.decision] += 1
        else:
            # Analyze what went wrong
            alternatives = decision.alternatives_considered
            if alternatives:
                better_option = alternatives[0][0]
                lesson = f"Should have chosen '{better_option}' instead of '{decision.decision}' for: {decision.context[:50]}"
            else:
                lesson = f"Strategy '{decision.decision}' failed for context: {decision.context[:50]}"
            self.agent.error_patterns[decision.decision] += 1

        decision.learned_lesson = lesson

        # Add to learning buffer
        self.agent.learning_buffer.append({
            'type': 'task_success' if success else 'task_failure',
            'pattern': decision.decision,
            'context': decision.context,
            'lesson': lesson,
            'timestamp': time.time()
        })

        logger.info(f"[MetaCognitive] Learned: {lesson}")

# ================================================================================
# KNOWLEDGE GRAPH ENGINE
# ================================================================================

class KnowledgeGraphEngine:
    """Manages contextual knowledge as a graph structure"""

    def __init__(self, agent_state: AdvancedAgentState):
        self.agent = agent_state

    def add_knowledge(self, content: str, node_type: str = 'fact', source: str = 'user_interaction') -> str:
        """Add bounded RAM-only knowledge; this is not authority or verified fact."""
        body = str(content or "")[:2048]
        if not body:
            return ""
        embedding = lite_embed([body])[0]
        node_id = uuid.uuid4().hex[:12]
        with self.agent.lock:
            related = self._find_related_nodes(embedding)
            node = KnowledgeNode(id=node_id, content=body, embedding=embedding,
                node_type=str(node_type)[:40], timestamp=time.time(),
                source=str(source)[:96], reliability=0.5)
            for related_id, similarity in related[:5]:
                node.connections[related_id] = float(similarity)
                old = self.agent.knowledge_graph.get(related_id)
                if old:
                    old.connections[node_id] = float(similarity)
            self.agent.knowledge_graph[node_id] = node
            while len(self.agent.knowledge_graph) > 512:
                oldest = next(iter(self.agent.knowledge_graph))
                self.agent.knowledge_graph.pop(oldest, None)
                for n in self.agent.knowledge_graph.values():
                    n.connections.pop(oldest, None)
        return node_id

    def _find_related_nodes(self, embedding: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Find nodes related to the given embedding"""
        similarities = []

        for node_id, node in self.agent.knowledge_graph.items():
            try:
                similarity = self._cosine_similarity(embedding, node.embedding)
                similarities.append((node_id, similarity))
            except Exception:
                continue

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        try:
            if len(vec1) != len(vec2) or not vec1:
                return 0.0
            a = np.asarray(vec1, dtype=float)
            b = np.asarray(vec2, dtype=float)
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                return 0.0
            denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
            return float(np.dot(a,b) / denominator) if denominator > 0 else 0.0
        except Exception:
            return 0.0

    def query_knowledge(self, query: str, top_k: int = 5) -> List[KnowledgeNode]:
        """Query the knowledge graph"""
        query_embedding = lite_embed([query])[0]
        related = self._find_related_nodes(query_embedding, top_k)

        results = []
        for node_id, similarity in related:
            if node_id in self.agent.knowledge_graph:
                node = self.agent.knowledge_graph[node_id]
                node.access_count += 1
                node.last_accessed = time.time()
                results.append(node)

        return results

    def get_context_subgraph(self, node_ids: List[str], depth: int = 2) -> Dict[str, KnowledgeNode]:
        """Get a subgraph centered on specific nodes"""
        subgraph = {}
        to_explore = set(node_ids)
        explored = set()

        for _ in range(depth):
            current_level = to_explore - explored
            if not current_level:
                break

            for node_id in current_level:
                if node_id in self.agent.knowledge_graph:
                    node = self.agent.knowledge_graph[node_id]
                    subgraph[node_id] = node
                    to_explore.update(node.connections.keys())
                explored.add(node_id)

        return subgraph

# ================================================================================
# TOOL ORCHESTRATION ENGINE
# ================================================================================

class ToolOrchestrator:
    """Orchestrates tool execution with dependency resolution and parallelization"""

    def __init__(self, agent_state: AdvancedAgentState):
        self.agent = agent_state
        self._circuit_lock = threading.RLock()
        self._tool_circuits: Dict[str, Dict[str, float]] = {}
        self._initialize_tools()

    def _initialize_tools(self):
        """Initialize tool registry with governance-aware metadata"""
        registry = {}
        base = {
            'memory_search': {
                'function': self._tool_memory_search,
                'can_parallel': True,
                'requires_network': False,
                'estimated_time': 0.5
            },
            'web_search': {
                'function': self._tool_web_search,
                'can_parallel': True,
                'requires_network': True,
                'estimated_time': 2.0
            },
            'calc': {
                'function': self._tool_calc,
                'can_parallel': True,
                'requires_network': False,
                'estimated_time': 0.1
            },
            'open_url': {
                'function': self._tool_open_url,
                'can_parallel': False,
                'requires_network': True,
                'estimated_time': 1.0
            },
            'agent': {
                'function': self._tool_agent_control,
                'can_parallel': False,
                'requires_network': False,
                'estimated_time': 0.5
            }
        }
        for tool_name, info in base.items():
            if _sm_tool_allowed(tool_name):
                registry[tool_name] = info
        if _sm_creative_studio_registry():
            registry['creative'] = {
                'function': self._tool_creative,
                'can_parallel': False,
                'requires_network': False,
                'estimated_time': 3.0
            }
        self.agent.tool_registry = registry

    def execute_tools(self, tool_calls: List[Tuple[str, Any]]) -> Dict[str, Any]:
        """Execute multiple tools with optimization"""
        results = {}

        # Group by parallelizability
        parallel_calls = []
        sequential_calls = []

        for tool_name, arg in list(tool_calls)[:32]:
            if tool_name in self.agent.tool_registry:
                tool_info = self.agent.tool_registry[tool_name]
                if tool_info['can_parallel'] and self.agent.parallel_execution_enabled:
                    parallel_calls.append((tool_name, arg))
                else:
                    sequential_calls.append((tool_name, arg))

        # Execute parallel tools
        if parallel_calls:
            parallel_results = self._execute_parallel(parallel_calls)
            results.update(parallel_results)

        # Sequential tools still run behind a hard deadline so UI callers cannot stall.
        for tool_name, arg in sequential_calls[:16]:
            result, duration, success = self._execute_bounded_tool(tool_name, arg)
            results[tool_name] = result
            self.agent.metrics.record_tool_execution(tool_name, duration, success)

        return results

    def _tool_timeout(self, tool_name: str) -> float:
        global_limit = float(getattr(config, "AI_AGENT_TOOL_TIMEOUT_SECONDS", 8.0) or 8.0)
        estimated = float((self.agent.tool_registry.get(tool_name) or {}).get("estimated_time", 1.0) or 1.0)
        return max(0.25, min(global_limit, max(estimated * 4.0, 1.0)))

    def _circuit_available(self, tool_name: str) -> bool:
        with self._circuit_lock:
            record = self._tool_circuits.get(tool_name) or {}
            return time.monotonic() >= float(record.get("open_until", 0.0) or 0.0)

    def _record_tool_result(self, tool_name: str, success: bool) -> None:
        threshold = max(1, int(getattr(config, "NEURON_CIRCUIT_FAILURE_THRESHOLD", 2) or 2))
        cooldown = max(1.0, float(getattr(config, "NEURON_CIRCUIT_COOLDOWN_SECONDS", 30.0) or 30.0))
        with self._circuit_lock:
            record = self._tool_circuits.setdefault(tool_name, {"failures": 0.0, "open_until": 0.0})
            if success:
                record["failures"] = 0.0
                record["open_until"] = 0.0
                return
            record["failures"] = float(record.get("failures", 0.0) or 0.0) + 1.0
            if record["failures"] >= threshold:
                record["open_until"] = time.monotonic() + cooldown

    def _execute_bounded_tool(self, tool_name: str, arg: Any, timeout: Optional[float] = None) -> Tuple[Any, float, bool]:
        if not self._circuit_available(tool_name):
            return "Circuit open: " + tool_name, 0.0, False
        if not _TOOL_WORKER_SLOTS.acquire(blocking=False):
            return "Deferred: tool capacity exhausted", 0.0, False
        deadline = max(0.1, min(float(timeout if timeout is not None else self._tool_timeout(tool_name)), 60.0))
        result_queue = queue.Queue(maxsize=1)
        started = time.monotonic()
        def runner():
            try:
                result_queue.put_nowait((True, self._execute_single_tool(tool_name, arg)))
            except Exception as exc:
                result_queue.put_nowait((False, "Error: " + str(exc)[:180]))
            finally:
                _TOOL_WORKER_SLOTS.release()
        thread = threading.Thread(target=runner, name="SM-Tool-"+str(tool_name), daemon=True)
        try:
            thread.start()
        except Exception:
            _TOOL_WORKER_SLOTS.release()
            return "Error: tool thread unavailable", 0.0, False
        thread.join(timeout=deadline)
        duration = time.monotonic() - started
        if thread.is_alive():
            self._record_tool_result(tool_name, False)
            return "INDETERMINATE: timeout; worker may still be running (not cancelled)", duration, False
        try:
            success, result = result_queue.get_nowait()
        except queue.Empty:
            success, result = False, "Error: tool returned no result"
        self._record_tool_result(tool_name, bool(success))
        return result, duration, bool(success)

    def _execute_parallel(self, tool_calls: List[Tuple[str, Any]]) -> Dict[str, Any]:
        """Run a finite batch; timed-out workers remain accounted for until exit."""
        calls = list(tool_calls)[:16]
        max_workers = max(1, min(int(self.agent.max_parallel_tools or 5), 8))
        deadline = max(0.25, min(float(getattr(config, "AI_AGENT_PARALLEL_BATCH_TIMEOUT_SECONDS", 12.0) or 12.0), 60.0))
        results = {}
        for offset in range(0, len(calls), max_workers):
            chunk = calls[offset:offset+max_workers]
            batch_result = queue.Queue()
            threads = []
            end = time.monotonic() + deadline
            for i,(name,arg) in enumerate(chunk, start=offset):
                key = str(name) + ("#" + str(i+1) if sum(1 for n,_ in calls[:i] if n==name) else "")
                if not _TOOL_WORKER_SLOTS.acquire(blocking=False):
                    results[key] = "Deferred: tool capacity exhausted"
                    continue
                def run(n=name,a=arg,k=key):
                    start=time.monotonic()
                    try:
                        answer=self._execute_single_tool(n,a)
                        batch_result.put((k,n,answer,time.monotonic()-start,True))
                    except Exception as exc:
                        batch_result.put((k,n,"Error: " + str(exc)[:180],time.monotonic()-start,False))
                    finally:
                        _TOOL_WORKER_SLOTS.release()
                thread=threading.Thread(target=run,daemon=True,name="SM-Parallel-"+str(name))
                thread.start()
                threads.append((key,name,thread))
            remaining=len(threads)
            while remaining and time.monotonic()<end:
                try:
                    key,name,answer,duration,success=batch_result.get(timeout=max(0.01,end-time.monotonic()))
                except queue.Empty:
                    break
                remaining-=1
                results[key]=answer
                self._record_tool_result(name,success)
                self.agent.metrics.record_tool_execution(name,duration,success)
            for key,name,thread in threads:
                if key not in results:
                    results[key]="INDETERMINATE: batch timeout; worker may still be running"
                    self._record_tool_result(name,False)
        return results

    def _execute_single_tool(self, tool_name: str, arg: Any) -> Any:
        if tool_name not in self.agent.tool_registry:
            raise ValueError("Unknown tool: " + str(tool_name))
        if not _sm_tool_allowed(tool_name) and tool_name != "creative":
            raise PermissionError(_sm_tool_error(tool_name))
        if tool_name in {"open_url", "agent"}:
            permitted, reason = _sm_action_permission("url" if tool_name == "open_url" else "desktop")
            if not permitted:
                raise PermissionError(reason)
        if tool_name == "web_search" and not _sm_network_allowed():
            raise PermissionError("network_forbidden_by_runtime_mode")
        return self.agent.tool_registry[tool_name]["function"](arg)

    # Tool implementations
    def _tool_memory_search(self, query: str) -> str:
        """Search memory/context"""
        if not _sm_module_approved("SarahMemoryCompare"):
            return _sm_tool_error("memory_search")
        try:
            from SarahMemoryCompare import compare_local_memory
            result = compare_local_memory(query)
            return result if result else "No relevant memories found"
        except Exception as e:
            return f"Memory search unavailable: {e}"

    def _tool_web_search(self, query: str) -> str:
        if not _sm_module_approved("SarahMemoryResearch"):
            return _sm_tool_error("web_search")
        if not _sm_network_allowed():
            return "Web search disabled in current mode"
        try:
            from SarahMemoryResearch import perform_web_search
            return perform_web_search(str(query)[:2000])
        except Exception:
            return "Web search unavailable"

    def _tool_calc(self, expression: str) -> str:
        """Perform calculation"""
        if not _sm_module_approved("SarahMemoryWebSYM"):
            return _sm_tool_error("calc")
        try:
            from SarahMemoryWebSYM import WebSemanticSynthesizer
            calc = WebSemanticSynthesizer()
            return str(calc.sarah_calculator(expression))
        except Exception as e:
            return f"Calculation failed: {e}"

    def _tool_open_url(self, url: str) -> str:
        return open_url_in_chrome(url)

    def _tool_agent_control(self, command: str) -> str:
        return handle_ai_agent_command(command)

    def _tool_creative(self, payload: Any) -> str:
        """Governed CreativeStudios bridge."""
        studios = _sm_creative_studio_registry()
        if not studios:
            return "CreativeStudios are not currently registered/approved in the SarahMemory core registry."
        try:
            if CanvasStudio is not None and _sm_module_approved("SarahMemoryCanvasStudio"):
                return f"CreativeStudios available: {', '.join(sorted(studios.keys()))}"
            return f"CreativeStudios registered: {', '.join(sorted(studios.keys()))}"
        except Exception as e:
            return f"Creative tool bridge failed: {e}"

# ================================================================================
# UNIFIED ADVANCED AGENT INTERFACE
# ================================================================================

def initialize_advanced_agent():
    """Initialize the advanced agent system idempotently."""
    try:
        ADVANCED_AGENT.initialize()
        logger.info("[AdvancedAgent] v8.0 systems initialized successfully")
        return True
    except Exception as e:
        logger.error(f"[AdvancedAgent] Initialization failed: {e}")
        return False


def shutdown_advanced_agent(timeout: float = 3.0) -> Dict[str, Any]:
    """Bounded lifecycle shutdown for workers, watcher, state, and WAL."""
    global _watch_thread_started
    _watch_stop_event.set()
    watcher_alive = False
    if _watch_thread is not None and _watch_thread.is_alive() and _watch_thread is not threading.current_thread():
        _watch_thread.join(timeout=max(0.1, min(float(timeout), 1.0)))
        watcher_alive = _watch_thread.is_alive()
    agent_started = bool(getattr(ADVANCED_AGENT, "_initialized", False)) or any(t.is_alive() for t in getattr(ADVANCED_AGENT, "executor_threads", []))
    agent_result = ADVANCED_AGENT.shutdown(timeout=max(0.1, float(timeout))) if agent_started else {"ok": True, "standby": True}
    checkpoint = checkpoint_context_history("PASSIVE") if bool(globals().get("_CONTEXT_INITIALIZED", False)) and _sm_flag("AI_FUNCTIONS_CONTEXT_PERSIST_ENABLED", False) and os.path.isfile(CONTEXT_DB_PATH) else {"ok": True, "standby": True}
    _watch_thread_started = bool(watcher_alive)
    return {"ok": bool(agent_result.get("ok")) and not watcher_alive, "agent": agent_result, "watcher_alive": watcher_alive, "checkpoint": checkpoint}

def advanced_agent_query(user_text: str, context: Optional[Dict] = None) -> str:
    """
    Main entry point for advanced agent query processing

    This orchestrates all v8.0 capabilities:
    - Hierarchical planning
    - Meta-cognitive reasoning
    - Knowledge graph querying
    - Parallel tool execution
    - Predictive intent modeling
    - Autonomous learning
    """
    start_time = time.time()
    try:
        init_context_history()
    except Exception:
        pass
    if not bool(getattr(ADVANCED_AGENT, "_initialized", False)):
        if _sm_flag("AI_FUNCTIONS_BACKGROUND_ENABLED", False) and _sm_flag("AI_AGENT_ENABLED", False):
            initialize_advanced_agent()
    ADVANCED_AGENT.interaction_count += 1
    kg_engine = None  # created lazily; avoids NameError if Step 3 fails

    try:
        # Step 1: Check for predicted intents
        predicted = None
        try:
            predicted = ADVANCED_AGENT.predicted_intents.get_nowait()
            if predicted.get("confidence", 0.0) > 0.6:
                logger.info(
                    f"[AdvancedAgent] Predicted intent: {predicted.get('intent')} "
                    f"(confidence: {predicted.get('confidence', 0.0):.2f})"
                )
        except queue.Empty:
            pass
        # Step 2: Classify intent
        intent = classify_intent(user_text) if (user_text and user_text.strip()) else "unknown"

        # Fast-path: identity/name questions should never be routed through the knowledge graph
        # (prevents repetitive "Based on what I know: Q/A ..." feedback loops)
        try:
            _q = (user_text or "").lower().strip()
            if _q and any(k in _q for k in (
                "your name",
                "who are you",
                "what are you",
                "identify yourself",
                "tell me your name",
                "what is ur name",
            )):
                result_text = "I'm Sarah — your SarahMemory AI companion."
                duration = time.time() - start_time
                try:
                    ADVANCED_AGENT.metrics.response_times.append(duration)
                    ADVANCED_AGENT.metrics.record_task_outcome(intent, True)
                except Exception:
                    pass
                return result_text
        except Exception:
            pass


        # Fast-path: creator/origin questions must NEVER be answered by external APIs or KG
        # (prevents generic LLM provider self-attribution like "I was created by OpenAI")
        try:
            _q2 = (user_text or "").lower().strip()
            if _q2 and any(k in _q2 for k in (
                "who created you",
                "who made you",
                "who built you",
                "who designed you",
                "who is your creator",
                "who developed you",
                "who programmed you",
                "who owns you",
            )):
                result_text = (
                    "I was created by Brian Lee Baros as part of the SarahMemory AI Companion Platform. "
                    "I can use different AI engines and tools, but my identity and behavior are defined by SarahMemory."
                )
                duration = time.time() - start_time
                try:
                    ADVANCED_AGENT.metrics.response_times.append(duration)
                    ADVANCED_AGENT.metrics.record_task_outcome(intent, True)
                except Exception:
                    pass
                return result_text
        except Exception:
            pass

        # Step 3: Query knowledge graph for context (only if we have a real query)
        relevant_knowledge = []
        context_summary = ""
        if user_text and user_text.strip():
            try:
                kg_engine = KnowledgeGraphEngine(ADVANCED_AGENT)
                relevant_knowledge = kg_engine.query_knowledge(user_text, top_k=5) or []
            except Exception as e:
                logger.debug("[AdvancedAgent] KG query failed: %s", e)
                relevant_knowledge = []

            if relevant_knowledge:
                safe_chunks = []
                for k in relevant_knowledge[:3]:
                    c = (getattr(k, "content", "") or "").strip()
                    if not c:
                        continue
                    safe_chunks.append(c[:50])
                context_summary = " | ".join(safe_chunks)
                logger.info(
                    f"[AdvancedAgent] Found {len(relevant_knowledge)} relevant knowledge nodes"
                )

        # Step 4: Create hierarchical plan
        planner = HierarchicalPlanner(ADVANCED_AGENT)
        task = planner.create_task(
            description=(user_text or ""),
            intent=intent,
            priority=TaskPriority.NORMAL
        )

        # Step 5: Meta-cognitive reasoning about approach
        reasoner = MetaCognitiveReasoner(ADVANCED_AGENT)

        # Generate approach options
        options = []
        if getattr(task, "subtasks", None):
            options.append(("hierarchical_decomposition", 0.8))
        if getattr(task, "tools_required", None):
            options.append(("direct_tool_execution", 0.7))
        if relevant_knowledge:
            options.append(("knowledge_based_response", 0.75))
        if not options:
            options.append(("conversational_response", 0.6))

        decision = reasoner.reason_about_decision(
            context=f"Query: {user_text} | Intent: {intent}" + (f" | KG: {context_summary}" if context_summary else ""),
            options=options
        )

        logger.info(
            f"[AdvancedAgent] Decision: {decision.decision} (confidence: {decision.confidence:.2f})"
        )
        # Step 6: Execute based on decision
        orchestrator = ToolOrchestrator(ADVANCED_AGENT)

        if decision.decision in {"hierarchical_decomposition", "direct_tool_execution"} and not _sm_flag("AI_FUNCTIONS_AUTO_TOOL_EXECUTION_ENABLED", False):
            result_text = "I'm unsure how to respond."
            success = False
        elif decision.decision == "hierarchical_decomposition":
            # Execute with full hierarchical planning
            execution_plan = planner.plan_execution(task)
            results = orchestrator.execute_tools(execution_plan)

            # Combine results
            result_text = _combine_tool_results(results, user_text)
            success = bool(results) and result_text not in {"I encountered errors with all available tools.", "I'm unsure how to respond."}

        elif decision.decision == "direct_tool_execution":
            # Execute tools directly
            tool_calls = [(tool, user_text) for tool in task.tools_required]
            results = orchestrator.execute_tools(tool_calls)
            result_text = _combine_tool_results(results, user_text)
            success = bool(results) and result_text not in {"I encountered errors with all available tools.", "I'm unsure how to respond."}

        elif decision.decision == "knowledge_based_response":
            # Use knowledge graph
            result_text = _generate_knowledge_response(relevant_knowledge, user_text)
            success = True

        else:
            # Conversational fallback
            result_text = _generate_conversational_response(user_text, context)
            success = True

        # Step 7: Learn from interaction
        if _sm_flag("AI_FUNCTIONS_LEARNING_ENABLED", False):
            ADVANCED_AGENT.learning_buffer.append({
                'type': 'task_success' if success else 'task_failure',
                'pattern': intent,
                'context_hash': hashlib.sha256(str(user_text).encode()).hexdigest(),
                'decision': decision.decision,
                'timestamp': time.time()
            })
            del ADVANCED_AGENT.learning_buffer[:-128]

        # Step 8: Add to knowledge graph
        # Step 8: Add to knowledge graph (only when we have a real answer)
        try:
            _rt_store = (result_text or '').strip()
            if _rt_store and success and _sm_flag("AI_FUNCTIONS_LEARNING_ENABLED", False):
                if kg_engine is None:
                    kg_engine = KnowledgeGraphEngine(ADVANCED_AGENT)
                kg_engine.add_knowledge(
                    content=f"Q: {(user_text or '')[:100]} | A: {_rt_store[:100]}",
                    node_type='experience',
                    source='interaction'
                )
        except Exception as e:
            logger.debug(f"[KnowledgeGraph] Store skipped: {e}")


        # Step 9: Reflect on outcome
        if _sm_flag("AI_FUNCTIONS_LEARNING_ENABLED", False):
            reasoner.reflect_on_outcome(decision, result_text, success)

        # Step 10: Record metrics
        duration = time.time() - start_time
        ADVANCED_AGENT.metrics.response_times.append(duration)
        del ADVANCED_AGENT.metrics.response_times[:-100]
        ADVANCED_AGENT.metrics.record_task_outcome(intent, success)

        # Step 11: Update user patterns for prediction
        current_time = datetime.now()
        pattern_key = f"{current_time.hour}_{current_time.weekday()}"
        if _sm_flag("AI_FUNCTIONS_LEARNING_ENABLED", False):
            patterns = ADVANCED_AGENT.user_patterns[pattern_key]
            patterns.append({
                'intent': intent, 'query_hash': hashlib.sha256(str(user_text).encode()).hexdigest(),
                'hour': current_time.hour, 'day': current_time.weekday(), 'timestamp': time.time()
            })
            del patterns[:-32]

        logger.info(f"[AdvancedAgent] Completed in {duration:.2f}s | Success: {success}")

        return result_text

    except Exception as e:
        logger.error(f"[AdvancedAgent] Query processing failed: {e}")
        return f"I encountered an error processing your request: {e}"

def _combine_tool_results(results: Dict[str, Any], original_query: str) -> str:
    """Combine results from multiple tools into coherent response"""
    if not results:
        return "I couldn't find a suitable response."

    # Filter out errors
    rejected = ("error:", "timeout:", "indeterminate:", "deferred:", "circuit open:",
                "action requires", "agent control denied:", "url opening denied:",
                "web search disabled", "memory search unavailable")
    successful_results = {k: v for k, v in results.items()
                         if v is not None and not str(v).strip().lower().startswith(rejected)}

    if not successful_results:
        return "I encountered errors with all available tools."

    # If single result, return it
    if len(successful_results) == 1:
        return list(successful_results.values())[0]

    # Combine multiple results intelligently
    combined = []
    for tool_name, result in successful_results.items():
        if result and len(str(result)) > 10:
            combined.append(f"[{tool_name}]: {result}")

    return "\n\n".join(combined) if combined else "Results obtained but require further processing."

def _generate_knowledge_response(knowledge_nodes: List[KnowledgeNode], query: str) -> str:
    """Generate response based on knowledge graph"""
    if not knowledge_nodes:
        return "I don't have enough knowledge to answer that."

    q = (query or "").lower().strip()

    # Identity/name questions must NEVER be answered from the knowledge graph
    # (prevents self-reinforcing Q/A echo loops)
    if any(k in q for k in (
        "your name",
        "who are you",
        "what are you",
        "identify yourself",
        "what is ur name",
        "tell me your name",
    )):
        return "I'm Sarah — your SarahMemory AI companion."

    # Creator/origin questions should not be answered from the knowledge graph
    if any(k in q for k in (
        "who created you",
        "who made you",
        "who built you",
        "who designed you",
        "who is your creator",
        "who developed you",
        "who programmed you",
        "who owns you",
    )):
        return (
            "I was created by Brian Lee Baros as part of the SarahMemory AI Companion Platform. "
            "I can use different AI engines and tools, but my identity and behavior are defined by SarahMemory."
        )


    # Keep incoming order (already ranked by embedding similarity)
    response_parts = []
    for node in knowledge_nodes[:5]:
        content = (getattr(node, "content", "") or "").strip()
        if len(content) < 20:
            continue

        # Filter out low-quality echo / scaffold templates
        low_quality_markers = (
            "i understand you said:",
            "could you provide more details",
            "based on what i know: q:",
            "q:",
            "a:",
        )
        cl = content.lower()
        if any(m in cl for m in low_quality_markers):
            continue

        response_parts.append(content)
        if len(response_parts) >= 3:
            break

    if response_parts:
        return "Based on what I know: " + " | ".join(response_parts)

    # Fallback when KG exists but nothing passes quality filters
    return ""


def _generate_conversational_response(text: str, context: Optional[Dict]) -> str:
    """Generate a conversational response"""
    text_lower = (text or "").lower().strip()

    # Identity / name
    if any(k in text_lower for k in (
        "your name",
        "who are you",
        "what are you",
        "identify yourself",
        "tell me your name",
        "what is ur name",
    )):
        return "I'm Sarah — your SarahMemory AI companion."


    # Creator/origin
    if any(k in text_lower for k in (
        "who created you",
        "who made you",
        "who built you",
        "who designed you",
        "who is your creator",
        "who developed you",
        "who programmed you",
        "who owns you",
    )):
        return (
            "I was created by Brian Lee Baros as part of the SarahMemory AI Companion Platform. "
            "I can use different AI engines and tools, but my identity and behavior are defined by SarahMemory."
        )

    # Greeting
    if any(_sm_phrase(text_lower, g) for g in ("hello", "hi", "hey")):
        return "Hello! How can I assist you today?"

    # Thanks
    if any(t in text_lower for t in ("thank", "thanks")):
        return "You're welcome! Let me know if you need anything else."

    # Capability query
    if "can you" in text_lower or "what can" in text_lower:
        return (
            "I'm an advanced AI companion that can help with research, reasoning, planning, "
            "problem-solving, and conversation. What would you like to work on?"
        )

    # Default (IMPORTANT: do NOT echo user text verbatim)
    return "I’m here and listening. Could you clarify what you’d like help with?"

# ================================================================================
# LEGACY COMPATIBILITY FUNCTIONS
# ================================================================================


# ================================================================================
# LEGACY FUNCTIONS (CRITICAL FOR BACKWARD COMPATIBILITY)
# ================================================================================

def _now_ms():
    return int(time.time()*1000)


def _start_input_watchers():
    global _watch_thread_started, _watch_thread, _last_mouse
    if _watch_thread_started and _watch_thread is not None and _watch_thread.is_alive():
        return
    _watch_stop_event.clear()
    _watch_thread_started = True
    def _loop():
        global _last_mouse
        try:
            import pyautogui as _pag
        except Exception:
            _pag = None
        while not _watch_stop_event.is_set():
            try:
                moved = False
                if _pag is not None:
                    pos = _pag.position()
                    if _last_mouse is None:
                        _last_mouse = pos
                    if pos != _last_mouse:
                        moved = True
                        _last_mouse = pos
                if moved:
                    _agent_state["last_human_ts"] = _now_ms()
                    if _agent_state["mode"] != "IDLE":
                        # If user is interacting, halt the agent
                        _agent_state["mode"] = "HALTED"
                        _agent_state["resume_eta_ms"] = _now_ms() + AI_AGENT_RESUME_DELAY
                if _watch_stop_event.wait(_sm_rhythm_interval("agent_watch", 0.1, {"loop": "agent_watch"})):
                    break
            except Exception:
                if _watch_stop_event.wait(_sm_rhythm_interval("agent_watch", 0.2, {"loop": "agent_watch", "exception": True})):
                    break
    import threading
    _watch_thread = threading.Thread(target=_loop, name="SM-AgentWatch", daemon=True)
    _watch_thread.start()


def agent_status():
    return dict(_agent_state)


def _update_gui_status():
    try:
        if hasattr(config, "status_bar"):
            config.status_bar.set_status(f"Agent: {_agent_state['mode']}")
    except Exception:
        pass


def intercept_agent_control_phrases(text: str):
    """Accept exact agent-control commands, not quoted phrases or substrings."""
    global _agent_state
    t = re.sub(r"[.!?]+$", "", str(text or "").strip().lower())
    if not t:
        return None
    if t in {p.strip().lower() for p in AI_AGENT_STOP_PHRASES}:
        _agent_state["mode"] = "HALTED"
        _agent_state["resume_eta_ms"] = _now_ms() + AI_AGENT_RESUME_DELAY
        _update_gui_status()
        return "Agent halted."
    if t in {p.strip().lower() for p in AI_AGENT_HALT_PHRASES}:
        _agent_state["mode"] = "HALTED"
        _update_gui_status()
        return "Paused."
    if t in {p.strip().lower() for p in AI_AGENT_RESUME_PHRASES}:
        if not _sm_flag("AI_AGENT_ENABLED", False):
            return "Agent control is disabled."
        if _now_ms() < _agent_state.get("resume_eta_ms", 0):
            return "Agent cooling down."
        _agent_state["mode"] = "ACTIVE"
        _update_gui_status()
        return "Resuming."
    return None


def agent_guard():
    """Human interruption guard; never self-activates from IDLE."""
    if not _sm_flag("AI_AGENT_ENABLED", False):
        return False
    _start_input_watchers()
    now = _now_ms()
    if now - _agent_state.get("last_human_ts", 0) < AI_AGENT_USER_ACTIVITY_TIMEOUT_MS:
        _agent_state["mode"] = "HALTED"
        _agent_state["resume_eta_ms"] = now + AI_AGENT_RESUME_DELAY
        _update_gui_status()
        return False
    return _agent_state.get("mode") == "ACTIVE" and now >= _agent_state.get("resume_eta_ms", 0)




# === Media Job Helpers (addons/media) ===
_media_jobs_q = queue.Queue(maxsize=64)
_media_results = {}
_media_lock = threading.RLock()

def _media_init():
    global _media_jobs_q, _media_results, _media_lock
    if _media_jobs_q is None:
        _media_jobs_q = queue.Queue()
    if _media_results is None:
        _media_results = {}
    if _media_lock is None:
        _media_lock = threading.RLock()

def submit_media_job(job: Dict[str, Any]) -> str:
    """Queue a bounded media job as a proposal, without executing it."""
    _media_init()
    if not isinstance(job, dict):
        raise ValueError("job must be a dict")
    job_id = str(job.get("job_id") or uuid.uuid4())[:120]
    record = {**job, "job_id": job_id, "ts_submit": time.time(), "execution_authority": False}
    with _media_lock:
        if job_id in _media_results:
            return job_id
        if len(_media_results) >= 128 or _media_jobs_q.full():
            raise RuntimeError("media_job_capacity_reached")
        _media_results[job_id] = {"status": "queued", "job": record, "result": None}
        try:
            _media_jobs_q.put_nowait(record)
        except queue.Full:
            _media_results.pop(job_id, None)
            raise RuntimeError("media_job_queue_full")
    return job_id

def poll_media_job() -> Optional[Dict[str, Any]]:
    """Non-blocking: returns next queued job dict or None."""
    _media_init()
    try:
        return _media_jobs_q.get_nowait()
    except Exception:
        return None

def store_media_result(job_id: str, result: Dict[str, Any], status: str = "done") -> None:
    """Store one bounded media result, retaining the existing public contract."""
    _media_init()
    if not job_id or not isinstance(result, dict):
        raise ValueError("job_id and dictionary result required")
    with _media_lock:
        rec = _media_results.get(job_id)
        if rec is None:
            if len(_media_results) >= 128:
                raise RuntimeError("media_result_capacity_reached")
            rec = {"status": "unknown", "job": None, "result": None}
        rec.update({"status": str(status)[:48], "result": dict(result), "ts_done": time.time()})
        _media_results[job_id] = rec

def get_media_result(job_id: str) -> Dict[str, Any]:
    """Return a snapshot rather than the mutable internal dictionary."""
    _media_init()
    with _media_lock:
        rec = _media_results.get(job_id)
        return dict(rec) if rec is not None else {"status": "missing", "job": None, "result": None}


# === Governed Plug-in Connectors ===

def local_memory_lookup(text):
    if not _sm_module_approved("SarahMemoryCompare"):
        return None
    try:
        from SarahMemoryCompare import compare_local_memory
        return compare_local_memory(text)
    except:
        return None


def web_research_query(text):
    if not _sm_network_allowed() or not _sm_module_approved("SarahMemoryResearch"):
        return None
    try:
        from SarahMemoryResearch import perform_web_search
        return perform_web_search(str(text or "")[:2000])
    except Exception:
        return None


def symbolic_calc_answer(text):
    """Delegate symbolic/arithmetic questions to SarahMemoryWebSYM safely."""
    if not _sm_module_approved("SarahMemoryWebSYM"):
        return None
    try:
        from SarahMemoryWebSYM import WebSemanticSynthesizer
        calc = WebSemanticSynthesizer()
        return str(calc.sarah_calculator(str(text or ""), str(text or "")))
    except Exception as e:
        logger.debug(f"[WebSYM] calculator failed: {e}")
        return None

def deep_learn_context(text):
    if not _sm_flag("AI_FUNCTIONS_LEARNING_ENABLED", False) or not _sm_module_approved("SarahMemoryDL"):
        return None
    try:
        from SarahMemoryDL import learn_from_input
        learn_from_input(str(text or "")[:2000])
    except Exception:
        pass
    return None

# === Additional Functions ===

def get_context():
    with _CONTEXT_LOCK:
        return [dict(item) if isinstance(item, dict) else item for item in context_buffer]


def clear_context():
    global context_buffer, context_embeddings
    with _CONTEXT_LOCK:
        context_buffer = []
        context_embeddings = []


def add_to_context(interaction):
    """Maintain a bounded RAM ring; persistent storage and sync are opt-in."""
    if not isinstance(interaction,dict):
        return
    item=dict(interaction)
    try:
        embedding=item.get("embedding")
        if not isinstance(embedding,(list,tuple,np.ndarray)) or not len(embedding):
            embedding=lite_embed(str(item.get("input") or item.get("final_response") or ""))[0]
        vector=[float(x) for x in list(embedding)[:4096]]
        if not vector or not all(math.isfinite(x) for x in vector):
            return
    except (TypeError,ValueError,OverflowError):
        return
    item["embedding"]=vector
    capacity=max(1,min(int(getattr(config,"CONTEXT_BUFFER_SIZE",20) or 20),256))
    with _CONTEXT_LOCK:
        context_buffer.append(item)
        context_embeddings.append(np.array(vector,dtype=float))
        del context_buffer[:-capacity]
        del context_embeddings[:-capacity]
    if _sm_flag("AI_FUNCTIONS_CONTEXT_PERSIST_ENABLED", False):
        try:
            with contextlib.closing(_context_db_connect()) as con:
                _ensure_context_schema(con)
                con.execute("INSERT INTO context_history (timestamp,input,embedding,final_response,source,intent) VALUES (?,?,?,?,?,?)",
                    (item.get("timestamp"),str(item.get("input") or "")[:4000],json.dumps(vector),
                     str(item.get("final_response") or "")[:4000],str(item.get("source") or "")[:96],str(item.get("intent") or "")[:96]))
                retention = max(128,min(int(getattr(config,"AI_FUNCTIONS_CONTEXT_MAX_ROWS",2048) or 2048),4096))
                cursor = con.execute("SELECT id FROM context_history ORDER BY id DESC LIMIT 1 OFFSET ?",(retention,))
                row = cursor.fetchone()
                if row:
                    con.execute("DELETE FROM context_history WHERE id IN (SELECT id FROM context_history WHERE id <= ? ORDER BY id ASC LIMIT 32)",(row[0],))
                con.commit()
        except Exception as exc:
            logger.warning("[Context] Persistence failed: %s",exc)
    if (_sm_network_allowed() and _sm_flag("AI_FUNCTIONS_CONTEXT_SYNC_ENABLED",False)
        and _sm_flag("ALLOW_HUB_SYNC",False)):
        hub=_get_hub()
        if hub and callable(getattr(hub,"context_update",None)) and isinstance(item.get("final_response"),str):
            _fire_and_forget(hub.context_update,item["final_response"][:4000],tags=[item.get("intent"),item.get("source")])


def retrieve_similar_context(current_embedding, top_n=3):
    try:
        current=np.asarray(current_embedding,dtype=float)
        if current.ndim!=1 or not np.isfinite(current).all():
            return []
        current_norm=float(np.linalg.norm(current))
        if current_norm<=0:
            return []
        with _CONTEXT_LOCK:
            pairs=list(zip(context_buffer,context_embeddings))
        ranked=[]
        for i,(record,emb) in enumerate(pairs):
            try:
                v=np.asarray(emb,dtype=float)
                if v.shape!=current.shape or not np.isfinite(v).all():
                    continue
                norm=float(np.linalg.norm(v))*current_norm
                if norm>0:
                    ranked.append((float(np.dot(current,v)/norm),i,dict(record)))
            except (TypeError,ValueError):
                continue
        ranked.sort(key=lambda row:(row[0],row[1]),reverse=True)
        return [row[2] for row in ranked[:max(0,min(int(top_n),32))]]
    except (ValueError,TypeError):
        return []


def log_ai_functions_event(event, details):
    """Bounded optional local event log, disabled by default."""
    if not _sm_flag("AI_FUNCTIONS_EVENT_LOG_ENABLED",False):
        return
    try:
        message=re.sub(r"(?i)(password|secret|token|api[_-]?key)\s*[:=]\s*\S+",r"\1=<redacted>",str(details or "")[:1024])
        path=os.path.join(config.DATASETS_DIR,"functions.db")
        os.makedirs(os.path.dirname(path),exist_ok=True)
        with contextlib.closing(sqlite3.connect(path,timeout=3.0)) as con:
            con.execute("CREATE TABLE IF NOT EXISTS ai_functions_events (id INTEGER PRIMARY KEY AUTOINCREMENT,timestamp TEXT,event TEXT,details TEXT)")
            con.execute("INSERT INTO ai_functions_events (timestamp,event,details) VALUES (?,?,?)",
                (datetime.now().isoformat(),str(event or "")[:96],message))
            con.commit()
    except Exception as exc:
        logger.warning("[AiFunctions] Event log unavailable: %s",exc)

if __name__ == '__main__':
    logger.info("Starting Enhanced SarahMemoryAiFunctions module test.")
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=5)
            voice_text = recognizer.recognize_google(audio)
            print("You said:", voice_text)
            if voice_text:
                response = route_intent_response(voice_text)
                print("AI responded:", response)
            else:
                print("No valid speech input detected.")
    except Exception as e:
        logger.error(f"Speech recognition failed: {e}")
        print("Speech recognition error:", e)
    logger.info("Enhanced SarahMemoryAiFunctions module testing complete.")

import os, time, logging
logger = logging.getLogger("AiFunctions")


def handle_ai_agent_command(command_text):
    """
    Execute desktop/UI/game commands with safety gating, human-activity halting, and consent checks.
    """
    import time
    import re as _re
    allowed, reason = _sm_action_permission("desktop")
    if not allowed:
        return "Agent control denied: " + reason
    if not _sm_user_command(str(command_text or "")):
        return "Agent control denied: explicit command required"
    try:
        import pyautogui as _pag
        _pag.FAILSAFE = True
    except Exception as e:
        logger.error(f"[AI-Agent] pyautogui unavailable: {e}")
        return "Desktop control is unavailable on this system."

    if not AI_AGENT_ENABLED:
        return "Agent control is disabled."

    # Normalize
    raw = command_text or ""
    cmd = raw.strip().lower()

    # String-only legacy entry points cannot verify a live user approval receipt.
    # Never transmit audio for cloud speech recognition to obtain permission.
    if any(_sm_phrase(cmd, str(risky)) for risky in AI_AGENT_REQUIRE_CONSENT):
        return "Action requires verified OperatorCore approval; direct agent command denied."

    # Human activity guard
    if not agent_guard():
        return f"Agent paused ({agent_status()['mode']})."

    # App lifecycle via Si
    if any(w in cmd for w in ("open ", "launch ", "start ", "focus ", "maximize ", "minimize ", "close ", "terminate ", "kill ")):
        try:
            from SarahMemorySi import manage_application_request
            ok = manage_application_request(cmd)
            return "Done." if ok else "I couldn't do that."
        except Exception as e:
            logger.warning(f"[AI-Agent] Si routing failed: {e}")

    # Basic UI actions
    if "move mouse to" in cmd:
        try:
            coords = cmd.replace("move mouse to","").strip()
            parts = [p.strip() for p in coords.replace("and",",").split(",") if p.strip()]
            x, y = int(float(parts[0])), int(float(parts[1]))
            _pag.moveTo(x, y, duration=0.15)
            return f"Moved mouse to {x},{y}."
        except Exception as e:
            return f"Invalid coordinates: {e}"

    if "double click" in cmd or "double-click" in cmd:
        _pag.doubleClick(); return "Double clicked."

    if "click" in cmd:
        _pag.click(); return "Clicked."

    # Typing
    m = _re.search(r"\btype\s+(.*)$", cmd)
    if m:
        text_to_type = m.group(1).strip().strip('"')
        _pag.typewrite(text_to_type, interval=0.02)
        return "Typed."

    # Press hotkey
    m = _re.search(r"\bpress\s+([a-z0-9\+\-]+)$", cmd)
    if m:
        key = m.group(1)
        if "+" in key:
            parts = [k.strip() for k in key.split("+")]
            _pag.hotkey(*parts)
        else:
            _pag.press(key)
        return "Pressed."

    if "scroll down" in cmd:
        _pag.scroll(-500); return "Scrolled down."
    if "scroll up" in cmd:
        _pag.scroll(500); return "Scrolled up."

    # Wait
    m = _re.search(r"\bwait\s+(\d+(?:\.\d+)?)", cmd)
    if m:
        time.sleep(min(3.0, max(0.0, float(m.group(1))))); return "Waiting done."

    # --- Game micro-controls (WASD etc.) ---
    def _hold(key, dur=0.2):
        _pag.keyDown(key); time.sleep(dur); _pag.keyUp(key)
    # Optional: parse "for X seconds"
    dur = 0.2
    dm = _re.search(r"\bfor\s+(\d+(?:\.\d+)?)\s*(?:sec|second|seconds|s)\b", cmd)
    if dm:
        dur = min(3.0, max(0.05, float(dm.group(1))))

    if any(w in cmd for w in ("forward","move forward","go forward")):
        _hold("w", dur); return "Moving forward."
    if any(w in cmd for w in ("back","backward","move back")):
        _hold("s", dur); return "Moving back."
    if any(w in cmd for w in ("left","strafe left","move left")):
        _hold("a", dur); return "Left."
    if any(w in cmd for w in ("right","strafe right","move right")):
        _hold("d", dur); return "Right."
    if "jump" in cmd:
        _hold("space", 0.1); return "Jump."
    if "crouch" in cmd or "duck" in cmd:
        _hold("ctrl", 0.1); return "Crouch."
    if "run" in cmd or "sprint" in cmd:
        _hold("shift", dur); return "Run."
    if "shoot" in cmd or "fire" in cmd:
        _pag.mouseDown(); time.sleep(dur); _pag.mouseUp(); return "Shoot."
    if "reload" in cmd:
        _hold("r", 0.1); return "Reload."
    if "use" in cmd or "interact" in cmd:
        _hold("e", 0.1); return "Use."
    if "open map" in cmd or "map" in cmd:
        _hold("tab", 0.1); return "Map."

    # Unknown
    return "I didn't recognize that command."


# [PATCH v7.7.2] Standardized chunking, transformer selection, and rerank utilities

def select_sentence_transformer(preferred: list = None):
    """
    Factory: returns object with .encode(list|str)->np.ndarray.

    Selection priority:
      1) explicit `preferred` list
      2) user-enabled models in SarahMemoryGlobals.MODEL_CONFIG (manual overrides)
      3) auto-selected resolver recommendation (only if Tier Rating != POOR)
      4) deterministic hash embed fallback (always works offline)

    IMPORTANT:
      - Never hardcode specific model repo IDs here.
      - Never download models implicitly; only load if present on disk.
    """
    import os
    import numpy as np

    # Safe globals
    try:
        import SarahMemoryGlobals as G
    except Exception:
        G = None

    try:
        from SarahMemoryGlobals import MODEL_CONFIG
    except Exception:
        MODEL_CONFIG = {}

    def _is_cloud() -> bool:
        try:
            return bool(getattr(G, "RUN_MODE", "local") == "cloud")
        except Exception:
            return False

    def _models_root() -> str:
        try:
            base_dir = getattr(G, "BASE_DIR", os.getcwd())
            models_dir = getattr(G, "MODELS_DIR", os.path.join(base_dir, "data", "models"))
            return os.path.join(models_dir, "SarahMemory")
        except Exception:
            return os.path.join(os.getcwd(), "data", "models", "SarahMemory")

    def _local_candidates(name: str) -> list:
        root = _models_root()
        n = str(name or "").strip()
        if not n or os.path.isabs(n) or ".." in n.replace("\\", "/").split("/"):
            return []
        base = os.path.realpath(root)
        raw = [os.path.join(root, n), os.path.join(root, n.replace("/", "_")),
               os.path.join(root, n.replace("-", "_")), os.path.join(root, n.replace("/", os.sep))]
        return [p for p in raw if os.path.commonpath((base, os.path.realpath(p))) == base]

    def _pick_auto_from_resolver() -> list:
        """
        Ask Globals.resolve_model('embeddings') for an ordered list.
        If Tier Rating == POOR and user didn't enable anything, this returns [].
        """
        out = []
        try:
            resolve_fn = getattr(G, "resolve_model", None)
            if callable(resolve_fn):
                res = resolve_fn("embeddings", text="", meta={"caller": "select_sentence_transformer"}) or {}
                sel = res.get("selected")
                fbs = res.get("fallbacks") or []
                if sel:
                    out.append(str(sel))
                out.extend([str(x) for x in fbs if x])
        except Exception:
            pass
        # de-dupe preserving order
        seen = set()
        ordered = []
        for x in out:
            if x and x not in seen:
                seen.add(x)
                ordered.append(x)
        return ordered

    # Build candidate list (ordered, de-duped)
    tried: list = []

    if isinstance(preferred, (list, tuple)) and preferred:
        for x in preferred:
            x = str(x or "").strip()
            if x and x not in tried:
                tried.append(x)

    # Manual/user-enabled flags always apply, even on POOR
    try:
        for name, en in (MODEL_CONFIG or {}).items():
            if en:
                n = str(name or "").strip()
                if n and n not in tried:
                    tried.append(n)
    except Exception:
        pass

    # Auto-selection only when allowed (resolver encodes POOR policy)
    if not tried:
        tried.extend(_pick_auto_from_resolver())

    # If cloud or no candidates, go straight to hash embed fallback
    if _is_cloud() or not tried:
        from SarahMemoryAdvCU import embed_text as _adv_embed

        class _HashWrap:
            def encode(self, x):
                if isinstance(x, str):
                    x = [x]
                # advcu embed_text supports list input in newer builds; fallback to per-item if needed
                try:
                    vecs = _adv_embed(x)  # type: ignore
                    return np.array(vecs)
                except Exception:
                    return np.vstack([np.array(_adv_embed(t)) for t in x])  # type: ignore

        return _HashWrap()

    # Try SentenceTransformer from local disk only
    for name in tried:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception:
            break

        local_paths = _local_candidates(name)
        local_path = next((p for p in local_paths if os.path.exists(p)), None)

        # If repo string isn't present on disk, skip (prevents downloads)
        if not local_path:
            continue

        try:
            model = SentenceTransformer(local_path)

            class _Wrap:
                def encode(self, x):
                    return model.encode(x)

            return _Wrap()
        except Exception:
            continue

    # Fallback: deterministic hash embed (offline-safe)
    from SarahMemoryAdvCU import embed_text as _adv_embed

    class _HashWrap:
        def encode(self, x):
            if isinstance(x, str):
                x = [x]
            try:
                vecs = _adv_embed(x)  # type: ignore
                return np.array(vecs)
            except Exception:
                return np.vstack([np.array(_adv_embed(t)) for t in x])  # type: ignore

    return _HashWrap()


def _build_provenance(used_calculator=False, used_local=False, used_web=False, api_name=None, model_name=None, used_fallback=False):
    if used_calculator: return {"source":"Calculator"}
    if used_local: return {"source":"Local"}
    if used_web and api_name and model_name: return {"source": f"API/{api_name} {model_name}"}
    if used_web: return {"source":"Web"}
    if used_fallback: return {"source":"Fallback"}
    return {"source":"Unknown"}


def _classify_intent(text):
    t=(text or "").strip().lower()
    if re.match(r'^[\s\d\+\-\*\/\(\)\.\^%]+$', t): return "Math"
    if "your name" in t or t.startswith("who are you") or "what are you" in t: return "Identity"
    if t.startswith("show me") or "image of" in t: return "MediaRequest"
    if t.startswith("open ") or "launch " in t: return "SystemControl"
    return "General"


# --- injected: on-demand ensure table for `response` ---

def _ensure_response_table(db_path=None):
    try:
        import sqlite3, os, logging
        try:
            import SarahMemoryGlobals as config
        except Exception:
            class config: pass
        if db_path is None:
            base = getattr(config, "BASE_DIR", os.getcwd())
            db_path = os.path.join(config.DATASETS_DIR, "system_logs.db")
        con = sqlite3.connect(db_path); cur = con.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS response (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, user TEXT, content TEXT, source TEXT, intent TEXT)'); con.commit(); con.close()
        logging.debug("[DB] ensured table `response` in %s", db_path)
    except Exception as e:
        try:
            import logging; logging.warning("[DB] ensure `response` failed: %s", e)
        except Exception:
            pass
# Database schema is created only by an explicit storage operation.

def add_to_context_entry(interaction_or_text, final_response=None, intent="chat", source="local"):
    """Compatibility context helper; inserts once via the governed RAM ring."""
    if isinstance(interaction_or_text,dict):
        data=dict(interaction_or_text)
    else:
        data={"input":str(interaction_or_text or ""),"final_response":str(final_response or ""),
              "intent":str(intent or "chat"),"source":str(source or "local")}
    data.setdefault("timestamp",datetime.now().isoformat())
    add_to_context(data)



def get_relevant_context(query: str, top_n: int = 3):
    """Phase B helper — retrieve top-N similar context turns for a query.

    Uses the existing in-RAM ring (context_buffer/context_embeddings) and
    falls back safely if embeddings are unavailable or disabled.
    """
    try:
        cfg = get_context_config()
        if not cfg.get("enabled", True):
            return []
    except Exception:
        cfg = {"enabled": True}

    try:
        if not query:
            return []
        # Use the same lite_embed path as the rest of the module
        emb = lite_embed(query)[0]
    except Exception:
        try:
            emb = _fallback_embed(query)
        except Exception:
            return []

    try:
        # Reuse the cosine similarity helper
        return retrieve_similar_context(emb, top_n)
    except Exception:
        return []



def init_context_history():
    """Hydrate context only when local persistence was explicitly opted into."""
    global _CONTEXT_INITIALIZED
    with _CONTEXT_LOCK:
        if _CONTEXT_INITIALIZED:
            return
        if not _sm_flag("AI_FUNCTIONS_CONTEXT_PERSIST_ENABLED", False):
            _CONTEXT_INITIALIZED=True
            return
        try:
            with contextlib.closing(_context_db_connect()) as con:
                _ensure_context_schema(con)
                rows=con.execute("SELECT timestamp,input,embedding FROM context_history ORDER BY id DESC LIMIT ?",
                    (max(1,min(int(getattr(config,"CONTEXT_BUFFER_SIZE",20) or 20),256)),)).fetchall()
            context_buffer.clear()
            context_embeddings.clear()
            for ts,ui,emb in reversed(rows):
                try:
                    vec=json.loads(emb) if emb else []
                    if not isinstance(vec,list) or not vec:
                        continue
                    floats=[float(v) for v in vec[:4096]]
                    if not all(math.isfinite(x) for x in floats):
                        continue
                    context_buffer.append({"timestamp":ts,"input":ui,"embedding":floats})
                    context_embeddings.append(np.asarray(floats,dtype=float))
                except (TypeError,ValueError):
                    continue
            _CONTEXT_INITIALIZED=True
        except Exception as exc:
            logger.warning("[Context] Initialization failed: %s",exc)

def route_query(user_text: str) -> str:
    """
    Main routing function with v8.0 advanced agent integration.

    IMPORTANT (Web UI behavior):
      - app.py will fall back to SarahMemoryReply.generate_reply when this returns
        the sentinel string "I'm unsure how to respond."
      - Therefore, for general chat that isn't handled by tools/knowledge graph,
        we intentionally return that sentinel so the richer reply pipeline (API/web/local DB)
        can run.
    """
    t = (user_text or "").strip()
    if not t:
        return "I didn't catch that."

    # Fast math path (prevents generic KG fallbacks on arithmetic)
    try:
        import re as _re
        if _re.match(r"^\s*[\d\s\+\-\*\/\(\)\.%\^]+\s*$", t):
            try:
                from SarahMemoryWebSYM import WebSemanticSynthesizer
                calc = WebSemanticSynthesizer()
                return str(calc.sarah_calculator(t))
            except Exception:
                # fall through to normal routing
                pass
    except Exception:
        pass


    # Fast identity guard for creator/origin (WebUI should never allow provider self-attribution)
    try:
        _ql = t.lower()
        if any(k in _ql for k in (
            "who created you",
            "who made you",
            "who built you",
            "who designed you",
            "who is your creator",
            "who developed you",
            "who programmed you",
            "who owns you",
        )):
            return (
                "I was created by Brian Lee Baros as part of the SarahMemory AI Companion Platform. "
                "I can use different AI engines and tools, but my identity and behavior are defined by SarahMemory."
            )
    except Exception:
        pass

    try:
        if getattr(config, "USE_ADVANCED_AGENT", True):
            r = advanced_agent_query(t)
            rs = (r or "").strip()

            # If the advanced agent produced a low-signal placeholder, let the full
            # reply pipeline (SarahMemoryReply -> SarahMemoryAPI/web/local DB) handle it.
            low_signal_prefixes = (
                "i have related information, but i need",
                "i have some related knowledge",
                "i’m here and listening",
                "i'm here and listening",
                "i understand you said:",
                "i don't have enough knowledge to answer that",
            )
            if not rs:
                return "I'm unsure how to respond."
            if rs.strip() == "I'm unsure how to respond.":
                return rs
            rsl = rs.lower()
            if any(rsl.startswith(p) for p in low_signal_prefixes):
                return "I'm unsure how to respond."
            return rs
        else:
            return _legacy_route_query(t)
    except Exception as e:
        logger.error(f"[RouteQuery] Error: {e}")
        return "I'm unsure how to respond."

def _legacy_route_query(user_text: str) -> str:
    t = str(user_text or "").strip()
    if not t:
        return "I didn't catch that."
    if t.lower().startswith(("https://", "http://", "www.")):
        return open_url_in_chrome(t)
    if re.fullmatch(r"[\d\s+*/().%^-]{1,200}",t) or t.lower().startswith("calculate "):
        return symbolic_calc_answer(t) or "I'm unsure how to respond."
    return "I'm unsure how to respond."

# ================================================================================
# MODULE INITIALIZATION
# ================================================================================

def _module_init():
    """Initialize module on import"""
    try:
        import_autostart = _sm_flag("ADVANCED_AGENT_IMPORT_AUTOSTART", False) and _sm_flag("AI_FUNCTIONS_IMPORT_AUTOSTART_APPROVED", False)

        # Initialize context history only when import autostart is enabled.
        # Cloud/public web WSGI boot should not hydrate DB-backed agent context
        # until a governed chat/agent path actually asks for it.
        if import_autostart:
            init_context_history()
        else:
            logger.info("[SarahMemoryAiFunctions] Context history import hydration held in governed standby")

        # Initialize advanced agent if enabled
        if getattr(config, "USE_ADVANCED_AGENT", True):
            if import_autostart:
                initialize_advanced_agent()
                logger.info("[SarahMemoryAiFunctions] v8.0 Advanced Agent loaded successfully")
            else:
                logger.info("[SarahMemoryAiFunctions] Advanced Agent held in governed import standby")
        else:
            logger.info("[SarahMemoryAiFunctions] Running in legacy mode")
    except Exception as e:
        logger.error(f"[SarahMemoryAiFunctions] Initialization error: {e}")

# Lifecycle cleanup is bounded and idempotent.
try:
    atexit.register(shutdown_advanced_agent)
except Exception:
    pass

# Auto-initialize on module load
_module_init()

# Legacy alias used by older test harnesses
route_intent_response = route_query

# ================================================================================
# EXPORTS
# ================================================================================

__all__ = [
    'route_query',
    'advanced_agent_query',
    'initialize_advanced_agent',
    'shutdown_advanced_agent',
    'checkpoint_context_history',
    'ADVANCED_AGENT',
    'Task',
    'TaskPriority',
    'TaskStatus',
    'HierarchicalPlanner',
    'MetaCognitiveReasoner',
    'KnowledgeGraphEngine',
    'ToolOrchestrator',
    'lite_embed',
    'submit_media_job',
    'poll_media_job',
    'store_media_result',
    'get_media_result',
]


# ================================================================================
# REMAINING LEGACY FUNCTIONS (MANUALLY ADDED)
# ================================================================================

def smart_reply(user_text: str) -> str:
    """High-level reply leveraging route_query with fallback"""
    try:
        return route_query(user_text)
    except Exception as e:
        logger.warning(f"[smart_reply] error: {e}")
        return "I'm working on that. Let me try again."

def create_chunks(text: str, chunk_size_tokens: int = 256, overlap_tokens: int = 64, language: str = "en") -> list:
    """Finite, bounded token chunker with compatible result schema."""
    size = max(1, min(int(chunk_size_tokens or 256), 4096))
    overlap = max(0, min(int(overlap_tokens or 0), size - 1))
    words = re.findall(r"\S+", str(text or ""))[:65536]
    chunks = []
    step = size - overlap
    for idx, start in enumerate(range(0, len(words), step)):
        if idx >= 512:
            break
        end = min(start + size, len(words))
        chunks.append({"id": idx, "text": " ".join(words[start:end]), "start": start, "end": end})
        if end == len(words):
            break
    return chunks

def detect_command_intent(text: str) -> str:
    """
    Lightweight intent router used by GUI/Reply:
    - returns 'image' for "show me ... image/picture" requests
    - returns 'math' if looks like an arithmetic expression
    - otherwise 'chat'
    """
    if not text:
        return "chat"
    import re
    low = text.lower().strip()
    if re.search(r"\bshow\s+me\s+(?:an?|some)?\s*(?:image|images|picture|pictures|pic|pics)\s+of\b", low):
        return "image"
    if re.search(r"^\s*\d+[\d\s\+\-\*\/\(\)\.]*$", low):
        return "math"
    return "chat"

def normalize_text(text: str) -> str:
    """Normalize text for processing"""
    try:
        return ' '.join((text or '').split())
    except Exception:
        return text

def open_url_in_chrome(url: str) -> str:
    """Use browser APIs, never os.startfile(command-line) composition."""
    permitted, reason = _sm_action_permission("url")
    if not permitted:
        return "URL opening denied: " + reason
    try:
        parsed = urllib.parse.urlsplit(str(url or "").strip())
        if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username or parsed.password:
            return "URL opening denied: public HTTPS URL required"
        import webbrowser
        opened = bool(webbrowser.open(parsed.geturl(), new=2))
        return "Opened " + parsed.geturl() if opened else "Browser did not confirm launch."
    except Exception as exc:
        return "URL opening failed: " + str(exc)[:160]

def draw_in_paint(subject: str = "dinosaur") -> str:
    """Legacy desktop drawing entry point; requires governed desktop execution."""
    permitted, reason = _sm_action_permission("desktop")
    if not permitted:
        return "Paint drawing denied: " + reason
    if not agent_guard():
        return "Paint drawing denied: agent not active or operator interacting"
    try:
        import SarahMemorySi as si
        fn = getattr(si, "manage_application_request", None)
        if not callable(fn) or not fn("open mspaint"):
            return "Paint launch could not be verified."
        return "Paint opened; drawing requires a verified CreativeStudios executor."
    except Exception:
        return "Paint launcher unavailable."



def _sm_norm(value: Any, limit: int = 8000) -> str:
    """Normalize surface task text without evaluating or executing content."""
    text = str(value or "").replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip()
    return text[:max(1, int(limit))]


def _sm_surface_topic_text(topic: str, title: str = "") -> str:
    topic = _sm_norm(topic or "the requested topic")
    if not topic:
        topic = "the requested topic"
    title_line = (title or topic.title() or "Document").strip()
    body = (
        f"{title_line}\n\n"
        f"This document provides a concise summary about {topic}. "
        f"It is intended as a starter draft that SarahMemory created directly in the requested application. "
        f"Key points may include definition, context, major characteristics, practical uses, and a short conclusion.\n\n"
        f"Overview\n"
        f"- {topic.title()} is the main subject of this document.\n"
        f"- This draft was generated locally to satisfy the user's requested workflow.\n\n"
        f"Summary\n"
        f"{topic.title()} can be described in practical terms using a short, readable structure. "
        f"The exact final wording can be refined later, but this provides a usable first draft.\n\n"
        f"Conclusion\n"
        f"This starter document gives a clean base for additional editing, formatting, and expansion."
    )
    return body


def _sm_website_scaffold(topic: str) -> dict:
    topic = _sm_norm(topic or 'website topic')
    title = html.escape(topic.title() if topic else 'Website Topic', quote=True)
    topic = html.escape(topic, quote=True)
    index_html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
  <title>{title}</title>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <nav>
      <a href=\"index.html\">Home</a> | <a href=\"about.html\">About</a>
    </nav>
  </header>
  <main>
    <h2>Welcome</h2>
    <p>This homepage is about {topic}.</p>
  </main>
</body>
</html>"""
    about_html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
  <title>About {title}</title>
</head>
<body>
  <header>
    <h1>About {title}</h1>
    <nav>
      <a href=\"index.html\">Home</a> | <a href=\"about.html\">About</a>
    </nav>
  </header>
  <main>
    <p>This page contains background information about {topic}.</p>
  </main>
</body>
</html>"""
    return {"index.html": index_html, "about.html": about_html}


def _sm_with_pag():
    if not _sm_action_permission("surface")[0]:
        return None
    try:
        import pyautogui as pag
        pag.FAILSAFE = True
        return pag
    except Exception:
        return None


def _sm_focus_app(app_name: str, timeout: float = 10.0) -> bool:
    if not _sm_action_permission("surface")[0]:
        return False
    try:
        import SarahMemorySi as _Si  # type: ignore
        if hasattr(_Si, 'wait_for_application_window'):
            _Si.wait_for_application_window(app_name, timeout=timeout, poll_s=0.35)
        if hasattr(_Si, 'focus_application'):
            if _Si.focus_application(app_name):
                return True
    except Exception:
        pass
    return False


def _sm_type_text(text: str, interval: float = 0.01, press_enter: bool = False) -> bool:
    if not _sm_action_permission("surface")[0]:
        return False
    try:
        import SarahMemorySi as _Si  # type: ignore
        fn = getattr(_Si, 'type_text_to_active_window', None)
        if callable(fn):
            return bool(fn(text, interval=interval, press_enter=press_enter))
    except Exception:
        pass
    pag = _sm_with_pag()
    if not pag:
        return False
    try:
        pag.write(str(text or ''), interval=max(0.0, float(interval or 0.0)))
        if press_enter:
            pag.press('enter')
        return True
    except Exception:
        return False


def _sm_hotkey(*keys: str) -> bool:
    if not _sm_action_permission("surface")[0]:
        return False
    try:
        import SarahMemorySi as _Si  # type: ignore
        fn = getattr(_Si, 'press_hotkey', None)
        if callable(fn):
            return bool(fn(*keys))
    except Exception:
        pass
    pag = _sm_with_pag()
    if not pag:
        return False
    try:
        pag.hotkey(*[str(k).lower() for k in keys if str(k).strip()])
        return True
    except Exception:
        return False


def _sm_open_and_focus(app_name: str) -> bool:
    if not _sm_action_permission("surface")[0]:
        return False
    try:
        import SarahMemorySi as _Si  # type: ignore
        if hasattr(_Si, 'manage_application_request'):
            if not _Si.manage_application_request(f"open {app_name}"):
                return False
            time.sleep(0.8)
            _sm_focus_app(app_name, timeout=10.0)
            return True
    except Exception:
        return False
    return False


def _sm_prepare_word_surface() -> bool:
    if not _sm_action_permission("surface")[0]:
        return False
    _sm_focus_app('winword', timeout=10.0)
    time.sleep(0.4)
    if not _sm_hotkey('ctrl', 'n'):
        return False
    time.sleep(0.8)
    return True


def _sm_prepare_excel_surface() -> bool:
    if not _sm_action_permission("surface")[0]:
        return False
    _sm_focus_app('excel', timeout=10.0)
    time.sleep(0.4)
    if not _sm_hotkey('ctrl', 'n'):
        return False
    time.sleep(0.8)
    return True


def _sm_prepare_notepad_surface() -> bool:
    if not _sm_action_permission("surface")[0]:
        return False
    _sm_focus_app('notepad', timeout=10.0)
    time.sleep(0.4)
    return True


def _sm_execute_browser_task(app_name: str, task: dict) -> dict:
    if not _sm_action_permission("surface")[0]:
        return {"ok": False, "reason": "governed_operatorcore_execution_required", "execution_authority": False}
    query = _sm_norm(task.get('search_query') or task.get('topic') or '')
    url = _sm_norm(task.get('target_url') or '')
    try:
        import SarahMemorySi as _Si  # type: ignore
        if query and hasattr(_Si, 'browser_search'):
            ok = bool(_Si.browser_search(app_name, query, timeout=10.0))
            return {'ok': ok, 'surface_open': ok, 'surface_focused': ok, 'working_context_ready': ok, 'content_applied': ok, 'result_verified': False, 'task_kind': 'browser_search', 'query': query}
        if url and hasattr(_Si, 'browser_open_url'):
            ok = bool(_Si.browser_open_url(app_name, url, timeout=10.0))
            return {'ok': ok, 'surface_open': ok, 'surface_focused': ok, 'working_context_ready': ok, 'content_applied': ok, 'result_verified': False, 'task_kind': 'browser_open_url', 'target_url': url}
    except Exception as e:
        return {'ok': False, 'error': str(e), 'task_kind': task.get('task_kind')}
    return {'ok': False, 'task_kind': task.get('task_kind'), 'reason': 'browser_helper_unavailable'}


def _sm_execute_word_task(task: dict, user_text: str = '') -> dict:
    if not _sm_action_permission("surface")[0]:
        return {"ok": False, "reason": "governed_operatorcore_execution_required", "execution_authority": False}
    if not _sm_open_and_focus('winword'):
        return {'ok': False, 'task_kind': task.get('task_kind'), 'reason': 'word_open_failed'}
    if task.get('task_kind') == 'open_named_document':
        name = _sm_norm(task.get('document_name') or task.get('title') or '')
        if not name:
            return {'ok': False, 'task_kind': 'open_named_document', 'reason': 'missing_document_name'}
        _sm_focus_app('winword', timeout=10.0)
        time.sleep(0.4)
        if not _sm_hotkey('ctrl', 'o'):
            return {'ok': False, 'task_kind': 'open_named_document', 'surface_open': True, 'surface_focused': True, 'reason': 'open_dialog_hotkey_failed'}
        time.sleep(0.8)
        ok = _sm_type_text(name, interval=0.01, press_enter=True)
        return {'ok': bool(ok), 'task_kind': 'open_named_document', 'surface_open': True, 'surface_focused': True, 'working_context_ready': True, 'content_applied': bool(ok), 'result_verified': False, 'document_name': name}
    if not _sm_prepare_word_surface():
        return {'ok': False, 'task_kind': task.get('task_kind'), 'surface_open': True, 'reason': 'word_surface_prepare_failed'}
    title = _sm_norm(task.get('title') or task.get('topic') or 'Document')
    topic = _sm_norm(task.get('topic') or user_text or 'the requested topic')
    body = _sm_norm(task.get('document_text') or _sm_surface_topic_text(topic, title))
    ok = _sm_type_text(body, interval=0.008, press_enter=False)
    return {'ok': bool(ok), 'task_kind': task.get('task_kind') or 'document_write', 'surface_open': True, 'surface_focused': True, 'working_context_ready': True, 'content_generated': True, 'content_applied': bool(ok), 'result_verified': False, 'candidate_text': body, 'title': title, 'topic': topic}


def _sm_execute_excel_task(task: dict) -> dict:
    if not _sm_action_permission("surface")[0]:
        return {"ok": False, "reason": "governed_operatorcore_execution_required", "execution_authority": False}
    if not _sm_open_and_focus('excel'):
        return {'ok': False, 'task_kind': task.get('task_kind'), 'reason': 'excel_open_failed'}
    if not _sm_prepare_excel_surface():
        return {'ok': False, 'task_kind': task.get('task_kind'), 'surface_open': True, 'reason': 'excel_surface_prepare_failed'}
    headers = task.get('headers') or ['Date','Description','Category','Debit','Credit','Balance']
    header_line = '\t'.join(str(h) for h in headers)
    sample_rows = ['\n' + '\t'.join(['', '', '', '', '', '']) for _ in range(8)]
    sheet_text = str(task.get('title') or 'Checkbook Register') + '\n' + header_line + ''.join(sample_rows)
    ok = _sm_type_text(sheet_text, interval=0.006, press_enter=False)
    return {'ok': bool(ok), 'task_kind': task.get('task_kind') or 'spreadsheet_template', 'surface_open': True, 'surface_focused': True, 'working_context_ready': True, 'content_generated': True, 'content_applied': bool(ok), 'result_verified': False, 'candidate_text': sheet_text}


def _sm_execute_editor_task(app_name: str, task: dict) -> dict:
    if not _sm_action_permission("surface")[0]:
        return {"ok": False, "reason": "governed_operatorcore_execution_required", "execution_authority": False}
    if not _sm_open_and_focus(app_name):
        return {'ok': False, 'task_kind': task.get('task_kind'), 'reason': f'{app_name}_open_failed'}
    if app_name == 'notepad':
        _sm_prepare_notepad_surface()
    topic = _sm_norm(task.get('topic') or 'website topic')
    pages = _sm_website_scaffold(topic)
    block = []
    for name, html in pages.items():
        block.append(f'<!-- {name} -->\n{html}\n')
    payload = '\n\n'.join(block)
    ok = _sm_type_text(payload, interval=0.004, press_enter=False)
    return {'ok': bool(ok), 'task_kind': task.get('task_kind') or 'website_scaffold', 'surface_open': True, 'surface_focused': True, 'working_context_ready': True, 'content_generated': True, 'content_applied': bool(ok), 'result_verified': False, 'candidate_text': payload, 'pages': list(pages.keys())}


def _sm_execute_paint_task(task: dict) -> dict:
    if not _sm_action_permission("surface")[0]:
        return {"ok": False, "reason": "governed_operatorcore_execution_required", "execution_authority": False}
    subject = _sm_norm(task.get('draw_subject') or task.get('shape') or 'shape')
    try:
        result_text = draw_in_paint(subject)
        ok = not str(result_text or '').lower().startswith(('paint opened but drawing failed', 'paint not found', 'could not draw'))
        return {'ok': bool(ok), 'task_kind': task.get('task_kind') or 'paint_draw', 'surface_open': True if 'paint' in str(result_text).lower() or ok else False, 'surface_focused': bool(ok), 'working_context_ready': bool(ok), 'content_generated': True, 'content_applied': bool(ok), 'result_verified': False, 'result': result_text, 'candidate_text': str(result_text)}
    except Exception as e:
        return {'ok': False, 'task_kind': task.get('task_kind') or 'paint_draw', 'error': str(e)}


def execute_surface_task(canonical_app: str, task: dict, user_text: str = '') -> dict:
    """Legacy surface adapter; fail closed unless caller supplies verified authority."""
    permitted, reason = _sm_action_permission("surface")
    if not permitted:
        return {"ok": False, "reason": reason, "execution_authority": False,
                "task_kind": str((task or {}).get("task_kind") or "unknown")}
    task = dict(task or {})
    if not _sm_user_command(user_text or str(task.get("command") or "")):
        return {"ok": False, "reason": "explicit_user_command_required", "execution_authority": False}
    app_name = _sm_norm(canonical_app or task.get("requested_app_exec") or task.get("requested_app") or "")
    kind = _sm_norm(task.get("task_kind") or "")
    if kind in {"browser_search", "browser_open_url"}:
        if not _sm_network_allowed():
            return {"ok": False, "reason": "network_forbidden_by_runtime_mode", "execution_authority": False}
        return _sm_execute_browser_task(app_name or "msedge", task)
    if app_name == "winword" or kind in {"document_write", "open_named_document"}:
        return _sm_execute_word_task(task, user_text=user_text)
    if app_name == "excel" or kind == "spreadsheet_template":
        return _sm_execute_excel_task(task)
    if app_name in {"notepad", "code", "dreamweaver"} and kind == "website_scaffold":
        return _sm_execute_editor_task(app_name, task)
    if app_name == "mspaint" or kind == "paint_draw":
        return _sm_execute_paint_task(task)
    return {"ok": False, "reason": "no_surface_executor_available", "execution_authority": False,
            "task_kind": kind or "unknown", "requested_app": app_name}


def rerank(query: str, candidates: list, cross_encoder=None) -> list:
    """
    Rerank candidates by relevance to query.
    If cross_encoder provided, use it; else return as-is.
    """
    if cross_encoder is None:
        return candidates
    try:
        pairs = [(query, c) for c in candidates]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _s in ranked]
    except Exception as e:
        logger.warning(f"[rerank] error: {e}")
        return candidates

def list_tools() -> list:
    """List governed tools with enabled state"""
    tools = [
        {"name": "memory_search", "description": "Search local memory"},
        {"name": "web_search", "description": "Search the web"},
        {"name": "calc", "description": "Calculate mathematical expressions"},
        {"name": "open_url", "description": "Open a URL in browser"},
        {"name": "agent", "description": "AI agent control"},
    ]
    if _sm_creative_studio_registry():
        tools.append({"name": "creative", "description": "CreativeStudios bridge"})
    for row in tools:
        row["enabled"] = _sm_tool_allowed(row["name"]) or (row["name"] == "creative" and bool(_sm_creative_studio_registry()))
    return tools

def exec_tool(name: str, arg: str = "") -> dict:
    """Compatibility facade; all calls use the same policy as ToolOrchestrator."""
    if name == "creative":
        return {"result": sorted(_sm_creative_studio_registry())} if _sm_creative_studio_registry() else {"error":"creative_unavailable"}
    if name not in {"memory_search","web_search","calc","open_url","agent"}:
        return {"error":"unknown_tool"}
    if not _sm_tool_allowed(name):
        return {"error":_sm_tool_error(name)}
    if name == "web_search" and not _sm_network_allowed():
        return {"error":"network_forbidden_by_runtime_mode"}
    if name in {"agent", "open_url"}:
        allowed, reason = _sm_action_permission("desktop" if name=="agent" else "url")
        if not allowed:
            return {"error":reason}
    try:
        if name == "memory_search": return {"result":local_memory_lookup(arg)}
        if name == "web_search": return {"result":web_research_query(arg)}
        if name == "calc": return {"result":symbolic_calc_answer(arg)}
        if name == "open_url": return {"result":open_url_in_chrome(arg)}
        return {"result":handle_ai_agent_command(arg)}
    except Exception as exc:
        return {"error":str(exc)[:180]}

def _simple_plan(text: str) -> list:
    """Simple task planning"""
    return [
        {"action": "understand", "description": "Understand the query"},
        {"action": "search", "description": "Search for information"},
        {"action": "respond", "description": "Formulate response"}
    ]

def plan_and_act(user_text: str, max_steps: int = 4) -> dict:
    """Return a bounded, non-executing proposal; never fabricate completion."""
    plan = _simple_plan(str(user_text or "")[:2000])[:max(0, min(int(max_steps), 16))]
    return {"plan": plan,
            "results": [{"step": s["action"], "result": "PROPOSED: " + s["description"], "executed": False} for s in plan],
            "final": "Proposal only; execution requires existing governed approval.",
            "status": "PROPOSED", "execution_started": False, "execution_authority": False}

# Update exports to include all functions
_LEGACY_EXPORTS = [
    'smart_reply', 'create_chunks', 'detect_command_intent', 'normalize_text',
    'open_url_in_chrome', 'draw_in_paint', 'rerank', 'list_tools', 'exec_tool',
        'execute_surface_task',
    'plan_and_act', 'execute_surface_task'
]

# Extend __all__ if it exists
try:
    __all__.extend(_LEGACY_EXPORTS)
except:
    pass


try:
    __all__.extend([
        'list_tools',
        'exec_tool',
        'execute_surface_task',
        '_sm_creative_studio_registry',
    ])
except Exception:
    pass

# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 START ---
# High-level agent runtime coordinator. It plans/stages only and routes execution
# decisions into CognitiveServices/OperatorCore. No direct tool execution.

class SarahMemoryAgentRuntime:
    """RAM-first coordinator for agentic tasks under SarahMemory governance."""

    def __init__(self, max_tasks: int = 64) -> None:
        self.max_tasks = max(8, int(max_tasks or 64))
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def submit_task(self, user_goal: str, *, context: Optional[Dict[str, Any]] = None, proposed_action: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        tid = uuid.uuid4().hex
        rhythm_context = dict(context or {})
        rhythm_context.update({"goal": str(user_goal or ""), "proposed_action": dict(proposed_action or {}), "loop": "sovereign_agent_submit"})
        rhythm_packet = _sm_rhythm_packet(rhythm_context, force_refresh=True)
        rhythm_guard = _sm_rhythm_throttle("sovereign_agent_submit", context=rhythm_context, min_interval_sec=float(rhythm_packet.get("agent_step_interval_sec", 0.75) or 0.75))
        task = {
            "task_id": tid,
            "created_ts": datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "goal": str(user_goal or "")[:2000],
            "status": "PROPOSED",
            "context": dict(context or {}),
            "proposed_action": dict(proposed_action or {}),
            "rhythm_cognition": rhythm_packet,
            "rhythm_guard": rhythm_guard,
            "one_way_broker": True,
            "direct_execution": False,
        }
        governance = {"ok": False, "decision": "DEFER", "allow": False}
        if not bool(rhythm_guard.get("allow", True)):
            task["status"] = "WAITING_CADENCE"
            task["governance"] = {"ok": True, "decision": "DEFER", "allow": False, "reason": "RhythmCognition anti-thrash cadence guard deferred this task."}
            self.tasks[tid] = task
            return {"ok": True, "task": task, "execution_started": False}
        try:
            import SarahMemoryCognitiveServices as _CogSvc  # type: ignore
            fn = getattr(_CogSvc, "govern_interop_broker_request", None)
            if callable(fn) and (proposed_action or {}).get("protocol"):
                governance = fn(proposed_action or {}, caller_context=context or {})
            else:
                fn2 = getattr(_CogSvc, "process_cognitive_request", None)
                if callable(fn2):
                    governance = fn2(str(user_goal or ""), caller_context=context or {}, proposed_action=proposed_action or {})
        except Exception as exc:
            governance = {"ok": False, "decision": "DEFER", "allow": False, "error": str(exc)}
        task["governance"] = governance
        task["status"] = "APPROVED" if bool(governance.get("allow")) else "WAITING_USER"
        self.tasks[tid] = task
        if len(self.tasks) > self.max_tasks:
            oldest = sorted(self.tasks.values(), key=lambda r: r.get("created_ts") or "")[:len(self.tasks)-self.max_tasks]
            for rec in oldest:
                self.tasks.pop(rec.get("task_id"), None)
        return {"ok": True, "task": task, "execution_started": False}

    def status(self, task_id: str = "") -> Dict[str, Any]:
        if task_id:
            return {"ok": bool(task_id in self.tasks), "task": self.tasks.get(task_id)}
        return {"ok": True, "count": len(self.tasks), "rhythm_cognition": _sm_rhythm_packet({"caller": "agent_runtime_status"}), "tasks": list(self.tasks.values())}


_AGENT_RUNTIME = SarahMemoryAgentRuntime()


def submit_sovereign_agent_task(user_goal: str, *, context: Optional[Dict[str, Any]] = None, proposed_action: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return _AGENT_RUNTIME.submit_task(user_goal, context=context, proposed_action=proposed_action)


def get_sovereign_agent_runtime_status(task_id: str = "") -> Dict[str, Any]:
    return _AGENT_RUNTIME.status(task_id)


try:
    __all__.extend([
        '_sm_rhythm_packet',
        '_sm_rhythm_interval',
        '_sm_rhythm_throttle',
    ])
except Exception:
    pass

# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 END ---
# ====================================================================
# END OF SarahMemoryAiFunctions.py v9.1.0a
# ====================================================================


# ---------------------------------------------------------------------------
# v9.1.0a: compatibility contracts used by existing CORE and API callers.
# All return evidence/proposals; no automatic execution or memory authority.
# ---------------------------------------------------------------------------
def plan_actions(intent, entities=None, user_context=None, device_context=None):
    """Diagnostics-compatible planning only (no tools, network, or side effects)."""
    query = str((user_context or {}).get("raw_text") or "") if isinstance(user_context, dict) else ""
    task = HierarchicalPlanner(ADVANCED_AGENT).create_task(query[:2000], str(intent or "unknown"))
    return [{"tool": name, "description": task.description[:200], "status": "PROPOSED",
             "requires_approval": True, "execution_authority": False}
            for name in task.tools_required[:16]]


def record_ai_response(intent, response, confidence=0.0, source="ai_functions"):
    """Personality compatibility: opt-in local context observation only."""
    if not _sm_flag("AI_FUNCTIONS_RECORD_RESPONSES_ENABLED", False):
        return {"ok": False, "reason": "response_recording_disabled", "execution_authority": False}
    add_to_context({"input": "", "final_response": str(response or "")[:2000],
                    "intent": str(intent or "")[:96], "source": str(source or "")[:96],
                    "timestamp": datetime.now().isoformat(),
                    "confidence": max(0.0, min(1.0, float(confidence or 0.0)))})
    return {"ok": True, "recorded": "ram_ring", "execution_authority": False}


def vector_backed_local_answer(text: str):
    """Reply-compatible local retrieval; never treats unverified memory as fact."""
    if not str(text or "").strip():
        return None
    try:
        from SarahMemoryAdvCU import resolve_local_fast_answer
        result = resolve_local_fast_answer(str(text)[:2000], allow_learning_record=False,
                                           approved_for_learning=False)
        if (isinstance(result, dict) and result.get("ok") and result.get("answer")
                and str(result.get("source") or "").lower() in {"verified_qa_cache", "verified_qa", "local_verified_qa", "verified_local_qa_cache"}):
            return str(result["answer"])[:4000]
    except Exception:
        pass
    return None


def run_automation(payload):
    """API compatibility: stage an automation proposal; never claim it ran."""
    if not isinstance(payload, dict):
        return {"ok": False, "reason": "payload_must_be_object", "execution_started": False}
    description = str(payload.get("goal") or payload.get("task") or payload.get("command") or "")[:2000]
    if not description:
        return {"ok": False, "reason": "task_required", "execution_started": False}
    return {"ok": False, "status": "AWAITING_GOVERNED_EXECUTOR", "goal": description,
            "reason": "automation_requires_existing_operatorcore_approval_and_executor",
            "execution_started": False, "execution_authority": False}

try:
    __all__.extend(["plan_actions", "record_ai_response", "vector_backed_local_answer", "run_automation"])
except Exception:
    pass

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryAiFunctions',
    "version": "v9.1.0a-alpha-sml-0.2",
    "category": 'AgentOrchestration',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['planning', 'coordination', 'bounded_tools'],
    "supported_missions": ['Conversation', 'Planning', 'Governance'],
    "supported_omega": ['Ω001'],
    "required_authority": ['Read'],
    "priority": 40,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryAiFunctions.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return dict(SML_ORGAN_METADATA)


def sml_health():
    """Truthful non-invasive vector; unknown values use numeric sentinels."""
    workers=list(getattr(ADVANCED_AGENT,"executor_threads",[]))
    active=sum(bool(t.is_alive()) for t in workers)
    started=bool(getattr(ADVANCED_AGENT,"_initialized",False))
    return {"status":"Operational" if started and active == 4 else "Degraded" if started else "Standby",
        "availability": (active / 4.0) if started else 0.0,
        "integrity":0.0,"performance":0.0,"reliability":0.0,
        "confidence":0.0,"latency_ms":0.0,"stability":0.0,"compatibility":0.0,
        "notes":["Unknown metrics are not asserted as healthy", "active_workers="+str(active),
                 "standby is intentional when autostart is disabled"]}


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryAiFunctions',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryAiFunctions', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---
