"""--==The SarahMemory Project==--
File: SarahMemoryAdaptive.py
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

ADAPTIVE BEHAVIOR ENGINE
======================================================

PURPOSE:
--------
This module serves as the "personality auto-tuning layer" that learns and adapts
to user preferences, patterns, timing, and operational habits. It sits beneath
SarahMemoryPersonality.py and provides the dynamic behavioral foundation.

KEY CAPABILITIES:
-----------------
1. EMOTIONAL INTELLIGENCE
   - Real-time sentiment analysis from user input
   - Plutchik's 8 primary emotions model (joy, trust, fear, surprise, sadness, disgust, anger, anticipation)
   - Emotional momentum and decay over time
   - Emotional memory persistence to database

2. ADAPTIVE LEARNING
   - User preference pattern recognition
   - Response style adaptation (verbosity, formality, humor)
   - Time-of-day behavioral adjustments
   - Session-based and long-term learning

3. SYSTEM AWARENESS
   - CPU/Memory-aware mode switching (lightweight/balanced/enhanced)
   - Resource-conscious emotional processing
   - Graceful degradation under load

4. REINFORCEMENT LEARNING
   - Positive/negative feedback integration
   - Interaction quality scoring
   - Behavioral adjustment based on outcomes

INTEGRATION POINTS:
-------------------
- SarahMemoryPersonality.py: Imports emotional learning functions
- SarahMemoryReply.py: Uses advanced_emotional_learning for response tuning
- SarahMemoryAPI.py: Imports simulate_emotion_response for API responses
- SarahMemoryAvatar.py: Imports load_emotional_state for visual expressions
- SarahMemorySystemLearn.py: Uses emotional metrics for learning
- SarahMemory-local_api_server.py: Imports update_personality

DATABASE TABLES:
----------------
- ai_learning.db: conversations (interaction history)
- personality1.db: traits (emotional state persistence)

===============================================================================
"""

from __future__ import annotations
# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "adaptive_engine"
# CATEGORY = "personality_adaptation"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "adaptive_behavior"
# FAMILY = "core_adaptation"
# GOVERNANCE_LEVEL = "bounded_read_only_advisory"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Adaptive behavior and emotional tuning engine. Learns preferences, tracks emotional state, adjusts response style, and degrades gracefully by resource mode."
# --- SARAHMETA END ---
import logging
import sqlite3
import datetime
import hashlib
import os
import platform
import random
import shutil
import time
import json
import math
import threading
import copy
import re
import contextvars
from contextlib import contextmanager, closing
import pathlib
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from collections import deque, OrderedDict
from enum import Enum

# ------------------------- Safe Imports -------------------------
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    psutil = None
    _HAS_PSUTIL = False

try:
    import SarahMemoryGlobals as config
except ImportError:
    # Fallback config stub for standalone testing
    class config:
        DATASETS_DIR = os.path.join(os.getcwd(), "data", "memory", "datasets")
        BASE_DIR = os.getcwd()
        DEBUG_MODE = True
        ENABLE_RESPONSE_LOG_TABLE = False


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Database paths
MEMORY_DB_PATH = os.path.join(config.DATASETS_DIR, "ai_learning.db")
EMOTION_DB_PATH = os.path.join(config.DATASETS_DIR, "personality1.db")

COMPUTE_PASSPORT_SCHEMA = "SarahMemory.ComputePassport/1.0-alpha"
ADAPTIVE_COMPUTE_PLAN_SCHEMA = "SarahMemory.AdaptiveComputePlan/1.0-alpha"
_COMPUTE_METRIC_KEYS = (
    "cpu_count", "cpu_pct", "ram_total_mb", "ram_avail_mb",
    "disk_free_gb", "disk_total_gb", "gpu_name",
    "gpu_vram_total_mb", "gpu_vram_free_mb", "gpu_temp_c", "cpu_temp_c",
)

# Sentiment lexicons (expanded for better coverage)
POSITIVE_KEYWORDS = frozenset([
    # Joy/Happiness
    "good", "great", "awesome", "fantastic", "nice", "love", "wonderful",
    "amazing", "excellent", "brilliant", "superb", "perfect", "beautiful",
    "happy", "glad", "pleased", "delighted", "thrilled", "excited", "joyful",
    # Trust/Appreciation
    "thank", "thanks", "appreciate", "grateful", "helpful", "kind", "trust",
    "reliable", "honest", "genuine", "sincere", "loyal", "faithful",
    # Satisfaction
    "satisfied", "content", "fulfilled", "accomplished", "successful", "proud",
    # Affirmation
    "yes", "agree", "correct", "right", "exactly", "absolutely", "definitely",
    "certainly", "sure", "okay", "ok", "fine", "cool", "neat", "sweet"
])

NEGATIVE_KEYWORDS = frozenset([
    # Anger
    "bad", "terrible", "awful", "hate", "angry", "furious", "upset", "mad",
    "annoyed", "irate", "rage", "frustrat", "irritat", "outrag", "hostile",
    # Sadness
    "sad", "depressed", "unhappy", "miserable", "disappointed", "heartbroken",
    "grief", "sorrow", "lonely", "hopeless", "despair", "gloomy", "melancholy",
    # Fear
    "scared", "afraid", "frightened", "terrified", "anxious", "worried",
    "nervous", "panic", "dread", "horror", "alarmed", "uneasy",
    # Disgust
    "disgusting", "gross", "revolting", "repulsive", "nasty", "vile", "sick",
    # Negation/Rejection
    "no", "wrong", "incorrect", "mistake", "error", "fail", "broken", "useless",
    "stupid", "dumb", "idiotic", "worst", "horrible", "pathetic", "garbage"
])

NEUTRAL_KEYWORDS = frozenset([
    "okay", "alright", "fine", "normal", "regular", "standard", "usual",
    "average", "moderate", "typical", "common", "ordinary"
])

# Surprise indicators
SURPRISE_INDICATORS = frozenset([
    "wow", "whoa", "really", "seriously", "omg", "unexpected", "surprise",
    "shocking", "incredible", "unbelievable", "astonishing", "amazing"
])

# Question indicators
QUESTION_INDICATORS = frozenset([
    "what", "why", "how", "when", "where", "who", "which", "whose", "whom",
    "can", "could", "would", "should", "will", "is", "are", "do", "does"
])


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class SystemMode(Enum):
    """System operating modes based on resource availability."""
    LIGHTWEIGHT = "lightweight"  # High CPU/Memory - minimal processing
    BALANCED = "balanced"        # Normal operation
    ENHANCED = "enhanced"        # Low resource usage - full features
    SAFE = "safe"                # Error recovery mode


class EmotionType(Enum):
    """Plutchik's wheel of emotions - 8 primary emotions."""
    JOY = "joy"
    TRUST = "trust"
    FEAR = "fear"
    SURPRISE = "surprise"
    SADNESS = "sadness"
    DISGUST = "disgust"
    ANGER = "anger"
    ANTICIPATION = "anticipation"


@dataclass
class EmotionalState:
    """
    Complete emotional state representation using Plutchik's model.
    All values are normalized to [0.0, 1.0] range.
    """
    # Primary emotions (Plutchik's wheel)
    joy: float = 0.5
    trust: float = 0.3
    fear: float = 0.2
    surprise: float = 0.1
    sadness: float = 0.1
    disgust: float = 0.05
    anger: float = 0.1
    anticipation: float = 0.3
    
    # Derived metrics
    emotional_balance: float = 0.0  # -1.0 (negative) to +1.0 (positive)
    openness: float = 0.6           # Willingness to engage
    engagement: float = 0.4         # Level of interaction depth
    
    # System state
    mode: str = "balanced"
    cpu: float = 0.0
    memory: float = 0.0
    
    # Metadata
    last_updated: str = ""
    adjustments: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "joy": round(self.joy, 4),
            "trust": round(self.trust, 4),
            "fear": round(self.fear, 4),
            "surprise": round(self.surprise, 4),
            "sadness": round(self.sadness, 4),
            "disgust": round(self.disgust, 4),
            "anger": round(self.anger, 4),
            "anticipation": round(self.anticipation, 4),
            "emotional_balance": round(self.emotional_balance, 4),
            "openness": round(self.openness, 4),
            "engagement": round(self.engagement, 4),
            "mode": self.mode,
            "cpu": round(self.cpu, 2),
            "memory": round(self.memory, 2),
            "last_updated": self.last_updated,
            "adjustments": self.adjustments[-10:] if self.adjustments else []  # Keep last 10
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalState':
        """Finite, validated reconstruction from persisted or untrusted JSON."""
        state = _validated_state(data)
        return cls(**{key: state[key] for key in cls.__dataclass_fields__})
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """Return the strongest emotion and its intensity."""
        emotions = {
            "joy": self.joy,
            "trust": self.trust,
            "fear": self.fear,
            "surprise": self.surprise,
            "sadness": self.sadness,
            "disgust": self.disgust,
            "anger": self.anger,
            "anticipation": self.anticipation
        }
        dominant = max(emotions.items(), key=lambda x: x[1])
        return dominant
    
    def get_emotion_label(self) -> str:
        """Get a human-readable emotion label."""
        name, intensity = self.get_dominant_emotion()
        
        # High intensity thresholds
        if self.anger >= 0.60:
            return "angry"
        if self.joy >= 0.60:
            return "happy"
        if self.sadness >= 0.55 or self.fear >= 0.55:
            return "sad"
        if self.surprise >= 0.55:
            return "surprised"
        if self.trust >= 0.60:
            return "trusting"
        if self.anticipation >= 0.55:
            return "eager"
        if self.disgust >= 0.50:
            return "disgusted"
        
        # Default based on balance
        if self.emotional_balance > 0.3:
            return "positive"
        elif self.emotional_balance < -0.3:
            return "negative"
        
        return "neutral"


@dataclass
class InteractionMetrics:
    """Metrics from a single user interaction."""
    sentiment_score: float = 0.0      # -1.0 to +1.0
    word_count: int = 0
    question_detected: bool = False
    surprise_detected: bool = False
    emotional_intensity: float = 0.0   # 0.0 to 1.0
    response_quality: float = 0.5      # Predicted quality
    timestamp: str = ""


# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger("SarahMemoryAdaptive")
if not logger.hasHandlers():
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
    ))
    logger.addHandler(_handler)
logger.setLevel(logging.DEBUG if getattr(config, 'DEBUG_MODE', False) else logging.INFO)


# ============================================================================
# GLOBAL STATE (Thread-Safe)
# ============================================================================

# Thread lock for state modifications
_state_lock = threading.RLock()

# Current emotional state (runtime)
STATE: Dict[str, Any] = {}

# Default emotional state template
DEFAULT_EMOTIONS: Dict[str, Any] = {
    "mode": "balanced",
    "cpu": 0.0,
    "memory": 0.0,
    "joy": 0.5,
    "trust": 0.3,
    "fear": 0.2,
    "surprise": 0.1,
    "sadness": 0.1,
    "disgust": 0.05,
    "anger": 0.1,
    "anticipation": 0.3,
    "emotional_balance": 0.0,
    "openness": 0.6,
    "engagement": 0.4,
    "last_updated": "",
    "adjustments": []
}

# Initialize STATE with defaults
STATE = copy.deepcopy(DEFAULT_EMOTIONS)

# Interaction history buffer (in-memory ring buffer)
_interaction_buffer: deque = deque(maxlen=100)

# Emotional momentum (tracks recent emotional trajectory)
_emotion_momentum: deque = deque(maxlen=20)



# ============================================================================
# ENTERPRISE RUNTIME BOUNDARIES -- local-first, bounded, advisory only
# ============================================================================
# Explicit scope is REQUIRED at the API/request boundary for multi-user use.
# The unscoped legacy interfaces continue to share the original STATE for
# backward compatibility. A scope identifier is hashed before persistence.
_ADAPTIVE_SCOPE = contextvars.ContextVar("sarahmemory_adaptive_scope", default="")
_SCOPED_STATES: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_SCOPED_MOMENTUM: "OrderedDict[str, deque]" = OrderedDict()
_SCOPED_INTERACTIONS: "OrderedDict[str, deque]" = OrderedDict()
_MAX_ACTIVE_SCOPES = 64
_SCHEMA_LOCK = threading.RLock()
_SCHEMA_READY: Dict[str, Tuple[int, int]] = {}
_EMOTION_FIELDS = ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation")
_STATE_FIELDS = frozenset(DEFAULT_EMOTIONS)
_ADAPTIVE_INPUT_LIMIT = 65536
_ADAPTIVE_TEXT_STORAGE_LIMIT = 16384


def _safe_float(value: Any, default: float = 0.0, lo: float = -1.0, hi: float = 1.0) -> float:
    """Reject nonfinite and out-of-range values at the adaptive trust boundary."""
    try:
        result = float(value)
        return max(lo, min(hi, result)) if math.isfinite(result) else default
    except (TypeError, ValueError, OverflowError):
        return default


def _validated_state(data: Any) -> Dict[str, Any]:
    """Accept only declared state keys, bounded finite floats and bounded history."""
    state = copy.deepcopy(DEFAULT_EMOTIONS)
    if not isinstance(data, dict):
        return state
    for field_name in _EMOTION_FIELDS + ("openness", "engagement"):
        state[field_name] = _safe_float(data.get(field_name), state[field_name], 0.0, 1.0)
    state["emotional_balance"] = _safe_float(data.get("emotional_balance"), 0.0)
    for name in ("cpu", "memory"):
        state[name] = _safe_float(data.get(name), 0.0, -1.0, 100.0)
    mode = str(data.get("mode", "balanced"))[:32]
    state["mode"] = mode if mode in {m.value for m in SystemMode} else "safe"
    state["last_updated"] = str(data.get("last_updated") or "")[:64]
    changes = data.get("adjustments")
    if isinstance(changes, list):
        state["adjustments"] = [str(x)[:180] for x in changes[-10:]]
    return state


def _scope_key() -> str:
    return _ADAPTIVE_SCOPE.get()


@contextmanager
def adaptive_scope(scope_id: str):
    """Bind a trusted, nonempty session/user/node key for one request.

    The caller MUST authenticate scope_id; this function does not authorize it.
    Scope identifiers are never stored in raw form. Nested scopes restore state.
    """
    if not isinstance(scope_id, str) or not scope_id.strip() or len(scope_id) > 512:
        raise ValueError("adaptive scope requires an authenticated, bounded identifier")
    digest = hashlib.sha256(scope_id.encode("utf-8")).hexdigest()
    token = _ADAPTIVE_SCOPE.set("scope_" + digest)
    try:
        yield
    finally:
        _ADAPTIVE_SCOPE.reset(token)


def _bounded_scoped_cache(cache: OrderedDict, key: str, constructor):
    if key not in cache:
        if len(cache) >= _MAX_ACTIVE_SCOPES:
            cache.popitem(last=False)
        cache[key] = constructor()
    cache.move_to_end(key)
    return cache[key]


def _current_state() -> Dict[str, Any]:
    key = _scope_key()
    if not key:
        return STATE
    return _bounded_scoped_cache(_SCOPED_STATES, key, lambda: copy.deepcopy(DEFAULT_EMOTIONS))


def _replace_current_state(state: Dict[str, Any]) -> Dict[str, Any]:
    global STATE
    validated = _validated_state(state)
    if _scope_key():
        _SCOPED_STATES[_scope_key()] = validated
        _SCOPED_STATES.move_to_end(_scope_key())
    else:
        STATE = validated
    return validated


def _momentum_buffer() -> deque:
    key = _scope_key()
    if not key:
        return _emotion_momentum
    return _bounded_scoped_cache(_SCOPED_MOMENTUM, key, lambda: deque(maxlen=20))


def _interaction_history_buffer() -> deque:
    key = _scope_key()
    if not key:
        return _interaction_buffer
    return _bounded_scoped_cache(_SCOPED_INTERACTIONS, key, lambda: deque(maxlen=100))


def clear_adaptive_runtime_scope() -> bool:
    """Forget the current in-memory scope only; does not delete persistent data."""
    key = _scope_key()
    if not key:
        return False
    with _state_lock:
        _SCOPED_STATES.pop(key, None)
        _SCOPED_MOMENTUM.pop(key, None)
        _SCOPED_INTERACTIONS.pop(key, None)
    return True


def _db_file_identity(db_path: str) -> Tuple[int, int]:
    try:
        s = os.stat(db_path)
        return (s.st_dev, s.st_ino)
    except OSError:
        return (-1, -1)


def _scope_allowed() -> bool:
    """Fail closed on multi-user hosts configured to require request scopes."""
    return bool(_scope_key()) or not bool(getattr(config, "ADAPTIVE_REQUIRE_SCOPE", False))


def _adaptive_can_store_raw() -> bool:
    """Legacy raw history defaults enabled; may be disabled by the owner."""
    return bool(getattr(config, "ADAPTIVE_STORE_RAW_CONVERSATIONS", True))


def _adaptive_storage_text(value: Any) -> str:
    return str(value or "")[:_ADAPTIVE_TEXT_STORAGE_LIMIT]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _sigmoid(x: float) -> float:
    """Stable finite logistic transform for backward compatibility."""
    x = _safe_float(x, 0.0, -1000.0, 1000.0)
    if x >= 0:
        ex = math.exp(-x)
        return 1.0 / (1.0 + ex)
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp finite values; nonfinite inputs cannot propagate to state or JSON."""
    return _safe_float(value, min_val, min_val, max_val)



def _get_system_resources() -> Tuple[float, float]:
    """Live pressure percentages; (-1, -1) means unavailable, never synthetic."""
    if not _HAS_PSUTIL or psutil is None:
        return (-1.0, -1.0)
    try:
        cpu = _safe_float(psutil.cpu_percent(interval=None), -1.0, 0.0, 100.0)
        mem = _safe_float(psutil.virtual_memory().percent, -1.0, 0.0, 100.0)
        return cpu, mem
    except (AttributeError, OSError, RuntimeError, ValueError) as exc:
        logger.warning("System telemetry unavailable: %s", type(exc).__name__)
        return (-1.0, -1.0)



def _get_timestamp() -> str:
    """Get current ISO timestamp."""
    return datetime.datetime.now().isoformat()


def _tokenize(text: str) -> List[str]:
    """Bounded Unicode-aware words, preserving negation contractions."""
    if not isinstance(text, str):
        return []
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)*", text[:_ADAPTIVE_INPUT_LIMIT].lower(), flags=re.UNICODE)



# ============================================================================
# DATABASE OPERATIONS - MEMORY DB (ai_learning.db)
# ============================================================================

def _ensure_memory_db_schema(conn: sqlite3.Connection) -> None:
    """Idempotent additive schema migration; never drops user data."""
    with conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
            user_input TEXT, ai_response TEXT, intent TEXT,
            sentiment_score REAL DEFAULT 0.0, emotional_state TEXT, session_id TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS interaction_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
            word_count INTEGER, sentiment_score REAL, emotional_intensity REAL,
            response_quality REAL, mode TEXT, session_id TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            preference_key TEXT UNIQUE NOT NULL, preference_value TEXT,
            confidence REAL DEFAULT 0.5, last_updated TEXT)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS adaptive_scoped_preferences (
            scope_id TEXT NOT NULL, preference_key TEXT NOT NULL,
            preference_value TEXT, confidence REAL NOT NULL,
            last_updated TEXT, PRIMARY KEY(scope_id, preference_key))""")
        compat = {
            "conversations": {"timestamp":"TEXT", "user_input":"TEXT", "ai_response":"TEXT",
                "intent":"TEXT", "sentiment_score":"REAL", "emotional_state":"TEXT", "session_id":"TEXT"},
            "interaction_metrics": {"timestamp":"TEXT", "word_count":"INTEGER", "sentiment_score":"REAL",
                "emotional_intensity":"REAL", "response_quality":"REAL", "mode":"TEXT", "session_id":"TEXT"},
            "user_preferences": {"preference_key":"TEXT", "preference_value":"TEXT", "confidence":"REAL", "last_updated":"TEXT"},
        }
        for table, spec in compat.items():
            columns = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            for column, kind in spec.items():
                if column not in columns:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id, timestamp)")



def connect_memory_db() -> Optional[sqlite3.Connection]:
    """Open memory database, initializing schema once per on-disk database identity."""
    conn = None
    try:
        os.makedirs(os.path.dirname(MEMORY_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(MEMORY_DB_PATH, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        with _SCHEMA_LOCK:
            identity = _db_file_identity(MEMORY_DB_PATH)
            if _SCHEMA_READY.get("memory:" + MEMORY_DB_PATH) != identity:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                _ensure_memory_db_schema(conn)
                _SCHEMA_READY["memory:" + MEMORY_DB_PATH] = _db_file_identity(MEMORY_DB_PATH)
        return conn
    except (OSError, sqlite3.Error) as exc:
        logger.error("Memory database unavailable: %s", type(exc).__name__)
        if conn is not None:
            conn.close()
        return None



def log_interaction_to_db(
    user_input: str,
    ai_response: str,
    intent: str = "unknown",
    sentiment_score: float = 0.0,
    emotional_state: Dict = None,
    session_id: str = None
) -> bool:
    """Record bounded, optionally redacted interaction data in one transaction."""
    if not bool(getattr(config, "ADAPTIVE_ENABLE_INTERACTION_LOGGING", True)) or not _scope_allowed():
        return False
    conn = connect_memory_db()
    if conn is None:
        return False
    try:
        scoped = _scope_key()
        # A trusted active scope always overrides untrusted supplied session IDs.
        effective_session = scoped or (str(session_id)[:128] if session_id else None)
        raw = _adaptive_can_store_raw()
        timestamp = _get_timestamp()
        emotional_json = json.dumps(_validated_state(emotional_state), allow_nan=False) if emotional_state else None
        with closing(conn), conn:
            conn.execute("""INSERT INTO conversations
                (timestamp, user_input, ai_response, intent, sentiment_score, emotional_state, session_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (timestamp,
                 _adaptive_storage_text(user_input) if raw else None,
                 _adaptive_storage_text(ai_response) if raw else None,
                 str(intent or "unknown")[:80],
                 _safe_float(sentiment_score), emotional_json, effective_session))
        return True
    except (TypeError, ValueError, sqlite3.Error) as exc:
        logger.warning("Adaptive interaction log failed: %s", type(exc).__name__)
        return False
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass



def get_recent_interactions(limit: int = 20) -> List[Dict]:
    """Return bounded history, scoped when an authenticated adaptive scope exists."""
    if not _scope_allowed():
        return []
    conn = connect_memory_db()
    if conn is None:
        return []
    try:
        bounded_limit = max(0, min(100, int(limit)))
        scope = _scope_key()
        with closing(conn):
            if scope:
                rows = conn.execute("""SELECT timestamp, user_input, ai_response, intent, sentiment_score
                    FROM conversations WHERE session_id = ? ORDER BY timestamp DESC, rowid DESC LIMIT ?""",
                    (scope, bounded_limit)).fetchall()
            else:
                rows = conn.execute("""SELECT timestamp, user_input, ai_response, intent, sentiment_score
                    FROM conversations WHERE session_id IS NULL ORDER BY timestamp DESC, rowid DESC LIMIT ?""",
                    (bounded_limit,)).fetchall()
        return [dict(row) for row in rows]
    except (ValueError, TypeError, sqlite3.Error) as exc:
        logger.warning("Adaptive interaction lookup failed: %s", type(exc).__name__)
        return []
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass



def get_interaction_count() -> int:
    """Count only the current session's records (legacy unscoped otherwise)."""
    if not _scope_allowed():
        return 0
    conn = connect_memory_db()
    if conn is None:
        return 0
    try:
        with closing(conn):
            if _scope_key():
                return int(conn.execute("SELECT COUNT(*) FROM conversations WHERE session_id = ?", (_scope_key(),)).fetchone()[0])
            return int(conn.execute("SELECT COUNT(*) FROM conversations WHERE session_id IS NULL").fetchone()[0])
    except sqlite3.Error as exc:
        logger.warning("Adaptive interaction count failed: %s", type(exc).__name__)
        return 0
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass



# ============================================================================
# DATABASE OPERATIONS - EMOTION DB (personality1.db)
# ============================================================================

def _ensure_emotion_db_schema(conn: sqlite3.Connection) -> None:
    """Additive legacy-safe schema repair. Unique constraints are never presumed."""
    with conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS traits (
            id INTEGER PRIMARY KEY AUTOINCREMENT, trait_name TEXT UNIQUE,
            description TEXT, last_updated TEXT, ts TEXT, trait TEXT, value REAL, source TEXT)""")
        compat = {"trait_name":"TEXT", "description":"TEXT", "last_updated":"TEXT", "ts":"TEXT",
                  "trait":"TEXT", "value":"REAL", "source":"TEXT"}
        cols = {r[1] for r in conn.execute("PRAGMA table_info(traits)")}
        for name, kind in compat.items():
            if name not in cols:
                conn.execute(f"ALTER TABLE traits ADD COLUMN {name} {kind}")
        conn.execute("""CREATE TABLE IF NOT EXISTS emotional_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
            emotion_snapshot TEXT, dominant_emotion TEXT, emotional_balance REAL,
            trigger_text TEXT, session_id TEXT)""")
        history_columns = {r[1] for r in conn.execute("PRAGMA table_info(emotional_history)")}
        if "session_id" not in history_columns:
            conn.execute("ALTER TABLE emotional_history ADD COLUMN session_id TEXT")
        conn.execute("""CREATE TABLE IF NOT EXISTS adaptive_scoped_state (
            scope_id TEXT PRIMARY KEY, state_json TEXT NOT NULL, last_updated TEXT)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_emotional_history_scope ON emotional_history(session_id, timestamp)")
        # Existing view or table is left untouched, never replaced.
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE name='emotion_states'").fetchone()
        if not exists:
            conn.execute("""CREATE VIEW emotion_states AS
                SELECT trait_name AS name, description AS value FROM traits""")




def connect_emotion_db() -> Optional[sqlite3.Connection]:
    """Open emotional persistence and initialize schema only as needed."""
    conn = None
    try:
        os.makedirs(os.path.dirname(EMOTION_DB_PATH), exist_ok=True)
        conn = sqlite3.connect(EMOTION_DB_PATH, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=5000")
        with _SCHEMA_LOCK:
            identity = _db_file_identity(EMOTION_DB_PATH)
            if _SCHEMA_READY.get("emotion:" + EMOTION_DB_PATH) != identity:
                conn.execute("PRAGMA journal_mode=WAL")
                _ensure_emotion_db_schema(conn)
                _SCHEMA_READY["emotion:" + EMOTION_DB_PATH] = _db_file_identity(EMOTION_DB_PATH)
        return conn
    except (OSError, sqlite3.Error) as exc:
        logger.error("Emotion database unavailable: %s", type(exc).__name__)
        if conn is not None:
            conn.close()
        return None



def load_emotional_state() -> Dict[str, Any]:
    """Load only the current trusted scope's state; legacy default remains local."""
    if not _scope_allowed():
        return copy.deepcopy(DEFAULT_EMOTIONS)
    conn = connect_emotion_db()
    with _state_lock:
        if conn is None:
            return copy.deepcopy(_current_state())
        try:
            with closing(conn):
                if _scope_key():
                    row = conn.execute("SELECT state_json FROM adaptive_scoped_state WHERE scope_id=?", (_scope_key(),)).fetchone()
                    if row is None:
                        state = copy.deepcopy(DEFAULT_EMOTIONS)
                    else:
                        try:
                            state = _validated_state(json.loads(row[0]))
                        except (ValueError, TypeError):
                            logger.warning("Corrupt scoped emotional state; using defaults")
                            state = copy.deepcopy(DEFAULT_EMOTIONS)
                else:
                    state = copy.deepcopy(DEFAULT_EMOTIONS)
                    for row in conn.execute("SELECT trait_name, description FROM traits ORDER BY rowid ASC"):
                        name, value = row[0], row[1]
                        if name in _STATE_FIELDS:
                            if name == "adjustments":
                                try:
                                    state[name] = json.loads(value or "[]")
                                except (ValueError, TypeError):
                                    state[name] = []
                            else:
                                state[name] = value
                    state = _validated_state(state)
            _replace_current_state(state)
            return copy.deepcopy(state)
        except sqlite3.Error as exc:
            logger.warning("Loading emotional state failed: %s", type(exc).__name__)
            return copy.deepcopy(_current_state())
        finally:
            try:
                conn.close()
            except sqlite3.Error:
                pass



def save_emotional_state(state: Dict[str, Any] = None) -> bool:
    """Atomic, validated save of only known adaptive traits in active scope."""
    if not _scope_allowed():
        return False
    with _state_lock:
        state_to_save = _validated_state(_current_state() if state is None else state)
    conn = connect_emotion_db()
    if conn is None:
        return False
    try:
        timestamp = _get_timestamp()
        with closing(conn), conn:
            if _scope_key():
                conn.execute("""INSERT INTO adaptive_scoped_state(scope_id,state_json,last_updated)
                    VALUES(?,?,?) ON CONFLICT(scope_id) DO UPDATE SET
                    state_json=excluded.state_json,last_updated=excluded.last_updated""",
                    (_scope_key(), json.dumps(state_to_save, allow_nan=False), timestamp))
            else:
                # UPDATE/INSERT works even on legacy tables without UNIQUE(trait_name).
                for trait_name, value in state_to_save.items():
                    serialized = json.dumps(value) if isinstance(value, list) else str(value)
                    numeric = _safe_float(value, 0.0, -1e6, 1e6) if isinstance(value, (float, int)) else 0.0
                    result = conn.execute("""UPDATE traits SET description=?, last_updated=?,
                        ts=?, trait=?, value=?, source=? WHERE trait_name=?""",
                        (serialized, timestamp, timestamp, trait_name, numeric,
                         "SarahMemoryAdaptive", trait_name))
                    if result.rowcount == 0:
                        # Legacy traits may require non-null ts/trait/value/source.
                        conn.execute("""INSERT INTO traits
                            (trait_name,description,last_updated,ts,trait,value,source)
                            VALUES(?,?,?,?,?,?,?)""",
                            (trait_name, serialized, timestamp, timestamp,
                             trait_name, numeric, "SarahMemoryAdaptive"))
        return True
    except (sqlite3.Error, TypeError, ValueError) as exc:
        logger.error("Saving emotional state failed: %s", type(exc).__name__)
        return False
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass



def log_emotional_history(emotion_snapshot: Dict, trigger_text: str = None) -> bool:
    """Write bounded emotional history; raw triggers are owner-configurable."""
    if not _scope_allowed() or not bool(getattr(config, "ADAPTIVE_ENABLE_INTERACTION_LOGGING", True)) or not bool(getattr(config, "ADAPTIVE_ENABLE_EMOTIONAL_HISTORY", True)):
        return False
    conn = connect_emotion_db()
    if conn is None:
        return False
    try:
        state = _validated_state(emotion_snapshot)
        dominant = max(_EMOTION_FIELDS, key=lambda x: state[x])
        with closing(conn), conn:
            conn.execute("""INSERT INTO emotional_history
                (timestamp,emotion_snapshot,dominant_emotion,emotional_balance,trigger_text,session_id)
                VALUES (?,?,?,?,?,?)""",
                (_get_timestamp(), json.dumps(state, allow_nan=False), dominant,
                 state["emotional_balance"],
                 _adaptive_storage_text(trigger_text)[:256] if _adaptive_can_store_raw() and trigger_text else None,
                 _scope_key() or None))
        return True
    except (sqlite3.Error, TypeError, ValueError) as exc:
        logger.error("Emotional history unavailable: %s", type(exc).__name__)
        return False
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass



# ============================================================================
# SENTIMENT ANALYSIS ENGINE
# ============================================================================

def analyze_sentiment(text: str) -> InteractionMetrics:
    """Deterministic bounded lexicon estimate, not a psychological diagnosis.

    Negation is applied locally to nearby sentiment tokens rather than reversing
    the polarity of an entire message. A standalone `no` is not negative affect.
    """
    words = _tokenize(text)
    if not words:
        return InteractionMetrics(timestamp=_get_timestamp())
    negation_words = {"not", "no", "never", "don't", "doesn't", "didn't", "won't", "can't", "couldn't",
                      "isn't", "aren't", "wasn't", "weren't", "cannot", "without", "hardly"}
    positive = 0.0
    negative = 0.0
    intensifiers = {"very", "extremely", "really", "so", "absolutely", "totally", "completely"}
    for i, word in enumerate(words):
        if word in negation_words or word in {"okay", "fine"}:
            continue
        pos = word in POSITIVE_KEYWORDS
        neg = word in NEGATIVE_KEYWORDS or any(word.startswith(root) for root in ("frustrat", "irritat", "outrag"))
        if not (pos or neg):
            continue
        window = words[max(0, i - 3):i]
        # Double negation resolves to ordinary polarity.
        flipped = sum(w in negation_words for w in window) % 2 == 1
        weight = 1.25 if any(w in intensifiers for w in window) else 1.0
        if (pos and not flipped) or (neg and flipped):
            positive += weight
        else:
            negative += weight
    raw = positive - negative
    score = _clamp(raw * 2.0 / max(1, len(words)), -1.0, 1.0)
    surprise = any(w in SURPRISE_INDICATORS for w in words)
    question = (isinstance(text, str) and text.rstrip().endswith("?")) or any(w in QUESTION_INDICATORS for w in words[:3])
    intensity = _clamp((positive + negative + sum(w in SURPRISE_INDICATORS for w in words)) * 3.0 / len(words))
    return InteractionMetrics(sentiment_score=round(score, 4), word_count=len(words),
                              question_detected=question, surprise_detected=surprise,
                              emotional_intensity=round(intensity, 4), response_quality=0.5,
                              timestamp=_get_timestamp())





# ---------------------------------------------------------------------------
# Hardware tier snapshot (delegates to SarahMemoryGlobals.hardware_score)
# - Does NOT load any 3rd party models. Pure telemetry for adaptive throttling.
# ---------------------------------------------------------------------------
def _get_hardware_tier_snapshot() -> Dict[str, Any]:
    """Return {'score': float|None, 'tier_rating': str|None, 'tier': str|None}.

    Best-effort: if Globals does not provide hardware_score(), returns empty.
    """
    try:
        hs_fn = getattr(config, "hardware_score", None)
        if callable(hs_fn):
            hs = hs_fn() or {}
            return {
                "score": hs.get("score", None),
                "tier_rating": hs.get("tier_rating", None),
                "tier": hs.get("tier", None),
            }
    except Exception:
        pass
    return {}
def determine_system_mode(cpu: float, memory: float) -> SystemMode:
    """Safe deterministic resource mode; missing/invalid metrics fail conservative."""
    if getattr(config, "SAFE_MODE", False):
        return SystemMode.SAFE
    if not all(isinstance(x, (float, int)) and math.isfinite(x) and 0 <= x <= 100 for x in (cpu, memory)):
        return SystemMode.LIGHTWEIGHT
    snap = _get_hardware_tier_snapshot()
    tier_rating = str(snap.get("tier_rating") or "").strip().lower()
    if tier_rating == "poor":
        return SystemMode.BALANCED if cpu < 20 and memory < 40 else SystemMode.LIGHTWEIGHT
    if cpu > 80 or memory > 85:
        return SystemMode.LIGHTWEIGHT
    if cpu < 50 and memory < 60:
        return SystemMode.ENHANCED
    return SystemMode.BALANCED





# ============================================================================
# EMOTIONAL LEARNING ENGINE
# ============================================================================

def advanced_emotional_learning(user_input: str) -> Dict[str, Any]:
    """Bounded deterministic tone observation and resource-adaptive presentation.

    Outputs influence presentation and cadence only. Not permission, identity,
    trust level, or a claim of true sentient emotions.
    """
    if not _scope_allowed():
        raise PermissionError("adaptive request scope required")
    with _state_lock:
        state = _current_state()
        cpu, memory = _get_system_resources()
        state["cpu"], state["memory"] = cpu, memory
        mode = determine_system_mode(cpu, memory)
        old_mode = state.get("mode", "balanced")
        now_monotonic = time.monotonic()
        # Hysteresis prevents rapid UI mode oscillation and per-request log storms.
        # Safety escalation and pressure recovery are never delayed.
        last_change = _safe_float(state.get("_mode_change_monotonic", 0.0), 0.0, 0.0, 1e15)
        if (mode.value != old_mode and mode in (SystemMode.BALANCED, SystemMode.ENHANCED)
                and old_mode in (SystemMode.BALANCED.value, SystemMode.ENHANCED.value,
                                 SystemMode.LIGHTWEIGHT.value)
                and now_monotonic - last_change < 10.0):
            mode = SystemMode(old_mode)
        state["mode"] = mode.value
        if old_mode != mode.value:
            state["_mode_change_monotonic"] = now_monotonic
            changes = state.setdefault("adjustments", [])
            changes.append(f"Mode changed: {old_mode} -> {mode.value} at {_get_timestamp()}")
            del changes[:-10]
            logger.info("[MODE] %s -> %s", old_mode, mode.value)
        metrics = analyze_sentiment(user_input)
        balance = metrics.sentiment_score
        # Resource pressure must NOT change meaning/sentiment.
        momentum = _momentum_buffer()
        momentum.append(balance)
        weighted_sum, weight_sum = 0.0, 0.0
        for i, value in enumerate(reversed(momentum)):
            weight = 0.5 ** i
            weighted_sum += value * weight
            weight_sum += weight
        emotional_balance = _clamp(weighted_sum / max(weight_sum, 1e-12), -1.0, 1.0)
        openness = _clamp(0.6 + metrics.sentiment_score * 0.08, 0.3, 0.95)
        engagement = _clamp(0.4 + 0.3 * min(metrics.word_count / 30.0, 1.0)
                            + 0.2 * metrics.emotional_intensity, 0.2, 1.0)
        # Conserve compute, do not pretend the user's engagement changed.
        if mode in (SystemMode.LIGHTWEIGHT, SystemMode.SAFE):
            engagement = min(engagement, 0.7)
        state.update({"emotional_balance": round(emotional_balance, 4),
                      "openness": round(openness, 4), "engagement": round(engagement, 4),
                      "last_updated": _get_timestamp()})
        _update_emotions_from_sentiment(metrics)
        packet = {"emotional_balance": state["emotional_balance"], "openness": state["openness"],
                  "engagement": state["engagement"], "mode": mode.value,
                  "sentiment_score": metrics.sentiment_score, "word_count": metrics.word_count}
        # The ephemeral nested packet is not persisted as user traits.
        state["emotion"] = copy.deepcopy(packet)
        return packet



def _update_emotions_from_sentiment(metrics: InteractionMetrics) -> None:
    """Smooth display affect toward baseline; never increase authorization/trust."""
    state = _current_state()
    sentiment = _safe_float(metrics.sentiment_score)
    intensity = _safe_float(metrics.emotional_intensity, 0.0, 0.0, 1.0)
    base = DEFAULT_EMOTIONS
    if sentiment > 0.1:
        state["joy"] = _clamp(state.get("joy", 0.5) + sentiment * intensity * 0.15)
        state["trust"] = _clamp(state.get("trust", 0.3) + sentiment * intensity * 0.10)
        state["anticipation"] = _clamp(state.get("anticipation", 0.3) + sentiment * 0.08)
        for field_name, amount in (("anger", 0.10), ("sadness", 0.08), ("fear", 0.05)):
            state[field_name] = _clamp(state.get(field_name, base[field_name]) - sentiment * amount)
    elif sentiment < -0.1:
        strength = abs(sentiment)
        for field_name, amount in (("anger", 0.20), ("sadness", 0.15), ("fear", 0.10)):
            state[field_name] = _clamp(state.get(field_name, base[field_name]) + strength * intensity * amount)
        state["joy"] = _clamp(state.get("joy", 0.5) - strength * 0.15)
        state["trust"] = _clamp(state.get("trust", 0.3) - strength * 0.10)
    if metrics.surprise_detected:
        state["surprise"] = _clamp(state.get("surprise", 0.1) + 0.25)
    # Stable, interaction-bounded decay: no autonomous loops or sleep delays.
    for name in _EMOTION_FIELDS:
        current = _safe_float(state.get(name), base[name], 0.0, 1.0)
        state[name] = round(_clamp(current + (base[name] - current) * 0.05), 6)



def simulate_emotion_response(input_type: str = "positive") -> Dict[str, Any]:
    """Explicit simulation for presentation/tests; persists to active scope only."""
    if not _scope_allowed():
        raise PermissionError("adaptive request scope required")
    if input_type not in {"positive", "negative", "neutral"}:
        logger.warning("Unknown emotion simulation type")
        return get_current_emotional_state()
    with _state_lock:
        state = _validated_state(load_emotional_state())
        if input_type == "positive":
            for name, strength in (("joy", 0.15), ("trust", 0.10), ("anticipation", 0.08)):
                state[name] = _clamp(state[name] + (1 - state[name]) * strength)
            for name, factor in (("anger", 0.90), ("fear", 0.92), ("sadness", 0.90)):
                state[name] = _clamp(state[name] * factor)
            state["emotional_balance"] = _clamp(state["emotional_balance"] + 0.1, -1, 1)
        elif input_type == "negative":
            for name, factor in (("anger", 0.25), ("fear", 0.20), ("sadness", 0.18)):
                state[name] = _clamp(state[name] + (1 - state[name]) * factor)
            for name, factor in (("joy", 0.85), ("trust", 0.90)):
                state[name] = _clamp(state[name] * factor)
            state["emotional_balance"] = _clamp(state["emotional_balance"] - 0.15, -1, 1)
        else:
            state["surprise"] = _clamp(state["surprise"] + (1 - state["surprise"]) * 0.05)
            for name in ("joy", "anger", "fear", "sadness", "trust"):
                state[name] = _clamp(state[name] + (DEFAULT_EMOTIONS[name] - state[name]) * 0.05)
        state["last_updated"] = _get_timestamp()
        _replace_current_state(state)
        snapshot = copy.deepcopy(state)
    if not save_emotional_state(snapshot):
        logger.warning("Emotion simulation not persisted")
    log_emotional_history(snapshot, "simulation:" + input_type)
    return snapshot



# ============================================================================
# PERSONALITY UPDATE SYSTEM
# ============================================================================

def update_personality(user_input: str, ai_response: str) -> Dict[str, Any]:
    """Observed adaptive signals only; feedback is learned explicitly elsewhere."""
    metrics = advanced_emotional_learning(user_input)
    sentiment = _safe_float(metrics.get("sentiment_score"))
    snapshot = get_current_emotional_state()
    saved = log_interaction_to_db(user_input, ai_response, sentiment_score=sentiment,
                                  emotional_state=snapshot, session_id=_scope_key() or None)
    with _state_lock:
        _interaction_history_buffer().append({
            "timestamp": _get_timestamp(), "sentiment": sentiment,
            "word_count": int(metrics.get("word_count", 0)),
            "metrics": copy.deepcopy(metrics)})
    if not save_emotional_state(snapshot):
        logger.warning("Adaptive emotional snapshot not persisted")
    log_emotional_history(snapshot, user_input)
    if not saved and getattr(config, "ADAPTIVE_ENABLE_INTERACTION_LOGGING", True):
        logger.warning("Adaptive interaction log not persisted")
    return metrics



def self_update_personality() -> Dict[str, Any]:
    """Evaluate bounded historical signals without silently mutating personality.

    User sentiment or total conversation count does not constitute feedback or
    permission to change durable personality preferences.
    """
    try:
        total = get_interaction_count()
        recent = get_recent_interactions(20)
        sentiments = [_safe_float(row.get("sentiment_score")) for row in recent]
        average = sum(sentiments) / len(sentiments) if sentiments else 0.0
        with _state_lock:
            engagement = _safe_float(_current_state().get("engagement"), 0.4, 0.0, 1.0)
        feedback = get_adaptive_feedback_summary()
        return {"status": "ok", "engagement": round(engagement, 4),
                "total_interactions": total, "avg_sentiment": round(average, 4),
                "explicit_feedback_count": feedback.get("count", 0),
                "average_response_quality": feedback.get("average_quality"),
                "preferences_mutated": False, "execution_authority": False}
    except Exception as exc:
        logger.warning("Adaptive assessment failed: %s", type(exc).__name__)
        return {"status": "error", "engagement": 0.0, "error": type(exc).__name__,
                "execution_authority": False}



# ============================================================================
# HELPER FUNCTIONS FOR EXTERNAL MODULES
# ============================================================================

def emotion_label_from_state(state: Dict = None) -> str:
    """
    Get a human-readable emotion label from the current state.
    
    Args:
        state: State dictionary (uses global STATE if None)
    
    Returns:
        String label: "angry", "happy", "sad", "surprised", "trusting", "neutral"
    """
    s = state if isinstance(state, dict) else get_current_emotional_state()
    
    try:
        if s.get("anger", 0) >= 0.60:
            return "angry"
        if s.get("joy", 0) >= 0.60:
            return "happy"
        if s.get("fear", 0) >= 0.55 or s.get("sadness", 0) >= 0.55:
            return "sad"
        if s.get("surprise", 0) >= 0.55:
            return "surprised"
        if s.get("trust", 0) >= 0.60:
            return "trusting"
        if s.get("anticipation", 0) >= 0.55:
            return "eager"
        
        # Check emotional balance
        balance = s.get("emotional_balance", 0)
        if balance > 0.3:
            return "positive"
        elif balance < -0.3:
            return "negative"
    except Exception:
        pass
    
    return "neutral"


def get_current_emotional_state() -> Dict[str, Any]:
    """Deep defensive copy of the active scope's adaptive runtime state."""
    if not _scope_allowed():
        return copy.deepcopy(DEFAULT_EMOTIONS)
    with _state_lock:
        return copy.deepcopy(_current_state())



def get_emotional_metrics() -> Dict[str, Any]:
    """Compatibility display metrics with state-bound safe finite values."""
    state = get_current_emotional_state()
    emotions = {name: _safe_float(state.get(name), DEFAULT_EMOTIONS[name], 0.0, 1.0)
                for name in _EMOTION_FIELDS}
    dominant_name = max(emotions, key=emotions.get)
    return {"emotional_balance": _safe_float(state.get("emotional_balance")),
            "openness": _safe_float(state.get("openness"), 0.6, 0, 1),
            "engagement": _safe_float(state.get("engagement"), 0.4, 0, 1),
            "dominant_emotion": dominant_name, "intensity": emotions[dominant_name],
            "mode": state.get("mode", "balanced"), "label": emotion_label_from_state(state)}



# ---------------------------------------------------------------------------
# RHYTHM COGNITION BRIDGE - v9.0.0
# ---------------------------------------------------------------------------
def get_adaptive_rhythm_basis(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read-only rhythm basis: tempo evidence, never urgency-based authority.

    Caller-supplied trusted emergency flags are distinct from lexical urgency.
    An authenticated request scope must be active for multi-user state.
    """
    ctx = dict(context or {}) if isinstance(context, dict) else {}
    metrics = get_emotional_metrics()
    state = get_current_emotional_state()
    dominant = str(metrics.get("dominant_emotion") or "neutral")
    intensity = _safe_float(metrics.get("intensity"), 0.0, 0.0, 1.0)
    text = str(ctx.get("text") or ctx.get("user_text") or ctx.get("command_text") or "")
    words = set(_tokenize(text))
    normalized = text.strip().lower()
    educational = normalized.startswith(("what is ", "what are ", "explain ",
                                          "define ", "describe ", "how does ",
                                          "tell me about "))
    explicit_urgency = {"urgent", "asap", "immediately", "emergency", "hurry"}
    hazard_terms = {"danger", "fire", "collision"}
    urgency = 0.68 if words.intersection(explicit_urgency) and not educational else 0.0
    if not educational and words.intersection(hazard_terms):
        urgency = max(urgency, 0.55)
    # Do not infer a real emergency from emotional presentation labels.
    if any(ctx.get(name) is True for name in ("emergency", "human_risk", "person_at_risk")):
        urgency = max(urgency, 0.92)
    cpu, memory = _get_system_resources()
    pressure = max(cpu, memory) if min(cpu, memory) >= 0 else -1
    pressure_tier = ("unknown" if pressure < 0 else "critical" if pressure >= 88 else
                     "high" if pressure >= 75 else "moderate" if pressure >= 55 else "normal")
    resource = {"cpu": cpu if cpu >= 0 else None, "memory": memory if memory >= 0 else None,
                "pressure_tier": pressure_tier}
    return {"source": "SarahMemoryAdaptive.get_adaptive_rhythm_basis",
            "dominant_emotion": dominant, "label": metrics.get("label") or dominant,
            "emotional_balance": metrics.get("emotional_balance", state.get("emotional_balance", 0.0)),
            "openness": metrics.get("openness", state.get("openness", 0.6)),
            "engagement": metrics.get("engagement", state.get("engagement", 0.4)),
            "intensity": intensity, "adaptive_mode": metrics.get("mode", state.get("mode", "balanced")),
            "urgency_score": round(urgency, 4), "resource": resource,
            "execution_authority": False,
            "notes": ["Adaptive presentation informs cadence only.",
                      "OperatorCore, AssuranceGate and safety governance remain required for actions."]}



def reset_emotional_state() -> Dict[str, Any]:
    """Reset only the active scope's presentation state, with explicit persistence."""
    if not _scope_allowed():
        raise PermissionError("adaptive request scope required")
    with _state_lock:
        state = copy.deepcopy(DEFAULT_EMOTIONS)
        state["last_updated"] = _get_timestamp()
        state["adjustments"] = ["Reset to defaults at " + state["last_updated"]]
        _replace_current_state(state)
        _momentum_buffer().clear()
        _interaction_history_buffer().clear()
    if not save_emotional_state(state):
        logger.warning("Adaptive reset persisted only in memory")
    return copy.deepcopy(state)



# ============================================================================
# USER PREFERENCE LEARNING
# ============================================================================

def learn_user_preference(key: str, value: str, confidence: float = 0.5) -> bool:
    """Persist an EXPLICIT supplied preference; infer nothing from sentiment."""
    if not _scope_allowed() or not isinstance(key, str) or not key.strip() or len(key) > 128 or not isinstance(value, str) or len(value) > 2048:
        return False
    confidence = _safe_float(confidence, 0.5, 0.0, 1.0)
    conn = connect_memory_db()
    if conn is None:
        return False
    try:
        with closing(conn), conn:
            if _scope_key():
                conn.execute("""INSERT INTO adaptive_scoped_preferences
                    (scope_id,preference_key,preference_value,confidence,last_updated)
                    VALUES(?,?,?,?,?) ON CONFLICT(scope_id,preference_key) DO UPDATE SET
                    preference_value=excluded.preference_value,
                    confidence=CASE WHEN preference_value=excluded.preference_value THEN
                        MAX(confidence,excluded.confidence) ELSE excluded.confidence END,
                    last_updated=excluded.last_updated""",
                    (_scope_key(), key, value, confidence, _get_timestamp()))
            else:
                # Legacy schema may not enforce unique preference_key.
                existing = conn.execute("SELECT preference_value,confidence FROM user_preferences WHERE preference_key=? LIMIT 1", (key,)).fetchone()
                if existing is not None:
                    certainty = max(_safe_float(existing[1], 0.0, 0, 1), confidence) if existing[0] == value else confidence
                    conn.execute("""UPDATE user_preferences SET preference_value=?,confidence=?,last_updated=? WHERE preference_key=?""",
                                 (value, certainty, _get_timestamp(), key))
                else:
                    conn.execute("""INSERT INTO user_preferences
                        (preference_key,preference_value,confidence,last_updated) VALUES(?,?,?,?)""",
                        (key, value, confidence, _get_timestamp()))
        return True
    except sqlite3.Error as exc:
        logger.warning("Preference update failed: %s", type(exc).__name__)
        return False
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass



def get_user_preference(key: str, default: str = None) -> Optional[str]:
    """Look up a preference exclusively within the current trust scope."""
    if not _scope_allowed():
        return default
    conn = connect_memory_db()
    if conn is None:
        return default
    try:
        with closing(conn):
            if _scope_key():
                row = conn.execute("""SELECT preference_value FROM adaptive_scoped_preferences
                    WHERE scope_id=? AND preference_key=?""", (_scope_key(), key)).fetchone()
            else:
                row = conn.execute("SELECT preference_value FROM user_preferences WHERE preference_key=?", (key,)).fetchone()
        return row[0] if row else default
    except sqlite3.Error:
        return default
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass



def get_all_preferences() -> Dict[str, Any]:
    """Return bounded preferences of the current trust scope only."""
    if not _scope_allowed():
        return {}
    conn = connect_memory_db()
    if conn is None:
        return {}
    try:
        with closing(conn):
            if _scope_key():
                rows = conn.execute("""SELECT preference_key,preference_value,confidence FROM adaptive_scoped_preferences
                    WHERE scope_id=? ORDER BY preference_key LIMIT 500""", (_scope_key(),)).fetchall()
            else:
                rows = conn.execute("SELECT preference_key,preference_value,confidence FROM user_preferences ORDER BY preference_key LIMIT 500").fetchall()
        return {row[0]: {"value": row[1], "confidence": _safe_float(row[2], 0.5, 0, 1)} for row in rows}
    except sqlite3.Error:
        return {}
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass





def record_adaptive_feedback(quality: float, *, source: str = "user", note: str = "") -> Dict[str, Any]:
    """Record authenticated explicit feedback; no automatic identity mutation.

    API callers are responsible for verifying that `source` represents an
    authenticated owner/user. Returns only a receipt and bounded quality.
    """
    if not _scope_allowed() or source != "user":
        return {"ok": False, "reason": "feedback_requires_verified_user", "execution_authority": False}
    if not isinstance(quality, (float, int)) or not math.isfinite(quality) or not 0 <= quality <= 1:
        return {"ok": False, "reason": "invalid_feedback_score", "execution_authority": False}
    conn = connect_memory_db()
    if conn is None:
        return {"ok": False, "reason": "persistence_unavailable", "execution_authority": False}
    try:
        with closing(conn), conn:
            cursor = conn.execute("""INSERT INTO interaction_metrics
                (timestamp,word_count,sentiment_score,emotional_intensity,response_quality,mode,session_id)
                VALUES(?,?,?,?,?,?,?)""",
                (_get_timestamp(), 0, 0.0, 0.0, round(quality, 4), "explicit_user_feedback", _scope_key() or None))
            receipt_id = int(cursor.lastrowid)
        return {"ok": True, "quality": round(quality, 4), "feedback_id": receipt_id,
                "scope_bound": bool(_scope_key()),
                "preferences_mutated": False, "execution_authority": False}
    except sqlite3.Error:
        return {"ok": False, "reason": "persistence_error", "execution_authority": False}
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


# ============================================================================
# GCAIOS COMPUTE PASSPORT AND ADAPTIVE COMPUTE FABRIC
# ============================================================================

def get_adaptive_response_profile(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Explainable advisory style selection, from explicit persisted preferences.

    No inferred sentiment is ever promoted into durable user identity. Time-of-
    day suggestions require an explicit system setting and never affect safety.
    """
    if not _scope_allowed():
        return {"ok": False, "reason": "scope_required", "execution_authority": False}
    ctx = dict(context or {}) if isinstance(context, dict) else {}
    prefs = get_all_preferences()
    defaults = {"verbosity": "balanced", "formality": "neutral", "humor_level": "moderate"}
    approved = {
        "verbosity": {"brief", "balanced", "detailed"},
        "formality": {"casual", "neutral", "formal"},
        "humor_level": {"none", "low", "moderate", "high"},
    }
    style = {}
    confidence = {}
    for key, fallback in defaults.items():
        entry = prefs.get(key, {})
        value = str(entry.get("value", fallback)).strip().lower()
        style[key] = value if value in approved[key] else fallback
        confidence[key] = _safe_float(entry.get("confidence", 0.0), 0.0, 0.0, 1.0) if value in approved[key] else 0.0
    hr = datetime.datetime.now().hour
    daypart = "morning" if 5 <= hr < 12 else "afternoon" if 12 <= hr < 17 else "evening" if 17 <= hr < 22 else "night"
    if getattr(config, "ADAPTIVE_TIME_OF_DAY_ENABLED", False):
        # Changes only *suggested* presentation energy, not scope or authority.
        energy = "soft" if daypart == "night" else "normal"
        time_adjustment = True
    else:
        energy = "normal"
        time_adjustment = False
    state = get_current_emotional_state()
    return {"ok": True, "style": style, "confidence": confidence,
            "daypart": daypart, "time_adjustment_enabled": time_adjustment,
            "presentation_energy": energy, "adaptive_mode": state.get("mode", "balanced"),
            "source": "explicit_preferences_or_defaults",
            "scope_bound": bool(_scope_key()), "execution_authority": False}


def get_adaptive_feedback_summary(limit: int = 20) -> Dict[str, Any]:
    """Aggregate explicit quality feedback; never confuse sentiment with reward."""
    if not _scope_allowed():
        return {"ok": False, "reason": "scope_required", "execution_authority": False}
    conn = connect_memory_db()
    if conn is None:
        return {"ok": False, "reason": "database_unavailable", "execution_authority": False}
    try:
        cap = max(1, min(100, int(limit)))
        with closing(conn):
            if _scope_key():
                rows = conn.execute("""SELECT response_quality FROM interaction_metrics
                    WHERE mode='explicit_user_feedback' AND session_id=? ORDER BY rowid DESC LIMIT ?""",
                    (_scope_key(), cap)).fetchall()
            else:
                rows = conn.execute("""SELECT response_quality FROM interaction_metrics
                    WHERE mode='explicit_user_feedback' AND session_id IS NULL ORDER BY rowid DESC LIMIT ?""",
                    (cap,)).fetchall()
        qualities = [_safe_float(row[0], 0.0, 0.0, 1.0) for row in rows]
        return {"ok": True, "count": len(qualities),
                "average_quality": round(sum(qualities)/len(qualities), 4) if qualities else None,
                "last_quality": qualities[0] if qualities else None,
                "explicit_user_feedback_only": True, "execution_authority": False}
    except (ValueError, sqlite3.Error) as exc:
        return {"ok": False, "reason": type(exc).__name__, "execution_authority": False}
    finally:
        try:
            conn.close()
        except sqlite3.Error:
            pass


def _compute_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _compute_int(value: Any, default: int = 0) -> int:
    try:
        number = float(value)
        return max(0, min(4096, int(number))) if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default



def _local_compute_metrics(context: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], List[str]]:
    """Collect bounded local telemetry without device IDs, serials, MACs, or hostnames."""
    ctx = dict(context or {}) if isinstance(context, dict) else {}
    metrics: Dict[str, Any] = {key: None for key in _COMPUTE_METRIC_KEYS}
    sources: List[str] = []

    try:
        metrics_fn = getattr(config, "get_system_metrics", None)
        if callable(metrics_fn):
            observed = metrics_fn() or {}
            if isinstance(observed, dict):
                for key in _COMPUTE_METRIC_KEYS:
                    if observed.get(key) is not None:
                        metrics[key] = observed.get(key)
                sources.append("SarahMemoryGlobals.get_system_metrics")
    except Exception:
        pass

    if _HAS_PSUTIL and psutil is not None:
        try:
            metrics["cpu_count"] = metrics.get("cpu_count") or psutil.cpu_count(logical=True)
            metrics["cpu_pct"] = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            metrics["ram_total_mb"] = metrics.get("ram_total_mb") or int(vm.total / (1024 * 1024))
            metrics["ram_avail_mb"] = int(vm.available / (1024 * 1024))
            root = str(getattr(config, "BASE_DIR", os.getcwd()) or os.getcwd())
            du = psutil.disk_usage(root if os.path.exists(root) else os.getcwd())
            metrics["disk_total_gb"] = metrics.get("disk_total_gb") or round(du.total / (1024 ** 3), 3)
            metrics["disk_free_gb"] = round(du.free / (1024 ** 3), 3)
            sources.append("psutil")
        except Exception:
            pass
    else:
        try:
            metrics["cpu_count"] = metrics.get("cpu_count") or os.cpu_count()
            usage = shutil.disk_usage(str(getattr(config, "BASE_DIR", os.getcwd()) or os.getcwd()))
            metrics["disk_total_gb"] = round(usage.total / (1024 ** 3), 3)
            metrics["disk_free_gb"] = round(usage.free / (1024 ** 3), 3)
            sources.append("python_stdlib")
        except Exception:
            pass

    # Context may fill genuinely unavailable evidence, but never overwrites local telemetry.
    supplied = ctx.get("hardware_metrics") if isinstance(ctx.get("hardware_metrics"), dict) else {}
    for key in _COMPUTE_METRIC_KEYS:
        if metrics.get(key) is None and supplied.get(key) is not None:
            metrics[key] = supplied.get(key)
            if "runtime_context" not in sources:
                sources.append("runtime_context")
    return metrics, sources


def build_compute_passport(context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a privacy-safe, non-authorizing machine capability passport.

    The passport describes what the current body can safely attempt. It does not
    identify the owner, arm hardware, select an executor, or mutate clocks.
    """
    ctx = dict(context or {}) if isinstance(context, dict) else {}
    # Only locally measured evidence enters a real machine-capacity passport.
    # Caller-supplied hardware metrics remain unverified and are not applied.
    metrics, sources = _local_compute_metrics({})
    verified_metrics = metrics
    supplied = ctx.get("hardware_metrics") if isinstance(ctx.get("hardware_metrics"), dict) else {}
    unverified_keys = [k for k in _COMPUTE_METRIC_KEYS if metrics.get(k) is None and supplied.get(k) is not None]
    if unverified_keys:
        sources.append("unverified_runtime_context_excluded")
    score: Optional[float] = None
    tier = "unknown"
    tier_rating = "Unknown"
    try:
        score_fn = getattr(config, "hardware_score", None)
        if callable(score_fn):
            scored = score_fn(verified_metrics) or {}
            if isinstance(scored, dict):
                score = _compute_number(scored.get("score"), 0.0)
                tier = str(scored.get("tier") or "unknown").strip().lower()
                tier_rating = str(scored.get("tier_rating") or "Unknown").strip()
                sources.append("SarahMemoryGlobals.hardware_score")
    except Exception:
        pass

    logical_cpu = _compute_int(metrics.get("cpu_count"))
    ram_total_mb = _compute_number(metrics.get("ram_total_mb"))
    ram_available_mb = _compute_number(metrics.get("ram_avail_mb"))
    disk_free_gb = _compute_number(metrics.get("disk_free_gb"), -1.0)
    vram_total_mb = _compute_number(metrics.get("gpu_vram_total_mb"))
    vram_available_mb = _compute_number(metrics.get("gpu_vram_free_mb"))
    architecture = str(platform.machine() or "unknown").strip().lower()
    operating_system = str(platform.system() or "unknown").strip()

    evidence = {
        "cpu_topology": _compute_int(verified_metrics.get("cpu_count")) > 0,
        "memory_capacity": _compute_number(verified_metrics.get("ram_total_mb")) > 0,
        "storage_capacity": verified_metrics.get("disk_free_gb") is not None,
        "architecture": architecture not in {"", "unknown"},
        "accelerator_capacity": _compute_number(verified_metrics.get("gpu_vram_total_mb")) > 0,
        "live_pressure": verified_metrics.get("cpu_pct") is not None and verified_metrics.get("ram_avail_mb") is not None,
    }
    essential = ("cpu_topology", "memory_capacity", "storage_capacity", "architecture")
    missing = [name for name, present in evidence.items() if not present]
    confidence = round(sum(1.0 for present in evidence.values() if present) / len(evidence), 4)
    unknown_machine = any(not evidence[name] for name in essential)

    # Eligibility is decided from locally observed telemetry, not caller context.
    trusted_ram = _compute_number(verified_metrics.get("ram_total_mb"))
    trusted_cpu = _compute_int(verified_metrics.get("cpu_count"))
    trusted_vram = _compute_number(verified_metrics.get("gpu_vram_total_mb"))
    if unknown_machine or trusted_ram < 4096 or trusted_cpu < 2:
        machine_class = "survival"
    elif trusted_ram < 8192 or trusted_cpu < 4:
        machine_class = "constrained"
    elif trusted_vram >= 8192 and trusted_ram >= 16384 and trusted_cpu >= 6:
        machine_class = "accelerated"
    elif trusted_ram >= 16384 and trusted_cpu >= 4:
        machine_class = "balanced"
    else:
        machine_class = "standard"

    fingerprint_basis = {
        "os": operating_system.lower(),
        "arch": architecture,
        "cpu_bucket": min(64, trusted_cpu),
        "ram_bucket_gb": int(max(0.0, trusted_ram) // 4096) * 4,
        "vram_bucket_gb": int(max(0.0, trusted_vram) // 1024),
    }
    passport_id = "compute_" + hashlib.sha256(
        json.dumps(fingerprint_basis, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]

    return {
        "ok": True,
        "schema": COMPUTE_PASSPORT_SCHEMA,
        "passport_id": passport_id,
        "generated_at": _get_timestamp(),
        "machine_class": machine_class,
        "unknown_machine": bool(unknown_machine),
        "confidence": confidence,
        "hardware_score": score,
        "tier": tier,
        "tier_rating": tier_rating,
        "platform": {
            "operating_system": operating_system,
            "architecture": architecture,
            "python_bits": 64 if "64" in platform.architecture()[0] else 32,
        },
        "capacity": {
            "logical_cpu": logical_cpu or None,
            "ram_total_mb": round(ram_total_mb, 2) if ram_total_mb > 0 else None,
            "ram_available_mb": round(ram_available_mb, 2) if ram_available_mb > 0 else None,
            "disk_free_gb": round(disk_free_gb, 3) if disk_free_gb >= 0 else None,
            "disk_total_gb": round(_compute_number(metrics.get("disk_total_gb")), 3) or None,
            "gpu_name": str(metrics.get("gpu_name") or "")[:160] or None,
            "vram_total_mb": round(vram_total_mb, 2) if vram_total_mb > 0 else None,
            "vram_available_mb": round(vram_available_mb, 2) if vram_available_mb > 0 else None,
        },
        "pressure": {
            "cpu_percent": round(_compute_number(metrics.get("cpu_pct")), 2) if metrics.get("cpu_pct") is not None else None,
            "memory_percent": round(100.0 * (1.0 - (ram_available_mb / ram_total_mb)), 2) if ram_total_mb > 0 and ram_available_mb >= 0 else None,
            "gpu_temp_c": round(_compute_number(metrics.get("gpu_temp_c")), 2) if metrics.get("gpu_temp_c") is not None else None,
            "cpu_temp_c": round(_compute_number(metrics.get("cpu_temp_c")), 2) if metrics.get("cpu_temp_c") is not None else None,
        },
        "evidence": evidence,
        "missing_evidence": missing,
        "unverified_context_metrics": unverified_keys,
        "sources": sorted(set(sources)),
        "privacy": {
            "hostname_included": False,
            "serial_numbers_included": False,
            "mac_addresses_included": False,
            "raw_device_ids_included": False,
        },
        "governance": {
            "discovery_is_activation": False,
            "unknown_machine_requires_safe_discovery": bool(unknown_machine),
            "clock_mutation_allowed": False,
            "execution_authority": False,
            "authority_owner": "SarahMemoryOperatorCore",
            "energy_budget_owner": "SarahMemoryEnergetics",
        },
        "execution_authority": False,
    }


def select_adaptive_compute_plan(
    workload: str = "cognition",
    *,
    passport: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Select a bounded software workload plan; no hardware settings are changed."""
    pp = dict(passport or build_compute_passport(context))
    workload_key = str(workload or "cognition").strip().lower().replace(" ", "_")[:80] or "cognition"
    machine_class = str(pp.get("machine_class") or "survival").lower()
    unknown = bool(pp.get("unknown_machine", True))
    pressure = pp.get("pressure") if isinstance(pp.get("pressure"), dict) else {}
    cpu_pressure = _safe_float(pressure.get("cpu_percent"), -1.0, 0.0, 100.0) if pressure.get("cpu_percent") is not None else -1.0
    memory_pressure = _safe_float(pressure.get("memory_percent"), -1.0, 0.0, 100.0) if pressure.get("memory_percent") is not None else -1.0

    profiles: Dict[str, Dict[str, Any]] = {
        "survival": {"workers": 1, "model_budget_gb": 1.0, "avatar_fps": 12, "avatar_scale": 0.50, "xr_hz": 10, "batch": 1},
        "constrained": {"workers": 1, "model_budget_gb": 2.0, "avatar_fps": 18, "avatar_scale": 0.65, "xr_hz": 15, "batch": 1},
        "standard": {"workers": 2, "model_budget_gb": 4.0, "avatar_fps": 24, "avatar_scale": 0.75, "xr_hz": 20, "batch": 2},
        "balanced": {"workers": 3, "model_budget_gb": 6.0, "avatar_fps": 30, "avatar_scale": 0.85, "xr_hz": 30, "batch": 2},
        "accelerated": {"workers": 4, "model_budget_gb": 10.0, "avatar_fps": 45, "avatar_scale": 1.0, "xr_hz": 45, "batch": 4},
    }
    budget = dict(profiles.get(machine_class) or profiles["survival"])
    reasons: List[str] = [f"Machine classified as {machine_class} from bounded local capability evidence."]
    mode = "SAFE_DISCOVERY" if unknown else machine_class.upper()

    if bool(getattr(config, "SAFE_MODE", False)):
        budget.update(profiles["survival"])
        mode = "SAFE_DISCOVERY"
        reasons.append("Global SAFE_MODE is active; conservative budgets only.")
    if max(cpu_pressure, memory_pressure) >= 90.0:
        budget.update(profiles["survival"])
        mode = "PRESSURE_RECOVERY"
        reasons.append("Live CPU or memory pressure is critical; workload reduced to survival profile.")
    elif max(cpu_pressure, memory_pressure) >= 78.0:
        budget["workers"] = 1
        budget["batch"] = 1
        budget["avatar_fps"] = min(int(budget["avatar_fps"]), 18)
        budget["xr_hz"] = min(int(budget["xr_hz"]), 15)
        budget["avatar_scale"] = min(float(budget["avatar_scale"]), 0.65)
        mode = "PRESSURE_GUARDED"
        reasons.append("Live CPU or memory pressure is elevated; concurrency and real-time rates reduced.")

    if unknown:
        budget.update(profiles["survival"])
        reasons.append("Essential machine evidence is missing; safe discovery profile is mandatory.")

    external_passport_unverified = passport is not None
    if external_passport_unverified:
        reasons.append("Externally supplied passport is unverified evidence; governance preflight remains mandatory.")

    heavy_workload = workload_key in {"training", "fine_tuning", "rendering", "xr", "world_simulation", "evolution"}
    if workload_key in {"background", "indexing", "sync", "diagnostics"}:
        budget["workers"] = 1
        budget["batch"] = 1
        reasons.append("Background work is intentionally single-worker to protect foreground cognition.")

    return {
        "ok": True,
        "schema": ADAPTIVE_COMPUTE_PLAN_SCHEMA,
        "plan_id": "plan_" + hashlib.sha256(
            f"{pp.get('passport_id')}:{workload_key}:{mode}:{budget}".encode("utf-8")
        ).hexdigest()[:20],
        "generated_at": _get_timestamp(),
        "workload": workload_key,
        "mode": mode,
        "machine_class": machine_class,
        "passport_id": pp.get("passport_id"),
        "decision": "ALLOW_WITH_CONSTRAINTS",
        "recommendation_only": True,
        "budgets": {
            "max_concurrent_workers": int(budget["workers"]),
            "max_batch_size": int(budget["batch"]),
            "local_model_memory_budget_gb": float(budget["model_budget_gb"]),
            "background_io_policy": "incremental_bounded",
        },
        "workloads": {
            "avatar": {
                "fps_cap": int(budget["avatar_fps"]),
                "render_scale": float(budget["avatar_scale"]),
                "temporal_history": machine_class not in {"survival", "constrained"} and mode not in {"PRESSURE_RECOVERY"},
                "neon_effects": mode not in {"SAFE_DISCOVERY", "PRESSURE_RECOVERY"},
            },
            "sarahnet_rt": {
                "target_state_hz": int(budget["xr_hz"]),
                "poll_limit": max(50, int(budget["xr_hz"]) * 4),
                "latest_wins": True,
                "ledger_per_frame": False,
            },
        },
        "requires_energetics_preflight": bool(heavy_workload or unknown or external_passport_unverified),
        "external_passport_unverified": external_passport_unverified,
        "requires_operator_core_for_execution": True,
        "clock_mutation_allowed": False,
        "reasons": reasons,
        "execution_authority": False,
    }


def adaptive_compute_self_test() -> Dict[str, Any]:
    """Side-effect-free contract checks using synthetic capability evidence."""
    base = {
        "passport_id": "synthetic",
        "pressure": {"cpu_percent": 10.0, "memory_percent": 20.0},
        "execution_authority": False,
    }
    unknown_plan = select_adaptive_compute_plan("avatar", passport={**base, "machine_class": "survival", "unknown_machine": True})
    fast_plan = select_adaptive_compute_plan("xr", passport={**base, "machine_class": "accelerated", "unknown_machine": False})
    checks = [
        {"name": "unknown_fails_safe", "passed": unknown_plan.get("mode") == "SAFE_DISCOVERY" and unknown_plan.get("budgets", {}).get("max_concurrent_workers") == 1},
        {"name": "accelerated_is_bounded", "passed": 1 <= int(fast_plan.get("budgets", {}).get("max_concurrent_workers") or 0) <= 4},
        {"name": "no_execution_authority", "passed": not unknown_plan.get("execution_authority") and not fast_plan.get("execution_authority")},
        {"name": "no_clock_mutation", "passed": not unknown_plan.get("clock_mutation_allowed") and not fast_plan.get("clock_mutation_allowed")},
    ]
    return {"ok": all(item["passed"] for item in checks), "checks": checks, "execution_authority": False}


# ============================================================================
# DIAGNOSTICS AND TESTING
# ============================================================================

def run_diagnostics() -> Dict[str, Any]:
    """Observe available resources without creating databases or changing state."""
    report: Dict[str, Any] = {"status": "ok", "checks": {}, "errors": [],
                              "execution_authority": False, "read_only": True}
    for name, path, table, count_name in (
        ("memory_db", MEMORY_DB_PATH, "conversations", "interaction_count"),
        ("emotion_db", EMOTION_DB_PATH, "traits", "trait_count"),
    ):
        if not os.path.isfile(path):
            report["checks"][name] = {"status": "warning", "reason": "database_not_initialized"}
            continue
        try:
            db_uri = pathlib.Path(path).absolute().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(db_uri, uri=True, timeout=1.0)) as conn:
                conn.execute("PRAGMA query_only=ON")
                value = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            report["checks"][name] = {"status": "ok", count_name: value}
        except (sqlite3.Error, OSError) as exc:
            report["checks"][name] = {"status": "error", "error_type": type(exc).__name__}
            report["errors"].append(f"{name} inaccessible")
    with _state_lock:
        state = _current_state()
        report["checks"]["state"] = {"status": "ok" if isinstance(state, dict) and state else "warning",
                                      "mode": str(state.get("mode", "unknown")) if isinstance(state, dict) else "unknown",
                                      "scoped": bool(_scope_key())}
    report["checks"]["dependencies"] = {"numpy": _HAS_NUMPY, "psutil": _HAS_PSUTIL}
    try:
        passport = build_compute_passport()
        plan = select_adaptive_compute_plan("diagnostics", passport=passport)
        checked = adaptive_compute_self_test()
        report["checks"]["adaptive_compute"] = {
            "status": "ok" if checked["ok"] else "error", "contract_test": checked,
            "passport_schema": passport["schema"], "unknown_machine": passport["unknown_machine"],
            "machine_class": passport["machine_class"], "plan_mode": plan["mode"]}
        if not checked["ok"]:
            report["errors"].append("adaptive_compute_contract_failed")
    except (ValueError, TypeError, RuntimeError) as exc:
        report["checks"]["adaptive_compute"] = {"status": "error", "error_type": type(exc).__name__}
        report["errors"].append("adaptive_compute_unavailable")
    if report["errors"] or any(x.get("status") == "warning" for x in report["checks"].values()):
        report["status"] = "degraded"
    return report



# ============================================================================
# MAIN BLOCK - TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SarahMemory Adaptive Behavior Engine - Test Suite")
    print("=" * 70)
    
    # Run diagnostics
    print("\n[1] Running Diagnostics...")
    diagnostics = run_diagnostics()
    print(f"    Status: {diagnostics['status']}")
    for check_name, check_result in diagnostics['checks'].items():
        print(f"    - {check_name}: {check_result}")
    
    # Test emotional learning
    print("\n[2] Testing Emotional Learning...")
    test_inputs = [
        "I had an awesome day and I love the results!",
        "This is terrible and I hate everything about it.",
        "What time is it?",
        "Wow! That's absolutely incredible!",
        "I'm feeling okay, nothing special."
    ]
    
    for test_input in test_inputs:
        metrics = advanced_emotional_learning(test_input)
        sentiment = analyze_sentiment(test_input)
        print(f"\n    Input: \"{test_input[:50]}...\"")
        print(f"    Sentiment: {sentiment.sentiment_score:.2f}, "
              f"Balance: {metrics['emotional_balance']:.2f}, "
              f"Engagement: {metrics['engagement']:.2f}")
    
    # Test personality update
    print("\n[3] Testing Personality Update...")
    user_text = "Thank you so much, this has been really helpful!"
    response_text = "You're very welcome! I'm glad I could help."
    result = update_personality(user_text, response_text)
    print(f"    User: \"{user_text}\"")
    print(f"    Response: \"{response_text}\"")
    print(f"    Metrics: {result}")
    
    # Test emotion simulation
    print("\n[4] Testing Emotion Simulation...")
    for emotion_type in ["positive", "negative", "neutral"]:
        state = simulate_emotion_response(emotion_type)
        label = emotion_label_from_state(state)
        print(f"    {emotion_type}: joy={state['joy']:.2f}, "
              f"anger={state['anger']:.2f}, label={label}")
    
    # Test self-update
    print("\n[5] Testing Self-Update...")
    snapshot = self_update_personality()
    print(f"    Status: {snapshot['status']}")
    print(f"    Engagement: {snapshot.get('engagement', 0):.2f}")
    print(f"    Total Interactions: {snapshot.get('total_interactions', 0)}")
    
    # Final state
    print("\n[6] Final Emotional State...")
    final_metrics = get_emotional_metrics()
    for key, value in final_metrics.items():
        print(f"    {key}: {value}")
    
    print("\n" + "=" * 70)
    print("Test Suite Complete")
    print("=" * 70)


# ============================================================================
# LEGACY/COMPATIBILITY EXPORTS
# ============================================================================

# Ensure response table (legacy support)
def _ensure_response_table(db_path: str = None) -> None:
    """Legacy function to ensure response table exists."""
    try:
        if db_path is None:
            db_path = os.path.join(config.DATASETS_DIR, "system_logs.db")
        
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                user TEXT,
                content TEXT,
                source TEXT,
                intent TEXT
            )
        """)
        conn.commit()
        conn.close()
        
        logger.debug(f"[DB] Ensured response table in {db_path}")
        
    except Exception as e:
        logger.warning(f"[DB] ensure response table failed: {e}")


# Auto-create response table if enabled
if getattr(config, "ENABLE_RESPONSE_LOG_TABLE", False):
    try:
        _ensure_response_table()
    except Exception:
        pass



# --- SM V8.0 TRI-LAYER PATCH 2026-05-20 ---
def build_emotion_affect_packet(user_input: str, context_packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build the Layer-3 Emotion/Affect packet. Emotion informs meaning; it does not authorize action."""
    try:
        import SarahMemoryCognitiveIdentityLayer as _CIL  # type: ignore
        return _CIL.build_emotion_affect_packet(user_input, context_packet=context_packet)
    except Exception as e:
        return {
            "packet_type": "EmotionAffectPacket",
            "schema": "SarahMemory.emotion_affect.v1.fallback",
            "primary_emotion": "neutral",
            "secondary_emotions": [],
            "sentiment_score": 0.0,
            "emotional_intensity": 0.0,
            "urgency": 0.0,
            "stress": 0.0,
            "confidence": 0.25,
            "input_role": "meaning_signal",
            "execution_authority": False,
            "error": type(e).__name__,
        }
# ====================================================================
# END OF SarahMemoryAdaptive.py v9.1.0a
# ====================================================================

# Enterprise compatibility contract: Existing public names and packet structures
# remain intact. For multi-user hosts, API/Bridge MUST wrap each authenticated
# request in `with adaptive_scope(trusted_identity):` before touching adaptive
# state, history, preferences or metrics. A missing scope retains original
# single-user behavior and MUST NOT be used for multi-user public services.
# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryAdaptive',
    "version": "v9.1.0a-candidate-sml-0.2",
    "category": 'Reasoning',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['adaptive_state', 'reasoning', 'compute_passport', 'adaptive_compute_planning'],
    "supported_missions": ['Conversation', 'Knowledge', 'Planning', 'Programming'],
    "supported_omega": ['Ω001', 'Ω002', 'Ω005', 'Ω010', 'Ω020', 'Ω030', 'Ω040'],
    "required_authority": ['Read'],
    "priority": 70,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryAdaptive.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return copy.deepcopy(SML_ORGAN_METADATA)


def sml_health():
    """Measured/observable local health; never manufacture perfect reliability."""
    files_exist = {"memory": os.path.isfile(MEMORY_DB_PATH),
                   "emotion": os.path.isfile(EMOTION_DB_PATH)}
    state_ok = bool(get_current_emotional_state())
    initialized = all(files_exist.values())
    status = "Healthy" if initialized and state_ok else "Degraded"
    return {"status": status,
            "availability": 1.0 if state_ok else 0.0,
            "integrity": None, "performance": None, "reliability": None,
            "confidence": 0.7 if initialized and state_ok else 0.35,
            "latency_ms": None, "stability": None, "compatibility": None,
            "notes": ["Local state accessible" if state_ok else "Local state unavailable",
                      "Persistence files present" if initialized else "Persistence not fully initialized",
                      "Runtime metrics not measured by this adapter"],
            "execution_authority": False}



def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryAdaptive',
        "sml_adapter": True,
        "metadata": copy.deepcopy(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryAdaptive', action=action, note=note or "organ observed packet", updates=updates)
    except Exception as exc:
        logger.warning("Adaptive SML packet update unavailable: %s", type(exc).__name__)
        return packet
# --- SML ORGAN ADAPTER END ---

# --- SML ADAPTIVE SPECIALIZATION START ---
def sml_update_adaptive_packet(packet, vector=None):
    """Update only the SML adaptive section; this does not alter facts/truth."""
    from SarahMemorySMLProtocol import get_protocol, SMLPacket
    pkt = packet if isinstance(packet, SMLPacket) else SMLPacket.from_dict(packet)
    return get_protocol().update_adaptive(pkt, vector=vector or None)
# --- SML ADAPTIVE SPECIALIZATION END ---
