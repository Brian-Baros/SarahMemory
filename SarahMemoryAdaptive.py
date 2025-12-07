"""--==The SarahMemory Project==--
File: SarahMemoryAdaptive.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-02
Time: 10:11:54
Author: © 2025 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
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

import logging
import sqlite3
import datetime
import os
import random
import time
import json
import math
import threading
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from collections import deque
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
        """Create from dictionary."""
        return cls(
            joy=float(data.get("joy", 0.5)),
            trust=float(data.get("trust", 0.3)),
            fear=float(data.get("fear", 0.2)),
            surprise=float(data.get("surprise", 0.1)),
            sadness=float(data.get("sadness", 0.1)),
            disgust=float(data.get("disgust", 0.05)),
            anger=float(data.get("anger", 0.1)),
            anticipation=float(data.get("anticipation", 0.3)),
            emotional_balance=float(data.get("emotional_balance", 0.0)),
            openness=float(data.get("openness", 0.6)),
            engagement=float(data.get("engagement", 0.4)),
            mode=str(data.get("mode", "balanced")),
            cpu=float(data.get("cpu", 0.0)),
            memory=float(data.get("memory", 0.0)),
            last_updated=str(data.get("last_updated", "")),
            adjustments=list(data.get("adjustments", []))
        )
    
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
STATE = DEFAULT_EMOTIONS.copy()

# Interaction history buffer (in-memory ring buffer)
_interaction_buffer: deque = deque(maxlen=100)

# Emotional momentum (tracks recent emotional trajectory)
_emotion_momentum: deque = deque(maxlen=20)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _sigmoid(x: float) -> float:
    """Compute sigmoid function for normalization."""
    try:
        if _HAS_NUMPY:
            return float(1.0 / (1.0 + np.exp(-x)))
        else:
            return 1.0 / (1.0 + math.exp(-x))
    except (OverflowError, ValueError):
        return 0.5


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a value to a specified range."""
    return max(min_val, min(max_val, value))


def _get_system_resources() -> Tuple[float, float]:
    """
    Get current CPU and memory usage percentages.
    Returns (cpu_percent, memory_percent).
    """
    if not _HAS_PSUTIL:
        return (50.0, 50.0)  # Default neutral values
    
    try:
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
        return (cpu, mem)
    except Exception as e:
        logger.warning(f"Failed to get system resources: {e}")
        return (50.0, 50.0)


def _get_timestamp() -> str:
    """Get current ISO timestamp."""
    return datetime.datetime.now().isoformat()


def _tokenize(text: str) -> List[str]:
    """Simple tokenization of text into lowercase words."""
    if not text:
        return []
    # Remove punctuation and split
    cleaned = ''.join(c.lower() if c.isalnum() or c.isspace() else ' ' for c in text)
    return cleaned.split()


# ============================================================================
# DATABASE OPERATIONS - MEMORY DB (ai_learning.db)
# ============================================================================

def _ensure_memory_db_schema(conn: sqlite3.Connection) -> None:
    """Ensure the memory database has all required tables."""
    cursor = conn.cursor()
    
    # Conversations table (interaction history)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_input TEXT,
            ai_response TEXT,
            intent TEXT,
            sentiment_score REAL DEFAULT 0.0,
            emotional_state TEXT,
            session_id TEXT
        )
    """)
    
    # Interaction metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interaction_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            word_count INTEGER,
            sentiment_score REAL,
            emotional_intensity REAL,
            response_quality REAL,
            mode TEXT
        )
    """)
    
    # User preferences learned over time
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            preference_key TEXT UNIQUE NOT NULL,
            preference_value TEXT,
            confidence REAL DEFAULT 0.5,
            last_updated TEXT
        )
    """)
    
    # Create indices for performance
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_timestamp 
        ON conversations(timestamp)
    """)
    
    conn.commit()


def connect_memory_db() -> Optional[sqlite3.Connection]:
    """
    Connect to the memory database (ai_learning.db).
    Creates the database and tables if they don't exist.
    
    Returns:
        sqlite3.Connection or None if connection fails
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(MEMORY_DB_PATH), exist_ok=True)
        
        conn = sqlite3.connect(MEMORY_DB_PATH, timeout=10.0)
        conn.row_factory = sqlite3.Row
        
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        
        _ensure_memory_db_schema(conn)
        
        logger.debug(f"Connected to memory database: {MEMORY_DB_PATH}")
        return conn
        
    except Exception as e:
        logger.error(f"Failed to connect to memory database: {e}")
        return None


def log_interaction_to_db(
    user_input: str,
    ai_response: str,
    intent: str = "unknown",
    sentiment_score: float = 0.0,
    emotional_state: Dict = None,
    session_id: str = None
) -> bool:
    """
    Log an interaction to the memory database.
    
    Args:
        user_input: The user's input text
        ai_response: The AI's response text
        intent: Classified intent of the interaction
        sentiment_score: Sentiment analysis score (-1.0 to 1.0)
        emotional_state: Current emotional state dictionary
        session_id: Optional session identifier
    
    Returns:
        True if logged successfully, False otherwise
    """
    try:
        conn = connect_memory_db()
        if not conn:
            return False
        
        timestamp = _get_timestamp()
        emotional_json = json.dumps(emotional_state) if emotional_state else None
        
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations 
            (timestamp, user_input, ai_response, intent, sentiment_score, emotional_state, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, user_input, ai_response, intent, sentiment_score, emotional_json, session_id))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Logged interaction at {timestamp}")
        return True
        
    except Exception as e:
        logger.error(f"Error logging interaction to database: {e}")
        return False


def get_recent_interactions(limit: int = 20) -> List[Dict]:
    """
    Retrieve recent interactions from the database.
    
    Args:
        limit: Maximum number of interactions to retrieve
    
    Returns:
        List of interaction dictionaries
    """
    try:
        conn = connect_memory_db()
        if not conn:
            return []
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT timestamp, user_input, ai_response, intent, sentiment_score
            FROM conversations
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
        
    except Exception as e:
        logger.error(f"Error retrieving interactions: {e}")
        return []


def get_interaction_count() -> int:
    """Get the total number of logged interactions."""
    try:
        conn = connect_memory_db()
        if not conn:
            return 0
        
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM conversations")
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
        
    except Exception as e:
        logger.error(f"Error getting interaction count: {e}")
        return 0


# ============================================================================
# DATABASE OPERATIONS - EMOTION DB (personality1.db)
# ============================================================================

def _ensure_emotion_db_schema(conn: sqlite3.Connection) -> None:
    """Ensure the emotion database has all required tables."""
    cursor = conn.cursor()
    
    # Traits table (emotional state persistence)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS traits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trait_name TEXT UNIQUE NOT NULL,
            description TEXT,
            last_updated TEXT
        )
    """)
    
    # Emotional history (time series)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emotional_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            emotion_snapshot TEXT,
            dominant_emotion TEXT,
            emotional_balance REAL,
            trigger_text TEXT
        )
    """)
    
    # Create legacy view for backward compatibility
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS emotion_states AS
        SELECT trait_name AS name, description AS value 
        FROM traits
    """)
    
    conn.commit()


def connect_emotion_db() -> Optional[sqlite3.Connection]:
    """
    Connect to the emotion database (personality1.db).
    Creates the database and tables if they don't exist.
    
    Returns:
        sqlite3.Connection or None if connection fails
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(EMOTION_DB_PATH), exist_ok=True)
        
        conn = sqlite3.connect(EMOTION_DB_PATH, timeout=10.0)
        conn.row_factory = sqlite3.Row
        
        # Enable WAL mode
        conn.execute("PRAGMA journal_mode=WAL")
        
        _ensure_emotion_db_schema(conn)
        
        logger.debug(f"Connected to emotion database: {EMOTION_DB_PATH}")
        return conn
        
    except Exception as e:
        logger.error(f"Failed to connect to emotion database: {e}")
        return None


def load_emotional_state() -> Dict[str, Any]:
    """
    Load the emotional state from the database.
    Falls back to defaults if loading fails.
    
    Returns:
        Dictionary containing the emotional state
    """
    global STATE
    
    emotions = DEFAULT_EMOTIONS.copy()
    
    try:
        conn = connect_emotion_db()
        if not conn:
            STATE = emotions.copy()
            return emotions
        
        cursor = conn.cursor()
        cursor.execute("SELECT trait_name, description FROM traits")
        
        for row in cursor.fetchall():
            trait_name = row["trait_name"]
            description = row["description"]
            
            # Parse numeric values
            if trait_name in ("cpu", "memory", "joy", "trust", "fear", "surprise",
                             "sadness", "disgust", "anger", "anticipation",
                             "emotional_balance", "openness", "engagement"):
                try:
                    emotions[trait_name] = float(description)
                except (ValueError, TypeError):
                    pass
            
            # Parse list values (JSON)
            elif trait_name == "adjustments":
                try:
                    emotions[trait_name] = json.loads(description) if description else []
                except (json.JSONDecodeError, TypeError):
                    emotions[trait_name] = [str(description)] if description else []
            
            # String values
            elif trait_name in ("mode", "last_updated"):
                emotions[trait_name] = str(description) if description else ""
            
            else:
                # Unknown trait, store as-is
                emotions[trait_name] = description
        
        conn.close()
        
        with _state_lock:
            STATE = emotions.copy()
        
        logger.debug(f"Loaded emotional state: {emotions}")
        return emotions
        
    except Exception as e:
        logger.error(f"Error loading emotional state: {e}")
        STATE = emotions.copy()
        return emotions


def save_emotional_state(state: Dict[str, Any] = None) -> bool:
    """
    Persist the emotional state to the database.
    
    Args:
        state: State dictionary to save (uses global STATE if None)
    
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        state_to_save = state or STATE
        if not state_to_save:
            return False
        
        conn = connect_emotion_db()
        if not conn:
            return False
        
        cursor = conn.cursor()
        timestamp = _get_timestamp()
        
        for trait_name, value in state_to_save.items():
            # Skip transient nested objects
            if trait_name == "emotion":
                continue
            
            # Format value for storage
            if isinstance(value, (int, float)):
                if _HAS_NUMPY and hasattr(value, 'item'):
                    store_value = f"{float(value.item()):.4f}"
                else:
                    store_value = f"{float(value):.4f}"
            elif isinstance(value, (list, dict)):
                try:
                    store_value = json.dumps(value)
                except (TypeError, ValueError):
                    store_value = str(value)
            else:
                store_value = str(value) if value is not None else ""
            
            # Upsert the trait
            cursor.execute("""
                INSERT INTO traits (trait_name, description, last_updated)
                VALUES (?, ?, ?)
                ON CONFLICT(trait_name) 
                DO UPDATE SET description = excluded.description, 
                              last_updated = excluded.last_updated
            """, (trait_name, store_value, timestamp))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Saved emotional state at {timestamp}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving emotional state: {e}")
        return False


def log_emotional_history(
    emotion_snapshot: Dict,
    trigger_text: str = None
) -> bool:
    """
    Log the current emotional state to history for analysis.
    
    Args:
        emotion_snapshot: Current emotional state snapshot
        trigger_text: Optional text that triggered the emotional change
    
    Returns:
        True if logged successfully
    """
    try:
        conn = connect_emotion_db()
        if not conn:
            return False
        
        timestamp = _get_timestamp()
        
        # Get dominant emotion
        emotions = {k: v for k, v in emotion_snapshot.items() 
                   if k in ("joy", "trust", "fear", "surprise", "sadness", 
                           "disgust", "anger", "anticipation")}
        dominant = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "neutral"
        balance = emotion_snapshot.get("emotional_balance", 0.0)
        
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO emotional_history 
            (timestamp, emotion_snapshot, dominant_emotion, emotional_balance, trigger_text)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, json.dumps(emotion_snapshot), dominant, balance, trigger_text))
        
        conn.commit()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Error logging emotional history: {e}")
        return False


# ============================================================================
# SENTIMENT ANALYSIS ENGINE
# ============================================================================

def analyze_sentiment(text: str) -> InteractionMetrics:
    """
    Perform sentiment analysis on input text.
    
    Uses lexicon-based analysis with multiple factors:
    - Positive/negative keyword counting
    - Surprise detection
    - Question detection
    - Intensity modifiers (very, extremely, etc.)
    
    Args:
        text: Input text to analyze
    
    Returns:
        InteractionMetrics with analysis results
    """
    if not text:
        return InteractionMetrics(timestamp=_get_timestamp())
    
    words = _tokenize(text)
    word_count = len(words)
    
    if word_count == 0:
        return InteractionMetrics(timestamp=_get_timestamp())
    
    # Count positive and negative words
    positive_count = sum(1 for w in words if w in POSITIVE_KEYWORDS)
    negative_count = sum(1 for w in words if w in NEGATIVE_KEYWORDS)
    surprise_count = sum(1 for w in words if w in SURPRISE_INDICATORS)
    
    # Check for questions
    question_detected = (
        text.strip().endswith("?") or
        any(w in QUESTION_INDICATORS for w in words[:3])
    )
    
    # Check for surprise
    surprise_detected = surprise_count > 0
    
    # Intensity modifiers
    intensifiers = {"very", "extremely", "really", "so", "absolutely", "totally", "completely"}
    intensity_multiplier = 1.0 + 0.2 * sum(1 for w in words if w in intensifiers)
    
    # Negation handling (simple)
    negations = {"not", "no", "never", "don't", "doesn't", "didn't", "won't", "can't", "couldn't"}
    negation_count = sum(1 for w in words if w in negations)
    
    # Calculate raw sentiment score
    raw_score = (positive_count - negative_count) * intensity_multiplier
    
    # Apply negation (flips sentiment partially)
    if negation_count > 0:
        raw_score *= -0.5
    
    # Normalize to [-1, 1] using sigmoid-like function
    sentiment_score = _clamp(raw_score / max(word_count, 1) * 2, -1.0, 1.0)
    
    # Calculate emotional intensity (how emotionally charged the text is)
    emotional_words = positive_count + negative_count + surprise_count
    emotional_intensity = _clamp(emotional_words / max(word_count, 1) * 3, 0.0, 1.0)
    
    return InteractionMetrics(
        sentiment_score=round(sentiment_score, 4),
        word_count=word_count,
        question_detected=question_detected,
        surprise_detected=surprise_detected,
        emotional_intensity=round(emotional_intensity, 4),
        response_quality=0.5,  # Default, updated by feedback
        timestamp=_get_timestamp()
    )


def determine_system_mode(cpu: float, memory: float) -> SystemMode:
    """
    Determine the appropriate system mode based on resource usage.
    
    Args:
        cpu: Current CPU usage percentage
        memory: Current memory usage percentage
    
    Returns:
        SystemMode enum value
    """
    # Check for safe mode flag in config
    if getattr(config, 'SAFE_MODE', False):
        return SystemMode.SAFE
    
    # High resource usage - lightweight mode
    if cpu > 80 or memory > 85:
        return SystemMode.LIGHTWEIGHT
    
    # Low resource usage - enhanced mode
    if cpu < 50 and memory < 60:
        return SystemMode.ENHANCED
    
    # Normal operation
    return SystemMode.BALANCED


# ============================================================================
# EMOTIONAL LEARNING ENGINE
# ============================================================================

def advanced_emotional_learning(user_input: str) -> Dict[str, Any]:
    """
    Perform advanced emotional learning from user input.
    
    This function:
    1. Analyzes sentiment of the input
    2. Adjusts emotional state based on analysis
    3. Considers system resources for mode switching
    4. Applies emotional momentum and decay
    5. Returns computed emotional metrics
    
    Args:
        user_input: The user's input text
    
    Returns:
        Dictionary containing emotional metrics:
        - emotional_balance: float (-1.0 to 1.0)
        - openness: float (0.0 to 1.0)
        - engagement: float (0.0 to 1.0)
    """
    global STATE, _emotion_momentum
    
    with _state_lock:
        # Get system resources
        cpu, memory = _get_system_resources()
        STATE["cpu"] = cpu
        STATE["memory"] = memory
        
        # Determine system mode
        mode = determine_system_mode(cpu, memory)
        old_mode = STATE.get("mode", "balanced")
        STATE["mode"] = mode.value
        
        # Log mode changes
        if old_mode != mode.value:
            adjustment = f"Mode changed: {old_mode} -> {mode.value} at {time.ctime()}"
            if "adjustments" not in STATE:
                STATE["adjustments"] = []
            STATE["adjustments"].append(adjustment)
            STATE["adjustments"] = STATE["adjustments"][-10:]  # Keep last 10
            logger.info(f"[MODE] {adjustment}")
        
        # Analyze sentiment
        metrics = analyze_sentiment(user_input)
        
        # Calculate emotional balance based on mode
        if mode == SystemMode.LIGHTWEIGHT:
            # Simplified calculation for high-load situations
            balance = metrics.sentiment_score
        elif mode == SystemMode.ENHANCED:
            # Full calculation with memory consideration
            balance = metrics.sentiment_score * (1.1 - memory / 100.0)
        else:
            # Balanced mode
            word_factor = min(1.0, len(_tokenize(user_input)) / 20.0)
            balance = metrics.sentiment_score * (1.0 - memory / 200.0) + word_factor * 0.1
        
        # Apply sigmoid normalization and scale to [-1, 1]
        emotional_balance = (_sigmoid(balance) - 0.5) * 2
        
        # Add to momentum buffer
        _emotion_momentum.append(emotional_balance)
        
        # Apply momentum smoothing (weighted average with recent history)
        if len(_emotion_momentum) > 1:
            weights = [0.5 ** i for i in range(len(_emotion_momentum))]
            weights.reverse()
            momentum_values = list(_emotion_momentum)
            weighted_sum = sum(w * v for w, v in zip(weights, momentum_values))
            emotional_balance = weighted_sum / sum(weights)
        
        # Calculate openness (willingness to engage)
        base_openness = 0.6
        sentiment_factor = (1 + metrics.sentiment_score) / 2  # 0 to 1
        openness = base_openness + (sentiment_factor - 0.5) * 0.2
        openness += (random.random() - 0.5) * 0.05  # Small random variation
        openness = _clamp(openness, 0.3, 0.95)
        
        # Calculate engagement (depth of interaction)
        word_factor = min(1.0, metrics.word_count / 30.0)
        intensity_factor = metrics.emotional_intensity
        resource_factor = 1.0 - (cpu / 200.0)  # Reduce engagement under load
        
        engagement = 0.4 + (word_factor * 0.3) + (intensity_factor * 0.2)
        engagement *= resource_factor
        engagement = _clamp(engagement, 0.2, 1.0)
        
        # Update state
        STATE["emotional_balance"] = round(emotional_balance, 4)
        STATE["openness"] = round(openness, 4)
        STATE["engagement"] = round(engagement, 4)
        STATE["last_updated"] = _get_timestamp()
        
        # Update primary emotions based on sentiment
        _update_emotions_from_sentiment(metrics)
        
        # Build return metrics
        emotion_metrics = {
            "emotional_balance": round(emotional_balance, 4),
            "openness": round(openness, 4),
            "engagement": round(engagement, 4),
            "mode": mode.value,
            "sentiment_score": metrics.sentiment_score,
            "word_count": metrics.word_count
        }
        
        # Store in STATE for other modules
        STATE["emotion"] = emotion_metrics
        
        logger.debug(f"[EMOTION] Balance: {emotional_balance:.2f}, "
                    f"Openness: {openness:.2f}, Engagement: {engagement:.2f}")
        
        return emotion_metrics


def _update_emotions_from_sentiment(metrics: InteractionMetrics) -> None:
    """
    Update the primary emotions based on interaction metrics.
    
    Args:
        metrics: InteractionMetrics from sentiment analysis
    """
    global STATE
    
    sentiment = metrics.sentiment_score
    intensity = metrics.emotional_intensity
    
    # Decay factor (emotions naturally decay toward baseline)
    decay = 0.05
    
    # Positive sentiment increases joy, trust, anticipation
    if sentiment > 0.1:
        STATE["joy"] = _clamp(STATE.get("joy", 0.5) + sentiment * intensity * 0.15, 0.0, 1.0)
        STATE["trust"] = _clamp(STATE.get("trust", 0.3) + sentiment * intensity * 0.10, 0.0, 1.0)
        STATE["anticipation"] = _clamp(STATE.get("anticipation", 0.3) + sentiment * 0.08, 0.0, 1.0)
        
        # Decrease negative emotions
        STATE["anger"] = _clamp(STATE.get("anger", 0.1) - sentiment * 0.10, 0.0, 1.0)
        STATE["sadness"] = _clamp(STATE.get("sadness", 0.1) - sentiment * 0.08, 0.0, 1.0)
        STATE["fear"] = _clamp(STATE.get("fear", 0.2) - sentiment * 0.05, 0.0, 1.0)
    
    # Negative sentiment increases anger, sadness, fear
    elif sentiment < -0.1:
        neg_strength = abs(sentiment)
        STATE["anger"] = _clamp(STATE.get("anger", 0.1) + neg_strength * intensity * 0.20, 0.0, 1.0)
        STATE["sadness"] = _clamp(STATE.get("sadness", 0.1) + neg_strength * intensity * 0.15, 0.0, 1.0)
        STATE["fear"] = _clamp(STATE.get("fear", 0.2) + neg_strength * intensity * 0.10, 0.0, 1.0)
        
        # Decrease positive emotions
        STATE["joy"] = _clamp(STATE.get("joy", 0.5) - neg_strength * 0.15, 0.0, 1.0)
        STATE["trust"] = _clamp(STATE.get("trust", 0.3) - neg_strength * 0.10, 0.0, 1.0)
    
    # Surprise detection
    if metrics.surprise_detected:
        STATE["surprise"] = _clamp(STATE.get("surprise", 0.1) + 0.25, 0.0, 1.0)
    else:
        # Surprise decays faster
        STATE["surprise"] = _clamp(STATE.get("surprise", 0.1) - decay * 2, 0.0, 1.0)
    
    # Apply general decay toward baseline
    baseline = {
        "joy": 0.5, "trust": 0.3, "fear": 0.2, "surprise": 0.1,
        "sadness": 0.1, "disgust": 0.05, "anger": 0.1, "anticipation": 0.3
    }
    
    for emotion, base_value in baseline.items():
        current = STATE.get(emotion, base_value)
        # Move toward baseline
        STATE[emotion] = current + (base_value - current) * decay


def simulate_emotion_response(input_type: str = "positive") -> Dict[str, Any]:
    """
    Simulate an emotional response based on input type.
    
    This function is used for testing and for situations where
    we need to trigger a specific emotional change.
    
    Args:
        input_type: One of "positive", "negative", or "neutral"
    
    Returns:
        Updated emotional state dictionary
    """
    global STATE
    
    logger.info(f"[EMOTION SIMULATION] Triggered by input_type: {input_type}")
    
    # Load current state from DB
    emotions = load_emotional_state()
    
    if input_type == "positive":
        # Boost positive emotions
        emotions["joy"] = _clamp(emotions["joy"] + (1 - emotions["joy"]) * 0.15, 0.0, 1.0)
        emotions["trust"] = _clamp(emotions["trust"] + (1 - emotions["trust"]) * 0.10, 0.0, 1.0)
        emotions["anticipation"] = _clamp(emotions["anticipation"] + (1 - emotions["anticipation"]) * 0.08, 0.0, 1.0)
        
        # Reduce negative emotions
        emotions["anger"] = _clamp(emotions["anger"] * 0.90, 0.0, 1.0)
        emotions["fear"] = _clamp(emotions["fear"] * 0.92, 0.0, 1.0)
        emotions["sadness"] = _clamp(emotions["sadness"] * 0.90, 0.0, 1.0)
        
        emotions["emotional_balance"] = _clamp(emotions.get("emotional_balance", 0) + 0.1, -1.0, 1.0)
    
    elif input_type == "negative":
        # Boost negative emotions
        emotions["anger"] = _clamp(emotions["anger"] + (1 - emotions["anger"]) * 0.25, 0.0, 1.0)
        emotions["fear"] = _clamp(emotions["fear"] + (1 - emotions["fear"]) * 0.20, 0.0, 1.0)
        emotions["sadness"] = _clamp(emotions["sadness"] + (1 - emotions["sadness"]) * 0.18, 0.0, 1.0)
        
        # Reduce positive emotions
        emotions["joy"] = _clamp(emotions["joy"] * 0.85, 0.0, 1.0)
        emotions["trust"] = _clamp(emotions["trust"] * 0.90, 0.0, 1.0)
        
        emotions["emotional_balance"] = _clamp(emotions.get("emotional_balance", 0) - 0.15, -1.0, 1.0)
    
    elif input_type == "neutral":
        # Small surprise boost for engagement
        emotions["surprise"] = _clamp(emotions["surprise"] + (1 - emotions["surprise"]) * 0.05, 0.0, 1.0)
        
        # Slight decay toward baseline
        for key in ["joy", "anger", "fear", "sadness", "trust"]:
            baseline = 0.3 if key in ("joy", "trust") else 0.1
            emotions[key] = emotions[key] + (baseline - emotions[key]) * 0.05
    
    else:
        logger.warning(f"Unknown emotion input type: {input_type}")
    
    # Update timestamp
    emotions["last_updated"] = _get_timestamp()
    
    # Save to database
    save_emotional_state(emotions)
    
    # Log to history
    log_emotional_history(emotions, f"simulation:{input_type}")
    
    # Update global state
    with _state_lock:
        STATE = emotions.copy()
    
    logger.debug(f"[EMOTION] After {input_type} simulation: joy={emotions['joy']:.2f}, "
                f"anger={emotions['anger']:.2f}, trust={emotions['trust']:.2f}")
    
    return emotions


# ============================================================================
# PERSONALITY UPDATE SYSTEM
# ============================================================================

def update_personality(user_input: str, ai_response: str) -> Dict[str, Any]:
    """
    Update the adaptive personality based on an interaction.
    
    This is the main entry point for the adaptive system, called after
    each user interaction. It:
    1. Logs the interaction to the database
    2. Performs emotional learning
    3. Calculates reinforcement factor
    4. Persists the emotional state
    
    Args:
        user_input: The user's input text
        ai_response: The AI's response text
    
    Returns:
        Dictionary containing emotional metrics
    """
    global STATE
    
    # Perform emotional learning
    metrics = advanced_emotional_learning(user_input)
    
    # Calculate reinforcement factor (for future learning)
    reinforcement_factor = 0.01 * (1 + abs(metrics.get("emotional_balance", 0)))
    
    # Log interaction to database
    sentiment_metrics = analyze_sentiment(user_input)
    success = log_interaction_to_db(
        user_input=user_input,
        ai_response=ai_response,
        intent=metrics.get("intent", "unknown"),
        sentiment_score=sentiment_metrics.sentiment_score,
        emotional_state=STATE.copy()
    )
    
    if success:
        logger.info(f"[ADAPTIVE] Interaction logged. Reinforcement: {reinforcement_factor:.4f}")
    else:
        logger.warning("[ADAPTIVE] Failed to log interaction to database")
    
    # Add to in-memory buffer
    _interaction_buffer.append({
        "timestamp": _get_timestamp(),
        "user_input": user_input[:200],  # Truncate for memory
        "sentiment": sentiment_metrics.sentiment_score,
        "metrics": metrics
    })
    
    # Persist emotional state
    try:
        save_emotional_state(STATE)
        log_emotional_history(STATE.copy(), user_input[:100])
    except Exception as e:
        logger.warning(f"Failed to persist emotional state: {e}")
    
    return metrics


def self_update_personality() -> Dict[str, Any]:
    """
    Perform a self-assessment and update based on interaction history.
    
    This function analyzes recent interactions to:
    1. Calculate engagement level based on interaction count
    2. Adjust baseline emotional values
    3. Learn patterns from history
    
    Returns:
        Dictionary with update status and metrics
    """
    try:
        # Get interaction count
        total_interactions = get_interaction_count()
        
        # Calculate engagement using tanh (saturates at high counts)
        if _HAS_NUMPY:
            engagement = min(1.0, float(np.tanh(total_interactions / 100.0)))
        else:
            engagement = min(1.0, math.tanh(total_interactions / 100.0))
        
        # Get recent interactions for pattern analysis
        recent = get_recent_interactions(limit=20)
        
        # Calculate average sentiment from recent interactions
        if recent:
            avg_sentiment = sum(r.get("sentiment_score", 0) for r in recent) / len(recent)
        else:
            avg_sentiment = 0.0
        
        # Update state
        with _state_lock:
            STATE["engagement"] = round(engagement, 4)
            
            # Adjust openness based on sentiment trend
            if avg_sentiment > 0.1:
                STATE["openness"] = _clamp(STATE.get("openness", 0.6) + 0.02, 0.3, 0.95)
            elif avg_sentiment < -0.1:
                STATE["openness"] = _clamp(STATE.get("openness", 0.6) - 0.02, 0.3, 0.95)
        
        logger.info(f"[SELF-UPDATE] Engagement: {engagement:.2f}, "
                   f"Avg sentiment: {avg_sentiment:.2f}, "
                   f"Total interactions: {total_interactions}")
        
        return {
            "status": "ok",
            "engagement": round(engagement, 4),
            "total_interactions": total_interactions,
            "avg_sentiment": round(avg_sentiment, 4)
        }
        
    except Exception as e:
        logger.error(f"Self-update failed: {e}")
        return {
            "status": "error",
            "engagement": 0.0,
            "error": str(e)
        }


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
    s = state or STATE or {}
    
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
    """
    Get a copy of the current emotional state.
    Thread-safe accessor for external modules.
    
    Returns:
        Copy of the current STATE dictionary
    """
    with _state_lock:
        return STATE.copy()


def get_emotional_metrics() -> Dict[str, float]:
    """
    Get simplified emotional metrics for display/API.
    
    Returns:
        Dictionary with key metrics:
        - emotional_balance
        - openness
        - engagement
        - dominant_emotion
        - intensity
    """
    with _state_lock:
        state = STATE.copy()
    
    # Find dominant emotion
    emotions = {k: state.get(k, 0) for k in 
                ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]}
    dominant_name, dominant_value = max(emotions.items(), key=lambda x: x[1])
    
    return {
        "emotional_balance": state.get("emotional_balance", 0.0),
        "openness": state.get("openness", 0.6),
        "engagement": state.get("engagement", 0.4),
        "dominant_emotion": dominant_name,
        "intensity": dominant_value,
        "mode": state.get("mode", "balanced"),
        "label": emotion_label_from_state(state)
    }


def reset_emotional_state() -> Dict[str, Any]:
    """
    Reset emotional state to defaults.
    Useful for testing or recovering from corrupted state.
    
    Returns:
        The reset state dictionary
    """
    global STATE
    
    with _state_lock:
        STATE = DEFAULT_EMOTIONS.copy()
        STATE["last_updated"] = _get_timestamp()
        STATE["adjustments"] = ["Reset to defaults at " + time.ctime()]
    
    save_emotional_state(STATE)
    logger.info("[ADAPTIVE] Emotional state reset to defaults")
    
    return STATE.copy()


# ============================================================================
# USER PREFERENCE LEARNING
# ============================================================================

def learn_user_preference(key: str, value: str, confidence: float = 0.5) -> bool:
    """
    Learn or update a user preference.
    
    Args:
        key: Preference identifier (e.g., "verbosity", "humor_level")
        value: Preference value
        confidence: Confidence level for this preference (0.0 to 1.0)
    
    Returns:
        True if saved successfully
    """
    try:
        conn = connect_memory_db()
        if not conn:
            return False
        
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO user_preferences (preference_key, preference_value, confidence, last_updated)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(preference_key)
            DO UPDATE SET 
                preference_value = excluded.preference_value,
                confidence = MAX(confidence, excluded.confidence),
                last_updated = excluded.last_updated
        """, (key, value, confidence, _get_timestamp()))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"[PREF] Learned preference: {key}={value} (confidence: {confidence:.2f})")
        return True
        
    except Exception as e:
        logger.error(f"Error learning preference: {e}")
        return False


def get_user_preference(key: str, default: str = None) -> Optional[str]:
    """
    Retrieve a learned user preference.
    
    Args:
        key: Preference identifier
        default: Default value if not found
    
    Returns:
        Preference value or default
    """
    try:
        conn = connect_memory_db()
        if not conn:
            return default
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT preference_value, confidence 
            FROM user_preferences 
            WHERE preference_key = ?
        """, (key,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return row["preference_value"]
        return default
        
    except Exception as e:
        logger.error(f"Error getting preference: {e}")
        return default


def get_all_preferences() -> Dict[str, Any]:
    """
    Get all learned user preferences.
    
    Returns:
        Dictionary of preference_key -> {value, confidence}
    """
    try:
        conn = connect_memory_db()
        if not conn:
            return {}
        
        cursor = conn.cursor()
        cursor.execute("""
            SELECT preference_key, preference_value, confidence 
            FROM user_preferences
        """)
        
        prefs = {}
        for row in cursor.fetchall():
            prefs[row["preference_key"]] = {
                "value": row["preference_value"],
                "confidence": row["confidence"]
            }
        
        conn.close()
        return prefs
        
    except Exception as e:
        logger.error(f"Error getting preferences: {e}")
        return {}


# ============================================================================
# DIAGNOSTICS AND TESTING
# ============================================================================

def run_diagnostics() -> Dict[str, Any]:
    """
    Run diagnostic checks on the adaptive system.
    
    Returns:
        Dictionary with diagnostic results
    """
    results = {
        "status": "ok",
        "checks": {},
        "errors": []
    }
    
    # Check memory database
    try:
        conn = connect_memory_db()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM conversations")
            count = cursor.fetchone()[0]
            conn.close()
            results["checks"]["memory_db"] = {"status": "ok", "interaction_count": count}
        else:
            results["checks"]["memory_db"] = {"status": "error", "message": "Connection failed"}
            results["errors"].append("Memory database connection failed")
    except Exception as e:
        results["checks"]["memory_db"] = {"status": "error", "message": str(e)}
        results["errors"].append(f"Memory DB: {e}")
    
    # Check emotion database
    try:
        conn = connect_emotion_db()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM traits")
            count = cursor.fetchone()[0]
            conn.close()
            results["checks"]["emotion_db"] = {"status": "ok", "trait_count": count}
        else:
            results["checks"]["emotion_db"] = {"status": "error", "message": "Connection failed"}
            results["errors"].append("Emotion database connection failed")
    except Exception as e:
        results["checks"]["emotion_db"] = {"status": "error", "message": str(e)}
        results["errors"].append(f"Emotion DB: {e}")
    
    # Check state
    results["checks"]["state"] = {
        "status": "ok" if STATE else "warning",
        "has_emotions": bool(STATE.get("joy")),
        "mode": STATE.get("mode", "unknown")
    }
    
    # Check dependencies
    results["checks"]["dependencies"] = {
        "numpy": _HAS_NUMPY,
        "psutil": _HAS_PSUTIL
    }
    
    # Overall status
    if results["errors"]:
        results["status"] = "degraded"
    
    return results


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
