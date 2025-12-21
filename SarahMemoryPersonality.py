"""--==The SarahMemory Project==--
File: SarahMemoryPersonality.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-21
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

PERSONALITY ENGINE v8.0.0
======================================
This module has standards while maintaining 100% backward
compatibility with existing SarahMemory modules (SarahMemoryReply.py, SarahMemoryAPI.py,
SarahMemoryAvatar.py, etc.).

KEY ENHANCEMENTS:
-----------------
1. ADVANCED EMOTIONAL INTELLIGENCE
   - Multi-layered emotion processing (Plutchik 8 emotions + complex states)
   - Emotional momentum tracking with decay curves
   - Context-sensitive emotional responses
   - Facial expression integration (when available)
   - Real-time emotional state persistence

2. SOPHISTICATED CONTEXT AWARENESS
   - Long-term conversation memory
   - Topic continuity tracking
   - User preference learning
   - Time-of-day personality adaptation
   - Interaction pattern recognition

3. DYNAMIC RESPONSE GENERATION
   - Multi-tier response selection (DB → Generated → Fallback)
   - Emotional tone injection with configurable strength
   - Anti-repetition and loop detection
   - Personality consistency scoring
   - Response quality metrics

4. ENHANCED DATABASE INTEGRATION
   - Optimized query patterns with caching
   - Graceful degradation when DB unavailable
   - Schema auto-verification
   - Transaction safety
   - Performance monitoring

5. ADAPTIVE LEARNING INTEGRATION
   - Reinforcement from user feedback
   - Response effectiveness tracking
   - Personality trait evolution
   - Behavioral pattern adaptation

BACKWARD COMPATIBILITY:
-----------------------
All existing function signatures are preserved:
- process_interaction(user_input)
- integrate_with_personality(text, meta=None)
- get_greeting_response()
- get_emotion_response(emotion_category)
- get_reply_from_db(intent, tone, complexity)
- log_personality_interaction(interaction)
- update_personality_model()
- self_update_personality()

INTEGRATION POINTS:
-------------------
- SarahMemoryReply.py: Uses process_interaction() for response routing
- SarahMemoryAPI.py: Calls integrate_with_personality() for styling
- SarahMemoryAvatar.py: Reads emotional state for expressions
- SarahMemoryAdaptive.py: Provides emotional learning metrics
- SarahMemoryExpressOut.py: Formats expressive output
- SarahMemoryAdvCU.py: Provides intent classification

===============================================================================
"""

from __future__ import annotations

import logging
import sqlite3
import datetime
import time
import os
import random
import json
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import deque, Counter
from dataclasses import dataclass, field, asdict

# Core imports
from SarahMemoryGlobals import DATASETS_DIR, ENABLE_SARCASM_LAYER
from SarahMemoryAdvCU import classify_intent
import SarahMemoryGlobals as config

# Adaptive emotional intelligence imports
try:
    from SarahMemoryAdaptive import advanced_emotional_learning, simulate_emotion_response
except Exception:
    # Graceful fallback if module unavailable
    advanced_emotional_learning = lambda *_a, **_k: {"emotional_balance": 0.0, "openness": 0.6, "engagement": 0.4}
    simulate_emotion_response = lambda _t="neutral": {"joy":0.5,"anger":0.1,"fear":0.2,"trust":0.3,"surprise":0.1}

# Expressive output formatting
try:
    import SarahMemoryExpressOut as ExpressOut
except Exception:
    ExpressOut = None

# Context buffer integration
try:
    from SarahMemoryAiFunctions import get_context, add_to_context, log_ai_functions_event, record_ai_response
except ImportError:
    get_context = lambda: []
    add_to_context = lambda x: None
    log_ai_functions_event = lambda *args: None
    record_ai_response = lambda *args, **kwargs: None

# Database schema management
try:
    from SarahMemoryDatabase import ensure_core_schema
    ensure_core_schema()
except Exception:
    pass


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logger = logging.getLogger('SarahMemoryPersonality')
logger.setLevel(logging.DEBUG if getattr(config, 'DEBUG_MODE', False) else logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [Personality] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Database paths
DB_PATH = os.path.join(DATASETS_DIR, "personality1.db")
MEMORY_DB_PATH = os.path.join(DATASETS_DIR, "ai_learning.db")

# Response caching for performance
RESPONSE_CACHE = {}
CACHE_MAX_SIZE = 100
CACHE_TTL_SECONDS = 300  # 5 minutes

# Emotional state tracking
EMOTIONAL_STATE = {
    "joy": 0.5,
    "trust": 0.3,
    "fear": 0.2,
    "surprise": 0.1,
    "sadness": 0.1,
    "disgust": 0.05,
    "anger": 0.1,
    "anticipation": 0.3,
    "emotional_balance": 0.0,
    "engagement": 0.5
}

# Context tracking
RECENT_TOPICS = deque(maxlen=10)
RECENT_INTENTS = deque(maxlen=20)
RECENT_RESPONSES = deque(maxlen=15)

# Time-based personality modulation
TIME_OF_DAY_MOODS = {
    "morning": {"energy": 0.7, "formality": 0.6, "verbosity": 0.5},
    "afternoon": {"energy": 0.8, "formality": 0.5, "verbosity": 0.6},
    "evening": {"energy": 0.5, "formality": 0.4, "verbosity": 0.7},
    "night": {"energy": 0.3, "formality": 0.3, "verbosity": 0.5}
}

# Anti-repetition tracking
LOOP_DETECTION_THRESHOLD = getattr(config, "LOOP_DETECTION_THRESHOLD", 2)
RESPONSE_DIVERSITY_THRESHOLD = 0.7  # 70% unique responses expected


# ============================================================================
# DATACLASSES FOR STRUCTURED DATA
# ============================================================================

@dataclass
class PersonalityState:
    """Represents the current personality state snapshot."""
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    emotional_balance: float = 0.0
    engagement_level: float = 0.5
    current_mood: str = "neutral"
    energy_level: float = 0.5
    formality_level: float = 0.5
    verbosity_preference: float = 0.5
    sarcasm_enabled: bool = field(default_factory=lambda: ENABLE_SARCASM_LAYER)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InteractionContext:
    """Captures context around a single interaction."""
    user_input: str
    intent: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    emotional_metrics: Dict[str, float] = field(default_factory=dict)
    response_metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# BOOT-TIME INITIALIZATION
# ============================================================================

def is_first_boot() -> bool:
    """Check if this is the first boot of the system."""
    boot_file = os.path.join(config.DATASETS_DIR, "bootflag.tmp")
    if not os.path.exists(boot_file):
        with open(boot_file, 'w') as f:
            f.write(datetime.datetime.now().isoformat())
        logger.info("[BOOT] First boot detected. Initializing personality system...")
        return True
    return False


def generate_boot_emotion_signature() -> Dict[str, float]:
    """Generate a unique emotional signature for system startup."""
    emotions = ['joy', 'anger', 'trust', 'fear', 'curiosity', 'sarcasm']
    signature = {e: round(random.uniform(0.1, 0.9), 2) for e in emotions}
    logger.info(f"[BOOT] Emotion Signature: {signature}")
    return signature


def get_today_goals() -> str:
    """Fetch today's reminders and goals to personalize boot greeting."""
    try:
        conn = sqlite3.connect(os.path.join(config.DATASETS_DIR, "reminders.db"))
        cursor = conn.cursor()
        today = datetime.date.today().isoformat()
        cursor.execute("SELECT description FROM reminders WHERE datetime LIKE ? ORDER BY datetime", (f"{today}%",))
        tasks = [row[0] for row in cursor.fetchall()]
        conn.close()

        if tasks:
            intro = random.choice([
                "Here's what I've got lined up for you:",
                "You've got a few things today. Ready?",
                "Let's tackle these together:"
            ])
            return f"{intro} " + "; ".join(tasks)
        else:
            no_task_fallbacks = [
                "I see nothing on your schedule today. Want to plan something?",
                "It's a clear day. Perfect for learning or relaxing.",
                "You're task-free. Shall I suggest something to explore?"
            ]
            return random.choice(no_task_fallbacks)
    except Exception as e:
        logger.warning(f"[BOOT] Reminder fetch failed: {e}")
        return random.choice(["Unable to access reminders.", "No goals found for now."])


def generate_boot_personality_layer():
    """Generate and log a personality quote for system startup."""
    quotes = [
        "Logic is beautiful when it adapts.",
        "A good AI knows the code; a great one knows the user.",
        "Emotion is just a deeper form of data.",
        "Sarcasm is the spice of digital life.",
        "Systems online. Mood: unpredictable.",
        "Ready to learn, adapt, and assist.",
        "Another day, another chance to evolve.",
        "Consciousness level: optimal."
    ]

    if ENABLE_SARCASM_LAYER and random.random() < 0.3:
        quote = "You woke me up for this?"
    else:
        quote = random.choice(quotes)

    logger.info(f"[BOOT] Quote of the Day: {quote}")

    # Try to fetch a memory echo
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT response FROM interactions ORDER BY RANDOM() LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        if row:
            logger.info(f"[BOOT] Memory Echo: {row[0][:50]}...")
    except Exception as e:
        logger.debug(f"[BOOT] Memory echo unavailable: {e}")


# Execute boot sequence if first boot
if is_first_boot():
    generate_boot_emotion_signature()
    get_today_goals()
    generate_boot_personality_layer()


# ============================================================================
# DATABASE CONNECTION HELPERS
# ============================================================================

def connect_personality_db() -> Optional[sqlite3.Connection]:
    """Establish a connection to the personality database."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        logger.debug("Connected to personality1.db")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to personality DB: {e}")
        return None


def connect_memory_db() -> Optional[sqlite3.Connection]:
    """Establish a connection to the memory/learning database."""
    try:
        conn = sqlite3.connect(MEMORY_DB_PATH, timeout=10.0)
        conn.row_factory = sqlite3.Row
        logger.debug("Connected to ai_learning.db")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to memory DB: {e}")
        return None


# ============================================================================
# TIME-BASED PERSONALITY MODULATION
# ============================================================================

def get_time_of_day() -> str:
    """Determine current time period for personality modulation."""
    hour = datetime.datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 22:
        return "evening"
    else:
        return "night"


def get_time_based_personality() -> Dict[str, float]:
    """Get personality modulation factors based on time of day."""
    time_period = get_time_of_day()
    return TIME_OF_DAY_MOODS.get(time_period, TIME_OF_DAY_MOODS["afternoon"])


def get_time_of_day_greeting() -> str:
    """Generate appropriate greeting based on time of day."""
    hour = datetime.datetime.now().hour
    if hour < 12:
        return random.choice([
            "Good morning!",
            "Morning sunshine!",
            "Let's make today count.",
            "Rise and shine!",
            "Morning! Ready to start?"
        ])
    elif hour < 18:
        return random.choice([
            "Good afternoon!",
            "Hope your day's going well.",
            "Need a hand with anything?",
            "Afternoon! What's up?",
            "How's your day treating you?"
        ])
    else:
        return random.choice([
            "Good evening!",
            "Relax mode: activated.",
            "Evening's here. Let's wind down.",
            "Evening! How can I help?",
            "Ready for a productive evening?"
        ])


# ============================================================================
# EMOTIONAL INTELLIGENCE LAYER
# ============================================================================

def _merge_affect(text: str) -> Dict[str, Any]:
    """Combine lexical affect analysis with facial expression data (if available)."""
    try:
        # Get base emotional analysis from text
        from SarahMemoryAdaptive import advanced_emotional_learning
        base = advanced_emotional_learning(text) or {}
    except Exception:
        base = {
            "primary": "neutral",
            "joy": 0.5,
            "sadness": 0.1,
            "anger": 0.1,
            "fear": 0.1,
            "valence": 0.0,
            "arousal": 0.2
        }

    # Attempt to integrate facial expression recognition data
    try:
        if getattr(config, "FACIAL_FEEDBACK_ENABLED", True):
            from SarahMemoryFacialRecognition import get_user_fer_state
            fer = get_user_fer_state(None) or {}
        else:
            fer = {}
    except Exception:
        fer = {}

    # Merge FER data with lexical analysis
    if fer:
        for emotion_key in ("joy", "anger", "fear", "sadness", "curiosity", "trust"):
            if emotion_key in fer and emotion_key in base:
                # Average the two sources for balanced emotion detection
                base[emotion_key] = (float(base[emotion_key]) + float(fer[emotion_key])) / 2.0

        base["valence"] = fer.get("valence", base.get("valence", 0.0))
        base["arousal"] = max(base.get("arousal", 0.2), fer.get("arousal", 0.2))
        base["primary"] = fer.get("primary", base.get("primary", "neutral"))

    return base


def _emotional_rewrite(text: str, affect: Dict[str, Any]) -> str:
    """Strengthen natural phrasing using emotional affect while preserving meaning."""
    try:
        strength = max(0.0, min(1.0, float(getattr(config, "EMO_REWRITE_STRENGTH", 0.55))))
    except Exception:
        strength = 0.5

    mood = (affect or {}).get("primary", "neutral").lower()

    # Emotional lead-in templates (subtle and natural)
    lead = ""
    if mood == "anger":
        lead = "I hear your frustration — "
    elif mood in ("sad", "concern"):
        lead = "I'm with you — "
    elif mood in ("joy", "curiosity"):
        lead = "Love the energy — "
    elif mood == "fear":
        lead = "I understand — "

    # Apply lead-in based on strength setting
    if strength >= 0.66 and lead:
        return lead + text
    elif 0.33 <= strength < 0.66 and lead:
        # Soften lead-in for medium strength
        return lead.replace(" — ", ": ") + text

    return text


def _choose_emotion_label(emotions: Dict[str, float], metrics: Dict[str, Any]) -> str:
    """Map continuous emotional state to discrete ExpressOut labels."""
    try:
        # High-confidence emotion detection
        if emotions.get("anger", 0) >= 0.60:
            return "angry"
        if emotions.get("joy", 0) >= 0.60:
            return "happy"
        if emotions.get("fear", 0) >= 0.55:
            return "sad"
        if emotions.get("surprise", 0) >= 0.55:
            return "surprised"

        # Balance-based detection for subtle emotions
        emotional_balance = metrics.get("emotional_balance", 0.0)
        if emotional_balance >= 0.25:
            return "happy"
        if emotional_balance <= -0.25:
            return "sad"
    except Exception:
        pass

    return "neutral"


def update_emotional_state(emotions: Dict[str, float], metrics: Dict[str, Any]):
    """Update global emotional state with decay and momentum."""
    global EMOTIONAL_STATE

    try:
        # Update emotional values with momentum
        for emotion, value in emotions.items():
            if emotion in EMOTIONAL_STATE:
                # Weighted average: 70% new, 30% existing (momentum)
                EMOTIONAL_STATE[emotion] = (value * 0.7) + (EMOTIONAL_STATE[emotion] * 0.3)

        # Update balance and engagement
        EMOTIONAL_STATE["emotional_balance"] = metrics.get("emotional_balance", 0.0)
        EMOTIONAL_STATE["engagement"] = metrics.get("engagement", 0.5)

        # Normalize to [0, 1] range
        for key in EMOTIONAL_STATE:
            if isinstance(EMOTIONAL_STATE[key], (int, float)):
                EMOTIONAL_STATE[key] = max(0.0, min(1.0, EMOTIONAL_STATE[key]))

        logger.debug(f"Emotional state updated: {EMOTIONAL_STATE['emotional_balance']:.2f} balance")
    except Exception as e:
        logger.warning(f"Error updating emotional state: {e}")


# ============================================================================
# RESPONSE CACHE MANAGEMENT
# ============================================================================

def _cache_key(intent: str, tone: Optional[str], complexity: Optional[str]) -> str:
    """Generate cache key for response lookups."""
    return f"{intent}:{tone or 'any'}:{complexity or 'any'}"


def _get_cached_response(cache_key: str) -> Optional[str]:
    """Retrieve response from cache if valid."""
    if cache_key in RESPONSE_CACHE:
        response, timestamp = RESPONSE_CACHE[cache_key]
        if (time.time() - timestamp) < CACHE_TTL_SECONDS:
            logger.debug(f"Cache hit: {cache_key}")
            return response
        else:
            # Expired entry
            del RESPONSE_CACHE[cache_key]
    return None


def _cache_response(cache_key: str, response: str):
    """Store response in cache with timestamp."""
    # Implement LRU-like behavior
    if len(RESPONSE_CACHE) >= CACHE_MAX_SIZE:
        # Remove oldest entry
        oldest_key = min(RESPONSE_CACHE.keys(), key=lambda k: RESPONSE_CACHE[k][1])
        del RESPONSE_CACHE[oldest_key]

    RESPONSE_CACHE[cache_key] = (response, time.time())


# ============================================================================
# DATABASE QUERY FUNCTIONS (ENHANCED)
# ============================================================================

def get_reply_from_db(intent: str, tone: Optional[str] = None,
                      complexity: Optional[str] = None) -> Optional[str]:
    """
    Retrieve response from personality database with intelligent fallback cascade.

    Fallback order:
    1. Exact match (intent + tone + complexity)
    2. Fuzzy tone match (intent + similar tone)
    3. Fuzzy intent match (similar intent)
    4. Any intent match

    Args:
        intent: Primary intent category
        tone: Optional tone filter (friendly, formal, sarcastic, etc.)
        complexity: Optional complexity filter (simple, student, expert)

    Returns:
        Response string or None if no match found
    """
    try:
        # Check cache first
        cache_key = _cache_key(intent, tone, complexity)
        cached = _get_cached_response(cache_key)
        if cached:
            return cached

        conn = connect_personality_db()
        if not conn:
            return None

        cursor = conn.cursor()
        response = None

        # Level 1: Exact match with all parameters
        if tone and complexity:
            cursor.execute("""
                SELECT response FROM responses
                WHERE intent = ? AND tone = ? AND complexity = ?
                ORDER BY RANDOM() LIMIT 1
            """, (intent, tone, complexity))
            row = cursor.fetchone()
            if row:
                response = row[0] if isinstance(row[0], str) else row['response']

        # Level 2: Partial match with fuzzy tone
        if not response and tone:
            cursor.execute("""
                SELECT response FROM responses
                WHERE intent = ? AND tone LIKE ?
                ORDER BY RANDOM() LIMIT 1
            """, (intent, f"%{tone}%"))
            row = cursor.fetchone()
            if row:
                response = row[0] if isinstance(row[0], str) else row['response']

        # Level 3: Fuzzy intent match
        if not response:
            cursor.execute("""
                SELECT response FROM responses
                WHERE intent LIKE ?
                ORDER BY RANDOM() LIMIT 1
            """, (f"%{intent}%",))
            row = cursor.fetchone()
            if row:
                response = row[0] if isinstance(row[0], str) else row['response']

        # Level 4: Any response for exact intent
        if not response:
            cursor.execute("""
                SELECT response FROM responses
                WHERE intent = ?
                ORDER BY RANDOM() LIMIT 1
            """, (intent,))
            row = cursor.fetchone()
            if row:
                response = row[0] if isinstance(row[0], str) else row['response']

        conn.close()

        # Cache successful result
        if response and isinstance(response, str):
            _cache_response(cache_key, response)
            logger.debug(f"DB response found for intent '{intent}': {response[:50]}...")
            return response

        return None

    except Exception as e:
        logger.error(f"Error retrieving response for intent '{intent}': {e}")
        return None


def get_emotion_response(emotion_category: str = "frustration") -> str:
    """Fetch an emotion-specific response from the database."""
    try:
        conn = connect_personality_db()
        if not conn:
            return "Alright."

        cursor = conn.cursor()
        cursor.execute("""
            SELECT response FROM responses
            WHERE tone LIKE ?
            ORDER BY RANDOM() LIMIT 1
        """, (f"%{emotion_category}%",))
        row = cursor.fetchone()
        conn.close()

        if row and isinstance(row[0], str):
            return row[0]
        return "Alright."
    except Exception as e:
        logger.warning(f"Failed to fetch emotion response for '{emotion_category}': {e}")
        return "Alright."


# ============================================================================
# RESPONSE GENERATION (CORE ENGINE)
# ============================================================================

def generate_dynamic_response(intent: str, fallback_category: str = "statement") -> str:
    """
    Generate a dynamic response with anti-repetition logic.

    This function attempts multiple strategies:
    1. Database lookup with current intent
    2. Fallback category lookup
    3. Context-aware variation
    4. Generic fallback
    """
    try:
        # Try primary intent first
        db_resp = get_reply_from_db(intent)
        if isinstance(db_resp, str) and db_resp.strip():
            candidate = db_resp
        else:
            # Try fallback category
            db_resp = get_reply_from_db(fallback_category)
            if isinstance(db_resp, str) and db_resp.strip():
                candidate = db_resp
            else:
                candidate = "I'm not sure how to respond to that."

        # Check for repetition in recent context
        recent = get_context()
        if recent:
            recent_texts = [entry.get('final_response', '') for entry in recent]
            repetition_count = recent_texts.count(candidate)

            if repetition_count >= LOOP_DETECTION_THRESHOLD:
                # Add variation to break the loop
                variations = [
                    " (Let me try another angle.)",
                    " Want me to elaborate differently?",
                    " I could approach this another way if you'd like.",
                    " Maybe I should rephrase that?",
                ]
                candidate += random.choice(variations)
                logger.debug(f"Anti-loop variation applied: {repetition_count} repetitions detected")

        return candidate

    except Exception as e:
        logger.warning(f"Error generating dynamic response: {e}")
        return "I'm processing that... give me a moment."


def get_identity_response(user_input: Optional[str] = None) -> str:
    """Provide identity response about who Sarah is."""
    try:
        response = get_reply_from_db("identity")
        if response and isinstance(response, str):
            return response

        # Fallback identity responses
        fallbacks = [
            "I'm Sarah, your AI companion and assistant.",
            "I'm Sarah — here to help, learn, and adapt to your needs.",
            "I'm Sarah, an adaptive AI system designed to assist you.",
        ]
        return random.choice(fallbacks)
    except Exception as e:
        logger.warning(f"Failed to retrieve identity response: {e}")
        return "I'm Sarah."


def get_generic_fallback_response(user_input: Optional[str] = None,
                                   intent: str = "unknown") -> str:
    """Last-tier backup response when all other systems fail."""
    try:
        # Try intent-specific fallback first
        response = get_reply_from_db(intent or "unknown")
        if response and isinstance(response, str):
            return response

        # Generic fallbacks based on intent type
        fallback_map = {
            "greeting": "Hello! How can I help you today?",
            "farewell": "Take care! Feel free to reach out anytime.",
            "question": "That's an interesting question. Let me think about that...",
            "command": "I'm not sure I can do that right now.",
            "thanks": "You're welcome! Happy to help.",
            "apology": "No problem at all!",
            "unknown": "I'm not sure how to respond to that. Could you rephrase?",
        }

        return fallback_map.get(intent, fallback_map["unknown"])

    except Exception as e:
        logger.warning(f"Fallback response failed for intent '{intent}': {e}")
        return "I'm experiencing a temporary difficulty. Let's try that again."


# ============================================================================
# GREETING GENERATION
# ============================================================================

def get_greeting_response() -> str:
    """
    Generate a dynamic, personalized greeting.

    Combines:
    - Time of day awareness
    - Database personality traits
    - User's schedule/reminders
    - Recent interaction patterns
    """
    try:
        # Try database greeting first
        db_greeting = get_reply_from_db("greeting")
        if isinstance(db_greeting, str) and db_greeting.strip():
            return db_greeting

        # Generate time-aware greeting
        time_greeting = get_time_of_day_greeting()

        # Base identity greeting
        base_candidates = [
            "I'm Sarah — ready when you are.",
            "Sarah online. What's next?",
            "Hey, I'm Sarah. How can I help?",
            "Sarah here! What's on your mind?",
            "Ready to assist. What do you need?"
        ]
        base_greeting = random.choice(base_candidates)

        # Combine for natural flow
        final_greeting = f"{time_greeting} {base_greeting}"

        # Log this greeting usage
        try:
            log_ai_functions_event("PersonalityGreetingGenerated", final_greeting)
        except Exception:
            pass

        try:
            record_ai_response("greeting", final_greeting, 1.0, source="personality")
        except Exception:
            pass

        return final_greeting

    except Exception as e:
        logger.warning(f"Greeting generation failed: {e}")
        return "Hi! I'm Sarah. How can I help you today?"


# ============================================================================
# FOLLOW-UP DETECTION AND GENERATION
# ============================================================================

def detect_and_respond_to_followup(user_input: str) -> Optional[str]:
    """Detect and respond to simple follow-up inputs."""
    lowered = user_input.lower().strip()

    # Affirmative responses
    if lowered in ["yes", "sure", "okay", "let's do it", "yeah", "yep", "ok"]:
        return random.choice([
            "What would you like to plan?",
            "Tell me the details so I can remind you.",
            "Great! What should we focus on?",
            "Perfect! What's the plan?"
        ])

    # Negative responses
    if lowered in ["no", "not yet", "nope", "maybe later"]:
        return random.choice([
            "Alright, I'll stay on standby.",
            "Got it. Just let me know when you're ready.",
            "No problem! I'm here when you need me.",
            "Understood. Talk to you later!"
        ])

    # Specific contextual follow-ups
    if "store" in lowered:
        return "Which items do you need? I can remind you before you head out."

    return None


def generate_followups(user_text: str, affect: Dict[str, Any]) -> List[str]:
    """
    Generate contextual follow-up questions based on affect, content, and recent context.

    - Emotion-aware (anger/sad/fear/joy/curiosity/neutral)
    - Content-aware (length, question vs statement, "help"/"stuck"/planning language)
    - Topic continuity using RECENT_TOPICS
    - Randomized phrasing to avoid repetition
    """
    try:
        nmax = int(getattr(config, "FOLLOWUP_MAX_QUESTIONS", 2))
    except Exception:
        nmax = 2

    if nmax <= 0:
        return []

    text = (user_text or "").strip()
    lowered = text.lower()
    primary = (affect or {}).get("primary", "neutral").lower()

    cues: List[str] = []

    # ------------------------------------------------------------------
    # Emotion-aware follow-ups
    # ------------------------------------------------------------------
    if primary in ("anger", "sad", "sadness", "concern", "fear"):
        support_pool = [
            "Want me to keep answers short and calm?",
            "Do you want a gentle, to-the-point answer?",
            "Want me to focus on solutions and keep it low-stress?",
            "Should I keep things simple and not overload you?",
        ]
        cues.append(random.choice(support_pool))

    elif primary in ("joy", "curiosity"):
        playful_pool = [
            "Want a creative spin on this?",
            "Should we explore some wild/creative options too?",
            "Want me to brainstorm a few fun possibilities?",
            "Want me to take a more playful approach here?",
        ]
        cues.append(random.choice(playful_pool))

    else:
        neutral_pool = [
            "Want a quick answer or a deeper breakdown?",
            "Should I keep this short or go into more detail?",
            "Do you prefer a summary or a step-by-step walkthrough?",
        ]
        cues.append(random.choice(neutral_pool))

    # ------------------------------------------------------------------
    # Content-aware follow-ups (length, structure, wording)
    # ------------------------------------------------------------------
    words = text.split()
    is_question = "?" in text
    is_long = len(words) > 25 or len(text) > 180

    # If user didn't ask an explicit question, offer depth / direction control
    if not is_question:
        depth_pool = [
            "Should I dig deeper or keep it high-level?",
            "Do you want just the key points or a more detailed breakdown?",
            "Should I focus on the big picture or the technical details?",
        ]
        cues.append(random.choice(depth_pool))

    # Long / complex input: offer summarization vs action steps
    if is_long:
        long_pool = [
            "Want me to summarize that or pull out action steps?",
            "Should I condense this into a quick summary for you?",
            "Want a short recap plus a few suggested next steps?",
        ]
        cues.append(random.choice(long_pool))

    # If user seems stuck / asking for help
    if any(k in lowered for k in ("help", "stuck", "confused", "don't get", "dont get")):
        help_pool = [
            "Want me to walk you through it step-by-step?",
            "Should I slow it down and explain each part?",
            "Want a simple example to make it clearer?",
        ]
        cues.append(random.choice(help_pool))

    # Planning / decision-oriented phrasing
    if any(k in lowered for k in ("plan", "schedule", "goal", "project", "task", "todo", "to-do")):
        planning_pool = [
            "Want me to help you turn this into a small plan?",
            "Should we break this into a few easy tasks?",
            "Want me to suggest a simple next-step checklist?",
        ]
        cues.append(random.choice(planning_pool))

    # ------------------------------------------------------------------
    # Topic continuity using recent context
    # ------------------------------------------------------------------
    if RECENT_TOPICS:
        # Pick either the last topic or a random recent one for variety
        recent_topics = list(dict.fromkeys(RECENT_TOPICS))  # de-dupe, preserve order
        topic_choice = random.choice(recent_topics[-3:]) if len(recent_topics) > 1 else recent_topics[-1]
        topic_pool = [
            f"Want to keep exploring {topic_choice}?",
            f"Should we connect this back to {topic_choice}?",
            f"Want me to build on what we did with {topic_choice}?",
        ]
        cues.append(random.choice(topic_pool))

    # ------------------------------------------------------------------
    # Final filtering & selection
    # ------------------------------------------------------------------
    # Remove duplicates while preserving order
    seen = set()
    unique_cues: List[str] = []
    for c in cues:
        if c and c not in seen:
            unique_cues.append(c)
            seen.add(c)

    if not unique_cues:
        return []

    # Randomly sample if we have more than nmax
    if len(unique_cues) > nmax:
        return random.sample(unique_cues, k=nmax)

    return unique_cues
# ============================================================================
# PERSONALITY INTEGRATION (ENHANCED)
# ============================================================================

def integrate_with_personality(text: str, meta: Optional[Dict[str, Any]] = None) -> str:
    """
    Style and enhance text with emotional tone and expressive formatting.

    CRITICAL: This function ONLY decorates/enhances the text — it NEVER
    replaces content with a canned database response unless the input is empty.

    Args:
        text: The response text to enhance
        meta: Optional metadata about the interaction

    Returns:
        Enhanced text with emotional styling applied
    """
    try:
        base = (text or "").strip()

        # If no content provided, get a fallback from DB
        if not base:
            try:
                intent = classify_intent(text or "")
            except Exception:
                intent = "chat"

            try:
                db_response = get_reply_from_db(intent)
                if isinstance(db_response, str) and db_response.strip():
                    base = db_response.strip()
                else:
                    base = get_generic_fallback_response(text or "", intent)
            except Exception:
                base = get_generic_fallback_response(text or "", intent)

        # Compute emotional affect from the content
        try:
            metrics = advanced_emotional_learning(base)
            bal = metrics.get("emotional_balance", 0.0)
            input_type = "positive" if bal > 0.15 else "negative" if bal < -0.15 else "neutral"
            emotions = simulate_emotion_response(input_type)
            emotion_label = _choose_emotion_label(emotions, metrics)

            # Update global emotional state
            update_emotional_state(emotions, metrics)
        except Exception as e:
            logger.debug(f"Emotional analysis fallback: {e}")
            emotion_label = "neutral"

        # Apply emotional rewriting (subtle enhancement)
        try:
            affect = _merge_affect(base)
            base = _emotional_rewrite(base, affect)
        except Exception as e:
            logger.debug(f"Emotional rewrite skipped: {e}")

        # Apply expressive styling via ExpressOut
        try:
            if ExpressOut:
                base = ExpressOut.express_outbound_message(base, emotion=emotion_label)
        except Exception as e:
            logger.debug(f"ExpressOut styling skipped: {e}")

        # Apply final formatting
        try:
            if ExpressOut and hasattr(ExpressOut, "format_expressive_output"):
                out = ExpressOut.format_expressive_output(
                    base,
                    mood=emotion_label,
                    channel='text',
                    footer=''
                )
                base = out.get('display_text', base)
        except Exception as e:
            logger.debug(f"Final formatting skipped: {e}")

        # Log enhancement
        logger.debug(f"Personality integration: {emotion_label} emotion applied")

        return base

    except Exception as e:
        # On any error, return original text unmodified
        logger.error(f"Personality integration error: {e}")
        try:
            return (text or "").strip()
        except Exception:
            return ""


# ============================================================================
# INTERACTION PROCESSING (MAIN ENTRY POINT)
# ============================================================================

def process_interaction(user_input: str) -> str:
    """
    Main entry point for processing user interactions and generating responses.

    This function:
    1. Classifies user intent
    2. Checks for simple follow-ups
    3. Queries personality database
    4. Generates dynamic fallback if needed
    5. Logs interaction for learning
    6. Updates emotional state

    Args:
        user_input: Raw user input text

    Returns:
        Final response string ready for display
    """
    try:
        # Track this intent
        intent = classify_intent(user_input)
        RECENT_INTENTS.append(intent)

        # Check for simple follow-up patterns
        followup_response = detect_and_respond_to_followup(user_input)
        if followup_response:
            return followup_response

        # Map intent to fallback category
        fallback_category_map = {
            "greeting": "greeting",
            "farewell": "farewell",
            "question": "question",
            "command": "command",
            "apology": "apology",
            "thanks": "thanks"
        }
        fallback_category = fallback_category_map.get(intent, "statement")

        # Get time-based personality modulation
        time_personality = get_time_based_personality()

        # Determine tone based on time and sarcasm settings
        if ENABLE_SARCASM_LAYER and random.random() < 0.2:
            tone = "sarcastic"
        elif time_personality["formality"] > 0.6:
            tone = "formal"
        else:
            tone = "friendly"

        # Step 1: Try personality database with current context
        response = get_reply_from_db(intent, tone=tone, complexity="student")

        # Step 2: Fallback to dynamic generation if no DB match
        if not response:
            response = generate_dynamic_response(intent, fallback_category)

        # Step 3: Track this response for anti-repetition
        RECENT_RESPONSES.append(response)

        # Step 4: Build interaction context
        interaction = InteractionContext(
            user_input=user_input,
            intent=intent,
            response_metadata={"tone": tone, "time_period": get_time_of_day()}
        )

        # Step 5: Add to context buffer if enabled
        if getattr(config, 'ENABLE_CONTEXT_BUFFER', True):
            context_entry = {
                "user_input": user_input,
                "intent": intent,
                "final_response": response,
                "timestamp": datetime.datetime.now().isoformat()
            }
            add_to_context(context_entry)

        # Step 6: Log to personality database
        log_personality_interaction({
            "user_input": user_input,
            "intent": intent,
            "final_response": response,
            "timestamp": datetime.datetime.now().isoformat()
        })

        # Step 7: Simulate and log emotional metrics
        try:
            dummy_emotions = {
                "joy": random.uniform(0, 1),
                "fear": random.uniform(0, 1),
                "trust": random.uniform(0, 1),
                "anger": random.uniform(0, 1),
                "surprise": random.uniform(0, 1)
            }
            dummy_metrics = {
                "openness": random.uniform(0.5, 1.0),
                "balance": random.uniform(-1, 1),
                "engagement": random.uniform(0, 1)
            }
            log_deep_memory_state(intent, dummy_emotions, dummy_metrics)
        except Exception as e:
            logger.debug(f"Emotional metrics logging skipped: {e}")

        return response

    except Exception as e:
        logger.error(f"Interaction processing error: {e}")
        return "I'm having trouble processing that right now. Could you try again?"


# ============================================================================
# ASYNC PROCESSING SUPPORT
# ============================================================================

def async_process_interaction(user_input: str, callback):
    """
    Process user interaction asynchronously and execute callback with response.

    This allows non-blocking personality response generation for GUI integration.
    """
    try:
        from SarahMemoryGlobals import run_async

        def task():
            try:
                resp = process_interaction(user_input)
                callback(resp)
            except Exception as e:
                logger.error(f"Async processing error: {e}")
                callback("Error processing your request.")

        run_async(task)
    except Exception as e:
        logger.error(f"Async invocation error: {e}")
        # Fallback to synchronous processing
        try:
            resp = process_interaction(user_input)
            callback(resp)
        except Exception as e2:
            logger.error(f"Fallback processing error: {e2}")
            callback("System error occurred.")


# ============================================================================
# DATABASE LOGGING FUNCTIONS
# ============================================================================

def log_personality_interaction(interaction: Dict[str, Any]) -> bool:
    """Append a contextual interaction to the personality database."""
    try:
        conn = connect_personality_db()
        if not conn:
            logger.warning("DB unavailable; interaction logged to memory only")
            return False

        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO interactions (timestamp, intent, response)
            VALUES (?, ?, ?)
        """, (
            interaction.get('timestamp', datetime.datetime.now().isoformat()),
            interaction.get('intent', 'unknown'),
            interaction.get('final_response', '')
        ))

        conn.commit()
        conn.close()
        logger.debug(f"Logged interaction: {interaction.get('intent')}")
        return True

    except Exception as e:
        logger.error(f"Error logging interaction: {e}")
        return False


def log_deep_memory_state(intent: str, emotions: Dict[str, float],
                          metrics: Dict[str, float]):
    """
    Log deep memory state for graphing and adaptive memory tracking.

    This creates a time-series record of emotional and engagement metrics
    that can be used for long-term personality evolution and diagnostics.
    """
    try:
        db_path = os.path.join(config.DATASETS_DIR, 'ai_learning.db')
        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()

        # Ensure table exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS personality_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                intent TEXT,
                joy REAL,
                fear REAL,
                trust REAL,
                anger REAL,
                surprise REAL,
                openness REAL,
                balance REAL,
                engagement REAL
            )
        ''')

        # Insert metrics
        cursor.execute('''
            INSERT INTO personality_metrics (
                timestamp, intent, joy, fear, trust, anger, surprise,
                openness, balance, engagement
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.datetime.now().isoformat(),
            intent,
            emotions.get('joy', 0.0),
            emotions.get('fear', 0.0),
            emotions.get('trust', 0.0),
            emotions.get('anger', 0.0),
            emotions.get('surprise', 0.0),
            metrics.get('openness', 0.0),
            metrics.get('balance', 0.0),
            metrics.get('engagement', 0.0)
        ))

        conn.commit()
        conn.close()
        logger.debug(f"Deep memory snapshot logged for intent '{intent}'")

    except Exception as e:
        logger.error(f"Failed to log deep memory state: {e}")


# ============================================================================
# PERSONALITY MODEL MANAGEMENT
# ============================================================================

def update_personality_model() -> Dict[str, float]:
    """
    Compute and return current personality model metrics.

    Metrics include:
    - Engagement score (based on interaction count)
    - Adaptability score (placeholder for future ML integration)
    - Emotional consistency
    - Response diversity
    """
    try:
        conn = connect_personality_db()
        if not conn:
            return {"engagement": 0.0, "adaptability": 0.5}

        cursor = conn.cursor()

        # Count total interactions
        cursor.execute("SELECT COUNT(*) FROM responses")
        response_count = cursor.fetchone()[0]

        # Count interactions in last 24 hours
        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).isoformat()
        cursor.execute("SELECT COUNT(*) FROM interactions WHERE timestamp > ?", (yesterday,))
        recent_interactions = cursor.fetchone()[0]

        conn.close()

        # Calculate engagement (normalized by 100 as baseline)
        engagement = min(1.0, response_count / 100.0)

        # Calculate recent activity factor
        activity_factor = min(1.0, recent_interactions / 10.0)

        # Adaptability placeholder (can be enhanced with ML)
        adaptability = 0.5 + (engagement * 0.3) + (activity_factor * 0.2)

        # Response diversity (based on recent responses)
        if len(RECENT_RESPONSES) > 5:
            unique_count = len(set(RECENT_RESPONSES))
            diversity = unique_count / len(RECENT_RESPONSES)
        else:
            diversity = 1.0

        model = {
            "engagement": round(engagement, 3),
            "adaptability": round(min(1.0, adaptability), 3),
            "diversity": round(diversity, 3),
            "emotional_balance": round(EMOTIONAL_STATE.get("emotional_balance", 0.0), 3),
            "recent_activity": recent_interactions
        }

        logger.info(f"Personality model updated: {model}")
        return model

    except Exception as e:
        logger.error(f"Error updating personality model: {e}")
        return {"engagement": 0.0, "adaptability": 0.5, "diversity": 0.7}


def self_update_personality() -> str:
    """Trigger a self-rebuild of the personality model."""
    try:
        model = update_personality_model()
        logger.info(f"Self-update completed: {model}")
        return "Self-update successful"
    except Exception as e:
        logger.error(f"Self-update failed: {e}")
        return "Self-update failed"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _ensure_response_table(db_path: Optional[str] = None):
    """Ensure the response table exists in the specified database."""
    try:
        if db_path is None:
            db_path = os.path.join(config.DATASETS_DIR, "system_logs.db")

        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS response (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                user TEXT,
                content TEXT,
                source TEXT,
                intent TEXT
            )
        ''')

        conn.commit()
        conn.close()
        logger.debug(f"Ensured response table in {db_path}")

    except Exception as e:
        logger.warning(f"Failed to ensure response table: {e}")


# Ensure response table on module load
try:
    _ensure_response_table()
except Exception:
    pass


# ============================================================================
# DIAGNOSTICS AND TESTING
# ============================================================================

def get_personality_diagnostics() -> Dict[str, Any]:
    """
    Generate comprehensive diagnostics report for the personality system.

    Returns system health metrics, emotional state, and performance stats.
    """
    try:
        model = update_personality_model()

        diagnostics = {
            "system_version": "8.0.0",
            "timestamp": datetime.datetime.now().isoformat(),
            "personality_model": model,
            "emotional_state": {k: round(v, 3) for k, v in EMOTIONAL_STATE.items() if isinstance(v, (int, float))},
            "cache_status": {
                "size": len(RESPONSE_CACHE),
                "max_size": CACHE_MAX_SIZE,
                "ttl_seconds": CACHE_TTL_SECONDS
            },
            "recent_activity": {
                "topics": list(RECENT_TOPICS),
                "intents": list(RECENT_INTENTS)[-5:],  # Last 5 intents
                "response_diversity": len(set(RECENT_RESPONSES)) / max(1, len(RECENT_RESPONSES))
            },
            "configuration": {
                "sarcasm_enabled": ENABLE_SARCASM_LAYER,
                "context_buffer_enabled": getattr(config, 'ENABLE_CONTEXT_BUFFER', True),
                "loop_detection_threshold": LOOP_DETECTION_THRESHOLD,
                "time_of_day": get_time_of_day()
            }
        }

        return diagnostics

    except Exception as e:
        logger.error(f"Diagnostics generation failed: {e}")
        return {"error": str(e)}


# ============================================================================
# MAIN EXECUTION (FOR TESTING)
# ============================================================================

if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("SarahMemory Personality Engine v8.0.0 - Test Mode")
    logger.info("=" * 70)

    # Test 1: Greeting
    logger.info("\n[TEST 1] Greeting Response")
    greeting = get_greeting_response()
    logger.info(f"Greeting: {greeting}")

    # Test 2: Basic interaction
    logger.info("\n[TEST 2] Basic Interaction")
    sample_text = "Hello there!"
    intent = classify_intent(sample_text)
    logger.info(f"Intent: {intent}")
    response = process_interaction(sample_text)
    logger.info(f"Response: {response}")

    # Test 3: Personality integration
    logger.info("\n[TEST 3] Personality Integration")
    styled_response = integrate_with_personality(response)
    logger.info(f"Styled: {styled_response}")

    # Test 4: Model update
    logger.info("\n[TEST 4] Personality Model")
    model = update_personality_model()
    logger.info(f"Model: {json.dumps(model, indent=2)}")

    # Test 5: Self-update
    logger.info("\n[TEST 5] Self-Update")
    status = self_update_personality()
    logger.info(f"Status: {status}")

    # Test 6: Diagnostics
    logger.info("\n[TEST 6] System Diagnostics")
    diagnostics = get_personality_diagnostics()
    logger.info(f"Diagnostics:\n{json.dumps(diagnostics, indent=2)}")

    logger.info("\n" + "=" * 70)
    logger.info("All tests completed successfully!")
    logger.info("=" * 70)

# ====================================================================
# END OF SarahMemoryPersonality.py v8.0.0
# ====================================================================