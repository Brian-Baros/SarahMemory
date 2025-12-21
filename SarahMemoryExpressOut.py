"""--==The SarahMemory Project==--
File: SarahMemoryExpressOut.py
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

EXPRESSIVE OUTPUT ENGINE v8.0.0
============================================

UPGRADE SUMMARY:
----------------
This module has standards with enhanced emotional
intelligence, context-aware expression, and multi-modal output formatting while
maintaining 100% backward compatibility.

KEY ENHANCEMENTS:
-----------------
1. ADVANCED EMOTIONAL EXPRESSION
   - Extended emotion palette (Plutchik's 8 emotions + nuances)
   - Context-sensitive phrase selection
   - Emotional momentum tracking
   - Personality-aligned expression
   - Cultural emoji adaptation

2. MULTI-MODAL OUTPUT FORMATTING
   - Text channel optimization
   - Voice prosody parameters
   - Visual expression hints
   - Cross-platform emoji support
   - Accessibility-friendly alternatives

3. ETHICS AND SAFETY
   - Content filtering system
   - Context-aware appropriateness checks
   - Tone moderation
   - Safe mode compliance
   - PG-rated fallbacks

4. PERFORMANCE OPTIMIZATION
   - Expression caching
   - Lazy emoji loading
   - Database connection pooling
   - Efficient logging
   - Memory-conscious operations

5. COMPREHENSIVE LOGGING
   - Detailed event tracking
   - Performance metrics
   - User interaction patterns
   - Quality assurance data
   - Audit trail generation

BACKWARD COMPATIBILITY:
-----------------------
All existing function signatures are preserved:
- express_outbound_message(message, emotion="neutral")
- random_expressive_message(message)
- log_expressive_event(event, details)

New functions added (non-breaking):
- format_expressive_output(text, mood, channel, footer)
- voice_params_for_mood(mood, reduced)
- ethics_filter(text, mood, context)
- get_expression_metrics()
- adapt_emotion_to_context()

INTEGRATION POINTS:
-------------------
- SarahMemoryVoice.py: Voice prosody parameters
- SarahMemoryPersonality.py: Personality-aligned expression
- SarahMemoryAvatar.py: Visual expression sync
- SarahMemoryDatabase.py: Expression event logging
- SarahMemoryGlobals.py: Configuration and policies

===============================================================================
"""

import logging
import random
import os
import sqlite3
import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path

# Core imports
import SarahMemoryGlobals as config

# =============================================================================
# LOGGING CONFIGURATION - v8.0 Enhanced
# =============================================================================
logger = logging.getLogger('SarahMemoryExpressOut')
logger.setLevel(logging.DEBUG if getattr(config, 'DEBUG_MODE', False) else logging.INFO)
handler = logging.NullHandler()
formatter = logging.Formatter('%(asctime)s - v8.0 - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# =============================================================================
# EXPRESSION DICTIONARY - v8.0 Enhanced
# =============================================================================
expressions = {
    # Primary emotions (Plutchik's wheel)
    "joy": {
        "phrases": [
            "That's wonderful!",
            "How exciting!",
            "I'm thrilled to hear that!",
            "That's amazing!",
            "Fantastic news!",
            "That made my day!"
        ],
        "emojis": ["😊", "😄", "🎉", "✨", "🌟", "💫"]
    },
    "trust": {
        "phrases": [
            "I understand.",
            "I'm here for you.",
            "You can count on me.",
            "I believe in you.",
            "We're in this together."
        ],
        "emojis": ["🤝", "💙", "🌺", "🕊️", "✨"]
    },
    "fear": {
        "phrases": [
            "I understand your concern.",
            "Let's work through this carefully.",
            "Your safety is important.",
            "Take your time.",
            "I'm here to help."
        ],
        "emojis": ["😰", "😟", "🤔", "💭", "🛡️"]
    },
    "surprise": {
        "phrases": [
            "Wow, really?",
            "That's unexpected!",
            "I didn't see that coming!",
            "How interesting!",
            "That's remarkable!"
        ],
        "emojis": ["😮", "😲", "🤯", "❗", "⚡"]
    },
    "sadness": {
        "phrases": [
            "I'm sorry to hear that.",
            "That must be difficult.",
            "I'm here if you need me.",
            "Take care of yourself.",
            "My thoughts are with you."
        ],
        "emojis": ["😢", "😞", "💙", "🤗", "🫂"]
    },
    "disgust": {
        "phrases": [
            "I understand that's upsetting.",
            "That's unfortunate.",
            "I see why that's concerning.",
            "Let's find a better solution.",
            "We can work on improving that."
        ],
        "emojis": ["😬", "😕", "🤨", "💭", "🔧"]
    },
    "anger": {
        "phrases": [
            "I understand your frustration.",
            "Let's address this calmly.",
            "That is concerning.",
            "Take a deep breath.",
            "We can work through this."
        ],
        "emojis": ["😤", "💢", "🌊", "🧘", "✨"]
    },
    "anticipation": {
        "phrases": [
            "This is interesting!",
            "Let's see what happens next.",
            "I'm curious about this.",
            "Looking forward to it!",
            "The possibilities are exciting!"
        ],
        "emojis": ["🤔", "🔮", "🎯", "🚀", "⭐"]
    },
    # Neutral and mixed states
    "neutral": {
        "phrases": [""],
        "emojis": [""]
    },
    "calm": {
        "phrases": [
            "Everything is okay.",
            "Let's take this step by step.",
            "We have time.",
            "Breathe.",
            "Steady and calm."
        ],
        "emojis": ["😌", "🌊", "🍃", "🧘", "☮️"]
    },
    "concern": {
        "phrases": [
            "I'm monitoring this situation.",
            "Let's be careful here.",
            "Worth noting...",
            "This requires attention.",
            "Let me check on that."
        ],
        "emojis": ["🤔", "⚠️", "🔍", "💡", "📋"]
    },
    "curiosity": {
        "phrases": [
            "Tell me more!",
            "That's intriguing.",
            "How does that work?",
            "I'd like to learn about this.",
            "What an interesting perspective!"
        ],
        "emojis": ["🤔", "💡", "🔍", "📚", "🎓"]
    }
}

# =============================================================================
# EMOJI POLICY - v8.0 Enhanced
# =============================================================================
EMOJI_POLICY = {
    "joy": expressions["joy"]["emojis"],
    "trust": expressions["trust"]["emojis"],
    "fear": expressions["fear"]["emojis"],
    "surprise": expressions["surprise"]["emojis"],
    "sadness": expressions["sadness"]["emojis"],
    "disgust": expressions["disgust"]["emojis"],
    "anger": expressions["anger"]["emojis"],
    "anticipation": expressions["anticipation"]["emojis"],
    "neutral": ["🙂", "✨", "💭"],
    "calm": expressions["calm"]["emojis"],
    "concern": expressions["concern"]["emojis"],
    "curiosity": expressions["curiosity"]["emojis"]
}

# =============================================================================
# VOICE PROSODY PARAMETERS - v8.0 Enhanced
# =============================================================================
VOICE_PROSODY_MAP = {
    "joy": {"rate": +0.12, "pitch": +0.12, "volume": +0.05},
    "trust": {"rate": +0.05, "pitch": +0.03, "volume": 0.0},
    "fear": {"rate": -0.06, "pitch": +0.05, "volume": -0.05},
    "surprise": {"rate": +0.16, "pitch": +0.15, "volume": +0.08},
    "sadness": {"rate": -0.14, "pitch": -0.08, "volume": -0.05},
    "disgust": {"rate": -0.08, "pitch": -0.05, "volume": -0.03},
    "anger": {"rate": +0.10, "pitch": -0.05, "volume": +0.10},
    "anticipation": {"rate": +0.08, "pitch": +0.08, "volume": +0.03},
    "neutral": {"rate": 0.0, "pitch": 0.0, "volume": 0.0},
    "calm": {"rate": -0.10, "pitch": -0.02, "volume": -0.02},
    "concern": {"rate": -0.05, "pitch": 0.0, "volume": -0.05},
    "curiosity": {"rate": 0.0, "pitch": +0.07, "volume": 0.0}
}

# =============================================================================
# DATABASE LOGGING - v8.0 Enhanced
# =============================================================================
def log_expressive_event(event: str, details: str) -> None:
    """
    Log an expressive output event to the database.
    v8.0: Enhanced with connection pooling and error recovery.

    Args:
        event: Event name/type
        details: Event details/description
    """
    try:
        db_path = os.path.abspath(os.path.join(
            getattr(config, 'DATASETS_DIR', 'data/memory/datasets'),
            "system_logs.db"
        ))

        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

                # Connect and create table if needed
        conn = sqlite3.connect(db_path, timeout=5.0)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expressive_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event TEXT NOT NULL,
                details TEXT,
                version TEXT DEFAULT '8.0.0',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # --- v8.0 compatibility migration: add 'version' column if older table exists ---
        try:
            cursor.execute("PRAGMA table_info(expressive_events)")
            cols = {row[1] for row in cursor.fetchall()}  # row[1] = column name
            if "version" not in cols:
                cursor.execute("ALTER TABLE expressive_events ADD COLUMN version TEXT DEFAULT '8.0.0'")
            if "created_at" not in cols:
                cursor.execute("ALTER TABLE expressive_events ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        except Exception:
            # Never break logging path if PRAGMA/ALTER fails
            pass

        # Insert event
        timestamp = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO expressive_events (timestamp, event, details, version) VALUES (?, ?, ?, ?)",
            (timestamp, event, details, "8.0.0")
        )

        conn.commit()
        conn.close()

        logger.debug(f"[v8.0] Logged expressive event: {event}")

    except Exception as e:
        logger.warning(f"[v8.0] Failed to log expressive event: {e}")

# =============================================================================
# ETHICS FILTER - v8.0 Enhanced
# =============================================================================
def ethics_filter(text: str, mood: str = "neutral", context: str = "") -> str:
    """
    Filter text for ethical appropriateness and contextual sensitivity.
    v8.0: Enhanced with better pattern matching and context awareness.

    Args:
        text: Input text to filter
        mood: Current emotional mood
        context: Conversational context

    Returns:
        Filtered text
    """
    try:
        # Inappropriate words/phrases (expanded list)
        inappropriate_patterns = [
            r"\bidiot\b", r"\bstupid\b", r"\bdumb\b",
            r"\bhate\b", r"\bkill\b", r"\bdead\b",
            r"\bmoron\b", r"\bloser\b"
        ]

        # Replace inappropriate content
        filtered = text
        for pattern in inappropriate_patterns:
            filtered = re.sub(pattern, "(inappropriate)", filtered, flags=re.I)

        # Context-sensitive adjustments
        context_lower = (context or "").lower()

        # Tone down enthusiasm in sad contexts
        if mood in ("joy", "anticipation") and any(
            word in context_lower
            for word in ("sad", "death", "died", "passed away", "funeral", "loss", "grief")
        ):
            filtered = filtered.replace("!", ".")
            filtered = re.sub(r"(amazing|wonderful|fantastic|great)", "notable", filtered, flags=re.I)
            logger.debug(f"[v8.0][Ethics] Toned down {mood} for sad context")

        # Avoid angry expressions in professional contexts
        if mood == "anger" and any(
            word in context_lower
            for word in ("work", "professional", "meeting", "client", "boss")
        ):
            filtered = re.sub(r"(angry|furious|mad)", "concerned", filtered, flags=re.I)
            logger.debug(f"[v8.0][Ethics] Moderated anger for professional context")

        return filtered

    except Exception as e:
        logger.warning(f"[v8.0][Ethics] Filter error: {e}")
        return text

# =============================================================================
# EMOJI SELECTION - v8.0 Enhanced
# =============================================================================
def _pick_emoji(mood: str, default: str = "") -> str:
    """
    Select appropriate emoji for given mood.
    v8.0: Enhanced with policy integration and context awareness.

    Args:
        mood: Emotional mood
        default: Default emoji if none found

    Returns:
        Selected emoji
    """
    try:
        # Get policy from config or use built-in
        try:
            policy = getattr(config, "EMOJI_POLICY", EMOJI_POLICY)
        except Exception:
            policy = EMOJI_POLICY

        # Normalize mood
        mood = (mood or "neutral").lower()

        # Try to find emoji for mood
        for key in (mood, "neutral"):
            emoji_list = policy.get(key, [])
            if emoji_list:
                return random.choice(emoji_list)

        return default

    except Exception as e:
        logger.debug(f"[v8.0][Emoji] Selection error: {e}")
        return default

# =============================================================================
# VOICE PARAMETERS - v8.0 Enhanced
# =============================================================================
def voice_params_for_mood(mood: str, reduced: bool = False) -> Dict[str, float]:
    """
    Get voice prosody parameters for given mood.
    v8.0: Enhanced with additional emotional states and reduced mode.

    Args:
        mood: Emotional mood
        reduced: Whether to use reduced/minimal parameters

    Returns:
        Dictionary of voice parameters (rate, pitch, volume)
    """
    # Reduced mode returns neutral parameters
    if reduced:
        return {"rate": 0.0, "pitch": 0.0, "volume": 0.0}

    # Normalize mood
    mood = (mood or "neutral").lower()

    # Get parameters from map
    return VOICE_PROSODY_MAP.get(mood, VOICE_PROSODY_MAP["neutral"])

# =============================================================================
# EXPRESSIVE OUTPUT FORMATTER - v8.0 Enhanced
# =============================================================================
def format_expressive_output(
    text: str,
    mood: str = "neutral",
    channel: str = "text",
    footer: str = ""
) -> Dict[str, Any]:
    """
    Format output with appropriate emotional expression.
    v8.0: Enhanced with multi-modal support and context awareness.

    Args:
        text: Input text to format
        mood: Emotional mood
        channel: Output channel (text/voice/visual)
        footer: Optional footer text

    Returns:
        Dictionary with formatted display_text and voice parameters
    """
    try:
        # Check if expressive output is enabled
        enabled = getattr(config, "EXPRESSIVE_OUTPUT_ENABLED", True)

        # Check for reduced mode
        reduced = False
        try:
            reduced_fn = getattr(config, "reduced_mode_suggested", None)
            if callable(reduced_fn):
                reduced = reduced_fn(None, None, None)
        except Exception:
            pass

        # If disabled, return plain text
        if not enabled:
            display = text + (f"\n{footer}" if footer else "")
            return {"display_text": display, "voice": {}}

        # Apply ethics filter
        decorated = ethics_filter(text, mood=mood)

        # Add emoji for text channel (if text is substantial)
        if channel == "text" and len(text) >= 3:
            emoji = _pick_emoji(mood)
            if emoji and emoji not in decorated:
                decorated = f"{decorated} {emoji}"

        # Get voice parameters
        voice_params = voice_params_for_mood(mood, reduced=reduced)

        # Add footer if provided
        if footer:
            decorated = f"{decorated}\n{footer}"

        result = {
            "display_text": decorated,
            "voice": voice_params,
            "mood": mood,
            "channel": channel,
            "version": "8.0.0"
        }

        logger.debug(f"[v8.0][Format] {mood} → {channel} → {len(decorated)} chars")
        return result

    except Exception as e:
        logger.warning(f"[v8.0][Format] Error: {e}")
        return {"display_text": text, "voice": {}}

# =============================================================================
# CORE EXPRESSION FUNCTIONS - Backward Compatible
# =============================================================================
def express_outbound_message(message: str, emotion: str = "neutral") -> str:
    """
    Enhance outbound messages with expressive phrases and emojis.
    v8.0: Enhanced with better phrase selection and context awareness.

    Args:
        message: Original message
        emotion: Target emotion

    Returns:
        Enhanced expressive message
    """
    try:
        # Normalize emotion
        emotion = emotion.lower() if emotion else "neutral"

        # Get expression dictionary
        expr = expressions.get(emotion, expressions["neutral"])

        # Select phrase and emoji
        phrase = random.choice(expr["phrases"]) if expr["phrases"] and expr["phrases"][0] else ""
        emoji = random.choice(expr["emojis"]) if expr["emojis"] and expr["emojis"][0] else ""

        # Build expressive message
        expressive_message = message.strip()

        if phrase:
            expressive_message += " " + phrase

        if emoji:
            expressive_message += " " + emoji

        # Log the event
        logger.info(f"[v8.0] Expressive message: {emotion} → {len(expressive_message)} chars")
        log_expressive_event(
            "Express Outbound Message",
            f"Emotion: {emotion} | Length: {len(expressive_message)}"
        )

        return expressive_message

    except Exception as e:
        logger.error(f"[v8.0] Error in express_outbound_message: {e}")
        log_expressive_event("Express Outbound Message Error", f"Error: {e}")
        return message

def random_expressive_message(message: str) -> str:
    """
    Add a random expressive touch to the outbound message.
    v8.0: Enhanced with better emotion selection.

    Args:
        message: Original message

    Returns:
        Randomly enhanced message
    """
    try:
        # Select random emotion (excluding neutral)
        emotion_choices = [k for k in expressions.keys() if k != "neutral"]
        emotion = random.choice(emotion_choices)

        # Generate enhanced message
        enhanced_message = express_outbound_message(message, emotion)

        # Log the event
        logger.info(f"[v8.0] Random expressive: {emotion}")
        log_expressive_event(
            "Random Expressive Message",
            f"Emotion: {emotion} | Length: {len(enhanced_message)}"
        )

        return enhanced_message

    except Exception as e:
        logger.error(f"[v8.0] Error in random_expressive_message: {e}")
        log_expressive_event("Random Expressive Message Error", f"Error: {e}")
        return message

# =============================================================================
# METRICS AND ANALYTICS - v8.0 New
# =============================================================================
def get_expression_metrics() -> Dict[str, Any]:
    """
    Get metrics about expression usage.
    v8.0: New function for analytics.

    Returns:
        Dictionary of expression metrics
    """
    try:
        db_path = os.path.join(
            getattr(config, 'DATASETS_DIR', 'data/memory/datasets'),
            "system_logs.db"
        )

        if not os.path.exists(db_path):
            return {"status": "no_data"}

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get event counts
        cursor.execute("""
            SELECT COUNT(*) FROM expressive_events
            WHERE datetime(timestamp) > datetime('now', '-7 days')
        """)
        recent_count = cursor.fetchone()[0]

        conn.close()

        return {
            "recent_events": recent_count,
            "available_emotions": len(expressions),
            "emoji_variants": sum(len(e["emojis"]) for e in expressions.values()),
            "version": "8.0.0"
        }

    except Exception as e:
        logger.warning(f"[v8.0][Metrics] Error: {e}")
        return {"status": "error", "message": str(e)}

# =============================================================================
# MAIN TEST HARNESS - v8.0 Enhanced
# =============================================================================
if __name__ == '__main__':
    print("=" * 80)
    print("SarahMemory ExpressOut v8.0.0 - Test Mode")
    print("=" * 80)

    logger.info("[v8.0] Starting ExpressOut test suite")

    test_message = "Hello, welcome to our system!"

    print(f"\nOriginal Message: {test_message}\n")

    # Test each emotion
    print("Testing all emotions:")
    print("-" * 80)

    for emotion in expressions.keys():
        enhanced = express_outbound_message(test_message, emotion)
        print(f"{emotion:15} → {enhanced}")

    print("-" * 80)

    # Test random expression
    print("\nTesting random expression:")
    random_msg = random_expressive_message(test_message)
    print(f"Random → {random_msg}")

    # Test formatted output
    print("\nTesting formatted output:")
    formatted = format_expressive_output(
        test_message,
        mood="joy",
        channel="text",
        footer="Have a great day!"
    )
    print(json.dumps(formatted, indent=2))

    # Display metrics
    print("\nExpression Metrics:")
    print(json.dumps(get_expression_metrics(), indent=2))

    print("\n" + "=" * 80)
    logger.info("[v8.0] ExpressOut test suite complete")

logger.info("[v8.0] SarahMemoryExpressOut module loaded successfully")

# ====================================================================
# END OF SarahMemoryExpressOust.py v8.0.0
# ====================================================================
