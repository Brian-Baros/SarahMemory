"""=== SarahMemory Project ===
File: SarahMemoryAdvCU.py
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

ADVANCED CONTEXT UNIT (AdvCU) 
===========================================================

PURPOSE:
--------
The Advanced Context Unit is one of the most powerful core modules in SarahMemory.
It serves as the central intelligence for understanding user intent and enabling
system automation capabilities.

KEY CAPABILITIES:
-----------------
1. INTENT CLASSIFICATION
   - Rule-based fast classification (offline-first)
   - Neural embedding-based classification (when models available)
   - Hybrid scoring with confidence levels
   - 19+ distinct intent categories

2. COMMAND PARSING
   - Action + Subject extraction
   - Math expression detection
   - URL/Domain recognition
   - App/Site identification
   - Structured ParsedCommand output

3. TEXT EMBEDDINGS
   - Sentence Transformer support (local models)
   - Fallback hash-based embeddings (always works offline)
   - Cosine similarity scoring

4. CODE INTROSPECTION
   - AST-based Python source analysis
   - URL/Domain/Email extraction from codebase
   - Prompt template detection
   - Delta-aware database storage

5. AUTOMATION SUPPORT
   - Desktop app launching vocabulary
   - System control commands
   - Window management
   - Agent control phrases

INTEGRATION POINTS:
-------------------
- SarahMemoryPersonality.py: Uses classify_intent
- SarahMemoryReply.py: Uses classify_intent
- SarahMemoryAPI.py: Uses classify_intent
- SarahMemoryAiFunctions.py: Uses classify_intent, embed_text
- SarahMemoryCompare.py: Uses evaluate_similarity, get_vector_score, embed_text
- SarahMemoryResearch.py: Uses classify_intent
- SarahMemorySOBJE.py: Uses sm_color_name_from_rgb
- SarahMemorySystemLearn.py: Uses classify_intent
- SarahMemoryGUI.py: Uses classify_intent
- SarahMemory-local_api_server.py: Uses classify_intent, parse_command
- UnifiedAvatarController.py: Uses module functions

DATABASE TABLES (functions.db):
-------------------------------
- code_corpus: Mined code snippets
- code_corpus_seen: Delta tracking for snippets
- advcu_log: Event logging

===============================================================================
"""

from __future__ import annotations

import re
import os
import json
import math
import time
import logging
import sqlite3
import ast
import io
import sys
import glob
import hashlib
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable
from enum import Enum
from collections import defaultdict

# ============================================================================
# SAFE IMPORTS
# ============================================================================

# NumPy (optional but preferred)
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

# OpenCV (optional for color detection)
try:
    import cv2 as _cv2
    _HAS_CV2 = True
except ImportError:
    _cv2 = None
    _HAS_CV2 = False

# Sentence Transformers (optional for neural embeddings)
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    SentenceTransformer = None
    _HAS_ST = False

# Import globals safely
try:
    import SarahMemoryGlobals as config
    from SarahMemoryGlobals import DATASETS_DIR, MODEL_CONFIG
except ImportError:
    # Fallback for standalone testing
    class config:
        DATASETS_DIR = os.path.join(os.getcwd(), "data", "memory", "datasets")
        BASE_DIR = os.getcwd()
        DEBUG_MODE = True
        LOCAL_ONLY_MODE = False
        SAFE_MODE = False
    DATASETS_DIR = config.DATASETS_DIR
    MODEL_CONFIG = {}


# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger("SarahMemoryAdvCU")
if not logger.handlers:
    logger.setLevel(logging.DEBUG if getattr(config, 'DEBUG_MODE', False) else logging.INFO)
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s] %(message)s"
    ))
    logger.addHandler(_handler)
logger.propagate = False


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

# Database and cache paths
SYSTEM_DB_PATH = os.path.join(DATASETS_DIR, "functions.db")
INTENT_CACHE_PATH = os.path.join(DATASETS_DIR, "intent_override_cache.json")
CODE_CORPUS_JSON = os.path.join(DATASETS_DIR, "code_corpus_cache.json")

# Ensure directories exist
os.makedirs(os.path.dirname(SYSTEM_DB_PATH), exist_ok=True)

# Offline mode detection
_OFFLINE_MODE = bool(
    getattr(config, "LOCAL_ONLY_MODE", False) or
    getattr(config, "SAFE_MODE", False) or
    os.getenv("HF_HUB_OFFLINE") == "1" or
    os.getenv("TRANSFORMERS_OFFLINE") == "1"
)

# Thread lock for model loading
_model_lock = threading.RLock()


# ============================================================================
# INTENT LABELS AND DESCRIPTIONS
# ============================================================================

class IntentType(Enum):
    """Enumeration of all supported intent types."""
    GREETING = "greeting"
    FAREWELL = "farewell"
    IDENTITY_QUERY = "identity_query"
    SMALLTALK = "smalltalk"
    MATH = "math"
    TIME_RELATED = "time_related"
    SYSTEM_CONTROL = "system_control"
    OPEN_APP = "open_app"
    OPEN_URL = "open_url"
    SEARCH_WEB = "search_web"
    PLAY_MEDIA = "play_media"
    WINDOW_MGMT = "window_mgmt"
    CLOSE_QUIT = "close_quit"
    AGENT_CONTROL = "agent_control"
    LOGIN = "login"
    SIGNUP = "signup"
    DEVICE_QUERY = "device_query"
    STATEMENT = "statement"
    COMMAND = "command"
    QUESTION = "question"
    TIME_QUERY = "TIME_QUERY"
    LOCATION_QUERY = "LOCATION_QUERY"
    CONTACT_LOOKUP = "CONTACT_LOOKUP"
    SEND_MESSAGE = "SEND_MESSAGE"


# Intent descriptions for documentation and UI
INTENT_DESCRIPTIONS: Dict[str, str] = {
    "greeting": "User greets or starts a conversation.",
    "farewell": "User ends or pauses conversation.",
    "identity_query": "User is asking who SarahMemory is or its role.",
    "smalltalk": "Friendly chatter, mood, casual questions.",
    "math": "User is performing or asking about a math calculation.",
    "time_related": "User asks about time, scheduling, or reminders.",
    "system_control": "User wants to control volume, brightness, etc.",
    "open_app": "Open a local application (browser, editor, etc.).",
    "open_url": "Navigate to a website or URL.",
    "search_web": "Search the internet for information.",
    "play_media": "Play or manage audio/video content.",
    "window_mgmt": "Manage windows (minimize, maximize, bring to front).",
    "close_quit": "Close apps or quit operations.",
    "agent_control": "Start/pause/stop behaviors of SarahMemory itself.",
    "login": "User wants to login or authenticate.",
    "signup": "User wants to sign up or create an account.",
    "device_query": "User asks about current device, hardware or environment.",
    "statement": "General statement that is not clearly an action command.",
    "command": "Explicit high-confidence executable command.",
    "question": "User is asking a question requiring information.",
    "TIME_QUERY": "User asks for current time.",
    "LOCATION_QUERY": "User asks about location.",
    "CONTACT_LOOKUP": "User wants to find a contact.",
    "SEND_MESSAGE": "User wants to send a message.",
}

# Intent priority order (more specific before broad)
INTENT_PRIORITY_ORDER: List[str] = [
    "login", "signup", "device_query", "time_related", "math",
    "system_control", "open_app", "open_url", "search_web",
    "play_media", "window_mgmt", "close_quit", "agent_control",
    "identity_query", "greeting", "farewell", "smalltalk",
    "question", "command", "statement",
]


# ============================================================================
# LEXICONS AND PATTERNS
# ============================================================================

# Action synonyms for command detection
ACTION_SYNONYMS: Dict[str, List[str]] = {
    "open_app": [
        "open", "launch", "start", "run", "fire up", "boot",
        "spin up", "bring up", "execute", "load"
    ],
    "open_url": [
        "go to", "navigate to", "take me to", "visit",
        "open website", "open page", "browse to", "load page"
    ],
    "search_web": [
        "search", "look up", "google", "find", "web search",
        "lookup", "search for", "find me", "research"
    ],
    "play_media": [
        "play", "resume", "start playing", "stream",
        "listen to", "watch", "queue"
    ],
    "close_quit": [
        "close", "quit", "exit", "terminate", "end",
        "kill", "stop", "shut down"
    ],
    "window_mgmt": [
        "maximize", "minimize", "focus", "switch to",
        "bring to front", "restore", "resize", "tile"
    ],
    "system_control": [
        "turn on", "turn off", "enable", "disable",
        "increase", "decrease", "raise", "lower",
        "mute", "unmute", "volume up", "volume down",
        "brightness", "wifi", "bluetooth"
    ],
    "agent_control": [
        "sarah stop now", "emergency stop", "abort mission",
        "halt", "pause", "resume", "continue", "go on",
        "stop what you're doing", "cancel that"
    ]
}

# Subject lexicon (apps and sites)
SUBJECT_LEXICON: Dict[str, Dict[str, Any]] = {
    # Browsers
    "chrome": {"kind": "app", "aka": ["google chrome", "chrome browser"], "exec": "chrome"},
    "edge": {"kind": "app", "aka": ["microsoft edge", "edge browser"], "exec": "msedge"},
    "firefox": {"kind": "app", "aka": ["mozilla firefox", "ff"], "exec": "firefox"},
    "safari": {"kind": "app", "aka": ["apple safari"], "exec": "safari"},
    "brave": {"kind": "app", "aka": ["brave browser"], "exec": "brave"},
    "opera": {"kind": "app", "aka": ["opera browser"], "exec": "opera"},

    # Editors and IDEs
    "notepad": {"kind": "app", "aka": ["note pad", "text editor"], "exec": "notepad"},
    "notepad++": {"kind": "app", "aka": ["notepad plus plus", "npp"], "exec": "notepad++"},
    "vscode": {"kind": "app", "aka": ["visual studio code", "vs code", "code"], "exec": "code"},
    "sublime": {"kind": "app", "aka": ["sublime text"], "exec": "sublime_text"},
    "atom": {"kind": "app", "aka": ["atom editor"], "exec": "atom"},
    "vim": {"kind": "app", "aka": ["vi", "neovim"], "exec": "vim"},

    # System utilities
    "calculator": {"kind": "app", "aka": ["calc", "windows calculator"], "exec": "calc"},
    "settings": {"kind": "app", "aka": ["control panel", "preferences", "system settings"], "exec": "ms-settings:"},
    "task manager": {"kind": "app", "aka": ["taskmgr", "process manager"], "exec": "taskmgr"},
    "file explorer": {"kind": "app", "aka": ["explorer", "files", "my computer"], "exec": "explorer"},
    "terminal": {"kind": "app", "aka": ["cmd", "command prompt", "powershell", "console"], "exec": "cmd"},

    # Media applications
    "spotify": {"kind": "app", "aka": ["spotify music"], "exec": "spotify"},
    "vlc": {"kind": "app", "aka": ["vlc player", "video lan", "vlc media player"], "exec": "vlc"},
    "media player": {"kind": "app", "aka": ["windows media player", "wmp"], "exec": "wmplayer"},
    "itunes": {"kind": "app", "aka": ["apple music"], "exec": "itunes"},

    # Communication
    "discord": {"kind": "app", "aka": ["discord chat"], "exec": "discord"},
    "slack": {"kind": "app", "aka": ["slack app"], "exec": "slack"},
    "teams": {"kind": "app", "aka": ["microsoft teams", "ms teams"], "exec": "teams"},
    "zoom": {"kind": "app", "aka": ["zoom meeting", "zoom app"], "exec": "zoom"},
    "skype": {"kind": "app", "aka": ["skype call"], "exec": "skype"},

    # Productivity
    "word": {"kind": "app", "aka": ["microsoft word", "ms word"], "exec": "winword"},
    "excel": {"kind": "app", "aka": ["microsoft excel", "ms excel"], "exec": "excel"},
    "powerpoint": {"kind": "app", "aka": ["microsoft powerpoint", "ms powerpoint", "ppt"], "exec": "powerpnt"},
    "outlook": {"kind": "app", "aka": ["microsoft outlook", "ms outlook"], "exec": "outlook"},

    # Games/Entertainment
    "steam": {"kind": "app", "aka": ["steam client"], "exec": "steam"},
    "epic": {"kind": "app", "aka": ["epic games", "epic launcher"], "exec": "EpicGamesLauncher"},

    # Websites/Services
    "google": {"kind": "site", "aka": ["google.com", "google search"], "url": "https://www.google.com"},
    "youtube": {"kind": "site", "aka": ["yt", "youtube.com"], "url": "https://www.youtube.com"},
    "gmail": {"kind": "site", "aka": ["mail.google.com", "g mail", "google mail"], "url": "https://mail.google.com"},
    "docs": {"kind": "site", "aka": ["google docs", "docs.google.com"], "url": "https://docs.google.com"},
    "drive": {"kind": "site", "aka": ["google drive", "drive.google.com"], "url": "https://drive.google.com"},
    "bing": {"kind": "site", "aka": ["bing.com", "bing search"], "url": "https://www.bing.com"},
    "duckduckgo": {"kind": "site", "aka": ["ddg", "duck duck go", "duckduckgo.com"], "url": "https://duckduckgo.com"},
    "twitter": {"kind": "site", "aka": ["x.com", "x", "twitter.com"], "url": "https://twitter.com"},
    "facebook": {"kind": "site", "aka": ["fb", "facebook.com"], "url": "https://www.facebook.com"},
    "instagram": {"kind": "site", "aka": ["ig", "instagram.com", "insta"], "url": "https://www.instagram.com"},
    "linkedin": {"kind": "site", "aka": ["linkedin.com"], "url": "https://www.linkedin.com"},
    "reddit": {"kind": "site", "aka": ["reddit.com"], "url": "https://www.reddit.com"},
    "github": {"kind": "site", "aka": ["github.com", "gh"], "url": "https://github.com"},
    "stackoverflow": {"kind": "site", "aka": ["stack overflow", "stackoverflow.com"], "url": "https://stackoverflow.com"},
    "amazon": {"kind": "site", "aka": ["amazon.com"], "url": "https://www.amazon.com"},
    "netflix": {"kind": "site", "aka": ["netflix.com"], "url": "https://www.netflix.com"},
    "wikipedia": {"kind": "site", "aka": ["wiki", "wikipedia.org"], "url": "https://www.wikipedia.org"},
}

# Greeting patterns
GREET_WORDS = re.compile(
    r"\b(hello|hi|hey|howdy|greetings|yo|good\s*morning|good\s*afternoon|"
    r"good\s*evening|hiya|what'?s\s*up|sup|hey\s*there|hola|bonjour)\b",
    re.IGNORECASE
)

# Farewell patterns
FAREWELL_WORDS = re.compile(
    r"\b(bye|goodbye|see\s*you|farewell|later|cya|exit|quit|"
    r"catch\s*you\s*later|talk\s*to\s*you\s*later|take\s*care|"
    r"good\s*night|gn|ttyl|peace\s*out)\b",
    re.IGNORECASE
)

# Identity phrases
IDENTITY_PHRASES: List[str] = [
    "your name", "who are you", "what's your name", "tell me your name",
    "identify yourself", "are you sarah", "which ai are you",
    "who am i talking to", "what are you", "what is your name",
    "who made you", "who created you", "what do you do"
]

# Question starters
QUESTION_STARTERS = re.compile(
    r"\b(what|how|why|where|when|who|which|can\s*you|could\s*you|"
    r"do\s*you|will\s*you|is\s*it|does\s*it|should\s*i|am\s*i|"
    r"are\s*you|may\s*i|shall\s*i|have\s*you|is\s*there|"
    r"how\s*much|how\s*many|which\s*one|would\s*you)\b",
    re.IGNORECASE
)

# Math patterns
MATH_SYMBOLS: Set[str] = set("+-*/^=%()[]")
MATH_REGEX = re.compile(r"[\d\s+\-*/^%().\[\]]+")
MATH_EXPR_REGEX = re.compile(r"\b(\d+[\s]*[\+\-\*/\^%][\s]*[\d\s\+\-\*/\^%()]+)\b")

# Time-related words
TIME_WORDS = re.compile(
    r"\b(now|today|tonight|tomorrow|yesterday|soon|later|next|this|"
    r"in|on|at|before|after|minutes?|hours?|seconds?|days?|weeks?|"
    r"months?|years?|morning|afternoon|evening|night|midnight|noon)\b",
    re.IGNORECASE
)

# URL and domain patterns
URL_REGEX = re.compile(r'\bhttps?://[^\s\'"<>]+', re.IGNORECASE)
DOMAIN_REGEX = re.compile(
    r'\b([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}\b',
    re.IGNORECASE
)
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ParsedCommand:
    """
    Structured representation of a parsed user command.
    Contains extracted intent, targets, and context.
    """
    intent: str
    raw_text: str
    confidence: float = 0.7
    math_expr: Optional[str] = None
    url: Optional[str] = None
    app: Optional[str] = None
    site: Optional[str] = None
    subject: Optional[str] = None
    action: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["extra"] = d.get("extra") or {}
        return d

    def is_actionable(self) -> bool:
        """Check if this command can be executed."""
        return self.intent in (
            "open_app", "open_url", "search_web", "play_media",
            "close_quit", "window_mgmt", "system_control", "agent_control",
            "math", "command"
        )

    def get_target(self) -> Optional[str]:
        """Get the primary target of this command."""
        return self.app or self.site or self.url or self.subject


@dataclass
class IntentResult:
    """
    Result of intent classification with metadata.
    """
    label: str
    confidence: float
    method: str  # "rule", "neural", "hybrid", "cache"
    alternatives: List[Tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "method": self.method,
            "alternatives": self.alternatives
        }


# ============================================================================
# TEXT UTILITIES
# ============================================================================

def _normalize_text(text: Any) -> str:
    """
    Normalize input text to a clean string.
    Handles None, dict with 'text' key, and other types.
    """
    if text is None:
        return ""
    if isinstance(text, str):
        return text.strip()
    if isinstance(text, dict) and "text" in text:
        return str(text["text"]).strip()
    return str(text).strip()


def tokenize(text: str) -> List[str]:
    """
    Simple tokenization into lowercase alphanumeric words.

    Args:
        text: Input text

    Returns:
        List of lowercase tokens
    """
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _contains_any(text: str, terms: List[str]) -> bool:
    """Check if text contains any of the given terms."""
    if not text or not terms:
        return False
    t = text.lower()
    return any(term.lower() in t for term in terms)


def _get_timestamp() -> str:
    """Get current ISO timestamp."""
    return datetime.now().isoformat()


# ============================================================================
# EMBEDDING SYSTEM
# ============================================================================

# Lazy-loaded sentence transformer model
_ST_EMBEDDER: Optional[Any] = None
_ST_EMBEDDER_LOADED: bool = False


def _hash_embed(text: str, dim: int = 64) -> List[float]:
    """
    Generate a deterministic embedding vector using hash functions.
    Works completely offline without any external dependencies.

    Args:
        text: Input text to embed
        dim: Dimension of output vector

    Returns:
        List of float values (embedding vector)
    """
    if not text:
        return [0.0] * dim

    # Use SHA-256 hash
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()

    # Generate dim floats from hash bytes
    vals = []
    for i in range(dim):
        b = h[i % len(h)]
        # Transform byte to float using sin for variation
        vals.append(math.sin((b + i) * 0.0174533))

    # Normalize vector
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]


def _get_local_st() -> Optional[Any]:
    """
    Get the local SentenceTransformer model.
    Lazy-loaded and cached for efficiency.

    Returns:
        SentenceTransformer model or None if unavailable
    """
    global _ST_EMBEDDER, _ST_EMBEDDER_LOADED

    if _ST_EMBEDDER_LOADED:
        return _ST_EMBEDDER

    with _model_lock:
        if _ST_EMBEDDER_LOADED:
            return _ST_EMBEDDER

        try:
            if not _HAS_ST:
                logger.debug("[AdvCU] sentence-transformers not installed")
                _ST_EMBEDDER_LOADED = True
                return None

            # Try to load local model only (no network)
            _ST_EMBEDDER = SentenceTransformer(
                "all-MiniLM-L6-v2",
                local_files_only=True
            )
            logger.info("[AdvCU] Loaded SentenceTransformer model")

        except Exception as e:
            logger.debug(f"[AdvCU] Could not load ST model: {e}")
            _ST_EMBEDDER = None

        _ST_EMBEDDER_LOADED = True
        return _ST_EMBEDDER


def embed_text(texts: Union[str, List[str]]) -> List[List[float]]:
    """
    Generate embeddings for input texts.
    Uses SentenceTransformer if available, falls back to hash-based embeddings.

    Args:
        texts: Single string or list of strings to embed

    Returns:
        List of embedding vectors (one per input text)
    """
    if isinstance(texts, str):
        texts = [texts]

    if not texts:
        return []

    # Try SentenceTransformer first
    st = _get_local_st()
    if st is not None:
        try:
            vecs = st.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            # Convert to Python lists for JSON compatibility
            return [
                v.tolist() if hasattr(v, "tolist") else list(v)
                for v in vecs
            ]
        except Exception as e:
            logger.warning(f"[AdvCU] ST encode failed, using fallback: {e}")

    # Fallback to hash-based embeddings
    return [_hash_embed(t) for t in texts]


# ============================================================================
# SIMILARITY FUNCTIONS
# ============================================================================

def evaluate_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between two texts.
    Based on token set overlap.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0.0 and 1.0
    """
    try:
        t1 = set(tokenize(text1))
        t2 = set(tokenize(text2))

        if not t1 and not t2:
            return 1.0
        if not t1 or not t2:
            return 0.0

        intersection = len(t1 & t2)
        union = len(t1 | t2)

        return intersection / union if union > 0 else 0.0

    except Exception as e:
        logger.error(f"[SIM] Jaccard similarity error: {e}")
        return 0.0


def get_vector_score(text1: str, text2: str) -> float:
    """
    Calculate cosine similarity between character-code vectors.
    Simple offline heuristic without neural models.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Similarity score between 0.0 and 1.0
    """
    try:
        # Convert to character code vectors
        a = [ord(c) for c in text1 if c.isalnum()]
        b = [ord(c) for c in text2 if c.isalnum()]

        if not a or not b:
            return 0.0

        # Pad shorter vector
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]

        # Calculate cosine similarity
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    except Exception as e:
        logger.error(f"[VEC] Vector score error: {e}")
        return 0.0


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity (-1.0 to 1.0)
    """
    if not vec1 or not vec2:
        return 0.0

    try:
        n = min(len(vec1), len(vec2))
        dot = sum(vec1[i] * vec2[i] for i in range(n))
        norm1 = math.sqrt(sum(vec1[i] ** 2 for i in range(n))) or 1.0
        norm2 = math.sqrt(sum(vec2[i] ** 2 for i in range(n))) or 1.0

        return dot / (norm1 * norm2)

    except Exception as e:
        logger.error(f"[COS] Cosine similarity error: {e}")
        return 0.0


# ============================================================================
# DETECTION HELPERS
# ============================================================================

def _likely_math(text: str) -> bool:
    """Check if text likely contains a math expression."""
    t = text.replace(" ", "")
    if not t:
        return False

    # Must contain at least one math symbol and one digit
    has_symbol = any(c in MATH_SYMBOLS for c in t)
    has_digit = any(c.isdigit() for c in t)

    if not (has_symbol and has_digit):
        return False

    # Avoid pure phone numbers or IDs
    if re.fullmatch(r"\d{7,}", t):
        return False

    # Check for basic arithmetic pattern
    if re.search(r"\d+\s*[\+\-\*/\^%]\s*\d+", text):
        return True

    return True


def _looks_like_url(text: str) -> bool:
    """Check if text contains a URL or domain."""
    if "http://" in text or "https://" in text:
        return True
    if DOMAIN_REGEX.search(text):
        return True
    return False


def _looks_like_login(text: str) -> bool:
    """Check if text indicates a login request."""
    t = text.lower()
    keywords = [
        "login", "log in", "sign in", "sign-in",
        "authenticate", "enter password", "enter pin"
    ]
    return any(k in t for k in keywords)


def _looks_like_signup(text: str) -> bool:
    """Check if text indicates a signup request."""
    t = text.lower()
    keywords = [
        "sign up", "signup", "create account",
        "register", "registration", "new account"
    ]
    return any(k in t for k in keywords)


def _looks_like_device_query(text: str) -> bool:
    """Check if text asks about the device/system."""
    t = text.lower()
    keywords = [
        "this device", "my phone", "my pc", "my computer",
        "my laptop", "current device", "what device am i using",
        "what am i using", "what system is this", "what os is this",
        "which browser", "which system", "which machine",
        "system info", "device info"
    ]
    return any(k in t for k in keywords)


def _looks_like_time_query(text: str) -> bool:
    """Check if text asks about time or scheduling."""
    t = text.lower()

    # Direct time questions
    if any(phrase in t for phrase in [
        "what time", "current time", "time now", "today's date",
        "what's today", "what day is", "schedule", "remind me",
        "set alarm", "calendar", "appointment"
    ]):
        return True

    # Time-related words with question context
    if TIME_WORDS.search(t) and QUESTION_STARTERS.search(t):
        return True

    return False


def _looks_like_greeting(text: str) -> bool:
    """Check if text is a greeting."""
    return bool(GREET_WORDS.search(text))


def _looks_like_farewell(text: str) -> bool:
    """Check if text is a farewell."""
    return bool(FAREWELL_WORDS.search(text))


def _looks_like_identity(text: str) -> bool:
    """Check if text asks about identity."""
    t = text.lower()
    return any(phrase in t for phrase in IDENTITY_PHRASES)


def _looks_like_system_control(text: str) -> bool:
    """Check if text is a system control command."""
    t = text.lower()
    keywords = [
        "volume", "brightness", "system", "mute", "unmute",
        "increase", "decrease", "turn on", "turn off",
        "shut down", "restart", "reboot", "sleep", "hibernate",
        "wifi", "bluetooth", "airplane mode"
    ]
    return any(kw in t for kw in keywords)


def _detect_action_subject(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Detect action and subject from text.

    Returns:
        (action_category, subject_key) tuple
    """
    t = text.lower()
    tokens = tokenize(text)
    joined = " ".join(tokens)

    # Detect action
    action_label = None
    for action_key, synonyms in ACTION_SYNONYMS.items():
        for syn in synonyms:
            if syn in joined or syn in t:
                action_label = action_key
                break
        if action_label:
            break

    # Detect subject
    subject_key = None
    for subj_key, meta in SUBJECT_LEXICON.items():
        # Check main key
        if subj_key in joined or subj_key in t:
            subject_key = subj_key
            break
        # Check aliases
        for alias in meta.get("aka", []):
            if alias in joined or alias in t:
                subject_key = subj_key
                break
        if subject_key:
            break

    return action_label, subject_key


# ============================================================================
# INTENT CLASSIFICATION
# ============================================================================

def _rule_based_intent(text: Any) -> str:
    """
    Fast rule-based intent classification.
    Prioritizes specific intents over broad ones.

    Args:
        text: Input text

    Returns:
        Intent label string
    """
    t = _normalize_text(text)
    if not t:
        return "statement"

    lt = t.lower()

    # Priority 1: Authentication flows
    if _looks_like_login(lt):
        return "login"
    if _looks_like_signup(lt):
        return "signup"

    # Priority 2: Device queries
    if _looks_like_device_query(lt):
        return "device_query"

    # Priority 3: Conversational intents
    if _looks_like_greeting(lt):
        return "greeting"
    if _looks_like_farewell(lt):
        return "farewell"
    if _looks_like_identity(lt):
        return "identity_query"

    # Priority 4: Math and time
    if _likely_math(t) or MATH_EXPR_REGEX.search(t):
        return "math"
    if _looks_like_time_query(lt):
        return "time_related"

    # Priority 5: System control
    if _looks_like_system_control(lt):
        return "system_control"

    # Priority 6: Action + subject combinations
    action, subject = _detect_action_subject(t)
    if action and subject:
        return action
    elif action:
        # Have action but no recognized subject
        return action

    # Priority 7: URL handling
    if _looks_like_url(t):
        if _contains_any(lt, ["search", "look up", "find", "google"]):
            return "search_web"
        return "open_url"

    # Priority 8: Command patterns
    cmd_starts = [
        "open", "go to", "launch", "start", "run", "play",
        "search", "close", "quit", "exit", "shut down", "restart"
    ]
    if any(lt.startswith(cmd) for cmd in cmd_starts):
        return "command"

    # Priority 9: Questions vs statements
    if QUESTION_STARTERS.search(lt):
        # Check for smalltalk questions
        smalltalk_phrases = [
            "how are you", "how's it going", "how do you feel",
            "are you okay", "what's up", "what are you doing"
        ]
        if any(phrase in lt for phrase in smalltalk_phrases):
            return "smalltalk"
        return "question"

    # Priority 10: Short informal greetings
    if lt in ("hi", "hey", "hello", "yo", "sup"):
        return "greeting"

    # Default: statement
    return "statement"


def _load_intent_override_cache() -> Dict[str, str]:
    """Load user-defined intent overrides from cache file."""
    try:
        if os.path.exists(INTENT_CACHE_PATH):
            with open(INTENT_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception as e:
        logger.warning(f"[Cache] Failed to read intent overrides: {e}")
    return {}


def classify_intent_with_confidence(
    text: Any,
    user_context: Any = None,
    device_context: Any = None
) -> Tuple[str, float]:
    """
    Classify intent with confidence score.
    Combines rule-based and optional neural classification.

    Args:
        text: Input text to classify
        user_context: Optional user context (reserved for future use)
        device_context: Optional device context (reserved for future use)

    Returns:
        Tuple of (intent_label, confidence_score)
    """
    t = _normalize_text(text)
    if not t:
        return "statement", 0.0

    # Check override cache first
    cache = _load_intent_override_cache()
    if t in cache:
        label = cache[t]
        log_advcu_event("Intent Override Hit", json.dumps({"text": t, "label": label}))
        return label, 1.0

    # Rule-based classification
    rule_label = _rule_based_intent(t)
    rule_conf = 0.7  # Base confidence for rule-based

    # Boost confidence for high-priority intents
    high_conf_intents = {
        "login", "signup", "device_query", "greeting",
        "farewell", "identity_query", "math"
    }
    if rule_label in high_conf_intents:
        rule_conf = 0.85

    # Log the classification
    log_advcu_event(
        "Classify Intent",
        json.dumps({
            "text": t[:100],
            "label": rule_label,
            "confidence": round(rule_conf, 3)
        })
    )

    return rule_label, float(rule_conf)


def classify_intent(
    text: Any,
    user_context: Any = None,
    device_context: Any = None
) -> str:
    """
    Primary intent classifier used across SarahMemory.

    This is the main entry point for intent classification, providing
    backward compatibility with existing code that expects just the label.

    Args:
        text: Input text to classify
        user_context: Optional user context (reserved)
        device_context: Optional device context (reserved)

    Returns:
        Intent label string
    """
    label, _ = classify_intent_with_confidence(
        text,
        user_context=user_context,
        device_context=device_context
    )
    return label


# ============================================================================
# COMMAND PARSING
# ============================================================================

def extract_math_expression(text: Any) -> Optional[str]:
    """
    Extract a math expression from text.

    Args:
        text: Input text

    Returns:
        Extracted math expression or None
    """
    t = _normalize_text(text)
    if not t:
        return None

    # Try regex pattern first
    m = MATH_EXPR_REGEX.search(t)
    if m:
        return m.group(1).strip()

    # Check if entire text looks like an expression
    cleaned = t.replace(" ", "")
    if all(ch.isdigit() or ch in MATH_SYMBOLS or ch == "." for ch in cleaned):
        return t.strip()

    return None


def is_math_query(text: Any) -> bool:
    """Check if text is primarily a math calculation."""
    return extract_math_expression(text) is not None


def is_time_query(text: Any) -> bool:
    """Check if text is a time/scheduling query."""
    t = _normalize_text(text)
    if not t:
        return False
    return _looks_like_time_query(t)


def parse_command(
    text: Any,
    user_context: Any = None,
    device_context: Any = None
) -> ParsedCommand:
    """
    Parse user input into a structured command.

    Extracts:
    - Intent classification
    - Math expressions
    - URLs and domains
    - App/site targets
    - Action and subject

    Args:
        text: Input text to parse
        user_context: Optional user context
        device_context: Optional device context

    Returns:
        ParsedCommand dataclass with extracted information
    """
    t = _normalize_text(text)
    label, confidence = classify_intent_with_confidence(t)

    # Initialize extraction results
    math_expr = None
    url = None
    app = None
    site = None
    subject = None
    action = None
    extra: Dict[str, Any] = {}

    # Extract math expression
    if label == "math" or _likely_math(t):
        math_expr = extract_math_expression(t)

    # Extract URLs
    if _looks_like_url(t):
        url_match = URL_REGEX.findall(t)
        if url_match:
            url = url_match[0]
        else:
            # Try to extract domain
            domain_match = DOMAIN_REGEX.search(t)
            if domain_match:
                site = domain_match.group(0)

    # Extract action and subject
    action, subj = _detect_action_subject(t)
    if subj:
        info = SUBJECT_LEXICON.get(subj, {})
        if info.get("kind") == "app":
            app = subj
        elif info.get("kind") == "site":
            site = subj
            if not url and info.get("url"):
                url = info["url"]
        subject = subj

    # Store context in extra
    extra["user_context"] = user_context
    extra["device_context"] = device_context
    extra["tokens"] = tokenize(t)

    return ParsedCommand(
        intent=label,
        raw_text=t,
        confidence=confidence,
        math_expr=math_expr,
        url=url,
        app=app,
        site=site,
        subject=subject,
        action=action,
        extra=extra
    )


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

# SQL schema for code corpus
_SCHEMA_SQL_CODE_CORPUS = """
CREATE TABLE IF NOT EXISTS code_corpus (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT,
    key TEXT,
    snippet TEXT,
    file TEXT,
    line INTEGER,
    ts TEXT
);
CREATE INDEX IF NOT EXISTS idx_code_corpus_kind ON code_corpus(kind);
CREATE INDEX IF NOT EXISTS idx_code_corpus_snip ON code_corpus(snippet);

CREATE TABLE IF NOT EXISTS code_corpus_seen (
    snip_hash TEXT PRIMARY KEY,
    file TEXT,
    line INTEGER,
    first_ts TEXT
);
CREATE INDEX IF NOT EXISTS idx_code_seen_file_line ON code_corpus_seen(file, line);

CREATE TABLE IF NOT EXISTS advcu_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT,
    event TEXT,
    details TEXT
);
"""


def _ensure_db() -> None:
    """Ensure all AdvCU database tables exist."""
    try:
        os.makedirs(os.path.dirname(SYSTEM_DB_PATH), exist_ok=True)
        with sqlite3.connect(SYSTEM_DB_PATH, timeout=10.0) as conn:
            conn.executescript(_SCHEMA_SQL_CODE_CORPUS)
            conn.commit()
    except Exception as e:
        logger.warning(f"[AdvCU] Database ensure failed: {e}")


def _ensure_code_corpus_db() -> None:
    """Alias for _ensure_db for backward compatibility."""
    _ensure_db()


def log_advcu_event(event: str, details: str) -> None:
    """
    Log an AdvCU event to the database.

    Args:
        event: Event type/name
        details: Event details (typically JSON)
    """
    try:
        _ensure_db()
        with sqlite3.connect(SYSTEM_DB_PATH, timeout=5.0) as conn:
            conn.execute(
                "INSERT INTO advcu_log(ts, event, details) VALUES (?, ?, ?)",
                (_get_timestamp(), event, details)
            )
            conn.commit()
    except Exception as e:
        logger.debug(f"[DB] log_advcu_event failed: {e}")


# ============================================================================
# CODE INTROSPECTION (AST Mining)
# ============================================================================

# Patterns for code mining
_RX_URL = re.compile(r'\bhttps?://[^\s\'"]+', re.IGNORECASE)
_RX_DOMAIN = re.compile(r'\b([a-z0-9-]+\.)+[a-z]{2,}\b', re.IGNORECASE)
_RX_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
_RX_PROMPTV = re.compile(
    r'(?i)(^|_)(prompt|system_prompt|sys_prompt|instruction|messages|template)$'
)
_RX_ROLEKEY = re.compile(r'(?i)"role"\s*:\s*"(system|user|assistant)"')
_RX_MSGSKEY = re.compile(r'(?i)"messages"\s*:\s*\[')


def _normalize_snippet(s: str) -> str:
    """Normalize a code snippet for storage."""
    s = s.replace("\r\n", "\n").strip()
    return s if len(s) <= 4000 else (s[:4000] + " …[truncated]")


class _NodeMiner(ast.NodeVisitor):
    """AST visitor to extract values from Python source."""

    def __init__(self, filename: str):
        self.filename = filename
        self.rows: List[Tuple[str, str, str, int]] = []  # (kind, key, snippet, line)

    def visit_Assign(self, node: ast.Assign):
        """Visit assignment nodes."""
        keynames = []
        for t in node.targets:
            if isinstance(t, ast.Name):
                keynames.append(t.id)
            elif isinstance(t, ast.Attribute):
                keynames.append(t.attr)

        is_prompty = any(
            _RX_PROMPTV.search(k) for k in keynames
            if isinstance(k, str)
        )

        self._capture_value(
            node.value,
            keyhint=";".join(keynames) if keynames else None,
            is_prompt=is_prompty,
            lineno=node.lineno
        )
        self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict):
        """Visit dict literals."""
        for k, v in zip(node.keys, node.values):
            kstr = self._const_to_str(k)
            self._capture_value(
                v,
                keyhint=kstr,
                is_prompt=False,
                lineno=getattr(v, "lineno", getattr(node, "lineno", 0))
            )
        self.generic_visit(node)

    def visit_List(self, node: ast.List):
        """Visit list literals."""
        for v in node.elts:
            self._capture_value(
                v,
                keyhint=None,
                is_prompt=False,
                lineno=getattr(v, "lineno", getattr(node, "lineno", 0))
            )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        """Visit constant values."""
        if isinstance(node.value, str):
            s = node.value.strip()
            if s:
                self._add_row('str', None, _normalize_snippet(s), getattr(node, 'lineno', 0))
                self._scan_regexes(s, getattr(node, 'lineno', 0))
        self.generic_visit(node)

    def _capture_value(
        self,
        value: ast.AST,
        keyhint: Optional[str],
        is_prompt: bool,
        lineno: int
    ):
        """Capture a value node recursively."""
        if isinstance(value, ast.Dict):
            for k, v in zip(value.keys, value.values):
                kstr = self._const_to_str(k)
                if isinstance(v, (ast.Dict, ast.List, ast.Tuple)):
                    self._capture_value(
                        v,
                        keyhint=kstr or keyhint,
                        is_prompt=is_prompt,
                        lineno=getattr(v, "lineno", lineno)
                    )
                else:
                    vs = self._const_to_str(v)
                    if vs:
                        kind = 'prompt' if (is_prompt or (kstr and _RX_PROMPTV.search(kstr))) else 'dict'
                        self._add_row(kind, kstr, _normalize_snippet(vs), lineno)
                        self._scan_regexes(vs, lineno)

        elif isinstance(value, (ast.List, ast.Tuple)):
            for v in value.elts:
                if isinstance(v, (ast.Dict, ast.List, ast.Tuple)):
                    self._capture_value(
                        v,
                        keyhint=keyhint,
                        is_prompt=is_prompt,
                        lineno=getattr(v, "lineno", lineno)
                    )
                else:
                    vs = self._const_to_str(v)
                    if vs:
                        kind = 'prompt' if is_prompt else 'list'
                        self._add_row(kind, keyhint, _normalize_snippet(vs), lineno)
                        self._scan_regexes(vs, lineno)

        else:
            vs = self._const_to_str(value)
            if vs:
                kind = 'prompt' if is_prompt else 'str'
                self._add_row(kind, keyhint, _normalize_snippet(vs), lineno)
                self._scan_regexes(vs, lineno)

    def _const_to_str(self, node: ast.AST) -> Optional[str]:
        """Extract string from constant node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        return None

    def _add_row(self, kind: str, key: Optional[str], snippet: str, line: int):
        """Add a row to results."""
        self.rows.append((kind, key or "", snippet, int(line)))

    def _scan_regexes(self, text: str, line: int):
        """Scan text for URLs, emails, domains."""
        for url in _RX_URL.findall(text):
            self._add_row('url', '', url.strip(), line)
        for em in _RX_EMAIL.findall(text):
            self._add_row('email', '', em.strip(), line)
        for dm in _RX_DOMAIN.findall(text):
            if "http" not in dm and "@" not in dm:
                self._add_row('domain', '', dm.strip(), line)


def _iter_core_files() -> List[str]:
    """Get list of core SarahMemory Python files."""
    base = getattr(config, "BASE_DIR", os.getcwd())
    return sorted(glob.glob(os.path.join(base, "SarahMemory*.py")))


def _scan_file(path: str) -> List[Tuple[str, str, str, int]]:
    """
    Scan a Python file and extract values.

    Args:
        path: Path to Python file

    Returns:
        List of (kind, key, snippet, line) tuples
    """
    try:
        with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
            src = f.read()

        tree = ast.parse(src)
        miner = _NodeMiner(os.path.basename(path))
        miner.visit(tree)

        # Check for JSON-like prompt payloads
        if '"messages"' in src and _RX_MSGSKEY.search(src):
            miner.rows.append(('prompt', 'messages', '[messages payload detected]', 0))

        return miner.rows

    except SyntaxError as se:
        logger.warning(f"[AdvCU] AST parse failed for {os.path.basename(path)}: {se}")
        return []
    except Exception as e:
        logger.warning(f"[AdvCU] scan_file error {os.path.basename(path)}: {e}")
        return []


def warm_code_corpus(force: bool = False) -> Dict[str, Any]:
    """
    Scan all SarahMemory*.py files and store in database.
    Uses delta-tracking to avoid duplicate inserts.

    Args:
        force: Force rescan even if cache exists

    Returns:
        Dictionary with scan statistics
    """
    _ensure_db()
    ts = datetime.utcnow().isoformat()
    files = _iter_core_files()

    # Check for recent cache
    if os.path.exists(CODE_CORPUS_JSON) and not force:
        try:
            with open(CODE_CORPUS_JSON, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if isinstance(cache, dict) and cache.get("_ok"):
                return cache
        except Exception:
            pass

    # Scan files
    rows = []
    scanned = 0
    for fp in files:
        mined = _scan_file(fp)
        for kind, key, snippet, line in mined:
            scanned += 1
            sh = hashlib.sha256((snippet or "").encode("utf-8")).hexdigest()
            rows.append((kind, key, snippet, os.path.basename(fp), int(line), ts, sh))

    # Insert new rows
    new_inserts = 0
    try:
        with sqlite3.connect(SYSTEM_DB_PATH, timeout=10.0) as conn:
            cur = conn.cursor()

            for kind, key, snip, file, line, tstamp, sh in rows:
                # Check if already seen
                cur.execute("SELECT 1 FROM code_corpus_seen WHERE snip_hash=?", (sh,))
                if cur.fetchone():
                    continue

                # Insert new
                cur.execute(
                    "INSERT OR REPLACE INTO code_corpus_seen(snip_hash, file, line, first_ts) VALUES (?, ?, ?, ?)",
                    (sh, file, line, tstamp)
                )
                cur.execute(
                    "INSERT INTO code_corpus(kind, key, snippet, file, line, ts) VALUES (?, ?, ?, ?, ?, ?)",
                    (kind, key, snip, file, line, tstamp)
                )
                new_inserts += 1

            conn.commit()

    except Exception as e:
        logger.warning(f"[AdvCU] DB delta write failed: {e}")

    # Save cache
    data = {
        "_ok": True,
        "generated": ts,
        "counts": {
            "files": len(files),
            "scanned": scanned,
            "new": new_inserts
        }
    }

    try:
        with open(CODE_CORPUS_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"[AdvCU] Cache write failed: {e}")

    logger.info(f"[AdvCU] Code corpus: {len(files)} files, {scanned} scanned, {new_inserts} new")
    return data


def get_corpus_stats() -> Dict[str, Any]:
    """Get statistics from the code corpus cache."""
    try:
        if os.path.exists(CODE_CORPUS_JSON):
            with open(CODE_CORPUS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"_ok": False}


def mine_hardcoded_knowledge(limit: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Retrieve mined code knowledge from database.

    Args:
        limit: Maximum rows to retrieve

    Returns:
        Dictionary grouped by kind (url, domain, email, prompt, etc.)
    """
    _ensure_db()
    out: Dict[str, List[Dict[str, Any]]] = {}

    try:
        with sqlite3.connect(SYSTEM_DB_PATH, timeout=10.0) as conn:
            cur = conn.cursor()

            if limit:
                cur.execute(
                    "SELECT kind, key, snippet, file, line FROM code_corpus LIMIT ?",
                    (int(limit),)
                )
            else:
                cur.execute("SELECT kind, key, snippet, file, line FROM code_corpus")

            for kind, key, snip, file, line in cur.fetchall():
                out.setdefault(kind, []).append({
                    "key": key,
                    "snippet": snip,
                    "file": file,
                    "line": line
                })

    except Exception as e:
        logger.warning(f"[AdvCU] read corpus failed: {e}")

    return out


def contextualize_with_code(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """
    Find code snippets relevant to a query.

    Args:
        query: Search query
        top_k: Number of results to return

    Returns:
        List of relevant snippets with scores
    """
    _ensure_db()

    # Embed query
    qv = embed_text(query)[0]

    # Retrieve all snippets
    rows: List[Tuple[str, str, str, str, int]] = []
    try:
        with sqlite3.connect(SYSTEM_DB_PATH, timeout=10.0) as conn:
            cur = conn.cursor()
            cur.execute("SELECT kind, key, snippet, file, line FROM code_corpus")
            rows = cur.fetchall()
    except Exception as e:
        logger.warning(f"[AdvCU] retrieval failed: {e}")
        return []

    # Score snippets
    scored = []
    # Sample for performance if too many rows
    sample = rows if len(rows) <= 20000 else rows[::max(1, len(rows) // 20000)]

    for kind, key, snip, file, line in sample:
        v = embed_text(snip)[0]
        score = cosine_similarity(qv, v)
        scored.append((score, kind, key, snip, file, line))

    # Sort by score
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return top_k
    out = []
    for score, kind, key, snip, file, line in scored[:max(1, top_k)]:
        out.append({
            "score": round(score, 4),
            "kind": kind,
            "key": key,
            "snippet": snip,
            "file": file,
            "line": line
        })

    return out


def verify_corpus_status() -> Dict[str, Any]:
    """
    Get current corpus status from database.

    Returns:
        Dictionary with db_rows, seen_hashes, and cache info
    """
    out = {"db_rows": 0, "seen_hashes": 0, "cache": {}}

    try:
        _ensure_db()
        with sqlite3.connect(SYSTEM_DB_PATH, timeout=5.0) as conn:
            cur = conn.cursor()

            cur.execute("SELECT COUNT(*) FROM code_corpus")
            out["db_rows"] = int(cur.fetchone()[0])

            cur.execute("SELECT COUNT(*) FROM code_corpus_seen")
            out["seen_hashes"] = int(cur.fetchone()[0])

    except Exception as e:
        logger.debug(f"[verify] database query failed: {e}")

    try:
        if os.path.exists(CODE_CORPUS_JSON):
            with open(CODE_CORPUS_JSON, "r", encoding="utf-8") as f:
                out["cache"] = json.load(f)
    except Exception:
        pass

    return out


def ensure_corpus_ready() -> None:
    """
    Ensure code corpus is ready.
    Safe to call multiple times.
    """
    try:
        stats = get_corpus_stats()
        if not stats.get("_ok"):
            warm_code_corpus(force=False)
    except Exception:
        warm_code_corpus(force=False)


# ============================================================================
# COLOR DETECTION
# ============================================================================

def sm_color_name_from_rgb(r: int, g: int, b: int) -> str:
    """
    Convert RGB values to a color name.

    Args:
        r: Red component (0-255)
        g: Green component (0-255)
        b: Blue component (0-255)

    Returns:
        Color name string
    """
    return sm_color_name_from_rgb_fine(r, g, b)


def sm_color_name_from_rgb_fine(r: int, g: int, b: int) -> str:
    """
    Fine-grained RGB to color name mapping.
    Uses HSV color space for accurate naming.

    Args:
        r: Red component (0-255)
        g: Green component (0-255)
        b: Blue component (0-255)

    Returns:
        Color name string
    """
    # Normalize to 0-1 range
    rp, gp, bp = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(rp, gp, bp), min(rp, gp, bp)
    v = mx
    d = mx - mn

    # Grayscale detection
    if d < 0.05:
        if v < 0.06:
            return "black"
        if v < 0.18:
            return "charcoal"
        if v < 0.30:
            return "dark gray"
        if v < 0.43:
            return "gray"
        if v < 0.70:
            return "silver"
        if v < 0.88:
            return "off-white"
        return "white"

    # Calculate saturation
    s = d / (mx or 1.0)

    # Calculate hue
    if mx == rp:
        h = ((gp - bp) / (d or 1e-9)) % 6
    elif mx == gp:
        h = ((bp - rp) / (d or 1e-9)) + 2
    else:
        h = ((rp - gp) / (d or 1e-9)) + 4
    h *= 60.0

    # Determine tone
    tone = "dark" if v < 0.15 else ("light" if v > 0.85 else "mid")

    # Map hue to color name
    if 0 <= h < 15 or h >= 345:
        return "red" if tone != "dark" else "dark red"
    elif 15 <= h < 45:
        return "orange" if tone != "dark" else "brown"
    elif 45 <= h < 60:
        return "gold" if tone != "dark" else "olive"
    elif 60 <= h < 80:
        return "chartreuse" if tone != "dark" else "olive"
    elif 80 <= h < 100:
        return "lime" if tone != "dark" else "olive green"
    elif 100 <= h < 130:
        return "spring green" if s > 0.5 else "pale green"
    elif 130 <= h < 170:
        return "teal" if tone == "dark" else "cyan"
    elif 170 <= h < 210:
        return "azure" if tone != "dark" else "steel blue"
    elif 210 <= h < 250:
        return "blue" if tone != "light" else "sky blue"
    elif 250 <= h < 280:
        return "indigo" if tone == "dark" else "violet"
    elif 280 <= h < 320:
        return "magenta" if tone != "dark" else "plum"
    elif 320 <= h < 345:
        return "rose" if tone != "dark" else "maroon"

    return "unknown"


def _dominant_color_and_name(bgr_roi) -> Dict[str, Any]:
    """
    Extract dominant color from an image region.

    Args:
        bgr_roi: BGR image region (numpy array)

    Returns:
        Dictionary with hex, rgb, hsl, name, confidence
    """
    if not _HAS_CV2 or not _HAS_NUMPY:
        return {
            "hex": "#000000",
            "rgb": [0, 0, 0],
            "hsl": [0, 0, 0],
            "name": "unknown",
            "confidence": 0.0
        }

    if bgr_roi is None or bgr_roi.size == 0:
        return {
            "hex": "#000000",
            "rgb": [0, 0, 0],
            "hsl": [0, 0, 0],
            "name": "unknown",
            "confidence": 0.0
        }

    try:
        # Convert to HSV for filtering
        hsv = _cv2.cvtColor(bgr_roi, _cv2.COLOR_BGR2HSV)
        h, s, v = _cv2.split(hsv)

        # Filter out very dark/desaturated pixels
        mask = (s > 18) & (v > 25)
        pix = bgr_roi[mask] if mask.any() else bgr_roi.reshape(-1, 3)

        # K-means clustering
        Z = np.float32(pix)
        K = 3
        criteria = (_cv2.TERM_CRITERIA_EPS + _cv2.TERM_CRITERIA_MAX_ITER, 24, 1.0)

        _, labels, centers = _cv2.kmeans(Z, K, None, criteria, 3, _cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.flatten(), minlength=K)
        idx = int(counts.argmax())
        b, g, r = centers[idx].astype(int).tolist()

        name = sm_color_name_from_rgb(r, g, b)
        conf = float(counts[idx] / max(1, counts.sum()))

    except Exception:
        # Fallback to mean
        b, g, r = pix.mean(axis=0).astype(int).tolist()
        name = sm_color_name_from_rgb(r, g, b)
        conf = 0.5

    # Calculate HSL
    rp, gp, bp = r / 255.0, g / 255.0, b / 255.0
    mx, mn = max(rp, gp, bp), min(rp, gp, bp)
    l = (mx + mn) / 2.0

    if mx == mn:
        h = 0.0
        s = 0.0
    else:
        d = mx - mn
        s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn + 1e-9)

        if mx == rp:
            h = ((gp - bp) / d + (6 if gp < bp else 0))
        elif mx == gp:
            h = ((bp - rp) / d + 2)
        else:
            h = ((rp - gp) / d + 4)

        h *= 60.0
        h %= 360.0

    hex_val = f"#{r:02x}{g:02x}{b:02x}"

    return {
        "hex": hex_val,
        "rgb": [int(r), int(g), int(b)],
        "hsl": [float(h), float(s), float(l)],
        "name": name,
        "confidence": conf
    }


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

def _log_corpus_boot_message() -> None:
    """Log a summary message about corpus state on module load."""
    try:
        stats = get_corpus_stats()
        counts = stats.get("counts") or {}
        files = counts.get("files", 0)
        scanned = counts.get("scanned", 0) or counts.get("rows", 0)
        new_lines = counts.get("new", 0)

        if files == 0 or scanned == 0:
            # Fallback to DB + filesystem
            try:
                with sqlite3.connect(SYSTEM_DB_PATH, timeout=5.0) as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT COUNT(*) FROM code_corpus")
                    scanned = int(cur.fetchone()[0])
            except Exception:
                scanned = 0

            try:
                files = len(_iter_core_files())
            except Exception:
                files = 0

        logger.info(f"[AdvCU] Code corpus: {files} files, {scanned} scanned, {new_lines} new")

    except Exception as e:
        logger.debug(f"[AdvCU] boot message skipped: {e}")


# Initialize on import
try:
    _ensure_db()
    ensure_corpus_ready()
except Exception as e:
    logger.debug(f"[AdvCU] initialization skipped: {e}")

_log_corpus_boot_message()


# ============================================================================
# DEMO AND TESTING
# ============================================================================

def _demo() -> None:
    """Run demonstration of AdvCU capabilities."""
    print("=" * 70)
    print("SarahMemory Advanced Context Unit (AdvCU) - Demo")
    print("=" * 70)

    tests = [
        "Hey Sarah, how are you?",
        "Open Chrome",
        "Go to google.com and then search for cute cats",
        "Search YouTube for lo-fi chill beats",
        "Launch Notepad quickly",
        "Please close the current window",
        "What is 5 + 5?",
        "When is my meeting tomorrow?",
        "Who are you?",
        "Bye for now",
        "Take me to https://github.com",
        "Find drivers for NVIDIA RTX 3080",
        "Turn the volume down",
        "Resume",
        "What time is it?",
        "Login to my account",
        "Create a new account",
    ]

    print("\n[Intent Classification Tests]\n")

    for phrase in tests:
        label, conf = classify_intent_with_confidence(phrase)
        parsed = parse_command(phrase)

        print(f"Input: \"{phrase}\"")
        print(f"  Intent: {label} (confidence: {conf:.2f})")
        print(f"  Actionable: {parsed.is_actionable()}")
        if parsed.get_target():
            print(f"  Target: {parsed.get_target()}")
        print()

    # Test embeddings
    print("\n[Embedding Tests]\n")

    test_texts = ["hello world", "goodbye world"]
    embeddings = embed_text(test_texts)
    print(f"Embedded {len(test_texts)} texts")
    print(f"Vector dimension: {len(embeddings[0])}")
    print(f"Similarity: {cosine_similarity(embeddings[0], embeddings[1]):.4f}")

    # Test corpus stats
    print("\n[Corpus Status]\n")

    status = verify_corpus_status()
    print(f"Database rows: {status['db_rows']}")
    print(f"Seen hashes: {status['seen_hashes']}")

    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    try:
        _demo()
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Intent Classification
    "classify_intent",
    "classify_intent_with_confidence",
    "parse_command",
    "ParsedCommand",
    "IntentResult",
    "INTENT_DESCRIPTIONS",
    "INTENT_PRIORITY_ORDER",

    # Embeddings
    "embed_text",

    # Similarity
    "evaluate_similarity",
    "get_vector_score",
    "cosine_similarity",

    # Detection Helpers
    "is_math_query",
    "is_time_query",
    "extract_math_expression",
    "tokenize",

    # Code Introspection
    "warm_code_corpus",
    "mine_hardcoded_knowledge",
    "contextualize_with_code",
    "get_corpus_stats",
    "verify_corpus_status",
    "ensure_corpus_ready",

    # Color Detection
    "sm_color_name_from_rgb",
    "sm_color_name_from_rgb_fine",
    "_dominant_color_and_name",

    # Database
    "log_advcu_event",

    # Lexicons
    "ACTION_SYNONYMS",
    "SUBJECT_LEXICON",
]

# ====================================================================
# END OF SarahMemoryAdvCU.py v8.0.0
# ====================================================================