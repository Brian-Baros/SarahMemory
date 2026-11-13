"""--==The SarahMemory Project==--
File: SarahMemoryAdvCU.py
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

ADVANCED CONTEXT UNIT (AdvCU)
===========================================================

--------
The Advanced Context Unit is one of the most powerful core modules in SarahMemory.
It serves as the central intelligence for understanding user intent and enabling
system automation capabilities.

KEY CAPABILITIES:
-----------------
1. INTENT CLASSIFICATION
- Rule-based fast classification (offline-first)
- Optional approved local neural embeddings for text similarity
- Conservative contextual scoring with bounded heuristic confidence
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

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "semantic_core"
# CATEGORY = "intent_and_context"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "advanced_context_unit"
# FAMILY = "core_cognition"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-11-01"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Advanced Context Unit for intent classification, command parsing, embeddings, semantic compression, and code introspection. Foundational cognitive core module."
# --- SARAHMETA END ---

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
import hmac
import hmac
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Set, Union, Callable
from enum import Enum
from collections import defaultdict, Counter
from contextlib import closing
from pathlib import Path

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
# Import is read-only: create database directories only on an explicit persistence operation.

# Offline mode detection
_OFFLINE_MODE = bool(
    getattr(config, "LOCAL_ONLY_MODE", False) or
    getattr(config, "SAFE_MODE", False) or
    os.getenv("HF_HUB_OFFLINE") == "1" or
    os.getenv("TRANSFORMERS_OFFLINE") == "1"
)

# Thread lock for model loading
_model_lock = threading.RLock()
_corpus_lock = threading.RLock()
_db_lock = threading.RLock()
_db_initialized_path = None
_ADVCU_MAX_INPUT = 16384
_ADVCU_MAX_SOURCE_FILES = 256
_ADVCU_MAX_SOURCE_BYTES = 1024 * 1024
_ADVCU_MAX_MINED_ROWS_PER_FILE = 1200
_ADVCU_MAX_CORPUS_RESULTS = 800



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
    DIAGNOSTICS = "diagnostics"
    CREATIVE = "creative"
    BUILD_REQUEST = "build_request"
    NETWORK_REQUEST = "network_request"
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
    "diagnostics": "User asks to run diagnostics or health checks on the system.",
    "creative": "User wants SarahMemory to generate or create an artifact such as an image, song, video, avatar, or webpage.",
    "build_request": "User wants SarahMemory to build or generate a website, page, application, dashboard, or project artifact.",
    "network_request": "User wants SarahMemory to use SarahNet, sync, broker, remote node, or mesh/network functionality.",
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
    "login", "signup", "device_query", "diagnostics", "creative", "build_request",
    "time_related", "math", "system_control", "open_app", "open_url", "search_web",
    "play_media", "window_mgmt", "close_quit", "agent_control", "network_request",
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
    "who made you", "who created you", "who programmed you", "who developed you",
    "what do you do", "what are you capable of", "what are your capabilities",
    "what can you do", "what can you do right now", "what are you able to do",
    "who is brian baros", "who is brian lee baros"
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

# Semantic compression keyword groups
_CREATIVE_VERBS = {
    "create", "make", "generate", "draw", "design", "compose",
    "produce", "build", "craft", "render", "write"
}
_CREATIVE_OUTPUT_KEYWORDS = {
    "image": ("image", "picture", "photo", "art", "drawing", "poster", "graphic", "wallpaper", "logo", "meme", "banner", "illustration", "sketch"),
    "video": ("video", "movie", "clip", "animation", "reel", "short"),
    "music": ("music", "song", "beat", "melody", "instrumental", "lyrics", "track"),
    "avatar": ("avatar", "character", "persona", "model"),
    "website": ("website", "webpage", "site", "landing page", "dashboard", "page"),
    "application": ("app", "application", "tool", "program", "script", "workflow"),
}
_NETWORK_HINTS = (
    "sarahnet", "network", "mesh", "broker", "sync", "remote node",
    "cloud", "peer", "server", "api hub"
)
_FILE_ACTION_HINTS = (
    "open", "save", "move", "copy", "delete", "rename", "preview",
    "launch", "browse", "display", "download", "export"
)

# Vision / perception semantic pools
_VISUAL_QUERY_TYPES: Tuple[str, ...] = (
    "identify_color",
    "read_text",
    "detect_objects",
    "detect_faces",
    "track_motion",
    "estimate_distance",
    "inspect_safety_zone",
    "compare_before_after",
    "assess_style_or_fit",
    "scene_summary",
)
_VISUAL_COLOR_HINTS = (
    "color", "colour", "shade", "dominant color", "what color", "what colour"
)
_VISUAL_TEXT_HINTS = (
    "read", "say", "text", "label", "sign", "license plate", "plate", "ocr",
    "serial", "screen", "display text", "what does"
)
_VISUAL_OBJECT_HINTS = (
    "object", "objects", "item", "items", "thing", "things", "see", "seeing",
    "in frame", "in the image", "in the picture", "in the scene", "around me",
    "what is there", "what's there", "what do you see"
)
_VISUAL_FACE_HINTS = (
    "face", "faces", "eye", "eyes", "eyecolor", "eye color", "facial",
    "person", "people", "employee", "worker", "recognize", "who is", "smiling"
)
_VISUAL_MOTION_HINTS = (
    "moving", "movement", "motion", "track", "tracking", "follow", "walking",
    "running", "approaching", "leaving", "entering", "exiting"
)
_VISUAL_DISTANCE_HINTS = (
    "distance", "far", "close", "near", "too close", "how far", "how close",
    "range", "proximity"
)
_VISUAL_SAFETY_HINTS = (
    "danger zone", "safe zone", "safety zone", "hazard zone", "restricted zone",
    "unsafe", "collision", "stop the machine", "stop machine", "emergency stop",
    "interlock", "too close to", "in the zone", "crossing the line"
)
_VISUAL_COMPARE_HINTS = (
    "before and after", "before/after", "compare", "difference", "changed",
    "has it changed", "what changed"
)
_VISUAL_STYLE_HINTS = (
    "look fat", "look good", "look bad", "fit", "style", "match", "flattering",
    "outfit", "dress this", "does this shirt", "does this dress", "does this outfit"
)
_VISUAL_SCENE_HINTS = (
    "summarize the scene", "scene summary", "describe the scene", "describe what you see",
    "what do you see", "what's happening", "what is happening", "summarize this image"
)
_VISUAL_DEVICE_HINTS = (
    "camera", "webcam", "vision", "image", "picture", "photo", "frame", "video",
    "screen", "monitor", "thermal", "infrared", "ir", "night vision", "autofocus",
    "zoom", "magnify", "sensor"
)
_VISUAL_SUBJECT_SYNONYMS: Dict[str, str] = {
    "tee": "shirt",
    "tshirt": "shirt",
    "t-shirt": "shirt",
    "top": "shirt",
    "hooded sweatshirt": "hoodie",
    "trousers": "pants",
    "slacks": "pants",
    "blue jeans": "jeans",
    "specs": "glasses",
    "spectacles": "glasses",
    "hard hat": "helmet",
    "hi vis": "safety vest",
    "high vis": "safety vest",
    "infra red": "infrared",
}
_VISUAL_SUBJECT_CANDIDATES: Tuple[str, ...] = (
    "shirt", "pants", "jeans", "shorts", "skirt", "dress", "jacket", "hoodie", "sweater",
    "coat", "hat", "cap", "helmet", "glasses", "shoes", "boots", "eyes", "eye", "face",
    "hair", "worker", "employee", "person", "people", "machine", "robot", "vehicle", "car",
    "road", "lane", "controller", "tool", "sign", "label", "screen", "display", "animal",
    "deer", "hog", "target", "zone", "door", "package", "box", "desk", "couch", "sofa"
)
_VISUAL_ACTION_HINTS = (
    "stop", "halt", "shutdown", "lockout", "disable", "track", "monitor", "follow",
    "alert", "warn", "notify", "report", "log"
)
_SEMANTIC_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "for", "of", "and", "or", "if", "then", "with", "into", "in",
    "on", "at", "from", "by", "about", "as", "that", "this", "these",
    "those", "it", "its", "me", "my", "your", "you", "we", "our", "us",
    "please", "can", "could", "would", "should", "what", "how", "why",
    "where", "when", "who", "which", "do", "does", "did", "have", "has",
    "had", "want", "need", "like", "show", "tell", "give", "there", "here",
    "new", "professional", "proper", "final", "full", "about"
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ParsedCommand:
    """
    Structured representation of a parsed user command.
    Contains extracted intent, targets, and compact semantic-routing context.
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
    intent_family: Optional[str] = None
    likely_lane: Optional[str] = None
    output_type: Optional[str] = None
    query_type: Optional[str] = None
    action_expectation: Optional[str] = None
    semantic_summary: Optional[str] = None
    attributes: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    module_hints: List[str] = field(default_factory=list)
    helper_payload: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["extra"] = d.get("extra") or {}
        d["helper_payload"] = d.get("helper_payload") or {}
        d["attributes"] = list(d.get("attributes") or [])
        d["keywords"] = list(d.get("keywords") or [])
        d["module_hints"] = list(d.get("module_hints") or [])
        return d

    def to_semantic_packet(self) -> Dict[str, Any]:
        """Return the compact semantic packet used by Neuron/helper lanes."""
        return {
            "intent": self.intent,
            "intent_confidence": self.confidence,
            "intent_family": self.intent_family,
            "likely_lane": self.likely_lane,
            "query_type": self.query_type,
            "action_expectation": self.action_expectation,
            "subject": self.subject,
            "attributes": list(self.attributes or []),
            "keywords": list(self.keywords or []),
            "output_type": self.output_type,
            "math_expr": self.math_expr,
            "action": self.action,
            "target": self.get_target(),
            "semantic_summary": self.semantic_summary,
            "module_hints": list(self.module_hints or []),
            "helper_payload": dict(self.helper_payload or {}),
            "execution_authority": False,
        }

    def is_actionable(self) -> bool:
        """Return a potential action candidate, never an authorization verdict."""
        return self.intent in (
            "open_app", "open_url", "search_web", "play_media",
            "close_quit", "window_mgmt", "system_control", "agent_control",
            "math", "command", "creative", "build_request", "network_request"
        ) or (self.likely_lane in ("action", "creative", "network"))

    def get_target(self) -> Optional[str]:
        """Get the primary target of this command."""
        return self.app or self.site or self.url or self.subject or self.output_type


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
    """Phrase-boundary matching: avoid 'ram' inside 'programming'."""
    return _advcu_any_phrase(_normalize_text(text), terms)


def _contains_phrase(text: str, phrase: str) -> bool:
    """Check for a whole-word/phrase match, avoiding accidental substring hits."""
    t = (text or "").lower().strip()
    p = (phrase or "").lower().strip()
    if not t or not p:
        return False
    pattern = r"(?<![A-Za-z0-9])" + re.escape(p) + r"(?![A-Za-z0-9])"
    return re.search(pattern, t) is not None


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
    """Deterministic signed feature hashing of lexical tokens and character trigrams.

    This is lexical similarity, not a trained semantic embedding model.
    """
    dim = max(8, min(2048, int(dim)))
    tokens = tokenize(_normalize_text(text)[:_ADVCU_MAX_INPUT])
    if not tokens:
        return [0.0] * dim
    vector = [0.0] * dim
    features = Counter("w:" + token for token in tokens)
    for token in tokens:
        padded = '^' + token + '$'
        for i in range(max(0, len(padded) - 2)):
            features['g:' + padded[i:i + 3]] += 0.35
    for feature, weight in features.items():
        digest = hashlib.blake2b(feature.encode('utf-8'), digest_size=8).digest()
        idx = int.from_bytes(digest[:4], 'big') % dim
        sign = -1 if digest[4] & 1 else 1
        vector[idx] += sign * min(weight, 4)
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else [0.0] * dim


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
            # Select embedding model via Globals resolver (exactly one per category + fallbacks)
            try:
                res = {}
                try:
                    res = config.resolve_model("embeddings", text="", meta=None, models_dir=getattr(config, "MODELS_DIR", None)) or {}
                except Exception:
                    res = {}
                selected = res.get("selected")
                fallbacks = res.get("fallbacks") or []
                candidates = [c for c in ([selected] + list(fallbacks)) if c]
            except Exception:
                candidates = []

            if not candidates:
                # POOR tier with no user overrides => core-only embeddings
                _ST_EMBEDDER = None
                logger.info("[AdvCU] Embeddings: core-only (no 3rd-party model selected)")
                _ST_EMBEDDER_LOADED = True
                return None

            # Try candidates in order; prefer local folder under MODELS_DIR
            models_dir = getattr(config, "MODELS_DIR", None) or os.path.join(os.getcwd(), "data", "models")
            model_root = Path(models_dir).resolve()
            for repo in candidates[:8]:
                try:
                    name = str(repo).strip().replace("\\", "/")
                    if not name or name.startswith("/") or ".." in name.split("/") or ":" in name:
                        continue
                    local_dir1 = model_root / name.replace("/", "_")
                    local_dir2 = model_root / name
                    selected_path = None
                    for candidate_path in (local_dir1, local_dir2):
                        resolved_path = candidate_path.resolve()
                        if resolved_path.is_relative_to(model_root) and resolved_path.is_dir():
                            selected_path = resolved_path
                            break
                    if selected_path is None:
                        continue  # A resolver identifier is not permission to fetch a model.
                    _ST_EMBEDDER = SentenceTransformer(str(selected_path), local_files_only=True)
                    logger.info("[AdvCU] Loaded local embedding model from approved models directory")
                    break
                except Exception as e:
                    logger.debug(f"[AdvCU] ST load failed for {repo}: {e}")
                    _ST_EMBEDDER = None
            if _ST_EMBEDDER is None:
                logger.debug("[AdvCU] No local SentenceTransformer model available")

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
    """Bounded lexical vector score (0..1), not a neural-semantic certainty."""
    a = _normalize_text(text1)
    b = _normalize_text(text2)
    if not a or not b:
        return 0.0
    if a.casefold() == b.casefold():
        return 1.0
    tokens_a, tokens_b = set(tokenize(a)), set(tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    lexical = len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))
    vector = max(0.0, cosine_similarity(_hash_embed(a, 256), _hash_embed(b, 256)))
    return round(max(0.0, min(1.0, 0.55 * lexical + 0.45 * vector)), 6)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Validate equal vector dimensions and finite values before cosine comparison."""
    try:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        a = [float(v) for v in vec1]
        b = [float(v) for v in vec2]
        if not all(math.isfinite(v) for v in a + b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return max(-1.0, min(1.0, dot / (na * nb))) if na and nb else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


# ============================================================================
# DETECTION HELPERS
# ============================================================================

def _advcu_phrase(text: str, term: str) -> bool:
    """Bounded whole-word or whole-phrase matching, never substring escalation."""
    if not isinstance(text, str) or not isinstance(term, str) or not term.strip():
        return False
    return bool(re.search(r"(?<![a-z0-9_])" + re.escape(term.strip().lower()) +
                          r"(?![a-z0-9_])", text.lower()))


def _advcu_any_phrase(text: str, terms) -> bool:
    return any(_advcu_phrase(text, term) for term in terms)


def _advcu_informational(text: str) -> bool:
    """Distinguish questions about actions from requests to execute those actions."""
    t = _normalize_text(text).lower().strip()
    if not t:
        return False
    if re.match(r"^(?:what|why|how|who|where|when|which|define|explain|describe|"
                r"tell me about|teach me|please explain|can you explain|could you explain|"
                r"show me how to|what does|what is|what are)\b", t):
        return True
    if re.match(r"^(?:can|could|would|will) you (?:tell|explain|describe|show|teach)\b", t):
        return True
    return False


def _advcu_direct_action(text: str) -> bool:
    """Recognize a requested operation, not a quoted instruction or a discussion."""
    t = _normalize_text(text).lower().strip()
    if not t or _advcu_informational(t):
        return False
    t = re.sub(r"^(?:hey sarah[, ]*|sarah[, ]*|please\s+)", "", t).strip()
    return bool(re.match(r"^(?:(?:can|could|would|will) you\s+|i want you to\s+|"
                         r"please\s+)?(?:open|launch|start|run|execute|close|quit|"
                         r"exit|stop|kill|pause|resume|turn|enable|disable|increase|"
                         r"decrease|set|play|search|find|go to|navigate to|take me to|"
                         r"show|create|make|generate|design|render|produce|compose|build|draw|write|send|save|delete|install|reboot|"
                         r"restart|shut down|mute|unmute|maximize|minimize|focus)\b", t))


def _advcu_explicit_visual(text: str) -> bool:
    t = _normalize_text(text).lower()
    strong = ("image", "photo", "picture", "camera", "webcam", "screenshot", "video frame",
              "in the scene", "what do you see", "look at", "in the image", "in this image", "color of", "colour of",
              "what color", "what colour", "read the sign", "read this sign", "read the label",
              "read the screen", "read this text", "read this label", "ocr", "license plate", "track motion",
              "track the", "safety zone", "hazard zone", "distance to", "how far away",
              "how close is", "what does my shirt say", "what does the sign say",
              "is this outfit", "does this shirt", "detect objects", "detect faces")
    return _advcu_any_phrase(t, strong)


def _likely_math(text: str) -> bool:
    """Arithmetic only; never treat model names, versions or UTF-8 as math."""
    return extract_math_expression(text) is not None


def _looks_like_url(text: str) -> bool:
    """Check if text contains a URL or domain."""
    if "http://" in text or "https://" in text:
        return True
    if DOMAIN_REGEX.search(text):
        return True
    return False


def _looks_like_login(text: str) -> bool:
    t = _normalize_text(text).lower()
    return bool(not _advcu_informational(t) and _advcu_any_phrase(t,
        ("login", "log in", "sign in", "sign-in", "authenticate", "enter password", "enter pin")))


def _looks_like_signup(text: str) -> bool:
    t = _normalize_text(text).lower()
    return bool(not _advcu_informational(t) and _advcu_any_phrase(t,
        ("sign up", "signup", "create account", "register account", "new account")))


def _looks_like_device_query(text: str) -> bool:
    """Hardware status/identity about this device, not generic software education."""
    t = _normalize_text(text).lower()
    if _advcu_informational(t) and not _advcu_any_phrase(t, ("my cpu", "my gpu", "my ram", "my device", "my computer", "my pc", "your cpu", "your gpu", "your ram", "your processor", "your hardware", "current cpu", "current gpu", "system status", "system stats", "hardware status")):
        return False
    explicit = ("my cpu", "my gpu", "my ram", "my computer", "my device", "my pc", "your cpu", "your gpu", "your ram", "your processor", "your hardware",
                "cpu usage", "cpu temp", "cpu temperature", "gpu temp", "gpu temperature",
                "cpu speed", "gpu speed", "disk space", "free space", "drive space",
                "how much ram", "hardware status", "device status", "system stats", "system info",
                "my hardware", "my processor", "nvidia-smi", "vram usage", "storage usage")
    if _advcu_any_phrase(t, explicit):
        return True
    if re.search(r"^(?:show|check|report|get|display|read|inspect)\b", t):
        return _advcu_any_phrase(t, ("gpu", "cpu", "ram", "vram", "hardware", "device", "disk", "storage", "temperature"))
    return False

def _looks_like_diagnostics(text: str) -> bool:
    t = text.lower()
    keywords = [
        "diagnostic",
        "diagnostics",
        "health check",
        "self test",
        "self-test",
        "system check",
        "run diagnostics",
    ]
    return any(k in t for k in keywords)


def _looks_like_time_query(text: str) -> bool:
    t = _normalize_text(text).lower()
    if not t:
        return False
    explicit = ("what time", "current time", "time now", "today's date", "what's today",
                "what day is", "schedule", "remind me", "set alarm", "calendar", "appointment",
                "meeting tomorrow", "meeting today", "meeting next", "date today")
    if _advcu_any_phrase(t, explicit):
        return True
    # Generic prepositions 'in/on/at/this' do not express a temporal question.
    temporal = ("today", "tonight", "tomorrow", "yesterday", "minutes", "minute", "hours",
                "hour", "weeks", "week", "month", "months", "year", "years", "noon",
                "midnight", "morning", "afternoon", "evening", "next friday", "next monday")
    return bool(_advcu_any_phrase(t, temporal) and
                (re.search(r"^(?:when|what time|how long|how many|how soon)\b", t) or
                 _advcu_any_phrase(t, ("remind", "meeting", "deadline", "schedule", "alarm"))))


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
    t = _normalize_text(text).lower()
    if not _advcu_direct_action(t):
        return False
    return _advcu_any_phrase(t, ("volume", "brightness", "mute", "unmute", "wifi", "bluetooth",
                                 "airplane mode", "restart", "reboot", "hibernate", "sleep",
                                 "turn on", "turn off", "shut down", "system settings"))


def _detect_action_subject(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Surface action and named target; quoted/discussed operations are not actions."""
    t = _normalize_text(text).lower()
    action_label = None
    if _advcu_direct_action(t) and not _advcu_informational(t):
        for action_key, synonyms in ACTION_SYNONYMS.items():
            if _advcu_any_phrase(t, synonyms):
                action_label = action_key
                break
    subject_key = None
    for key, meta in SUBJECT_LEXICON.items():
        if _advcu_phrase(t, key) or _advcu_any_phrase(t, meta.get("aka", ())):
            subject_key = key
            break
    return action_label, subject_key




_SURFACE_DRAW_SHAPES = ("circle", "square", "triangle", "rectangle", "oval", "line", "star")
_SURFACE_APP_ALIASES = {
    "word": "winword", "microsoft word": "winword", "ms word": "winword", "winword": "winword",
    "excel": "excel", "microsoft excel": "excel", "ms excel": "excel",
    "powerpoint": "powerpnt", "microsoft powerpoint": "powerpnt", "ms powerpoint": "powerpnt", "powerpnt": "powerpnt",
    "paint": "mspaint", "microsoft paint": "mspaint", "mspaint": "mspaint",
    "notepad": "notepad", "vscode": "code", "visual studio code": "code", "vs code": "code", "code": "code",
    "dreamweaver": "dreamweaver",
}


def _canonical_surface_exec(app_name: Optional[str]) -> str:
    app = str(app_name or '').strip().lower()
    if not app:
        return ''
    app = app.replace('.exe', '').strip()
    return _SURFACE_APP_ALIASES.get(app, app)


def extract_surface_task(text: str, preferred_app: Optional[str] = None) -> Dict[str, Any]:
    t = _normalize_text(text)
    low = t.lower()
    if not _advcu_direct_action(low) or _advcu_informational(low):
        return {}
    result: Dict[str, Any] = {}
    requested_app = ''
    for alias in sorted(_SURFACE_APP_ALIASES, key=len, reverse=True):
        if _advcu_phrase(low, alias):
            requested_app = alias
            break
    if preferred_app and not requested_app:
        requested_app = str(preferred_app)
    canonical = _canonical_surface_exec(requested_app)
    if canonical:
        result['requested_app'] = requested_app or canonical
        result['requested_app_exec'] = canonical
    open_named = re.search(r'\bopen\b.*?\b(?:document|file|spreadsheet|workbook|project|page|website)\b\s*(?:named|called)\s+(.+?)(?:\s+(?:and\s+(?:write|create|make)|about|with)\b|$)', t, re.I)
    if open_named:
        result['task_kind'] = 'open_named_document'
        result['document_name'] = open_named.group(1).strip('" .')[:256]
        return result
    title_m = re.search(r'\btitled\s+(.+?)(?:\s+and\s+(?:write|create|make)|\s+about|\s+with|$)', t, re.I)
    if title_m:
        result['title'] = title_m.group(1).strip('" .')[:256]
    about_matches = re.findall(r'\b(?:about|on|regarding)\s+([^?.!,;]+)', t, re.I)
    if about_matches:
        result['topic'] = about_matches[-1].strip(' ?.,')[:256]
    if _advcu_any_phrase(low, ('website','webpage','homepage','about page')):
        result['task_kind'] = 'website_scaffold'
        pages=[]
        if _advcu_any_phrase(low, ('homepage','home page')): pages.append('index')
        if _advcu_phrase(low,'about page'): pages.append('about')
        result['pages'] = pages or ['index','about']
        return result
    if _advcu_any_phrase(low, ('checkbook','track my spending','spreadsheet','workbook')) and _advcu_any_phrase(low, ('create','make','build','write')):
        result['task_kind'] = 'spreadsheet_template'
        result['template_kind'] = 'checkbook' if _advcu_any_phrase(low, ('checkbook','spending')) else 'table'
        return result
    if _advcu_phrase(low,'draw'):
        result['task_kind'] = 'paint_draw'
        shape_m = re.search(r'\b(' + '|'.join(_SURFACE_DRAW_SHAPES) + r')\b', low)
        result['draw_subject'] = shape_m.group(1) if shape_m else 'drawing'
        return result
    if _advcu_any_phrase(low, ('write','create','make','build')) and _advcu_any_phrase(low, ('document','report','essay','letter','page')):
        result['task_kind'] = 'write_document'
        result['document_text'] = t[:2048]
    return result

def _extract_keywords(text: str, limit: int = 8) -> List[str]:
    """Extract compact routing keywords from text."""
    out: List[str] = []
    seen: Set[str] = set()
    for tok in tokenize(text):
        t = tok.lower().strip()
        if not t or t in _SEMANTIC_STOPWORDS:
            continue
        if len(t) == 1 and not t.isdigit():
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
        if len(out) >= limit:
            break
    return out


def _flatten_subject_tokens(subject: Optional[str]) -> Set[str]:
    if not subject:
        return set()
    return set(tokenize(subject.lower()))


def _extract_attributes(
    text: str,
    subject: Optional[str] = None,
    action: Optional[str] = None,
    output_type: Optional[str] = None,
    limit: int = 8
) -> List[str]:
    """Extract descriptive attributes that are not the primary subject/action."""
    subject_tokens = _flatten_subject_tokens(subject)
    action_tokens = _flatten_subject_tokens(action)
    output_tokens = _flatten_subject_tokens(output_type)
    ignored = subject_tokens | action_tokens | output_tokens | _SEMANTIC_STOPWORDS | _CREATIVE_VERBS

    attrs: List[str] = []
    seen: Set[str] = set()
    for tok in tokenize(text):
        t = tok.lower().strip()
        if not t or t in ignored:
            continue
        if len(t) == 1 and not t.isdigit():
            continue
        if t not in seen:
            seen.add(t)
            attrs.append(t)
        if len(attrs) >= limit:
            break
    return attrs

def _infer_subject_from_text(
    text: str,
    output_type: Optional[str],
    action: Optional[str],
    existing_subject: Optional[str] = None
) -> Optional[str]:
    """Infer the semantic topic/subject when no explicit app/site subject exists."""
    if existing_subject:
        return existing_subject
    keywords = _extract_keywords(text, limit=10)
    ignored: Set[str] = set()
    if output_type:
        ignored |= _flatten_subject_tokens(output_type)
    if action:
        ignored |= _flatten_subject_tokens(action)
    ignored |= _CREATIVE_VERBS
    for kw in keywords:
        if kw in ignored:
            continue
        return kw
    return None




def _normalize_visual_subject(subject: Optional[str]) -> Optional[str]:
    """Normalize a likely vision subject into a stable canonical phrase."""
    s = _normalize_text(subject).lower().strip(" ?!,.:-")
    if not s:
        return None
    s = re.sub(r"\b(my|the|this|that|these|those|a|an)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return None
    if s in _VISUAL_SUBJECT_SYNONYMS:
        s = _VISUAL_SUBJECT_SYNONYMS[s]
    for k, v in _VISUAL_SUBJECT_SYNONYMS.items():
        if s == k or s.endswith(" " + k):
            return v
    return s


def _contains_any_phrase(text: str, phrases: Tuple[str, ...]) -> bool:
    return _advcu_any_phrase(_normalize_text(text), phrases)


# --- SARAHMEMORY REALITY PATCH 2026-07-23: model/knowledge routing guards ---
_MODEL_ARCHITECTURE_HINTS: Tuple[str, ...] = (
    "local model", "sovereign model", "llm", "large language model", "tokenizer", "tokenize",
    "adapter", "lora", "qlora", "fine tune", "finetune", "train", "training", "qist", "sel",
    "semantic tokenization", "contextual drift", "hallucination", "activation", "ablation", "model forensics",
)
_GENERAL_KNOWLEDGE_STARTERS: Tuple[str, ...] = (
    "what is ", "what are ", "who is ", "who are ", "why ", "how does ", "how do ",
    "explain ", "define ", "describe ", "tell me about ",
)

def _looks_like_model_architecture_request(text: str) -> bool:
    lt = _normalize_text(text).lower()
    strong = tuple(x for x in _MODEL_ARCHITECTURE_HINTS if x not in
                   ("train", "training", "adapter", "activation", "model forensics"))
    if _advcu_any_phrase(lt, strong):
        return True
    return bool(_advcu_any_phrase(lt, ("train", "training", "adapter", "activation")) and
                _advcu_any_phrase(lt, ("model", "llm", "tokenizer", "ai", "neural network")))


def _looks_like_general_knowledge_question(text: str) -> bool:
    lt = _normalize_text(text).lower()
    if not lt:
        return False
    if _looks_like_model_architecture_request(lt):
        return False
    visual_guard_phrases = ("what do you see", "look at", "in this image", "in the picture", "in the photo")
    if any(p in lt for p in visual_guard_phrases):
        return False
    visual_guard_words = ("image", "picture", "photo", "camera", "webcam", "screenshot", "frame", "video", "screen")
    if any(re.search(rf"(?<![a-z0-9_]){re.escape(w)}(?![a-z0-9_])", lt) for w in visual_guard_words):
        return False
    return any(lt.startswith(prefix) for prefix in _GENERAL_KNOWLEDGE_STARTERS)

# --- END SARAHMEMORY REALITY PATCH 2026-07-23 ---


def _detect_visual_query_type(
    text: str, subject: Optional[str] = None, output_type: Optional[str] = None,
    intent_label: Optional[str] = None,
) -> Optional[str]:
    """Require perceptual evidence; generic words like 'say' never open a vision lane."""
    lt = _normalize_text(text).lower()
    if not lt or _looks_like_model_architecture_request(lt):
        return None
    # Mentioning an image in an instructional, editing, or generation request
    # is not evidence that the user is asking to perceive an optical frame.
    if intent_label in ("creative", "build_request"):
        return None
    if _advcu_direct_action(lt) and _advcu_any_phrase(lt, ("generate", "create", "make", "draw", "design", "open", "launch", "save", "write", "render", "produce")):
        return None
    if re.match(r"^(?:how (?:do|can|to)|what (?:is|are)|explain|define)\b", lt) and not _advcu_any_phrase(lt,
        ("what color", "what colour", "color of", "colour of", "what do you see", "what is in", "read the", "in this image", "in the image", "what is written", "how far", "how close")):
        return None
    if not _advcu_explicit_visual(lt):
        return None
    if _contains_any_phrase(lt, _VISUAL_SAFETY_HINTS):
        return "inspect_safety_zone"
    if _advcu_any_phrase(lt, ("before and after", "compare the image", "compare these photos")):
        return "compare_before_after"
    if _advcu_any_phrase(lt, ("look good", "look bad", "look fat", "outfit", "fit", "style")):
        return "assess_style_or_fit"
    if _advcu_any_phrase(lt, ("what color", "what colour", "color of", "colour of", "dominant color", "identify color")):
        return "identify_color"
    if _advcu_any_phrase(lt, ("read the", "read this", "ocr", "license plate", "what does my shirt say", "what does the sign say", "text in")):
        return "read_text"
    if _advcu_any_phrase(lt, ("track", "motion", "moving", "movement")):
        return "track_motion"
    if _advcu_any_phrase(lt, ("distance", "how far", "how close", "proximity")):
        return "estimate_distance"
    if _advcu_any_phrase(lt, ("face", "faces", "facial", "eye color", "eyes")):
        return "detect_faces"
    if _advcu_any_phrase(lt, ("what do you see", "describe the scene", "summarize this image")):
        return "scene_summary"
    return "detect_objects" if _advcu_any_phrase(lt, ("object", "objects", "identify", "detect")) else "scene_summary"


def _extract_visual_subject_from_text(text: str, query_type: Optional[str], fallback_subject: Optional[str] = None) -> Optional[str]:
    """Best-effort extraction of the subject being visually queried."""
    lt = _normalize_text(text).lower()
    patterns = [
        r"(?:color|colour)\s+(?:of|for)\s+(?:my|the|this|that)?\s*([a-z0-9 _-]{1,48})",
        r"what\s+(?:color|colour)\s+(?:is|are)\s+(?:my|the|this|that)?\s*([a-z0-9 _-]{1,48})",
        r"what\s+does\s+(?:my|the|this|that)?\s*([a-z0-9 _-]{1,48})\s+say",
        r"read\s+(?:the|my|this|that)?\s*([a-z0-9 _-]{1,48})",
        r"does\s+(?:my|the|this|that)?\s*([a-z0-9 _-]{1,48})\s+make\s+me\s+look",
        r"is\s+(?:the|my|this|that)?\s*([a-z0-9 _-]{1,48})\s+too\s+close",
        r"is\s+(?:the|my|this|that)?\s*([a-z0-9 _-]{1,48})\s+in\s+(?:the\s+)?(?:danger|safe|safety|hazard)\s+zone",
        r"track\s+(?:the|my|this|that)?\s*([a-z0-9 _-]{1,48})",
        r"compare\s+(?:the|my|this|that)?\s*([a-z0-9 _-]{1,48})",
    ]
    for pat in patterns:
        m = re.search(pat, lt)
        if m:
            subj = _normalize_visual_subject(m.group(1))
            if subj:
                return subj

    cleaned = re.sub(r"[^a-z0-9\s_-]", " ", lt)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    for cand in sorted(_VISUAL_SUBJECT_CANDIDATES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(cand)}\b", cleaned):
            return _normalize_visual_subject(cand)

    return _normalize_visual_subject(fallback_subject)


def _visual_attributes_for_query_type(query_type: Optional[str]) -> List[str]:
    mapping = {
        "identify_color": ["color"],
        "read_text": ["text"],
        "detect_objects": ["objects"],
        "detect_faces": ["faces"],
        "track_motion": ["motion"],
        "estimate_distance": ["distance"],
        "inspect_safety_zone": ["distance", "safety_zone"],
        "compare_before_after": ["comparison", "change"],
        "assess_style_or_fit": ["style", "fit"],
        "scene_summary": ["scene"],
    }
    return list(mapping.get(query_type or "", []))


def _infer_action_expectation(text: str, query_type: Optional[str], intent_family: Optional[str], lane: Optional[str]) -> Optional[str]:
    lt = _normalize_text(text).lower()
    if query_type == "inspect_safety_zone":
        return "governed_action_candidate" if _advcu_direct_action(lt) and _advcu_any_phrase(lt, _VISUAL_ACTION_HINTS) else "answer_with_monitoring_guidance"
    if query_type in ("track_motion", "estimate_distance"):
        return "answer_with_monitoring_guidance" if _advcu_any_phrase(lt, ("track", "monitor", "follow", "alert", "notify", "report", "warn")) else "answer_only"
    if (lane == "action" or intent_family == "ACTION") and _advcu_direct_action(lt):
        return "governed_action_candidate"
    return "answer_only"


def _merge_priority_attributes(priority: List[str], existing: List[str], limit: int = 8) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for seq in (priority or [], existing or []):
        for item in seq:
            t = _normalize_text(item).lower().strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
            if len(out) >= limit:
                return out
    return out


def _detect_output_type(text: str, intent_label: str, action: Optional[str] = None) -> Optional[str]:
    t = _normalize_text(text).lower()
    if _looks_like_model_architecture_request(t):
        return "model_architecture"
    if intent_label in ("diagnostics", "device_query"):
        return "system_status"
    if _advcu_informational(t) and not _advcu_explicit_visual(t):
        return "text_answer"
    if intent_label == "math":
        return "answer"
    # Artifact type only when it was actually requested, not merely mentioned.
    if (_advcu_direct_action(t) or intent_label in ("creative", "build_request")) and _advcu_any_phrase(t, _CREATIVE_VERBS):
        for kind, hints in _CREATIVE_OUTPUT_KEYWORDS.items():
            if _advcu_any_phrase(t, hints):
                return kind
    if action in ("open_app", "open_url", "search_web", "play_media", "window_mgmt", "close_quit", "system_control"):
        return "action"
    if intent_label in ("diagnostics", "device_query"):
        return "system_status"
    return None


def _detect_intent_family_and_lane(
    text: str,
    intent_label: str,
    action: Optional[str],
    output_type: Optional[str],
    math_expr: Optional[str]
) -> Tuple[str, str]:
    """Map parsed meaning to an intent family + primary lane."""
    t = (text or "").lower()

    if intent_label == "diagnostics":
        return "SYSTEM", "system"

    # SARAHMEMORY REALITY PATCH 2026-07-23:
    # Route local-model/tokenizer/SEL/QIST work into model governance, not creative/avatar lanes.
    if output_type == "model_architecture" or _looks_like_model_architecture_request(t):
        return "MODEL_GOVERNANCE", "model"

    if any(h in t for h in _NETWORK_HINTS) or intent_label == "network_request":
        return "NETWORK", "network"

    if output_type in ("image", "video", "music", "avatar", "website"):
        return "CREATIVE", "creative"

    if action in ("open_app", "open_url", "search_web", "play_media", "window_mgmt", "close_quit", "system_control", "agent_control"):
        return "ACTION", "action"

    if math_expr or intent_label in (
        "math", "question", "identity_query", "smalltalk", "greeting",
        "farewell", "statement", "time_related", "device_query"
    ):
        return "ANSWER", "answer"

    if any(v in t for v in _CREATIVE_VERBS) and any(
        hint in t for hints in _CREATIVE_OUTPUT_KEYWORDS.values() for hint in hints
    ):
        return "CREATIVE", "creative"

    if any(v in t for v in _FILE_ACTION_HINTS):
        return "ACTION", "action"

    return "ANSWER", "answer"


def _module_hints_for_packet(lane: str, output_type: Optional[str], intent_label: str) -> List[str]:
    """Suggest likely owner/helper modules for downstream routing."""
    if lane == "system":
        return ["SarahMemoryDiagnostics"]
    if lane == "network":
        return ["SarahMemoryNetwork", "appnet"]
    if lane == "creative":
        if output_type == "video":
            return ["SarahMemoryVideoEditorCore", "SarahMemoryCanvasStudio"]
        if output_type == "music":
            return ["SarahMemoryMusicGenerator", "SarahMemoryLyricsToSong"]
        if output_type == "avatar":
            return ["SarahMemoryCanvasStudio"]
        if output_type == "website":
            return ["SarahMemoryCanvasStudio", "SarahMemoryAiFunctions"]
        return ["SarahMemoryCanvasStudio"]
    if lane == "action":
        return ["SarahMemoryAiFunctions"]
    if lane == "model":
        return ["SarahMemoryLLM", "SarahMemoryPreTokenAnalyzer", "SarahMemoryQuantumSafe", "SarahMemorySynapes"]
    if intent_label == "math":
        return ["SarahMemoryLogicCalc"]
    return ["SarahMemoryWebSYM", "SarahMemoryResearch"]


def _infer_query_type(
    text: str,
    intent_label: str,
    output_type: Optional[str],
    math_expr: Optional[str],
    visual_query_type: Optional[str] = None,
) -> str:
    """Produce a normalized query_type field for compact semantic packets."""
    t = (text or "").lower()
    if math_expr or intent_label == "math":
        return "deterministic_math"
    if visual_query_type:
        return visual_query_type
    if intent_label == "diagnostics":
        return "diagnostics"
    if output_type == "model_architecture" or intent_label == "model_governance":
        return "model_architecture_request"
    if output_type in ("image", "video", "music", "avatar"):
        return "creative_request"
    if output_type == "website":
        return "build_request"
    if any(k in t for k in _NETWORK_HINTS):
        return "network_request"
    if intent_label in ("open_app", "open_url", "search_web", "play_media", "window_mgmt", "close_quit", "system_control", "agent_control", "command"):
        return "action_request"
    if intent_label in ("question", "identity_query", "device_query", "time_related"):
        return "identity_or_definition" if "what is" in t or "who is" in t else "question"
    return "statement"


def _build_semantic_summary(
    intent_label: str,
    subject: Optional[str],
    attributes: List[str],
    output_type: Optional[str],
    query_type: Optional[str],
    math_expr: Optional[str]
) -> str:
    """Generate a concise semantic summary for routing and logging."""
    if math_expr:
        return f"user is asking to solve math expression {math_expr}"
    bits: List[str] = []
    if query_type:
        bits.append(query_type.replace("_", " "))
    if output_type and output_type != "answer":
        bits.append(f"output={output_type}")
    if subject:
        bits.append(f"subject={subject}")
    if attributes:
        bits.append("attributes=" + ", ".join(attributes[:4]))
    if not bits:
        bits.append(f"intent={intent_label}")
    return "user request: " + " | ".join(bits)


def _build_helper_payload_from_command(cmd: ParsedCommand) -> Dict[str, Any]:
    """Build the compressed semantic helper payload for helper-only LLM use."""
    payload: Dict[str, Any] = {
        "intent": cmd.intent,
        "intent_family": cmd.intent_family,
        "task": None,
        "topic": cmd.subject,
        "attribute": (cmd.attributes[0] if cmd.attributes else None),
        "attributes": list(cmd.attributes[:4]),
        "keywords": list(cmd.keywords[:6]),
        "query_type": cmd.query_type,
        "action_expectation": cmd.action_expectation,
        "output_type": cmd.output_type,
        "math_expr": cmd.math_expr,
        "semantic_summary": cmd.semantic_summary,
    }
    surface_task = (cmd.extra or {}).get("surface_task") if isinstance(cmd.extra, dict) else None
    if isinstance(surface_task, dict) and surface_task:
        payload["surface_task"] = dict(surface_task)

    if cmd.math_expr:
        payload["task"] = "solve exact expression"
    elif cmd.output_type in ("image", "video", "music", "avatar", "website"):
        payload["task"] = f"fill missing creative/build details for {cmd.output_type}"
    elif cmd.intent_family == "ANSWER":
        payload["task"] = "infer concise likely concept"
    elif cmd.intent_family == "ACTION":
        payload["task"] = "fill missing action target details only"
    elif cmd.intent_family == "NETWORK":
        payload["task"] = "fill missing network intent details only"
    else:
        payload["task"] = "fill missing semantic gap only"

    if cmd.query_type in _VISUAL_QUERY_TYPES:
        payload["vision"] = {
            "query_type": cmd.query_type,
            "requested_subject": cmd.subject,
            "requested_attributes": list((cmd.attributes or [])[:4]),
            "action_expectation": cmd.action_expectation or "answer_only",
            "needs_optical_input": True,
            "module_hints": [m for m in (cmd.module_hints or []) if m in ("SarahMemorySOBJE", "SarahMemoryFacialRecognition")],
        }

    return {k: v for k, v in payload.items() if v not in (None, "", [], {})}



# ============================================================================
# INTENT CLASSIFICATION
# ============================================================================

def _rule_based_intent(text: Any) -> str:
    """Offline deterministic semantic classification; no action authority granted."""
    t = _normalize_text(text)
    if not t:
        return "statement"
    lt = t.lower()
    if _looks_like_diagnostics(lt) and not _advcu_informational(lt):
        return "diagnostics"
    if _looks_like_device_query(lt):
        return "device_query"
    if _looks_like_time_query(lt):
        return "time_related"
    if _likely_math(t):
        return "math"
    if _looks_like_general_knowledge_question(lt) or _advcu_informational(lt):
        return "identity_query" if _looks_like_identity(lt) else "question"
    if _looks_like_login(lt):
        return "login"
    if _looks_like_signup(lt):
        return "signup"
    if _advcu_direct_action(t):
        if _advcu_any_phrase(t, ("stop", "halt", "pause", "resume", "cancel")) and _advcu_any_phrase(t, ("sarah", "agent", "agents", "mission", "task")):
            return "agent_control"
        if _looks_like_system_control(t):
            return "system_control"
        action, subj = _detect_action_subject(t)
        if action:
            return action
        if _advcu_any_phrase(lt, _NETWORK_HINTS):
            return "network_request"
        if _advcu_any_phrase(lt, _CREATIVE_VERBS) and any(_advcu_any_phrase(lt, hints) for hints in _CREATIVE_OUTPUT_KEYWORDS.values()):
            if _advcu_any_phrase(lt, _CREATIVE_OUTPUT_KEYWORDS["website"] + _CREATIVE_OUTPUT_KEYWORDS["application"]):
                return "build_request"
            return "creative"
        return "command"
    if _looks_like_greeting(lt):
        return "greeting"
    if _looks_like_farewell(lt):
        return "farewell"
    if _looks_like_identity(lt):
        return "identity_query"
    if _looks_like_url(t):
        return "question" if t.strip().endswith('?') else "statement"
    if QUESTION_STARTERS.search(lt) or t.strip().endswith('?'):
        return "smalltalk" if _advcu_any_phrase(lt, ("how are you", "how's it going", "how do you feel", "are you okay", "what's up")) else "question"
    return "statement"


def _load_intent_override_cache() -> Dict[str, str]:
    """Read bounded, validated local overrides; user-file contents never grant execution."""
    try:
        if not os.path.isfile(INTENT_CACHE_PATH) or os.path.getsize(INTENT_CACHE_PATH) > 256 * 1024:
            return {}
        with open(INTENT_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        allowed = {x.value for x in IntentType} | {"model_governance"}
        if not isinstance(data, dict):
            return {}
        return {k: v for k, v in list(data.items())[:512]
                if isinstance(k, str) and 0 < len(k) < 512 and isinstance(v, str) and v in allowed}
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("[AdvCU] override cache unavailable: %s", exc)
        return {}


def classify_intent_with_confidence(text: Any, user_context: Any = None, device_context: Any = None) -> Tuple[str, float]:
    """Deterministic intent label with heuristic (not calibrated probability) score."""
    t = _normalize_text(text)[:_ADVCU_MAX_INPUT]
    if not t:
        return "statement", 0.0
    cache = _load_intent_override_cache()
    if t in cache:
        return cache[t], 0.92
    label = _rule_based_intent(t)
    evidence = {
        "math": 0.94, "open_app": 0.88, "open_url": 0.85,
        "greeting": 0.86, "farewell": 0.86, "device_query": 0.80,
        "system_control": 0.82, "diagnostics": 0.83, "time_related": 0.79,
        "question": 0.78, "identity_query": 0.83, "statement": 0.56,
    }
    confidence = evidence.get(label, 0.73)
    if len(t) < 5 and label not in ("greeting", "farewell"):
        confidence = min(confidence, 0.55)
    if label in ("command", "agent_control", "close_quit") and not _advcu_direct_action(t):
        confidence = min(confidence, 0.45)
    # Database audit is explicitly opt-in. Never persist private input by default.
    if bool(getattr(config, 'ADVCU_ENABLE_EVENT_LOGGING', False)):
        log_advcu_event("Classify Intent", json.dumps({"text_sha256": hashlib.sha256(t.encode('utf-8')).hexdigest(),
                                                        "label": label, "heuristic_confidence": confidence}))
    return label, float(confidence)


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
    """Return a standalone numeric expression without reinterpreting identifiers."""
    t = _normalize_text(text)
    if not t or len(t) > _ADVCU_MAX_INPUT:
        return None
    math = r"-?\d+(?:\.\d+)?(?:\s*[+*/%^\-]\s*-?\d+(?:\.\d+)?)+"
    # Do not accept expressions embedded in identifiers or version-number chains.
    for m in re.finditer(r"(?<![\w.])" + math + r"(?![\w.])", t):
        candidate = m.group(0)
        if re.search(r"[+*/%^\-]", candidate):
            return candidate.strip()
    bare = t.strip().rstrip('?!')
    if re.fullmatch(r"[\d\s().+*/%^\-]+", bare) and re.search(r"[+*/%^\-]", bare):
        return bare
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
    - Compact semantic packet data for governed routing

    Args:
        text: Input text to parse
        user_context: Optional user context
        device_context: Optional device context

    Returns:
        ParsedCommand dataclass with extracted information
    """
    t = _normalize_text(text)
    label, confidence = classify_intent_with_confidence(t, user_context=user_context, device_context=device_context)

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

    output_type = _detect_output_type(t, label, action=action)
    # SARAHMEMORY REALITY PATCH 2026-07-23:
    # Correct the old creative/avatar misroute for local-model/tokenizer/SEL/QIST work.
    if output_type == "model_architecture":
        label = "model_governance"
        confidence = max(float(confidence or 0.0), 0.82)
        action = None
    surface_task = extract_surface_task(t, preferred_app=app or subject)
    if surface_task.get("requested_app") and not app:
        app = str(surface_task.get("requested_app"))
        if not subject:
            subject = app

    if label == "agent_control":
        action = "agent_control"
    elif label == "diagnostics":
        action = None
    elif label in ("device_query", "time_related", "identity_query", "question") and not _advcu_direct_action(t):
        action = None
    # Semantic subject fallback from natural language/topic extraction
    if not math_expr:
        subject = _infer_subject_from_text(
            t,
            output_type=output_type,
            action=action,
            existing_subject=subject
        )

    visual_query_type = _detect_visual_query_type(
        t,
        subject=subject,
        output_type=output_type,
        intent_label=label,
    )
    if visual_query_type:
        is_question_form = bool(re.match(r"^\s*(is|are|does|do|what|which|who|can|could|should|would|how)\b", t.lower()))
        if is_question_form and label in ("close_quit", "command", "system_control"):
            label = "question"
            confidence = max(float(confidence or 0.0), 0.72)
        if is_question_form and action in ("close_quit", "system_control"):
            action = None
        if output_type in ("music", "video", "avatar", "website", "application", "action") and not any(v in t.lower() for v in _CREATIVE_VERBS):
            output_type = None
        visual_subject = _extract_visual_subject_from_text(t, visual_query_type, fallback_subject=subject)
        if visual_subject:
            subject = visual_subject

    intent_family, likely_lane = _detect_intent_family_and_lane(
        t,
        label,
        action=action,
        output_type=output_type,
        math_expr=math_expr
    )
    if _advcu_informational(t) and output_type != "model_architecture" and not visual_query_type:
        intent_family, likely_lane = "ANSWER", "answer"
        action = None
    query_type = _infer_query_type(t, label, output_type, math_expr, visual_query_type=visual_query_type)
    if surface_task and surface_task.get("task_kind"):
        task_kind = str(surface_task.get("task_kind") or "").strip().lower()
        intent_family = "ACTION"
        likely_lane = "action"
        if task_kind == "website_scaffold":
            output_type = "website"
            query_type = "website_scaffold"
        elif task_kind == "spreadsheet_template":
            output_type = "spreadsheet"
            query_type = "spreadsheet_template"
        elif task_kind == "paint_draw":
            output_type = "image"
            query_type = "paint_draw"
        elif task_kind == "open_named_document":
            output_type = output_type or "document"
            query_type = "open_named_document"
        else:
            output_type = output_type or "document"
            query_type = "document_write"
    keywords = _extract_keywords(t)
    attributes = _extract_attributes(t, subject=subject, action=action, output_type=output_type)
    if visual_query_type:
        attributes = _merge_priority_attributes(_visual_attributes_for_query_type(visual_query_type), attributes)
    # SARAHMEMORY REALITY PATCH 2026-07-23:
    # Model governance requests may be advisory or build candidates, but they are never avatar output.
    if output_type == "model_architecture":
        attributes = _merge_priority_attributes(["local_model", "tokenizer", "governance"], attributes)
    action_expectation = _infer_action_expectation(t, query_type, intent_family, likely_lane)
    semantic_summary = _build_semantic_summary(
        label,
        subject=subject,
        attributes=attributes,
        output_type=output_type,
        query_type=query_type,
        math_expr=math_expr
    )
    module_hints = _module_hints_for_packet(likely_lane, output_type, label)
    if visual_query_type:
        vision_helpers = ["SarahMemorySOBJE", "SarahMemoryFacialRecognition"]
        module_hints = vision_helpers + [m for m in module_hints if m not in vision_helpers]

    # Store context in extra
    extra["user_context"] = user_context
    extra["device_context"] = device_context
    extra["tokens"] = tokenize(t)
    extra["query_type"] = query_type
    extra["intent_family"] = intent_family
    extra["likely_lane"] = likely_lane
    extra["output_type"] = output_type
    extra["action_expectation"] = action_expectation
    extra["module_hints"] = module_hints
    if surface_task:
        extra["surface_task"] = dict(surface_task)
    if visual_query_type:
        extra["vision"] = {
            "query_type": visual_query_type,
            "requested_subject": subject,
            "requested_attributes": list(_visual_attributes_for_query_type(visual_query_type)),
            "action_expectation": action_expectation,
        }

    parsed = ParsedCommand(
        intent=label,
        raw_text=t,
        confidence=confidence,
        math_expr=math_expr,
        url=url,
        app=app,
        site=site,
        subject=subject,
        action=action,
        intent_family=intent_family,
        likely_lane=likely_lane,
        output_type=output_type,
        query_type=query_type,
        action_expectation=action_expectation,
        semantic_summary=semantic_summary,
        attributes=attributes,
        keywords=keywords,
        module_hints=module_hints,
        extra=extra
    )
    parsed.helper_payload = _build_helper_payload_from_command(parsed)
    extra["helper_payload"] = dict(parsed.helper_payload)
    extra["semantic_packet"] = parsed.to_semantic_packet()
    return parsed


def build_semantic_packet(
    text: Any,
    user_context: Any = None,
    device_context: Any = None
) -> Dict[str, Any]:
    """
    Build the canonical compact semantic packet used by Neuron and helper lanes.

    This is the governed semantic compression output described by SM_FULL_IO.TXT.
    """
    parsed = parse_command(text, user_context=user_context, device_context=device_context)
    packet = parsed.to_semantic_packet()
    packet["raw_text"] = parsed.raw_text
    packet["target"] = parsed.get_target()
    return packet


def build_helper_payload(
    text: Any,
    user_context: Any = None,
    device_context: Any = None
) -> Dict[str, Any]:
    """
    Build the compressed helper-only LLM payload.

    Third-party/helper models should receive this compact payload instead of the
    raw prompt whenever local structure is already understood.
    """
    parsed = parse_command(text, user_context=user_context, device_context=device_context)
    return dict(parsed.helper_payload or _build_helper_payload_from_command(parsed))




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


def _ensure_db() -> bool:
    """Initialize existing schema once per path, only when a write is requested."""
    global _db_initialized_path
    if _db_initialized_path == SYSTEM_DB_PATH and os.path.isfile(SYSTEM_DB_PATH):
        return True
    with _db_lock:
        if _db_initialized_path == SYSTEM_DB_PATH and os.path.isfile(SYSTEM_DB_PATH):
            return True
        try:
            os.makedirs(os.path.dirname(SYSTEM_DB_PATH), exist_ok=True)
            with closing(sqlite3.connect(SYSTEM_DB_PATH, timeout=5.0)) as conn:
                conn.executescript(_SCHEMA_SQL_CODE_CORPUS)
                conn.commit()
            _db_initialized_path = SYSTEM_DB_PATH
            return True
        except (sqlite3.Error, OSError) as exc:
            logger.warning("[AdvCU] Database setup unavailable: %s", exc)
            return False


def _advcu_read_connection():
    """SQLite read-only URI; avoids unintentionally creating functions.db."""
    return sqlite3.connect(Path(SYSTEM_DB_PATH).resolve().as_uri() + "?mode=ro",
                           timeout=2.0, uri=True)


def _ensure_code_corpus_db() -> None:
    """Alias for _ensure_db for backward compatibility."""
    _ensure_db()


def log_advcu_event(event: str, details: str) -> None:
    """Explicit bounded audit persistence; never auto-called without opt-in."""
    if not bool(getattr(config, 'ADVCU_ENABLE_EVENT_LOGGING', False)):
        return
    try:
        _ensure_db()
        with sqlite3.connect(SYSTEM_DB_PATH, timeout=3.0) as conn:
            conn.execute("INSERT INTO advcu_log(ts, event, details) VALUES (?, ?, ?)",
                         (_get_timestamp(), str(event)[:128], str(details)[:1024]))
    except (OSError, sqlite3.Error) as exc:
        logger.debug("[AdvCU] audit persistence unavailable: %s", exc)


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


def _advcu_sensitive_literal(value: str, key: str = "") -> bool:
    """Conservatively omit private source literals; no attempt to store raw secrets."""
    s = str(value or '')
    k = str(key or '').lower()
    if len(s) > 2048 or any(ord(c) < 32 and c not in '\n\t\r' for c in s):
        return True
    if re.search(r"(?:^|[_\W])(?:password|passwd|secret|credential|private_key|access_token|refresh_token|api_key|client_secret|auth_token|bearer)(?:$|[_\W])", k):
        return True
    if re.search(r"(?:-----BEGIN [^-]+PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|"
                 r"(?:password|passwd|secret|api[_-]?key|access[_-]?token|bearer)\s*[:=]\s*[^\s]+)", s, re.I):
        return True
    if _RX_EMAIL.search(s) or re.search(r"https?://[^\s/@]+:[^\s/@]+@", s):
        return True
    return False


class _NodeMiner(ast.NodeVisitor):
    """AST visitor to extract values from Python source."""

    def __init__(self, filename: str):
        self.filename = filename
        self.rows: List[Tuple[str, str, str, int]] = []
        self._seen = set()

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
        """Raw unlabeled literals are excluded: source context is needed for privacy filtering."""
        return None

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
        """Bounded, sensitive-data filtered source metadata; never archive credentials."""
        if len(self.rows) >= _ADVCU_MAX_MINED_ROWS_PER_FILE:
            return
        key = str(key or '')[:128]
        snippet = str(snippet or '').strip()
        if not snippet or _advcu_sensitive_literal(snippet, key):
            return
        snippet = _normalize_snippet(snippet)[:1024]
        ident = (kind, key, snippet, int(line))
        if ident not in self._seen:
            self._seen.add(ident)
            self.rows.append(ident)

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
    """Enumerate a bounded set of ordinary CORE source files without recursion."""
    base = Path(getattr(config, "BASE_DIR", os.getcwd())).resolve()
    if not base.is_dir():
        return []
    files=[]
    for path in sorted(base.glob("SarahMemory*.py")):
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size > _ADVCU_MAX_SOURCE_BYTES:
                continue
            if path.resolve().parent != base:
                continue
            files.append(str(path))
            if len(files) >= _ADVCU_MAX_SOURCE_FILES:
                break
        except OSError:
            continue
    return files


def _scan_file(path: str) -> List[Tuple[str, str, str, int]]:
    """Bounded AST-only source analysis; never execute inspected code."""
    try:
        entry = Path(path)
        if entry.is_symlink() or not entry.is_file() or entry.stat().st_size > _ADVCU_MAX_SOURCE_BYTES:
            return []
        with entry.open('r', encoding='utf-8', errors='replace') as handle:
            src = handle.read(_ADVCU_MAX_SOURCE_BYTES + 1)
        if len(src.encode('utf-8', errors='ignore')) > _ADVCU_MAX_SOURCE_BYTES:
            return []
        tree = ast.parse(src, filename=str(entry))
        miner = _NodeMiner(entry.name)
        miner.visit(tree)
        if '"messages"' in src and _RX_MSGSKEY.search(src) and len(miner.rows) < _ADVCU_MAX_MINED_ROWS_PER_FILE:
            miner.rows.append(('prompt', 'messages', '[messages payload detected]', 0))
        return miner.rows
    except (OSError, SyntaxError, ValueError, UnicodeError) as exc:
        logger.debug("[AdvCU] cannot AST-scan %s: %s", os.path.basename(path), exc)
        return []


def warm_code_corpus(force: bool = False) -> Dict[str, Any]:
    """Bounded incremental AST corpus refresh, never called on normal boot.

    Original code_corpus/code_corpus_seen schema is preserved. Modified or removed
    source files replace only their own indexed data inside a transaction.
    """
    with _corpus_lock:
        files = _iter_core_files()
        fingerprints = {}
        for name in files:
            try:
                stat = os.stat(name)
                fingerprints[os.path.basename(name)] = [int(stat.st_size), int(stat.st_mtime_ns)]
            except OSError:
                continue
        cache = get_corpus_stats()
        old_fingerprints = cache.get('file_fingerprints') if isinstance(cache, dict) else None
        if not isinstance(old_fingerprints, dict):
            old_fingerprints = {}
        if bool(cache.get('_ok')) and not force and fingerprints == old_fingerprints:
            return cache
        changed = sorted(key for key in fingerprints if force or fingerprints[key] != old_fingerprints.get(key))
        removed = sorted(set(old_fingerprints) - set(fingerprints))
        new_rows = []
        scanned = 0
        for name in changed:
            fp = next((x for x in files if os.path.basename(x) == name), None)
            if fp is None:
                continue
            for kind, key, snippet, line in _scan_file(fp):
                scanned += 1
                # Include original location in PK to preserve repeated literals in other files.
                raw = '\x1f'.join((name, str(line), kind, key, snippet))
                digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()
                new_rows.append((kind, key, snippet, name, int(line), digest))
        result = {'_ok': False, 'generated': datetime.utcnow().isoformat(),
                  'counts': {'files': len(files), 'scanned': scanned, 'new': 0,
                             'changed_files': len(changed), 'removed_files': len(removed)},
                  'file_fingerprints': fingerprints}
        if not _ensure_db():
            result['error'] = 'database_unavailable'
            return result
        try:
            with closing(sqlite3.connect(SYSTEM_DB_PATH, timeout=6.0)) as conn:
                conn.execute('BEGIN IMMEDIATE')
                for name in changed + removed:
                    conn.execute('DELETE FROM code_corpus WHERE file=?', (name,))
                    conn.execute('DELETE FROM code_corpus_seen WHERE file=?', (name,))
                stamp = result['generated']
                for kind, key, snippet, name, line, digest in new_rows:
                    conn.execute('INSERT OR IGNORE INTO code_corpus_seen(snip_hash,file,line,first_ts) VALUES (?,?,?,?)',
                                 (digest, name, line, stamp))
                    conn.execute('INSERT INTO code_corpus(kind,key,snippet,file,line,ts) VALUES (?,?,?,?,?,?)',
                                 (kind, key, snippet, name, line, stamp))
                conn.commit()
            result['_ok'] = True
            result['counts']['new'] = len(new_rows)
            # Cache is optimization only: atomic replacement after committed DB transaction.
            os.makedirs(os.path.dirname(CODE_CORPUS_JSON), exist_ok=True)
            tmp = CODE_CORPUS_JSON + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as handle:
                json.dump(result, handle, indent=2)
            os.replace(tmp, CODE_CORPUS_JSON)
        except (OSError, sqlite3.Error) as exc:
            result['_ok'] = False
            result['error'] = type(exc).__name__
            logger.warning('[AdvCU] incremental code corpus failed: %s', exc)
        return result


def get_corpus_stats() -> Dict[str, Any]:
    """Read existing cache only, with size bound and schema check."""
    try:
        if os.path.isfile(CODE_CORPUS_JSON) and os.path.getsize(CODE_CORPUS_JSON) < 256 * 1024:
            with open(CODE_CORPUS_JSON, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {'_ok': False}
    except (OSError, ValueError, TypeError):
        pass
    return {'_ok': False}


def mine_hardcoded_knowledge(limit: Optional[int] = None) -> Dict[str, List[Dict[str, Any]]]:
    """Read a bounded, already prepared local source corpus."""
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not os.path.isfile(SYSTEM_DB_PATH):
        return out
    requested = 500 if limit is None else max(0, min(800, int(limit)))
    if requested == 0:
        return out
    try:
        with closing(_advcu_read_connection()) as conn:
            cur = conn.execute('SELECT kind,key,snippet,file,line FROM code_corpus ORDER BY id DESC LIMIT ?', (requested,))
            for kind, key, snippet, file, line in cur:
                if not _advcu_sensitive_literal(snippet, key):
                    out.setdefault(kind, []).append({'key': key, 'snippet': snippet, 'file': file, 'line': line})
    except (OSError, sqlite3.Error, ValueError) as exc:
        logger.debug('[AdvCU] code knowledge read failed: %s', exc)
    return out


def contextualize_with_code(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    """Bounded local lexical shortlist then vector scoring; never bulk-load a corpus."""
    query = _normalize_text(query)[:512]
    if not query or not os.path.isfile(SYSTEM_DB_PATH):
        return []
    n = max(1, min(32, int(top_k)))
    terms = [x for x in tokenize(query) if len(x) > 2][:6]
    try:
        with closing(_advcu_read_connection()) as conn:
            if terms:
                filters = ' OR '.join('(lower(snippet) LIKE ? OR lower(key) LIKE ?)' for _ in terms)
                args = [p for t in terms for p in ('%' + t.replace('%', '').replace('_', '') + '%',) * 2]
                rows = list(conn.execute('SELECT kind,key,snippet,file,line FROM code_corpus WHERE ' + filters +
                                         ' ORDER BY id DESC LIMIT ?', (*args, _ADVCU_MAX_CORPUS_RESULTS)))
            else:
                rows = list(conn.execute('SELECT kind,key,snippet,file,line FROM code_corpus ORDER BY id DESC LIMIT ?',
                                         (_ADVCU_MAX_CORPUS_RESULTS,)))
    except (OSError, sqlite3.Error) as exc:
        logger.debug('[AdvCU] corpus retrieval failed: %s', exc)
        return []
    candidates = []
    for kind, key, snip, file, line in rows:
        if _advcu_sensitive_literal(snip, key):
            continue
        lexical = evaluate_similarity(query, (key or '') + ' ' + (snip or ''))
        candidates.append((lexical, kind, key, snip, file, line))
    candidates.sort(key=lambda record: record[0], reverse=True)
    shortlisted = candidates[:max(64, n * 3)]
    if not shortlisted:
        return []
    # One model batch, and a lexical fallback when no local embedding is selected.
    try:
        vectors = embed_text([query] + [rec[3][:1024] for rec in shortlisted])
        query_vec = vectors[0]
    except (ValueError, TypeError, IndexError):
        vectors = []
        query_vec = None
    scored = []
    for idx, (lexical, kind, key, snip, file, line) in enumerate(shortlisted):
        vector_score = max(0.0, cosine_similarity(query_vec, vectors[idx + 1])) if query_vec and len(vectors) > idx + 1 else 0.0
        score = min(1.0, max(0.0, 0.65 * lexical + 0.35 * vector_score))
        scored.append({'score': round(score, 4), 'kind': kind, 'key': key,
                       'snippet': snip, 'file': file, 'line': line})
    return sorted(scored, key=lambda rec: rec['score'], reverse=True)[:n]


def verify_corpus_status() -> Dict[str, Any]:
    """Read-only corpus diagnostics; never mutate or create a database."""
    out = {'db_rows': 0, 'seen_hashes': 0, 'cache': get_corpus_stats()}
    if not os.path.isfile(SYSTEM_DB_PATH):
        return out
    try:
        with closing(_advcu_read_connection()) as conn:
            out['db_rows'] = int(conn.execute('SELECT COUNT(*) FROM code_corpus').fetchone()[0])
            out['seen_hashes'] = int(conn.execute('SELECT COUNT(*) FROM code_corpus_seen').fetchone()[0])
    except (OSError, sqlite3.Error):
        out['status'] = 'unavailable'
    return out


def ensure_corpus_ready() -> None:
    """Explicit request to verify the bounded incremental source index."""
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
    """Stable bounded HSV-like RGB classification; rejects non-finite channels."""
    try:
        from colorsys import rgb_to_hsv
        vals = [float(v) for v in (r, g, b)]
        if not all(math.isfinite(v) and 0 <= v <= 255 for v in vals):
            return 'unknown'
        rr, gg, bb = [v / 255.0 for v in vals]
        h, saturation, value = rgb_to_hsv(rr, gg, bb)
    except (ValueError, TypeError, OverflowError):
        return 'unknown'
    if saturation < 0.15:
        if value < 0.06: return 'black'
        if value < 0.18: return 'charcoal'
        if value < 0.30: return 'dark gray'
        if value < 0.43: return 'gray'
        if value < 0.70: return 'silver'
        if value < 0.88: return 'off-white'
        return 'white'
    degrees = h * 360.0
    tone = 'dark' if value < 0.15 else ('light' if value > 0.85 else 'mid')
    if degrees < 15 or degrees >= 345: return 'dark red' if tone == 'dark' else 'red'
    if degrees < 45: return 'brown' if tone == 'dark' else 'orange'
    if degrees < 60: return 'olive' if tone == 'dark' else 'gold'
    if degrees < 80: return 'olive' if tone == 'dark' else 'chartreuse'
    if degrees < 100: return 'olive green' if tone == 'dark' else 'lime'
    if degrees < 130: return 'spring green' if saturation > 0.5 else 'pale green'
    if degrees < 170: return 'teal' if tone == 'dark' else 'cyan'
    if degrees < 210: return 'steel blue' if tone == 'dark' else 'azure'
    if degrees < 250: return 'sky blue' if tone == 'light' else 'blue'
    if degrees < 280: return 'indigo' if tone == 'dark' else 'violet'
    if degrees < 320: return 'plum' if tone == 'dark' else 'magenta'
    if degrees < 345: return 'maroon' if tone == 'dark' else 'rose'
    return 'unknown'


def _dominant_color_and_name(bgr_roi) -> Dict[str, Any]:
    """Bounded color identification with graceful empty/tiny-frame handling."""
    missing = {"hex": "#000000", "rgb": [0, 0, 0], "hsl": [0, 0, 0],
               "name": "unknown", "confidence": 0.0}
    if not _HAS_CV2 or not _HAS_NUMPY or bgr_roi is None:
        return missing
    try:
        roi = np.asarray(bgr_roi)
        if roi.ndim != 3 or roi.shape[2] != 3 or roi.size == 0:
            return missing
        # Deterministic bounded sampling protects CPU and memory on full-HD frames.
        flattened = roi.reshape(-1, 3)
        step = max(1, (len(flattened) + 4095) // 4096)
        pixels = flattened[::step].astype(np.uint8)
        hsv = _cv2.cvtColor(pixels.reshape(-1, 1, 3), _cv2.COLOR_BGR2HSV).reshape(-1, 3)
        mask = (hsv[:, 1] > 18) & (hsv[:, 2] > 25)
        selected = pixels[mask] if mask.any() else pixels
        if selected.shape[0] == 0:
            return missing
        groups = min(3, selected.shape[0])
        if groups == 1:
            b, g, r = map(int, selected[0])
            confidence = 1.0
        else:
            z = np.float32(selected)
            criteria = (_cv2.TERM_CRITERIA_EPS + _cv2.TERM_CRITERIA_MAX_ITER, 24, 1.0)
            _, labels, centers = _cv2.kmeans(z, groups, None, criteria, 2, _cv2.KMEANS_PP_CENTERS)
            counts = np.bincount(labels.flatten(), minlength=groups)
            dominant_idx = int(counts.argmax())
            b, g, r = [int(max(0, min(255, round(x)))) for x in centers[dominant_idx]]
            confidence = float(counts[dominant_idx] / max(1, counts.sum()))
        from colorsys import rgb_to_hls
        hue, lightness, saturation = rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
        return {"hex": f"#{r:02x}{g:02x}{b:02x}", "rgb": [r, g, b],
                "hsl": [float(hue * 360.0), float(saturation), float(lightness)],
                "name": sm_color_name_from_rgb(r, g, b), "confidence": confidence}
    except (ValueError, TypeError, AttributeError, OverflowError, _cv2.error) as exc:
        logger.debug("[AdvCU] color processing unavailable: %s", exc)
        return missing


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

def _log_corpus_boot_message() -> None:
    """Lightweight import-time summary: no source scan or database creation."""
    try:
        cache = get_corpus_stats()
        counts = cache.get('counts') or {}
        logger.debug("[AdvCU] corpus cache: files=%s rows=%s", counts.get('files', 0), counts.get('scanned', 0))
    except Exception:
        pass


def _advcu_warm_code_corpus_on_import() -> bool:
    try:
        env = os.getenv("SARAH_ADVCU_WARM_CODE_CORPUS_ON_IMPORT", "0").strip().lower()
        if env in ("1", "true", "yes", "on"):
            return True
        return bool(getattr(config, "ADVCU_WARM_CODE_CORPUS_ON_IMPORT", False))
    except Exception:
        return False


# Import must be light.  The code corpus can still be warmed manually, but normal
# boot/runtime imports should not scan/write thousands of rows just because a
# helper module was imported by MSDC, SystemLearn, Neuron, or Chat.
try:
    if _advcu_warm_code_corpus_on_import():
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
    "build_semantic_packet",
    "build_helper_payload",
    "ParsedCommand",
    "IntentResult",
    "INTENT_DESCRIPTIONS",
    "INTENT_PRIORITY_ORDER",
    "get_advcu_weight_status",
    "train_advcu_local_weights",
    "score_advcu_local_weights",

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



# -----------------------------------------------------------------------------
# V10/V9G Canonical SelfAware Query Packet bridge
# -----------------------------------------------------------------------------
def _sm_word_or_phrase_match(norm: str, term: str, language_packet: Optional[Dict[str, Any]] = None) -> bool:
    """Safe token/phrase matcher: fan != fantasy, ram != programming."""
    t = str(term or "").strip().lower()
    if not t:
        return False
    try:
        import SarahMemoryCognitiveIdentityLayer as _CIL  # type: ignore
        verdict = _CIL.candidate_blocked_by_language_packet(t, language_packet or {})
        if isinstance(verdict, dict) and verdict.get("blocked"):
            return False
    except Exception:
        pass
    return re.search(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", str(norm or "").lower()) is not None


def _sm_any_word_or_phrase(norm: str, terms: Tuple[str, ...], language_packet: Optional[Dict[str, Any]] = None) -> bool:
    return any(_sm_word_or_phrase_match(norm, term, language_packet) for term in (terms or ()))


def build_selfaware_canonical_query_packet(text: str, context_packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """AdvCU bridge to the deterministic PreToken SelfAware case frame.

    Falls back to a phrase-safe local classifier when PreToken is unavailable.
    """
    try:
        import SarahMemoryPreTokenAnalyzer as _PTA  # type: ignore
        fn = getattr(_PTA, 'build_selfaware_canonical_query_packet', None)
        if callable(fn):
            pkt = fn(text, context_packet=context_packet)
            if isinstance(pkt, dict):
                return pkt
    except Exception:
        pass

    raw = str(text or '')
    norm = re.sub(r'\s+', ' ', raw.strip().lower().replace('temperture', 'temperature'))
    try:
        import SarahMemoryCognitiveIdentityLayer as _CIL  # type: ignore
        lang = _CIL.build_language_context_packet(raw, context_packet=context_packet)
    except Exception:
        lang = {}

    component = (
        'cpu' if _sm_any_word_or_phrase(norm, ('cpu','processor'), lang) else
        'gpu' if _sm_any_word_or_phrase(norm, ('gpu','graphics','video card'), lang) else
        'motherboard' if _sm_any_word_or_phrase(norm, ('motherboard','mainboard','baseboard','system board'), lang) else
        ''
    )
    if _sm_any_word_or_phrase(norm, ('temperature','temp','thermal','heat'), lang):
        metric = 'temperature'
    elif _sm_any_word_or_phrase(norm, ('fan','rpm'), lang):
        metric = 'fan_speed'
    else:
        metric = 'identity'

    kind = (
        'temperature' if metric == 'temperature' else
        'fan_speed' if metric == 'fan_speed' else
        component if component in ('cpu','gpu','motherboard') else
        'network' if _sm_any_word_or_phrase(norm, ('network','ethernet','wi-fi','wifi'), lang) else
        'general_system_fact'
    )
    return {
        'packet_type':'CanonicalQueryPacket',
        'version':'V10_V9G_CANONICAL_QUERY_PACKET',
        'raw_text':raw,
        'normalized_text':norm,
        'domain':'selfaware_body' if kind!='general_system_fact' else 'chat',
        'intent':'body_fact_query' if kind!='general_system_fact' else 'general_chat',
        'requested_component':component,
        'requested_metric':metric,
        'fact_kind':kind,
        'target':component if metric=='temperature' else '',
        'read_only':True,
        'action_taken':False,
        'language_context_packet': lang,
    }

# --- SM V8.0 TRI-LAYER PATCH 2026-05-20 ---
# Packet-aware context bridge. Existing classify_intent/parse_command remain intact.
def build_contextual_intent_packet(
    text: Any,
    language_context_packet: Optional[Dict[str, Any]] = None,
    context_packet: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return AdvCU intent data with Layer-2 language/phrase-lock protection."""
    raw = str(text or "")
    try:
        import SarahMemoryCognitiveIdentityLayer as _CIL  # type: ignore
        lang = language_context_packet if isinstance(language_context_packet, dict) else _CIL.build_language_context_packet(raw, context_packet=context_packet)
    except Exception:
        lang = language_context_packet if isinstance(language_context_packet, dict) else {"raw_text": raw, "phrase_locks": [], "blocked_substring_matches": []}
    label, confidence = classify_intent_with_confidence(raw)
    parsed = parse_command(raw)
    route_guard = {
        "checked": False,
        "no_substring_routing_inside_locked_phrases": False,
        "blocked": [],
    }
    try:
        import SarahMemoryCognitiveIdentityLayer as _CIL  # type: ignore
        route_guard["checked"] = True
        route_guard["no_substring_routing_inside_locked_phrases"] = True
        for candidate in ("fan", "ram", "ai", "ui", "os", "cpu", "gpu"):
            verdict = _CIL.candidate_blocked_by_language_packet(candidate, lang)
            if verdict.get("blocked"):
                route_guard["blocked"].append(verdict)
        # Specific protection: game-lore proper noun questions must not become hardware/system control.
        if lang.get("context_domain") == "gaming_lore" and str(label).lower() in {"system_control", "device_query", "diagnostics"}:
            route_guard["corrected_intent_from"] = label
            label = "question"
            confidence = max(float(confidence or 0.0), 0.76)
    except Exception:
        pass
    out = parsed.to_dict() if hasattr(parsed, "to_dict") else dict(getattr(parsed, "__dict__", {}))
    out.update({
        "packet_type": "ContextualIntentPacket",
        "schema": "SarahMemory.advcu.contextual_intent.v1",
        "intent": label,
        "confidence": float(confidence or 0.0),
        "language_context_packet": lang,
        "route_guard": route_guard,
        "execution_authority": False,
    })
    if get_advcu_weight_status()["ready"]:
        out["local_weight_advice"] = score_advcu_local_weights(raw)
    return out


def classify_intent_contextual(
    text: Any,
    language_context_packet: Optional[Dict[str, Any]] = None,
    context_packet: Optional[Dict[str, Any]] = None,
) -> Tuple[str, float, Dict[str, Any]]:
    pkt = build_contextual_intent_packet(text, language_context_packet=language_context_packet, context_packet=context_packet)
    return str(pkt.get("intent") or "statement"), float(pkt.get("confidence") or 0.0), pkt



# ---------------------------------------------------------------------------
# SARAHMEMORY v9 LOCAL FAST-ANSWER RESOLVER
# ---------------------------------------------------------------------------
# AdvCU owns semantic understanding and pool selection. app.py should call this
# resolver instead of carrying large answer dictionaries. The resolver stays
# local-first: SQLite pools only, local LLM only if a caller separately enables
# that lane, no web/API/shell/device authority.

_SM_ADVCU_FAST_ANSWER_SCHEMA = "SarahMemory.AdvCU.local_fast_answer.v1"


def _advcu_demo_static_facts_enabled() -> bool:
    """Return True only for explicit dry-run/demo static concept tests."""
    try:
        val = str(os.getenv("SARAHMEMORY_DEMO_STATIC_FACTS", "")).strip().lower()
        if val in {"1", "true", "yes", "on", "dryrun", "demo"}:
            return True
    except Exception:
        pass
    try:
        import SarahMemoryGlobals as _cfg  # type: ignore
        return bool(getattr(_cfg, "DEMO_STATIC_FACTS_ENABLED", False))
    except Exception:
        return False


_SM_ADVCU_CORE_CONCEPTS = {
    "api": "An API, or application programming interface, is a defined way for software systems to communicate. It exposes rules, requests, responses, and data formats so one program can safely ask another program or service to do something or provide information.",
    "application programming interface": "An application programming interface, or API, is a defined contract that lets software systems communicate through agreed requests, responses, and data formats.",
    "python": "Python is a high-level programming language known for readable syntax, broad libraries, and common use in automation, web development, data work, scripting, and AI tooling.",
    "photosynthesis": "Photosynthesis is the process plants, algae, and some bacteria use to convert light energy, carbon dioxide, and water into sugars, releasing oxygen as a byproduct.",
    "ram": "RAM, or Random Access Memory, is a computer's short-term working memory. It temporarily holds active data and instructions so the CPU can access them quickly while software is running.",
    "cpu": "A CPU, or central processing unit, is the main processor that executes instructions and coordinates much of a computer's work.",
    "gpu": "A GPU, or graphics processing unit, is a processor optimized for parallel workloads such as graphics, video, and many AI computations.",
    "database": "A database is an organized store of data that software can query, update, and manage. In SarahMemory, SQLite databases act as bounded local memory pools.",
    "sqlite": "SQLite is a lightweight file-based relational database engine. It stores tables, indexes, and records inside local database files without requiring a separate database server.",
    "sql": "SQL, or Structured Query Language, is a language used to query and manage relational databases.",
    "tokenizer": "A tokenizer converts text into smaller units called tokens so a model or language system can process meaning, patterns, and context more consistently.",
    "model": "In AI, a model is a learned or configured system that maps inputs to outputs. In SarahMemory, model authority is governed separately from the architecture that routes and validates it.",
    "sel": "SEL is SarahMemory's semantic expression layer. It frames a request into governed meaning fields such as goal, intent, target, authority, safety, verification, rollback, and memory.",
    "qist": "QIST is SarahMemory's quantum-inspired semantic ranking layer. It classically ranks possible meaning and route candidates without using quantum hardware or granting execution authority.",
}


def _advcu_extract_definition_subject(text: str) -> str:
    lt = _normalize_text(text).lower().strip()
    patterns = (
        r"^what\s+is\s+an?\s+(.+?)\??$",
        r"^what\s+is\s+(.+?)\??$",
        r"^what\s+are\s+(.+?)\??$",
        r"^define\s+(.+?)\??$",
        r"^explain\s+(.+?)\??$",
        r"^describe\s+(.+?)\??$",
        r"^tell\s+me\s+about\s+(.+?)\??$",
    )
    for pat in patterns:
        m = re.match(pat, lt)
        if m:
            subj = (m.group(1) or "").strip(" ?.!")
            subj = re.sub(r"\b(the|a|an)\b", " ", subj)
            subj = re.sub(r"\s+", " ", subj).strip()
            return subj[:96]
    return ""


def _advcu_candidate_from_core_concepts(query: str) -> Optional[Dict[str, Any]]:
    if not _advcu_demo_static_facts_enabled():
        return None
    subject = _advcu_extract_definition_subject(query)
    keys = []
    if subject:
        keys.append(subject)
    norm = _normalize_text(query).lower().strip(" ?.!")
    keys.append(norm)
    # Common article/acronym variants.
    keys.extend([
        norm.replace("what is an ", "", 1),
        norm.replace("what is a ", "", 1),
        norm.replace("what is ", "", 1),
        norm.replace("define ", "", 1),
        norm.replace("explain ", "", 1),
    ])
    for key in keys:
        k = re.sub(r"\s+", " ", str(key or "").lower().strip(" ?.!")).strip()
        if not k:
            continue
        ans = _SM_ADVCU_CORE_CONCEPTS.get(k)
        if ans and _advcu_answer_quality_ok(ans):
            return {
                "answer": ans,
                "source": "demo_advcu_core_concept",
                "runtime_authority": False,
                "dryrun_only": True,
                "db": None,
                "table": None,
                "confidence": 0.91,
                "method": "AdvCU.bounded_core_concepts",
            }
    return None


def _advcu_answer_quality_ok(answer: Any) -> bool:
    text = _normalize_text(answer)
    raw = str(answer or "").strip()
    if not text or len(text) < 3:
        return False
    # Hot-path fast answers must be concise answer material, not whole papers,
    # logs, abstracts, or copied documentation pages from local databases.
    if len(raw) > 1100:
        return False
    low = text.lower()
    bad = (
        "i'm not sure how to respond",
        "i am not sure how to respond",
        "not sure how to respond",
        "could you rephrase",
        "please rephrase",
        "try rephrasing",
        "provide more details",
        "couldn't solve",
        "could not solve",
        "unable to answer",
        "answer unavailable",
        "no answer found",
        "no local match",
        "request denied by policy",
        "user confirmation required",
        "traceback",
        "error:",
        "exception",
    )
    if any(b in low for b in bad):
        return False
    paper_markers = (
        "abstract", "1 introduction", "references", "bibliography", "@", "doi", "arxiv",
        "we present", "we propose", "this paper", "compiler from llvm",
    )
    if sum(1 for b in paper_markers if b in low) >= 2:
        return False
    return True

def _advcu_infer_fast_query_type(text: str, parsed: Optional[ParsedCommand] = None) -> str:
    lt = _normalize_text(text).lower()
    parsed = parsed if isinstance(parsed, ParsedCommand) else None
    query_type = str(getattr(parsed, "query_type", "") or "").strip().lower()
    if _looks_like_model_architecture_request(lt):
        return "model_governance"
    if any(k in lt for k in ("sarahmemoryapi.py", "app.py", "function", "class", "route", "where is", "defined", "code corpus")):
        return "project_code_question"
    if _looks_like_general_knowledge_question(lt):
        return "definition"
    if query_type:
        return query_type
    return "general_chat"


def _advcu_safe_answer_only(text: str, parsed: Optional[ParsedCommand] = None) -> bool:
    """Presentation-only route. Non-authorizing regardless of phrases in questions."""
    t = _normalize_text(text)
    if not t or len(t) > _ADVCU_MAX_INPUT:
        return False
    if parsed is not None:
        if parsed.likely_lane in ("action", "network", "creative") and parsed.action_expectation == "governed_action_candidate":
            return False
        if parsed.intent in ("login", "signup", "agent_control", "system_control") and _advcu_direct_action(t):
            return False
    return bool(_advcu_informational(t) or (parsed is not None and parsed.intent in ("question", "math", "identity_query", "device_query", "time_related")))


def _advcu_candidate_from_search_answers(query: str) -> Optional[Dict[str, Any]]:
    """Return only verified QA-cache records as high-confidence local knowledge."""
    try:
        import SarahMemoryDatabase as _SMDB  # type: ignore
        answers = _SMDB.search_answers(query, include_unverified=False, return_records=True)
        if isinstance(answers, list):
            for rec in answers[:5]:
                ans = rec.get("answer") if isinstance(rec, dict) else rec
                if _advcu_answer_quality_ok(ans):
                    verified = bool(rec.get("verified")) if isinstance(rec, dict) else False
                    if not verified:
                        continue
                    return {
                        "answer": _normalize_text(ans),
                        "source": "verified_local_qa_cache",
                        "source_type": str(rec.get("source_type") or rec.get("source") or "verified_qa_cache") if isinstance(rec, dict) else "verified_qa_cache",
                        "db": "ai_learning.db",
                        "table": "qa_cache",
                        "confidence": 0.88,
                        "verified": True,
                        "verification_state": str(rec.get("verification_state") or "VERIFIED") if isinstance(rec, dict) else "VERIFIED",
                        "method": "SarahMemoryDatabase.search_answers(verified_only)",
                    }
    except TypeError:
        # Older Database contract: do not promote legacy cache into high confidence.
        return None
    except Exception:
        return None
    return None


def _advcu_candidate_from_semantic_pools(query: str, intent: str, query_type: str) -> Optional[Dict[str, Any]]:
    try:
        import SarahMemoryDatabase as _SMDB  # type: ignore
        fn = getattr(_SMDB, "search_semantic_memory_pools", None)
        if callable(fn):
            result = fn(query, intent=intent, query_type=query_type, max_hits=8, time_budget_sec=1.25)
            if isinstance(result, dict):
                for hit in result.get("hits") or []:
                    ans = hit.get("answer") or hit.get("snippet")
                    if _advcu_answer_quality_ok(ans):
                        return {
                            "answer": _normalize_text(ans),
                            "source": hit.get("source") or "local_semantic_db",
                            "db": hit.get("db"),
                            "table": hit.get("table"),
                            "purpose": hit.get("purpose"),
                            "confidence": max(0.0, min(0.65, float(hit.get("score") or 0.50))),
                            "method": result.get("method") or "meta_routed_sqlite_semantic_pool_search",
                            "db_hits": result.get("hits") or [],
                            "routes_checked": result.get("routes_checked"),
                            "latency_ms": result.get("latency_ms"),
                        }
    except Exception:
        return None
    return None


def _advcu_candidate_from_dataset_pool(query: str) -> Optional[Dict[str, Any]]:
    try:
        import SarahMemoryDatabase as _SMDB  # type: ignore
        result = _SMDB.search_local_dataset_pool(query, max_hits=5, time_budget_sec=1.0, max_db_files=8)
        if isinstance(result, dict):
            for hit in result.get("hits") or []:
                ans = hit.get("snippet")
                if _advcu_answer_quality_ok(ans):
                    return {
                        "answer": _normalize_text(ans),
                        "source": "local_dataset_pool",
                        "db": hit.get("db"),
                        "table": hit.get("table"),
                        "confidence": max(0.0, min(0.58, float(hit.get("score") or 0.45))),
                        "method": result.get("method") or "direct_sqlite_dataset_pool_search",
                        "db_hits": result.get("hits") or [],
                        "latency_ms": result.get("latency_ms"),
                    }
    except Exception:
        return None
    return None


def _advcu_record_semantic_sample_if_allowed(
    *,
    text: str,
    answer: str,
    candidate: Dict[str, Any],
    semantic_packet: Dict[str, Any],
    sel_packet: Optional[Dict[str, Any]],
    qist_result: Optional[Dict[str, Any]],
    approved_for_learning: bool = False,
) -> Dict[str, Any]:
    """Best-effort Synapes ledger hook. Never required for answer release."""
    try:
        import SarahMemorySynapes as _SMSyn  # type: ignore
        fn = getattr(_SMSyn, "record_semantic_answer_sample", None)
        if callable(fn):
            return fn(
                raw_text=text,
                answer=answer,
                semantic_packet=semantic_packet,
                sel_packet=sel_packet if isinstance(sel_packet, dict) else {},
                qist_result=qist_result if isinstance(qist_result, dict) else {},
                source={
                    "source": candidate.get("source"),
                    "db": candidate.get("db"),
                    "table": candidate.get("table"),
                    "method": candidate.get("method"),
                },
                confidence=float(candidate.get("confidence") or 0.0),
                approved_for_learning=bool(approved_for_learning),
            )
    except Exception as exc:
        return {"ok": False, "recorded": False, "error": str(exc)}
    return {"ok": False, "recorded": False, "reason": "synapes_recorder_unavailable"}


def resolve_local_fast_answer(
    text: Any,
    context_packet: Optional[Dict[str, Any]] = None,
    sel_packet: Optional[Dict[str, Any]] = None,
    qist_result: Optional[Dict[str, Any]] = None,
    allow_learning_record: bool = False,
    approved_for_learning: bool = False,
) -> Dict[str, Any]:
    """Resolve an answer-only request from governed local memory pools.

    This is the intended replacement for app.py answer-pool bloat. It combines:
      - AdvCU semantic packet + intent/query type
      - meta.db routed SQLite pools through SarahMemoryDatabase
      - optional Synapes semantic/tokenization ledger record

    It never grants execution authority and never uses network/web/API.
    """
    raw = _normalize_text(text)
    started = time.time()
    out: Dict[str, Any] = {
        "ok": False,
        "schema": _SM_ADVCU_FAST_ANSWER_SCHEMA,
        "answer": "",
        "source": "unavailable",
        "confidence": 0.0,
        "execution_authority": False,
        "network_used": False,
        "web_access": False,
        "api_access": False,
        "filesystem_write": bool(allow_learning_record and approved_for_learning) or bool(getattr(config, "ADVCU_ENABLE_EVENT_LOGGING", False)),
        "local_only": True,
        "errors": [],
    }
    if not raw:
        out["errors"].append("empty_query")
        return out

    try:
        parsed = parse_command(raw)
    except Exception as exc:
        parsed = None
        out["errors"].append(f"parse_command:{exc}")

    intent = str(getattr(parsed, "intent", "question") or "question")
    query_type = _advcu_infer_fast_query_type(raw, parsed)
    try:
        semantic_packet = build_semantic_packet(raw)
    except Exception:
        semantic_packet = parsed.to_semantic_packet() if isinstance(parsed, ParsedCommand) else {"raw_text": raw, "intent": intent, "query_type": query_type}
    semantic_packet["query_type"] = query_type
    semantic_packet["execution_authority"] = False

    out.update({
        "intent": intent,
        "query_type": query_type,
        "semantic_packet": semantic_packet,
        "sel_packet": sel_packet if isinstance(sel_packet, dict) else {},
        "qist_result": qist_result if isinstance(qist_result, dict) else {},
    })

    if not _advcu_safe_answer_only(raw, parsed):
        out["blocked_reason"] = "not_answer_only_or_risk_terms_present"
        return out

    candidates: List[Dict[str, Any]] = []
    _terms_for_hotpath = []
    try:
        import SarahMemoryDatabase as _SMDB_terms  # type: ignore
        _terms_for_hotpath = list(getattr(_SMDB_terms, "_sm_dataset_query_terms", lambda _q: [])(raw) or [])
    except Exception:
        _terms_for_hotpath = []
    _short_or_noisy_hotpath = bool(_terms_for_hotpath) and all((" " not in str(t) and len(str(t)) <= 3) for t in _terms_for_hotpath)

    getters = [
        lambda: _advcu_candidate_from_core_concepts(raw),
        lambda: _advcu_candidate_from_search_answers(raw),
        lambda: _advcu_candidate_from_semantic_pools(raw, intent, query_type),
    ]
    if not _short_or_noisy_hotpath:
        getters.append(lambda: _advcu_candidate_from_dataset_pool(raw))

    for getter in getters:
        candidate = getter()
        if isinstance(candidate, dict) and _advcu_answer_quality_ok(candidate.get("answer")):
            candidates.append(candidate)
            break

    if not candidates:
        out["blocked_reason"] = "no_local_semantic_answer"
        out["latency_ms"] = int((time.time() - started) * 1000)
        return out

    best = candidates[0]
    answer = _normalize_text(best.get("answer"))
    out.update({
        "ok": True,
        "answer": answer,
        "source": best.get("source") or "local_semantic_db",
        "confidence": float(best.get("confidence") or 0.70),
        "db": best.get("db"),
        "table": best.get("table"),
        "purpose": best.get("purpose"),
        "method": best.get("method"),
        "db_hits": best.get("db_hits") or [],
        "routes_checked": best.get("routes_checked"),
        "presentation_only": True,
        "execution_authority": False,
        "network_used": False,
        "latency_ms": int((time.time() - started) * 1000),
        "source_verified": bool(best.get("verified", False)),
    })

    if allow_learning_record and approved_for_learning:
        out["synapes_record"] = _advcu_record_semantic_sample_if_allowed(
            text=raw,
            answer=answer,
            candidate=best,
            semantic_packet=semantic_packet,
            sel_packet=sel_packet if isinstance(sel_packet, dict) else {},
            qist_result=qist_result if isinstance(qist_result, dict) else {},
            approved_for_learning=bool(approved_for_learning),
        )
    return out


# Backward-friendly alias for callers that think in terms of lookup rather than resolution.
lookup_local_knowledge_answer = resolve_local_fast_answer

# ====================================================================
# END OF SarahMemoryAdvCU.py v9.1.0a
# ====================================================================

# ==========================================================================
# ADVCU LOCAL ADAPTIVE WEIGHTS (v1) — GOVERNED / IN-MEMORY / ADVISORY ONLY
# ==========================================================================
# This is SarahMemory-owned, dependency-free learned scoring rather than
# third-party model weights or a canned answer pool.  No model downloads,
# on-import training, external calls, automatic disk writes, or action grants.
# Weights are deliberately volatile until an operator approves persistence.
_ADVCU_WEIGHT_SCHEMA = "SarahMemory.AdvCU.local_linear_weights.v1"
_ADVCU_WEIGHT_DIMENSION = 512
_ADVCU_WEIGHT_MAX_SAMPLES_PER_CALL = 128
_ADVCU_WEIGHT_MAX_TOTAL_SAMPLES = 100000
_ADVCU_WEIGHT_MAX_TEXT_LENGTH = 2048
_ADVCU_WEIGHT_LOCK = threading.RLock()
_ADVCU_WEIGHT_ROWS: Dict[str, Dict[int, float]] = {}
_ADVCU_WEIGHT_LABEL_COUNTS: Dict[str, int] = {}
_ADVCU_WEIGHT_SAMPLES_SEEN = 0
_ADVCU_WEIGHT_REVISION = 0


def _advcu_weight_features(text: str) -> Dict[int, float]:
    """Stable bounded sparse features; no raw input is retained in the model."""
    tokens = tokenize(text[:_ADVCU_WEIGHT_MAX_TEXT_LENGTH])[:64]
    if not tokens:
        return {}
    features = Counter("w:" + word for word in tokens)
    features.update("b:" + a + "/" + b for a, b in zip(tokens, tokens[1:]))
    result: Dict[int, float] = {0: 1.0}  # intercept, never hashed
    for feature, count in features.items():
        # A local stable hash, not Python hash() (which varies across processes).
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = 1 + (int.from_bytes(digest, "big") % (_ADVCU_WEIGHT_DIMENSION - 1))
        result[index] = result.get(index, 0.0) + min(3.0, float(count))
    norm = math.sqrt(sum(value * value for value in result.values()))
    return {index: value / norm for index, value in result.items()}


def get_advcu_weight_status() -> Dict[str, Any]:
    """Read-only, non-identifying evidence of this module's learned weight state."""
    with _ADVCU_WEIGHT_LOCK:
        counts = dict(_ADVCU_WEIGHT_LABEL_COUNTS)
        return {
            "schema": _ADVCU_WEIGHT_SCHEMA,
            "implementation": "local_supervised_sparse_linear",
            "dimension": _ADVCU_WEIGHT_DIMENSION,
            "revision": _ADVCU_WEIGHT_REVISION,
            "samples_seen": _ADVCU_WEIGHT_SAMPLES_SEEN,
            "trained_labels": sorted(counts),
            "learned_parameters": sum(len(row) for row in _ADVCU_WEIGHT_ROWS.values()),
            "ready": len(counts) >= 2 and _ADVCU_WEIGHT_SAMPLES_SEEN >= 4,
            "persistent": False,
            "autonomous_training": False,
            "calibrated_probability": False,
            "execution_authority": False,
        }


def train_advcu_local_weights(
    samples: List[Dict[str, str]], *, approved_for_learning: bool = False,
    learning_rate: float = 0.12, epochs: int = 3,
    expected_weight_revision: Optional[int] = None,
    include_training_receipt: bool = False,
    dataset_commitment: Optional[str] = None,
) -> Dict[str, Any]:
    """Learn only from explicitly approved labeled examples, transactionally.

    Nothing about a user's ordinary input, a model answer, or an agent request
    constitutes learning approval. Training never changes the rule classifier. Optional revision fencing rejects stale approved writes atomically.
    """
    if approved_for_learning is not True:
        return {"ok": False, "reason": "explicit_learning_approval_required", "execution_authority": False}
    if type(include_training_receipt) is not bool:
        return {"ok": False, "reason": "invalid_receipt_request", "execution_authority": False}
    if dataset_commitment is not None and (
            not include_training_receipt or not isinstance(dataset_commitment, str) or
            len(dataset_commitment) != 64 or
            any(ch not in "0123456789abcdef" for ch in dataset_commitment)):
        return {"ok": False, "reason": "invalid_dataset_commitment", "execution_authority": False}
    if expected_weight_revision is not None and (
            isinstance(expected_weight_revision, bool) or
            not isinstance(expected_weight_revision, int) or
            not 0 <= expected_weight_revision <= _ADVCU_WEIGHT_MAX_TOTAL_SAMPLES):
        return {"ok": False, "reason": "invalid_expected_weight_revision", "execution_authority": False}
    if not isinstance(samples, (list, tuple)) or not (1 <= len(samples) <= _ADVCU_WEIGHT_MAX_SAMPLES_PER_CALL):
        return {"ok": False, "reason": "sample_count_out_of_bounds", "execution_authority": False}
    if (isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 4 or
            isinstance(learning_rate, bool) or not isinstance(learning_rate, (float, int)) or
            not math.isfinite(float(learning_rate)) or not 0.01 <= learning_rate <= 0.3):
        return {"ok": False, "reason": "invalid_training_hyperparameters", "execution_authority": False}
    allowed_labels = {item.value for item in IntentType} | {"model_governance"}
    validated: List[Tuple[str, Dict[int, float]]] = []
    for item in samples:
        if not isinstance(item, dict):
            return {"ok": False, "reason": "invalid_sample", "execution_authority": False}
        text = item.get("text")
        label = item.get("label")
        if (not isinstance(text, str) or not text.strip() or len(text) > _ADVCU_WEIGHT_MAX_TEXT_LENGTH or
                not isinstance(label, str) or label not in allowed_labels):
            return {"ok": False, "reason": "invalid_sample_label_or_text", "execution_authority": False}
        # Reject unpaired Unicode surrogates before feature hashing; malformed
        # caller text must return a governed denial, never raise mid-batch.
        try:
            text.encode("utf-8")
        except UnicodeError:
            return {"ok": False, "reason": "invalid_sample_unicode", "execution_authority": False}
        features = _advcu_weight_features(text)
        if not features:
            return {"ok": False, "reason": "empty_features", "execution_authority": False}
        validated.append((label, features))

    with _ADVCU_WEIGHT_LOCK:
        global _ADVCU_WEIGHT_ROWS, _ADVCU_WEIGHT_LABEL_COUNTS
        global _ADVCU_WEIGHT_SAMPLES_SEEN, _ADVCU_WEIGHT_REVISION
        if expected_weight_revision is not None and _ADVCU_WEIGHT_REVISION != expected_weight_revision:
            return {"ok": False, "reason": "weight_revision_mismatch",
                    "revision": _ADVCU_WEIGHT_REVISION, "execution_authority": False}
        if _ADVCU_WEIGHT_SAMPLES_SEEN + len(validated) > _ADVCU_WEIGHT_MAX_TOTAL_SAMPLES:
            return {"ok": False, "reason": "lifetime_sample_limit", "execution_authority": False}
        # Copy-on-write: rejection/interruption never leaves a partially trained model.
        rows = {label: dict(values) for label, values in _ADVCU_WEIGHT_ROWS.items()}
        counts = dict(_ADVCU_WEIGHT_LABEL_COUNTS)
        for label, _ in validated:
            rows.setdefault(label, {})
            counts[label] = counts.get(label, 0) + 1
        labels = sorted(rows)
        for _ in range(epochs):
            for truth, features in validated:
                logits = {label: sum(rows[label].get(i, 0.0) * x for i, x in features.items())
                          for label in labels}
                maximum = max(logits.values())
                exp_scores = {label: math.exp(value - maximum) for label, value in logits.items()}
                denominator = sum(exp_scores.values())
                for label in labels:
                    delta = float(learning_rate) * ((1.0 if label == truth else 0.0) -
                                                     exp_scores[label] / denominator)
                    row = rows[label]
                    for index, value in features.items():
                        new_value = max(-4.0, min(4.0, row.get(index, 0.0) + delta * value))
                        if abs(new_value) > 1e-12:
                            row[index] = new_value
                        else:
                            row.pop(index, None)
        _ADVCU_WEIGHT_ROWS = rows
        _ADVCU_WEIGHT_LABEL_COUNTS = counts
        _ADVCU_WEIGHT_SAMPLES_SEEN += len(validated)
        _ADVCU_WEIGHT_REVISION += 1
        result = {"ok": True, "revision": _ADVCU_WEIGHT_REVISION,
                  "samples_accepted": len(validated), "samples_seen": _ADVCU_WEIGHT_SAMPLES_SEEN,
                  "learned_parameters": sum(len(row) for row in rows.values()),
                  "memory_only": True, "advisory_only": True, "execution_authority": False}
        if include_training_receipt:
            # Caller-owned, bounded metadata; no raw text or fingerprint retained.
            # A supplied commitment is an assertion, not independently verified.
            result["training_receipt"] = {
                "schema": "SarahMemory.AdvCU.training_receipt.v1",
                "revision_before": _ADVCU_WEIGHT_REVISION - 1,
                "revision_after": _ADVCU_WEIGHT_REVISION,
                "samples_accepted": len(validated),
                "samples_seen_total": _ADVCU_WEIGHT_SAMPLES_SEEN,
                "epochs": epochs,
                "dataset_commitment": dataset_commitment,
                "commitment_verified": False,
                "audit_ledger_written": False,
                "persisted": False,
                "execution_authority": False,
            }
        return result


def score_advcu_local_weights(text: Any, *, top_k: int = 3) -> Dict[str, Any]:
    """Offer *uncalibrated* evidence for human/governed routing; never execute."""
    status = get_advcu_weight_status()
    response: Dict[str, Any] = {
        "schema": _ADVCU_WEIGHT_SCHEMA,
        "ready": False,
        "advisory_only": True,
        "execution_authority": False,
        "calibrated_probability": False,
        "revision": status["revision"],
        "suggestions": [],
    }
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 5:
        response["reason"] = "invalid_top_k"
        return response
    if not isinstance(text, str) or not text.strip() or len(text) > _ADVCU_WEIGHT_MAX_TEXT_LENGTH:
        response["reason"] = "invalid_text"
        return response
    # Malformed Unicode is not valid scoring evidence; fail closed.
    try:
        text.encode("utf-8")
    except UnicodeError:
        response["reason"] = "invalid_text_unicode"
        return response
    features = _advcu_weight_features(text)
    with _ADVCU_WEIGHT_LOCK:
        # Bind the reported revision and readiness to the exact scoring snapshot.
        # An approved training/import can otherwise advance revision between
        # the initial status read and acquisition of this lock.
        response["revision"] = _ADVCU_WEIGHT_REVISION
        if len(_ADVCU_WEIGHT_LABEL_COUNTS) < 2 or _ADVCU_WEIGHT_SAMPLES_SEEN < 4:
            response["reason"] = "insufficient_approved_training"
            return response
        logits = {label: sum(row.get(i, 0.0) * val for i, val in features.items())
                  for label, row in _ADVCU_WEIGHT_ROWS.items()}
        # Conservative open-set check: unseen vocabulary should not be forced
        # into an unrelated trained category. Hash collisions remain possible.
        known_indices = {index for row in _ADVCU_WEIGHT_ROWS.values() for index in row}
        feature_mass = sum(abs(value) for index, value in features.items() if index != 0)
        covered_mass = sum(abs(value) for index, value in features.items()
                           if index != 0 and index in known_indices)
        coverage = (covered_mass / feature_mass) if feature_mass else 0.0
        revision = _ADVCU_WEIGHT_REVISION
    if not logits:
        response["reason"] = "empty_model"
        return response
    highest = max(logits.values())
    exponentials = {label: math.exp(score - highest) for label, score in logits.items()}
    total = sum(exponentials.values())
    ranked = sorted(exponentials, key=lambda label: (-exponentials[label], label))
    top_score = exponentials[ranked[0]] / total
    runner_up = exponentials[ranked[1]] / total if len(ranked) > 1 else 0.0
    response.update({
        "ready": True,
        "revision": revision,
        "feature_coverage": round(coverage, 6),
        "method": "local_approved_supervised_sparse_linear",
    })
    if coverage < 0.25 or top_score < 0.55 or top_score - runner_up < 0.10:
        response["abstained"] = True
        response["reason"] = "weak_or_unfamiliar_evidence"
        return response
    response["abstained"] = False
    response["suggestions"] = [
        {"label": label, "relative_score": round(exponentials[label] / total, 6)}
        for label in ranked[:top_k]
    ]
    return response


# ==========================================================================
# ADVCU EXPLICIT WEIGHT CHECKPOINT (v2) — NO FILE I/O OR AUTO-RESTORE
# ==========================================================================
def export_advcu_weight_checkpoint(*, approved_for_export: bool = False) -> Dict[str, Any]:
    """Return a JSON-compatible integrity-protected snapshot only on approval.

    The caller owns storage, access control, audit and rollback decisions.
    This function never writes to disk or grants routing/execution authority.
    """
    if approved_for_export is not True:
        return {"ok": False, "reason": "explicit_export_approval_required", "execution_authority": False}
    with _ADVCU_WEIGHT_LOCK:
        payload = {
            "schema": _ADVCU_WEIGHT_SCHEMA,
            "dimension": _ADVCU_WEIGHT_DIMENSION,
            "revision": _ADVCU_WEIGHT_REVISION,
            "samples_seen": _ADVCU_WEIGHT_SAMPLES_SEEN,
            "label_counts": dict(sorted(_ADVCU_WEIGHT_LABEL_COUNTS.items())),
            "rows": {label: {str(index): value for index, value in sorted(row.items())}
                     for label, row in sorted(_ADVCU_WEIGHT_ROWS.items())},
        }
    # Never advertise a checkpoint that the bounded importer cannot accept.
    # The model remains unchanged; the caller may retain its prior snapshot.
    try:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        encoded = canonical.encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        return {"ok": False, "reason": "checkpoint_serialization_failed",
                "execution_authority": False, "persistent": False}
    if len(encoded) > 250000:
        return {"ok": False, "reason": "checkpoint_size_limit",
                "execution_authority": False, "persistent": False}
    return {"ok": True, "payload": payload,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "execution_authority": False, "persistent": False}


def sign_advcu_weight_checkpoint(checkpoint: Any, *, approved_for_signing: bool = False,
                                 provenance_key: Optional[bytes] = None) -> Dict[str, Any]:
    """Explicitly attach HMAC-SHA256 provenance to an existing valid snapshot.

    Key management and authorization remain entirely with the caller. A shared
    secret proves possession of that secret, not an independently audited identity.
    The secret is never returned or persisted by AdvCU.
    """
    denied = {"ok": False, "execution_authority": False}
    if approved_for_signing is not True:
        return {**denied, "reason": "explicit_signing_approval_required"}
    if not isinstance(provenance_key, bytes) or not 32 <= len(provenance_key) <= 65536:
        return {**denied, "reason": "invalid_provenance_key"}
    if not isinstance(checkpoint, dict) or checkpoint.get("ok") is not True:
        return {**denied, "reason": "invalid_checkpoint"}
    payload, digest = checkpoint.get("payload"), checkpoint.get("sha256")
    if not isinstance(payload, dict) or not isinstance(digest, str) or len(digest) != 64:
        return {**denied, "reason": "invalid_checkpoint"}
    try:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(canonical.encode("utf-8")) > 250000:
            raise ValueError("oversized")
        if not hmac.compare_digest(hashlib.sha256(canonical.encode("utf-8")).hexdigest(), digest):
            raise ValueError("digest")
    except (TypeError, ValueError, OverflowError, RecursionError, UnicodeError):
        return {**denied, "reason": "checkpoint_validation_failed"}
    # Bind the signature to the canonical payload and its integrity digest.
    signature = hmac.new(provenance_key, (digest + ":" + canonical).encode("utf-8"), hashlib.sha256).hexdigest()
    signed = dict(checkpoint)
    signed["provenance"] = {"algorithm": "HMAC-SHA256", "signature": signature}
    return signed


def import_advcu_weight_checkpoint(checkpoint: Any, *, approved_for_import: bool = False,
                                   expected_revision: Optional[int] = None,
                                   trusted_provenance_key: Optional[bytes] = None,
                                   minimum_checkpoint_revision: Optional[int] = None) -> Dict[str, Any]:
    """Validate and atomically restore approved local weights from a snapshot.

    SHA-256 detects corruption, not authenticity. Supplying a trusted key
    requires a matching HMAC signature before any state change. Without a key,
    the original integrity-only import behavior is preserved for compatibility.
    This does not train, persist, change the classifier or execute commands.
    minimum_checkpoint_revision is a caller-owned rollback floor; AdvCU does
    not persist a trusted floor, so callers must retain it in governed storage.
    """
    if approved_for_import is not True:
        return {"ok": False, "reason": "explicit_import_approval_required", "execution_authority": False}
    if not isinstance(checkpoint, dict) or checkpoint.get("ok") is not True:
        return {"ok": False, "reason": "invalid_checkpoint", "execution_authority": False}
    # Reject malformed caller revision fences before potentially expensive
    # checkpoint serialization. The final comparison still occurs under the
    # model lock, so this preflight cannot authorize a stale concurrent write.
    if expected_revision is not None and (
            type(expected_revision) is not int or
            not 0 <= expected_revision <= _ADVCU_WEIGHT_MAX_TOTAL_SAMPLES):
        return {"ok": False, "reason": "revision_conflict", "execution_authority": False}
    # Reject oversized/unusable authentication keys before parsing attacker-
    # controlled checkpoint payloads. Approval and revision preflight remain first.
    if trusted_provenance_key is not None and (
            not isinstance(trusted_provenance_key, bytes) or
            not 32 <= len(trusted_provenance_key) <= 65536):
        return {"ok": False, "reason": "invalid_provenance_key", "execution_authority": False}
    # Validate caller-owned rollback floor before serializing untrusted data.
    # The in-lock comparison below remains authoritative for state changes.
    if minimum_checkpoint_revision is not None and (
            type(minimum_checkpoint_revision) is not int or
            not 0 <= minimum_checkpoint_revision <= _ADVCU_WEIGHT_MAX_TOTAL_SAMPLES):
        return {"ok": False, "reason": "invalid_revision_floor", "execution_authority": False}
    payload, digest = checkpoint.get("payload"), checkpoint.get("sha256")
    if not isinstance(payload, dict) or not isinstance(digest, str) or len(digest) != 64:
        return {"ok": False, "reason": "invalid_checkpoint", "execution_authority": False}
    # Bound the checkpoint's nested weight structure before JSON serialization.
    # An untrusted oversized mapping must not require a full canonical copy
    # merely to discover that its shape exceeds the model's fixed limits.
    preflight_counts = payload.get("label_counts")
    preflight_rows = payload.get("rows")
    if (not isinstance(preflight_counts, dict) or not isinstance(preflight_rows, dict) or
            len(preflight_counts) > len(IntentType) + 1 or
            len(preflight_rows) > len(IntentType) + 1 or
            any(not isinstance(row, dict) or len(row) > _ADVCU_WEIGHT_DIMENSION or
                any(not isinstance(index, str) or len(index) > 3 or
                    not index.isascii() or not index.isdecimal()
                    for index in row)
                for row in preflight_rows.values())):
        return {"ok": False, "reason": "checkpoint_validation_failed", "execution_authority": False}
    try:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(canonical.encode("utf-8")) > 250000:
            raise ValueError("oversized")
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if not __import__("hmac").compare_digest(actual, digest):
            raise ValueError("checksum")
        if payload.get("schema") != _ADVCU_WEIGHT_SCHEMA or payload.get("dimension") != _ADVCU_WEIGHT_DIMENSION:
            raise ValueError("schema")
        revision, seen = payload.get("revision"), payload.get("samples_seen")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in (revision, seen)):
            raise ValueError("counter")
        # Imported revisions must remain within the same bounded domain as
        # approved training and revision-fenced checkpoint consumers. An
        # authenticated but out-of-range revision must not poison local state.
        if seen > _ADVCU_WEIGHT_MAX_TOTAL_SAMPLES or revision > _ADVCU_WEIGHT_MAX_TOTAL_SAMPLES:
            raise ValueError("limit")
        # Each approved training revision consumes at least one sample.
        # Reject internally inconsistent checkpoints before replacing state.
        if revision > seen:
            raise ValueError("revision_exceeds_samples")
        counts, encoded_rows = payload.get("label_counts"), payload.get("rows")
        if not isinstance(counts, dict) or not isinstance(encoded_rows, dict):
            raise ValueError("shape")
        allowed = {item.value for item in IntentType} | {"model_governance"}
        if set(counts) != set(encoded_rows) or len(counts) > len(allowed):
            raise ValueError("labels")
        if any(label not in allowed or isinstance(n, bool) or not isinstance(n, int) or n < 1
               for label, n in counts.items()) or sum(counts.values()) != seen:
            raise ValueError("counts")
        rows = {}
        for label, encoded in encoded_rows.items():
            if not isinstance(encoded, dict) or len(encoded) > _ADVCU_WEIGHT_DIMENSION:
                raise ValueError("row")
            row = {}
            for key, value in encoded.items():
                if not isinstance(key, str) or not key.isascii() or not key.isdecimal():
                    raise ValueError("index")
                index = int(key)
                if str(index) != key or index >= _ADVCU_WEIGHT_DIMENSION:
                    raise ValueError("index")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 4.0:
                    raise ValueError("weight")
                row[index] = float(value)
            rows[label] = row
    except (ValueError, TypeError, OverflowError, RecursionError, UnicodeError):
        return {"ok": False, "reason": "checkpoint_validation_failed", "execution_authority": False}
    if trusted_provenance_key is not None:
        provenance = checkpoint.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("algorithm") != "HMAC-SHA256":
            return {"ok": False, "reason": "provenance_required", "execution_authority": False}
        signature = provenance.get("signature")
        if not isinstance(signature, str) or len(signature) != 64:
            return {"ok": False, "reason": "provenance_invalid", "execution_authority": False}
        expected_signature = hmac.new(trusted_provenance_key,
                                      (digest + ":" + canonical).encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected_signature):
            return {"ok": False, "reason": "provenance_invalid", "execution_authority": False}
    global _ADVCU_WEIGHT_ROWS, _ADVCU_WEIGHT_LABEL_COUNTS, _ADVCU_WEIGHT_SAMPLES_SEEN, _ADVCU_WEIGHT_REVISION
    with _ADVCU_WEIGHT_LOCK:
        if minimum_checkpoint_revision is not None and revision < minimum_checkpoint_revision:
            return {"ok": False, "reason": "checkpoint_rollback_rejected", "execution_authority": False}
        if expected_revision is not None and (isinstance(expected_revision, bool) or
                not isinstance(expected_revision, int) or expected_revision != _ADVCU_WEIGHT_REVISION):
            return {"ok": False, "reason": "revision_conflict", "execution_authority": False}
        _ADVCU_WEIGHT_ROWS = rows
        _ADVCU_WEIGHT_LABEL_COUNTS = dict(counts)
        _ADVCU_WEIGHT_SAMPLES_SEEN = seen
        _ADVCU_WEIGHT_REVISION = revision
    return {"ok": True, "revision": revision, "samples_seen": seen,
            "memory_only": True, "execution_authority": False}


# ==========================================================================
# ADVCU BOUNDED HOLDOUT EVALUATION (v3) — READ-ONLY / ADVISORY
# ==========================================================================
def fingerprint_advcu_holdout_texts(
    texts: List[str], *, approved_for_fingerprinting: bool = False,
    fingerprint_key: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Create caller-owned keyed overlap tokens; never retain raw text or keys.

    The operator must independently collect training fingerprints at training
    time and safeguard the secret key. This does not prove dataset independence.
    """
    if approved_for_fingerprinting is not True:
        return {"ok": False, "reason": "explicit_fingerprinting_approval_required", "execution_authority": False}
    if not isinstance(fingerprint_key, bytes) or not 32 <= len(fingerprint_key) <= 65536:
        return {"ok": False, "reason": "invalid_fingerprint_key", "execution_authority": False}
    if not isinstance(texts, (list, tuple)) or not 1 <= len(texts) <= 128:
        return {"ok": False, "reason": "sample_count_out_of_bounds", "execution_authority": False}
    if any(not isinstance(t, str) or not t.strip() or len(t) > _ADVCU_WEIGHT_MAX_TEXT_LENGTH for t in texts):
        return {"ok": False, "reason": "invalid_text", "execution_authority": False}
    # Validate all normalized inputs before hashing any fingerprint. Unicode
    # surrogates must fail closed rather than escape as encoding exceptions.
    normalized = []
    try:
        for text in texts:
            normalized.append(" ".join(text.split()).casefold().encode("utf-8"))
    except UnicodeError:
        return {"ok": False, "reason": "invalid_text_unicode", "execution_authority": False}
    tokens = [hmac.new(fingerprint_key, item, hashlib.sha256).hexdigest()
              for item in normalized]
    return {"ok": True, "fingerprints": tokens, "count": len(tokens),
            "retained_by_advcu": False, "execution_authority": False}


def bind_advcu_training_fingerprints(
    fingerprints: List[str], *, approved_for_binding: bool = False,
    fingerprint_key: Optional[bytes] = None, weight_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Caller-owned HMAC commitment to exact fingerprints and a weight revision.

    The caller must independently establish that these fingerprints represent
    the approved training set. No text, key, or commitment is retained.
    """
    denied = {"ok": False, "execution_authority": False}
    if approved_for_binding is not True:
        return {**denied, "reason": "explicit_binding_approval_required"}
    # Match the bounded fingerprinting key contract; reject excessive keys
    # before canonicalizing caller-owned training fingerprint collections.
    if not isinstance(fingerprint_key, bytes) or not 32 <= len(fingerprint_key) <= 65536:
        return {**denied, "reason": "invalid_fingerprint_key"}
    if (isinstance(weight_revision, bool) or not isinstance(weight_revision, int) or
            not 0 <= weight_revision <= _ADVCU_WEIGHT_MAX_TOTAL_SAMPLES):
        return {**denied, "reason": "invalid_weight_revision"}
    if (not isinstance(fingerprints, (list, tuple)) or
            len(fingerprints) > _ADVCU_WEIGHT_MAX_TOTAL_SAMPLES or
            any(not isinstance(token, str) or len(token) != 64 or
                any(ch not in "0123456789abcdef" for ch in token)
                for token in fingerprints)):
        return {**denied, "reason": "invalid_training_fingerprints"}
    # Sorting and deduplication avoid order-dependent commitments.
    canonical = json.dumps([weight_revision, sorted(set(fingerprints))],
                           separators=(",", ":"), allow_nan=False)
    commitment = hmac.new(fingerprint_key,
                          ("AdvCU-training-fingerprints-v1:" + canonical).encode("utf-8"),
                          hashlib.sha256).hexdigest()
    return {"ok": True, "weight_revision": weight_revision,
            "fingerprint_count": len(set(fingerprints)),
            "commitment": commitment, "execution_authority": False,
            "retained_by_advcu": False}


def evaluate_advcu_local_weights(
    samples: List[Dict[str, str]], *, approved_for_evaluation: bool = False,
    require_unique_texts: bool = False,
    training_text_fingerprints: Optional[List[str]] = None,
    fingerprint_key: Optional[bytes] = None,
    expected_weight_revision: Optional[int] = None,
    training_fingerprint_commitment: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate explicitly supplied labeled holdout samples without training.

    Returns aggregate counts only, not the input text. Evaluation is not proof
    of generalization, calibration, security compliance, or execution authority.
    Callers must ensure holdout examples were not included in training.
    Optional caller-managed HMAC fingerprints reject exact normalized training
    overlap; absence of a match does not establish independent holdout data.
    Set require_unique_texts=True to reject repeated evaluation text after
    whitespace/case normalization; this does NOT prove training independence.
    expected_weight_revision optionally fences the frozen model snapshot against
    the caller's approved revision. A supplied training_fingerprint_commitment
    authenticates caller-provided fingerprints to that revision under the shared
    key, but cannot prove they originated from the actual training examples.
    """
    base = {"ok": False, "execution_authority": False, "trained": False,
            "advisory_only": True}
    if approved_for_evaluation is not True:
        return {**base, "reason": "explicit_evaluation_approval_required"}
    if not isinstance(require_unique_texts, bool):
        return {**base, "reason": "invalid_uniqueness_policy"}
    # Optional caller-owned revision fence: prevent stale holdout reports.
    # A revision alone does not authenticate training fingerprints or datasets.
    if expected_weight_revision is not None and (
            isinstance(expected_weight_revision, bool) or
            not isinstance(expected_weight_revision, int) or
            not 0 <= expected_weight_revision <= _ADVCU_WEIGHT_MAX_TOTAL_SAMPLES):
        return {**base, "reason": "invalid_expected_weight_revision"}
    if not isinstance(samples, (list, tuple)) or not 1 <= len(samples) <= 128:
        return {**base, "reason": "sample_count_out_of_bounds"}
    # Opt-in exact normalized-text overlap screening against caller-provided
    # keyed training fingerprints. Neither the secret nor tokens are retained.
    overlap_screened = training_text_fingerprints is not None
    training_tokens: Set[str] = set()
    if overlap_screened:
        if (not isinstance(fingerprint_key, bytes) or len(fingerprint_key) < 32 or
                not isinstance(training_text_fingerprints, (list, tuple)) or
                len(training_text_fingerprints) > _ADVCU_WEIGHT_MAX_TOTAL_SAMPLES or
                any(not isinstance(token, str) or len(token) != 64 or
                    any(ch not in "0123456789abcdef" for ch in token)
                    for token in training_text_fingerprints)):
            return {**base, "reason": "invalid_training_fingerprints"}
        training_tokens = set(training_text_fingerprints)
    elif fingerprint_key is not None:
        return {**base, "reason": "training_fingerprints_required"}
    if training_fingerprint_commitment is not None:
        if (not overlap_screened or expected_weight_revision is None or
                not isinstance(training_fingerprint_commitment, str) or
                len(training_fingerprint_commitment) != 64 or
                any(ch not in "0123456789abcdef" for ch in training_fingerprint_commitment)):
            return {**base, "reason": "invalid_training_fingerprint_commitment"}
        binding = bind_advcu_training_fingerprints(
            training_text_fingerprints, approved_for_binding=True,
            fingerprint_key=fingerprint_key, weight_revision=expected_weight_revision)
        if (not binding["ok"] or not hmac.compare_digest(
                binding["commitment"], training_fingerprint_commitment)):
            return {**base, "reason": "training_fingerprint_commitment_mismatch"}
    allowed = {item.value for item in IntentType} | {"model_governance"}
    validated = []
    seen_texts: Set[str] = set()  # ephemeral; never retained in model or report
    for sample in samples:
        if not isinstance(sample, dict):
            return {**base, "reason": "invalid_sample"}
        text, label = sample.get("text"), sample.get("label")
        if (not isinstance(text, str) or not text.strip() or
                len(text) > _ADVCU_WEIGHT_MAX_TEXT_LENGTH or
                not isinstance(label, str) or label not in allowed):
            return {**base, "reason": "invalid_sample"}
        if require_unique_texts:
            # Normalization is deterministic, local, and does not expose text.
            # Reject duplicate texts even when their labels disagree; repeated
            # observations would otherwise inflate evaluation sample counts.
            key = " ".join(text.split()).casefold()
            if key in seen_texts:
                return {**base, "reason": "duplicate_evaluation_text"}
            seen_texts.add(key)
        if overlap_screened:
            token = hmac.new(fingerprint_key, " ".join(text.split()).casefold().encode("utf-8"),
                             hashlib.sha256).hexdigest()
            if token in training_tokens:
                return {**base, "reason": "training_evaluation_overlap_detected"}
        validated.append((text, label))
    # Freeze model once, so concurrent training cannot mix revisions in a report.
    with _ADVCU_WEIGHT_LOCK:
        rows = {label: dict(row) for label, row in _ADVCU_WEIGHT_ROWS.items()}
        counts = dict(_ADVCU_WEIGHT_LABEL_COUNTS)
        revision = _ADVCU_WEIGHT_REVISION
        seen = _ADVCU_WEIGHT_SAMPLES_SEEN
    if expected_weight_revision is not None and revision != expected_weight_revision:
        return {**base, "reason": "weight_revision_mismatch", "revision": revision}
    if len(counts) < 2 or seen < 4:
        return {**base, "reason": "insufficient_approved_training", "revision": revision}
    correct = 0
    abstained = 0
    covered = 0
    known = {index for row in rows.values() for index in row}
    confusion: Dict[str, Dict[str, int]] = {}
    for text, truth in validated:
        features = _advcu_weight_features(text)
        mass = sum(abs(v) for i, v in features.items() if i != 0)
        overlap = sum(abs(v) for i, v in features.items() if i != 0 and i in known)
        coverage = overlap / mass if mass else 0.0
        logits = {label: sum(row.get(i, 0.0) * v for i, v in features.items())
                  for label, row in rows.items()}
        maximum = max(logits.values())
        scores = {label: math.exp(value - maximum) for label, value in logits.items()}
        total = sum(scores.values())
        ranked = sorted(scores, key=lambda label: (-scores[label], label))
        best = scores[ranked[0]] / total
        second = scores[ranked[1]] / total if len(ranked) > 1 else 0.0
        if coverage < 0.25 or best < 0.55 or best - second < 0.10:
            predicted = "__abstain__"
            abstained += 1
        else:
            predicted = ranked[0]
            covered += 1
            correct += int(predicted == truth)
        confusion.setdefault(truth, {})[predicted] = confusion.setdefault(truth, {}).get(predicted, 0) + 1
    # Aggregate selective classification metrics. Abstentions count as misses
    # for recall, but not as false positive predictions for any label.
    # This reports observed behavior, not calibrated probabilities or proof
    # that examples were independently held out.
    per_label = {}
    for label in sorted(set(confusion) | {pred for row in confusion.values()
                                               for pred in row if pred != "__abstain__"}):
        support = sum(confusion.get(label, {}).values())
        tp = confusion.get(label, {}).get(label, 0)
        predicted_count = sum(row.get(label, 0) for row in confusion.values())
        per_label[label] = {
            "support": support,
            "predicted": predicted_count,
            "correct": tp,
            "precision": round(tp / predicted_count, 6) if predicted_count else None,
            "recall": round(tp / support, 6) if support else None,
        }
    supported = [v for v in per_label.values() if v["support"]]
    macro_recall = (round(sum(v["recall"] for v in supported) / len(supported), 6)
                    if supported else None)
    n = len(validated)
    return {**base, "ok": True, "revision": revision, "evaluated": n,
            "correct": correct, "abstained": abstained, "answered": covered,
            "accuracy_all": round(correct / n, 6),
            "accuracy_answered": round(correct / covered, 6) if covered else None,
            "coverage": round(covered / n, 6), "confusion_counts": confusion,
            "per_label": per_label, "macro_recall": macro_recall,
            "holdout_independence_verified": False,
            "training_overlap_screened": overlap_screened,
            "evaluation_text_uniqueness_verified": require_unique_texts,
            "calibrated_probability": False}


# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryAdvCU',
    "version": "v9.1.0a-alpha-sml-0.2",
    "category": 'Reasoning',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['mission_discovery', 'reasoning'],
    "supported_missions": ['Conversation', 'Knowledge', 'Planning', 'Programming'],
    "supported_omega": ['Ω001', 'Ω002', 'Ω005', 'Ω010', 'Ω020', 'Ω030', 'Ω040'],
    "required_authority": ['Read'],
    "priority": 70,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryAdvCU.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return dict(SML_ORGAN_METADATA)


def sml_health():
    """Read-only self-evidence, not fabricated availability/latency percentages."""
    available = os.path.isfile(SYSTEM_DB_PATH)
    return {
        "status": "DEGRADED" if not available else "AVAILABLE",
        "availability": 1.0 if available else 0.0,
        "integrity": None,
        "performance": None,
        "reliability": None,
        "confidence": 0.5 if available else 0.25,
        "latency_ms": None,
        "stability": None,
        "compatibility": None,
        "notes": ["Classifier available; SQLite corpus present" if available else
                  "Classifier available; SQLite corpus absent; no measured runtime health"],
    }


def sml_diagnostics():
    """Read-only diagnostic report without implicit DB creation."""
    health = sml_health()
    return {"status": health["status"], "component": "SarahMemoryAdvCU",
            "sml_adapter": True, "metadata": dict(SML_ORGAN_METADATA), "health": health,
            "db_present": os.path.isfile(SYSTEM_DB_PATH), "neural_embedder_loaded": _ST_EMBEDDER is not None,
            "execution_authority": False}


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryAdvCU', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

# --- SML ADVCU SPECIALIZATION START ---
def sml_classify_packet_mission(packet, text=""):
    """Use SML Mission Engine to classify/refine the packet mission."""
    from SarahMemorySMLProtocol import get_protocol, SMLPacket
    pkt = packet if isinstance(packet, SMLPacket) else SMLPacket.from_dict(packet)
    if text and not pkt.payload.get("raw_request"):
        pkt.payload["raw_request"] = str(text)
    return get_protocol().classify_mission(pkt)
# --- SML ADVCU SPECIALIZATION END ---

# ==========================================================================
# ADVCU OPTIONAL TRAINING RECEIPT AUTHENTICATION — CALLER-OWNED, NO I/O
# ==========================================================================
def sign_advcu_training_receipt(receipt: Any, *, approved_for_signing: bool = False,
                                provenance_key: Optional[bytes] = None) -> Dict[str, Any]:
    """Authenticate a bounded v1 receipt; never authenticate the underlying dataset.

    Caller owns the secret and storage. No ledger write or execution permission.
    """
    denied = {"ok": False, "execution_authority": False}
    if approved_for_signing is not True:
        return {**denied, "reason": "explicit_signing_approval_required"}
    if not isinstance(provenance_key, bytes) or len(provenance_key) < 32:
        return {**denied, "reason": "invalid_provenance_key"}
    if not isinstance(receipt, dict) or set(receipt) != {
            "schema", "revision_before", "revision_after", "samples_accepted",
            "samples_seen_total", "epochs", "dataset_commitment", "commitment_verified",
            "audit_ledger_written", "persisted", "execution_authority"}:
        return {**denied, "reason": "invalid_training_receipt"}
    if receipt.get("schema") != "SarahMemory.AdvCU.training_receipt.v1":
        return {**denied, "reason": "invalid_training_receipt"}
    before, after = receipt["revision_before"], receipt["revision_after"]
    accepted, total, epochs = (receipt[k] for k in ("samples_accepted", "samples_seen_total", "epochs"))
    if (any(type(x) is not int for x in (before, after, accepted, total, epochs)) or
            not (0 <= before < _ADVCU_WEIGHT_MAX_TOTAL_SAMPLES and after == before + 1) or
            not (1 <= accepted <= _ADVCU_WEIGHT_MAX_SAMPLES_PER_CALL and
                 accepted <= total <= _ADVCU_WEIGHT_MAX_TOTAL_SAMPLES) or
            not 1 <= epochs <= 4 or
            any(receipt[k] is not False for k in
                ("commitment_verified", "audit_ledger_written", "persisted", "execution_authority"))):
        return {**denied, "reason": "invalid_training_receipt"}
    commitment = receipt["dataset_commitment"]
    if commitment is not None and (not isinstance(commitment, str) or len(commitment) != 64 or
                                   any(c not in "0123456789abcdef" for c in commitment)):
        return {**denied, "reason": "invalid_training_receipt"}
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
    signature = hmac.new(provenance_key, ("AdvCU.training_receipt.v1:" + canonical).encode("utf-8"),
                         hashlib.sha256).hexdigest()
    return {"ok": True, "receipt": dict(receipt), "algorithm": "HMAC-SHA256",
            "signature": signature, "execution_authority": False, "persisted": False}


def verify_advcu_training_receipt(signed_receipt: Any, *,
                                  trusted_provenance_key: Optional[bytes] = None,
                                  expected_weight_revision: Optional[int] = None,
                                  previously_accepted_signatures: Optional[Any] = None) -> Dict[str, Any]:
    """Verify receipt with optional revision/replay fences; caller maintains replay history.

    A successful replay_token must be durably recorded by the caller to reject
    future repeats. This read-only check cannot guarantee atomic cross-process replay defense.
    """
    denied = {"ok": False, "execution_authority": False}
    if not isinstance(trusted_provenance_key, bytes) or len(trusted_provenance_key) < 32:
        return {**denied, "reason": "invalid_provenance_key"}
    if expected_weight_revision is not None and (
            type(expected_weight_revision) is not int or
            not 0 <= expected_weight_revision <= _ADVCU_WEIGHT_MAX_TOTAL_SAMPLES):
        return {**denied, "reason": "invalid_expected_revision"}
    # Replay history belongs to the caller; this function never stores or mutates it.
    if previously_accepted_signatures is not None:
        if not isinstance(previously_accepted_signatures, (set, frozenset)) or len(previously_accepted_signatures) > 10000 or any(
                not isinstance(item, str) or len(item) != 64 or
                any(ch not in "0123456789abcdef" for ch in item)
                for item in previously_accepted_signatures):
            return {**denied, "reason": "invalid_replay_history"}
    if not isinstance(signed_receipt, dict) or signed_receipt.get("algorithm") != "HMAC-SHA256":
        return {**denied, "reason": "invalid_signed_receipt"}
    signature = signed_receipt.get("signature")
    if not isinstance(signature, str) or len(signature) != 64 or any(c not in "0123456789abcdef" for c in signature):
        return {**denied, "reason": "invalid_signed_receipt"}
    # Revalidate all receipt fields before accepting any signature.
    checked = sign_advcu_training_receipt(signed_receipt.get("receipt"), approved_for_signing=True,
                                          provenance_key=trusted_provenance_key)
    if not checked["ok"] or not hmac.compare_digest(checked["signature"], signature):
        return {**denied, "reason": "receipt_authentication_failed"}
    if (expected_weight_revision is not None and
            signed_receipt["receipt"]["revision_after"] != expected_weight_revision):
        return {**denied, "reason": "weight_revision_mismatch"}
    if (previously_accepted_signatures is not None and
            signature in previously_accepted_signatures):
        return {**denied, "reason": "receipt_replay_detected"}
    return {"ok": True, "authenticated": True, "dataset_verified": False,
            "audit_ledger_written": False, "execution_authority": False,
            "replay_token": signature}


# Caller-owned atomic in-process replay admission; no persistence or external I/O.
def accept_advcu_training_receipt_once(signed_receipt: Any, *,
                                      trusted_provenance_key: Optional[bytes] = None,
                                      accepted_signatures: Any = None,
                                      replay_lock: Any = None,
                                      expected_weight_revision: Optional[int] = None) -> Dict[str, Any]:
    """Verify and atomically record one receipt in a caller-owned set.

    The caller MUST supply the same lock for every accessor of the shared set.
    This only protects cooperating threads in one process; durable cross-process
    replay prevention requires an external transactional authority.
    """
    denied = {"ok": False, "execution_authority": False}
    if type(accepted_signatures) is not set or replay_lock is None or not (
            callable(getattr(replay_lock, "acquire", None)) and
            callable(getattr(replay_lock, "release", None))):
        return {**denied, "reason": "caller_replay_state_required"}
    # Authenticate before entering the critical section to minimize lock hold time.
    verified = verify_advcu_training_receipt(
        signed_receipt, trusted_provenance_key=trusted_provenance_key,
        expected_weight_revision=expected_weight_revision)
    if not verified.get("ok"):
        return verified
    token = verified["replay_token"]
    with replay_lock:
        if len(accepted_signatures) > 10000 or any(
                not isinstance(x, str) or len(x) != 64 or
                any(c not in "0123456789abcdef" for c in x)
                for x in accepted_signatures):
            return {**denied, "reason": "invalid_replay_history"}
        if token in accepted_signatures:
            return {**denied, "reason": "receipt_replay_detected"}
        if len(accepted_signatures) >= 10000:
            return {**denied, "reason": "replay_history_capacity_reached"}
        accepted_signatures.add(token)
    return {**verified, "replay_recorded": True, "persisted": False}



# Caller-managed replay-history checkpoint; no disk I/O, execution, or ledger writes.
def export_advcu_replay_checkpoint(accepted_signatures: Any, *,
                                   approved_for_export: bool = False,
                                   provenance_key: Optional[bytes] = None,
                                   checkpoint_revision: Optional[int] = None) -> Dict[str, Any]:
    """Authenticate bounded caller-managed replay state, optionally with a revision.

    Revision fencing only works when the caller supplies and durably protects a
    trusted minimum revision. This function performs no persistence or I/O.
    """
    denied = {"ok": False, "execution_authority": False}
    if approved_for_export is not True:
        return {**denied, "reason": "explicit_export_approval_required"}
    if not isinstance(provenance_key, bytes) or len(provenance_key) < 32:
        return {**denied, "reason": "invalid_provenance_key"}
    if checkpoint_revision is not None and (type(checkpoint_revision) is not int or
                                            not 0 <= checkpoint_revision <= 2**63 - 1):
        return {**denied, "reason": "invalid_checkpoint_revision"}
    if type(accepted_signatures) not in (set, frozenset) or len(accepted_signatures) > 10000 or any(
            type(item) is not str or len(item) != 64 or
            any(ch not in "0123456789abcdef" for ch in item)
            for item in accepted_signatures):
        return {**denied, "reason": "invalid_replay_history"}
    version = 2 if checkpoint_revision is not None else 1
    payload = {"schema": f"SarahMemory.AdvCU.replay_checkpoint.v{version}",
               "signatures": sorted(accepted_signatures)}
    if version == 2:
        payload["revision"] = checkpoint_revision
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    signature = hmac.new(provenance_key,
                         (f"AdvCU.replay_checkpoint.v{version}:" + canonical).encode("utf-8"),
                         hashlib.sha256).hexdigest()
    return {"ok": True, "checkpoint": payload, "signature": signature,
            "algorithm": "HMAC-SHA256", "persisted": False,
            "execution_authority": False}


def verify_advcu_replay_checkpoint(signed_checkpoint: Any, *,
                                    trusted_provenance_key: Optional[bytes] = None,
                                    minimum_checkpoint_revision: Optional[int] = None,
                                    expected_checkpoint_revision: Optional[int] = None,
                                    previous_signed_checkpoint: Optional[Dict[str, Any]] = None,
                                    maximum_new_signatures: Optional[int] = None,
                                    expected_previous_signature: Optional[str] = None) -> Dict[str, Any]:
    """Verify authenticated history and optionally reject older revisions.

    The minimum revision MUST come from trusted caller-managed monotonic state;
    this API cannot prevent rollback if the caller also rolls back that state.
    """
    denied = {"ok": False, "execution_authority": False}
    if not isinstance(trusted_provenance_key, bytes) or len(trusted_provenance_key) < 32:
        return {**denied, "reason": "invalid_provenance_key"}
    if minimum_checkpoint_revision is not None and (
            type(minimum_checkpoint_revision) is not int or
            not 0 <= minimum_checkpoint_revision <= 2**63 - 1):
        return {**denied, "reason": "invalid_minimum_checkpoint_revision"}
    if expected_checkpoint_revision is not None and (
            type(expected_checkpoint_revision) is not int or
            not 0 <= expected_checkpoint_revision <= 2**63 - 1):
        return {**denied, "reason": "invalid_expected_checkpoint_revision"}
    if (minimum_checkpoint_revision is not None and
            expected_checkpoint_revision is not None and
            expected_checkpoint_revision < minimum_checkpoint_revision):
        return {**denied, "reason": "inconsistent_checkpoint_revision_bounds"}
    if maximum_new_signatures is not None:
        if (type(maximum_new_signatures) is not int or
                not 0 <= maximum_new_signatures <= 10000):
            return {**denied, "reason": "invalid_maximum_new_signatures"}
        if previous_signed_checkpoint is None:
            return {**denied, "reason": "previous_checkpoint_required"}
    # Caller-owned trust anchor: bind continuity to a specific prior checkpoint.
    # A signature alone is not a trusted anchor unless protected by the caller.
    if expected_previous_signature is not None:
        if (type(expected_previous_signature) is not str or
                len(expected_previous_signature) != 64 or
                any(ch not in "0123456789abcdef" for ch in expected_previous_signature)):
            return {**denied, "reason": "invalid_expected_previous_signature"}
        if previous_signed_checkpoint is None:
            return {**denied, "reason": "previous_checkpoint_required"}
    if (not isinstance(signed_checkpoint, dict) or set(signed_checkpoint) != {
            "ok", "checkpoint", "signature", "algorithm", "persisted", "execution_authority"} or
            signed_checkpoint.get("ok") is not True or
            signed_checkpoint.get("algorithm") != "HMAC-SHA256" or
            signed_checkpoint.get("persisted") is not False or
            signed_checkpoint.get("execution_authority") is not False):
        return {**denied, "reason": "invalid_replay_checkpoint"}
    payload = signed_checkpoint.get("checkpoint")
    if not isinstance(payload, dict):
        return {**denied, "reason": "invalid_replay_checkpoint"}
    schema = payload.get("schema")
    if schema == "SarahMemory.AdvCU.replay_checkpoint.v1" and set(payload) == {"schema", "signatures"}:
        version = 1
    elif schema == "SarahMemory.AdvCU.replay_checkpoint.v2" and set(payload) == {"schema", "signatures", "revision"}:
        version = 2
        if type(payload["revision"]) is not int or not 0 <= payload["revision"] <= 2**63 - 1:
            return {**denied, "reason": "invalid_replay_checkpoint"}
    else:
        return {**denied, "reason": "invalid_replay_checkpoint"}
    entries = payload.get("signatures")
    if (not isinstance(entries, list) or len(entries) > 10000 or any(
            type(item) is not str or len(item) != 64 or
            any(ch not in "0123456789abcdef" for ch in item) for item in entries) or
            entries != sorted(set(entries))):
        return {**denied, "reason": "invalid_replay_checkpoint"}
    signature = signed_checkpoint.get("signature")
    if type(signature) is not str or len(signature) != 64 or any(
            ch not in "0123456789abcdef" for ch in signature):
        return {**denied, "reason": "invalid_replay_checkpoint"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    expected = hmac.new(trusted_provenance_key,
                        (f"AdvCU.replay_checkpoint.v{version}:" + canonical).encode("utf-8"),
                        hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return {**denied, "reason": "replay_checkpoint_authentication_failed"}
    if minimum_checkpoint_revision is not None:
        if version != 2:
            return {**denied, "reason": "checkpoint_revision_required"}
        if payload["revision"] < minimum_checkpoint_revision:
            return {**denied, "reason": "stale_checkpoint_revision"}
    if expected_checkpoint_revision is not None:
        if version != 2:
            return {**denied, "reason": "checkpoint_revision_required"}
        if payload["revision"] != expected_checkpoint_revision:
            return {**denied, "reason": "unexpected_checkpoint_revision"}
    if previous_signed_checkpoint is not None:
        # Verify both authenticated snapshots before comparing sequence/history.
        # The caller must independently protect the trusted prior checkpoint.
        previous = verify_advcu_replay_checkpoint(
            previous_signed_checkpoint, trusted_provenance_key=trusted_provenance_key)
        if not previous.get("ok"):
            return {**denied, "reason": "invalid_previous_checkpoint"}
        if (expected_previous_signature is not None and
                not hmac.compare_digest(previous_signed_checkpoint["signature"],
                                        expected_previous_signature)):
            return {**denied, "reason": "unexpected_previous_checkpoint"}
        if version != 2 or "checkpoint_revision" not in previous:
            return {**denied, "reason": "checkpoint_revision_required"}
        previous_revision = previous["checkpoint_revision"]
        if previous_revision == 2**63 - 1 or payload["revision"] != previous_revision + 1:
            return {**denied, "reason": "checkpoint_sequence_discontinuity"}
        current_entries = set(entries)
        if not previous["accepted_signatures"].issubset(current_entries):
            return {**denied, "reason": "checkpoint_history_discontinuity"}
        if (maximum_new_signatures is not None and
                len(current_entries) - previous["count"] > maximum_new_signatures):
            return {**denied, "reason": "checkpoint_growth_limit_exceeded"}
    result = {"ok": True, "accepted_signatures": set(entries), "count": len(entries),
              "persisted": False, "execution_authority": False}
    if version == 2:
        result["checkpoint_revision"] = payload["revision"]
    return result

# Explicit, caller-authorized checkpoint recovery validation; no mutation or persistence.
def validate_advcu_replay_checkpoint_recovery(signed_checkpoint: Any, *,
                                              approved_for_recovery: bool = False,
                                              trusted_provenance_key: Optional[bytes] = None,
                                              trusted_checkpoint_signature: Optional[str] = None,
                                              minimum_checkpoint_revision: Optional[int] = None,
                                              maximum_recovered_signatures: int = 10000) -> Dict[str, Any]:
    """Validate a recovery candidate against a caller-protected exact trust anchor.

    This read-only operation never replaces caller state. The caller must
    independently authorize and atomically commit any recovery, and must keep
    the anchor and revision floor in trusted durable storage.
    """
    denied = {"ok": False, "execution_authority": False}
    if approved_for_recovery is not True:
        return {**denied, "reason": "explicit_recovery_approval_required"}
    if (type(trusted_checkpoint_signature) is not str or
            len(trusted_checkpoint_signature) != 64 or
            any(ch not in "0123456789abcdef" for ch in trusted_checkpoint_signature)):
        return {**denied, "reason": "trusted_checkpoint_signature_required"}
    if (type(minimum_checkpoint_revision) is not int or
            not 0 <= minimum_checkpoint_revision <= 2**63 - 1):
        return {**denied, "reason": "trusted_revision_floor_required"}
    if (type(maximum_recovered_signatures) is not int or
            not 0 <= maximum_recovered_signatures <= 10000):
        return {**denied, "reason": "invalid_recovery_capacity"}
    verified = verify_advcu_replay_checkpoint(
        signed_checkpoint, trusted_provenance_key=trusted_provenance_key,
        minimum_checkpoint_revision=minimum_checkpoint_revision)
    if not verified.get("ok"):
        return verified
    if not hmac.compare_digest(signed_checkpoint["signature"], trusted_checkpoint_signature):
        return {**denied, "reason": "untrusted_recovery_checkpoint"}
    if verified["count"] > maximum_recovered_signatures:
        return {**denied, "reason": "recovery_capacity_exceeded"}
    return {**verified, "recovery_validated": True, "recovery_committed": False,
            "audit_ledger_written": False, "execution_authority": False}


# Read-only recovery planning; the caller owns any later atomic commit and audit.
def plan_advcu_replay_checkpoint_recovery(signed_checkpoint: Any, *,
                                          approved_for_recovery: bool = False,
                                          trusted_provenance_key: Optional[bytes] = None,
                                          trusted_checkpoint_signature: Optional[str] = None,
                                          minimum_checkpoint_revision: Optional[int] = None,
                                          current_accepted_signatures: Any = None,
                                          maximum_recovered_signatures: int = 10000,
                                          maximum_state_changes: int = 10000) -> Dict[str, Any]:
    """Produce a bounded, non-executable before/after recovery plan.

    Current state is caller-supplied and untrusted; the plan is advisory only.
    The caller must recheck state under its own lock before any authorized commit.
    """
    denied = {"ok": False, "execution_authority": False}
    if type(current_accepted_signatures) not in (set, frozenset) or (
            len(current_accepted_signatures) > 10000 or any(
                type(item) is not str or len(item) != 64 or
                any(ch not in "0123456789abcdef" for ch in item)
                for item in current_accepted_signatures)):
        return {**denied, "reason": "invalid_current_replay_history"}
    if (type(maximum_state_changes) is not int or
            not 0 <= maximum_state_changes <= 10000):
        return {**denied, "reason": "invalid_recovery_change_limit"}
    validated = validate_advcu_replay_checkpoint_recovery(
        signed_checkpoint, approved_for_recovery=approved_for_recovery,
        trusted_provenance_key=trusted_provenance_key,
        trusted_checkpoint_signature=trusted_checkpoint_signature,
        minimum_checkpoint_revision=minimum_checkpoint_revision,
        maximum_recovered_signatures=maximum_recovered_signatures)
    if not validated.get("ok"):
        return validated
    before = current_accepted_signatures
    after = validated["accepted_signatures"]
    added = len(after - before)
    removed = len(before - after)
    if added + removed > maximum_state_changes:
        return {**denied, "reason": "recovery_change_limit_exceeded"}
    def commitment(entries: Any) -> str:
        canonical = json.dumps(sorted(entries), separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("ascii")).hexdigest()
    return {"ok": True, "recovery_plan_only": True,
            "recovery_committed": False, "audit_ledger_written": False,
            "execution_authority": False,
            "checkpoint_revision": validated["checkpoint_revision"],
            "checkpoint_signature": signed_checkpoint["signature"],
            "before_count": len(before), "after_count": len(after),
            "added_count": added, "removed_count": removed,
            "before_sha256": commitment(before),
            "after_sha256": commitment(after),
            "requires_caller_state_recheck": True,
            "requires_caller_atomic_commit": True,
            "requires_caller_audit": True}


# Advisory freshness check; callers must hold their own lock during final commit.
def verify_advcu_recovery_plan_freshness(recovery_plan: Any, *,
                                         approved_for_recovery: bool = False,
                                         current_accepted_signatures: Any = None,
                                         expected_checkpoint_signature: Optional[str] = None,
                                         expected_checkpoint_revision: Optional[int] = None) -> Dict[str, Any]:
    """Check that a previously produced plan still matches caller-supplied state.

    This does not authenticate a plan or authorize a commit. The caller must
    protect the plan and trust anchor, verify the checkpoint independently,
    and recheck state under a lock immediately before its own atomic commit.
    """
    denied = {"ok": False, "execution_authority": False}
    if approved_for_recovery is not True:
        return {**denied, "reason": "explicit_recovery_approval_required"}
    if (type(current_accepted_signatures) not in (set, frozenset) or
            len(current_accepted_signatures) > 10000 or any(
                type(item) is not str or len(item) != 64 or
                any(ch not in "0123456789abcdef" for ch in item)
                for item in current_accepted_signatures)):
        return {**denied, "reason": "invalid_current_replay_history"}
    if (type(expected_checkpoint_signature) is not str or
            len(expected_checkpoint_signature) != 64 or
            any(ch not in "0123456789abcdef" for ch in expected_checkpoint_signature)):
        return {**denied, "reason": "trusted_checkpoint_signature_required"}
    if (type(expected_checkpoint_revision) is not int or
            not 0 <= expected_checkpoint_revision <= 2**63 - 1):
        return {**denied, "reason": "trusted_checkpoint_revision_required"}
    required = {"ok", "recovery_plan_only", "recovery_committed", "audit_ledger_written",
                "execution_authority", "checkpoint_revision", "checkpoint_signature",
                "before_count", "after_count", "added_count", "removed_count",
                "before_sha256", "after_sha256", "requires_caller_state_recheck",
                "requires_caller_atomic_commit", "requires_caller_audit"}
    if (type(recovery_plan) is not dict or set(recovery_plan) != required or
            recovery_plan.get("ok") is not True or
            recovery_plan.get("recovery_plan_only") is not True or
            recovery_plan.get("recovery_committed") is not False or
            recovery_plan.get("audit_ledger_written") is not False or
            recovery_plan.get("execution_authority") is not False or
            any(recovery_plan.get(k) is not True for k in (
                "requires_caller_state_recheck", "requires_caller_atomic_commit",
                "requires_caller_audit")) or
            any(type(recovery_plan.get(k)) is not int or not 0 <= recovery_plan[k] <= 10000
                for k in ("before_count", "after_count", "added_count", "removed_count")) or
            any(type(recovery_plan.get(k)) is not str or len(recovery_plan[k]) != 64 or
                any(ch not in "0123456789abcdef" for ch in recovery_plan[k])
                for k in ("before_sha256", "after_sha256", "checkpoint_signature")) or
            type(recovery_plan.get("checkpoint_revision")) is not int or
            not 0 <= recovery_plan["checkpoint_revision"] <= 2**63 - 1 or
            recovery_plan["before_count"] + recovery_plan["added_count"] -
                recovery_plan["removed_count"] != recovery_plan["after_count"]):
        return {**denied, "reason": "invalid_recovery_plan"}
    if (not hmac.compare_digest(recovery_plan["checkpoint_signature"],
                                expected_checkpoint_signature) or
            recovery_plan["checkpoint_revision"] != expected_checkpoint_revision):
        return {**denied, "reason": "untrusted_recovery_plan_anchor"}
    canonical = json.dumps(sorted(current_accepted_signatures),
                           separators=(",", ":"), ensure_ascii=True)
    current_digest = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    if (len(current_accepted_signatures) != recovery_plan["before_count"] or
            not hmac.compare_digest(current_digest, recovery_plan["before_sha256"])):
        return {**denied, "reason": "recovery_plan_stale"}
    return {"ok": True, "recovery_plan_fresh": True,
            "recovery_committed": False, "audit_ledger_written": False,
            "execution_authority": False, "requires_caller_atomic_commit": True,
            "requires_caller_audit": True, "requires_caller_state_recheck": True}


# Authenticated advisory recovery plans. Caller retains keys, trust anchors and authority.
def sign_advcu_recovery_plan(recovery_plan: Any, *,
                             approved_for_signing: bool = False,
                             provenance_key: Optional[bytes] = None) -> Dict[str, Any]:
    """Sign a bounded recovery plan; never authorize or perform recovery."""
    denied = {"ok": False, "execution_authority": False}
    if approved_for_signing is not True:
        return {**denied, "reason": "explicit_plan_signing_approval_required"}
    if type(provenance_key) is not bytes or len(provenance_key) < 32:
        return {**denied, "reason": "invalid_provenance_key"}
    if type(recovery_plan) is not dict:
        return {**denied, "reason": "invalid_recovery_plan"}
    # The existing strict schema validator checks the plan and its trust anchor.
    anchor = recovery_plan.get("checkpoint_signature")
    revision = recovery_plan.get("checkpoint_revision")
    if (type(anchor) is not str or type(revision) is not int):
        return {**denied, "reason": "invalid_recovery_plan"}
    # Validate structure with a synthetic bounded before-state only when empty.
    # Otherwise validate all plan fields explicitly without inventing state.
    required = {"ok", "recovery_plan_only", "recovery_committed",
                "audit_ledger_written", "execution_authority", "checkpoint_revision",
                "checkpoint_signature", "before_count", "after_count", "added_count",
                "removed_count", "before_sha256", "after_sha256",
                "requires_caller_state_recheck", "requires_caller_atomic_commit",
                "requires_caller_audit"}
    if (set(recovery_plan) != required or recovery_plan.get("ok") is not True or
            recovery_plan.get("recovery_plan_only") is not True or
            any(recovery_plan.get(k) is not False for k in (
                "recovery_committed", "audit_ledger_written", "execution_authority")) or
            any(recovery_plan.get(k) is not True for k in (
                "requires_caller_state_recheck", "requires_caller_atomic_commit",
                "requires_caller_audit")) or
            not 0 <= revision <= 2**63 - 1 or
            any(type(recovery_plan.get(k)) is not int or
                not 0 <= recovery_plan[k] <= 10000 for k in (
                    "before_count", "after_count", "added_count", "removed_count")) or
            recovery_plan["before_count"] + recovery_plan["added_count"] -
                recovery_plan["removed_count"] != recovery_plan["after_count"] or
            any(type(recovery_plan.get(k)) is not str or
                len(recovery_plan[k]) != 64 or
                any(c not in "0123456789abcdef" for c in recovery_plan[k])
                for k in ("checkpoint_signature", "before_sha256", "after_sha256"))):
        return {**denied, "reason": "invalid_recovery_plan"}
    canonical = json.dumps(recovery_plan, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False)
    mac = hmac.new(provenance_key,
                   ("AdvCU.recovery_plan.v1:" + canonical).encode("ascii"),
                   hashlib.sha256).hexdigest()
    return {"ok": True, "schema": "SarahMemory.AdvCU.recovery_plan_signature.v1",
            "plan": dict(recovery_plan), "signature": mac,
            "algorithm": "HMAC-SHA256", "execution_authority": False,
            "recovery_committed": False}


def verify_advcu_signed_recovery_plan(signed_plan: Any, *,
                                       trusted_provenance_key: Optional[bytes] = None,
                                       approved_for_verification: bool = False) -> Dict[str, Any]:
    """Authenticate an advisory plan; freshness and commit authorization are separate."""
    denied = {"ok": False, "execution_authority": False}
    if approved_for_verification is not True:
        return {**denied, "reason": "explicit_plan_verification_approval_required"}
    if type(trusted_provenance_key) is not bytes or len(trusted_provenance_key) < 32:
        return {**denied, "reason": "invalid_provenance_key"}
    if (type(signed_plan) is not dict or set(signed_plan) != {
            "ok", "schema", "plan", "signature", "algorithm",
            "execution_authority", "recovery_committed"} or
            signed_plan.get("ok") is not True or
            signed_plan.get("schema") != "SarahMemory.AdvCU.recovery_plan_signature.v1" or
            signed_plan.get("algorithm") != "HMAC-SHA256" or
            signed_plan.get("execution_authority") is not False or
            signed_plan.get("recovery_committed") is not False or
            type(signed_plan.get("signature")) is not str or
            len(signed_plan["signature"]) != 64 or
            any(c not in "0123456789abcdef" for c in signed_plan["signature"])):
        return {**denied, "reason": "invalid_signed_recovery_plan"}
    # Reuse the same strict schema and canonicalization; no state is modified.
    candidate = sign_advcu_recovery_plan(
        signed_plan["plan"], approved_for_signing=True,
        provenance_key=trusted_provenance_key)
    if not candidate.get("ok"):
        return {**denied, "reason": "invalid_recovery_plan"}
    if not hmac.compare_digest(candidate["signature"], signed_plan["signature"]):
        return {**denied, "reason": "recovery_plan_authentication_failed"}
    return {"ok": True, "plan_authenticated": True, "plan_fresh": False,
            "recovery_committed": False, "execution_authority": False,
            "requires_caller_state_recheck": True, "requires_caller_audit": True}


# Optional authenticated expiry/replay gate; advisory only, with caller-owned clock/state.
def sign_advcu_timed_recovery_plan(signed_plan: Any, *,
                                   approved_for_signing: bool = False,
                                   provenance_key: Optional[bytes] = None,
                                   issued_at: Any = None, expires_at: Any = None,
                                   nonce: Any = None) -> Dict[str, Any]:
    """Bind a previously authenticated advisory plan to a bounded time and nonce."""
    denied = {"ok": False, "execution_authority": False}
    if approved_for_signing is not True:
        return {**denied, "reason": "explicit_plan_signing_approval_required"}
    checked = verify_advcu_signed_recovery_plan(
        signed_plan, trusted_provenance_key=provenance_key,
        approved_for_verification=True)
    if not checked.get("ok"):
        return {**denied, "reason": "invalid_signed_recovery_plan"}
    if (type(issued_at) is not int or type(expires_at) is not int or
            not 0 <= issued_at < expires_at <= 2**63 - 1 or
            expires_at - issued_at > 86400 or
            type(nonce) is not str or len(nonce) != 64 or
            any(c not in "0123456789abcdef" for c in nonce)):
        return {**denied, "reason": "invalid_time_window_or_nonce"}
    payload = {"schema": "SarahMemory.AdvCU.timed_recovery_plan.v1",
               "signed_plan": signed_plan, "issued_at": issued_at,
               "expires_at": expires_at, "nonce": nonce}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False)
    signature = hmac.new(provenance_key,
                         ("AdvCU.timed_recovery_plan.v1:" + canonical).encode("ascii"),
                         hashlib.sha256).hexdigest()
    return {"ok": True, "payload": payload, "signature": signature,
            "execution_authority": False, "recovery_committed": False}


def verify_advcu_timed_recovery_plan(envelope: Any, *,
                                     approved_for_verification: bool = False,
                                     trusted_provenance_key: Optional[bytes] = None,
                                     trusted_now: Any = None,
                                     accepted_nonces: Any = None,
                                     maximum_clock_skew: int = 0) -> Dict[str, Any]:
    """Read-only expiry/replay screening; caller must atomically reserve nonce."""
    denied = {"ok": False, "execution_authority": False}
    if approved_for_verification is not True:
        return {**denied, "reason": "explicit_plan_verification_approval_required"}
    if (type(trusted_now) is not int or not 0 <= trusted_now <= 2**63 - 1 or
            type(maximum_clock_skew) is not int or not 0 <= maximum_clock_skew <= 300 or
            type(accepted_nonces) not in (set, frozenset) or
            len(accepted_nonces) > 10000):
        return {**denied, "reason": "invalid_caller_trust_state"}
    if (type(envelope) is not dict or set(envelope) != {
            "ok", "payload", "signature", "execution_authority", "recovery_committed"} or
            envelope.get("ok") is not True or
            envelope.get("execution_authority") is not False or
            envelope.get("recovery_committed") is not False or
            type(envelope.get("signature")) is not str or
            len(envelope["signature"]) != 64 or
            any(c not in "0123456789abcdef" for c in envelope["signature"])):
        return {**denied, "reason": "invalid_timed_envelope"}
    payload = envelope["payload"]
    if (type(payload) is not dict or set(payload) != {
            "schema", "signed_plan", "issued_at", "expires_at", "nonce"} or
            payload.get("schema") != "SarahMemory.AdvCU.timed_recovery_plan.v1"):
        return {**denied, "reason": "invalid_timed_envelope"}
    # Recompute the authenticated envelope using the same strict signer.
    expected = sign_advcu_timed_recovery_plan(
        payload["signed_plan"], approved_for_signing=True,
        provenance_key=trusted_provenance_key, issued_at=payload["issued_at"],
        expires_at=payload["expires_at"], nonce=payload["nonce"])
    if not expected.get("ok"):
        return {**denied, "reason": "invalid_timed_envelope"}
    if not hmac.compare_digest(expected["signature"], envelope["signature"]):
        return {**denied, "reason": "timed_envelope_authentication_failed"}
    if trusted_now + maximum_clock_skew < payload["issued_at"]:
        return {**denied, "reason": "recovery_plan_not_yet_valid"}
    if trusted_now >= payload["expires_at"] + maximum_clock_skew:
        return {**denied, "reason": "recovery_plan_expired"}
    if payload["nonce"] in accepted_nonces:
        return {**denied, "reason": "recovery_plan_replay_detected"}
    return {"ok": True, "time_window_valid": True, "nonce_not_seen": True,
            "plan_authenticated": True, "plan_fresh": False,
            "execution_authority": False, "recovery_committed": False,
            "requires_caller_atomic_nonce_reservation": True,
            "requires_caller_state_recheck": True, "requires_caller_audit": True}


def preflight_advcu_timed_nonce_reservation(envelope: Any, *,
        approved_for_verification: bool = False,
        trusted_provenance_key: Optional[bytes] = None,
        trusted_now: Any = None, accepted_nonces: Any = None,
        maximum_clock_skew: int = 0, maximum_accepted_nonces: int = 10000) -> Dict[str, Any]:
    """Read-only capacity/freshness preflight; never reserves or commits a nonce.

    Caller must revalidate and atomically reserve under its own lock/transaction.
    This result is not an authorization, nor a race-free reservation guarantee.
    """
    denied = {"ok": False, "execution_authority": False, "recovery_committed": False}
    if approved_for_verification is not True:
        return {**denied, "reason": "explicit_plan_verification_approval_required"}
    if (type(maximum_accepted_nonces) is not int or
            not 1 <= maximum_accepted_nonces <= 10000 or
            type(accepted_nonces) not in (set, frozenset) or
            len(accepted_nonces) > maximum_accepted_nonces):
        return {**denied, "reason": "invalid_nonce_reservation_capacity"}
    if any(type(item) is not str or len(item) != 64 or
           any(ch not in "0123456789abcdef" for ch in item)
           for item in accepted_nonces):
        return {**denied, "reason": "invalid_accepted_nonce_entry"}
    verified = verify_advcu_timed_recovery_plan(
        envelope, approved_for_verification=True,
        trusted_provenance_key=trusted_provenance_key, trusted_now=trusted_now,
        accepted_nonces=accepted_nonces, maximum_clock_skew=maximum_clock_skew)
    if not verified.get("ok"):
        return {**denied, "reason": verified.get("reason", "verification_failed")}
    if len(accepted_nonces) >= maximum_accepted_nonces:
        return {**denied, "reason": "nonce_reservation_capacity_exhausted"}
    return {"ok": True, "nonce": envelope["payload"]["nonce"],
            "remaining_capacity_after_reservation": maximum_accepted_nonces - len(accepted_nonces) - 1,
            "plan_authenticated": True, "plan_fresh": False,
            "nonce_reserved": False, "execution_authority": False,
            "recovery_committed": False,
            "requires_caller_atomic_nonce_reservation": True,
            "requires_caller_state_recheck": True, "requires_caller_audit": True}


# Caller-attested reservation evidence; does not reserve or authenticate the caller's transaction.
def attest_advcu_nonce_reservation(envelope: Any, *,
        approved_for_attestation: bool = False,
        trusted_provenance_key: Optional[bytes] = None,
        trusted_now: Any = None, accepted_nonces: Any = None,
        reservation_id: Any = None, maximum_accepted_nonces: int = 10000) -> Dict[str, Any]:
    """Sign bounded evidence of a caller-reported completed nonce reservation.

    Caller MUST have atomically reserved the nonce before invoking this method.
    A signed receipt proves only that the signer attested to the supplied state;
    it does not prove transaction atomicity, durable storage, or audit completion.
    """
    denied = {"ok": False, "execution_authority": False,
              "recovery_committed": False, "audit_ledger_written": False}
    if approved_for_attestation is not True:
        return {**denied, "reason": "explicit_reservation_attestation_approval_required"}
    if (type(maximum_accepted_nonces) is not int or
            not 1 <= maximum_accepted_nonces <= 10000 or
            type(accepted_nonces) not in (set, frozenset) or
            not 1 <= len(accepted_nonces) <= maximum_accepted_nonces or
            any(type(n) is not str or len(n) != 64 or
                any(c not in "0123456789abcdef" for c in n)
                for n in accepted_nonces) or
            type(reservation_id) is not str or len(reservation_id) != 64 or
            any(c not in "0123456789abcdef" for c in reservation_id) or
            type(trusted_now) is not int or not 0 <= trusted_now <= 2**63 - 1 or
            type(trusted_provenance_key) is not bytes or
            not 16 <= len(trusted_provenance_key) <= 4096):
        return {**denied, "reason": "invalid_caller_reservation_evidence"}
    # Check authentication and time with an empty replay set: the nonce MUST
    # already be present in caller-supplied post-reservation state.
    verified = verify_advcu_timed_recovery_plan(
        envelope, approved_for_verification=True,
        trusted_provenance_key=trusted_provenance_key,
        trusted_now=trusted_now, accepted_nonces=frozenset())
    if not verified.get("ok"):
        return {**denied, "reason": verified.get("reason", "invalid_timed_plan")}
    nonce = envelope["payload"]["nonce"]
    if nonce not in accepted_nonces:
        return {**denied, "reason": "nonce_not_in_caller_reserved_state"}
    receipt = {"schema": "SarahMemory.AdvCU.nonce_reservation_attestation.v1",
               "nonce": nonce, "reservation_id": reservation_id,
               "observed_at": trusted_now,
               "timed_plan_signature": envelope["signature"],
               "accepted_nonce_count": len(accepted_nonces),
               "caller_attestation_only": True}
    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False)
    signature = hmac.new(trusted_provenance_key,
        ("AdvCU.nonce_reservation_attestation.v1:" + canonical).encode("ascii"),
        hashlib.sha256).hexdigest()
    return {"ok": True, "receipt": receipt, "signature": signature,
            "nonce_reserved_by_advcu": False, "execution_authority": False,
            "recovery_committed": False, "audit_ledger_written": False,
            "requires_caller_audit": True}

def verify_advcu_nonce_reservation_attestation(attestation: Any, *,
        approved_for_verification: bool = False,
        trusted_provenance_key: Optional[bytes] = None) -> Dict[str, Any]:
    """Authenticate caller attestation without asserting transaction correctness."""
    denied = {"ok": False, "execution_authority": False}
    if approved_for_verification is not True:
        return {**denied, "reason": "explicit_attestation_verification_approval_required"}
    if (type(trusted_provenance_key) is not bytes or
            not 16 <= len(trusted_provenance_key) <= 4096 or
            type(attestation) is not dict or set(attestation) != {
                "ok", "receipt", "signature", "nonce_reserved_by_advcu",
                "execution_authority", "recovery_committed",
                "audit_ledger_written", "requires_caller_audit"} or
            attestation.get("ok") is not True or
            any(attestation.get(k) is not False for k in (
                "nonce_reserved_by_advcu", "execution_authority",
                "recovery_committed", "audit_ledger_written")) or
            attestation.get("requires_caller_audit") is not True or
            type(attestation.get("signature")) is not str or
            len(attestation["signature"]) != 64 or
            any(c not in "0123456789abcdef" for c in attestation["signature"])):
        return {**denied, "reason": "invalid_reservation_attestation"}
    r = attestation["receipt"]
    if (type(r) is not dict or set(r) != {
            "schema", "nonce", "reservation_id", "observed_at",
            "timed_plan_signature", "accepted_nonce_count",
            "caller_attestation_only"} or
            r.get("schema") != "SarahMemory.AdvCU.nonce_reservation_attestation.v1" or
            r.get("caller_attestation_only") is not True or
            type(r.get("observed_at")) is not int or
            not 0 <= r["observed_at"] <= 2**63 - 1 or
            type(r.get("accepted_nonce_count")) is not int or
            not 1 <= r["accepted_nonce_count"] <= 10000 or
            any(type(r.get(k)) is not str or len(r[k]) != 64 or
                any(c not in "0123456789abcdef" for c in r[k])
                for k in ("nonce", "reservation_id", "timed_plan_signature"))):
        return {**denied, "reason": "invalid_reservation_attestation"}
    canonical = json.dumps(r, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False)
    expected = hmac.new(trusted_provenance_key,
        ("AdvCU.nonce_reservation_attestation.v1:" + canonical).encode("ascii"),
        hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, attestation["signature"]):
        return {**denied, "reason": "reservation_attestation_authentication_failed"}
    return {"ok": True, "attestation_authenticated": True,
            "atomic_reservation_verified": False, "execution_authority": False,
            "requires_caller_audit": True}


def reconcile_advcu_nonce_reservation_attestation(attestation: Any, *,
        approved_for_reconciliation: bool = False,
        trusted_provenance_key: Optional[bytes] = None,
        trusted_reservations: Any = None,
        maximum_reservations: int = 10000) -> Dict[str, Any]:
    """Read-only reconciliation against a caller-protected nonce-to-ID snapshot.

    The snapshot must originate from the caller's authoritative transaction store.
    Matching evidence does not establish atomicity, durability, or audit completion.
    Caller must synchronize/recheck before acting; no state is changed here.
    """
    denied = {"ok": False, "execution_authority": False,
              "recovery_committed": False, "audit_ledger_written": False}
    if approved_for_reconciliation is not True:
        return {**denied, "reason": "explicit_reconciliation_approval_required"}
    if (type(maximum_reservations) is not int or
            not 1 <= maximum_reservations <= 10000 or
            type(trusted_reservations) is not dict or
            not 1 <= len(trusted_reservations) <= maximum_reservations):
        return {**denied, "reason": "invalid_trusted_reservation_snapshot"}
    for nonce, reservation_id in trusted_reservations.items():
        if any(type(value) is not str or len(value) != 64 or
               any(c not in "0123456789abcdef" for c in value)
               for value in (nonce, reservation_id)):
            return {**denied, "reason": "invalid_trusted_reservation_snapshot"}
    verified = verify_advcu_nonce_reservation_attestation(
        attestation, approved_for_verification=True,
        trusted_provenance_key=trusted_provenance_key)
    if not verified.get("ok"):
        return {**denied, "reason": verified.get("reason", "invalid_attestation")}
    receipt = attestation["receipt"]
    if receipt["nonce"] not in trusted_reservations:
        return {**denied, "reason": "nonce_absent_from_trusted_reservations"}
    if not hmac.compare_digest(trusted_reservations[receipt["nonce"]],
                               receipt["reservation_id"]):
        return {**denied, "reason": "reservation_id_mismatch"}
    if len(trusted_reservations) < receipt["accepted_nonce_count"]:
        return {**denied, "reason": "trusted_snapshot_count_below_attested_count"}
    return {"ok": True, "attestation_authenticated": True,
            "reservation_snapshot_matched": True,
            "atomic_reservation_verified": False, "durability_verified": False,
            "execution_authority": False, "recovery_committed": False,
            "audit_ledger_written": False, "requires_caller_audit": True,
            "requires_caller_state_recheck": True}


def verify_advcu_reservation_history_continuity(previous_snapshot: Any,
        current_snapshot: Any, *, approved_for_verification: bool = False,
        maximum_reservations: int = 10000) -> Dict[str, Any]:
    """Read-only check for monotonic nonce-to-reservation-ID history.

    Both snapshots MUST be independently protected by the caller. This
    function cannot establish their provenance, durability, atomicity or
    freshness. No reservation, execution, or audit write is performed.
    """
    denied = {"ok": False, "execution_authority": False,
              "recovery_committed": False, "audit_ledger_written": False}
    if approved_for_verification is not True:
        return {**denied, "reason": "explicit_continuity_verification_approval_required"}
    if (type(maximum_reservations) is not int or
            not 1 <= maximum_reservations <= 10000 or
            type(previous_snapshot) is not dict or
            type(current_snapshot) is not dict or
            len(previous_snapshot) > maximum_reservations or
            len(current_snapshot) > maximum_reservations):
        return {**denied, "reason": "invalid_reservation_history_snapshot"}
    for snapshot in (previous_snapshot, current_snapshot):
        for nonce, reservation_id in snapshot.items():
            if any(type(value) is not str or len(value) != 64 or
                   any(char not in "0123456789abcdef" for char in value)
                   for value in (nonce, reservation_id)):
                return {**denied, "reason": "invalid_reservation_history_snapshot"}
    for nonce, reservation_id in previous_snapshot.items():
        if nonce not in current_snapshot:
            return {**denied, "reason": "previous_reservation_missing"}
        if not hmac.compare_digest(reservation_id, current_snapshot[nonce]):
            return {**denied, "reason": "conflicting_reservation_identifier"}
    return {"ok": True, "history_continuous": True,
            "previous_count": len(previous_snapshot),
            "current_count": len(current_snapshot),
            "new_reservations": len(current_snapshot) - len(previous_snapshot),
            "atomic_reservation_verified": False,
            "durability_verified": False, "execution_authority": False,
            "recovery_committed": False, "audit_ledger_written": False,
            "requires_caller_audit": True,
            "requires_caller_state_recheck": True}


def sign_advcu_reservation_snapshot(snapshot: Any, *,
        approved_for_signing: bool = False,
        provenance_key: Optional[bytes] = None,
        maximum_reservations: int = 10000) -> Dict[str, Any]:
    """Authenticate caller-supplied reservation history without changing it.

    Caller owns key custody, snapshot provenance, persistence, and audit.
    This commitment is not proof of reservation atomicity or freshness.
    """
    denied = {"ok": False, "execution_authority": False,
              "audit_ledger_written": False}
    if approved_for_signing is not True:
        return {**denied, "reason": "explicit_snapshot_signing_approval_required"}
    if (type(provenance_key) is not bytes or not 32 <= len(provenance_key) <= 4096 or
            type(maximum_reservations) is not int or
            not 1 <= maximum_reservations <= 10000 or
            type(snapshot) is not dict or len(snapshot) > maximum_reservations):
        return {**denied, "reason": "invalid_snapshot_or_key"}
    for nonce, reservation_id in snapshot.items():
        if any(type(value) is not str or len(value) != 64 or
               any(char not in "0123456789abcdef" for char in value)
               for value in (nonce, reservation_id)):
            return {**denied, "reason": "invalid_reservation_history_snapshot"}
    canonical = json.dumps(snapshot, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False)
    signature = hmac.new(provenance_key,
        ("AdvCU.reservation_snapshot.v1:" + canonical).encode("ascii"),
        hashlib.sha256).hexdigest()
    return {"ok": True, "schema": "SarahMemory.AdvCU.reservation_snapshot.v1",
            "snapshot": dict(snapshot), "signature": signature,
            "algorithm": "HMAC-SHA256", "execution_authority": False,
            "audit_ledger_written": False, "requires_caller_audit": True}


def verify_advcu_reservation_snapshot(commitment: Any, *,
        approved_for_verification: bool = False,
        trusted_provenance_key: Optional[bytes] = None,
        maximum_reservations: int = 10000) -> Dict[str, Any]:
    """Read-only authentication of a bounded reservation snapshot commitment."""
    denied = {"ok": False, "execution_authority": False,
              "audit_ledger_written": False}
    if approved_for_verification is not True:
        return {**denied, "reason": "explicit_snapshot_verification_approval_required"}
    if (type(commitment) is not dict or
            set(commitment) != {"ok", "schema", "snapshot", "signature",
                                "algorithm", "execution_authority",
                                "audit_ledger_written", "requires_caller_audit"} or
            commitment.get("ok") is not True or
            commitment.get("schema") != "SarahMemory.AdvCU.reservation_snapshot.v1" or
            commitment.get("algorithm") != "HMAC-SHA256" or
            commitment.get("execution_authority") is not False or
            commitment.get("audit_ledger_written") is not False or
            commitment.get("requires_caller_audit") is not True or
            type(commitment.get("signature")) is not str or
            len(commitment["signature"]) != 64 or
            any(c not in "0123456789abcdef" for c in commitment["signature"])):
        return {**denied, "reason": "invalid_snapshot_commitment"}
    signed = sign_advcu_reservation_snapshot(
        commitment["snapshot"], approved_for_signing=True,
        provenance_key=trusted_provenance_key,
        maximum_reservations=maximum_reservations)
    if not signed.get("ok"):
        return {**denied, "reason": "invalid_snapshot_or_key"}
    if not hmac.compare_digest(signed["signature"], commitment["signature"]):
        return {**denied, "reason": "snapshot_authentication_failed"}
    return {"ok": True, "snapshot_authenticated": True,
            "reservation_count": len(commitment["snapshot"]),
            "atomic_reservation_verified": False, "durability_verified": False,
            "execution_authority": False, "audit_ledger_written": False,
            "requires_caller_audit": True}


def verify_advcu_authenticated_reservation_continuity(previous_commitment: Any,
        current_commitment: Any, *, approved_for_verification: bool = False,
        trusted_provenance_key: Optional[bytes] = None,
        maximum_reservations: int = 10000,
        trusted_previous_signature: Optional[str] = None) -> Dict[str, Any]:
    """Read-only continuity check between authenticated reservation snapshots.

    Caller must anchor the previous signature to trusted durable history;
    authentication alone cannot establish freshness or prevent a pair rollback.
    No nonce reservation, persistence, approval grant, or execution occurs.
    """
    denied = {"ok": False, "execution_authority": False,
              "recovery_committed": False, "audit_ledger_written": False}
    if approved_for_verification is not True:
        return {**denied, "reason": "explicit_continuity_verification_approval_required"}
    if (type(trusted_previous_signature) is not str or
            len(trusted_previous_signature) != 64 or
            any(c not in "0123456789abcdef" for c in trusted_previous_signature)):
        return {**denied, "reason": "trusted_previous_signature_required"}
    for label, commitment in (("previous", previous_commitment),
                              ("current", current_commitment)):
        verified = verify_advcu_reservation_snapshot(
            commitment, approved_for_verification=True,
            trusted_provenance_key=trusted_provenance_key,
            maximum_reservations=maximum_reservations)
        if not verified.get("ok"):
            return {**denied, "reason": label + "_snapshot_authentication_failed"}
    if not hmac.compare_digest(previous_commitment["signature"],
                               trusted_previous_signature):
        return {**denied, "reason": "previous_snapshot_anchor_mismatch"}
    continuity = verify_advcu_reservation_history_continuity(
        previous_commitment["snapshot"], current_commitment["snapshot"],
        approved_for_verification=True, maximum_reservations=maximum_reservations)
    if not continuity.get("ok"):
        return {**denied, "reason": continuity.get("reason", "history_not_continuous")}
    return {"ok": True, "authenticated_continuity": True,
            "previous_count": continuity["previous_count"],
            "current_count": continuity["current_count"],
            "new_reservations": continuity["new_reservations"],
            "atomic_reservation_verified": False, "durability_verified": False,
            "freshness_verified": False, "execution_authority": False,
            "recovery_committed": False, "audit_ledger_written": False,
            "requires_caller_audit": True, "requires_caller_state_recheck": True}


def sign_advcu_reservation_revision(previous_commitment: Any,
        current_commitment: Any, *, previous_revision: int,
        approved_for_signing: bool = False,
        provenance_key: Optional[bytes] = None,
        trusted_previous_signature: Optional[str] = None,
        maximum_reservations: int = 10000) -> Dict[str, Any]:
    """Sign one bounded, monotonic reservation revision; never persist or execute.

    Caller must retain a trusted previous revision AND its snapshot signature.
    A signature alone cannot prove the caller's durable rollback anchor.
    """
    denied = {"ok": False, "execution_authority": False,
              "audit_ledger_written": False}
    if approved_for_signing is not True:
        return {**denied, "reason": "explicit_revision_signing_approval_required"}
    if (type(previous_revision) is not int or
            not 0 <= previous_revision < 2**63 - 1):
        return {**denied, "reason": "invalid_previous_revision"}
    checked = verify_advcu_authenticated_reservation_continuity(
        previous_commitment, current_commitment,
        approved_for_verification=True,
        trusted_provenance_key=provenance_key,
        trusted_previous_signature=trusted_previous_signature,
        maximum_reservations=maximum_reservations)
    if not checked.get("ok"):
        return {**denied, "reason": checked.get("reason", "continuity_failed")}
    revision = previous_revision + 1
    payload = {"previous_revision": previous_revision, "revision": revision,
               "previous_signature": previous_commitment["signature"],
               "current_signature": current_commitment["signature"]}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True)
    signature = hmac.new(provenance_key,
        ("AdvCU.reservation_revision.v1:" + canonical).encode("ascii"),
        hashlib.sha256).hexdigest()
    return {"ok": True, "schema": "SarahMemory.AdvCU.reservation_revision.v1",
            "payload": payload, "signature": signature,
            "execution_authority": False, "audit_ledger_written": False,
            "requires_caller_audit": True}


def verify_advcu_reservation_revision(previous_commitment: Any,
        current_commitment: Any, revision_receipt: Any, *,
        trusted_previous_revision: int,
        trusted_previous_signature: Optional[str] = None,
        approved_for_verification: bool = False,
        trusted_provenance_key: Optional[bytes] = None,
        maximum_reservations: int = 10000) -> Dict[str, Any]:
    """Read-only revision verification against caller-owned rollback anchors."""
    denied = {"ok": False, "execution_authority": False,
              "audit_ledger_written": False, "recovery_committed": False}
    if approved_for_verification is not True:
        return {**denied, "reason": "explicit_revision_verification_approval_required"}
    if (type(trusted_previous_revision) is not int or
            not 0 <= trusted_previous_revision < 2**63 - 1 or
            type(revision_receipt) is not dict or
            set(revision_receipt) != {"ok", "schema", "payload", "signature",
                                      "execution_authority", "audit_ledger_written",
                                      "requires_caller_audit"} or
            revision_receipt.get("ok") is not True or
            revision_receipt.get("schema") != "SarahMemory.AdvCU.reservation_revision.v1" or
            revision_receipt.get("execution_authority") is not False or
            revision_receipt.get("audit_ledger_written") is not False or
            revision_receipt.get("requires_caller_audit") is not True or
            type(revision_receipt.get("signature")) is not str or
            len(revision_receipt["signature"]) != 64 or
            any(c not in "0123456789abcdef" for c in revision_receipt["signature"])):
        return {**denied, "reason": "invalid_revision_receipt"}
    expected = sign_advcu_reservation_revision(
        previous_commitment, current_commitment,
        previous_revision=trusted_previous_revision,
        approved_for_signing=True, provenance_key=trusted_provenance_key,
        trusted_previous_signature=trusted_previous_signature,
        maximum_reservations=maximum_reservations)
    if not expected.get("ok"):
        return {**denied, "reason": expected.get("reason", "revision_continuity_failed")}
    if (type(revision_receipt.get("payload")) is not dict or
            revision_receipt["payload"] != expected["payload"]):
        return {**denied, "reason": "revision_anchor_or_payload_mismatch"}
    if not hmac.compare_digest(revision_receipt["signature"], expected["signature"]):
        return {**denied, "reason": "revision_authentication_failed"}
    return {"ok": True, "revision_authenticated": True,
            "revision": expected["payload"]["revision"],
            "freshness_verified": False, "durability_verified": False,
            "execution_authority": False, "recovery_committed": False,
            "audit_ledger_written": False, "requires_caller_audit": True,
            "requires_caller_state_recheck": True}



def verify_advcu_reservation_revision_chain(commitments: Any,
        revision_receipts: Any, *, trusted_start_revision: int,
        trusted_start_signature: Optional[str] = None,
        approved_for_verification: bool = False,
        trusted_provenance_key: Optional[bytes] = None,
        maximum_reservations: int = 10000,
        maximum_chain_links: int = 128) -> Dict[str, Any]:
    """Verify a bounded, caller-anchored chain; no writes or authority grants.

    The caller must protect the starting revision and signature independently.
    Verification proves neither freshness nor durable reservation history.
    """
    denied = {"ok": False, "execution_authority": False,
              "recovery_committed": False, "audit_ledger_written": False}
    if approved_for_verification is not True:
        return {**denied, "reason": "explicit_chain_verification_approval_required"}
    if (type(maximum_chain_links) is not int or
            not 1 <= maximum_chain_links <= 128 or
            type(maximum_reservations) is not int or
            not 1 <= maximum_reservations <= 10000 or
            type(trusted_start_revision) is not int or
            not 0 <= trusted_start_revision < 2**63 - 1 or
            type(trusted_start_signature) is not str or
            len(trusted_start_signature) != 64 or
            any(c not in "0123456789abcdef" for c in trusted_start_signature) or
            type(commitments) not in (list, tuple) or
            type(revision_receipts) not in (list, tuple) or
            not 1 <= len(revision_receipts) <= maximum_chain_links or
            len(commitments) != len(revision_receipts) + 1 or
            trusted_start_revision + len(revision_receipts) >= 2**63):
        return {**denied, "reason": "invalid_revision_chain_bounds"}
    if (type(commitments[0]) is not dict or
            type(commitments[0].get("signature")) is not str or
            not hmac.compare_digest(commitments[0]["signature"],
                                    trusted_start_signature)):
        return {**denied, "reason": "starting_snapshot_anchor_mismatch"}
    for index, receipt in enumerate(revision_receipts):
        result = verify_advcu_reservation_revision(
            commitments[index], commitments[index + 1], receipt,
            trusted_previous_revision=trusted_start_revision + index,
            trusted_previous_signature=commitments[index]["signature"],
            approved_for_verification=True,
            trusted_provenance_key=trusted_provenance_key,
            maximum_reservations=maximum_reservations)
        if not result.get("ok"):
            return {**denied, "reason": "chain_link_verification_failed",
                    "failed_link": index, "link_reason": result.get("reason", "invalid_link")}
    return {"ok": True, "chain_authenticated": True,
            "links_verified": len(revision_receipts),
            "final_revision": trusted_start_revision + len(revision_receipts),
            "final_snapshot_signature": commitments[-1]["signature"],
            "freshness_verified": False, "durability_verified": False,
            "atomic_reservation_verified": False, "execution_authority": False,
            "recovery_committed": False, "audit_ledger_written": False,
            "requires_caller_audit": True, "requires_caller_state_recheck": True}


def verify_advcu_reservation_revision_checkpoint(commitments: Any,
        revision_receipts: Any, *, trusted_start_revision: int,
        trusted_start_signature: Optional[str], trusted_end_revision: int,
        trusted_end_signature: Optional[str],
        approved_for_verification: bool = False,
        trusted_provenance_key: Optional[bytes] = None,
        maximum_reservations: int = 10000,
        maximum_chain_links: int = 128) -> Dict[str, Any]:
    """Verify a bounded chain between two independently caller-trusted anchors.

    No implicit checkpoint trust, key management, state writes or audit claims.
    A valid chain does not establish freshness of either external anchor.
    """
    denied = {"ok": False, "execution_authority": False,
              "recovery_committed": False, "audit_ledger_written": False}
    if approved_for_verification is not True:
        return {**denied, "reason": "explicit_checkpoint_verification_approval_required"}
    if (type(trusted_end_revision) is not int or
            not 0 <= trusted_end_revision < 2**63 or
            type(trusted_end_signature) is not str or
            len(trusted_end_signature) != 64 or
            any(c not in "0123456789abcdef" for c in trusted_end_signature)):
        return {**denied, "reason": "invalid_trusted_end_checkpoint"}
    checked = verify_advcu_reservation_revision_chain(
        commitments, revision_receipts,
        trusted_start_revision=trusted_start_revision,
        trusted_start_signature=trusted_start_signature,
        approved_for_verification=True,
        trusted_provenance_key=trusted_provenance_key,
        maximum_reservations=maximum_reservations,
        maximum_chain_links=maximum_chain_links)
    if not checked.get("ok"):
        return {**denied, "reason": checked.get("reason", "chain_verification_failed"),
                **({"failed_link": checked["failed_link"]} if "failed_link" in checked else {})}
    if (checked["final_revision"] != trusted_end_revision or
            not hmac.compare_digest(checked["final_snapshot_signature"],
                                    trusted_end_signature)):
        return {**denied, "reason": "trusted_end_checkpoint_mismatch"}
    return {"ok": True, "checkpoint_continuity_verified": True,
            "links_verified": checked["links_verified"],
            "final_revision": checked["final_revision"],
            "final_snapshot_signature": checked["final_snapshot_signature"],
            "freshness_verified": False, "durability_verified": False,
            "atomic_reservation_verified": False, "execution_authority": False,
            "recovery_committed": False, "audit_ledger_written": False,
            "requires_caller_audit": True, "requires_caller_state_recheck": True}


def verify_advcu_reservation_checkpoint_segments(segments: Any, *,
        trusted_start_revision: int, trusted_start_signature: Optional[str],
        trusted_end_revision: int, trusted_end_signature: Optional[str],
        approved_for_verification: bool = False,
        trusted_provenance_key: Optional[bytes] = None,
        maximum_reservations: int = 10000,
        maximum_segments: int = 8,
        maximum_total_links: int = 128) -> Dict[str, Any]:
    """Read-only bounded verification of consecutive caller-anchored segments.

    Each segment is (commitments, revision_receipts). Intermediate anchors
    are derived from authenticated prior segments, never silently trusted.
    Neither durability nor freshness nor execution permission is established.
    """
    denied = {"ok": False, "execution_authority": False,
              "recovery_committed": False, "audit_ledger_written": False}
    if approved_for_verification is not True:
        return {**denied, "reason": "explicit_segment_verification_approval_required"}
    if (type(maximum_segments) is not int or not 1 <= maximum_segments <= 8 or
            type(maximum_total_links) is not int or not 1 <= maximum_total_links <= 128 or
            type(maximum_reservations) is not int or not 1 <= maximum_reservations <= 10000 or
            type(segments) not in (list, tuple) or
            not 1 <= len(segments) <= maximum_segments or
            type(trusted_start_revision) is not int or
            not 0 <= trusted_start_revision < 2**63 or
            type(trusted_end_revision) is not int or
            not 0 <= trusted_end_revision < 2**63 or
            type(trusted_start_signature) is not str or
            len(trusted_start_signature) != 64 or
            any(c not in "0123456789abcdef" for c in trusted_start_signature) or
            type(trusted_end_signature) is not str or
            len(trusted_end_signature) != 64 or
            any(c not in "0123456789abcdef" for c in trusted_end_signature)):
        return {**denied, "reason": "invalid_segment_bounds_or_anchors"}
    total_links = 0
    for segment in segments:
        if (type(segment) not in (list, tuple) or len(segment) != 2 or
                type(segment[0]) not in (list, tuple) or
                type(segment[1]) not in (list, tuple) or
                not 1 <= len(segment[1]) <= 128 or
                len(segment[0]) != len(segment[1]) + 1):
            return {**denied, "reason": "invalid_segment_structure"}
        total_links += len(segment[1])
        if total_links > maximum_total_links:
            return {**denied, "reason": "total_link_capacity_exceeded"}
    if trusted_start_revision + total_links != trusted_end_revision:
        return {**denied, "reason": "segment_revision_anchor_mismatch"}
    revision, signature = trusted_start_revision, trusted_start_signature
    for index, (commitments, receipts) in enumerate(segments):
        checked = verify_advcu_reservation_revision_chain(
            commitments, receipts, trusted_start_revision=revision,
            trusted_start_signature=signature, approved_for_verification=True,
            trusted_provenance_key=trusted_provenance_key,
            maximum_reservations=maximum_reservations,
            maximum_chain_links=maximum_total_links)
        if not checked.get("ok"):
            return {**denied, "reason": "segment_verification_failed",
                    "failed_segment": index,
                    "segment_reason": checked.get("reason", "invalid_segment")}
        revision, signature = checked["final_revision"], checked["final_snapshot_signature"]
    if not hmac.compare_digest(signature, trusted_end_signature):
        return {**denied, "reason": "segment_end_signature_mismatch"}
    return {"ok": True, "segments_verified": len(segments),
            "links_verified": total_links, "final_revision": revision,
            "final_snapshot_signature": signature, "freshness_verified": False,
            "durability_verified": False, "atomic_reservation_verified": False,
            "execution_authority": False, "recovery_committed": False,
            "audit_ledger_written": False, "requires_caller_audit": True,
            "requires_caller_state_recheck": True}


def verify_advcu_reservation_intermediate_anchors(segments: Any,
        intermediate_anchors: Any, *, trusted_start_revision: int,
        trusted_start_signature: Optional[str], trusted_end_revision: int,
        trusted_end_signature: Optional[str],
        approved_for_verification: bool = False,
        trusted_provenance_key: Optional[bytes] = None,
        maximum_reservations: int = 10000,
        maximum_segments: int = 8,
        maximum_total_links: int = 128) -> Dict[str, Any]:
    """Read-only verification of independently supplied intermediate checkpoints.

    Exactly one caller-trusted (revision, signature) anchor is required at
    each internal segment boundary. These anchors MUST be independently
    authenticated by the caller; this function does not confer trust on them.
    """
    denied = {"ok": False, "execution_authority": False,
              "recovery_committed": False, "audit_ledger_written": False}
    if approved_for_verification is not True:
        return {**denied, "reason": "explicit_intermediate_verification_approval_required"}
    if (type(maximum_segments) is not int or
            not 1 <= maximum_segments <= 8 or
            type(maximum_total_links) is not int or
            not 1 <= maximum_total_links <= 128 or
            type(maximum_reservations) is not int or
            not 1 <= maximum_reservations <= 10000):
        return {**denied, "reason": "invalid_segment_bounds_or_anchors"}
    if (type(segments) not in (list, tuple) or
            not 2 <= len(segments) <= maximum_segments or
            type(intermediate_anchors) not in (list, tuple) or
            len(intermediate_anchors) != len(segments) - 1):
        return {**denied, "reason": ("invalid_segment_bounds_or_anchors"
                if type(segments) in (list, tuple) and len(segments) > maximum_segments
                else "invalid_intermediate_anchor_count")}
    for anchor in intermediate_anchors:
        if (type(anchor) not in (tuple, list) or len(anchor) != 2 or
                type(anchor[0]) is not int or
                not 0 <= anchor[0] < 2**63 or
                type(anchor[1]) is not str or len(anchor[1]) != 64 or
                any(ch not in "0123456789abcdef" for ch in anchor[1])):
            return {**denied, "reason": "invalid_intermediate_anchor"}
    # Reject contradictory or out-of-order caller checkpoints before the
    # more expensive cryptographic segment traversal.  This preflight does
    # not confer trust on any supplied anchor or mutate caller state.
    if (type(trusted_start_revision) is not int or
            not 0 <= trusted_start_revision < 2**63 or
            type(trusted_end_revision) is not int or
            not 0 <= trusted_end_revision < 2**63):
        return {**denied, "reason": "invalid_revision_anchor"}
    # Reject malformed caller trust anchors before traversing revision history.
    # Authentication still occurs in the delegated cryptographic verifier.
    if (type(trusted_start_signature) is not str or
            len(trusted_start_signature) != 64 or
            any(ch not in "0123456789abcdef" for ch in trusted_start_signature) or
            type(trusted_end_signature) is not str or
            len(trusted_end_signature) != 64 or
            any(ch not in "0123456789abcdef" for ch in trusted_end_signature)):
        return {**denied, "reason": "invalid_terminal_signature_anchor"}
    # Enforce the caller's aggregate budget before cryptographic traversal.
    # Reject oversized segments without iterating through individual receipts.
    total_links = 0
    for segment in segments:
        if (type(segment) not in (tuple, list) or len(segment) != 2 or
                type(segment[0]) not in (tuple, list) or
                type(segment[1]) not in (tuple, list) or
                not 1 <= len(segment[1]) <= 128 or
                len(segment[0]) != len(segment[1]) + 1):
            return {**denied, "reason": "invalid_segment_structure"}
        total_links += len(segment[1])
        if total_links > maximum_total_links:
            return {**denied, "reason": "total_link_capacity_exceeded"}
    # Fail closed on a contradictory terminal revision before any HMAC work.
    # This is only a structural preflight; trust remains caller-managed.
    if (trusted_start_revision + total_links != trusted_end_revision or
            trusted_start_revision + total_links >= 2**63):
        return {**denied, "reason": "segment_revision_anchor_mismatch"}
    expected_revision = trusted_start_revision
    for index, (segment, anchor) in enumerate(zip(segments[:-1], intermediate_anchors)):
        if (type(segment) not in (tuple, list) or len(segment) != 2 or
                type(segment[1]) not in (tuple, list) or
                not 1 <= len(segment[1]) <= 128):
            return {**denied, "reason": "invalid_segment_structure"}
        expected_revision += len(segment[1])
        if (expected_revision >= trusted_end_revision or
                anchor[0] != expected_revision):
            return {**denied, "reason": "intermediate_revision_order_mismatch",
                    "failed_boundary": index}
    checked = verify_advcu_reservation_checkpoint_segments(
        segments, trusted_start_revision=trusted_start_revision,
        trusted_start_signature=trusted_start_signature,
        trusted_end_revision=trusted_end_revision,
        trusted_end_signature=trusted_end_signature,
        approved_for_verification=True,
        trusted_provenance_key=trusted_provenance_key,
        maximum_reservations=maximum_reservations,
        maximum_segments=maximum_segments,
        maximum_total_links=maximum_total_links)
    if not checked.get("ok"):
        return {**denied, "reason": checked.get("reason", "segment_verification_failed"),
                **({"failed_segment": checked["failed_segment"]}
                   if "failed_segment" in checked else {})}
    revision = trusted_start_revision
    for index, segment in enumerate(segments[:-1]):
        revision += len(segment[1])
        signature = segment[0][-1]["signature"]
        anchor_revision, anchor_signature = intermediate_anchors[index]
        if (revision != anchor_revision or
                not hmac.compare_digest(signature, anchor_signature)):
            return {**denied, "reason": "intermediate_anchor_mismatch",
                    "failed_boundary": index}
    return {"ok": True, "intermediate_anchors_verified": len(intermediate_anchors),
            "segments_verified": checked["segments_verified"],
            "links_verified": checked["links_verified"],
            "final_revision": checked["final_revision"],
            "final_snapshot_signature": checked["final_snapshot_signature"],
            "freshness_verified": False, "durability_verified": False,
            "atomic_reservation_verified": False, "execution_authority": False,
            "recovery_committed": False, "audit_ledger_written": False,
            "requires_caller_audit": True, "requires_caller_state_recheck": True}
