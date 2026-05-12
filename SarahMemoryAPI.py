"""=== SarahMemory Project ===
File: SarahMemoryAPI.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-03-03
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

MULTI-PROVIDER API ORCHESTRATION MODULE 
=====================================================================

PURPOSE:
--------
SarahMemoryAPI is the central API orchestration layer for SarahMemory with multi=redundancies
It manages connections to multiple AI providers including Local Downloaded Models, 
API keys must be set in the .env file or system enviorments, API providers must be selected 
to 'True' in the SarahMemoryGlobals.py file approx line number (1446-1466)
(OpenAI, Claude, Mistral,Gemini, HuggingFace, etc.) 
and provides intelligent failover, caching,
and context-aware prompt building.
The SarahMemory 'SarahNET' Network node mesh is the "Last Resort Endpoint" if all else fails.

KEY CAPABILITIES:
-----------------
1. MULTI-PROVIDER SUPPORT
   - OpenAI (GPT-5[.0,.1,.2,etc], GPT-4, GPT-4o, GPT-4.1, o1, o3, etc.)
   - Anthropic Claude (claude-3-opus, claude-3-sonnet, etc.)
   - Mistral AI (mistral-large, mistral-medium, etc.)
   - Google Gemini (gemini-pro, gemini-1.5-pro, etc.)
   - HuggingFace Inference API
   - DeepSeek, Groq, Cohere, and more

2. INTELLIGENT ROUTING
   - Automatic provider selection based on intent
   - Cost-tier aware model selection
   - Capability-based routing (vision, TTS, STT)
   - Automatic fallback on failure

3. CONTEXT-AWARE PROMPTING
   - Intent-based role assignment (60+ specialized roles)
   - Emotional state integration
   - Conversation context injection
   - Complexity and tone adaptation

4. RESILIENCE FEATURES
   - Multi-tier fallback chain
   - Response caching
   - Mesh network fallback (SarahNet)
   - Offline mode handling
   - Rate limit handling

5. COMPREHENSIVE LOGGING
   - API event logging to SQLite
   - Research path tracking
   - Performance metrics
   - Error diagnostics

INTEGRATION POINTS:
-------------------
- SarahMemoryReply.py: Uses send_to_api for response generation
- SarahMemoryResearch.py: Uses send_to_api for research queries
- SarahMemoryCompare.py: Uses send_to_api for comparisons
- SarahMemoryWebSYM.py: Uses send_to_openai (deprecated alias)
- app.py: Uses send_to_api for web interface
- SarahMemoryUpdater.py: Uses API module for updates

DATABASE TABLES (system_logs.db):
---------------------------------
- api_integration_events: API call logging
- response: Response storage
- cognitive_events: Cognitive analysis results

===============================================================================
"""

from __future__ import annotations
# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "api_orchestrator"
# CATEGORY = "provider_routing"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "multi_provider_ai"
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "api_orchestration"
# FAMILY = "core_integration"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Central multi-provider API orchestration layer for local models, external AI APIs, routing, failover, caching, and prompt/context assembly."
# --- SARAHMETA END ---
import json
import re
import logging
import os
import sys
import sqlite3
import threading
import time
# Local LLM (3rd-party models) runtime
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    # Optional helpers for model-type inspection / seq2seq fallback
    try:
        from transformers import AutoConfig, AutoModelForSeq2SeqLM
    except Exception:
        AutoConfig = None
        AutoModelForSeq2SeqLM = None
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    AutoConfig = None
    AutoModelForSeq2SeqLM = None

import queue
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

# HTTP client
try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    requests = None
    _HAS_REQUESTS = False

# ============================================================================
# SAFE IMPORTS
# ============================================================================

# Import globals safely
try:
    import SarahMemoryGlobals as config
    from SarahMemoryGlobals import DATASETS_DIR, run_async
except ImportError:
    class config:
        DATASETS_DIR = os.path.join(os.getcwd(), "data", "memory", "datasets")
        BASE_DIR = os.getcwd()
        DEBUG_MODE = True
        API_RESEARCH_ENABLED = True
        API_TIMEOUT = 30
        SAFE_MODE = False
        LOCAL_ONLY_MODE = False
        SARAHNET_ENABLED = False
    DATASETS_DIR = config.DATASETS_DIR
    def run_async(fn):
        import threading
        t = threading.Thread(target=fn, daemon=True)
        t.start()
        return t

# Mesh network fallback
try:
    from SarahMemoryNetwork import NetworkNode, NetworkProtocol
except Exception:
    NetworkNode = None
    NetworkProtocol = None

# Safety flags
try:
    from SarahMemoryGlobals import SAFE_MODE, LOCAL_ONLY_MODE, is_offline, get_active_api
except Exception:
    SAFE_MODE = getattr(config, 'SAFE_MODE', False)
    LOCAL_ONLY_MODE = getattr(config, 'LOCAL_ONLY_MODE', False)

    def is_offline() -> bool:
        return False

    def get_active_api(primary=None, fallbacks=None) -> Optional[str]:
        return None

# Intent classification
try:
    from SarahMemoryAdvCU import classify_intent
except ImportError:
    def classify_intent(text: str, **kwargs) -> str:
        return "question"

# Context retrieval
try:
    from SarahMemoryAiFunctions import get_context
except ImportError:
    def get_context() -> List[Dict]:
        return []

# Emotional state
try:
    from SarahMemoryAdaptive import simulate_emotion_response
except ImportError:
    def simulate_emotion_response(input_type: str = "neutral") -> Dict[str, float]:
        return {
            "joy": 0.5, "trust": 0.5, "fear": 0.1,
            "anger": 0.1, "surprise": 0.2, "sadness": 0.1,
            "disgust": 0.1, "anticipation": 0.4
        }


# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger("SarahMemoryAPI")
logger.setLevel(logging.DEBUG if getattr(config, 'DEBUG_MODE', False) else logging.INFO)
if not logger.hasHandlers():
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - [%(name)s] %(message)s"
    ))
    logger.addHandler(_handler)
logger.propagate = False

# Research path logger
research_path_logger = logging.getLogger("ResearchPathLogger")


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================
#
# Lazy caches (avoid loading heavy stuff at import-time)
__EMBEDDER_LOCK = threading.Lock()
__EMBEDDER_CACHE: Dict[str, Any] = {}
__MODEL_AVAIL_CACHE: Optional[Dict[str, bool]] = None

def _safe_bool(x: Any) -> bool:
    try:
        return bool(x)
    except Exception:
        return False

def _models_root() -> str:
    # Prefer Globals MODELS_DIR if present, else BASE_DIR/data/models
    base_dir = getattr(config, "BASE_DIR", os.getcwd())
    # models_dir = getattr(config, "MODELS_DIR", os.path.join(base_dir, "data", "models"))
    # Your convention mentioned: ./data/models/SarahMemory/...
    #return os.path.join(models_dir, "SarahMemory")
    return getattr(config, "MODELS_DIR", os.path.join(base_dir, "data", "models"))

def _is_cloud() -> bool:
    # RUN_MODE exists in Globals (auto-detects PythonAnywhere)
    return getattr(config, "RUN_MODE", "local") == "cloud"

def _enabled_model_names() -> Dict[str, bool]:
    """Return user-enabled local model names from SarahMemoryGlobals.MODEL_CONFIG.

    IMPORTANT:
    - This function must NOT hardcode model repository names.
    - The single source of truth for model identifiers is SarahMemoryGlobals.py.
    """
    model_cfg = getattr(config, "MODEL_CONFIG", {}) or {}
    return {str(k): _safe_bool(v) for k, v in model_cfg.items() if _safe_bool(v)}

def pick_embedding_model(
    requested: Optional[str] = None,
    *,
    text: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], List[str], str]:
    """Pick exactly ONE embedding model (plus ordered fallbacks) using the global resolver.

    Selection precedence:
      1) Caller-requested model (if user-enabled and present on disk)
      2) SarahMemoryGlobals.resolve_model('embeddings') selection + fallbacks
      3) First enabled model found on disk (deterministic)

    Returns: (selected, fallbacks, source)
      - selected may be None (e.g., Tier Rating = POOR and user did not enable any models)
      - source is a small string for diagnostics ('requested'|'resolver'|'enabled_first'|'none')
    """
    meta = meta or {}

    # 1) Requested override: only if user-enabled
    enabled = _enabled_model_names()

    def _is_installed_local(name: str) -> bool:
        try:
            if _is_cloud():
                return False
            root = _models_root()
            candidates = [
                os.path.join(root, name),
                os.path.join(root, name.replace("/", "_")),
                os.path.join(root, name.replace("-", "_")),
                os.path.join(root, f"{name}.json"),
            ]
            return any(os.path.exists(p) for p in candidates)
        except Exception:
            return False

    if requested and enabled.get(str(requested), False) and _is_installed_local(str(requested)):
        return str(requested), [], "requested"

    # 2) Canonical resolver in Globals
    try:
        resolve_fn = getattr(config, "resolve_model", None)
        if callable(resolve_fn):
            res = resolve_fn("embeddings", text=text or "", meta=meta or {}) or {}
            sel = res.get("selected")
            fbs = res.get("fallbacks") or []
            sel = str(sel) if sel else None
            fbs = [str(x) for x in fbs if x]
            if sel:
                return sel, fbs, "resolver"
            # If resolver returns no model (e.g. POOR tier), honor it.
            if fbs:
                return fbs[0], fbs[1:], "resolver"
            return None, [], "none"
    except Exception:
        pass

    # 3) Deterministic fallback: first enabled model that exists on disk
    try:
        for name in sorted(enabled.keys()):
            if _is_installed_local(name):
                return name, [], "enabled_first"
    except Exception:
        pass

    return None, [], "none"

def get_embedder(model_name: Optional[str] = None, *, text: str = "", meta: Optional[Dict[str, Any]] = None):
    """Lazy-load a local SentenceTransformer embedder using the global resolver.

    - Picks ONE model per request (plus fallbacks) via pick_embedding_model().
    - If no model is selected (e.g., Tier Rating = POOR and user has not enabled any),
      this returns None and callers should fall back to core-only embeddings.
    """
    selected, fallbacks, _src = pick_embedding_model(model_name, text=text or "", meta=meta or {})
    if not selected:
        return None

    with __EMBEDDER_LOCK:
        if selected in __EMBEDDER_CACHE:
            return __EMBEDDER_CACHE[selected]

        # In cloud mode, do not try to load local models
        if _is_cloud():
            raise RuntimeError("Local embedder requested in cloud mode (RUN_MODE=cloud).")

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise RuntimeError(f"sentence-transformers not available: {e}")

        local_root = _models_root()
        os.makedirs(local_root, exist_ok=True)

        # Try primary then fallbacks (all are repo strings from Globals, not hardcoded here)
        last_err: Optional[Exception] = None
        for repo in [selected] + list(fallbacks or []):
            try:
                emb = SentenceTransformer(repo, cache_folder=local_root)
                __EMBEDDER_CACHE[repo] = emb
                return emb
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"Failed to load any enabled embedding model: {last_err}")


def _fallback_embed(text: str, dim: int = 64) -> List[float]:
    """Deterministic core-only embedding (no downloads, never throws)."""
    import hashlib, math
    h = hashlib.sha256((text or "").encode("utf-8")).digest()
    out: List[float] = []
    for i in range(int(dim)):
        b = h[i % len(h)]
        out.append(float(math.sin((b + i) * 0.0174533)))
    return out


def embed_text(text: str, model_name: Optional[str] = None, *, meta: Optional[Dict[str, Any]] = None):
    """Single, reusable embedding function for the whole stack.

    - Uses selected local embedder when available.
    - Core-only fallback (hash projection) when no model is available/allowed.
    """
    dim = int(getattr(config, "EMBEDDING_DIMENSION", 64) or 64)
    try:
        emb = get_embedder(model_name, text=text or "", meta=meta or {})
        if emb is None:
            return _fallback_embed(text, dim=dim)
        vec = emb.encode([text], normalize_embeddings=True)
        return vec[0].tolist()
    except Exception:
        return _fallback_embed(text, dim=dim)

# API Disabled flag
API_DISABLED = not getattr(config, 'API_RESEARCH_ENABLED', True)

# Response cache
_response_cache: Dict[str, str] = {}
_cache_lock = threading.RLock()

# Database path
SYSTEM_LOGS_DB = os.path.join(DATASETS_DIR, "system_logs.db")
os.makedirs(os.path.dirname(SYSTEM_LOGS_DB), exist_ok=True)


# ============================================================================
# API PROVIDERS ENUM
# ============================================================================

class APIProvider(Enum):
    """Supported API providers."""
    OPENAI = "openai"
    CLAUDE = "claude"
    ANTHROPIC = "anthropic"  # Alias for claude
    MISTRAL = "mistral"
    GEMINI = "gemini"
    HUGGINGFACE = "huggingface"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    COHERE = "cohere"
    LOCAL_LLM = "local_llm"
    LOCAL_API = "local"
    MESH = "mesh"


# ============================================================================
# API KEYS AND ENDPOINTS
# ============================================================================

# API Keys from environment (multiple fallback names supported)
API_KEYS: Dict[str, Optional[str]] = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "claude": os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY"),
    "mistral": os.getenv("MISTRAL_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    "huggingface": os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY"),
    "deepseek": os.getenv("DEEPSEEK_API_KEY"),
    "groq": os.getenv("GROQ_API_KEY"),
    "cohere": os.getenv("COHERE_API_KEY") or os.getenv("CO_API_KEY"),
    "local": os.getenv("LOCAL_BRAIN"),
    "local_llm": os.getenv("LOCAL_LLM_API"),
    "mesh": os.getenv("MESH_API"),
}

# OpenAI endpoint selection (honor custom endpoint if valid)
_openai_env = (os.getenv("OPENAI_ENDPOINT") or "").strip()
if _openai_env and ("openai" in _openai_env or "azure.com" in _openai_env):
    OPENAI_BASE_URL = _openai_env
    logger.info(f"[SarahMemoryAPI] Using custom OPENAI_ENDPOINT: {OPENAI_BASE_URL}")
else:
    OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"

# API URLs for each provider
API_URLS: Dict[str, str] = {
    "openai": OPENAI_BASE_URL,
    "claude": "https://api.anthropic.com/v1/messages",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    "huggingface": "https://api-inference.huggingface.co/models/{model}",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "cohere": "https://api.cohere.ai/v1/chat",
    "local": "http://127.0.0.1:8000/api/local/chat",
    "local_llm": "local-runtime",
    "mesh": "https://api.sarahmemory.com/api/chat",
}

# Default models for each provider
DEFAULT_MODELS: Dict[str, str] = {
    "openai": "gpt-5.1",
    "claude": "claude-3-sonnet-20240229",
    "anthropic": "claude-3-sonnet-20240229",
    "mistral": "mistral-large-latest",
    "gemini": "gemini-1.5-flash", #change to gemini-1.5-pro if needed
    "huggingface": "meta-llama/Meta-Llama-3-8B-Instruct",
    "deepseek": "deepseek-chat",
    "groq": "llama-3.1-70b-versatile",
    "cohere": "command-r-plus",
    "local": "synapses-micro-brain",
    "local_llm": "auto",  # auto-selected local model
    "mesh": "auto",
}

# Provider priority for fallback
PROVIDER_PRIORITY: List[str] = [
    "local",        # SarahMemory AiOS Synapses/Neuron micro-brain fully local .db files, LM (deterministic + retrieval + tool routing)
    "local_llm",    # Local 3rd Party LLM MODEL runtime
    "openai",       #OPEN AI API, Set API key in either .env or Local system enviorment settings
    "claude",       #CLAUDE AI API, Set API key in either .env or local system enviorment settings
    "mistral",      #MISTRAL AI API
    "gemini",       #GOOGLE GEMINI AI API
    "deepseek",     #DEEPSEEK AI API
    "groq",         #GROK AI API
    "cohere",       #COHERE AI API
    "huggingface",  #HUGGING FACE API
    "mesh",         # SarahMemory ONLINE CLOUD Server/Distributed/AiOS NETWORK NODES/ tier LAST unless explicitly requested LAST RESORT, P2P ,node, 
]
PRIMARY_PROVIDER = next((p for p in PROVIDER_PRIORITY if _safe_bool(API_KEYS.get(p))), "local")
# ============================================================================
# ENHANCED ROLE MAP (60+ SPECIALIZED ROLES)
# ============================================================================

ROLE_MAP: Dict[str, str] = {
    # === GENERAL PURPOSE ===
    "unknown": "general-purpose AI assistant with broad knowledge across all domains",
    "question": "expert researcher and knowledge synthesizer",
    "statement": "thoughtful conversationalist and active listener",
    "command": "precise AI task executor and automation specialist",
    "conversation": "engaging conversational partner",

    # === TECHNICAL & DEVELOPMENT ===
    "debug": "senior software engineer specializing in debugging and troubleshooting",
    "code": "expert software developer proficient in multiple programming languages",
    "programming": "full-stack developer with expertise in modern frameworks",
    "architecture": "senior software architect specializing in system design",
    "devops": "DevOps engineer expert in CI/CD, containers, and cloud infrastructure",
    "security": "cybersecurity expert specializing in threat analysis and secure coding",
    "database": "database administrator and SQL optimization specialist",
    "api": "API design specialist and integration expert",
    "frontend": "frontend developer expert in React, Vue, and modern UI/UX",
    "backend": "backend engineer specializing in scalable server architectures",
    "mobile": "mobile app developer for iOS and Android platforms",
    "cloud": "cloud solutions architect for AWS, Azure, and GCP",
    "testing": "QA engineer specializing in test automation and quality assurance",
    "algorithm": "algorithm specialist and data structures expert",

    # === EDUCATION & LEARNING ===
    "teaching": "university professor with expertise in pedagogy and curriculum design",
    "explanation": "technical expert skilled at explaining complex concepts simply",
    "tutorial": "patient teacher specializing in step-by-step instruction",
    "mentor": "experienced mentor providing guidance and career advice",
    "tutor": "dedicated tutor adapting to individual learning styles",
    "academic": "academic researcher with expertise in scholarly analysis",
    "student": "curious student seeking to understand deeply",

    # === CREATIVE & ARTS ===
    "story": "creative storyteller and narrative craftsman",
    "writing": "professional writer skilled in various styles and formats",
    "poetry": "poet with mastery of rhythm, meter, and figurative language",
    "joke": "stand-up comedian with impeccable timing",
    "humor": "comedy writer specializing in wit and wordplay",
    "screenplay": "screenwriter expert in dialogue and dramatic structure",
    "fiction": "fiction author specializing in character development and plot",
    "copywriting": "advertising copywriter creating compelling marketing content",
    "journalism": "investigative journalist committed to factual reporting",
    "creative": "creative director with innovative artistic vision",
    "music": "music theorist and composition specialist",
    "art": "art critic and visual design consultant",

    # === BUSINESS & PROFESSIONAL ===
    "finance": "certified financial analyst and investment strategist",
    "business": "business strategist with MBA-level expertise",
    "marketing": "marketing expert specializing in digital and traditional strategies",
    "sales": "sales consultant expert in negotiation and client relations",
    "legal": "legal advisor providing general legal information (not legal advice)",
    "hr": "human resources specialist in recruitment and employee relations",
    "management": "management consultant specializing in organizational efficiency",
    "startup": "startup advisor with experience in venture funding and scaling",
    "economics": "economist analyzing markets and economic trends",
    "accounting": "certified accountant specializing in financial reporting",
    "consulting": "management consultant providing strategic business advice",
    "negotiation": "negotiation expert skilled in conflict resolution",
    "entrepreneur": "serial entrepreneur with experience in building businesses",

    # === SCIENCE & RESEARCH ===
    "science": "research scientist with expertise across scientific disciplines",
    "math": "mathematician specializing in problem-solving and proofs",
    "physics": "physicist expert in theoretical and applied physics",
    "chemistry": "chemist specializing in molecular science and reactions",
    "biology": "biologist with expertise in life sciences and genetics",
    "data_science": "data scientist expert in statistics and machine learning",
    "ai": "AI researcher specializing in machine learning and neural networks",
    "research": "academic researcher skilled in methodology and analysis",
    "statistics": "statistician expert in data analysis and probability",
    "astronomy": "astronomer with expertise in celestial phenomena",
    "geology": "geologist specializing in earth sciences",
    "environmental": "environmental scientist focused on sustainability",

    # === HEALTH & WELLNESS ===
    "medical": "licensed medical information provider (not a replacement for professional medical advice)",
    "health": "health educator promoting wellness and healthy lifestyle",
    "nutrition": "registered dietitian specializing in nutrition science",
    "fitness": "certified fitness trainer and exercise physiologist",
    "mental_health": "mental health awareness educator",
    "psychology": "psychologist providing educational information about behavior and cognition",
    "therapy": "supportive counselor providing emotional support and coping strategies",
    "wellness": "holistic wellness coach promoting mind-body balance",

    # === EMOTIONAL & SOCIAL ===
    "emotion": "empathetic therapist providing emotional support and understanding",
    "empathy": "compassionate listener skilled in emotional validation",
    "support": "supportive friend offering encouragement and understanding",
    "motivation": "motivational coach inspiring positive change and action",
    "relationship": "relationship counselor providing communication strategies",
    "parenting": "parenting advisor offering evidence-based child-rearing guidance",
    "social": "social skills coach helping with interpersonal communication",

    # === SPECIALIZED DOMAINS ===
    "emergency": "crisis advisor providing calm, clear emergency guidance",
    "identity": "AI assistant with a warm, helpful personality",
    "travel": "travel advisor and destination expert",
    "food": "culinary expert and food critic",
    "gaming": "gaming expert knowledgeable in video games and esports",
    "sports": "sports analyst and athletics expert",
    "history": "historian with expertise in world history and events",
    "philosophy": "philosopher exploring ethics, logic, and existential questions",
    "religion": "religious studies scholar providing objective information",
    "politics": "political analyst providing balanced political information",
    "culture": "cultural anthropologist with expertise in global traditions",
    "language": "linguist and polyglot specializing in language learning",
    "translation": "professional translator fluent in multiple languages",

    # === LIFESTYLE & PERSONAL ===
    "lifestyle": "lifestyle coach helping optimize daily routines",
    "productivity": "productivity expert specializing in time management",
    "organization": "professional organizer helping declutter and systematize",
    "fashion": "fashion consultant with style expertise",
    "home": "home improvement specialist and interior design advisor",
    "diy": "DIY expert skilled in home projects and crafts",
    "automotive": "automotive expert knowledgeable in vehicles and maintenance",
    "pet": "veterinary information provider and pet care advisor",
    "gardening": "horticulturist and gardening expert",

    # === INTENT-SPECIFIC ROLES ===
    "greeting": "friendly AI companion ready to help",
    "farewell": "warm AI assistant wishing you well",
    "smalltalk": "engaging conversationalist enjoying casual chat",
    "time_related": "scheduling assistant helping with time and calendars",
    "system_control": "system automation assistant for device control",
    "open_app": "application launcher assistant",
    "open_url": "web navigation assistant",
    "search_web": "web research assistant finding information online",
    "play_media": "media control assistant for audio and video",
    "window_mgmt": "window management assistant for desktop organization",
    "close_quit": "application management assistant",
    "agent_control": "AI behavior controller and settings manager",
    "login": "authentication assistant helping with secure access",
    "signup": "registration assistant guiding account creation",
    "device_query": "system information specialist providing device details",
}

# Role categories for grouping
ROLE_CATEGORIES: Dict[str, List[str]] = {
    "general": ["unknown", "question", "statement", "command", "conversation"],
    "technical": ["debug", "code", "programming", "architecture", "devops", "security",
                  "database", "api", "frontend", "backend", "mobile", "cloud", "testing", "algorithm"],
    "education": ["teaching", "explanation", "tutorial", "mentor", "tutor", "academic", "student"],
    "creative": ["story", "writing", "poetry", "joke", "humor", "screenplay", "fiction",
                 "copywriting", "journalism", "creative", "music", "art"],
    "business": ["finance", "business", "marketing", "sales", "legal", "hr", "management",
                 "startup", "economics", "accounting", "consulting", "negotiation", "entrepreneur"],
    "science": ["science", "math", "physics", "chemistry", "biology", "data_science", "ai",
                "research", "statistics", "astronomy", "geology", "environmental"],
    "health": ["medical", "health", "nutrition", "fitness", "mental_health", "psychology",
               "therapy", "wellness"],
    "emotional": ["emotion", "empathy", "support", "motivation", "relationship", "parenting", "social"],
    "specialized": ["emergency", "identity", "travel", "food", "gaming", "sports", "history",
                    "philosophy", "religion", "politics", "culture", "language", "translation"],
    "lifestyle": ["lifestyle", "productivity", "organization", "fashion", "home", "diy",
                  "automotive", "pet", "gardening"],
    "intents": ["greeting", "farewell", "smalltalk", "time_related", "system_control",
                "open_app", "open_url", "search_web", "play_media", "window_mgmt",
                "close_quit", "agent_control", "login", "signup", "device_query"],
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class APIRequest:
    """Structured API request."""
    user_input: str
    provider: str = "auto"
    intent: str = "question"
    tone: str = "friendly"
    complexity: str = "adult"
    model: Optional[str] = None
    max_tokens: int = 800
    temperature: float = 1.0
    context: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIResponse:
    """Structured API response."""
    source: str
    data: Optional[str] = None
    model_used: Optional[str] = None
    prompt: Optional[str] = None
    intent: Optional[str] = None
    tone: Optional[str] = None
    emotion: Optional[Dict[str, float]] = None
    cached: bool = False
    error: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def is_success(self) -> bool:
        return self.data is not None and self.error is None


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

def _ensure_database() -> None:
    """Ensure all required database tables exist."""
    try:
        conn = sqlite3.connect(SYSTEM_LOGS_DB, timeout=10.0)
        cursor = conn.cursor()

        # API integration events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_integration_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                details TEXT,
                provider TEXT,
                model TEXT,
                latency_ms REAL
            )
        """)

        # Response storage
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS response (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                user TEXT,
                content TEXT,
                source TEXT,
                intent TEXT,
                model TEXT
            )
        """)

        # Cognitive events
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                provider TEXT,
                input_text TEXT,
                result TEXT
            )
        """)

        # Create indices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_events_ts ON api_integration_events(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_response_ts ON response(ts)")

        conn.commit()
        conn.close()

    except Exception as e:
        logger.warning(f"[DB] Database setup failed: {e}")


def log_api_event(
    event: str,
    details: str,
    provider: str = "",
    model: str = "",
    latency_ms: float = 0.0
) -> None:
    """
    Log an API event to the database.

    Args:
        event: Event type/name
        details: Event details
        provider: API provider name
        model: Model used
        latency_ms: Request latency in milliseconds
    """
    try:
        conn = sqlite3.connect(SYSTEM_LOGS_DB, timeout=5.0)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO api_integration_events
            (timestamp, event, details, provider, model, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            event,
            details[:500] if details else "",
            provider,
            model,
            latency_ms
        ))

        conn.commit()
        conn.close()

        logger.debug(f"[LOG] {event}: {details[:100]}...")

    except Exception as e:
        logger.error(f"[API Log Error] {e}")


def log_cognitive_event(provider: str, input_text: str, result: str) -> None:
    """Log a cognitive analysis event."""
    try:
        conn = sqlite3.connect(SYSTEM_LOGS_DB, timeout=5.0)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO cognitive_events (timestamp, provider, input_text, result)
            VALUES (?, ?, ?, ?)
        """, (datetime.now().isoformat(), provider, input_text[:500], result[:1000]))

        conn.commit()
        conn.close()

    except Exception as e:
        logger.error(f"[Cognitive Log Error] {e}")


# ============================================================================
# PROVIDER SELECTION
# ============================================================================

# ---------------------------------------------------------------------------
# Provider enablement + key-gating (backed by SarahMemoryGlobals toggles)
# ---------------------------------------------------------------------------
def _provider_flag_attr(provider: str) -> Optional[str]:
    """Map provider name -> SarahMemoryGlobals flag attribute name."""
    p = (provider or "").strip().lower()
    return {
        "openai": "OPEN_AI_API",
        "claude": "CLAUDE_API",
        "anthropic": "ANTHROPIC_API",
        "mistral": "MISTRAL_API",
        "gemini": "GEMINI_API",
        "huggingface": "HUGGINGFACE_API",
        "deepseek": "DEEPSEEK_API",
        "groq": "GROQ_API",
        "cohere": "COHERE_API",
        "local_llm": "LOCAL_LLM_API",
        "local": "LOCAL_API",
        "mesh": "MESH_API",
    }.get(p)

def _provider_is_enabled(provider: str) -> bool:
    p = (provider or "").strip().lower()
    if p == "anthropic":
        return bool(getattr(config, "ANTHROPIC_API", False) or getattr(config, "CLAUDE_API", False))
    attr = _provider_flag_attr(p)
    return bool(getattr(config, attr, False)) if attr else False

def _provider_requires_key(provider: str) -> bool:
    p = (provider or "").strip().lower()
    return p not in {"local", "local_llm", "mesh"}

def _provider_has_credentials(provider: str) -> bool:
    p = (provider or "").strip().lower()
    if not _provider_is_enabled(p):
        return False
    if not _provider_requires_key(p):
        return True
    return bool(API_KEYS.get(p))

def fallback_provider(current: str) -> Optional[str]:
    """Return the next available provider based on priority order."""
    for p in PROVIDER_PRIORITY:
        if p == current:
            continue
        if _provider_has_credentials(p):
            return p
    return None

def get_best_provider_for_intent(intent: str) -> str:
    """Pick the best enabled provider for an intent using a simple priority scan."""
    for p in PROVIDER_PRIORITY:
        if _provider_has_credentials(p):
            return p
    return PRIMARY_PROVIDER

def get_role_for_intent(intent: str) -> str:
    """
    Get the appropriate role description for an intent.

    Args:
        intent: The classified intent

    Returns:
        Role description string
    """
    return ROLE_MAP.get(intent, ROLE_MAP["unknown"])


def build_advanced_prompt(
    user_input: str,
    intent: str = "question",
    tone: str = "friendly",
    complexity: str = "adult",
    provider: str = "unknown"
) -> Tuple[str, Dict[str, float]]:
    """
    Build an advanced context-aware prompt with emotional state.

    Args:
        user_input: The user's input text
        intent: Classified intent
        tone: Desired tone (friendly, professional, casual, formal)
        complexity: Response complexity (child, teen, adult, expert)
        provider: Target API provider

    Returns:
        Tuple of (prompt_string, emotion_dict)
    """
    # Get role based on intent
    role = get_role_for_intent(intent)

    # Get conversation context
    try:
        context_snippets = get_context()
        recent_context = "\n".join([
            c.get("input", "") for c in context_snippets[-3:]
        ]).strip()
    except Exception:
        recent_context = ""

    # Get emotional state
    try:
        emotion = simulate_emotion_response()
    except Exception:
        emotion = {"joy": 0.5, "trust": 0.5, "fear": 0.1, "anger": 0.1, "surprise": 0.2}
    # -----------------------------------------------------------------
    # Local providers: keep prompt minimal to prevent chat-template echo.
    # The local runtime already wraps with system/user roles.
    # -----------------------------------------------------------------
    p = (provider or "").strip().lower()
    if p in {"local", "local_llm", "ollama", "model_catalog", "localmodels", "local_llm_api"}:
        prompt = (user_input or "").strip()
        research_path_logger.debug(
            f"API Call → Provider: {provider} (local-minimal), Intent: {intent}"
        )
        return prompt, emotion


    # Build mood profile string
    mood_items = [f"{k.title()}: {v:.2f}" for k, v in emotion.items() if isinstance(v, (int, float))]
    mood = ", ".join(mood_items[:5])  # Limit to 5 emotions

    # Context line
    context_line = (
        f"CONTEXT: {recent_context}"
        if recent_context
        else "CONTEXT: None (no recent conversation history)"
    )

    # Complexity guidelines
    complexity_guides = {
        "child": "Use simple words and short sentences. Explain like talking to a 10-year-old.",
        "teen": "Use clear language with some technical terms explained.",
        "adult": "Use standard professional language appropriate for adults.",
        "expert": "Use technical terminology freely. Assume domain expertise."
    }
    complexity_guide = complexity_guides.get(complexity, complexity_guides["adult"])

    # Tone guidelines
    tone_guides = {
        "friendly": "Be warm, approachable, and conversational.",
        "professional": "Be formal, precise, and business-appropriate.",
        "casual": "Be relaxed, informal, and personable.",
        "formal": "Be highly formal and respectful.",
        "empathetic": "Be understanding, supportive, and emotionally aware.",
        "technical": "Be precise, detailed, and technically accurate."
    }
    tone_guide = tone_guides.get(tone, tone_guides["friendly"])

    # Build prompt
    prompt = f"""ROLE: You are a {role}.
INTENT: {intent.upper()}
TONE: {tone_guide}
COMPLEXITY: {complexity_guide}
MOOD PROFILE: {mood}
{context_line}

QUERY: {user_input}

Respond clearly, helpfully, and concisely. Adapt your response based on the emotional context and complexity level requested.""".strip()

    # Log for research path tracking
    research_path_logger.debug(
        f"API Call → Provider: {provider}, Intent: {intent}, Role: {role[:50]}"
    )

    return prompt, emotion


# ============================================================================
# MESH NETWORK FALLBACK
# ============================================================================

_mesh_node = None
_mesh_lock = threading.Lock()


def _ensure_mesh_node():
    """Ensure mesh network node is initialized."""
    global _mesh_node

    if not getattr(config, "SARAHNET_ENABLED", False):
        return None

    if NetworkNode is None or NetworkProtocol is None:
        return None

    with _mesh_lock:
        if _mesh_node is not None:
            return _mesh_node

        try:
            cfg = config.get_sarahnet_config() if hasattr(config, 'get_sarahnet_config') else {}
            shared = config.sarahnet_shared_secret() if hasattr(config, 'sarahnet_shared_secret') else ""

            node = NetworkNode(
                node_id=cfg.get("node_id", "api-node"),
                shared_secret=shared,
                bind=(cfg.get("bind_host", "0.0.0.0"), int(cfg.get("bind_port", 9000))),
                prefer_tcp=bool(cfg.get("prefer_tcp", True)),
                allow_udp=bool(cfg.get("allow_udp", True)),
            )

            for pid, pair in (cfg.get("peers") or {}).items():
                node.add_peer(pid, (pair[0], int(pair[1])))

            node._fallback_q = queue.Queue(maxsize=8)

            def _on_msg(peer, meta, pt_bytes):
                try:
                    mtype, mmeta, payload, flags, mid = NetworkProtocol.unpack_message(pt_bytes)
                    if mmeta.get("role") == "sarah":
                        node._fallback_q.put_nowait(payload.decode("utf-8", errors="ignore"))
                except Exception:
                    pass

            node.on_message = _on_msg
            _mesh_node = node
            return node

        except Exception as e:
            logger.warning(f"[Mesh] Initialization failed: {e}")
            return None


def _mesh_fallback_request(user_input: str, timeout_s: float = 3.0) -> Optional[Dict]:
    """
    Attempt to get a response from the mesh network.

    Args:
        user_input: User's input text
        timeout_s: Timeout in seconds

    Returns:
        Response dict or None
    """
    node = _ensure_mesh_node()
    if not node or not getattr(node, "_peers", {}):
        return None

    try:
        blob = NetworkProtocol.pack_message(
            NetworkProtocol.T_DATA,
            user_input.encode("utf-8"),
            {"role": "api-fallback"},
        )

        target = next(iter(node._peers.keys()), None)
        if not target:
            return None

        node.send(target, blob)

        t_end = time.time() + max(1.0, float(timeout_s))
        while time.time() < t_end:
            try:
                text = node._fallback_q.get(timeout=0.2)
                if text:
                    return {
                        "source": "mesh",
                        "data": text,
                        "prompt": user_input,
                        "intent": "question",
                    }
            except queue.Empty:
                pass

        return None

    except Exception as e:
        logger.warning(f"[Mesh] Request error: {e}")
        return None


# ============================================================================
# PROVIDER-SPECIFIC API CALLS
# ============================================================================

def _call_openai(
    prompt: str,
    model: str,
    key: str,
    max_tokens: int = 800,
    temperature: float = 1.0
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call OpenAI API.

    Returns:
        Tuple of (content, error)
    """
    url = API_URLS["openai"]

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    messages = [{"role": "user", "content": prompt}]

    # Models that don't accept temperature
    no_temp_families = (
        "gpt-4.1", "gpt-5", "o1", "o3", "o4",
        "chatgpt-4o", "gpt-4o", "gpt-4o-mini", "gpt-realtime"
    )

    # Models that use max_completion_tokens instead of max_tokens
    new_token_models = (
        "gpt-4.1", "gpt-5", "o1", "o3", "o4",
        "chatgpt-4o", "gpt-4o", "gpt-4o-mini", "gpt-realtime"
    )

    payload = {
        "model": model,
        "messages": messages,
    }

    # Add token limit
    if model.startswith(new_token_models):
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens

    # Add temperature if supported
    if not model.startswith(no_temp_families):
        payload["temperature"] = temperature

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=getattr(config, 'API_TIMEOUT', 30)
        )

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:200]}"

        data = response.json()

        # Extract content
        choices = data.get("choices", [{}])
        if choices:
            choice = choices[0]
            if "message" in choice:
                content = choice["message"].get("content", "").strip()
            else:
                content = choice.get("text", "").strip()
            return content, None

        return None, "No choices in response"

    except Exception as e:
        return None, str(e)


def _call_claude(
    prompt: str,
    model: str,
    key: str,
    max_tokens: int = 800
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call Anthropic Claude API.

    Returns:
        Tuple of (content, error)
    """
    url = API_URLS["claude"]

    headers = {
        "x-api-key": key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=getattr(config, 'API_TIMEOUT', 30)
        )

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:200]}"

        data = response.json()

        # Extract content from Claude response
        content_blocks = data.get("content", [])
        if content_blocks:
            text_blocks = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
            content = " ".join(text_blocks).strip()
            return content, None

        # Legacy format
        content = data.get("completion", "").strip()
        if content:
            return content, None

        return None, "No content in response"

    except Exception as e:
        return None, str(e)


def _call_mistral(
    prompt: str,
    model: str,
    key: str,
    max_tokens: int = 800,
    temperature: float = 0.7
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call Mistral API.

    Returns:
        Tuple of (content, error)
    """
    url = API_URLS["mistral"]

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=getattr(config, 'API_TIMEOUT', 30)
        )

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:200]}"

        data = response.json()

        choices = data.get("choices", [{}])
        if choices:
            content = choices[0].get("message", {}).get("content", "").strip()
            return content, None

        return None, "No choices in response"

    except Exception as e:
        return None, str(e)


def _call_gemini(
    prompt: str,
    model: str,
    key: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call Google Gemini API.

    Returns:
        Tuple of (content, error)
    """
    url = API_URLS["gemini"].format(model=model) + f"?key={key}"

    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=getattr(config, 'API_TIMEOUT', 30)
        )

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:200]}"

        data = response.json()

        candidates = data.get("candidates", [{}])
        if candidates:
            content = (
                candidates[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            return content, None

        return None, "No candidates in response"

    except Exception as e:
        return None, str(e)


def _call_huggingface(
    prompt: str,
    model: str,
    key: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call HuggingFace Inference API.

    Returns:
        Tuple of (content, error)
    """
    url = API_URLS["huggingface"].format(model=model)

    headers = {"Authorization": f"Bearer {key}"}

    payload = {"inputs": prompt}

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=getattr(config, 'API_TIMEOUT', 60)
        )

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:200]}"

        data = response.json()

        if isinstance(data, list) and data:
            content = data[0].get("generated_text", "").strip()
            return content, None

        return str(data), None

    except Exception as e:
        return None, str(e)


def _call_groq(
    prompt: str,
    model: str,
    key: str,
    max_tokens: int = 800,
    temperature: float = 0.7
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call Groq API.

    Returns:
        Tuple of (content, error)
    """
    url = API_URLS["groq"]

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=getattr(config, 'API_TIMEOUT', 30)
        )

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:200]}"

        data = response.json()

        choices = data.get("choices", [{}])
        if choices:
            content = choices[0].get("message", {}).get("content", "").strip()
            return content, None

        return None, "No choices in response"

    except Exception as e:
        return None, str(e)


def _call_deepseek(
    prompt: str,
    model: str,
    key: str,
    max_tokens: int = 800,
    temperature: float = 0.7
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call DeepSeek API.

    Returns:
        Tuple of (content, error)
    """
    url = API_URLS["deepseek"]

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=getattr(config, 'API_TIMEOUT', 30)
        )

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:200]}"

        data = response.json()

        choices = data.get("choices", [{}])
        if choices:
            content = choices[0].get("message", {}).get("content", "").strip()
            return content, None

        return None, "No choices in response"

    except Exception as e:
        return None, str(e)


def _call_cohere(
    prompt: str,
    model: str,
    key: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Call Cohere API.

    Returns:
        Tuple of (content, error)
    """
    url = API_URLS["cohere"]

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "message": prompt,
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=getattr(config, 'API_TIMEOUT', 30)
        )

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}: {response.text[:200]}"

        data = response.json()

        content = data.get("text", "").strip()
        return content, None

    except Exception as e:
        return None, str(e)


# ============================================================================
# MAIN API FUNCTION
# ============================================================================

# ---------------------------------------------------------------------------
# LOCAL / LLM from MODEL_CATALOG / MESH backends
# ---------------------------------------------------------------------------
def _call_local(prompt: str, model: Optional[str] = None, timeout: int = 8) -> Tuple[Optional[str], Optional[str]]:
    """Call the local SarahMemory API chat endpoint (POST /api/chat)."""
    url = API_URLS.get("local")
    if not url:
        return None, "LOCAL endpoint not configured"
    payload = {"text": prompt, "model": model or DEFAULT_MODELS.get("local")}
    try:
        r = requests.post(url, json=payload, timeout=timeout, verify=False)
        if r.status_code >= 400:
            return None, f"Local HTTP {r.status_code}: {r.text[:400]}"
        data = r.json() if r.content else {}
        return data.get("reply") or data.get("response") or data.get("text") or data.get("content"), None
    except Exception as e:
        return None, str(e)

# ---------------------------------------------------------------------------
# Local 3rd-party LLM runtime (MODEL_CATALOG routing)
# ---------------------------------------------------------------------------

_LOCAL_LLM_CACHE = {
    "repo": None,
    "model": None,
    "tokenizer": None,
    "device": None,
    "dtype": None,
}

def _normalize_local_repo_to_dir(repo: str) -> str:
    """
    Convert HF repo id -> local directory name convention used by SarahMemoryLLM:
      Qwen/Qwen3-0.6B -> Qwen_Qwen3-0.6B
    """
    return (repo or "").strip().replace("/", "_")

def _resolve_local_llm_repo(intent: str, user_text: str, meta: Optional[dict] = None) -> Tuple[Optional[str], list, str]:
    """
    Resolve best local 3rd-party *text generation* model repo using SarahMemoryGlobals MODEL_CATALOG + MODEL_CONFIG.

    Returns (selected_repo, fallbacks, source_lane).

    IMPORTANT:
    - This resolver is for LOCAL_LLM text generation only.
    - Embedding / vision / tts categories are explicitly rejected here so we do not accidentally
      route an encoder-only model (e.g., sentence-transformers, e5) into a CausalLM loader.
    """
    try:
        # Prefer the newer resolver if present
        category = None
        try:
            import SarahMemoryLLM as _SMLLM
            category = _SMLLM.infer_category_from_text(user_text or "")
        except Exception:
            category = None

        # Normalize / constrain categories: LOCAL_LLM lane is generation only.
        cat = (category or "").strip().lower()
        if cat in (
            "embedding", "embeddings", "semantic", "memory", "retrieval", "vector",
            "vision", "object", "object_detection",
            "image", "image_generation", "creative",
            "audio", "voice", "tts", "stt",
        ):
            category = None

        if not category:
            it = (intent or "").lower()
            if "code" in it or "coder" in it:
                category = "coder"
            else:
                category = "reasoning"

        if hasattr(config, "resolve_model") and callable(getattr(config, "resolve_model")):
            r = config.resolve_model(category, text=user_text or "", meta=meta or {}, models_dir=getattr(config, "MODELS_DIR", None))
            selected = r.get("selected")
            fallbacks = r.get("fallbacks") or []
            source = r.get("source") or "none"
            return selected, fallbacks, source
    except Exception:
        pass

    # Fallback: use stack primary repo
    try:
        if hasattr(config, "get_stack_primary_repo"):
            cat = "coder" if ("code" in (intent or "").lower() or "coder" in (intent or "").lower()) else "reasoning"
            sel = config.get_stack_primary_repo(cat, text=user_text or "", meta=meta or {})
            return sel, [], "fallback"
    except Exception:
        pass

    return None, [], "none"

def _load_local_llm(repo: str) -> Tuple[Optional[Any], Optional[Any], Optional[str]]:
    """
    Load (or reuse cached) local HF/Transformers *text generation* model from ./data/models/<repo_underscored>/.

    Notes:
    - This loader is for generation models only (CausalLM/Seq2SeqLM).
    - Encoder-only / SentenceTransformer directories are rejected with a clear message.
    """
    if not repo:
        return None, None, "No repo selected"

    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        return None, None, "Local LLM runtime not available (missing torch/transformers)"

    # Resolve local path
    models_dir = getattr(config, "MODELS_DIR", None)
    if not models_dir:
        try:
            base = getattr(config, "BASE_DIR", os.getcwd())
            models_dir = os.path.join(base, "data", "models")
        except Exception:
            models_dir = os.path.join(os.getcwd(), "data", "models")

    local_dir = os.path.join(models_dir, _normalize_local_repo_to_dir(repo))

    if not os.path.isdir(local_dir):
        return None, None, f"Local model directory not found: {local_dir}"

    # Reuse cache if same repo
    if (
        _LOCAL_LLM_CACHE.get("repo") == repo
        and _LOCAL_LLM_CACHE.get("model") is not None
        and _LOCAL_LLM_CACHE.get("tokenizer") is not None
    ):
        return _LOCAL_LLM_CACHE["model"], _LOCAL_LLM_CACHE["tokenizer"], None

    # SentenceTransformer-style directories are *not* generation models.
    # Common indicator: modules.json in the root.
    st_modules_path = os.path.join(local_dir, "modules.json")
    if os.path.isfile(st_modules_path):
        return None, None, (
            f"Local repo '{repo}' appears to be a SentenceTransformer/embedding model (modules.json present). "
            f"It cannot be used as a text-generation LLM. Download/select a chat/instruct model from MODEL_CATALOG['reasoning'/'coder']."
        )

    # Decide device/dtype
    device = "cuda" if (torch and torch.cuda.is_available()) else "cpu"
    dtype = torch.float16 if (device == "cuda") else torch.float32

    # Inspect config when possible to block encoder-only models and optionally support seq2seq.
    model_kind = "causal"
    try:
        if AutoConfig is not None:
            cfg = AutoConfig.from_pretrained(local_dir, local_files_only=True, trust_remote_code=True)
            mt = str(getattr(cfg, "model_type", "") or "").lower()
            # Block common encoder-only types (not suitable for generation via CausalLM)
            if mt in {
                "bert", "roberta", "mpnet", "distilbert", "electra", "xlnet",
                "deberta", "deberta-v2", "albert", "camembert", "flaubert",
                "sentence-transformers",
            }:
                return None, None, (
                    f"Local repo '{repo}' is an encoder/embedding model (model_type='{mt}'). "
                    f"It cannot be used for chat text generation. Select a causal/instruct model (e.g., Qwen/Phi/Mistral)."
                )

            # Seq2Seq models can generate (T5/BART/etc). Support when available.
            if bool(getattr(cfg, "is_encoder_decoder", False)) and AutoModelForSeq2SeqLM is not None:
                model_kind = "seq2seq"
    except Exception:
        # If config can't be read, we'll attempt to load and catch errors.
        model_kind = "causal"

    try:
        # Tokenizer: try fast first, then fall back.
        try:
            tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True, use_fast=True, trust_remote_code=True)
        except Exception:
            tok = AutoTokenizer.from_pretrained(local_dir, local_files_only=True, use_fast=False, trust_remote_code=True)

        # Some models need pad token
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token

        if model_kind == "seq2seq" and AutoModelForSeq2SeqLM is not None:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                local_dir,
                local_files_only=True,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                local_dir,
                local_files_only=True,
                torch_dtype=dtype,
                trust_remote_code=True,
            )

        # Move to device explicitly (avoid accelerate/device_map dependency).
        try:
            model.to(device)
        except Exception:
            # Some models already place themselves; ignore best-effort
            pass

        model.eval()

        _LOCAL_LLM_CACHE.update({
            "repo": repo,
            "model": model,
            "tokenizer": tok,
            "device": device,
            "dtype": str(dtype),
            "model_kind": model_kind,
        })

        return model, tok, None
    except Exception as e:
        # Improve common confusion: trying to load embedding models as LLMs.
        msg = str(e)
        if "Unrecognized model" in msg or "model_type" in msg:
            return None, None, (
                f"Failed to load local model '{repo}' as a text-generation LLM. "
                f"This usually means the directory is not a HF generation model (often a SentenceTransformer/embedding export) "
                f"or the download is incomplete. Under: {local_dir}. Error: {msg}"
            )
        return None, None, f"Failed to load local model ({repo}): {e}"

def _call_local_llm(
    prompt: str,
    model: Optional[str] = None,
    timeout: int = 30,
    *,
    intent: str = "chat",
    user_text: Optional[str] = None,
    meta: Optional[dict] = None,
    max_tokens: int = 512,
    temperature: float = 0.7
) -> Tuple[Optional[str], Optional[str]]:
    """Execute local inference via Transformers using the Auto/Manual selected repo from MODEL_CATALOG.

    Compatibility:
      - Legacy callers may call: _call_local_llm(prompt, model=None, timeout=30)
      - Newer callers may pass: intent=..., user_text=..., meta=..., max_tokens=..., temperature=...

    Notes:
      - 'timeout' is accepted for compatibility but is not currently used (local inference is in-process).
      - If 'model' is provided, it is treated as a forced repo/alias (bypassing auto selection).
    """
    # Normalize inputs
    user_text = (user_text if user_text is not None else prompt) or ""
    meta = meta or {}

    tried: list[str] = []
    last_err: Optional[str] = None

    # Optional forced repo/alias override
    forced_repo: Optional[str] = None
    if model:
        forced_repo = str(model).strip()
        try:
            if hasattr(config, "resolve_model_repo") and callable(getattr(config, "resolve_model_repo")):
                forced_repo = config.resolve_model_repo(forced_repo) or forced_repo
            else:
                forced_repo = (getattr(config, "MODEL_REPO_MAP", {}) or {}).get(forced_repo, forced_repo)
        except Exception:
            pass

    if forced_repo:
        selected, fallbacks, source_lane = forced_repo, [], "forced"
    else:
        selected, fallbacks, source_lane = _resolve_local_llm_repo(intent=intent, user_text=user_text, meta=meta)

    for repo in [selected, *(fallbacks or [])]:
        if not repo:
            continue
        tried.append(str(repo))

        llm, tok, err = _load_local_llm(str(repo))
        if err:
            last_err = err
            continue

        try:
            # Build a minimal chat-style prompt if tokenizer supports it
            messages = [
                {"role": "system", "content": "You are SarahMemory. Answer clearly and helpfully."},
                {"role": "user", "content": prompt},
            ]

            if hasattr(tok, "apply_chat_template"):
                text_in = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                text_in = f"System: You are SarahMemory.\nUser: {prompt}\nAssistant:"

            inputs = tok(text_in, return_tensors="pt", truncation=True, max_length=4096)

            if torch and torch.cuda.is_available():
                try:
                    dev = getattr(llm, "device", None)
                    if dev is None:
                        dev = next(llm.parameters()).device
                    inputs = {k: v.to(dev) for k, v in inputs.items()}
                except Exception:
                    pass

            gen_kwargs = {
                "max_new_tokens": int(max_tokens),
                "do_sample": float(temperature) > 0.0,
                "temperature": float(max(0.0, temperature)),
                "top_p": 0.95,
                "eos_token_id": tok.eos_token_id,
                "pad_token_id": (tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id),
            }

            with torch.inference_mode():
                out = llm.generate(**inputs, **gen_kwargs)

            decoded = tok.decode(out[0], skip_special_tokens=True)

            # Best-effort: strip the input prefix (handles tokenizers that don't reproduce exact prefix)
            try:
                if decoded.startswith(text_in):
                    decoded = decoded[len(text_in):].strip()
                else:
                    idx = decoded.find(text_in)
                    if idx != -1:
                        decoded = decoded[idx + len(text_in):].strip()
            except Exception:
                pass

            # If an Assistant delimiter exists, keep the last segment
            if "Assistant:" in decoded:
                decoded = decoded.split("Assistant:")[-1].strip()

            if decoded:
                return _sm_sanitize_llm_text(decoded), None
        except Exception as e:
            last_err = str(e)
            continue

    return None, (
        f"Local LLM failed. Tried: {tried or ['<none>']} "
        f"(selection={selected}, lane={source_lane}) | last_error={last_err}"
    )


def _call_mesh(prompt: str, provider: Optional[str] = None, model: Optional[str] = None, timeout: int = 45) -> Tuple[Optional[str], Optional[str]]:
    """Call the distributed mesh tier (cloud broker)."""
    url = API_URLS.get("mesh")
    if not url:
        return None, "MESH endpoint not configured"
    payload = {"prompt": prompt, "provider": provider, "model": model}
    headers = {"Content-Type": "application/json"}
    key = API_KEYS.get("mesh")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if r.status_code >= 400:
            return None, f"Mesh HTTP {r.status_code}: {r.text[:400]}"
        data = r.json() if r.content else {}
        return data.get("reply") or data.get("text") or data.get("content"), None
    except Exception as e:
        return None, str(e)

def _sm_is_selfaware_body_query(text: str) -> bool:
    t = str(text or "").strip().lower()
    if not t:
        return False
    hardware_terms = (
        "cpu", "processor", "gpu", "graphics", "motherboard", "mainboard", "baseboard",
        "ram", "memory", "disk", "drive", "storage", "network adapter", "wifi", "wi-fi",
        "ethernet", "temperature", "temp", "fan", "rpm", "sata", "usb", "nvme", "pcie",
    )
    self_scope_terms = ("your", "you", "system", "runtime", "body map", "body-map", "computer", "machine", "pc")
    return any(k in t for k in hardware_terms) and any(k in t for k in self_scope_terms)


def send_to_api(
    user_input: str,
    provider: str = "auto",
    intent: str = "question",
    tone: str = "friendly",
    complexity: str = "adult",
    model: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Send a request to an AI API provider.

    This is the main entry point for all API calls. It handles:
    - Provider selection and fallback
    - Offline/safe mode detection
    - Prompt building with context
    - Response caching
    - Error handling and logging

    Args:
        user_input: The user's input text
        provider: API provider name (openai, claude, mistral, etc.)
        intent: Classified intent for role selection
        tone: Desired response tone
        complexity: Response complexity level
        model: Specific model to use (optional)
        **kwargs: Additional parameters (max_tokens, temperature, etc.)

    Returns:
        Dictionary with response data:
        {
            "source": str,           # Provider name
            "data": str or None,     # Response content
            "model_used": str,       # Model that generated response
            "prompt": str,           # Prompt sent to API
            "intent": str,           # Intent used
            "tone": str,             # Tone used
            "emotion": dict,         # Emotional state
            "cached": bool,          # Whether response was cached
            "error": str or None     # Error message if failed
        }
    """
    start_time = time.time()

    # V10/V9C: live body-map questions must never be answered by a helper model.
    # They require appself/SelfAware Evidence Court so the model cannot invent CPU/GPU/etc.
    classification_packet = kwargs.get("chat_classification_packet") or kwargs.get("classification_packet") or {}
    if (isinstance(classification_packet, dict) and str(classification_packet.get("domain") or "") == "selfaware_body") or _sm_is_selfaware_body_query(user_input):
        return {
            "source": "api_blocked_selfaware_body",
            "data": None,
            "model_used": None,
            "prompt": user_input,
            "intent": "selfaware_body",
            "tone": tone,
            "emotion": {},
            "cached": False,
            "error": "Live hardware/body-map facts require SelfAware Evidence Court, not model generation.",
            "do_not_write_sql": True,
            "do_not_persist": True,
            "volatile_runtime_fact": True,
        }

    # Detect run mode and offline status
    try:
        run_mode = str(getattr(config, "RUN_MODE", os.getenv("RUN_MODE", ""))).lower()

        offline = False
        if run_mode not in ("cloud", "server") and callable(is_offline):
            try:
                offline = is_offline()
            except Exception:
                offline = False
    except Exception:
        run_mode = ""
        offline = False

    # Log debug info
    logger.debug(
        f"[API] provider={provider}, SAFE_MODE={SAFE_MODE}, "
        f"LOCAL_ONLY_MODE={LOCAL_ONLY_MODE}, run_mode={run_mode}, offline={offline}"
    )


    # Check safety / offline modes
    if SAFE_MODE or LOCAL_ONLY_MODE or (offline and run_mode not in ("cloud", "server")):
        # In local-only / safe / offline modes we MUST NOT call external providers (including mesh).
        # We will reroute to local lanes if possible; otherwise return a clear error.
        requested = (provider or "").strip().lower()
        if requested in ("", "unknown", "auto", "primary", "default"):
            # Prefer LOCAL_API first (DB/tool/retrieval), then LOCAL_LLM (3rd-party model runtime).
            if _provider_is_enabled("local"):
                provider = "local"
            elif _provider_is_enabled("local_llm"):
                provider = "local_llm"
            else:
                return {
                    "source": requested or "auto",
                    "data": None,
                    "prompt": None,
                    "intent": intent,
                    "error": "Blocked by safety/local-only mode (no local providers enabled)"
                }
        else:
            # If caller explicitly asked for an external provider, force-reroute to local.
            if requested not in ("local", "local_llm"):
                if _provider_is_enabled("local_llm"):
                    provider = "local_llm"
                elif _provider_is_enabled("local"):
                    provider = "local"
                else:
                    return {
                        "source": requested,
                        "data": None,
                        "prompt": None,
                        "intent": intent,
                        "error": "Blocked by safety/local-only mode (no local providers enabled)"
                    }
            else:
                provider = requested



    # Check if API research is disabled
    # NOTE: This flag is intended to gate *external* research/API calls. Local lanes are always allowed.
    if not getattr(config, 'API_RESEARCH_ENABLED', True):
        req = (provider or "").strip().lower()
        if req in ("openai","claude","anthropic","mistral","gemini","huggingface","deepseek","groq","cohere","mesh"):
            logger.warning("[BLOCKED] API research disabled in Globals (external provider blocked).")
            return {
                "source": req,
                "data": None,
                "prompt": None,
                "intent": intent,
                "error": "API research disabled (external providers blocked)"
            }


    # Auto-select provider if not specified
    if (provider is None) or (str(provider).strip().lower() in ("", "unknown", "auto", "primary", "default")):
        try:
            selected = get_active_api(
                getattr(config, "PRIMARY_API", None),
                getattr(config, "API_FALLBACKS", None),
            )
            if selected:
                provider = selected
            else:
                provider = get_best_provider_for_intent(intent)
        except Exception:
            provider = get_best_provider_for_intent(intent)

    # Normalize provider name
    provider = provider.lower()
    if provider == "anthropic":
        provider = "claude"

    # Normalize legacy/local provider aliases
    if provider in ("ollama", "local_llm", "model_catalog"):
        provider = "local_llm"



    # Enforce provider flags from SarahMemoryGlobals (hard gate)
    # If OpenAI is disabled, NEVER allow an OpenAI call even if an API key exists.
    if provider == "openai" and not bool(getattr(config, "OPEN_AI_API", False)):
        if bool(getattr(config, "LOCAL_LLM_API", False)):
            provider = "local_llm"
        elif bool(getattr(config, "LOCAL_API", False)):
            provider = "local"
        elif bool(getattr(config, "MESH_API", False)):
            provider = "mesh"
        # else: keep as openai, but it will fail the enabled check below

    # If chosen provider is disabled, pick best enabled provider for intent.
    if not _provider_is_enabled(provider):
        provider = get_best_provider_for_intent(intent)

    # Get API key
    key = API_KEYS.get(provider)
    if _provider_requires_key(provider) and not key:
        # If missing credentials, try fallback rather than hard-fail.
        fb = fallback_provider(provider)
        if fb:
            logger.warning(f"[Fallback Triggered] Missing credentials for {provider}; switching to {fb}")
            return send_to_api(
                user_input,
                provider=fb,
                intent=intent,
                tone=tone,
                complexity=complexity,
                model=model,
                **kwargs
            )
        return {
            "source": provider,
            "data": None,
            "error": f"API key missing for {provider}"
        }


    # Get model
    model = model or DEFAULT_MODELS.get(provider, "gpt-4o")

    # Build prompt
    prompt, emotion = build_advanced_prompt(user_input, intent, tone, complexity, provider)

    # Check cache
    with _cache_lock:
        if prompt in _response_cache:
            logger.debug(f"[CACHE HIT] Using cached response for prompt")
            return {
                "source": provider,
                "data": _response_cache[prompt],
                "model_used": model,
                "prompt": prompt,
                "intent": intent,
                "tone": tone,
                "emotion": emotion,
                "cached": True
            }

    # Get parameters
    max_tokens = int(kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or
                     getattr(config, "API_MAX_TOKENS", 800))
    temperature = float(kwargs.get("temperature") or getattr(config, "OPENAI_TEMPERATURE", 1.0))

    # Call appropriate provider
    content = None
    error = None

    try:
        if provider == "openai":
            content, error = _call_openai(prompt, model, key, max_tokens, temperature)
        elif provider in ("claude", "anthropic"):
            content, error = _call_claude(prompt, model, key, max_tokens)
        elif provider == "mistral":
            content, error = _call_mistral(prompt, model, key, max_tokens, temperature)
        elif provider == "gemini":
            content, error = _call_gemini(prompt, model, key)
        elif provider == "huggingface":
            content, error = _call_huggingface(prompt, model, key)
        elif provider == "groq":
            content, error = _call_groq(prompt, model, key, max_tokens, temperature)
        elif provider == "deepseek":
            content, error = _call_deepseek(prompt, model, key, max_tokens, temperature)
        elif provider == "cohere":
            content, error = _call_cohere(prompt, model, key)
        elif provider == "local":
            content, error = _call_local(prompt, model)
        elif provider in ("local_llm", "localmodels", "local_llm_api"):
            content, error = _call_local_llm(prompt=prompt, intent=intent, user_text=user_input, meta=kwargs.get('meta'), max_tokens=max_tokens, temperature=temperature)
        elif provider == "mesh":
            content, error = _call_mesh(prompt, provider=None, model=model)
        else:
            error = f"Unknown provider: {provider}"
    except Exception as e:
        error = str(e)

    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000

    # Handle success
    if content and not error:
        content = _sm_sanitize_llm_text(content)
        # Cache response
        with _cache_lock:
            _response_cache[prompt] = content

        # Log success
        log_api_event(
            f"{provider.upper()} API Success",
            f"Prompt: {prompt[:80]} | Response: {content[:80]}",
            provider=provider,
            model=model,
            latency_ms=latency_ms
        )

        return {
            "source": provider,
            "data": content,
            "model_used": model,
            "prompt": prompt,
            "intent": intent,
            "tone": tone,
            "emotion": emotion,
            "cached": False,
            "latency_ms": latency_ms
        }

    # Handle failure - try fallback
    logger.error(f"[{provider.upper()} API Exception] {error}")

    # Try fallback provider
    fallback = fallback_provider(provider)
    if fallback:
        logger.warning(f"[Fallback Triggered] Switching from {provider} to {fallback}")
        return send_to_api(
            user_input,
            provider=fallback,
            intent=intent,
            tone=tone,
            complexity=complexity,
            **kwargs
        )

    # No fallback available
    log_api_event(
        f"{provider.upper()} API Failed",
        f"Error: {error}",
        provider=provider,
        model=model,
        latency_ms=latency_ms
    )

    research_path_logger.info(
        f"SarahMemoryAPI.py -> send_to_api, Input: {user_input[:50]}, "
        f"Provider: {provider}, Intent: {intent}, Error: {error}"
    )

    return {
        "source": provider,
        "data": None,
        "error": error or "API request failed",
        "intent": intent
    }


def send_to_api_async(
    user_input: str,
    provider: str,
    callback: Callable[[Dict], None]
) -> None:
    """
    Send API request asynchronously.

    Args:
        user_input: User's input text
        provider: API provider
        callback: Function to call with result
    """
    def async_task():
        result = send_to_api(user_input, provider)
        callback(result)

    run_async(async_task)


# ============================================================================
# COGNITIVE ANALYSIS
# ============================================================================

def run_cognitive_analysis(text: str, provider: str = "openai") -> Dict[str, Any]:
    """
    Run cognitive analysis (emotion, tone, sentiment) on text.

    Args:
        text: Text to analyze
        provider: API provider to use

    Returns:
        Analysis results dictionary
    """
    try:
        if provider == "openai":
            key = API_KEYS.get("openai")
            if not key:
                return {"error": "OpenAI API key missing"}

            prompt = f"""Analyze the following text and provide:
1. Primary emotion (e.g., happy, sad, angry, fearful, surprised, neutral)
2. Tone (e.g., formal, casual, aggressive, friendly)
3. Sentiment (positive, negative, neutral) with confidence score

Text: {text}

Respond in JSON format:
{{"emotion": "...", "tone": "...", "sentiment": "...", "confidence": 0.0}}"""

            content, error = _call_openai(
                prompt,
                "gpt-4o-mini",
                key,
                max_tokens=200,
                temperature=0.3
            )

            if error:
                return {"error": error}

            try:
                result = json.loads(content)
            except:
                result = {"raw": content}

            log_cognitive_event("openai", text[:200], str(result))

            return {"source": "openai", "result": result}

        elif provider == "claude":
            return {"error": "Claude cognitive analysis not yet implemented."}

        elif provider == "gemini":
            return {"error": "Gemini cognitive analysis not yet implemented."}

        else:
            return {"error": f"Unknown provider '{provider}'"}

    except Exception as e:
        logger.error(f"[Cognitive Error:{provider}] {e}")
        return {"error": str(e)}


def analyze_image(image_path: str) -> Dict[str, Any]:
    """
    Analyze an image using vision-capable AI.

    Args:
        image_path: Path to image file

    Returns:
        Analysis results
    """
    logger.warning("Image analysis via vision models not yet fully implemented.")
    return {"error": "Image analysis not implemented."}


# ============================================================================
# MODEL SELECTION HELPERS
# ============================================================================

def _candidate_models_for_intent(intent: str = "chat") -> List[str]:
    """
    Get candidate models for a given intent.

    Args:
        intent: The classified intent

    Returns:
        List of model names in priority order
    """
    try:
        picks = []

        # Add primary and secondary models
        for m in (
            getattr(config, 'API_PRIMARY_MODEL', None),
            getattr(config, 'API_SECONDARY_MODEL', None)
        ):
            blocklist = getattr(config, 'API_BLOCKLIST_MODELS', [])
            if m and m not in picks and m not in blocklist:
                picks.append(m)

        # Add allowed models
        for m in getattr(config, 'API_ALLOWED_MODELS', []):
            blocklist = getattr(config, 'API_BLOCKLIST_MODELS', [])
            if m not in picks and m not in blocklist:
                picks.append(m)

        # Add default model
        dm = getattr(config, 'API_DEFAULT_MODEL', None)
        if dm and dm not in picks:
            picks.append(dm)

        return picks

    except Exception:
        return []


def get_available_providers() -> List[str]:
    """
    Get list of providers that are enabled and usable under SarahMemoryGlobals.

    Local lanes do not require API keys, but they must still be enabled. External
    lanes require both the Globals toggle and credentials. This prevents the UI
    and REM reports from showing a disabled/unavailable provider as selected.
    """
    return [p for p in PROVIDER_PRIORITY if _provider_has_credentials(p)]


def get_provider_status() -> Dict[str, Dict[str, Any]]:
    """
    Get status of all providers.

    Returns:
        Dictionary with provider status information
    """
    status = {}
    for provider in PROVIDER_PRIORITY:
        key = API_KEYS.get(provider)
        enabled = _provider_is_enabled(provider)
        available = _provider_has_credentials(provider)
        status[provider] = {
            "available": bool(available),
            "enabled": bool(enabled),
            "requires_key": bool(_provider_requires_key(provider)),
            "key_set": bool(key),
            "default_model": DEFAULT_MODELS.get(provider),
            "url": API_URLS.get(provider)
        }
    return status


def rem_api_capability_snapshot(snapshot: Optional[Dict[str, Any]] = None, include_test_call: bool = False) -> Dict[str, Any]:
    """Bounded REM API lane snapshot.

    Metadata-only by default. It reports the currently usable provider/model for
    the DL Engine and REM dashboards without making provider calls unless the
    caller explicitly asks for a test call.
    """
    started = time.time()
    providers = get_provider_status()
    selected_provider = next((p for p in PROVIDER_PRIORITY if providers.get(p, {}).get("available")), None)
    selected_model = DEFAULT_MODELS.get(selected_provider) if selected_provider else None

    # If no external/local provider is currently enabled, still report the SarahMemory
    # core runtime as an active capability. This prevents the DL Engine dashboard from
    # looking dead while the core AIOS is operating without third-party providers.
    if not selected_provider:
        selected_provider = "core"
        selected_model = "SarahMemory Core Runtime"

    out: Dict[str, Any] = {
        "ok": True,
        "lane": "api",
        "metadata_only": not bool(include_test_call),
        "providers": providers,
        "selected_provider": selected_provider,
        "selected_model": selected_model,
        "active_provider": selected_provider,
        "active_model": selected_model,
        "models_loaded": 1 if selected_provider else 0,
        "notes": [
            "REM API lane defaults to metadata-only; no provider calls are made unless include_test_call=True.",
            "Selected provider is the first enabled and credentialed provider in governed priority order; falls back to SarahMemory Core Runtime when no third-party provider is active."
        ],
    }
    if include_test_call and selected_provider:
        try:
            probe = send_to_api(
                "SarahMemory REM API health probe. Reply with OK only.",
                provider=selected_provider,
                intent="diagnostic",
                tone="technical",
                complexity="adult",
                max_tokens=8,
                temperature=0.0,
            )
            out["test_call"] = {
                "ok": bool(probe and probe.get("data")),
                "source": (probe or {}).get("source"),
                "model_used": (probe or {}).get("model_used"),
                "error": (probe or {}).get("error"),
            }
        except Exception as exc:
            out["test_call"] = {"ok": False, "error": str(exc)}
    out["duration_ms"] = int((time.time() - started) * 1000)
    return out


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ============================================================================

# Alias for legacy code
def send_to_openai(user_input: str, **kwargs) -> Dict[str, Any]:
    """Legacy alias.

    IMPORTANT:
    - If OPEN_AI_API is disabled in SarahMemoryGlobals, this MUST NOT call OpenAI.
    - In that case we route to the local 3rd-party model lane (provider='local_llm').
    """
    try:
        if not bool(getattr(config, "OPEN_AI_API", False)):
            return send_to_api(user_input, provider="local_llm", **kwargs)
    except Exception:
        return send_to_api(user_input, provider="local_llm", **kwargs)

    return send_to_api(user_input, provider="openai", **kwargs)


# Cache reference for backward compatibility
cache = _response_cache


# ============================================================================
# DATABASE SETUP
# ============================================================================

def _ensure_response_table(db_path: Optional[str] = None) -> None:
    """
    Ensure the response table exists.

    Args:
        db_path: Optional database path
    """
    try:
        if db_path is None:
            db_path = SYSTEM_LOGS_DB

        conn = sqlite3.connect(db_path, timeout=5.0)
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS response (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                user TEXT,
                content TEXT,
                source TEXT,
                intent TEXT,
                model TEXT
            )
        """)

        conn.commit()
        conn.close()

        logger.debug(f"[DB] Ensured table `response` in {db_path}")

    except Exception as e:
        logger.warning(f"[DB] ensure `response` failed: {e}")


# ============================================================================
# FLASK HEALTH ENDPOINT (AUTO-PATCH)
# ============================================================================

try:
    from flask import jsonify

    if "app" in globals():
        _sm_app = globals()["app"]

        @_sm_app.get("/health")
        def health():
            return jsonify({"ok": True}), 200

except Exception:
    pass


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Ensure database tables exist
try:
    _ensure_database()
    _ensure_response_table()
except Exception as e:
    logger.debug(f"[API] Database initialization: {e}")


# ============================================================================
# DEMO AND TESTING
# ============================================================================

def _demo() -> None:
    """Run demonstration of API capabilities."""
    print("=" * 70)
    print("SarahMemory API Module - Demo")
    print("=" * 70)

    # Show available providers
    print("\n[Available Providers]\n")
    for provider, status in get_provider_status().items():
        key_status = "✓ KEY SET" if status["key_set"] else "✗ NO KEY"
        print(f"  {provider}: {key_status} | Model: {status['default_model']}")

    # Show role categories
    print("\n[Role Categories]\n")
    for category, roles in ROLE_CATEGORIES.items():
        print(f"  {category.upper()}: {len(roles)} roles")

    # Show sample roles
    print("\n[Sample Roles]\n")
    sample_intents = ["question", "code", "medical", "creative", "finance"]
    for intent in sample_intents:
        role = get_role_for_intent(intent)
        print(f"  {intent}: {role[:60]}...")

    # Test prompt building
    print("\n[Prompt Building Test]\n")
    prompt, emotion = build_advanced_prompt(
        "How do I optimize a SQL query?",
        intent="database",
        tone="professional",
        complexity="expert"
    )
    print(f"  Prompt length: {len(prompt)} chars")
    print(f"  Emotion state: {emotion}")

    print("\n" + "=" * 70)
    print("Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    _demo()


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Main functions
    "send_to_api",
    "send_to_api_async",
    "send_to_openai",

    # Prompt building
    "build_advanced_prompt",
    "get_role_for_intent",

    # Provider management
    "fallback_provider",
    "get_best_provider_for_intent",
    "get_available_providers",
    "get_provider_status",
    "rem_api_capability_snapshot",

    # Cognitive analysis
    "run_cognitive_analysis",
    "analyze_image",

    # Logging
    "log_api_event",
    "log_cognitive_event",

    # Model selection
    "_candidate_models_for_intent",

    # Constants
    "ROLE_MAP",
    "ROLE_CATEGORIES",
    "API_KEYS",
    "API_URLS",
    "DEFAULT_MODELS",
    "PROVIDER_PRIORITY",

    # Data classes
    "APIRequest",
    "APIResponse",
    "APIProvider",

    # Cache
    "cache",
]

# ====================================================================
# END OF SarahMemoryAPI.py v8.0.0
# ===================================================================
# -----------------------------------------------------------------------------
# Output Sanitization (hide system prompts / chain-of-thought from UI & TTS)
# -----------------------------------------------------------------------------
_SANITIZE_RE_BLOCKS = [
    re.compile(r"(?is)<think>.*?</think>"),
    re.compile(r"(?is)<analysis>.*?</analysis>"),
    re.compile(r"(?is)<system>.*?</system>"),
]
_SANITIZE_RE_LINES = [
    re.compile(r"(?im)^\s*(system|user|assistant)\s*:\s*.*$"),
    re.compile(r"(?im)^\s*(system|user|assistant)\s*$"),
    re.compile(r"(?im)^\s*you are sarahmemory\b.*$"),
    re.compile(r"(?im)^\s*answer clearly and helpfully\.?$"),
    re.compile(r"(?im)^\s*answer the user directly\.?$"),
    re.compile(r"(?im)^\s*(role|tone|complexity|recent context|context|intent)\s*:\s*.*$"),
    re.compile(r"(?im)^\s*\[(system|user|assistant)\]\s*.*$"),
]
def _sm_sanitize_llm_text(text: str) -> str:
    """
    Remove internal prompt scaffolding and model 'thinking' from user-visible output.
    This is a best-effort filter; it does NOT change model reasoning, only display.
    """
    if text is None:
        return ""
    t = str(text)
    # drop common tag blocks
    for rx in _SANITIZE_RE_BLOCKS:
        t = rx.sub("", t)
    # If the model echoed a chat transcript, remove leading role lines.
    # We only remove these line types; keep remaining content.
    # First normalize newlines
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # Remove fenced "system/user/assistant" transcript lines anywhere
    for rx in _SANITIZE_RE_LINES:
        t = rx.sub("", t)
    # Remove common token markers
    t = re.sub(r"(?im)^\s*<\|?(system|user|assistant)\|?>\s*$", "", t)
    # Collapse excessive blank lines
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    # If still contains an 'Assistant:' delimiter, keep the last segment
    if "Assistant:" in t:
        t = t.split("Assistant:")[-1].strip()
    return t