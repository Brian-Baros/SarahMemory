"""--==The SarahMemory Project==--
File: SarahMemoryResearch.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-07-11
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

WORLD-CLASS ENTERPRISE RESEARCH ENGINE

This module provides comprehensive multi-source research capabilities for the
SarahMemory AiOS platform, integrating:

✓ LOCAL DATABASE RESEARCH
- QA Cache with semantic similarity scoring
- Personality response database
- Imported dataset full-text search
- Offline LLM ensemble reasoning
- Vector embedding-based retrieval
- Static fallback knowledge base

✓ WEB RESEARCH PIPELINE
- Wikipedia API with structured extraction
- DuckDuckGo HTML parsing with anti-CAPTCHA
- Free Dictionary API integration
- OpenLibrary book/author search
- StackOverflow (when enabled)
- Reddit, WikiHow, Quora (configurable)
- Internet Archive (configurable)
- Intelligent query preprocessing
- Multi-source aggregation

✓ API RESEARCH LAYER
- OpenAI GPT-4/3.5 integration
- Claude Anthropic models
- Mistral AI models
- Google Gemini models
- HuggingFace models
- Automatic provider fallback chain
- Intent-based model selection
- Cost-tier optimization

✓ ADVANCED FEATURES
- Parallel multi-source execution
- Response caching with TTL
- Query intent classification
- Source credibility scoring
- Result deduplication
- Answer synthesis and summarization
- Research path logging
- Performance metrics tracking
- Graceful degradation (offline mode)
- Registry integration (Windows)

✓ INTEGRATION WITH SARAHMEMORY ECOSYSTEM
- SarahMemoryReply.py: Primary consumer for response generation
- SarahMemoryWebSYM.py: Mathematical and symbolic synthesis
- SarahMemoryAPI.py: External LLM provider routing
- SarahMemoryDatabase.py: Persistent storage layer
- SarahMemoryAdvCU.py: Intent classification
- SarahMemoryAiFunctions.py: Context management
- SarahMemoryPersonality.py: Emotional tone integration

DESIGN PRINCIPLES:
------------------
1. MULTI-TIER FALLBACK: Local → Web → API → Static
2. PERFORMANCE FIRST: Cached results, parallel execution
3. OFFLINE CAPABLE: Works without internet connection
4. TRANSPARENT LOGGING: Full research path traceability
5. CONFIGURABLE: All sources toggleable via Globals
6. ENTERPRISE READY: Production-grade error handling
7. BACKWARD COMPATIBLE: All existing functions preserved

===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "B"
# ROLE = "research_engine"
# CATEGORY = "research_and_retrieval"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "research"
# HARDWARE_DOMAIN = "network_optional"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "research"
# FAMILY = "knowledge"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-07-11"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Multi-tier research engine with local datasets/cache, web research, API research, caching, synthesis, and graceful offline degradation."
# --- SARAHMETA END ---

import asyncio
try:
    import aiohttp  # type: ignore
except Exception:
    aiohttp = None  # type: ignore
import json
import os
import re
import time
import hashlib
import html
import logging
import sqlite3
import threading
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore
# import data.mods.v800.sm_v800_research_rssfeedread  # noqa

# =======================================================================
# WINDOWS REGISTRY (OPTIONAL - GRACEFUL DEGRADATION)
# ============================================================================
try:
    import winreg  # type: ignore
    WINREG_AVAILABLE = True
except Exception:
    winreg = None  # type: ignore
    WINREG_AVAILABLE = False

# ============================================================================
# SARAHMEMORY CORE IMPORTS
# ============================================================================
try:
    import SarahMemoryGlobals as config
except ImportError:
    # Minimal config fallback for headless/offline environments
    class config:
        BASE_DIR = os.getcwd()
        DATASETS_DIR = os.path.join(BASE_DIR, "data", "datasets")
        MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
        LOGS_DIR = os.path.join(BASE_DIR, "data", "logs")
        LOCAL_DATA_ENABLED = True
        LOCAL_ONLY_MODE = True
        WEB_RESEARCH_ENABLED = False
        API_RESEARCH_ENABLED = False
        DUCKDUCKGO_RESEARCH_ENABLED = False
        WIKIPEDIA_RESEARCH_ENABLED = True
        FREE_DICTIONARY_RESEARCH_ENABLED = False
        OPENLIBRARY_RESEARCH_ENABLED = False
        STACKOVERFLOW_RESEARCH_ENABLED = False
        REDDIT_RESEARCH_ENABLED = False
        WIKIHOW_RESEARCH_ENABLED = False
        QUORA_RESEARCH_ENABLED = False
        INTERNET_ARCHIVE_RESEARCH_ENABLED = False
        OPEN_AI_API = False
        CLAUDE_API = False
        MISTRAL_API = False
        GEMINI_API = False
        HUGGINGFACE_API = False
        MODEL_CONFIG = {}

from SarahMemoryAdvCU import classify_intent
from SarahMemoryAiFunctions import get_context, add_to_context
from SarahMemoryAPI import send_to_api
try:
    from SarahMemoryDatabase import (
        search_answers,
        search_responses,
        vector_search,
        vector_search_qa_cache,
        search_local_dataset_pool,
    )
except Exception:
    from SarahMemoryDatabase import search_answers, search_responses, vector_search, vector_search_qa_cache
    search_local_dataset_pool = None  # type: ignore
from SarahMemoryGlobals import import_other_data, MODEL_CONFIG
from SarahMemoryWebSYM import WebSemanticSynthesizer

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

# Main System Logger
logger = logging.getLogger("SarahMemoryResearch")
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Research Path Logger (separate detailed log for query tracing)
debug_log_path = os.path.join(config.BASE_DIR, "data", "logs", "research.log")
os.makedirs(os.path.dirname(debug_log_path), exist_ok=True)

research_debug_logger = logging.getLogger("ResearchDebug")
research_debug_logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(debug_log_path, mode='a', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
if not research_debug_logger.hasHandlers():
    research_debug_logger.addHandler(file_handler)

# Research Path Logger for GUI integration
research_path_logger = logging.getLogger('ResearchPathLogger')

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

# Web API Endpoints
WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
FREE_DICTIONARY_API = "https://www.thefreedictionary.com/"
OPENLIBRARY_API = "https://openlibrary.org/search.json?q="
DUCKDUCKGO_HTML = "https://duckduckgo.com/html/?q="
STACKOVERFLOW_API = "https://api.stackexchange.com/2.3/search?order=desc&sort=relevance&site=stackoverflow&intitle="
REDDIT_API = "https://www.reddit.com/search.json?q="
WIKIHOW_API = "https://www.wikihow.com/api.php?action=query&list=search&srsearch="
QUORA_BASE = "https://www.quora.com/search?q="
INTERNET_ARCHIVE_API = "https://archive.org/advancedsearch.php?q="

# Cache Configuration
CACHE_TTL_SECONDS = 3600  # 1 hour default
MAX_CACHE_SIZE = 10000

# Request Timeouts (seconds)
HTTP_TIMEOUT_SHORT = 10
HTTP_TIMEOUT_MEDIUM = 20
HTTP_TIMEOUT_LONG = 30

# Parallel Execution Limits
MAX_CONCURRENT_WEB_REQUESTS = 5
MAX_CONCURRENT_API_REQUESTS = 3

# ============================================================================
# ENHANCED STATIC KNOWLEDGE BASE
# ============================================================================

STATIC_FACTS = {
    # Physics & Constants
    "what is the speed of light": "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 186,282 miles per second).",
    "what is planck's constant": "Planck's constant (h) is approximately 6.62607015 × 10⁻³⁴ joule-seconds, a fundamental constant in quantum mechanics.",
    "what is the gravitational constant": "The gravitational constant (G) is approximately 6.674 × 10⁻¹¹ N⋅m²/kg², governing gravitational attraction.",
    "what is avogadro's number": "Avogadro's number is approximately 6.022 × 10²³, representing the number of atoms or molecules in one mole.",
    
    # Chemistry & Matter
    "what is the boiling point of water": "The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure (1 atm).",
    "what is the freezing point of water": "The freezing point of water is 0 degrees Celsius (32 degrees Fahrenheit) at standard atmospheric pressure.",
    "what is the periodic table": "The periodic table organizes all known chemical elements by atomic number, electron configuration, and recurring chemical properties.",
    
    # Mathematics
    "what is pi": "Pi (π) is a mathematical constant approximately equal to 3.14159265359, representing the ratio of a circle's circumference to its diameter.",
    "what is euler's number": "Euler's number (e) is approximately 2.71828182846, the base of natural logarithms and fundamental to exponential growth.",
    "what is the golden ratio": "The golden ratio (φ) is approximately 1.61803398875, appearing frequently in nature, art, and architecture.",
    "what is the fibonacci sequence": "The Fibonacci sequence (0, 1, 1, 2, 3, 5, 8, 13, ...) is generated by adding the two previous numbers.",
    
    # Computer Science
    "what is binary": "Binary is a base-2 number system using only 0 and 1, fundamental to all digital computing systems.",
    "what is an algorithm": "An algorithm is a step-by-step procedure or formula for solving a problem or performing a computation.",
    "what is machine learning": "Machine learning is a subset of artificial intelligence where computers learn from data without explicit programming.",
    
    # Astronomy
    "what is the distance to the sun": "The average distance from Earth to the Sun is about 149.6 million kilometers (93 million miles), defined as one Astronomical Unit (AU).",
    "what is a light year": "A light-year is the distance light travels in one year, approximately 9.461 trillion kilometers (5.879 trillion miles).",
    "how many planets are in our solar system": "Our solar system has 8 officially recognized planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
    
    # Earth Science
    "what is the earth's radius": "Earth's mean radius is approximately 6,371 kilometers (3,959 miles).",
    "what is the earth's circumference": "Earth's equatorial circumference is approximately 40,075 kilometers (24,901 miles).",
    "what is the deepest ocean": "The Pacific Ocean is the deepest, with the Mariana Trench reaching approximately 11,034 meters (36,201 feet) at its deepest point.",
    
    # Biology
    "what is dna": "DNA (deoxyribonucleic acid) is the molecule that carries genetic instructions for life, consisting of two strands forming a double helix.",
    "what is photosynthesis": "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen.",
    "how many chromosomes do humans have": "Humans have 46 chromosomes (23 pairs) in most cells, with sex cells (gametes) having 23 chromosomes.",

    # General local-first fallback examples. These are stable, non-network facts
    # used only after local QA/personality/dataset lookup fails or is quarantined.
    "who is elon musk": "Elon Musk is a businessman and entrepreneur best known for his involvement with companies such as Tesla, SpaceX, X, Neuralink, The Boring Company, and xAI. For current roles, dates, or financial details, SarahMemory should verify with a current source when Web/API research is explicitly enabled.",
    "how do you make a sandwich": "To make a basic sandwich: choose bread, add a spread if wanted, add a filling such as meat, cheese, eggs, peanut butter, or vegetables, add toppings like lettuce or tomato, season lightly, close the sandwich, then cut and serve it.",
    "how to make a sandwich": "To make a basic sandwich: choose bread, add a spread if wanted, add a filling such as meat, cheese, eggs, peanut butter, or vegetables, add toppings like lettuce or tomato, season lightly, close the sandwich, then cut and serve it.",
    "how do you make beef stew": "To make basic beef stew: cut beef into chunks, season it, brown it in a pot, add onion, garlic, carrots, potatoes, celery, broth, and herbs, then simmer gently until the beef is tender. Thicken if needed and adjust seasoning before serving.",
    "how to make beef stew": "To make basic beef stew: cut beef into chunks, season it, brown it in a pot, add onion, garlic, carrots, potatoes, celery, broth, and herbs, then simmer gently until the beef is tender. Thicken if needed and adjust seasoning before serving.",
}

# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ResearchSource(Enum):
    """Enumeration of research data sources."""
    LOCAL_QA_CACHE = "local_qa_cache"
    LOCAL_PERSONALITY = "local_personality"
    LOCAL_DATASETS = "local_datasets"
    LOCAL_LLM_ENSEMBLE = "local_llm_ensemble"
    LOCAL_WEBSYM = "local_websym"
    LOCAL_STATIC = "local_static"
    WEB_WIKIPEDIA = "web_wikipedia"
    WEB_DUCKDUCKGO = "web_duckduckgo"
    WEB_DICTIONARY = "web_dictionary"
    WEB_OPENLIBRARY = "web_openlibrary"
    WEB_STACKOVERFLOW = "web_stackoverflow"
    WEB_REDDIT = "web_reddit"
    WEB_WIKIHOW = "web_wikihow"
    WEB_QUORA = "web_quora"
    WEB_ARCHIVE = "web_archive"
    API_OPENAI = "api_openai"
    API_CLAUDE = "api_claude"
    API_MISTRAL = "api_mistral"
    API_GEMINI = "api_gemini"
    API_HUGGINGFACE = "api_huggingface"
    NONE = "none"


@dataclass
class ResearchResult:
    """Structured research result with metadata."""
    source: ResearchSource
    content: str
    confidence: float = 0.0
    query: str = ""
    intent: str = "unknown"
    timestamp: float = field(default_factory=time.time)
    latency_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for backward compatibility."""
        return {
            "source": self.source.value if isinstance(self.source, ResearchSource) else self.source,
            "intent": self.intent,
            "data": self.content,
            "snippet": self.content,  # Backward compatibility
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata
        }


@dataclass
class ResearchMetrics:
    """Performance metrics for research operations."""
    total_queries: int = 0
    cache_hits: int = 0
    local_successes: int = 0
    web_successes: int = 0
    api_successes: int = 0
    failures: int = 0
    avg_latency_ms: float = 0.0
    last_reset: float = field(default_factory=time.time)
    
    def record_success(self, source: ResearchSource, latency_ms: int):
        """Record successful query."""
        self.total_queries += 1
        if source.value.startswith("local"):
            self.local_successes += 1
        elif source.value.startswith("web"):
            self.web_successes += 1
        elif source.value.startswith("api"):
            self.api_successes += 1
        
        # Update average latency (exponential moving average)
        alpha = 0.3
        self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms
    
    def record_cache_hit(self):
        """Record cache hit."""
        self.cache_hits += 1
    
    def record_failure(self):
        """Record failure."""
        self.failures += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(self.total_queries, 1),
            "local_successes": self.local_successes,
            "web_successes": self.web_successes,
            "api_successes": self.api_successes,
            "failures": self.failures,
            "success_rate": (self.total_queries - self.failures) / max(self.total_queries, 1),
            "avg_latency_ms": self.avg_latency_ms,
            "uptime_hours": (time.time() - self.last_reset) / 3600
        }


# ============================================================================
# GLOBAL CACHES & METRICS
# ============================================================================

# Research result cache with TTL
research_cache: Dict[str, Tuple[ResearchResult, float]] = {}
cache_lock = threading.Lock()

# Performance metrics
metrics = ResearchMetrics()

# Query deduplication (prevent redundant simultaneous queries)
active_queries: Set[str] = set()
active_queries_lock = threading.Lock()

# ============================================================================
# CACHE MANAGEMENT
# ============================================================================

def get_cache_key(query: str) -> str:
    """Generate cache key from query."""
    return hashlib.md5(query.lower().strip().encode()).hexdigest()


def get_cached_result(query: str) -> Optional[ResearchResult]:
    """Retrieve cached result if available and not expired."""
    cache_key = get_cache_key(query)
    
    with cache_lock:
        if cache_key in research_cache:
            result, expiry_time = research_cache[cache_key]
            
            if time.time() < expiry_time:
                metrics.record_cache_hit()
                research_debug_logger.debug(f"[CACHE HIT] Query: {query}")
                return result
            else:
                # Expired - remove from cache
                del research_cache[cache_key]
                research_debug_logger.debug(f"[CACHE EXPIRED] Query: {query}")
    
    return None


def cache_result(query: str, result: ResearchResult, ttl: int = CACHE_TTL_SECONDS):
    """Store result in cache with TTL."""
    cache_key = get_cache_key(query)
    expiry_time = time.time() + ttl
    
    with cache_lock:
        # Enforce cache size limit
        if len(research_cache) >= MAX_CACHE_SIZE:
            # Remove oldest 10% of entries
            items = sorted(research_cache.items(), key=lambda x: x[1][1])
            to_remove = items[:MAX_CACHE_SIZE // 10]
            for key, _ in to_remove:
                del research_cache[key]
        
        research_cache[cache_key] = (result, expiry_time)
        research_debug_logger.debug(f"[CACHE STORE] Query: {query}, TTL: {ttl}s")


def clear_cache():
    """Clear all cached results."""
    with cache_lock:
        research_cache.clear()
    logger.info("[CACHE] Cleared all cached results")


# ============================================================================
# LOCAL-ONLY QUERY NORMALIZATION / SCORING HELPERS
# ============================================================================

_LOCAL_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "did", "do", "does",
    "for", "from", "give", "how", "i", "in", "is", "it", "me", "of", "on", "or", "please",
    "should", "tell", "that", "the", "this", "to", "up", "was", "were", "what", "when", "where",
    "which", "who", "why", "would", "you", "about", "define", "describe", "explain", "information",
}


def _normalize_local_query_text(text: str) -> str:
    """Normalize query/content for local-only matching without external dependencies."""
    try:
        return re.sub(r"\s+", " ", re.sub(r"[^a-zA-Z0-9_ .:\-/]", " ", str(text or "").lower())).strip()
    except Exception:
        return str(text or "").lower().strip()


def _extract_local_query_terms(query: str, *, max_terms: int = 12) -> List[str]:
    """Extract useful local-search terms from a natural-language question."""
    normalized = _normalize_local_query_text(query)
    words = [w for w in re.findall(r"[a-z0-9][a-z0-9_\-]{1,}", normalized) if w not in _LOCAL_QUERY_STOPWORDS]
    deduped: List[str] = []
    for word in words:
        if word not in deduped:
            deduped.append(word)
        if len(deduped) >= max_terms:
            break

    phrases: List[str] = []
    for i in range(0, max(0, len(words) - 1)):
        phrase = f"{words[i]} {words[i + 1]}".strip()
        if phrase and phrase not in phrases:
            phrases.append(phrase)
        if len(phrases) >= 4:
            break

    return [*deduped, *phrases]


def _local_text_match_score(query: str, content: str) -> float:
    """Bounded lexical relevance score for local database/dataset fallback."""
    q = _normalize_local_query_text(query)
    c = _normalize_local_query_text(content)
    if not q or not c:
        return 0.0
    if q in c:
        return 1.0

    terms = _extract_local_query_terms(query)
    if not terms:
        return 0.0

    word_terms = [t for t in terms if " " not in t]
    phrase_terms = [t for t in terms if " " in t]
    if not word_terms and not phrase_terms:
        return 0.0

    word_hits = sum(1 for t in word_terms if re.search(rf"\b{re.escape(t)}\b", c))
    phrase_hits = sum(1 for t in phrase_terms if t in c)
    word_score = word_hits / max(len(word_terms), 1)
    phrase_score = min(1.0, phrase_hits / max(len(phrase_terms), 1)) if phrase_terms else 0.0
    coverage = max(word_score, (word_score * 0.75) + (phrase_score * 0.25))

    if word_hits == 1 and not phrase_hits and len(word_terms) > 1:
        coverage = min(coverage, 0.34)

    return max(0.0, min(1.0, coverage))


def _best_local_snippet(content: str, query: str, *, max_chars: int = 900) -> str:
    """Return a compact local snippet around the strongest query term."""
    raw = re.sub(r"\s+", " ", str(content or "")).strip()
    if not raw:
        return ""

    low = raw.lower()
    best_idx = -1
    for term in _extract_local_query_terms(query):
        if not term:
            continue
        idx = low.find(term.lower())
        if idx >= 0:
            best_idx = idx
            break

    if best_idx < 0:
        return raw[:max_chars].strip()

    start = max(0, best_idx - max_chars // 3)
    end = min(len(raw), start + max_chars)
    snippet = raw[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(raw):
        snippet += "..."
    return snippet


def _looks_like_local_no_answer(text: str) -> bool:
    t = str(text or "").strip().lower()
    if not t:
        return True
    bad = (
        "sorry, i was unable to find",
        "i couldn't find reliable information",
        "could not find reliable information",
        "no engine produced an answer",
        "local llm failed",
        "api key missing",
        "blocked by safety/local-only mode",
    )
    return any(marker in t for marker in bad)


# ============================================================================
# LOW-QUALITY CACHE QUARANTINE
# ============================================================================

def _is_low_quality_research_text(text: str) -> bool:
    """True when a local hit is only a stored failure/fallback, not knowledge."""
    t = str(text or "").strip().lower()
    if not t:
        return True
    blocked = (
        "i'm not sure how to respond",
        "i am not sure how to respond",
        "not sure how to respond",
        "could you rephrase",
        "please rephrase",
        "i don't know how to respond",
        "i do not know how to respond",
        "no engine produced an answer",
        "provide more constraints",
        "enable an applicable tier",
        "still researching",
        "couldn't find reliable information",
        "could not find reliable information",
        "unable to answer",
        "answer unavailable",
        "local_none",
        "fallback failure",
        "traceback",
        "exception during",
        "error:",
    )
    return any(p in t for p in blocked)



def _is_procedural_howto_query(text: str) -> bool:
    """Detect procedural how-to prompts that should not trigger broad storage scans.

    These prompts are better served by vetted cache/static/local LLM. Scanning the
    SQLite dataset pool or imported file datasets for them is both low-value and
    risky on systems with mechanical backup/storage drives.
    """
    try:
        t = str(text or "").lower().strip()
        t = re.sub(r"[^a-z0-9\s']+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if not t:
            return False
        blocked = (
            " gpu", " cpu", " motherboard", " ram", " vram", " drive", " disk", " nvme", " hdd",
            "what type of", "what version", "what time", "who is", "who are", "where is", "when is",
        )
        if any(b in f" {t}" for b in blocked):
            return False
        return bool(
            re.search(r"\bhow\s+to\s+(make|cook|prepare|build|create|fix|clean|write|use|install|bake)\b", t)
            or re.search(r"\bhow\s+(?:do|can|should|would)\s+(?:i|you|we|someone|one)?\s*(make|cook|prepare|build|create|fix|clean|write|use|install|bake)\b", t)
            or re.search(r"\b(recipe|steps)\s+(?:for|to)\b", t)
        )
    except Exception:
        return False

def _normalize_static_lookup_query(query: str) -> str:
    """Normalize punctuation and common prompt wrappers for static fallback lookup."""
    q = str(query or "").strip().lower()
    q = re.sub(r"[^a-z0-9_\-\s']+", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    # Keep question phrases when they are part of STATIC_FACTS keys, but normalize
    # user variants like "can you tell me how to...".
    replacements = (
        ("can you tell me how to ", "how to "),
        ("tell me how to ", "how to "),
        ("can you tell me who is ", "who is "),
        ("tell me who is ", "who is "),
        ("can you tell me what is ", "what is "),
        ("tell me what is ", "what is "),
    )
    for src, dst in replacements:
        if q.startswith(src):
            q = dst + q[len(src):].strip()
            break
    return q


def _external_research_allowed(kind: str = "web") -> bool:
    """Governed external research gate.

    LOCAL_ONLY_MODE and SARAH_FORCE_OFFLINE always deny web/API lanes.  Web and
    third-party API lanes must be explicitly armed by Globals flags; static/local
    DB/local model lanes remain available.
    """
    try:
        if bool(getattr(config, "LOCAL_ONLY_MODE", True)):
            return False
        if str(os.getenv("SARAH_FORCE_OFFLINE", "")).strip().lower() in ("1", "true", "yes", "on"):
            return False
        k = str(kind or "web").strip().lower()
        if k == "api":
            return bool(getattr(config, "API_RESEARCH_ENABLED", False))
        return bool(getattr(config, "WEB_RESEARCH_ENABLED", False))
    except Exception:
        return False


def _research_source_label(result: "ResearchResult", lane: str) -> "ResearchResult":
    """Attach source/lane labels without changing the existing result schema."""
    try:
        if isinstance(result.metadata, dict):
            result.metadata.setdefault("lane", lane)
            result.metadata.setdefault("source_label", result.source.value if hasattr(result.source, "value") else str(result.source))
            result.metadata.setdefault("local_only", lane.startswith("local"))
    except Exception:
        pass
    return result

# ============================================================================
# LOCAL RESEARCH ENGINE
# ============================================================================

class LocalResearch:
    """
    Local database and offline research engine.
    
    Searches through:
    - QA cache (previous questions/answers)
    - Personality response database
    - Imported local datasets
    - Offline LLM models
    - WebSYM semantic synthesis
    - Static fallback knowledge
    """
    
    @staticmethod
    def search(query: str, intent: str) -> Optional[ResearchResult]:
        """
        Comprehensive local search across all local sources.
        
        Args:
            query: User query
            intent: Classified intent from AdvCU
            
        Returns:
            ResearchResult if found, None otherwise
        """
        start_time = time.time()
        research_debug_logger.debug(f"[LOCAL SEARCH START] Query: '{query}' | Intent: {intent}")
        
        try:
            # Priority 1: QA Cache (fastest, highest confidence)
            result = LocalResearch._search_qa_cache(query, intent)
            if result:
                result.latency_ms = int((time.time() - start_time) * 1000)
                research_debug_logger.debug(f"[LOCAL SUCCESS] Source: QA Cache, Latency: {result.latency_ms}ms")
                return result
            
            # Priority 2: Personality Response Database
            result = LocalResearch._search_personality(query, intent)
            if result:
                result.latency_ms = int((time.time() - start_time) * 1000)
                research_debug_logger.debug(f"[LOCAL SUCCESS] Source: Personality, Latency: {result.latency_ms}ms")
                return result
            
            # Priority 3: Static exact facts before any broad SQLite/vector/import scan.
            # This preserves local-first behavior while preventing simple stable questions
            # (for example, vetted STATIC_FACTS entries) from waking heavy storage lanes.
            result = LocalResearch._search_static(query, intent)
            if result:
                result.latency_ms = int((time.time() - start_time) * 1000)
                research_debug_logger.debug(f"[LOCAL SUCCESS] Source: Static Fast Path, Latency: {result.latency_ms}ms")
                return result

            # Priority 4: Procedural how-to fast path. Do NOT run broad SQLite/vector/import scans
            # for simple how-to prompts; use static/local LLM first to avoid mechanical-drive churn.
            if _is_procedural_howto_query(query):
                result = LocalResearch._search_static(query, intent)
                if result:
                    result.latency_ms = int((time.time() - start_time) * 1000)
                    research_debug_logger.debug(f"[LOCAL SUCCESS] Source: Static How-To, Latency: {result.latency_ms}ms")
                    return result

                result = LocalResearch._search_llm_ensemble(query, intent)
                if result:
                    result.latency_ms = int((time.time() - start_time) * 1000)
                    research_debug_logger.debug(f"[LOCAL SUCCESS] Source: Local LLM How-To, Latency: {result.latency_ms}ms")
                    return result

                if not bool(getattr(config, "LOCAL_HOWTO_ALLOW_DATASET_SCAN", False)):
                    logger.info("[LOCAL HOWTO] Dataset/vector/import scans skipped by anti-thrash policy.")
                    return None

            # Priority 5: Local LLM generation for low-risk general knowledge.
            # WAVE7: Use the selected local model before broad DB/vector scans so
            # general questions do not grind storage and do not fall back to static seeds.
            if bool(getattr(config, "LOCAL_LLM_FIRST_FOR_GENERAL", True)):
                result = LocalResearch._search_llm_ensemble(query, intent)
                if result:
                    result.latency_ms = int((time.time() - start_time) * 1000)
                    research_debug_logger.debug(f"[LOCAL SUCCESS] Source: Local LLM First, Latency: {result.latency_ms}ms")
                    return result

            # Priority 6: Direct SQLite dataset pool search
            # Phase 7 embedding may be skipped by optimized boot policy. That must
            # not disable local knowledge retrieval. Search is bounded and project-local.
            result = LocalResearch._search_sqlite_dataset_pool(query, intent)
            if result:
                result.latency_ms = int((time.time() - start_time) * 1000)
                research_debug_logger.debug(f"[LOCAL SUCCESS] Source: SQLite Dataset Pool, Latency: {result.latency_ms}ms")
                return result

            # Priority 6: Local Vector Memory
            result = LocalResearch._search_vector_memory(query, intent)
            if result:
                result.latency_ms = int((time.time() - start_time) * 1000)
                research_debug_logger.debug(f"[LOCAL SUCCESS] Source: Vector Memory, Latency: {result.latency_ms}ms")
                return result

            # Priority 7: Imported Local Text Datasets
            result = LocalResearch._search_datasets(query, intent)
            if result:
                result.latency_ms = int((time.time() - start_time) * 1000)
                research_debug_logger.debug(f"[LOCAL SUCCESS] Source: Imported Text Datasets, Latency: {result.latency_ms}ms")
                return result
            
            # Priority 8: Static fallback is disabled by default in Wave7.
            # It is kept only for explicit diagnostics/demo mode so hardcoded answers
            # cannot mask a local LLM routing failure.
            if bool(getattr(config, "LOCAL_STATIC_FALLBACK_ENABLED", False)):
                result = LocalResearch._search_static(query, intent)
                if result:
                    result.latency_ms = int((time.time() - start_time) * 1000)
                    research_debug_logger.debug(f"[LOCAL SUCCESS] Source: Static, Latency: {result.latency_ms}ms")
                    return result

            # Priority 9: Offline LLM Ensemble final attempt
            result = LocalResearch._search_llm_ensemble(query, intent)
            if result:
                result.latency_ms = int((time.time() - start_time) * 1000)
                research_debug_logger.debug(f"[LOCAL SUCCESS] Source: LLM Ensemble, Latency: {result.latency_ms}ms")
                return result
            
            # Priority 10: WebSYM Semantic Synthesis
            result = LocalResearch._search_websym(query, intent)
            if result:
                result.latency_ms = int((time.time() - start_time) * 1000)
                research_debug_logger.debug(f"[LOCAL SUCCESS] Source: WebSYM, Latency: {result.latency_ms}ms")
                return result
            
            logger.info("[LOCAL] No local match found across all sources")
            return None
            
        except Exception as e:
            logger.error(f"[LOCAL ERROR] Search failed: {e}")
            research_debug_logger.error(f"[LOCAL ERROR] {str(e)}")
            return None
    
    @staticmethod
    def _search_qa_cache(query: str, intent: str) -> Optional[ResearchResult]:
        """Search QA cache with semantic similarity."""
        try:
            cached_answers = search_answers(query)
            if cached_answers:
                clean_answers = [a for a in cached_answers if not _is_low_quality_research_text(a)]
                if not clean_answers:
                    research_debug_logger.debug("[QA CACHE QUARANTINE] Only low-quality fallback answers matched; skipping QA cache.")
                    return None
                answer = clean_answers[0]
                research_debug_logger.debug(f"[QA CACHE HIT] Result: {answer[:100]}")
                logger.info("[Class 1] Found valid match in QA Cache")
                
                return ResearchResult(
                    source=ResearchSource.LOCAL_QA_CACHE,
                    content=answer,
                    confidence=0.95,
                    query=query,
                    intent=intent,
                    metadata={"file": "QA Cache", "method": "direct_match"}
                )
        except Exception as e:
            research_debug_logger.debug(f"[QA CACHE ERROR] {str(e)}")
        
        return None
    
    @staticmethod
    def _search_personality(query: str, intent: str) -> Optional[ResearchResult]:
        """Search personality response database."""
        try:
            response_answers = search_responses(query)
            if response_answers:
                clean_answers = [a for a in response_answers if not _is_low_quality_research_text(a)]
                if not clean_answers:
                    research_debug_logger.debug("[PERSONALITY QUARANTINE] Only low-quality fallback responses matched; skipping Personality DB.")
                    return None
                answer = clean_answers[0]
                research_debug_logger.debug(f"[PERSONALITY HIT] Result: {answer[:100]}")
                logger.info("[Class 1] Found valid match in Personality Responses")
                
                return ResearchResult(
                    source=ResearchSource.LOCAL_PERSONALITY,
                    content=answer,
                    confidence=0.90,
                    query=query,
                    intent=intent,
                    metadata={"file": "Personality1", "method": "response_match"}
                )
        except Exception as e:
            research_debug_logger.debug(f"[PERSONALITY ERROR] {str(e)}")
        
        return None
    
    @staticmethod
    def _search_sqlite_dataset_pool(query: str, intent: str) -> Optional[ResearchResult]:
        """Search the local SQLite dataset pool directly.

        This is the Phase-7-safe fallback. It does not require prebuilt vector
        embeddings and never touches Web/API providers. It is bounded by row,
        table, hit, and time limits inside SarahMemoryDatabase.
        """
        try:
            if not callable(search_local_dataset_pool):
                research_debug_logger.debug("[SQLITE POOL] search_local_dataset_pool unavailable")
                return None

            pool = search_local_dataset_pool(
                query,
                max_hits=int(getattr(config, "LOCAL_DATASET_POOL_MAX_HITS", 5) or 5),
                time_budget_sec=float(getattr(config, "LOCAL_DATASET_POOL_TIME_BUDGET_SEC", 1.25) or 1.25),
                max_tables_per_db=int(getattr(config, "LOCAL_DATASET_POOL_MAX_TABLES_PER_DB", 20) or 20),
                max_rows_per_table=int(getattr(config, "LOCAL_DATASET_POOL_MAX_ROWS_PER_TABLE", 4) or 4),
                max_db_files=int(getattr(config, "LOCAL_DATASET_POOL_MAX_DBS", 12) or 12),
            )
            if not isinstance(pool, dict) or not pool.get("ok"):
                return None

            hits = pool.get("hits") if isinstance(pool.get("hits"), list) else []
            hits = [h for h in hits if not _is_low_quality_research_text(h.get("snippet"))]
            if not hits:
                research_debug_logger.debug("[SQLITE POOL QUARANTINE] Only low-quality fallback rows matched; skipping SQLite pool.")
                return None

            best_score = float(hits[0].get("score") or 0.0)
            if best_score < 0.42:
                return None

            combined = "\n\n".join([
                f"From {h.get('db')}::{h.get('table')} [score={float(h.get('score') or 0.0):.2f}]: {h.get('snippet')}"
                for h in hits[:3]
                if str(h.get("snippet") or "").strip()
            ]).strip()

            if not combined:
                return None

            return ResearchResult(
                source=ResearchSource.LOCAL_DATASETS,
                content=combined,
                confidence=max(0.60, min(0.90, best_score)),
                query=query,
                intent=intent,
                metadata={
                    "method": "direct_sqlite_dataset_pool_search",
                    "hit_count": len(hits),
                    "db_count": pool.get("db_count"),
                    "tables_checked": pool.get("tables_checked"),
                    "phase7_required": False,
                    "files": [f"{h.get('db')}::{h.get('table')}" for h in hits[:8]],
                },
            )
        except Exception as e:
            research_debug_logger.debug(f"[SQLITE POOL ERROR] {str(e)}")
            return None


    @staticmethod
    def _search_vector_memory(query: str, intent: str) -> Optional[ResearchResult]:
        """Search local vectorized memory stores without web/API access."""
        try:
            if not bool(getattr(config, "LOCAL_VECTOR_SEARCH_ENABLED", True)):
                research_debug_logger.debug("[VECTOR MEMORY] Disabled by LOCAL_VECTOR_SEARCH_ENABLED=False")
                return None
            top_n = max(1, min(3, int(getattr(config, "LOCAL_VECTOR_SEARCH_TOP_N", 3) or 3)))
            qa_hits = vector_search_qa_cache(query, top_n=top_n) or []
            mem_hits = vector_search(query, top_n=top_n) or []

            combined = []
            for score, text in qa_hits:
                if _is_low_quality_research_text(text):
                    continue
                lexical = _local_text_match_score(query, str(text))
                if lexical >= 0.35:
                    combined.append((float(score), lexical, str(text), "qa_cache"))
            for score, text in mem_hits:
                if _is_low_quality_research_text(text):
                    continue
                lexical = _local_text_match_score(query, str(text))
                if lexical >= 0.35:
                    combined.append((float(score), lexical, str(text), "voice_logs"))

            combined.sort(key=lambda item: (item[1], item[0]), reverse=True)
            if not combined:
                return None

            top = combined[:3]
            content = "\n\n".join([f"[{src} score={score:.3f} lexical={lexical:.2f}] {text[:600]}" for score, lexical, text, src in top])
            best = max(0.0, min(0.92, float(top[0][1])))
            return ResearchResult(
                source=ResearchSource.LOCAL_DATASETS,
                content=content,
                confidence=max(0.58, best),
                query=query,
                intent=intent,
                metadata={"method": "local_vector_search", "hits": len(combined), "top_sources": [src for _, _, _, src in top]}
            )
        except Exception as e:
            research_debug_logger.debug(f"[VECTOR MEMORY ERROR] {str(e)}")
            return None

    @staticmethod
    def _search_datasets(query: str, intent: str) -> Optional[ResearchResult]:
        """Search imported local datasets using ranked lexical/entity matching.

        This is intentionally local-only and offline-safe. It fixes the old exact
        sentence lookup behavior where natural-language questions only matched
        when the complete question existed verbatim in the dataset pool.
        """
        try:
            if not bool(getattr(config, "LOCAL_IMPORTED_DATA_SCAN_ENABLED", False)):
                research_debug_logger.debug("[DATASETS] Imported text scan skipped by LOCAL_IMPORTED_DATA_SCAN_ENABLED=False")
                return None
            results = []
            data = import_other_data() or {}
            for file, content in data.items():
                text = str(content or "")
                if not text.strip():
                    continue
                if _is_low_quality_research_text(text):
                    continue
                score = _local_text_match_score(query, text)
                if score >= 0.42:
                    snippet = _best_local_snippet(text, query, max_chars=900).replace("\n", " ")
                    research_debug_logger.debug(f"[DATASET HIT] File: {file} | score={score:.3f}")
                    results.append({"file": str(file), "snippet": snippet, "score": score})

            results.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
            if results:
                logger.info(f"[Class 1] Found {len(results)} scored matches in imported datasets")
                combined_content = "\n\n".join([
                    f"From {r['file']} [score={float(r['score']):.2f}]: {r['snippet']}" for r in results[:3]
                ])
                best_score = float(results[0].get("score") or 0.0)
                return ResearchResult(
                    source=ResearchSource.LOCAL_DATASETS,
                    content=combined_content,
                    confidence=max(0.58, min(0.88, best_score)),
                    query=query,
                    intent=intent,
                    metadata={"files": [r["file"] for r in results[:8]], "count": len(results), "method": "ranked_local_dataset_search"}
                )
        except Exception as e:
            research_debug_logger.debug(f"[DATASETS ERROR] {str(e)}")
        return None

    @staticmethod
    def _search_llm_ensemble(query: str, intent: str) -> Optional[ResearchResult]:
        """Generate an answer with the configured local text-generation LLM.

        Earlier builds only proved an embedding model could process the query and
        returned a status sentence. That did not answer general knowledge questions.
        This lane now calls SarahMemoryAPI provider='local_llm' when enabled and
        returns the actual generated answer. No web/API provider is used here.
        """
        try:
            if not bool(getattr(config, "LOCAL_LLM_API", False) or getattr(config, "LOCAL_LLM_ENABLED", False)):
                research_debug_logger.debug("[LOCAL LLM] Disabled by Globals; skipping local generation")
                return None

            logger.info("[Class 1] Attempting local LLM response synthesis")
            prompt = (
                "Answer the user's question using your local model knowledge. "
                "Do not claim live web access. If local evidence is insufficient, say so briefly.\n\n"
                f"Question: {query}"
            )
            response = _send_to_api_with_resolver(
                prompt,
                provider="local_llm",
                intent=intent or "question",
                tone="neutral",
                complexity="adult",
            )
            if isinstance(response, dict):
                text = str(response.get("data") or response.get("reply") or "").strip()
                if text and not _looks_like_local_no_answer(text):
                    return ResearchResult(
                        source=ResearchSource.LOCAL_LLM_ENSEMBLE,
                        content=text,
                        confidence=0.66,
                        query=query,
                        intent=intent,
                        metadata={
                            "method": "local_llm_generation",
                            "provider": response.get("source") or "local_llm",
                            "model_used": response.get("model_used"),
                        },
                    )
                if response.get("error"):
                    research_debug_logger.debug(f"[LOCAL LLM ERROR] {response.get('error')}")
        except Exception as e:
            research_debug_logger.debug(f"[LOCAL LLM ENSEMBLE ERROR] {str(e)}")
        return None

    @staticmethod
    def _search_websym(query: str, intent: str) -> Optional[ResearchResult]:
        """Search using WebSYM semantic synthesizer."""
        try:
            logger.info("[Class 1] Attempting WebSYM Fallback")
            synthesized = WebSemanticSynthesizer.synthesize_response("", query)
            
            if synthesized and "couldn't find reliable information" not in synthesized.lower():
                research_debug_logger.debug(f"[WEBSYM SUCCESS] Result: {synthesized[:100]}")
                
                return ResearchResult(
                    source=ResearchSource.LOCAL_WEBSYM,
                    content=synthesized,
                    confidence=0.75,
                    query=query,
                    intent=intent,
                    metadata={"method": "semantic_synthesis"}
                )
        except Exception as e:
            research_debug_logger.debug(f"[WEBSYM ERROR] {str(e)}")
        
        return None
    
    @staticmethod
    def _search_static(query: str, intent: str) -> Optional[ResearchResult]:
        """Search static fallback knowledge base."""
        try:
            lookup = _normalize_static_lookup_query(query)
            static_fact = STATIC_FACTS.get(lookup) or STATIC_FACTS.get(str(query or "").lower().strip())
            if static_fact:
                logger.info("[Class 1] Static fallback match found")
                research_debug_logger.debug(f"[STATIC HIT] Query: {query}")
                
                return ResearchResult(
                    source=ResearchSource.LOCAL_STATIC,
                    content=static_fact,
                    confidence=1.0,
                    query=query,
                    intent=intent,
                    metadata={"type": "hardcoded_fact"}
                )
        except Exception as e:
            research_debug_logger.debug(f"[STATIC ERROR] {str(e)}")
        
        return None


# ============================================================================
# WEB RESEARCH ENGINE
# ============================================================================

class WebResearch:
    """
    Web-based research engine with multiple source support.
    
    Supports:
    - Wikipedia (structured API)
    - DuckDuckGo (HTML parsing with anti-CAPTCHA)
    - Free Dictionary (word definitions)
    - OpenLibrary (books and authors)
    - StackOverflow (programming questions)
    - Reddit (community discussions)
    - WikiHow (how-to guides)
    - Quora (Q&A community)
    - Internet Archive (historical content)
    """
    
    @staticmethod
    async def fetch_json(url: str, params: Optional[Dict] = None, timeout: int = HTTP_TIMEOUT_MEDIUM) -> Optional[Dict]:
        """Fetch JSON response from URL."""
        if aiohttp is None:
            logger.warning("[WEB] aiohttp unavailable; JSON fetch skipped for %s", url)
            return None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=timeout) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"[WEB] HTTP {response.status} for {url}")
                        return None
        except asyncio.TimeoutError:
            logger.warning(f"[WEB] Timeout for {url}")
            return None
        except Exception as e:
            logger.warning(f"[WEB JSON ERROR] {url}: {e}")
            return None
    
    @staticmethod
    async def fetch_html(url: str, timeout: int = HTTP_TIMEOUT_MEDIUM) -> Optional[str]:
        """Fetch HTML response from URL."""
        if aiohttp is None:
            logger.warning("[WEB] aiohttp unavailable; HTML fetch skipped for %s", url)
            return None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logger.warning(f"[WEB] HTTP {response.status} for {url}")
                        return None
        except asyncio.TimeoutError:
            logger.warning(f"[WEB] Timeout for {url}")
            return None
        except Exception as e:
            logger.warning(f"[WEB HTML ERROR] {url}: {e}")
            return None
    
    @staticmethod
    def preprocess_query(query: str) -> str:
        """Preprocess query for web search optimization."""
        prefixes = [
            "what is", "who is", "tell me about", "give me information on",
            "explain", "define", "describe", "how does", "why does",
            "when did", "where is", "which is"
        ]
        
        query_lower = query.lower().strip()
        for prefix in prefixes:
            if query_lower.startswith(prefix):
                query = query_lower[len(prefix):].strip()
                break
        
        # Remove punctuation at end
        query = query.rstrip('?.!')
        
        research_debug_logger.debug(f"[PREPROCESS] '{query_lower}' → '{query}'")
        return query
    
    @staticmethod
    async def fetch_sources(query: str) -> Optional[ResearchResult]:
        """
        Fetch information from multiple web sources concurrently.
        
        Args:
            query: User query
            
        Returns:
            ResearchResult with synthesized information
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in research_cache:
            cached_result, expiry = research_cache[cache_key]
            if time.time() < expiry:
                return cached_result
        
        clean_query = WebResearch.preprocess_query(query)
        logger.info(f"[WEB RESEARCH] Preprocessed Query: '{clean_query}'")
        
        # Collect raw data from multiple sources
        raw = {}
        tasks = []
        
        # Wikipedia
        if config.WIKIPEDIA_RESEARCH_ENABLED:
            tasks.append(WebResearch._fetch_wikipedia(clean_query))
        
        # DuckDuckGo
        if config.DUCKDUCKGO_RESEARCH_ENABLED:
            tasks.append(WebResearch._fetch_duckduckgo(clean_query))
        
        # Free Dictionary
        if config.FREE_DICTIONARY_RESEARCH_ENABLED:
            tasks.append(WebResearch._fetch_dictionary(clean_query))
        
        # OpenLibrary
        if config.OPENLIBRARY_RESEARCH_ENABLED:
            tasks.append(WebResearch._fetch_openlibrary(clean_query))
        
        # StackOverflow
        if config.STACKOVERFLOW_RESEARCH_ENABLED:
            tasks.append(WebResearch._fetch_stackoverflow(clean_query))
        
        # Reddit
        if config.REDDIT_RESEARCH_ENABLED:
            tasks.append(WebResearch._fetch_reddit(clean_query))
        
        # WikiHow
        if config.WIKIHOW_RESEARCH_ENABLED:
            tasks.append(WebResearch._fetch_wikihow(clean_query))
        
        # Quora
        if config.QUORA_RESEARCH_ENABLED:
            tasks.append(WebResearch._fetch_quora(clean_query))
        
        # Internet Archive
        if config.INTERNET_ARCHIVE_RESEARCH_ENABLED:
            tasks.append(WebResearch._fetch_archive(clean_query))
        
        # Execute all tasks concurrently
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.warning(f"[WEB] Task {i} failed: {result}")
                elif result:
                    source_name, content = result
                    raw[source_name] = content
        
        # Synthesize response from collected data
        if raw:
            combined_raw_text = "\n\n".join([
                f"[{source}] {content}" for source, content in raw.items() if isinstance(content, str)
            ])
            
            synthesized_response = WebSemanticSynthesizer.synthesize_response(combined_raw_text, query)
            
            if synthesized_response and "still researching" not in synthesized_response.lower():
                latency_ms = int((time.time() - start_time) * 1000)
                
                result = ResearchResult(
                    source=ResearchSource.WEB_WIKIPEDIA,  # Primary source
                    content=synthesized_response,
                    confidence=0.80,
                    query=query,
                    intent="research",
                    latency_ms=latency_ms,
                    metadata={"sources": list(raw.keys()), "raw_count": len(raw)}
                )
                
                # Cache the result
                cache_result(query, result)
                
                return result
            
            # If synthesis failed, return first valid raw result
            for val in raw.values():
                if val and isinstance(val, str):
                    latency_ms = int((time.time() - start_time) * 1000)
                    
                    result = ResearchResult(
                        source=ResearchSource.WEB_DUCKDUCKGO,
                        content=val,
                        confidence=0.70,
                        query=query,
                        intent="research",
                        latency_ms=latency_ms,
                        metadata={"sources": list(raw.keys())}
                    )
                    
                    return result
        
        logger.warning("[WEB] No web results found")
        return None
    
    @staticmethod
    async def _fetch_wikipedia(query: str) -> Optional[Tuple[str, str]]:
        """Fetch from Wikipedia API."""
        try:
            url = WIKIPEDIA_API + query.replace(" ", "_")
            data = await WebResearch.fetch_json(url, timeout=HTTP_TIMEOUT_SHORT)
            
            if data and "extract" in data:
                extract = data["extract"]
                research_debug_logger.debug(f"[WIKIPEDIA] Found: {extract[:100]}")
                return ("Wikipedia", extract)
        except Exception as e:
            research_debug_logger.debug(f"[WIKIPEDIA ERROR] {str(e)}")
        
        return None
    
    @staticmethod
    async def _fetch_duckduckgo(query: str) -> Optional[Tuple[str, str]]:
        """Fetch from DuckDuckGo HTML."""
        try:
            url = DUCKDUCKGO_HTML + query.replace(" ", "+")
            html_text = await WebResearch.fetch_html(url, timeout=HTTP_TIMEOUT_SHORT)
            
            if html_text:
                # Check for CAPTCHA
                if WebResearch._is_ddg_blocked(html_text):
                    research_debug_logger.debug("[DUCKDUCKGO] CAPTCHA detected")
                    return None
                
                soup = BeautifulSoup(html_text, 'html.parser')
                snippet = soup.select_one('a.result__snippet')
                
                if snippet:
                    text = snippet.text.strip()
                    research_debug_logger.debug(f"[DUCKDUCKGO] Found: {text[:100]}")
                    return ("DuckDuckGo", text)
        except Exception as e:
            research_debug_logger.debug(f"[DUCKDUCKGO ERROR] {str(e)}")
        
        return None
    
    @staticmethod
    async def _fetch_dictionary(query: str) -> Optional[Tuple[str, str]]:
        """Fetch from Free Dictionary."""
        try:
            url = FREE_DICTIONARY_API + query
            html_text = await WebResearch.fetch_html(url, timeout=HTTP_TIMEOUT_SHORT)
            
            if html_text:
                soup = BeautifulSoup(html_text, 'html.parser')
                definition = soup.select_one('div.definition')
                
                if definition:
                    text = definition.text.strip()
                    research_debug_logger.debug(f"[DICTIONARY] Found: {text[:100]}")
                    return ("FreeDictionary", text)
        except Exception as e:
            research_debug_logger.debug(f"[DICTIONARY ERROR] {str(e)}")
        
        return None
    
    @staticmethod
    async def _fetch_openlibrary(query: str) -> Optional[Tuple[str, str]]:
        """Fetch from OpenLibrary API."""
        try:
            url = OPENLIBRARY_API + query
            data = await WebResearch.fetch_json(url, timeout=HTTP_TIMEOUT_SHORT)
            
            if data and "docs" in data and data["docs"]:
                first_result = data["docs"][0]
                title = first_result.get("title", "")
                author = first_result.get("author_name", ["Unknown"])[0]
                year = first_result.get("first_publish_year", "Unknown")
                
                text = f"{title} by {author} (published {year})"
                research_debug_logger.debug(f"[OPENLIBRARY] Found: {text}")
                return ("OpenLibrary", text)
        except Exception as e:
            research_debug_logger.debug(f"[OPENLIBRARY ERROR] {str(e)}")
        
        return None
    
    @staticmethod
    async def _fetch_stackoverflow(query: str) -> Optional[Tuple[str, str]]:
        """Fetch from StackOverflow API."""
        try:
            url = STACKOVERFLOW_API + query
            data = await WebResearch.fetch_json(url, timeout=HTTP_TIMEOUT_SHORT)
            
            if data and "items" in data and data["items"]:
                first_result = data["items"][0]
                title = first_result.get("title", "")
                score = first_result.get("score", 0)
                
                text = f"{title} (Score: {score})"
                research_debug_logger.debug(f"[STACKOVERFLOW] Found: {text}")
                return ("StackOverflow", text)
        except Exception as e:
            research_debug_logger.debug(f"[STACKOVERFLOW ERROR] {str(e)}")
        
        return None
    
    @staticmethod
    async def _fetch_reddit(query: str) -> Optional[Tuple[str, str]]:
        """Fetch from Reddit API."""
        try:
            url = REDDIT_API + query
            data = await WebResearch.fetch_json(url, timeout=HTTP_TIMEOUT_SHORT)
            
            if data and "data" in data and "children" in data["data"] and data["data"]["children"]:
                first_post = data["data"]["children"][0]["data"]
                title = first_post.get("title", "")
                score = first_post.get("score", 0)
                
                text = f"{title} (Score: {score})"
                research_debug_logger.debug(f"[REDDIT] Found: {text}")
                return ("Reddit", text)
        except Exception as e:
            research_debug_logger.debug(f"[REDDIT ERROR] {str(e)}")
        
        return None
    
    @staticmethod
    async def _fetch_wikihow(query: str) -> Optional[Tuple[str, str]]:
        """Fetch from WikiHow."""
        try:
            url = WIKIHOW_API + query
            data = await WebResearch.fetch_json(url, timeout=HTTP_TIMEOUT_SHORT)
            
            if data and "query" in data and "search" in data["query"]:
                results = data["query"]["search"]
                if results:
                    title = results[0].get("title", "")
                    research_debug_logger.debug(f"[WIKIHOW] Found: {title}")
                    return ("WikiHow", title)
        except Exception as e:
            research_debug_logger.debug(f"[WIKIHOW ERROR] {str(e)}")
        
        return None
    
    @staticmethod
    async def _fetch_quora(query: str) -> Optional[Tuple[str, str]]:
        """Fetch from Quora (basic scraping)."""
        try:
            url = QUORA_BASE + query.replace(" ", "+")
            html_text = await WebResearch.fetch_html(url, timeout=HTTP_TIMEOUT_SHORT)
            
            if html_text:
                soup = BeautifulSoup(html_text, 'html.parser')
                # Note: Quora structure may change; this is a basic example
                question = soup.select_one('span.q-text')
                
                if question:
                    text = question.text.strip()
                    research_debug_logger.debug(f"[QUORA] Found: {text[:100]}")
                    return ("Quora", text)
        except Exception as e:
            research_debug_logger.debug(f"[QUORA ERROR] {str(e)}")
        
        return None
    
    @staticmethod
    async def _fetch_archive(query: str) -> Optional[Tuple[str, str]]:
        """Fetch from Internet Archive."""
        try:
            url = INTERNET_ARCHIVE_API + query + "&output=json"
            data = await WebResearch.fetch_json(url, timeout=HTTP_TIMEOUT_SHORT)
            
            if data and "response" in data and "docs" in data["response"]:
                docs = data["response"]["docs"]
                if docs:
                    first_doc = docs[0]
                    title = first_doc.get("title", "")
                    research_debug_logger.debug(f"[ARCHIVE] Found: {title}")
                    return ("InternetArchive", title)
        except Exception as e:
            research_debug_logger.debug(f"[ARCHIVE ERROR] {str(e)}")
        
        return None
    
    @staticmethod
    def _is_ddg_blocked(text: str) -> bool:
        """Check if DuckDuckGo has blocked the request with CAPTCHA."""
        if not text:
            return False
        
        text_lower = text.lower()
        blocking_keywords = [
            'error-lite@duckduckgo.com',
            'select all squares containing a duck',
            'please complete the following challenge',
            'bots use duckduckgo too',
            'captcha',
            'human verification'
        ]
        
        return any(keyword in text_lower for keyword in blocking_keywords)


# ============================================================================

# -----------------------------------------------------------------------------
# Resolver-aware API call shim (keeps model names centralized in SarahMemoryGlobals.py)
# -----------------------------------------------------------------------------
def _send_to_api_with_resolver(query: str, *, provider: str, intent: str, tone: str = "neutral", complexity: str = "adult") -> Optional[Dict[str, Any]]:
    """Call SarahMemoryAPI.send_to_api with non-breaking resolver hints.
    - Adds meta.sm_model_resolve + meta.preferred_model_repo when possible.
    - Never assumes send_to_api signature; retries without extra kwargs on TypeError.
    """
    meta2: Dict[str, Any] = {}
    try:
        rm_fn = getattr(config, "resolve_model", None)
        if callable(rm_fn):
            rm = rm_fn("reasoning", text=query, meta={}) or {}
            meta2.setdefault("sm_model_resolve", {})["reasoning"] = rm
            if isinstance(rm, dict) and rm.get("selected"):
                meta2["preferred_model_repo"] = rm.get("selected")
    except Exception:
        pass

    # Attempt: send meta (newer signature)
    try:
        return send_to_api(
            query,
            provider=provider,
            intent=intent,
            tone=tone,
            complexity=complexity,
            meta=meta2,
        )
    except TypeError:
        pass
    except Exception:
        # allow caller to handle provider fallback
        return None

    # Fallback: legacy signature
    try:
        return send_to_api(
            query,
            provider=provider,
            intent=intent,
            tone=tone,
            complexity=complexity,
        )
    except Exception:
        return None


# API RESEARCH ENGINE
# ============================================================================

class APIResearch:
    """
    External API research engine with multi-provider support.
    
    Supports:
    - OpenAI (GPT-4, GPT-3.5)
    - Anthropic Claude
    - Mistral AI
    - Google Gemini
    - HuggingFace
    
    Features automatic fallback chain and provider selection.
    """
    
    @staticmethod
    def query(query: str, intent: str) -> Optional[ResearchResult]:
        """
        Query external API providers with automatic fallback.
        
        Args:
            query: User query
            intent: Classified intent
            
        Returns:
            ResearchResult if successful, None otherwise
        """
        start_time = time.time()
        
        try:
            # Global kill switch check
            if not config.API_RESEARCH_ENABLED:
                logger.warning("[API] API Research globally disabled via config.API_RESEARCH_ENABLED")
                research_debug_logger.debug("[API BLOCKED] Global kill switch")
                return None
            
            # Determine provider priority order
            provider_priority = APIResearch._get_provider_priority()
            
            if not provider_priority:
                logger.warning("[API] No AI provider enabled in config")
                return None
            
            # Try each provider in order
            for provider in provider_priority:
                try:
                    logger.info(f"[API] Attempting provider: {provider}")
                    
                    result = _send_to_api_with_resolver(
                        query,
                        provider=provider,
                        intent=intent,
                        tone="neutral",
                        complexity="adult"
                    )
                    
                    if result and result.get("data"):
                        latency_ms = int((time.time() - start_time) * 1000)
                        
                        # Map provider to ResearchSource
                        source_map = {
                            "openai": ResearchSource.API_OPENAI,
                            "claude": ResearchSource.API_CLAUDE,
                            "mistral": ResearchSource.API_MISTRAL,
                            "gemini": ResearchSource.API_GEMINI,
                            "huggingface": ResearchSource.API_HUGGINGFACE
                        }
                        
                        source = source_map.get(provider, ResearchSource.API_OPENAI)
                        
                        research_result = ResearchResult(
                            source=source,
                            content=result.get("data", ""),
                            confidence=0.85,
                            query=query,
                            intent=result.get("intent", intent),
                            latency_ms=latency_ms,
                            metadata={
                                "provider": provider,
                                "original_source": result.get("source", provider)
                            }
                        )
                        
                        logger.info(f"[API SUCCESS] Provider: {provider}, Latency: {latency_ms}ms")
                        return research_result
                    
                    else:
                        logger.warning(f"[API] Provider {provider} returned no data")
                
                except Exception as e:
                    logger.warning(f"[API ERROR] Provider {provider}: {e}")
                    research_debug_logger.debug(f"[API FAILURE] {provider}: {str(e)}")
                    continue
            
            logger.warning("[API] All providers failed")
            return None
        
        except Exception as e:
            logger.error(f"[API] Fatal error: {e}")
            research_debug_logger.error(f"[API FATAL] {str(e)}")
            return None
    
    @staticmethod
    def _get_provider_priority() -> List[str]:
        """Get list of enabled providers in priority order."""
        priority_order = ["openai", "claude", "mistral", "gemini", "huggingface"]
        enabled = []
        
        for provider in priority_order:
            config_key = f"{provider.upper()}_API"
            if hasattr(config, config_key) and getattr(config, config_key):
                enabled.append(provider)
        
        return enabled


# ============================================================================
# PARALLEL RESEARCH COORDINATOR
# ============================================================================

async def parallel_research(query: str) -> ResearchResult:
    """
    Execute parallel research across all available sources.
    
    This is the main entry point for research operations. It coordinates
    local, web, and API research in parallel and returns the best result.
    
    Args:
        query: User query
        
    Returns:
        ResearchResult with best available information
    """
    start_time = time.time()
    
    # Check cache first
    cached = get_cached_result(query)
    if cached:
        logger.info(f"[RESEARCH] Cache hit for query: '{query}'")
        return cached
    
    # Check for duplicate active query
    query_key = get_cache_key(query)
    with active_queries_lock:
        if query_key in active_queries:
            logger.info(f"[RESEARCH] Duplicate query in progress: '{query}'")
            # Wait a bit and check cache again
            await asyncio.sleep(0.5)
            cached = get_cached_result(query)
            if cached:
                return cached
        else:
            active_queries.add(query_key)
    
    try:
        # Classify intent
        intent = classify_intent(query)
        research_debug_logger.debug(f"[PARALLEL START] Query: '{query}', Intent: {intent}")
        
        # Define parallel tasks
        async def local_task():
            """Local research task."""
            if config.LOCAL_DATA_ENABLED:
                return await asyncio.to_thread(LocalResearch.search, query, intent)
            return None
        
        async def web_task():
            """Web research task."""
            if _external_research_allowed("web"):
                try:
                    return await WebResearch.fetch_sources(query)
                except Exception as e:
                    logger.warning(f"[PARALLEL WEB ERROR] {e}")
            return None
        
        async def api_task():
            """API research task."""
            if _external_research_allowed("api"):
                return await asyncio.to_thread(APIResearch.query, query, intent)
            return None
        
        # Execute all tasks in parallel
        logger.info(f"[RESEARCH] Starting parallel search for: '{query}'")
        results = await asyncio.gather(local_task(), web_task(), api_task(), return_exceptions=True)
        
        # Process results (filter None and exceptions)
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                source_name = ["local", "web", "api"][i]
                logger.warning(f"[RESEARCH] {source_name} task failed: {result}")
            elif result and isinstance(result, ResearchResult):
                valid_results.append(result)
        
        # Select best result based on confidence and source priority
        if valid_results:
            # Sort by confidence (descending)
            valid_results.sort(key=lambda r: r.confidence, reverse=True)
            best_result = _research_source_label(valid_results[0], "parallel_research")
            
            # Calculate total latency
            total_latency = int((time.time() - start_time) * 1000)
            best_result.latency_ms = total_latency
            
            # Record metrics
            metrics.record_success(best_result.source, total_latency)
            
            # Cache result
            cache_result(query, best_result)
            
            logger.info(f"[RESEARCH SUCCESS] Source: {best_result.source.value}, "
                       f"Confidence: {best_result.confidence:.2f}, "
                       f"Latency: {total_latency}ms")
            
            research_debug_logger.debug(f"[PARALLEL COMPLETE] Best source: {best_result.source.value}")
            
            return best_result
        
        # No results found from any source
        metrics.record_failure()
        logger.warning(f"[RESEARCH FAILED] No results for query: '{query}'")
        
        return ResearchResult(
            source=ResearchSource.NONE,
            content="Sorry, I was unable to find any reliable information using all available sources.",
            confidence=0.0,
            query=query,
            intent=intent,
            latency_ms=int((time.time() - start_time) * 1000)
        )
    
    finally:
        # Remove from active queries
        with active_queries_lock:
            active_queries.discard(query_key)


# ============================================================================
# PUBLIC API (BACKWARD COMPATIBILITY)
# ============================================================================

def get_local_research_data(query: str, intent: Optional[str] = None, *, allow_local_llm: bool = True) -> Dict[str, Any]:
    """Offline-safe local knowledge entry point.

    This function never calls WebResearch or external API providers. It is the
    correct fallback for /api/chat when the UI is in ANY mode but Globals has
    Web/API disabled or LOCAL_ONLY_MODE is active.
    """
    started = time.time()
    q = str(query or "").strip()
    if not q:
        return {
            "source": "local_none",
            "intent": intent or "unknown",
            "data": "",
            "snippet": "",
            "confidence": 0.0,
            "latency_ms": 0,
            "metadata": {"method": "local_only", "reason": "empty_query"},
        }

    try:
        resolved_intent = str(intent or classify_intent(q) or "question")
    except Exception:
        resolved_intent = str(intent or "question")

    try:
        if not bool(getattr(config, "LOCAL_DATA_ENABLED", True)):
            return {
                "source": "local_disabled",
                "intent": resolved_intent,
                "data": "Local data is disabled by LOCAL_DATA_ENABLED=False.",
                "snippet": "Local data is disabled by LOCAL_DATA_ENABLED=False.",
                "confidence": 0.0,
                "latency_ms": int((time.time() - started) * 1000),
                "metadata": {"method": "local_only", "local_data_enabled": False},
            }

        original_llm_flag = None
        if not allow_local_llm:
            try:
                original_llm_flag = getattr(config, "LOCAL_LLM_API", None)
                setattr(config, "LOCAL_LLM_API", False)
            except Exception:
                original_llm_flag = None
        try:
            result = LocalResearch.search(q, resolved_intent)
        finally:
            if original_llm_flag is not None:
                try:
                    setattr(config, "LOCAL_LLM_API", original_llm_flag)
                except Exception:
                    pass

        if result and isinstance(result, ResearchResult):
            result = _research_source_label(result, "local_only")
            out = result.to_dict()
            out.setdefault("latency_ms", int((time.time() - started) * 1000))
            out.setdefault("metadata", {})
            if isinstance(out.get("metadata"), dict):
                out["metadata"].setdefault("method", "local_only")
                out["metadata"]["web_allowed"] = False
                out["metadata"]["external_api_allowed"] = False
            return out

        return {
            "source": "local_none",
            "intent": resolved_intent,
            "data": "",
            "snippet": "",
            "confidence": 0.0,
            "latency_ms": int((time.time() - started) * 1000),
            "metadata": {"method": "local_only", "web_allowed": False, "external_api_allowed": False},
        }
    except Exception as e:
        logger.error(f"[LOCAL RESEARCH ERROR] {e}")
        return {
            "source": "local_error",
            "intent": resolved_intent,
            "data": f"Local research failed: {e}",
            "snippet": f"Local research failed: {e}",
            "confidence": 0.0,
            "latency_ms": int((time.time() - started) * 1000),
            "metadata": {"method": "local_only", "error": str(e)},
        }


def get_research_data(query: str) -> Dict[str, Any]:
    """
    Main entry point for research operations (backward compatible).
    
    This function maintains backward compatibility with existing code
    while leveraging the new parallel research infrastructure.
    
    Args:
        query: User query
        
    Returns:
        Dictionary with research results in legacy format
    """
    logger.info(f"[RESEARCH] Query: '{query}'")
    research_debug_logger.debug(f"[ENTRY POINT] get_research_data called with: {query}")
    
    try:
        # Run parallel research
        result = asyncio.run(parallel_research(query))
        
        # Convert to legacy dictionary format
        return result.to_dict()
    
    except Exception as e:
        logger.error(f"[RESEARCH ERROR] {e}")
        research_debug_logger.error(f"[FATAL ERROR] {str(e)}")
        metrics.record_failure()
        
        return {
            "source": "error",
            "intent": "unknown",
            "data": f"Research failed: {str(e)}",
            "confidence": 0.0
        }


# ============================================================================
# UTILITY FUNCTIONS (BACKWARD COMPATIBILITY)
# ============================================================================

def fetch_image_url(query: str) -> Optional[str]:
    """
    Fetch image URL from DuckDuckGo image search.
    
    Args:
        query: Image search query
        
    Returns:
        Image URL if found, None otherwise
    """
    try:
        import urllib.request
        import urllib.parse
        
        q = urllib.parse.quote_plus(query)
        url = f"https://duckduckgo.com/?q={q}&iax=images&ia=images"
        
        req = urllib.request.Request(url, headers={"User-Agent": "SarahMemoryBot/8.0"})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SHORT) as response:
            page = response.read().decode("utf-8", errors="ignore")
        
        # Check for CAPTCHA
        if WebResearch._is_ddg_blocked(page):
            logger.warning("[IMAGE] DuckDuckGo CAPTCHA detected")
            return None
        
        # Extract image URL
        match = re.search(r'imgurl=([^&\s]+)', page)
        if match:
            image_url = urllib.parse.unquote(match.group(1))
            logger.info(f"[IMAGE] Found: {image_url[:100]}")
            return image_url
        
        return None
    
    except Exception as e:
        logger.warning(f"[IMAGE ERROR] {e}")
        return None


def fetch_web_snippets(query: str, max_len: int = 800) -> str:
    """
    Fetch web snippets from DuckDuckGo.
    
    Args:
        query: Search query
        max_len: Maximum snippet length
        
    Returns:
        Concatenated snippets
    """
    try:
        import urllib.request
        import urllib.parse
        
        q = urllib.parse.quote_plus(query)
        url = f"https://duckduckgo.com/html/?q={q}"
        
        req = urllib.request.Request(url, headers={"User-Agent": "SarahMemoryBot/8.0"})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SHORT) as response:
            page = response.read().decode("utf-8", errors="ignore")
        
        # Check for CAPTCHA
        if WebResearch._is_ddg_blocked(page):
            return "[DDG_CAPTCHA]"
        
        # Extract text
        text = re.sub(r"<[^>]+>", " ", page)
        text = re.sub(r"\s+", " ", text).strip()
        
        if WebResearch._is_ddg_blocked(text):
            return "[DDG_CAPTCHA]"
        
        return text[:max_len] + ("..." if len(text) > max_len else "")
    
    except Exception as e:
        logger.warning(f"[SNIPPET ERROR] {e}")
        return ""


# ============================================================================
# DATABASE INTEGRATION
# ============================================================================

def _ensure_response_table(db_path: Optional[str] = None):
    """
    Ensure response table exists in database.
    
    This is called automatically on module import to ensure
    database schema is up to date.
    """
    try:
        if db_path is None:
            db_path = os.path.join(config.DATASETS_DIR, "system_logs.db")
        
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
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
        
        logger.debug(f"[DB] Ensured table 'response' in {db_path}")
    
    except Exception as e:
        logger.warning(f"[DB] Failed to ensure 'response' table: {e}")


# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

def get_research_metrics() -> Dict[str, Any]:
    """Get current research performance metrics."""
    return metrics.get_stats()


def reset_research_metrics():
    """Reset performance metrics."""
    global metrics
    metrics = ResearchMetrics()
    logger.info("[METRICS] Reset performance metrics")


def log_research_stats():
    """Log current research statistics."""
    stats = metrics.get_stats()
    
    logger.info("=" * 60)
    logger.info("RESEARCH ENGINE STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Total Queries:     {stats['total_queries']}")
    logger.info(f"Cache Hits:        {stats['cache_hits']} ({stats['cache_hit_rate']*100:.1f}%)")
    logger.info(f"Local Successes:   {stats['local_successes']}")
    logger.info(f"Web Successes:     {stats['web_successes']}")
    logger.info(f"API Successes:     {stats['api_successes']}")
    logger.info(f"Failures:          {stats['failures']}")
    logger.info(f"Success Rate:      {stats['success_rate']*100:.1f}%")
    logger.info(f"Avg Latency:       {stats['avg_latency_ms']:.0f}ms")
    logger.info(f"Uptime:            {stats['uptime_hours']:.2f} hours")
    logger.info("=" * 60)


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Ensure database tables exist
try:
    _ensure_response_table()
except Exception:
    pass

# Log initialization
logger.info("=" * 80)
logger.info("SarahMemory Research Engine v9.0.0 - World-Class Edition")
logger.info("=" * 80)
logger.info("Features Enabled:")
logger.info(f"  ✓ Local Database Research:  {config.LOCAL_DATA_ENABLED}")
logger.info(f"  ✓ Web Research:             {config.WEB_RESEARCH_ENABLED}")
logger.info(f"  ✓ API Research:             {config.API_RESEARCH_ENABLED}")
logger.info(f"  ✓ Wikipedia:                {config.WIKIPEDIA_RESEARCH_ENABLED}")
logger.info(f"  ✓ DuckDuckGo:               {config.DUCKDUCKGO_RESEARCH_ENABLED}")
logger.info(f"  ✓ Result Caching:           Enabled (TTL: {CACHE_TTL_SECONDS}s)")
logger.info(f"  ✓ Parallel Execution:       Enabled")
logger.info(f"  ✓ Research Path Logging:    {debug_log_path}")
logger.info("=" * 80)

# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Main API
    'get_research_data',
    'get_local_research_data',
    'parallel_research',
    
    # Research Classes
    'LocalResearch',
    'WebResearch',
    'APIResearch',
    
    # Data Structures
    'ResearchResult',
    'ResearchSource',
    'ResearchMetrics',
    
    # Utility Functions
    'fetch_image_url',
    'fetch_web_snippets',
    
    # Cache Management
    'get_cached_result',
    'cache_result',
    'clear_cache',
    
    # Performance Monitoring
    'get_research_metrics',
    'reset_research_metrics',
    'log_research_stats',
    
    # Constants
    'STATIC_FACTS',
]

# =============================================================================
# __SM_EVOLUTION_HELPERS__
# Helper for SarahMemoryEvolution: turn an issues payload into a research query
# while remaining compatible with existing get_research_data().
# =============================================================================

from typing import Any, Dict, List

def research_patch_suggestions_from_payload(payload: Dict[str, Any], max_chars: int = 6000) -> str:
    """Generate a compact research query from an Evolution payload and call get_research_data()."""
    try:
        issues = payload.get("issues") or []
        lines: List[str] = []
        for it in issues[:10]:
            lines.append(f"- {it.get('issue_id')}: {it.get('summary','')}")
        query = (
            "Python monkey patch strategy for the following issues. "
            "Constraints: no core edits, headless-safe, no new deps.\n"
            + "\n".join(lines)
        )
        query = query[:max_chars]
        r = get_research_data(query)  # type: ignore
        if isinstance(r, dict):
            import json as _json
            return _json.dumps(r, indent=2)[:15000]
        return str(r)[:15000]
    except Exception as e:
        return f"Research helper failed: {type(e).__name__}: {e}"

# =============================================================================
# SARAH_REM_RESEARCH_LANE_V1
# Bounded idle-time research lane for REM Sleep. Summary-only; no bulk download.
# =============================================================================

def rem_research_dream_tick(snapshot: Optional[Dict[str, Any]] = None, max_queries: int = 2) -> Dict[str, Any]:
    """Run a bounded REM research tick with hard timeout protection.

    This lane is intentionally narrow. It can produce short summaries for REM
    dashboards, but it must not bulk-download data or block the dream loop.
    """
    import concurrent.futures

    started = time.time()
    timeout_seconds = max(3.0, float(os.getenv("SARAH_REM_RESEARCH_TIMEOUT_SECONDS", "20") or 20))
    max_q = max(0, min(3, int(max_queries or 0)))
    bloat_guard = {
        "summary_only": True,
        "no_bulk_download": True,
        "max_queries": max_q,
        "timeout_seconds": timeout_seconds,
    }
    out: Dict[str, Any] = {
        "ok": True,
        "lane": "research",
        "web_allowed": bool(getattr(config, "WEB_RESEARCH_ENABLED", False) or getattr(config, "API_RESEARCH_ENABLED", False)),
        "queries_run": 0,
        "results": [],
        "bloat_guard": bloat_guard,
    }

    if max_q <= 0:
        out.update({"status": "skipped", "reason": "max_queries=0"})
        out["duration_ms"] = int((time.time() - started) * 1000)
        return out

    if not out["web_allowed"]:
        out.update({"status": "disabled", "reason": "Research disabled by governance flags"})
        out["duration_ms"] = int((time.time() - started) * 1000)
        return out

    queries = [
        "SarahMemory self-study code map improvement strategy",
        "safe bounded idle-time learning without database bloat",
        "governed autonomous agent sandbox rollback policy",
    ][:max_q]

    def _one_query(query: str) -> Dict[str, Any]:
        try:
            # Prefer the existing API research path because it respects provider flags.
            api = send_to_api(
                query,
                provider="auto",
                intent="research",
                tone="technical",
                complexity="adult",
                max_tokens=300,
                temperature=0.2,
            )
            data = str((api or {}).get("data") or "").strip()
            if len(data) > 900:
                data = data[:900].rstrip() + "..."
            return {
                "query": query,
                "source": str((api or {}).get("source") or "api_auto"),
                "summary": data or str((api or {}).get("error") or "No summary returned."),
                "web_allowed": True,
                "timed_out": False,
            }
        except Exception as exc:
            return {"query": query, "source": "research_error", "summary": str(exc), "web_allowed": True, "timed_out": False}

    for query in queries:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_one_query, query)
        try:
            item = future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            item = {
                "query": query,
                "source": "timeout_guard",
                "summary": f"REM research query exceeded {timeout_seconds:.1f}s timeout and was skipped.",
                "web_allowed": True,
                "timed_out": True,
            }
        finally:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        out["results"].append(item)
        out["queries_run"] = int(out.get("queries_run") or 0) + 1

    out["duration_ms"] = int((time.time() - started) * 1000)
    return out

# ====================================================================
# END OF SarahMemoryResearch.py v9.0.0
# ====================================================================
