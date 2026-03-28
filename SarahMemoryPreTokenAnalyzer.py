"""--==The SarahMemory Project==--
File: SarahMemoryPreTokenAnalyzer.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2026-03-28
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
"""
# Purpose:
#   Governed pre-token semantic analysis, symbolic compression, ambiguity control,
#   clarification-state preservation, and compact downstream payload preparation
#   for SarahMemory AiOS.
#
# Core mission:
#   Reduce token waste BEFORE local models, 3rd-party local models, or online APIs
#   ever see the user's full prompt.
#
# Design rules:
#   - Local-first, deterministic, auditable.
#   - Does NOT execute actions and does NOT answer the user directly.
#   - Works stand-alone even if AdvCU / SentenceTransformers are unavailable.
#   - When available, borrows AdvCU intent classification, command parsing,
#     semantic packet shaping, helper payload shaping, and embeddings.
#   - Maintains conceptual-query clarification state instead of losing task memory.
#   - Uses multiple fallback tiers so the module scales across consumer,
#     commercial, industrial, and mission-critical environments.
#
# Position in framework:
#   Context Packet / Cognitive Governance
#   -> Pre-Token Semantic Analysis
#   -> AdvCU semantic compression / helper payloads
#   -> Neuron routing
#   -> downstream deterministic or helper-model execution

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PROJECT_VERSION = "8.0.0"
MODULE_NAME = "SarahMemoryPreTokenAnalyzer"

# ---------------------------------------------------------------------------
# Lazy external contracts (safe / non-fatal)
# ---------------------------------------------------------------------------

_CONFIG = None
_ADVCU_CACHE: Optional[Dict[str, Any]] = None
_ST_MODEL = None
_ST_LOCK = threading.RLock()
_ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}
_ANALYSIS_CACHE_LOCK = threading.RLock()
_ANALYSIS_CACHE_MAX = 256


def _get_config() -> Any:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG
    try:
        import SarahMemoryGlobals as _cfg  # type: ignore
        _CONFIG = _cfg
        return _CONFIG
    except Exception:
        class _FallbackConfig:
            BASE_DIR = os.getcwd()
            DATA_DIR = os.path.join(BASE_DIR, "data")
            DATASETS_DIR = os.path.join(DATA_DIR, "memory", "datasets")
            DEBUG_MODE = False
            SAFE_MODE = False
            LOCAL_ONLY_MODE = False
            RUN_MODE = "local"
            DEVICE_MODE = "local_agent"
            DEVICE_PROFILE = "Standard"
        _CONFIG = _FallbackConfig()
        return _CONFIG


def _get_advcu_contracts() -> Dict[str, Any]:
    """
    Lazy-load AdvCU surfaces to avoid hard import cycles.
    """
    global _ADVCU_CACHE
    if _ADVCU_CACHE is not None:
        return _ADVCU_CACHE

    contracts: Dict[str, Any] = {
        "available": False,
        "classify_intent": None,
        "classify_intent_with_confidence": None,
        "parse_command": None,
        "build_semantic_packet": None,
        "build_helper_payload": None,
        "embed_text": None,
        "tokenize": None,
    }
    try:
        import SarahMemoryAdvCU as _advcu  # type: ignore
        contracts.update(
            {
                "available": True,
                "classify_intent": getattr(_advcu, "classify_intent", None),
                "classify_intent_with_confidence": getattr(_advcu, "classify_intent_with_confidence", None),
                "parse_command": getattr(_advcu, "parse_command", None),
                "build_semantic_packet": getattr(_advcu, "build_semantic_packet", None),
                "build_helper_payload": getattr(_advcu, "build_helper_payload", None),
                "embed_text": getattr(_advcu, "embed_text", None),
                "tokenize": getattr(_advcu, "tokenize", None),
            }
        )
    except Exception:
        pass

    _ADVCU_CACHE = contracts
    return _ADVCU_CACHE


def _get_local_st() -> Any:
    """
    Optional local SentenceTransformer loader.
    - Local-files-only when possible.
    - Never fatal.
    """
    global _ST_MODEL
    if _ST_MODEL is not None:
        return _ST_MODEL
    with _ST_LOCK:
        if _ST_MODEL is not None:
            return _ST_MODEL
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception:
            _ST_MODEL = None
            return None

        cfg = _get_config()
        offline = bool(
            getattr(cfg, "SAFE_MODE", False)
            or getattr(cfg, "LOCAL_ONLY_MODE", False)
            or os.getenv("HF_HUB_OFFLINE") == "1"
            or os.getenv("TRANSFORMERS_OFFLINE") == "1"
        )

        candidates: List[str] = []
        try:
            resolve_fn = getattr(cfg, "resolve_model", None)
            if callable(resolve_fn):
                res = resolve_fn("embeddings", text="", meta=None, models_dir=getattr(cfg, "MODELS_DIR", None)) or {}
                selected = res.get("selected")
                fallbacks = res.get("fallbacks") or []
                candidates = [str(x) for x in ([selected] + list(fallbacks)) if x]
        except Exception:
            candidates = []

        try:
            models_dir = getattr(cfg, "MODELS_DIR", os.path.join(getattr(cfg, "BASE_DIR", os.getcwd()), "data", "models"))
        except Exception:
            models_dir = os.path.join(os.getcwd(), "data", "models")

        for repo in candidates:
            try:
                local_dir_1 = os.path.join(models_dir, repo)
                local_dir_2 = os.path.join(models_dir, repo.replace("/", "_"))
                if os.path.isdir(local_dir_1):
                    _ST_MODEL = SentenceTransformer(local_dir_1, local_files_only=bool(offline))
                    return _ST_MODEL
                if os.path.isdir(local_dir_2):
                    _ST_MODEL = SentenceTransformer(local_dir_2, local_files_only=bool(offline))
                    return _ST_MODEL
                _ST_MODEL = SentenceTransformer(repo, local_files_only=bool(offline))
                return _ST_MODEL
            except Exception:
                continue

        _ST_MODEL = None
        return None


# ---------------------------------------------------------------------------
# Thresholds / weights / limits
# ---------------------------------------------------------------------------

FULL_COMPRESS_THRESHOLD = 0.86
PARTIAL_COMPRESS_THRESHOLD = 0.66
CLARIFY_THRESHOLD = 0.40
MIN_ALIGNMENT_FOR_FULL = 0.54
MIN_ALIGNMENT_FOR_PARTIAL = 0.38

MAX_TEXT_LEN = 16000
MAX_CLARIFICATION_CHOICES = 5
MAX_ATTRIBUTES = 8
MAX_SAFE_SEGMENTS = 6
MAX_SYMBOL_LINES = 12
MAX_COMPACT_ATTRS = 6
MAX_RAW_FALLBACK_CHARS = 480
MAX_SUMMARY_CHARS = 180
MAX_CACHE_KEY_CHARS = 256

FALLBACK_FULL = 1
FALLBACK_PARTIAL = 2
FALLBACK_RAW_HOLD = 3
FALLBACK_BOUNDED_OPTIONS = 4
FALLBACK_SAFE_GENERIC = 5

# ---------------------------------------------------------------------------
# Core vocab / heuristics
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "get",
    "give", "hello", "hey", "hi", "i", "in", "into", "is", "it", "me", "my",
    "of", "on", "or", "please", "right", "show", "tell", "than", "that", "the",
    "this", "to", "today", "turn", "up", "what", "with", "would", "write", "you",
    "your", "their", "our", "we", "they", "them", "his", "her", "hers", "him",
}

AMBIGUOUS_REFERENTS = {"it", "that", "this", "those", "these", "one", "ones", "thing", "latest", "recent", "last"}

ACTION_KEYWORDS = {
    "open", "close", "launch", "start", "stop", "run", "move", "copy", "delete",
    "remove", "rename", "create", "write", "save", "download", "upload", "set",
    "turn", "switch", "enable", "disable", "install", "uninstall", "erase", "make",
    "fix", "update", "edit", "export", "build", "generate", "clear", "empty",
}

QUESTION_KEYWORDS = {
    "what", "when", "where", "who", "why", "how", "temperature", "forecast",
    "status", "diagnostics", "is", "are", "does", "do", "can", "could", "should",
}

WEATHER_KEYWORDS = {
    "weather", "temperature", "forecast", "rain", "snow", "humid", "humidity",
    "wind", "conditions", "degrees", "temp", "climate",
}

DOC_KEYWORDS = {
    "document", "doc", "word", "report", "letter", "essay", "pdf", "spreadsheet",
    "slides", "presentation", "write", "draft", "edit", "export",
}

SYSTEM_KEYWORDS = {
    "diagnostics", "diagnostic", "gpu", "cpu", "ram", "vram", "health", "status",
    "thermal", "temperature", "overheating", "service", "system", "performance",
}

DRIVER_KEYWORDS = {
    "driver", "keyboard", "mouse", "rgb", "bluetooth", "wifi", "audio", "mic",
    "printer", "device", "usb", "caps", "capslock", "caps_lock",
}

NETWORK_KEYWORDS = {
    "network", "sarahnet", "remote", "node", "broker", "sync", "mesh", "cloud",
    "lan", "wan", "vpn", "dns",
}

CREATIVE_KEYWORDS = {
    "image", "picture", "draw", "music", "song", "lyrics", "video", "avatar",
    "art", "paint", "render", "animate", "animation",
}

RESEARCH_KEYWORDS = {
    "research", "browser", "search", "look up", "lookup", "find", "fetch", "scrape",
    "web", "website", "article", "news",
}

EMAIL_KEYWORDS = {"email", "emails", "mail", "spam", "unsubscribe", "inbox", "trash"}
SCHEDULE_KEYWORDS = {"daily", "weekly", "monthly", "schedule", "remind", "tomorrow", "today", "tonight", "every"}

SAFE_COMMON_ABBREVIATIONS = {
    "ai": "artificial_intelligence",
    "api": "application_programming_interface",
    "cpu": "central_processing_unit",
    "gpu": "graphics_processing_unit",
    "ram": "random_access_memory",
    "vram": "video_random_access_memory",
    "pdf": "portable_document_format",
    "rgb": "red_green_blue",
    "usb": "universal_serial_bus",
    "wifi": "wireless_network",
    "lan": "local_area_network",
    "wan": "wide_area_network",
    "tts": "text_to_speech",
    "stt": "speech_to_text",
    "ui": "user_interface",
    "ux": "user_experience",
    "sms": "short_message_service",
    "db": "database",
    "sql": "structured_query_language",
    "llm": "large_language_model",
    "slm": "small_language_model",
    "lm": "language_model",
    "plc": "programmable_logic_controller",
    "cnc": "computer_numerical_control",
    "cad": "computer_aided_design",
    "cam": "computer_aided_manufacturing",
}

US_STATE_ABBREVIATIONS = {
    "AL": ["Alabama"], "AK": ["Alaska"], "AZ": ["Arizona"], "AR": ["Arkansas"],
    "CA": ["California"], "CO": ["Colorado"], "CT": ["Connecticut"], "DE": ["Delaware"],
    "FL": ["Florida"], "GA": ["Georgia"], "HI": ["Hawaii"], "ID": ["Idaho"],
    "IL": ["Illinois"], "IN": ["Indiana"], "IA": ["Iowa"], "KS": ["Kansas"],
    "KY": ["Kentucky"], "LA": ["Louisiana", "Los Angeles, California"], "ME": ["Maine"],
    "MD": ["Maryland"], "MA": ["Massachusetts"], "MI": ["Michigan"], "MN": ["Minnesota"],
    "MS": ["Mississippi"], "MO": ["Missouri"], "MT": ["Montana"], "NE": ["Nebraska"],
    "NV": ["Nevada"], "NH": ["New Hampshire"], "NJ": ["New Jersey"], "NM": ["New Mexico"],
    "NY": ["New York"], "NC": ["North Carolina"], "ND": ["North Dakota"], "OH": ["Ohio"],
    "OK": ["Oklahoma"], "OR": ["Oregon"], "PA": ["Pennsylvania"], "RI": ["Rhode Island"],
    "SC": ["South Carolina"], "SD": ["South Dakota"], "TN": ["Tennessee"], "TX": ["Texas"],
    "UT": ["Utah"], "VT": ["Vermont"], "VA": ["Virginia"], "WA": ["Washington"],
    "WV": ["West Virginia"], "WI": ["Wisconsin"], "WY": ["Wyoming"], "DC": ["District of Columbia"],
}

LEXICAL_AMBIGUITIES = {
    "edge": {"slot": "software_or_geometry", "candidates": ["Microsoft Edge", "literal edge"]},
    "word": {"slot": "software_or_term", "candidates": ["Microsoft Word", "generic word/document term"]},
    "plant": {"slot": "site_or_lifeform", "candidates": ["industrial facility", "living plant"]},
    "latest": {"slot": "scope", "candidates": ["latest file", "latest model", "latest message", "latest result"]},
}

DOMAIN_KEYWORD_MAP = {
    "WEATHER": WEATHER_KEYWORDS,
    "DOC": DOC_KEYWORDS,
    "SYS": SYSTEM_KEYWORDS,
    "DRV": DRIVER_KEYWORDS,
    "NET": NETWORK_KEYWORDS,
    "CRT": CREATIVE_KEYWORDS,
    "RSH": RESEARCH_KEYWORDS,
    "EMAIL": EMAIL_KEYWORDS,
    "SCHED": SCHEDULE_KEYWORDS,
}

RISKY_ACTION_PATTERNS = (
    "delete", "erase", "format", "shutdown", "reboot", "install", "uninstall",
    "move", "copy", "rename", "driver", "wipe", "destroy", "flash", "reimage",
)

OPAQUE_CHAIN_RE = re.compile(r"\b(?:[A-Z]\.){4,}[A-Z]?\b")
UPPER_TOKEN_RE = re.compile(r"\b[A-Z]{2,8}\b")
WORD_RE = re.compile(r"[A-Za-z0-9_+#/-]+")
SPACE_RE = re.compile(r"\s+")
TRAILING_LOC_RE = re.compile(r"\bin\s+([A-Za-z][A-Za-z\s,.-]{1,64})$", re.I)
NUMERIC_TOKEN_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")
TIME_SCOPE_RE = re.compile(r"\b(today|tomorrow|tonight|this morning|this afternoon|this evening|next week|next month|later)\b", re.I)

# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------


def _context_bool(context_packet: Optional[Dict[str, Any]], *keys: str, default: bool = False) -> bool:
    context_packet = context_packet if isinstance(context_packet, dict) else {}
    meta = context_packet.get("meta") if isinstance(context_packet.get("meta"), dict) else {}
    for key in keys:
        value = context_packet.get(key, meta.get(key, None))
        if isinstance(value, bool):
            return value
        if value is None:
            continue
        try:
            sval = str(value).strip().lower()
        except Exception:
            continue
        if sval in {"1", "true", "yes", "on"}:
            return True
        if sval in {"0", "false", "no", "off"}:
            return False
    return default


def _use_fast_local_path(context_packet: Optional[Dict[str, Any]] = None) -> bool:
    if _context_bool(context_packet, "pretoken_force_advcu", "pta_force_advcu", default=False):
        return False
    if _context_bool(context_packet, "pretoken_fast_only", "pta_fast_only", default=False):
        return True
    env = str(os.getenv("SARAH_PRETOKEN_FAST_LOCAL", "auto") or "auto").strip().lower()
    if env in {"0", "false", "no", "off"}:
        return False
    return True


def _alignment_embeddings_enabled(context_packet: Optional[Dict[str, Any]] = None) -> bool:
    if _context_bool(context_packet, "pretoken_force_alignment", "pta_force_alignment", default=False):
        return True
    if _context_bool(context_packet, "pretoken_disable_alignment", "pta_disable_alignment", default=False):
        return False
    mode = str(os.getenv("SARAH_PRETOKEN_ALIGNMENT_MODE", "local_only") or "local_only").strip().lower()
    return mode in {"full", "embeddings", "hybrid"}


def _empty_advcu_contracts() -> Dict[str, Any]:
    return {
        "available": False,
        "classify_intent": None,
        "classify_intent_with_confidence": None,
        "parse_command": None,
        "build_semantic_packet": None,
        "build_helper_payload": None,
        "embed_text": None,
        "tokenize": None,
    }


def _empty_advcu_bundle() -> Dict[str, Any]:
    return {
        "available": False,
        "intent": "",
        "intent_confidence": 0.0,
        "subject": "",
        "attributes": [],
        "output_type": "",
        "query_type": "",
        "likely_lane": "",
        "semantic_summary": "",
        "module_hints": [],
        "helper_payload": {},
        "semantic_packet": {},
        "parsed_command": {},
        "likely_domain": "",
        "math_expr": None,
    }


def _local_route_snapshot(normalized_text: str, lowered: str, words: Sequence[str], domain_scores: Dict[str, float]) -> Dict[str, Any]:
    text_words = {w.lower() for w in words}
    has_action = _contains_any(lowered, ACTION_KEYWORDS)
    has_question = lowered.endswith("?") or _contains_any(lowered, QUESTION_KEYWORDS)
    has_schedule = _contains_any(lowered, SCHEDULE_KEYWORDS)

    domain = _best_domain(domain_scores)
    lane = _detect_lane(domain, lowered)
    query_type = _detect_query_type(lowered, lane)
    output_type = _detect_output_type(lowered, domain)
    subject = _extract_subject(words, domain)
    attributes = _extract_attributes(words, subject)
    override_strength = 0.42
    force_override = False

    if _contains_any(lowered, WEATHER_KEYWORDS) and ("temperature" in lowered or "forecast" in lowered or "weather" in lowered):
        domain = "WEATHER"
        lane = "answer"
        query_type = "factual_weather_forecast" if ("forecast" in lowered or "tomorrow" in lowered or "next" in lowered) else "factual_weather_current"
        output_type = "text_answer"
        subject = "forecast" if "forecast" in lowered else "temperature"
        attributes = [a for a in attributes if a not in {"give", "me"}]
        override_strength = 0.96
        force_override = True
    elif any(x in lowered for x in ("keyboard", "caps lock", "caps_lock", "rgb", "printer", "mouse", "wifi", "bluetooth", "mic", "audio driver", "device driver")):
        domain = "DRV"
        lane = "action"
        query_type = "device_control"
        output_type = "action_result"
        if "keyboard" in lowered:
            subject = "keyboard"
        elif "printer" in lowered:
            subject = "printer"
        elif "mouse" in lowered:
            subject = "mouse"
        elif "wifi" in lowered:
            subject = "wifi"
        else:
            subject = subject if subject not in {"request", "turn"} else "device"
        attrs = []
        if "rgb" in lowered:
            attrs.append("rgb")
        color = _extract_color(lowered)
        if color:
            attrs.append(color)
        if "caps lock" in lowered or "caps_lock" in lowered or "caps" in text_words:
            attrs.append("caps_lock")
        if " off" in lowered or lowered.endswith("off"):
            attrs.append("off")
        attributes = list(dict.fromkeys(attrs + [a for a in attributes if a not in {"turn", "sure", "make"}]))[:MAX_ATTRIBUTES]
        override_strength = 0.97
        force_override = True
    elif any(x in lowered for x in ("unsubscribe", "spam", "inbox", "mail", "email", "trash")):
        domain = "EMAIL"
        lane = "action" if (has_action or "unsubscribe" in lowered or "empty" in lowered or has_schedule) else "answer"
        query_type = "email_automation" if (has_action or has_schedule) else "email_request"
        output_type = "action_result" if lane == "action" else "text_answer"
        subject = "email"
        attrs = []
        for cue in ("unsubscribe", "spam", "empty", "trash", "daily", "weekly", "monthly"):
            if cue in lowered:
                attrs.append(cue)
        attributes = list(dict.fromkeys(attrs + attributes))[:MAX_ATTRIBUTES]
        override_strength = 0.95
        force_override = True
    elif any(x in lowered for x in ("open word", "open excel", "open powerpoint", "write report", "write letter", "document", "report", "presentation", "slides", "spreadsheet", "pdf")):
        domain = "DOC"
        lane = "action"
        query_type = "document_action"
        output_type = "document"
        if "word" in lowered:
            subject = "word"
        elif "report" in lowered:
            subject = "report"
        elif "presentation" in lowered or "slides" in lowered:
            subject = "presentation"
        elif "spreadsheet" in lowered or "excel" in lowered:
            subject = "spreadsheet"
        override_strength = 0.86
        force_override = True
    elif any(x in lowered for x in ("diagnostics", "diagnostic", "gpu", "cpu", "thermal", "overheating", "ram", "vram")):
        domain = "SYS"
        lane = "system"
        query_type = "system_status"
        output_type = "text_answer" if not has_action else "action_result"
        override_strength = 0.84
        force_override = True
    elif any(x in lowered for x in ("sarahnet", "node", "mesh", "broker", "sync", "remote")):
        domain = "NET"
        lane = "network"
        query_type = "network_operation"
        output_type = "action_result"
        override_strength = 0.82
        force_override = True
    elif any(x in lowered for x in ("draw", "image", "picture", "song", "music", "video", "avatar", "render", "animate")):
        domain = "CRT"
        lane = "creative"
        query_type = "creative_request"
        output_type = "artifact"
        override_strength = 0.80
        force_override = True
    elif any(x in lowered for x in ("research", "search", "look up", "lookup", "browser", "article", "news", "website")):
        domain = "RSH"
        lane = "answer"
        query_type = "research_request"
        output_type = "text_answer"
        override_strength = 0.72
        force_override = True
    elif has_action:
        domain = "ACT"
        lane = "action"
        query_type = "action_request"
        output_type = "action_result"
        override_strength = 0.70
        force_override = True
    elif has_question:
        domain = "QRY" if domain == "GEN" else domain
        lane = "answer"
        query_type = "question"
        output_type = "text_answer"
        override_strength = max(override_strength, 0.58)

    return {
        "likely_domain": domain,
        "likely_lane": lane,
        "query_type": query_type,
        "output_type": output_type,
        "subject": subject,
        "attributes": list(attributes)[:MAX_ATTRIBUTES],
        "action_detected": bool(has_action),
        "has_schedule": bool(has_schedule),
        "override_strength": float(_clamp(override_strength)),
        "force_override": bool(force_override),
    }


def _should_consult_advcu(normalized_text: str, local_profile: Dict[str, Any], ambiguities: Sequence[Dict[str, Any]], context_packet: Optional[Dict[str, Any]] = None) -> bool:
    if _context_bool(context_packet, "pretoken_force_advcu", "pta_force_advcu", default=False):
        return True
    if _context_bool(context_packet, "pretoken_disable_advcu", "pta_disable_advcu", default=False):
        return False
    if not _use_fast_local_path(context_packet):
        return True
    if OPAQUE_CHAIN_RE.search(normalized_text or "") and len(normalized_text or "") <= 96:
        return False
    domain = str(local_profile.get("likely_domain") or "GEN")
    strength = float(local_profile.get("override_strength") or 0.0)
    lexical_only = all(isinstance(a, dict) and str(a.get("ambiguity_type") or "") in {"lexical", "scope"} for a in ambiguities) if ambiguities else True
    if strength >= 0.90 and domain in {"WEATHER", "DRV", "EMAIL"} and lexical_only:
        return False
    if strength >= 0.84 and domain in {"DOC", "SYS", "NET", "CRT", "RSH"} and lexical_only:
        return False
    if len(normalized_text) <= 96 and strength >= 0.74 and lexical_only:
        return False
    return True


def _merge_local_and_advcu(local_profile: Dict[str, Any], advcu_bundle: Dict[str, Any], domain_scores: Dict[str, float], lowered: str, words: Sequence[str]) -> Dict[str, Any]:
    advcu_bundle = advcu_bundle if isinstance(advcu_bundle, dict) else _empty_advcu_bundle()
    adv_domain = str(advcu_bundle.get("likely_domain") or "").strip().upper()
    adv_lane = str(advcu_bundle.get("likely_lane") or "").strip().lower()
    adv_query_type = str(advcu_bundle.get("query_type") or "").strip().lower()
    local_domain = str(local_profile.get("likely_domain") or _best_domain(domain_scores)).strip().upper() or "GEN"
    local_lane = str(local_profile.get("likely_lane") or _detect_lane(local_domain, lowered)).strip().lower() or "answer"
    local_strength = float(local_profile.get("override_strength") or 0.0)
    force_local = bool(local_profile.get("force_override"))

    domain = local_domain
    if not force_local:
        if local_domain in {"GEN", "QRY", "ACT"} and adv_domain and adv_domain not in {"GEN", "QRY", "ACT"}:
            domain = adv_domain
        elif adv_domain and adv_domain == local_domain:
            domain = adv_domain

    lane = local_lane
    if not force_local and domain in {"GEN", "QRY", "ACT"} and adv_lane in {"answer", "action", "creative", "system", "network"}:
        lane = adv_lane

    if domain in {"WEATHER"}:
        lane = "answer"
    elif domain in {"DOC", "DRV", "EMAIL", "SCHED", "ACT"}:
        lane = "action"
    elif domain == "SYS":
        lane = "system"
    elif domain == "NET":
        lane = "network"
    elif domain == "CRT":
        lane = "creative"
    elif domain in {"RSH", "QRY"}:
        lane = "answer"

    query_type = str(local_profile.get("query_type") or "").strip().lower()
    if not query_type or (not force_local and query_type in {"general_request", "question", "command", "action_request"} and adv_query_type and adv_query_type not in {"statement", "question"}):
        query_type = adv_query_type or _detect_query_type(lowered, lane)
    if domain == "WEATHER" and query_type in {"statement", "question", "general_request"}:
        query_type = "factual_weather_forecast" if ("forecast" in lowered or "tomorrow" in lowered or "next" in lowered) else "factual_weather_current"
    elif domain == "DRV" and query_type in {"statement", "question", "general_request", "command"}:
        query_type = "device_control"
    elif domain == "EMAIL" and query_type in {"statement", "question", "general_request", "command"}:
        query_type = "email_automation" if ("unsubscribe" in lowered or "empty" in lowered or _contains_any(lowered, SCHEDULE_KEYWORDS)) else "email_request"
    elif domain == "DOC" and query_type in {"statement", "question", "general_request", "command"}:
        query_type = "document_action"
    elif lane == "action" and query_type in {"statement", "general_request"}:
        query_type = "action_request"

    output_type = _prefer_output_type(advcu_bundle.get("output_type"), str(local_profile.get("output_type") or _detect_output_type(lowered, domain)), domain)

    subject = str(local_profile.get("subject") or "").strip()
    if (not subject or subject in {"request", "turn", "open", "write", "make"}) and advcu_bundle.get("subject"):
        subject = str(advcu_bundle.get("subject") or "").strip()
    if not subject or subject in {"request", "turn", "open", "write", "make"}:
        subject = _extract_subject(words, domain)
    if domain == "DRV" and (not subject or subject in {"turn", "request"}):
        subject = "keyboard" if "keyboard" in lowered else "device"
    if domain == "EMAIL":
        subject = "email"

    local_attrs = list(local_profile.get("attributes") or [])
    adv_attrs = list(advcu_bundle.get("attributes") or []) if isinstance(advcu_bundle.get("attributes"), list) else []
    attributes = []
    for val in (local_attrs + adv_attrs + _extract_attributes(words, subject)):
        sval = str(val).strip().lower()
        if not sval or sval in STOPWORDS or sval == str(subject).strip().lower():
            continue
        if sval in {"sure", "make", "give", "me"}:
            continue
        if sval not in attributes:
            attributes.append(sval)
    attributes = attributes[:MAX_ATTRIBUTES]

    semantic_summary = str(advcu_bundle.get("semantic_summary") or "").strip()
    if force_local or not semantic_summary or local_strength >= 0.84:
        semantic_summary = _build_semantic_summary(lane, domain, subject, attributes, lowered)

    return {
        "likely_domain": domain,
        "likely_lane": lane,
        "query_type": query_type or _detect_query_type(lowered, lane),
        "output_type": output_type,
        "subject": subject or "request",
        "attributes": attributes,
        "semantic_summary": semantic_summary,
        "action_detected": bool(local_profile.get("action_detected") or _contains_any(lowered, ACTION_KEYWORDS)),
        "override_strength": local_strength,
        "force_override": force_local,
    }


def _build_estimated_downstream_raw_text(raw_text: str, semantic_summary: str, helper_payload: Optional[Dict[str, Any]] = None, semantic_packet: Optional[Dict[str, Any]] = None, clarification_question: str = "", ambiguities: Optional[Sequence[Dict[str, Any]]] = None) -> str:
    payload = {
        "user_text": _normalize_text(raw_text),
        "semantic_summary": _normalize_text(semantic_summary),
        "helper_payload": _minify_helper_payload(helper_payload or {}),
        "semantic_packet": semantic_packet or {},
        "clarification_question": _normalize_text(clarification_question),
        "ambiguities": [
            {
                "slot": str(a.get("slot") or ""),
                "raw": str(a.get("raw") or ""),
                "candidates": list(a.get("candidates") or [])[:MAX_CLARIFICATION_CHOICES],
            }
            for a in list(ambiguities or [])[:3]
            if isinstance(a, dict)
        ],
    }
    return _normalize_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False))


def _should_run_semantic_alignment(local_profile: Dict[str, Any], ambiguities: Sequence[Dict[str, Any]], context_packet: Optional[Dict[str, Any]] = None) -> bool:
    if _context_bool(context_packet, "pretoken_force_alignment", "pta_force_alignment", default=False):
        return True
    if _context_bool(context_packet, "pretoken_disable_alignment", "pta_disable_alignment", default=False):
        return False
    domain = str(local_profile.get("likely_domain") or "GEN")
    strength = float(local_profile.get("override_strength") or 0.0)
    if ambiguities:
        return True
    if domain in {"WEATHER", "DRV", "EMAIL", "DOC", "SYS", "NET", "CRT", "RSH"} and strength >= 0.78:
        return False
    return domain in {"GEN", "QRY", "ACT"}


def _default_alignment_score(raw_text: str, symbolic_packet_text: str, semantic_summary: str, likely_domain: str, likely_lane: str, ambiguities: Sequence[Dict[str, Any]]) -> float:
    raw_norm = _normalize_text(raw_text).lower()
    sym_norm = _normalize_text(symbolic_packet_text).lower()
    raw_tokens = set(_word_tokens(raw_norm))
    sym_tokens = set(_word_tokens(sym_norm))
    lexical = len(raw_tokens & sym_tokens) / max(1, len(raw_tokens | sym_tokens)) if raw_tokens else 0.0
    structural = 0.50
    if likely_domain in sym_norm:
        structural += 0.16
    if likely_lane in {"action", "answer", "creative", "system", "network"}:
        structural += 0.06
    if likely_domain == "WEATHER" and any(x in raw_norm for x in ("weather", "temperature", "forecast")):
        structural += 0.14
    if likely_domain == "DRV" and any(x in raw_norm for x in ("keyboard", "rgb", "caps")):
        structural += 0.18
    if likely_domain == "EMAIL" and any(x in raw_norm for x in ("unsubscribe", "spam", "trash")):
        structural += 0.18
    if likely_domain == "DOC" and any(x in raw_norm for x in ("word", "report", "document")):
        structural += 0.14
    if ambiguities:
        structural -= 0.18
    if semantic_summary:
        summary_tokens = set(_word_tokens(_normalize_text(semantic_summary).lower()))
        lexical = max(lexical, len(raw_tokens & summary_tokens) / max(1, len(raw_tokens | summary_tokens))) if raw_tokens else lexical
    return _clamp(0.42 * lexical + 0.58 * structural)


def analyze_text(raw_text: str, context_packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Primary entrypoint.

    Returns a deterministic analysis bundle suitable for:
    - SarahMemoryAdvCU semantic compression intake
    - SarahMemoryNeuron clarification state / query memory
    - SarahMemoryReply targeted clarification wording
    - SarahMemoryAPI local/helper compact prompt assembly
    """
    context_packet = context_packet if isinstance(context_packet, dict) else {}
    started_ts = time.time()

    raw_text = _coerce_text(raw_text)
    normalized_text = _normalize_text(raw_text)
    cache_key = _analysis_cache_key(normalized_text, context_packet)
    cached = _cache_get(cache_key)
    if cached is not None:
        result = copy.deepcopy(cached)
        result["cache_hit"] = True
        result["analysis_ts"] = started_ts
        result["elapsed_ms"] = round((time.time() - started_ts) * 1000.0, 3)
        return result

    query_id = str(context_packet.get("query_id") or context_packet.get("id") or _new_id("pta"))
    concept_group_id = str(
        context_packet.get("concept_group_id")
        or _get_nested(context_packet, ("meta", "concept_group_id"), "")
        or query_id
    )

    lowered = normalized_text.lower()
    words = _word_tokens(normalized_text)
    domain_scores = _score_domains(lowered, words)
    local_profile = _local_route_snapshot(normalized_text, lowered, words, domain_scores)

    abbreviation_hits, ambiguities, opaque_segments = _extract_abbreviation_findings(
        normalized_text=normalized_text,
        lowered=lowered,
        likely_domain=str(local_profile.get("likely_domain") or "GEN"),
        likely_lane=str(local_profile.get("likely_lane") or "answer"),
    )
    ambiguities.extend(_detect_reference_ambiguities(normalized_text, str(local_profile.get("likely_lane") or "answer"), str(local_profile.get("likely_domain") or "GEN"), context_packet))
    ambiguities.extend(_detect_scope_ambiguities(normalized_text, str(local_profile.get("likely_lane") or "answer"), str(local_profile.get("likely_domain") or "GEN"), context_packet))
    ambiguities.extend(_detect_action_ambiguities(normalized_text, str(local_profile.get("likely_lane") or "answer"), str(local_profile.get("likely_domain") or "GEN")))
    ambiguities.extend(_detect_execution_ambiguities(normalized_text, str(local_profile.get("likely_lane") or "answer"), str(local_profile.get("likely_domain") or "GEN")))
    ambiguities = _dedupe_ambiguities(ambiguities)

    consult_advcu = _should_consult_advcu(normalized_text, local_profile, ambiguities, context_packet)
    advcu_contracts = _empty_advcu_contracts()
    advcu_bundle = _empty_advcu_bundle()
    if consult_advcu:
        advcu_contracts = _get_advcu_contracts()
        advcu_bundle = _advcu_analyze(normalized_text, context_packet, advcu_contracts)

    merged_profile = _merge_local_and_advcu(local_profile, advcu_bundle, domain_scores, lowered, words)
    likely_domain = str(merged_profile.get("likely_domain") or "GEN")
    likely_lane = str(merged_profile.get("likely_lane") or "answer")
    query_type = str(merged_profile.get("query_type") or _detect_query_type(lowered, likely_lane))
    output_type = _prefer_output_type(merged_profile.get("output_type"), _detect_output_type(lowered, likely_domain), likely_domain)
    action_detected = bool(merged_profile.get("action_detected") or _contains_any(lowered, ACTION_KEYWORDS))

    subject = str(merged_profile.get("subject") or advcu_bundle.get("subject") or _extract_subject(words, likely_domain) or "request")
    attributes = list(merged_profile.get("attributes") or []) or list(advcu_bundle.get("attributes") or []) or _extract_attributes(words, subject)
    attributes = attributes[:MAX_ATTRIBUTES]
    semantic_summary = str(merged_profile.get("semantic_summary") or advcu_bundle.get("semantic_summary") or _build_semantic_summary(likely_lane, likely_domain, subject, attributes, normalized_text))

    known_vocab_score = _score_known_vocab(words, abbreviation_hits)
    context_anchor_score = _score_context_anchor(normalized_text, ambiguities)
    domain_match_score = float(domain_scores.get(likely_domain, max(domain_scores.values()) if domain_scores else 0.0))
    entity_resolution_score = _score_entity_resolution(ambiguities, abbreviation_hits)
    syntax_stability_score = _score_syntax_stability(normalized_text)
    repeat_history_score = _score_repeat_history(normalized_text, context_packet)
    ambiguity_score = _score_ambiguity(ambiguities, likely_lane)
    opaque_text_score = _score_opaque_text(normalized_text, opaque_segments)
    execution_risk_score = _score_execution_risk(lowered=lowered, likely_lane=likely_lane, ambiguities=ambiguities)
    advcu_confidence = _clamp(float(advcu_bundle.get("intent_confidence") or 0.0))
    local_override_strength = _clamp(float(merged_profile.get("override_strength") or 0.0))

    compression_confidence = _clamp(
        0.24 * context_anchor_score
        + 0.20 * domain_match_score
        + 0.16 * known_vocab_score
        + 0.10 * repeat_history_score
        + 0.10 * entity_resolution_score
        + 0.10 * syntax_stability_score
        + 0.08 * advcu_confidence
        + 0.12 * local_override_strength
        - 0.22 * ambiguity_score
        - 0.22 * execution_risk_score
        - 0.32 * opaque_text_score
    )

    maintainability_score = _score_maintainability(
        likely_lane=likely_lane,
        likely_domain=likely_domain,
        subject=subject,
        attributes=attributes,
        ambiguities=ambiguities,
    )

    compression_mode = _decide_compression_mode(
        compression_confidence=compression_confidence,
        ambiguity_score=ambiguity_score,
        opaque_text_score=opaque_text_score,
        execution_risk_score=execution_risk_score,
        action_detected=action_detected,
    )

    symbolic_packet_partial = _build_symbolic_packet_partial(
        normalized_text=normalized_text,
        lowered=lowered,
        likely_lane=likely_lane,
        likely_domain=likely_domain,
        query_type=query_type,
        subject=subject,
        attributes=attributes,
        output_type=output_type,
        ambiguities=ambiguities,
        abbreviation_hits=abbreviation_hits,
        compression_mode=compression_mode,
        advcu_bundle=advcu_bundle,
    )
    symbolic_packet_text = _packet_lines_to_text(symbolic_packet_partial)

    if _should_run_semantic_alignment(merged_profile, ambiguities, context_packet):
        if _alignment_embeddings_enabled(context_packet):
            semantic_alignment_score = _semantic_alignment_score(
                raw_text=normalized_text,
                symbolic_packet_text=symbolic_packet_text,
                semantic_summary=semantic_summary,
                advcu_bundle=advcu_bundle,
                contracts=advcu_contracts,
            )
        else:
            semantic_alignment_score = _default_alignment_score(
                raw_text=normalized_text,
                symbolic_packet_text=symbolic_packet_text,
                semantic_summary=semantic_summary,
                likely_domain=likely_domain,
                likely_lane=likely_lane,
                ambiguities=ambiguities,
            )
    else:
        semantic_alignment_score = _default_alignment_score(
            raw_text=normalized_text,
            symbolic_packet_text=symbolic_packet_text,
            semantic_summary=semantic_summary,
            likely_domain=likely_domain,
            likely_lane=likely_lane,
            ambiguities=ambiguities,
        )

    if not ambiguities and local_override_strength >= 0.84 and likely_domain in {"WEATHER", "DRV", "EMAIL", "DOC", "SYS", "NET", "CRT", "RSH"}:
        semantic_alignment_score = max(semantic_alignment_score, 0.84)

    if compression_mode == "full" and semantic_alignment_score < MIN_ALIGNMENT_FOR_FULL:
        compression_mode = "partial"
    elif compression_mode == "partial" and semantic_alignment_score < MIN_ALIGNMENT_FOR_PARTIAL:
        if ambiguities:
            compression_mode = "clarify_first"
        elif local_override_strength < 0.72 or likely_domain in {"GEN", "QRY", "ACT"}:
            compression_mode = "raw_only"

    clarification_required = bool(
        compression_mode == "clarify_first"
        or any(isinstance(a, dict) and not a.get("resolved") for a in ambiguities)
    )
    clarification_type, clarification_options = _build_clarification_shape(ambiguities, likely_lane, likely_domain)
    clarification_question = ""
    if clarification_required:
        clarification_question = build_clarification_question(
            {
                "ambiguities": ambiguities,
                "likely_lane": likely_lane,
                "likely_domain": likely_domain,
                "normalized_text": normalized_text,
                "symbolic_packet_partial": symbolic_packet_partial,
            }
        )

    fallback_level = _compute_fallback_level(compression_mode, clarification_required, opaque_text_score, clarification_options)
    finalized = finalize_symbolic_packet({"symbolic_packet_partial": symbolic_packet_partial, "ambiguities": ambiguities})

    compact_llm_payload = build_llm_compact_payload(
        {
            "query_id": query_id,
            "concept_group_id": concept_group_id,
            "normalized_text": normalized_text,
            "compression_mode": compression_mode,
            "likely_lane": likely_lane,
            "likely_domain": likely_domain,
            "query_type": query_type,
            "output_type": output_type,
            "subject": subject,
            "attributes": attributes,
            "semantic_summary": semantic_summary,
            "symbolic_packet_partial": symbolic_packet_partial,
            "resolved_packet": finalized.get("resolved_packet", []),
            "advcu_bundle": advcu_bundle,
            "ambiguities": ambiguities,
            "needs_clarification": clarification_required,
            "clarification_type": clarification_type,
            "clarification_question": clarification_question,
            "clarification_options": clarification_options,
            "fallback_level": fallback_level,
        },
        include_raw_fallback=(compression_mode in {"raw_only", "clarify_first"}),
    )

    compact_json_text = json.dumps(compact_llm_payload, separators=(",", ":"), ensure_ascii=False)
    compact_line_text = _compact_payload_to_line(compact_llm_payload)
    helper_text = _normalize_text(json.dumps(_get_nested(advcu_bundle, ("helper_payload",), {}) or {}, separators=(",", ":"), ensure_ascii=False))
    semantic_packet_text = _normalize_text(json.dumps(_get_nested(advcu_bundle, ("semantic_packet",), {}) or {}, separators=(",", ":"), ensure_ascii=False))
    raw_downstream_text = _build_estimated_downstream_raw_text(
        raw_text=normalized_text,
        semantic_summary=semantic_summary,
        helper_payload=advcu_bundle.get("helper_payload") or {},
        semantic_packet=advcu_bundle.get("semantic_packet") or {},
        clarification_question=clarification_question,
        ambiguities=ambiguities,
    )

    token_metrics = estimate_token_reduction(
        raw_text=normalized_text,
        symbolic_packet_text=symbolic_packet_text,
        compact_llm_text=compact_json_text,
        helper_payload=advcu_bundle.get("helper_payload") or {},
        compact_line_text=compact_line_text,
        semantic_packet_text=semantic_packet_text,
        raw_downstream_text=raw_downstream_text,
    )

    best_payload_text, best_payload_strategy = _select_best_payload_text(
        compact_json_text=compact_json_text,
        compact_line_text=compact_line_text,
        helper_text=helper_text,
        symbolic_packet_text=symbolic_packet_text,
        semantic_packet_text=semantic_packet_text,
        raw_text=normalized_text,
        needs_clarification=clarification_required,
        raw_downstream_text=raw_downstream_text,
    )

    downstream_bundle = {
        "best_payload_strategy": best_payload_strategy,
        "best_payload_text": best_payload_text,
        "compact_json_text": compact_json_text,
        "compact_line_text": compact_line_text,
        "helper_payload_text": helper_text,
        "symbolic_packet_text": symbolic_packet_text,
        "semantic_packet_text": semantic_packet_text,
        "raw_downstream_text": raw_downstream_text,
    }

    result = {
        "ok": True,
        "module": MODULE_NAME,
        "version": PROJECT_VERSION,
        "analysis_ts": started_ts,
        "elapsed_ms": round((time.time() - started_ts) * 1000.0, 3),
        "cache_hit": False,
        "query_id": query_id,
        "concept_group_id": concept_group_id,
        "pending_query_id": query_id,
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "compression_mode": compression_mode,
        "compression_confidence": round(compression_confidence, 6),
        "ambiguity_score": round(ambiguity_score, 6),
        "maintainability_score": round(maintainability_score, 6),
        "execution_risk_score": round(execution_risk_score, 6),
        "context_anchor_score": round(context_anchor_score, 6),
        "domain_match_score": round(domain_match_score, 6),
        "known_vocab_score": round(known_vocab_score, 6),
        "entity_resolution_score": round(entity_resolution_score, 6),
        "syntax_stability_score": round(syntax_stability_score, 6),
        "repeat_history_score": round(repeat_history_score, 6),
        "opaque_text_score": round(opaque_text_score, 6),
        "semantic_alignment_score": round(semantic_alignment_score, 6),
        "local_override_strength": round(local_override_strength, 6),
        "advcu_consulted": bool(consult_advcu),
        "likely_domain": likely_domain,
        "likely_lane": likely_lane,
        "query_type": query_type,
        "output_type": output_type,
        "subject": subject,
        "attributes": attributes,
        "semantic_summary": semantic_summary[:MAX_SUMMARY_CHARS],
        "action_detected": bool(action_detected),
        "known_abbreviations": abbreviation_hits,
        "ambiguities": ambiguities,
        "opaque_segments": opaque_segments,
        "safe_segments": _extract_safe_segments(normalized_text, ambiguities, opaque_segments),
        "symbolic_packet_partial": symbolic_packet_partial,
        "symbolic_packet_text": symbolic_packet_text,
        "resolved_packet": finalized.get("resolved_packet", []),
        "needs_clarification": bool(clarification_required),
        "clarification_required": bool(clarification_required),
        "clarification_type": clarification_type,
        "clarification_options": clarification_options,
        "clarification_question": clarification_question,
        "fallback_level": fallback_level,
        "state": _state_from_mode(compression_mode, clarification_required),
        "domain_scores": {k: round(v, 6) for k, v in domain_scores.items()},
        "context_packet_meta": _minimal_context_meta(context_packet),
        "token_metrics": token_metrics,
        "advcu_bundle": advcu_bundle,
        "compact_llm_payload": compact_llm_payload,
        "compact_llm_text": compact_json_text,
        "compact_llm_line": compact_line_text,
        "downstream_bundle": downstream_bundle,
    }

    _cache_put(cache_key, result)
    return result


def build_clarification_question(analysis_result: Dict[str, Any]) -> str:
    """
    Create a focused clarification prompt using unresolved slots only.
    """
    analysis_result = analysis_result if isinstance(analysis_result, dict) else {}
    ambiguities = analysis_result.get("ambiguities") if isinstance(analysis_result.get("ambiguities"), list) else []
    if not ambiguities:
        return "Please clarify the unresolved part of your request."

    primary = None
    for item in ambiguities:
        if isinstance(item, dict) and not item.get("resolved"):
            primary = item
            break
    if primary is None:
        primary = ambiguities[0] if ambiguities else {}
    if not isinstance(primary, dict):
        return "Please clarify the unresolved part of your request."

    slot = str(primary.get("slot") or "value").replace("_", " ")
    raw = str(primary.get("raw") or "that part")
    candidates = primary.get("candidates") if isinstance(primary.get("candidates"), list) else []
    trimmed = [str(c) for c in candidates[:MAX_CLARIFICATION_CHOICES] if str(c).strip()]
    if trimmed:
        numbered = [f"{i + 1}) {item}" for i, item in enumerate(trimmed)]
        prefix = "Please clarify" if primary.get("ambiguity_type") != "scope" else "Please narrow"
        return f"{prefix} the {slot} '{raw}'. Options: " + " | ".join(numbered)
    return f"Please clarify what you mean by '{raw}' for the {slot}."



def merge_clarification_answer(frame: Dict[str, Any], user_answer: str) -> Dict[str, Any]:
    """
    Merge a clarification response into an existing analysis frame.

    Clarification is an amendment to the same conceptual query, not a new query.
    """
    merged = copy.deepcopy(frame if isinstance(frame, dict) else {})
    user_answer = _normalize_text(_coerce_text(user_answer))
    merged.setdefault("clarification_history", [])
    merged["clarification_history"].append({"ts": time.time(), "answer": user_answer})
    ambiguities = merged.get("ambiguities") if isinstance(merged.get("ambiguities"), list) else []

    resolved_any = False
    for item in ambiguities:
        if not isinstance(item, dict) or item.get("resolved"):
            continue
        candidates = item.get("candidates") if isinstance(item.get("candidates"), list) else []
        chosen = _resolve_candidate_from_answer(candidates, user_answer)
        if chosen:
            item["resolved"] = True
            item["resolved_value"] = chosen
            item["resolved_ts"] = time.time()
            resolved_any = True

    unresolved_remaining = [a for a in ambiguities if isinstance(a, dict) and not a.get("resolved")]
    merged["ambiguities"] = ambiguities
    merged["unresolved_slots"] = unresolved_remaining
    merged["clarification_answer"] = user_answer

    if resolved_any and not unresolved_remaining:
        merged["needs_clarification"] = False
        merged["clarification_required"] = False
        merged["clarification_question"] = ""
        merged["compression_mode"] = "partial"
        merged["state"] = "clarified"
    elif unresolved_remaining:
        merged["needs_clarification"] = True
        merged["clarification_required"] = True
        merged["clarification_question"] = build_clarification_question(merged)
        merged["state"] = "awaiting_clarification"

    finalized = finalize_symbolic_packet(merged)
    merged["resolved_packet"] = finalized.get("resolved_packet", merged.get("symbolic_packet_partial", []))
    merged["state"] = finalized.get("state", merged.get("state", "parsed_partial"))

    compact_llm_payload = build_llm_compact_payload(merged, include_raw_fallback=bool(merged.get("needs_clarification")))
    compact_llm_text = json.dumps(compact_llm_payload, separators=(",", ":"), ensure_ascii=False)
    compact_llm_line = _compact_payload_to_line(compact_llm_payload)
    merged_raw_downstream = _build_estimated_downstream_raw_text(
        raw_text=str(merged.get("normalized_text") or merged.get("raw_text") or ""),
        semantic_summary=str(merged.get("semantic_summary") or ""),
        helper_payload=_get_nested(merged, ("advcu_bundle", "helper_payload"), {}) or {},
        semantic_packet=_get_nested(merged, ("advcu_bundle", "semantic_packet"), {}) or {},
        clarification_question=str(merged.get("clarification_question") or ""),
        ambiguities=merged.get("ambiguities") if isinstance(merged.get("ambiguities"), list) else [],
    )
    best_payload_text, best_payload_strategy = _select_best_payload_text(
        compact_json_text=compact_llm_text,
        compact_line_text=compact_llm_line,
        helper_text=_normalize_text(json.dumps(_get_nested(merged, ("advcu_bundle", "helper_payload"), {}) or {}, separators=(",", ":"), ensure_ascii=False)),
        symbolic_packet_text=_packet_lines_to_text(merged.get("resolved_packet") or merged.get("symbolic_packet_partial") or []),
        semantic_packet_text=_normalize_text(json.dumps(_get_nested(merged, ("advcu_bundle", "semantic_packet"), {}) or {}, separators=(",", ":"), ensure_ascii=False)),
        raw_text=str(merged.get("normalized_text") or merged.get("raw_text") or ""),
        needs_clarification=bool(merged.get("needs_clarification")),
        raw_downstream_text=merged_raw_downstream,
    )

    merged["compact_llm_payload"] = compact_llm_payload
    merged["compact_llm_text"] = compact_llm_text
    merged["compact_llm_line"] = compact_llm_line
    merged["downstream_bundle"] = {
        "best_payload_strategy": best_payload_strategy,
        "best_payload_text": best_payload_text,
        "compact_json_text": compact_llm_text,
        "compact_line_text": compact_llm_line,
        "helper_payload_text": _normalize_text(json.dumps(_get_nested(merged, ("advcu_bundle", "helper_payload"), {}) or {}, separators=(",", ":"), ensure_ascii=False)),
        "symbolic_packet_text": _packet_lines_to_text(merged.get("resolved_packet") or merged.get("symbolic_packet_partial") or []),
        "semantic_packet_text": _normalize_text(json.dumps(_get_nested(merged, ("advcu_bundle", "semantic_packet"), {}) or {}, separators=(",", ":"), ensure_ascii=False)),
        "raw_downstream_text": merged_raw_downstream,
    }
    merged["token_metrics"] = estimate_token_reduction(
        raw_text=str(merged.get("normalized_text") or merged.get("raw_text") or ""),
        symbolic_packet_text=_packet_lines_to_text(merged.get("resolved_packet") or merged.get("symbolic_packet_partial") or []),
        compact_llm_text=compact_llm_text,
        helper_payload=_get_nested(merged, ("advcu_bundle", "helper_payload"), {}) or {},
        compact_line_text=compact_llm_line,
        semantic_packet_text=_normalize_text(json.dumps(_get_nested(merged, ("advcu_bundle", "semantic_packet"), {}) or {}, separators=(",", ":"), ensure_ascii=False)),
        raw_downstream_text=merged_raw_downstream,
    )
    return merged



def finalize_symbolic_packet(analysis_result_or_frame: Dict[str, Any]) -> Dict[str, Any]:
    """
    Replace resolved ambiguities and produce the best available finalized symbolic packet.
    """
    frame = copy.deepcopy(analysis_result_or_frame if isinstance(analysis_result_or_frame, dict) else {})
    packet = frame.get("symbolic_packet_partial") if isinstance(frame.get("symbolic_packet_partial"), list) else []
    ambiguities = frame.get("ambiguities") if isinstance(frame.get("ambiguities"), list) else []

    resolved_map: Dict[str, str] = {}
    unresolved = []
    for item in ambiguities:
        if not isinstance(item, dict):
            continue
        slot = str(item.get("slot") or "value")
        raw = str(item.get("raw") or "")
        key = f"{slot}:{raw}".lower()
        if item.get("resolved") and item.get("resolved_value"):
            resolved_map[key] = str(item.get("resolved_value"))
        else:
            unresolved.append(item)

    resolved_packet: List[str] = []
    for line in packet:
        line = str(line)
        if line.startswith("AMBIG["):
            replacement = _replacement_for_ambiguity_line(line, ambiguities, resolved_map)
            if replacement:
                resolved_packet.append(replacement)
            elif unresolved:
                resolved_packet.append(line)
            continue
        resolved_packet.append(line)

    if not unresolved:
        resolved_packet = [x for x in resolved_packet if not x.startswith("STATE[")]
        resolved_packet.append("STATE[resolved]")
        state = "resolved"
    else:
        state = "awaiting_clarification"
        if not any(x.startswith("STATE[") for x in resolved_packet):
            resolved_packet.append("STATE[awaiting_clarification]")

    return {"ok": True, "resolved_packet": resolved_packet, "state": state, "unresolved_slots": unresolved}



def build_clarification_frame_record(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight persistence-ready record aligned with the conceptual-query model.
    """
    analysis_result = analysis_result if isinstance(analysis_result, dict) else {}
    now = time.time()
    resolved_packet = analysis_result.get("resolved_packet") if isinstance(analysis_result.get("resolved_packet"), list) else []
    return {
        "frame_id": str(analysis_result.get("frame_id") or _new_id("clarframe")),
        "session_id": str(_get_nested(analysis_result, ("context_packet_meta", "session_id"), "")),
        "query_id": str(analysis_result.get("query_id") or _new_id("query")),
        "parent_query_id": str(analysis_result.get("query_id") or ""),
        "concept_group_id": str(analysis_result.get("concept_group_id") or analysis_result.get("query_id") or ""),
        "original_input": str(analysis_result.get("raw_text") or ""),
        "normalized_text": str(analysis_result.get("normalized_text") or ""),
        "partial_packet_json": json.dumps(analysis_result.get("symbolic_packet_partial") or [], ensure_ascii=False),
        "unresolved_slots_json": json.dumps(analysis_result.get("ambiguities") or [], ensure_ascii=False),
        "clarification_question": str(analysis_result.get("clarification_question") or ""),
        "clarification_answer": str(analysis_result.get("clarification_answer") or ""),
        "resolved_packet_json": json.dumps(resolved_packet, ensure_ascii=False),
        "state": str(analysis_result.get("state") or "parsed_partial"),
        "confidence": float(analysis_result.get("compression_confidence") or 0.0),
        "created_ts": now,
        "updated_ts": now,
    }



def build_llm_compact_payload(analysis_result: Dict[str, Any], include_raw_fallback: bool = False) -> Dict[str, Any]:
    """
    Build the shortest safe payload for a local/3rd-party/online model.

    This is the main token-reduction output of the module.
    """
    analysis_result = analysis_result if isinstance(analysis_result, dict) else {}
    packet = analysis_result.get("resolved_packet") if isinstance(analysis_result.get("resolved_packet"), list) and analysis_result.get("resolved_packet") else analysis_result.get("symbolic_packet_partial")
    if not isinstance(packet, list):
        packet = []
    packet = _compact_packet_lines(packet)
    advcu_bundle = analysis_result.get("advcu_bundle") if isinstance(analysis_result.get("advcu_bundle"), dict) else {}
    helper_payload = advcu_bundle.get("helper_payload") if isinstance(advcu_bundle.get("helper_payload"), dict) else {}
    unresolved = [a for a in list(analysis_result.get("ambiguities") or []) if isinstance(a, dict) and not a.get("resolved")]

    compact = {
        "v": PROJECT_VERSION,
        "m": str(analysis_result.get("compression_mode") or "partial"),
        "fb": int(analysis_result.get("fallback_level") or FALLBACK_PARTIAL),
        "ln": str(analysis_result.get("likely_lane") or advcu_bundle.get("likely_lane") or "answer"),
        "dm": str(analysis_result.get("likely_domain") or advcu_bundle.get("likely_domain") or "GEN"),
        "pkt": [str(x) for x in packet[:MAX_SYMBOL_LINES]],
    }

    if unresolved:
        compact["amb"] = [
            {
                "t": _symbol_safe(item.get("ambiguity_type") or "generic"),
                "s": _symbol_safe(item.get("slot") or "value"),
                "r": _symbol_safe(item.get("raw") or "unknown"),
                "c": [_symbol_safe(x) for x in list(item.get("candidates") or [])[:MAX_CLARIFICATION_CHOICES]],
            }
            for item in unresolved[:3]
        ]
        if analysis_result.get("needs_clarification"):
            compact["cq"] = _symbol_safe(str(analysis_result.get("clarification_question") or "clarify unresolved slot"))

    # Only add helper payload / summary when the packet is too generic to stand alone.
    if not packet or str(analysis_result.get('likely_domain') or 'GEN') in {'GEN', 'ACT', 'QRY'}:
        compact["i"] = str(advcu_bundle.get("intent") or analysis_result.get("query_type") or "question")
        compact["qt"] = str(analysis_result.get("query_type") or advcu_bundle.get("query_type") or "general_request")
        subject = _symbol_safe(analysis_result.get("subject") or advcu_bundle.get("subject") or "request")
        if subject:
            compact["s"] = subject
        attrs = [_symbol_safe(x) for x in list(analysis_result.get("attributes") or advcu_bundle.get("attributes") or [])[:3]]
        if attrs:
            compact["at"] = attrs
        min_helper = _minify_helper_payload(helper_payload)
        if min_helper:
            compact["hp"] = min_helper
    elif advcu_bundle.get('math_expr'):
        compact['mx'] = _symbol_safe(str(advcu_bundle.get('math_expr')))

    if include_raw_fallback:
        compact["raw"] = str(analysis_result.get("normalized_text") or analysis_result.get("raw_text") or "")[:MAX_RAW_FALLBACK_CHARS]

    return {k: v for k, v in compact.items() if v not in (None, "", [], {})}



def compact_for_model(raw_text: str, context_packet: Optional[Dict[str, Any]] = None, include_raw_fallback: bool = False) -> Dict[str, Any]:
    """
    Convenience wrapper for downstream model callers.
    """
    analysis = analyze_text(raw_text, context_packet=context_packet)
    payload = build_llm_compact_payload(analysis, include_raw_fallback=include_raw_fallback or bool(analysis.get("needs_clarification")))
    payload_text = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    payload_line = _compact_payload_to_line(payload)
    raw_downstream_text = _build_estimated_downstream_raw_text(
        raw_text=str(analysis.get("normalized_text") or raw_text or ""),
        semantic_summary=str(analysis.get("semantic_summary") or ""),
        helper_payload=_get_nested(analysis, ("advcu_bundle", "helper_payload"), {}) or {},
        semantic_packet=_get_nested(analysis, ("advcu_bundle", "semantic_packet"), {}) or {},
        clarification_question=str(analysis.get("clarification_question") or ""),
        ambiguities=analysis.get("ambiguities") if isinstance(analysis.get("ambiguities"), list) else [],
    )
    best_payload_text, best_payload_strategy = _select_best_payload_text(
        compact_json_text=payload_text,
        compact_line_text=payload_line,
        helper_text=_normalize_text(json.dumps(_get_nested(analysis, ("advcu_bundle", "helper_payload"), {}) or {}, separators=(",", ":"), ensure_ascii=False)),
        symbolic_packet_text=str(analysis.get("symbolic_packet_text") or ""),
        semantic_packet_text=_normalize_text(json.dumps(_get_nested(analysis, ("advcu_bundle", "semantic_packet"), {}) or {}, separators=(",", ":"), ensure_ascii=False)),
        raw_text=str(analysis.get("normalized_text") or raw_text or ""),
        needs_clarification=bool(analysis.get("needs_clarification")),
        raw_downstream_text=raw_downstream_text,
    )
    return {
        "ok": True,
        "analysis": analysis,
        "payload": payload,
        "payload_text": payload_text,
        "payload_line": payload_line,
        "payload_text_best": best_payload_text,
        "payload_strategy_best": best_payload_strategy,
        "token_metrics": estimate_token_reduction(
            raw_text=str(analysis.get("normalized_text") or raw_text or ""),
            symbolic_packet_text=str(analysis.get("symbolic_packet_text") or ""),
            compact_llm_text=payload_text,
            helper_payload=_get_nested(analysis, ("advcu_bundle", "helper_payload"), {}) or {},
            compact_line_text=payload_line,
            semantic_packet_text=_normalize_text(json.dumps(_get_nested(analysis, ("advcu_bundle", "semantic_packet"), {}) or {}, separators=(",", ":"), ensure_ascii=False)),
            raw_downstream_text=raw_downstream_text,
        ),
    }



def estimate_token_reduction(
    raw_text: str,
    symbolic_packet_text: str,
    compact_llm_text: str,
    helper_payload: Optional[Dict[str, Any]] = None,
    compact_line_text: str = "",
    semantic_packet_text: str = "",
    raw_downstream_text: str = "",
) -> Dict[str, Any]:
    """
    Approximate token reduction against two baselines:
    1) raw user sentence only
    2) estimated downstream raw orchestration payload
    """
    raw_text = _normalize_text(raw_text)
    symbolic_packet_text = _normalize_text(symbolic_packet_text)
    compact_llm_text = _normalize_text(compact_llm_text)
    compact_line_text = _normalize_text(compact_line_text)
    semantic_packet_text = _normalize_text(semantic_packet_text)
    helper_text = _normalize_text(json.dumps(helper_payload or {}, separators=(",", ":"), ensure_ascii=False))
    raw_downstream_text = _normalize_text(raw_downstream_text)

    raw_tokens = _estimate_token_count(raw_text)
    raw_downstream_tokens = _estimate_token_count(raw_downstream_text or raw_text)
    symbolic_tokens = 0 if _is_effectively_empty_payload_text(symbolic_packet_text) else _estimate_token_count(symbolic_packet_text)
    compact_tokens = 0 if _is_effectively_empty_payload_text(compact_llm_text) else _estimate_token_count(compact_llm_text)
    compact_line_tokens = 0 if _is_effectively_empty_payload_text(compact_line_text) else _estimate_token_count(compact_line_text)
    helper_tokens = 0 if _is_effectively_empty_payload_text(helper_text) else _estimate_token_count(helper_text)
    semantic_packet_tokens = 0 if _is_effectively_empty_payload_text(semantic_packet_text) else _estimate_token_count(semantic_packet_text)

    candidates = {
        "symbolic_packet": symbolic_tokens,
        "compact_json": compact_tokens,
        "compact_line": compact_line_tokens,
        "helper_payload": helper_tokens,
        "advcu_semantic_packet": semantic_packet_tokens,
    }
    positive = {k: v for k, v in candidates.items() if v > 0}
    best_name = min(positive, key=positive.get) if positive else "raw"
    best = positive.get(best_name, raw_downstream_tokens)

    user_saved = max(0, raw_tokens - best)
    user_ratio = round((user_saved / max(1, raw_tokens)), 6)
    downstream_saved = max(0, raw_downstream_tokens - best)
    downstream_ratio = round((downstream_saved / max(1, raw_downstream_tokens)), 6)

    return {
        "approx_raw_user_tokens": raw_tokens,
        "approx_raw_downstream_tokens": raw_downstream_tokens,
        "approx_symbolic_tokens": symbolic_tokens,
        "approx_compact_tokens": compact_tokens,
        "approx_compact_line_tokens": compact_line_tokens,
        "approx_helper_tokens": helper_tokens,
        "approx_advcu_semantic_packet_tokens": semantic_packet_tokens,
        "best_downstream_strategy": best_name,
        "best_downstream_tokens": best,
        "tokens_saved_vs_user": user_saved,
        "reduction_ratio_vs_user": user_ratio,
        "tokens_saved_vs_downstream": downstream_saved,
        "reduction_ratio_vs_downstream": downstream_ratio,
    }





def clear_analysis_cache() -> None:
    with _ANALYSIS_CACHE_LOCK:
        _ANALYSIS_CACHE.clear()


# ---------------------------------------------------------------------------
# Internal helper logic
# ---------------------------------------------------------------------------


def _advcu_analyze(normalized_text: str, context_packet: Dict[str, Any], contracts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "available": bool(contracts.get("available")),
        "intent": "question",
        "intent_confidence": 0.0,
        "subject": "",
        "attributes": [],
        "output_type": "",
        "query_type": "",
        "likely_lane": "",
        "semantic_summary": "",
        "module_hints": [],
        "helper_payload": {},
        "semantic_packet": {},
        "parsed_command": {},
        "likely_domain": "",
        "math_expr": None,
    }
    if not normalized_text or not contracts.get("available"):
        return out

    user_context = context_packet.get("user_context")
    device_context = context_packet.get("device_context")

    try:
        classify_fn = contracts.get("classify_intent_with_confidence")
        if callable(classify_fn):
            label, conf = classify_fn(normalized_text, user_context=user_context, device_context=device_context)
            out["intent"] = str(label or "question")
            out["intent_confidence"] = float(conf or 0.0)
        else:
            classify_basic = contracts.get("classify_intent")
            if callable(classify_basic):
                out["intent"] = str(classify_basic(normalized_text, user_context=user_context, device_context=device_context) or "question")
                out["intent_confidence"] = 0.60
    except Exception:
        pass

    try:
        parse_fn = contracts.get("parse_command")
        if callable(parse_fn):
            parsed = parse_fn(normalized_text, user_context=user_context, device_context=device_context)
            parsed_dict = parsed.to_dict() if hasattr(parsed, "to_dict") else {}
            out["parsed_command"] = parsed_dict if isinstance(parsed_dict, dict) else {}
            semantic_packet = parsed.to_semantic_packet() if hasattr(parsed, "to_semantic_packet") else {}
            out["semantic_packet"] = semantic_packet if isinstance(semantic_packet, dict) else {}
            out["subject"] = str(_first_non_empty(parsed_dict.get("subject"), semantic_packet.get("subject"), ""))
            out["attributes"] = list(_first_non_empty(parsed_dict.get("attributes"), semantic_packet.get("attributes"), []))
            out["output_type"] = str(_first_non_empty(parsed_dict.get("output_type"), semantic_packet.get("output_type"), ""))
            out["query_type"] = str(_first_non_empty(parsed_dict.get("query_type"), semantic_packet.get("query_type"), ""))
            out["likely_lane"] = str(_first_non_empty(parsed_dict.get("likely_lane"), semantic_packet.get("likely_lane"), ""))
            out["semantic_summary"] = str(_first_non_empty(parsed_dict.get("semantic_summary"), semantic_packet.get("semantic_summary"), ""))
            out["module_hints"] = list(_first_non_empty(parsed_dict.get("module_hints"), semantic_packet.get("module_hints"), []))
            out["helper_payload"] = dict(_first_non_empty(parsed_dict.get("helper_payload"), semantic_packet.get("helper_payload"), {}))
            out["math_expr"] = _first_non_empty(parsed_dict.get("math_expr"), semantic_packet.get("math_expr"), None)
            out["likely_domain"] = _infer_domain_from_advcu(out)
    except Exception:
        pass

    try:
        helper_fn = contracts.get("build_helper_payload")
        if callable(helper_fn):
            helper_payload = helper_fn(normalized_text, user_context=user_context, device_context=device_context)
            if isinstance(helper_payload, dict) and helper_payload:
                out["helper_payload"] = helper_payload
    except Exception:
        pass

    try:
        build_packet_fn = contracts.get("build_semantic_packet")
        if callable(build_packet_fn):
            semantic_packet = build_packet_fn(normalized_text, user_context=user_context, device_context=device_context)
            if isinstance(semantic_packet, dict) and semantic_packet:
                out["semantic_packet"] = semantic_packet
                out["subject"] = str(_first_non_empty(out.get("subject"), semantic_packet.get("subject"), ""))
                out["attributes"] = list(_first_non_empty(out.get("attributes"), semantic_packet.get("attributes"), []))
                out["output_type"] = str(_first_non_empty(out.get("output_type"), semantic_packet.get("output_type"), ""))
                out["query_type"] = str(_first_non_empty(out.get("query_type"), semantic_packet.get("query_type"), ""))
                out["likely_lane"] = str(_first_non_empty(out.get("likely_lane"), semantic_packet.get("likely_lane"), ""))
                out["semantic_summary"] = str(_first_non_empty(out.get("semantic_summary"), semantic_packet.get("semantic_summary"), ""))
                out["module_hints"] = list(_first_non_empty(out.get("module_hints"), semantic_packet.get("module_hints"), []))
                out["helper_payload"] = dict(_first_non_empty(out.get("helper_payload"), semantic_packet.get("helper_payload"), {}))
                out["math_expr"] = _first_non_empty(out.get("math_expr"), semantic_packet.get("math_expr"), None)
                out["likely_domain"] = _infer_domain_from_advcu(out)
    except Exception:
        pass

    return out



def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"



def _coerce_text(value: Any) -> str:
    try:
        text = "" if value is None else str(value)
    except Exception:
        text = ""
    return text[:MAX_TEXT_LEN]



def _normalize_text(text: str) -> str:
    text = _coerce_text(text).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("\t", " ")
    text = SPACE_RE.sub(" ", text).strip()
    return text



def _analysis_cache_key(normalized_text: str, context_packet: Dict[str, Any]) -> str:
    session_id = str(context_packet.get("session_id") or _get_nested(context_packet, ("meta", "session_id"), "") or "")[:64]
    source = str(context_packet.get("source") or "")[:32]
    mode = str(context_packet.get("mode") or "")[:16]
    last_hist = ""
    history = context_packet.get("history") if isinstance(context_packet.get("history"), list) else []
    if history:
        item = history[-1]
        if isinstance(item, dict):
            last_hist = str(item.get("text") or item.get("input") or "")
        else:
            last_hist = str(item)
    raw = f"{session_id}|{source}|{mode}|{normalized_text[:MAX_CACHE_KEY_CHARS]}|{last_hist[:64]}"
    return hashlib.sha256(raw.encode("utf-8", errors="ignore")).hexdigest()



def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    with _ANALYSIS_CACHE_LOCK:
        item = _ANALYSIS_CACHE.get(key)
        if item is None:
            return None
        return copy.deepcopy(item)



def _cache_put(key: str, value: Dict[str, Any]) -> None:
    with _ANALYSIS_CACHE_LOCK:
        if len(_ANALYSIS_CACHE) >= _ANALYSIS_CACHE_MAX:
            oldest_key = next(iter(_ANALYSIS_CACHE.keys()), None)
            if oldest_key:
                _ANALYSIS_CACHE.pop(oldest_key, None)
        _ANALYSIS_CACHE[key] = copy.deepcopy(value)



def _word_tokens(text: str) -> List[str]:
    return WORD_RE.findall(text or "")



def _contains_any(text: str, options: Iterable[str]) -> bool:
    text = text or ""
    for item in options:
        if item and item in text:
            return True
    return False



def _score_domains(lowered: str, words: Sequence[str]) -> Dict[str, float]:
    scores: Dict[str, float] = {k: 0.0 for k in DOMAIN_KEYWORD_MAP}
    text_words = {w.lower() for w in words}
    for domain, keys in DOMAIN_KEYWORD_MAP.items():
        hits = 0
        for key in keys:
            key_low = key.lower()
            if " " in key_low:
                if key_low in lowered:
                    hits += 1
            elif key_low in text_words:
                hits += 1
        total = max(1, len(keys))
        scores[domain] = _clamp(hits / total)

    if any(x in lowered for x in ("temperature", "forecast", "weather", "degrees")):
        scores["WEATHER"] = max(scores.get("WEATHER", 0.0), 0.72)
    if any(x in lowered for x in ("keyboard", "caps lock", "caps_lock", "rgb", "printer", "mouse", "wifi", "bluetooth")):
        scores["DRV"] = max(scores.get("DRV", 0.0), 0.82)
    if any(x in lowered for x in ("unsubscribe", "spam", "inbox", "mail", "email", "trash")):
        scores["EMAIL"] = max(scores.get("EMAIL", 0.0), 0.86)
    if any(x in lowered for x in ("open word", "write report", "document", "report", "presentation", "slides", "spreadsheet", "pdf")):
        scores["DOC"] = max(scores.get("DOC", 0.0), 0.78)
    if any(x in lowered for x in ("diagnostics", "diagnostic", "gpu", "cpu", "thermal", "overheating", "ram", "vram")):
        scores["SYS"] = max(scores.get("SYS", 0.0), 0.74)
    if any(x in lowered for x in ("sarahnet", "node", "mesh", "broker", "sync", "remote")):
        scores["NET"] = max(scores.get("NET", 0.0), 0.76)
    if any(x in lowered for x in ("research", "browser", "search", "look up", "lookup", "article", "news", "website")):
        scores["RSH"] = max(scores.get("RSH", 0.0), 0.70)
    if any(x in lowered for x in ("daily", "weekly", "monthly", "every day", "every week", "every month")):
        scores["SCHED"] = max(scores.get("SCHED", 0.0), 0.62)
    if _contains_any(lowered, ACTION_KEYWORDS):
        scores.setdefault("ACT", 0.0)
        scores["ACT"] = max(scores["ACT"], 0.42)
    if _contains_any(lowered, QUESTION_KEYWORDS):
        scores.setdefault("QRY", 0.0)
        scores["QRY"] = max(scores["QRY"], 0.28)
    return scores





def _best_domain(domain_scores: Dict[str, float]) -> str:
    if not domain_scores:
        return "GEN"
    best_key = "GEN"
    best_val = 0.0
    for key, val in domain_scores.items():
        if float(val) > best_val:
            best_key, best_val = key, float(val)
    return best_key if best_val > 0.0 else "GEN"



def _select_best_domain(advcu_domain: str, heuristic_domain: str) -> str:
    adv = str(advcu_domain or "").strip().upper()
    heur = str(heuristic_domain or "").strip().upper()
    if adv and adv not in {"GEN", "QRY", "ACT"}:
        return adv
    if heur and heur != "GEN":
        return heur
    if adv:
        return adv
    return "GEN"



def _detect_lane(likely_domain: str, lowered: str) -> str:
    if likely_domain == "WEATHER":
        return "answer"
    if likely_domain in {"DOC", "DRV", "ACT", "SCHED", "EMAIL"}:
        return "action"
    if likely_domain in {"CRT"}:
        return "creative"
    if likely_domain in {"SYS"}:
        return "system"
    if likely_domain in {"NET"}:
        return "network"
    if likely_domain in {"RSH", "QRY"}:
        return "answer"
    if lowered.endswith("?") or _contains_any(lowered, QUESTION_KEYWORDS):
        return "answer"
    if _contains_any(lowered, ACTION_KEYWORDS):
        return "action"
    return "answer"





def _detect_query_type(lowered: str, likely_lane: str) -> str:
    if "temperature" in lowered or "forecast" in lowered or "weather" in lowered:
        return "factual_weather_forecast" if ("forecast" in lowered or "tomorrow" in lowered or "next" in lowered) else "factual_weather_current"
    if any(x in lowered for x in ("unsubscribe", "spam", "trash", "inbox", "mail", "email")):
        return "email_automation" if ("unsubscribe" in lowered or "empty" in lowered or _contains_any(lowered, SCHEDULE_KEYWORDS)) else "email_request"
    if any(x in lowered for x in ("keyboard", "caps lock", "caps_lock", "rgb", "printer", "mouse", "wifi", "bluetooth")):
        return "device_control"
    if any(x in lowered for x in ("open word", "write report", "document", "report", "presentation", "slides", "spreadsheet", "pdf")):
        return "document_action"
    if likely_lane == "system":
        return "system_status"
    if likely_lane == "network":
        return "network_operation"
    if likely_lane == "creative":
        return "creative_request"
    if likely_lane == "action":
        return "action_request"
    if lowered.endswith("?"):
        return "question"
    return "general_request"





def _detect_output_type(lowered: str, likely_domain: str) -> str:
    if likely_domain == "DOC":
        return "document"
    if likely_domain == "CRT":
        return "artifact"
    if likely_domain in {"WEATHER", "SYS", "QRY", "EMAIL"}:
        return "text_answer"
    if likely_domain in {"DRV", "ACT", "NET", "SCHED"}:
        return "action_result"
    return "text"


def _prefer_output_type(advcu_output_type: Any, detected_output_type: str, likely_domain: str) -> str:
    adv = str(advcu_output_type or '').strip().lower()
    detected = str(detected_output_type or '').strip().lower()
    if likely_domain == 'DOC':
        return 'document'
    if likely_domain == 'CRT':
        return 'artifact'
    if likely_domain == 'EMAIL':
        return 'action_result'
    if likely_domain == 'WEATHER':
        return 'text_answer'
    if adv and adv not in {'text', 'answer', 'action'}:
        return adv
    return detected or adv or 'text'



def _extract_subject(words: Sequence[str], likely_domain: str) -> str:
    lowered_words = [w.lower() for w in words]
    preferred = {
        "WEATHER": ["weather", "temperature", "forecast"],
        "DOC": ["document", "report", "letter", "word", "pdf"],
        "SYS": ["gpu", "cpu", "system", "diagnostics", "thermal", "status"],
        "DRV": ["keyboard", "mouse", "printer", "device", "driver", "rgb"],
        "NET": ["network", "node", "broker", "sync"],
        "CRT": ["image", "music", "video", "avatar", "song"],
        "RSH": ["browser", "research", "search", "web"],
        "EMAIL": ["email", "mail", "inbox", "spam"],
        "SCHED": ["schedule", "reminder", "task"],
    }
    for item in preferred.get(likely_domain, []):
        if item in lowered_words:
            return item
    for item in words:
        low = item.lower()
        if low not in STOPWORDS and len(low) > 2:
            return low
    return "request"



def _extract_attributes(words: Sequence[str], subject: str) -> List[str]:
    subject = (subject or "").lower()
    attrs: List[str] = []
    for item in words:
        low = item.lower()
        if low == subject or low in STOPWORDS:
            continue
        if len(low) < 2:
            continue
        attrs.append(low)
    return attrs[:MAX_ATTRIBUTES]



def _build_semantic_summary(likely_lane: str, likely_domain: str, subject: str, attributes: Sequence[str], normalized_text: str) -> str:
    if likely_domain == "WEATHER":
        return f"weather-oriented {likely_lane} request about {subject}"
    if likely_domain == "DOC":
        return f"document-oriented {likely_lane} request about {subject}"
    if likely_domain == "SYS":
        return f"system diagnostics request about {subject}"
    if likely_domain == "DRV":
        return f"device-control request about {subject}"
    if likely_domain == "CRT":
        return f"creative request about {subject}"
    if likely_domain == "NET":
        return f"network request about {subject}"
    if likely_domain == "RSH":
        return f"research request about {subject}"
    if likely_domain == "EMAIL":
        return f"email management request about {subject}"
    attrs = ", ".join(list(attributes)[:3]) if attributes else normalized_text[:80]
    return f"{likely_lane} request involving {subject} with attributes {attrs}"



def _extract_abbreviation_findings(
    normalized_text: str,
    lowered: str,
    likely_domain: str,
    likely_lane: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    abbreviation_hits: List[Dict[str, Any]] = []
    ambiguities: List[Dict[str, Any]] = []
    opaque_segments: List[str] = []

    for match in OPAQUE_CHAIN_RE.finditer(normalized_text):
        token = match.group(0)
        letters_only = token.replace(".", "")
        if letters_only.lower() in SAFE_COMMON_ABBREVIATIONS:
            abbreviation_hits.append({
                "raw": token,
                "normalized": SAFE_COMMON_ABBREVIATIONS[letters_only.lower()],
                "kind": "safe_common",
                "confidence": 0.95,
            })
            continue
        opaque_segments.append(token)
        ambiguities.append({
            "ambiguity_type": "lexical",
            "slot": "abbreviation_chain",
            "raw": token,
            "candidates": [],
            "confidence": 0.05,
            "reason": "opaque_dotted_chain",
            "resolved": False,
        })

    for token in UPPER_TOKEN_RE.findall(normalized_text):
        if token.lower() in SAFE_COMMON_ABBREVIATIONS:
            abbreviation_hits.append({
                "raw": token,
                "normalized": SAFE_COMMON_ABBREVIATIONS[token.lower()],
                "kind": "safe_common",
                "confidence": 0.97,
            })
            continue

        state_candidates = US_STATE_ABBREVIATIONS.get(token.upper())
        if state_candidates:
            confidence = 0.90 if len(state_candidates) == 1 else 0.42
            if len(state_candidates) == 1:
                abbreviation_hits.append({
                    "raw": token,
                    "normalized": state_candidates[0],
                    "kind": "state_abbreviation",
                    "confidence": confidence,
                })
            else:
                ambiguities.append({
                    "ambiguity_type": "lexical",
                    "slot": "location",
                    "raw": token,
                    "candidates": state_candidates,
                    "confidence": confidence,
                    "reason": "ambiguous_state_or_city_abbreviation",
                    "resolved": False,
                })
            continue

        if len(token) >= 5:
            ambiguities.append({
                "ambiguity_type": "lexical",
                "slot": "abbreviation",
                "raw": token,
                "candidates": [],
                "confidence": 0.18,
                "reason": "unknown_uppercase_token",
                "resolved": False,
            })

    for raw, spec in LEXICAL_AMBIGUITIES.items():
        pattern = rf"\b{re.escape(raw)}\b"
        if re.search(pattern, lowered):
            if raw == "word" and ("open" in lowered or "write" in lowered or likely_domain == "DOC"):
                abbreviation_hits.append({
                    "raw": raw,
                    "normalized": "Microsoft Word",
                    "kind": "lexical_context_resolved",
                    "confidence": 0.91,
                })
            elif raw == "edge" and ("browser" in lowered or "open" in lowered or "website" in lowered):
                abbreviation_hits.append({
                    "raw": raw,
                    "normalized": "Microsoft Edge",
                    "kind": "lexical_context_resolved",
                    "confidence": 0.88,
                })
            elif raw == "plant" and ("report" in lowered or "factory" in lowered or likely_lane == "action"):
                abbreviation_hits.append({
                    "raw": raw,
                    "normalized": "industrial facility",
                    "kind": "lexical_context_resolved",
                    "confidence": 0.74,
                })
            else:
                ambiguities.append({
                    "ambiguity_type": "lexical",
                    "slot": str(spec.get("slot") or "value"),
                    "raw": raw,
                    "candidates": list(spec.get("candidates") or []),
                    "confidence": 0.41,
                    "reason": "lexical_ambiguity",
                    "resolved": False,
                })

    if likely_domain == "WEATHER" and "temp" in lowered:
        abbreviation_hits.append({"raw": "temp", "normalized": "temperature", "kind": "domain_common", "confidence": 0.94})
    if likely_domain == "DOC" and "doc" in lowered:
        abbreviation_hits.append({"raw": "doc", "normalized": "document", "kind": "domain_common", "confidence": 0.92})

    return _dedupe_hits(abbreviation_hits), _dedupe_ambiguities(ambiguities), _dedupe_strings(opaque_segments)



def _detect_reference_ambiguities(normalized_text: str, likely_lane: str, likely_domain: str, context_packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    lowered = normalized_text.lower()
    findings: List[Dict[str, Any]] = []
    has_local_anchor = bool(_get_nested(context_packet, ("meta", "last_target"), "") or context_packet.get("target") or context_packet.get("selection"))
    for token in _word_tokens(lowered):
        if token in AMBIGUOUS_REFERENTS and not has_local_anchor:
            findings.append({
                "ambiguity_type": "reference",
                "slot": "reference",
                "raw": token,
                "candidates": [],
                "confidence": 0.22 if likely_lane == "action" else 0.32,
                "reason": "ambiguous_referent",
                "resolved": False,
            })
    return findings



def _detect_scope_ambiguities(normalized_text: str, likely_lane: str, likely_domain: str, context_packet: Dict[str, Any]) -> List[Dict[str, Any]]:
    lowered = normalized_text.lower()
    findings: List[Dict[str, Any]] = []
    if "latest" in lowered or "recent" in lowered:
        findings.append({
            "ambiguity_type": "scope",
            "slot": "scope",
            "raw": "latest" if "latest" in lowered else "recent",
            "candidates": ["latest file", "latest message", "latest model", "latest result"],
            "confidence": 0.36,
            "reason": "scope_ambiguous_latest_recent",
            "resolved": False,
        })
    if TIME_SCOPE_RE.search(normalized_text) and likely_domain not in {"WEATHER", "SCHED"}:
        match = TIME_SCOPE_RE.search(normalized_text)
        findings.append({
            "ambiguity_type": "scope",
            "slot": "time_scope",
            "raw": match.group(1) if match else "time",
            "candidates": [],
            "confidence": 0.40,
            "reason": "time_scope_needs_context",
            "resolved": False,
        })
    return findings



def _detect_action_ambiguities(normalized_text: str, likely_lane: str, likely_domain: str) -> List[Dict[str, Any]]:
    lowered = normalized_text.lower()
    findings: List[Dict[str, Any]] = []
    if likely_lane == "action":
        for raw in ("fix it", "update it", "make it better"):
            if raw in lowered:
                findings.append({
                    "ambiguity_type": "action",
                    "slot": "operation",
                    "raw": raw,
                    "candidates": ["diagnose", "repair", "upgrade", "optimize"],
                    "confidence": 0.28,
                    "reason": "vague_action",
                    "resolved": False,
                })
    return findings



def _detect_execution_ambiguities(normalized_text: str, likely_lane: str, likely_domain: str) -> List[Dict[str, Any]]:
    lowered = normalized_text.lower()
    findings: List[Dict[str, Any]] = []
    if likely_lane == "action" and ("use" in lowered and ("app" in lowered or "driver" in lowered or "api" in lowered)):
        findings.append({
            "ambiguity_type": "execution",
            "slot": "execution_method",
            "raw": "execution method",
            "candidates": ["driver", "application", "direct system control", "research", "api"],
            "confidence": 0.33,
            "reason": "execution_method_ambiguous",
            "resolved": False,
        })
    return findings



def _dedupe_hits(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        key = (str(item.get("raw")), str(item.get("normalized")), str(item.get("kind")))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out



def _dedupe_ambiguities(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        key = (str(item.get("ambiguity_type")), str(item.get("slot")), str(item.get("raw")), str(item.get("reason")))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out



def _dedupe_strings(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out



def _score_known_vocab(words: Sequence[str], abbreviation_hits: Sequence[Dict[str, Any]]) -> float:
    if not words:
        return 0.0
    recognized = 0
    for word in words:
        low = word.lower()
        if low in STOPWORDS or low in SAFE_COMMON_ABBREVIATIONS or low.isdigit() or len(low) > 2:
            recognized += 1
    recognized += len(abbreviation_hits)
    denom = max(1, len(words) + len(abbreviation_hits))
    return _clamp(recognized / denom)



def _score_context_anchor(normalized_text: str, ambiguities: Sequence[Dict[str, Any]]) -> float:
    if not normalized_text:
        return 0.0
    if not ambiguities:
        return 0.95
    score = 0.85
    lowered = normalized_text.lower()
    for item in ambiguities:
        raw = str(item.get("raw") or "").lower()
        if raw and raw in lowered:
            score -= 0.08
        if item.get("candidates"):
            score -= 0.04
        if item.get("ambiguity_type") in {"reference", "scope", "execution"}:
            score -= 0.06
    return _clamp(score)



def _score_entity_resolution(ambiguities: Sequence[Dict[str, Any]], abbreviation_hits: Sequence[Dict[str, Any]]) -> float:
    if ambiguities and not abbreviation_hits:
        return 0.25
    if abbreviation_hits and not ambiguities:
        return 0.92
    if abbreviation_hits and ambiguities:
        return 0.55
    return 0.70



def _score_syntax_stability(normalized_text: str) -> float:
    if not normalized_text:
        return 0.0
    words = _word_tokens(normalized_text)
    if not words:
        return 0.0
    punctuation_penalty = normalized_text.count("?") * 0.04 + normalized_text.count("!") * 0.03
    chain_penalty = 0.22 if OPAQUE_CHAIN_RE.search(normalized_text) else 0.0
    ratio_penalty = 0.20 if len(words) < 2 else 0.0
    return _clamp(1.0 - punctuation_penalty - chain_penalty - ratio_penalty)



def _score_repeat_history(normalized_text: str, context_packet: Dict[str, Any]) -> float:
    history = context_packet.get("history") if isinstance(context_packet.get("history"), list) else []
    if not history:
        return 0.55
    normalized = normalized_text.lower()
    for item in history[-8:]:
        txt = ""
        if isinstance(item, dict):
            txt = str(item.get("text") or item.get("input") or "")
        elif isinstance(item, str):
            txt = item
        txt = txt.strip().lower()
        if txt and txt == normalized:
            return 0.94
        if txt and txt in normalized:
            return 0.76
    return 0.58



def _score_ambiguity(ambiguities: Sequence[Dict[str, Any]], likely_lane: str) -> float:
    if not ambiguities:
        return 0.0
    base = min(0.95, 0.22 * len(ambiguities))
    if likely_lane == "action":
        base += 0.16
    elif likely_lane == "network":
        base += 0.10
    for item in ambiguities:
        t = str(item.get("ambiguity_type") or "")
        if t == "execution":
            base += 0.07
        elif t == "reference":
            base += 0.05
    return _clamp(base)



def _score_opaque_text(normalized_text: str, opaque_segments: Sequence[str]) -> float:
    if opaque_segments:
        return _clamp(0.55 + 0.18 * len(opaque_segments))
    uppercase_tokens = UPPER_TOKEN_RE.findall(normalized_text)
    if len(uppercase_tokens) >= 4:
        return 0.42
    return 0.0



def _score_execution_risk(lowered: str, likely_lane: str, ambiguities: Sequence[Dict[str, Any]]) -> float:
    base = 0.12 if likely_lane == "answer" else 0.28
    if likely_lane in {"action", "network", "system"}:
        base += 0.12
    if ambiguities:
        base += 0.18
    for pattern in RISKY_ACTION_PATTERNS:
        if pattern in lowered:
            base += 0.08
    return _clamp(base)



def _score_maintainability(likely_lane: str, likely_domain: str, subject: str, attributes: Sequence[str], ambiguities: Sequence[Dict[str, Any]]) -> float:
    score = 0.92
    if not subject or subject == "request":
        score -= 0.16
    if len(attributes) > 6:
        score -= 0.09
    if ambiguities:
        score -= 0.18
    if likely_domain == "GEN":
        score -= 0.08
    if likely_lane not in {"answer", "action", "creative", "system", "network"}:
        score -= 0.10
    return _clamp(score)



def _decide_compression_mode(compression_confidence: float, ambiguity_score: float, opaque_text_score: float, execution_risk_score: float, action_detected: bool) -> str:
    if opaque_text_score >= 0.75:
        return "raw_only"
    if execution_risk_score >= 0.80 and ambiguity_score >= 0.45:
        return "clarify_first"
    if compression_confidence >= FULL_COMPRESS_THRESHOLD and ambiguity_score < 0.28:
        return "full"
    if compression_confidence >= PARTIAL_COMPRESS_THRESHOLD:
        return "partial"
    if compression_confidence >= CLARIFY_THRESHOLD or (ambiguity_score >= 0.40 and action_detected):
        return "clarify_first"
    return "raw_only"



def _build_symbolic_packet_partial(
    normalized_text: str,
    lowered: str,
    likely_lane: str,
    likely_domain: str,
    query_type: str,
    subject: str,
    attributes: Sequence[str],
    output_type: str,
    ambiguities: Sequence[Dict[str, Any]],
    abbreviation_hits: Sequence[Dict[str, Any]],
    compression_mode: str,
    advcu_bundle: Optional[Dict[str, Any]] = None,
) -> List[str]:
    packet: List[str] = []
    advcu_bundle = advcu_bundle if isinstance(advcu_bundle, dict) else {}
    domain = likely_domain if likely_domain != "GEN" else _lane_to_packet_type(likely_lane)

    if likely_domain == "WEATHER":
        loc_value = _extract_location_value(normalized_text, ambiguities, abbreviation_hits)
        mode = "forecast" if ("forecast" in lowered or "tomorrow" in lowered) else "current"
        day_hint = "+1" if "tomorrow" in lowered else "+0"
        packet.append(f"QRY[WEATHER {mode} day:{day_hint} loc:{loc_value}]")
    elif likely_domain == "DOC":
        op = "OPEN" if "open" in lowered else "CREATE"
        packet.append(f"DOC[{op} app:{_extract_doc_app(lowered)}]")
        task_hint = _extract_doc_task(lowered, subject, attributes)
        if task_hint:
            packet.append(f"CTX[task:{_symbol_safe(task_hint)}]")
        if subject and subject.lower() not in {"word", "document", "report"}:
            packet.append(f"CTX[subject:{_symbol_safe(subject)}]")
    elif likely_domain == "SYS":
        metric = _extract_system_metric(lowered, subject, attributes)
        packet.append(f"SYS[DIAG target:{_symbol_safe(subject or 'system')} metric:{_symbol_safe(metric)}]")
    elif likely_domain == "DRV":
        packet.append(f"DRV[{_extract_driver_packet(lowered, subject, attributes)}]")
    elif likely_domain == "NET":
        packet.append(f"NET[{_extract_network_packet(lowered, subject, attributes)}]")
    elif likely_domain == "CRT":
        packet.append(f"CRT[{_extract_creative_packet(lowered, subject, attributes)}]")
    elif likely_domain == "RSH":
        packet.append(f"RSH[BROWSER query:{_symbol_safe(_collapse_for_packet(normalized_text))}]")
    elif likely_domain == "EMAIL":
        packet.extend(_extract_email_packet(lowered, subject, attributes))
    elif likely_domain == "SCHED":
        packet.append(f"SCHED[pattern:{_extract_schedule_pattern(lowered)}]")
    else:
        packet_type = _lane_to_packet_type(likely_lane)
        packet.append(f"{packet_type}[domain:{_symbol_safe(domain)} query_type:{_symbol_safe(query_type)}]")
        if subject and subject != 'request':
            packet.append(f"CTX[subject:{_symbol_safe(subject)}]")

    # Only add lightweight CTX lines when they contribute new information.
    if output_type and likely_domain not in {"WEATHER", "DOC", "SYS", "DRV", "NET", "CRT", "RSH", "EMAIL", "SCHED"}:
        packet.append(f"CTX[output_type:{_symbol_safe(output_type)}]")
    if attributes and likely_domain not in {"WEATHER", "EMAIL"}:
        compact_attrs = [a for a in attributes[:3] if a and a.lower() not in {str(subject).lower(), 'word', 'document'}]
        if compact_attrs:
            packet.append(f"CTX[attributes:{_symbol_safe(','.join(compact_attrs))}]")

    # Use AdvCU semantic lines only for generic domains or when they add non-duplicate math/action context.
    advcu_packet = advcu_bundle.get("semantic_packet") if isinstance(advcu_bundle.get("semantic_packet"), dict) else {}
    if advcu_packet and likely_domain in {"GEN", "ACT", "QRY"}:
        compact_sem_packet = _advcu_semantic_packet_to_symbolic(advcu_packet)
        for line in compact_sem_packet:
            if line not in packet:
                packet.append(line)
    elif advcu_packet and advcu_packet.get('math_expr'):
        packet.append(f"CTX[math_expr:{_symbol_safe(advcu_packet.get('math_expr'))}]")

    for item in ambiguities:
        if not isinstance(item, dict):
            continue
        slot = _symbol_safe(str(item.get("slot") or "value"))
        raw = _symbol_safe(str(item.get("raw") or ""))
        packet.append(f"AMBIG[{slot}:{raw}]")

    packet = _compact_packet_lines(packet)

    if compression_mode == "raw_only":
        packet.append("STATE[raw_only]")
    elif ambiguities or compression_mode == "clarify_first":
        packet.append("STATE[awaiting_clarification]")
    elif compression_mode == "full":
        packet.append("STATE[resolved]")
    else:
        packet.append("STATE[parsed_partial]")

    return packet[: MAX_SYMBOL_LINES]



def _lane_to_packet_type(lane: str) -> str:
    mapping = {"answer": "QRY", "action": "ACT", "creative": "CRT", "system": "SYS", "network": "NET"}
    return mapping.get(lane, "QRY")



def _extract_location_value(normalized_text: str, ambiguities: Sequence[Dict[str, Any]], abbreviation_hits: Sequence[Dict[str, Any]]) -> str:
    for item in ambiguities:
        if isinstance(item, dict) and item.get("slot") == "location":
            return _symbol_safe(str(item.get("raw") or "unknown"))

    city_state = re.search(r"\bin\s+([A-Za-z][A-Za-z\s.'-]{1,48}?),\s*([A-Za-z]{2}|[A-Za-z][A-Za-z\s.'-]{2,32})\b", normalized_text)
    if city_state:
        city = _normalize_text(city_state.group(1))
        state_raw = _normalize_text(city_state.group(2))
        state_norm = state_raw
        for item in abbreviation_hits:
            if isinstance(item, dict) and str(item.get("raw") or "").strip().lower() == state_raw.lower() and item.get("kind") == "state_abbreviation":
                state_norm = str(item.get("normalized") or state_raw)
                break
        if len(state_raw) == 2 and state_raw.upper() in US_STATE_ABBREVIATIONS and state_norm == state_raw:
            cand = US_STATE_ABBREVIATIONS.get(state_raw.upper()) or []
            if len(cand) == 1:
                state_norm = cand[0]
        return _symbol_safe(f"{city}_{state_norm}")

    for item in abbreviation_hits:
        if isinstance(item, dict) and item.get("kind") == "state_abbreviation":
            return _symbol_safe(str(item.get("normalized") or "unknown"))
    match = TRAILING_LOC_RE.search(normalized_text)
    if match:
        return _symbol_safe(match.group(1).strip())
    return "unknown"





def _extract_doc_app(lowered: str) -> str:
    if "word" in lowered:
        return "Word"
    if "pdf" in lowered:
        return "PDF"
    if "slides" in lowered or "presentation" in lowered:
        return "Slides"
    if "spreadsheet" in lowered:
        return "Spreadsheet"
    return "Document"


def _extract_doc_task(lowered: str, subject: str, attributes: Sequence[str]) -> str:
    for key in ('report', 'letter', 'essay', 'draft', 'presentation', 'spreadsheet'):
        if key in lowered:
            return key
    for item in attributes:
        if item.lower() not in {'word', 'document'}:
            return item
    return subject if subject and subject.lower() not in {'word', 'document'} else ''


def _compact_packet_lines(packet: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for line in packet:
        line = str(line).strip()
        if not line or line in seen:
            continue
        # Drop redundant generic lines when stronger domain-specific lines already exist.
        if line.startswith('QRY[query_type:') and any(x.startswith(('DOC[', 'DRV[', 'NET[', 'SYS[', 'CRT[', 'RSH[', 'ACT[', 'SCHED[')) for x in out):
            continue
        if line.startswith('CTX[output_type:') and any(x.startswith(('DOC[', 'DRV[', 'NET[', 'SYS[', 'CRT[', 'RSH[', 'ACT[', 'SCHED[', 'QRY[WEATHER')) for x in out):
            continue
        seen.add(line)
        out.append(line)
    return out



def _extract_system_metric(lowered: str, subject: str, attributes: Sequence[str]) -> str:
    if "thermal" in lowered or "overheating" in lowered or "temperature" in lowered:
        return "thermal_status"
    if "status" in lowered:
        return "status"
    if subject and subject != "request":
        return subject
    return attributes[0] if attributes else "health"



def _extract_driver_packet(lowered: str, subject: str, attributes: Sequence[str]) -> str:
    target = subject or "device"
    if "keyboard" in lowered and "rgb" in lowered:
        color = _extract_color(lowered) or "unknown"
        caps = "off" if ("caps lock off" in lowered or "caps_lock off" in lowered or "caps off" in lowered) else "unknown"
        return f"KEYBOARD rgb:{_symbol_safe(color)} caps_lock:{_symbol_safe(caps)}"
    return f"target:{_symbol_safe(target)} params:{_symbol_safe(','.join(attributes[:4]))}"



def _extract_network_packet(lowered: str, subject: str, attributes: Sequence[str]) -> str:
    if "sync" in lowered:
        return "SYNC action:start"
    if "node" in lowered:
        return f"NODE target:{_symbol_safe(subject)} params:{_symbol_safe(','.join(attributes[:3]))}"
    return f"target:{_symbol_safe(subject or 'network')} params:{_symbol_safe(','.join(attributes[:4]))}"



def _extract_creative_packet(lowered: str, subject: str, attributes: Sequence[str]) -> str:
    if "image" in lowered or "picture" in lowered or "draw" in lowered:
        mode = "image"
    elif "video" in lowered:
        mode = "video"
    elif "song" in lowered or "music" in lowered:
        mode = "music"
    else:
        mode = subject or "artifact"
    return f"mode:{_symbol_safe(mode)} topic:{_symbol_safe(','.join(attributes[:4]) or subject or 'request')}"



def _extract_email_packet(lowered: str, subject: str, attributes: Sequence[str]) -> List[str]:
    packet: List[str] = []
    if "unsubscribe" in lowered and "spam" in lowered:
        packet.append("ACT[EMAIL unsubscribe:spam]")
    if "empty" in lowered and "trash" in lowered:
        packet.append("ACT[EMAIL_TRASH empty:spam]")
    if ("daily" in lowered or "every day" in lowered) and packet:
        packet.append("SCHED[pattern:daily]")
    if not packet:
        packet.append(f"ACT[EMAIL target:{_symbol_safe(subject or 'mail')} params:{_symbol_safe(','.join(attributes[:4]))}]")
    return packet



def _extract_schedule_pattern(lowered: str) -> str:
    if "daily" in lowered or "every day" in lowered:
        return "daily"
    if "weekly" in lowered or "every week" in lowered:
        return "weekly"
    if "monthly" in lowered or "every month" in lowered:
        return "monthly"
    return "one_time"



def _extract_color(lowered: str) -> str:
    for color in ("red", "green", "blue", "yellow", "purple", "orange", "white", "black", "pink"):
        if color in lowered:
            return color
    return ""



def _extract_safe_segments(normalized_text: str, ambiguities: Sequence[Dict[str, Any]], opaque_segments: Sequence[str]) -> List[str]:
    text = normalized_text
    for item in ambiguities:
        if not isinstance(item, dict):
            continue
        raw = str(item.get("raw") or "").strip()
        if raw:
            text = re.sub(rf"\b{re.escape(raw)}\b", " ", text, flags=re.I)
    for raw in opaque_segments:
        text = text.replace(raw, " ")
    parts = [x.strip() for x in re.split(r",| and | then |;", text, flags=re.I) if x.strip()]
    out: List[str] = []
    for p in parts:
        if p and p not in out:
            out.append(p)
    return out[:MAX_SAFE_SEGMENTS]



def _build_clarification_shape(ambiguities: Sequence[Dict[str, Any]], likely_lane: str, likely_domain: str) -> Tuple[str, List[str]]:
    unresolved = [a for a in ambiguities if isinstance(a, dict) and not a.get("resolved")]
    if not unresolved:
        return "none", []
    primary = unresolved[0]
    ambiguity_type = str(primary.get("ambiguity_type") or "generic")
    options = [str(x) for x in list(primary.get("candidates") or [])[:MAX_CLARIFICATION_CHOICES]]
    if not options and ambiguity_type == "execution":
        options = ["driver", "application", "direct system control", "research", "api"]
    return ambiguity_type, options



def _compute_fallback_level(compression_mode: str, clarification_required: bool, opaque_text_score: float, clarification_options: Sequence[str]) -> int:
    if compression_mode == "full":
        return FALLBACK_FULL
    if compression_mode == "partial" and not clarification_required:
        return FALLBACK_PARTIAL
    if compression_mode in {"partial", "clarify_first"} and clarification_required and clarification_options:
        return FALLBACK_BOUNDED_OPTIONS
    if compression_mode == "raw_only" and opaque_text_score < 0.75:
        return FALLBACK_RAW_HOLD
    return FALLBACK_SAFE_GENERIC



def _state_from_mode(compression_mode: str, clarification_required: bool) -> str:
    if clarification_required:
        return "awaiting_clarification"
    if compression_mode == "full":
        return "resolved"
    if compression_mode == "partial":
        return "parsed_partial"
    if compression_mode == "raw_only":
        return "raw_hold"
    return "new"



def _replacement_for_ambiguity_line(line: str, ambiguities: Sequence[Dict[str, Any]], resolved_map: Dict[str, str]) -> str:
    content = line[6:-1] if line.startswith("AMBIG[") and line.endswith("]") else ""
    if ":" not in content:
        return ""
    slot, raw = content.split(":", 1)
    key = f"{slot}:{raw}".lower()
    value = resolved_map.get(key)
    if not value:
        return ""
    slot_norm = _symbol_safe(slot)
    value_norm = _symbol_safe(value)
    return f"CTX[{slot_norm}:{value_norm}]"



def _resolve_candidate_from_answer(candidates: Sequence[str], user_answer: str) -> str:
    if not candidates:
        return ""
    answer = user_answer.strip().lower()
    if not answer:
        return ""
    if answer.isdigit():
        idx = int(answer) - 1
        if 0 <= idx < len(candidates):
            return str(candidates[idx])
    for candidate in candidates:
        c = str(candidate)
        cl = c.lower()
        if answer == cl or answer in cl or cl in answer:
            return c
    return ""



def _packet_lines_to_text(lines: Sequence[str]) -> str:
    if not lines:
        return ""
    return "\n".join(str(x).strip() for x in lines if str(x).strip())



def _symbol_safe(value: Any) -> str:
    s = _normalize_text(_coerce_text(value)).replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_+:/,.-]", "", s)
    return s[:120] or "unknown"



def _collapse_for_packet(text: str) -> str:
    text = _normalize_text(text)
    words = _word_tokens(text)
    return "_".join(w.lower() for w in words[:10]) or "request"



def _estimate_token_count(text: str) -> int:
    if not text:
        return 0
    rough_words = len(_word_tokens(text))
    punctuation = sum(1 for ch in text if ch in "{}[]():,._-/|\n")
    return max(1, int(round(rough_words * 1.30 + punctuation * 0.20)))


def _is_effectively_empty_payload_text(text: str) -> bool:
    norm = _normalize_text(text)
    return norm in {"", "{}", "[]", "null", "None"}



def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return values[-1] if values else None



def _infer_domain_from_advcu(bundle: Dict[str, Any]) -> str:
    lane = str(bundle.get("likely_lane") or "").strip().lower()
    output_type = str(bundle.get("output_type") or "").strip().lower()
    summary = str(bundle.get("semantic_summary") or "").lower()
    subject = str(bundle.get("subject") or "").lower()
    helper = bundle.get("helper_payload") if isinstance(bundle.get("helper_payload"), dict) else {}
    keywords = [str(x).lower() for x in list(helper.get("keywords") or [])]
    joined = " ".join([summary, subject] + keywords)

    if any(x in joined for x in ("weather", "forecast", "temperature")):
        return "WEATHER"
    if any(x in joined for x in ("word", "document", "report", "pdf", "presentation", "slides", "spreadsheet")) or output_type == "document":
        return "DOC"
    if any(x in joined for x in ("keyboard", "mouse", "printer", "rgb", "caps", "bluetooth", "wifi", "driver", "device")):
        return "DRV"
    if any(x in joined for x in ("unsubscribe", "spam", "trash", "email", "mail", "inbox")):
        return "EMAIL"
    if any(x in joined for x in ("network", "node", "mesh", "broker", "sync", "remote")) or lane == "network":
        return "NET"
    if any(x in joined for x in ("diagnostics", "gpu", "cpu", "thermal", "overheating", "ram", "vram")) or lane == "system":
        return "SYS"
    if lane == "creative":
        return "CRT"
    if any(x in joined for x in ("research", "browser", "search", "look up", "website", "article", "news")):
        return "RSH"
    if lane == "answer":
        return "QRY"
    if lane == "action":
        return "ACT"
    return "GEN"





def _advcu_semantic_packet_to_symbolic(packet: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    lane = str(packet.get("likely_lane") or "answer").lower()
    packet_type = _lane_to_packet_type(lane)
    query_type = _symbol_safe(packet.get("query_type") or "general_request")
    target = _symbol_safe(packet.get("target") or packet.get("subject") or "request")
    lines.append(f"{packet_type}[query_type:{query_type} target:{target}]")

    output_type = packet.get("output_type")
    if output_type:
        lines.append(f"CTX[output_type:{_symbol_safe(output_type)}]")

    attributes = packet.get("attributes") if isinstance(packet.get("attributes"), list) else []
    if attributes:
        lines.append(f"CTX[attributes:{_symbol_safe(','.join(str(x) for x in attributes[:4]))}]")

    action = packet.get("action")
    if action:
        lines.append(f"CTX[action:{_symbol_safe(action)}]")

    math_expr = packet.get("math_expr")
    if math_expr:
        lines.append(f"CTX[math_expr:{_symbol_safe(math_expr)}]")

    return lines[:6]



def _embed_strings(texts: Sequence[str], contracts: Dict[str, Any]) -> List[List[float]]:
    texts = [str(t or "") for t in texts]
    embed_fn = contracts.get("embed_text")
    if callable(embed_fn):
        try:
            vecs = embed_fn(texts)
            if isinstance(vecs, list) and vecs:
                return [list(v) for v in vecs]
        except Exception:
            pass

    st = _get_local_st()
    if st is not None:
        try:
            vecs = st.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]
        except Exception:
            pass

    return [_hash_embed(t) for t in texts]



def _hash_embed(text: str, dim: int = 64) -> List[float]:
    if not text:
        return [0.0] * dim
    h = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    vals = []
    for i in range(dim):
        b = h[i % len(h)]
        vals.append(math.sin((b + i) * 0.0174533))
    norm = math.sqrt(sum(v * v for v in vals)) or 1.0
    return [v / norm for v in vals]



def _cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    n = min(len(v1), len(v2))
    if n <= 0:
        return 0.0
    dot = sum(float(v1[i]) * float(v2[i]) for i in range(n))
    a = math.sqrt(sum(float(v1[i]) ** 2 for i in range(n))) or 1.0
    b = math.sqrt(sum(float(v2[i]) ** 2 for i in range(n))) or 1.0
    return _clamp(dot / (a * b), -1.0, 1.0)



def _semantic_alignment_score(raw_text: str, symbolic_packet_text: str, semantic_summary: str, advcu_bundle: Dict[str, Any], contracts: Dict[str, Any]) -> float:
    raw_norm = _normalize_text(raw_text).lower()
    sym_norm = _normalize_text(symbolic_packet_text).lower()
    sum_norm = _normalize_text(semantic_summary).lower()
    if not raw_norm or not sym_norm:
        return 0.0

    raw_tokens = set(_word_tokens(raw_norm))
    sym_tokens = set(_word_tokens(sym_norm))
    sum_tokens = set(_word_tokens(sum_norm)) if sum_norm else set()
    overlap = len(raw_tokens & sym_tokens) / max(1, len(raw_tokens | sym_tokens))
    summary_overlap = len(raw_tokens & sum_tokens) / max(1, len(raw_tokens | sum_tokens)) if sum_tokens else 0.0

    structural = 0.42
    if "weather" in sym_norm and any(x in raw_norm for x in ("weather", "temperature", "forecast")):
        structural += 0.20
    if "drv[" in sym_norm and any(x in raw_norm for x in ("keyboard", "rgb", "caps", "printer", "mouse")):
        structural += 0.24
    if "act[email" in sym_norm and any(x in raw_norm for x in ("unsubscribe", "spam", "trash", "email")):
        structural += 0.24
    if "doc[" in sym_norm and any(x in raw_norm for x in ("word", "report", "document", "presentation", "spreadsheet")):
        structural += 0.18
    if "state[awaiting_clarification]" in sym_norm:
        structural -= 0.10

    advcu_helper = advcu_bundle.get("helper_payload") if isinstance(advcu_bundle.get("helper_payload"), dict) else {}
    helper_text = _normalize_text(json.dumps(_minify_helper_payload(advcu_helper), separators=(",", ":"), ensure_ascii=False)).lower()
    helper_tokens = set(_word_tokens(helper_text)) if helper_text else set()
    helper_overlap = len(raw_tokens & helper_tokens) / max(1, len(raw_tokens | helper_tokens)) if helper_tokens else 0.0

    score = _clamp(0.34 * overlap + 0.18 * summary_overlap + 0.18 * helper_overlap + 0.30 * structural)

    embed_fn = contracts.get("embed_text") if isinstance(contracts, dict) else None
    if callable(embed_fn):
        try:
            vecs = _embed_strings([raw_norm, sym_norm, sum_norm or raw_norm], contracts)
            sim_raw_sym = (_cosine_similarity(vecs[0], vecs[1]) + 1.0) / 2.0
            sim_raw_sum = (_cosine_similarity(vecs[0], vecs[2]) + 1.0) / 2.0
            score = _clamp(0.68 * score + 0.22 * sim_raw_sym + 0.10 * sim_raw_sum)
        except Exception:
            pass
    return score





def _minify_helper_payload(helper_payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(helper_payload, dict):
        return {}
    out = {
        "i": helper_payload.get("intent"),
        "f": helper_payload.get("intent_family"),
        "t": helper_payload.get("task"),
        "tp": helper_payload.get("topic"),
        "a": helper_payload.get("attribute"),
        "as": list(helper_payload.get("attributes") or [])[:4],
        "k": list(helper_payload.get("keywords") or [])[:6],
        "qt": helper_payload.get("query_type"),
        "ae": helper_payload.get("action_expectation"),
        "o": helper_payload.get("output_type"),
        "mx": helper_payload.get("math_expr"),
        "s": helper_payload.get("semantic_summary"),
    }
    if isinstance(helper_payload.get("vision"), dict):
        vision = helper_payload["vision"]
        out["v"] = {
            "qt": vision.get("query_type"),
            "rs": vision.get("requested_subject"),
            "ra": list(vision.get("requested_attributes") or [])[:4],
            "ae": vision.get("action_expectation"),
            "ni": vision.get("needs_optical_input"),
        }
    return {k: v for k, v in out.items() if v not in (None, "", [], {})}



def _compact_payload_to_line(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    ordered_keys = ["v", "m", "fb", "i", "cf", "ln", "dm", "qt", "ot", "s", "at", "sum", "pkt", "amb", "cq", "raw"]
    parts: List[str] = []
    for key in ordered_keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if isinstance(value, list):
            compact = ";".join(_symbol_safe(x) for x in value[:6])
        elif isinstance(value, dict):
            compact = _symbol_safe(json.dumps(value, separators=(",", ":"), ensure_ascii=False))
        else:
            compact = _symbol_safe(value)
        parts.append(f"{key}={compact}")
    return "|".join(parts)



def _select_best_payload_text(
    compact_json_text: str,
    compact_line_text: str,
    helper_text: str,
    symbolic_packet_text: str,
    semantic_packet_text: str,
    raw_text: str,
    needs_clarification: bool,
    raw_downstream_text: str = "",
) -> Tuple[str, str]:
    candidates = {
        "compact_json": compact_json_text,
        "compact_line": compact_line_text,
        "helper_payload": helper_text,
        "symbolic_packet": symbolic_packet_text,
        "advcu_semantic_packet": semantic_packet_text,
    }
    usable = {k: _normalize_text(v) for k, v in candidates.items() if _normalize_text(v) and not _is_effectively_empty_payload_text(v)}
    baseline = _normalize_text(raw_downstream_text or raw_text)
    baseline_tokens = _estimate_token_count(baseline)
    if needs_clarification:
        priority = ["compact_json", "compact_line", "symbolic_packet", "helper_payload", "advcu_semantic_packet"]
        for key in priority:
            if key in usable:
                return usable[key], key
    best_key = None
    best_tokens = None
    for key, value in usable.items():
        tokens = _estimate_token_count(value)
        if best_tokens is None or tokens < best_tokens:
            best_key = key
            best_tokens = tokens
    if best_key is not None and best_tokens is not None:
        if best_tokens <= baseline_tokens:
            return usable[best_key], best_key
    return baseline, "raw_downstream"





def _minimal_context_meta(context_packet: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "session_id": str(context_packet.get("session_id") or _get_nested(context_packet, ("meta", "session_id"), "")),
        "source": str(context_packet.get("source") or ""),
        "mode": str(context_packet.get("mode") or ""),
    }



def _get_nested(obj: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur



def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


# ---------------------------------------------------------------------------
# Optional SQL schema helpers
# ---------------------------------------------------------------------------

CLARIFICATION_FRAMES_SQL = """
CREATE TABLE IF NOT EXISTS clarification_frames (
    frame_id TEXT PRIMARY KEY,
    session_id TEXT,
    query_id TEXT,
    parent_query_id TEXT,
    concept_group_id TEXT,
    original_input TEXT,
    normalized_text TEXT,
    partial_packet_json TEXT,
    unresolved_slots_json TEXT,
    clarification_question TEXT,
    clarification_answer TEXT,
    resolved_packet_json TEXT,
    state TEXT,
    confidence REAL,
    created_ts REAL,
    updated_ts REAL
);
""".strip()

CONTEXT_HISTORY_CLARIFICATION_SQL = """
CREATE TABLE IF NOT EXISTS context_history_clarification (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    concept_group_id TEXT,
    query_id TEXT,
    frame_id TEXT,
    original_input TEXT,
    clarification_question TEXT,
    clarification_answer TEXT,
    resolved_packet_json TEXT,
    state TEXT,
    created_ts REAL,
    updated_ts REAL
);
""".strip()


__all__ = [
    "analyze_text",
    "build_clarification_question",
    "merge_clarification_answer",
    "finalize_symbolic_packet",
    "build_clarification_frame_record",
    "build_llm_compact_payload",
    "compact_for_model",
    "estimate_token_reduction",
    "clear_analysis_cache",
    "CLARIFICATION_FRAMES_SQL",
    "CONTEXT_HISTORY_CLARIFICATION_SQL",
]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    samples = [
        "Give me the temperature right now in Nacogdoches, TX",
        "Open Word and write report for LA plant",
        "Turn my keyboard rgb blue and make sure caps lock is off",
        "G.Y.M.F.A.I.T.H.B.I.W.Y.A.",
        "Unsubscribe to spam and empty spam trash daily",
    ]
    demo_context = {
        "session_id": "demo-session",
        "source": "cli",
        "mode": "LOCAL",
        "pretoken_fast_only": True,
        "pretoken_disable_alignment": True,
    }
    for sample in samples:
        print("\n" + "=" * 90)
        print("INPUT:", sample)
        analysis = analyze_text(sample, context_packet=demo_context)
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
        if analysis.get("needs_clarification"):
            print("CLARIFICATION:", analysis.get("clarification_question"))
            merged = merge_clarification_answer(analysis, "2")
            print("MERGED:", json.dumps(merged.get("resolved_packet"), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ====================================================================
# END OF SarahMemoryPreTokenAnalyzer.py v8.0.0
# ====================================================================
