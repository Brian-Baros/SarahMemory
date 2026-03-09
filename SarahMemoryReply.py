"""--==The SarahMemory Project==--
File: SarahMemoryReply.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-03-01
Time: 10:11:54
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
https://store.sarahmemory.com
==============================================================================="""
# =============================================================================
#  Description:
#    Enhanced local DB path with vector-based recall and one-shot bootstrap of embeddings when missing.
#
#  PURPOSE
#  -------
#  Central reply-routing and response generation for the SarahMemory platform.
#  This module connects GUI input to the multi-pipeline reasoning system:
#   - Local fast paths (math, simple facts from DB/cache)
#   - Online API (OpenAI multi-model fallback via SarahMemoryAPI.py)
#   - Web Research (snippets, images, light scraping via SarahMemoryResearch.py)
#   - Offline/Local datasets (via SarahMemoryDatabase.py and friends)
#   - Personality integration and graceful fallbacks
#
#  DESIGN GOALS
#  ------------
#  * No syntax errors. No brittle kwargs. Backward-compatible entry points.
#  * Never block the GUI thread: heavy work happens outside or is short.
#  * Always return a unified "bundle" dict to the GUI:
#       {
#         "response": str,          # plain text ready for display
#         "image_url": Optional[str],  # a thumbnailable URL for GUI to render
#         "links": List[str],       # any hyperlinks to show in the chat
#         "meta": { ... },          # metadata for logging/analytics
#       }
#  * Operate both ONLINE and OFFLINE.
#  * Respect global configuration knobs set in SarahMemoryGlobals.py.
#  * Avoid renaming/removing existing public defs:
#       - generate_reply(self, user_text)
#       - route_reply(*args, **kwargs)     (back-compat alias)
#
#  SAFETY
#  ------
#  * "Safe math" evaluator supports + - * / ( ) and floats only.
#  * Web fetching is best-effort, never fatal to the routing.
#  * All imports are guarded; if a module is missing, we degrade gracefully.
#
#  LOGGING
#  -------
#  Uses logger "SarahMemoryReply". Emits INFO/WARNING/ERROR for traceability.
#  DB writes are optional; if DB layer is missing, logging continues in memory.
#
# =============================================================================

from __future__ import annotations

import os
import re
import sys
import json
import math
import time
import html
import queue
import threading
import types
import random
import string
import base64
import logging
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logger = logging.getLogger("SarahMemoryReply")
if not logger.handlers:
    _h = logging.StreamHandler(stream=sys.stdout)
    _fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# --- v7.7.4: uniform bundle stamper (ensures provenance + latency) ---
def _stamp_bundle(bundle: dict) -> dict:
    try:
        if not isinstance(bundle, dict):
            return bundle
        meta = bundle.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}

        source = meta.get("source") or bundle.get("source")
        if not source:
            pipe = meta.get("pipeline") or []
            source = pipe[-1] if pipe else ("offline" if meta.get("offline") else "local")
        intent = meta.get("intent") or bundle.get("intent") or "chat"

        response_text = _sm_sanitize_user_text(
            bundle.get("response") or bundle.get("presentation_reply") or bundle.get("reply") or bundle.get("content") or ""
        )

        meta["source"] = source
        meta["intent"] = intent
        meta.setdefault("outward_formatter", "SarahMemoryReply")
        meta.setdefault("presentation_only", True)
        meta.setdefault("latency_ms", 0)
        meta.pop("raw_answer", None)
        meta.pop("canonical_answer", None)
        meta.pop("trace_internal", None)

        bundle["ok"] = bool(bundle.get("ok", True))
        bundle["source"] = source
        bundle["intent"] = intent
        bundle["response"] = response_text
        bundle["presentation_reply"] = response_text
        bundle["reply"] = response_text
        bundle["content"] = response_text

        if not isinstance(bundle.get("links"), list):
            bundle["links"] = list(bundle.get("links") or [])
        if not isinstance(bundle.get("artifacts"), list):
            bundle["artifacts"] = list(bundle.get("artifacts") or [])
        if not isinstance(bundle.get("actions"), list):
            bundle["actions"] = list(bundle.get("actions") or [])
        if not isinstance(bundle.get("errors"), list):
            bundle["errors"] = list(bundle.get("errors") or [])

        image_url = bundle.get("image_url")
        if image_url and not any(isinstance(a, dict) and a.get("path") == image_url for a in bundle["artifacts"]):
            bundle["artifacts"].append({
                "type": "image",
                "path": image_url,
                "display_ready": True,
                "download_ready": True,
                "source": source,
            })

        bundle["meta"] = meta
    except Exception:
        pass
    return bundle


# === AUTO-PATCH CALL (module-level) ===
try:
    from SarahMemoryDatabase import ensure_core_schema
    ensure_core_schema()
except Exception:
    pass

# -----------------------------------------------------------------------------
# Guarded imports from sibling modules (graceful degradation)
# -----------------------------------------------------------------------------
def _try_neuron_reply(text_in: str, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Use SarahMemoryNeuron as the orchestration brain when available."""
    try:
        from SarahMemoryNeuron import neuron_route  # type: ignore
    except Exception:
        return None
    try:
        res = neuron_route(text_in, meta=meta)
        if not getattr(res, "ok", False):
            return None
        reply_txt = getattr(res, "reply", None)
        if not reply_txt:
            return None
        try:
            meta.setdefault("pipeline", []).append(f"neuron:{getattr(res,'source','neuron')}")
            meta["neuron_intent"] = getattr(res, "intent", None)
            meta["neuron_confidence"] = getattr(res, "confidence", None)
            meta["neuron_trace"] = getattr(res, "trace", None)
        except Exception:
            pass
        return _sm_make_outward_bundle(str(reply_txt), meta=meta, artifacts=_sm_creative_artifacts_from_meta(meta))
    except Exception as e:
        try:
            logger.warning(f"[neuron] route failed: {e}")
        except Exception:
            pass
        return None

try:
    import SarahMemoryGlobals as config
except Exception as _e:
    class _Config:
        # Minimal defaults when Globals is unavailable (offline-safe)
        API_PRIMARY_MODEL   = os.getenv("SARAH_OPENAI_PRIMARY_MODEL") or os.getenv("OPENAI_MODEL") or "default"
        API_SECONDARY_MODEL = os.getenv("SARAH_OPENAI_SECONDARY_MODEL") or os.getenv("OPENAI_MODEL_SECONDARY") or "default"
        API_DEFAULT_MODEL   = os.getenv("SARAH_OPENAI_DEFAULT_MODEL") or os.getenv("OPENAI_MODEL_DEFAULT") or "default"
        API_RESEARCH_ENABLED = True
        WEB_RESEARCH_ENABLED = True
        GUI_ALLOW_IMAGES = True
        LOCAL_ONLY_MODE = False
        SAFE_MODE = False
        def is_offline(self) -> bool:
            # Best-effort: treat no internet as offline (caller may override)
            return False
    config = _Config()
    logger.warning("[Globals] Missing or failed import; using minimal defaults: %s", _e)


try:
    from SarahMemoryGlobals import sm_is_core_module_approved, sm_get_registered_core_modules
except Exception:
    def sm_is_core_module_approved(_module_name: str) -> bool:
        return True
    def sm_get_registered_core_modules() -> dict:
        return {}

# Database / memory layer
try:
    from SarahMemoryDatabase import (
        search_answers,
        log_ai_functions_event,
        store_response_history,
        load_quick_facts,
        store_answer,
    )
except Exception as _e:
    logger.warning("[Database] Optional DB features unavailable: %s", _e)
    def search_answers(q: str) -> List[str]:
        return []
    def log_ai_functions_event(name: str, payload: Dict[str, Any]) -> None:
        pass
    def store_response_history(bundle: Dict[str, Any]) -> None:
        pass
    def load_quick_facts() -> Dict[str, str]:
        return {}

# Personality and adaptive layers
try:
    from SarahMemoryPersonality import (
        get_identity_response,
        get_generic_fallback_response,
        integrate_with_personality,
    )
except Exception as _e:
    logger.warning("[Personality] Fallback personality only: %s", _e)
    def get_identity_response(_: str) -> str:
        return "I'm Sarah, your AI companion."
    def get_generic_fallback_response(_: str) -> str:
        return "I'm still thinking—try rephrasing or ask something else!"
    def integrate_with_personality(text: str, meta: Dict[str, Any] | None = None) -> str:
        """Fallback shim that proxies to Personality.integrate_with_personality if present.
        Accepts (text[, meta]) to remain compatible with earlier/later signatures.
        """
        try:
            from SarahMemoryPersonality import integrate_with_personality as _pint
            try:
                # Prefer newer signature if available
                return _pint(text, meta=meta)  # type: ignore[call-arg]
            except TypeError:
                # Older versions only accept (text)
                return _pint(text)  # type: ignore[misc]
        except Exception:
            return text


# Web & research utilities
try:
    import SarahMemoryResearch as research
except Exception as _e:
    research = types.SimpleNamespace()
    logger.warning("[Research] Limited research features: %s", _e)

# API abstraction (multi-model OpenAI)
try:
    from SarahMemoryAPI import send_to_api
except Exception as _e:
    send_to_api = None
    logger.warning("[API] send_to_api unavailable: %s", _e)

# Intent & utility helpers
try:
    from SarahMemoryAiFunctions import (


        route_intent_response as _core_route_intent_response,
        detect_command_intent,
        normalize_text,
    )


except Exception as _e:
    _core_route_intent_response = None
    def detect_command_intent(_t: str) -> str: return "chat"
    def normalize_text(t: str) -> str: return (t or "").strip()
    logger.warning("[AiFunctions] Limited intent helpers: %s", _e)

# GUI utility hooks (OPTIONAL) - we don't depend on them here
try:
    from SarahMemoryGUI import voice, avatar
except Exception:
    voice = avatar = None


# --- Safe fallback: local vector-backed answer helper (returns None if unavailable) ---
def _vector_backed_local_answer(text: str):
    try:
        from SarahMemoryAiFunctions import vector_backed_local_answer as _v
        return _v(text)
    except Exception:
        return None


_LAST_MODEL_USED = None
# =============================================================================
# Data structures
# =============================================================================
@dataclass
class ReplyBundle:
    response: str
    image_url: Optional[str] = None
    links: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        response_text = _sm_sanitize_user_text(d.get("response") or "")
        if d.get("links") is None:
            d["links"] = []
        if d.get("meta") is None:
            d["meta"] = {}
        d.setdefault("artifacts", [])
        d.setdefault("actions", [])
        d.setdefault("errors", [])
        d["ok"] = True
        d["response"] = response_text
        d["presentation_reply"] = response_text
        d["reply"] = response_text
        d["content"] = response_text
        # Never expose raw/canonical internals through the outward bundle.
        try:
            meta = d.get("meta") or {}
            if isinstance(meta, dict):
                meta.pop("raw_answer", None)
                meta.pop("canonical_answer", None)
                meta.pop("trace_internal", None)
                d["meta"] = meta
        except Exception:
            pass
        return d



def _sm_make_outward_bundle(
    presentation_text: str,
    *,
    image_url: Optional[str] = None,
    links: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    actions: Optional[List[Dict[str, Any]]] = None,
    errors: Optional[List[str]] = None,
) -> Dict[str, Any]:
    meta = dict(meta or {})
    text = _sm_sanitize_user_text(presentation_text or "")
    bundle = ReplyBundle(text, image_url=image_url, links=list(links or []), meta=meta).to_dict()
    bundle["artifacts"] = list(artifacts or [])
    bundle["actions"] = list(actions or [])
    bundle["errors"] = list(errors or [])
    bundle["ok"] = True
    bundle["response"] = text
    bundle["presentation_reply"] = text
    bundle["reply"] = text
    bundle["content"] = text
    if image_url:
        bundle["artifacts"].append({
            "type": "image",
            "path": image_url,
            "display_ready": True,
            "download_ready": True,
            "source": meta.get("source") or "image",
        })
    return _stamp_bundle(bundle)

def _sm_creative_artifacts_from_meta(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    artifacts: List[Dict[str, Any]] = []
    try:
        trace = meta.get("neuron_trace") or {}
        raw_art = trace.get("artifacts") or meta.get("artifacts") or {}
        if isinstance(raw_art, dict):
            for key, value in raw_art.items():
                if value in (None, "", [], {}):
                    continue
                art_type = "file"
                if isinstance(value, str):
                    lv = value.lower()
                    if lv.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
                        art_type = "image"
                    elif lv.endswith((".mp4", ".mov", ".avi", ".webm")):
                        art_type = "video"
                    elif lv.endswith((".mp3", ".wav", ".flac", ".ogg")):
                        art_type = "audio"
                    elif lv.endswith((".html", ".htm")):
                        art_type = "webpage"
                artifacts.append({
                    "name": key,
                    "type": art_type,
                    "path": value,
                    "display_ready": True,
                    "download_ready": True,
                    "source": meta.get("source") or "creative",
                })
    except Exception:
        pass
    return artifacts

# =============================================================================
# Utilities — safe math & simple parsing
# =============================================================================
_MATH_ALLOWED = set("0123456789+-*/(). %")

def _looks_like_math(text: str) -> bool:
    t = (text or "").replace("what is", "").replace("=", "").strip()
    if not t:
        return False
    return all(c in _MATH_ALLOWED for c in t)

def _eval_safe_math(expr: str) -> Optional[str]:
    """
    Ultra-safe evaluator: supports + - * / % and parentheses.
    No names, no functions, no pow operator to avoid surprises.
    """
    try:
        clean = expr.replace("what is", "").replace("=", "").strip()
        if not clean or not _looks_like_math(clean):
            return None
        # Disallow exponent to keep it modest
        if "^" in clean or "**" in clean:
            return None
        # Evaluate in empty builtins
        result = eval(clean, {"__builtins__": {}}, {})
        return f"{clean} = {result}"
    except Exception as e:
        logger.debug("Math eval error: %s", e)
        return None

_IMG_TRIGGERS = ("show me a photo of ", "show me an image of ", "image of ", "picture of ", "show me ", "display ")

def _maybe_image_subject(text: str) -> Optional[str]:
    t = (text or "").strip().lower()
    for trig in _IMG_TRIGGERS:
        if t.startswith(trig):
            subj = t[len(trig):].strip().strip('"').strip("'")
            if subj:
                return subj
    # also handle "photo of X" mid-sentence
    m = re.search(r"\b(photo|image|picture)\s+of\s+(.+)$", t)
    if m:
        return m.group(2).strip()
    return None

_URL_RE = re.compile(r"(https?://[^\s]+)", re.I)

def _extract_links(s: str) -> List[str]:
    if not s:
        return []
    return _URL_RE.findall(s)


# =============================================================================
# Core pipeline helpers
# =============================================================================
def _try_local_db(text: str) -> Optional[str]:
    """Query local fast facts/answers first (offline friendly) with vector fallback."""
    try:
        facts = load_quick_facts()
        key = (text or "").strip().lower()
        if key in facts:
            return facts[key]
    except Exception as e:
        logger.debug("Quick facts load failed: %s", e)
    try:
        hits = search_answers(text)
        if hits:
            return hits[0]
    except Exception as e:
        logger.debug("Local DB search failed: %s", e)
    # Vector-based recall (auto-bootstrap if needed)
    vec_ans = _vector_backed_local_answer(text)
    if vec_ans:
        return vec_ans
    return None



def _try_api(text: str) -> Optional[str]:
    """Use multi-model OpenAI (via SarahMemoryAPI.send_to_api) and fall back to APIResearch.query.
    Always return a plain string or None."""
    if config is not None and getattr(config, "LOCAL_ONLY_MODE", False):
        return None

    # Primary path: direct SarahMemoryAPI -> send_to_api
    try:
        if send_to_api is not None:
            # Choose provider from Globals (single source of truth); default to current primary if available.
            try:
                _prov = None
                if hasattr(config, "get_active_api"):
                    _prov = config.get_active_api()  # type: ignore[attr-defined]
                if not _prov:
                    _prov = getattr(config, "PRIMARY_API", None)
                if not _prov:
                    _prov = "local_llm"
            except Exception:
                _prov = "local_llm"
            if not isinstance(_prov, str):
                _prov = str(_prov)
            resp = send_to_api(user_input=text, provider=_prov, intent="chat")

            # Track which model was actually used, if any
            try:
                globals()["_LAST_MODEL_USED"] = resp.get("model_used") if isinstance(resp, dict) else None
            except Exception:
                globals()["_LAST_MODEL_USED"] = None

            # If the API layer explicitly says "no API key", treat it as
            # "no cloud available" and let the pipeline fall through to
            # web/local/fallback instead of echoing that to the user.
            if isinstance(resp, dict):
                data_field = resp.get("data") or resp.get("text") or resp.get("response")
                if isinstance(data_field, str) and "API key" in data_field.lower():
                    return "I'm running in offline mode, so I can only use my built-in knowledge right now."


            # Normalize to a plain text string
            if isinstance(resp, dict):
                # common shapes: {'data': str} or {'choices':[{'message':{'content':...}}]}
                txt = resp.get("data") or resp.get("text") or resp.get("response")
                if not txt and "choices" in resp:
                    try:
                        ch0 = (resp.get("choices") or [{}])[0]
                        if isinstance(ch0, dict):
                            if isinstance(ch0.get("message"), dict):
                                txt = (ch0["message"].get("content") or "").strip()
                            else:
                                txt = (ch0.get("text") or "").strip()
                    except Exception:
                        txt = None
                if txt:
                    return str(txt).strip()
            elif isinstance(resp, str) and resp.strip():

                return resp.strip()
    except Exception as e:
        logger.debug("API call failed: %s", e)

    # Secondary path: research aggregator (normalizes 'data' consistently)
    try:
        if 'research' in globals() and hasattr(research, 'APIResearch'):
            r = research.APIResearch.query(text, intent="chat")
            if isinstance(r, dict):
                txt = r.get("data") or r.get("summary")
                if isinstance(txt, dict) and "summary" in txt:
                    txt = txt.get("summary")
                if txt:
                    return str(txt).strip()
    except Exception as e:
        logger.debug("API fallback failed: %s", e)

    return None


def _try_web(text: str) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Web research quick path: returns (summary, image_url, links)
    Uses best-effort helpers defined in SarahMemoryResearch.py
    """
    if config is not None and getattr(config, "LOCAL_ONLY_MODE", False):
        return None, None, []
    if not hasattr(research, "fetch_web_snippets") and not hasattr(research, "fetch_image_url"):
        return None, None, []
    summary = None
    image = None
    links: List[str] = []
    try:
        if hasattr(research, "fetch_web_snippets"):
            summary = research.fetch_web_snippets(text) or None
            # Detect CAPTCHA or blocked responses and treat as no-result
            if summary and ('DDG_CAPTCHA' in summary or 'captcha' in summary.lower()):
                summary = None
            links.extend(_extract_links(summary or ""))
    except Exception as e:
        logger.debug("Web snippet fetch failed: %s", e)
    try:
        subj = _maybe_image_subject(text)
        if subj and getattr(config, "GUI_ALLOW_IMAGES", True) and hasattr(research, "fetch_image_url"):
            image = research.fetch_image_url(subj) or None
    except Exception as e:
        logger.debug("Image fetch failed: %s", e)
    return summary, image, links


# -----------------------------------------------------------------------------
# Output sanitation (final-answer only)
# -----------------------------------------------------------------------------
# Some local LLMs echo role transcripts (system/user/assistant) and hidden thought tags.
# This scrubber ensures the GUI/TTS sees only the final answer.

def _sm_sanitize_user_text(text: str) -> str:
    """
    Sanitize model output so the UI/TTS only sees the final user-facing answer.
    Removes common prompt scaffolding (ROLE/INTENT/MOOD/TEMP/etc) and hidden blocks.
    """
    if not text:
        return ""
    t = str(text).replace("\r\n", "\n").replace("\r", "\n").strip()

    # Strip hidden reasoning blocks (common tags)
    try:
        t = re.sub(r"(?is)<\s*(think|analysis)\s*>.*?<\s*/\s*\1\s*>", "", t)
        t = re.sub(r"(?is)\[\s*(think|analysis)\s*\].*?\[\s*/\s*\1\s*\]", "", t)
    except Exception:
        pass

    # If an explicit assistant marker exists, keep only what follows the LAST one.
    try:
        matches = list(re.finditer(r"(?im)^(assistant)\s*:?\s*$", t))
        if matches:
            t = t[matches[-1].end():].strip()
        else:
            if re.search(r"(?im)^assistant\s*:\s*", t):
                parts = re.split(r"(?im)^assistant\s*:\s*", t)
                if parts:
                    t = parts[-1].strip()
    except Exception:
        pass

    # Remove common prompt header lines that local models sometimes echo.
    try:
        drop_pat = re.compile(
            r"(?i)^\s*(role|intent|tone|complexity|mood(?:\s*profile)?|context|query|parameters?|"
            r"temp(?:erature)?|top[_\s]?p|max[_\s]?tokens?|model(?:_used)?|provider|source)\s*[:=].*$"
        )
        cleaned = []
        for line in (t.split("\n")):
            if drop_pat.match(line or ""):
                continue
            # drop purely decorative provenance markers
            if re.match(r"(?i)^\s*\[\s*source\s*:[^\]]*\]\s*$", line or ""):
                continue
            if (line or "").strip() in ("[]", "[ ]"):
                continue
            cleaned.append(line)
        t = "\n".join(cleaned).strip()
    except Exception:
        pass

    # Strip trailing empty provenance tag (legacy)
    try:
        t = re.sub(r"\s*\[\s*\]\s*$", "", t).strip()
    except Exception:
        pass

    return t

def _finalize_text(raw, meta):
    txt = _sm_sanitize_user_text((raw or "").strip())
    if not txt:
        return ""
    if (meta or {}).get("intent") in {"identity", "math", "image", "web", "local"}:
        return txt
    # Personality styling + gentle follow-ups (inline to avoid GUI changes)
    try:
        from SarahMemoryPersonality import integrate_with_personality as _pint
        styled = _pint(txt)
    except Exception:
        styled = txt
    # Minimal follow-up generation using adaptive state
    try:
        from SarahMemoryAdaptive import advanced_emotional_learning
        emo = advanced_emotional_learning(txt) or {}
        bal = float(emo.get("emotional_balance", 0.0))
        cues = []
        if bal < -0.2:
            cues.append("Want me to keep this short and calm?")
        if bal > 0.25:
            cues.append("Want a creative spin on this?")
        if "?" not in txt:
            cues.append("Should I dig deeper on that?")
        if cues:
            styled = styled + "\n\n• " + "\n• ".join(cues[:2])
    except Exception:
        pass
    return styled
    try:
        return integrate_with_personality(txt)    # 1-arg call
    except Exception:
        return txt

# =============================================================================
# PUBLIC: Back-compat alias and main entry
# =============================================================================
def route_reply(*args, **kwargs) -> Dict[str, Any]:
    """
    Back-compat alias retained for older call sites.
    Accepts arbitrary kwargs (e.g., gui=self) and ignores unknown ones.
    """
    try:
        user_text = kwargs.get("user_text")
        if user_text is None and args:
            user_text = args[0]
        return generate_reply(None, user_text or "")
    except Exception as e:
        logger.error("route_reply failure: %s", e)
        return ReplyBundle("I'm having trouble routing that request.", meta={"error": str(e)}).to_dict()


def generate_reply(self, user_text: str) -> Dict[str, Any]:
    started = time.time()
    # Early bailout on interrupt
    try:
        # Use a local alias to avoid shadowing the module-level `config`
        import SarahMemoryGlobals as _cfg
        if getattr(_cfg, 'INTERRUPT_FLAG', False):
            setattr(_cfg, 'INTERRUPT_FLAG', False)
            bundle = {
                'response': 'Okay, stopping.',
                'image_url': None,
                'links': [],
                'meta': {'intent': 'interrupt', 'source': 'local'}
            }
            try:
                from SarahMemoryDatabase import store_response_history
                store_response_history(user_text, bundle['response'])
            except Exception:
                pass
            return _stamp_bundle(bundle)
    except Exception:
        pass

    """
    Main entry used by the GUI.
    Returns a bundle dict: {response, image_url?, links[], meta{}}
    """
    started = time.time()
    text_in = normalize_text(user_text) if 'normalize_text' in globals() else (user_text or "").strip()
    meta: Dict[str, Any] = {
        "intent": "chat",
        "pipeline": [],
        "offline": False,
        "model": None,
        "latency_ms": 0,
    }
    # --- System Management Intents via SMAPI ---
    try:
        from SarahMemorySMAPI import sm_api
        if text_in.lower() in ("system status", "check system", "status report"):
            meta["intent"] = "system_status"
            status = sm_api.get_system_status()
            out = json.dumps(status, indent=2)
            bundle = ReplyBundle(out, meta=meta).to_dict()
            return _stamp_bundle(bundle)

        if text_in.lower().startswith("set "):
            parts = text_in.split()
            if len(parts) >= 3:
                key = parts[1]
                value = " ".join(parts[2:])
                if sm_api.set_user_setting(key, value):
                    out = f"Updated setting '{key}' to '{value}'."
                else:
                    out = f"Failed to update '{key}'."
                bundle = ReplyBundle(out, meta=meta).to_dict()
                return _stamp_bundle(bundle)
    except Exception:
        pass

# ---------------------------------------------------------------------
# OFFLINE CHECK
# ---------------------------------------------------------------------


    # ---------------------------------------------------------------------
    # OFFLINE CHECK
    # ---------------------------------------------------------------------
    try:
        meta["offline"] = bool(config.is_offline()) if hasattr(config, "is_offline") else False
    except Exception:
        meta["offline"] = False


    # (0) Command/Agent control fast path
    try:
        from SarahMemoryAdvCU import classify_intent as _sm_classify
        _intent_guess = _sm_classify(text_in)
    except Exception:
        _intent_guess = None
    if _intent_guess == "command" or (text_in.split()[:1] and text_in.split()[0].lower() in ("open","launch","start","focus","maximize","minimize","close","exit","quit","type","press","click","double","scroll","move","drag")):
        if not getattr(config, "AI_AGENT_ENABLED", True):
            out = _finalize_text("Agent control is disabled in settings.", meta)
            bundle = ReplyBundle(out, meta=meta).to_dict()
            _store_history_safe(bundle)
            _trigger_av_voice_safe(self, out)
            return _stamp_bundle(bundle)
        # Route application lifecycle to Si, direct UI to AiFunctions
        handled = False
        first_word = (text_in.split()[:1] or [""])[0].lower()
        try:
            if first_word in ("open","launch","start","focus","maximize","minimize","close","exit","quit"):
                from SarahMemorySi import manage_application_request
                handled = bool(manage_application_request(text_in))
            else:
                from SarahMemoryAiFunctions import handle_ai_agent_command
                handle_ai_agent_command(text_in)
                handled = True
        except Exception as _e:
            logger.warning(f"Agent command failed: {_e}")
        msg = "Okay." if handled else "I tried, but that didn't seem to work."
        out = _finalize_text(msg, meta)
        bundle = ReplyBundle(out, meta=meta).to_dict()
        _store_history_safe(bundle)
        _trigger_av_voice_safe(self, out)
        return _stamp_bundle(bundle)

    # ---------------------------------------------------------------------
    # FAST PATHS
    # ---------------------------------------------------------------------
    # (1) Identity / greeting
    norm = re.sub(r"[^\w\s]", "", (text_in or "").lower()).strip()
    if norm in ("who are you", "what is your name"):
        ans = get_identity_response(text_in)  # canonical identity line
        meta["intent"] = "identity"
        out = _finalize_text(ans, meta)       # still run through personality
        bundle = ReplyBundle(out, meta=meta).to_dict()
        try:
            store_response_history(bundle)    # optional if DB present
        except Exception:
            pass
        # optional: voice/avatar trigger if your GUI hooks are available
        _trigger_av_voice_safe(self, out)
        # Store in QA cache (local + cloud) before returning
        try:
            if user_text and bundle.get("response"):
                store_answer(user_text, bundle["response"])
        except Exception as e:
            logger.warning(f"[QA STORE] Failed to store answer: {e}")

        return _stamp_bundle(bundle)

    # (2) Math
    if _looks_like_math(text_in):
        m = _eval_safe_math(text_in)
        if m:
            meta["intent"] = "math"
            meta["pipeline"].append("math")
            out = _finalize_text(m, meta)
            bundle = ReplyBundle(out, meta=meta).to_dict()
            _store_history_safe(bundle)
            _trigger_av_voice_safe(self, out)
            # Store in QA cache (local + cloud) before returning
            try:
                if user_text and bundle.get("response"):
                    store_answer(user_text, bundle["response"])
            except Exception as e:
                logger.warning(f"[QA STORE] Failed to store answer: {e}")
            return _stamp_bundle(bundle)

    # (3) Explicit image
    subj = _maybe_image_subject(text_in)
    if subj and getattr(config, "GUI_ALLOW_IMAGES", True):
        meta["intent"] = "image"
        meta["pipeline"].append("image")
        img_url = None
        if hasattr(research, "fetch_image_url"):
            try:
                img_url = research.fetch_image_url(subj)
            except Exception as e:
                logger.debug("image_url fetch error: %s", e)
        txt = f"Here is an image for **{subj}**." if img_url else f"I couldn't find an image for **{subj}**."
        out = _finalize_text(txt, meta)
        bundle = ReplyBundle(out, image_url=img_url, meta=meta).to_dict()
        _store_history_safe(bundle)
        _trigger_av_voice_safe(self, out)
        # Store in QA cache (local + cloud) before returning
        try:
            if user_text and bundle.get("response"):
                store_answer(user_text, bundle["response"])
        except Exception as e:
            logger.warning(f"[QA STORE] Failed to store answer: {e}")
        return _stamp_bundle(bundle)


    # ---------------------------------------------------------------------
    # LOCAL DB
    # ---------------------------------------------------------------------
    meta["pipeline"].append("local")
    local_ans = _try_local_db(text_in)
    if local_ans:
        out = _finalize_text(local_ans, meta)
        bundle = ReplyBundle(out, meta=meta).to_dict()
        _store_history_safe(bundle)
        _trigger_av_voice_safe(self, out)
        # Store in QA cache (local + cloud) before returning
        try:
            if user_text and bundle.get("response"):
                store_answer(user_text, bundle["response"])
        except Exception as e:
            logger.warning(f"[QA STORE] Failed to store answer: {e}")
        return _stamp_bundle(bundle)

    # ---------------------------------------------------------------------
    # NEURON ORCHESTRATOR (preferred multi-tier routing)
    # ---------------------------------------------------------------------
    try:
        nb = _try_neuron_reply(text_in, meta)
        if nb and nb.get("response"):
            out = _finalize_text(nb["response"], meta)
            nb["response"] = out
            _store_history_safe(nb)
            _trigger_av_voice_safe(self, out)
            try:
                if user_text and nb.get("response"):
                    store_answer(user_text, nb["response"])
            except Exception as e:
                logger.warning(f"[QA STORE] Failed to store answer: {e}")
            return _stamp_bundle(nb)
    except Exception:
        pass

    # ---------------------------------------------------------------------
    # API ONLINE (if allowed & not offline)
    # ---------------------------------------------------------------------
    api_txt = None
    if not meta["offline"] and not getattr(config, "LOCAL_ONLY_MODE", False):
        meta["pipeline"].append("api")
        api_txt = _try_api(text_in)
        if api_txt:
            out = _finalize_text(api_txt, meta)
            try:
                meta["model"] = globals().get("_LAST_MODEL_USED")
            except Exception:
                pass
            try:
                import SarahMemoryGlobals as _cfg
                setattr(_cfg, "LAST_PRIMARY_MODEL_USED", meta.get("model"))
            except Exception:
                pass

            links = _extract_links(out)
            bundle = ReplyBundle(out, links=links, meta=meta).to_dict()
            _store_history_safe(bundle)
            _trigger_av_voice_safe(self, out)
            # Store in QA cache (local + cloud) before returning
            try:
                if user_text and bundle.get("response"):
                    store_answer(user_text, bundle["response"])
            except Exception as e:
                logger.warning(f"[QA STORE] Failed to store answer: {e}")
            return _stamp_bundle(bundle)

    # ---------------------------------------------------------------------
    # WEB RESEARCH (if allowed & not offline)
    # ---------------------------------------------------------------------
    web_summary = web_image = None
    web_links: List[str] = []
    if not meta["offline"] and getattr(config, "WEB_RESEARCH_ENABLED", True):
        meta["pipeline"].append("web")
        web_summary, web_image, web_links = _try_web(text_in)
        if web_summary or web_image:
            out = _finalize_text(web_summary or "Here is what I found online.", meta)
            bundle = ReplyBundle(out, image_url=web_image, links=web_links, meta=meta).to_dict()
            _store_history_safe(bundle)
            _trigger_av_voice_safe(self, out)
            # Store in QA cache (local + cloud) before returning
            try:
                if user_text and bundle.get("response"):
                    store_answer(user_text, bundle["response"])
            except Exception as e:
                logger.warning(f"[QA STORE] Failed to store answer: {e}")
            return _stamp_bundle(bundle)

    # ---------------------------------------------------------------------
    # FINAL FALLBACK (personality-driven)
    # ---------------------------------------------------------------------
    meta["pipeline"].append("fallback")
    fallback = get_generic_fallback_response(text_in) or "I'm here, but I couldn't find an answer."
    out = _finalize_text(fallback, meta)

    bundle = ReplyBundle(out, meta=meta).to_dict()
    _store_history_safe(bundle)
    _trigger_av_voice_safe(self, out)

    # Latency
    meta["latency_ms"] = int((time.time() - started) * 1000)
    bundle["meta"]["latency_ms"] = meta["latency_ms"]

    # Store in QA cache (local + cloud) before returning
    try:
        if user_text and bundle.get("response"):
            store_answer(user_text, bundle["response"])
    except Exception as e:
        logger.warning(f"[QA STORE] Failed to store answer: {e}")

    return _stamp_bundle(bundle)


# =============================================================================
# Helpers — side effects are non-fatal
# =============================================================================
def _store_history_safe(bundle: Dict[str, Any]) -> None:
    try:
        safe_bundle = dict(bundle or {})
        safe_bundle["response"] = _sm_sanitize_user_text(safe_bundle.get("presentation_reply") or safe_bundle.get("response") or "")
        safe_bundle["presentation_reply"] = safe_bundle.get("response") or ""
        store_response_history(safe_bundle)
    except Exception as e:
        logger.debug("History store skipped: %s", e)

def _trigger_av_voice_safe(self_ref, text: str) -> None:
    if not text:
        return

    spoken_text = _sm_sanitize_user_text(text)
    if not spoken_text:
        return

    def _worker() -> None:
        try:
            if voice is not None:
                try:
                    if hasattr(voice, "synthesize_voice_async"):
                        voice.synthesize_voice_async(spoken_text)
                    elif hasattr(voice, "speak_text_async"):
                        voice.speak_text_async(spoken_text)
                    elif hasattr(voice, "queue_text"):
                        voice.queue_text(spoken_text)
                    elif hasattr(voice, "enqueue_speech"):
                        voice.enqueue_speech(spoken_text)
                    elif hasattr(voice, "synthesize_voice"):
                        voice.synthesize_voice(spoken_text)
                    elif hasattr(voice, "speak_text"):
                        voice.speak_text(spoken_text)
                except Exception as e:
                    logger.debug("Voice hook failed: %s", e)

            if avatar is not None:
                try:
                    dur = max(1.0, len(spoken_text.split()) / 2.0)
                    if hasattr(avatar, "simulate_lip_sync_async"):
                        avatar.simulate_lip_sync_async(dur)
                except Exception as e:
                    logger.debug("Avatar hook failed: %s", e)
        except Exception as e:
            logger.debug("AV hooks failed: %s", e)

    try:
        t = threading.Thread(target=_worker, name="SarahMemoryReplyAV", daemon=True)
        t.start()
    except Exception:
        try:
            _worker()
        except Exception as e:
            logger.debug("AV worker launch failed: %s", e)


# =============================================================================
# Diagnostics (optional, safe to import/run)
# =============================================================================
def _self_test() -> Dict[str, Any]:
    """Minimal self-test for CI / diagnostics."""
    samples = [
        "what is 5 + 5",
        "show me an image of winnie the pooh",
        "who are you?",
        "tell me specials on amazon today",
    ]
    results = []
    for s in samples:
        b = generate_reply(None, s)
        results.append({"q": s, "ok": bool(b and b.get("response"))})
    return {"ok": all(r["ok"] for r in results), "cases": results}


if __name__ == "__main__":
    out = _self_test()
    print(json.dumps(out, indent=2))
def render_provenance_footer(source_label, intent_label):
    try: return f"[Source: {source_label}] (Intent: {intent_label})"
    except Exception: return "[Source: Unknown] (Intent: undetermined)"
def _append_source_intent_to_reply(text, provenance=None, intent=None):
    src_lab=(provenance or {}).get("source","Unknown"); lab=intent or "undetermined"
    foot=render_provenance_footer(src_lab, lab)
    if text and not text.endswith("\n"): text=text+"\n"
    return (text or "") + foot


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
try:
    _ensure_response_table()
except Exception:
    pass

def _stamp_bundle(bundle: dict) -> dict:
    try:
        if not isinstance(bundle, dict):
            return bundle
        meta = bundle.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}

        source = meta.get("source") or bundle.get("source")
        if not source:
            pipe = meta.get("pipeline") or []
            source = pipe[-1] if pipe else ("offline" if meta.get("offline") else "local")
        intent = meta.get("intent") or bundle.get("intent") or "chat"

        response_text = _sm_sanitize_user_text(
            bundle.get("response") or bundle.get("presentation_reply") or bundle.get("reply") or bundle.get("content") or ""
        )

        meta["source"] = source
        meta["intent"] = intent
        meta.setdefault("outward_formatter", "SarahMemoryReply")
        meta.setdefault("presentation_only", True)
        meta.setdefault("latency_ms", 0)
        meta.pop("raw_answer", None)
        meta.pop("canonical_answer", None)
        meta.pop("trace_internal", None)

        bundle["ok"] = bool(bundle.get("ok", True))
        bundle["source"] = source
        bundle["intent"] = intent
        bundle["response"] = response_text
        bundle["presentation_reply"] = response_text
        bundle["reply"] = response_text
        bundle["content"] = response_text

        if not isinstance(bundle.get("links"), list):
            bundle["links"] = list(bundle.get("links") or [])
        if not isinstance(bundle.get("artifacts"), list):
            bundle["artifacts"] = list(bundle.get("artifacts") or [])
        if not isinstance(bundle.get("actions"), list):
            bundle["actions"] = list(bundle.get("actions") or [])
        if not isinstance(bundle.get("errors"), list):
            bundle["errors"] = list(bundle.get("errors") or [])

        image_url = bundle.get("image_url")
        if image_url and not any(isinstance(a, dict) and a.get("path") == image_url for a in bundle["artifacts"]):
            bundle["artifacts"].append({
                "type": "image",
                "path": image_url,
                "display_ready": True,
                "download_ready": True,
                "source": source,
            })

        bundle["meta"] = meta
    except Exception:
        pass
    return bundle
# --- v7.7.4: enforce provenance label in responses ---
def _sm_enforce_provenance(bundle):
    """
    Legacy hook. Keep provenance in meta ONLY; never append to the visible response text.
    Also acts as a last-chance sanitizer for bundle['response'].
    """
    try:
        if not isinstance(bundle, dict):
            return bundle
        meta = bundle.get("meta") if isinstance(bundle.get("meta"), dict) else {}
        src = meta.get("source") or meta.get("Source") or meta.get("provider") or bundle.get("source") or "Local"
        meta["source"] = src
        bundle["meta"] = meta
        if "response" in bundle and isinstance(bundle.get("response"), str):
            bundle["response"] = _sm_sanitize_user_text(bundle.get("response") or "")
    except Exception:
        pass
    return bundle

try:
    if 'generate_reply' in globals():
        _orig = generate_reply
        def generate_reply(*args, **kwargs):
            out = _orig(*args, **kwargs)
            if isinstance(out, dict):
                out = _sm_enforce_provenance(out)
                out = _stamp_bundle(out)
                return out
            return out
except Exception:
    pass


# v7.7.4 fallback: ensure emotion logging doesn't fail if table name differs
def _log_emotion_safe(emotion: str, intensity: float = 0.5):
    try:
        import os, sqlite3, time
        db = os.path.join(os.getcwd(), 'data', 'memory', 'datasets', 'system_logs.db')
        os.makedirs(os.path.dirname(db), exist_ok=True)
        con = sqlite3.connect(db); cur = con.cursor()
        cur.execute("INSERT INTO emotion_states(ts, emotion, intensity) VALUES (?,?,?)",
                    (time.strftime('%Y-%m-%dT%H:%M:%S'), emotion, float(intensity)))
        con.commit(); con.close()
    except Exception:
        try:
            con = sqlite3.connect(db); cur = con.cursor()
            cur.execute("INSERT INTO traits(ts, trait, value) VALUES (?,?,?)",
                        (time.strftime('%Y-%m-%dT%H:%M:%S'), emotion, float(intensity)))
            con.commit(); con.close()
        except Exception:
            pass

# ====================================================================
# END OF SarahMemoryReply.py v8.0.0
# ====================================================================