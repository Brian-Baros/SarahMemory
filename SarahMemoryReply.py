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
'The SarahMemory Companion AI-Bot Platform, SarahMemory AiOS, and all Parts of the SarahMemory Project are property of SOFTDEV0 LLC., & Brian Lee Baros'
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
# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "reply_router"
# CATEGORY = "response_generation"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "reply_pipeline"
# HARDWARE_DOMAIN = "audio_visual_optional"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "reply"
# FAMILY = "presentation"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Central reply-routing and response-bundle generator connecting GUI/WebUI input to local reasoning, research, personality styling, and presentation-safe output."
# --- SARAHMETA END ---
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

# -----------------------------------------------------------------------------
# Reply context / response cleanup / avatar speech cues
# -----------------------------------------------------------------------------
_FOLLOWUP_CONTEXT: Dict[str, Dict[str, Any]] = {}
_FOLLOWUP_MAX_AGE_SEC = 1800.0
_FOLLOWUP_MAX_ITEMS = 256

YES_RE = re.compile(r"^(yes|yeah|yep|yup|sure|ok|okay|please|do it|go on|continue|tell me more)\b", re.I)
NO_RE = re.compile(r"^(no|nope|nah|never mind|nevermind|stop|cancel|forget it|no thanks)\b", re.I)
IDENTITY_RE = re.compile(r"\b(what\s+is\s+your\s+name|who\s+are\s+you|who\s+(made|created|built|developed|engineered|designed)\s+you)\b", re.I)
BAD_UI_TRAIL_RE = re.compile(r"(?:\s*,?\s*\[\s*\]\s*)+$")
BAD_DIG_DEEPER_RE = re.compile(r"\n?\s*[•*-]?\s*Should\s+I\s+dig\s+deeper\s+on\s+that\?\s*(?:\[\s*\])?\s*$", re.I)


def _sm_prune_followup_context() -> None:
    try:
        now = time.time()
        stale = [k for k, v in (_FOLLOWUP_CONTEXT or {}).items() if (now - float((v or {}).get("ts") or 0.0)) > _FOLLOWUP_MAX_AGE_SEC]
        for k in stale:
            _FOLLOWUP_CONTEXT.pop(k, None)
        if len(_FOLLOWUP_CONTEXT) > _FOLLOWUP_MAX_ITEMS:
            ordered = sorted(_FOLLOWUP_CONTEXT.items(), key=lambda kv: float((kv[1] or {}).get("ts") or 0.0), reverse=True)
            keep = dict(ordered[:_FOLLOWUP_MAX_ITEMS])
            _FOLLOWUP_CONTEXT.clear()
            _FOLLOWUP_CONTEXT.update(keep)
    except Exception:
        pass


def _sm_current_session_key(meta: Optional[Dict[str, Any]] = None) -> str:
    meta = dict(meta or {})
    for k in ("session_id", "sessionId", "sid", "chat_id", "chatId", "client_id", "clientId", "user_id", "userId"):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    try:
        from flask import has_request_context, request  # type: ignore
        if has_request_context():
            payload = request.get_json(silent=True) or {}
            if isinstance(payload, dict):
                for k in ("session_id", "sessionId", "sid", "chat_id", "chatId", "client_id", "clientId", "user_id", "userId"):
                    v = payload.get(k)
                    if isinstance(v, str) and v.strip():
                        return v.strip()
            ua = (request.headers.get("User-Agent") or "")[:120]
            ip = request.headers.get("X-Forwarded-For") or request.remote_addr or "unknown"
            return f"{ip}|{ua}"
    except Exception:
        pass
    return str(meta.get("source") or "default")


def _sm_clean_reply_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    out = s
    try:
        out = out.replace("ðŸ™‚", "🙂").replace("ðŸ˜Š", "😊")
        out = BAD_DIG_DEEPER_RE.sub("", out)
        out = BAD_UI_TRAIL_RE.sub("", out)
        out = re.sub(r"^(The answer is\s+)+", "", out, flags=re.I)
    except Exception:
        pass
    return out.strip()


def _sm_topic_from_exchange(q: str, a: str) -> str:
    qn = (q or "").strip()
    if qn and len(qn) <= 120:
        return qn
    an = _sm_clean_reply_text(a or "")
    if an:
        first = re.split(r"(?<=[.!?])\s+|\n", an, maxsplit=1)[0].strip()
        if first:
            return first[:120]
    return (qn[:120] or "that topic").strip() or "that topic"


def _sm_should_fallback_identity(user_text: str, local_answer: str) -> bool:
    try:
        if not IDENTITY_RE.search(user_text or ""):
            return False
    except Exception:
        return False
    a = (local_answer or "").strip().lower()
    if not a:
        return True
    if len(a) < 10:
        return True
    if "openai" in a:
        return True
    if "[]" in a or "ðÿ" in a or "ðŸ" in a:
        return True
    return False


def _estimate_speech_duration_seconds(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.8
    words = max(1, len(t.split()))
    secs = words / 2.2
    if secs < 0.8:
        secs = 0.8
    if secs > 12.0:
        secs = 12.0
    return float(secs)


def _basic_mouth_cues(text: str, duration_s: float) -> List[Dict[str, Any]]:
    steps = max(6, min(30, int(duration_s * 10)))
    cues: List[Dict[str, Any]] = []
    if steps <= 1:
        return [{"t": 0, "v": 0.6}, {"t": int(duration_s * 1000), "v": 0.0}]
    for i in range(steps):
        t_ms = int((i / max(1, steps - 1)) * duration_s * 1000)
        v = 0.15 + (0.55 if (i % 2 == 0) else 0.30)
        cues.append({"t": t_ms, "v": v})
    cues.append({"t": int(duration_s * 1000), "v": 0.0})
    return cues


def _sm_attach_avatar_speech(meta: Dict[str, Any], text: str) -> None:
    try:
        if not isinstance(meta, dict):
            return
        txt = _sm_clean_reply_text(_sm_sanitize_user_text(text or "", for_tts=True))
        if not txt:
            return
        dur_s = _estimate_speech_duration_seconds(txt)
        meta["avatar_speech"] = {
            "speak": True,
            "duration_ms": int(dur_s * 1000),
            "cues": _basic_mouth_cues(txt, dur_s),
            "ts": time.time(),
        }
    except Exception:
        pass


def _sm_store_followup_context(meta: Optional[Dict[str, Any]], user_text: str, reply_text: str) -> None:
    try:
        if not user_text or not reply_text:
            return
        meta = dict(meta or {})
        if str(meta.get("intent") or "").lower() in {"interrupt", "followup_no"}:
            return
        _sm_prune_followup_context()
        key = _sm_current_session_key(meta)
        _FOLLOWUP_CONTEXT[key] = {
            "q": str(user_text or "").strip(),
            "a": str(reply_text or "").strip(),
            "topic": _sm_topic_from_exchange(str(user_text or ""), str(reply_text or "")),
            "ts": time.time(),
        }
    except Exception:
        pass


def _sm_resolve_followup_input(user_text: str, meta: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    text = (user_text or "").strip()
    if not text:
        return None, text
    meta = meta if isinstance(meta, dict) else {}
    _sm_prune_followup_context()
    key = _sm_current_session_key(meta)
    prev = _FOLLOWUP_CONTEXT.get(key)
    if not prev:
        return None, text
    try:
        if (time.time() - float(prev.get("ts") or 0.0)) > _FOLLOWUP_MAX_AGE_SEC:
            _FOLLOWUP_CONTEXT.pop(key, None)
            return None, text
    except Exception:
        pass

    if len(text) <= 40 and YES_RE.match(text):
        topic = prev.get("topic") or _sm_topic_from_exchange(str(prev.get("q") or ""), str(prev.get("a") or ""))
        meta["followup_original_text"] = text
        meta["followup_of"] = str(prev.get("q") or topic)
        meta.setdefault("pipeline", []).append("followup_yes")
        return None, f"Go deeper on this and explain more clearly: {topic}"

    if len(text) <= 40 and NO_RE.match(text):
        _FOLLOWUP_CONTEXT.pop(key, None)
        meta["intent"] = "followup_no"
        meta.setdefault("pipeline", []).append("followup_no")
        bundle = ReplyBundle(
            "No problem — what would you like to talk about next?",
            meta=meta,
        ).to_dict()
        return bundle, text

    return None, text


# --- v7.7.4: uniform bundle stamper (ensures provenance + latency) ---
def _stamp_bundle(bundle: dict) -> dict:
    try:
        if not isinstance(bundle, dict):
            return bundle

        meta = bundle.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}

        prompt_text = meta.get("_prompt_text")
        source = meta.get("source") or bundle.get("source")
        if not source:
            pipe = meta.get("pipeline") or []
            source = pipe[-1] if pipe else ("offline" if meta.get("offline") else "local")
        intent = meta.get("intent") or bundle.get("intent") or "chat"

        response_text = _sm_sanitize_user_text(
            bundle.get("response") or bundle.get("presentation_reply") or bundle.get("reply") or bundle.get("content") or "",
            user_text=prompt_text,
        )
        response_text = _sm_clean_reply_text(response_text)

        if _sm_should_fallback_identity(str(prompt_text or meta.get("followup_of") or ""), response_text):
            response_text = "My name is Sarah — your SarahMemory AI companion (SarahMemory AiOS)."

        if not response_text:
            vision_txt = _sm_render_vision_reply(meta, bundle)
            if vision_txt:
                response_text = vision_txt

        reply_gate = meta.get("reply_gate") if isinstance(meta.get("reply_gate"), dict) else {}
        reply_allowed = bool(reply_gate.get("reply_allowed", True))
        if response_text:
            _sm_attach_avatar_speech(meta, response_text)
            if reply_allowed:
                meta.setdefault(
                    "followup_hint",
                    "If you want more detail, reply: \"yes\" or ask a specific angle (history, examples, pros/cons).",
                )
            else:
                meta.setdefault(
                    "followup_hint",
                    "This result is not finalized yet. SarahMemory is holding outward success until verification is complete.",
                )
                meta.setdefault("completion_verified", False)

        meta["source"] = source
        meta["intent"] = intent
        meta.setdefault("outward_formatter", "SarahMemoryReply")
        meta.setdefault("presentation_only", True)
        meta.setdefault("latency_ms", 0)
        meta.pop("raw_answer", None)
        meta.pop("canonical_answer", None)
        meta.pop("trace_internal", None)
        meta.pop("_prompt_text", None)

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
        if response_text and bool((meta.get("reply_gate") if isinstance(meta.get("reply_gate"), dict) else {}).get("reply_allowed", True)):
            _sm_store_followup_context(meta, str(prompt_text or meta.get("followup_original_text") or meta.get("followup_of") or ""), response_text)
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

        try:
            res_dict = res.to_dict() if hasattr(res, "to_dict") else {
                "ok": getattr(res, "ok", False),
                "reply": getattr(res, "reply", ""),
                "confidence": getattr(res, "confidence", None),
                "intent": getattr(res, "intent", None),
                "source": getattr(res, "source", "neuron"),
                "artifacts": getattr(res, "artifacts", {}) or {},
                "trace": getattr(res, "trace", {}) or {},
            }
        except Exception:
            res_dict = {
                "ok": True,
                "reply": getattr(res, "reply", "") or "",
                "confidence": getattr(res, "confidence", None),
                "intent": getattr(res, "intent", None),
                "source": getattr(res, "source", "neuron"),
                "artifacts": {},
                "trace": {},
            }

        reply_txt = str(res_dict.get("reply") or "").strip()
        trace = res_dict.get("trace") if isinstance(res_dict.get("trace"), dict) else {}
        artifacts_raw = res_dict.get("artifacts") or {}
        actions = list(res_dict.get("actions") or []) if isinstance(res_dict.get("actions"), list) else []

        try:
            meta.setdefault("pipeline", []).append(f"neuron:{res_dict.get('source') or 'neuron'}")
            meta["neuron_intent"] = res_dict.get("intent")
            meta["neuron_confidence"] = res_dict.get("confidence")
            meta["neuron_trace"] = trace
            if isinstance(trace.get("vision_findings"), dict):
                meta["vision_findings"] = trace.get("vision_findings")
        except Exception:
            pass

        gate = _sm_reply_gate_from_neuron(text_in, res_dict, meta)
        if gate.get("active"):
            compass_packet = gate.get("packet") if isinstance(gate.get("packet"), dict) else {}
            meta["compass"] = compass_packet
            meta["reply_gate"] = {
                "active": True,
                "origin": gate.get("origin"),
                "reply_allowed": bool(compass_packet.get("reply_allowed", False)),
                "continue_allowed": bool(compass_packet.get("continue_allowed", False)),
                "status": str(compass_packet.get("status") or ""),
                "next_required_step": str(compass_packet.get("next_required_step") or ""),
                "reason": str(compass_packet.get("reason") or ""),
            }
            if not bool(compass_packet.get("reply_allowed", False)):
                reply_txt = _sm_reply_text_for_compass_hold(text_in, res_dict, compass_packet)
                meta["presentation_state"] = "hold"
            else:
                meta["presentation_state"] = "final"

        if not reply_txt:
            reply_txt = _sm_render_vision_reply(meta, {"trace": trace, "artifacts": artifacts_raw}) or ""
        if not reply_txt:
            return None

        reply_txt = _sm_humanize_ticket_reply(reply_txt, str(res_dict.get("source") or ""), artifacts_raw)

        artifacts = []
        try:
            if isinstance(artifacts_raw, list):
                artifacts.extend([a for a in artifacts_raw if isinstance(a, dict)])
            elif isinstance(artifacts_raw, dict):
                for key, value in artifacts_raw.items():
                    if value in (None, "", [], {}):
                        continue
                    artifacts.append({
                        "name": key,
                        "type": "file",
                        "path": value if isinstance(value, str) else json.dumps(value),
                        "display_ready": True,
                        "download_ready": True,
                        "source": str(res_dict.get("source") or "neuron"),
                    })
        except Exception:
            artifacts = []

        try:
            artifacts.extend(_sm_creative_artifacts_from_meta({
                "source": str(res_dict.get("source") or "neuron"),
                "artifacts": artifacts_raw,
                "neuron_trace": trace,
            }))
        except Exception:
            pass

        return _sm_make_outward_bundle(
            str(reply_txt),
            meta=meta,
            artifacts=artifacts,
            actions=actions,
            raw_answer=str(reply_txt),
            trace_internal=trace if isinstance(trace, dict) else {},
        )
    except Exception as e:
        try:
            logger.warning(f"[neuron] route failed: {e}")
        except Exception:
            pass
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

# V10/V9C persistence guard for volatile SelfAware body facts.
def _sm_is_volatile_body_fact_query(text: str) -> bool:
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


def _sm_meta_blocks_persistence(meta: Optional[Dict[str, Any]] = None, text: str = "") -> bool:
    meta = meta if isinstance(meta, dict) else {}
    if bool(meta.get("do_not_write_sql") or meta.get("do_not_persist") or meta.get("do_not_learn") or meta.get("volatile_runtime_fact")):
        return True
    pkg = meta.get("verified_artifact_package") if isinstance(meta.get("verified_artifact_package"), dict) else {}
    if pkg and bool(pkg.get("volatile") or pkg.get("do_not_write_sql") or pkg.get("volatile_runtime_fact")):
        return True
    cp = meta.get("chat_classification_packet") if isinstance(meta.get("chat_classification_packet"), dict) else {}
    if cp and str(cp.get("domain") or "") == "selfaware_body":
        return True
    return _sm_is_volatile_body_fact_query(text)


try:
    _sm_raw_store_answer = store_answer  # type: ignore[name-defined]
    def store_answer(query, answer):  # type: ignore[no-redef]
        if _sm_meta_blocks_persistence(text=str(query or "")):
            logger.info("[V10/V9C] Skipped QA cache store for volatile SelfAware body fact query.")
            return False
        return _sm_raw_store_answer(query, answer)
except Exception:
    pass

try:
    _sm_raw_search_answers = search_answers  # type: ignore[name-defined]
    def search_answers(query):  # type: ignore[no-redef]
        if _sm_is_volatile_body_fact_query(str(query or "")):
            logger.info("[V10/V9C] Skipped QA cache recall for volatile SelfAware body fact query.")
            return []
        return _sm_raw_search_answers(query)
except Exception:
    pass


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
        meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
        response_text = _sm_sanitize_user_text(d.get("response") or "", user_text=meta.get("_prompt_text"))
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
    raw_answer: Optional[str] = None,
    canonical_answer: Optional[str] = None,
    trace_internal: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    meta = dict(meta or {})
    if raw_answer is not None:
        meta["raw_answer"] = raw_answer
    if canonical_answer is not None:
        meta["canonical_answer"] = canonical_answer
    if trace_internal is not None:
        meta["trace_internal"] = trace_internal
    text = _sm_sanitize_user_text(presentation_text or "", user_text=meta.get("_prompt_text"))
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

def _sm_strip_selfaware_artifact_text(text: str, kind: str = "") -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    try:
        value = re.sub(r"^\s*Verified\s+SelfAware\s+fact\s*\([^)]*\)\s*:\s*", "", value, flags=re.I).strip()
        labels = {
            "cpu": "CPU",
            "gpu": "GPU",
            "motherboard": "Motherboard",
            "memory": "Memory",
            "network": "Network adapters",
            "network_card": "Network adapters",
            "operating_system": "Operating platform",
        }
        for label in [labels.get(str(kind or "").lower(), ""), "CPU", "GPU", "Motherboard", "Memory", "Network adapters", "Operating platform"]:
            if label and value.lower().startswith((label + " =").lower()):
                value = value[len(label) + 2:].strip()
                break
    except Exception:
        pass
    return value.strip()


def _sm_format_thermal_artifact_response(user_text: str, package: Dict[str, Any], artifact_text: str) -> Optional[str]:
    try:
        pkg = package if isinstance(package, dict) else {}
        value = pkg.get("artifact_value") if isinstance(pkg.get("artifact_value"), dict) else {}
        binding = pkg.get("thermal_sensor_binding") if isinstance(pkg.get("thermal_sensor_binding"), dict) else {}
        if not value and binding:
            value = binding
        if not isinstance(value, dict) or not value.get("thermal_case"):
            return None
        selected = value.get("selected_reading") if isinstance(value.get("selected_reading"), dict) else {}
        component = str(value.get("requested_component") or selected.get("component") or "body_thermal").replace("_", " ")
        temp = selected.get("temperature_c")
        source_type = str(selected.get("source_type") or "").replace("_", " ")
        status = str(value.get("sensor_binding_status") or "unavailable").replace("_", " ")
        if temp not in (None, ""):
            if component == "cpu" and "motherboard" in source_type:
                return f"I do not currently have a direct CPU temperature reading from a CPU thermal probe. This CPU is verified on my motherboard, and the motherboard exposes a CPU-related thermal sensor. Based on that verified board sensor, my CPU temperature is currently {temp}°C."
            if component == "gpu":
                return f"My GPU temperature is currently {temp}°C."
            if component == "motherboard":
                return f"My motherboard-related temperature reading is currently {temp}°C."
            return f"My currently verified {component} temperature is {temp}°C."
        if component == "cpu":
            related = value.get("related_readings") if isinstance(value.get("related_readings"), dict) else {}
            gpu_temp = related.get("gpu_temp_c")
            if gpu_temp not in (None, ""):
                return f"I do not currently have a verified CPU temperature reading. The related live thermal reading I can verify is GPU temperature at {gpu_temp}°C."
            return "I do not currently have a verified CPU temperature reading. Direct CPU sensor visibility is unavailable, and no mapped motherboard CPU-related thermal sensor has been approved yet."
        return f"I do not currently have a verified {component} temperature reading. Sensor binding status: {status}."
    except Exception:
        return None


def present_verified_artifact_answer(user_text: str, package: Dict[str, Any], meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Presentation boundary for volatile SelfAware hardware/body artifacts."""
    meta = dict(meta or {})
    pkg = package if isinstance(package, dict) else {}
    kind = str(pkg.get("artifact_kind") or meta.get("fact_kind") or "hardware_fact").strip().lower()
    artifact_text = _sm_strip_selfaware_artifact_text(str(pkg.get("artifact_text") or ""), kind=kind)
    if not artifact_text:
        artifact_text = _sm_strip_selfaware_artifact_text(str(pkg.get("artifact_value") or ""), kind=kind)
    low = str(user_text or "").lower()
    wants_court = any(k in low for k in ("evidence court", "court result", "quorum", "confidence", "proof", "show evidence"))
    court = pkg.get("evidence_court") if isinstance(pkg.get("evidence_court"), dict) else {}
    if wants_court:
        response = f"SelfAware verified this {kind.replace('_', ' ')} at {court.get('quorum') or 'unknown quorum'} quorum with {court.get('confidence') or 'unknown'} confidence: {artifact_text}."
    elif kind == "temperature":
        response = _sm_format_thermal_artifact_response(user_text, pkg, artifact_text) or f"My currently verified thermal result is: {artifact_text}."
    elif kind == "cpu":
        article = "an" if artifact_text[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
        response = f"I currently have {article} {artifact_text}."
    elif kind == "gpu":
        response = f"My currently verified graphics hardware is {artifact_text}."
    elif kind == "motherboard":
        response = f"My currently verified motherboard is {artifact_text}."
    elif kind in {"network", "network_card", "wifi_card", "ethernet_card", "bluetooth_card", "lan"}:
        response = f"My currently verified network adapter summary is: {artifact_text}. Sensitive IP and MAC details are not exposed."
    else:
        response = f"My currently verified {kind.replace('_', ' ')} result is: {artifact_text}."
    meta.update({
        "source": meta.get("source") or "selfaware_verified_artifact",
        "engine": meta.get("engine") or "SarahMemoryReply.present_verified_artifact_answer",
        "intent": meta.get("intent") or "system_status",
        "verified_artifact_package": pkg,
        "compare_mode": "integrity_security_only",
        "do_not_reverify": True,
        "do_not_write_sql": True,
        "do_not_persist": True,
        "do_not_learn": True,
        "do_not_promote_to_memory": True,
        "volatile_runtime_fact": True,
    })
    return _sm_make_outward_bundle(
        response,
        meta=meta,
        artifacts=[{"name": "verified_artifact_package", "type": "volatile_verified_hardware_fact", "verified_artifact_package": pkg, "display_ready": False, "download_ready": False}],
        raw_answer=response,
    )


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

def _try_websym_math(text: str) -> Optional[str]:
    """Prefer WebSYM calculator-style math answers when available; otherwise return None."""
    try:
        from SarahMemoryWebSYM import WebSemanticSynthesizer, is_math_expression  # type: ignore
    except Exception:
        return None

    q = (text or "").strip()
    if not q:
        return None

    try:
        if hasattr(WebSemanticSynthesizer, "is_math_query"):
            if not bool(WebSemanticSynthesizer.is_math_query(q)):
                if not (callable(is_math_expression) and is_math_expression(q)):
                    return None
        else:
            if not (callable(is_math_expression) and is_math_expression(q)):
                return None
    except Exception:
        pass

    try:
        if hasattr(WebSemanticSynthesizer, "sarah_calculator"):
            ans = WebSemanticSynthesizer.sarah_calculator(q, original_query=q)
            ans = str(ans or "").strip()
            return ans or None
    except Exception:
        return None
    return None


def _sm_extract_vision_findings(meta: Optional[Dict[str, Any]], bundle: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    meta = meta if isinstance(meta, dict) else {}
    bundle = bundle if isinstance(bundle, dict) else {}
    candidates = [
        bundle.get("vision_findings"),
        meta.get("vision_findings"),
        (meta.get("neuron_trace") or {}).get("vision_findings") if isinstance(meta.get("neuron_trace"), dict) else None,
        (bundle.get("trace") or {}).get("vision_findings") if isinstance(bundle.get("trace"), dict) else None,
    ]
    for cand in candidates:
        if isinstance(cand, dict) and cand:
            return cand
    return None


def _sm_render_vision_reply(meta: Optional[Dict[str, Any]], bundle: Optional[Dict[str, Any]] = None) -> str:
    data = _sm_extract_vision_findings(meta, bundle)
    if not isinstance(data, dict) or not data:
        return ""

    request = data.get("vision_request") if isinstance(data.get("vision_request"), dict) else {}
    findings = data.get("vision_findings") if isinstance(data.get("vision_findings"), dict) else data

    query_type = str(request.get("query_type") or findings.get("query_type") or "").strip().lower()
    subject = str(findings.get("resolved_subject") or request.get("requested_subject") or findings.get("subject") or "item").strip() or "item"
    resolved_attributes = findings.get("resolved_attributes") if isinstance(findings.get("resolved_attributes"), dict) else {}
    observations = findings.get("observed_subjects") if isinstance(findings.get("observed_subjects"), list) else []
    detections = findings.get("detections") if isinstance(findings.get("detections"), list) else []
    ocr = findings.get("ocr") if isinstance(findings.get("ocr"), dict) else {}
    tracking = findings.get("tracking") if isinstance(findings.get("tracking"), dict) else {}
    safety = findings.get("safety") if isinstance(findings.get("safety"), dict) else {}
    distance = findings.get("distance") if isinstance(findings.get("distance"), dict) else {}

    color = resolved_attributes.get("color") or findings.get("color")
    if query_type == "identify_color" and color:
        return f"It looks like your {subject} is {color}."

    if query_type == "read_text":
        text_val = str(ocr.get("text") or findings.get("text") or "").strip()
        if text_val:
            return f"The visible text says: {text_val}"
        return "I couldn't read any clear text from the current view."

    if query_type == "detect_objects":
        labels = []
        for d in detections:
            if isinstance(d, dict):
                label = d.get("label") or d.get("name") or d.get("class")
                if label:
                    labels.append(str(label))
        labels.extend([str(x) for x in observations if x])
        labels = list(dict.fromkeys(labels))
        if labels:
            return "I can see: " + ", ".join(labels[:8]) + "."

    if query_type == "detect_faces":
        face_count = findings.get("face_count")
        if isinstance(face_count, int):
            if face_count == 1:
                return "I can see one face."
            return f"I can see {face_count} faces."

    if query_type == "track_motion":
        moving = tracking.get("moving_objects")
        if isinstance(moving, int):
            if moving <= 0:
                return "I do not detect significant motion right now."
            return f"I detect motion from {moving} moving object{'s' if moving != 1 else ''}."

    if query_type == "estimate_distance":
        val = distance.get("value") or findings.get("distance_value")
        unit = distance.get("unit") or findings.get("distance_unit") or "units"
        if val not in (None, ""):
            return f"The estimated distance to the {subject} is about {val} {unit}."

    if query_type == "inspect_safety_zone":
        violation = safety.get("hazard_zone_violation")
        if violation is True:
            return "A safety issue is detected: the observed subject appears to be inside the configured danger zone."
        if violation is False:
            return "The observed subject appears to be outside the configured danger zone."

    if query_type == "assess_style_or_fit":
        summary = str(findings.get("style_assessment") or findings.get("summary") or "").strip()
        if summary:
            return summary

    if query_type == "scene_summary":
        summary = str(findings.get("summary") or "").strip()
        if summary:
            return summary

    if color:
        return f"It looks like the {subject} is {color}."
    if observations:
        return "From the current view, I can see: " + ", ".join([str(x) for x in observations[:8]]) + "."
    return ""




def _sm_is_operational_neuron_result(source: str, artifacts: Any, trace: Any) -> bool:
    src = str(source or "").strip().lower()
    if src in {"action_ticket", "creative_ticket", "ingress_router", "governance"}:
        return True
    try:
        primary_lane = str((trace or {}).get("primary_lane") or "").strip().lower()
        if primary_lane in {"action", "creative", "system", "network"}:
            return True
    except Exception:
        pass
    try:
        if isinstance(artifacts, dict):
            for key in ("action_ticket", "job_ticket", "route_ticket", "execution_plan", "execution_result", "smget"):
                if key in artifacts:
                    return True
    except Exception:
        pass
    return False


def _sm_humanize_ticket_reply(reply_txt: str, source: str, artifacts: Any) -> str:
    raw = str(reply_txt or "").strip()
    src = str(source or "").strip().lower()
    artifacts = artifacts if isinstance(artifacts, dict) else {}

    if raw.startswith("ACTION_TICKET::") or src == "action_ticket":
        ticket = artifacts.get("action_ticket") if isinstance(artifacts.get("action_ticket"), dict) else {}
        action_name = str(ticket.get("action") or "requested action").replace("_", " ").strip()
        return f"The action request has been staged for governed execution: {action_name}."

    if raw.startswith("CREATIVE_JOB_TICKET::") or src == "creative_ticket":
        ticket = artifacts.get("job_ticket") if isinstance(artifacts.get("job_ticket"), dict) else {}
        kind = str(ticket.get("kind") or "creative").replace("_", " ").strip()
        return f"The {kind} job has been prepared and is awaiting completion and verification."

    if src == "ingress_router" and raw:
        if "direct internal execution completed" in raw.lower():
            return "The request was handed into the governed execution path, but final completion is still awaiting verification."

    return raw


def _sm_build_compass_plan_state_from_neuron(res_dict: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    trace = res_dict.get("trace") if isinstance(res_dict.get("trace"), dict) else {}
    artifacts = res_dict.get("artifacts") if isinstance(res_dict.get("artifacts"), dict) else {}
    source = str(res_dict.get("source") or "").strip().lower()
    reply_txt = str(res_dict.get("reply") or "").strip()

    plan_state: Dict[str, Any] = {}
    proposed_action: Dict[str, Any] = {}

    if isinstance(trace.get("compass"), dict):
        plan_state["compass"] = dict(trace.get("compass") or {})

    if isinstance(trace.get("governance"), dict):
        gov = dict(trace.get("governance") or {})
        plan_state["governance"] = gov
        if isinstance(gov.get("smget"), dict):
            plan_state["smget"] = dict(gov.get("smget") or {})
            proposed_action["smget"] = dict(gov.get("smget") or {})

    if isinstance(artifacts.get("smget"), dict):
        plan_state["smget"] = dict(artifacts.get("smget") or {})
        proposed_action["smget"] = dict(artifacts.get("smget") or {})

    if source == "action_ticket":
        ticket = artifacts.get("action_ticket") if isinstance(artifacts.get("action_ticket"), dict) else {}
        plan_state.update({
            "task_ticketed": True,
            "working_context_ready": False,
            "content_generated": False,
            "content_applied": False,
            "result_verified": False,
            "compare_passed": False,
        })
        if ticket:
            proposed_action.update(ticket)
    elif source == "creative_ticket":
        ticket = artifacts.get("job_ticket") if isinstance(artifacts.get("job_ticket"), dict) else {}
        plan_state.update({
            "task_ticketed": True,
            "working_context_ready": True,
            "content_generated": True,
            "content_applied": False,
            "result_verified": False,
            "compare_passed": False,
        })
        if ticket:
            proposed_action.update(ticket)
    elif source == "ingress_router":
        route_ticket = artifacts.get("route_ticket") if isinstance(artifacts.get("route_ticket"), dict) else {}
        execution_result = artifacts.get("execution_result") if isinstance(artifacts.get("execution_result"), dict) else {}
        execution_plan = artifacts.get("execution_plan") if isinstance(artifacts.get("execution_plan"), dict) else {}
        if route_ticket:
            proposed_action.update(route_ticket)
        if execution_plan:
            proposed_action.setdefault("execution_plan", execution_plan)
        if execution_result:
            plan_state["execution_result"] = execution_result
            details = execution_result.get("details") if isinstance(execution_result.get("details"), dict) else {}
            if isinstance(details.get("compass"), dict):
                plan_state["compass"] = dict(details.get("compass") or {})
            for key, value in details.items():
                if key not in plan_state and isinstance(value, (bool, str, int, float, dict, list)):
                    plan_state[key] = value
            if execution_result.get("executed") is True:
                plan_state.setdefault("content_applied", True)
                plan_state.setdefault("working_context_ready", True)
            elif execution_result.get("attempted") is True:
                plan_state.setdefault("working_context_ready", True)
        if reply_txt:
            plan_state.setdefault("reply", reply_txt)
    else:
        if reply_txt:
            plan_state.setdefault("candidate_text", reply_txt)

    return plan_state, proposed_action


def _sm_reply_gate_from_neuron(user_text: str, res_dict: Dict[str, Any], meta: Dict[str, Any]) -> Dict[str, Any]:
    trace = res_dict.get("trace") if isinstance(res_dict.get("trace"), dict) else {}
    artifacts = res_dict.get("artifacts") if isinstance(res_dict.get("artifacts"), dict) else {}
    source = str(res_dict.get("source") or "").strip().lower()

    direct_compass = trace.get("compass") if isinstance(trace.get("compass"), dict) else {}
    if direct_compass:
        return {"active": True, "packet": dict(direct_compass), "origin": "trace"}

    if not _sm_is_operational_neuron_result(source, artifacts, trace):
        return {"active": False, "packet": {}, "origin": None}

    try:
        from SarahMemoryCognitiveCompass import get_compass_packet  # type: ignore
    except Exception:
        return {"active": False, "packet": {}, "origin": None}

    caller_context = dict(meta or {})
    caller_context.setdefault("intent", res_dict.get("intent"))
    caller_context.setdefault("primary_lane", trace.get("primary_lane") or res_dict.get("intent") or source)
    caller_context.setdefault("source", source or "neuron")
    caller_context.setdefault("consult_thinker", False)
    if isinstance(trace.get("governance"), dict):
        caller_context["governance"] = dict(trace.get("governance") or {})

    plan_state, proposed_action = _sm_build_compass_plan_state_from_neuron(res_dict)
    try:
        packet = get_compass_packet(
            user_text,
            caller_context=caller_context,
            plan_state=plan_state,
            proposed_action=proposed_action,
            force_refresh=False,
        )
        if isinstance(packet, dict) and packet:
            return {"active": True, "packet": packet, "origin": "computed"}
    except Exception as e:
        try:
            logger.debug("[reply] compass gate failed: %s", e)
        except Exception:
            pass
    return {"active": False, "packet": {}, "origin": None}


def _sm_reply_text_for_compass_hold(user_text: str, res_dict: Dict[str, Any], compass_packet: Dict[str, Any]) -> str:
    packet = dict(compass_packet or {})
    status = str(packet.get("status") or "").strip().upper()
    next_step = str(packet.get("next_required_step") or "").strip().upper()
    reason = str(packet.get("reason") or "").strip()
    source = str(res_dict.get("source") or "").strip().lower()
    artifacts = res_dict.get("artifacts") if isinstance(res_dict.get("artifacts"), dict) else {}

    if status in {"SAFE_ABORT", "RED_ZONE"}:
        return reason or "The request was stopped by governance before a safe verified result was available."

    if source == "action_ticket":
        ticket = artifacts.get("action_ticket") if isinstance(artifacts.get("action_ticket"), dict) else {}
        action_name = str(ticket.get("action") or "requested action").replace("_", " ").strip()
        base = f"The action request has been staged for governed execution: {action_name}."
    elif source == "creative_ticket":
        ticket = artifacts.get("job_ticket") if isinstance(artifacts.get("job_ticket"), dict) else {}
        kind = str(ticket.get("kind") or "creative").replace("_", " ").strip()
        base = f"The {kind} job has been prepared, but it is not verified as complete yet."
    elif source == "ingress_router":
        base = "The request has been handed into the governed execution path, but final completion is not verified yet."
    else:
        base = "The request is still in progress and has not reached a verified completion state yet."

    if status == "COMPARE_PENDING":
        return reason or (base + " Reply is being held until verification and Compare release succeed.")
    if status == "WAITING_ON_SIDEKICK":
        return reason or (base + " The execution path is still waiting on the responsible module or executor.")
    if status in {"PROCEDURAL_GAP", "UNVERIFIED_COMPLETION", "MINOR_DRIFT", "NO_PROGRESS", "LOOP_RISK", "ESCAPE_HATCH_TRIGGERED"}:
        return reason or (base + " The next required step is " + (next_step.replace("_", " ").lower() if next_step else "pending") + ".")
    return reason or base

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
    if _sm_is_volatile_body_fact_query(text):
        return None
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

def _sm_sanitize_user_text(text: str, for_tts: bool = False, user_text: Optional[str] = None) -> str:
    """
    Sanitize model output so the UI/TTS only sees the final user-facing answer.
    Removes prompt scaffolding, hidden blocks, markdown noise, provenance labels,
    echoed prompts, mojibake, control bytes, and decorative symbols.
    """
    if not text:
        return ""

    t = html.unescape(str(text)).replace("\r\n", "\n").replace("\r", "\n").strip()

    # Remove control bytes early while preserving tab/newline.
    try:
        t = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", t)
        t = t.replace("\ufffd", " ")
    except Exception:
        pass

    # Strip hidden reasoning blocks (common tags)
    try:
        t = re.sub(r"(?is)<\s*(think|analysis)\s*>.*?<\s*/\s*\1\s*>", "", t)
        t = re.sub(r"(?is)\[\s*(think|analysis)\s*\].*?\[\s*/\s*\1\s*\]", "", t)
    except Exception:
        pass

    # If explicit transcript markers exist, keep only the visible assistant content.
    try:
        t = re.sub(r"(?im)^\s*(system|developer|tool)\s*:\s*.*$", "", t)
        matches = list(re.finditer(r"(?im)^assistant\s*:\s*$", t))
        if matches:
            t = t[matches[-1].end():].strip()
        elif re.search(r"(?im)^assistant\s*:\s*", t):
            parts = re.split(r"(?im)^assistant\s*:\s*", t)
            if parts:
                t = parts[-1].strip()
        t = re.sub(r"(?im)^\s*(user|human|prompt|question)\s*:\s*", "", t)
    except Exception:
        pass

    # Remove common prompt header lines that local models sometimes echo.
    try:
        drop_pat = re.compile(
            r"(?i)^\s*(role|intent|tone|complexity|mood(?:\s*profile)?|context|query|parameters?|"
            r"temp(?:erature)?|top[_\s]?p|max[_\s]?tokens?|model(?:_used)?|provider|source)\s*[:=].*$"
        )
        cleaned = []
        for line in t.split("\n"):
            if drop_pat.match(line or ""):
                continue
            if re.match(r"(?i)^\s*\[\s*source\s*:[^\]]*\]\s*$", line or ""):
                continue
            if re.match(r"(?i)^\s*\(\s*intent\s*:[^\)]*\)\s*$", line or ""):
                continue
            if (line or "").strip() in ("[]", "[ ]"):
                continue
            cleaned.append(line)
        t = "\n".join(cleaned).strip()
    except Exception:
        pass

    # Remove legacy provenance/footer markers that should stay in meta only.
    try:
        t = re.sub(r"(?im)^\s*\[\s*source\s*:[^\]]*\]\s*$", "", t)
        t = re.sub(r"(?im)^\s*\(\s*intent\s*:[^\)]*\)\s*$", "", t)
        t = re.sub(r"\s*\[\s*source\s*:[^\]]*\]\s*", " ", t, flags=re.I)
        t = re.sub(r"\s*\(\s*intent\s*:[^\)]*\)\s*", " ", t, flags=re.I)
        t = re.sub(r"\s*\[\s*\]\s*$", "", t).strip()
    except Exception:
        pass

    # Collapse markdown/code formatting into plain readable text.
    try:
        t = re.sub(r"(?is)```.*?```", lambda m: re.sub(r"^```[a-zA-Z0-9_-]*\n?|\n?```$", "", m.group(0).strip()), t)
        t = re.sub(r"`([^`]+)`", r"\1", t)
        t = re.sub(r"!\[([^\]]*)\]\([^\)]+\)", r"\1", t)
        t = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", t)
        t = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", t)
        t = re.sub(r"(?m)^\s*>+\s*", "", t)
        t = re.sub(r"(?m)^\s*[-*•◦▪▫]+\s+", "", t)
        t = re.sub(r"(?m)^\s*\d+[\.)]\s+", "", t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
        t = re.sub(r"__([^_]+)__", r"\1", t)
        t = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", t)
        t = re.sub(r"(?<!_)_([^_]+)_(?!_)", r"\1", t)
        t = re.sub(r"~~([^~]+)~~", r"\1", t)
    except Exception:
        pass

    # Remove echoed prompt if the first visible line/paragraph is the user's question.
    try:
        if user_text:
            def _norm_cmp(s: str) -> str:
                s = html.unescape(str(s or ""))
                s = re.sub(r"(?i)^\s*(user|human|prompt|question)\s*:\s*", "", s)
                s = s.strip().strip(" \"'`*_[](){}<>")
                s = re.sub(r"\s+", " ", s)
                s = re.sub(r"[\s\-–—:;,.!?]+$", "", s)
                return s.lower()

            prompt_norm = _norm_cmp(user_text)
            if prompt_norm:
                lines = t.split("\n")
                while lines:
                    first = lines[0].strip()
                    if not first:
                        lines.pop(0)
                        continue
                    first_norm = _norm_cmp(first)
                    first_para = first_norm
                    if len(lines) > 1 and lines[1].strip() == "":
                        para_lines = []
                        for line in lines:
                            if not line.strip() and para_lines:
                                break
                            if line.strip():
                                para_lines.append(line.strip())
                        if para_lines:
                            first_para = _norm_cmp(" ".join(para_lines))
                    if first_norm == prompt_norm or first_para == prompt_norm:
                        lines.pop(0)
                        while lines and not lines[0].strip():
                            lines.pop(0)
                        continue
                    break
                t = "\n".join(lines).strip()
    except Exception:
        pass

    # Strip decorative symbols and emoji-like characters that cause literal speech.
    try:
        t = t.replace("™", " ").replace("®", " ").replace("©", " ")
        t = t.replace("•", "\n").replace("▪", "\n").replace("▫", "\n").replace("◦", "\n")
        t = re.sub(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]", " ", t)
        t = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", t)
    except Exception:
        pass

    # Remove common mojibake fragments and malformed tail noise.
    try:
        t = re.sub(r"(?:Ã.|Â.|ð.|Ð.){2,}", " ", t)
        t = re.sub(r"\b(?:Ã[^\s]{0,7}|Â[^\s]{0,7}|ð[^\s]{0,7}|Ð[^\s]{0,7})\b", " ", t)
        t = re.sub(r"[ÂÃÐð]+(?=\s|$)", " ", t)
        t = re.sub(r"(?:\s*[ÂÃÐð�][^\w\n]{0,4})+\s*$", "", t)
        t = re.sub(r"(?:\s*[^\w\s\.,;:!?\)\]\}\"'/%-]){2,}\s*$", "", t)
    except Exception:
        pass

    if for_tts:
        try:
            t = t.replace("&", " and ")
            t = re.sub(r"[*/_~`#|^<>\[\]{}\\]+", " ", t)
            t = re.sub(r"(?m)^\s*[-–—]+\s*", "", t)
            t = t.replace("\n", ". ")
        except Exception:
            pass
    else:
        try:
            t = re.sub(r"[ \t]{2,}", " ", t)
            t = re.sub(r" ?\n ?", "\n", t)
        except Exception:
            pass

    # Final whitespace / punctuation normalization.
    try:
        t = re.sub(r"\n{3,}", "\n\n", t)
        t = re.sub(r"[ \t]{2,}", " ", t)
        t = re.sub(r"\s+([,.;:!?])", r"\1", t)
        t = re.sub(r"([,.;:!?]){2,}", lambda m: m.group(0)[0], t)
        t = re.sub(r"[ \t]+\n", "\n", t)
        t = re.sub(r"\n[ \t]+", "\n", t)
        t = re.sub(r"(?:\s*[,;:]+\s*)+$", "", t)
        t = t.strip(" \t\n-–—*_`#|:;,")
    except Exception:
        pass

    return t.strip()


def _finalize_text(raw, meta):
    meta = meta or {}
    txt = _sm_clean_reply_text(_sm_sanitize_user_text((raw or "").strip(), user_text=meta.get("_prompt_text")))
    if not txt:
        txt = _sm_render_vision_reply(meta, None)
    if not txt:
        return ""

    if meta.get("intent") in {"identity", "math", "image", "web", "local", "interrupt", "system_status", "followup_no"}:
        return txt

    styled = txt
    try:
        try:
            styled = integrate_with_personality(txt, meta=meta)
        except TypeError:
            styled = integrate_with_personality(txt)
    except Exception:
        styled = txt

    styled = _sm_clean_reply_text(_sm_sanitize_user_text(styled, user_text=meta.get("_prompt_text")))
    return styled or txt


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
    original_text = normalize_text(user_text) if 'normalize_text' in globals() else (user_text or "").strip()
    text_in = original_text
    meta: Dict[str, Any] = {
        "intent": "chat",
        "pipeline": [],
        "offline": False,
        "model": None,
        "latency_ms": 0,
        "_prompt_text": original_text,
    }
    followup_bundle, text_in = _sm_resolve_followup_input(text_in, meta)
    if isinstance(followup_bundle, dict):
        return _stamp_bundle(followup_bundle)
    if text_in != original_text:
        meta["effective_prompt_text"] = text_in
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
        m = _eval_safe_math(text_in) or _try_websym_math(text_in)
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
            nb_meta = nb.get("meta") if isinstance(nb.get("meta"), dict) else {}
            reply_gate = nb_meta.get("reply_gate") if isinstance(nb_meta.get("reply_gate"), dict) else {}
            if bool(reply_gate.get("active")) and not bool(reply_gate.get("reply_allowed", True)):
                out = _sm_clean_reply_text(_sm_sanitize_user_text(nb.get("response") or "", user_text=nb_meta.get("_prompt_text") or original_text))
            else:
                out = _finalize_text(nb.get("response") or "", nb_meta)
            nb["response"] = out
            nb["presentation_reply"] = out
            nb["reply"] = out
            nb["content"] = out
            try:
                latency_ms = int((time.time() - started) * 1000)
                nb_meta["latency_ms"] = latency_ms
                nb["meta"] = nb_meta
            except Exception:
                pass
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
        safe_meta = safe_bundle.get("meta") if isinstance(safe_bundle.get("meta"), dict) else {}
        if _sm_meta_blocks_persistence(safe_meta, text=str(safe_meta.get("_prompt_text") or safe_bundle.get("user_input") or safe_bundle.get("query") or "")):
            logger.info("[V10/V9C] Skipped response history store for volatile SelfAware body fact.")
            return
        safe_bundle["response"] = _sm_sanitize_user_text(
            safe_bundle.get("presentation_reply") or safe_bundle.get("response") or "",
            user_text=safe_meta.get("_prompt_text"),
        )
        safe_bundle["presentation_reply"] = safe_bundle.get("response") or ""
        store_response_history(safe_bundle)
    except Exception as e:
        logger.debug("History store skipped: %s", e)

def _trigger_av_voice_safe(self_ref, text: str) -> None:
    if not text:
        return

    spoken_text = _sm_sanitize_user_text(text, for_tts=True)
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


# Back-compat helpers retained but neutered: provenance stays in meta only.
def render_provenance_footer(source_label, intent_label):
    return ""


def _append_source_intent_to_reply(text, provenance=None, intent=None):
    return _sm_sanitize_user_text(text or "")


# --- injected: on-demand ensure table for `response` ---
def _ensure_response_table(db_path=None):
    try:
        import sqlite3
        import os
        import logging
        try:
            import SarahMemoryGlobals as config
        except Exception:
            class config:
                DATASETS_DIR = os.path.join(os.getcwd(), "data", "memory", "datasets")
        if db_path is None:
            db_path = os.path.join(config.DATASETS_DIR, "system_logs.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS response ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "ts TEXT, user TEXT, content TEXT, source TEXT, intent TEXT)"
        )
        con.commit()
        con.close()
        logging.debug("[DB] ensured table `response` in %s", db_path)
    except Exception as e:
        try:
            import logging
            logging.warning("[DB] ensure `response` failed: %s", e)
        except Exception:
            pass


try:
    _ensure_response_table()
except Exception:
    pass


def _sm_enforce_provenance(bundle):
    """
    Legacy hook. Keep provenance in meta ONLY; never append to visible response text.
    Also acts as a last-chance sanitizer for bundle['response'].
    """
    try:
        if not isinstance(bundle, dict):
            return bundle
        meta = bundle.get("meta") if isinstance(bundle.get("meta"), dict) else {}
        meta["source"] = meta.get("source") or meta.get("Source") or meta.get("provider") or bundle.get("source") or "local"
        bundle["meta"] = meta
        if "response" in bundle and isinstance(bundle.get("response"), str):
            bundle["response"] = _sm_sanitize_user_text(
                bundle.get("response") or "",
                user_text=meta.get("_prompt_text"),
            )
    except Exception:
        pass
    return bundle


# v7.7.4 fallback: ensure emotion logging doesn't fail if table name differs
def _log_emotion_safe(emotion: str, intensity: float = 0.5):
    try:
        import os
        import sqlite3
        import time
        db = os.path.join(os.getcwd(), "data", "memory", "datasets", "system_logs.db")
        os.makedirs(os.path.dirname(db), exist_ok=True)
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute(
            "INSERT INTO emotion_states(ts, emotion, intensity) VALUES (?,?,?)",
            (time.strftime("%Y-%m-%dT%H:%M:%S"), emotion, float(intensity)),
        )
        con.commit()
        con.close()
    except Exception:
        try:
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.execute(
                "INSERT INTO traits(ts, trait, value) VALUES (?,?,?)",
                (time.strftime("%Y-%m-%dT%H:%M:%S"), emotion, float(intensity)),
            )
            con.commit()
            con.close()
        except Exception:
            pass


# ====================================================================
# END OF SarahMemoryReply.py v8.0.0
# ====================================================================
