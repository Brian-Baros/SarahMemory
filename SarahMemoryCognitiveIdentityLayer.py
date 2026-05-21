"""--== SarahMemory Project ==--
File: SarahMemoryCognitiveIdentityLayer.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0

Tri-layer identity/context packet helpers for SarahMemory AiOS.

Purpose:
- Build governed, read-only input packets for the Artificial Living System stack.
- Layer 1: Six-question governance seed support.
- Layer 2: language/context identity ring support.
- Layer 3: emotion/affect ring support.

Doctrine:
- This module does not execute actions.
- This module does not authorize actions.
- This module does not persist private content by default.
- Helper models are evidence only; packet construction is deterministic-first.
"""
from __future__ import annotations

import os
import re
import time
import uuid
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None  # type: ignore

logger = logging.getLogger("SarahMemoryCognitiveIdentityLayer")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

MODULE_NAME = "SarahMemoryCognitiveIdentityLayer"
MODULE_VERSION = "8.0.0"

QUESTION_WORDS = ("who", "why", "what", "when", "where", "how")
DETERMINERS = {"a", "an", "the", "this", "that", "these", "those", "my", "your", "our", "his", "her", "their"}
PRONOUNS = {"i", "me", "you", "he", "she", "it", "we", "they", "them", "us", "him", "her", "myself", "yourself"}
PREPOSITIONS = {"in", "on", "at", "by", "with", "to", "from", "into", "onto", "over", "under", "between", "near", "inside", "outside", "about", "for", "of"}
CONJUNCTIONS = {"and", "or", "but", "because", "if", "then", "while", "although", "unless"}
PARTICLES = {"is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "not", "no"}

KNOWN_COMPOUNDS = {
    "final fantasy", "sarahmemory aios", "sarah memory", "sarahnet", "cognitive triforce",
    "motor servo device controller", "visual studio code", "new york", "los angeles",
    "star trek", "world of warcraft", "runes of magic", "hogwarts legacy",
}

ACTION_VERBS = {
    "open", "close", "turn", "start", "stop", "launch", "run", "create", "make", "delete", "move",
    "copy", "rename", "search", "find", "show", "tell", "ask", "scan", "read", "write", "build",
    "patch", "fix", "update", "repair", "analyze", "compare", "look", "watch", "learn",
}

POSITIVE_WORDS = {"good", "great", "excellent", "thanks", "thank", "love", "awesome", "happy", "glad", "perfect", "wonderful"}
NEGATIVE_WORDS = {"bad", "wrong", "broken", "hate", "angry", "mad", "sad", "scared", "afraid", "problem", "issue", "error", "stupid", "frustrated"}
URGENCY_WORDS = {"now", "urgent", "immediately", "emergency", "asap", "quick", "fast", "hurry", "stop", "abort", "danger", "critical"}
CONCERN_WORDS = {"careful", "concern", "worried", "unsafe", "risk", "danger", "harm", "injury", "unknown"}


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _safe_text(value: Any, limit: int = 2000) -> str:
    s = str(value or "")
    if len(s) > limit:
        return s[:limit].rstrip() + " …"
    return s


def _normalize(text: Any) -> str:
    raw = _safe_text(text, 20000).replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"[\t\f\v]+", " ", raw)
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?|[^\w\s]", text or "", flags=re.UNICODE)


def _word_tokens(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)?", text or "", flags=re.UNICODE)


def _dedupe(values: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for v in values:
        key = json.dumps(v, sort_keys=True, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


def _known_compound_phrases(lowered: str) -> List[str]:
    out = []
    for phrase in sorted(KNOWN_COMPOUNDS, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", lowered):
            out.append(" ".join(w.capitalize() if w else w for w in phrase.split()))
    return out


def _quoted_phrases(raw: str) -> List[str]:
    return [m.group(1).strip() for m in re.finditer(r"[\"“”']([^\"“”']{2,120})[\"“”']", raw or "") if m.group(1).strip()]


def _titlecase_phrases(raw: str) -> List[str]:
    # Conservative: use proper-noun runs after anchors like from/in/about/called/named.
    out: List[str] = []
    anchor_pat = r"(?:from|in|about|called|named|titled|for|of)\s+(([A-Z][A-Za-z0-9]+)(?:\s+[A-Z][A-Za-z0-9]+){1,5})"
    for m in re.finditer(anchor_pat, raw or ""):
        phrase = m.group(1).strip()
        if phrase and phrase.lower() not in {"who is", "what is", "where is", "when did"}:
            out.append(phrase)
    return out


def _proper_nouns(raw: str, phrase_locks: List[str]) -> List[str]:
    out = []
    for phrase in phrase_locks:
        out.append(phrase)
    # Single proper noun after "who is" etc. Example: "Who is Cloud from Final Fantasy"
    m = re.search(r"\b(?:who|what)\s+(?:is|are|was|were)\s+([A-Z][A-Za-z0-9_'-]{1,80})\b", raw or "", flags=re.I)
    if m:
        out.append(m.group(1))
    return _dedupe(out)


def _basic_pos_tags(words: List[str]) -> Dict[str, List[str]]:
    nouns: List[str] = []
    verbs: List[str] = []
    pronouns: List[str] = []
    adjectives: List[str] = []
    adverbs: List[str] = []
    prepositions: List[str] = []
    conjunctions: List[str] = []
    determiners: List[str] = []
    particles: List[str] = []

    for w in words:
        lw = w.lower()
        if lw in PRONOUNS:
            pronouns.append(w)
        elif lw in PREPOSITIONS:
            prepositions.append(w)
        elif lw in CONJUNCTIONS:
            conjunctions.append(w)
        elif lw in DETERMINERS:
            determiners.append(w)
        elif lw in PARTICLES:
            particles.append(w)
        elif lw in ACTION_VERBS or lw.endswith("ing") or lw.endswith("ed"):
            verbs.append(w)
        elif lw.endswith("ly"):
            adverbs.append(w)
        elif lw.endswith(("ous", "ful", "ive", "al", "ic", "able", "ible")):
            adjectives.append(w)
        else:
            nouns.append(w)
    return {
        "nouns": _dedupe(nouns),
        "verbs": _dedupe(verbs),
        "pronouns": _dedupe(pronouns),
        "adjectives": _dedupe(adjectives),
        "adverbs": _dedupe(adverbs),
        "prepositions": _dedupe(prepositions),
        "conjunctions": _dedupe(conjunctions),
        "determiners": _dedupe(determiners),
        "particles": _dedupe(particles),
    }


def _subject_object(raw: str, words: List[str], phrase_locks: List[str]) -> Tuple[str, str]:
    text = raw or ""
    m = re.search(r"\b(?:who|what)\s+(?:is|are|was|were)\s+([^?.,;]+)", text, flags=re.I)
    if m:
        span = m.group(1).strip()
        # Split on prepositions while preserving compound objects.
        m2 = re.match(r"(.+?)\s+(?:from|in|of|about|for)\s+(.+)$", span, flags=re.I)
        if m2:
            return m2.group(1).strip(), m2.group(2).strip()
        return span.strip(), ""
    for q in QUESTION_WORDS:
        if words and words[0].lower() == q and len(words) > 1:
            return words[1], ""
    # Imperative: first verb and rest as object-ish topic.
    for i, w in enumerate(words):
        if w.lower() in ACTION_VERBS:
            return words[i + 1] if i + 1 < len(words) else "", " ".join(words[i + 2:i + 8])
    return (words[0] if words else ""), ""


def _context_domain(raw: str, phrase_locks: List[str]) -> str:
    lower = (raw or "").lower()
    locked_lower = " | ".join(p.lower() for p in phrase_locks)
    if "final fantasy" in locked_lower or any(k in lower for k in ("video game", "game character", "rpg", "cloud strife")):
        return "gaming_lore"
    if any(k in lower for k in ("webcam", "camera", "frame", "snapshot", "see", "vision")):
        return "vision"
    if any(k in lower for k in ("cpu", "gpu", "motherboard", "drive", "sata", "usb", "fan", "temperature")):
        return "selfaware_body"
    if any(k in lower for k in ("file", "folder", "path", "zip", "patch", "code")):
        return "software_engineering"
    if any(k in lower for k in ("emotion", "feel", "tone", "angry", "happy", "sad")):
        return "emotion_affect"
    return "general"


def _purpose_hint(raw: str, words: List[str]) -> str:
    lower = (raw or "").lower().strip()
    if lower.startswith(("who", "what", "when", "where", "why", "how")):
        return "answer_question"
    if any(w.lower() in ACTION_VERBS for w in words):
        return "request_action_or_operation"
    if "because" in lower or "so that" in lower:
        return "explain_reason_or_goal"
    return "conversation_or_context"


def _substring_blocks(raw: str, phrase_locks: List[str]) -> List[Dict[str, Any]]:
    lower_locks = [p.lower() for p in phrase_locks]
    candidates = ["fan", "ram", "ai", "usb", "cpu", "gpu", "ui", "os"]
    blocks: List[Dict[str, Any]] = []
    for lock in lower_locks:
        compact = re.sub(r"\s+", "", lock)
        for cand in candidates:
            if cand in compact and cand != compact and cand not in lock.split():
                blocks.append({
                    "candidate": cand,
                    "inside_phrase": lock,
                    "reason": "candidate is substring inside locked phrase, not standalone token",
                })
    return _dedupe(blocks)


def candidate_blocked_by_language_packet(candidate: str, language_packet: Dict[str, Any]) -> Dict[str, Any]:
    cand = str(candidate or "").strip().lower()
    if not cand:
        return {"blocked": False, "candidate": cand}
    for item in language_packet.get("blocked_substring_matches") or []:
        if isinstance(item, dict) and str(item.get("candidate") or "").lower() == cand:
            return {"blocked": True, "candidate": cand, "reason": item.get("reason"), "inside_phrase": item.get("inside_phrase")}
    return {"blocked": False, "candidate": cand}


def build_language_context_packet(raw_text: str, context_packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    raw = _safe_text(raw_text, 20000)
    normalized = _normalize(raw)
    words = _word_tokens(normalized)
    lower = normalized.lower()
    quoted = _quoted_phrases(raw)
    titlecase = _titlecase_phrases(raw)
    known = _known_compound_phrases(lower)
    phrase_locks = _dedupe([p for p in quoted + titlecase + known if p and len(p) > 1])
    pos = _basic_pos_tags(words)
    subject, obj = _subject_object(raw, words, phrase_locks)
    context_domain = _context_domain(raw, phrase_locks)
    purpose_hint = _purpose_hint(raw, words)
    blocked = _substring_blocks(raw, phrase_locks)

    ambiguity_score = 0.0
    if not words:
        ambiguity_score += 0.4
    if len(words) <= 2 and not phrase_locks:
        ambiguity_score += 0.2
    if any(w.lower() in {"it", "this", "that", "they", "them"} for w in words) and not context_packet:
        ambiguity_score += 0.2
    if blocked:
        ambiguity_score = max(0.0, ambiguity_score - 0.1)  # phrase locking improves safety
    ambiguity_score = max(0.0, min(1.0, ambiguity_score))

    return {
        "packet_type": "LanguageContextPacket",
        "schema": "SarahMemory.language_context.v1",
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "packet_id": "lang-" + uuid.uuid4().hex[:12],
        "ts": _now_iso(),
        "raw_text": raw,
        "normalized_text": normalized,
        "tokens": _tokens(normalized),
        "words": words,
        "phrase_locks": phrase_locks,
        "proper_nouns": _proper_nouns(raw, phrase_locks),
        "compound_phrases": phrase_locks,
        "subject": subject,
        "object": obj,
        "parts_of_speech": pos,
        "nouns": pos.get("nouns", []),
        "verbs": pos.get("verbs", []),
        "pronouns": pos.get("pronouns", []),
        "adjectives": pos.get("adjectives", []),
        "adverbs": pos.get("adverbs", []),
        "prepositions": pos.get("prepositions", []),
        "conjunctions": pos.get("conjunctions", []),
        "determiners": pos.get("determiners", []),
        "particles": pos.get("particles", []),
        "context_domain": context_domain,
        "purpose_hint": purpose_hint,
        "ambiguity_score": ambiguity_score,
        "blocked_substring_matches": blocked,
        "requires_clarification": ambiguity_score >= 0.65,
        "doctrine": {
            "no_substring_routing_inside_locked_phrases": True,
            "language_context_before_governance": True,
            "packet_is_evidence_not_authority": True,
        },
        "context_meta": {
            "has_context_packet": isinstance(context_packet, dict),
            "source": (context_packet or {}).get("source") if isinstance(context_packet, dict) else None,
        },
    }


def _lexical_emotion_estimate(text: str) -> Dict[str, float]:
    words = [w.lower() for w in _word_tokens(text)]
    total = max(1, len(words))
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    urg = sum(1 for w in words if w in URGENCY_WORDS)
    con = sum(1 for w in words if w in CONCERN_WORDS)
    exclaim = str(text or "").count("!")
    caps = sum(1 for w in re.findall(r"\b[A-Z]{3,}\b", text or ""))
    sentiment = max(-1.0, min(1.0, (pos - neg) / max(1, pos + neg))) if (pos or neg) else 0.0
    intensity = min(1.0, (neg + urg + exclaim + caps) / 6.0)
    return {
        "sentiment_score": sentiment,
        "urgency": min(1.0, (urg / total) * 4.0 + min(0.4, exclaim * 0.1)),
        "stress": min(1.0, (neg + con + caps) / 6.0),
        "intensity": intensity,
        "concern": min(1.0, (con / total) * 4.0),
    }


def build_emotion_affect_packet(text: str, context_packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    raw = _safe_text(text, 20000)
    lexical = _lexical_emotion_estimate(raw)
    adaptive: Dict[str, Any] = {}
    emotion_state: Dict[str, Any] = {}
    try:
        import SarahMemoryAdaptive as _Adaptive  # type: ignore
        fn = getattr(_Adaptive, "advanced_emotional_learning", None)
        if callable(fn):
            adaptive = fn(raw) or {}
        load = getattr(_Adaptive, "load_emotional_state", None)
        if callable(load):
            emotion_state = load() or {}
    except Exception as e:
        adaptive = {"error": str(e)}

    candidates: Dict[str, float] = {}
    for key in ("joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"):
        try:
            candidates[key] = float(emotion_state.get(key, adaptive.get(key, 0.0)) or 0.0)
        except Exception:
            candidates[key] = 0.0

    # Add deterministic text-side signal without allowing it to overrule Adaptive.
    if lexical["sentiment_score"] > 0.25:
        candidates["joy"] = max(candidates.get("joy", 0.0), min(1.0, 0.45 + lexical["sentiment_score"] * 0.35))
        candidates["trust"] = max(candidates.get("trust", 0.0), 0.45)
    elif lexical["sentiment_score"] < -0.25:
        candidates["anger"] = max(candidates.get("anger", 0.0), min(1.0, 0.35 + abs(lexical["sentiment_score"]) * 0.30))
        candidates["sadness"] = max(candidates.get("sadness", 0.0), 0.30)
    if lexical["urgency"] > 0.35:
        candidates["anticipation"] = max(candidates.get("anticipation", 0.0), lexical["urgency"])
    if lexical["concern"] > 0.25:
        candidates["fear"] = max(candidates.get("fear", 0.0), lexical["concern"])

    ranked = sorted(candidates.items(), key=lambda kv: kv[1], reverse=True)
    primary = ranked[0][0] if ranked and ranked[0][1] > 0.05 else "neutral"
    secondary = [k for k, v in ranked[1:4] if v > 0.20]
    emotional_balance = float(adaptive.get("emotional_balance", lexical["sentiment_score"]) or 0.0) if isinstance(adaptive, dict) else lexical["sentiment_score"]
    intensity = max(float(lexical["intensity"]), abs(emotional_balance), max(candidates.values() or [0.0]))

    tone = "neutral_direct"
    if lexical["urgency"] >= 0.65:
        tone = "calm_urgent"
    elif primary in {"anger", "fear", "sadness"}:
        tone = "calm_supportive"
    elif primary in {"joy", "trust"}:
        tone = "clear_positive"

    return {
        "packet_type": "EmotionAffectPacket",
        "schema": "SarahMemory.emotion_affect.v1",
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "packet_id": "emo-" + uuid.uuid4().hex[:12],
        "ts": _now_iso(),
        "primary_emotion": primary,
        "secondary_emotions": secondary,
        "emotion_scores": candidates,
        "sentiment_score": max(-1.0, min(1.0, emotional_balance)),
        "emotional_intensity": max(0.0, min(1.0, intensity)),
        "urgency": max(0.0, min(1.0, lexical["urgency"])),
        "stress": max(0.0, min(1.0, lexical["stress"])),
        "confidence": 0.64 if adaptive else 0.42,
        "sarcasm_likelihood": 0.15 if re.search(r"\b(yeah right|sure buddy|obviously)\b", raw, re.I) else 0.0,
        "engagement": float(adaptive.get("engagement", 0.5) or 0.5) if isinstance(adaptive, dict) else 0.5,
        "openness": float(adaptive.get("openness", 0.5) or 0.5) if isinstance(adaptive, dict) else 0.5,
        "source": "text",
        "input_role": "meaning_signal",
        "output_constraints": {
            "tone": tone,
            "avoid_humor": primary in {"anger", "fear", "sadness"} or lexical["urgency"] > 0.5,
            "ask_clarifying_question": False,
            "emotion_does_not_authorize_action": True,
        },
        "doctrine": {
            "emotion_informs_meaning": True,
            "emotion_never_overrides_truth": True,
            "emotion_never_overrides_governance": True,
        },
        "adaptive_metrics": {k: v for k, v in adaptive.items() if k not in {"raw", "history"}} if isinstance(adaptive, dict) else {},
    }


def _entry_point_from_language(language: Dict[str, Any], context_packet: Optional[Dict[str, Any]] = None) -> str:
    ctx = context_packet if isinstance(context_packet, dict) else {}
    event_type = str(ctx.get("event_type") or ctx.get("source") or "").lower()
    if "webcam" in event_type or "camera" in event_type or "sensor" in event_type:
        return "WHAT"
    if "dream" in event_type or "rem" in event_type:
        return "WHY"
    if "schedule" in event_type or "timer" in event_type:
        return "WHEN"
    words = [str(w).lower() for w in (language.get("words") or [])]
    for q in QUESTION_WORDS:
        if words and words[0] == q:
            return q.upper()
    return "WHAT"


def build_six_question_seed_packet(
    text: str,
    *,
    language_context_packet: Optional[Dict[str, Any]] = None,
    emotion_affect_packet: Optional[Dict[str, Any]] = None,
    context_packet: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a non-authorizing seed packet for the six-question governance loop.

    This is not the final CognitiveServices verdict. It lets upstream routers,
    logs, Reply, and Neuron carry the WHO/WHY/WHAT/WHEN/WHERE/HOW frame before
    final SMGET authorization.
    """
    language = language_context_packet if isinstance(language_context_packet, dict) else build_language_context_packet(text, context_packet=context_packet)
    emotion = emotion_affect_packet if isinstance(emotion_affect_packet, dict) else build_emotion_affect_packet(text, context_packet=context_packet)
    ctx = context_packet if isinstance(context_packet, dict) else {}
    entry = _entry_point_from_language(language, ctx)
    questions = {
        "WHO": {
            "question": "Who is involved, asking, affected, or approving?",
            "seed": {
                "subject": language.get("subject"),
                "pronouns": language.get("pronouns") or [],
                "caller": ctx.get("caller") or ctx.get("source") or "unknown",
            },
        },
        "WHY": {
            "question": "Why is this needed or meaningful?",
            "seed": {
                "purpose_hint": language.get("purpose_hint"),
                "emotion": emotion.get("primary_emotion"),
                "urgency": emotion.get("urgency"),
            },
        },
        "WHAT": {
            "question": "What is being requested, observed, changed, or affected?",
            "seed": {
                "object": language.get("object"),
                "verbs": language.get("verbs") or [],
                "context_domain": language.get("context_domain"),
            },
        },
        "WHEN": {
            "question": "When is this allowed, relevant, remembered, or expired?",
            "seed": {
                "event_time": ctx.get("event_time") or ctx.get("ts"),
                "scheduled": ctx.get("scheduled"),
            },
        },
        "WHERE": {
            "question": "Where is the context, device, body part, file, surface, or impact?",
            "seed": {
                "source": ctx.get("source"),
                "surface": ctx.get("surface"),
                "device": ctx.get("device") or ctx.get("body_part"),
            },
        },
        "HOW": {
            "question": "How is this verified, executed, audited, failed safe, and rolled back?",
            "seed": {
                "method_hint": ctx.get("method") or ctx.get("execution_mode"),
                "requires_clarification": language.get("requires_clarification"),
                "blocked_substring_matches": language.get("blocked_substring_matches") or [],
            },
        },
    }
    return {
        "packet_type": "SixQuestionSeedPacket",
        "schema": "SarahMemory.six_question_seed.v1",
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "packet_id": "sixseed-" + uuid.uuid4().hex[:12],
        "ts": _now_iso(),
        "entry_point": entry,
        "questions": questions,
        "loop_closed": False,
        "execution_authority": False,
        "doctrine": {
            "any_point_can_start": True,
            "any_order": True,
            "all_six_questions_interconnect": True,
            "seed_packet_is_not_authorization": True,
        },
    }


def build_tri_layer_input_packet(text: str, context_packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    language = build_language_context_packet(text, context_packet=context_packet)
    emotion = build_emotion_affect_packet(text, context_packet=context_packet)
    six_seed = build_six_question_seed_packet(
        text,
        language_context_packet=language,
        emotion_affect_packet=emotion,
        context_packet=context_packet,
    )
    identity = {}
    try:
        import SarahMemoryCognitiveSelf as _CogSelf  # type: ignore
        fn = getattr(_CogSelf, "resolve_active_identity", None)
        if callable(fn):
            identity = fn(context_packet or {}) or {}
    except Exception:
        identity = {}
    return {
        "packet_type": "TriLayerInputPacket",
        "schema": "SarahMemory.tri_layer_input.v2",
        "module": MODULE_NAME,
        "module_version": MODULE_VERSION,
        "packet_id": "tri-" + uuid.uuid4().hex[:12],
        "ts": _now_iso(),
        "language_context_packet": language,
        "six_question_seed_packet": six_seed,
        "emotion_affect_packet": emotion,
        "identity_packet": identity,
        "loop_status": "PRE_GOVERNANCE",
        "execution_authority": False,
        "doctrine": {
            "input_can_start_from_any_source": True,
            "language_context_before_routing": True,
            "six_question_loop_before_action": True,
            "emotion_as_input_and_output_signal": True,
            "governance_required_before_action": True,
        },
    }
