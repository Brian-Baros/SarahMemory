# --==The SarahMemory Project==--
# File: SarahMemoryPreTokenAnalyzer.py
# Purpose: Governed pre-token semantic analysis and ambiguity control for SarahMemory AiOS
# Part of the SarahMemory Companion AI-bot Platform
# Author: Draft generated for Brian Lee Baros / SarahMemory Project
# Version: 8.0.0-draft
# Notes:
# - This module is intentionally local-first and deterministic.
# - It does NOT execute actions and it does NOT answer the user directly.
# - It prepares text for SarahMemoryAdvCU / SarahMemoryNeuron by:
#     * normalizing raw text
#     * detecting abbreviations and opaque text
#     * scoring ambiguity / compression safety
#     * building compact symbolic packet drafts
#     * generating targeted clarification questions
#     * merging clarification answers into the same conceptual query frame
#
# Enterprise design goals:
# - Safe by default
# - Readable / auditable output
# - No third-party dependencies required
# - Graceful degradation when context is limited

from __future__ import annotations

import copy
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PROJECT_VERSION = "8.0.0-draft"
MODULE_NAME = "SarahMemoryPreTokenAnalyzer"


SURFACE_APP_ALIASES = {
    "word": "winword",
    "microsoft word": "winword",
    "ms word": "winword",
    "winword": "winword",
    "excel": "excel",
    "microsoft excel": "excel",
    "ms excel": "excel",
    "powerpoint": "powerpnt",
    "microsoft powerpoint": "powerpnt",
    "ms powerpoint": "powerpnt",
    "powerpnt": "powerpnt",
    "paint": "mspaint",
    "microsoft paint": "mspaint",
    "mspaint": "mspaint",
    "notepad": "notepad",
    "visual studio code": "code",
    "vs code": "code",
    "vscode": "code",
    "code": "code",
    "dreamweaver": "dreamweaver",
    "edge": "msedge",
    "microsoft edge": "msedge",
    "msedge": "msedge",
    "chrome": "chrome",
    "google chrome": "chrome",
    "firefox": "firefox",
    "brave": "brave",
    "opera": "opera",
}

_SURFACE_VERBS = {"open", "create", "write", "make", "draw", "search", "find", "look", "launch", "start"}
_VALID_SURFACE_APPS = set(SURFACE_APP_ALIASES.values())


def _canonical_surface_app(name: str) -> str:
    raw = _normalize_text(_coerce_text(name)).lower().replace('.exe', '').strip()
    return SURFACE_APP_ALIASES.get(raw, raw)


def _extract_surface_task_contract(text: str, preferred_app: Optional[str] = None) -> Dict[str, Any]:
    return extract_surface_task(text=text, preferred_app=preferred_app)


def _surface_topic_from_text(raw: str) -> str:
    patterns = [
        r"\b(?:about|on|for)\s+(.+)$",
        r"\bsummary\s+(?:about|of|on)?\s*(.+)$",
        r"\bsearch\s+(?:for\s+)?(.+)$",
        r"\bdraw\s+(?:me\s+)?(?:a|an)?\s*(?:picture|image)?\s*(?:of\s+)?(.+)$",
    ]
    for pat in patterns:
        m = re.search(pat, raw, re.I)
        if m:
            val = _normalize_text(m.group(1).strip(' ?.,;:'))
            if val:
                return val
    return ""


def extract_surface_task(text: str, preferred_app: Optional[str] = None) -> Dict[str, Any]:
    """Deterministic surface-task extraction for desktop/browser workflows."""
    raw = _normalize_text(_coerce_text(text))
    if not raw:
        return {}
    lowered = raw.lower()

    preferred = _canonical_surface_app(preferred_app or "") if preferred_app else ""
    if preferred in _SURFACE_VERBS or preferred not in _VALID_SURFACE_APPS:
        preferred = ""

    alias_candidates = sorted(SURFACE_APP_ALIASES.keys(), key=len, reverse=True)

    def _find_requested_app() -> str:
        if preferred:
            return preferred
        for alias in alias_candidates:
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lowered):
                return _canonical_surface_app(alias)
        return ""

    requested_app = _find_requested_app()
    task: Dict[str, Any] = {}
    if requested_app:
        task["requested_app"] = requested_app
        task["requested_app_exec"] = requested_app

    browser_apps = {"msedge", "chrome", "firefox", "brave", "opera"}
    editor_apps = {"notepad", "code", "dreamweaver"}
    doc_apps = {"winword", *editor_apps}

    if requested_app in browser_apps:
        m = re.search(r"\bsearch\s+(?:for\s+)?(.+)$", raw, re.I)
        if m:
            query = _normalize_text(m.group(1).strip(' ?.,;:'))
            if query:
                task.update({"task_kind": "browser_search", "search_query": query, "topic": query})
                return task
        m = re.search(r"\b(?:go to|navigate to|open)\s+(https?://\S+|[a-z0-9.-]+\.[a-z]{2,}(?:/\S*)?)", lowered)
        if m:
            url = str(m.group(1) or "").strip()
            if url and not url.startswith('http'):
                url = 'https://' + url
            if url:
                task.update({"task_kind": "browser_open_url", "target_url": url, "topic": url})
                return task

    if requested_app in doc_apps:
        m = re.search(r"\bopen\s+(?:the\s+)?document\s+named\s+(.+)$", raw, re.I)
        if m:
            name = _normalize_text(m.group(1).strip(' .,'))
            if name:
                task.update({"task_kind": "open_named_document", "document_name": name, "title": name})
                return task

    if requested_app == 'excel' or 'spreadsheet' in lowered or 'checkbook' in lowered:
        if 'checkbook' in lowered or 'track my spending' in lowered or 'spending' in lowered:
            task.update({
                "task_kind": "spreadsheet_template",
                "template_kind": "checkbook",
                "title": "Checkbook Register",
                "headers": ["Date", "Description", "Category", "Debit", "Credit", "Balance"],
                "topic": "track spending",
            })
            if not requested_app:
                task["requested_app"] = 'excel'
                task["requested_app_exec"] = 'excel'
            return task

    if (requested_app in editor_apps or requested_app == 'winword' or 'website' in lowered or 'homepage' in lowered or 'about page' in lowered):
        if 'website' in lowered or 'homepage' in lowered or 'about page' in lowered:
            topic_match = re.search(r"\bmake\s+it\s+about\s+(.+)$", raw, re.I) or re.search(r"\babout\s+(.+)$", raw, re.I)
            topic = _normalize_text(topic_match.group(1).strip(' .,')) if topic_match else 'website topic'
            pages = ['index.html']
            if 'homepage' in lowered:
                pages = ['index.html']
            if 'about page' in lowered:
                pages = ['index.html', 'about.html']
            task.update({"task_kind": "website_scaffold", "topic": topic, "pages": pages, "title": topic.title()})
            if not requested_app:
                task["requested_app"] = 'notepad'
                task["requested_app_exec"] = 'notepad'
            return task

    if requested_app == 'mspaint' or ('paint' in lowered and 'draw' in lowered):
        shape_match = re.search(r"\bdraw\s+(?:me\s+)?(?:a|an)?\s*(circle|square|rectangle|triangle|line|oval|star)\b", lowered)
        if shape_match:
            subject = shape_match.group(1)
            task.update({"task_kind": "paint_draw", "draw_subject": subject, "shape": subject, "topic": subject})
            if not requested_app:
                task["requested_app"] = 'mspaint'
                task["requested_app_exec"] = 'mspaint'
            return task
        m = re.search(r"\bdraw\s+(?:me\s+)?(?:a|an)?\s*(?:picture|image)?\s*(?:of\s+)?(.+)$", raw, re.I)
        if m:
            subject = _normalize_text(m.group(1).strip(' .,'))
            if subject:
                task.update({"task_kind": "paint_draw", "draw_subject": subject, "topic": subject})
                if not requested_app:
                    task["requested_app"] = 'mspaint'
                    task["requested_app_exec"] = 'mspaint'
                return task

    if requested_app in doc_apps or any(k in lowered for k in ('document', 'report', 'summary', 'essay', 'letter')):
        topic = _surface_topic_from_text(raw)
        title = ''
        title_m = re.search(r"\btitled\s+(.+?)(?:\s+and\s+write|$)", raw, re.I)
        if title_m:
            title = _normalize_text(title_m.group(1).strip(' .,'))
        if requested_app or topic or 'write' in lowered or 'summary' in lowered or 'report' in lowered or 'document' in lowered:
            task.update({
                "task_kind": "document_write",
                "topic": topic or 'the requested topic',
                "title": title or ((topic or 'Document').title()),
            })
            if not requested_app:
                task["requested_app"] = 'winword'
                task["requested_app_exec"] = 'winword'
            return task

    return task

# ---------------------------------------------------------------------------
# Thresholds / weights
# ---------------------------------------------------------------------------

FULL_COMPRESS_THRESHOLD = 0.85
PARTIAL_COMPRESS_THRESHOLD = 0.65
CLARIFY_THRESHOLD = 0.40

MAX_TEXT_LEN = 12000
MAX_CLARIFICATION_CHOICES = 5

# ---------------------------------------------------------------------------
# Core vocab / heuristics
# ---------------------------------------------------------------------------

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "get",
    "give", "hello", "hey", "hi", "i", "in", "into", "is", "it", "me", "my",
    "of", "on", "or", "please", "right", "show", "tell", "than", "that", "the",
    "this", "to", "today", "turn", "up", "what", "with", "would", "write", "you",
}

ACTION_KEYWORDS = {
    "open", "close", "launch", "start", "stop", "run", "move", "copy", "delete",
    "remove", "rename", "create", "write", "save", "download", "upload", "set",
    "turn", "switch", "enable", "disable", "install", "uninstall", "erase",
}

QUESTION_KEYWORDS = {
    "what", "when", "where", "who", "why", "how", "temperature", "forecast",
    "status", "diagnostics", "is", "are", "does", "do", "can", "could",
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
}

US_STATE_ABBREVIATIONS = {
    "AL": ["Alabama"],
    "AK": ["Alaska"],
    "AZ": ["Arizona"],
    "AR": ["Arkansas"],
    "CA": ["California"],
    "CO": ["Colorado"],
    "CT": ["Connecticut"],
    "DE": ["Delaware"],
    "FL": ["Florida"],
    "GA": ["Georgia"],
    "HI": ["Hawaii"],
    "ID": ["Idaho"],
    "IL": ["Illinois"],
    "IN": ["Indiana"],
    "IA": ["Iowa"],
    "KS": ["Kansas"],
    "KY": ["Kentucky"],
    "LA": ["Louisiana", "Los Angeles, California"],
    "ME": ["Maine"],
    "MD": ["Maryland"],
    "MA": ["Massachusetts"],
    "MI": ["Michigan"],
    "MN": ["Minnesota"],
    "MS": ["Mississippi"],
    "MO": ["Missouri"],
    "MT": ["Montana"],
    "NE": ["Nebraska"],
    "NV": ["Nevada"],
    "NH": ["New Hampshire"],
    "NJ": ["New Jersey"],
    "NM": ["New Mexico"],
    "NY": ["New York"],
    "NC": ["North Carolina"],
    "ND": ["North Dakota"],
    "OH": ["Ohio"],
    "OK": ["Oklahoma"],
    "OR": ["Oregon"],
    "PA": ["Pennsylvania"],
    "RI": ["Rhode Island"],
    "SC": ["South Carolina"],
    "SD": ["South Dakota"],
    "TN": ["Tennessee"],
    "TX": ["Texas"],
    "UT": ["Utah"],
    "VT": ["Vermont"],
    "VA": ["Virginia"],
    "WA": ["Washington"],
    "WV": ["West Virginia"],
    "WI": ["Wisconsin"],
    "WY": ["Wyoming"],
    "DC": ["District of Columbia"],
}

DOMAIN_KEYWORD_MAP = {
    "WEATHER": WEATHER_KEYWORDS,
    "DOC": DOC_KEYWORDS,
    "SYS": SYSTEM_KEYWORDS,
    "DRV": DRIVER_KEYWORDS,
    "NET": NETWORK_KEYWORDS,
    "CRT": CREATIVE_KEYWORDS,
    "RSH": RESEARCH_KEYWORDS,
}

RISKY_ACTION_PATTERNS = (
    "delete",
    "erase",
    "format",
    "shutdown",
    "reboot",
    "install",
    "uninstall",
    "move",
    "copy",
    "rename",
    "driver",
)

OPAQUE_CHAIN_RE = re.compile(r"\b(?:[A-Z]\.){4,}[A-Z]?\b")
UPPER_TOKEN_RE = re.compile(r"\b[A-Z]{2,8}\b")
WORD_RE = re.compile(r"[A-Za-z0-9_+#/-]+")
SPACE_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Public contract helpers
# ---------------------------------------------------------------------------


def analyze_text(raw_text: str, context_packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Primary entrypoint.

    Returns a deterministic analysis bundle suitable for:
    - SarahMemoryAdvCU semantic compression intake
    - SarahMemoryNeuron clarification state / query memory
    - SarahMemoryReply targeted clarification wording
    """
    context_packet = context_packet if isinstance(context_packet, dict) else {}
    started_ts = time.time()
    query_id = str(context_packet.get("query_id") or context_packet.get("id") or _new_id("pta"))
    concept_group_id = str(
        context_packet.get("concept_group_id")
        or context_packet.get("meta", {}).get("concept_group_id")
        or query_id
    )

    raw_text = _coerce_text(raw_text)
    normalized_text = _normalize_text(raw_text)
    lowered = normalized_text.lower()
    words = _word_tokens(normalized_text)
    non_stop_words = [w for w in words if w.lower() not in STOPWORDS]

    domain_scores = _score_domains(lowered, words)
    likely_domain = _best_domain(domain_scores)
    likely_lane = _detect_lane(likely_domain, lowered)
    query_type = _detect_query_type(lowered, likely_lane)
    action_detected = _contains_any(lowered, ACTION_KEYWORDS)

    abbreviation_hits, ambiguities, opaque_segments = _extract_abbreviation_findings(
        normalized_text=normalized_text,
        lowered=lowered,
        likely_domain=likely_domain,
    )

    subject = _extract_subject(words, likely_domain)
    attributes = _extract_attributes(words, subject)
    output_type = _detect_output_type(lowered, likely_domain)
    surface_task = _extract_surface_task_contract(normalized_text, preferred_app=subject)
    if surface_task:
        task_kind = str(surface_task.get("task_kind") or "").strip().lower()
        likely_lane = "action"
        if task_kind == "website_scaffold":
            likely_domain = "DOC"
            query_type = "website_scaffold"
            output_type = "website"
        elif task_kind == "spreadsheet_template":
            likely_domain = "DOC"
            query_type = "spreadsheet_template"
            output_type = "spreadsheet"
        elif task_kind == "paint_draw":
            likely_domain = "CRT"
            query_type = "paint_draw"
            output_type = "image"
        elif task_kind == "browser_search":
            likely_domain = "RSH"
            query_type = "browser_search"
            output_type = "action_result"
        elif task_kind == "browser_open_url":
            likely_domain = "RSH"
            query_type = "browser_open_url"
            output_type = "action_result"
        elif task_kind == "open_named_document":
            likely_domain = "DOC"
            query_type = "open_named_document"
            output_type = "document"
        else:
            likely_domain = "DOC"
            query_type = "document_write"
            output_type = "document"
        if surface_task.get("requested_app"):
            subject = str(surface_task.get("requested_app"))
    semantic_summary = _build_semantic_summary(
        likely_lane=likely_lane,
        likely_domain=likely_domain,
        subject=subject,
        attributes=attributes,
        normalized_text=normalized_text,
    )

    known_vocab_score = _score_known_vocab(words, abbreviation_hits)
    context_anchor_score = _score_context_anchor(normalized_text, ambiguities)
    domain_match_score = max(domain_scores.values()) if domain_scores else 0.0
    entity_resolution_score = _score_entity_resolution(ambiguities, abbreviation_hits)
    syntax_stability_score = _score_syntax_stability(normalized_text)
    repeat_history_score = _score_repeat_history(normalized_text, context_packet)
    ambiguity_score = _score_ambiguity(ambiguities, likely_lane)
    opaque_text_score = _score_opaque_text(normalized_text, opaque_segments)
    execution_risk_score = _score_execution_risk(
        lowered=lowered,
        likely_lane=likely_lane,
        ambiguities=ambiguities,
    )

    compression_confidence = _clamp(
        0.28 * context_anchor_score
        + 0.22 * domain_match_score
        + 0.18 * known_vocab_score
        + 0.12 * repeat_history_score
        + 0.10 * entity_resolution_score
        + 0.10 * syntax_stability_score
        - 0.25 * ambiguity_score
        - 0.30 * execution_risk_score
        - 0.35 * opaque_text_score
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
    )

    clarification_question = ""
    needs_clarification = compression_mode == "clarify_first" or bool(ambiguities and action_detected)
    if needs_clarification:
        clarification_question = build_clarification_question(
            {
                "ambiguities": ambiguities,
                "likely_lane": likely_lane,
                "likely_domain": likely_domain,
                "normalized_text": normalized_text,
                "symbolic_packet_partial": symbolic_packet_partial,
            }
        )

    result = {
        "ok": True,
        "module": MODULE_NAME,
        "version": PROJECT_VERSION,
        "analysis_ts": started_ts,
        "elapsed_ms": round((time.time() - started_ts) * 1000.0, 3),
        "query_id": query_id,
        "concept_group_id": concept_group_id,
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
        "likely_domain": likely_domain,
        "likely_lane": likely_lane,
        "query_type": query_type,
        "output_type": output_type,
        "subject": subject,
        "attributes": attributes,
        "action_detected": bool(action_detected),
        "known_abbreviations": abbreviation_hits,
        "ambiguities": ambiguities,
        "opaque_segments": opaque_segments,
        "safe_segments": _extract_safe_segments(normalized_text, ambiguities, opaque_segments),
        "symbolic_packet_partial": symbolic_packet_partial,
        "needs_clarification": bool(needs_clarification),
        "clarification_question": clarification_question,
        "semantic_summary": semantic_summary,
        "surface_task": surface_task,
        "state": _state_from_mode(compression_mode),
        "domain_scores": {k: round(v, 6) for k, v in domain_scores.items()},
        "context_packet_meta": _minimal_context_meta(context_packet),
    }
    return result



def build_clarification_question(analysis_result: Dict[str, Any]) -> str:
    """
    Create a focused clarification prompt using only unresolved slots.
    """
    analysis_result = analysis_result if isinstance(analysis_result, dict) else {}
    ambiguities = analysis_result.get("ambiguities") if isinstance(analysis_result.get("ambiguities"), list) else []
    if not ambiguities:
        return "Please clarify the unresolved part of your request."

    primary = ambiguities[0] if isinstance(ambiguities[0], dict) else {}
    slot = str(primary.get("slot") or "value").replace("_", " ")
    raw = str(primary.get("raw") or "that part")
    candidates = primary.get("candidates") if isinstance(primary.get("candidates"), list) else []
    if candidates:
        trimmed = [str(c) for c in candidates[:MAX_CLARIFICATION_CHOICES] if str(c).strip()]
        if len(trimmed) == 1:
            return f"Do you mean {trimmed[0]} for '{raw}'?"
        if len(trimmed) == 2:
            return f"Do you mean {trimmed[0]} or {trimmed[1]} for '{raw}'?"
        left = ", ".join(trimmed[:-1])
        return f"Please clarify the {slot} '{raw}'. Do you mean {left}, or {trimmed[-1]}?"
    return f"Please clarify what you mean by '{raw}' for the {slot}."



def merge_clarification_answer(frame: Dict[str, Any], user_answer: str) -> Dict[str, Any]:
    """
    Merge a clarification response into an existing analysis frame.

    Clarification is an amendment to the same conceptual query, not a new query.
    """
    merged = copy.deepcopy(frame if isinstance(frame, dict) else {})
    user_answer = _normalize_text(_coerce_text(user_answer))
    merged.setdefault("clarification_history", [])
    merged["clarification_history"].append(
        {
            "ts": time.time(),
            "answer": user_answer,
        }
    )
    ambiguities = merged.get("ambiguities") if isinstance(merged.get("ambiguities"), list) else []

    resolved_any = False
    for item in ambiguities:
        if not isinstance(item, dict):
            continue
        if item.get("resolved"):
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

    if resolved_any and not unresolved_remaining:
        merged["needs_clarification"] = False
        merged["clarification_question"] = ""
        merged["compression_mode"] = "partial"
        merged["state"] = "clarified"
    elif unresolved_remaining:
        merged["needs_clarification"] = True
        merged["clarification_question"] = build_clarification_question(merged)
        merged["state"] = "awaiting_clarification"

    finalized = finalize_symbolic_packet(merged)
    merged["resolved_packet"] = finalized.get("resolved_packet", merged.get("symbolic_packet_partial", []))
    merged["state"] = finalized.get("state", merged.get("state", "parsed_partial"))
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

    return {
        "ok": True,
        "resolved_packet": resolved_packet,
        "state": state,
        "unresolved_slots": unresolved,
    }



def build_clarification_frame_record(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightweight persistence-ready record aligned with the conceptual-query model.
    """
    analysis_result = analysis_result if isinstance(analysis_result, dict) else {}
    now = time.time()
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
        "clarification_answer": "",
        "resolved_packet_json": json.dumps([], ensure_ascii=False),
        "state": str(analysis_result.get("state") or "parsed_partial"),
        "confidence": float(analysis_result.get("compression_confidence") or 0.0),
        "created_ts": now,
        "updated_ts": now,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
    if _contains_any(lowered, ACTION_KEYWORDS):
        scores.setdefault("ACT", 0.0)
        scores["ACT"] = max(scores["ACT"], 0.55)
    if _contains_any(lowered, QUESTION_KEYWORDS):
        scores.setdefault("QRY", 0.0)
        scores["QRY"] = max(scores["QRY"], 0.55)
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



def _detect_lane(likely_domain: str, lowered: str) -> str:
    if likely_domain in {"WEATHER", "QRY", "RSH"}:
        return "answer"
    if likely_domain in {"DOC", "DRV", "ACT"}:
        return "action"
    if likely_domain in {"CRT"}:
        return "creative"
    if likely_domain in {"SYS"}:
        return "system"
    if likely_domain in {"NET"}:
        return "network"
    if lowered.endswith("?") or _contains_any(lowered, QUESTION_KEYWORDS):
        return "answer"
    if _contains_any(lowered, ACTION_KEYWORDS):
        return "action"
    return "answer"



def _detect_query_type(lowered: str, likely_lane: str) -> str:
    if likely_lane == "system":
        return "system_status"
    if likely_lane == "network":
        return "network_operation"
    if likely_lane == "creative":
        return "creative_request"
    if likely_lane == "action":
        return "command"
    if "temperature" in lowered or "forecast" in lowered or "weather" in lowered:
        return "factual_weather"
    if lowered.endswith("?"):
        return "question"
    return "general_request"



def _detect_output_type(lowered: str, likely_domain: str) -> str:
    if likely_domain == "DOC":
        return "document"
    if likely_domain == "CRT":
        return "artifact"
    if likely_domain in {"WEATHER", "SYS", "QRY"}:
        return "text_answer"
    if likely_domain in {"DRV", "ACT", "NET"}:
        return "action_result"
    return "text"



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
    return attrs[:8]



def _build_semantic_summary(
    likely_lane: str,
    likely_domain: str,
    subject: str,
    attributes: Sequence[str],
    normalized_text: str,
) -> str:
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
    attrs = ", ".join(list(attributes)[:3]) if attributes else normalized_text[:80]
    return f"{likely_lane} request involving {subject} with attributes {attrs}"



def _extract_abbreviation_findings(
    normalized_text: str,
    lowered: str,
    likely_domain: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    abbreviation_hits: List[Dict[str, Any]] = []
    ambiguities: List[Dict[str, Any]] = []
    opaque_segments: List[str] = []

    # Dotted acronym chains first: most suspicious.
    for match in OPAQUE_CHAIN_RE.finditer(normalized_text):
        token = match.group(0)
        letters_only = token.replace(".", "")
        if letters_only.lower() in SAFE_COMMON_ABBREVIATIONS:
            abbreviation_hits.append(
                {
                    "raw": token,
                    "normalized": SAFE_COMMON_ABBREVIATIONS[letters_only.lower()],
                    "kind": "safe_common",
                    "confidence": 0.95,
                }
            )
            continue
        opaque_segments.append(token)
        ambiguities.append(
            {
                "slot": "abbreviation_chain",
                "raw": token,
                "candidates": [],
                "confidence": 0.05,
                "reason": "opaque_dotted_chain",
                "resolved": False,
            }
        )

    # Uppercase abbreviations and state-style tokens.
    for token in UPPER_TOKEN_RE.findall(normalized_text):
        if token.lower() in SAFE_COMMON_ABBREVIATIONS:
            abbreviation_hits.append(
                {
                    "raw": token,
                    "normalized": SAFE_COMMON_ABBREVIATIONS[token.lower()],
                    "kind": "safe_common",
                    "confidence": 0.97,
                }
            )
            continue
        state_candidates = US_STATE_ABBREVIATIONS.get(token.upper())
        if state_candidates:
            confidence = 0.90 if len(state_candidates) == 1 else 0.42
            if len(state_candidates) == 1:
                abbreviation_hits.append(
                    {
                        "raw": token,
                        "normalized": state_candidates[0],
                        "kind": "state_abbreviation",
                        "confidence": confidence,
                    }
                )
            else:
                ambiguities.append(
                    {
                        "slot": "location",
                        "raw": token,
                        "candidates": state_candidates,
                        "confidence": confidence,
                        "reason": "ambiguous_state_or_city_abbreviation",
                        "resolved": False,
                    }
                )
            continue

        if len(token) >= 5:
            ambiguities.append(
                {
                    "slot": "abbreviation",
                    "raw": token,
                    "candidates": [],
                    "confidence": 0.18,
                    "reason": "unknown_uppercase_token",
                    "resolved": False,
                }
            )

    # Context-specific shorthand normalization.
    if likely_domain == "WEATHER" and "temp" in lowered:
        abbreviation_hits.append(
            {
                "raw": "temp",
                "normalized": "temperature",
                "kind": "domain_common",
                "confidence": 0.94,
            }
        )
    if likely_domain == "DOC" and "doc" in lowered:
        abbreviation_hits.append(
            {
                "raw": "doc",
                "normalized": "document",
                "kind": "domain_common",
                "confidence": 0.92,
            }
        )
    return _dedupe_hits(abbreviation_hits), _dedupe_ambiguities(ambiguities), _dedupe_strings(opaque_segments)



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
        key = (str(item.get("slot")), str(item.get("raw")), str(item.get("reason")))
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
        if item.get("slot") in {"location", "abbreviation"}:
            score -= 0.05
    return _clamp(score)



def _score_entity_resolution(
    ambiguities: Sequence[Dict[str, Any]],
    abbreviation_hits: Sequence[Dict[str, Any]],
) -> float:
    if not ambiguities:
        return 0.95 if abbreviation_hits else 0.80
    unresolved_penalty = 0.15 * len(ambiguities)
    return _clamp(0.75 + (0.04 * len(abbreviation_hits)) - unresolved_penalty)



def _score_syntax_stability(normalized_text: str) -> float:
    if not normalized_text:
        return 0.0
    punct = sum(1 for ch in normalized_text if ch in "[]{}<>=|~")
    weird_ratio = punct / max(1, len(normalized_text))
    newline_penalty = 0.04 if "\n" in normalized_text else 0.0
    long_penalty = 0.06 if len(normalized_text) > 600 else 0.0
    return _clamp(0.96 - weird_ratio - newline_penalty - long_penalty)



def _score_repeat_history(normalized_text: str, context_packet: Dict[str, Any]) -> float:
    history_candidates = []
    meta = context_packet.get("meta") if isinstance(context_packet.get("meta"), dict) else {}
    for key in ("last_normalized_text", "prior_normalized_text", "repeat_hint"):
        value = meta.get(key) or context_packet.get(key)
        if isinstance(value, str) and value.strip():
            history_candidates.append(_normalize_text(value))
    if not history_candidates:
        return 0.50
    normalized_text = normalized_text.lower()
    for item in history_candidates:
        item_low = item.lower()
        if normalized_text == item_low:
            return 0.95
        if normalized_text in item_low or item_low in normalized_text:
            return 0.80
    return 0.55



def _score_ambiguity(ambiguities: Sequence[Dict[str, Any]], likely_lane: str) -> float:
    if not ambiguities:
        return 0.0
    base = min(0.90, 0.24 * len(ambiguities))
    if likely_lane == "action":
        base += 0.10
    return _clamp(base)



def _score_opaque_text(normalized_text: str, opaque_segments: Sequence[str]) -> float:
    if not normalized_text:
        return 0.0
    if not opaque_segments:
        weird = sum(1 for ch in normalized_text if ch in "^~`|")
        return _clamp(weird / max(8.0, len(normalized_text) / 4.0))
    raw_penalty = sum(len(seg) for seg in opaque_segments) / max(1.0, len(normalized_text))
    return _clamp(0.45 + raw_penalty)



def _score_execution_risk(lowered: str, likely_lane: str, ambiguities: Sequence[Dict[str, Any]]) -> float:
    risk = 0.10
    if likely_lane == "action":
        risk += 0.25
    for token in RISKY_ACTION_PATTERNS:
        if token in lowered:
            risk += 0.10
    if ambiguities:
        risk += 0.12 * len(ambiguities)
    return _clamp(risk)



def _score_maintainability(
    likely_lane: str,
    likely_domain: str,
    subject: str,
    attributes: Sequence[str],
    ambiguities: Sequence[Dict[str, Any]],
) -> float:
    score = 0.92
    if likely_lane == "action":
        score -= 0.03
    if likely_domain == "GEN":
        score -= 0.05
    if not subject:
        score -= 0.08
    if len(attributes) > 8:
        score -= 0.05
    if ambiguities:
        score -= 0.07 * len(ambiguities)
    return _clamp(score)



def _decide_compression_mode(
    compression_confidence: float,
    ambiguity_score: float,
    opaque_text_score: float,
    execution_risk_score: float,
    action_detected: bool,
) -> str:
    if execution_risk_score >= 0.80:
        return "clarify_first"
    if opaque_text_score >= 0.75:
        return "raw_only"
    if ambiguity_score >= 0.70 and action_detected:
        return "clarify_first"
    if compression_confidence >= FULL_COMPRESS_THRESHOLD:
        return "full"
    if compression_confidence >= PARTIAL_COMPRESS_THRESHOLD:
        return "partial"
    if compression_confidence >= CLARIFY_THRESHOLD:
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
) -> List[str]:
    packet: List[str] = []

    domain = likely_domain if likely_domain != "GEN" else _lane_to_packet_type(likely_lane)

    # Weather
    if likely_domain == "WEATHER":
        loc_value = _extract_location_value(normalized_text, ambiguities, abbreviation_hits)
        mode = "forecast" if "forecast" in lowered else "current"
        packet.append(f"QRY[WEATHER {mode} loc:{loc_value}]")
    # Document
    elif likely_domain == "DOC":
        if "open" in lowered:
            packet.append(f"DOC[OPEN app:{_extract_doc_app(lowered)}]")
        else:
            packet.append(f"DOC[CREATE app:{_extract_doc_app(lowered)}]")
        if subject:
            packet.append(f"CTX[subject:{_symbol_safe(subject)}]")
    # System
    elif likely_domain == "SYS":
        metric = _extract_system_metric(lowered, subject, attributes)
        packet.append(f"SYS[DIAG target:{_symbol_safe(subject or 'system')} metric:{_symbol_safe(metric)}]")
    # Driver
    elif likely_domain == "DRV":
        packet.append(f"DRV[{_extract_driver_packet(lowered, subject, attributes)}]")
    # Network
    elif likely_domain == "NET":
        packet.append(f"NET[{_extract_network_packet(lowered, subject, attributes)}]")
    # Creative
    elif likely_domain == "CRT":
        packet.append(f"CRT[{_extract_creative_packet(lowered, subject, attributes)}]")
    # Research
    elif likely_domain == "RSH":
        packet.append(f"RSH[BROWSER query:{_symbol_safe(_collapse_for_packet(normalized_text))}]")
    else:
        packet_type = _lane_to_packet_type(likely_lane)
        packet.append(f"{packet_type}[domain:{_symbol_safe(domain)} query_type:{_symbol_safe(query_type)}]")
        if subject:
            packet.append(f"CTX[subject:{_symbol_safe(subject)}]")

    if output_type:
        packet.append(f"CTX[output_type:{_symbol_safe(output_type)}]")

    if attributes:
        packet.append(f"CTX[attributes:{_symbol_safe(','.join(attributes[:4]))}]")

    for item in ambiguities:
        if not isinstance(item, dict):
            continue
        slot = _symbol_safe(str(item.get("slot") or "value"))
        raw = _symbol_safe(str(item.get("raw") or ""))
        packet.append(f"AMBIG[{slot}:{raw}]")

    if compression_mode == "raw_only":
        packet.append("STATE[raw_only]")
    elif ambiguities or compression_mode == "clarify_first":
        packet.append("STATE[awaiting_clarification]")
    elif compression_mode == "full":
        packet.append("STATE[resolved]")
    else:
        packet.append("STATE[parsed_partial]")

    return packet



def _lane_to_packet_type(lane: str) -> str:
    mapping = {
        "answer": "QRY",
        "action": "ACT",
        "creative": "CRT",
        "system": "SYS",
        "network": "NET",
    }
    return mapping.get(lane, "QRY")



def _extract_location_value(
    normalized_text: str,
    ambiguities: Sequence[Dict[str, Any]],
    abbreviation_hits: Sequence[Dict[str, Any]],
) -> str:
    for item in ambiguities:
        if isinstance(item, dict) and item.get("slot") == "location":
            return _symbol_safe(str(item.get("raw") or "unknown"))
    for item in abbreviation_hits:
        if isinstance(item, dict) and item.get("kind") == "state_abbreviation":
            return _symbol_safe(str(item.get("normalized") or "unknown"))
    match = re.search(r"\bin\s+([A-Za-z][A-Za-z\s,.-]{1,64})$", normalized_text)
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



def _extract_color(lowered: str) -> str:
    for color in ("red", "green", "blue", "yellow", "purple", "orange", "white", "black", "pink"):
        if color in lowered:
            return color
    return ""



def _extract_safe_segments(
    normalized_text: str,
    ambiguities: Sequence[Dict[str, Any]],
    opaque_segments: Sequence[str],
) -> List[str]:
    text = normalized_text
    for item in ambiguities:
        raw = str(item.get("raw") or "")
        if raw:
            text = text.replace(raw, " ")
    for raw in opaque_segments:
        text = text.replace(raw, " ")
    parts = [p.strip() for p in re.split(r"[,;]", text) if p.strip()]
    return parts[:6]



def _state_from_mode(mode: str) -> str:
    mapping = {
        "full": "resolved",
        "partial": "parsed_partial",
        "clarify_first": "awaiting_clarification",
        "raw_only": "raw_only",
    }
    return mapping.get(mode, "parsed_partial")



def _replacement_for_ambiguity_line(
    line: str,
    ambiguities: Sequence[Dict[str, Any]],
    resolved_map: Dict[str, str],
) -> str:
    inner = line[len("AMBIG["):-1] if line.startswith("AMBIG[") and line.endswith("]") else ""
    if ":" not in inner:
        return ""
    slot, raw = inner.split(":", 1)
    key = f"{slot}:{raw}".lower()
    resolved = resolved_map.get(key)
    if resolved:
        prefix = "LOC" if slot == "location" else "CTX"
        return f"{prefix}[{_symbol_safe(slot)}:{_symbol_safe(resolved)}]"
    for item in ambiguities:
        if not isinstance(item, dict):
            continue
        if str(item.get("slot") or "") == slot and str(item.get("raw") or "") == raw:
            if item.get("resolved") and item.get("resolved_value"):
                prefix = "LOC" if slot == "location" else "CTX"
                return f"{prefix}[{_symbol_safe(slot)}:{_symbol_safe(str(item.get('resolved_value')))}]"
    return ""



def _resolve_candidate_from_answer(candidates: Sequence[str], user_answer: str) -> str:
    answer = _normalize_text(user_answer).lower()
    if not answer:
        return ""
    for candidate in candidates:
        cand = _normalize_text(candidate).lower()
        if answer == cand or answer in cand or cand in answer:
            return candidate
    # lightweight abbreviation fallbacks
    answer_words = set(_word_tokens(answer))
    for candidate in candidates:
        cand_words = set(_word_tokens(candidate.lower()))
        if answer_words and cand_words and (answer_words & cand_words):
            return candidate
    return ""



def _collapse_for_packet(text: str) -> str:
    words = _word_tokens(text)
    compact = "_".join(w.lower() for w in words[:12] if w)
    return compact[:120] or "request"



def _symbol_safe(value: Any) -> str:
    try:
        text = "" if value is None else str(value)
    except Exception:
        text = ""
    text = text.strip().replace(" ", "_")
    text = re.sub(r"[^A-Za-z0-9_.,:+#/-]", "", text)
    return text[:160] or "unknown"



def _minimal_context_meta(context_packet: Dict[str, Any]) -> Dict[str, Any]:
    meta = context_packet.get("meta") if isinstance(context_packet.get("meta"), dict) else {}
    return {
        "session_id": str(context_packet.get("session_id") or meta.get("session_id") or ""),
        "source": str(context_packet.get("source") or ""),
        "mode": str(context_packet.get("mode") or ""),
        "local_only": bool(meta.get("local_only") or False),
        "safe_mode": bool(meta.get("safe_mode") or False),
    }



def _get_nested(obj: Dict[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur



def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        num = float(value)
    except Exception:
        return low
    if num < low:
        return low
    if num > high:
        return high
    return num


# ---------------------------------------------------------------------------
# Optional SQL schema helper
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


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")

    samples = [
        "Give me the temperature right now in Nacogdoches, TX",
        "Open Word and write report for LA plant",
        "Turn my keyboard rgb blue and make sure caps lock is off",
        "G.Y.M.F.A.I.T.H.B.I.W.Y.A.",
    ]

    for sample in samples:
        print("\n" + "=" * 90)
        print("INPUT:", sample)
        analysis = analyze_text(sample, context_packet={"session_id": "demo-session", "source": "cli", "mode": "LOCAL"})
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
        if analysis.get("needs_clarification"):
            print("CLARIFICATION:", analysis.get("clarification_question"))
            if analysis.get("ambiguities"):
                merged = merge_clarification_answer(analysis, "Louisiana")
                print("MERGED:", json.dumps(merged.get("resolved_packet"), indent=2, ensure_ascii=False))

# =============================================================================
# V10/V9C Chat Classification Skeleton
# =============================================================================
def build_chat_classification_skeleton(text: str, context_packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return a deterministic pre-token skeleton for /api/chat.

    This does not execute actions and does not answer. It separates addressing
    SarahMemory from identity intent, and marks live hardware/body-map questions
    for SelfAware Evidence Court routing.
    """
    raw = _coerce_text(text)
    normalized = _normalize_text(raw).lower()
    addressed = bool(re.search(r"(?<![a-z0-9])(sarahmemory|sarah ai|sarah)(?![a-z0-9])", normalized))
    hardware_terms = (
        "cpu", "processor", "gpu", "graphics", "motherboard", "mainboard", "baseboard",
        "ram", "memory", "disk", "drive", "storage", "network adapter", "wifi", "wi-fi",
        "ethernet", "temperature", "temp", "fan", "rpm", "sata", "usb", "nvme", "pcie",
    )
    self_scope_terms = ("your", "you", "system", "runtime", "body map", "body-map", "computer", "machine", "pc")
    is_selfaware = any(k in normalized for k in hardware_terms) and any(k in normalized for k in self_scope_terms)
    identity_patterns = (
        r"\bwhat\s+(?:is|'s)\s+your\s+name\b",
        r"\bwho\s+are\s+you\b",
        r"\byour\s+name\b",
        r"\bwhat\s+version\b",
        r"\bwho\s+(?:made|created|built|designed|engineered|developed)\s+you\b",
    )
    is_identity = (not is_selfaware) and any(re.search(p, normalized) for p in identity_patterns)
    fact_kind = ""
    if is_selfaware:
        if "gpu" in normalized or "graphics" in normalized:
            fact_kind = "gpu"
        elif "cpu" in normalized or "processor" in normalized:
            fact_kind = "temperature" if "temp" in normalized or "temperature" in normalized else "cpu"
        elif "motherboard" in normalized or "mainboard" in normalized or "baseboard" in normalized:
            fact_kind = "motherboard"
        elif "ram" in normalized or "memory" in normalized:
            fact_kind = "memory"
        elif "network" in normalized or "wifi" in normalized or "wi-fi" in normalized or "ethernet" in normalized:
            fact_kind = "network_card"
        elif "drive" in normalized or "disk" in normalized or "storage" in normalized:
            fact_kind = "storage_topology"
        else:
            fact_kind = "general_system_fact"
    domain = "selfaware_body" if is_selfaware else ("identity" if is_identity else "general")
    return {
        "contract": "V10_V9C_UNIVERSAL_RUNTIME_BODY_MEMORY_AUTHORITY",
        "module": MODULE_NAME,
        "raw_text": raw,
        "normalized_text": normalized,
        "addressed_bot": addressed,
        "addressed_name": "SarahMemory" if addressed else "",
        "domain_hint": domain,
        "intent_hint": "hardware_fact" if is_selfaware else ("identity_query" if is_identity else "general"),
        "fact_kind_hint": fact_kind,
        "requires_evidence_court": bool(is_selfaware),
        "context_packet_seen": bool(isinstance(context_packet, dict)),
    }
