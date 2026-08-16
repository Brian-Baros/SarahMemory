"""--==The SarahMemory Project==--
File: SarahMemoryCompare.py
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

RESPONSE COMPARISON ENGINE v9.0.0
==============================================

This module provides multi-source comparison, advanced semantic analysis,
and comprehensive quality metrics while maintaining backward compatibility
with existing SarahMemory modules.

This file is patched to hardcode the Compare GuardDogs required by the
SarahMemory foundation framework:
1. Output Validity Guard
2. Source Consensus Guard
3. Intent Policy Guard

These guarddogs act as the final answer acceptance judge before a response
may move into presentation generation.
===============================================================================
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "guarddog_engine"
# CATEGORY = "response_validation"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = ""
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "compare_guarddogs"
# FAMILY = "core_quality_control"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-07-11"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Response comparison and final-answer judging engine implementing Output Validity, Source Consensus, and Intent Policy GuardDogs."
# --- SARAHMETA END ---

import datetime
import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Union

import numpy as np

# =============================================================================
# CORE SARAHMEMORY IMPORTS
# =============================================================================
import SarahMemoryGlobals as config
from SarahMemoryGlobals import (


    API_RESPONSE_CHECK_TRAINER,
    API_RESEARCH_ENABLED,
    COMPARE_VOTE,
    DATASETS_DIR,
    DEBUG_MODE,
    LOCAL_DATA_ENABLED,
    WEB_RESEARCH_ENABLED,
)

# ARILE organ sentinel helper. This file reports local variance to the central
# SarahMemoryARILE.py engine without owning ARILE authority.
try:
    from SarahMemoryARILE import ARILESentinelBase, arile_emit, arile_should_run
except Exception:  # pragma: no cover
    ARILESentinelBase = object  # type: ignore
    arile_emit = None  # type: ignore
    def arile_should_run(lane: str, source: str = "unknown", default: bool = True) -> bool:
        return bool(default)

class LocalARILESentinel(ARILESentinelBase):
    organ_name = __name__

    def report(self, failure_type: str, summary: str, severity: float = 0.50, **data) -> None:
        try:
            if callable(arile_emit):
                arile_emit(source=__name__, organ=self.organ_name, kind="organ_variance", failure_type=failure_type, severity=severity, confidence=0.82, risk="high" if severity >= 0.75 else "medium", summary=summary, requires_governance=severity >= 0.75, retention="security_audit" if severity >= 0.75 else "diagnostic", data=data)
        except Exception:
            pass

_local_arile_sentinel = LocalARILESentinel()

COMPARE_THRESHOLD_VALUE = float(getattr(config, "COMPARE_THRESHOLD_VALUE", 0.65) or 0.65)

# =============================================================================
# MODULE IMPORTS WITH ERROR HANDLING
# =============================================================================
try:
    from SarahMemoryWebSYM import WebSemanticSynthesizer
except Exception:
    WebSemanticSynthesizer = None

try:
    from SarahMemoryDatabase import (
        record_qa_feedback,
        auto_correct_dataset_entry,
        tokenize_text,
    )
except Exception:
    record_qa_feedback = None
    auto_correct_dataset_entry = None
    tokenize_text = None

try:
    from SarahMemoryAdvCU import evaluate_similarity, get_vector_score
    from SarahMemoryAdvCU import embed_text as fallback_embed
except Exception:
    evaluate_similarity = None
    get_vector_score = None
    fallback_embed = None

try:
    from SarahMemoryAPI import send_to_api
except Exception:
    send_to_api = None

try:
    from SarahMemoryResearch import APIResearch
except Exception:
    APIResearch = None

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logger = logging.getLogger("SarahMemoryCompare")
logger.setLevel(logging.DEBUG if DEBUG_MODE else logging.INFO)
_handler = logging.NullHandler()
_formatter = logging.Formatter("%(asctime)s - v8.0 - %(name)s - %(levelname)s - %(message)s")
_handler.setFormatter(_formatter)
if not logger.hasHandlers():
    logger.addHandler(_handler)

# =============================================================================
# OFFLINE-FIRST EMBEDDER
# =============================================================================
class _LiteEmbedder:
    """
    Deterministic, zero-dependency text embedder for offline mode.
    """

    def __init__(self, dim: int = 128):
        self.dim = int(dim or 128)
        logger.info(f"[v8.0] LiteEmbedder initialized (dim={self.dim})")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        out: List[np.ndarray] = []
        for t in texts:
            h = hashlib.sha256((t or "").encode("utf-8")).digest()
            expanded = h * ((self.dim // 32) + 1)
            v = np.frombuffer(expanded, dtype=np.uint8)[: self.dim].astype(np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            else:
                v = np.ones(self.dim, dtype=np.float32) / np.sqrt(self.dim)
            out.append(v)
        return np.stack(out, axis=0)

    def __repr__(self) -> str:
        return f"<LiteEmbedder(dim={self.dim})>"


# =============================================================================
# INTERNAL NORMALIZATION / POLICY HELPERS
# =============================================================================
def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(x) for x in value)
    if isinstance(value, dict):
        value = value.get("text") or value.get("response") or value.get("reply") or ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _looks_like_placeholder(text: str) -> bool:
    t = _normalize_text(text).lower()
    if not t:
        return True
    placeholders = {
        "none",
        "null",
        "n/a",
        "na",
        "unknown",
        "todo",
        "tbd",
        "pass",
        "placeholder",
        "coming soon",
        "no response",
        "empty",
    }
    return t in placeholders


def _looks_malformed(text: str) -> bool:
    t = _normalize_text(text)
    if not t:
        return True
    if len(t) < 2:
        return True
    if t.count("{") != t.count("}"):
        return True
    if t.count("[") != t.count("]"):
        return True
    return False


def _is_echo_only(user_text: str, generated_response: str) -> bool:
    return _normalize_text(user_text).lower() == _normalize_text(generated_response).lower()


def _threshold_for_intent(intent: str) -> float:
    it = (intent or "general").strip().lower()
    thresholds = {
        "math": 0.85,
        "logic": 0.85,
        "calculation": 0.85,
        "diagnostics": 0.80,
        "system": 0.80,
        "status": 0.80,
        "code": 0.75,
        "research": 0.70,
        "general": 0.65,
        "question": 0.65,
        "creative": 0.45,
        "image": 0.45,
        "music": 0.45,
        "video": 0.45,
        "chat": 0.55,
    }
    return float(thresholds.get(it, COMPARE_THRESHOLD_VALUE))


def _allowed_no_source_pass(intent: str, response_text: str) -> float:
    """
    Conservative no-source confidence floor.
    Used only when no comparison sources are available.
    """
    it = (intent or "general").lower().strip()
    txt = _normalize_text(response_text)
    if not txt:
        return 0.0
    if it in {"math", "logic", "calculation"}:
        return 0.72 if re.search(r"-?\d", txt) else 0.55
    if it in {"diagnostics", "system", "status"}:
        return 0.68
    if it in {"creative", "image", "music", "video"}:
        return 0.58
    return 0.56


def _safe_enabled(module_name: str) -> bool:
    """
    Governance-aware capability check.
    Discovery is not activation.
    If Globals exposes approval helpers, use them. Otherwise default to True for
    backward compatibility.
    """
    try:
        fn = getattr(config, "is_core_module_approved", None)
        if callable(fn):
            return bool(fn(module_name))
    except Exception:
        pass
    try:
        fn2 = getattr(config, "is_core_file_approved", None)
        if callable(fn2):
            return bool(fn2(module_name))
    except Exception:
        pass
    return True


def _safe_fetch_local_hits(user_text: str) -> Optional[str]:
    if not LOCAL_DATA_ENABLED:
        return None
    if not _safe_enabled("SarahMemoryDatabase.py"):
        return None
    try:
        from SarahMemoryDatabase import search_answers as _search_answers  # type: ignore
        hits = _search_answers(user_text)
        if hits:
            first = hits[0]
            if isinstance(first, dict):
                return _normalize_text(first.get("text") or first.get("answer") or first.get("response"))
            return _normalize_text(first)
    except Exception as e:
        logger.debug(f"[v8.0][Sources] Local search_answers failed: {e}")

    try:
        from SarahMemoryPersonality import get_reply_from_db  # type: ignore
        local_resp = get_reply_from_db("general")
        if local_resp:
            if isinstance(local_resp, list):
                return _normalize_text(local_resp[0])
            return _normalize_text(local_resp)
    except Exception as e:
        logger.debug(f"[v8.0][Sources] Personality fallback failed: {e}")
    return None


# =============================================================================
# SENTENCE MODEL SELECTOR
# =============================================================================
def get_active_sentence_model():
    """
    Offline-first sentence embedding model selector.
    Contract:
      - Exactly ONE embeddings model is selected per call via Globals resolver.
      - Offline/safe/local-only mode falls back to _LiteEmbedder.
      - Never hardcodes remote repo strings here.
    """
    offline = bool(
        getattr(config, "LOCAL_ONLY_MODE", False)
        or getattr(config, "SAFE_MODE", False)
        or os.getenv("HF_HUB_OFFLINE") == "1"
        or os.getenv("TRANSFORMERS_OFFLINE") == "1"
    )

    candidates: List[str] = []
    try:
        resolver = getattr(config, "resolve_model", None)
        if callable(resolver):
            res = resolver("embeddings", text="", meta=None, models_dir=getattr(config, "MODELS_DIR", None)) or {}
            selected = res.get("selected")
            fallbacks = res.get("fallbacks") or []
            candidates = [c for c in ([selected] + list(fallbacks)) if c]
    except Exception:
        candidates = []

    if not candidates or offline:
        if offline:
            logger.info("[v8.0][Embedder] Offline mode → LiteEmbedder")
        return _LiteEmbedder()

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        logger.warning(f"[v8.0][Embedder] sentence-transformers unavailable → {e}")
        return _LiteEmbedder()

    for repo in candidates:
        try:
            model = SentenceTransformer(repo, local_files_only=True)
            logger.info(f"[v8.0][Embedder] Loaded local model: {repo}")
            return model
        except Exception as e:
            logger.debug(f"[v8.0][Embedder] Load failed ({repo}): {e}")

    logger.info("[v8.0][Embedder] No local embeddings model loaded → LiteEmbedder")
    return _LiteEmbedder()


# =============================================================================
# SOURCE FETCHING
# =============================================================================
def fetch_all_sources(
    user_text: str,
    intent: str,
    allow_local: bool = True,
    allow_web: bool = True,
    allow_api: bool = True,
) -> Dict[str, Any]:
    """
    Fetch responses from enabled comparison sources.

    Backward compatible:
    - Existing callers can still use fetch_all_sources(user_text, intent)
    - Additional flags are optional and default to True
    """
    sources: Dict[str, Any] = {}

    if allow_local and LOCAL_DATA_ENABLED:
        local_text = _safe_fetch_local_hits(user_text)
        if local_text:
            sources["local"] = local_text

    if allow_web and WEB_RESEARCH_ENABLED and WebSemanticSynthesizer and _safe_enabled("SarahMemoryWebSYM.py"):
        try:
            if hasattr(WebSemanticSynthesizer, "synthesize_response"):
                web_resp = WebSemanticSynthesizer.synthesize_response("", user_text)
            else:
                web_resp = None
            web_text = _normalize_text(web_resp)
            if len(web_text) > 10:
                sources["web"] = web_text
                logger.debug(f"[v8.0][Sources] Web synthesis success: {len(web_text)} chars")
        except Exception as e:
            logger.warning(f"[v8.0][Sources] Web source error: {e}")

    if allow_api and API_RESEARCH_ENABLED and send_to_api and _safe_enabled("SarahMemoryAPI.py"):
        try:
            chosen_model = None
            try:
                select_api_model = getattr(config, "select_api_model", None)
                if callable(select_api_model):
                    chosen_model = select_api_model(intent=intent)
            except Exception:
                chosen_model = None

            resp = send_to_api(
                user_input=user_text,
                provider=getattr(config, "PRIMARY_API", "openai"),
                intent=intent,
                model=chosen_model,
            )
            if isinstance(resp, dict) and resp.get("data"):
                source_name = resp.get("source", "api")
                sources[source_name] = _normalize_text(resp.get("data"))
            elif APIResearch:
                try:
                    result = APIResearch.query(user_text, intent)  # type: ignore[attr-defined]
                    if isinstance(result, dict) and result.get("data"):
                        sources[result.get("source", "api")] = _normalize_text(result.get("data"))
                except Exception as e2:
                    logger.debug(f"[v8.0][Sources] Legacy API fallback failed: {e2}")
        except Exception as e:
            logger.warning(f"[v8.0][Sources] API source error: {e}")

    logger.info(f"[v8.0][Sources] Fetched {len(sources)} sources: {list(sources.keys())}")
    return sources


# =============================================================================
# QUALITY / METRICS / GUARDDOG SUPPORT
# =============================================================================
def calculate_similarity_metrics(text1: str, text2: str) -> Dict[str, float]:
    """
    Calculate multi-layer similarity metrics between two texts.
    """
    text1 = _normalize_text(text1)
    text2 = _normalize_text(text2)
    metrics = {"similarity_score": 0.0, "vector_score": 0.0, "token_overlap": 0.0}

    try:
        if tokenize_text:
            tokens1 = tokenize_text(text1)
            tokens2 = tokenize_text(text2)
            if not isinstance(tokens1, list):
                tokens1 = list(tokens1) if tokens1 else []
            if not isinstance(tokens2, list):
                tokens2 = list(tokens2) if tokens2 else []
        else:
            tokens1 = re.findall(r"\w+", text1.lower())
            tokens2 = re.findall(r"\w+", text2.lower())

        overlap = len(set(tokens1).intersection(set(tokens2)))
        total = max(len(set(tokens1).union(set(tokens2))), 1)
        metrics["token_overlap"] = round(overlap / total, 3)

        if get_vector_score:
            try:
                metrics["vector_score"] = round(float(get_vector_score(text1, text2)), 3)
            except Exception as e:
                logger.debug(f"[v8.0][Metrics] Vector score failed: {e}")
        elif fallback_embed:
            try:
                v1 = fallback_embed(text1)
                v2 = fallback_embed(text2)
                if v1 is not None and v2 is not None:
                    arr1 = np.asarray(v1).reshape(-1).astype(np.float32)
                    arr2 = np.asarray(v2).reshape(-1).astype(np.float32)
                    denom = float(np.linalg.norm(arr1) * np.linalg.norm(arr2))
                    if denom > 0:
                        metrics["vector_score"] = round(float(np.dot(arr1, arr2) / denom), 3)
            except Exception as e:
                logger.debug(f"[v8.0][Metrics] Fallback embed score failed: {e}")

        if evaluate_similarity:
            try:
                metrics["similarity_score"] = round(float(evaluate_similarity(text1, text2)), 3)
            except Exception as e:
                logger.debug(f"[v8.0][Metrics] Semantic similarity failed: {e}")
        else:
            metrics["similarity_score"] = round(
                (metrics["token_overlap"] * 0.65) + (metrics["vector_score"] * 0.35), 3
            )
    except Exception as e:
        logger.warning(f"[v8.0][Metrics] Metric calculation error: {e}")

    return metrics


def get_comparison_metrics(text1: str, text2: str) -> Dict[str, float]:
    """
    Public compatibility helper for raw comparison metrics.
    """
    return calculate_similarity_metrics(text1, text2)


def calibrate_confidence(metrics: Dict[str, float], intent: str = "general") -> float:
    """
    Convert raw metrics into a calibrated confidence score.
    """
    sim = float(metrics.get("similarity_score") or 0.0)
    vec = float(metrics.get("vector_score") or 0.0)
    tok = float(metrics.get("token_overlap") or 0.0)

    it = (intent or "general").lower().strip()
    if it in {"math", "logic", "diagnostics", "system", "status"}:
        conf = (sim * 0.45) + (vec * 0.35) + (tok * 0.20)
    else:
        conf = (sim * 0.50) + (vec * 0.30) + (tok * 0.20)
    return round(max(0.0, min(1.0, conf)), 3)


def validate_response_quality(
    user_text: str,
    generated_response: str,
    intent: str = "general",
) -> Dict[str, Any]:
    """
    GuardDog A - Output Validity Guard.
    """
    response_text = _normalize_text(generated_response)
    checks = {
        "non_empty": bool(response_text),
        "not_placeholder": not _looks_like_placeholder(response_text),
        "not_malformed": not _looks_malformed(response_text),
        "not_echo_only": not _is_echo_only(user_text, response_text),
    }

    issues: List[str] = []
    if not checks["non_empty"]:
        issues.append("empty_response")
    if not checks["not_placeholder"]:
        issues.append("placeholder_response")
    if not checks["not_malformed"]:
        issues.append("malformed_response")
    if not checks["not_echo_only"]:
        issues.append("echo_only_response")

    it = (intent or "general").lower().strip()
    if it in {"math", "logic", "calculation"} and not re.search(r"-?\d", response_text):
        checks["math_shape"] = False
        issues.append("math_missing_numeric_result")
    else:
        checks["math_shape"] = True

    if it in {"diagnostics", "system", "status"} and len(response_text) < 3:
        checks["system_shape"] = False
        issues.append("system_response_too_short")
    else:
        checks["system_shape"] = True

    accepted = not issues
    decision = "PASS" if accepted else ("REPAIRABLE" if response_text else "REJECT")
    return {
        "accepted": accepted,
        "decision": decision,
        "issues": issues,
        "checks": checks,
        "normalized_response": response_text,
    }


def multi_source_consensus(
    generated_response: str,
    response_pool: Dict[str, Any],
    intent: str = "general",
) -> Dict[str, Any]:
    """
    GuardDog B - Source Consensus Guard.
    """
    response_text = _normalize_text(generated_response)
    all_scores: List[Dict[str, Any]] = []
    best_match: Optional[Dict[str, Any]] = None
    best_score = 0.0

    for source_name, source_value in (response_pool or {}).items():
        source_text = _normalize_text(source_value)
        if not source_text:
            continue
        metrics = calculate_similarity_metrics(response_text, source_text)
        confidence = calibrate_confidence(metrics, intent=intent)
        entry = {
            "source": source_name,
            "confidence": confidence,
            "response": source_text,
            **metrics,
        }
        all_scores.append(entry)
        if confidence > best_score:
            best_score = confidence
            best_match = entry

    threshold = _threshold_for_intent(intent)
    strong = [s for s in all_scores if float(s.get("confidence") or 0.0) >= threshold]
    majority_ratio = 0.0
    if all_scores:
        majority_ratio = round(len(strong) / max(len(all_scores), 1), 3)

    if best_match and best_score >= threshold:
        decision = "PASS"
    elif best_match and best_score >= max(0.45, threshold - 0.15):
        decision = "WEAK_PASS"
    elif all_scores:
        decision = "CONFLICT"
    else:
        decision = "NO_SOURCES"

    return {
        "decision": decision,
        "best_match": best_match,
        "best_score": round(best_score, 3),
        "all_scores": all_scores,
        "strong_count": len(strong),
        "total_count": len(all_scores),
        "majority_ratio": majority_ratio,
        "threshold": threshold,
    }


def advanced_compare_reply(user_text: str, generated_response: str, intent: str = "general") -> Dict[str, Any]:
    """
    Non-breaking advanced wrapper. Uses the same core path as compare_reply().
    """
    return compare_reply(user_text, generated_response, intent=intent)


# =============================================================================
# RESPONSE COMPARISON - GUARDDOG ENFORCED
# =============================================================================
def compare_reply(user_text: str, generated_response: str, intent: str = "general") -> Dict[str, Any]:
    """
    Compare generated response against multiple sources with the 3 Compare GuardDogs.

    GuardDog A - Output Validity Guard
    GuardDog B - Source Consensus Guard
    GuardDog C - Intent Policy Guard
    """
    if not API_RESPONSE_CHECK_TRAINER:
        logger.info("[v8.0][Compare] Comparison disabled (API_RESPONSE_CHECK_TRAINER=False)")
        return {
            "status": "SKIPPED",
            "feedback": "API response comparison is disabled.",
            "confidence": 0.0,
            "accepted": True,
            "decision": "SKIPPED",
            "guarddogs": {},
            "version": "9.0.0",
        }

    timestamp = datetime.datetime.now().isoformat()
    try:
        response_text = _normalize_text(generated_response)

        # ---------------------------------------------------------------------
        # GuardDog A - Output Validity Guard
        # ---------------------------------------------------------------------
        validity = validate_response_quality(user_text, response_text, intent=intent)
        if not validity.get("accepted"):
            result = {
                "status": "REJECT",
                "feedback": "Output validity guard rejected candidate response.",
                "confidence": 0.0,
                "similarity_score": 0.0,
                "vector_score": 0.0,
                "token_overlap": 0.0,
                "source": None,
                "intent": intent,
                "api_response": "",
                "threshold": _threshold_for_intent(intent),
                "all_scores": [],
                "accepted": False,
                "decision": "REJECT",
                "recommended_next": "retry_same_category",
                "guarddogs": {
                    "validity": validity,
                    "consensus": {"decision": "SKIPPED"},
                    "intent_policy": {"decision": "SKIPPED"},
                },
                "version": "9.0.0",
            }
            logger.info(f"[v8.0][Compare] Validity REJECT | Issues: {validity.get('issues')}")
            return result

        # ---------------------------------------------------------------------
        # Fetch comparison sources
        # ---------------------------------------------------------------------
        allow_web = not bool(getattr(config, "LOCAL_ONLY_MODE", False) or getattr(config, "SAFE_MODE", False))
        allow_api = allow_web
        response_pool = fetch_all_sources(
            user_text,
            intent,
            allow_local=True,
            allow_web=allow_web,
            allow_api=allow_api,
        )

        # ---------------------------------------------------------------------
        # GuardDog B - Source Consensus Guard
        # ---------------------------------------------------------------------
        consensus = multi_source_consensus(response_text, response_pool, intent=intent)

        # ---------------------------------------------------------------------
        # GuardDog C - Intent Policy Guard
        # ---------------------------------------------------------------------
        threshold = float(consensus.get("threshold") or _threshold_for_intent(intent))
        confidence = float(consensus.get("best_score") or 0.0)
        best_match = consensus.get("best_match") or {}
        decision = "REJECT"
        accepted = False
        recommended_next = "fallback_to_lower_deterministic_tier"
        status = "MISS"
        feedback = "MISS"

        if consensus.get("decision") == "PASS":
            accepted = True
            decision = "ACCEPT"
            recommended_next = "none"
            status = "HIT"
            feedback = "HIT"
        elif consensus.get("decision") == "WEAK_PASS":
            accepted = True
            decision = "PASS_WITH_WARNING"
            recommended_next = "none"
            status = "HIT"
            feedback = "HIT"
        elif consensus.get("decision") == "NO_SOURCES":
            confidence = _allowed_no_source_pass(intent, response_text)
            if confidence >= max(0.55, threshold - 0.15):
                accepted = True
                decision = "ACCEPT_SELF_CONSISTENT"
                recommended_next = "none"
                status = "HIT"
                feedback = "HIT"
            else:
                accepted = False
                decision = "RETRY_SAME_CATEGORY"
                recommended_next = "retry_same_category"
        else:
            if confidence >= max(0.50, threshold - 0.20):
                decision = "RETRY_SAME_CATEGORY"
                recommended_next = "retry_same_category"
            else:
                decision = "FALLBACK_TO_LOWER_TIER"
                recommended_next = "fallback_to_lower_deterministic_tier"

        if record_qa_feedback:
            try:
                record_qa_feedback(
                    user_text,
                    score=1 if accepted else 0,
                    feedback=(
                        f"{decision} | confidence={confidence:.3f} | "
                        f"source={best_match.get('source', 'none')} | "
                        f"intent={intent} | timestamp={timestamp}"
                    ),
                )
            except Exception as e:
                logger.warning(f"[v8.0][Compare] Failed to record QA feedback: {e}")

        audit_data = {
            "user_text": user_text,
            "response_given": response_text,
            "status": status,
            "feedback": feedback,
            "decision": decision,
            "accepted": accepted,
            "intent": intent,
            "threshold": threshold,
            "timestamp": timestamp,
            "guarddogs": {
                "validity": validity,
                "consensus": consensus,
                "intent_policy": {
                    "decision": decision,
                    "accepted": accepted,
                    "recommended_next": recommended_next,
                },
            },
            "version": "9.0.0",
        }

        try:
            audit_dir = os.path.join(DATASETS_DIR, "logs", "compare_audits")
            os.makedirs(audit_dir, exist_ok=True)
            audit_filename = f"compare_audit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            audit_path = os.path.join(audit_dir, audit_filename)
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(audit_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"[v8.0][Compare] Failed to save audit: {e}")

        if COMPARE_VOTE:
            logger.info("[v8.0][COMPARE_VOTE] User feedback requested: Was this helpful? [Yes/No]")

        result = {
            "status": status,
            "feedback": feedback,
            "confidence": round(confidence, 3),
            "score": round(confidence, 3),
            "final_score": round(confidence, 3),
            "rating": round(confidence, 3),
            "similarity_score": float(best_match.get("similarity_score", 0.0) if best_match else 0.0),
            "vector_score": float(best_match.get("vector_score", 0.0) if best_match else 0.0),
            "token_overlap": float(best_match.get("token_overlap", 0.0) if best_match else 0.0),
            "source": best_match.get("source") if best_match else None,
            "intent": intent,
            "api_response": best_match.get("response", "") if best_match else "",
            "threshold": threshold,
            "all_scores": consensus.get("all_scores", []),
            "accepted": accepted,
            "decision": decision,
            "recommended_next": recommended_next,
            "guarddogs": {
                "validity": validity,
                "consensus": {
                    "decision": consensus.get("decision"),
                    "majority_ratio": consensus.get("majority_ratio"),
                    "strong_count": consensus.get("strong_count"),
                    "total_count": consensus.get("total_count"),
                },
                "intent_policy": {
                    "decision": decision,
                    "accepted": accepted,
                    "recommended_next": recommended_next,
                },
            },
            "version": "9.0.0",
        }

        logger.info(
            f"[v8.0][Compare] Result: {status} | Decision: {decision} | "
            f"Confidence: {confidence:.3f} | Source: {result.get('source')}"
        )
        return result

    except Exception as e:
        logger.error(f"[v8.0][Compare] Comparison error: {e}", exc_info=True)
        return {
            "status": "ERROR",
            "feedback": str(e),
            "confidence": 0.0,
            "score": 0.0,
            "final_score": 0.0,
            "rating": 0.0,
            "accepted": False,
            "decision": "ERROR",
            "recommended_next": "fallback_to_lower_deterministic_tier",
            "guarddogs": {},
            "version": "9.0.0",
        }


# =============================================================================
# COMPARE MULTIPLE CANDIDATES
# =============================================================================
def compare_candidates(
    user_text: str,
    candidates: Dict[str, str],
    intent: str = "general",
    *,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compare multiple candidate replies against a shared source pool and select a winner.
    """
    meta = meta or {}
    allow_web = not bool(meta.get("offline") or meta.get("local_only") or getattr(config, "LOCAL_ONLY_MODE", False))
    allow_api = allow_web

    sources = fetch_all_sources(
        user_text,
        intent,
        allow_local=True,
        allow_web=allow_web,
        allow_api=allow_api,
    )

    scored: Dict[str, Any] = {}
    winner_key: Optional[str] = None
    winner_reply: str = ""
    winner_score: float = -1.0
    winner_source: Optional[str] = None

    if sources:
        for key, reply in (candidates or {}).items():
            reply_text = _normalize_text(reply)
            if not reply_text:
                continue
            consensus = multi_source_consensus(reply_text, sources, intent=intent)
            best_score = float(consensus.get("best_score") or 0.0)
            best_match = consensus.get("best_match") or {}
            scored[key] = {
                "reply": reply_text,
                "best_score": best_score,
                "best_source": best_match.get("source"),
                "best_match": best_match,
                "consensus": consensus,
            }
            if best_score > winner_score:
                winner_key = key
                winner_reply = reply_text
                winner_score = best_score
                winner_source = best_match.get("source")
    else:
        for key, reply in (candidates or {}).items():
            reply_text = _normalize_text(reply)
            if not reply_text:
                continue
            result = compare_reply(user_text, reply_text, intent=intent)
            score = float(
                result.get("confidence")
                or result.get("score")
                or result.get("final_score")
                or result.get("rating")
                or 0.0
            )
            scored[key] = {
                "reply": reply_text,
                "best_score": score,
                "best_source": result.get("source"),
                "best_match": result,
                "consensus": result.get("guarddogs", {}).get("consensus", {}),
            }
            if score > winner_score:
                winner_key = key
                winner_reply = reply_text
                winner_score = score
                winner_source = result.get("source")

    accepted_candidates = [k for k, v in scored.items() if float(v.get("best_score") or 0.0) >= _threshold_for_intent(intent)]
    majority = None
    if accepted_candidates:
        ratio = round(len(accepted_candidates) / max(len(scored), 1), 3)
        majority = {"key": winner_key, "ratio": ratio}

    return {
        "ok": True,
        "winner_key": winner_key,
        "winner_reply": winner_reply,
        "winner_score": (winner_score if winner_key is not None else 0.0),
        "winner_source": winner_source,
        "majority": majority,
        "scored": scored,
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def comparison_summary_line(status: str, confidence: float, source_label: str, intent_label: str) -> str:
    try:
        return (
            f"[v8.0][Comparison] Status: {status} | "
            f"Confidence: {float(confidence):.3f} | "
            f"Source: {source_label} | "
            f"Intent: {intent_label}"
        )
    except Exception:
        return "[v8.0][Comparison] Status: UNKNOWN | Confidence: 0.0 | Source: Unknown | Intent: undetermined"


def _maybe_store_compare_hit(
    confidence: Optional[float],
    threshold: Optional[float],
    query: str,
    reply: str,
    intent: str,
    compare_source: str,
    dbmod=None,
) -> None:
    if confidence is None:
        return
    threshold = float(threshold if threshold is not None else COMPARE_THRESHOLD_VALUE)

    if float(confidence) >= threshold:
        try:
            if dbmod is None:
                import SarahMemoryDatabase as dbmod  # type: ignore
            if hasattr(dbmod, "store_comparison_outcome"):
                dbmod.store_comparison_outcome(
                    query=query,
                    reply=reply,
                    intent=intent,
                    source=compare_source,
                    confidence=confidence,
                )
                logger.debug(f"[v8.0] Stored comparison hit: {float(confidence):.3f}")
        except Exception as e:
            logger.debug(f"[v8.0] Failed to store comparison hit: {e}")


# =============================================================================
# DATABASE TABLE INITIALIZATION
# =============================================================================
def _ensure_response_table(db_path: Optional[str] = None) -> None:
    try:
        import sqlite3

        if db_path is None:
            base = getattr(config, "BASE_DIR", os.getcwd())
            datasets_dir = getattr(config, "DATASETS_DIR", os.path.join(base, "data", "memory", "datasets"))
            db_path = os.path.join(datasets_dir, "system_logs.db")

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS response (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                user TEXT,
                content TEXT,
                source TEXT,
                intent TEXT,
                confidence REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
        conn.close()
        logger.debug(f"[v8.0][DB] Ensured 'response' table in {db_path}")
    except Exception as e:
        logger.warning(f"[v8.0][DB] Failed to ensure response table: {e}")


try:
    _ensure_response_table()
except Exception:
    pass


# =============================================================================
# MODE FLAGS
# =============================================================================
try:
    LOCAL_ONLY = bool(getattr(config, "LOCAL_ONLY_MODE", False))
    WEB_OK = bool(getattr(config, "WEB_RESEARCH_ENABLED", False))
    API_OK = bool(getattr(config, "API_RESEARCH_ENABLED", False))
    logger.debug(f"[v8.0][Config] LOCAL_ONLY={LOCAL_ONLY}, WEB_OK={WEB_OK}, API_OK={API_OK}")
except Exception as e:
    logger.warning(f"[v8.0][Config] Failed to load mode flags: {e}")
    LOCAL_ONLY = False
    WEB_OK = False
    API_OK = False


# =============================================================================
# __SM_EVOLUTION_HELPERS__
# =============================================================================
def select_best_suggestion(user_text: str, candidates: Dict[str, str], intent: str = "code") -> Dict[str, Any]:
    """
    Compare candidate suggestion texts and choose the best.
    Backward compatible return shape is preserved.
    """
    scores: Dict[str, float] = {}
    best_key: Optional[str] = None
    best_score: float = float("-inf")
    best_text: str = ""

    try:
        multi = compare_candidates(user_text, candidates, intent=intent)
        scored = multi.get("scored", {}) or {}
        for key, payload in scored.items():
            score_f = float(payload.get("best_score") or 0.0)
            scores[key] = score_f
            if score_f > best_score:
                best_key = key
                best_score = score_f
                best_text = _normalize_text(payload.get("reply"))
        if best_key is None:
            best_key = multi.get("winner_key")
            best_text = _normalize_text(multi.get("winner_reply"))
            best_score = float(multi.get("winner_score") or 0.0)
    except Exception:
        for key, txt in (candidates or {}).items():
            if not txt:
                continue
            try:
                result = compare_reply(user_text, txt, intent=intent)  # type: ignore
                score = None
                if isinstance(result, dict):
                    score = result.get("confidence")
                    if score is None:
                        score = result.get("score")
                    if score is None:
                        score = result.get("final_score")
                    if score is None:
                        score = result.get("rating")
                score_f = float(score or 0.0)
            except Exception:
                score_f = 0.0
            scores[key] = score_f
            if score_f > best_score:
                best_score = score_f
                best_key = key
                best_text = _normalize_text(txt)

    return {
        "ok": True,
        "intent": intent,
        "best_key": best_key,
        "best_score": (best_score if best_key is not None else 0.0),
        "scores": scores,
        "best_text": best_text,
    }


# =============================================================================
# MAIN TEST HARNESS
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("SarahMemory Compare v9.0.0 - Test Mode")
    print("=" * 80)

    input_text = input("\nEnter prompt for test comparison: ").strip()
    if not input_text:
        input_text = "What is the weather today?"

    test_response = input("Enter AI-generated response: ").strip()
    if not test_response:
        test_response = "The weather today is sunny and warm."

    intent = input("Enter intent (default: general): ").strip() or "general"

    print("\n" + "=" * 80)
    print("Running Comparison...")
    print("=" * 80)

    result = compare_reply(input_text, test_response, intent=intent)

    print("\n" + "=" * 80)
    print("Comparison Results:")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    print("=" * 80)

    if result.get("status") != "ERROR":
        summary = comparison_summary_line(
            result.get("status", "UNKNOWN"),
            result.get("confidence", 0.0),
            result.get("source", "unknown"),
            result.get("intent", "undetermined"),
        )
        print(f"\n{summary}\n")

logger.info("[v8.0] SarahMemoryCompare module loaded successfully")

# -----------------------------------------------------------------------------
# V10/V9G SelfAware answer-shape guard
# -----------------------------------------------------------------------------
def compare_selfaware_answer_contract(user_text: str, response_text: str, canonical_packet: Optional[Dict[str, Any]] = None, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Reject SelfAware replies that answer the wrong metric/component.

    Example: CPU-temperature question answered with CPU identity is rejected.
    This is deterministic and does not call external models.
    """
    pkt = dict(canonical_packet or {})
    raw = _normalize_text(user_text).lower() if '_normalize_text' in globals() else str(user_text or '').lower()
    reply = _normalize_text(response_text).lower() if '_normalize_text' in globals() else str(response_text or '').lower()
    metric = str(pkt.get('requested_metric') or '').lower()
    comp = str(pkt.get('requested_component') or pkt.get('target') or '').lower()
    reasons = []
    accepted = True
    if metric == 'temperature':
        has_temp = any(x in reply for x in ('°c', ' c', 'temperature', 'thermal', 'temp'))
        identity_only = any(x in reply for x in ('physical cores', 'logical threads', 'processor')) and not has_temp
        if identity_only or not has_temp:
            accepted = False; reasons.append('temperature_query_not_answered_as_temperature')
        if comp == 'cpu' and 'gpu temperature' in reply and 'cpu temperature' not in reply and 'cpu thermal' not in reply:
            accepted = False; reasons.append('gpu_temperature_substituted_for_cpu_temperature')
    if metric == 'bios_version' and ('server version' in reply or 'my name is' in reply):
        accepted = False; reasons.append('hardware_bios_version_hijacked_by_identity_version')
    if metric == 'connectivity' and 'network adapters =' in reply and ('ethernet' in raw or 'wi-fi' in raw or 'wifi' in raw):
        accepted = False; reasons.append('connectivity_question_received_adapter_dump')
    return {'accepted': bool(accepted), 'decision': 'ACCEPT' if accepted else 'REJECT_INTENT_ALIGNMENT', 'reasons': reasons, 'compare_mode': 'selfaware_answer_contract', 'sql_recorded': False}

# --- SM V8.0 TRI-LAYER PATCH 2026-05-20 ---
def validate_tri_layer_packet(packet: Dict[str, Any]) -> Dict[str, Any]:
    """GuardDog validation for tri-layer input/governance packets before presentation/action."""
    pkt = packet if isinstance(packet, dict) else {}
    failures: List[str] = []
    lang = pkt.get("language_context_packet") if isinstance(pkt.get("language_context_packet"), dict) else {}
    emo = pkt.get("emotion_affect_packet") if isinstance(pkt.get("emotion_affect_packet"), dict) else {}
    six = pkt.get("six_question_governance_packet") or pkt.get("six_question_governance")
    six = six if isinstance(six, dict) else {}
    if lang and lang.get("doctrine", {}).get("no_substring_routing_inside_locked_phrases") is not True:
        failures.append("language_phrase_lock_doctrine_missing")
    if emo and emo.get("doctrine", {}).get("emotion_never_overrides_governance") is not True:
        failures.append("emotion_governance_doctrine_missing")
    if six and six.get("doctrine", {}).get("no_loop_no_action") is not True:
        failures.append("six_question_no_loop_no_action_missing")
    return {
        "ok": not failures,
        "guard": "tri_layer_packet_guarddog",
        "failures": failures,
        "allow_presentation": not failures,
        "allow_execution": False,
        "notes": "Packet validation only; execution still requires SMGET/OperatorCore.",
    }


# -----------------------------------------------------------------------------
# SM V8 Robotic Body Expansion - embodied action contract guard
# -----------------------------------------------------------------------------
def validate_embodied_action_contract(action_contract: Dict[str, Any], body_witness: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """GuardDog validation for robot/body action claims.

    Compare confirms the proposed contract does not claim unverified body
    capability and does not bypass SMGET/OperatorCore. It does not execute.
    """
    contract = action_contract if isinstance(action_contract, dict) else {}
    witness = body_witness if isinstance(body_witness, dict) else {}
    failures: List[str] = []
    warnings: List[str] = []
    joined = " ".join([
        str(contract.get("action_type") or "").lower(),
        str(contract.get("capability_name") or "").lower(),
        str(contract.get("executor_name") or "").lower(),
        str(contract.get("target") or "").lower(),
        str(contract.get("normalized_text") or "").lower(),
    ])
    is_robotic = any(k in joined for k in ("robot", "servo", "gripper", "locomotion", "arm", "hand", "leg", "torque", "force", "walk", "move")) or str(contract.get("risk_level") or "").startswith("TIER_ROBOT")
    if is_robotic:
        if not bool(contract.get("requires_confirmation") or contract.get("confirmed") or contract.get("user_confirmed")):
            warnings.append("robotic_action_confirmation_not_visible")
        if str(contract.get("execution_mode") or "").lower() == "apply" and not bool(contract.get("body_part_driver_verified")):
            failures.append("robotic_apply_claims_unverified_body_driver")
        if str(contract.get("executor_name") or "") not in {"robotic_body_staging_executor", "emergency_instinct_executor"} and "robot" in joined:
            warnings.append("robotic_contract_executor_not_recognized_as_robotic_staging_path")
        if not bool(contract.get("safe_stop_available") or (contract.get("safety_envelope") or {}).get("safe_stop_required")):
            failures.append("robotic_contract_missing_safe_stop")
        declared = witness.get("declared_parts") if isinstance(witness.get("declared_parts"), list) else []
        target = str(contract.get("body_part") or contract.get("target_body_part") or contract.get("target") or "").lower().replace(" ", "_")
        if target and declared and target not in declared:
            failures.append("target_body_part_not_declared")
    return {
        "ok": bool(not failures),
        "guard": "embodied_action_contract_guarddog",
        "is_robotic_body_action": bool(is_robotic),
        "failures": failures,
        "warnings": warnings,
        "allow_execution": False,
        "decision": "ACCEPT_STRUCTURE" if not failures else "REJECT_EMBODIED_CONTRACT",
        "doctrine": {
            "compare_validates_claims_not_execution": True,
            "no_hallucinated_actuation": True,
            "body_witness_required_for_robotic_apply": True,
        },
    }

# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 START ---
# GuardDog helpers for agentic/interoperability envelopes.

def validate_sovereign_agent_packet(packet: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    pkt = dict(packet or {})
    message_type = str(pkt.get("message_type") or pkt.get("type") or pkt.get("action_type") or "unknown").lower()
    execution_like = message_type in {"execute", "tool_call", "command", "driver_action", "robot_motion", "filesystem_write"}
    reasons = []
    passed = True
    if execution_like:
        passed = False
        reasons.append("Packet is execution-like; Compare requires SMGET/OperatorCore route, not direct adapter execution.")
    if bool(pkt.get("bidirectional")) and not bool(pkt.get("explicit_bidirectional_authority")):
        passed = False
        reasons.append("Bidirectional packet lacks explicit governed exception.")
    if not reasons:
        reasons.append("Packet is acceptable as passive evidence/manifest/query only.")
    return {
        "ok": True,
        "passed": passed,
        "decision": "PASS" if passed else "FAIL",
        "one_way_broker": True,
        "direct_execution_allowed": False,
        "reasons": reasons,
    }


def compare_interop_envelope(envelope: Optional[Dict[str, Any]] = None, expected_protocol: str = "") -> Dict[str, Any]:
    pkt = dict(envelope or {})
    review = validate_sovereign_agent_packet(pkt)
    protocol = str(pkt.get("protocol") or pkt.get("adapter") or "").lower()
    if expected_protocol and protocol != str(expected_protocol).lower():
        review["passed"] = False
        review["decision"] = "FAIL"
        review.setdefault("reasons", []).append("Protocol does not match expected adapter contract.")
    return review
# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 END ---


def compare_agent_adapter_result_contract(
    prompt: str = "",
    adapter_result: Optional[Dict[str, Any]] = None,
    *,
    task_truth: Optional[Dict[str, Any]] = None,
    firewall_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Verify Terminal Bay read-only adapter results before release to Chat/UI.

    SARAHMEMORY_PATCH_NOTE 2026-08-04:
    Compare does not authorize execution. It verifies release/presentation only.
    Accepted adapters are explicitly bounded:
      - passported_local_get_v1 for api.local.health_check
      - passported_external_get_v1 for research.public_web / research.approved_api
    Both lanes remain GET-only, non-mutating, source-scoped, and task-truth bound.
    """
    result = adapter_result if isinstance(adapter_result, dict) else {}
    truth = task_truth if isinstance(task_truth, dict) else {}
    firewall = firewall_result if isinstance(firewall_result, dict) else {}
    failures: List[str] = []
    warnings: List[str] = []

    adapter_name = str(result.get("adapter") or "")
    skill_id = str(truth.get("skill_id") or "")
    method = str(result.get("method") or "GET").upper()
    responses = [x for x in list(result.get("responses") or []) if isinstance(x, dict)]

    if bool(result.get("blocked")) or not bool(result.get("ok")):
        failures.append("adapter_result_not_ok")
    if method != "GET":
        failures.append("adapter_method_not_get")
    if not responses:
        failures.append("adapter_response_required")

    mutation_flags = (
        result.get("mutated"), result.get("write_performed"), result.get("shell_execution"),
        result.get("file_mutation"), result.get("driver_action"), result.get("devbridge_apply"),
        result.get("memory_write"),
    )
    if any(bool(x) for x in mutation_flags):
        failures.append("adapter_claims_mutation")

    if adapter_name == "passported_local_get_v1":
        if skill_id != "api.local.health_check":
            failures.append("unexpected_skill_id")
        expected_state = "verified_read_only_adapter_result"
    elif adapter_name == "passported_external_get_v1":
        if skill_id not in {"research.public_web", "research.approved_api"}:
            failures.append("unexpected_skill_id")
        expected_state = "verified_external_read_only_adapter_result"
    else:
        failures.append("unknown_adapter_contract")
        expected_state = "adapter_result_rejected"

    allowed_sources = set(str(x or "").rstrip("/") for x in list(truth.get("allowed_sources") or []))
    observed_sources = set(str(x.get("url") or "").rstrip("/") for x in responses if isinstance(x, dict))
    if allowed_sources and observed_sources and not observed_sources.issubset(allowed_sources):
        failures.append("observed_source_outside_task_truth")
    for response in responses:
        if str(response.get("method") or "GET").upper() != "GET":
            failures.append("response_method_not_get")
        if not bool(response.get("ok")):
            failures.append("response_not_ok")
        if bool(response.get("mutated") or response.get("file_mutation") or response.get("shell_execution")):
            failures.append("response_claims_mutation")
        if adapter_name == "passported_external_get_v1":
            url = str(response.get("url") or "")
            if not url.lower().startswith("https://"):
                failures.append("external_response_not_https")

    if str(firewall.get("verdict") or "ALLOW").upper() == "DENY":
        failures.append("firewall_denied_result")
    if not bool(truth.get("compare_required", True)):
        warnings.append("compare_not_marked_required_in_task_truth")

    failures = sorted(set(failures))
    warnings = sorted(set(warnings))
    accepted = not failures
    return {
        "ok": accepted,
        "accepted": accepted,
        "decision": "PASS" if accepted else "FAIL",
        "verified_answer_state": expected_state if accepted else "adapter_result_rejected",
        "failures": failures,
        "warnings": warnings,
        "source_count": len(observed_sources),
        "adapter": adapter_name,
        "skill_id": skill_id,
        "execution_authority": False,
        "presentation_allowed": accepted,
        "doctrine": {
            "compare_verifies_release_not_execution": True,
            "read_only_get_only": True,
            "task_truth_required": True,
            "external_get_allowed_only_when_passported_and_scoped": True,
        },
    }

# ====================================================================
# END OF SarahMemoryCompare.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryCompare',
    "version": "v9.0.0-alpha-sml-0.2",
    "category": 'Governance',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['comparison', 'governance'],
    "supported_missions": ['Conversation', 'Execution', 'Governance', 'Security'],
    "supported_omega": ['Ω001', 'Ω050', 'Ω060'],
    "required_authority": ['Read', 'Research'],
    "priority": 90,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryCompare.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return dict(SML_ORGAN_METADATA)


def sml_health():
    """Return a local SML health vector without side effects."""
    return {
        "status": "Healthy",
        "availability": 1.0,
        "integrity": 1.0,
        "performance": 1.0,
        "reliability": 1.0,
        "confidence": 0.75,
        "latency_ms": 0.0,
        "stability": 1.0,
        "compatibility": 1.0,
        "notes": ["SML adapter present"],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryCompare',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryCompare', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

# --- SML COMPARE SPECIALIZATION START ---
def sml_verify_packet(packet, confidence=None, note="Compare verified SML packet"):
    """Mark packet verification and optionally update SML confidence."""
    from SarahMemorySMLProtocol import SMLPacket, sml_touch_packet
    pkt = packet if isinstance(packet, SMLPacket) else SMLPacket.from_dict(packet)
    updates = {}
    if confidence is not None:
        try:
            updates["confidence"] = max(0.0, min(1.0, float(confidence)))
        except Exception:
            pass
    return sml_touch_packet(pkt, organ="SarahMemoryCompare", action="verify", omega="Ω050", note=note, updates=updates)
# --- SML COMPARE SPECIALIZATION END ---

