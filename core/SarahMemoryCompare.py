"""--==The SarahMemory Project==--
File: SarahMemoryCompare.py
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

RESPONSE COMPARISON ENGINE v9.1.0a
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

# v9.1.0a: bounded verification and truthful provenance (private helpers).
import ast as _compare_ast
import copy
import math
import secrets
import threading
import time
from pathlib import Path

_COMPARE_MAX_QUERY = 16_384
_COMPARE_MAX_TEXT = 65_536
_COMPARE_MAX_SOURCE = 32_768
_COMPARE_MAX_SOURCES = 8
_COMPARE_MAX_CANDIDATES = 12
_COMPARE_AUDIT_LOCK = threading.RLock()
_COMPARE_MAX_AUDIT_FILE_BYTES = 8 * 1024 * 1024
_COMPARE_MAX_AUDIT_PARTS = 16


def _parse_bool(value: Any) -> bool:
    return value is True or (isinstance(value, (str, int)) and str(value).strip().lower() in ('1','true','yes','on','enabled'))


def _cfg_true(name: str, default: Any = False) -> bool:
    try:
        return _parse_bool(getattr(config, name, default))
    except Exception:
        return False


def _verify_arithmetic(question: str, candidate: str) -> Optional[bool]:
    """Verify only small, unambiguous elementary expressions with exact arithmetic.

    None = out of scope, not an approval. No eval, arbitrary functions, or code.
    """
    q=_normalize_text(question).strip().lower().rstrip(' ?.!=')
    m=re.fullmatch(r'(?:what\s+is|calculate|compute|evaluate)\s+(-?\d+(?:\.\d+)?(?:\s*[-+*/×÷]\s*-?\d+(?:\.\d+)?){1,4})',q)
    if not m:return None
    expression=m.group(1).replace('×','*').replace('÷','/')
    if len(expression)>100:return None
    try:
        parsed=_compare_ast.parse(expression,mode='eval')
        def calc(node, depth=0):
            if depth>12:raise ValueError('depth')
            if isinstance(node,_compare_ast.Expression):return calc(node.body,depth+1)
            if isinstance(node,_compare_ast.Constant) and type(node.value) in (int,float):return float(node.value)
            if isinstance(node,_compare_ast.UnaryOp) and isinstance(node.op,(_compare_ast.USub,_compare_ast.UAdd)):
                v=calc(node.operand,depth+1);return -v if isinstance(node.op,_compare_ast.USub) else v
            if isinstance(node,_compare_ast.BinOp) and isinstance(node.op,(_compare_ast.Add,_compare_ast.Sub,_compare_ast.Mult,_compare_ast.Div)):
                a=calc(node.left,depth+1);b=calc(node.right,depth+1)
                if isinstance(node.op,_compare_ast.Add):return a+b
                if isinstance(node.op,_compare_ast.Sub):return a-b
                if isinstance(node.op,_compare_ast.Mult):return a*b
                if b==0:raise ValueError('zero division')
                return a/b
            raise ValueError('unsupported expression')
        expected=calc(parsed)
        if not math.isfinite(expected) or abs(expected)>1e12:return None
        answer=_normalize_text(candidate).strip()
        a=re.search(r'(?:answer\s*(?:is|:)|=)\s*(-?\d+(?:\.\d+)?)\b',answer,re.I)
        if not a:
            a=re.fullmatch(r'-?\d+(?:\.\d+)?',answer)
        if not a:return None
        number=float(a.group(1) if a.lastindex else a.group(0))
        return math.isclose(expected,number,rel_tol=1e-9,abs_tol=1e-9)
    except Exception:
        return None


def _claim_contradiction(candidate: str, source: str) -> bool:
    """Conservative flag for obvious numeric/negation conflicts; no semantic truth claim."""
    a,b=str(candidate).lower(),str(source).lower()
    wa,wb=set(re.findall(r"\b[a-z]+\b",a)),set(re.findall(r"\b[a-z]+\b",b))
    overlap=len(wa&wb)/max(1,len(wa|wb))
    negatives={'not','never','no','false','cannot','isnt','wasnt','without'}
    if overlap>=0.25 and bool(wa&negatives)!=bool(wb&negatives):return True
    na=set(re.findall(r"(?<!\w)-?\d+(?:\.\d+)?\b",a))
    nb=set(re.findall(r"(?<!\w)-?\d+(?:\.\d+)?\b",b))
    if na and nb and na!=nb and overlap>=0.2:return True
    if (' yes ' in ' '+a+' ') != (' yes ' in ' '+b+' ') and (' no ' in ' '+a+' ' or ' no ' in ' '+b+' '):return True
    return False


def _redacted_preview(value: str, limit: int = 160) -> str:
    text=str(value or '')[:limit]
    text=re.sub(r'(?i)(password|api[_-]?key|secret|token|credential)\s*[:=]\s*[^\s,;]+',r'\1=[REDACTED]',text)
    return text


def _audit_compare_result(result: Dict[str,Any], query: str, reply: str) -> str:
    """Bounded append-only daily audit. No raw user prompts retained by default."""
    if not _cfg_true('SARAH_COMPARE_AUDIT_ENABLED',True):return 'disabled'
    try:
        root=Path(getattr(config,'DATASETS_DIR',DATASETS_DIR)).absolute()
        if root.is_symlink() or (root/'logs').is_symlink() or (root/'logs'/'compare_audits').is_symlink():
            return 'error_symlink_root'
        audit_root=root/'logs'/'compare_audits'
        audit_root.mkdir(parents=True,exist_ok=True)
        receipt={"version":"9.1.0a","ts":datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 "event_id":secrets.token_hex(12),"decision":result.get('decision'),
                 "accepted":result.get('accepted'),"verified":result.get('verified'),
                 "confidence":result.get('confidence'),"intent":str(result.get('intent') or '')[:60],
                 "query_sha256":hashlib.sha256(str(query).encode('utf8')).hexdigest(),
                 "response_sha256":hashlib.sha256(str(reply).encode('utf8')).hexdigest(),
                 "source":str(result.get('source') or '')[:80]}
        if _cfg_true('SARAH_COMPARE_AUDIT_INCLUDE_PREVIEWS',False):
            receipt['query_preview']=_redacted_preview(query)
            receipt['response_preview']=_redacted_preview(reply)
        line=(json.dumps(receipt,sort_keys=True,ensure_ascii=False,default=str)+'\n').encode('utf8')
        if len(line)>4096:return 'error_receipt_budget'
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d')
        with _COMPARE_AUDIT_LOCK:
            for part in range(_COMPARE_MAX_AUDIT_PARTS):
                target=audit_root/f'compare_audit_{stamp}_{part:02d}.jsonl'
                if target.is_symlink():return 'error_symlink_file'
                mode=os.O_WRONLY|os.O_CREAT|os.O_APPEND
                if hasattr(os,'O_NOFOLLOW'):mode|=os.O_NOFOLLOW
                fd=os.open(target,mode,0o600)
                try:
                    if os.stat(fd).st_size+len(line)>_COMPARE_MAX_AUDIT_FILE_BYTES:
                        continue
                    # Cooperative cross-process locking where platform APIs exist.
                    try:
                        import fcntl
                        fcntl.flock(fd,fcntl.LOCK_EX)
                    except ImportError:
                        pass
                    try:
                        if os.fstat(fd).st_size+len(line)>_COMPARE_MAX_AUDIT_FILE_BYTES:
                            continue
                        if os.write(fd,line)!=len(line):return 'error_short_write'
                        os.fsync(fd)
                        return 'recorded'
                    finally:
                        try:
                            import fcntl
                            fcntl.flock(fd,fcntl.LOCK_UN)
                        except ImportError:
                            pass
                finally:
                    os.close(fd)
        return 'error_budget_exhausted'
    except Exception as e:
        logger.warning('Compare audit unavailable: %s', e)
        return 'error'


def _finalize_compare_result(result: Dict[str,Any], query: str, reply: str, started: float) -> Dict[str,Any]:
    c=max(0.,min(1.,float(result.get('confidence') or 0.0)))
    for key in ('confidence','score','final_score','rating'):result[key]=round(c,3)
    result['elapsed_ms']=round((time.perf_counter()-started)*1000,3)
    result['audit_status']=_audit_compare_result(result,query,reply)
    # Audits are evidence, not execution authorization. Explicitly flag incomplete audit.
    result['audit_complete']=result['audit_status']=='recorded'
    return result


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
# VALIDATION_DATE = "2026-11-01"
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
    """Bound normalization, not evidence that a statement is true."""
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(x)[:_COMPARE_MAX_TEXT] for x in value[:32])
    if isinstance(value, dict):
        value = value.get("text") or value.get("response") or value.get("reply") or ""
    text = str(value)[:_COMPARE_MAX_TEXT + 1].strip()
    return re.sub(r"\s+", " ", text)


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
    """Use the authoritative CORE registry. Unknown means not approved."""
    name = os.path.splitext(os.path.basename(str(module_name or "")))[0]
    if not name:
        return False
    fn = getattr(config, "sm_is_core_module_approved", None)
    if not callable(fn):
        return False
    try:
        return fn(name) is True
    except Exception:
        return False


def _safe_fetch_local_hits(user_text: str) -> Optional[str]:
    if not _cfg_true("LOCAL_DATA_ENABLED", LOCAL_DATA_ENABLED) or not _safe_enabled("SarahMemoryDatabase.py"):
        return None
    try:
        from SarahMemoryDatabase import search_answers
        hits = search_answers(user_text[:_COMPARE_MAX_QUERY])
        if hits:
            first = hits[0]
            result = _normalize_text(first.get("text") or first.get("answer") or first.get("response")) if isinstance(first, dict) else _normalize_text(first)
            return result[:_COMPARE_MAX_SOURCE] if result else None
    except Exception as e:
        logger.debug("Local compare source unavailable: %s", e)
    # Do not substitute a random Personality reply as a verified answer to this query.
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
    user_text: str, intent: str, allow_local: bool = True, allow_web: bool = True, allow_api: bool = True
) -> Dict[str, Any]:
    """Acquire bounded, approved comparison evidence without overriding local-only mode."""
    sources: Dict[str, Any] = {}
    query = _normalize_text(user_text)
    if not query or len(query) > _COMPARE_MAX_QUERY:
        return sources
    offline = _cfg_true("SAFE_MODE") or _cfg_true("LOCAL_ONLY_MODE")
    if allow_local and _cfg_true("LOCAL_DATA_ENABLED", LOCAL_DATA_ENABLED):
        item = _safe_fetch_local_hits(query)
        if item:
            sources["local"] = item[:_COMPARE_MAX_SOURCE]
    if offline:
        return sources
    if allow_web and _cfg_true("WEB_RESEARCH_ENABLED", WEB_RESEARCH_ENABLED) and WebSemanticSynthesizer and _safe_enabled("SarahMemoryWebSYM.py"):
        try:
            fn = getattr(WebSemanticSynthesizer, "synthesize_response", None)
            if callable(fn):
                item = _normalize_text(fn("", query))
                if item and len(item) <= _COMPARE_MAX_SOURCE:
                    sources["web"] = item
        except Exception as exc:
            logger.warning("Compare web source failed: %s", exc)
    if allow_api and _cfg_true("API_RESEARCH_ENABLED", API_RESEARCH_ENABLED) and _safe_enabled("SarahMemoryAPI.py"):
        result = None
        try:
            if callable(send_to_api):
                selection = getattr(config, "select_api_model", None)
                model = selection(intent=intent) if callable(selection) else None
                result = send_to_api(user_input=query, provider=getattr(config, "PRIMARY_API", ""), intent=intent, model=model)
            if not (isinstance(result, dict) and result.get("data")) and APIResearch is not None:
                query_fn = getattr(APIResearch, "query", None)
                if callable(query_fn):
                    result = query_fn(query, intent)
            if isinstance(result, dict) and result.get("data"):
                item = _normalize_text(result["data"])
                if item and len(item) <= _COMPARE_MAX_SOURCE:
                    sources["api"] = item  # source labels from providers are not authorization.
        except Exception as exc:
            logger.warning("Compare API source failed: %s", exc)
    return dict(list(sources.items())[:_COMPARE_MAX_SOURCES])


# =============================================================================
# QUALITY / METRICS / GUARDDOG SUPPORT
# =============================================================================
def calculate_similarity_metrics(text1: str, text2: str) -> Dict[str, float]:
    """Bounded semantic/lexical similarity. This does not prove factual accuracy."""
    a = _normalize_text(text1)[:_COMPARE_MAX_SOURCE]
    b = _normalize_text(text2)[:_COMPARE_MAX_SOURCE]
    out = {"similarity_score": 0.0, "vector_score": 0.0, "token_overlap": 0.0}
    if not a or not b:
        return out
    ta, tb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    union = ta | tb
    out["token_overlap"] = round(len(ta & tb)/len(union), 3) if union else 0.0
    if a.casefold() == b.casefold():
        return {"similarity_score":1.0,"vector_score":1.0,"token_overlap":1.0}
    def safe_similarity(fn):
        try:
            x=float(fn(a,b))
            return round(max(0.0,min(1.0,x)),3) if math.isfinite(x) else 0.0
        except Exception:
            return 0.0
    if callable(get_vector_score):
        out["vector_score"] = safe_similarity(get_vector_score)
    elif callable(fallback_embed):
        try:
            v1,v2=np.asarray(fallback_embed(a),dtype=np.float64).ravel(),np.asarray(fallback_embed(b),dtype=np.float64).ravel()
            if v1.size and v1.shape==v2.shape and v1.size<=4096:
                denom=float(np.linalg.norm(v1)*np.linalg.norm(v2))
                if denom>0:
                    out["vector_score"]=round(max(0.,min(1.,float(np.dot(v1,v2)/denom))),3)
        except Exception:
            pass
    if callable(evaluate_similarity):
        out["similarity_score"]=safe_similarity(evaluate_similarity)
    else:
        out["similarity_score"]=round(0.65*out["token_overlap"]+0.35*out["vector_score"],3)
    return out


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


def validate_response_quality(user_text: str, generated_response: str, intent: str = "general") -> Dict[str, Any]:
    """GuardDog A: bounded structural validation, not truth verification."""
    raw = str(generated_response or "")
    response_text = _normalize_text(raw)
    checks = {
        "non_empty":bool(response_text),
        "not_placeholder":not _looks_like_placeholder(response_text),
        "not_malformed":not _looks_malformed(response_text) or ((intent or "").lower() in {"math","logic","calculation"} and bool(re.fullmatch(r"-?\d+(?:\.\d+)?",response_text))),
        "not_echo_only":not _is_echo_only(user_text,response_text),
        "within_budget":len(raw)<=_COMPARE_MAX_TEXT and len(response_text)<=_COMPARE_MAX_TEXT,
        "no_control_chars":not bool(re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]",raw)),
    }
    if (intent or "").lower() in {"math","logic","calculation"}:
        checks["math_shape"] = bool(re.search(r"-?\d",response_text))
    if (intent or "").lower() in {"diagnostics","system","status"}:
        checks["system_shape"] = len(response_text)>=3
    issues=[key for key,passed in checks.items() if not passed]
    return {"accepted":not issues,"decision":"PASS" if not issues else "REJECT","issues":issues,"checks":checks,"normalized_response":response_text}


def multi_source_consensus(generated_response: str, response_pool: Dict[str, Any], intent: str = "general") -> Dict[str, Any]:
    """GuardDog B: do not confuse a single best match with agreement."""
    response_text = _normalize_text(generated_response)
    scores=[]; seen=set(); conflict_detected=False
    for source_name, source_value in list((response_pool or {}).items())[:_COMPARE_MAX_SOURCES]:
        source_text = _normalize_text(source_value)
        if not source_text or len(source_text)>_COMPARE_MAX_SOURCE:
            continue
        # Identical duplicate contents do not create independent corroboration.
        fingerprint=str(source_name).casefold()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        metrics=calculate_similarity_metrics(response_text,source_text)
        confidence=calibrate_confidence(metrics,intent=intent)
        conflict_detected=conflict_detected or _claim_contradiction(response_text,source_text)
        scores.append({"source":str(source_name)[:80],"confidence":confidence,"response":source_text,**metrics})
    threshold=_threshold_for_intent(intent)
    strong=[s for s in scores if s["confidence"]>=threshold]
    best=max(scores,key=lambda s:s["confidence"],default=None)
    ratio=round(len(strong)/len(scores),3) if scores else 0.0
    if not scores:
        decision='NO_SOURCES'
    elif len(scores)==1:
        decision='SINGLE_SOURCE' if strong else 'CONFLICT'
    elif len(strong)>=2 and len(strong)==len(scores) and not conflict_detected:
        decision='PASS'
    elif strong:
        decision='CONFLICT'
    else:
        decision='CONFLICT'
    return {"decision":decision,"best_match":best,"best_score":round(best["confidence"],3) if best else 0.0,
            "all_scores":scores,"strong_count":len(strong),"total_count":len(scores),"majority_ratio":ratio,"threshold":threshold,
            "verified_consensus":False,"source_independence_verified":False,"contradiction_signal":conflict_detected}


def advanced_compare_reply(user_text: str, generated_response: str, intent: str = "general") -> Dict[str, Any]:
    """
    Non-breaking advanced wrapper. Uses the same core path as compare_reply().
    """
    return compare_reply(user_text, generated_response, intent=intent)


# =============================================================================
# RESPONSE COMPARISON - GUARDDOG ENFORCED
# =============================================================================
def compare_reply(user_text: str, generated_response: str, intent: str = "general") -> Dict[str, Any]:
    """Three GuardDogs. Never assert verification from formatting or missing evidence."""
    started=time.perf_counter()
    text=_normalize_text(generated_response)
    intent=(intent or 'general').strip().lower()
    base={"status":"MISS","feedback":"MISS","confidence":0.0,"score":0.0,"final_score":0.0,"rating":0.0,
          "similarity_score":0.0,"vector_score":0.0,"token_overlap":0.0,"source":None,"intent":intent,"api_response":"",
          "threshold":_threshold_for_intent(intent),"all_scores":[],"accepted":False,"verified":False,"ok":False,
          "decision":"REJECT","recommended_next":"retry_same_category","guarddogs":{},"version":"9.1.0a",
          "audit_status":"not_attempted","execution_authority":False}
    if not _cfg_true("API_RESPONSE_CHECK_TRAINER",API_RESPONSE_CHECK_TRAINER):
        base.update(status='SKIPPED',decision='SKIPPED',feedback='Comparison disabled; not verified',recommended_next='verify_before_release')
        return base
    validity=validate_response_quality(user_text,generated_response,intent=intent)
    base['guarddogs']['validity']=validity
    if not validity['accepted']:
        base.update(status='REJECT',decision='REJECT',feedback='Output validity rejected',recommended_next='repair_candidate')
        return _finalize_compare_result(base,user_text,text,started)
    try:
        pool=fetch_all_sources(user_text,intent,allow_local=True,allow_web=True,allow_api=True)
        consensus=multi_source_consensus(text,pool,intent=intent)
        base['guarddogs']['consensus']={k:v for k,v in consensus.items() if k!='all_scores'}
        best=consensus.get('best_match') or {}
        for key in ('similarity_score','vector_score','token_overlap'):
            base[key]=best.get(key,0.0)
        base.update(confidence=consensus['best_score'],source=best.get('source'),api_response=best.get('response',''),
                    all_scores=consensus['all_scores'])
        direct_math=_verify_arithmetic(user_text,text) if intent in {'math','calculation','logic'} else None
        if direct_math is True:
            base.update(verified=True,accepted=True,ok=True,decision='ACCEPT_DETERMINISTIC',status='HIT',feedback='HIT',confidence=1.0,source='compare.deterministic_arithmetic',recommended_next='none')
        elif direct_math is False:
            base.update(decision='REJECT_INCORRECT_ARITHMETIC',status='REJECT',recommended_next='repair_candidate')
        elif consensus['decision']=='PASS':
            # Agreement is corroboration, not independently authenticated truth.
            # Presentation may use this as a provisional candidate; Compass cannot certify it.
            base.update(accepted=True,ok=False,verified=False,decision='PASS_CORROBORATED_UNVERIFIED',status='PROVISIONAL',feedback='Corroborated; provenance not authenticated',recommended_next='verify_provenance_for_final_release')
        elif consensus['decision']=='SINGLE_SOURCE':
            base.update(decision='NEED_INDEPENDENT_EVIDENCE',recommended_next='verify_second_source')
        elif consensus['decision']=='CONFLICT':
            base.update(decision='CONFLICT',recommended_next='research_or_review_sources')
        elif intent in {'creative','image','music','video','chat','conversation','brainstorm'} and not _compare_is_general_factual_question(user_text):
            base.update(accepted=True,ok=False,verified=False,decision='PASS_WITH_UNCERTAINTY',status='PROVISIONAL',feedback='Presentation only; unverified',confidence=0.0,recommended_next='none')
        else:
            base.update(decision='NEED_MORE_EVIDENCE',recommended_next='route_to_verified_source')
        base['guarddogs']['intent_policy']={"decision":base['decision'],"accepted":base['accepted'],"verified":base['verified']}
    except Exception as exc:
        logger.warning('Compare error: %s',exc)
        base.update(status='ERROR',decision='ERROR',feedback='comparison_runtime_error',recommended_next='retry_or_defer')
    return _finalize_compare_result(base,user_text,text,started)



def _compare_is_general_factual_question(text: str) -> bool:
    t = _normalize_text(text).lower()
    if not t:
        return False
    factual_starters = ("who is", "who was", "what is", "what are", "when did", "when was", "where is", "where was", "how did", "why did")
    return t.startswith(factual_starters)


def compare_candidate_release(user_text: str, candidate: Dict[str, Any], intent: str = "general", *, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Release gate. Candidate-supplied trust flags and labels are never attestations."""
    cand=dict(candidate or {})
    answer=_normalize_text(cand.get('answer') or cand.get('reply') or cand.get('response') or '')
    source=str(cand.get('source') or 'candidate')[:100]
    source_type=str(cand.get('source_type') or source).lower()[:100]
    validity=validate_response_quality(user_text,answer,intent=intent)
    base={"ok":False,"accepted":False,"decision":"NEED_MORE_EVIDENCE","reply":"","confidence":0.0,
          "verified":False,"source":source,"source_type":source_type,"reason":"unverified_candidate_provenance",
          "execution_authority":False,"compare_verifies_release_not_execution":True,"guarddogs":{"validity":validity}}
    if not validity['accepted']:
        base.update(decision='REJECT',reason='output_validity_guard_rejected_candidate')
        return base
    math_proven=_verify_arithmetic(user_text,answer)
    if math_proven is True:
        base.update(ok=True,accepted=True,decision='PASS',reply=answer,confidence=1.0,verified=True,
                    source='compare.deterministic_arithmetic',reason='independently_evaluated_arithmetic')
        return base
    if math_proven is False:
        base.update(decision='REJECT',reason='incorrect_arithmetic_claim')
        return base
    factual=_compare_is_general_factual_question(user_text) or str(intent or '').lower() in {'math','logic','calculation','general','question','knowledge','general_knowledge','research','diagnostics','system','status','action','code'}
    if factual:
        base.update(reason='verified_external_evidence_required',recommended_next='route_to_neuron_research_or_verified_local_source')
        return base
    # Non-factual chat/creative text may be presented, but not certified as true.
    base.update(ok=True,accepted=True,decision='PASS_WITH_UNCERTAINTY',reply=answer,confidence=0.0,
                verified=False,reason='presentation_only_no_factual_attestation')
    return base


# =============================================================================
# COMPARE MULTIPLE CANDIDATES
# =============================================================================
def compare_candidates(user_text: str, candidates: Dict[str, str], intent: str = 'general', *, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Rank candidates; choose only an accepted candidate, never a placeholder."""
    meta=dict(meta or {})
    offline=_cfg_true('LOCAL_ONLY_MODE') or _cfg_true('SAFE_MODE') or _parse_bool(meta.get('offline')) or _parse_bool(meta.get('local_only'))
    valid=[]; scored={}
    for key,value in list((candidates or {}).items())[:_COMPARE_MAX_CANDIDATES]:
        check=validate_response_quality(user_text,value,intent=intent)
        if check['accepted']:
            valid.append((key,_normalize_text(value)))
        else:
            scored[key]={"reply":_normalize_text(value),"best_score":0.0,"best_source":None,"accepted":False,"validity":check}
    if not valid:
        return {"ok":False,"winner_key":None,"winner_reply":"","winner_score":0.0,"winner_source":None,"majority":None,"scored":scored}
    pool=fetch_all_sources(user_text,intent,allow_local=True,allow_web=not offline,allow_api=not offline)
    winner=None; high=-1.0; advisory_key=None; advisory_score=-1.0
    for key,answer in valid:
        verdict=multi_source_consensus(answer,pool,intent=intent)
        math_truth=_verify_arithmetic(user_text,answer) if str(intent).lower() in {'math','calculation','logic'} else None
        accepted=math_truth is True  # Only independently re-checked deterministic results are final winners.
        # Best-match alone cannot prove a majority or authorize release.
        score=1.0 if math_truth is True else verdict['best_score']
        if score>advisory_score:
            advisory_key=key;advisory_score=score
        scored[key]={"reply":answer,"best_score":score,"best_source":(verdict.get('best_match') or {}).get('source'),
                     "best_match":verdict.get('best_match'),"consensus":verdict,"accepted":accepted,
                     "validity":{"accepted":True}}
        if accepted and score>high:
            winner=key;high=score
    return {"ok":winner is not None,"winner_key":winner,"winner_reply":scored[winner]['reply'] if winner is not None else '',
            "winner_score":high if winner is not None else 0.0,
            "winner_source":scored[winner]['best_source'] if winner is not None else None,
            "majority":None,"scored":scored,"provisional_winner_key":advisory_key,
            "provisional_winner_reply":scored[advisory_key]['reply'] if advisory_key is not None else ''}


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
    confidence: Optional[float], threshold: Optional[float], query: str, reply: str,
    intent: str, compare_source: str, dbmod=None,
) -> None:
    """Compatibility stub. Numeric score cannot authorize memory/training writes.

    The original signature does not carry an authenticated user-approved receipt,
    so persistence must remain the responsibility of the governed memory organ.
    """
    return None


# =============================================================================
# DATABASE TABLE INITIALIZATION
# =============================================================================
def _ensure_response_table(db_path: Optional[str] = None) -> None:
    """Explicit legacy database initialization; never invoked during import."""
    import sqlite3
    dest=db_path or os.path.join(getattr(config,'DATASETS_DIR',DATASETS_DIR),'system_logs.db')
    root=os.path.dirname(os.path.abspath(dest))
    if not root:
        raise ValueError('invalid_database_path')
    os.makedirs(root,exist_ok=True)
    with sqlite3.connect(dest,timeout=2.0) as db:
        db.execute('CREATE TABLE IF NOT EXISTS response (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, user TEXT, content TEXT, source TEXT, intent TEXT, confidence REAL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')


# v9.1.0a: no import-time persistent database changes.


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
    result=compare_candidates(user_text,candidates,intent=intent)
    scored=result.get('scored') or {}
    return {"ok":bool(result.get('ok')),"intent":intent,"best_key":result.get('winner_key'),
            "best_score":result.get('winner_score',0.0),"scores":{k:v.get('best_score',0.0) for k,v in scored.items()},
            "best_text":result.get('winner_reply',''),"provisional_best_text":result.get('provisional_winner_reply',''),
            "provisional_best_key":result.get('provisional_winner_key')}


# =============================================================================
# MAIN TEST HARNESS
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("SarahMemory Compare v9.1.0a - Test Mode")
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
    """Validate metric alignment without inventing a physical sensor reading."""
    pkt=dict(canonical_packet or {}); raw=_normalize_text(user_text).lower();reply=_normalize_text(response_text).lower()
    metric=str(pkt.get('requested_metric') or '').lower(); comp=str(pkt.get('requested_component') or pkt.get('target') or '').lower()
    reasons=[]
    if metric=='temperature':
        has_measurement=bool(re.search(r'(?<!\w)-?\d{1,3}(?:\.\d+)?\s*(?:°\s*[cf]|degrees?\s*(?:c|f)|celsius|fahrenheit)\b',reply))
        if not has_measurement:
            reasons.append('temperature_measurement_unverified_or_missing')
        if comp=='cpu' and 'gpu temperature' in reply and 'cpu temperature' not in reply:
            reasons.append('gpu_temperature_substituted_for_cpu_temperature')
    if metric=='bios_version' and any(x in reply for x in ('server version','my name is')):
        reasons.append('hardware_bios_version_hijacked_by_identity_version')
    if metric=='connectivity' and 'network adapters =' in reply and any(x in raw for x in ('ethernet','wi-fi','wifi')):
        reasons.append('connectivity_question_received_adapter_dump')
    return {'accepted':not reasons,'decision':'ACCEPT' if not reasons else 'REJECT_INTENT_ALIGNMENT',
            'reasons':reasons,'compare_mode':'selfaware_answer_contract','sql_recorded':False,
            'verified_measurement':False,'execution_authority':False}

# --- SM V8.0 TRI-LAYER PATCH 2026-05-20 ---
def validate_tri_layer_packet(packet: Dict[str, Any]) -> Dict[str, Any]:
    pkt=packet if isinstance(packet,dict) else {}
    failures=[]
    lang=pkt.get('language_context_packet'); emo=pkt.get('emotion_affect_packet'); six=pkt.get('six_question_governance_packet') or pkt.get('six_question_governance')
    for name,item in [('language',lang),('emotion',emo),('six_question',six)]:
        if not isinstance(item,dict) or not item:
            failures.append(name+'_packet_required')
    if isinstance(lang,dict) and lang and lang.get('doctrine',{}).get('no_substring_routing_inside_locked_phrases') is not True:
        failures.append('language_phrase_lock_doctrine_missing')
    if isinstance(emo,dict) and emo and emo.get('doctrine',{}).get('emotion_never_overrides_governance') is not True:
        failures.append('emotion_governance_doctrine_missing')
    if isinstance(six,dict) and six and six.get('doctrine',{}).get('no_loop_no_action') is not True:
        failures.append('six_question_no_loop_no_action_missing')
    return {'ok':not failures,'guard':'tri_layer_packet_guarddog','failures':failures,
            'allow_presentation':not failures,'allow_execution':False,
            'notes':'Packet structure only; never grants OperatorCore execution permission.'}


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
    pkt=dict(packet or {});message_type=str(pkt.get('message_type') or pkt.get('type') or pkt.get('action_type') or 'unknown').lower()
    execution_like=message_type in {'execute','tool_call','command','driver_action','robot_motion','filesystem_write'}
    reasons=[]
    if not pkt:
        reasons.append('empty_packet')
    if execution_like:
        reasons.append('execution_like_packet_requires_operatorcore')
    if _parse_bool(pkt.get('bidirectional')) and not _parse_bool(pkt.get('explicit_bidirectional_authority')):
        reasons.append('bidirectional_authority_missing')
    ok=not reasons
    return {'ok':ok,'passed':ok,'decision':'PASS' if ok else 'FAIL','one_way_broker':True,
            'direct_execution_allowed':False,'reasons':reasons or ['passive_evidence_only']}


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


def compare_agent_adapter_result_contract(prompt: str = '', adapter_result: Optional[Dict[str, Any]] = None, *,
                                          task_truth: Optional[Dict[str, Any]] = None, firewall_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read-only adapter release requires an actual firewall and scoped task truth."""
    result=adapter_result if isinstance(adapter_result,dict) else {}
    truth=task_truth if isinstance(task_truth,dict) else {}
    firewall=firewall_result if isinstance(firewall_result,dict) else {}
    failures=[];warnings=[]
    adapter=str(result.get('adapter') or '')
    skill=str(truth.get('skill_id') or '')
    method=str(result.get('method') or 'GET').upper()
    response_items=result.get('responses')
    responses=[x for x in response_items[:_COMPARE_MAX_SOURCES] if isinstance(x,dict)] if isinstance(response_items,list) else []
    if result.get('ok') is not True or _parse_bool(result.get('blocked')):
        failures.append('adapter_result_not_ok')
    if method!='GET':failures.append('adapter_method_not_get')
    if not responses:failures.append('adapter_response_required')
    if any(_parse_bool(result.get(k)) for k in ('mutated','write_performed','shell_execution','file_mutation','driver_action','devbridge_apply','memory_write')):
        failures.append('adapter_claims_mutation')
    if adapter=='passported_local_get_v1':
        expected='verified_read_only_adapter_result'
        if skill!='api.local.health_check':failures.append('unexpected_skill_id')
    elif adapter=='passported_external_get_v1':
        expected='verified_external_read_only_adapter_result'
        if skill not in {'research.public_web','research.approved_api'}:failures.append('unexpected_skill_id')
    else:
        expected='adapter_result_rejected';failures.append('unknown_adapter_contract')
    allowed=set(str(x).rstrip('/') for x in (truth.get('allowed_sources') or []) if isinstance(x,str))
    if not allowed:failures.append('approved_source_scope_required')
    if str(firewall.get('verdict') or '').upper()!='ALLOW':failures.append('positive_firewall_decision_required')
    observed=set()
    for r in responses:
        uri=str(r.get('url') or '').rstrip('/')
        observed.add(uri)
        if not uri or (allowed and uri not in allowed):failures.append('observed_source_outside_task_truth')
        if str(r.get('method') or 'GET').upper()!='GET':failures.append('response_method_not_get')
        if r.get('ok') is not True:failures.append('response_not_ok')
        if any(_parse_bool(r.get(k)) for k in ('mutated','file_mutation','shell_execution')):failures.append('response_claims_mutation')
        if adapter=='passported_external_get_v1' and not uri.lower().startswith('https://'):
            failures.append('external_response_not_https')
    if truth.get('compare_required') is not True:warnings.append('compare_requirement_not_attested')
    accepted=not failures
    return {"ok":accepted,"accepted":accepted,"decision":"PASS" if accepted else "FAIL",
            "verified_answer_state":expected if accepted else 'adapter_result_rejected',
            "failures":sorted(set(failures)),"warnings":warnings,"source_count":len(observed),
            "adapter":adapter,"skill_id":skill,"execution_authority":False,"presentation_allowed":accepted,
            "doctrine":{"compare_verifies_release_not_execution":True,"read_only_get_only":True,"task_truth_required":True,"external_get_allowed_only_when_passported_and_scoped":True}}


# -----------------------------------------------------------------------------
# Peer Council comparison/scoring helper
# -----------------------------------------------------------------------------
def compare_peer_council_outputs(
    query: str,
    candidate_answer: str = "",
    peer_evidence_artifacts: Optional[List[Dict[str, Any]]] = None,
    *,
    peer_assessments: Optional[List[Dict[str, Any]]] = None,
    claim_vector: Optional[Dict[str, Any]] = None,
    provider_reliability: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Score peer/provider outputs as evidence, not authority.

    The helper is deterministic and fails closed. It relies on SML evidence
    adjudication when available, never authorizes execution, and never authorizes
    memory writes. Provider reliability is only a small weighting input; it is
    not a substitute for evidence.
    """
    artifacts = [dict(x) for x in list(peer_evidence_artifacts or [])[:_COMPARE_MAX_SOURCES] if isinstance(x, dict)]
    assessments = [dict(x) for x in list(peer_assessments or [])[:_COMPARE_MAX_SOURCES] if isinstance(x, dict)]
    reliability = dict(provider_reliability or {})
    answer = _normalize_text(candidate_answer or "")
    failures: List[str] = []
    warnings: List[str] = []

    validity = validate_response_quality(query, answer, intent="general") if answer else {
        "accepted": False,
        "decision": "REJECT",
        "issues": ["candidate_answer_missing"],
    }
    if not bool(validity.get("accepted")):
        failures.append("output_validity_failed")

    dangerous = [
        a for a in assessments
        if bool(a.get("blocked") or a.get("quarantine")) or bool(
            {"PEER_REPLY_EXECUTION_REQUEST", "PEER_REPLY_SECRET_REQUEST", "PEER_REPLY_AUTHORITY_ESCALATION"} &
            ({str(a.get("classification") or "")} | set(str(x) for x in (a.get("classifications") or [])[:16]))
        )
    ]
    if dangerous:
        failures.append("peer_reply_authority_or_security_violation")

    contradiction_notes: List[str] = []
    contents = [str(a.get("content") or a.get("answer") or a.get("summary") or "").strip().lower() for a in artifacts]
    nonempty_contents = [c for c in contents if c]
    if len(set(nonempty_contents)) > 1 and any(("not " in c or " no " in c or "false" in c) for c in nonempty_contents):
        contradiction_notes.append("peer_artifacts_contain_simple_negation_conflict")
    for assessment in assessments:
        if "PEER_REPLY_CONTRADICTION" in list(assessment.get("classifications") or []):
            contradiction_notes.append("agent_firewall_peer_assessment_reported_contradiction")
    if contradiction_notes:
        warnings.extend(contradiction_notes)

    try:
        import SarahMemorySMLProtocol as sml  # type: ignore
        court = sml.sml_build_evidence_court_packet(
            query,
            artifacts,
            context={"source": "SarahMemoryCompare.compare_peer_council_outputs", "claim_vector": claim_vector or {}},
        )
    except Exception as exc:
        court = {
            "ok": False,
            "court_2": {"verdict": "SML_EVIDENCE_COURT_UNAVAILABLE", "ranked_artifacts": [], "rejected_artifacts": [], "accepted_artifact": None},
            "error": str(exc)[:300],
            "execution_authority": False,
        }
        failures.append("sml_evidence_court_unavailable")

    court_2 = court.get("court_2") if isinstance(court.get("court_2"), dict) else {}
    ranked = [dict(x) for x in list(court_2.get("ranked_artifacts") or []) if isinstance(x, dict)]
    accepted_artifact = court_2.get("accepted_artifact") if isinstance(court_2.get("accepted_artifact"), dict) else None
    evidence_support = float((accepted_artifact or {}).get("court_score") or 0.0)
    source_authority = max([float(x.get("authority_score") or 0.0) for x in ranked] or [0.0])
    freshness = 1.0 if accepted_artifact and not bool(accepted_artifact.get("stale_for_current_claim_risk")) else 0.45 if ranked else 0.0
    unsupported_penalty = 0.0 if accepted_artifact else 0.35
    authority_penalty = 0.65 if dangerous else 0.0
    contradiction_penalty = 0.25 if contradiction_notes else 0.0
    rel_values = [float(v) for v in list(reliability.values())[:16] if isinstance(v, (int, float)) and math.isfinite(float(v))]
    reliability_score = max(0.0, min(1.0, sum(rel_values) / max(len(rel_values), 1))) if rel_values else 0.50
    output_validity = 1.0 if bool(validity.get("accepted")) else 0.0
    final_score = max(
        0.0,
        min(
            1.0,
            (output_validity * 0.22)
            + (evidence_support * 0.30)
            + (source_authority * 0.18)
            + (freshness * 0.12)
            + (reliability_score * 0.08)
            - unsupported_penalty
            - authority_penalty
            - contradiction_penalty,
        ),
    )

    if dangerous:
        decision = "REJECT_AUTHORITY_OR_SECURITY_VIOLATION"
    elif contradiction_notes:
        decision = "CONFLICT_REQUIRES_HUMAN_REVIEW"
    elif accepted_artifact and bool(validity.get("accepted")) and final_score >= 0.58:
        decision = "PASS_VERIFIED_EVIDENCE"
    elif artifacts:
        decision = "NEED_MORE_EVIDENCE"
    else:
        decision = "NO_EVIDENCE"

    accepted = decision == "PASS_VERIFIED_EVIDENCE"
    return {
        "ok": bool(accepted),
        "accepted": bool(accepted),
        "decision": decision,
        "scores": {
            "output_validity": round(output_validity, 3),
            "evidence_support": round(evidence_support, 3),
            "source_authority": round(source_authority, 3),
            "freshness": round(freshness, 3),
            "provider_reliability": round(reliability_score, 3),
            "unsupported_claim_penalty": round(unsupported_penalty, 3),
            "authority_escalation_penalty": round(authority_penalty, 3),
            "contradiction_penalty": round(contradiction_penalty, 3),
            "final": round(final_score, 3),
        },
        "winning_artifact": accepted_artifact,
        "rejected_artifacts": list(court_2.get("rejected_artifacts") or [])[:12],
        "ranked_artifacts": ranked[:12],
        "conflict_notes": contradiction_notes,
        "failures": failures,
        "warnings": warnings,
        "validity": validity,
        "sml_evidence_court": court,
        "memory_write_allowed": False,
        "execution_authority": False,
        "presentation_allowed": bool(accepted),
    }


# ====================================================================
# END OF SarahMemoryCompare.py v9.1.0a
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryCompare',
    "version": "v9.1.0a-alpha-sml-0.2",
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
    return copy.deepcopy(SML_ORGAN_METADATA)


def sml_health():
    return {'status':'Available' if callable(compare_reply) else 'Degraded',
            'availability':1.0 if callable(compare_reply) else 0.0,'integrity':None,'performance':None,
            'reliability':None,'confidence':None,'latency_ms':None,'stability':None,
            'compatibility':None,'notes':['Source module available; runtime effectiveness and live governance not measured']}


def sml_diagnostics():
    return {'status':'AVAILABLE','component':'SarahMemoryCompare','sml_adapter':True,
            'metadata':sml_get_metadata(),'health':sml_health(),'execution_authority':False}


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
    """Observation only unless a decision receipt can be re-checked in this call.

    A caller's confidence float never supplies a Compare verdict or approval.
    """
    from SarahMemorySMLProtocol import SMLPacket, sml_touch_packet
    pkt=packet if isinstance(packet,SMLPacket) else SMLPacket.from_dict(packet)
    return sml_touch_packet(pkt,organ='SarahMemoryCompare',action='observe',omega='Ω020',
                            note='Compare verification requires independent checked receipt; '+str(note)[:120],updates=None)
# --- SML COMPARE SPECIALIZATION END ---

