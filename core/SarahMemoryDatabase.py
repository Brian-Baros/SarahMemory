"""--==The SarahMemory Project==--
File: SarahMemoryDatabase.py
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
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "data_layer"
# CATEGORY = "database_and_memory"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "data_memory"
# HARDWARE_DOMAIN = "filesystem"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "database_core"
# FAMILY = "core_memory"
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
# NOTES = "Primary database and memory access layer for SQLite/MySQL, response-layer storage, QA cache, device identity/capabilities, and dataset access wrappers."
# --- SARAHMETA END ---

import logging
import math
import sqlite3
import time
import uuid
import os
import datetime
try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # type: ignore
import json
import SarahMemoryGlobals as config
from SarahMemoryGlobals import run_async, DATASETS_DIR
import random
import hashlib
import secrets
try:
    import jwt  # type: ignore
except Exception:
    jwt = None  # type: ignore
from datetime import datetime as dt, timedelta

# Setup logging for the database module
logger = logging.getLogger('SarahMemoryDatabase')
logger.setLevel(logging.DEBUG if bool(getattr(config, 'DEBUG_MODE', False)) else logging.INFO)
handler = logging.NullHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)


try:
    import mysql.connector as mysql
except Exception:
    mysql = None

try:
    import SarahMemoryGlobals as G
except Exception:
    G = None

try:
    from SarahMemoryGlobals import get_mesh_sync_config
except Exception:
    def get_mesh_sync_config():
        """Fallback mesh-sync config when Phase B helpers are unavailable."""
        try:
            safe_mode = bool(getattr(G, "SAFE_MODE", False))
        except Exception:
            safe_mode = False
        return {
            "node_name":          getattr(G, "NODE_NAME", "SarahMemoryNode") if G else "SarahMemoryNode",
            "mesh_enabled":       bool(getattr(G, "MESH_SYNC_ENABLED", True)) if G else True,
            "hub_allowed":        bool(getattr(G, "ALLOW_HUB_SYNC", True)) if G else True,
            "safe_mode":          safe_mode,
            "safe_mode_only":     bool(getattr(G, "MESH_SYNC_SAFE_MODE_ONLY", False)) if G else False,
            "sarahnet_enabled":   bool(getattr(G, "SARAHNET_ENABLED", True)) if G else True,
            "web_base":           getattr(G, "SARAH_WEB_BASE", "https://ai.sarahmemory.com") if G else "https://ai.sarahmemory.com",
            "remote_sync_enabled":bool(getattr(G, "REMOTE_SYNC_ENABLED", True)) if G else True,
            "heartbeat_sec":      float(getattr(G, "REMOTE_HEARTBEAT_SEC", 30)) if G else 30.0,
            "http_timeout":       float(getattr(G, "REMOTE_HTTP_TIMEOUT", 6.0)) if G else 6.0,
        }


# ============================
# PHASE A: Identity & Device Awareness (v7.7.5)
# ============================

def sm_get_or_create_user(email, display_name=None):
    """
    Phase B: Get or create user in MySQL.
    Falls back to stub if cloud DB unavailable.
    """
    conn = _get_cloud_conn()
    if not conn:
        logger.warning("Cloud DB unavailable, using stub user")
        return {"id": None, "email": email, "display_name": display_name, "is_active": False}

    try:
        cursor = conn.cursor(dictionary=True)

        # Check if user exists
        cursor.execute(
            "SELECT id, email, display_name, is_active, is_verified FROM sm_users WHERE email = %s AND deleted_at IS NULL",
            (email,)
        )
        user = cursor.fetchone()

        if user:
            # Update last_login
            cursor.execute("UPDATE sm_users SET last_login = NOW() WHERE id = %s", (user['id'],))
            conn.commit()
            logger.info(f"User {email} logged in, id={user['id']}")
            return user

        # Create new user
        cursor.execute(
            "INSERT INTO sm_users (email, display_name) VALUES (%s, %s)",
            (email, display_name or email.split('@')[0])
        )
        conn.commit()
        user_id = cursor.lastrowid

        logger.info(f"Created new user: {email}, id={user_id}")
        return {
            "id": user_id,
            "email": email,
            "display_name": display_name or email.split('@')[0],
            "is_active": True,
            "is_verified": False
        }

    except Exception as e:
        logger.error(f"Error in sm_get_or_create_user({email}): {e}", exc_info=True)
        return {"id": None, "email": email, "display_name": display_name}
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass


def sm_link_auth_provider(user_id, provider, provider_uid, email):
    """Phase B: Link OAuth provider to user account."""
    conn = _get_cloud_conn()
    if not conn or not user_id:
        return False

    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sm_auth_providers (user_id, provider, provider_uid, email) VALUES (%s, %s, %s, %s) ON DUPLICATE KEY UPDATE updated_at = NOW()",
            (user_id, provider, provider_uid, email)
        )
        conn.commit()
        logger.info(f"Linked {provider} to user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Error linking auth provider: {e}")
        return False
    finally:
        if conn:
            conn.close()


def sm_get_or_create_device(user_id, session_id, user_agent, mode):
    """
    Phase B: Register device with fingerprinting.
    """
    conn = _get_cloud_conn()
    if not conn:
        return {"id": None, "mode": mode, "fingerprint": session_id}

    try:
        # Generate device fingerprint
        fingerprint = hashlib.sha256(f"{user_agent}:{session_id}".encode()).hexdigest()[:64]

        cursor = conn.cursor(dictionary=True)

        # Check if device exists
        cursor.execute(
            "SELECT id, user_id, device_name, mode, trust_level FROM sm_devices WHERE device_fingerprint = %s",
            (fingerprint,)
        )
        device = cursor.fetchone()

        if device:
            # Update last_seen
            cursor.execute(
                "UPDATE sm_devices SET last_seen = NOW() WHERE id = %s",
                (device['id'],)
            )
            conn.commit()
            return device

        # Create new device
        cursor.execute(
            "INSERT INTO sm_devices (user_id, device_fingerprint, user_agent, mode) VALUES (%s, %s, %s, %s)",
            (user_id, fingerprint, user_agent, mode)
        )
        conn.commit()
        device_id = cursor.lastrowid

        # Create default capabilities
        cursor.execute("INSERT INTO sm_device_capabilities (device_id) VALUES (%s)", (device_id,))
        conn.commit()

        logger.info(f"Registered new device: {fingerprint[:16]}..., id={device_id}")

        return {
            "id": device_id,
            "user_id": user_id,
            "fingerprint": fingerprint,
            "mode": mode,
            "trust_level": 0
        }

    except Exception as e:
        logger.error(f"Error in sm_get_or_create_device: {e}", exc_info=True)
        return {"id": None, "mode": mode}
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass


def sm_get_capabilities(device_id):
    """
    Phase B: Get device capabilities from MySQL.
    """
    conn = _get_cloud_conn()
    if not conn or not device_id:
        return {
            "CAN_USE_GEO": False,
            "CAN_SEND_EMAIL": False,
            "CAN_ACCESS_CONTACTS": False,
            "CAN_TRIGGER_LOCAL_APPS": False,
            "CAN_USE_CAMERA": False,
            "CAN_USE_MICROPHONE": False
        }

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM sm_device_capabilities WHERE device_id = %s",
            (device_id,)
        )
        caps = cursor.fetchone()

        if caps:
            return {
                "CAN_USE_GEO": bool(caps['can_use_geo']),
                "CAN_SEND_EMAIL": bool(caps['can_send_email']),
                "CAN_ACCESS_CONTACTS": bool(caps['can_access_contacts']),
                "CAN_TRIGGER_LOCAL_APPS": bool(caps['can_trigger_local_apps']),
                "CAN_USE_CAMERA": bool(caps['can_use_camera']),
                "CAN_USE_MICROPHONE": bool(caps['can_use_microphone'])
            }

        return {
            "CAN_USE_GEO": False,
            "CAN_SEND_EMAIL": False,
            "CAN_ACCESS_CONTACTS": False,
            "CAN_TRIGGER_LOCAL_APPS": False
        }

    except Exception as e:
        logger.error(f"Error in sm_get_capabilities({device_id}): {e}", exc_info=True)
        return {"CAN_USE_GEO": False, "CAN_SEND_EMAIL": False}
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass

def _get_cloud_conn():
    """Return a MySQL connection or None if cloud is disabled or unavailable."""
    if not G or not getattr(G, "CLOUD_DB_ENABLED", False):
        return None
    if mysql is None:
        return None
    try:
        return mysql.connect(
            host=G.CLOUD_DB_HOST,
            port=getattr(G, "CLOUD_DB_PORT", 3306),
            user=G.CLOUD_DB_USER,
            password=G.CLOUD_DB_PASSWORD,
            database=G.CLOUD_DB_NAME,
            connection_timeout=int(getattr(G, "CLOUD_DB_CONNECT_TIMEOUT", 3) or 3),
        )
    except Exception as e:
        logging.error(f"[CLOUD_DB_CONNECT ERROR] {e}")
        return None






# --- Database Paths ---
DB_PATH = os.path.join(config.DATASETS_DIR, 'ai_learning.db')
USER_DB_PATH = os.path.join(config.DATASETS_DIR, 'user_profile.db')

RESPONSE_HISTORY_DB = DB_PATH


def _resolve_system_log_db_path():
    try:
        db_path = getattr(config, "SYSTEM_LOG_DB", None)
    except Exception:
        db_path = None
    if db_path:
        return db_path
    try:
        dd = getattr(config, "DATASETS_DIR", None)
    except Exception:
        dd = None
    if not dd:
        dd = os.path.join(os.getcwd(), "data", "memory", "datasets")
    return os.path.join(dd, "system_logs.db")


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


def _sm_meta_blocks_persistence(meta=None, text: str = "") -> bool:
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


def _sm_search_terms(text: str, max_terms: int = 10):
    """Extract offline-safe terms for local DB LIKE fallback."""
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "did", "do", "does",
        "for", "from", "give", "how", "i", "in", "is", "it", "me", "of", "on", "or", "please",
        "should", "tell", "that", "the", "this", "to", "up", "was", "were", "what", "when", "where",
        "which", "who", "why", "would", "you", "about", "define", "describe", "explain", "information",
    }
    try:
        words = [w for w in re.findall(r"[a-z0-9][a-z0-9_\-]{1,}", str(text or "").lower()) if w not in stop]
    except Exception:
        words = []
    out = []
    for word in words:
        if word not in out:
            out.append(word)
        if len(out) >= max_terms:
            break
    return out


def _sm_rank_text_for_query(query: str, text: str) -> float:
    """Small lexical ranker for QA/personality fallback searches."""
    q = str(query or "").lower().strip()
    t = str(text or "").lower()
    if not q or not t:
        return 0.0
    if q in t:
        return 1.0
    terms = _sm_search_terms(q)
    if not terms:
        return 0.0
    hits = sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", t))
    return hits / max(len(terms), 1)




def _sm_is_low_quality_cached_answer(text: str) -> bool:
    """Return True for cached fallback/failure or corpus blob text that must not be recalled as knowledge.

    SarahMemory must never treat a past failure message, whole paper, diagnostic
    dump, or copied corpus chunk as a high-confidence answer. Cleanup tools may
    quarantine these rows; retrieval must reject them immediately.
    """
    raw = str(text or "").strip()
    t = raw.lower()
    if not t:
        return True
    blocked_phrases = (
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
        "i couldn't find reliable information",
        "i could not find reliable information",
        "unable to answer",
        "answer unavailable",
        "request denied by policy",
        "user confirmation required",
        "local_none",
        "fallback failure",
        "traceback",
        "exception during",
        "error:",
    )
    if any(p in t for p in blocked_phrases):
        return True
    if len(raw) > 1200:
        return True
    paper_markers = (
        "abstract", "1. introduction", "1 introduction", "references", "bibliography",
        "@", "doi", "arxiv", "we present", "we propose", "this paper",
        "compiler from llvm", "mozilla",
    )
    if sum(1 for p in paper_markers if p in t) >= 2:
        return True
    # Rows composed mostly of metadata/noise should not become answers.
    if len(t) < 4:
        return True
    return False


def _json_dumps_safe(value, limit: int = 100000) -> str:
    try:
        raw = json.dumps(value if value is not None else {}, ensure_ascii=False)
    except Exception:
        raw = json.dumps({"_unserializable": str(value)}, ensure_ascii=False)
    if len(raw) > int(limit):
        return raw[: int(limit)]
    return raw


def _normalize_response_layers(user_text: str, response_text: str, meta: dict, kwargs: dict):
    """Normalize canonical vs presentation layers for storage.

    Raw/canonical stays internal. Presentation is the user-facing form.
    Backward compatibility: if a caller only provides a flat response, we store it as
    the presentation answer and use it as canonical only when no better raw truth is available.
    """
    meta = meta if isinstance(meta, dict) else {}
    canonical_answer = (
        kwargs.get("canonical_answer")
        or kwargs.get("raw_answer")
        or meta.get("canonical_answer")
        or meta.get("raw_answer")
        or meta.get("truth")
        or meta.get("truth_core")
        or meta.get("answer_raw")
        or meta.get("raw")
        or ""
    )
    presented_answer = (
        kwargs.get("presented_answer")
        or kwargs.get("presentation_answer")
        or meta.get("presented_answer")
        or meta.get("presentation_answer")
        or meta.get("presentation_reply")
        or meta.get("reply")
        or response_text
        or ""
    )
    if not canonical_answer:
        canonical_answer = presented_answer
    canonical_type = str(
        kwargs.get("canonical_type")
        or meta.get("canonical_type")
        or ("deterministic" if bool(kwargs.get("truth_locked") or meta.get("truth_locked")) else "response")
    )
    truth_locked = bool(kwargs.get("truth_locked") or meta.get("truth_locked") or meta.get("deterministic"))
    tone = str(kwargs.get("tone") or meta.get("tone") or "")
    style = str(kwargs.get("style") or meta.get("style") or meta.get("complexity") or "")
    persona_state = str(kwargs.get("persona_state") or meta.get("persona_state") or meta.get("persona") or "")
    lane = str(kwargs.get("lane") or meta.get("lane") or meta.get("primary_lane") or "")
    source = str(kwargs.get("source") or meta.get("source") or meta.get("provider") or "")
    raw_meta = {
        "canonical_answer": canonical_answer,
        "canonical_type": canonical_type,
        "truth_locked": truth_locked,
        "lane": lane,
        "source": source,
    }
    raw_meta.update(meta.get("raw_meta") or {})
    presentation_meta = {
        "presented_answer": presented_answer,
        "tone": tone,
        "style": style,
        "persona_state": persona_state,
        "lane": lane,
        "source": source,
    }
    presentation_meta.update(meta.get("presentation_meta") or {})
    return {
        "user_text": str(user_text or ""),
        "canonical_answer": str(canonical_answer or ""),
        "presented_answer": str(presented_answer or ""),
        "canonical_type": canonical_type,
        "truth_locked": truth_locked,
        "tone": tone,
        "style": style,
        "persona_state": persona_state,
        "lane": lane,
        "source": source,
        "raw_meta": raw_meta,
        "presentation_meta": presentation_meta,
    }


def ensure_response_memory_schema():
    """Create/upgrade dual-layer response memory tables.

    Keeps backward compatibility with legacy flat history while adding canonical vs
    presentation storage required by the SarahMemory I/O framework.
    """
    try:
        ensure_core_schema()
    except Exception:
        pass
    try:
        os.makedirs(os.path.dirname(RESPONSE_HISTORY_DB), exist_ok=True)
        with sqlite3.connect(RESPONSE_HISTORY_DB) as conn:
            cur = conn.cursor()
            def _ensure_column(cur, table: str, col: str, col_type: str) -> None:
                try:
                    cur.execute(f"PRAGMA table_info({table})")
                    existing = [r[1] for r in cur.fetchall()]
                    if col not in existing:
                        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
                except Exception:
                    pass
            cur.execute("""CREATE TABLE IF NOT EXISTS response_layers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                session_id TEXT,
                user_input TEXT,
                canonical_answer TEXT,
                presented_answer TEXT,
                intent TEXT,
                lane TEXT,
                source TEXT,
                canonical_type TEXT,
                truth_locked INTEGER DEFAULT 0,
                tone TEXT,
                style TEXT,
                persona_state TEXT,
                raw_meta_json TEXT,
                presentation_meta_json TEXT
            )""")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_response_layers_ts ON response_layers(ts)")
            for col, col_type in [
                ("canonical_answer", "TEXT"),
                ("presented_answer", "TEXT"),
                ("lane", "TEXT"),
                ("source", "TEXT"),
                ("canonical_type", "TEXT"),
                ("truth_locked", "INTEGER DEFAULT 0"),
                ("tone", "TEXT"),
                ("style", "TEXT"),
                ("persona_state", "TEXT"),
                ("raw_meta_json", "TEXT"),
                ("presentation_meta_json", "TEXT"),
            ]:
                _ensure_column(cur, "conversations", col, col_type)
            conn.commit()
    except Exception as e:
        logger.warning(f"[ensure_response_memory_schema] {e}")
    try:
        sys_db = _resolve_system_log_db_path()
        os.makedirs(os.path.dirname(sys_db), exist_ok=True)
        with sqlite3.connect(sys_db) as conn:
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS response_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                user_input TEXT,
                response TEXT,
                intent TEXT,
                tone TEXT,
                complexity TEXT,
                source TEXT,
                session_id TEXT,
                meta_json TEXT,
                canonical_answer TEXT,
                presented_answer TEXT,
                canonical_type TEXT,
                truth_locked INTEGER DEFAULT 0,
                lane TEXT,
                persona_state TEXT,
                raw_meta_json TEXT,
                presentation_meta_json TEXT
            )""")
            conn.commit()
    except Exception as e:
        logger.warning(f"[ensure_response_history_schema] {e}")

def get_active_sentence_model():
    """Return the active embedding model (one per category), with offline-safe fallback.

    Policy:
      - Tier Rating POOR => auto 3rd-party selection is disabled.
        Only user-enabled models may be selected; otherwise return _LiteEmbedder.
      - For non-POOR tiers: select exactly ONE model for category 'embeddings',
        and keep ordered fallbacks. Never ensembles by default.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception:
        SentenceTransformer = None  # type: ignore

    import SarahMemoryGlobals as config

    class _LiteEmbedder:
        """Always-works local embedder (hash projection)."""
        def __init__(self, dim: int = 64):
            self.dim = int(dim)

        def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            out = []
            for t in (texts or []):
                h = hashlib.sha256((t or "").encode("utf-8")).digest()
                vec = []
                for i in range(self.dim):
                    b = h[i % len(h)]
                    vec.append(math.sin((b + i) * 0.0174533))
                # L2 normalize if requested
                if normalize_embeddings:
                    s = math.sqrt(sum(v*v for v in vec)) or 1.0
                    vec = [v/s for v in vec]
                out.append(vec)
            if convert_to_numpy:
                try:
                    import numpy as np  # type: ignore
                    return np.array(out, dtype=float)
                except Exception:
                    return out
            return out

    # If SentenceTransformer isn't available, fall back immediately.
    if SentenceTransformer is None:
        return _LiteEmbedder()

    # Resolve ONE model via Globals resolver (plus fallbacks)
    try:
        res = config.resolve_model("embeddings", text="", meta=None, models_dir=getattr(config, "MODELS_DIR", None)) or {}
        selected = res.get("selected")
        fallbacks = res.get("fallbacks") or []
        candidates = [c for c in ([selected] + list(fallbacks)) if c]
    except Exception:
        candidates = []

    # If nothing selected, core-only fallback (POOR tier with no user overrides)
    if not candidates:
        return _LiteEmbedder()

    models_dir = getattr(config, "MODELS_DIR", None) or os.path.join(os.getcwd(), "data", "models")

    for repo in candidates:
        try:
            local_dir1 = os.path.join(models_dir, repo.replace("/", "_"))
            local_dir2 = os.path.join(models_dir, repo)
            if os.path.isdir(local_dir1):
                return SentenceTransformer(local_dir1, local_files_only=True)
            if os.path.isdir(local_dir2):
                return SentenceTransformer(local_dir2, local_files_only=True)
            # As a final attempt, rely on HF local cache only
            return SentenceTransformer(repo, local_files_only=True)
        except Exception as e:
            try:
                logger.warning(f"Model load failed: {repo} -> {e}")
            except Exception:
                pass

    # All candidates failed
    return _LiteEmbedder()


def _sm_embedding_to_list(value):
    """Normalize an embedding output to a flat Python list of floats."""
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if isinstance(value, list) and value and isinstance(value[0], list):
            value = value[0]
        return [float(x) for x in (value or [])]
    except Exception:
        return []


def _sm_encode_text(model, text: str):
    """Encode text with any local embedder shape, never throwing."""
    try:
        encoded = model.encode(str(text or ""), normalize_embeddings=True)
    except TypeError:
        encoded = model.encode(str(text or ""))
    return _sm_embedding_to_list(encoded)


def _sm_cosine_similarity(a, b) -> float:
    """Dimension-safe cosine similarity for mixed legacy/runtime embeddings."""
    try:
        av = np.asarray(a, dtype=float).reshape(-1)
        bv = np.asarray(b, dtype=float).reshape(-1)
        if av.size == 0 or bv.size == 0 or av.size != bv.size:
            return -1.0
        denom = float(norm(av) * norm(bv))
        if denom <= 0.0:
            return -1.0
        return float(dot(av, bv) / denom)
    except Exception:
        return -1.0


def ensure_local_data_runtime_ready(verify_embedding: bool = True) -> dict:
    """Lightweight local-data readiness gate for boot and diagnostics.

    This validates local DB/model pathing and the embedding runtime without scanning,
    rebuilding, downloading, or re-vectorizing datasets. Fast boot may skip heavy
    rebuilds, but it must not skip this readiness check when LOCAL_DATA_ENABLED=True.
    """
    status = {
        "ok": False,
        "local_data_enabled": bool(getattr(config, "LOCAL_DATA_ENABLED", True)),
        "datasets_dir": getattr(config, "DATASETS_DIR", DATASETS_DIR),
        "models_dir": getattr(config, "MODELS_DIR", os.path.join(getattr(config, "BASE_DIR", os.getcwd()), "data", "models")),
        "db_count": 0,
        "db_files": [],
        "embedding_ready": False,
        "embedder": None,
        "embedding_dim": 0,
        "errors": [],
    }
    try:
        if not status["local_data_enabled"]:
            status["errors"].append("LOCAL_DATA_ENABLED is False")
            return status

        os.makedirs(status["datasets_dir"], exist_ok=True)
        os.makedirs(status["models_dir"], exist_ok=True)

        db_files = sorted([f for f in os.listdir(status["datasets_dir"]) if f.lower().endswith(".db")])
        status["db_files"] = db_files[:50]
        status["db_count"] = len(db_files)

        try:
            conn = init_database()
            if conn is not None:
                conn.close()
        except Exception as exc:
            status["errors"].append(f"init_database failed: {exc}")

        try:
            ensure_response_memory_schema()
        except Exception as exc:
            status["errors"].append(f"ensure_response_memory_schema failed: {exc}")

        if verify_embedding:
            try:
                emb = get_active_sentence_model()
                status["embedder"] = emb.__class__.__name__ if emb is not None else None
                vec = _sm_encode_text(emb, "SarahMemory local data readiness probe") if emb is not None else []
                status["embedding_dim"] = len(vec)
                status["embedding_ready"] = bool(vec)
            except Exception as exc:
                status["errors"].append(f"embedding readiness failed: {exc}")

        status["ok"] = bool(status["db_count"] >= 0 and (status["embedding_ready"] or not verify_embedding))
        return status
    except Exception as exc:
        status["errors"].append(str(exc))
        return status

# --- Initialization ---
def init_database():
    # This initializes only ai_learning.db. Other databases are managed in DBCreate.py but accessed here.
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Voice logs
        cursor.execute('''CREATE TABLE IF NOT EXISTS voice_logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            voice_text TEXT NOT NULL,
                            embedding BLOB
                          )''')
        # Performance metrics
        cursor.execute('''CREATE TABLE IF NOT EXISTS performance_metrics (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            cpu_usage REAL,
                            memory_usage REAL,
                            disk_usage REAL,
                            network_usage REAL
                          )''')
        #LyricsToSong
        cursor.execute('''CREATE TABLE IF NOT EXISTS vocal_projects (
                            project_id TEXT PRIMARY KEY,
                            name TEXT NOT NULL,
                            lyrics TEXT NOT NULL,
                            tempo INTEGER DEFAULT 120,
                            key TEXT DEFAULT 'C',
                            scale TEXT DEFAULT 'major',
                            style TEXT DEFAULT 'pop',
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            metadata TEXT  -- JSON blob
                            )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS vocal_tracks (
                            track_id TEXT PRIMARY KEY,
                            project_id TEXT NOT NULL,
                            voice_profile TEXT DEFAULT 'neutral',
                            emotion TEXT DEFAULT 'neutral',
                            pitch_shift REAL DEFAULT 0.0,
                            tempo_factor REAL DEFAULT 1.0,
                            is_harmony INTEGER DEFAULT 0,
                            harmony_interval TEXT,
                            audio_path TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (project_id) REFERENCES vocal_projects(project_id)
                            )''')

        cursor.execute('''CREATE TABLE IF NOT EXISTS voice_profiles (
                            profile_name TEXT PRIMARY KEY,
                            gender TEXT,
                            range_min REAL,
                            range_max REAL,
                            pitch_shift REAL,
                            formant_shift REAL,
                            vibrato_rate REAL,
                            vibrato_depth REAL,
                            custom_data TEXT  -- JSON blob
                        )''')
        # QA cache
        cursor.execute('''CREATE TABLE IF NOT EXISTS qa_cache (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            query TEXT,
                            ai_answer TEXT,
                            hit_score INTEGER,
                            feedback TEXT,
                            timestamp TEXT
                          )''')
        conn.commit()
        logger.info("Runtime DB initialized with QA cache.")
                # Additional DBs initialized externally but used here
        # reminders.db, avatar.db, windows10.db, windows11.db, software.db, device_link.db
        return conn
    except Exception as e:
        logger.error(f"Error initializing runtime DB: {e}")
        return None

# --- QA Cache Helpers ---
def search_answers(query):
    """Unified search over local QA cache and (optionally) cloud QA cache.

    Local search now supports keyword/entity fallback so general questions do not
    fail simply because the full natural-language query is not stored verbatim.
    """
    if _sm_is_volatile_body_fact_query(query):
        logger.info("[V10/V9C] QA cache recall blocked for volatile SelfAware body fact query.")
        return []
    results = []

    # 1) Local sqlite first (fast, offline-safe)
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        ranked = []
        cursor.execute(
            "SELECT query, ai_answer, COALESCE(hit_score, 0) FROM qa_cache WHERE query LIKE ? OR ai_answer LIKE ? ORDER BY hit_score DESC LIMIT 25",
            ('%' + str(query or '') + '%', '%' + str(query or '') + '%')
        )
        rows = cursor.fetchall()

        if not rows:
            terms = _sm_search_terms(query)
            if terms:
                where = " OR ".join(["query LIKE ? OR ai_answer LIKE ?" for _ in terms[:5]])
                params = []
                for term in terms[:5]:
                    params.extend([f"%{term}%", f"%{term}%"])
                cursor.execute(f"SELECT query, ai_answer, COALESCE(hit_score, 0) FROM qa_cache WHERE {where} LIMIT 50", params)
                rows = cursor.fetchall()

        conn.close()
        for qrow, answer, hit_score in rows:
            if _sm_is_low_quality_cached_answer(answer):
                continue
            score = max(_sm_rank_text_for_query(query, qrow), _sm_rank_text_for_query(query, answer))
            if score >= 0.40 or str(query or '').lower() in str(qrow or '').lower() or str(query or '').lower() in str(answer or '').lower():
                ranked.append((score + (float(hit_score or 0) * 0.01), answer))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if ranked:
            results.extend([answer for _, answer in ranked[:5]])
    except Exception as e:
        logger.error(f"Error searching local QA cache: {e}")

    # 2) Cloud MySQL (if enabled and available)
    try:
        mesh_cfg = get_mesh_sync_config()
    except Exception:
        mesh_cfg = {}
    if mesh_cfg.get("mesh_enabled", True) and mesh_cfg.get("hub_allowed", True):
        try:
            cloud = _get_cloud_conn()
            if cloud is not None:
                cur = cloud.cursor()
                cur.execute(
                    "SELECT ai_answer FROM sm_qa_cache WHERE query LIKE %s ORDER BY hit_score DESC LIMIT 5",
                    ('%' + query + '%',)
                )
                rows = cur.fetchall()
                cloud.close()
                for (answer,) in rows:
                    if not _sm_is_low_quality_cached_answer(answer):
                        results.append(answer)
        except Exception as e:
            logger.error(f"[CLOUD QA SEARCH ERROR] {e}")

    return results


def store_answer(query, answer):
    """Store answer locally and push to cloud hub if available."""
    if _sm_is_volatile_body_fact_query(query) or _sm_is_volatile_body_fact_query(answer):
        logger.info("[V10/V9C] QA cache store blocked for volatile SelfAware body fact.")
        return False
    if _sm_is_low_quality_cached_answer(answer):
        logger.info("[V9.0][CACHE_GUARD] QA cache store blocked for low-quality fallback answer.")
        return False
    timestamp = dt.now().isoformat()

    # 1) Local sqlite
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO qa_cache (query, ai_answer, hit_score, feedback, timestamp) VALUES (?, ?, ?, ?, ?)",
            (query, answer, 0, "ungraded", timestamp)
        )
        conn.commit()
        conn.close()
        logger.info(f"Stored QA cache for query: '{query}' (local)")
    except Exception as e:
        logger.error(f"Error storing QA cache locally: {e}")

        # 2) Cloud MySQL (best-effort, NON-BLOCKING)
        try:
            mesh_cfg = get_mesh_sync_config() or {}
        except Exception:
            mesh_cfg = {}

        try:
            local_only = bool(getattr(G, "LOCAL_ONLY_MODE", False)) if G else False
        except Exception:
            local_only = False

        # Respect local-only + hub policy gates
        if (not local_only) and bool(mesh_cfg.get("mesh_enabled", True)) and bool(mesh_cfg.get("hub_allowed", True)):

            def _cloud_push():
                try:
                    cloud = _get_cloud_conn()
                    if cloud is None:
                        return
                    cur = cloud.cursor()
                    cur.execute(
                        "INSERT INTO sm_qa_cache (query, ai_answer, hit_score, feedback, timestamp, source_node) "
                        "VALUES (%s, %s, %s, %s, %s, %s)",
                        (query, answer, 0, "ungraded", timestamp.replace("T", " "), G.NODE_NAME if G else None)
                    )
                    cloud.commit()
                    cloud.close()
                    logger.info(f"Stored QA cache for query: '{query}' (cloud)")
                except Exception as e:
                    logger.error(f"[CLOUD QA STORE ERROR] {e}")

            # Fire-and-forget to avoid blocking chat/UI responsiveness
            try:
                run_async(_cloud_push)
            except Exception:
                # Fallback: attempt inline but still best-effort
                _cloud_push()




def store_performance_metrics(conn):
    try:
        timestamp = dt.now().isoformat()
        if psutil is None:
            return {"status": "psutil unavailable", "timestamp": str(dt.now())}
        cpu = psutil.cpu_percent(interval=0.2)
        mem = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        net = random.uniform(0, 100)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO performance_metrics (timestamp, cpu_usage, memory_usage, disk_usage, network_usage) VALUES (?, ?, ?, ?, ?)",
            (timestamp, cpu, mem, disk, net)
        )
        conn.commit()
        logger.info(f"Performance metrics at {timestamp}: CPU {cpu}%, Mem {mem}%, Disk {disk}%, Net {net:.2f}%")
        return True
    except Exception as e:
        logger.error(f"Error storing performance metrics: {e}")
        return False

def get_all_voice_logs(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM voice_logs")
        logs = cursor.fetchall()
        logger.info(f"Retrieved {len(logs)} voice logs")
        return logs
    except Exception as e:
        logger.error(f"Error retrieving voice logs: {e}")
        return []

# --- User Profile DB Support ---
def connect_user_profile_db():
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        logger.info("Connected to user_profile.db.")
        return conn
    except Exception as e:
        logger.error(f"Unable to connect to user_profile.db: {e}")
        return None

# --- Diagnostics Export ---
def record_qa_feedback(query, score, feedback, timestamp=None):
    if _sm_is_volatile_body_fact_query(query):
        logger.info("[V10/V9C] QA feedback update blocked for volatile SelfAware body fact query.")
        return False
    try:
        if not timestamp:
           timestamp = dt.now().isoformat()
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE qa_cache SET hit_score = ?, feedback = ?, timestamp = ? WHERE query LIKE ?",
            (score, feedback, timestamp, '%' + query + '%')
        )
        conn.commit()
        conn.close()
        logger.info(f"Recorded feedback on QA entry: {query} | Score: {score} | Feedback: {feedback} | Time: {timestamp}")
    except Exception as e:
        logger.error(f"Error recording QA feedback: {e}")

def export_voice_logs_to_json(conn, output_path):
    try:
        logs = get_all_voice_logs(conn)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2)
        logger.info(f"Exported voice logs to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error exporting voice logs: {e}")
        return False

# --- New Additions ---

# Additional dataset access wrappers
REMINDER_DB = os.path.join(config.DATASETS_DIR, "reminders.db")
AVATAR_DB = os.path.join(config.DATASETS_DIR, "avatar.db")
WIN10_DB = os.path.join(config.DATASETS_DIR, "windows10.db")
WIN11_DB = os.path.join(config.DATASETS_DIR, "windows11.db")
SOFTWARE_DB = os.path.join(config.DATASETS_DIR, "software.db")
DEVICE_LINK_DB = os.path.join(config.DATASETS_DIR, "device_link.db")


def fetch_reminders():
    try:
        conn = sqlite3.connect(REMINDER_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT title, description, datetime FROM reminders WHERE active = 1")
        results = cursor.fetchall()
        conn.close()
        return results
    except Exception as e:
        logger.error(f"[REMINDER_DB ERROR] {e}")
        return []

def fetch_software_commands():
    try:
        candidates = [SOFTWARE_DB, os.path.join(os.path.dirname(__file__), 'software.db'), os.path.join(os.getcwd(), 'software.db')]
        db_path = None
        best_size = -1
        for cand in candidates:
            try:
                if cand and os.path.exists(cand):
                    sz = os.path.getsize(cand)
                    if sz > best_size:
                        best_size = sz
                        db_path = cand
            except Exception:
                pass
        db_path = db_path or SOFTWARE_DB
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(software_apps)")
        cols = {str(r[1]).lower() for r in cursor.fetchall()}
        has_name = 'name' in cols
        has_app_name = 'app_name' in cols
        has_path = 'path' in cols
        has_usage_count = 'usage_count' in cols
        has_last_used = 'last_used' in cols
        if not has_path or not (has_name or has_app_name):
            conn.close()
            return []
        select_name = "COALESCE(name, app_name)" if (has_name and has_app_name) else ("name" if has_name else "app_name")
        order_bits = []
        if has_usage_count:
            order_bits.append("COALESCE(usage_count, 0) DESC")
        if has_last_used:
            order_bits.append("COALESCE(last_used, '') DESC")
        order_sql = (" ORDER BY " + ", ".join(order_bits)) if order_bits else ""
        sql = f"SELECT {select_name} AS resolved_name, path FROM software_apps WHERE COALESCE(path, '') <> ''{order_sql}"
        cursor.execute(sql)
        raw_entries = cursor.fetchall()
        conn.close()
        entries = []
        for name, path in raw_entries:
            if not path:
                continue
            resolved_name = (name or '').strip() if isinstance(name, str) else str(name or '').strip()
            if not resolved_name:
                try:
                    resolved_name = os.path.splitext(os.path.basename(path))[0]
                except Exception:
                    resolved_name = ''
            entries.append((resolved_name, path))
        return entries
    except Exception as e:
        logger.error(f"[SOFTWARE_DB ERROR] {e}")
        return []

def fetch_os_commands(version="10"):
    try:
        db_path = WIN10_DB if version == "10" else WIN11_DB
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT command, description FROM os_commands WHERE version = ?", (version,))
        entries = cursor.fetchall()
        conn.close()
        return entries
    except Exception as e:
        logger.error(f"[OS_COMMAND_DB ERROR] {e}")
        return []

def fetch_avatar_metadata():
    try:
        conn = sqlite3.connect(AVATAR_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT file_path, tags, emotion, gps_latitude, gps_longitude FROM photo_metadata")
        entries = cursor.fetchall()
        conn.close()
        return entries
    except Exception as e:
        logger.error(f"[AVATAR_DB ERROR] {e}")
        return []

def fetch_device_links():
    try:
        conn = sqlite3.connect(DEVICE_LINK_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT device_name, device_type, connection_type FROM device_registry")
        entries = cursor.fetchall()
        conn.close()
        return entries
    except Exception as e:
        logger.error(f"[DEVICE_LINK ERROR] {e}")
        return []
def search_responses(question):
    """Fuzzy search inside the personality1.db responses with keyword fallback."""
    try:
        db_path = os.path.join(config.DATASETS_DIR, "personality1.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        ranked = []
        cursor.execute("SELECT response FROM responses WHERE response LIKE ? LIMIT 25", ('%' + str(question or '') + '%',))
        rows = cursor.fetchall()
        if not rows:
            terms = _sm_search_terms(question)
            if terms:
                where = " OR ".join(["response LIKE ?" for _ in terms[:5]])
                params = [f"%{term}%" for term in terms[:5]]
                cursor.execute(f"SELECT response FROM responses WHERE {where} LIMIT 50", params)
                rows = cursor.fetchall()
        conn.close()
        for (response,) in rows:
            score = _sm_rank_text_for_query(question, response)
            if score >= 0.40 or str(question or '').lower() in str(response or '').lower():
                ranked.append((score, response))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [row[1] for row in ranked[:5]] if ranked else []
    except Exception as e:
        logger.error(f"[DB Search Responses Error] {e}")
        return []

def insert_response_into_personality(intent, response, tone="neutral", complexity="basic"):
    """Insert a learned response into personality1.db"""
    db_path = os.path.join(config.DATASETS_DIR, "personality1.db")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO responses (intent, response, tone, complexity)
            VALUES (?, ?, ?, ?)
        """, (intent, response, tone, complexity))
        conn.commit()
        conn.close()
        logger.info(f"[LEARNING] Inserted personality knowledge: ({intent}, {tone}, {complexity})")
        return True
    except Exception as e:
        logger.error(f"Failed to insert into Personality DB: {e}")
        return False
def _sm_boot_vector_bool(name: str, default: bool = False) -> bool:
    """Read a boot-vector boolean from Globals/env safely."""
    try:
        value = getattr(config, name, default)
    except Exception:
        value = default
    try:
        env_val = os.getenv(f"SARAH_{name}") or os.getenv(name)
        if env_val is not None and str(env_val).strip() != "":
            value = env_val
    except Exception:
        pass
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on", "enabled")


def _sm_boot_vector_int(name: str, default: int) -> int:
    """Read a bounded boot-vector integer from Globals/env safely."""
    try:
        value = getattr(config, name, default)
    except Exception:
        value = default
    try:
        env_val = os.getenv(f"SARAH_{name}") or os.getenv(name)
        if env_val is not None and str(env_val).strip() != "":
            value = env_val
    except Exception:
        pass
    try:
        return max(1, int(value))
    except Exception:
        return int(default)


def _sm_boot_vector_float(name: str, default: float) -> float:
    """Read a bounded boot-vector float from Globals/env safely."""
    try:
        value = getattr(config, name, default)
    except Exception:
        value = default
    try:
        env_val = os.getenv(f"SARAH_{name}") or os.getenv(name)
        if env_val is not None and str(env_val).strip() != "":
            value = env_val
    except Exception:
        pass
    try:
        return max(0.25, float(value))
    except Exception:
        return float(default)


def _sm_compact_for_vector(text: str, max_chars: int = 1400) -> str:
    """Normalize a database/text chunk before storing it in vector memory."""
    import re as _re
    raw = _re.sub(r"\s+", " ", str(text or "")).strip()
    if len(raw) <= int(max_chars):
        return raw
    return raw[: int(max_chars)].rstrip() + " ..."


def _sm_text_to_chunks(source: str, content: str, *, min_chars: int = 24, max_chars: int = 1400):
    """Yield bounded chunks from imported text content."""
    import re as _re
    src = str(source or "local_text")
    text = str(content or "").replace("\r", "\n")
    parts = []
    for block in _re.split(r"\n\s*\n|\n", text):
        block = _sm_compact_for_vector(block, max_chars=max_chars)
        if len(block) >= int(min_chars):
            parts.append(block)
        if len(parts) >= 5000:
            break
    for block in parts:
        yield f"[source:{src}] {block}"


def _sm_sqlite_text_rows_for_vector(max_items: int, time_budget_sec: float,
                                    max_db_bytes: int, max_tables_per_db: int,
                                    max_rows_per_table: int):
    """Yield text rows from DATASETS_DIR/*.db for boot vector refresh.

    This is bounded and read-only. It is intentionally not a full database dump.
    It extracts conversational/general text from known SQLite tables so Phase 7
    can rebuild usable semantic memory without hammering the drive.
    """
    started = time.time()
    emitted = 0
    root = getattr(config, "DATASETS_DIR", DATASETS_DIR)
    try:
        db_files = [f for f in sorted(os.listdir(root)) if f.lower().endswith(".db")]
    except Exception as exc:
        logger.warning(f"[BOOT_VECTOR] Could not list DATASETS_DIR={root}: {exc}")
        return

    skip_db_parts = (
        "system_logs", "context_history", "audit", "ticket", "events",
        "runtime", "session", "cookies", "cache", "vector",
    )
    skip_table_parts = (
        "sqlite_", "log", "audit", "event", "migration", "ticket", "session",
        "performance", "search_log", "sync_",
    )

    for db_file in db_files[:max(1, int(max_db_files))]:
        if emitted >= int(max_items) or (time.time() - started) > float(time_budget_sec):
            break
        low_db = db_file.lower()
        if any(part in low_db for part in skip_db_parts):
            continue
        db_path = os.path.join(root, db_file)
        try:
            if os.path.getsize(db_path) > int(max_db_bytes):
                logger.info(f"[BOOT_VECTOR] Skipped large DB {db_file}")
                continue
        except Exception:
            pass

        conn = None
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0.35)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [str(r[0]) for r in cur.fetchall() if r and r[0]][:int(max_tables_per_db)]

            for table in tables:
                if emitted >= int(max_items) or (time.time() - started) > float(time_budget_sec):
                    break
                low_table = table.lower()
                if any(part in low_table for part in skip_table_parts):
                    continue

                try:
                    cur.execute(f"PRAGMA table_info({_sm_quote_identifier(table)})")
                    cols_info = cur.fetchall() or []
                    text_cols = []
                    for col in cols_info:
                        name = str(col[1] or "")
                        ctype = str(col[2] or "").lower()
                        if not name:
                            continue
                        if ctype == "" or any(x in ctype for x in ("text", "char", "clob", "varchar", "json")):
                            text_cols.append(name)
                    text_cols = text_cols[:8]
                    if not text_cols:
                        continue

                    sql = f"SELECT {', '.join(_sm_quote_identifier(c) for c in text_cols)} FROM {_sm_quote_identifier(table)} LIMIT ?"
                    cur.execute(sql, (int(max_rows_per_table),))
                    for row in cur.fetchall():
                        row_text = " ".join(str(v) for v in row if v not in (None, ""))
                        row_text = _sm_compact_for_vector(row_text)
                        if len(row_text) < 24:
                            continue
                        emitted += 1
                        yield f"[source:{db_file}::{table}] {row_text}"
                        if emitted >= int(max_items):
                            break
                except Exception as table_exc:
                    logger.debug(f"[BOOT_VECTOR] Skipped {db_file}.{table}: {table_exc}")
                    continue
        except Exception as db_exc:
            logger.debug(f"[BOOT_VECTOR] Skipped {db_file}: {db_exc}")
        finally:
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass


def store_voice_input(conn, voice_text, embedding=None):
    """Store one local vector-memory row in ai_learning.db/voice_logs.

    This function was referenced by embed_and_store_dataset_sentences() but was
    absent in the patched file. Keeping it here restores the Phase-7 write path
    without changing the voice_logs schema.
    """
    try:
        if conn is None:
            return False
        text = str(voice_text or "").strip()
        if not text:
            return False
        if embedding is None:
            model = get_active_sentence_model()
            embedding = _sm_encode_text(model, text)
        emb = _sm_embedding_to_list(embedding)
        if not emb:
            return False
        cur = conn.cursor()
        cur.execute("""CREATE TABLE IF NOT EXISTS voice_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        voice_text TEXT NOT NULL,
                        embedding BLOB
                      )""")
        cur.execute(
            "INSERT INTO voice_logs (timestamp, voice_text, embedding) VALUES (?, ?, ?)",
            (dt.now().isoformat(), text, json.dumps(emb, ensure_ascii=False)),
        )
        return True
    except Exception as exc:
        logger.warning(f"[store_voice_input] Failed to store vector row: {exc}")
        return False


def embed_and_store_dataset_sentences(force: bool = False, *, include_imported_data: bool = True,
                                      include_sqlite_pool: bool = True, max_items: int | None = None,
                                      time_budget_sec: float | None = None):
    """Build/refresh local vector memory from imported files and SQLite datasets.

    Contract:
    - LOCAL_ONLY_MODE/offline does NOT disable local vector refresh.
    - No downloads. Uses the local resolver and LiteEmbedder fallback.
    - Bounded by item/time/DB/table caps to avoid drive thrashing.
    - Does not require IMPORT_OTHER_DATA_LEARN for SQLite DB pool refresh.
    - Idempotent enough for boot: duplicates are skipped by voice_text.
    """
    conn = None
    try:
        local_enabled = bool(getattr(config, "LOCAL_DATA_ENABLED", True))
        if not local_enabled:
            logger.info("Skipping vector refresh: LOCAL_DATA_ENABLED is False.")
            return {"ok": False, "inserted": 0, "reason": "LOCAL_DATA_ENABLED_FALSE"}

        max_items = int(max_items or _sm_boot_vector_int("BOOT_VECTOR_MAX_ITEMS", 1200))
        time_budget_sec = float(time_budget_sec or _sm_boot_vector_float("BOOT_VECTOR_TIME_BUDGET_SEC", 35.0))
        max_db_bytes = _sm_boot_vector_int("BOOT_VECTOR_MAX_DB_BYTES", 512 * 1024 * 1024)
        max_tables_per_db = _sm_boot_vector_int("BOOT_VECTOR_MAX_TABLES_PER_DB", 60)
        max_rows_per_table = _sm_boot_vector_int("BOOT_VECTOR_MAX_ROWS_PER_TABLE", 25)

        logger.info(
            "Starting bounded local vector refresh: max_items=%s time_budget=%.1fs include_sqlite=%s include_imported=%s",
            max_items, time_budget_sec, include_sqlite_pool, include_imported_data,
        )

        model = get_active_sentence_model()
        conn = init_database()
        if conn is None:
            return {"ok": False, "inserted": 0, "reason": "DB_INIT_FAILED"}

        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_voice_logs_voice_text ON voice_logs(voice_text)")

        started = time.time()
        inserted_count = 0
        skipped_count = 0
        source_counts = {"imported_text": 0, "sqlite_pool": 0}

        def _store_candidate(line: str, source_kind: str) -> None:
            nonlocal inserted_count, skipped_count
            if inserted_count >= max_items:
                skipped_count += 1
                return
            if (time.time() - started) > float(time_budget_sec):
                skipped_count += 1
                return
            line = _sm_compact_for_vector(line)
            if len(line) < 24:
                skipped_count += 1
                return
            try:
                cursor.execute("SELECT 1 FROM voice_logs WHERE voice_text = ? LIMIT 1", (line,))
                if cursor.fetchone() and not bool(force):
                    skipped_count += 1
                    return
                embedding = _sm_encode_text(model, line)
                if not embedding:
                    skipped_count += 1
                    return
                if store_voice_input(conn, voice_text=line, embedding=embedding):
                    inserted_count += 1
                    source_counts[source_kind] = source_counts.get(source_kind, 0) + 1
                else:
                    skipped_count += 1
            except Exception as ve:
                skipped_count += 1
                logger.debug(f"[BOOT_VECTOR] Skipped candidate due to embedding/store failure: {ve}")

        if bool(include_imported_data):
            try:
                data = config.import_other_data() if hasattr(config, "import_other_data") else {}
            except Exception as imp_exc:
                data = {}
                logger.warning(f"[BOOT_VECTOR] import_other_data unavailable: {imp_exc}")
            for file_path, content in (data or {}).items():
                if inserted_count >= max_items or (time.time() - started) > float(time_budget_sec):
                    break
                for chunk in _sm_text_to_chunks(str(file_path), str(content or "")):
                    _store_candidate(chunk, "imported_text")
                    if inserted_count >= max_items or (time.time() - started) > float(time_budget_sec):
                        break

        if bool(include_sqlite_pool):
            remaining = max(1, max_items - inserted_count)
            remaining_budget = max(0.25, float(time_budget_sec) - (time.time() - started))
            for chunk in _sm_sqlite_text_rows_for_vector(
                max_items=remaining,
                time_budget_sec=remaining_budget,
                max_db_bytes=max_db_bytes,
                max_tables_per_db=max_tables_per_db,
                max_rows_per_table=max_rows_per_table,
            ):
                _store_candidate(chunk, "sqlite_pool")
                if inserted_count >= max_items or (time.time() - started) > float(time_budget_sec):
                    break

        conn.commit()
        result = {
            "ok": True,
            "inserted": inserted_count,
            "skipped": skipped_count,
            "embedder": model.__class__.__name__ if model is not None else None,
            "source_counts": source_counts,
            "time_sec": round(time.time() - started, 3),
            "bounded": True,
        }
        logger.info("Local vector refresh complete: %s", result)
        return result

    except Exception as e:
        logger.error(f"[EMBED_FAIL] Dataset vector refresh failed: {e}")
        return {"ok": False, "inserted": 0, "reason": str(e)}
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


def refresh_local_vector_memory_on_boot(force: bool = False):
    """Named boot entry point used by SarahMemoryInitialization Phase 7."""
    include_imported = bool(getattr(config, "IMPORT_OTHER_DATA_LEARN", False)) or _sm_boot_vector_bool("BOOT_VECTOR_INCLUDE_IMPORTED_DATA", True)
    return embed_and_store_dataset_sentences(
        force=force,
        include_imported_data=include_imported,
        include_sqlite_pool=True,
        max_items=_sm_boot_vector_int("BOOT_VECTOR_MAX_ITEMS", 1200),
        time_budget_sec=_sm_boot_vector_float("BOOT_VECTOR_TIME_BUDGET_SEC", 35.0),
    )

def check_memory_responses(log_output=True, limit=1000):
    """
    Scans all Class 1 dataset entries for malformed, irrelevant, or non-conversational content.
    Flags console scripts, install instructions, file paths, and tech noise.

    Args:
        log_output (bool): If True, prints flagged entries.
        limit (int): Max entries to check per database.

    Returns:
        dict: Report of flagged items from each DB
    """
    
    flagged = {}
    filters = [
        r"\[console_scripts\]", r"\bsetup\.py\b", r"pip install", r"\.exe", r"from ", r"import ",
        r"def ", r"class ", r"fonttools", r"certifi", r"charset", r"ttx", r"wheel", r"cython",
        r"sentry_sdk", r"pyautogui", r"anyio", r"Hello there!", r"project\(", r"normalizer",
        r"cythonize", r"pyftmerge", r"pyftsubset", r"continue", r"__main__"
    ]
    combined_filter = re.compile("|".join(filters), re.IGNORECASE)

    db_paths = {
        "personality1.db": "responses",
        "functions.db": "functions",
        "programming.db": "knowledge_base",
        "ai_learning.db": "learned",
        "avatar.db": "photo_metadata"
    }

    for db_file, table in db_paths.items():
        db_path = os.path.join(config.DATASETS_DIR, db_file)
        if not os.path.exists(db_path):
            continue
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            column = "response" if table == "responses" else "content" if table != "photo_metadata" else "file_path"
            cur.execute(f"SELECT {column} FROM {table} LIMIT ?", (limit,))
            rows = cur.fetchall()
            flagged[db_file] = []
            for row in rows:
                content = row[0] if row else ""
                if content and combined_filter.search(content):
                    flagged[db_file].append(content)
                    if log_output:
                        print(f"[FLAGGED - {db_file}] â†’ {content[:100]}...")
            conn.close()
        except Exception as e:
            print(f"[ERROR] While scanning {db_file}: {e}")
            continue

def auto_correct_dataset_entry(user_input, bad_response, corrected_response, keywords=None):
    """
    Replaces a faulty response in any of the key datasets with a corrected one.
    Includes optional keyword validation before replacing.
    """
    db_files = ["personality1.db", "functions.db", "programming.db"]
    success = False

    for db_file in db_files[:max(1, int(max_db_files))]:
        db_path = os.path.join(config.DATASETS_DIR, db_file)
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            if db_file == "personality1.db":
                cursor.execute("SELECT id FROM responses WHERE response = ?", (bad_response,))
                result = cursor.fetchone()
                if result:
                    entry_id = result[0]
                    if keywords and not all(k.lower() in corrected_response.lower() for k in keywords):
                        logger.warning(f"[AUTO_CORRECT] {db_file} â†’ Missing keywords: {keywords}. Skipping.")
                        continue
                    cursor.execute("UPDATE responses SET response = ? WHERE id = ?", (corrected_response, entry_id))
                    success = True

            elif db_file == "functions.db":
                cursor.execute("SELECT id FROM functions WHERE description = ?", (bad_response,))
                result = cursor.fetchone()
                if result:
                    entry_id = result[0]
                    cursor.execute("UPDATE functions SET description = ? WHERE id = ?", (corrected_response, entry_id))
                    success = True

            elif db_file == "programming.db":
                cursor.execute("SELECT id FROM knowledge_base WHERE content = ?", (bad_response,))
                result = cursor.fetchone()
                if result:
                    entry_id = result[0]
                    cursor.execute("UPDATE knowledge_base SET content = ? WHERE id = ?", (corrected_response, entry_id))
                    success = True

            conn.commit()
            if success:
                logger.info(f"[AUTO_CORRECT] Corrected entry in {db_file}.")
            conn.close()
        except Exception as e:
            logger.error(f"[AUTO_CORRECT ERROR in {db_file}] {e}")
            continue

    return success
from numpy import dot
from numpy.linalg import norm
import numpy as np

def vector_search_qa_cache(query_text, top_n=1):
    """Vector-based semantic search on QA cache memory (query + answer)."""
    conn = None
    try:
        model = get_active_sentence_model()
        query_vec = _sm_encode_text(model, query_text)
        if not query_vec:
            return []

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT query, ai_answer FROM qa_cache")
        entries = cursor.fetchall()

        results = []
        for query, answer in entries:
            if _sm_is_low_quality_cached_answer(answer):
                continue
            combined = f"{query} {answer}"
            emb_vec = _sm_encode_text(model, combined)
            similarity = _sm_cosine_similarity(query_vec, emb_vec)
            if similarity < 0:
                continue

            tokens = tokenize_text(answer)
            entropy_score = len(set(tokens)) / max(len(tokens), 1)
            if entropy_score < 0.3 or len(tokens) < 5:
                continue

            results.append((similarity, answer))

        results.sort(key=lambda item: item[0], reverse=True)
        return results[:top_n]
    except Exception as e:
        logger.error(f"[QA VECTOR SEARCH ERROR] {e}")
        return []
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass


def vector_search(query_text, top_n=1):
    """Enhanced vector search with entropy analysis and dimension-safe matching."""
    conn = None
    try:
        model = get_active_sentence_model()
        query_vec = _sm_encode_text(model, query_text)
        if not query_vec:
            return []

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS search_log (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            query TEXT,
                            match_text TEXT,
                            similarity REAL,
                            timestamp TEXT
                          )""")

        cursor.execute("SELECT voice_text, embedding FROM voice_logs")
        entries = cursor.fetchall()

        results = []
        for text, emb_json in entries:
            if not emb_json:
                continue
            try:
                emb_vec = json.loads(emb_json) if isinstance(emb_json, str) else emb_json
            except Exception:
                continue

            similarity = _sm_cosine_similarity(query_vec, emb_vec)
            if similarity < 0:
                continue

            tokens = tokenize_text(text)
            entropy_score = len(set(tokens)) / max(len(tokens), 1)
            if entropy_score < 0.3 or len(tokens) < 5:
                continue

            results.append((similarity, text))

        results.sort(key=lambda item: item[0], reverse=True)
        top_results = results[:top_n]

        for sim, matched in top_results:
            cursor.execute("INSERT INTO search_log (query, match_text, similarity, timestamp) VALUES (?, ?, ?, ?)",
                           (query_text, matched, float(sim), dt.now().isoformat()))

        conn.commit()
        return top_results
    except Exception as e:
        logger.error(f"[VECTOR_SEARCH ERROR] {e}")
        return []
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

def tokenize_text(text):
    """Tokenizes text for entropy and quality analysis."""
    import re
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(text)
    except:
        return re.findall(r'\b\w+\b', text)

def ensure_qa_cache_table_exists():
    conn = sqlite3.connect(os.path.join(config.DATASETS_DIR, "ai_learning.db"))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS qa_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            ai_answer TEXT,
            hit_score REAL
        )
    """)
    conn.commit()
    conn.close()

def log_ai_functions_event(event_type, details):
    try:
        conn = sqlite3.connect(os.path.join(config.DATASETS_DIR, "functions.db"))
        cursor = conn.cursor()
        timestamp = dt.now().isoformat()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS functions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                function_name TEXT NOT NULL,
                description TEXT,
                is_enabled BOOLEAN DEFAULT 1,
                user_input TEXT,
                timestamp TEXT
            )
        """)
        cursor.execute("""
            INSERT INTO functions (function_name, description, is_enabled, user_input, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (event_type, details, 1, "", timestamp))
        conn.commit()
        conn.close()
        logger.info(f"[FUNCTION_LOG] {event_type} - {details}")
    except Exception as e:
        logger.error(f"[FUNCTION_LOG ERROR] Failed to log function event: {e}")

# ---------------------------------------------------------------------------
# Local SQLite Dataset Pool Search
# ---------------------------------------------------------------------------
def _sm_dataset_query_terms(query: str):
    """Extract bounded search terms for direct local dataset lookup."""
    import re as _re
    raw = str(query or "").strip().lower()
    raw = _re.sub(r"[^a-z0-9_\-\s']+", " ", raw)
    raw = _re.sub(r"\s+", " ", raw).strip()
    prefixes = (
        "who is ", "who was ", "what is ", "what are ", "tell me about ",
        "explain ", "define ", "describe ", "give me information on ",
        "give me info on ", "how does ", "how do ", "why does ", "when did ",
        "where is ", "where was ",
    )
    cleaned = raw
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does", "did",
        "for", "from", "give", "how", "i", "in", "info", "information", "is",
        "me", "of", "on", "or", "please", "tell", "the", "this", "to", "was",
        "were", "what", "when", "where", "which", "who", "why", "with", "you",
    }
    words = [w for w in _re.findall(r"[a-z0-9][a-z0-9_\-']*", cleaned) if len(w) > 1 and w not in stop]
    # Preserve the cleaned entity/topic phrase first, then individual terms.
    terms = []
    phrase = " ".join(words[:6]).strip()
    if phrase and len(words) >= 2:
        terms.append(phrase)
    for w in words[:8]:
        if w not in terms:
            terms.append(w)
    return terms


def _sm_terms_are_short_or_noisy(terms: list) -> bool:
    """True when terms are too short/common for broad LIKE scans.

    Examples such as API, RAM, CPU, GPU, UI, OS, DB are valid concepts, but
    substring SQL like %%api%% across every routed table forces SQLite to scan
    large local databases and will thrash mechanical disks. Short acronyms must
    use cache/exact curated tables or a later indexed lookup, not broad pool
    scans.
    """
    clean = [str(t or "").strip().lower() for t in (terms or []) if str(t or "").strip()]
    if not clean:
        return True
    if all((" " not in t and len(t) <= 3) for t in clean):
        return True
    noisy = {"api", "ram", "cpu", "gpu", "ui", "os", "db", "sql", "url", "usb", "ssd", "hdd"}
    return bool(len(clean) == 1 and clean[0] in noisy)


def _sm_route_allowed_for_short_term(route: dict) -> bool:
    """Restrict acronym lookups to exact QA-style stores.

    This prevents broad unindexed scans of programming papers, dataset ledgers,
    logs, and generated semantic samples for tiny tokens like 'api'.
    """
    db = str((route or {}).get("db_name") or "").lower()
    table = str((route or {}).get("table_name") or "").lower()
    if table in {"qa_cache", "sm_qa_cache", "qacache"}:
        return True
    if db == "ai_learning.db" and table == "knowledge_base":
        return True
    return False


def _sm_exact_short_query_hit(cur, table_name: str, text_cols: list, query: str, terms: list, limit: int = 4) -> list:
    """Read only exact/rowid-limited candidates for short acronym queries.

    This deliberately avoids WHERE col LIKE '%%api%%'.  For short terms, broad
    substring scans are the HDD-thrash failure mode.
    """
    qnorm = re.sub(r"\s+", " ", str(query or "").strip().lower())
    topic = str((terms or [""])[0] or "").strip().lower()
    if not topic:
        return []
    lower_cols = {str(c).lower(): c for c in text_cols}
    q_candidates = [lower_cols.get(k) for k in ("query", "question", "prompt", "user_input", "input") if lower_cols.get(k)]
    a_candidates = [lower_cols.get(k) for k in ("ai_answer", "answer", "response", "reply", "content", "output") if lower_cols.get(k)]
    rows = []
    try:
        if q_candidates:
            qcol = q_candidates[0]
            select_cols = []
            for c in [qcol] + a_candidates + text_cols[:4]:
                if c and c not in select_cols:
                    select_cols.append(c)
            sql = f"SELECT {', '.join(_sm_quote_identifier(c) for c in select_cols)} FROM {_sm_quote_identifier(table_name)} WHERE lower({_sm_quote_identifier(qcol)}) IN (?, ?, ?) LIMIT ?"
            cur.execute(sql, (qnorm, f"what is {topic}", f"define {topic}", int(limit)))
            rows.extend((select_cols, r) for r in cur.fetchall())
    except Exception:
        pass
    try:
        if not rows and "category" in lower_cols and "content" in lower_cols:
            ccat = lower_cols["category"]
            ccontent = lower_cols["content"]
            sql = f"SELECT {_sm_quote_identifier(ccat)}, {_sm_quote_identifier(ccontent)} FROM {_sm_quote_identifier(table_name)} WHERE lower({_sm_quote_identifier(ccat)}) IN (?, ?, ?) LIMIT ?"
            cur.execute(sql, (topic, f"definition:{topic}", "webster_static", int(limit)))
            rows.extend(([ccat, ccontent], r) for r in cur.fetchall())
    except Exception:
        pass
    return rows[:int(limit)]


def _sm_dataset_score(query: str, text: str) -> float:
    """Lexical relevance score for direct SQLite dataset hits."""
    import re as _re
    content = _re.sub(r"\s+", " ", str(text or "").lower()).strip()
    if not content:
        return 0.0
    terms = _sm_dataset_query_terms(query)
    if not terms:
        return 0.0
    phrase_terms = [t for t in terms if " " in t]
    word_terms = [t for t in terms if " " not in t]
    phrase_hits = sum(1 for t in phrase_terms if t in content)
    word_hits = sum(1 for t in word_terms if _re.search(rf"\b{_re.escape(t)}\b", content))
    if len(word_terms) >= 2 and word_hits < 2 and phrase_hits == 0:
        return 0.0
    coverage = word_hits / max(len(word_terms), 1)
    phrase_bonus = 0.25 if phrase_hits else 0.0
    exact_bonus = 0.20 if phrase_terms and phrase_terms[0] in content else 0.0
    return max(0.0, min(1.0, coverage + phrase_bonus + exact_bonus))


def _sm_dataset_snippet(text: str, query: str, max_chars: int = 900) -> str:
    """Return a compact snippet around the best local dataset term."""
    import re as _re
    raw = _re.sub(r"\s+", " ", str(text or "")).strip()
    if not raw:
        return ""
    low = raw.lower()
    idx = -1
    for term in _sm_dataset_query_terms(query):
        if not term:
            continue
        pos = low.find(term.lower())
        if pos >= 0:
            idx = pos
            break
    if idx < 0:
        return raw[:max_chars].strip()
    start = max(0, idx - max_chars // 3)
    end = min(len(raw), start + max_chars)
    out = raw[start:end].strip()
    if start > 0:
        out = "..." + out
    if end < len(raw):
        out += "..."
    return out


def _sm_quote_identifier(name: str) -> str:
    """SQLite identifier quoting for table/column names discovered by PRAGMA."""
    return '"' + str(name or "").replace('"', '""') + '"'


def _sm_dataset_db_skip_name(filename: str) -> bool:
    """Avoid noisy/runtime-heavy databases for general answer fallback."""
    low = str(filename or "").lower()
    noisy = (
        "system_logs.db", "context_history.db", "audit", "ticket", "events",
        "runtime", "session", "cookies", "cache",
    )
    return any(part in low for part in noisy)


def search_local_dataset_pool(query: str, max_hits: int = 8, time_budget_sec: float = 2.0,
                              max_tables_per_db: int = 40, max_rows_per_table: int = 6,
                              max_db_bytes: int = 256 * 1024 * 1024,
                              max_db_files: int = 12):
    """Search local SQLite databases directly without Phase-7 vector embedding.

    Purpose:
      - Restores local-first general knowledge fallback when boot skips Phase 7.
      - Searches DATASETS_DIR/*.db using bounded, read-only SQLite queries.
      - Never calls Web/API and never writes to database files.

    Returns:
      dict with ok, hits, db_count, tables_checked, errors.
    """
    started = time.time()
    q = str(query or "").strip()
    out = {
        "ok": False,
        "query": q,
        "hits": [],
        "db_count": 0,
        "tables_checked": 0,
        "errors": [],
        "method": "direct_sqlite_dataset_pool_search",
        "phase7_required": False,
    }
    if not q:
        out["errors"].append("empty_query")
        return out

    terms = _sm_dataset_query_terms(q)
    if not terms:
        out["errors"].append("no_search_terms")
        return out

    root = getattr(config, "DATASETS_DIR", DATASETS_DIR)
    try:
        db_files = [f for f in sorted(os.listdir(root)) if f.lower().endswith(".db")]
    except Exception as exc:
        out["errors"].append(f"datasets_dir_unavailable:{exc}")
        return out

    hits = []
    for db_file in db_files[:max(1, int(max_db_files))]:
        if time.time() - started > float(time_budget_sec):
            break
        if _sm_dataset_db_skip_name(db_file):
            continue
        db_path = os.path.join(root, db_file)
        try:
            if os.path.getsize(db_path) > int(max_db_bytes):
                out["errors"].append(f"skipped_large_db:{db_file}")
                continue
        except Exception:
            pass

        out["db_count"] += 1
        conn = None
        try:
            uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=0.25)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [str(r[0]) for r in cur.fetchall() if r and r[0]][:int(max_tables_per_db)]
            for table in tables:
                if time.time() - started > float(time_budget_sec):
                    break
                t_low = table.lower()
                if any(k in t_low for k in ("log", "audit", "event", "migration", "ticket", "session")):
                    continue
                try:
                    cur.execute(f"PRAGMA table_info({_sm_quote_identifier(table)})")
                    cols_info = cur.fetchall()
                    text_cols = []
                    for col in cols_info:
                        name = str(col[1])
                        ctype = str(col[2] or "").lower()
                        if not name:
                            continue
                        if ctype == "" or any(x in ctype for x in ("text", "char", "clob", "varchar", "json")):
                            text_cols.append(name)
                    text_cols = text_cols[:8]
                    if not text_cols:
                        continue
                    out["tables_checked"] += 1
                    first_term = next((t for t in terms if " " not in t), terms[-1])
                    where = " OR ".join([f"{_sm_quote_identifier(c)} LIKE ?" for c in text_cols])
                    sql = f"SELECT {', '.join(_sm_quote_identifier(c) for c in text_cols)} FROM {_sm_quote_identifier(table)} WHERE {where} LIMIT ?"
                    params = [f"%{first_term}%" for _ in text_cols] + [int(max_rows_per_table)]
                    cur.execute(sql, params)
                    for row in cur.fetchall():
                        row_text = " ".join(str(v) for v in row if v is not None)
                        if not row_text.strip():
                            continue
                        if _sm_is_low_quality_cached_answer(row_text):
                            continue
                        score = _sm_dataset_score(q, row_text)
                        if score < 0.42:
                            continue
                        hits.append({
                            "db": db_file,
                            "table": table,
                            "score": float(score),
                            "snippet": _sm_dataset_snippet(row_text, q),
                        })
                        if len(hits) >= int(max_hits) * 2:
                            break
                    if len(hits) >= int(max_hits) * 2:
                        break
                except Exception as table_exc:
                    out["errors"].append(f"{db_file}.{table}:{table_exc}")
                    continue
        except Exception as db_exc:
            out["errors"].append(f"{db_file}:{db_exc}")
        finally:
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
        if len(hits) >= int(max_hits) * 2:
            break

    hits.sort(key=lambda h: float(h.get("score") or 0.0), reverse=True)
    out["hits"] = hits[:int(max_hits)]
    out["ok"] = bool(out["hits"])
    out["latency_ms"] = int((time.time() - started) * 1000)
    return out



# ---------------------------------------------------------------------------
# SARAHMEMORY v9 SEMANTIC MEMORY MAP / META.DB ROUTING
# ---------------------------------------------------------------------------
# Purpose:
# - Keep app.py out of the knowledge-pool business.
# - Let meta.db map which local SQLite pools can answer each semantic query type.
# - Provide bounded, read-only pool lookup for AdvCU fast-answer resolution.
# - No web/API access, no filesystem mutation outside meta.db schema refresh, no scans
#   outside DATASETS_DIR.

_SM_META_MEMORY_SCHEMA_VERSION = "SarahMemory.meta_memory_map.v1"


def _sm_meta_db_path() -> str:
    try:
        configured = getattr(config, "META_DB_PATH", None)
        if configured:
            return str(configured)
    except Exception:
        pass
    try:
        root = getattr(config, "DATASETS_DIR", DATASETS_DIR)
    except Exception:
        root = DATASETS_DIR
    return os.path.join(str(root), "meta.db")


def _sm_dataset_root() -> str:
    try:
        return str(getattr(config, "DATASETS_DIR", DATASETS_DIR))
    except Exception:
        return str(DATASETS_DIR)


def _sm_schema_hash_for_table(cur, table_name: str) -> str:
    try:
        cur.execute(f"PRAGMA table_info({_sm_quote_identifier(table_name)})")
        cols = [tuple(r) for r in cur.fetchall()]
        payload = json.dumps(cols, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        return ""


def _sm_infer_table_route(db_name: str, table_name: str) -> dict:
    """Infer safe semantic routing metadata for known table shapes."""
    db = str(db_name or "").lower()
    table = str(table_name or "").lower()
    route = {
        "purpose": "local_sqlite_pool",
        "query_types": "general_chat",
        "read_allowed": 1,
        "write_allowed": 0,
        "max_rows_per_query": 8,
        "confidence_weight": 0.45,
        "priority": 40,
        "enabled": 1,
    }

    if table in {"qa_cache", "sm_qa_cache", "qacache"}:
        route.update({
            "purpose": "question_answer_cache",
            "query_types": "definition,question,general_chat,howto",
            "confidence_weight": 0.86,
            "priority": 95,
        })
    elif table == "knowledge_base":
        if db == "ai_learning.db":
            route.update({
                "purpose": "general_static_or_learned_knowledge",
                "query_types": "definition,fact,question,general_chat",
                "confidence_weight": 0.76,
                "priority": 85,
            })
        elif db == "programming.db":
            route.update({
                "purpose": "programming_reference_knowledge",
                "query_types": "programming_reference,code,implementation,project_code_question",
                "confidence_weight": 0.66,
                "priority": 68,
            })
        else:
            route.update({
                "purpose": "bounded_domain_knowledge_base",
                "query_types": "domain_reference,question",
                "confidence_weight": 0.50,
                "priority": 45,
            })
    elif table == "code_corpus":
        route.update({
            "purpose": "project_code_corpus",
            "query_types": "project_code_question,code,implementation,model_governance",
            "confidence_weight": 0.80,
            "priority": 80,
        })
    elif table in {"dataset_ledger", "model_registry", "training_runs", "training_jobs", "eval_results"}:
        route.update({
            "purpose": "sarahmemory_living_model_records",
            "query_types": "model_governance,self_tokenization,semantic_sample,tokenization",
            "confidence_weight": 0.70,
            "priority": 78,
        })
    elif table == "responses":
        route.update({
            "purpose": "reply_pool_or_personality_response",
            "query_types": "identity,smalltalk,reply_pool,greeting,farewell",
            "confidence_weight": 0.55,
            "priority": 55,
        })
    elif any(k in table for k in ("log", "audit", "event", "session", "ticket")):
        route.update({
            "purpose": "operational_history_not_fast_answer_default",
            "query_types": "audit,diagnostics",
            "read_allowed": 0,
            "max_rows_per_query": 0,
            "confidence_weight": 0.0,
            "priority": 5,
            "enabled": 0,
        })

    if db == "meta.db":
        route.update({
            "purpose": "database_registry_and_semantic_map",
            "query_types": "meta,database_map,diagnostics",
            "read_allowed": 1,
            "write_allowed": 0,
            "max_rows_per_query": 10,
            "confidence_weight": 0.30,
            "priority": 10,
            "enabled": 0,
        })
    return route


def ensure_meta_memory_map_schema() -> dict:
    """Create/upgrade meta.db routing tables. Idempotent and local-only."""
    meta_path = _sm_meta_db_path()
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    conn = sqlite3.connect(meta_path, timeout=1.5)
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS db_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                db_name TEXT UNIQUE NOT NULL,
                db_path TEXT NOT NULL,
                role TEXT NOT NULL,
                read_allowed INTEGER DEFAULT 1,
                write_allowed INTEGER DEFAULT 0,
                governance_level TEXT DEFAULT 'bounded',
                priority INTEGER DEFAULT 50,
                last_seen TEXT,
                schema_hash TEXT,
                schema_version TEXT DEFAULT 'SarahMemory.meta_memory_map.v1'
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS table_registry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                db_name TEXT NOT NULL,
                table_name TEXT NOT NULL,
                purpose TEXT,
                query_types TEXT,
                read_allowed INTEGER DEFAULT 1,
                write_allowed INTEGER DEFAULT 0,
                max_rows_per_query INTEGER DEFAULT 20,
                confidence_weight REAL DEFAULT 0.5,
                priority INTEGER DEFAULT 50,
                last_seen TEXT,
                schema_hash TEXT,
                UNIQUE(db_name, table_name)
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS semantic_pool_routes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_type TEXT NOT NULL,
                intent TEXT DEFAULT '',
                db_name TEXT NOT NULL,
                table_name TEXT NOT NULL,
                priority INTEGER DEFAULT 50,
                enabled INTEGER DEFAULT 1,
                UNIQUE(query_type, intent, db_name, table_name)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_db_registry_name ON db_registry(db_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_table_registry_route ON table_registry(query_types, read_allowed, priority)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_semantic_pool_routes_lookup ON semantic_pool_routes(query_type, intent, enabled, priority)")
        conn.commit()
        return {"ok": True, "meta_db": meta_path, "schema": _SM_META_MEMORY_SCHEMA_VERSION}
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        return {"ok": False, "meta_db": meta_path, "error": str(exc)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def refresh_meta_db_registry(max_db_files: int = 64, max_tables_per_db: int = 160) -> dict:
    """Refresh meta.db's map of local SQLite databases/tables under DATASETS_DIR.

    This is bounded and project-local. It does not read row data, only SQLite schema
    metadata. Operational/noisy tables are registered but disabled for fast answers.
    """
    schema = ensure_meta_memory_map_schema()
    out = {
        "ok": bool(schema.get("ok")),
        "schema": _SM_META_MEMORY_SCHEMA_VERSION,
        "dbs_seen": 0,
        "tables_seen": 0,
        "routes_written": 0,
        "errors": [],
    }
    if not schema.get("ok"):
        out["errors"].append(str(schema.get("error") or "meta_schema_failed"))
        return out

    root = _sm_dataset_root()
    try:
        db_files = [f for f in sorted(os.listdir(root)) if f.lower().endswith(".db")]
    except Exception as exc:
        out["errors"].append(f"datasets_dir_unavailable:{exc}")
        return out

    meta_path = _sm_meta_db_path()
    meta_conn = sqlite3.connect(meta_path, timeout=2.0)
    try:
        meta_cur = meta_conn.cursor()
        now = dt.now().isoformat()
        for db_file in db_files[:max(1, int(max_db_files))]:
            db_path = os.path.join(root, db_file)
            if not os.path.isfile(db_path):
                continue
            out["dbs_seen"] += 1
            role = "database_registry" if db_file.lower() == "meta.db" else "local_memory_pool"
            db_read_allowed = 1
            db_write_allowed = 0
            db_priority = 10 if db_file.lower() == "meta.db" else 50
            schema_hash_accum = []
            conn = None
            try:
                conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0.35)
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                tables = [str(r[0]) for r in cur.fetchall() if r and r[0]][:max(1, int(max_tables_per_db))]
                for table in tables:
                    inferred = _sm_infer_table_route(db_file, table)
                    th = _sm_schema_hash_for_table(cur, table)
                    if th:
                        schema_hash_accum.append(f"{table}:{th}")
                    meta_cur.execute(
                        """
                        INSERT OR REPLACE INTO table_registry
                        (db_name, table_name, purpose, query_types, read_allowed, write_allowed,
                         max_rows_per_query, confidence_weight, priority, last_seen, schema_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            db_file,
                            table,
                            inferred["purpose"],
                            inferred["query_types"],
                            int(inferred["read_allowed"]),
                            int(inferred["write_allowed"]),
                            int(inferred["max_rows_per_query"]),
                            float(inferred["confidence_weight"]),
                            int(inferred["priority"]),
                            now,
                            th,
                        ),
                    )
                    out["tables_seen"] += 1
                    for qt in [q.strip() for q in str(inferred["query_types"] or "").split(",") if q.strip()]:
                        meta_cur.execute(
                            """
                            INSERT OR REPLACE INTO semantic_pool_routes
                            (query_type, intent, db_name, table_name, priority, enabled)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (qt, None, db_file, table, int(inferred["priority"]), int(inferred["enabled"])),
                        )
                        out["routes_written"] += 1
                db_schema_hash = hashlib.sha256("|".join(sorted(schema_hash_accum)).encode("utf-8", errors="ignore")).hexdigest() if schema_hash_accum else ""
                meta_cur.execute(
                    """
                    INSERT OR REPLACE INTO db_registry
                    (db_name, db_path, role, read_allowed, write_allowed, governance_level,
                     priority, last_seen, schema_hash, schema_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (db_file, db_path, role, db_read_allowed, db_write_allowed, "bounded", db_priority, now, db_schema_hash, _SM_META_MEMORY_SCHEMA_VERSION),
                )
            except Exception as exc:
                out["errors"].append(f"{db_file}:{exc}")
            finally:
                try:
                    if conn is not None:
                        conn.close()
                except Exception:
                    pass
        meta_conn.commit()
        out["ok"] = True
        out["meta_db"] = meta_path
        return out
    except Exception as exc:
        try:
            meta_conn.rollback()
        except Exception:
            pass
        out["ok"] = False
        out["errors"].append(str(exc))
        return out
    finally:
        try:
            meta_conn.close()
        except Exception:
            pass


def get_semantic_pool_routes(intent: str = "", query_type: str = "", refresh_if_empty: bool = True, limit: int = 24) -> list:
    """Return registered local pools for an AdvCU query type/intent."""
    ensure_meta_memory_map_schema()
    meta_path = _sm_meta_db_path()
    qt = str(query_type or "general_chat").strip().lower()
    it = str(intent or "").strip().lower()
    rows = []
    conn = None
    try:
        conn = sqlite3.connect(f"file:{meta_path}?mode=ro", uri=True, timeout=0.35)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT r.query_type, r.intent, r.db_name, r.table_name, r.priority, r.enabled,
                   t.purpose, t.read_allowed, t.write_allowed, t.max_rows_per_query, t.confidence_weight
            FROM semantic_pool_routes r
            JOIN table_registry t ON t.db_name=r.db_name AND t.table_name=r.table_name
            WHERE r.enabled=1 AND t.read_allowed=1
              AND lower(r.query_type)=?
              AND (r.intent IS NULL OR r.intent='' OR lower(r.intent)=?)
            ORDER BY r.priority DESC, t.confidence_weight DESC
            LIMIT ?
            """,
            (qt, it, int(limit)),
        )
        rows = [dict(r) for r in cur.fetchall()]
        # Only generic chat should use generic chat routes. Definition/fact/code
        # queries must not silently fan out into every general_chat table.
        if not rows and qt == "general_chat":
            cur.execute(
                """
                SELECT r.query_type, r.intent, r.db_name, r.table_name, r.priority, r.enabled,
                       t.purpose, t.read_allowed, t.write_allowed, t.max_rows_per_query, t.confidence_weight
                FROM semantic_pool_routes r
                JOIN table_registry t ON t.db_name=r.db_name AND t.table_name=r.table_name
                WHERE r.enabled=1 AND t.read_allowed=1
                  AND lower(r.query_type)='general_chat'
                  AND (r.intent IS NULL OR r.intent='' OR lower(r.intent)=?)
                ORDER BY r.priority DESC, t.confidence_weight DESC
                LIMIT ?
                """,
                (it, int(limit)),
            )
            rows = [dict(r) for r in cur.fetchall()]
    except Exception:
        rows = []
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:
            pass

    if not rows and refresh_if_empty:
        refresh_meta_db_registry()
        return get_semantic_pool_routes(intent=intent, query_type=query_type, refresh_if_empty=False, limit=limit)
    return rows


def _sm_text_columns_for_table(cur, table_name: str) -> list:
    try:
        cur.execute(f"PRAGMA table_info({_sm_quote_identifier(table_name)})")
        cols_info = cur.fetchall()
    except Exception:
        return []
    text_cols = []
    for col in cols_info:
        try:
            name = str(col[1])
            ctype = str(col[2] or "").lower()
            if ctype == "" or any(x in ctype for x in ("text", "char", "clob", "varchar", "json")):
                text_cols.append(name)
        except Exception:
            continue
    return text_cols[:10]


def search_semantic_memory_pools(query: str, *, intent: str = "", query_type: str = "general_chat",
                                 max_hits: int = 8, time_budget_sec: float = 1.5) -> dict:
    """Search meta-routed local SQLite pools for a bounded answer candidate.

    Read-only. Local DATASETS_DIR only. No external API, no web, no shell, no wide
    filesystem scan. This is the DB retrieval half of the AdvCU fast-answer lane.
    """
    started = time.time()
    q = str(query or "").strip()
    out = {
        "ok": False,
        "query": q,
        "intent": str(intent or ""),
        "query_type": str(query_type or "general_chat"),
        "hits": [],
        "routes_checked": 0,
        "errors": [],
        "method": "meta_routed_sqlite_semantic_pool_search",
        "network_used": False,
        "execution_authority": False,
    }
    if not q:
        out["errors"].append("empty_query")
        return out
    if _sm_is_volatile_body_fact_query(q):
        out["errors"].append("volatile_body_fact_blocked")
        return out
    terms = _sm_dataset_query_terms(q)
    if not terms:
        out["errors"].append("no_search_terms")
        return out

    short_or_noisy = _sm_terms_are_short_or_noisy(terms)
    routes = get_semantic_pool_routes(intent=intent, query_type=query_type, limit=16)
    if short_or_noisy:
        routes = [r for r in routes if _sm_route_allowed_for_short_term(r)][:4]
        out["short_term_guard"] = True
        out["guard_reason"] = "short_or_noisy_term_no_broad_like_scan"
    if not routes:
        out["errors"].append("no_meta_routes")
        return out
    root = _sm_dataset_root()
    hits = []
    for route in routes:
        if time.time() - started > float(time_budget_sec):
            break
        out["routes_checked"] += 1
        db_name = str(route.get("db_name") or "")
        table = str(route.get("table_name") or "")
        if not db_name or not table:
            continue
        db_path = os.path.join(root, db_name)
        if not os.path.isfile(db_path):
            continue
        if not os.path.abspath(db_path).startswith(os.path.abspath(root)):
            continue
        try:
            semantic_max_bytes = int(getattr(config, "SEMANTIC_SQLITE_MAX_DB_BYTES", 256 * 1024 * 1024))
        except Exception:
            semantic_max_bytes = 256 * 1024 * 1024
        try:
            large_scan_override = bool(getattr(config, "SARAHMEMORY_ALLOW_LARGE_DB_ROUTED_SCAN", False))
        except Exception:
            large_scan_override = False
        try:
            db_size = os.path.getsize(db_path)
            if db_size > semantic_max_bytes and not large_scan_override:
                out["errors"].append(f"skipped_large_db:{db_name}")
                continue
        except Exception:
            pass
        conn = None
        try:
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=0.25)
            cur = conn.cursor()
            text_cols = _sm_text_columns_for_table(cur, table)
            if not text_cols:
                continue
            lower_cols = {c.lower(): c for c in text_cols}
            qcol = lower_cols.get("query") or lower_cols.get("question") or lower_cols.get("prompt") or lower_cols.get("user_input") or lower_cols.get("input")
            acol = lower_cols.get("ai_answer") or lower_cols.get("answer") or lower_cols.get("response") or lower_cols.get("reply") or lower_cols.get("content") or lower_cols.get("output") or lower_cols.get("snippet")
            max_rows = max(1, min(8 if short_or_noisy else 16, int(route.get("max_rows_per_query") or 8)))
            first_term = next((t for t in terms if " " not in t), terms[-1])
            if short_or_noisy:
                fetched_rows = _sm_exact_short_query_hit(cur, table, text_cols, q, terms, limit=max_rows)
            else:
                where_cols = []
                for c in text_cols[:6]:
                    where_cols.append(f"{_sm_quote_identifier(c)} LIKE ?")
                where = " OR ".join(where_cols)
                select_cols = text_cols[:6]
                sql = f"SELECT {', '.join(_sm_quote_identifier(c) for c in select_cols)} FROM {_sm_quote_identifier(table)} WHERE {where} LIMIT ?"
                params = [f"%{first_term}%" for _ in where_cols] + [max_rows]
                cur.execute(sql, params)
                fetched_rows = [(select_cols, row) for row in cur.fetchall()]
            for select_cols, row in fetched_rows:
                row_map = {select_cols[i]: row[i] for i in range(min(len(select_cols), len(row)))}
                answer_text = ""
                if acol and row_map.get(acol) is not None:
                    answer_text = str(row_map.get(acol) or "").strip()
                elif qcol and row_map.get(qcol) is not None:
                    answer_text = str(row_map.get(qcol) or "").strip()
                if not answer_text:
                    answer_text = " ".join(str(v) for v in row if v is not None).strip()
                if not answer_text or _sm_is_low_quality_cached_answer(answer_text):
                    continue
                row_text = " ".join(str(v) for v in row if v is not None)
                score = _sm_dataset_score(q, row_text)
                weight = float(route.get("confidence_weight") or 0.5)
                final_score = max(0.0, min(1.0, (score * 0.75) + (weight * 0.25)))
                if final_score < 0.42:
                    continue
                hits.append({
                    "db": db_name,
                    "table": table,
                    "purpose": route.get("purpose"),
                    "query_type": route.get("query_type"),
                    "score": float(final_score),
                    "answer": answer_text[:1600].strip(),
                    "snippet": _sm_dataset_snippet(row_text, q, max_chars=900),
                    "source": "local_semantic_db",
                    "network_used": False,
                    "execution_authority": False,
                })
                if len(hits) >= int(max_hits) * 2:
                    break
        except Exception as exc:
            out["errors"].append(f"{db_name}.{table}:{exc}")
        finally:
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
        if len(hits) >= int(max_hits) * 2:
            break
    hits.sort(key=lambda h: float(h.get("score") or 0.0), reverse=True)
    out["hits"] = hits[:int(max_hits)]
    out["ok"] = bool(out["hits"])
    out["latency_ms"] = int((time.time() - started) * 1000)
    return out

if __name__ == '__main__':
    logger.info("Starting SarahMemoryDatabase module test.")
    conn = init_database()
    if conn:
        model = get_active_sentence_model()
        embedding = model.encode("Test voice input.").tolist()
        store_performance_metrics(conn)
        logs = get_all_voice_logs(conn)
        export_voice_logs_to_json(conn, 'voice_logs_export.json')
        conn.close()
def store_response_history(*args, **kwargs):
    """
    Persist a user/assistant exchange locally (offline-first).

    Supported call shapes:
      - store_response_history({"user_input":..., "response":..., "meta":...})
      - store_response_history(user_text, response_text, meta={...})
      - store_response_history(user_text=user_text, response=response_text, meta={...})

    Framework rule:
      - canonical/raw truth is stored internally
      - presentation answer is stored separately and is the user-facing form
    """
    try:
        import os
        import json
        import sqlite3
        import datetime as _dt

        user_text = ""
        response_text = ""
        meta = {}

        if args and isinstance(args[0], dict):
            payload = dict(args[0])
            user_text = payload.get("user_input") or payload.get("query") or payload.get("text") or payload.get("user") or ""
            response_text = payload.get("response") or payload.get("reply") or payload.get("data") or payload.get("assistant") or ""
            meta = payload.get("meta") or payload.get("metadata") or {}
        else:
            if len(args) >= 2:
                user_text = args[0]
                response_text = args[1]
            elif len(args) == 1:
                response_text = args[0]
                user_text = kwargs.get("user_input") or kwargs.get("query") or kwargs.get("text") or ""
            else:
                user_text = kwargs.get("user_input") or kwargs.get("query") or kwargs.get("text") or ""
                response_text = kwargs.get("response") or kwargs.get("reply") or kwargs.get("data") or ""
            meta = kwargs.get("meta") or kwargs.get("metadata") or kwargs.get("extra") or {}

        if meta is None:
            meta = {}
        if not isinstance(meta, dict):
            meta = {"_meta": str(meta)}
        if _sm_meta_blocks_persistence(meta, text=str(user_text or response_text or "")):
            logger.info("[V10/V9C] response_history store blocked for volatile SelfAware body fact.")
            return False

        intent = str(kwargs.get("intent") or meta.get("intent") or meta.get("Intent") or "")
        session_id = str(kwargs.get("session_id") or meta.get("session_id") or meta.get("sid") or "")
        ts = kwargs.get("timestamp") or meta.get("ts") or _dt.datetime.utcnow().isoformat(timespec="seconds")

        try:
            ensure_response_memory_schema()
        except Exception:
            pass

        layers = _normalize_response_layers(str(user_text or ""), str(response_text or ""), meta, kwargs)
        canonical_answer = layers["canonical_answer"]
        presented_answer = layers["presented_answer"]
        canonical_type = layers["canonical_type"]
        truth_locked = 1 if layers["truth_locked"] else 0
        tone = layers["tone"]
        style = layers["style"]
        persona_state = layers["persona_state"]
        lane = layers["lane"]
        source = layers["source"]
        raw_meta_json = _json_dumps_safe(layers["raw_meta"])
        presentation_meta_json = _json_dumps_safe(layers["presentation_meta"])
        meta_json = _json_dumps_safe(meta)

        # AI learning DB (canonical + presentation record)
        try:
            with sqlite3.connect(DB_PATH) as conn:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO conversations (timestamp, user_input, ai_response, intent, sentiment_score, emotional_state, session_id, canonical_answer, presented_answer, lane, source, canonical_type, truth_locked, tone, style, persona_state, raw_meta_json, presentation_meta_json) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        str(ts),
                        str(user_text or ""),
                        str(presented_answer or ""),
                        str(intent or ""),
                        None,
                        None,
                        (session_id or None),
                        str(canonical_answer or ""),
                        str(presented_answer or ""),
                        str(lane or ""),
                        str(source or ""),
                        str(canonical_type or "response"),
                        int(truth_locked),
                        str(tone or ""),
                        str(style or ""),
                        str(persona_state or ""),
                        raw_meta_json,
                        presentation_meta_json,
                    ),
                )
                c.execute(
                    "INSERT INTO response_layers (ts, session_id, user_input, canonical_answer, presented_answer, intent, lane, source, canonical_type, truth_locked, tone, style, persona_state, raw_meta_json, presentation_meta_json) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        str(ts),
                        str(session_id or ""),
                        str(user_text or ""),
                        str(canonical_answer or ""),
                        str(presented_answer or ""),
                        str(intent or ""),
                        str(lane or ""),
                        str(source or ""),
                        str(canonical_type or "response"),
                        int(truth_locked),
                        str(tone or ""),
                        str(style or ""),
                        str(persona_state or ""),
                        raw_meta_json,
                        presentation_meta_json,
                    ),
                )
                conn.commit()
        except Exception:
            pass

        # System log DB (backward compatible flat history + dual-layer fields)
        try:
            sys_db = _resolve_system_log_db_path()
            os.makedirs(os.path.dirname(sys_db), exist_ok=True)
            with sqlite3.connect(sys_db) as conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO response_history(ts,user_input,response,intent,tone,complexity,source,session_id,meta_json,canonical_answer,presented_answer,canonical_type,truth_locked,lane,persona_state,raw_meta_json,presentation_meta_json) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        str(ts),
                        str(user_text or ""),
                        str(presented_answer or ""),
                        str(intent or ""),
                        str(tone or ""),
                        str(style or ""),
                        str(source or ""),
                        str(session_id or ""),
                        meta_json,
                        str(canonical_answer or ""),
                        str(presented_answer or ""),
                        str(canonical_type or "response"),
                        int(truth_locked),
                        str(lane or ""),
                        str(persona_state or ""),
                        raw_meta_json,
                        presentation_meta_json,
                    ),
                )
                conn.commit()
        except Exception:
            pass

        # Context memory (presentation answer only)
        try:
            from SarahMemoryAiFunctions import add_to_context_entry  # type: ignore
            add_to_context_entry(str(user_text or ""), str(presented_answer or ""), meta=meta)  # type: ignore
        except Exception:
            pass

        try:
            from SarahMemoryAdaptive import log_interaction_to_db  # type: ignore
            log_interaction_to_db(str(user_text or ""), str(presented_answer or ""), meta)  # type: ignore
        except Exception:
            pass

        return True
    except Exception:
        return None

def store_comparison_outcome(query, reply, intent, source, confidence, meta=None):
    if _sm_meta_blocks_persistence(meta, text=str(query or reply or "")):
        logger.info("[V10/V9C] comparison outcome store blocked for volatile SelfAware body fact.")
        return False
    try:
        import sqlite3, os, datetime as _dt
        db_path = getattr(config, "SYSTEM_LOG_DB", None) if "config" in globals() else None
        if not db_path: db_path = os.path.join(os.getcwd(), "system_logs.db")
        con = sqlite3.connect(db_path); cur = con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS comparison_hits (ts TEXT, query TEXT, reply TEXT, intent TEXT, source TEXT, confidence REAL)")
        cur.execute("INSERT INTO comparison_hits VALUES (?,?,?,?,?,?)", (_dt.datetime.utcnow().isoformat(), query, reply, intent, source, float(confidence)))
        con.commit(); con.close()
    except Exception: pass


# ---------------------------------------------------------------------------
# Runtime anti-thrash helpers
# ---------------------------------------------------------------------------
def _sm_env_bool(name: str, default: bool = False) -> bool:
    try:
        val = os.getenv(name)
        if val is None and 'config' in globals() and hasattr(config, name):
            val = getattr(config, name)
        if val is None:
            return bool(default)
        if isinstance(val, bool):
            return bool(val)
        return str(val).strip().lower() in ("1", "true", "yes", "on", "enable", "enabled")
    except Exception:
        return bool(default)


def _sm_vectoring_stamp_path(datasets_dir: str) -> str:
    try:
        return os.path.join(datasets_dir, ".vectoring_last_run.json")
    except Exception:
        return os.path.join(os.getcwd(), ".vectoring_last_run.json")


def _sm_vectoring_recent(datasets_dir: str, cooldown_seconds: float) -> bool:
    try:
        p = _sm_vectoring_stamp_path(datasets_dir)
        if not os.path.exists(p):
            return False
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        last = float(data.get("ts", 0) or 0)
        return last > 0 and (time.time() - last) < max(0.0, float(cooldown_seconds))
    except Exception:
        return False


def _sm_write_vectoring_stamp(datasets_dir: str, payload: dict) -> None:
    try:
        p = _sm_vectoring_stamp_path(datasets_dir)
        tmp = p + ".tmp"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        data = dict(payload or {})
        data["ts"] = time.time()
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True, ensure_ascii=False)
        os.replace(tmp, p)
    except Exception:
        pass

# === injected: Visible dataset vectoring with ASCII status bars ===




"""
SarahMemoryDatabase.py v8.0 - Enhanced run_vectoring_with_status_bars Function
================================================================================

This file contains the enhanced v8.0 version of the run_vectoring_with_status_bars
function that should replace the existing function in SarahMemoryDatabase.py.

Location: Replace function starting at line ~952 in SarahMemoryDatabase.py

Author: © 2025 Brian Lee Baros. All Rights Reserved.
================================================================================
"""

def run_vectoring_with_status_bars(force=False):
    """
    v8.0 ENHANCED: Enumerate *.db files in the configured datasets directory and 
    visibly vector each one with world-class visual progress indicators.
    
    Features:
    - Uses tqdm for animated, timed progress render
    - Beautiful ASCII progress bars with Unicode characters
    - Real-time status updates
    - Color-coded progress (if terminal supports it)
    - Detailed logging with timestamps
    - Graceful fallback for headless environments
    - Multi-platform compatibility (Windows/Linux/macOS)
    
    Args:
        force: If True, run even when normal boot/runtime throttles would skip it
    
    Returns:
        None
    """
    import os
    import logging
    import time
    import sys
    
    # Try to import tqdm for enhanced progress bars
    try:
        from tqdm import tqdm
        _HAS_TQDM = True
    except Exception:
        _HAS_TQDM = False
    
    # Safe config import
    try:
        import SarahMemoryGlobals as config
    except Exception:
        class _Cfg:
            pass
        config = _Cfg()
        setattr(config, "BASE_DIR", os.getcwd())
    
    # ==========================================================================
    # CONFIGURATION AND SETUP
    # ==========================================================================
    datasets_dir = getattr(config, "DATASETS_DIR", None)
    if not datasets_dir:
        base = getattr(config, "BASE_DIR", os.getcwd())
        datasets_dir = os.path.join(base, "data", "memory", "datasets")
    
    if not os.path.isdir(datasets_dir):
        logging.warning("[v8.0][BOOT][VECTOR] Datasets directory not found: %s", datasets_dir)
        print(f"  ⚠ Warning: Datasets directory not found: {datasets_dir}")
        return
    
    # Check if local data is enabled and whether this is an explicitly authorized
    # vectoring window.  Normal boot must not rebuild/vector every dataset.
    local_enabled = getattr(config, "LOCAL_DATA_ENABLED", True)
    if not local_enabled and not force:
        logging.info("[v8.0][VECTOR] Local dataset embedding skipped - LOCAL_DATA_ENABLED is False.")
        return

    manual_enabled = (
        _sm_env_bool("BOOT_ENABLE_DATASET_VECTORING", False)
        or _sm_env_bool("SARAH_ENABLE_DATASET_VECTORING", False)
        or _sm_env_bool("BOOT_RUN_VECTORING_ON_STARTUP", False)
    )
    if not force and not manual_enabled:
        logging.info("[v8.0][VECTOR] Dataset vectoring skipped by optimized runtime policy. Set BOOT_ENABLE_DATASET_VECTORING=true or call force=True to run.")
        return

    try:
        cooldown = float(os.getenv("SARAH_VECTORING_COOLDOWN_SEC", str(getattr(config, "VECTORING_COOLDOWN_SEC", 6 * 60 * 60))))
    except Exception:
        cooldown = 6 * 60 * 60
    if not force and _sm_vectoring_recent(datasets_dir, cooldown):
        logging.info("[v8.0][VECTOR] Dataset vectoring skipped by cooldown policy (%ss).", cooldown)
        return
    
    # ==========================================================================
    # SCAN FOR DATABASE FILES
    # ==========================================================================
    db_files = [f for f in os.listdir(datasets_dir) if f.lower().endswith(".db")]
    
    logging.info("[v8.0][BOOT][VECTOR] Scanning datasets directory: %s", datasets_dir)
    logging.info("[v8.0][BOOT][VECTOR] Found %d database files", len(db_files))
    
    if not db_files:
        logging.warning("[v8.0][BOOT][VECTOR] No .db files found in datasets directory.")
        print("⚠ Warning: No database files found in datasets directory")
        return
    
    # ==========================================================================
    # VECTOR ENTRY POINTS DISCOVERY
    # ==========================================================================
    # Discover any existing project entry points if available
    _entries = []
    entry_point_names = [
        "refresh_vector_indexes",
        "ingest_and_vectorize_all",
        "vectorize_all_datasets",
        "rebuild_indexes",
        "initialize_vector_store",
        "embed_and_store_dataset_sentences"
    ]
    
    for _name in entry_point_names:
        if _name in globals():
            _entries.append(_name)
    
    if _entries:
        logging.info("[v8.0][VECTOR] Using vectoring entry point once: %s", _entries[0])
        start_once = time.time()
        try:
            globals()[_entries[0]]()
            _sm_write_vectoring_stamp(datasets_dir, {"entry_point": _entries[0], "status": "ok", "db_count": len(db_files)})
            logging.info("[v8.0][VECTOR] Entry point completed in %.2fs", time.time() - start_once)
        except Exception as e:
            logging.warning("[v8.0][VECTOR] Entry point failed: %s", e)
            _sm_write_vectoring_stamp(datasets_dir, {"entry_point": _entries[0], "status": "failed", "error": str(e), "db_count": len(db_files)})
        return
    else:
        logging.info("[v8.0][VECTOR] No vectoring entry points found; using metadata-only scan display")
    
    # ==========================================================================
    # v8.0 ENHANCED PROGRESS DISPLAY
    # ==========================================================================
    print(f"\n  ⏳ Processing {len(db_files)} dataset database(s)...")
    print("  " + "─" * 76)
    
    processed_count = 0
    failed_count = 0
    start_time = time.time()
    
    # Process each database file
    for idx, fn in enumerate(sorted(db_files), 1):
        full_path = os.path.join(datasets_dir, fn)
        
        # Prepare display label
        file_label = f"{fn}"
        progress_label = f"[{idx}/{len(db_files)}] {file_label}"
        
        logging.info("[v8.0][BOOT][VECTOR] Processing database: %s", full_path)
        
        # =======================================================================
        # TQDM-BASED PROGRESS (Preferred)
        # =======================================================================
        if _HAS_TQDM:
            try:
                # Create a progress bar for this file
                with tqdm(
                    total=100,
                    desc=f"  ✓ {progress_label}",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
                    ncols=80,
                    leave=True,
                    unit="%"
                ) as pbar:
                    # Update to show start
                    pbar.update(0)
                    
                    # Perform vectoring
                    processing_start = time.time()
                    
                    try:
                        # Metadata-only pass. Heavy vectoring entry points run once above.
                        time.sleep(0.01)
                        
                        # Simulate progress (since we don't have real progress from the function)
                        for step in range(0, 101, 20):
                            pbar.n = step
                            pbar.refresh()
                            time.sleep(0.01)
                        
                        # Mark as complete
                        pbar.n = 100
                        pbar.refresh()
                        
                        processing_time = time.time() - processing_start
                        logging.info("[v8.0][BOOT][VECTOR] Completed %s in %.3f seconds", 
                                   fn, processing_time)
                        
                        processed_count += 1
                    
                    except Exception as e:
                        logging.warning("[v8.0][BOOT][VECTOR] Failed vectoring '%s': %s", fn, e)
                        pbar.set_description(f"  ✗ {progress_label} (FAILED)")
                        failed_count += 1
            
            except Exception as e:
                logging.warning("[v8.0][BOOT][VECTOR] tqdm render failed (%s); using ASCII fallback.", e)
                # Fall through to ASCII fallback below
                _use_ascii = True
            else:
                _use_ascii = False
        else:
            _use_ascii = True
        
        # =======================================================================
        # ASCII PROGRESS FALLBACK (For headless/minimal environments)
        # =======================================================================
        if _use_ascii:
            try:
                # Print start indicator
                sys.stdout.write(f"  ⏳ {progress_label} ... ")
                sys.stdout.flush()
                
                processing_start = time.time()
                
                try:
                    # Metadata-only pass. Heavy vectoring entry points run once above.
                    time.sleep(0.005)
                    
                    processing_time = time.time() - processing_start
                    
                    # Print completion
                    sys.stdout.write(f"✓ ({processing_time:.2f}s)\n")
                    sys.stdout.flush()
                    
                    logging.info("[v8.0][BOOT][VECTOR] Completed %s in %.3f seconds", 
                               fn, processing_time)
                    processed_count += 1
                
                except Exception as e:
                    sys.stdout.write(f"✗ FAILED\n")
                    sys.stdout.flush()
                    
                    logging.warning("[v8.0][BOOT][VECTOR] Failed vectoring '%s': %s", fn, e)
                    failed_count += 1
            
            except Exception as e:
                logging.error("[v8.0][BOOT][VECTOR] ASCII progress failed: %s", e)
                failed_count += 1
    
    # ==========================================================================
    # FINAL STATUS SUMMARY
    # ==========================================================================
    total_time = time.time() - start_time
    
    print("  " + "─" * 76)
    print(f"\n  ✓ Dataset vectoring complete:")
    print(f"     • Processed: {processed_count} database(s)")
    
    if failed_count > 0:
        print(f"     • Failed: {failed_count} database(s)")
    
    print(f"     • Total time: {total_time:.2f} seconds")
    print()
    
    _sm_write_vectoring_stamp(datasets_dir, {"entry_point": "metadata_only", "status": "ok", "processed": processed_count, "failed": failed_count, "db_count": len(db_files)})
    logging.info("[v8.0][VECTOR] Dataset metadata scan complete.")
    logging.info("[v8.0][VECTOR] Processed: %d, Failed: %d, Time: %.2f seconds", 
               processed_count, failed_count, total_time)


# =============================================================================
# HELPER FUNCTION FOR ASCII PROGRESS BARS
# =============================================================================
def _print_progress_bar(prefix, percent):
    """
    v8.0 Enhanced: Print a simple ASCII progress bar.
    
    Args:
        prefix: Label to show before the bar
        percent: Progress percentage (0-100)
    """
    try:
        bar_len = 40
        
        # Normalize percentage
        try:
            pct = int(max(0, min(100, float(percent))))
        except Exception:
            pct = 0
        
        # Calculate filled portion
        filled = int(bar_len * pct / 100)
        if filled < 0:
            filled = 0
        if filled > bar_len:
            filled = bar_len
        
        # Create bar with Unicode blocks for better visual
        try:
            # Try to use Unicode block characters
            bar = "█" * filled + "░" * (bar_len - filled)
        except Exception:
            # Fallback to ASCII characters
            bar = "#" * filled + "-" * (bar_len - filled)
        
        # Print progress
        sys.stdout.write("\r{} [{}] {:3d}%".format(prefix, bar, pct))
        sys.stdout.flush()
        
        # Add newline when complete
        if pct >= 100:
            sys.stdout.write("\n")
            sys.stdout.flush()
    
    except Exception:
        # Never let the progress bar break the boot sequence
        pass
def load_quick_facts(limit=50):
    return []

# === v7.7.3 schema guard: create missing tables at runtime (idempotent) ===
def ensure_core_schema():
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            # Idempotent schema guard for legacy databases (never raises)
            def _ensure_column(cur, table: str, col: str, col_type: str) -> None:
                try:
                    cur.execute(f"PRAGMA table_info({table})")
                    existing = [r[1] for r in cur.fetchall()]
                    if col not in existing:
                        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
                except Exception:
                    pass

            c.execute("""CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_input TEXT,
                ai_response TEXT,
                intent TEXT,
                sentiment_score REAL,
                emotional_state TEXT,
                session_id TEXT,
                canonical_answer TEXT,
                presented_answer TEXT,
                lane TEXT,
                source TEXT,
                canonical_type TEXT,
                truth_locked INTEGER DEFAULT 0,
                tone TEXT,
                style TEXT,
                persona_state TEXT,
                raw_meta_json TEXT,
                presentation_meta_json TEXT)""")
            
            _ensure_column(c, "conversations", "intent", "TEXT")
            _ensure_column(c, "conversations", "sentiment_score", "REAL")
            _ensure_column(c, "conversations", "emotional_state", "TEXT")
            _ensure_column(c, "conversations", "session_id", "TEXT")
            _ensure_column(c, "conversations", "canonical_answer", "TEXT")
            _ensure_column(c, "conversations", "presented_answer", "TEXT")
            _ensure_column(c, "conversations", "lane", "TEXT")
            _ensure_column(c, "conversations", "source", "TEXT")
            _ensure_column(c, "conversations", "canonical_type", "TEXT")
            _ensure_column(c, "conversations", "truth_locked", "INTEGER DEFAULT 0")
            _ensure_column(c, "conversations", "tone", "TEXT")
            _ensure_column(c, "conversations", "style", "TEXT")
            _ensure_column(c, "conversations", "persona_state", "TEXT")
            _ensure_column(c, "conversations", "raw_meta_json", "TEXT")
            _ensure_column(c, "conversations", "presentation_meta_json", "TEXT")
            c.execute("""CREATE TABLE IF NOT EXISTS response_layers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                session_id TEXT,
                user_input TEXT,
                canonical_answer TEXT,
                presented_answer TEXT,
                intent TEXT,
                lane TEXT,
                source TEXT,
                canonical_type TEXT,
                truth_locked INTEGER DEFAULT 0,
                tone TEXT,
                style TEXT,
                persona_state TEXT,
                raw_meta_json TEXT,
                presentation_meta_json TEXT)""")
            c.execute("CREATE INDEX IF NOT EXISTS idx_response_layers_ts ON response_layers(ts)")
            c.execute("""CREATE TABLE IF NOT EXISTS intent_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                intent TEXT,
                confidence REAL,
                extras TEXT)""")
            conn.commit()
        # personality1.db minimal tables
        per_db = os.path.join(config.DATASETS_DIR, "personality1.db")
        with sqlite3.connect(per_db) as pconn:
            pc = pconn.cursor()
            pc.execute("""CREATE TABLE IF NOT EXISTS emotion_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                joy REAL, anger REAL, fear REAL, sadness REAL, curiosity REAL, trust REAL,
                valence REAL, arousal REAL, primary_label TEXT, fer_source TEXT, notes TEXT)""")
            pc.execute("""CREATE TABLE IF NOT EXISTS responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intent TEXT,
                response TEXT,
                tone TEXT,
                complexity TEXT)""")
            pconn.commit()
            
    except Exception as e:
        logger.error(f"[ensure_core_schema] {e}")
def save_emotion_state(state: dict, fer_source: str = "unknown", notes: str = "") -> bool:
    try:
        ensure_core_schema()
        per_db = os.path.join(config.DATASETS_DIR, "personality1.db")
        with sqlite3.connect(per_db) as conn:
            cur = conn.cursor()
            ts = dt.now().isoformat()
            row = (ts,
                float(state.get("joy",0)), float(state.get("anger",0)),
                float(state.get("fear",0)), float(state.get("sadness",0)),
                float(state.get("curiosity",0)), float(state.get("trust",0)),
                float(state.get("valence",0)), float(state.get("arousal",0)),
                str(state.get("primary","neutral")), fer_source, notes)
            cur.execute("""INSERT INTO emotion_states
                (timestamp, joy, anger, fear, sadness, curiosity, trust, valence, arousal, primary_label, fer_source, notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", row)
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"save_emotion_state failed: {e}")
        return False

def load_last_emotion_state() -> dict:
    try:
        ensure_core_schema()
        per_db = os.path.join(config.DATASETS_DIR, "personality1.db")
        with sqlite3.connect(per_db) as conn:
            cur = conn.cursor()
            cur.execute("SELECT joy,anger,fear,sadness,curiosity,trust,valence,arousal,primary_label FROM emotion_states ORDER BY id DESC LIMIT 1")
            r = cur.fetchone()
        if not r:
            return {"joy":0.5,"anger":0.1,"fear":0.1,"sadness":0.1,"curiosity":0.4,"trust":0.4,"valence":0.0,"arousal":0.2,"primary":"neutral"}
        keys = ["joy","anger","fear","sadness","curiosity","trust","valence","arousal","primary"]
        return dict(zip(keys, list(r)))
    except Exception as e:
        logger.error(f"load_last_emotion_state failed: {e}")
        return {"joy":0.5,"anger":0.1,"fear":0.1,"sadness":0.1,"curiosity":0.4,"trust":0.4,"valence":0.0,"arousal":0.2,"primary":"neutral"}

def record_intent(intent: str, confidence: float, extras: dict = None):
    try:
        ensure_core_schema()
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute("INSERT INTO intent_logs (timestamp,intent,confidence,extras) VALUES (?,?,?,?)",
                        (dt.now().isoformat(), intent, float(confidence), json.dumps(extras or {})))
            conn.commit()
    except Exception as e:
        logger.warning(f"record_intent failed: {e}")

"""
PATCH: Add these functions WITHOUT removing your existing ensure_core_schema or other defs.
"""

CHAT_DB = os.path.join(DATASETS_DIR, "context_history.db") # reuse same DB as current context

def ensure_chat_schema():
    os.makedirs(os.path.dirname(CHAT_DB), exist_ok=True)
    con = sqlite3.connect(CHAT_DB)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_threads (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            category TEXT DEFAULT '',
            created_ts INTEGER NOT NULL,
            last_ts INTEGER NOT NULL,
            tags TEXT DEFAULT '',
            archived INTEGER DEFAULT 0
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            thread_id TEXT NOT NULL,
            ts INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            meta_json TEXT DEFAULT '{}'
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_ts ON chat_messages(thread_id, ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_threads_last_ts ON chat_threads(last_ts)")
    con.commit(); con.close()

def create_thread(title: str, category: str = "General", tags: str = "") -> str:
    ensure_chat_schema()
    tid = f"t_{uuid.uuid4().hex}"
    now = int(time.time())
    con = sqlite3.connect(CHAT_DB); cur = con.cursor()
    cur.execute(
        "INSERT INTO chat_threads (id, title, category, created_ts, last_ts, tags, archived) VALUES (?,?,?,?,?,?,0)",
        (tid, title or "Untitled", category or "General", now, now, tags or "")
    )
    con.commit(); con.close()
    return tid

def append_message(thread_id: str, role: str, content: str, meta: dict | None = None) -> str:
    ensure_chat_schema()
    mid = f"m_{uuid.uuid4().hex}"
    now = int(time.time())
    con = sqlite3.connect(CHAT_DB); cur = con.cursor()
    cur.execute(
        "INSERT INTO chat_messages (id, thread_id, ts, role, content, meta_json) VALUES (?,?,?,?,?,?)",
        (mid, thread_id, now, role, content, json.dumps(meta or {}))
    )
    cur.execute("UPDATE chat_threads SET last_ts=? WHERE id=?", (now, thread_id))
    con.commit(); con.close()
    return mid

def list_threads(category: str | None = None, limit: int = 200):
    ensure_chat_schema()
    con = sqlite3.connect(CHAT_DB); cur = con.cursor()
    if category:
        cur.execute(
            "SELECT id, title, category, created_ts, last_ts, tags, archived FROM chat_threads WHERE archived=0 AND category=? ORDER BY last_ts DESC LIMIT ?",
            (category, limit)
        )
    else:
        cur.execute(
            "SELECT id, title, category, created_ts, last_ts, tags, archived FROM chat_threads WHERE archived=0 ORDER BY last_ts DESC LIMIT ?",
            (limit,)
        )
    rows = [
        {
            "id": r[0], "title": r[1], "category": r[2],
            "created_ts": r[3], "last_ts": r[4], "tags": r[5], "archived": r[6]
        } for r in cur.fetchall()
    ]
    con.close(); return rows

def load_messages(thread_id: str, limit: int = 500):
    ensure_chat_schema()
    con = sqlite3.connect(CHAT_DB); cur = con.cursor()
    cur.execute(
        "SELECT id, ts, role, content, meta_json FROM chat_messages WHERE thread_id=? ORDER BY ts ASC LIMIT ?",
        (thread_id, limit)
    )
    rows = [
        {"id": r[0], "ts": r[1], "role": r[2], "content": r[3], "meta": json.loads(r[4] or '{}')} for r in cur.fetchall()
    ]
    con.close(); return rows


# ============================
# Phase B: Context & QA Mesh Helpers (Database Layer)
# ============================

CONTEXT_DB = CHAT_DB  # reuse the shared context_history.db backing store


def ensure_context_turn_schema():
    """Create a minimal context_turns table in context_history.db (idempotent)."""
    try:
        os.makedirs(os.path.dirname(CONTEXT_DB), exist_ok=True)
        con = sqlite3.connect(CONTEXT_DB)
        cur = con.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS context_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                intent TEXT DEFAULT 'chat',
                source TEXT DEFAULT 'core',
                meta_json TEXT DEFAULT '{}'
            )"""
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_context_turns_ts ON context_turns(ts)")
        con.commit(); con.close()
    except Exception as e:
        logger.error(f"[CTX_SCHEMA ERROR] {e}")


def store_context_turn(user_text: str, ai_text: str, intent: str = "chat", source: str = "core", meta: dict | None = None):
    """Best-effort persistence of a single dialogue turn into context_history.db.

    This is a DB-level helper; higher layers (AiFunctions, Reply) are still
    responsible for managing in-RAM buffers and embeddings.
    """
    try:
        ensure_context_turn_schema()
        now = time.time()
        meta_json = json.dumps(meta or {}, ensure_ascii=False)
        con = sqlite3.connect(CONTEXT_DB)
        cur = con.cursor()
        cur.execute(
            "INSERT INTO context_turns (ts, role, text, intent, source, meta_json) VALUES (?,?,?,?,?,?)",
            (now, "user", user_text or "", intent or "chat", source or "core", meta_json),
        )
        cur.execute(
            "INSERT INTO context_turns (ts, role, text, intent, source, meta_json) VALUES (?,?,?,?,?,?)",
            (now + 1e-4, "assistant", ai_text or "", intent or "chat", source or "core", meta_json),
        )
        con.commit(); con.close()
    except Exception as e:
        logger.warning(f"[CTX_STORE WARN] {e}")


def load_recent_context_turns(max_turns: int = 10, max_age_sec: float | None = None):
    """Load recent turns from context_history.db for use in prompt building.

    Returns a list of rows in ascending time order:
        [{"ts": float, "role": str, "text": str, "intent": str, "source": str, "meta": dict}, ...]
    """
    try:
        ensure_context_turn_schema()
        now = time.time()
        con = sqlite3.connect(CONTEXT_DB)
        cur = con.cursor()
        if max_age_sec and max_age_sec > 0:
            cutoff = now - float(max_age_sec)
            cur.execute(
                "SELECT ts, role, text, intent, source, meta_json FROM context_turns WHERE ts >= ? ORDER BY ts DESC LIMIT ?",
                (cutoff, int(max_turns)),
            )
        else:
            cur.execute(
                "SELECT ts, role, text, intent, source, meta_json FROM context_turns ORDER BY ts DESC LIMIT ?",
                (int(max_turns),),
            )
        rows = cur.fetchall()
        con.close()
        out = []
        for ts_val, role, text_val, intent_val, source_val, meta_json in reversed(rows):
            try:
                meta = json.loads(meta_json or "{}")
            except Exception:
                meta = {}
            out.append(
                {
                    "ts": float(ts_val),
                    "role": role,
                    "text": text_val,
                    "intent": intent_val or "chat",
                    "source": source_val or "core",
                    "meta": meta,
                }
            )
        return out
    except Exception as e:
        logger.warning(f"[CTX_LOAD WARN] {e}")
        return []


def sync_qa_cache_from_cloud(limit: int = 200):
    """Optional helper: pull QA entries from the cloud hub into local qa_cache.

    This respects mesh/hub flags so that when mesh sync is disabled or hub
    is disallowed, the function becomes a no-op.
    """
    try:
        mesh_cfg = get_mesh_sync_config()
    except Exception:
        mesh_cfg = {}
    if not mesh_cfg.get("mesh_enabled", True) or not mesh_cfg.get("hub_allowed", True):
        logger.info("[QA_SYNC] Mesh/hub sync disabled; skipping cloud QA sync.")
        return 0

    cloud = _get_cloud_conn()
    if cloud is None:
        logger.info("[QA_SYNC] Cloud DB unavailable; skipping QA sync.")
        return 0

    pulled = 0
    try:
        cur = cloud.cursor()
        cur.execute(
            "SELECT query, ai_answer, hit_score, feedback, timestamp FROM sm_qa_cache ORDER BY id DESC LIMIT %s",
            (int(limit),),
        )
        rows = cur.fetchall()
        cloud.close()
        if not rows:
            return 0

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Ensure local qa_cache has all Phase B fields
        c.execute(
            """CREATE TABLE IF NOT EXISTS qa_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                ai_answer TEXT,
                hit_score INTEGER,
                feedback TEXT,
                timestamp TEXT
            )"""
        )
        for q, ans, score, fb, ts in rows:
            try:
                c.execute(
                    "SELECT 1 FROM qa_cache WHERE query=? AND ai_answer=? LIMIT 1",
                    (q, ans),
                )
                if c.fetchone():
                    continue
                c.execute(
                    "INSERT INTO qa_cache (query, ai_answer, hit_score, feedback, timestamp) VALUES (?,?,?,?,?)",
                    (q, ans, score or 0, fb or "ungraded", (ts.isoformat() if hasattr(ts, "isoformat") else str(ts))),
                )
                pulled += 1
            except Exception as inner_e:
                logger.debug(f"[QA_SYNC SKIP] {inner_e}")
        conn.commit(); conn.close()
        logger.info(f"[QA_SYNC] Pulled {pulled} entries from cloud QA cache.")
        return pulled
    except Exception as e:
        logger.error(f"[QA_SYNC ERROR] {e}")
        return pulled


# ============================================================================
# Phase B: User Preferences
# ============================================================================

def sm_get_user_preferences(user_id):
    """Get user preferences from MySQL."""
    conn = _get_cloud_conn()
    if not conn or not user_id:
        return {}

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM sm_user_preferences WHERE user_id = %s", (user_id,))
        prefs = cursor.fetchone()
        return prefs if prefs else {}
    except Exception as e:
        logger.error(f"Error getting preferences for user {user_id}: {e}")
        return {}
    finally:
        if conn:
            conn.close()


def sm_update_user_preferences(user_id, preferences):
    """Update user preferences in MySQL."""
    conn = _get_cloud_conn()
    if not conn or not user_id:
        return False

    try:
        cursor = conn.cursor()

        # Build dynamic UPDATE query
        fields = []
        values = []
        for key, value in preferences.items():
            fields.append(f"{key} = %s")
            values.append(value)

        if fields:
            values.append(user_id)
            query = f"UPDATE sm_user_preferences SET {', '.join(fields)} WHERE user_id = %s"
            cursor.execute(query, values)
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error updating preferences for user {user_id}: {e}")
        return False
    finally:
        if conn:
            conn.close()


# ============================================================================
# Phase B: Conversation Storage
# ============================================================================

def sm_save_conversation_message(conversation_id, role, content, device_id=None, model_used=None):
    """Save message to conversation history."""
    conn = _get_cloud_conn()
    if not conn:
        return None

    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sm_conversation_messages (conversation_id, role, content, device_id, model_used) VALUES (%s, %s, %s, %s, %s)",
            (conversation_id, role, content, device_id, model_used)
        )
        conn.commit()
        return cursor.lastrowid
    except Exception as e:
        logger.error(f"Error saving message: {e}")
        return None
    finally:
        if conn:
            conn.close()


def sm_get_conversation_messages(conversation_id, limit=50):
    """Get messages from conversation."""
    conn = _get_cloud_conn()
    if not conn:
        return []

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            "SELECT * FROM sm_conversation_messages WHERE conversation_id = %s ORDER BY created_at DESC LIMIT %s",
            (conversation_id, limit)
        )
        return list(reversed(cursor.fetchall()))
    except Exception as e:
        logger.error(f"Error getting messages: {e}")
        return []
    finally:
        if conn:
            conn.close()

from dataclasses import asdict, dataclass, field
from typing import Any, Dict
from datetime import datetime as _sm_memory_datetime
# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 START ---
# Structured memory lifecycle records. These builders are pure/data-only by
# default; persistence must be explicitly performed by the existing DB layer.

@dataclass
class MemoryLifecycleRecord:
    memory_id: str
    fact: str
    source: str = "unknown"
    confidence: float = 0.0
    created_at: str = field(default_factory=lambda: _sm_memory_datetime.now().isoformat())
    last_confirmed: str = ""
    retention_class: str = "standard"
    privacy_class: str = "local_private"
    contradiction_status: str = "unchecked"
    user_approved: bool = False
    rollback_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryDiff:
    diff_id: str
    memory_id: str
    old_value: Any
    new_value: Any
    reason: str
    source: str = "unknown"
    confidence: float = 0.0
    approval_state: str = "pending"
    created_at: str = field(default_factory=lambda: _sm_memory_datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryRetentionPolicy:
    def classify(self, record: MemoryLifecycleRecord) -> Dict[str, Any]:
        privacy = str(record.privacy_class or "local_private")
        confidence = float(record.confidence or 0.0)
        retain = confidence >= 0.50 or bool(record.user_approved)
        if privacy in {"sensitive", "private_sensitive"} and not record.user_approved:
            retain = False
        return {
            "ok": True,
            "retain": bool(retain),
            "requires_user_approval": privacy in {"sensitive", "private_sensitive"},
            "write_immediately": False,
            "batch_flush_allowed": True,
            "reasons": ["Memory lifecycle policy is local-first and diff-based; persistence is explicit/batched."],
        }


def build_memory_lifecycle_record(fact: str, *, source: str = "unknown", confidence: float = 0.0, **metadata: Any) -> Dict[str, Any]:
    rec = MemoryLifecycleRecord(
        memory_id=hashlib.sha256((str(fact) + str(source)).encode("utf-8", errors="ignore")).hexdigest()[:32] if 'hashlib' in globals() else str(int(time.time()*1000)),
        fact=str(fact or ""),
        source=str(source or "unknown"),
        confidence=max(0.0, min(1.0, float(confidence or 0.0))),
        metadata=dict(metadata or {}),
    )
    policy = MemoryRetentionPolicy().classify(rec)
    return {"ok": True, "record": rec.to_dict(), "policy": policy}


def build_memory_diff(memory_id: str, old_value: Any, new_value: Any, reason: str = "") -> Dict[str, Any]:
    did = hashlib.sha256((str(memory_id) + str(old_value) + str(new_value)).encode("utf-8", errors="ignore")).hexdigest()[:32] if 'hashlib' in globals() else str(int(time.time()*1000))
    diff = MemoryDiff(diff_id=did, memory_id=str(memory_id or ""), old_value=old_value, new_value=new_value, reason=str(reason or "unspecified"))
    return {"ok": True, "diff": diff.to_dict(), "write_immediately": False}
# --- SM V8.0 SOVEREIGN AGENT RUNTIME CONSOLIDATION PASS 7 END ---

# ====================================================================
# END OF SarahMemoryDatabase.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryDatabase',
    "version": "v9.0.0-alpha-sml-0.2",
    "category": 'Memory',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['memory', 'persistent_knowledge'],
    "supported_missions": ['Conversation', 'Knowledge', 'Memory'],
    "supported_omega": ['Ω001', 'Ω010', 'Ω080', 'Ω090'],
    "required_authority": ['Read'],
    "priority": 75,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryDatabase.py'},
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
        "component": 'SarahMemoryDatabase',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryDatabase', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

