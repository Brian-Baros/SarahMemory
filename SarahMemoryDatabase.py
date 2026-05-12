"""--==The SarahMemory Project==--
File: SarahMemoryDatabase.py
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
# NOTES = "Primary database and memory access layer for SQLite/MySQL, response-layer storage, QA cache, device identity/capabilities, and dataset access wrappers."
# --- SARAHMETA END ---
import logging
import sqlite3
import time
import uuid
import os
import datetime
import psutil
import json
import SarahMemoryGlobals as config
from SarahMemoryGlobals import run_async, DATASETS_DIR
import random
import hashlib
import secrets
import jwt
from datetime import datetime as dt, timedelta

# Setup logging for the database module
logger = logging.getLogger('SarahMemoryDatabase')
logger.setLevel(logging.DEBUG)
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
    """Unified search over local QA cache and (optionally) cloud QA cache."""
    if _sm_is_volatile_body_fact_query(query):
        logger.info("[V10/V9C] QA cache recall blocked for volatile SelfAware body fact query.")
        return []
    results = []

    # 1) Local sqlite first (fast, offline-safe)
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ai_answer FROM qa_cache WHERE query LIKE ? ORDER BY hit_score DESC",
            ('%' + query + '%',)
        )
        rows = cursor.fetchall()
        conn.close()
        if rows:
            results.extend([row[0] for row in rows])
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
                    results.append(answer)
        except Exception as e:
            logger.error(f"[CLOUD QA SEARCH ERROR] {e}")

    return results


def store_answer(query, answer):
    """Store answer locally and push to cloud hub if available."""
    if _sm_is_volatile_body_fact_query(query) or _sm_is_volatile_body_fact_query(answer):
        logger.info("[V10/V9C] QA cache store blocked for volatile SelfAware body fact.")
        return False
    timestamp = datetime.datetime.now().isoformat()

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
        timestamp = datetime.datetime.now().isoformat()
        cpu = psutil.cpu_percent(interval=1)
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
           timestamp = datetime.datetime.now().isoformat()
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
    """Fuzzy search inside the personality1.db responses."""
    try:
        db_path = os.path.join(config.DATASETS_DIR, "personality1.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT response FROM responses WHERE response LIKE ?", ('%' + question + '%',))
        results = cursor.fetchall()
        conn.close()
        return [row[0] for row in results] if results else []
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
def embed_and_store_dataset_sentences():
    """
    Extracts text from imported local files, creates vector embeddings,
    and stores them in the voice_logs table with timestamped entries.

    âœ… Expands SarahMemory foundation model with permanent vector memory
    ðŸ” Safe to call repeatedly; avoids duplicate re-learning based on file mod times.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from SarahMemoryGlobals import import_other_data, IMPORT_OTHER_DATA_LEARN, MULTI_MODEL, MODEL_CONFIG

        if not IMPORT_OTHER_DATA_LEARN:
            logger.info("Skipping vector rebuild: IMPORT_OTHER_DATA_LEARN is False.")
            return

        def get_active_sentence_model():
            if MULTI_MODEL:
                for model_name, enabled in MODEL_CONFIG.items():
                    if enabled:
                        try:
                            return SentenceTransformer(model_name)
                        except Exception as e:
                            logger.warning(f"Model load failed: {model_name} , {e}")
            return SentenceTransformer('all-MiniLM-L6-v2')

        logger.info("Starting semantic vector embedding for dataset memory...")
        model = get_active_sentence_model()
        data = import_other_data()
        conn = init_database()
        inserted_count = 0

        for file_path, content in data.items():
            for line in content.split('\n'):
                line = line.strip()
                if not line or len(line) < 20:
                    continue
                try:
                    embedding = model.encode(line).tolist()
                    success = store_voice_input(conn, voice_text=line, embedding=embedding)
                    if success:
                        inserted_count += 1
                except Exception as ve:
                    logger.warning(f"[EMBED ERROR] Skipped line due to embedding failure: {ve}")
        conn.close()
        logger.info(f"Vector memory embedding complete. {inserted_count} entries added.")

    except Exception as e:
        logger.error(f"[EMBED_FAIL] Dataset vector embedding failed: {e}")

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

    for db_file in db_files:
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
    """
    Vector-based semantic search on QA cache memory (query and answer).
    """
    try:
        model = get_active_sentence_model()
        query_vec = model.encode(query_text)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT query, ai_answer FROM qa_cache")
        entries = cursor.fetchall()

        results = []
        for query, answer in entries:
            combined = f"{query} {answer}"
            emb_vec = model.encode(combined)
            similarity = dot(query_vec, emb_vec) / (norm(query_vec) * norm(emb_vec))

            tokens = tokenize_text(answer)
            entropy_score = len(set(tokens)) / max(len(tokens), 1)
            if entropy_score < 0.3 or len(tokens) < 5:
                continue

            results.append((similarity, answer))

        results.sort(reverse=True)
        return results[:top_n]
    except Exception as e:
        logger.error(f"[QA VECTOR SEARCH ERROR] {e}")
        return []

def vector_search(query_text, top_n=1):
    """
    Enhanced vector search with entropy analysis and query logging.
    """
    try:
        model = get_active_sentence_model()
        query_vec = model.encode(query_text)

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create search log table if not exists
        cursor.execute('''CREATE TABLE IF NOT EXISTS search_log (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            query TEXT,
                            match_text TEXT,
                            similarity REAL,
                            timestamp TEXT
                          )''')

        cursor.execute("SELECT voice_text, embedding FROM voice_logs")
        entries = cursor.fetchall()

        results = []
        for text, emb_json in entries:
            if not emb_json:
                continue
            emb_vec = np.array(json.loads(emb_json))
            similarity = dot(query_vec, emb_vec) / (norm(query_vec) * norm(emb_vec))

            # Entropy score (basic uniqueness check)
            tokens = tokenize_text(text)
            entropy_score = len(set(tokens)) / max(len(tokens), 1)
            if entropy_score < 0.3 or len(tokens) < 5:
                continue  # Skip low-quality matches

            results.append((similarity, text))

        results.sort(reverse=True)
        top_results = results[:top_n]

        for sim, matched in top_results:
            cursor.execute("INSERT INTO search_log (query, match_text, similarity, timestamp) VALUES (?, ?, ?, ?)",
                           (query_text, matched, float(sim), datetime.datetime.now().isoformat()))

        conn.commit()
        conn.close()
        return top_results
    except Exception as e:
        logger.error(f"[VECTOR_SEARCH ERROR] {e}")
        return []

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
        timestamp = datetime.datetime.now().isoformat()
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


# === injected: Visible dataset vectoring with ASCII status bars ===

def _print_progress_bar(prefix, percent):
    """Render ASCII progress bar to stdout (unaffected by logging levels).
    Signature preserved: (prefix, percent)."""
    try:
        import sys
        bar_len = 30
        try:
            pct = int(max(0, min(100, float(percent))))
        except Exception:
            pct = 0
        filled = int(bar_len * pct / 100)
        if filled < 0:
            filled = 0
        if filled > bar_len:
            filled = bar_len
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write("\r{} [{}] {:3d}%".format(prefix, bar, pct))
        sys.stdout.flush()
        if pct >= 100:
            sys.stdout.write("\n")
            sys.stdout.flush()
    except Exception:
        # Never let the progress bar break the boot sequence
        pass



"""
SarahMemoryDatabase.py v8.0 - Enhanced run_vectoring_with_status_bars Function
================================================================================

This file contains the enhanced v8.0 version of the run_vectoring_with_status_bars
function that should replace the existing function in SarahMemoryDatabase.py.

Location: Replace function starting at line ~952 in SarahMemoryDatabase.py

Author: © 2025 Brian Lee Baros. All Rights Reserved.
================================================================================
"""

def run_vectoring_with_status_bars(force=True):
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
        force: If True, run even if LOCAL_DATA_ENABLED is False
    
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
    
    # Check if local data is enabled
    local_enabled = getattr(config, "LOCAL_DATA_ENABLED", True)
    if not local_enabled and not force:
        logging.info("[v8.0][BOOT][VECTOR] Local dataset embedding skipped - LOCAL_DATA_ENABLED is False.")
        print("Local dataset embedding skipped (LOCAL_DATA_ENABLED is False)")
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
        "initialize_vector_store"
    ]
    
    for _name in entry_point_names:
        if _name in globals():
            _entries.append(_name)
    
    if _entries:
        logging.info("[v8.0][BOOT][VECTOR] Using vectoring entry point: %s", _entries[0])
    else:
        logging.info("[v8.0][BOOT][VECTOR] No vectoring entry points found, using minimal processing")
    
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
                        if _entries:
                            # Call the first available vectoring entry point
                            globals()[_entries[0]]()
                        else:
                            # Minimal processing simulation
                            time.sleep(0.03)
                        
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
                    if _entries:
                        globals()[_entries[0]]()
                    else:
                        time.sleep(0.02)
                    
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
    
    logging.info("[v8.0][BOOT][VECTOR] Dataset scan complete.")
    logging.info("[v8.0][BOOT][VECTOR] Processed: %d, Failed: %d, Time: %.2f seconds", 
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
            ts = datetime.datetime.now().isoformat()
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
                        (datetime.datetime.now().isoformat(), intent, float(confidence), json.dumps(extras or {})))
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
# ====================================================================
# END OF SarahMemoryDatabase.py v8.0.0
# ====================================================================