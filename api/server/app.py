# --==The SarahMemory Project==--
# File: /app/server/app.py
# ULTIMATE merged Flask server for SarahMemory (v8.0.0)
# Part of the SarahMemory Companion AI-bot Platform
# Author: © 2025 Brian Lee Baros. All Rights Reserved.
# www.linkedin.com/in/brian-baros-29962a176
# https://www.facebook.com/bbaros
# brian.baros@sarahmemory.com
# 'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
# https://www.sarahmemory.com
# https://api.sarahmemory.com
# https://ai.sarahmemory.com
# - Serves Web UI
# - Hub (HMAC) endpoints
# - Node registration / embeddings / context / jobs
# - Leaderboard + wallet (with Ledger module preference + local fallback)
# - Settings/Themes/Voices + Contacts + Reminders + Cleanup Tools
# - Calendar/Chat History fetchers for Web UI
# - File ingest / remote transfer
# - Camera/Mic/Voice toggles + simple telecom stubs
# - Safe fallbacks against missing core modules

from __future__ import annotations
import os, sys, json, time, glob, sqlite3, hmac, hashlib, base64
from pathlib import Path
from decimal import Decimal
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, send_file, g, session, abort
# --- Flask CORS (safe import for CLI testing & WSGI) ---
try:
    from flask_cors import CORS
    _CORS_AVAILABLE = True
except Exception as e:
    CORS = None  # type: ignore
    _CORS_AVAILABLE = False
    print("[WARN] flask_cors not available:", e)

from dotenv import load_dotenv
load_dotenv()
import re
import jwt
import bcrypt
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import wraps
from datetime import datetime, timedelta
import logging # Explicitly import logging

# ---------------------------------------------------------------------------
# Path resolution (prefer SarahMemoryGlobals; fallback to local server layout)
# ---------------------------------------------------------------------------
# Configure basic logging for the app.py directly
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
app_logger = logging.getLogger(__name__)
logger = app_logger  # consistent alias



# ------------------OLD V8 Root-----------------------
#ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#if ROOT not in sys.path:
#    sys.path.insert(0, ROOT)
#-------------------------------------------------------
# ---------------------------------------------------------------------------
# NEW V8 Root/Path resolution (prefer SarahMemoryGlobals; fallback to local server layout)
# ---------------------------------------------------------------------------

def _find_project_root(start_dir: str, max_up: int = 6) -> str:
    """
    Walk upward from start_dir to locate SarahMemoryGlobals.py (project root marker).
    This fixes cases where app.py runs from /api/server and only adds /api to sys.path.
    """
    cur = os.path.abspath(start_dir)
    for _ in range(max_up):
        marker = os.path.join(cur, "SarahMemoryGlobals.py")
        if os.path.exists(marker):
            return cur
        parent = os.path.abspath(os.path.join(cur, ".."))
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start_dir)

# Start from app.py directory
_THIS_DIR = os.path.abspath(os.path.dirname(__file__))

# Candidate roots:
# 1) parent (existing behavior)
# 2) grandparent (common: api/server -> api -> project)
# 3) auto-discovered marker walk
ROOT_PARENT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
ROOT_GRANDPARENT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
ROOT_DISCOVERED = _find_project_root(_THIS_DIR)

# Insert best root first
for p in (ROOT_DISCOVERED, ROOT_GRANDPARENT, ROOT_PARENT):
    if p and p not in sys.path:
        sys.path.insert(0, p)



# Attempt to load SarahMemoryGlobals for consistent pathing and versions
try:
    import SarahMemoryGlobals as config
    BASE_DIR = getattr(config, "BASE_DIR", os.getcwd())
    PUBLIC_DIR = getattr(config, "PUBLIC_DIR", os.path.join(BASE_DIR, "public_html"))
    WEB_DIR = getattr(config, "WEB_DIR", os.path.join(PUBLIC_DIR, "web"))
    DATA_DIR = getattr(config, "DATA_DIR", os.path.join(BASE_DIR, "data"))
    PROJECT_VERSION = getattr(config, "PROJECT_VERSION", "8.0.0") # Ensure v8.0.0 as per spec
except Exception as e:
    app_logger.warning(f"SarahMemoryGlobals (config) import failed or missing attributes. Falling back to local defaults: {e}")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # /api/server
    PUBLIC_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))  # /api
    WEB_DIR = PUBLIC_DIR  # serve index.html from /api
    DATA_DIR = os.path.join(BASE_DIR, "data")  # /api/server/data
    PROJECT_VERSION = "8.0.0" # Ensure v8.0.0 as per spec


# Identity / branding (server-side source of truth)
BRAND_NAME = "Sarah"
PLATFORM_NAME = "SarahMemory AiOS"
CREATOR_NAME = "Brian Lee Baros"
ORG_NAME = "SOFTDEV0 LLC"

def _identity_payload():
    return {
        "name": BRAND_NAME,
        "platform": PLATFORM_NAME,
        "version": PROJECT_VERSION,
        "creator": CREATOR_NAME,
        "organization": ORG_NAME,
        "build": "webui-server",
    }

def _is_identity_question(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    keys = [
        "what is your name", "who are you", "your name",
        "version", "what version", "version number",
        "who made you", "who created you", "creator",
        "who designed you", "designer", "engineer",
        "who engineered you", "who built you",
        "brian lee baros", "softdev0",
    ]
    return any(k in t for k in keys)

# Prefer server/static as templates if the SPA build exists
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(SERVER_DIR, "static")
TEMPLATE_DIR = SERVER_DIR if os.path.exists(os.path.join(STATIC_DIR, "index.html")) else WEB_DIR

# Web UI dist root (Lovable/Vite build output)
# Expected: <PROJECT_ROOT>/data/ui/v8/
UI_DIST_DIR = os.path.abspath(os.path.join(SERVER_DIR, "..", "..", "data", "ui", "v8"))
WALLETS_DIR = os.path.join(DATA_DIR, "wallets")
META_DB = os.path.join(DATA_DIR, "meta.db") # merged meta DB
LOGS_DIR = os.path.join(DATA_DIR, "logs") # Default to DATA_DIR/logs

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(WALLETS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Global runtime state (kept intentionally small and fast)
# ---------------------------------------------------------------------------
APP_VERSION = PROJECT_VERSION  # API/UI convenience alias

# Persistent state file (safe JSON, kept in DATA_DIR)
STATE_DB = os.path.join(DATA_DIR, "server_state.json")  # JSON, not sqlite
WALLET_DB = os.path.join(DATA_DIR, "wallets.db")        # sqlite (created on demand)

# Simple feature toggles (web UI can control these)
MIC_ON = True
TTS_ON = True

MIC_ENABLED = MIC_ON
TTS_ENABLED = TTS_ON
VOICE_OUTPUT_ON = TTS_ON
VOICE_OUTPUT_ENABLED = TTS_ON
# Small in-memory cache for hot endpoints (rankings/wallet/etc.)
_CACHE = {}
def _cache_get(key: str):
    item = _CACHE.get(key)
    if not item:
        return None
    value, expires_at = item
    if expires_at and time.time() > expires_at:
        _CACHE.pop(key, None)
        return None
    return value

def _cache_set(key: str, value, ttl_s: float = 0.0):
    expires_at = (time.time() + ttl_s) if ttl_s and ttl_s > 0 else None
    _CACHE[key] = (value, expires_at)

def _cache_invalidate(prefix: str = ""):
    if not prefix:
        _CACHE.clear()
        return
    for k in list(_CACHE.keys()):
        if k.startswith(prefix):
            _CACHE.pop(k, None)

def load_state() -> dict:
    """Load persisted server state. Never raises."""
    try:
        if os.path.exists(STATE_DB):
            with open(STATE_DB, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def save_state(state_or_key, value=None) -> None:
    """Persist server state safely.
    - If called with a dict, overwrites state.
    - If called with (key, value), updates that key.
    Never raises.
    """
    try:
        if value is None and isinstance(state_or_key, dict):
            state = state_or_key or {}
        else:
            key = str(state_or_key)
            state = load_state()
            state[key] = value
        tmp = STATE_DB + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state or {}, f, indent=2, sort_keys=True)
        os.replace(tmp, STATE_DB)
    except Exception:
        pass

# Load persisted toggles at boot
_boot_state = load_state()
if isinstance(_boot_state, dict):
    MIC_ON = bool(_boot_state.get("MIC_ON", MIC_ON))
    TTS_ON = bool(_boot_state.get("TTS_ON", TTS_ON))
MIC_ENABLED = MIC_ON
TTS_ENABLED = TTS_ON
VOICE_OUTPUT_ON = TTS_ON
VOICE_OUTPUT_ENABLED = TTS_ON

# Optional core modules
ledger_mod = None
try:
    import SarahMemoryLedger as ledger_mod
except ImportError: # Use specific ImportError for module not found
    app_logger.info("SarahMemoryLedger module not found. Ledger functionality will be basic.")
except Exception as e:
    app_logger.error(f"Error importing SarahMemoryLedger: {e}")


net_mod = None
try:
    import SarahMemoryNetwork as net_mod
except ImportError:
    app_logger.info("SarahMemoryNetwork module not found. Hub functionality will be basic.")
except Exception as e:
    app_logger.error(f"Error importing SarahMemoryNetwork: {e}")

# Flask app (templates under WEB_DIR so /api/index.html is found)
app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="/api/static",
    template_folder=TEMPLATE_DIR
)

# Ensure Flask has a secret key for session cookies (used by /api/ui/bootstrap)
SECRET_KEY_FILE = os.path.join(DATA_DIR, ".secret_key")

def get_or_create_secret_key() -> str:
    try:
        _ensure_dir(DATA_DIR)
        if os.path.exists(SECRET_KEY_FILE):
            with open(SECRET_KEY_FILE, "r", encoding="utf-8") as f:
                k = (f.read() or "").strip()
                if k:
                    return k
        k = os.environ.get("SECRET_KEY") or secrets.token_hex(32)
        with open(SECRET_KEY_FILE, "w", encoding="utf-8") as f:
            f.write(k)
        try:
            os.chmod(SECRET_KEY_FILE, 0o600)
        except Exception:
            pass
        return k
    except Exception:
        # Fallback: ephemeral (sessions won't persist across restarts)
        return os.environ.get("SECRET_KEY") or secrets.token_hex(32)

try:
    if not app.config.get("SECRET_KEY"):
        app.config["SECRET_KEY"] = get_or_create_secret_key()
except Exception:
    # Not fatal; sessions will simply not persist.
    pass

# Apply CORS *after* app is created
# Tighten CORS based on env config
ALLOWED_ORIGINS = [o.strip() for o in (os.getenv("CORS_ORIGINS", "") or "").split(",") if o.strip()]
if not ALLOWED_ORIGINS:
    # Dev + known frontends fallback
    ALLOWED_ORIGINS = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5055",
        "http://127.0.0.1:5055",
        "https://ai.sarahmemory.com",
        "https://api.sarahmemory.com",
    ]

if _CORS_AVAILABLE:
    try:
        CORS(
            app,
            resources={r"/api/*": {"origins": ALLOWED_ORIGINS}},
            supports_credentials=True,
        )
    except Exception as e:
        app_logger.error(f"CORS config failed: {e}")
else:
    app_logger.warning("Flask-CORS not installed; CORS disabled (same-origin still works).")

# --- SarahMemoryGITtalk (TEMP ADMIN TOOL) ---
try:
    # Only enable when you explicitly turn it on
    if os.environ.get("SARAH_GITTALK_ENABLED", "0").strip().lower() in ("1", "true", "yes", "on"):
        mod_path = Path(__file__).resolve().parent / "data" / "mods" / "v800"
        if mod_path.exists() and str(mod_path) not in sys.path:
            sys.path.insert(0, str(mod_path))

        from SarahMemoryGITtalk import create_gittalk_blueprint  # noqa
        app.register_blueprint(create_gittalk_blueprint(url_prefix="/api/gittalk"))
        app_logger.info("SarahMemoryGITtalk blueprint mounted at /api/gittalk")
except Exception as e:
    app_logger.warning(f"SarahMemoryGITtalk not mounted: {e}")
# --- end SarahMemoryGITtalk ---

try:
    from SarahMemoryDatabase import init_database
    init_database()  # ensures ai_learning.db + qa_cache exist
except ImportError:
    app_logger.warning("SarahMemoryDatabase not found. Skipping database initialization.")
except Exception as e:
    app_logger.error(f"DB init failed in app.py: {e}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connect_sqlite(path: str):
    """Establishes an SQLite database connection with row_factory set to sqlite3.Row."""
    try:
        con = sqlite3.connect(path, timeout=5.0)
        con.row_factory = sqlite3.Row
        return con
    except sqlite3.Error as e:
        app_logger.error(f"Failed to connect to SQLite DB at {path}: {e}")
        raise # Re-raise to be handled by caller

def _safe_getattr(mod, name, default=None):
    """Safely gets an attribute from a module, returning a default if not found or an error occurs."""
    try:
        return getattr(mod, name)
    except AttributeError:
        # app_logger.debug(f"Attribute '{name}' not found in module {mod.__name__}.")
        return default
    except Exception as e:
        app_logger.error(f"Error accessing attribute '{name}' from module {mod.__name__}: {e}")
        return default

def _ensure_dir(p: str):
    """Ensures a directory exists, logging any errors."""
    try:
        os.makedirs(p, exist_ok=True)
    except OSError as e:
        app_logger.error(f"Failed to create directory {p}: {e}")

# Cache global paths to avoid recalculation on every request
_cached_globals_paths = None
def _globals_paths():
    """
    Locate key SarahMemory paths from SarahMemoryGlobals.py.
    Returns a dict with: DATA_DIR, ROOT_DIR, SANDBOX_DIR, ADDONS_DIR, MODS_DIR, SETTINGS_DIR
    """
    global _cached_globals_paths
    if _cached_globals_paths is not None:
        return _cached_globals_paths

    # Defaults (work on PythonAnywhere / headless Linux too)
    root_dir = os.path.abspath(Path(__file__).resolve().parents[2])  # v800 patch: stable BASE_DIR
    data_dir = os.path.join(root_dir, "data")
    sandbox_dir = os.path.join(root_dir, "sandbox")
    addons_dir = os.path.join(data_dir, "addons")
    mods_dir = os.path.join(root_dir, "mods")
    settings_dir = os.path.join(data_dir, "settings")

    try:
        import SarahMemoryGlobals as smg  # type: ignore
        root_dir = os.path.abspath(getattr(smg, "ROOT_DIR", root_dir))
        data_dir = os.path.abspath(getattr(smg, "DATA_DIR", data_dir))
        sandbox_dir = os.path.abspath(getattr(smg, "SANDBOX_DIR", sandbox_dir))
        addons_dir = os.path.abspath(getattr(smg, "ADDONS_DIR", addons_dir))
        mods_dir = os.path.abspath(getattr(smg, "MODS_DIR", mods_dir))
        settings_dir = os.path.abspath(getattr(smg, "SETTINGS_DIR", settings_dir))
    except Exception:
        pass

    # Ensure dirs exist (best-effort)
    for d in (data_dir, sandbox_dir, addons_dir, mods_dir, settings_dir):
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass

    _cached_globals_paths = {
        "ROOT_DIR": root_dir,
        "DATA_DIR": data_dir,
        "SANDBOX_DIR": sandbox_dir,
        "ADDONS_DIR": addons_dir,
        "MODS_DIR": mods_dir,
        "SETTINGS_DIR": settings_dir,
    }
    return _cached_globals_paths


def _globals_dir(key: str, default_rel: str) -> str:
    """Return a string path from _globals_paths()[key].
    Falls back to CWD/default_rel if missing or invalid."""
    try:
        d = _globals_paths()
        if isinstance(d, dict):
            v = d.get(key)
            if isinstance(v, (str, bytes, os.PathLike)):
                return os.fspath(v)
    except Exception:
        pass
    return os.path.join(os.path.abspath(os.getcwd()), default_rel)



def _get_hub_hmac_secret() -> str:
    """Shared secret for node/hub HMAC signing.

    Priority:
      1) env HUB_HMAC_SECRET / SARAH_HUB_HMAC_SECRET
      2) SarahMemoryGlobals.HUB_HMAC_SECRET (if present)
    """
    try:
        import SarahMemoryGlobals as G
        v = getattr(G, "HUB_HMAC_SECRET", "") or ""
        if v:
            return str(v)
    except Exception:
        pass
    return (os.environ.get("HUB_HMAC_SECRET") or os.environ.get("SARAH_HUB_HMAC_SECRET") or "").strip()

def _sign_ok(body: bytes, signature: str) -> bool:
    """Verify X-Sarah-Signature as hex(HMAC_SHA256(secret, body)).

    If no secret is configured, allow ONLY localhost requests (dev-safe fallback).
    """
    secret = _get_hub_hmac_secret()
    sig = (signature or "").strip()
    if not secret:
        # No secret configured — do not expose signature-less auth to the internet.
        # Accept only loopback for local development.
        try:
            ra = request.remote_addr or ""
        except Exception:
            ra = ""
        return ra in ("127.0.0.1", "::1", "localhost")
    if not sig:
        return False
    try:
        mac = hmac.new(secret.encode("utf-8"), body or b"", hashlib.sha256).hexdigest()
        return hmac.compare_digest(mac, sig)
    except Exception:
        return False

# ---------------------------------------------------------------------------
# Wallet / Ledger
# ---------------------------------------------------------------------------
def _wallet_path_simple(node: str) -> str:
    safe = "".join(ch for ch in node if ch.isalnum() or ch in ("_", "-")) or "anon"
    return os.path.join(WALLETS_DIR, f"wallet-{safe}.srh")

def ensure_wallet_simple(node: str):
    """Ensure minimal wallet tables exist."""
    con = None
    try:
        con = _connect_sqlite(WALLET_DB)
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS wallet (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE,
                balance TEXT DEFAULT '0'
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                user_id TEXT,
                delta TEXT,
                note TEXT
            )
        """)
        con.commit()
        return True
    except Exception as e:
        logger.exception("ensure_wallet_simple failed: %s", e)
        return False
    finally:
        try:
            if con is not None:
                con.close()
        except Exception:
            pass


def get_balance_simple(path: str) -> Decimal:
    balance = Decimal("0")
    con = None
    try:
        con = _connect_sqlite(path)
        cur = con.cursor()
        cur.execute("SELECT balance FROM wallet WHERE id=1")
        row = cur.fetchone()
        balance = Decimal(row) if row and row is not None else Decimal("0")
    except sqlite3.Error as e:
        app_logger.error(f"Failed to get simple wallet balance from {path}: {e}")
    finally:
        if con: con.close()
    return balance

def read_top_nodes(limit=10):
    """Return top nodes for the public leaderboard.

    Preferred source (when enabled): GoogieHost MySQL table `sm_network_nodes`
      - ordered by `trust_score` DESC
      - limited to `limit`

    Fallback source: local SQLite wallet (legacy/demo)
    """
    # --- Cloud MySQL path (preferred) ---
    try:
        cloud_enabled = str(os.getenv("CLOUD_DB_ENABLED", "false")).strip().lower() in ("1", "true", "yes", "on")
        if cloud_enabled:
            # Local import so the server can still boot even if MySQL client isn't installed.
            try:
                import pymysql  # type: ignore
            except Exception:
                pymysql = None

            if pymysql is not None:
                host = os.getenv("CLOUD_DB_HOST") or ""
                name = os.getenv("CLOUD_DB_NAME") or ""
                user = os.getenv("CLOUD_DB_USER") or ""
                pwd = os.getenv("CLOUD_DB_PASSWORD") or ""
                port = int(os.getenv("CLOUD_DB_PORT") or "3306")

                if host and name and user and pwd:
                    con = None
                    try:
                        con = pymysql.connect(
                            host=host,
                            user=user,
                            password=pwd,
                            database=name,
                            port=port,
                            connect_timeout=5,
                            read_timeout=5,
                            write_timeout=5,
                            cursorclass=pymysql.cursors.DictCursor,
                            charset="utf8mb4",
                        )
                        with con.cursor() as cur:
                            cur.execute(
                                """
                                SELECT node_name, node_id, ip_address, is_online, trust_score
                                FROM sm_network_nodes
                                ORDER BY trust_score DESC, id ASC
                                LIMIT %s
                                """,
                                (max(1, int(limit)),),
                            )
                            rows = cur.fetchall() or []
                        leaders = []
                        rank = 1
                        for r in rows:
                            leaders.append(
                                {
                                    "rank": rank,
                                    "name": (r.get("node_name") or r.get("node_id") or "").strip() or f"Node-{rank}",
                                    "org": "SarahMemory Node",
                                    "rep": float(r.get("trust_score") or 0),
                                    "node_id": r.get("node_id"),
                                    "is_online": int(r.get("is_online") or 0),
                                    "ip": r.get("ip_address"),
                                }
                            )
                            rank += 1
                        return leaders
                    except Exception as e:
                        logger.debug("read_top_nodes cloud MySQL failed: %s", e)
                    finally:
                        try:
                            if con is not None:
                                con.close()
                        except Exception:
                            pass
    except Exception as e:
        logger.debug("read_top_nodes cloud config failed: %s", e)

    # --- Local fallback path (SQLite wallet) ---
    ensure_wallet_simple()
    con = None
    try:
        con = _connect_sqlite(WALLET_DB)
        cur = con.cursor()
        cur.execute("SELECT user_id, balance FROM wallet")
        rows = cur.fetchall() or []
        data = []
        for r in rows:
            uid = r[0]
            bal = Decimal(str(r[1] if r[1] is not None else "0"))
            data.append({
                "rank": 0,
                "name": uid,
                "org": "Local Wallet",
                "rep": float(bal),
                "user_id": uid,
                "balance": str(bal),
            })
        data.sort(key=lambda x: Decimal(str(x.get("rep", 0))), reverse=True)
        # fill ranks
        for i, item in enumerate(data[: max(1, int(limit))], start=1):
            item["rank"] = i
        return data[: max(1, int(limit))]
    except Exception as e:
        logger.debug("read_top_nodes sqlite fallback failed: %s", e)
        return []
    finally:
        try:
            if con is not None:
                con.close()
        except Exception:
            pass




def ensure_meta_db():
    con = None
    try:
        con = _connect_sqlite(META_DB)
        cur = con.cursor()
        # Hub/node tables (for network sync)
        cur.execute("""CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            last_ts REAL,
            meta TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, node_id TEXT, context_id TEXT, vector TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS contexts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, node_id TEXT, text TEXT, tags TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS job_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, node_id TEXT, job_id TEXT, result TEXT
        )""")
        # Knowledge marketplace + receipts
        cur.execute("""CREATE TABLE IF NOT EXISTS knowledge_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, requester TEXT, topic TEXT, reward TEXT, status TEXT, provider TEXT, answer TEXT
        )""")
        cur.execute("""CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, payload TEXT, sig TEXT, valid INTEGER
        )""")
        con.commit()
    except sqlite3.Error as e:
        app_logger.error(f"Failed to ensure meta DB at {META_DB}: {e}")
    finally:
        if con: con.close()
ensure_meta_db()


# ---------------------------------------------------------------------------
# Core routes (UI + API)
# ---------------------------------------------------------------------------

def _get_runtime_meta_safe():
    """
    Lightweight wrapper around SarahMemoryGlobals.get_runtime_meta (Phase A1).
    Returns a small dict with runtime identity and safety flags that is safe to
    serialize to logs and JSON responses. If SarahMemoryGlobals is missing or
    incomplete, falls back to conservative defaults.
    """
    try:
        import SarahMemoryGlobals as G
        meta_fn = _safe_getattr(G, "get_runtime_meta")
        if callable(meta_fn):
            meta = meta_fn() or {}
        else:
            meta = {}
        # Ensure baseline keys exist even if get_runtime_meta() is older than v7.7.5
        meta.setdefault("project_version", getattr(G, "PROJECT_VERSION", PROJECT_VERSION))
        meta.setdefault("author", getattr(G, "AUTHOR", "Brian Lee Baros"))
        meta.setdefault("revision_start_date", getattr(G, "REVISION_START_DATE", ""))
        meta.setdefault("run_mode", getattr(G, "RUN_MODE", "local"))
        meta.setdefault("device_mode", getattr(G, "DEVICE_MODE", "local_agent"))
        meta.setdefault("device_profile", getattr(G, "DEVICE_PROFILE", "Standard"))
        meta.setdefault("safe_mode", getattr(G, "SAFE_MODE", False))
        meta.setdefault("local_only", getattr(G, "LOCAL_ONLY_MODE", False))
        meta.setdefault("node_name", getattr(G, "NODE_NAME", "SarahMemoryNode"))
        return meta
    except Exception as e:
        app_logger.warning(f"Error getting runtime meta from SarahMemoryGlobals, falling back: {e}")
        # Fail-safe identity snapshot if globals are unavailable.
        return {
            "project_version": PROJECT_VERSION,
            "author": "Brian Lee Baros",
            "revision_start_date": "",
            "run_mode": "local",
            "device_mode": "local_agent",
            "device_profile": "Standard",
            "safe_mode": False,
            "local_only": False,
            "node_name": "SarahMemoryNode",
        }
try:
    import SarahMemoryCognitiveServices as cog
    COG_AVAILABLE = True
except Exception as e:
    app_logger.warning(f"CognitiveServices not available: {e}")
    cog = None
    COG_AVAILABLE = True

@app.before_request
def _cognitive_guard():
    if not COG_AVAILABLE:
        return None

    # Only guard API endpoints (avoid slowing static/template hits)
    p = (request.path or "")
    if not p.startswith("/api/"):
        return None

    # Pull a small amount of text to analyze (don’t log secrets)
    data = request.get_json(silent=True) if request.method in ("POST","PUT","PATCH") else None
    msg = ""
    if isinstance(data, dict):
        # common fields
        msg = str(data.get("message") or data.get("text") or data.get("q") or "")[:4000]

    # Example: call a lightweight analyzer (sentiment/risk tagging/etc.)
    # Store result for the endpoint to use (no blocking by default)
    try:
        g.cognitive = {"ok": True, "sentiment": cog.analyze_text(msg) if msg else None}
    except Exception as e:
        g.cognitive = {"ok": False, "error": str(e)}

    return None



@app.route("/api/session/bootstrap", methods=['POST'])
def api_session_bootstrap():
    """
    Phase A3 — Session Bootstrap API.
    Single canonical handshake endpoint used by Web UI (app.js) at startup.
    Aligns client and server runtime identity and exposes core feature flags.
    """
    try:
        payload = request.get_json(silent=True) or {} # jsonify handles non-JSON, no need for force=True
    except Exception as e:
        app_logger.warning(f"Failed to parse JSON for bootstrap, proceeding with empty payload: {e}")
        payload = {}

    client_info = {
        "env": (payload.get("client_env") or request.args.get("client_env") or "").strip(),
        "platform": (payload.get("platform") or request.args.get("platform") or "").strip(),
        "ui_version": (payload.get("ui_version") or request.args.get("ui_version") or "").strip(),
        "agent_name": (payload.get("agent_name") or request.args.get("agent_name") or "").strip(),
        "bridge": (payload.get("bridge") or request.args.get("bridge") or "").strip(),
    }

    runtime = _get_runtime_meta_safe()

    # Camera/mic/voice toggles (default to False if never touched yet)
    # Using app.config for Flask global state rather than globals()
    camera_enabled = app.config.get("CAMERA_ENABLED", False)
    mic_enabled = app.config.get("MIC_ENABLED", False)
    voice_enabled = app.config.get("VOICE_OUTPUT_ENABLED", False)

    features = {
        "camera": camera_enabled,
        "microphone": mic_enabled,
        "voice_output": voice_enabled,
        "hub_enabled": bool(net_mod is not None),
        "wallet_enabled": True, # Assume wallet is always enabled if META_DB is there
        "ledger_module": bool(ledger_mod is not None),
        "file_transfer": True, # Assume file transfer is always enabled
    }

    env = {
        "api_base": request.host_url.rstrip("/"),
        "web_root": request.host_url.rstrip("/") + "/api/",
    }

    return jsonify({
        "ok": True,
        "version": PROJECT_VERSION,
        "runtime": runtime,
        "client": client_info,
        "features": features,
        "env": env,
        "ts": time.time(),
    })

@app.route("/api/")
def api_index():
    """API root health banner (JSON).

    NOTE:
    - The Ranking SPA is served at "/" (root_index).
    - "/api/" is reserved for programmatic health/status checks used by the frontend heartbeat.
    """
    return jsonify(
        {
            "ok": True,
            "running": True,
            "service": "SarahMemory API",
            "version": PROJECT_VERSION,
        }
    )


def _req_host() -> str:
    """Return host without port, lowercased."""
    try:
        return (request.host or "").split(":", 1)[0].strip().lower()
    except Exception:
        return ""


def _want_ui_for_request() -> bool:
    """Host-based routing for the dual server.

    Local:
      - 127.0.0.1 / localhost -> Web UI
    Cloud:
      - ai.sarahmemory.com    -> Web UI
      - api.sarahmemory.com   -> Network Hub

    Default is hub, unless it matches UI conditions.
    """
    host = _req_host()
    if host in ("127.0.0.1", "localhost"):
        return True
    if host.startswith("ai."):
        return True
    return False

@app.route("/")
def root_index():
    """Serve the Ranking/Web UI (static SPA) at the site root.

    PythonAnywhere serves /assets and /static via static mappings, but "/" must be handled
    by Flask. If the UI build is present, return static/index.html; otherwise fall back
    to the API banner.
    """
    # Prefer the Web UI (Lovable/Vite dist) for local + ai.* host.
    if _want_ui_for_request():
        ui_index = os.path.join(UI_DIST_DIR, "index.html")
        if os.path.isfile(ui_index):
            return send_from_directory(UI_DIST_DIR, "index.html")

    # Otherwise, show the Network Hub landing (legacy /api/server/static/index.html)
    hub_index = os.path.join(STATIC_DIR, "index.html")
    if os.path.isfile(hub_index):
        return send_from_directory(STATIC_DIR, "index.html")
    return redirect("/api/")


# -----------------------------------------------------------------------------
# Web UI dist asset serving (local + ai.sarahmemory.com)
# -----------------------------------------------------------------------------

@app.route("/assets/<path:filename>")
def ui_assets(filename):
    if _want_ui_for_request():
        base = os.path.join(UI_DIST_DIR, "assets")
        if os.path.isdir(base):
            return send_from_directory(base, filename)
    return abort(404)


@app.route("/themes/<path:filename>")
def ui_themes(filename):
    if _want_ui_for_request():
        base = os.path.join(UI_DIST_DIR, "themes")
        if os.path.isdir(base):
            return send_from_directory(base, filename)
    return abort(404)


@app.route("/favicon.ico")
def ui_favicon():
    if _want_ui_for_request() and os.path.isfile(os.path.join(UI_DIST_DIR, "favicon.ico")):
        return send_from_directory(UI_DIST_DIR, "favicon.ico")
    return abort(404)


@app.route("/robots.txt")
def ui_robots():
    if _want_ui_for_request() and os.path.isfile(os.path.join(UI_DIST_DIR, "robots.txt")):
        return send_from_directory(UI_DIST_DIR, "robots.txt")
    return abort(404)


@app.route("/placeholder.svg")
def ui_placeholder():
    if _want_ui_for_request() and os.path.isfile(os.path.join(UI_DIST_DIR, "placeholder.svg")):
        return send_from_directory(UI_DIST_DIR, "placeholder.svg")
    return abort(404)


@app.route("/<path:path>")
def ui_spa_fallback(path):
    """SPA fallback for non-/api routes.

    Vite builds often use client-side routing; unknown paths must return index.html.
    """
    # Never hijack API routes
    if path.startswith("api/"):
        return abort(404)

    if _want_ui_for_request():
        candidate = os.path.join(UI_DIST_DIR, path)
        if os.path.isfile(candidate):
            return send_from_directory(UI_DIST_DIR, path)
        # Client-side route: return index.html
        ui_index = os.path.join(UI_DIST_DIR, "index.html")
        if os.path.isfile(ui_index):
            return send_from_directory(UI_DIST_DIR, "index.html")

    # Fallback to hub static if present
    candidate = os.path.join(STATIC_DIR, path)
    if os.path.isfile(candidate):
        return send_from_directory(STATIC_DIR, path)
    return abort(404)

@app.route("/api/static/<path:filename>")
def static_serv(filename):
    return send_from_directory(STATIC_DIR, filename)

# Loose assets for the hub index (icons, hero image, QR code, etc.)
# This lets relative URLs like "SOFTDEV0_LLC_Logo.png" work from /api/
# by serving them from either STATIC_DIR or the project BASE_DIR.
ASSET_EXTS = {
    "png", "jpg", "jpeg", "gif", "webp", "svg", "ico", "bmp",
    "css", "js", "map", "json", "txt", "xml"
}

@app.route("/api/<path:filename>")
def api_loose_assets(filename: str):
    # Do not interfere with explicit API endpoints like /api/health or /api/leaderboard.
    # Flask prefers static rules (/api/health) over this dynamic one, so those will still win.
    if "." not in filename:
        # No extension: let the real API routes handle it (or 404 there).
        # We just return a 404 JSON so this route doesn't claim it.
        return jsonify({"error": "not an asset"}), 404

    ext = filename.rsplit(".", 1).lower()
    if ext not in ASSET_EXTS:
        return jsonify({"error": "unsupported asset type", "file": filename}), 404

    # Try in /api/server/static first (STATIC_DIR), then in the project root (BASE_DIR).
    # Using iter for potential performance gain if many routes.
    for root in (STATIC_DIR, BASE_DIR):
        candidate = os.path.join(root, filename)
        if os.path.exists(candidate):
            return send_from_directory(root, filename)

    return jsonify({"error": "asset not found", "file": filename}), 404

@app.route("/api/leaderboard")
def api_leaderboard():
    cache_key = 'leaderboard:10'
    cached = _cache_get(cache_key)
    if cached is not None:
        return jsonify(cached)
    payload = {'leaders': read_top_nodes(limit=10)}
    _cache_set(cache_key, payload, ttl_s=5.0)
    return jsonify(payload)

def _perform_health_checks():
    """
    Fast + safe health checks.

    Returns: (ok: bool, notes: list[str], main_running: bool)

    Notes are short machine-readable strings so the UI / SarahNet rendezvous can decide
    whether to fall back (CLOUD/LAN/OFF) without crashing the API.
    """
    import json as _json  # local import to avoid boot-time surprises

    notes = []
    ok = True

    # 1) Core modules importability (best-effort)
    for mod_name in ("SarahMemoryGlobals", "SarahMemoryVoice", "SarahMemoryDatabase", "SarahMemoryAPI"):
        try:
            __import__(mod_name)
        except Exception as e:
            ok = False
            notes.append(f"import_failed:{mod_name}:{e}")

    # 2) server_state.json readable (STATE_DB is JSON, not sqlite)
    try:
        if os.path.exists(STATE_DB):
            try:
                with open(STATE_DB, "r", encoding="utf-8") as f:
                    _json.load(f)
            except Exception as e:
                ok = False
                notes.append(f"state_json_invalid:{e}")
        else:
            notes.append("state_json_missing")
    except Exception as e:
        ok = False
        notes.append(f"state_json_check_failed:{e}")

    # 3) meta.db reachable (sqlite)
    try:
        con = _connect_sqlite(META_DB)
        con.execute("CREATE TABLE IF NOT EXISTS _health_ping (id INTEGER PRIMARY KEY, ts TEXT)")
        con.close()
    except Exception as e:
        ok = False
        notes.append(f"sqlite_meta_db_failed:{e}")

    # 4) Main process running flag (desktop installs). Safe on cloud.
    main_running = False
    try:
        fn = globals().get("_is_running")
        if callable(fn):
            main_running = bool(fn())
    except Exception as e:
        notes.append(f"main_running_check_failed:{e}")

    return bool(ok), (notes if isinstance(notes, list) else []), bool(main_running)


@app.get("/api/health")
def api_health():
    """
    Universal health endpoint.
    - running      → HTTP API is responding (True if this function is hit)
    - main_running → optional desktop launcher (SarahMemoryMain) process check
                     used on Windows/Linux desktop installs only.
    """
    ok, notes, main_running = _perform_health_checks()
    status = "ok" if ok else "down"

    return jsonify(
        {
            "ok": ok,
            "status": status,
            "running": True,  # API is up if we're here
            "main_running": main_running,
            "version": PROJECT_VERSION,
            "ts": time.time(),
            "notes": notes,
        }
    )

@app.route("/api/chat", methods=['POST'])
def api_chat():
    """
    Primary chat endpoint used by the Web UI (app.js).
    Expects JSON like:
      { "text": "user message here", "files":  }
    Returns JSON like:
      {
        "ok": true,
        "reply": "<assistant text>",
        "meta": { "source": "api", "engine": "route_intent_response" }
      }
    """
    try:
        payload = request.get_json(silent=True) or {}
        # Optional metadata (kept lightweight; used by UI when available)
        intent = str(payload.get("intent") or "")
        tone = str(payload.get("tone") or "")
        complexity = str(payload.get("complexity") or "")
        avatar_request = bool(payload.get("avatar_request") or payload.get("avatar") or False)
        text = (payload.get("text") or "").strip()

        if not text:
            return jsonify({
                "ok": False,
                "error": "Missing 'text' in request body.",
                "meta": {"source": "api", "reason": "no_text"},
            }), 400

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Identity guardrails (keep branding consistent; prevent model/provider drift)
        # ------------------------------------------------------------------
        if _is_identity_question(text):
            ident = _identity_payload()
            low = (text or "").strip().lower()

            if "version" in low:
                reply = (
                    f"My name is {ident['name']} — your {ident['platform']} companion. "
                    f"Server version: {ident['version']}."
                )
            elif any(k in low for k in (
                "who made you", "who created you", "creator", "who built you",
                "who designed you", "designer", "engineer", "who engineered you",
            )):
                reply = f"I was created by {ident['creator']} ({ident['organization']}) as part of {ident['platform']}."
            elif "mission" in low:
                reply = f"My mission is to help you as {ident['platform']} — fast, accurate, and user-controlled."
            elif "brian lee baros" in low:
                reply = f"{ident['creator']} is the creator/lead engineer of the {ident['platform']} project."
            else:
                reply = f"I'm {ident['name']} — your {ident['platform']} companion."

            return jsonify({
                "ok": True,
                "reply": reply,
                "identity": ident,
                "meta": {"source": "identity_guard", "version": ident["version"]},
            }), 200

        reply_str = ""
        engine_source = "api" # Default source

        # ------------------------------------------------------------------
        # Try the lightweight router (commands, mouse moves, URL opens)
        # ------------------------------------------------------------------
        router_result = None
        try:
            import SarahMemoryAiFunctions as F
            # Centralize the Cognitive loop call here as per AGI spec
            route_fn = _safe_getattr(F, "route_user_input_through_cognitive_loop") # New function name
                                                                                   # or existing route_intent_response
            if callable(route_fn):
                # Assuming route_user_input_through_cognitive_loop takes text and optional hints
                router_result = route_fn(text, intent=intent, tone=tone, complexity=complexity)
                engine_source = "cognitive_loop"
        except ImportError:
            app_logger.warning("SarahMemoryAiFunctions not found. Skipping cognitive loop.")
        except Exception as e:
            app_logger.error(f"Error in SarahMemoryAiFunctions router: {e}", exc_info=True)
            router_result = None # Ensure fallback if router itself errors

        # If the router clearly handled it, use its response.
        if router_result:
            if isinstance(router_result, str) and router_result.strip():
                # Avoid using internal "I'm unsure" as final reply if a full AI pipeline can do better
                if router_result.strip() != "I'm unsure how to respond.":
                    reply_str = router_result.strip()
            elif isinstance(router_result, dict) and router_result.get("handled"):
                reply_str = (
                    router_result.get("response")
                    or router_result.get("text")
                    or ""
                ).strip()
        # ------------------------------------------------------------------
        # Fallback: full SarahMemory reply pipeline (same as GUI path)
        # ------------------------------------------------------------------
        if not reply_str:
            bundle = None
            try:
                from SarahMemoryReply import generate_reply
                # generate_reply is defined as generate_reply(self, user_text, **kwargs)
                # For API usage we pass self=None and extended args
                bundle = generate_reply(None, text, intent=intent, tone=tone, complexity=complexity)
                engine_source = "sarahmemory_reply"
            except ImportError:
                app_logger.warning("SarahMemoryReply module not found. Attempting direct API call fallback.")
            except Exception as e:
                app_logger.error(f"SarahMemoryReply.generate_reply failed: {e}", exc_info=True)

            if bundle:
                # Normal path: unify bundle —> string
                if isinstance(bundle, dict):
                    reply_str = str(
                        bundle.get("response")
                        or bundle.get("text")
                        or ""
                    ).strip()
                else:
                    reply_str = str(bundle or "").strip()

        # ------------------------------------------------------------------
        # Last-ditch fallback: Direct API call if all else fails
        # ------------------------------------------------------------------
        if not reply_str:
            try:
                from SarahMemoryAPI import send_to_api as _send_to_api
                api_res = _send_to_api(
                    text,
                    provider="openai",
                    intent=intent,
                    tone=tone,
                    complexity=complexity,
                )
                if isinstance(api_res, dict):
                    reply_str = str(api_res.get("data") or "").strip()
                else:
                    reply_str = str(api_res).strip()
                engine_source = "direct_api_fallback"
            except ImportError:
                app_logger.warning("SarahMemoryAPI module not found. Cannot perform direct API call fallback.")
            except Exception as api_e:
                # If we get here, everything failed — real 500
                msg = (
                    f"SarahMemoryReply.generate_reply failed (or not imported); "
                    f"API fallback failed: {api_e}"
                )
                app_logger.error(msg, exc_info=True)
                return jsonify({
                    "ok": False,
                    "error": "Internal server error: Failed to generate reply.",
                    "meta": {"source": "api", "reason": "all_reply_methods_failed"},
                }), 500

        # Final safety: never return None; if still empty, provide general fallback
        reply_str = reply_str or "I'm sorry, I couldn't generate a response at this time."

        meta_out = {
            "source": engine_source,
            "version": PROJECT_VERSION,
        }
        if avatar_request:
            meta_out["avatar_request"] = avatar_request

        return jsonify({
            "ok": True,
            "reply": reply_str,
            "meta": meta_out,
        }), 200

    except Exception as e:
        # Hard catch-all so the Web UI sees a clean JSON error instead of a stack trace
        app_logger.exception("api_chat failed unexpectedly.")
        return jsonify({
            "ok": False,
            "error": "Internal server error during chat processing.",
            "meta": {"source": "api", "reason": "uncaught_exception"},
        }), 500

@app.get("/api/state")
def api_state():
    """
    Runtime state snapshot.
    Combines persisted server_state.json with a live main_running check.
    Never raises.
    """
    try:
        state = load_state()
        if not isinstance(state, dict):
            state = {}

        # Live truth: mirror /api/health main_running computation
        main_running = False
        try:
            fn = globals().get("_is_running")
            if callable(fn):
                main_running = bool(fn())
        except Exception:
            main_running = False

        state["main_running"] = main_running
        state.setdefault("ok", True)
        state.setdefault("notes", [])
        state.setdefault("source", "state_db_plus_live")
        state["ts"] = time.time()

        # Optional: persist the refreshed truth so UI/other callers stay consistent
        try:
            save_state(state)
        except Exception:
            pass

        return jsonify({
            "ok": True,
            "state": state,
            "ts": time.time(),
            "version": PROJECT_VERSION,
        }), 200

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": str(e),
            "ts": time.time(),
            "version": PROJECT_VERSION,
        }), 200


# ---------------------------------------------------------------------------
# Media Job Contract (v8.0.0) - JSON-first API for image/video/audio/3D
# ---------------------------------------------------------------------------

@app.route("/api/media/job", methods=["POST"])
def api_media_job_submit():
    """Submit a media generation job. Engine execution is handled by mods/add-ons."""
    try:
        payload = request.get_json(silent=True) or {}
        job = payload.get("job") or payload  # allow direct job dict
        import SarahMemoryAiFunctions as F
        job_id = F.submit_media_job(job)
        return jsonify({"ok": True, "job_id": job_id}), 200
    except Exception as e:
        app_logger.error(f"api_media_job_submit failed: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 400

@app.route("/api/media/job/poll", methods=["POST"])
def api_media_job_poll():
    """Poll the next queued media job (for worker/add-on processes)."""
    try:
        import SarahMemoryAiFunctions as F
        job = F.poll_media_job()
        return jsonify({"ok": True, "job": job}), 200
    except Exception as e:
        app_logger.error(f"api_media_job_poll failed: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/api/media/result/<job_id>", methods=["GET"])
def api_media_job_result(job_id):
    """Get status/result for a media job."""
    try:
        import SarahMemoryAiFunctions as F
        rec = F.get_media_result(job_id)
        return jsonify({"ok": True, "data": rec}), 200
    except Exception as e:
        app_logger.error(f"api_media_job_result failed: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 404

@app.route("/api/media/result/<job_id>/store", methods=["POST"])
def api_media_job_store(job_id):
    """Store a media result (for worker/add-on processes)."""
    try:
        payload = request.get_json(silent=True) or {}
        result = payload.get("result") or {}
        status = payload.get("status") or "done"
        import SarahMemoryAiFunctions as F
        F.store_media_result(job_id, result, status=status)
        # Best-effort: if AvatarPanelAPI is active, try to display it
        try:
            from SarahMemoryAvatarPanel import AvatarPanelAPI
            api = AvatarPanelAPI()
            api.display_media_result(result)
        except Exception:
            pass
        return jsonify({"ok": True}), 200
    except Exception as e:
        app_logger.error(f"api_media_job_store failed: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/request-knowledge", methods=['POST'])
def api_request_knowledge():
    data = request.get_json(silent=True) or {}
    requester = (data.get("requester") or data.get("from") or "").strip()
    topic = (data.get("topic") or data.get("notes") or "").strip()
    amount = data.get("amount") or data.get("reward") or "0" # Keep as string for Decimal conversion

    # Validate inputs
    if not requester:
        return jsonify({"error": "Requester ID is required."}), 400
    if not topic:
        return jsonify({"error": "Knowledge topic is required."}), 400

    try:
        amount_decimal = Decimal(str(amount)) # Ensure convertible to Decimal
        if amount_decimal < 0:
            return jsonify({"error": "Reward amount cannot be negative."}), 400
    except Exception:
        return jsonify({"error": "Invalid reward amount format."}), 400

    con = None
    try:
        con = _connect_sqlite(META_DB)
        cur = con.cursor()
        cur.execute("INSERT INTO knowledge_requests(ts, requester, topic, reward, status) VALUES (?,?,?,?,?)",
                    (time.time(), requester, topic, str(amount_decimal), "open"))
        rid = cur.lastrowid
        con.commit()
        ensure_wallet_simple(requester) # Ensure wallet for requester
        return jsonify({"request_id": rid, "status": "open"}), 201
    except sqlite3.Error as e:
        app_logger.error(f"Failed to record knowledge request to {META_DB}: {e}", exc_info=True)
        return jsonify({"error": "Failed to record knowledge request due to database error."}), 500
    finally:
        if con: con.close()


@app.route("/api/wallet/<node>")
def api_wallet_view(node):
    con = None
    try:
        p = ensure_wallet_simple(node)
        con = _connect_sqlite(p)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT balance, reputation, last_rep_ts, rep_daily FROM wallet WHERE id=1")
        r = cur.fetchone()
        if not r:
            return jsonify({"error": f"Wallet data not found for node: {node}"}), 404

        cur.execute("SELECT ts,delta,memo FROM txs ORDER BY id DESC LIMIT 50")
        txs = [dict(row) for row in cur.fetchall()] if hasattr(cur, "fetchall") else []

        return jsonify({
            "node": node,
            "balance": r["balance"],
            "reputation": float(r["reputation"] or 0.0),
            "last_rep_ts": float(r["last_rep_ts"] or 0.0),
            "rep_daily": float(r["rep_daily"] or 0.0),
            "txs": txs
        })
    except sqlite3.Error as e:
        app_logger.error(f"SQLite error fetching wallet details for node {node}: {e}", exc_info=True)
        return jsonify({"error": "Database error fetching wallet details"}), 500
    except Exception as e:
        app_logger.exception(f"Unexpected error fetching wallet for node {node}.")
        return jsonify({"error": str(e)}), 500
    finally:
        if con: con.close()

@app.post("/api/hub/ping")
def hub_ping():
    body = request.get_data()
    sig = request.headers.get("X-Sarah-Signature", "")
    if not _sign_ok(body, sig):
        return jsonify({"ok": False, "err": "Unauthorized: Invalid or missing signature"}), 401
    try:
        payload = request.get_json(silent=True) or {}
        return jsonify({"ok": True, "now": time.time(), "echo": payload})
    except Exception as e:
        app_logger.error(f"Error processing hub_ping request: {e}", exc_info=True)
        return jsonify({"ok": False, "err": f"Internal server error: {str(e)}"}), 500


@app.post("/api/hub/job")
def hub_job():
    body = request.get_data()
    sig = request.headers.get("X-Sarah-Signature", "")
    if not _sign_ok(body, sig):
        return jsonify({"ok": False, "err": "Unauthorized: Invalid or missing signature"}), 401
    try:
        payload = request.get_json(silent=True) or {}
        jid = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest() # Specify encoding

        # optional light persistence for debugging
        jobs_dir = os.path.join(DATA_DIR, "jobs")
        _ensure_dir(jobs_dir)
        try:
            with open(os.path.join(jobs_dir, f"job-{int(time.time())}-{jid}.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            app_logger.warning(f"Failed to persist hub job to disk: {e}")

        # TODO: Integrate with SarahMemoryNetwork (net_mod) for proper job handling
        if net_mod and _safe_getattr(net_mod, "process_hub_job"):
             try:
                 net_mod.process_hub_job(jid, payload)
                 app_logger.info(f"Hub job {jid} processed by SarahMemoryNetwork.")
             except Exception as e:
                 app_logger.error(f"Error in SarahMemoryNetwork processing hub job {jid}: {e}", exc_info=True)
                 # Don't fail the hub_job API, just log the internal processing error

        return jsonify({"ok": True, "job_id": jid}), 200
    except Exception as e:
        app_logger.error(f"Error processing hub_job request: {e}", exc_info=True)
        return jsonify({"ok": False, "err": f"Internal server error: {str(e)}"}), 500


@app.post("/api/hub/reply")
def hub_reply():
    body = request.get_data()
    sig = request.headers.get("X-Sarah-Signature", "")
    if not _sign_ok(body, sig):
        return jsonify({"ok": False, "err": "Unauthorized: Invalid or missing signature"}), 401
    try:
        payload = request.get_json(silent=True) or {}
        # optional light persistence for debugging
        receipts_dir = os.path.join(DATA_DIR, "receipts")
        _ensure_dir(receipts_dir)
        try:
            reply_id = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
            with open(os.path.join(receipts_dir, f"reply-{int(time.time())}-{reply_id}.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            app_logger.warning(f"Failed to persist hub reply receipt to disk: {e}")

        # TODO: Integrate with SarahMemoryNetwork (net_mod) for proper reply handling
        if net_mod and _safe_getattr(net_mod, "process_hub_reply"):
             try:
                 net_mod.process_hub_reply(payload)
                 app_logger.info("Hub reply processed by SarahMemoryNetwork.")
             except Exception as e:
                 app_logger.error(f"Error in SarahMemoryNetwork processing hub reply: {e}", exc_info=True)

        return jsonify({"ok": True}), 200
    except Exception as e:
        app_logger.error(f"Error processing hub_reply request: {e}", exc_info=True)
        return jsonify({"ok": False, "err": f"Internal server error: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# API Key guard + Node/Embedding/Context/Jobs endpoints
# ---------------------------------------------------------------------------
SARAH_API_KEY = os.environ.get("SARAH_API_KEY", "") # Keep variable name consistent

def _api_key_auth_ok() -> bool:
    """
    Optional lightweight auth for admin-ish endpoints.
    Accepts either:
      - X-API-Key: <key>
      - Authorization: Bearer <key>
    """
    # allow local / dev with no auth if explicitly configured
    try:
        if config is not None and getattr(config, "ALLOW_NOAUTH_LOCAL", False):
            return True
    except Exception:
        pass

    api_key = (os.environ.get("SARAHMEMORY_API_KEY") or os.environ.get("API_KEY") or "").strip()
    if not api_key:
        # Backward compatible: no key configured => open
        return True

    hdr = (request.headers.get("X-API-Key") or "").strip()
    if hdr and hmac.compare_digest(hdr, api_key):
        return True

    auth_header = (request.headers.get("Authorization") or "").strip()
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
        if token and hmac.compare_digest(token, api_key):
            return True

    return False

@app.post("/api/register-node")
def api_register_node():
    if not _api_key_auth_ok():
        return jsonify({"error": "Unauthorized: Invalid or missing API key"}), 401
    data = request.get_json(silent=True) or {}
    node_id = (data.get("node_id") or "").strip() or "unknown_node"
    # Ensure meta is a JSON string, assume simple dump if already dict
    meta = json.dumps(data.get("meta") or {})
    con = None
    try:
        con = _connect_sqlite(META_DB)
        cur = con.cursor()
        cur.execute("INSERT INTO nodes(node_id,last_ts,meta) VALUES(?,?,?) "
                    "ON CONFLICT(node_id) DO UPDATE SET last_ts=excluded.last_ts, meta=excluded.meta",
                    (node_id, time.time(), meta))
        con.commit()
        ensure_wallet_simple(node_id)
        _cache_invalidate('leaderboard')
        return jsonify({"ok": True}), 200
    except sqlite3.Error as e:
        app_logger.error(f"Failed to register node {node_id} to {META_DB}: {e}", exc_info=True)
        return jsonify({"error": "Database error during node registration."}), 500
    finally:
        if con: con.close()


@app.route("/api/receive-embedding", methods=['POST'])
def api_receive_embedding():
    if not _api_key_auth_ok():
        return jsonify({"error": "Unauthorized: Invalid or missing API key"}), 401
    data = request.get_json(silent=True) or {}
    node_id = (data.get("node_id") or "").strip() or "unknown_node"
    embedding_data = data.get("embedding")
    context_id = data.get("context_id")

    if not embedding_data:
        return jsonify({"error": "Missing 'embedding' data."}), 400
    if not context_id:
        return jsonify({"error": "Missing 'context_id'."}), 400

    vector = json.dumps(embedding_data)
    con = None
    try:
        con = _connect_sqlite(META_DB)
        cur = con.cursor()
        cur.execute("INSERT INTO embeddings(ts,node_id,context_id,vector) VALUES(?,?,?,?)",
                    (time.time(), node_id, context_id, vector))
        con.commit()
        return jsonify({"ok": True}), 200
    except sqlite3.Error as e:
        app_logger.error(f"Failed to receive embedding for node {node_id} to {META_DB}: {e}", exc_info=True)
        return jsonify({"error": "Database error receiving embedding."}), 500
    finally:
        if con: con.close()

@app.route("/api/context-update", methods=['POST'])
def api_context_update():
    if not _api_key_auth_ok():
        return jsonify({"error": "Unauthorized: Invalid or missing API key"}), 401
    data = request.get_json(silent=True) or {}
    node_id = (data.get("node_id") or "").strip() or "unknown_node"
    text = data.get("text")
    tags_data = data.get("tags")

    if not text:
        return jsonify({"error": "Missing 'text' for context update."}), 400

    tags = json.dumps(tags_data if isinstance(tags_data, list) else [])
    con = None
    try:
        con = _connect_sqlite(META_DB)
        cur = con.cursor()
        cur.execute("INSERT INTO contexts(ts,node_id,text,tags) VALUES(?,?,?,?)",
                    (time.time(), node_id, text, tags))
        con.commit()
        return jsonify({"ok": True}), 200
    except sqlite3.Error as e:
        app_logger.error(f"Failed to update context for node {node_id} to {META_DB}: {e}", exc_info=True)
        return jsonify({"error": "Database error during context update."}), 500
    finally:
        if con: con.close()

@app.route("/api/jobs", methods=['POST'])
def api_jobs_post():
    if not _api_key_auth_ok():
        return jsonify({"error": "Unauthorized: Invalid or missing API key"}), 401
    data = request.get_json(silent=True) or {}
    node_id = (data.get("node_id") or "").strip() or "unknown_node"
    job_id = (data.get("job_id") or "").strip() or "unknown_job"
    result_data = data.get("result")

    if not result_data:
        return jsonify({"error": "Missing 'result' data for job."}), 400

    result = json.dumps(result_data)
    con = None
    try:
        con = _connect_sqlite(META_DB)
        cur = con.cursor()
        cur.execute("INSERT INTO job_results(ts,node_id,job_id,result) VALUES(?,?,?,?)",
                    (time.time(), node_id, job_id, result))
        con.commit()
        return jsonify({"ok": True}), 200
    except sqlite3.Error as e:
        app_logger.error(f"Failed to post job results for node {node_id} and job {job_id} to {META_DB}: {e}", exc_info=True)
        return jsonify({"error": "Database error posting job results."}), 500
    finally:
        if con: con.close()

# ---------------------------------------------------------------------------
# WebUI helper endpoints (Themes/Voices/Settings/Contacts/Reminders/Cleanup)
# ---------------------------------------------------------------------------
@app.after_request
def add_security_headers(resp):
    """Attach basic security headers (safe defaults for WebUI + API)."""
    try:
        # Version / identity
        resp.headers["X-SarahMemory-Version"] = str(PROJECT_VERSION)

        # Standard hardening headers
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"

        # NOTE: CSP can be strict; keep it permissive enough for current WebUI.
        # Tighten later once all asset/CDN usage is finalized.
        if "Content-Security-Policy" not in resp.headers:
            resp.headers["Content-Security-Policy"] = (
                "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:; "
                "connect-src 'self' https://api.sarahmemory.com https://ai.sarahmemory.com; "
                "img-src 'self' data: blob: https:; "
                "media-src 'self' data: blob: https:; "
                "style-src 'self' 'unsafe-inline' https:; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https:;"
            )
    except Exception as e:
        try:
            app_logger.error(f"Failed to add security headers: {e}")
        except Exception:
            pass

    # Optional FE speech script injection (gated)
    if os.getenv("SARAH_FE_SPEECH", "0") == "1":
        try:
            ct = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" in ct:
                data = resp.get_data(as_text=True)
                if data and "<html" in data.lower() and 'id="sm-fe-speech"' not in data:
                    tag = "\n<script id=\"sm-fe-speech\" src=\"/api/fe/v800/speech.js\" defer></script>\n"
                    lower = data.lower()
                    i = lower.rfind("</head>")
                    if i != -1:
                        resp.set_data(data[:i] + tag + data[i:])
                        resp.headers.pop("Content-Length", None)
        except Exception as e:
            try:
                app_logger.warning(f"Speech script injection failed: {e}")
            except Exception:
                pass

    return resp

# Centralized settings file path (robust for headless/WSGI environments)
# NOTE: Avoid KeyError at import-time if _globals_paths() returns a partial dict during early init.
try:
    _gp = _globals_paths() or {}
    _settings_dir = _gp.get("SETTINGS_DIR") or os.path.join(_gp.get("DATA_DIR", os.path.join(os.getcwd(), "data")), "settings")
    try:
        os.makedirs(_settings_dir, exist_ok=True)
    except Exception:
        pass
    SETTINGS_FILE = os.path.join(_settings_dir, "settings.json")  # SETTINGS_DIR/settings.json
except Exception:
    SETTINGS_FILE = os.path.join(os.getcwd(), "settings.json")

@app.route("/get_user_setting")
def get_user_setting():
    key = request.args.get("key", "").strip()
    if not key:
        return jsonify({"error": "Setting key is required."}), 400

    data = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            app_logger.error(f"Error reading settings file {SETTINGS_FILE}: {e}")
            data = {} # On error, treat as empty settings

    return jsonify({"value": data.get(key, "")})

@app.route("/set_user_setting", methods=['POST'])
def set_user_setting():
    payload = request.get_json(silent=True) or {}
    key = payload.get("key")
    val = payload.get("value")

    if key is None:
        return jsonify({"status": "error", "error": "Setting key is required."}), 400

    data = {}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            app_logger.error(f"Error reading settings file {SETTINGS_FILE} for update: {e}")
            data = {} # If file is corrupted, start fresh with new setting

    data[key] = val
    _ensure_dir(os.path.dirname(SETTINGS_FILE)) # Ensure settings directory exists
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return jsonify({"status":"ok"})
    except IOError as e:
        app_logger.error(f"Error writing settings file {SETTINGS_FILE}: {e}", exc_info=True)
        return jsonify({"status":"error", "error": f"Failed to save setting: {e}"}), 500


# Themes routes are fine, pathing should be robust now.

@app.route("/get_available_voices")
def get_available_voices():
    """Return available TTS voices for the WebUI.
    Prefer the richer SarahMemoryVoice bridge (v8.0) so we see both
    system voices and any registered custom voices (.pt models).
    Fallback to a direct pyttsx3 probe if that fails.
    """
    # First try the unified SarahMemoryVoice API
    sm_list_voices = None
    try:
        from SarahMemoryVoice import list_voices as sm_list_voices_func
        sm_list_voices = sm_list_voices_func
    except ImportError:
        app_logger.info("SarahMemoryVoice module not found for listing voices.")
    except Exception as e:
        app_logger.error(f"Error importing SarahMemoryVoice.list_voices: {e}", exc_info=True)

    if sm_list_voices:
        try:
            voices = sm_list_voices() or []
            if voices:
                return jsonify(voices)
        except Exception as e:
            app_logger.warning(f" get_available_voices via SarahMemoryVoice failed: {e}", exc_info=True)

    # Fallback: query local OS voices directly via pyttsx3
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty("voices") or []
        out = []
        for v in voices:
            name_val = getattr(v, "name", "") or getattr(v, "id", "")
            out.append({
                "id": getattr(v, "id", ""),
                "name": name_val
            })
        return jsonify(out)
    except ImportError:
        app_logger.info("pyttsx3 not installed. Cannot get local OS voices.")
    except Exception as e:
        app_logger.error(f"Error getting voices via pyttsx3 fallback: {e}", exc_info=True)

    return jsonify([]) # Return empty list if all methods fail


# Helper function for cleanup routes to reduce repetition
def _call_cleanup_module_func(func_name: str, *args, **kwargs):
    """Helper to call functions from SarahMemoryCleanup and handle responses."""
    try:
        import SarahMemoryCleanup as C
        fn = _safe_getattr(C, func_name)
        if callable(fn):
            result = fn(*args, **kwargs)
            return jsonify({"status": "ok", "result": str(result)}), 200
        app_logger.warning(f"SarahMemoryCleanup function '{func_name}' not found or not callable.")
        return jsonify({"status": "noop", "error": f"Cleanup function '{func_name}' not found."}), 404
    except ImportError:
        app_logger.error("SarahMemoryCleanup module not found.")
        return jsonify({"status": "error", "error": "SarahMemoryCleanup module not available."}), 503
    except Exception as e:
        app_logger.exception(f"Error in SarahMemoryCleanup function '{func_name}'.")
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/cleanup/backup_all")
def cleanup_backup_all():
    return _call_cleanup_module_func("backup_all")

@app.route("/cleanup/restore_latest")
def cleanup_restore_latest():
    return _call_cleanup_module_func("restore_latest")

@app.route("/cleanup/clear_range", methods=['POST'])
def cleanup_clear_range():
    payload = request.get_json(silent=True) or {}
    db_name = payload.get("db", "context_history.db")
    seconds = int(payload.get("seconds", 0) or 0)
    return _call_cleanup_module_func("clear_range", db_name, seconds if seconds > 0 else None)

@app.route("/cleanup/tidy_logs")
def cleanup_tidy_logs():
    return _call_cleanup_module_func("tidy_logs")


# Camera/Mic/Voice toggles
@app.route("/toggle_camera")
def toggle_camera():
    state = request.args.get("state","").lower() in ("true","1","yes","on")
    app.config["CAMERA_ON"] = state # Use app.config for global state
    return jsonify({"status":"ok","camera": state})

@app.route("/toggle_microphone", methods=["POST"])
def toggle_microphone():
    """
    Enable/disable microphone capture for the UI.
    Accepts JSON: { "enabled": true/false }
    """
    try:
        data = request.get_json(silent=True) or {}
        desired = bool(data.get("enabled", True))

        global MIC_ON, MIC_ENABLED
        MIC_ON = desired
        MIC_ON = desired
        MIC_ENABLED = MIC_ON

        try:
            save_state("MIC_ON", bool(desired))
        except Exception:
            pass

        return jsonify({"ok": True, "mic_enabled": bool(desired)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/toggle_voice_output", methods=["POST"])
def toggle_voice_output():
    """
    Enable/disable voice output for the UI.
    Accepts JSON: { "enabled": true/false }
    """
    try:
        data = request.get_json(silent=True) or {}
        desired = bool(data.get("enabled", True))

        global VOICE_OUTPUT_ON, VOICE_OUTPUT_ENABLED
        VOICE_OUTPUT_ON = desired
        TTS_ON = desired
        TTS_ENABLED = TTS_ON
        VOICE_OUTPUT_ON = TTS_ON
        VOICE_OUTPUT_ENABLED = TTS_ON

        try:
            save_state("TTS_ON", bool(desired))
        except Exception:
            pass

        return jsonify({"ok": True, "voice_output_enabled": bool(desired)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.route("/check_call_active")
def check_call_active():
    return jsonify({"active": app.config.get("CALL_ACTIVE", False)}) # Use app.config

@app.route("/initiate_call", methods=['POST'])
def initiate_call():
    data = request.get_json(silent=True) or {}
    number = (data.get("number") or "").strip()
    app.config["CALL_ACTIVE"] = bool(number)  # Use app.config
    return jsonify({"status":"call_started","to":number})

# File transfer / ingest
@app.route("/send_file_to_remote", methods=['POST'])
def send_file_to_remote():
    payload = request.get_json(silent=True) or {}
    fname = payload.get("filename")
    b64 = payload.get("data")

    if not fname or not b64:
        return jsonify({"status": "error", "error": "Missing filename or data."}), 400

    try:
        data = base64.b64decode(b64.encode("utf-8"))
    except Exception as e:
        return jsonify({"status": "error", "error": f"Invalid base64 data: {e}"}), 400

    if os.name == "nt":
        out_dir = os.path.join(os.environ.get("USERPROFILE"), "Downloads") if "USERPROFILE" in os.environ else r"C:\Users\Public\Downloads"
    else:
        out_dir = os.path.join(DATA_DIR, "downloads") # Use DATA_DIR for cross-platform and server-safe

    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, os.path.basename(fname)) # Use basename to prevent path traversal
    try:
        with open(out_path, "wb") as f:
            f.write(data)
        return jsonify({"message": f"Sent file to remote user (saved locally): {fname}", "path": out_path}), 200
    except IOError as e:
        app_logger.error(f"Failed to save remote file {out_path}: {e}", exc_info=True)
        return jsonify({"status": "error", "error": f"Failed to save file locally: {e}"}), 500


@app.route("/ingest_local_file", methods=['POST'])
def ingest_local_file():
    payload = request.get_json(silent=True) or {}
    fname = payload.get("filename")
    b64 = payload.get("data")

    if not fname or not b64:
        return jsonify({"status": "error", "error": "Missing filename or data."}), 400

    _paths = _globals_paths()
    DATASETS_DIR = _paths["DATASETS_DIR"]
    DOCUMENTS_DIR = _paths["DOCUMENTS_DIR"]
    try:
        data = base64.b64decode(b64.encode("utf-8"))
    except Exception as e:
        return jsonify({"status": "error", "error": f"Invalid base64 data: {e}"}), 400

    out_dir = DOCUMENTS_DIR or DATASETS_DIR # Default to DOCUMENTS_DIR if available, else DATASETS_DIR
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, os.path.basename(fname)) # Use basename to prevent path traversal
    try:
        with open(out_path, "wb") as f:
            f.write(data)
        return jsonify({"message": f"Stored file in local documents: {fname}", "path": out_path}), 200
    except IOError as e:
        app_logger.error(f"Failed to ingest local file {out_path}: {e}", exc_info=True)
        return jsonify({"status": "error", "error": f"Failed to store file locally: {e}"}), 500


# Contacts
USER_DB_PATH = _globals_paths() # Cache user DB path

def _init_contacts_db(db_path):
    """Helper to initialize contacts table."""
    con = None
    try:
        con = _connect_sqlite(db_path)
        cur = con.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS contacts (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, number TEXT)")
        con.commit()
    except sqlite3.Error as e:
        app_logger.error(f"Failed to initialize contacts database at {db_path}: {e}")
        raise # Re-raise to ensure caller knows about failure
    finally:
        if con: con.close()


@app.route("/get_all_contacts")
def get_all_contacts():
    con = None
    try:
        _init_contacts_db(USER_DB_PATH) # Ensure table exists
        con = _connect_sqlite(USER_DB_PATH)
        cur = con.cursor()
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT id, name, number FROM contacts ORDER BY name COLLATE NOCASE")
        rows = [dict(r) for r in cur.fetchall()]
        return jsonify({"contacts": rows})
    except Exception as e:
        app_logger.exception(f"Error fetching contacts from {USER_DB_PATH}.")
        return jsonify({"error": "Failed to retrieve contacts."}), 500
    finally:
        if con: con.close()

@app.route("/add_contact", methods=['POST'])
def add_contact():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    number = (data.get("number") or "").strip()

    if not name or not number:
        return jsonify({"status":"error", "error":"Name and number are required to add contact."}), 400

    con = None
    try:
        _init_contacts_db(USER_DB_PATH) # Ensure table exists
        con = _connect_sqlite(USER_DB_PATH)
        cur = con.cursor()
        cur.execute("INSERT INTO contacts(name,number) VALUES(?,?)",(name,number))
        con.commit()
        return jsonify({"status":"ok"}), 200
    except sqlite3.Error as e:
        app_logger.error(f"Failed to add contact {name} to {USER_DB_PATH}: {e}", exc_info=True)
        return jsonify({"status":"error", "error": "Database error adding contact."}), 500
    finally:
        if con: con.close()

@app.route("/delete_contact", methods=['POST'])
def delete_contact():
    data = request.get_json(silent=True) or {}
    rid = data.get("id")
    if not isinstance(rid, int):
        return jsonify({"status": "error", "error": "Invalid contact ID provided."}), 400

    con = None
    try:
        con = _connect_sqlite(USER_DB_PATH)
        cur = con.cursor()
        cur.execute("DELETE FROM contacts WHERE id=?", (rid,))
        if cur.rowcount == 0:
            return jsonify({"status": "error", "error": f"Contact with ID {rid} not found."}), 404
        con.commit()
        return jsonify({"status":"deleted", "id": rid}), 200
    except sqlite3.Error as e:
        app_logger.error(f"Failed to delete contact with ID {rid} from {USER_DB_PATH}: {e}", exc_info=True)
        return jsonify({"status":"error", "error": "Database error deleting contact."}), 500
    finally:
        if con: con.close()

# Reminders
REMINDERS_DB_PATH = os.path.join(_globals_dir("DATA_DIR", "data"), "reminders.db")

def _init_reminders_db(db_path):
    """Helper to initialize reminders table."""
    con = None
    try:
        con = _connect_sqlite(db_path)
        cur = con.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS reminders (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, time TEXT, note TEXT)')
        con.commit()
    except sqlite3.Error as e:
        app_logger.error(f"Failed to initialize reminders database at {db_path}: {e}")
        raise # Re-raise to ensure caller knows about failure
    finally:
        if con: con.close()

@app.route("/get_reminders")
def get_reminders():
    con = None
    try:
        _init_reminders_db(REMINDERS_DB_PATH) # Ensure table exists
        con = _connect_sqlite(REMINDERS_DB_PATH)
        cur = con.cursor()
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute('SELECT id, title, time, note FROM reminders ORDER BY time ASC')
        rows = [dict(r) for r in cur.fetchall()]
        return jsonify({'reminders': rows})
    except Exception as e:
        app_logger.exception(f"Error fetching reminders from {REMINDERS_DB_PATH}.")
        return jsonify({"error": "Failed to retrieve reminders."}), 500
    finally:
        if con: con.close()

@app.route("/save_reminder", methods=['POST'])
def save_reminder():
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    time_s = (payload.get("time") or "").strip()
    note = payload.get("note") or ""

    if not title or not time_s:
        return jsonify({"status":"error", "error":"Title and time are required to save reminder."}), 400

    con = None
    try:
        _init_reminders_db(REMINDERS_DB_PATH) # Ensure table exists
        con = _connect_sqlite(REMINDERS_DB_PATH)
        cur = con.cursor()
        cur.execute('INSERT INTO reminders(title, time, note) VALUES(?,?,?)',(title, time_s, note))
        con.commit()
        rid = cur.lastrowid
        return jsonify({"status":"ok","id":rid}), 200
    except sqlite3.Error as e:
        app_logger.error(f"Failed to save reminder '{title}' to {REMINDERS_DB_PATH}: {e}", exc_info=True)
        return jsonify({"status":"error", "error": "Database error saving reminder."}), 500
    finally:
        if con: con.close()

@app.route("/delete_reminder", methods=['POST'])
def delete_reminder():
    payload = request.get_json(silent=True) or {}
    rid = payload.get("id")

    if not isinstance(rid, int):
        return jsonify({"status": "error", "error": "Invalid reminder ID provided."}), 400

    con = None
    try:
        con = _connect_sqlite(REMINDERS_DB_PATH)
        cur = con.cursor()
        cur.execute('DELETE FROM reminders WHERE id=?', (rid,))
        if cur.rowcount == 0:
            return jsonify({"status": "error", "error": f"Reminder with ID {rid} not found."}), 404
        con.commit()
        return jsonify({"status":"deleted", "id": rid}), 200
    except sqlite3.Error as e:
        app_logger.exception(f"Failed to delete reminder with ID {rid} from {REMINDERS_DB_PATH}.")
        return jsonify({"status":"error", "error": "Database error deleting reminder."}), 500
    finally:
        if con: con.close()

@app.route("/run_automation_trigger", methods=['POST'])
def run_automation_trigger():
    payload = request.get_json(silent=True) or {}
    try:
        import SarahMemoryAiFunctions as F
        run_automation_func = _safe_getattr(F, "run_automation")
        if callable(run_automation_func):
            res = run_automation_func(payload)
            return jsonify({"status":"ok","result":str(res)}), 200
        app_logger.warning("SarahMemoryAiFunctions.run_automation not found or not callable.")
        return jsonify({"status":"noop", "message":"Automation function not available."}), 404
    except ImportError:
        app_logger.error("SarahMemoryAiFunctions module not found for automation trigger.")
        return jsonify({"status":"error", "error":"Automation module not available."}), 503
    except Exception as e:
        app_logger.exception("Error running automation trigger.")
        return jsonify({"status":"error", "error":str(e)}), 500

# Calendar + Chat history (for Web UI)
CHAT_HISTORY_DB_PATH = os.path.join(_globals_dir("DATA_DIR", "data"), "context_history.db")

@app.route("/get_chat_threads_by_date")
def get_chat_threads_by_date():
    date_filter = request.args.get("date", "").strip()  # YYYY-MM-DD
    con = None
    try:
        con = _connect_sqlite(CHAT_HISTORY_DB_PATH)
        cur = con.cursor()
        q = "SELECT id, timestamp, user_input AS preview FROM conversations"
        params = []
        if date_filter:
            q += " WHERE date(timestamp)=?"
            params.append(date_filter)
        q += " ORDER BY timestamp DESC" # Order by newest first
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute(q, tuple(params))
        rows = [dict(r) for r in cur.fetchall()]
        return jsonify({"threads": rows})
    except sqlite3.Error as e:
        app_logger.error(f"Failed to fetch chat threads by date from {CHAT_HISTORY_DB_PATH}: {e}", exc_info=True)
        return jsonify({"error": "Database error fetching chat threads."}), 500
    except Exception as e:
        app_logger.exception(f"Unexpected error fetching chat threads by date.")
        return jsonify({"error": str(e)}), 500
    finally:
        if con: con.close()

@app.route("/get_conversation_by_id")
def get_conversation_by_id():
    convo_id = request.args.get("id")
    if not convo_id:
        return jsonify({"error": "Conversation ID is required."}), 400

    con = None
    try:
        con = _connect_sqlite(CHAT_HISTORY_DB_PATH)
        cur = con.cursor()
        # Assuming conversations table has role, text, and metadata
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        cur.execute("SELECT role, text, metadata AS meta FROM conversations WHERE id = ?", (convo_id,))
        rows = [dict(r) for r in cur.fetchall()]
        if not rows:
            return jsonify({"error": f"Conversation with ID {convo_id} not found."}), 404
        return jsonify(rows)
    except sqlite3.Error as e:
        app_logger.error(f"Failed to fetch conversation by ID {convo_id} from {CHAT_HISTORY_DB_PATH}: {e}", exc_info=True)
        return jsonify({"error": "Database error fetching conversation."}), 500
    except Exception as e:
        app_logger.exception(f"Unexpected error fetching conversation by ID {convo_id}.")
        return jsonify({"error": str(e)}), 500
    finally:
        if con: con.close()

# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
@app.get("/get_theme_files") # Use app.get for GET requests
def get_theme_files():
    final_themes_dir = None
    try:
        import SarahMemoryGlobals as G
        # Prioritize checking THEMES_DIR from SarahMemoryGlobals
        if hasattr(G, "THEMES_DIR"):
            final_themes_dir = G.THEMES_DIR
    except Exception:
        pass # Fallback to local logic if SarahMemoryGlobals has issues

    if final_themes_dir is None: # If not found via Globals, use local logic
        # Re-evaluating path for local fallback to ensure accuracy
        base_dir_local = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_dir_local = os.path.join(base_dir_local, "data")
        themes_dirA_local = os.path.join(data_dir_local, "mods", "themes")
        themes_dirB_local = os.path.join(data_dir_local, "themes")

        if os.path.isdir(themes_dirA_local):
            final_themes_dir = themes_dirA_local
        elif os.path.isdir(themes_dirB_local):
            final_themes_dir = themes_dirB_local
        else:
            final_themes_dir = themes_dirA_local # Default to this even if it doesn't exist yet

    files = []
    if final_themes_dir and os.path.isdir(final_themes_dir):
        for dp, dn, fnames in os.walk(final_themes_dir):
            for f in fnames:
                # Optimized check for file extensions
                if f.lower().endswith((".css", ".json", ".yml", ".yaml", ".toml", ".png", ".jpg", ".jpeg", ".svg", ".ttf", ".otf")):
                    rel = os.path.relpath(os.path.join(dp, f), final_themes_dir).replace("\\", "/")
                    files.append(rel)
    else:
        app_logger.warning(f"Theme directory '{final_themes_dir}' not found or is not a directory.")

    # Determine active_root for jsonify
    # This logic still refers to the old A/B distinction for `active_root`
    # It might be more robust to derive `active_root` from `final_themes_dir` if it's dynamic
    data_dir_for_json_path = DATA_DIR # Use the global DATA_DIR
    themes_dirA_for_json_path = os.path.join(data_dir_for_json_path, "mods", "themes")
    themes_dirB_for_json_path = os.path.join(data_dir_for_json_path, "themes")

    if os.path.isdir(themes_dirB_for_json_path): # Prefer /data/themes if it contains actual themes
        active_root = "/api/data/themes"
    elif os.path.isdir(themes_dirA_for_json_path): # Then /data/mods/themes
        active_root = "/api/data/mods/themes"
    else: # Fallback
        active_root = "/api/data/mods/themes" # Defaulting to the mods path

    return jsonify({"root": active_root, "count": len(files), "files": sorted(files)})

@app.route("/api/data/themes/<path:filename>")
def serve_theme_file_A(filename):
    data_dir_for_serving = DATA_DIR # Use the determined global DATA_DIR
    root = os.path.join(data_dir_for_serving, "themes")
    # Basic path traversal protection
    if ".." in filename or filename.startswith("/"):
        return jsonify({"error": "Invalid path"}), 400
    try:
        return send_from_directory(root, filename)
    except Exception as e:
        app_logger.error(f"Error serving theme file from {root}/{filename}: {e}")
        return jsonify({"error": "Theme file not found or accessible"}), 404


@app.route("/api/data/mods/themes/<path:filename>")
def serve_theme_file_B(filename):
    data_dir_for_serving = DATA_DIR # Use the determined global DATA_DIR
    root = os.path.join(data_dir_for_serving, "mods", "themes")
    # Basic path traversal protection
    if ".." in filename or filename.startswith("/"):
        return jsonify({"error": "Invalid path"}), 400
    try:
        return send_from_directory(root, filename)
    except Exception as e:
        app_logger.error(f"Error serving theme file from {root}/{filename}: {e}")
        return jsonify({"error": "Theme file not found or accessible"}), 404


# --- Boot Launcher / Health (idempotent server-side autostart) ---
import subprocess

PID_FILE = os.path.join(DATA_DIR, "sarahmemory.pid") # Using global DATA_DIR

def _is_running():
    """Checks if SarahMemoryMain process is already running based on PID file."""
    try:
        if not os.path.exists(PID_FILE):
            return False
        with open(PID_FILE, "r") as f:
            pid_s = (f.read() or "").strip()
        if not pid_s:
            return False
        pid = int(pid_s)
        # Best-effort: os.kill(pid, 0) works on POSIX; on Windows, it might just raise an error
        # rather than allowing os.kill(pid, 0) to check existence. subprocess.os.name handles.
        if os.name == "posix": # Linux/macOS
            try:
                os.kill(pid, 0) # Check if process exists
                return True
            except OSError: # Process does not exist
                return False
        elif os.name == "nt": # Windows
            import ctypes
            # Check if PID is active on Windows
            kernel32 = ctypes.WinDLL('kernel32')
            handle = kernel32.OpenProcess(0x1000, False, pid) # PROCESS_QUERY_LIMITED_INFORMATION
            if handle is not None:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            app_logger.warning(f"Unknown OS type '{os.name}'. Cannot reliably check PID {pid}.")
            return False # Conservative default
    except (ValueError, IOError) as e:
        app_logger.debug(f"PID file read error or invalid PID: {e}")
        return False
    except Exception as e:
        app_logger.error(f"Unexpected error in _is_running: {e}", exc_info=True)
        return False

def _write_pid(pid: int):
    """Writes the current process PID to a file."""
    try:
        _ensure_dir(DATA_DIR) # Ensure DATA_DIR exists before writing PID
        with open(PID_FILE, "w") as f:
            f.write(str(pid))
    except (IOError, OSError) as e:
        app_logger.error(f"Failed to write PID file {PID_FILE}: {e}")
    except Exception as e:
        app_logger.error(f"Unexpected error writing PID file: {e}", exc_info=True)


def _start_sarah_main():
    """Spawn the canonical boot chain (SarahMemoryMain.py) in background."""
    try:
        if _is_running():
            app_logger.info("SarahMemoryMain is already running. Skipping new spawn.")
            return True
    except Exception:
        pass

    main_py_path = os.path.join(BASE_DIR, "SarahMemoryMain.py")
    if not os.path.exists(main_py_path):
        app_logger.error(f"SarahMemoryMain.py not found at {main_py_path}. Cannot start main process.")
        return False

    # Prefer the currently running interpreter, then common venv locations, then system python.
    candidates = [
        [sys.executable, main_py_path],
        [os.path.join(BASE_DIR, "venv", "Scripts", "python.exe"), main_py_path],   # Windows venv
        [os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe"), main_py_path], # Windows .venv
        [os.path.join(BASE_DIR, "venv", "bin", "python3"), main_py_path],         # Linux/mac venv
        [os.path.join(BASE_DIR, ".venv", "bin", "python3"), main_py_path],
        ["python", main_py_path],
        ["python3", main_py_path],
    ]

    # Filter invalid interpreter paths (except bare commands)
    final_candidates = []
    for cmd in candidates:
        try:
            exe = cmd[0]
            if os.path.isabs(exe) and not os.path.exists(exe):
                continue
            final_candidates.append(cmd)
        except Exception:
            continue

    # Try each candidate until one spawns successfully
    for cmd in final_candidates:
        try:
            app_logger.info(f"Attempting to start SarahMemoryMain: {cmd}")
            proc = subprocess.Popen(
                cmd,
                cwd=BASE_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
            )
            try:
                _write_pid(proc.pid)
            except Exception:
                pass
            return True
        except Exception as e:
            app_logger.warning(f"Failed to start SarahMemoryMain with {cmd}: {e}")

    return False

@app.post("/api/launch")
def api_launch():
    try:
        if _is_running():
            return jsonify({"ok": True, "running": True, "msg": "SarahMemoryMain is already running."}), 200
        ok = _start_sarah_main()
        return jsonify({"ok": bool(ok), "running": bool(ok), "msg": "SarahMemoryMain launched successfully." if ok else "Failed to launch SarahMemoryMain."}), (200 if ok else 500)
    except Exception as e:
        app_logger.exception("Error during launch API call.")
        return jsonify({"ok": False, "error": str(e), "msg": "Internal server error during launch."}), 500


# ============================================================================
# Phase B: Authentication System
# ============================================================================

# JWT Configuration (Variables are kept as per your original file for .env consistency)
JWT_SECRET = os.getenv("SARAH_JWT_SECRET") or os.getenv("JWT_SECRET_KEY")
if not JWT_SECRET:
    app_logger.critical("JWT_SECRET is not set. Using default insecure key. THIS IS DANGEROUS FOR PRODUCTION!")
    JWT_SECRET = "change-this-secret-key-in-production"

JWT_ALGORITHM = 'HS256'
JWT_EXP_DELTA_DAYS = 7

def generate_jwt_token(user_id, email, display_name): # Added display_name
    """Generate JWT token for user."""
    payload = {
        'user_id': user_id,
        'email': email,
        'display_name': display_name, # Include display_name in token
        'exp': datetime.utcnow() + timedelta(days=JWT_EXP_DELTA_DAYS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token):
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        # Basic validation: ensure essential keys are present
        if 'user_id' in payload and 'email' in payload and 'exp' in payload:
            return payload
        app_logger.warning("JWT payload missing essential keys.")
        return None
    except jwt.ExpiredSignatureError:
        app_logger.info("Expired JWT token received.")
        return None
    except jwt.InvalidTokenError:
        app_logger.warning("Invalid JWT token received.")
        return None
    except Exception as e:
        app_logger.error(f"Unexpected error during JWT verification: {e}", exc_info=True)
        return None

def require_auth(f):
    """Decorator to require authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'Authentication required. Token missing.'}), 401

        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Authentication failed. Invalid or expired token.'}), 401

        request.user_id = payload
        request.user_email = payload
        request.user_display_name = payload # Store display_name
        return f(*args, **kwargs)
    return decorated_function


@app.route('/api/auth/register', methods=['POST'])
def auth_register():
    """Phase B: Register new user account."""
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        pin = data.get('pin', '')
        display_name = data.get('display_name', '') # Keep display_name in input

        # Validate input
        if not email or '@' not in email or '.' not in email: # More robust email check
            return jsonify({'error': 'Invalid email format.'}), 400
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters.'}), 400
        if not pin or not pin.isdigit() or len(pin) != 4: # Strict 4-digit check
            return jsonify({'error': 'PIN must be exactly 4 digits.'}), 400
        if not display_name: # Ensure display name
            display_name = email.split('@', 1)[0] # Default if not provided

        # Import database functions
        try:
            from SarahMemoryDatabase import sm_get_user_by_email, sm_create_user, _get_cloud_conn, sm_insert_email_verification
        except ImportError:
            app_logger.error("SarahMemoryDatabase module not found for authentication.")
            return jsonify({'error': 'Database module unavailable.'}), 503
        except Exception as e:
            app_logger.error(f"Error importing SarahMemoryDatabase functions: {e}", exc_info=True)
            return jsonify({'error': 'Database module configuration error.'}), 503

        # Check if user already exists
        conn = None
        try:
            conn = _get_cloud_conn()
            if not conn:
                return jsonify({'error': 'Cloud database connection unavailable.'}), 503

            existing_user = sm_get_user_by_email(email, conn) # Pass connection to avoid re-opening
            if existing_user: # sm_get_user_by_email should return None if not found
                return jsonify({'error': 'Email already registered.'}), 409

            # Hash password and PIN
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            pin_hash = bcrypt.hashpw(pin.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            # Create user in database
            user_id = sm_create_user(email, display_name, password_hash, pin_hash, conn) # Pass connection
            if not user_id:
                raise Exception("Failed to create user in database.")

            # Generate and insert verification code
            verification_code = secrets.token_urlsafe(18)
            sm_insert_email_verification(user_id, email, verification_code, request.remote_addr, request.headers.get('User-Agent', ''), conn)

            # Send verification email
            send_verification_email(email, verification_code)

            return jsonify({
                'success': True,
                'user_id': user_id,
                'message': 'Registration successful. Please check your email for verification code.'
            }), 201

        except Exception as e:
            app_logger.exception(f" Registration failed for {email}.")
            if conn: conn.rollback() # Rollback on error
            return jsonify({'error': f'Registration failed: {str(e)}'}), 500
        finally:
            if conn: conn.close()

    except Exception as e:
        app_logger.exception(f" Unhandled error during register route processing.")
        return jsonify({'error': 'Internal server error during registration.'}), 500


@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    """Phase B: Login user with email, password, and PIN."""
    try:
        data = request.get_json(silent=True) or {}
        email = (data.get('email') or '').strip().lower()
        password = data.get('password') or ''
        pin = data.get('pin') or ''

        if not email or not password or not pin:
            return jsonify({'error': 'Email, password, and PIN are required.'}), 400

        # Import database function (cloud user auth)
        try:
            from SarahMemoryDatabase import _get_cloud_conn, sm_get_user_auth_data, sm_update_last_login
        except ImportError:
            app_logger.error("SarahMemoryDatabase module not found for authentication.")
            return jsonify({'error': 'Database module unavailable.'}), 503
        except Exception as e:
            app_logger.error(f"Error importing SarahMemoryDatabase functions: {e}", exc_info=True)
            return jsonify({'error': 'Database module configuration error.'}), 503

        conn = None
        try:
            conn = _get_cloud_conn()
            if not conn:
                return jsonify({'error': 'Cloud database connection unavailable.'}), 503

            user_auth = sm_get_user_auth_data(email, conn)
            if not user_auth:
                return jsonify({'error': 'Invalid credentials.'}), 401

            # Normalize auth record
            def _field(obj, *names, default=None):
                if isinstance(obj, dict):
                    for n in names:
                        if n in obj and obj[n] is not None:
                            return obj[n]
                try:
                    for n in names:
                        try:
                            v = obj[n]
                            if v is not None:
                                return v
                        except Exception:
                            pass
                except Exception:
                    pass
                return default

            user_id = _field(user_auth, 'user_id', 'id', 'uid', default=email)
            display_name = _field(user_auth, 'display_name', 'name', 'username', default=email.split('@')[0])
            pw_hash = _field(user_auth, 'password_hash', 'pass_hash', 'password', 'pw_hash', default=None)
            pin_hash = _field(user_auth, 'pin_hash', 'pinhash', 'pin', default=None)
            is_active = _field(user_auth, 'is_active', 'active', default=1)

            if str(is_active) in ("0", "false", "False", "no", "NO"):
                return jsonify({'error': 'Account disabled. Please contact support.'}), 403

            if not pw_hash or not bcrypt.checkpw(password.encode('utf-8'), str(pw_hash).encode('utf-8')):
                return jsonify({'error': 'Invalid credentials.'}), 401

            if not pin_hash or not bcrypt.checkpw(pin.encode('utf-8'), str(pin_hash).encode('utf-8')):
                return jsonify({'error': 'Invalid credentials.'}), 401

            try:
                sm_update_last_login(user_id, conn)
            except Exception:
                pass

            token = generate_jwt_token(user_id, email, display_name)
            return jsonify({
                'ok': True,
                'token': token,
                'user': {'user_id': user_id, 'email': email, 'display_name': display_name}
            }), 200

        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

    except Exception as e:
        app_logger.error(f"auth_login failed: {e}", exc_info=True)
        return jsonify({'error': 'Login failed.'}), 500

@app.get("/api/auth/verify-email")
def auth_verify_email():
    """Phase B: Verify email with code."""
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        code = data.get('code', '').strip()

        if not email or not code:
            return jsonify({'error': 'Email and verification code are required.'}), 400

        try:
            from SarahMemoryDatabase import _get_cloud_conn, sm_get_user_by_email, sm_get_verification_entry, sm_verify_user_email
        except ImportError:
            app_logger.error("SarahMemoryDatabase module not found for email verification.")
            return jsonify({'error': 'Database module unavailable.'}), 503
        except Exception as e:
            app_logger.error(f"Error importing SarahMemoryDatabase functions: {e}", exc_info=True)
            return jsonify({'error': 'Database module configuration error.'}), 503

        conn = None
        try:
            conn = _get_cloud_conn()
            if not conn:
                return jsonify({'error': 'Cloud database connection unavailable.'}), 503

            user = sm_get_user_by_email(email, conn)
            if not user:
                return jsonify({'error': 'User not found.'}), 404

            verification_entry = sm_get_verification_entry(user, code, conn)

            if not verification_entry:
                return jsonify({'error': 'Invalid or expired verification code.'}), 400

            # Additional check if it's already verified
            if verification_entry.get('verified_at'):
                return jsonify({'error': 'Email already verified. Please try logging in.'}), 409

            # Mark as verified
            sm_verify_user_email(user, verification_entry, conn)

            return jsonify({'success': True, 'message': 'Email verified successfully.'}), 200

        except Exception as e:
            app_logger.exception(f" Email verification failed for {email}.")
            if conn: conn.rollback() # Rollback on error
            return jsonify({'error': f'Verification failed: {str(e)}'}), 500
        finally:
            if conn: conn.close()

    except Exception as e:
        app_logger.exception(f" Unhandled error during email verification route processing.")
        return jsonify({'error': 'Internal server error during email verification.'}), 500

@app.route('/api/user/preferences', methods=['GET', 'PUT', 'POST'])
@require_auth
def user_preferences():
    """Phase B: Get or update user preferences."""
    conn = None
    try:
        from SarahMemoryDatabase import sm_get_user_preferences, sm_update_user_preferences, _get_cloud_conn
        conn = _get_cloud_conn()
        if not conn:
            return jsonify({'error': 'Cloud database connection unavailable.'}), 503

        if request.method == 'GET':
            prefs = sm_get_user_preferences(request.user_id, conn)
            return jsonify(prefs), 200

        elif request.method == 'PUT':
            data = request.json
            success = sm_update_user_preferences(request.user_id, data, conn)
            if success:
                return jsonify({'success': True}), 200
            else:
                return jsonify({'error': 'Failed to update preferences.'}), 500
    except ImportError:
        app_logger.error("SarahMemoryDatabase module not found for user preferences.")
        return jsonify({'error': 'Database module unavailable.'}), 503
    except Exception as e:
        app_logger.exception(f" Preferences operation failed for user {request.user_id}.")
        return jsonify({'error': f'Operation failed: {str(e)}'}), 500
    finally:
        if conn: conn.close()


def send_verification_email(email, code):
    """Phase B: Send verification email with code."""
    smtp_host = os.getenv('SMTP_HOST')
    smtp_port = int(os.getenv('SMTP_PORT', 587))
    smtp_user = os.getenv('SMTP_USER')
    smtp_password = os.getenv('SMTP_PASSWORD')
    smtp_from = os.getenv('SMTP_FROM_EMAIL', 'noreply@sarahmemory.com')

    if not smtp_user or not smtp_password or not smtp_host:
        app_logger.warning(" SMTP not fully configured (missing host, user, or password). Skipping email to %s.", email)
        return

    msg = MIMEMultipart('alternative')
    msg = 'SarahMemory Email Verification'
    msg = smtp_from
    msg = email

    text = f"""
Welcome to SarahMemory!

Your verification code is: {code}

This code expires in 15 minutes.

If you didn't request this, please ignore this email.
    """

    html = f"""
<html>
  <body style="font-family: Arial, sans-serif;">
    <h2>Welcome to SarahMemory!</h2>
    <p>Your verification code is:</p>
    <h1 style="background: #5f9ef7; color: white; padding: 20px; text-align: center; font-size: 32px; letter-spacing: 5px;">
      {code}
    </h1>
    <p>This code expires in 15 minutes.</p>
    <p style="color: #666; font-size: 12px;">If you didn't request this, please ignore this email.</p>
  </body>
</html>
    """

    msg.attach(MIMEText(text, 'plain'))
    msg.attach(MIMEText(html, 'html'))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_from, email, msg.as_string())
        app_logger.info(" Verification email sent to %s.", email)
    except smtplib.SMTPAuthenticationError:
        app_logger.error(f" SMTP authentication error for user {smtp_user}. Check SMTP_PASSWORD.")
    except smtplib.SMTPException as e:
        app_logger.error(f" SMTP error sending email to {email}: {e}", exc_info=True)
    except Exception as e:
        app_logger.error(f" Unexpected error sending email to {email}: {e}", exc_info=True)

# ===========================================================================
# AVATAR PANEL / MULTIMEDIA / VIDEO CONFERENCE API ROUTES
# ===========================================================================
# These routes integrate with SarahMemoryAvatarPanel.py to provide
# multimedia display, avatar animation, desktop mirror, and video conferencing

_avatar_panel_api = None # Global instance for caching the API object


def get_avatar_panel_api():
    """Get or create the Avatar Panel API instance, caching it."""
    global _avatar_panel_api
    if _avatar_panel_api is None:
        try:
            # Prefer importing from UnifiedAvatarController as per AGI spec
            from UnifiedAvatarController import get_panel_api
            _avatar_panel_api = get_panel_api()
            if _avatar_panel_api:
                app_logger.info("Successfully loaded Avatar Panel API via UnifiedAvatarController.")
            else:
                app_logger.warning("UnifiedAvatarController.get_panel_api returned None.")
        except ImportError:
            try: # Fallback to older SarahMemoryAvatarPanel if UnifiedAvatarController is not ready
                from SarahMemoryAvatarPanel import get_panel_api as smap_get_panel_api
                _avatar_panel_api = smap_get_panel_api()
                if _avatar_panel_api:
                    app_logger.info("Successfully loaded Avatar Panel API via SarahMemoryAvatarPanel (fallback).")
                else:
                    app_logger.warning("SarahMemoryAvatarPanel.get_panel_api returned None.")
            except ImportError:
                app_logger.error("Neither UnifiedAvatarController nor SarahMemoryAvatarPanel found. Avatar features disabled.")
            except Exception as e:
                app_logger.error(f" Error loading panel API via SarahMemoryAvatarPanel: {e}", exc_info=True)
        except Exception as e:
            app_logger.error(f" Error loading panel API via UnifiedAvatarController: {e}", exc_info=True)
    return _avatar_panel_api

def _avatar_api_response_wrapper(func):
    """Decorator to standardize responses for avatar panel API calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        api = get_avatar_panel_api()
        if not api:
            return jsonify({"error": "Avatar panel not available or initialized."}), 503
        try:
            result = func(api, *args, **kwargs)
            return jsonify(result), 200
        except Exception as e:
            app_logger.exception(f"Error in avatar API endpoint '{request.path}'.")
            return jsonify({"error": str(e), "message": "Failed to perform avatar action."}), 500
    return wrapper

@app.route("/api/avatar/state", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_get_state(api):
    return api.get_state()

@app.route("/api/avatar/mode", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_set_mode(api):
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "AVATAR_2D")
    return api.set_mode(mode)

@app.route("/api/avatar/emotion", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_set_emotion(api):
    data = request.get_json(silent=True) or {}
    emotion = data.get("emotion", "neutral")
    intensity = data.get("intensity", 1.0)
    # Validate intensity is a float between 0.0 and 1.0
    try:
        intensity = float(intensity)
        if not (0.0 <= intensity <= 1.0):
            raise ValueError("Intensity must be between 0.0 and 1.0")
    except ValueError as e:
        return jsonify({"error": f"Invalid intensity parameter: {e}"}), 400
    return api.set_emotion(emotion, intensity)

@app.route("/api/avatar/frame", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_get_frame(api):
    width = int(request.args.get("width", 300))
    height = int(request.args.get("height", 300))
    format = request.args.get("format", "base64") # "base64" or "binary" if streaming
    # Consider validating format here
    return api.get_avatar_frame(width, height, format)

@app.route("/api/avatar/lipsync", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_control_lipsync(api):
    data = request.get_json(silent=True) or {}
    action = data.get("action", "start")
    duration = data.get("duration", 0.0)
    if action == "start":
        return api.start_lip_sync(float(duration))
    elif action == "stop":
        return api.stop_lip_sync()
    else:
        return jsonify({"error": "Invalid action for lipsync. Must be 'start' or 'stop'."}), 400

@app.route("/api/avatar/conference/start", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_conference_start(api):
    data = request.get_json(silent=True) or {}
    peer_id = data.get("peer_id", "")
    video = data.get("video", True)
    audio = data.get("audio", True)
    if not peer_id:
        return jsonify({"error": "Peer ID is required to start a conference."}), 400
    return api.start_call(peer_id, video, audio)

@app.route("/api/avatar/conference/answer", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_conference_answer(api):
    data = request.get_json(silent=True) or {}
    peer_id = data.get("peer_id", "")
    if not peer_id:
        return jsonify({"error": "Peer ID is required to answer a conference."}), 400
    return api.answer_call(peer_id)

@app.route("/api/avatar/conference/end", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_conference_end(api):
    return api.end_call()

@app.route("/api/avatar/conference/toggle", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_conference_toggle(api):
    data = request.get_json(silent=True) or {}
    media_type = data.get("type", "video") # "video" or "audio"
    if media_type == "video":
        return api.toggle_call_video()
    elif media_type == "audio":
        return api.toggle_call_audio()
    else:
        return jsonify({"error": "Invalid media type. Must be 'video' or 'audio'."}), 400

@app.route("/api/avatar/conference/info", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_conference_info(api):
    return api.get_call_info()

@app.route("/api/avatar/media/image", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_display_image(api):
    data = request.get_json(silent=True) or {}
    image_path = data.get("path", "")
    if not image_path:
        return jsonify({"error": "Image path is required to display image."}), 400
    return api.display_image(image_path)

@app.route("/api/avatar/media/video", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_display_video(api):
    data = request.get_json(silent=True) or {}
    video_path = data.get("path", "")
    loop = data.get("loop", False)
    if not video_path:
        return jsonify({"error": "Video path is required to display video."}), 400
    return api.display_video(video_path, loop)

@app.route("/api/avatar/media/stop", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_stop_media(api):
    return api.stop_media()

@app.route("/api/avatar/media/info", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_media_info(api):
    return api.get_media_info()

@app.route("/api/avatar/desktop/mirror", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_desktop_mirror(api):
    data = request.get_json(silent=True) or {}
    action = data.get("action", "start")
    if action == "start":
        return api.start_desktop_mirror()
    elif action == "stop":
        return api.stop_desktop_mirror()
    else:
        return jsonify({"error": "Invalid action for desktop mirror. Must be 'start' or 'stop'."}), 400

@app.route("/api/avatar/panel/size", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_set_panel_size(api):
    data = request.get_json(silent=True) or {}
    width = data.get("width", 480)
    height = data.get("height", 360)
    try: # Validate as integers
        width = int(width)
        height = int(height)
    except ValueError:
        return jsonify({"error": "Width and height must be integers."}), 400
    return api.set_panel_size(width, height)

@app.route("/api/avatar/panel/maximize", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_toggle_maximize(api):
    return api.toggle_maximize()

@app.route("/api/avatar/panel/popout", methods=['POST'])
@_avatar_api_response_wrapper
def avatar_toggle_popout(api):
    return api.toggle_popout()

# ---------------- Additional v8.0 API endpoints (merged from app-new.py) ----------------

def get_config_snapshot():
    """Return a small config snapshot that the WebUI can query."""
    try:
        import SarahMemoryGlobals as G
        meta = {}
        meta.setdefault("project_version", getattr(G, "PROJECT_VERSION", PROJECT_VERSION))
        meta.setdefault("author", getattr(G, "AUTHOR", "Brian Lee Baros"))
        meta.setdefault("revision_start_date", getattr(G, "REVISION_START_DATE", ""))
        meta.setdefault("run_mode", getattr(G, "RUN_MODE", "local"))
        meta.setdefault("device_mode", getattr(G, "DEVICE_MODE", "local_agent"))
        meta.setdefault("device_profile", getattr(G, "DEVICE_PROFILE", "Standard"))
        meta.setdefault("safe_mode", getattr(G, "SAFE_MODE", False))
        meta.setdefault("local_only", getattr(G, "LOCAL_ONLY_MODE", False)) # Changed from LOCAL_ONLY for consistency
        meta.setdefault("node_name", getattr(G, "NODE_NAME", "SarahMemory"))
        meta.setdefault("api_root", getattr(G, "API_ROOT", "/api"))
        return meta
    except Exception as e:
        app_logger.warning(f"Error getting config snapshot from SarahMemoryGlobals, falling back: {e}")
        # Minimal fallback identity snapshot if globals are unavailable.
        return {
            "project_version": PROJECT_VERSION,
            "author": "Brian Lee Baros",
            "revision_start_date": "",
            "run_mode": "local",
            "device_mode": "local_agent",
            "device_profile": "Standard",
            "safe_mode": False,
            "local_only": False,
            "node_name": "SarahMemory",
            "api_root": "/api",
        }

@app.route("/api/settings")
def api_settings():
    meta = get_config_snapshot()
    return jsonify({
        "ok": True,
        "settings": meta,
        # WebUI bootstrap hint: the frontend can choose to speak this via its own
        # browser TTS engine. Server-side TTS cannot play in a remote browser.
        "intro": {
            "text": "Hi! I'm Sarah — ready when you are. Try asking me anything.",
            "should_speak": True,
        },
        "ts": time.time(), # Added timestamp for consistency
    })


@app.route("/api/ui/bootstrap", methods=["GET"])
def api_ui_bootstrap():
    """One-call bootstrap for the React/Vite WebUI.

    The WebUI can call this once on page load.
    - Returns identity/config + capability flags.
    - Returns an intro message that the browser can speak.
    - Uses a session cookie to avoid repeating the intro on every refresh.
    """
    meta = get_config_snapshot()

    # Session-based one-time intro flag.
    already = bool(session.get("intro_spoken"))
    if not already:
        session["intro_spoken"] = True

    # Capability detection for the WebUI.
    # NOTE: Do not reference core_speak_text here because the TTS helper block
    # is initialized further down in this file.
    tts_ok = False
    try:
        from SarahMemoryVoice import speak_text as _s
        tts_ok = callable(_s)
    except Exception:
        tts_ok = False
    avatar_ok = True
    try:
        import SarahMemoryAvatar as _A
        avatar_ok = True
    except Exception:
        avatar_ok = False

    return jsonify({
        "ok": True,
        "settings": meta,
        "capabilities": {
            "tts_server": bool(tts_ok),
            "avatar": bool(avatar_ok),
            "media_jobs": True,
        },
        "intro": {
            "text": "Hi! I'm Sarah — ready when you are. Try asking me anything.",
            "should_speak": (not already),
        },
        "ts": time.time(),
    }), 200

# --------------------------- TTS / VOICE HELPERS --------------------------

core_speak_text = None
try:
    from SarahMemoryVoice import speak_text as core_speak_text_func
    core_speak_text = core_speak_text_func
except ImportError:
    app_logger.info("SarahMemoryVoice module not found for TTS.")
except Exception as e:
    app_logger.error(f"Error importing SarahMemoryVoice.speak_text: {e}", exc_info=True)


@app.route("/api/tts/speak", methods=['POST'])
def api_tts_speak():
    """
    Minimal TTS bridge for the Web UI.
    Expected JSON:
      { "text": "...", "voice": "default", "rate": 1.0 }
    """
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    voice = (data.get("voice") or "default").strip()
    rate_str = data.get("rate") # Keep as string/int for initial parsing

    if not text:
        return jsonify({"ok": False, "error": "Missing text for TTS."}), 400

    try:
        rate = float(rate_str) if rate_str is not None else 1.0
        if not (0.1 <= rate <= 5.0): # Example range, adjust as needed
             return jsonify({"ok": False, "error": "Speech rate must be between 0.1 and 5.0."}), 400
    except ValueError:
        return jsonify({"ok": False, "error": "Invalid speech rate format."}), 400


    if core_speak_text is None:
        return jsonify({
            "ok": False,
            "error": "TTS engine not available on this server.",
        }), 501

    try:
        # Assuming core_speak_text can handle these parameters
        core_speak_text(text, voice_name=voice, rate=rate)
        return jsonify({"ok": True}), 200
    except Exception as e:
        app_logger.exception(f"Error during TTS speak request for text: '{text}...'")
        return jsonify({"ok": False, "error": f"Failed to speak text: {e}"}), 500

@app.route("/api/logs/events")
def api_logs_events():
    """
    Return the last N lines of api_events.log so the Web UI can show them.
    This reads the log file created by Flask's basic logging, not `log_event()`.
    """
    N = int(request.args.get("limit", 200)) # Limit to last N lines
    path = os.path.join(LOGS_DIR, "api_events.log") # Expecting a JSON log file

    if not os.path.exists(path):
        return jsonify({"ok": True, "events": [], "message":f"Log file {os.path.basename(path)} not found."}), 200

    events = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            # Read all lines and then slice for performance for large files, or use deque.
            # For simplicity, reading all and slicing.
            lines = f.readlines()
            # If the file is very large, consider reading from end-of-file for performance
            # Or use a more sophisticated log reader.

            # This is a bit slow for very large files, but robust for typical usage
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    # If a line isn't valid JSON, still append it as raw to show problem
                    events.append({"raw": line, "error": "Invalid JSON format in log line"})
                except Exception as e:
                    app_logger.warning(f"Error parsing log line: {e} | Line: {line}")
                    events.append({"raw": line, "error": f"Parsing error: {str(e)}"})
        return jsonify({"ok": True, "events": events}), 200
    except IOError as e:
        app_logger.error(f"Error reading API events log file {path}: {e}")
        return jsonify({"ok": False, "error": f"Failed to read API events log: {e}"}), 500
    except Exception as e:
        app_logger.exception(f"Unexpected error when fetching API events log.")
        return jsonify({"ok": False, "error": str(e)}), 500


# --------------------------- SIMPLE PING ----------------------------------

@app.route("/api/ping")
def api_ping():
    ok, notes, main_running = _perform_health_checks() # Include health check in ping
    return jsonify({
        "ok": True,
        "pong": True,
        "ts": time.time(),
        "version": PROJECT_VERSION,
        "health_status": "ok" if ok else "warning",
        "running": True,
        "main_running": main_running,
    })
@app.route("/api/ledger/top-nodes")
def api_top_nodes():
    limit_str = request.args.get("limit", "10")
    try:
        limit = int(limit_str)
        if not (1 <= limit <= 100): # Reasonable limit
            raise ValueError("Limit must be between 1 and 100.")
    except ValueError as e:
        return jsonify({"ok": False, "error": f"Invalid limit parameter: {e}"}), 400

    leaders = read_top_nodes(limit=limit)
    return jsonify({"ok": True, "leaders": leaders}), 200

# --------------------------- SIMPLE SETTINGS SNAPSHOT ---------------------

@app.route("/api/download/<path:filename>")
def api_download(filename):
    """Download a file that lives under DATA_DIR (safe path enforced)."""
    if not filename:
        return jsonify({"ok": False, "error": "Missing filename"}), 400

    # Normalize and enforce containment within DATA_DIR
    try:
        base = os.path.abspath(DATA_DIR)
        full_path = os.path.abspath(os.path.join(base, filename))
        common_path = os.path.commonpath([base, full_path])
    except Exception:
        return jsonify({"ok": False, "error": "Invalid path"}), 400

    if common_path != base:
        app_logger.warning("Attempted download outside DATA_DIR: %s", full_path)
        return jsonify({"ok": False, "error": "Invalid path"}), 400

    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        return jsonify({"ok": False, "error": "File not found"}), 404

    try:
        # Use send_file so nested paths are fine after the containment check.
        return send_file(full_path, as_attachment=True, download_name=os.path.basename(full_path))
    except TypeError:
        # Flask <2.0 compatibility: download_name not supported
        return send_file(full_path, as_attachment=True)


# -----------------------------------------------------------------------------
# Optional dependency shim: bleach
# -----------------------------------------------------------------------------
# appsys.py relies on `bleach.clean()` for HTML sanitization. On some minimal
# installs, `bleach` may not be present. To keep APPSYS online without forcing
# extra installs, we provide a conservative fallback implementation.
try:
    import bleach  # type: ignore
except Exception:  # pragma: no cover
    try:
        import types as _types
        import re as _re
        import html as _html
        _bleach_mod = _types.ModuleType("bleach")

        def _fallback_clean(text, tags=None, attributes=None, strip=False, strip_comments=True, **kwargs):
            try:
                s = "" if text is None else str(text)
            except Exception:
                s = ""
            # Remove HTML comments (basic)
            if strip_comments:
                s = _re.sub(r"<!--.*?-->", "", s, flags=_re.DOTALL)
            if strip:
                # Drop all tags
                s = _re.sub(r"<[^>]+>", "", s)
                return s
            # Escape everything (safest default)
            return _html.escape(s, quote=True)

        _bleach_mod.clean = _fallback_clean  # type: ignore
        import sys as _sys
        _sys.modules["bleach"] = _bleach_mod
    except Exception:
        # If even the shim fails, appsys import will raise and be logged.
        pass

# --- v8 local system endpoints (Files / OS utilities) ---
def _ensure_api_import_paths():
    """Make api/server modules importable in all launch modes."""
    try:
        server_dir = os.path.abspath(os.path.dirname(__file__))      # .../api/server
        api_dir = os.path.abspath(os.path.join(server_dir, ".."))   # .../api
        proj_dir = os.path.abspath(os.path.join(api_dir, ".."))     # project root
        for p in (server_dir, api_dir, proj_dir):
            if p and p not in sys.path:
                sys.path.insert(0, p)
    except Exception:
        pass

try:
    _ensure_api_import_paths()
    try:
        # When imported as a package (e.g., `from api.server.app import app`)
        from . import appsys as _appsys  # type: ignore
    except Exception:
        # When executed with api/server on sys.path (e.g., `python api/server/app.py`)
        import appsys as _appsys  # type: ignore

    _appsys.init_app(app)
except Exception as _e:
    try:
        app_logger.error(f"appsys init failed: {_e}", exc_info=True)
    except Exception:
        pass

# --- v8 MCP broker endpoints (SarahNet one-way broker) ---
try:
    _ensure_api_import_paths()
    try:
        from . import appnet as _appnet  # type: ignore
    except Exception:
        import appnet as _appnet  # type: ignore

    _appnet.init_app(app, _connect_sqlite, META_DB, _api_key_auth_ok, _sign_ok)
except Exception as _e:
    try:
        app_logger.error(f"appnet init failed: {_e}", exc_info=True)
    except Exception:
        pass



# ============================================================================
# UI Event Speech Support (Opt-in)
# ============================================================================

@app.post("/api/ui/event")
def api_ui_event():
    """
    Programmatic UI event trigger for speech/notifications.
    Body: {"event": "panel_open", "detail": "Files", "speak": "Opening File Manager"}
    """
    try:
        data = request.get_json(silent=True) or {}
        event = (data.get("event") or "unknown").strip() or "unknown"
        detail = (data.get("detail") or "").strip()
        speak = (data.get("speak") or "").strip()

        app_logger.info(f"UI event: {event} | {detail}")

        if speak and os.getenv("SARAH_UI_SPEECH_LOCAL", "0") == "1":
            try:
                from SarahMemoryVoice import speak_text  # type: ignore
                speak_text(speak, blocking=False)
            except Exception:
                pass

        return jsonify({"ok": True, "event": event}), 200
    except Exception as e:
        app_logger.error(f"UI event failed: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 500


# --- Terminal API (DEVELOPERSMODE gated by SarahMemoryTerminal) ---
from flask import request, jsonify
import SarahMemoryTerminal as smterm

@app.post("/api/terminal/execute")
def api_terminal_execute():
    payload = request.get_json(silent=True) or {}
    result = smterm.terminal_api_execute(payload, caller="Flask:/api/terminal/execute")
    return jsonify(result), (200 if result.get("ok") else 403 if result.get("blocked") else 400)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5055))
    app_logger.info(f"Starting SarahMemory Flask API server on http://0.0.0.0:{port}")
    # Initializing app.config with default values for toggles
    app.config.setdefault("CAMERA_ON", False)
    app.config.setdefault("MIC_ON", False)
    app.config.setdefault("VOICE_OUTPUT_ON", True)
    app.config.setdefault("TELECOM_ENABLED", False)  # For telecom stateub

    # In development, use debug=True for reloader and debugger
    # In production, use a WSGI server like Gunicorn/uWSGI
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    if debug_mode:
        app_logger.warning("Running in DEBUG mode. Do NOT use in production!")

    app.run(host="0.0.0.0", port=port, debug=debug_mode)

# ====================================================================
# END OF app.py v8.0.0
# ====================================================================