#--==The SarahMemory Project==--
#File: /app/server/app.py
# ULTIMATE merged Flask server for SarahMemory (v8.0.0)
#Part of the SarahMemory Companion AI-bot Platform
#Author: Â© 2025 Brian Lee Baros. All Rights Reserved.
#www.linkedin.com/in/brian-baros-29962a176
#https://www.facebook.com/bbaros
#brian.baros@sarahmemory.com
#'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
#https://www.sarahmemory.com
#https://api.sarahmemory.com
#https://ai.sarahmemory.com

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
from decimal import Decimal
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for, send_file
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()
import jwt
import bcrypt
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import wraps
from datetime import datetime, timedelta
# ---------------------------------------------------------------------------
# Path resolution (prefer SarahMemoryGlobals; fallback to local server layout)
# ---------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import SarahMemoryGlobals as config  # rich path config if available
    BASE_DIR   = getattr(config, "BASE_DIR", os.getcwd())
    PUBLIC_DIR = getattr(config, "PUBLIC_DIR", os.path.join(BASE_DIR, "public_html"))
    WEB_DIR    = getattr(config, "WEB_DIR", os.path.join(PUBLIC_DIR, "web"))
    DATA_DIR   = getattr(config, "DATA_DIR", os.path.join(BASE_DIR, "data"))
    PROJECT_VERSION = getattr(config, "PROJECT_VERSION", "7.7.5")
except Exception:
    BASE_DIR   = os.path.dirname(os.path.abspath(__file__))      # /api/server
    PUBLIC_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))   # /api
    WEB_DIR    = PUBLIC_DIR                                      # serve index.html from /api
    DATA_DIR   = os.path.join(BASE_DIR, "data")                  # /api/server/data
    PROJECT_VERSION = "7.7.5"

STATIC_DIR   = os.path.join(BASE_DIR, "static")                  # /api/server/static
WALLETS_DIR  = os.path.join(DATA_DIR, "wallets")
META_DB      = os.path.join(DATA_DIR, "meta.db")                 # merged meta DB
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(WALLETS_DIR, exist_ok=True)



# Optional core modules
try:
    import SarahMemoryLedger as ledger_mod
except Exception:
    ledger_mod = None
try:
    import SarahMemoryNetwork as net_mod
except Exception:
    net_mod = None

# Flask app (templates under WEB_DIR so /api/index.html is found)
app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    static_url_path="/api/static",
    template_folder=WEB_DIR
)

# Apply CORS *after* app is created
# Allow all your public frontends to call this API (GoogieHost + PythonAnywhere + local dev)
try:
    from flask_cors import CORS as _CORS_ENSURE
    _CORS_ENSURE(
        app,
        resources={
            r"/api/*": {
                "origins": [
                    "https://api.sarahmemory.com",
                    "https://www.sarahmemory.com",
                    "https://ai.sarahmemory.com",
                    "https://softdev0.pythonanywhere.com",
                    "http://127.0.0.1:5000",
                    "http://127.0.0.1:8080",
                    "http://localhost:5000",
                    "http://localhost:8080",
                ]
            }
        },
        supports_credentials=True,  # ðŸ”¥ THIS is critical for credentials:'include'
    )
except Exception:
    # If flask-cors is not installed, API still works on same-origin calls.
    pass

try:
    from SarahMemoryDatabase import init_database
    init_database()  # ensures ai_learning.db + qa_cache exist
except Exception as e:
    print("[WARN] DB init failed in app.py:", e)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _connect_sqlite(path: str):
    con = sqlite3.connect(path, timeout=5.0)
    con.row_factory = sqlite3.Row
    return con

def _safe_getattr(mod, name, default=None):
    try:
        return getattr(mod, name)
    except Exception:
        return default

def _ensure_dir(p: str):
    try: os.makedirs(p, exist_ok=True)
    except Exception: pass

def _globals_paths():
    # Resolve additional dirs used by endpoints; prefer SarahMemoryGlobals if present
    try:
        import SarahMemoryGlobals as G
        DATASETS_DIR = getattr(G, "DATASETS_DIR", os.path.join(BASE_DIR, "data", "memory", "datasets"))
        SETTINGS_DIR = getattr(G, "SETTINGS_DIR", os.path.join(BASE_DIR, "data", "settings"))
        THEMES_DIR   = getattr(G, "THEMES_DIR", os.path.join(BASE_DIR, "data", "mods", "themes"))
        DOCUMENTS_DIR= getattr(G, "DOCUMENTS_DIR", os.path.join(BASE_DIR, "documents"))
        USER_DB_PATH = getattr(G, "USER_DB_PATH", os.path.join(DATASETS_DIR, "user_profile.db"))
    except Exception:
        DATASETS_DIR = os.path.join(BASE_DIR, "data", "memory", "datasets")
        SETTINGS_DIR = os.path.join(BASE_DIR, "data", "settings")
        THEMES_DIR   = os.path.join(BASE_DIR, "data", "mods", "themes")
        DOCUMENTS_DIR= os.path.join(BASE_DIR, "documents")
        USER_DB_PATH = os.path.join(DATASETS_DIR, "user_profile.db")
    for d in (DATASETS_DIR, SETTINGS_DIR, THEMES_DIR, DOCUMENTS_DIR):
        _ensure_dir(d)
    return DATASETS_DIR, SETTINGS_DIR, THEMES_DIR, DOCUMENTS_DIR, USER_DB_PATH

# ---------------------------------------------------------------------------
# Wallet / Ledger
# ---------------------------------------------------------------------------
def _wallet_path_simple(node: str) -> str:
    safe = "".join(ch for ch in node if ch.isalnum() or ch in ("-","_")) or "anon"
    return os.path.join(WALLETS_DIR, f"wallet-{safe}.srh")

def ensure_wallet_simple(node: str):
    p = _wallet_path_simple(node)
    con = _connect_sqlite(p); cur = con.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS wallet (
        id INTEGER PRIMARY KEY,
        balance TEXT,
        reputation REAL DEFAULT 0.0,
        last_rep_ts REAL DEFAULT (strftime('%s','now')),
        rep_daily REAL DEFAULT 0.0
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS txs (
        id INTEGER PRIMARY KEY, ts REAL, delta TEXT, memo TEXT
    )""")
    cur.execute("SELECT COUNT(1) FROM wallet WHERE id=1")
    if not (cur.fetchone() or [0])[0]:
        cur.execute("INSERT INTO wallet(id,balance) VALUES(1, ?)", ("0",))
    con.commit(); con.close()
    return p

def get_balance_simple(path: str) -> Decimal:
    con = _connect_sqlite(path); cur = con.cursor()
    cur.execute("SELECT balance FROM wallet WHERE id=1")
    row = con.fetchone(); con.close()
    return Decimal(row["balance"] if row and row["balance"] is not None else "0")

def read_top_nodes(limit=10):
    # Prefer richer ledger if present
    try:
        if ledger_mod and hasattr(ledger_mod, "top_nodes"):
            res = ledger_mod.top_nodes()
            try:
                return res.get_json().get("leaders", [])[:limit]
            except Exception:
                pass
    except Exception:
        pass
    rows = []
    for fn in glob.glob(os.path.join(WALLETS_DIR, "wallet-*.srh")):
        try:
            node = os.path.basename(fn).replace("wallet-","").replace(".srh","")
            con = _connect_sqlite(fn); cur = con.cursor()
            cur.execute("SELECT balance, reputation FROM wallet WHERE id=1")
            r = cur.fetchone(); con.close()
            bal = Decimal(str(r["balance"])) if r and r["balance"] is not None else Decimal("0")
            rep = float(r["reputation"]) if r and r["reputation"] is not None else 0.0
        except Exception:
            bal = Decimal("0"); rep = 0.0
        rows.append({"node": node, "balance": str(bal), "reputation": rep})
    rows.sort(key=lambda r: (Decimal(r["balance"]), r["reputation"]), reverse=True)
    return rows[:limit]

# ---------------------------------------------------------------------------
# DB: meta tables (merged)
# ---------------------------------------------------------------------------
def ensure_meta_db():
    con = _connect_sqlite(META_DB); cur = con.cursor()
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
    con.commit(); con.close()
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
        meta_fn = getattr(G, "get_runtime_meta", None)
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
    except Exception:
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


@app.route("/api/session/bootstrap", methods=["GET", "POST"])
def api_session_bootstrap():
    """
    Phase A3 â€” Session Bootstrap API.

    Single canonical handshake endpoint used by Web UI (app.js) at startup.
    Aligns client and server runtime identity and exposes core feature flags.
    """
    try:
        payload = request.get_json(force=True, silent=True) or {}
    except Exception:
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
    camera_enabled = bool(globals().get("CAMERA_ENABLED", False))
    mic_enabled = bool(globals().get("MIC_ENABLED", False))
    voice_enabled = bool(globals().get("VOICE_OUTPUT_ENABLED", False))

    features = {
        "camera": camera_enabled,
        "microphone": mic_enabled,
        "voice_output": voice_enabled,
        "hub_enabled": bool(net_mod is not None),
        "wallet_enabled": True,
        "ledger_module": bool(ledger_mod is not None),
        "file_transfer": True,
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
    """
    API root:

    - If a hub/leaderboard index.html exists, serve that.
      We check both WEB_DIR and STATIC_DIR so this works on all layouts:
        - Local:   BASE_DIR/api/server/static/index.html
        - PA:      /home/Softdev0/SarahMemory/api/server/static/index.html
        - Web:     https://www.sarahmemory.com/api/server/static/index.html
    - Otherwise, return a simple JSON health banner.
    """
    candidates = []

    # Template-based (if you ever put Jinja templates into WEB_DIR)
    idx_web = os.path.join(WEB_DIR, "index.html")
    if os.path.exists(idx_web):
        try:
            leaders = read_top_nodes(limit=10)
            return render_template("index.html", leaders=leaders)
        except Exception:
            candidates.append(idx_web)

    # Pure static fallback (the hub index you mentioned)
    idx_static = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(idx_static):
        try:
            return send_file(idx_static)
        except Exception:
            candidates.append(idx_static)

    # Last resort: JSON banner
    return jsonify(
        {
            "ok": True,
            "service": "SarahMemory API",
            "version": PROJECT_VERSION,
            "note": "No index.html found in WEB_DIR or STATIC_DIR",
            "checked": candidates,
        }
    )

@app.route("/")
def root_redirect():
    return redirect("/api/")

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

    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ASSET_EXTS:
        return jsonify({"error": "unsupported asset type", "file": filename}), 404

    # Try in /api/server/static first (STATIC_DIR), then in the project root (BASE_DIR).
    for root in (STATIC_DIR, BASE_DIR):
        candidate = os.path.join(root, filename)
        if os.path.exists(candidate):
            return send_from_directory(root, filename)

    return jsonify({"error": "asset not found", "file": filename}), 404

@app.route("/api/leaderboard")
def api_leaderboard():
    return jsonify({"leaders": read_top_nodes(limit=10)})

@app.route("/api/health")
def api_health():
    """
    Universal health endpoint.

    - running      â†’ HTTP API is responding (True if this function is hit)
    - main_running â†’ optional desktop launcher (SarahMemoryMain) process check
                     used on Windows/Linux desktop installs only.
    """
    ok, notes = True, []

    if not os.path.isdir(WALLETS_DIR):
        ok = False
        notes.append("wallets folder missing")

    try:
        con = _connect_sqlite(META_DB)
        con.execute("SELECT 1")
        con.close()
    except Exception:
        ok = False
        notes.append("meta DB inaccessible")

    status = "ok" if ok else "down"

    # On PythonAnywhere / pure-API mode, there may be no SarahMemoryMain process.
    main_running = False
    try:
        checker = globals().get("_is_running", None)
        if callable(checker):
            main_running = bool(checker())
    except Exception:
        main_running = False

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

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Primary chat endpoint used by the Web UI (app.js).

    Expects JSON like:
      { "text": "user message here", "files": [...] }

    Returns JSON like:
      {
        "ok": true,
        "reply": "<assistant text>",
        "meta": { "source": "api", "engine": "route_intent_response" }
      }
    """
    import logging

    try:
        payload = request.get_json(force=True, silent=True) or {}
        text = (payload.get("text") or "").strip()

        if not text:
            return jsonify({
                "ok": False,
                "error": "Missing 'text' in request.",
                "meta": {"source": "api", "reason": "no_text"},
            }), 400

        # Optional hints from the Web UI (wonâ€™t break older callers)
        intent = (payload.get("intent") or "question").strip()
        tone = (payload.get("tone") or "friendly").strip()
        complexity = (payload.get("complexity") or "adult").strip()

        reply_str = ""

        # ------------------------------------------------------------------
        # Try the lightweight router (commands, mouse moves, URL opens)
        # ------------------------------------------------------------------
        router_result = None
        try:
            import SarahMemoryAiFunctions as F
            router = getattr(F, "route_intent_response", None)
            if callable(router):
                router_result = router(text)
        except Exception:
            router = None
            router_result = None

        # If the router clearly handled it (non-empty string or dict), use it.
        # BUT treat the generic fallback "I'm unsure how to respond." as NOT handled,
        # so the full AI pipeline can still answer normal questions.
        if isinstance(router_result, str) and router_result.strip():
            if router_result.strip() != "I'm unsure how to respond.":
                reply_str = router_result.strip()
            else:
                router_result = None
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
                # generate_reply is defined as generate_reply(self, user_text)
                # For API usage we pass self=None
                bundle = generate_reply(None, text)
            except Exception as e:
                # Last-chance fallback: direct API call if available
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
                except Exception as api_e:
                    # If we get here, everything failed â†’ real 500
                    msg = (
                        f"SarahMemoryReply.generate_reply failed: {e}; "
                        f"API fallback failed: {api_e}"
                    )
                    try:
                        logging.getLogger("SarahMemoryAPI").error(msg)
                    except Exception:
                        pass
                    return jsonify({
                        "ok": False,
                        "error": msg,
                        "meta": {"source": "api", "reason": "reply_and_api_failed"},
                    }), 500
            else:
                # Normal path: unify bundle â†’ string
                if isinstance(bundle, dict):
                    reply_str = str(
                        bundle.get("response")
                        or bundle.get("text")
                        or ""
                    ).strip()
                else:
                    reply_str = str(bundle or "").strip()

        # Final safety: never return None
        reply_str = reply_str or ""

        return jsonify({
            "ok": True,
            "reply": reply_str,
            "meta": {
                "source": "api",
                "engine": "route_intent_response",
                "version": PROJECT_VERSION,
            },
        }), 200

    except Exception as e:
        # Hard catch-all so the Web UI sees a clean JSON error instead of a stack trace
        try:
            app.logger.exception("api_chat failed: %s", e)
        except Exception:
            pass

        return jsonify({
            "ok": False,
            "error": str(e),
            "meta": {"source": "api", "reason": "exception"},
        }), 500

@app.route("/api/request-knowledge", methods=["POST"])
def api_request_knowledge():
    data = request.get_json(force=True, silent=True) or {}
    requester = (data.get("requester") or data.get("from") or "").strip()
    topic = (data.get("topic") or data.get("notes") or "").strip() or "unspecified"
    amount = data.get("amount") or data.get("reward") or 0
    try: amount = str(Decimal(str(amount)))
    except Exception: amount = "0"
    if not requester:
        return jsonify({"error":"requester required"}), 400
    con = _connect_sqlite(META_DB); cur = con.cursor()
    cur.execute("INSERT INTO knowledge_requests(ts, requester, topic, reward, status) VALUES (?,?,?,?,?)",
                (time.time(), requester, topic, amount, "open"))
    rid = cur.lastrowid; con.commit(); con.close()
    ensure_wallet_simple(requester)
    return jsonify({"request_id": rid, "status":"open"}), 201

@app.route("/api/wallet/<node>")
def api_wallet_view(node):
    try:
        p = ensure_wallet_simple(node)
        con = _connect_sqlite(p); cur = con.cursor()
        cur.execute("SELECT balance, reputation, last_rep_ts, rep_daily FROM wallet WHERE id=1")
        r = cur.fetchone()
        cur.execute("SELECT ts,delta,memo FROM txs ORDER BY id DESC LIMIT 50")
        txs = [dict(ts=row[0], delta=row[1], memo=row[2]) for row in cur.fetchall()]
        con.close()
        return jsonify({"node": node, "balance": r[0], "reputation": r[1], "last_rep_ts": r[2], "rep_daily": r[3], "txs": txs})
    except Exception as e:
        return jsonify({"error": str(e)}), 404



# ---------------------------------------------------------------------------
# Hub (HMAC) endpoints
# ---------------------------------------------------------------------------
HUB_SECRET = os.environ.get("SARAH_HUB_SECRET", "")
def _sign_ok(body: bytes, sig: str) -> bool:
    if not HUB_SECRET:
        return True
    try:
        mac = hmac.new(HUB_SECRET.encode("utf-8"), body or b"", hashlib.sha256).hexdigest()
        return hmac.compare_digest(mac, (sig or ""))
    except Exception:
        return False

@app.post("/api/hub/ping")
def hub_ping():
    body = request.get_data(); sig = request.headers.get("X-Sarah-Signature","")
    if not _sign_ok(body, sig): return jsonify({"ok":False,"err":"auth"}), 401
    payload = request.get_json(force=True, silent=True) or {}
    return jsonify({"ok": True, "now": time.time(), "echo": payload})

@app.post("/api/hub/job")
def hub_job():
    body = request.get_data(); sig = request.headers.get("X-Sarah-Signature","")
    if not _sign_ok(body, sig): return jsonify({"ok":False,"err":"auth"}), 401
    payload = request.get_json(force=True, silent=True) or {}
    jid = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    # optional light persistence for debugging
    try:
        jobs_dir = os.path.join(DATA_DIR, "jobs"); _ensure_dir(jobs_dir)
        with open(os.path.join(jobs_dir, f"job-{int(time.time())}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception: pass
    return jsonify({"ok": True, "job_id": jid})

@app.post("/api/hub/reply")
def hub_reply():
    body = request.get_data(); sig = request.headers.get("X-Sarah-Signature","")
    if not _sign_ok(body, sig): return jsonify({"ok":False,"err":"auth"}), 401
    payload = request.get_json(force=True, silent=True) or {}
    try:
        receipts = os.path.join(DATA_DIR, "receipts"); _ensure_dir(receipts)
        with open(os.path.join(receipts, f"r-{int(time.time())}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception: pass
    return jsonify({"ok": True})

# ---------------------------------------------------------------------------
# API Key guard + Node/Embedding/Context/Jobs endpoints
# ---------------------------------------------------------------------------
def _auth_ok():
    expected = os.environ.get("SARAH_API_KEY")
    if not expected: return True
    auth = request.headers.get("Authorization","")
    if auth.startswith("Bearer "):
        token = auth.split(" ",1)[1].strip()
        return token == expected
    return False

@app.route("/api/register-node", methods=["POST"])
def api_register_node():
    if not _auth_ok(): return jsonify({"error":"unauthorized"}), 401
    data = request.get_json(silent=True, force=True) or {}
    node_id = (data.get("node_id") or "").strip() or "unknown"
    meta = json.dumps(data.get("meta") or {})
    con = _connect_sqlite(META_DB); cur = con.cursor()
    cur.execute("INSERT INTO nodes(node_id,last_ts,meta) VALUES(?,?,?) "
                "ON CONFLICT(node_id) DO UPDATE SET last_ts=excluded.last_ts, meta=excluded.meta",
                (node_id, time.time(), meta))
    con.commit(); con.close()
    ensure_wallet_simple(node_id)
    return jsonify({"ok": True})

@app.route("/api/receive-embedding", methods=["POST"])
def api_receive_embedding():
    if not _auth_ok(): return jsonify({"error":"unauthorized"}), 401
    data = request.get_json(silent=True, force=True) or {}
    node_id = (data.get("node_id") or "").strip() or "unknown"
    vector = json.dumps(data.get("embedding"))
    ctx = data.get("context_id")
    con = _connect_sqlite(META_DB); cur = con.cursor()
    cur.execute("INSERT INTO embeddings(ts,node_id,context_id,vector) VALUES(?,?,?,?)",
                (time.time(), node_id, ctx, vector))
    con.commit(); con.close()
    return jsonify({"ok": True})

@app.route("/api/context-update", methods=["POST"])
def api_context_update():
    if not _auth_ok(): return jsonify({"error":"unauthorized"}), 401
    data = request.get_json(silent=True, force=True) or {}
    node_id = (data.get("node_id") or "").strip() or "unknown"
    text = data.get("text") or ""
    tags = json.dumps(data.get("tags") or [])
    con = _connect_sqlite(META_DB); cur = con.cursor()
    cur.execute("INSERT INTO contexts(ts,node_id,text,tags) VALUES(?,?,?,?)",
                (time.time(), node_id, text, tags))
    con.commit(); con.close()
    return jsonify({"ok": True})

@app.route("/api/jobs", methods=["POST"])
def api_jobs_post():
    if not _auth_ok(): return jsonify({"error":"unauthorized"}), 401
    data = request.get_json(silent=True, force=True) or {}
    node_id = (data.get("node_id") or "").strip() or "unknown"
    job_id = (data.get("job_id") or "").strip() or "unknown"
    result = json.dumps(data.get("result"))
    con = _connect_sqlite(META_DB); cur = con.cursor()
    cur.execute("INSERT INTO job_results(ts,node_id,job_id,result) VALUES(?,?,?,?)",
                (time.time(), node_id, job_id, result))
    con.commit(); con.close()
    return jsonify({"ok": True})

# ---------------------------------------------------------------------------
# WebUI helper endpoints (Themes/Voices/Settings/Contacts/Reminders/Cleanup)
# ---------------------------------------------------------------------------
@app.after_request
def add_security_headers(resp):
    try:
        resp.headers["X-Project-Version"] = PROJECT_VERSION
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self' https://api.sarahmemory.com; "
            "img-src 'self' data: https://api.sarahmemory.com; "
            "style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'"
        )
    except Exception: pass
    return resp

# Settings
@app.route("/get_user_setting")
def get_user_setting():
    _, SETTINGS_DIR, _, _, _ = _globals_paths()
    key = request.args.get("key", "")
    path = os.path.join(SETTINGS_DIR, "settings.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify({"value": data.get(key, "")})
        except Exception:
            pass
    return jsonify({"value": ""})

@app.route("/set_user_setting", methods=["POST"])
def set_user_setting():
    _, SETTINGS_DIR, _, _, _ = _globals_paths()
    payload = request.get_json(silent=True) or {}
    key = payload.get("key"); val = payload.get("value")
    path = os.path.join(SETTINGS_DIR, "settings.json")
    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    data[key] = val
    _ensure_dir(SETTINGS_DIR)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return jsonify({"status":"ok"})

# Themes
# Voices (local OS voices via pyttsx3 if present)
@app.route("/get_available_voices")
def get_available_voices():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        return jsonify([{"id": v.id, "name": getattr(v, "name", v.id)} for v in voices])
    except Exception:
        return jsonify([])

# Cleanup tools
@app.route("/cleanup/backup_all")
def cleanup_backup_all():
    try:
        import SarahMemoryCleanup as C
        fn = _safe_getattr(C, "backup_all")
        if callable(fn): return jsonify({"status":"ok","result": str(fn())})
    except Exception as e:
        return jsonify({"status":"error","error": str(e)}), 500
    return jsonify({"status":"noop"})

@app.route("/cleanup/restore_latest")
def cleanup_restore_latest():
    try:
        import SarahMemoryCleanup as C
        fn = _safe_getattr(C, "restore_latest")
        if callable(fn): return jsonify({"status":"ok","result": str(fn())})
    except Exception as e:
        return jsonify({"status":"error","error": str(e)}), 500
    return jsonify({"status":"noop"})

@app.route("/cleanup/clear_range", methods=["POST"])
def cleanup_clear_range():
    payload = request.get_json(silent=True) or {}
    db_name = payload.get("db", "context_history.db")
    seconds = int(payload.get("seconds", 0) or 0)
    try:
        import SarahMemoryCleanup as C
        fn = _safe_getattr(C, "clear_range")
        if callable(fn): return jsonify({"status":"ok","result": str(fn(db_name, seconds if seconds>0 else None))})
    except Exception as e:
        return jsonify({"status":"error","error": str(e)}), 500
    return jsonify({"status":"noop"})

@app.route("/cleanup/tidy_logs")
def cleanup_tidy_logs():
    try:
        import SarahMemoryCleanup as C
        fn = _safe_getattr(C, "tidy_logs")
        if callable(fn): return jsonify({"status":"ok","result": str(fn())})
    except Exception as e:
        return jsonify({"status":"error","error": str(e)}), 500
    return jsonify({"status":"noop"})

# Camera/Mic/Voice toggles
@app.route("/toggle_camera")
def toggle_camera():
    state = request.args.get("state","").lower() in ("true","1","yes","on")
    globals()["CAMERA_ENABLED"] = state
    return jsonify({"status":"ok","camera": state})

@app.route("/toggle_microphone")
def toggle_microphone():
    state = request.args.get("state","").lower() in ("true","1","yes","on")
    globals()["MIC_ENABLED"] = state
    return jsonify({"status":"ok","microphone": state})

@app.route("/toggle_voice_output")
def toggle_voice_output():
    state = request.args.get("state","").lower() in ("true","1","yes","on")
    globals()["VOICE_OUTPUT_ENABLED"] = state
    return jsonify({"status":"ok","voice_output": state})

# Telecom (light stubs)
@app.route("/check_call_active")
def check_call_active():
    return jsonify({"active": bool(globals().get("CALL_ACTIVE", False))})

@app.route("/initiate_call", methods=["POST"])
def initiate_call():
    data = request.get_json(silent=True) or {}
    number = (data.get("number") or "").strip()
    globals()["CALL_ACTIVE"] = True if number else False
    return jsonify({"status":"call_started","to":number})

# File transfer / ingest
@app.route("/send_file_to_remote", methods=["POST"])
def send_file_to_remote():
    payload = request.get_json(silent=True) or {}
    fname = payload.get("filename") or "file.bin"
    b64   = payload.get("data") or ""
    try: data = base64.b64decode(b64.encode("utf-8"))
    except Exception: data = b""
    if os.name == "nt":
        # Windows: keep original Downloads semantics
        out_dir = r"C:\Users\Public\Downloads"
    else:
        # Cross-platform / server-safe: drop into data/downloads
        out_dir = os.path.join(DATA_DIR, "downloads")
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "wb") as f: f.write(data)
    return jsonify({"message": f"Sent file to remote user: {fname}", "path": out_path})

@app.route("/ingest_local_file", methods=["POST"])
def ingest_local_file():
    payload = request.get_json(silent=True) or {}
    fname = payload.get("filename") or "file.bin"
    b64   = payload.get("data") or ""
    DATASETS_DIR, _, _, DOCUMENTS_DIR, _ = _globals_paths()
    try: data = base64.b64decode(b64.encode("utf-8"))
    except Exception: data = b""
    out_dir = DOCUMENTS_DIR or DATASETS_DIR
    _ensure_dir(out_dir)
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "wb") as f: f.write(data)
    return jsonify({"message": f"Stored file in local documents: {fname}", "path": out_path})

# Contacts
@app.route("/get_all_contacts")
def get_all_contacts():
    _, _, _, _, USER_DB_PATH = _globals_paths()
    con = sqlite3.connect(USER_DB_PATH); cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS contacts (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, number TEXT)")
    con.commit()
    cur.execute("SELECT id, name, number FROM contacts ORDER BY name COLLATE NOCASE")
    rows = [{"id":r[0], "name":r[1], "number":r[2]} for r in cur.fetchall()]
    con.close()
    return jsonify({"contacts": rows})

@app.route("/add_contact", methods=["POST"])
def add_contact():
    _, _, _, _, USER_DB_PATH = _globals_paths()
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    number = (data.get("number") or "").strip()
    if not name or not number:
        return jsonify({"status":"error","error":"Missing name/number"}), 400
    con = sqlite3.connect(USER_DB_PATH); cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS contacts (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, number TEXT)")
    con.commit()
    cur.execute("INSERT INTO contacts(name,number) VALUES(?,?)",(name,number))
    con.commit(); con.close()
    return jsonify({"status":"ok"})

@app.route("/delete_contact", methods=["POST"])
def delete_contact():
    _, _, _, _, USER_DB_PATH = _globals_paths()
    rid = int((request.get_json(silent=True) or {}).get("id", 0) or 0)
    con = sqlite3.connect(USER_DB_PATH); cur = con.cursor()
    cur.execute("DELETE FROM contacts WHERE id=?", (rid,))
    con.commit(); con.close()
    return jsonify({"status":"deleted"})

# Reminders
@app.route("/get_reminders")
def get_reminders():
    DATASETS_DIR, *_ = _globals_paths()
    db = os.path.join(DATASETS_DIR, "reminders.db")
    con = sqlite3.connect(db); cur = con.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS reminders (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, time TEXT, note TEXT)')
    con.commit()
    cur.execute('SELECT id, title, time, note FROM reminders ORDER BY time ASC')
    rows = [{'id':r[0], 'title':r[1], 'time':r[2], 'note':r[3] or ''} for r in cur.fetchall()]
    con.close()
    return jsonify({'reminders': rows})

@app.route("/save_reminder", methods=["POST"])
def save_reminder():
    DATASETS_DIR, *_ = _globals_paths()
    db = os.path.join(DATASETS_DIR, "reminders.db")
    payload = request.get_json(silent=True) or {}
    title = (payload.get("title") or "").strip()
    time_s = (payload.get("time") or "").strip()
    note = payload.get("note") or ""
    if not title or not time_s:
        return jsonify({"status":"error","error":"Missing title/time"}), 400
    con = sqlite3.connect(db); cur = con.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS reminders (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT, time TEXT, note TEXT)')
    con.commit()
    cur.execute('INSERT INTO reminders(title, time, note) VALUES(?,?,?)',(title, time_s, note))
    con.commit(); rid = cur.lastrowid; con.close()
    return jsonify({"status":"ok","id":rid})

@app.route("/delete_reminder", methods=["POST"])
def delete_reminder():
    DATASETS_DIR, *_ = _globals_paths()
    db = os.path.join(DATASETS_DIR, "reminders.db")
    rid = (request.get_json(silent=True) or {}).get("id")
    con = sqlite3.connect(db); cur = con.cursor()
    cur.execute('DELETE FROM reminders WHERE id=?', (rid,))
    con.commit(); con.close()
    return jsonify({"status":"deleted"})

@app.route("/run_automation_trigger", methods=["POST"])
def run_automation_trigger():
    try:
        import SarahMemoryAiFunctions as F
        payload = request.get_json(silent=True) or {}
        fn = _safe_getattr(F, "run_automation")
        if callable(fn):
            res = fn(payload)
            return jsonify({"status":"ok","result":str(res)})
    except Exception:
        pass
    return jsonify({"status":"ok","message":"Trigger acknowledged"})

# Calendar + Chat history (for Web UI)
@app.route("/get_chat_threads_by_date")
def get_chat_threads_by_date():
    DATASETS_DIR, *_ = _globals_paths()
    db = os.path.join(DATASETS_DIR, "context_history.db")
    date_filter = request.args.get("date", "")  # YYYY-MM-DD
    con = sqlite3.connect(db); con.row_factory = sqlite3.Row
    cur = con.cursor()
    q = "SELECT id, timestamp, user_input AS preview FROM conversations"
    if date_filter:
        q += " WHERE date(timestamp)=?"
        cur.execute(q, (date_filter,))
    else:
        cur.execute(q)
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return jsonify({"threads": rows})

@app.route("/get_conversation_by_id")
def get_conversation_by_id():
    DATASETS_DIR, *_ = _globals_paths()
    db = os.path.join(DATASETS_DIR, "context_history.db")
    convo_id = request.args.get("id")
    con = sqlite3.connect(db); con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT role, text, metadata AS meta FROM conversations WHERE id = ?", (convo_id,))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return jsonify(rows)

# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------
@app.get("/get_theme_files")
def get_theme_files():
    try:
        import SarahMemoryGlobals as G
        data_dir  = getattr(G, "DATA_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data"))
        themes_dir= getattr(G, "THEMES_DIR", os.path.join(data_dir, "mods", "themes"))
    except Exception:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data_dir = os.path.join(base_dir, "data")
        themes_dirA = os.path.join(data_dir, "mods", "themes")
        themes_dirB = os.path.join(data_dir, "themes")
        themes_dir = themes_dirA if os.path.isdir(themes_dirA) else themes_dirB

    files = []
    if os.path.isdir(themes_dir):
        for dp, dn, fnames in os.walk(themes_dir):
            for f in fnames:
                if f.lower().endswith((".css", ".json", ".yml", ".yaml", ".toml", ".png", ".jpg", ".jpeg", ".svg", ".ttf", ".otf")):
                    rel = os.path.relpath(os.path.join(dp, f), themes_dir).replace("\\", "/")
                    files.append(rel)

    rootA = "/api/data/themes"
    rootB = "/api/data/mods/themes"
    active_root = rootA if os.path.isdir(os.path.join(data_dir, "themes")) else rootB
    return jsonify({"root": active_root, "count": len(files), "files": sorted(files)})

@app.route("/api/data/themes/<path:filename>")
def serve_theme_file_A(filename):
    try:
        import SarahMemoryGlobals as G
        data_dir = getattr(G, "DATA_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data"))
    except Exception:
        data_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data")
    root = os.path.join(data_dir, "themes")
    return send_from_directory(root, filename)

@app.route("/api/data/mods/themes/<path:filename>")
def serve_theme_file_B(filename):
    try:
        import SarahMemoryGlobals as G
        data_dir = getattr(G, "DATA_DIR", os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data"))
    except Exception:
        data_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "data")
    root = os.path.join(data_dir, "mods", "themes")
    return send_from_directory(root, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5055))
    app.run(host="0.0.0.0", port=port)

# --- Boot Launcher / Health (idempotent server-side autostart) ---
import subprocess

PID_FILE = os.path.join(DATA_DIR, "sarahmemory.pid")

def _is_running():
    try:
        if not os.path.exists(PID_FILE):
            return False
        with open(PID_FILE, "r") as f:
            pid_s = (f.read() or "").strip()
        if not pid_s:
            return False
        pid = int(pid_s)
        # Best-effort: os.kill(pid, 0) works on POSIX; ignore failures elsewhere
        try:
            os.kill(pid, 0)
            return True
        except Exception:
            return False
    except Exception:
        return False

def _write_pid(pid: int):
    try:
        with open(PID_FILE, "w") as f:
            f.write(str(pid))
    except Exception:
        pass

def _start_sarah_main():
    """Spawn the canonical boot chain (SarahMemoryMain.py) in background."""
    try:
        candidates = [
            [os.path.join("venv","Scripts","python.exe"), "SarahMemoryMain.py"], # Windows venv
            [os.path.join("venv","bin","python"), "SarahMemoryMain.py"],         # Linux/mac venv
            ["python", "SarahMemoryMain.py"],
            ["python3", "SarahMemoryMain.py"],
        ]
        for cmd in candidates:
            try:
                # Work from BASE_DIR so imports/paths match desktop & server layout
                proc = subprocess.Popen(cmd, cwd=BASE_DIR)
                _write_pid(proc.pid)
                return True
            except Exception:
                continue
        return False
    except Exception:
        return False

@app.route("/api/launch", methods=["POST"])
def api_launch():
    try:
        if _is_running():
            return jsonify({"ok": True, "running": True, "msg": "already running"})
        ok = _start_sarah_main()
        return jsonify({"ok": bool(ok), "running": bool(ok)}), (200 if ok else 500)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

# [removed duplicate /api/health]


# ============================================================================
# Phase B: Authentication System
# ============================================================================

# JWT Configuration
JWT_SECRET = os.getenv('JWT_SECRET_KEY', 'change-this-secret-key-in-production')
JWT_ALGORITHM = 'HS256'
JWT_EXP_DELTA_DAYS = 7

def generate_jwt_token(user_id, email):
    """Generate JWT token for user."""
    payload = {
        'user_id': user_id,
        'email': email,
        'exp': datetime.utcnow() + timedelta(days=JWT_EXP_DELTA_DAYS),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_jwt_token(token):
    """Verify JWT token and return payload."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def require_auth(f):
    """Decorator to require authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'Authentication required'}), 401

        payload = verify_jwt_token(token)
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401

        request.user_id = payload['user_id']
        request.user_email = payload['email']
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
        display_name = data.get('display_name', '')

        # Validate input
        if not email or '@' not in email:
            return jsonify({'error': 'Invalid email'}), 400
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        if not pin or len(pin) != 4 or not pin.isdigit():
            return jsonify({'error': 'PIN must be 4 digits'}), 400

        # Import database functions
        try:
            from SarahMemoryDatabase import sm_get_or_create_user, _get_cloud_conn
        except Exception as e:
            return jsonify({'error': 'Database module unavailable'}), 503

        # Check if user already exists
        existing = sm_get_or_create_user(email)
        if existing and existing.get('id'):
            return jsonify({'error': 'Email already registered'}), 409

        # Hash password and PIN
        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        pin_hash = bcrypt.hashpw(pin.encode(), bcrypt.gensalt()).decode()

        # Create user in database
        conn = _get_cloud_conn()
        if not conn:
            return jsonify({'error': 'Database unavailable'}), 503

        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sm_users (email, display_name, password_hash, security_pin_hash) VALUES (%s, %s, %s, %s)",
                (email, display_name or email.split('@')[0], password_hash, pin_hash)
            )
            conn.commit()
            user_id = cursor.lastrowid

            # Generate verification code
            verification_code = ''.join([str(secrets.randbelow(10)) for _ in range(6)])
            cursor.execute(
                "INSERT INTO sm_email_verifications (user_id, email, verification_code, expires_at, ip_address, user_agent) VALUES (%s, %s, %s, DATE_ADD(NOW(), INTERVAL 15 MINUTE), %s, %s)",
                (user_id, email, verification_code, request.remote_addr, request.headers.get('User-Agent', ''))
            )
            conn.commit()

            # Send verification email
            send_verification_email(email, verification_code)

            return jsonify({
                'success': True,
                'user_id': user_id,
                'message': 'Registration successful. Please check your email for verification code.'
            }), 201

        finally:
            conn.close()

    except Exception as e:
        print(f"[Phase B] Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500


@app.route('/api/auth/login', methods=['POST'])
def auth_login():
    """Phase B: Login user with email, password, and PIN."""
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        pin = data.get('pin', '')

        if not email or not password or not pin:
            return jsonify({'error': 'Email, password, and PIN required'}), 400

        # Import database function
        try:
            from SarahMemoryDatabase import _get_cloud_conn
        except:
            return jsonify({'error': 'Database module unavailable'}), 503

        conn = _get_cloud_conn()
        if not conn:
            return jsonify({'error': 'Database unavailable'}), 503

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, email, display_name, password_hash, security_pin_hash, is_active, is_verified FROM sm_users WHERE email = %s AND deleted_at IS NULL",
                (email,)
            )
            user = cursor.fetchone()

            if not user:
                return jsonify({'error': 'Invalid credentials'}), 401

            if not user['is_active']:
                return jsonify({'error': 'Account disabled'}), 403

            # Verify password
            if not bcrypt.checkpw(password.encode(), user['password_hash'].encode()):
                return jsonify({'error': 'Invalid credentials'}), 401

            # Verify PIN
            if not bcrypt.checkpw(pin.encode(), user['security_pin_hash'].encode()):
                return jsonify({'error': 'Invalid PIN'}), 401

            # Check if email verified
            if not user['is_verified']:
                return jsonify({'error': 'Email not verified'}), 403

            # Generate JWT token
            token = generate_jwt_token(user['id'], user['email'])

            # Update last_login
            cursor.execute("UPDATE sm_users SET last_login = NOW() WHERE id = %s", (user['id'],))
            conn.commit()

            return jsonify({
                'success': True,
                'token': token,
                'user': {
                    'id': user['id'],
                    'email': user['email'],
                    'display_name': user['display_name']
                }
            }), 200

        finally:
            conn.close()

    except Exception as e:
        print(f"[Phase B] Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500


@app.route('/api/auth/verify-email', methods=['POST'])
def auth_verify_email():
    """Phase B: Verify email with code."""
    try:
        data = request.json
        email = data.get('email', '').strip().lower()
        code = data.get('code', '').strip()

        if not email or not code:
            return jsonify({'error': 'Email and code required'}), 400

        try:
            from SarahMemoryDatabase import _get_cloud_conn
        except:
            return jsonify({'error': 'Database module unavailable'}), 503

        conn = _get_cloud_conn()
        if not conn:
            return jsonify({'error': 'Database unavailable'}), 503

        try:
            cursor = conn.cursor(dictionary=True)

            # Get user
            cursor.execute("SELECT id FROM sm_users WHERE email = %s", (email,))
            user = cursor.fetchone()
            if not user:
                return jsonify({'error': 'User not found'}), 404

            # Check verification code
            cursor.execute(
                "SELECT id FROM sm_email_verifications WHERE user_id = %s AND verification_code = %s AND expires_at > NOW() AND verified_at IS NULL",
                (user['id'], code)
            )
            verification = cursor.fetchone()

            if not verification:
                return jsonify({'error': 'Invalid or expired code'}), 400

            # Mark as verified
            cursor.execute("UPDATE sm_email_verifications SET verified_at = NOW() WHERE id = %s", (verification['id'],))
            cursor.execute("UPDATE sm_users SET is_verified = TRUE, email_verified_at = NOW() WHERE id = %s", (user['id'],))
            conn.commit()

            return jsonify({'success': True, 'message': 'Email verified successfully'}), 200

        finally:
            conn.close()

    except Exception as e:
        print(f"[Phase B] Verification error: {e}")
        return jsonify({'error': 'Verification failed'}), 500


@app.route('/api/user/preferences', methods=['GET', 'PUT'])
@require_auth
def user_preferences():
    """Phase B: Get or update user preferences."""
    try:
        from SarahMemoryDatabase import sm_get_user_preferences, sm_update_user_preferences

        if request.method == 'GET':
            prefs = sm_get_user_preferences(request.user_id)
            return jsonify(prefs), 200

        elif request.method == 'PUT':
            data = request.json
            success = sm_update_user_preferences(request.user_id, data)
            if success:
                return jsonify({'success': True}), 200
            else:
                return jsonify({'error': 'Update failed'}), 500

    except Exception as e:
        print(f"[Phase B] Preferences error: {e}")
        return jsonify({'error': 'Operation failed'}), 500


def send_verification_email(email, code):
    """Phase B: Send verification email with code."""
    try:
        smtp_host = os.getenv('SMTP_HOST', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', 587))
        smtp_user = os.getenv('SMTP_USER', '')
        smtp_password = os.getenv('SMTP_PASSWORD', '')
        smtp_from = os.getenv('SMTP_FROM_EMAIL', 'noreply@sarahmemory.com')

        if not smtp_user or not smtp_password:
            print("[Phase B] SMTP not configured, skipping email")
            return

        msg = MIMEMultipart('alternative')
        msg['Subject'] = 'SarahMemory Email Verification'
        msg['From'] = smtp_from
        msg['To'] = email

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

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_from, email, msg.as_string())

        print(f"[Phase B] Verification email sent to {email}")

    except Exception as e:
        print(f"[Phase B] Error sending email: {e}")