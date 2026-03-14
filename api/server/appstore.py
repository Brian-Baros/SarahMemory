"""
# --==The SarahMemory Project==--
# File: /app/server/appstore.py
# Purpose: SarahMemory Power Store Endpoints
# Part of the SarahMemory Companion AI-bot Platform
# Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
# www.linkedin.com/in/brian-baros-29962a176
# https://www.facebook.com/bbaros
# brian.baros@sarahmemory.com
# 'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
# https://www.sarahmemory.com
# https://api.sarahmemory.com
# https://ai.sarahmemory.com
# https://store.sarahmemory.com
#==============================================================================================
# Design goals:
# - NO endpoint collisions with ANY of the other app*.py files
# - Everything is namespaced under /api/store/
# - Reference implementation for store + storefront integrations (PayPal/Printify/Kittl)
#
# Key rules:
# - No secrets in frontend. All secrets live in PythonAnywhere .env and are read server-side.
# - "Kitchen sink" module: store/auth/products/payments/integrations, without endpoint collisions.
"""
from __future__ import annotations
# --- SARAHMETA START ---
# GRADE = "B"
# ROLE = "api_bridge"
# CATEGORY = "storefront_operations"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "api_server"
# API_DOMAIN = "store"
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "store_api"
# FAMILY = "commerce"
# GOVERNANCE_LEVEL = "restricted"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "Storefront and Power Store API bridge under /api/store/* for auth, products, payments/integrations, generated product concepts, and server-side secret handling."
# --- SARAHMETA END ---
import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from flask import Blueprint, jsonify, request

bp2 = Blueprint("appstore_v800", __name__)

# Injected by app.py at init_app time
_CONNECT_SQLITE: Optional[Callable[..., Any]] = None
_META_DB: Optional[str] = None
_API_KEY_AUTH_OK: Optional[Callable[[], bool]] = None
_SIGN_OK: Optional[Callable[[bytes, str], bool]] = None


# ----------------------------- helpers ---------------------------------

def _now() -> float:
    return float(time.time())


def _jok(data: Any = None, **meta: Any):
    payload = {"ok": True}
    if data is not None:
        payload["data"] = data
    if meta:
        payload["meta"] = meta
    return jsonify(payload)


def _jerr(msg: str, code: str = "error", http: int = 400, **meta: Any):
    payload = {"ok": False, "error": msg, "error_code": code}
    if meta:
        payload["meta"] = meta
    return jsonify(payload), http


def _get_env(name: str, default: str = "") -> str:
    try:
        v = os.getenv(name, default)
        return v if v is not None else default
    except Exception:
        return default


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _as_str(x: Any, default: str = "") -> str:
    try:
        if x is None:
            return default
        return str(x)
    except Exception:
        return default


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    s = s.strip()
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))


def _sha256_bytes(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _hash_pw(pw: str, salt: str) -> str:
    # PBKDF2-HMAC-SHA256, modest iterations to keep PA CPU ok
    iterations = _as_int(_get_env("STORE_PBKDF2_ITERS", "120000"), 120000)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt.encode("utf-8"), iterations, dklen=32)
    return _b64url(dk)


def _token_secret() -> str:
    # Stable secret from env; fallback to STORE_ADMIN_PASSWORD hashed if missing
    sec = _get_env("STORE_TOKEN_SECRET", "").strip()
    if sec:
        return sec
    # hard fallback (still server-side)
    seed = (_get_env("STORE_ADMIN_EMAIL", "admin@sarahmemory.com") + ":" + _get_env("STORE_ADMIN_PASSWORD", ""))
    return _b64url(_sha256_bytes(seed.encode("utf-8")))


def _sign_token(payload: Dict[str, Any]) -> str:
    header = {"alg": "HS256", "typ": "SMT"}  # SarahMemory Token
    sec = _token_secret().encode("utf-8")
    h = _b64url(json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    p = _b64url(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    msg = f"{h}.{p}".encode("utf-8")
    sig = _b64url(hmac.new(sec, msg, hashlib.sha256).digest())
    return f"{h}.{p}.{sig}"


def _verify_token(token: str) -> Tuple[bool, Dict[str, Any]]:
    try:
        parts = token.strip().split(".")
        if len(parts) != 3:
            return False, {}
        h, p, sig = parts
        sec = _token_secret().encode("utf-8")
        msg = f"{h}.{p}".encode("utf-8")
        exp_sig = _b64url(hmac.new(sec, msg, hashlib.sha256).digest())
        if not hmac.compare_digest(exp_sig, sig):
            return False, {}
        payload = json.loads(_b64url_decode(p).decode("utf-8"))
        if not isinstance(payload, dict):
            return False, {}
        exp = _as_int(payload.get("exp", 0), 0)
        if exp and _now() > exp:
            return False, {}
        return True, payload
    except Exception:
        return False, {}


def _auth_header_token() -> str:
    # Supports: Authorization: Bearer <token>  OR  X-Store-Token: <token>
    try:
        auth = _as_str(request.headers.get("Authorization") or "")
        if auth.lower().startswith("bearer "):
            return auth.split(" ", 1)[1].strip()
    except Exception:
        pass
    return _as_str(request.headers.get("X-Store-Token") or "")


def _is_admin() -> bool:
    # Admin gate: API-key auth OR store token (role=admin)
    try:
        if _API_KEY_AUTH_OK and _API_KEY_AUTH_OK():
            return True
    except Exception:
        pass
    tok = _auth_header_token()
    if not tok:
        return False
    ok, payload = _verify_token(tok)
    if not ok:
        return False
    return _as_str(payload.get("role") or "") == "admin"


def _is_user() -> bool:
    # User gate: API-key auth OR store token (role=user/admin)
    try:
        if _API_KEY_AUTH_OK and _API_KEY_AUTH_OK():
            return True
    except Exception:
        pass
    tok = _auth_header_token()
    if not tok:
        return False
    ok, payload = _verify_token(tok)
    if not ok:
        return False
    role = _as_str(payload.get("role") or "")
    return role in ("user", "admin")


def _db():
    if not _CONNECT_SQLITE:
        raise RuntimeError("CONNECT_SQLITE not injected")
    if not _META_DB:
        raise RuntimeError("META_DB not injected")
    return _CONNECT_SQLITE(_META_DB)


def _ensure_tables():
    # Idempotent schema
    with _db() as con:
        cur = con.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS store_kv (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL,
            meta TEXT NOT NULL,
            exp REAL NOT NULL
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS store_users (
            user_id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            pw_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_ts REAL NOT NULL,
            reset_token TEXT NOT NULL,
            reset_exp REAL NOT NULL
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS store_products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            price REAL NOT NULL,
            category TEXT NOT NULL,
            image_url TEXT NOT NULL,
            tags TEXT NOT NULL,
            source TEXT NOT NULL,
            created_ts REAL NOT NULL,
            updated_ts REAL NOT NULL
        )""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS store_generation_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            req TEXT NOT NULL,
            resp TEXT NOT NULL,
            created_ts REAL NOT NULL
        )""")
        con.commit()


def _kv_prune(cur):
    now = _now()
    cur.execute("DELETE FROM store_kv WHERE exp > 0 AND exp <= ?", (now,))


def _require_json() -> Dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return {}
    return payload


def _cors_ok(resp):
    # If app.py already does CORS globally, this is redundant but safe.
    try:
        origin = request.headers.get("Origin") or "*"
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Store-Token"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    except Exception:
        pass
    return resp


# ----------------------------- lifecycle ---------------------------------

def init_app(app, connect_sqlite, meta_db_path, api_key_auth_ok=None, sign_ok=None):
    """
    Called by app.py to inject shared helpers and mount the blueprint.
    """
    global _CONNECT_SQLITE, _META_DB, _API_KEY_AUTH_OK, _SIGN_OK
    _CONNECT_SQLITE = connect_sqlite
    _META_DB = meta_db_path
    _API_KEY_AUTH_OK = api_key_auth_ok
    _SIGN_OK = sign_ok

    try:
        _ensure_tables()
    except Exception:
        # Don't hard-fail mount; endpoints will raise explicit errors on use.
        pass

    app.register_blueprint(bp2)
    return True


# ----------------------------- endpoints ---------------------------------

@bp2.route("/api/store/health", methods=["GET", "OPTIONS"])
def api_store_health():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    try:
        _ensure_tables()
        return _cors_ok(_jok({"status": "ok", "ts": _now()}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "store_health_fail", 500))


# ---- Simple KV Store (namespaced) ----

@bp2.route("/api/store/set", methods=["POST", "OPTIONS"])
def api_store_set():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    payload = _require_json()
    k = _as_str(payload.get("key") or payload.get("k") or "").strip()
    v = payload.get("value") if "value" in payload else payload.get("v")
    meta = payload.get("meta") or {}
    ttl = _as_int(payload.get("ttl") or 0, 0)
    if not k:
        return _cors_ok(_jerr("Missing key", "missing_key", 400))
    if v is None:
        v = ""
    try:
        _ensure_tables()
        exp = (_now() + ttl) if ttl > 0 else 0.0
        with _db() as con:
            cur = con.cursor()
            _kv_prune(cur)
            cur.execute(
                "INSERT INTO store_kv(k,v,meta,exp) VALUES(?,?,?,?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v, meta=excluded.meta, exp=excluded.exp",
                (k, json.dumps(v), json.dumps(meta), float(exp)),
            )
            con.commit()
        return _cors_ok(_jok({"key": k, "stored": True, "exp": exp}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "store_set_fail", 500))


@bp2.route("/api/store/get", methods=["GET", "OPTIONS"])
def api_store_get():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    k = _as_str(request.args.get("key") or request.args.get("k") or "").strip()
    if not k:
        return _cors_ok(_jerr("Missing key", "missing_key", 400))
    try:
        _ensure_tables()
        with _db() as con:
            cur = con.cursor()
            _kv_prune(cur)
            cur.execute("SELECT v, meta, exp FROM store_kv WHERE k=?", (k,))
            row = cur.fetchone()
            con.commit()
        if not row:
            return _cors_ok(_jok({"hit": False, "key": k}))
        v_json, meta_json, exp = row
        v = json.loads(v_json) if v_json else None
        meta = json.loads(meta_json) if meta_json else {}
        return _cors_ok(_jok({"hit": True, "key": k, "value": v, "meta": meta, "exp": exp}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "store_get_fail", 500))


@bp2.route("/api/store/del", methods=["POST", "OPTIONS"])
def api_store_del():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    payload = _require_json()
    k = _as_str(payload.get("key") or payload.get("k") or "").strip()
    if not k:
        return _cors_ok(_jerr("Missing key", "missing_key", 400))
    try:
        _ensure_tables()
        with _db() as con:
            cur = con.cursor()
            cur.execute("DELETE FROM store_kv WHERE k=?", (k,))
            n = cur.rowcount
            con.commit()
        return _cors_ok(_jok({"key": k, "deleted": int(n or 0)}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "store_del_fail", 500))


# ---- Store Auth (server-side, no FE secrets) ----

@bp2.route("/api/store/auth/register", methods=["POST", "OPTIONS"])
def api_store_auth_register():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    payload = _require_json()
    email = _as_str(payload.get("email") or "").strip().lower()
    password = _as_str(payload.get("password") or "")
    if not email or not password:
        return _cors_ok(_jerr("Missing email/password", "missing_fields", 400))
    try:
        _ensure_tables()
        user_id = "u_" + uuid.uuid4().hex
        salt = _b64url(os.urandom(16))
        pw_hash = _hash_pw(password, salt)
        with _db() as con:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO store_users(user_id,email,pw_hash,salt,created_ts,reset_token,reset_exp) "
                "VALUES(?,?,?,?,?,?,?)",
                (user_id, email, pw_hash, salt, _now(), "", 0.0),
            )
            con.commit()
        # Issue user token
        token = _sign_token({"sub": user_id, "email": email, "role": "user", "iat": int(_now()), "exp": int(_now()) + 86400})
        return _cors_ok(_jok({"user_id": user_id, "token": token, "role": "user"}))
    except Exception as e:
        msg = str(e)
        if "UNIQUE constraint failed" in msg or "unique constraint" in msg.lower():
            return _cors_ok(_jerr("Email already registered", "email_exists", 409))
        return _cors_ok(_jerr(msg, "register_fail", 500))


@bp2.route("/api/store/auth/login", methods=["POST", "OPTIONS"])
def api_store_auth_login():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    payload = _require_json()
    email = _as_str(payload.get("email") or "").strip().lower()
    password = _as_str(payload.get("password") or "")
    if not email or not password:
        return _cors_ok(_jerr("Missing email/password", "missing_fields", 400))
    try:
        _ensure_tables()
        with _db() as con:
            cur = con.cursor()
            cur.execute("SELECT user_id, pw_hash, salt FROM store_users WHERE email=?", (email,))
            row = cur.fetchone()
        if not row:
            return _cors_ok(_jerr("Invalid credentials", "bad_creds", 401))
        user_id, pw_hash, salt = row
        calc = _hash_pw(password, salt)
        if not hmac.compare_digest(calc, pw_hash):
            return _cors_ok(_jerr("Invalid credentials", "bad_creds", 401))
        token = _sign_token({"sub": user_id, "email": email, "role": "user", "iat": int(_now()), "exp": int(_now()) + 86400})
        return _cors_ok(_jok({"user_id": user_id, "token": token, "role": "user"}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "login_fail", 500))


@bp2.route("/api/store/auth/password-reset", methods=["POST", "OPTIONS"])
def api_store_auth_password_reset():
    """
    Simple reset workflow:
    - If payload has {email} only: issue reset_token (returned for now; can be emailed later).
    - If payload has {email, reset_token, new_password}: apply reset.
    """
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    payload = _require_json()
    email = _as_str(payload.get("email") or "").strip().lower()
    if not email:
        return _cors_ok(_jerr("Missing email", "missing_email", 400))
    try:
        _ensure_tables()
        reset_token = _as_str(payload.get("reset_token") or "")
        new_password = _as_str(payload.get("new_password") or "")
        if not reset_token and not new_password:
            # Issue
            tok = "rt_" + uuid.uuid4().hex
            exp = _now() + 3600.0
            with _db() as con:
                cur = con.cursor()
                cur.execute("UPDATE store_users SET reset_token=?, reset_exp=? WHERE email=?", (tok, float(exp), email))
                if cur.rowcount <= 0:
                    return _cors_ok(_jerr("Unknown email", "unknown_email", 404))
                con.commit()
            return _cors_ok(_jok({"email": email, "reset_token": tok, "reset_exp": exp}))
        # Apply
        if not reset_token or not new_password:
            return _cors_ok(_jerr("Missing reset_token/new_password", "missing_fields", 400))
        with _db() as con:
            cur = con.cursor()
            cur.execute("SELECT user_id, reset_token, reset_exp FROM store_users WHERE email=?", (email,))
            row = cur.fetchone()
            if not row:
                return _cors_ok(_jerr("Unknown email", "unknown_email", 404))
            user_id, rt, rex = row
            if not rt or not hmac.compare_digest(_as_str(rt), reset_token):
                return _cors_ok(_jerr("Invalid reset token", "bad_reset_token", 401))
            if rex and float(rex) < _now():
                return _cors_ok(_jerr("Reset token expired", "reset_expired", 401))
            salt = _b64url(os.urandom(16))
            pw_hash = _hash_pw(new_password, salt)
            cur.execute(
                "UPDATE store_users SET pw_hash=?, salt=?, reset_token=?, reset_exp=? WHERE user_id=?",
                (pw_hash, salt, "", 0.0, user_id),
            )
            con.commit()
        return _cors_ok(_jok({"email": email, "reset": True}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "reset_fail", 500))


@bp2.route("/api/store/auth/admin/login", methods=["POST", "OPTIONS"])
def api_store_auth_admin_login():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    payload = _require_json()
    email = _as_str(payload.get("email") or "").strip().lower()
    password = _as_str(payload.get("password") or "")
    admin_email = _get_env("STORE_ADMIN_EMAIL", "").strip().lower()
    admin_pw = _get_env("STORE_ADMIN_PASSWORD", "")
    if not admin_email or not admin_pw:
        return _cors_ok(_jerr("Admin credentials not configured", "admin_not_configured", 503))
    if email != admin_email or password != admin_pw:
        return _cors_ok(_jerr("Invalid credentials", "bad_creds", 401))
    token = _sign_token({"sub": "admin", "email": admin_email, "role": "admin", "iat": int(_now()), "exp": int(_now()) + 86400})
    return _cors_ok(_jok({"token": token, "role": "admin"}))


# ---- Storefront: trends + products + generation log ----

def _product_row_to_obj(row):
    (pid, name, description, price, category, image_url, tags, source, created_ts, updated_ts) = row
    return {
        "id": int(pid),
        "name": name,
        "description": description,
        "price": float(price),
        "category": category,
        "image_url": image_url,
        "tags": json.loads(tags) if tags else [],
        "source": source,
        "created_ts": float(created_ts),
        "updated_ts": float(updated_ts),
    }


@bp2.route("/api/store/trends", methods=["POST", "OPTIONS"])
def api_store_trends():
    """
    Lightweight placeholder. Returns stable trend buckets.
    You can swap in SarahMemory AI ranking later without breaking the contract.
    """
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    payload = _require_json()
    niche = _as_str(payload.get("niche") or payload.get("category") or "general")
    region = _as_str(payload.get("region") or "US")
    data = {
        "niche": niche,
        "region": region,
        "generated_ts": _now(),
        "trends": [
            {"label": f"{niche} essentials", "score": 0.86},
            {"label": f"{niche} minimalist", "score": 0.79},
            {"label": f"{niche} premium", "score": 0.74},
            {"label": f"{niche} gifting", "score": 0.71},
        ],
    }
    return _cors_ok(_jok(data))


@bp2.route("/api/store/products/generate", methods=["POST", "OPTIONS"])
def api_store_products_generate():
    """
    Generate product concepts server-side (no vendor calls). Output is a list of product objects.
    If you later wire SarahMemoryAI, keep this envelope stable.
    """
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    payload = _require_json()
    niche = _as_str(payload.get("niche") or payload.get("category") or "general")
    count = max(1, min(24, _as_int(payload.get("count") or 8, 8)))

    # Deterministic-ish seed so refresh isn't random noise if user repeats same ask
    seed = hashlib.sha256((niche + "|" + json.dumps(payload, sort_keys=True)).encode("utf-8")).hexdigest()[:8]

    products = []
    for i in range(count):
        name = f"{niche.title()} Item {i+1}"
        desc = f"AI-curated concept for {niche} (batch {seed})."
        price = round(19.99 + (i * 2.5), 2)
        products.append({
            "id": 0,
            "name": name,
            "description": desc,
            "price": price,
            "category": niche,
            "image_url": payload.get("image_url") or "",
            "tags": [niche, "ai", "concept"],
            "source": "generated",
        })

    # Optional: log request/response
    try:
        _ensure_tables()
        with _db() as con:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO store_generation_log(req, resp, created_ts) VALUES(?,?,?)",
                (json.dumps(payload), json.dumps(products), _now()),
            )
            con.commit()
    except Exception:
        pass

    return _cors_ok(_jok({"products": products, "seed": seed}))


@bp2.route("/api/store/products", methods=["GET", "OPTIONS"])
def api_store_products_list():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    try:
        _ensure_tables()
        q = _as_str(request.args.get("q") or "").strip().lower()
        cat = _as_str(request.args.get("category") or "").strip().lower()
        limit = max(1, min(200, _as_int(request.args.get("limit") or 50, 50)))
        offset = max(0, _as_int(request.args.get("offset") or 0, 0))

        with _db() as con:
            cur = con.cursor()
            sql = "SELECT id,name,description,price,category,image_url,tags,source,created_ts,updated_ts FROM store_products"
            params = []
            where = []
            if q:
                where.append("(lower(name) LIKE ? OR lower(description) LIKE ?)")
                params.extend([f"%{q}%", f"%{q}%"])
            if cat:
                where.append("lower(category)=?")
                params.append(cat)
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY updated_ts DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            cur.execute(sql, tuple(params))
            rows = cur.fetchall() or []
        products = [_product_row_to_obj(r) for r in rows]
        return _cors_ok(_jok({"products": products, "limit": limit, "offset": offset}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "products_list_fail", 500))


@bp2.route("/api/store/products/bulk", methods=["POST", "OPTIONS"])
def api_store_products_bulk():
    """
    Admin-only bulk upsert:
      { products: [ {name, description, price, category, image_url, tags, source, id?}, ... ] }
    """
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_admin():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))
    payload = _require_json()
    items = payload.get("products") or payload.get("items") or []
    if not isinstance(items, list):
        return _cors_ok(_jerr("products must be a list", "bad_payload", 400))
    try:
        _ensure_tables()
        upserted = 0
        now = _now()
        with _db() as con:
            cur = con.cursor()
            for it in items:
                if not isinstance(it, dict):
                    continue
                pid = _as_int(it.get("id") or 0, 0)
                name = _as_str(it.get("name") or "").strip()
                if not name:
                    continue
                desc = _as_str(it.get("description") or "")
                price = _as_float(it.get("price") or 0.0, 0.0)
                category = _as_str(it.get("category") or "general")
                image_url = _as_str(it.get("image_url") or it.get("imageUrl") or it.get("image") or "")
                tags = it.get("tags") or []
                if not isinstance(tags, list):
                    tags = []
                source = _as_str(it.get("source") or "manual")

                if pid > 0:
                    cur.execute(
                        "UPDATE store_products SET name=?, description=?, price=?, category=?, image_url=?, tags=?, source=?, updated_ts=? WHERE id=?",
                        (name, desc, float(price), category, image_url, json.dumps(tags), source, now, pid),
                    )
                    if cur.rowcount > 0:
                        upserted += 1
                        continue
                cur.execute(
                    "INSERT INTO store_products(name,description,price,category,image_url,tags,source,created_ts,updated_ts) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (name, desc, float(price), category, image_url, json.dumps(tags), source, now, now),
                )
                upserted += 1
            con.commit()
        return _cors_ok(_jok({"upserted": upserted}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "products_bulk_fail", 500))


@bp2.route("/api/store/generation-log", methods=["POST", "OPTIONS"])
def api_store_generation_log():
    """
    Log any generation activity (frontend -> backend). Admin or user token accepted.
    """
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_user():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))
    payload = _require_json()
    try:
        _ensure_tables()
        with _db() as con:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO store_generation_log(req, resp, created_ts) VALUES(?,?,?)",
                (json.dumps(payload.get("req") or payload), json.dumps(payload.get("resp") or {}), _now()),
            )
            con.commit()
        return _cors_ok(_jok({"logged": True}))
    except Exception as e:
        return _cors_ok(_jerr(str(e), "genlog_fail", 500))


# ---- PayPal proxy endpoints (secrets in .env) ----

@bp2.route("/api/store/paypal/config", methods=["GET", "OPTIONS"])
def api_store_paypal_config():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    client_id = _get_env("PAYPAL_CLIENT_ID", "").strip()
    env = _get_env("PAYPAL_ENV", "sandbox").strip()  # sandbox|live
    if not client_id:
        return _cors_ok(_jerr("PayPal not configured", "paypal_not_configured", 503))
    return _cors_ok(_jok({"clientId": client_id, "environment": env}))


def _paypal_basic_auth() -> str:
    cid = _get_env("PAYPAL_CLIENT_ID", "").strip()
    sec = _get_env("PAYPAL_SECRET", "").strip()
    raw = f"{cid}:{sec}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("utf-8")


def _paypal_api_base() -> str:
    env = _get_env("PAYPAL_ENV", "sandbox").strip().lower()
    return "https://api-m.paypal.com" if env == "live" else "https://api-m.sandbox.paypal.com"


def _http_json(url: str, method: str = "GET", headers: Optional[Dict[str, str]] = None, body: Optional[Dict[str, Any]] = None) -> Tuple[int, Any]:
    # No extra deps; use urllib
    import urllib.request
    import urllib.error

    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
            try:
                return resp.status, json.loads(raw.decode("utf-8"))
            except Exception:
                return resp.status, raw.decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw.decode("utf-8"))
        except Exception:
            return e.code, raw.decode("utf-8", errors="ignore")
    except Exception as e:
        return 0, {"error": str(e)}


@bp2.route("/api/store/paypal/create-order", methods=["POST", "OPTIONS"])
def api_store_paypal_create_order():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_user():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))

    cid = _get_env("PAYPAL_CLIENT_ID", "").strip()
    sec = _get_env("PAYPAL_SECRET", "").strip()
    if not cid or not sec:
        return _cors_ok(_jerr("PayPal not configured", "paypal_not_configured", 503))

    payload = _require_json()
    # Accept multiple client payload shapes:
    # A) StoreUI: { currency, total, items:[{name,quantity,price}] }
    # B) Simple: { amount, currency }
    # C) Advanced: { order: <full PayPal order body> }
    currency = _as_str(payload.get("currency") or "USD").upper()
    order_body = payload.get("order") if isinstance(payload.get("order"), dict) else None

    if order_body is None:
        total = payload.get("total")
        amount = payload.get("amount")
        items = payload.get("items") if isinstance(payload.get("items"), list) else []

        value = None
        if total is not None:
            try:
                value = f"{float(total):.2f}"
            except Exception:
                value = _as_str(total)
        elif amount is not None:
            try:
                value = f"{float(amount):.2f}"
            except Exception:
                value = _as_str(amount)

        if not value:
            return _cors_ok(_jerr("Missing total/amount", "missing_amount", 400))

        purchase_unit = {"amount": {"currency_code": currency, "value": value}}

        # Optional itemization for better receipts
        norm_items = []
        for it in items:
            if not isinstance(it, dict):
                continue
            nm = _as_str(it.get("name") or "").strip()
            qty = _as_int(it.get("quantity") or 1, 1)
            pr = it.get("price")
            try:
                pr_val = f"{float(pr):.2f}"
            except Exception:
                pr_val = _as_str(pr or "0.00")
            if nm:
                norm_items.append({
                    "name": nm,
                    "quantity": str(max(1, qty)),
                    "unit_amount": {"currency_code": currency, "value": pr_val}
                })
        if norm_items:
            purchase_unit["items"] = norm_items

        order_body = {
            "intent": "CAPTURE",
            "purchase_units": [purchase_unit],
        }

    # Step 1: get access token
    base = _paypal_api_base()
    st, tok = _http_json(
        f"{base}/v1/oauth2/token",
        method="POST",
        headers={"Authorization": _paypal_basic_auth(), "Content-Type": "application/x-www-form-urlencoded"},
        body=None,
    )
    if st == 0:
        return _cors_ok(_jerr("PayPal network error", "paypal_network_error", 502, details=tok))
    # Above call is form-encoded; our helper sends JSON if body not None.
    # Do proper token call using urllib directly:
    try:
        import urllib.request, urllib.parse, urllib.error
        data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/v1/oauth2/token",
            data=data,
            headers={"Authorization": _paypal_basic_auth(), "Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
            tok = json.loads(raw.decode("utf-8"))
            st = resp.status
    except Exception as e:
        return _cors_ok(_jerr("PayPal token error", "paypal_token_error", 502, details=str(e)))

    access = _as_str(tok.get("access_token") or "")
    if not access:
        return _cors_ok(_jerr("PayPal token missing", "paypal_token_missing", 502, details=tok))

    st2, created = _http_json(
        f"{base}/v2/checkout/orders",
        method="POST",
        headers={"Authorization": f"Bearer {access}"},
        body=order_body,
    )
    if st2 not in (200, 201):
        return _cors_ok(_jerr("PayPal create order failed", "paypal_create_fail", 502, status=st2, details=created))
    return _cors_ok(_jok(created))


@bp2.route("/api/store/paypal/capture-order", methods=["POST", "OPTIONS"])
def api_store_paypal_capture_order():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_user():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))

    cid = _get_env("PAYPAL_CLIENT_ID", "").strip()
    sec = _get_env("PAYPAL_SECRET", "").strip()
    if not cid or not sec:
        return _cors_ok(_jerr("PayPal not configured", "paypal_not_configured", 503))

    payload = _require_json()
    order_id = _as_str(payload.get("orderID") or payload.get("order_id") or payload.get("orderId") or payload.get("id") or "").strip()
    if not order_id:
        return _cors_ok(_jerr("Missing order id", "missing_order_id", 400))

    base = _paypal_api_base()
    # token
    try:
        import urllib.request, urllib.parse
        data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/v1/oauth2/token",
            data=data,
            headers={"Authorization": _paypal_basic_auth(), "Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            tok = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return _cors_ok(_jerr("PayPal token error", "paypal_token_error", 502, details=str(e)))

    access = _as_str(tok.get("access_token") or "")
    if not access:
        return _cors_ok(_jerr("PayPal token missing", "paypal_token_missing", 502, details=tok))

    st, cap = _http_json(
        f"{base}/v2/checkout/orders/{order_id}/capture",
        method="POST",
        headers={"Authorization": f"Bearer {access}"},
        body={},
    )
    if st not in (200, 201):
        return _cors_ok(_jerr("PayPal capture failed", "paypal_capture_fail", 502, status=st, details=cap))
    return _cors_ok(_jok(cap))


# ---- Printify proxy endpoints (secrets in .env) ----

def _printify_headers() -> Dict[str, str]:
    tok = _get_env("PRINTIFY_API_TOKEN", "").strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


@bp2.route("/api/store/printify/generate-image", methods=["POST", "OPTIONS"])
def api_store_printify_generate_image():
    """
    Reference endpoint: you can wire this to your own image pipeline.
    For now it simply echoes intent + returns placeholder.
    """
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_user():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))
    payload = _require_json()
    prompt = _as_str(payload.get("prompt") or payload.get("text") or "")
    return _cors_ok(_jok({"ok": True, "prompt": prompt, "imageUrl": "", "image_url": "", "note": "stub"}))


@bp2.route("/api/store/printify/create-product", methods=["POST", "OPTIONS"])
def api_store_printify_create_product():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_user():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))

    tok = _get_env("PRINTIFY_API_TOKEN", "").strip()
    shop_id = _get_env("PRINTIFY_SHOP_ID", "").strip()
    if not tok or not shop_id:
        return _cors_ok(_jerr("Printify not configured", "printify_not_configured", 503))

    payload = _require_json()
    # If client sends simplified payload, return a stub (until you wire full Printify product builder)
    if isinstance(payload, dict) and ("designUrl" in payload or "design_url" in payload):
        design_url = _as_str(payload.get("designUrl") or payload.get("design_url") or "")
        keywords = payload.get("keywords") if isinstance(payload.get("keywords"), list) else []
        stub = {
            "id": "printify_stub_" + uuid.uuid4().hex,
            "name": "Printify Product (stub)",
            "description": "Server-side stub response. Implement full Printify product builder to go live.",
            "price": 0.0,
            "imageUrl": design_url,
            "variants": [],
            "category": "printify",
        }
        return _cors_ok(_jok(stub))

    st, resp = _http_json(
        f"https://api.printify.com/v1/shops/{shop_id}/products.json",
        method="POST",
        headers=_printify_headers(),
        body=payload,
    )
    if st not in (200, 201):
        return _cors_ok(_jerr("Printify create-product failed", "printify_create_fail", 502, status=st, details=resp))
    return _cors_ok(_jok(resp))


@bp2.route("/api/store/printify/products", methods=["GET", "OPTIONS"])
def api_store_printify_products():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_user():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))

    tok = _get_env("PRINTIFY_API_TOKEN", "").strip()
    shop_id = _get_env("PRINTIFY_SHOP_ID", "").strip()
    if not tok or not shop_id:
        return _cors_ok(_jerr("Printify not configured", "printify_not_configured", 503))

    st, resp = _http_json(
        f"https://api.printify.com/v1/shops/{shop_id}/products.json",
        method="GET",
        headers=_printify_headers(),
        body=None,
    )
    if st not in (200, 201):
        return _cors_ok(_jerr("Printify list-products failed", "printify_list_fail", 502, status=st, details=resp))
    # Normalize list for frontend
    try:
        if isinstance(resp, dict) and isinstance(resp.get("data"), list):
            return _cors_ok(_jok({"products": resp.get("data")}))
        if isinstance(resp, list):
            return _cors_ok(_jok({"products": resp}))
    except Exception:
        pass
    return _cors_ok(_jok(resp))


# ---- Kittl proxy endpoints (secrets in .env) ----

@bp2.route("/api/store/kittl/trending-templates", methods=["GET", "OPTIONS"])
def api_store_kittl_trending_templates():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_user():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))
    api_key = _get_env("KITTL_API_KEY", "").strip()
    if not api_key:
        # Fail-soft; keep contract stable
        return _cors_ok(_jok({"templates": [], "note": "kittl not configured"}))
    # No public stable Kittl API contract assumed; treat as stub until you wire it.
    return _cors_ok(_jok({"templates": [], "note": "stub"}))


@bp2.route("/api/store/kittl/create-design", methods=["POST", "OPTIONS"])
def api_store_kittl_create_design():
    if request.method == "OPTIONS":
        return _cors_ok(_jok({"preflight": True}))
    if not _is_user():
        return _cors_ok(_jerr("Unauthorized", "unauthorized", 401))
    api_key = _get_env("KITTL_API_KEY", "").strip()
    if not api_key:
        return _cors_ok(_jerr("Kittl not configured", "kittl_not_configured", 503))
    payload = _require_json()
    # Stub response
    kid = "kittl_" + uuid.uuid4().hex
    keywords = payload.get("keywords") if isinstance(payload.get("keywords"), list) else []
    name = "Kittl Design (stub)"
    imageUrl = ""
    category = "kittl"
    return _cors_ok(_jok({"id": kid, "name": name, "imageUrl": imageUrl, "keywords": keywords, "category": category}))
