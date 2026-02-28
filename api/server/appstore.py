# --==The SarahMemory Project==--
# File: /api/server/appstore.py
# Purpose: SarahMemory Power Store Endpoints
# Design goals:
# - NO endpoint collisions with ANY of the other app*.py files
# - Everything is namespaced under /api/store/
# - Reference implementation for the "store" concept (KV + commerce helpers)
# - All secrets/keys live server-side in .env (PythonAnywhere), never in frontend
#
# Notes
# - This file intentionally fails soft for 3rd-party integrations.
# - If a vendor is not configured, endpoints return {ok:false, error:"..."} with 501/503.
# - Storage is SQLite via app.py injected _connect_sqlite(meta_db).

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import time
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple, List

from flask import Blueprint, jsonify, request

# Stable blueprint name (prevents register collisions across reloads)
bp2 = Blueprint("appstore_v800", __name__)

# Injected by app.py at init_app time
_CONNECT_SQLITE: Optional[Callable[..., Any]] = None
_META_DB: Optional[str] = None
_API_KEY_AUTH_OK: Optional[Callable[[], bool]] = None
_SIGN_OK: Optional[Callable[[bytes, str], bool]] = None

PROJECT_VERSION = os.environ.get("PROJECT_VERSION", "8.0.0")

# ----------------------------- helpers ---------------------------------

def _now() -> float:
    return time.time()


def _iso(ts: Optional[float] = None) -> str:
    return datetime.utcfromtimestamp(ts or _now()).isoformat() + "Z"


def _j() -> Dict[str, Any]:
    return request.get_json(silent=True) or {}


def _ok(**kw) -> Tuple[Any, int]:
    return jsonify({"ok": True, **kw}), 200


def _err(msg: str, code: int = 400, **kw) -> Tuple[Any, int]:
    return jsonify({"ok": False, "error": msg, **kw}), code


def _require_injected() -> bool:
    return bool(_CONNECT_SQLITE and _META_DB)


def _auth_required() -> Optional[Tuple[Any, int]]:
    """Lightweight API-key gate.

    - If app.py injects _API_KEY_AUTH_OK, we enforce it.
    - If not injected, endpoint remains open (dev mode).
    """
    try:
        if _API_KEY_AUTH_OK and not _API_KEY_AUTH_OK():
            return _err("unauthorized", 401)
    except Exception:
        return _err("unauthorized", 401)
    return None


def _safe_key(k: str) -> str:
    s = (k or "").strip()
    s = s.replace("\\", "/")
    s = s.split("/")[-1]
    s = re.sub(r"[^a-zA-Z0-9._:-]+", "_", s)
    s = s.replace("..", "_")
    if not s:
        s = "key_" + uuid.uuid4().hex
    return s


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _hmac_sign(payload: bytes, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def _token_secret() -> str:
    # Use a dedicated secret if provided; otherwise fallback to Flask SECRET_KEY.
    sec = _env("STORE_TOKEN_SECRET") or _env("SECRET_KEY")
    if not sec:
        # Fail-safe: deterministic but not ideal; user should set STORE_TOKEN_SECRET.
        sec = "sarahmemory-store-default-secret"
    return sec


def _make_token(claims: Dict[str, Any], ttl_s: int = 60 * 60 * 24) -> str:
    exp = int(_now()) + int(ttl_s)
    body = {**claims, "exp": exp, "iat": int(_now()), "v": 1}
    raw = json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = _hmac_sign(raw, _token_secret())
    tok = base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=") + "." + sig
    return tok


def _verify_token(token: str) -> Optional[Dict[str, Any]]:
    try:
        t = (token or "").strip()
        if not t or "." not in t:
            return None
        b64, sig = t.split(".", 1)
        pad = "=" * (-len(b64) % 4)
        raw = base64.urlsafe_b64decode((b64 + pad).encode("utf-8"))
        exp_sig = _hmac_sign(raw, _token_secret())
        if not hmac.compare_digest(sig, exp_sig):
            return None
        claims = json.loads(raw.decode("utf-8"))
        if int(claims.get("exp", 0)) < int(_now()):
            return None
        return claims
    except Exception:
        return None


def _bearer_claims() -> Optional[Dict[str, Any]]:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return _verify_token(auth.split(" ", 1)[1].strip())
    # Optional: allow token in json for simple clients
    p = _j()
    if isinstance(p.get("token"), str):
        return _verify_token(str(p.get("token")))
    return None


def _connect():
    if not _require_injected():
        raise RuntimeError("store backend not initialized")
    return _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]


# ----------------------------- DB schema ---------------------------------

def _ensure_tables() -> None:
    if not _require_injected():
        return

    try:
        conn = _connect()
        cur = conn.cursor()

        # KV store with metadata + TTL
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS store_kv (
              k TEXT PRIMARY KEY,
              v_json TEXT NOT NULL,
              meta_json TEXT NOT NULL,
              created_ts REAL NOT NULL,
              updated_ts REAL NOT NULL,
              expires_ts REAL
            )
            """
        )

        # Simple user store for store-auth (reference only)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS store_users (
              user_id TEXT PRIMARY KEY,
              email TEXT UNIQUE NOT NULL,
              pass_hash TEXT NOT NULL,
              salt TEXT NOT NULL,
              created_ts REAL NOT NULL,
              updated_ts REAL NOT NULL,
              disabled INTEGER NOT NULL DEFAULT 0
            )
            """
        )

        # Products
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS store_products (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL,
              description TEXT NOT NULL,
              price REAL NOT NULL,
              category TEXT NOT NULL,
              image_url TEXT NOT NULL,
              rating REAL NOT NULL DEFAULT 4.8,
              reviews INTEGER NOT NULL DEFAULT 127,
              tags_json TEXT NOT NULL DEFAULT '[]',
              source TEXT NOT NULL DEFAULT 'local',
              created_ts REAL NOT NULL,
              updated_ts REAL NOT NULL
            )
            """
        )

        # Generation log
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS store_generation_log (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              kind TEXT NOT NULL,
              payload_json TEXT NOT NULL,
              result_json TEXT NOT NULL,
              created_ts REAL NOT NULL
            )
            """
        )

        conn.commit()
        conn.close()
    except Exception:
        # Fail soft; app.py should log
        try:
            conn.close()  # type: ignore
        except Exception:
            pass


# ----------------------------- endpoints ---------------------------------

@bp2.route("/api/store/health", methods=["GET"])
def store_health():
    if not _require_injected():
        return _err("store not initialized", 503)
    return _ok(status="ok", ts=_now(), version=PROJECT_VERSION)


# -------- KV store (reference) --------

@bp2.route("/api/store/set", methods=["POST"])
def store_set():
    gate = _auth_required()
    if gate:
        return gate
    if not _require_injected():
        return _err("store not initialized", 503)

    p = _j()
    k = _safe_key(str(p.get("key") or p.get("k") or ""))
    v = p.get("value")
    meta = p.get("meta") if isinstance(p.get("meta"), dict) else {}
    ttl = p.get("ttl_s") if p.get("ttl_s") is not None else p.get("ttl")

    expires_ts = None
    try:
        if ttl is not None:
            expires_ts = float(_now()) + float(ttl)
    except Exception:
        expires_ts = None

    try:
        v_json = json.dumps(v, ensure_ascii=False)
        meta_json = json.dumps(meta, ensure_ascii=False)
    except Exception:
        return _err("value/meta not JSON serializable", 400)

    try:
        conn = _connect()
        cur = conn.cursor()
        ts = _now()
        cur.execute(
            """
            INSERT INTO store_kv (k, v_json, meta_json, created_ts, updated_ts, expires_ts)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(k) DO UPDATE SET
              v_json=excluded.v_json,
              meta_json=excluded.meta_json,
              updated_ts=excluded.updated_ts,
              expires_ts=excluded.expires_ts
            """,
            (k, v_json, meta_json, ts, ts, expires_ts),
        )
        conn.commit()
        conn.close()
        return _ok(key=k, expires_ts=expires_ts)
    except Exception as e:
        try:
            conn.close()  # type: ignore
        except Exception:
            pass
        return _err(f"db error: {e}", 500)


@bp2.route("/api/store/get", methods=["GET"])
def store_get():
    gate = _auth_required()
    if gate:
        return gate
    if not _require_injected():
        return _err("store not initialized", 503)

    k = _safe_key(str(request.args.get("key") or request.args.get("k") or ""))

    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("SELECT v_json, meta_json, created_ts, updated_ts, expires_ts FROM store_kv WHERE k=?", (k,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return _err("not found", 404, key=k)

        v_json, meta_json, created_ts, updated_ts, expires_ts = row
        if expires_ts is not None and float(expires_ts) < _now():
            # expire + delete
            cur.execute("DELETE FROM store_kv WHERE k=?", (k,))
            conn.commit()
            conn.close()
            return _err("expired", 404, key=k)

        conn.close()
        return _ok(
            key=k,
            value=json.loads(v_json),
            meta=json.loads(meta_json),
            created_ts=created_ts,
            updated_ts=updated_ts,
            expires_ts=expires_ts,
        )
    except Exception as e:
        try:
            conn.close()  # type: ignore
        except Exception:
            pass
        return _err(f"db error: {e}", 500)


@bp2.route("/api/store/del", methods=["POST"])
def store_del():
    gate = _auth_required()
    if gate:
        return gate
    if not _require_injected():
        return _err("store not initialized", 503)

    p = _j()
    k = _safe_key(str(p.get("key") or p.get("k") or ""))
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("DELETE FROM store_kv WHERE k=?", (k,))
        conn.commit()
        deleted = cur.rowcount
        conn.close()
        return _ok(key=k, deleted=deleted)
    except Exception as e:
        try:
            conn.close()  # type: ignore
        except Exception:
            pass
        return _err(f"db error: {e}", 500)


# -------- Store Auth (reference) --------

def _pw_hash(password: str, salt: str) -> str:
    # PBKDF2-SHA256 (built-in) for portability
    pw = (password or "").encode("utf-8")
    s = (salt or "").encode("utf-8")
    dk = hashlib.pbkdf2_hmac("sha256", pw, s, 150_000)
    return base64.b64encode(dk).decode("utf-8")


def _email_norm(email: str) -> str:
    return (email or "").strip().lower()


def _user_public(row) -> Dict[str, Any]:
    return {"user_id": row[0], "email": row[1], "disabled": bool(row[5])}


@bp2.route("/api/store/auth/register", methods=["POST"])
def store_register():
    # Public endpoint
    if not _require_injected():
        return _err("store not initialized", 503)

    p = _j()
    email = _email_norm(str(p.get("email") or ""))
    password = str(p.get("password") or "")
    if not email or "@" not in email:
        return _err("invalid email", 400)
    if len(password) < 6:
        return _err("password too short", 400)

    user_id = "u_" + uuid.uuid4().hex
    salt = uuid.uuid4().hex
    ph = _pw_hash(password, salt)

    try:
        conn = _connect()
        cur = conn.cursor()
        ts = _now()
        cur.execute(
            "INSERT INTO store_users (user_id, email, pass_hash, salt, created_ts, updated_ts, disabled) VALUES (?, ?, ?, ?, ?, ?, 0)",
            (user_id, email, ph, salt, ts, ts),
        )
        conn.commit()
        # Return token
        token = _make_token({"sub": user_id, "email": email, "role": "user"})
        conn.close()
        return _ok(user={"user_id": user_id, "email": email}, token=token)
    except Exception as e:
        try:
            conn.close()  # type: ignore
        except Exception:
            pass
        # Likely duplicate
        return _err(f"register failed: {e}", 400)


@bp2.route("/api/store/auth/login", methods=["POST"])
def store_login():
    # Public endpoint
    if not _require_injected():
        return _err("store not initialized", 503)

    p = _j()
    email = _email_norm(str(p.get("email") or ""))
    password = str(p.get("password") or "")

    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute("SELECT user_id, email, pass_hash, salt, created_ts, disabled FROM store_users WHERE email=?", (email,))
        row = cur.fetchone()
        if not row:
            conn.close()
            return _err("invalid credentials", 401)
        if int(row[5]) == 1:
            conn.close()
            return _err("account disabled", 403)

        exp = _pw_hash(password, row[3])
        if not hmac.compare_digest(str(exp), str(row[2])):
            conn.close()
            return _err("invalid credentials", 401)

        token = _make_token({"sub": row[0], "email": row[1], "role": "user"})
        conn.close()
        return _ok(user={"user_id": row[0], "email": row[1]}, token=token)
    except Exception as e:
        try:
            conn.close()  # type: ignore
        except Exception:
            pass
        return _err(f"login failed: {e}", 500)


@bp2.route("/api/store/auth/password-reset", methods=["POST"])
def store_password_reset():
    # Reference implementation: generate a reset token (no email sending here)
    if not _require_injected():
        return _err("store not initialized", 503)

    p = _j()
    email = _email_norm(str(p.get("email") or ""))
    if not email:
        return _err("email required", 400)

    # Always return ok for privacy; token is only returned if STORE_DEBUG_RESET=1
    debug = _env("STORE_DEBUG_RESET", "0") == "1"
    token = _make_token({"email": email, "role": "reset"}, ttl_s=15 * 60)
    if debug:
        return _ok(message="reset token issued", reset_token=token)
    return _ok(message="if the account exists, reset instructions will be delivered")


@bp2.route("/api/store/auth/admin/login", methods=["POST"])
def store_admin_login():
    # Admin is controlled by env vars (PythonAnywhere .env)
    p = _j()
    email = _email_norm(str(p.get("email") or ""))
    password = str(p.get("password") or "")

    admin_email = _email_norm(_env("STORE_ADMIN_EMAIL"))
    admin_pass = _env("STORE_ADMIN_PASSWORD")

    if not admin_email or not admin_pass:
        return _err("admin auth not configured", 503)

    if email != admin_email or not hmac.compare_digest(password, admin_pass):
        return _err("invalid credentials", 401)

    token = _make_token({"sub": "admin", "email": admin_email, "role": "admin"}, ttl_s=12 * 60 * 60)
    return _ok(user={"user_id": "admin", "email": admin_email}, token=token)


def _require_admin() -> Optional[Tuple[Any, int]]:
    claims = _bearer_claims()
    if not claims or str(claims.get("role")) != "admin":
        return _err("admin required", 403)
    return None


# -------- Trends / Products (Power Store) --------

@bp2.route("/api/store/trends", methods=["POST"])
def store_trends():
    # Public endpoint; optionally gate with api key
    gate = _auth_required()
    if gate:
        return gate

    p = _j()
    q = str(p.get("query") or p.get("q") or "").strip()
    limit = int(p.get("limit") or 10)
    limit = max(1, min(limit, 50))

    # Reference: deterministic pseudo-trends
    base = [
        "wireless earbuds",
        "fitness tracker",
        "bluetooth speaker",
        "USB-C cable",
        "gaming mouse",
        "mechanical keyboard",
        "portable SSD",
        "smart home plug",
        "webcam 1080p",
        "phone tripod",
    ]
    if q:
        base = [f"{q} {t}" for t in ("deal", "new", "top", "best", "2026", "bundle")] + base

    trends = [{"keyword": base[i % len(base)], "score": round(0.9 - (i * 0.01), 3)} for i in range(limit)]
    return _ok(trends=trends, source="reference")


def _product_row_to_obj(row) -> Dict[str, Any]:
    return {
        "id": int(row[0]),
        "name": row[1],
        "description": row[2],
        "price": float(row[3]),
        "category": row[4],
        "image_url": row[5],
        "rating": float(row[6]),
        "reviews": int(row[7]),
        "tags": json.loads(row[8] or "[]"),
        "source": row[9],
    }


@bp2.route("/api/store/products", methods=["GET"])
def store_products_list():
    gate = _auth_required()
    if gate:
        return gate
    if not _require_injected():
        return _err("store not initialized", 503)

    q = str(request.args.get("q") or "").strip().lower()
    category = str(request.args.get("category") or "").strip().lower()
    limit = int(request.args.get("limit") or 50)
    offset = int(request.args.get("offset") or 0)
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    try:
        conn = _connect()
        cur = conn.cursor()

        where = []
        args: List[Any] = []
        if q:
            where.append("(lower(name) LIKE ? OR lower(description) LIKE ?)")
            args.extend([f"%{q}%", f"%{q}%"])
        if category:
            where.append("lower(category) = ?")
            args.append(category)

        sql = "SELECT id,name,description,price,category,image_url,rating,reviews,tags_json,source FROM store_products"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY updated_ts DESC LIMIT ? OFFSET ?"
        args.extend([limit, offset])

        cur.execute(sql, tuple(args))
        rows = cur.fetchall() or []
        conn.close()
        products = [_product_row_to_obj(r) for r in rows]
        return _ok(products=products, count=len(products))
    except Exception as e:
        try:
            conn.close()  # type: ignore
        except Exception:
            pass
        return _err(f"db error: {e}", 500)


@bp2.route("/api/store/products/bulk", methods=["POST"])
def store_products_bulk_upsert():
    gate = _auth_required()
    if gate:
        return gate
    admin = _require_admin()
    if admin:
        return admin
    if not _require_injected():
        return _err("store not initialized", 503)

    p = _j()
    items = p.get("products") or p.get("items")
    if not isinstance(items, list) or not items:
        return _err("products list required", 400)

    def _coerce_price(x) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    inserted = 0
    updated = 0

    try:
        conn = _connect()
        cur = conn.cursor()
        ts = _now()

        for it in items:
            if not isinstance(it, dict):
                continue
            pid = it.get("id")
            name = str(it.get("name") or it.get("title") or "").strip() or "Untitled"
            desc = str(it.get("description") or "").strip()
            price = _coerce_price(it.get("price"))
            category = str(it.get("category") or "electronics").strip()
            image_url = str(it.get("image_url") or it.get("image") or it.get("imageUrl") or "").strip()
            if not image_url:
                image_url = "https://placehold.co/600x400?text=SarahMemory"
            rating = float(it.get("rating") or 4.8)
            reviews = int(it.get("reviews") or 127)
            tags = it.get("tags") if isinstance(it.get("tags"), list) else []
            tags_json = json.dumps(tags, ensure_ascii=False)
            source = str(it.get("source") or "bulk").strip()

            if pid:
                cur.execute(
                    """
                    UPDATE store_products
                    SET name=?, description=?, price=?, category=?, image_url=?, rating=?, reviews=?, tags_json=?, source=?, updated_ts=?
                    WHERE id=?
                    """,
                    (name, desc, price, category, image_url, rating, reviews, tags_json, source, ts, int(pid)),
                )
                if cur.rowcount:
                    updated += 1
                else:
                    # id not found -> insert new
                    cur.execute(
                        """
                        INSERT INTO store_products (name,description,price,category,image_url,rating,reviews,tags_json,source,created_ts,updated_ts)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (name, desc, price, category, image_url, rating, reviews, tags_json, source, ts, ts),
                    )
                    inserted += 1
            else:
                cur.execute(
                    """
                    INSERT INTO store_products (name,description,price,category,image_url,rating,reviews,tags_json,source,created_ts,updated_ts)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (name, desc, price, category, image_url, rating, reviews, tags_json, source, ts, ts),
                )
                inserted += 1

        conn.commit()
        conn.close()
        return _ok(inserted=inserted, updated=updated)
    except Exception as e:
        try:
            conn.close()  # type: ignore
        except Exception:
            pass
        return _err(f"db error: {e}", 500)


@bp2.route("/api/store/products/generate", methods=["POST"])
def store_products_generate():
    gate = _auth_required()
    if gate:
        return gate
    if not _require_injected():
        return _err("store not initialized", 503)

    p = _j()
    keywords = p.get("keywords") or p.get("trends") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    if not isinstance(keywords, list):
        keywords = []

    count = int(p.get("count") or p.get("limit") or 4)
    count = max(1, min(count, 20))

    # Reference generator: creates deterministic products
    def mk(i: int, kw: str) -> Dict[str, Any]:
        kw_s = str(kw or "Smart Gadget").strip() or "Smart Gadget"
        price = round(19.99 + (i * 10) + (len(kw_s) % 7) * 3.0, 2)
        return {
            "name": kw_s.title(),
            "description": f"AI-curated product based on trend: {kw_s}",
            "price": price,
            "category": str(p.get("category") or "electronics"),
            "image_url": str(p.get("image_url") or "https://placehold.co/600x400?text=SarahMemory"),
            "rating": 4.8,
            "reviews": 127,
            "tags": ["AI-Curated", "Trend-Based", "Smart"],
            "source": "generator",
        }

    seed = keywords if keywords else ["wireless bluetooth headphones", "smart fitness tracker", "portable speaker", "usb-c fast charging cable"]
    items = [mk(i, seed[i % len(seed)]) for i in range(count)]

    # Persist
    try:
        conn = _connect()
        cur = conn.cursor()
        ts = _now()
        for it in items:
            cur.execute(
                """
                INSERT INTO store_products (name,description,price,category,image_url,rating,reviews,tags_json,source,created_ts,updated_ts)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    it["name"],
                    it["description"],
                    float(it["price"]),
                    it["category"],
                    it["image_url"],
                    float(it["rating"]),
                    int(it["reviews"]),
                    json.dumps(it.get("tags") or [], ensure_ascii=False),
                    it.get("source") or "generator",
                    ts,
                    ts,
                ),
            )
        conn.commit()

        # Return inserted rows (latest)
        cur.execute(
            "SELECT id,name,description,price,category,image_url,rating,reviews,tags_json,source FROM store_products ORDER BY id DESC LIMIT ?",
            (len(items),),
        )
        rows = cur.fetchall() or []
        conn.close()
        products = list(reversed([_product_row_to_obj(r) for r in rows]))

        _log("products_generate", p, {"count": len(products)})
        return _ok(products=products, generated=len(products))
    except Exception as e:
        try:
            conn.close()  # type: ignore
        except Exception:
            pass
        return _err(f"db error: {e}", 500)


def _log(kind: str, payload: Any, result: Any) -> None:
    if not _require_injected():
        return
    try:
        conn = _connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO store_generation_log (kind,payload_json,result_json,created_ts) VALUES (?,?,?,?)",
            (kind, json.dumps(payload, ensure_ascii=False), json.dumps(result, ensure_ascii=False), _now()),
        )
        conn.commit()
        conn.close()
    except Exception:
        try:
            conn.close()  # type: ignore
        except Exception:
            pass


@bp2.route("/api/store/generation-log", methods=["POST"])
def store_generation_log_write():
    gate = _auth_required()
    if gate:
        return gate
    if not _require_injected():
        return _err("store not initialized", 503)

    p = _j()
    kind = str(p.get("kind") or "log")
    _log(kind, p.get("payload"), p.get("result"))
    return _ok(message="logged")


# -------- PayPal proxy (server-side secrets) --------

@bp2.route("/api/store/paypal/config", methods=["GET"])
def store_paypal_config():
    gate = _auth_required()
    if gate:
        return gate

    client_id = _env("PAYPAL_CLIENT_ID")
    env = _env("PAYPAL_ENV", "sandbox") or "sandbox"
    if not client_id:
        return _err("paypal not configured", 503)
    return _ok(clientId=client_id, environment=env)


def _paypal_token() -> Optional[str]:
    client_id = _env("PAYPAL_CLIENT_ID")
    secret = _env("PAYPAL_SECRET")
    env = (_env("PAYPAL_ENV", "sandbox") or "sandbox").lower()
    if not client_id or not secret:
        return None

    base = "https://api-m.paypal.com" if env == "live" else "https://api-m.sandbox.paypal.com"

    try:
        import urllib.request
        import urllib.parse

        data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode("utf-8")
        req = urllib.request.Request(base + "/v1/oauth2/token", data=data, method="POST")
        auth = base64.b64encode(f"{client_id}:{secret}".encode("utf-8")).decode("utf-8")
        req.add_header("Authorization", f"Basic {auth}")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8")
        j = json.loads(body)
        return str(j.get("access_token") or "") or None
    except Exception:
        return None


def _paypal_base() -> str:
    env = (_env("PAYPAL_ENV", "sandbox") or "sandbox").lower()
    return "https://api-m.paypal.com" if env == "live" else "https://api-m.sandbox.paypal.com"


@bp2.route("/api/store/paypal/create-order", methods=["POST"])
def store_paypal_create_order():
    gate = _auth_required()
    if gate:
        return gate

    tok = _paypal_token()
    if not tok:
        return _err("paypal not configured", 503)

    p = _j()
    currency = str(p.get("currency") or "USD")
    value = str(p.get("value") or p.get("amount") or "0.00")

    payload = {
        "intent": "CAPTURE",
        "purchase_units": [{"amount": {"currency_code": currency, "value": value}}],
    }

    try:
        import urllib.request

        req = urllib.request.Request(_paypal_base() + "/v2/checkout/orders", data=json.dumps(payload).encode("utf-8"), method="POST")
        req.add_header("Authorization", f"Bearer {tok}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8")
        j = json.loads(body)
        return _ok(order=j)
    except Exception as e:
        return _err(f"paypal create-order failed: {e}", 502)


@bp2.route("/api/store/paypal/capture-order", methods=["POST"])
def store_paypal_capture_order():
    gate = _auth_required()
    if gate:
        return gate

    tok = _paypal_token()
    if not tok:
        return _err("paypal not configured", 503)

    p = _j()
    order_id = str(p.get("orderID") or p.get("order_id") or p.get("id") or "").strip()
    if not order_id:
        return _err("order_id required", 400)

    try:
        import urllib.request

        req = urllib.request.Request(_paypal_base() + f"/v2/checkout/orders/{order_id}/capture", data=b"{}", method="POST")
        req.add_header("Authorization", f"Bearer {tok}")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8")
        j = json.loads(body)
        return _ok(capture=j)
    except Exception as e:
        return _err(f"paypal capture-order failed: {e}", 502)


# -------- Printify proxy (server-side secrets) --------

@bp2.route("/api/store/printify/generate-image", methods=["POST"])
def store_printify_generate_image():
    gate = _auth_required()
    if gate:
        return gate

    token = _env("PRINTIFY_API_TOKEN")
    if not token:
        return _err("printify not configured", 503)

    # This endpoint is a placeholder: Printify is product/print; image gen is usually external.
    # Expect your own image generator to run and return a URL.
    p = _j()
    prompt = str(p.get("prompt") or "").strip()
    if not prompt:
        return _err("prompt required", 400)

    # Fail-soft: return a deterministic placeholder until you wire a real generator.
    img = f"https://placehold.co/1024x1024?text={uuid.uuid4().hex[:8]}"
    _log("printify_generate_image", {"prompt": prompt}, {"image_url": img})
    return _ok(image_url=img, engine="placeholder")


@bp2.route("/api/store/printify/create-product", methods=["POST"])
def store_printify_create_product():
    gate = _auth_required()
    if gate:
        return gate

    token = _env("PRINTIFY_API_TOKEN")
    shop_id = _env("PRINTIFY_SHOP_ID")
    if not token or not shop_id:
        return _err("printify not configured", 503)

    p = _j()
    # Reference: accept payload and just log it; real Printify call can be added later.
    _log("printify_create_product", p, {"queued": True})
    return _ok(status="queued", note="wire Printify product create here")


@bp2.route("/api/store/printify/products", methods=["GET"])
def store_printify_products():
    gate = _auth_required()
    if gate:
        return gate

    token = _env("PRINTIFY_API_TOKEN")
    shop_id = _env("PRINTIFY_SHOP_ID")
    if not token or not shop_id:
        return _err("printify not configured", 503)

    # Reference: return empty list (real Printify list call can be added later)
    return _ok(products=[], source="placeholder")


# -------- Kittl proxy (server-side secrets) --------

@bp2.route("/api/store/kittl/trending-templates", methods=["GET"])
def store_kittl_trending_templates():
    gate = _auth_required()
    if gate:
        return gate

    token = _env("KITTL_API_KEY")
    if not token:
        return _err("kittl not configured", 503)

    # Reference: placeholder response until wired
    templates = [
        {"id": "tmpl_" + uuid.uuid4().hex[:10], "name": "Minimal Logo", "score": 0.92},
        {"id": "tmpl_" + uuid.uuid4().hex[:10], "name": "Vintage Badge", "score": 0.89},
        {"id": "tmpl_" + uuid.uuid4().hex[:10], "name": "Streetwear Typo", "score": 0.86},
    ]
    return _ok(templates=templates, source="placeholder")


@bp2.route("/api/store/kittl/create-design", methods=["POST"])
def store_kittl_create_design():
    gate = _auth_required()
    if gate:
        return gate

    token = _env("KITTL_API_KEY")
    if not token:
        return _err("kittl not configured", 503)

    p = _j()
    title = str(p.get("title") or "Design").strip()
    _log("kittl_create_design", p, {"created": True})
    return _ok(design={"id": "dsgn_" + uuid.uuid4().hex[:10], "title": title}, source="placeholder")


# ----------------------------- init ---------------------------------

def init_appstore(
    connect_sqlite: Callable[..., Any],
    meta_db_path: str,
    api_key_auth_ok: Optional[Callable[[], bool]] = None,
    sign_ok: Optional[Callable[[bytes, str], bool]] = None,
) -> Blueprint:
    """Inject storage/auth helpers and return blueprint."""
    global _CONNECT_SQLITE, _META_DB, _API_KEY_AUTH_OK, _SIGN_OK
    _CONNECT_SQLITE = connect_sqlite
    _META_DB = meta_db_path
    _API_KEY_AUTH_OK = api_key_auth_ok
    _SIGN_OK = sign_ok

    _ensure_tables()
    return bp2


def init_app(
    app,
    connect_sqlite: Callable[..., Any],
    meta_db_path: str,
    api_key_auth_ok: Optional[Callable[[], bool]] = None,
    sign_ok: Optional[Callable[[bytes, str], bool]] = None,
) -> None:
    """Preferred: inject + register blueprint into Flask app (single-call)."""
    if "appstore_v800" in getattr(app, "blueprints", {}):
        return

    init_appstore(connect_sqlite, meta_db_path, api_key_auth_ok=api_key_auth_ok, sign_ok=sign_ok)
    _ensure_tables()
    app.register_blueprint(bp2)
