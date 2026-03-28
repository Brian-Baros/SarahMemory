"""
 --==The SarahMemory Project==--
 File: /app/server/appnet2.py
 Purpose: SarahNet "Bravo" — Identity/Trust + Overlay Tunnel Control + Virtual DNS/Name Directory
 Part of the SarahMemory Companion AI-bot Platform
 Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
 www.linkedin.com/in/brian-baros-29962a176
 https://www.facebook.com/bbaros
 brian.baros@sarahmemory.com
 'The SarahMemory Companion AI-Bot Platform, SarahMemory AiOS, and all Parts of the SarahMemory Project are property of SOFTDEV0 LLC., & Brian Lee Baros'
 https://www.sarahmemory.com
 https://api.sarahmemory.com
 https://ai.sarahmemory.com
 https://store.sarahmemory.com
==============================================================================================
 Design goals:
 - NO endpoint collisions with appnet.py
 - Everything is namespaced under /api/net2/*
 - Pure control-plane (HTTP + sqlite). No OS-level VPN claims here.
 - Cross-platform safe: Windows/Linux/macOS/headless/cloud.
 - app.py should call:
     import appnet
     import appnet2
     appnet.init_app(app, CONNECT_SQLITE, META_DB, api_key_auth_ok=..., sign_ok=...)
     appnet2.init_app(app, CONNECT_SQLITE, META_DB, api_key_auth_ok=..., sign_ok=...)
"""

from __future__ import annotations
# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "api_bridge"
# CATEGORY = "network_identity_and_trust"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "api_server"
# API_DOMAIN = "net2"
# HARDWARE_DOMAIN = "network"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "sarahnet_control_plane_api"
# FAMILY = "core_network"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "SarahNet Bravo control-plane API under /api/net2/* for node identity, trust tiers, challenge/attest flows, virtual DNS/name directory, and overlay tunnel session control."
# --- SARAHMETA END ---
import base64
import hashlib
import json
import os
import time
import uuid
from typing import Any, Callable, Dict, Optional

from flask import Blueprint, jsonify, request

bp2 = Blueprint("appnet2_v800", __name__)

# Injected by app.py at init_app time
_CONNECT_SQLITE: Optional[Callable[..., Any]] = None
_META_DB: Optional[str] = None
_API_KEY_AUTH_OK: Optional[Callable[[], bool]] = None
_SIGN_OK: Optional[Callable[[bytes, str], bool]] = None

# ----------------------------- helpers ---------------------------------

def _now() -> float:
    return time.time()

def _j() -> Dict[str, Any]:
    return request.get_json(silent=True) or {}

def _ok(**kw):
    return jsonify({"ok": True, **kw}), 200

def _err(msg: str, code: int = 400, **kw):
    return jsonify({"ok": False, "error": msg, **kw}), code

def _require_injected() -> bool:
    return bool(_CONNECT_SQLITE and _META_DB)

def _body_bytes() -> bytes:
    try:
        return request.get_data(cache=True) or b""
    except Exception:
        return b""

def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"

def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")

def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"), validate=False)

def _verify_auth(body_bytes: bytes) -> bool:
    """
    Accept either:
      - X-Sarah-Signature verified by injected _SIGN_OK(body, sig)
      - API key verified by injected _API_KEY_AUTH_OK()
    If nothing injected, allow (dev mode).
    """
    sig = (request.headers.get("X-Sarah-Signature") or "").strip()
    if sig and _SIGN_OK:
        try:
            return bool(_SIGN_OK(body_bytes, sig))
        except Exception:
            return False

    if _API_KEY_AUTH_OK:
        try:
            return bool(_API_KEY_AUTH_OK())
        except Exception:
            return False

    return True

def _ensure_tables() -> None:
    if not _require_injected():
        return
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()

        # -----------------------------
        # Node Identity & Trust
        # -----------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net2_nodes (
                node_id TEXT PRIMARY KEY,
                pubkey TEXT,
                created_ts REAL,
                last_ts REAL,
                meta_json TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_net2_nodes_last_ts ON net2_nodes(last_ts)")

        # Challenges (nonce/expiry) for attest flow
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net2_challenges (
                id TEXT PRIMARY KEY,
                node_id TEXT,
                nonce TEXT,
                created_ts REAL,
                expires_ts REAL,
                used INTEGER DEFAULT 0
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_net2_challenges_node ON net2_challenges(node_id, expires_ts)")

        # Trust tier (lightweight)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net2_trust (
                node_id TEXT PRIMARY KEY,
                tier INTEGER DEFAULT 0,
                score REAL DEFAULT 0,
                updated_ts REAL
            )
        """)

        # -----------------------------
        # Virtual DNS / Name Directory
        # -----------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net2_dns (
                name TEXT PRIMARY KEY,
                rtype TEXT,
                value TEXT,
                ttl INTEGER DEFAULT 60,
                owner_node TEXT,
                updated_ts REAL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_net2_dns_owner ON net2_dns(owner_node, updated_ts)")

        # -----------------------------
        # Overlay Tunnel Sessions (control-plane)
        # This is NOT OS-level VPN. It's a SarahNet overlay session artifact.
        # -----------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net2_tunnel_sessions (
                session_id TEXT PRIMARY KEY,
                from_node TEXT,
                to_node TEXT,
                created_ts REAL,
                expires_ts REAL,
                policy_json TEXT,
                status TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_net2_tunnel_sessions_to ON net2_tunnel_sessions(to_node, expires_ts)")

        con.commit()
    except Exception:
        try:
            if con:
                con.rollback()
        except Exception:
            pass
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

def _purge_expired_challenges(cur) -> int:
    """
    Opportunistic cleanup to keep META_DB lean.
    Purges:
      - expired challenges
      - used challenges older than a small grace window
    """
    try:
        now = _now()
        used_grace = int(os.environ.get("SARAHNET_CHALLENGE_USED_GRACE_SEC", "300") or 300)  # 5 min default
        used_before = now - max(60, min(86400, used_grace))

        # Expired
        cur.execute("DELETE FROM net2_challenges WHERE expires_ts < ?", (now,))
        expired_n = int(getattr(cur, "rowcount", 0) or 0)

        # Used & old
        cur.execute("DELETE FROM net2_challenges WHERE used=1 AND created_ts < ?", (used_before,))
        used_n = int(getattr(cur, "rowcount", 0) or 0)

        return expired_n + used_n
    except Exception:
        return 0

# ----------------------------- endpoints ---------------------------------

@bp2.get("/api/net2/ping")
def net2_ping():
    return _ok(
        pong=True,
        ts=_now(),
        version=os.environ.get("PROJECT_VERSION", "8.0.0"),
        module="appnet2",
    )

@bp2.get("/api/net2/health")
def net2_health():
    payload = {
        "ok": True,
        "enabled": bool(_CONNECT_SQLITE and _META_DB),
        "ts": _now(),
        "version": os.environ.get("PROJECT_VERSION", "8.0.0"),
    }
    if not payload["enabled"]:
        payload["ok"] = False
        payload["reason"] = "no_storage"
        return jsonify(payload), 200
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1;")
        cur.fetchone()
        con.close()
        payload["storage"] = "sqlite"
    except Exception as e:
        payload["ok"] = False
        payload["reason"] = "db_error"
        payload["detail"] = str(e)
    return jsonify(payload), 200

# ------------------ Identity / Trust ------------------

@bp2.post("/api/net2/node/register")
def net2_node_register():
    raw = _body_bytes()
    if not _verify_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Storage not configured (META_DB).", 500)

    data = _j()
    node_id = (data.get("node_id") or "").strip()
    pubkey = (data.get("pubkey") or "").strip()
    meta = data.get("meta") or {}

    if not node_id:
        return _err("Missing node_id")
    if pubkey and len(pubkey) > 4096:
        return _err("pubkey too large")

    try:
        meta_json = json.dumps(meta, ensure_ascii=False)
    except Exception:
        meta_json = "{}"

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        now = _now()

        # housekeeping
        _purge_expired_challenges(cur)

        cur.execute(
            "INSERT INTO net2_nodes(node_id,pubkey,created_ts,last_ts,meta_json) VALUES(?,?,?,?,?) "
            "ON CONFLICT(node_id) DO UPDATE SET pubkey=excluded.pubkey, last_ts=excluded.last_ts, meta_json=excluded.meta_json",
            (node_id, pubkey or None, now, now, meta_json),
        )
        cur.execute(
            "INSERT INTO net2_trust(node_id,tier,score,updated_ts) VALUES(?,?,?,?) "
            "ON CONFLICT(node_id) DO NOTHING",
            (node_id, 0, 0.0, now),
        )
        con.commit()
        return _ok(registered=True, node_id=node_id)
    except Exception as e:
        return _err("DB error", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

@bp2.post("/api/net2/node/challenge")
def net2_node_challenge():
    raw = _body_bytes()
    if not _verify_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Storage not configured (META_DB).", 500)

    data = _j()
    node_id = (data.get("node_id") or "").strip()
    if not node_id:
        return _err("Missing node_id")

    # Nonce: random 32 bytes base64
    nonce = _b64e(os.urandom(32))
    cid = _new_id("chal")
    now = _now()
    ttl = int(os.environ.get("SARAHNET_CHALLENGE_TTL_SEC", "120") or 120)
    expires = now + max(30, min(600, ttl))

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()

        # housekeeping
        purged = _purge_expired_challenges(cur)

        cur.execute(
            "INSERT INTO net2_challenges(id,node_id,nonce,created_ts,expires_ts,used) VALUES(?,?,?,?,?,0)",
            (cid, node_id, nonce, now, expires),
        )
        con.commit()
        return _ok(challenge_id=cid, node_id=node_id, nonce=nonce, expires_ts=expires, purged=purged)
    except Exception as e:
        return _err("DB error", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@bp2.get("/api/net2/node/list")
def net2_node_list():
    """
    Debug/ops endpoint (read-only):
      - returns recently-seen nodes
    """
    raw = _body_bytes()
    if not _verify_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Storage not configured (META_DB).", 500)

    limit = int(request.args.get("limit") or 50)
    limit = max(1, min(500, limit))

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute("SELECT node_id, created_ts, last_ts, meta_json FROM net2_nodes ORDER BY last_ts DESC LIMIT ?", (limit,))
        rows = cur.fetchall() or []
        out = []
        for r in rows:
            try:
                meta = json.loads(r[3] or "{}")
            except Exception:
                meta = {}
            out.append({"node_id": r[0], "created_ts": r[1], "last_ts": r[2], "meta": meta})
        return _ok(nodes=out, count=len(out), ts=_now())
    except Exception as e:
        return _err("DB error", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@bp2.post("/api/net2/node/attest")
def net2_node_attest():
    """
    Attestation placeholder:
    - We mark challenge used and update node last_ts.
    - Signature verification belongs in node code + injected auth, or later cryptographic expansion.
    Body:
      { "node_id": "...", "challenge_id": "...", "proof": {..} }
    """
    raw = _body_bytes()
    if not _verify_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Storage not configured (META_DB).", 500)

    data = _j()
    node_id = (data.get("node_id") or "").strip()
    challenge_id = (data.get("challenge_id") or "").strip()
    proof = data.get("proof") or {}

    if not node_id or not challenge_id:
        return _err("Missing node_id/challenge_id")

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()

        # housekeeping
        purged = _purge_expired_challenges(cur)

        cur.execute("SELECT nonce, expires_ts, used FROM net2_challenges WHERE id=? AND node_id=? LIMIT 1", (challenge_id, node_id))
        row = cur.fetchone()
        if not row:
            return _err("Unknown challenge", 404)
        if int(row[2] or 0) == 1:
            return _err("Challenge already used", 409)
        if float(row[1] or 0) < _now():
            return _err("Challenge expired", 409)

        # Mark used
        cur.execute("UPDATE net2_challenges SET used=1 WHERE id=? AND node_id=?", (challenge_id, node_id))

        # Update last_ts on node
        cur.execute("UPDATE net2_nodes SET last_ts=? WHERE node_id=?", (_now(), node_id))

        con.commit()
        return _ok(attested=True, node_id=node_id, challenge_id=challenge_id, purged=purged, note="proof stored/accepted (verification policy external)")
    except Exception as e:
        return _err("DB error", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

@bp2.post("/api/net2/node/challenge_and_attest")
def net2_node_challenge_and_attest():
    """
    Single-call operational convenience to avoid copy/paste errors.
    Body:
      { "node_id": "...", "proof": {..} }
    Returns:
      { challenge_id, nonce, expires_ts, attested:true, ... }
    """
    raw = _body_bytes()
    if not _verify_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Storage not configured (META_DB).", 500)

    data = _j()
    node_id = (data.get("node_id") or "").strip()
    proof = data.get("proof") or {}
    if not node_id:
        return _err("Missing node_id")

    # Create challenge
    nonce = _b64e(os.urandom(32))
    cid = _new_id("chal")
    now = _now()
    ttl = int(os.environ.get("SARAHNET_CHALLENGE_TTL_SEC", "120") or 120)
    expires = now + max(30, min(600, ttl))

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()

        # housekeeping
        purged = _purge_expired_challenges(cur)

        cur.execute(
            "INSERT INTO net2_challenges(id,node_id,nonce,created_ts,expires_ts,used) VALUES(?,?,?,?,?,0)",
            (cid, node_id, nonce, now, expires),
        )

        # Immediately attest (mark used) – this preserves your current policy (verification external)
        cur.execute("UPDATE net2_challenges SET used=1 WHERE id=? AND node_id=?", (cid, node_id))
        cur.execute("UPDATE net2_nodes SET last_ts=? WHERE node_id=?", (_now(), node_id))

        con.commit()
        return _ok(
            node_id=node_id,
            challenge_id=cid,
            nonce=nonce,
            expires_ts=expires,
            attested=True,
            purged=purged,
            note="challenge+attest bundled (verification policy external)",
        )
    except Exception as e:
        return _err("DB error", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

@bp2.get("/api/net2/node/profile")
def net2_node_profile():
    if not _require_injected():
        return _err("Storage not configured (META_DB).", 500)
    node_id = (request.args.get("node_id") or "").strip()
    if not node_id:
        return _err("Missing node_id")

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute("SELECT node_id,pubkey,created_ts,last_ts,meta_json FROM net2_nodes WHERE node_id=? LIMIT 1", (node_id,))
        row = cur.fetchone()
        if not row:
            return _ok(found=False, node_id=node_id)
        try:
            meta = json.loads(row[4] or "{}")
        except Exception:
            meta = {}
        cur.execute("SELECT tier,score,updated_ts FROM net2_trust WHERE node_id=? LIMIT 1", (node_id,))
        tr = cur.fetchone() or (0, 0.0, None)
        return _ok(
            found=True,
            node={"node_id": row[0], "pubkey": row[1], "created_ts": row[2], "last_ts": row[3], "meta": meta},
            trust={"tier": int(tr[0] or 0), "score": float(tr[1] or 0.0), "updated_ts": tr[2]},
        )
    except Exception as e:
        return _err("DB error", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

# ------------------ Virtual DNS / Directory ------------------

@bp2.post("/api/net2/dns/upsert")
def net2_dns_upsert():
    raw = _body_bytes()
    if not _verify_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Storage not configured (META_DB).", 500)

    data = _j()
    name = (data.get("name") or "").strip().lower()
    rtype = (data.get("rtype") or "A").strip().upper()
    value = (data.get("value") or "").strip()
    ttl = int(data.get("ttl") or 60)
    owner_node = (data.get("owner_node") or "").strip()

    if not name or not value or not owner_node:
        return _err("Missing name/value/owner_node")
    if len(name) > 255 or len(value) > 2048:
        return _err("Record too large")
    if rtype not in ("A", "AAAA", "CNAME", "TXT", "SRV"):
        return _err("Invalid rtype")

    ttl = max(10, min(86400, ttl))

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute(
            "INSERT INTO net2_dns(name,rtype,value,ttl,owner_node,updated_ts) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET rtype=excluded.rtype,value=excluded.value,ttl=excluded.ttl,owner_node=excluded.owner_node,updated_ts=excluded.updated_ts",
            (name, rtype, value, ttl, owner_node, _now()),
        )
        con.commit()
        return _ok(saved=True, name=name, rtype=rtype)
    except Exception as e:
        return _err("DB error", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

@bp2.get("/api/net2/dns/resolve")
def net2_dns_resolve():
    if not _require_injected():
        return _err("Storage not configured (META_DB).", 500)

    name = (request.args.get("name") or "").strip().lower()
    if not name:
        return _err("Missing name")

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute("SELECT name,rtype,value,ttl,owner_node,updated_ts FROM net2_dns WHERE name=? LIMIT 1", (name,))
        row = cur.fetchone()
        if not row:
            return _ok(found=False, name=name)
        return _ok(found=True, record={
            "name": row[0],
            "rtype": row[1],
            "value": row[2],
            "ttl": int(row[3] or 60),
            "owner_node": row[4],
            "updated_ts": row[5],
        })
    except Exception as e:
        return _err("DB error", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

# ------------------ Overlay Tunnel Sessions (control-plane) ------------------

@bp2.post("/api/net2/tunnel/issue")
def net2_tunnel_issue():
    """
    Issue an overlay tunnel session artifact (NOT OS VPN).
    Body:
      { "from_node":"..", "to_node":"..", "policy": {...}, "ttl_sec": 600 }
    """
    raw = _body_bytes()
    if not _verify_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Storage not configured (META_DB).", 500)

    data = _j()
    from_node = (data.get("from_node") or "").strip()
    to_node = (data.get("to_node") or "").strip()
    policy = data.get("policy") or {}
    ttl_sec = int(data.get("ttl_sec") or 600)

    if not from_node or not to_node:
        return _err("Missing from_node/to_node")
    ttl_sec = max(60, min(86400, ttl_sec))

    try:
        policy_json = json.dumps(policy, ensure_ascii=False)
    except Exception:
        policy_json = "{}"

    sid = _new_id("tun")
    now = _now()
    expires = now + ttl_sec

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute(
            "INSERT INTO net2_tunnel_sessions(session_id,from_node,to_node,created_ts,expires_ts,policy_json,status) "
            "VALUES(?,?,?,?,?,?,?)",
            (sid, from_node, to_node, now, expires, policy_json, "active"),
        )
        con.commit()
        return _ok(session_id=sid, from_node=from_node, to_node=to_node, expires_ts=expires)
    except Exception as e:
        return _err("DB error", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

@bp2.get("/api/net2/tunnel/status")
def net2_tunnel_status():
    if not _require_injected():
        return _err("Storage not configured (META_DB).", 500)
    session_id = (request.args.get("session_id") or "").strip()
    if not session_id:
        return _err("Missing session_id")

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute(
            "SELECT session_id,from_node,to_node,created_ts,expires_ts,policy_json,status "
            "FROM net2_tunnel_sessions WHERE session_id=? LIMIT 1",
            (session_id,),
        )
        row = cur.fetchone()
        if not row:
            return _ok(found=False, session_id=session_id)
        try:
            policy = json.loads(row[5] or "{}")
        except Exception:
            policy = {}
        return _ok(found=True, session={
            "session_id": row[0],
            "from_node": row[1],
            "to_node": row[2],
            "created_ts": row[3],
            "expires_ts": row[4],
            "policy": policy,
            "status": row[6],
        })
    except Exception as e:
        return _err("DB error", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

# ---------------------------------------------------------------------
# init_app (called by app.py ONCE)
# ---------------------------------------------------------------------
def init_app(app, connect_sqlite, meta_db_path: str, api_key_auth_ok=None, sign_ok=None) -> None:
    global _CONNECT_SQLITE, _META_DB, _API_KEY_AUTH_OK, _SIGN_OK
    _CONNECT_SQLITE = connect_sqlite
    _META_DB = meta_db_path
    _API_KEY_AUTH_OK = api_key_auth_ok
    _SIGN_OK = sign_ok

    # Prevent double-register
    if "appnet2_v800" in getattr(app, "blueprints", {}):
        return

    _ensure_tables()
    app.register_blueprint(bp2)