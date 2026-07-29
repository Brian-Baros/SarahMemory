"""--==The SarahMemory Project==--
File: api/server/appnet2.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-06-06
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

Purpose: SarahNet "Bravo" — Identity/Trust + Overlay Tunnel Control + Virtual DNS/Name Directory
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
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-06-06"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
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

# ARILE boundary helper. API files emit compact variance signals only; the central
# backend engine remains SarahMemoryARILE.py.
try:
    from SarahMemoryARILE import arile_emit, arile_endpoint_guard
except Exception:  # pragma: no cover
    arile_emit = None  # type: ignore
    arile_endpoint_guard = None  # type: ignore

def _arile_api_emit(failure_type: str, summary: str, severity: float = 0.55, **data):
    try:
        if callable(arile_emit):
            arile_emit(source=f"api.server.{__name__}", kind="api_boundary_variance", failure_type=failure_type, severity=severity, confidence=0.85, risk="high" if severity >= 0.75 else "medium", summary=summary, requires_governance=severity >= 0.75, retention="security_audit" if severity >= 0.75 else "diagnostic", data=data)
    except Exception:
        pass

def _arile_check_request(endpoint_name: str, risk: str = "low") -> str:
    try:
        if callable(arile_endpoint_guard):
            return arile_endpoint_guard(endpoint_name, {"method": getattr(request, "method", ""), "content_length": getattr(request, "content_length", 0) or 0, "remote_addr": getattr(request, "remote_addr", "")}, risk=risk)
    except Exception:
        pass
    return "allow"


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

    # loopback fallback when no auth verifier is injected
    try:
        remote = str(getattr(request, "remote_addr", "") or "").strip().lower()
        return remote in ("", "127.0.0.1", "::1", "localhost")
    except Exception:
        return False

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

        # -----------------------------
        # Governed outbound AI-agent passport/task broker
        # Control-plane only: no agent process is launched by this table.
        # -----------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net2_agent_tasks (
                task_id TEXT PRIMARY KEY,
                passport_id TEXT UNIQUE,
                agent_id TEXT,
                destination_node TEXT,
                purpose TEXT,
                origin_lane TEXT,
                allowed_lanes_json TEXT,
                allowed_capabilities_json TEXT,
                allowed_resources_json TEXT,
                transport_json TEXT,
                status TEXT,
                created_ts REAL,
                departed_ts REAL,
                returned_ts REAL,
                reviewed_ts REAL,
                result_summary TEXT,
                result_payload_hash TEXT,
                capture_report_path TEXT,
                review_verdict TEXT,
                metadata_json TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_net2_agent_tasks_destination ON net2_agent_tasks(destination_node,status,created_ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_net2_agent_tasks_agent ON net2_agent_tasks(agent_id,status,created_ts)")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net2_agent_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                task_id TEXT,
                passport_id TEXT,
                agent_id TEXT,
                event_type TEXT,
                verdict TEXT,
                detail TEXT,
                payload_hash TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_net2_agent_events_task ON net2_agent_events(task_id,ts)")

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
        version=os.environ.get("PROJECT_VERSION", "9.0.0"),
        module="appnet2",
    )

@bp2.get("/api/net2/health")
def net2_health():
    payload = {
        "ok": True,
        "enabled": bool(_CONNECT_SQLITE and _META_DB),
        "ts": _now(),
        "version": os.environ.get("PROJECT_VERSION", "9.0.0"),
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
# Governed AI-agent passport/task control plane
# ---------------------------------------------------------------------
def _agent_modules() -> tuple:
    try:
        import SarahMemoryAgentFirewall as firewall  # type: ignore
    except Exception:
        firewall = None
    try:
        import SarahMemoryTrustRegistry as registry  # type: ignore
    except Exception:
        registry = None
    return firewall, registry


def _agent_event(task_id: str, passport_id: str, agent_id: str, event_type: str, verdict: str, detail: str = "", payload_hash: str = "") -> None:
    if not _require_injected():
        return
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        con.execute(
            "INSERT INTO net2_agent_events(ts,task_id,passport_id,agent_id,event_type,verdict,detail,payload_hash) VALUES(?,?,?,?,?,?,?,?)",
            (_now(), str(task_id)[:180], str(passport_id)[:180], str(agent_id)[:180], str(event_type)[:96], str(verdict)[:64], str(detail)[:2000], str(payload_hash)[:128]),
        )
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
    try:
        from SarahMemoryLedger import record_governance_receipt  # type: ignore
        record_governance_receipt(
            "sarahnet_agent",
            event_type,
            subject_id=agent_id,
            task_id=task_id,
            lane="sarahnet_agent",
            verdict=verdict,
            risk="high" if verdict in ("DENY", "BLOCKED") else "medium",
            retention_class="agent_passport",
            payload_hash=payload_hash,
            summary=detail or event_type,
            metadata={"passport_id": passport_id, "execution_authority": False},
        )
    except Exception:
        pass


def _confirmed(data: Dict[str, Any]) -> bool:
    value = data.get("confirmed", data.get("user_approved", data.get("confirmation", False)))
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes", "approved", "i approve", "confirm")


@bp2.post("/api/net2/agent/issue")
def net2_agent_issue():
    """Issue and queue a bounded outbound task. No process is launched here."""
    body_bytes = _body_bytes()
    if not _verify_auth(body_bytes):
        return _err("Authentication required", 401)
    data = _j()
    if not _confirmed(data):
        return _err("Explicit user approval required", 403, execution_authority=False)
    firewall, registry = _agent_modules()
    if firewall is None or not callable(getattr(firewall, "issue_outbound_agent_passport", None)):
        return _err("AgentFirewall passport issuer unavailable", 503)
    _ensure_tables()
    task_id = str(data.get("task_id") or _new_id("agenttask"))[:180]
    agent_id = str(data.get("agent_id") or "").strip()[:180]
    purpose = str(data.get("purpose") or data.get("task") or "").strip()[:4000]
    destination = str(data.get("destination_node") or "").strip()[:180]
    if not agent_id or not purpose or not destination:
        return _err("agent_id, purpose/task, and destination_node are required")
    origin_lane = str(data.get("origin_lane") or "research")[:96]
    allowed_lanes = data.get("allowed_lanes") if isinstance(data.get("allowed_lanes"), list) else [origin_lane]
    allowed_caps = data.get("allowed_capabilities") if isinstance(data.get("allowed_capabilities"), list) else ["research", "return_data"]
    allowed_res = data.get("allowed_resources") if isinstance(data.get("allowed_resources"), list) else []
    denied_res = data.get("denied_resources") if isinstance(data.get("denied_resources"), list) else ["core/*", ".env", "credentials", "shell", "device_control"]
    issued = firewall.issue_outbound_agent_passport(
        agent_id=agent_id,
        agent_name=str(data.get("agent_name") or agent_id),
        purpose=purpose,
        task_id=task_id,
        origin_lane=origin_lane,
        allowed_lanes=allowed_lanes,
        allowed_capabilities=allowed_caps,
        allowed_resources=allowed_res,
        denied_resources=denied_res,
        maximum_risk_tier=str(data.get("maximum_risk_tier") or "low"),
        ttl_seconds=int(data.get("ttl_seconds") or 3600),
        one_time_use=bool(data.get("one_time_use", True)),
        network_allowed=True,
        filesystem_allowed=bool(data.get("filesystem_allowed", False)),
        shell_allowed=False,
        device_allowed=False,
        memory_allowed=bool(data.get("memory_allowed", False)),
        user_approved=True,
        meta={"destination_node": destination, "source": "appnet2"},
    )
    if not issued.get("ok"):
        return _err("Passport issuance failed", 400, detail=str(issued.get("error") or "unknown"))
    passport = issued.get("passport") if isinstance(issued.get("passport"), dict) else {}
    creds = issued.get("departure_credentials") if isinstance(issued.get("departure_credentials"), dict) else {}
    passport_id = str(passport.get("passport_id") or creds.get("passport_id") or "")
    transport = {
        "schema": "SARAHMEMORY_AGENT_TASK_ENVELOPE_V1",
        "task_id": task_id,
        "passport_id": passport_id,
        "agent_id": agent_id,
        "purpose": purpose,
        "origin_lane": origin_lane,
        "allowed_lanes": allowed_lanes,
        "allowed_capabilities": allowed_caps,
        "allowed_resources": allowed_res,
        "credentials": creds,
        "execution_authority": False,
    }
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        con.execute(
            """INSERT INTO net2_agent_tasks(task_id,passport_id,agent_id,destination_node,purpose,origin_lane,
               allowed_lanes_json,allowed_capabilities_json,allowed_resources_json,transport_json,status,created_ts,metadata_json)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (task_id, passport_id, agent_id, destination, purpose, origin_lane, json.dumps(allowed_lanes), json.dumps(allowed_caps), json.dumps(allowed_res), json.dumps(transport, ensure_ascii=False), "issued", _now(), json.dumps({"issuer": "SarahMemory", "user_approved": True})),
        )
        con.commit()
    except Exception as exc:
        try:
            if con:
                con.rollback()
        except Exception:
            pass
        try:
            if registry and callable(getattr(registry, "revoke_agent_passport", None)):
                registry.revoke_agent_passport(passport_id, reason="broker_queue_failure", user_approved=True)
        except Exception:
            pass
        return _err("Agent task queue failure", 500, detail=str(exc))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass
    _agent_event(task_id, passport_id, agent_id, "PASSPORT_ISSUED", "ISSUED", "Governed outbound agent task queued.")
    return _ok(task_id=task_id, passport_id=passport_id, status="issued", destination_node=destination, passport=passport, warning="No agent process was launched. Call /api/net2/agent/depart after transport acceptance.", execution_authority=False)


@bp2.post("/api/net2/agent/depart")
def net2_agent_depart():
    body_bytes = _body_bytes()
    if not _verify_auth(body_bytes):
        return _err("Authentication required", 401)
    data = _j()
    if not _confirmed(data):
        return _err("Explicit user approval required", 403)
    task_id = str(data.get("task_id") or "").strip()
    passport_id = str(data.get("passport_id") or "").strip()
    if not task_id and not passport_id:
        return _err("task_id or passport_id required")
    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        con.row_factory = __import__("sqlite3").Row
        if task_id:
            row = con.execute("SELECT * FROM net2_agent_tasks WHERE task_id=? LIMIT 1", (task_id,)).fetchone()
        else:
            row = con.execute("SELECT * FROM net2_agent_tasks WHERE passport_id=? LIMIT 1", (passport_id,)).fetchone()
        if not row:
            return _err("Agent task not found", 404)
        rec = dict(row); task_id = rec["task_id"]; passport_id = rec["passport_id"]
        _, registry = _agent_modules()
        if registry is None:
            return _err("TrustRegistry unavailable", 503)
        marked = registry.mark_agent_departed(passport_id, transport_ref=str(data.get("transport_ref") or f"sarahnet:{rec.get('destination_node')}"), user_approved=True)
        if not marked.get("ok"):
            return _err("Passport departure rejected", 409, detail=str(marked.get("error")))
        con.execute("UPDATE net2_agent_tasks SET status='departed',departed_ts=? WHERE task_id=?", (_now(), task_id)); con.commit()
        _agent_event(task_id, passport_id, rec["agent_id"], "AGENT_DEPARTED", "DEPARTED", "Passport-bearing task released to governed transport.")
        return _ok(task_id=task_id, passport_id=passport_id, status="departed", destination_node=rec.get("destination_node"), execution_authority=False)
    except Exception as exc:
        return _err("Departure error", 500, detail=str(exc))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@bp2.get("/api/net2/agent/poll")
def net2_agent_poll():
    if not _verify_auth(b""):
        return _err("Authentication required", 401)
    destination = str(request.args.get("destination_node") or "").strip()
    limit = max(1, min(50, int(request.args.get("limit") or 10)))
    if not destination:
        return _err("destination_node required")
    _ensure_tables(); con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        rows = con.execute("SELECT task_id,passport_id,agent_id,purpose,origin_lane,transport_json,created_ts,departed_ts FROM net2_agent_tasks WHERE destination_node=? AND status='departed' ORDER BY departed_ts ASC LIMIT ?", (destination, limit)).fetchall()
        tasks = []
        for row in rows:
            try:
                envelope = json.loads(row[5] or "{}")
            except Exception:
                envelope = {}
            tasks.append({"task_id": row[0], "passport_id": row[1], "agent_id": row[2], "purpose": row[3], "origin_lane": row[4], "transport_envelope": envelope, "created_ts": row[6], "departed_ts": row[7], "execution_authority": False})
        return _ok(destination_node=destination, tasks=tasks, count=len(tasks))
    except Exception as exc:
        return _err("Poll error", 500, detail=str(exc))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@bp2.post("/api/net2/agent/return")
def net2_agent_return():
    body_bytes = _body_bytes()
    if not _verify_auth(body_bytes):
        return _err("Authentication required", 401)
    data = _j(); firewall, registry = _agent_modules()
    if firewall is None:
        return _err("AgentFirewall unavailable", 503)
    if registry is None:
        return _err("TrustRegistry unavailable", 503)
    task_id = str(data.get("task_id") or "").strip(); passport_id = str(data.get("passport_id") or "").strip(); agent_id = str(data.get("agent_id") or "").strip()
    if not task_id or not passport_id or not agent_id:
        return _err("task_id, passport_id, and agent_id required")

    # Validate the broker work-order identity before AgentFirewall records a
    # one-time return. This prevents a mismatched task envelope from consuming a
    # valid passport and keeps the Neuron Axis entirely outside the inbound lane.
    preflight_con = None
    try:
        preflight_con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        preflight_row = preflight_con.execute(
            "SELECT agent_id,passport_id,status FROM net2_agent_tasks WHERE task_id=? LIMIT 1",
            (task_id,),
        ).fetchone()
    except Exception as exc:
        return _err("Return identity preflight failed", 500, detail=str(exc))
    finally:
        try:
            if preflight_con:
                preflight_con.close()
        except Exception:
            pass
    if not preflight_row or str(preflight_row[0]) != agent_id or str(preflight_row[1]) != passport_id:
        _agent_event(task_id, passport_id, agent_id, "AGENT_RETURN_DENIED", "DENY", "broker_task_identity_mismatch")
        return _ok(
            task_id=task_id,
            passport_id=passport_id,
            status="quarantined",
            firewall_verdict={
                "verdict": "DENY",
                "reason": "broker_task_identity_mismatch",
                "containment_state": "QUARANTINED",
                "passport_verified": False,
            },
            execution_authority=False,
            note="Return identity did not match the outbound broker work order. No passport was consumed and no data reached Neuron.",
        )

    payload_hash = str(data.get("payload_hash") or _sha256_hex(json.dumps(data.get("result") or data.get("result_summary") or "", sort_keys=True, default=str).encode("utf-8", "ignore")))
    packet = {
        "headers": {
            "User-Agent": str(data.get("agent_name") or "SarahMemory outbound AI-agent return"),
            "X-SarahMemory-Agent-Id": agent_id,
            "X-SarahMemory-Passport-Id": passport_id,
            "X-SarahMemory-Agent-Signature": str(data.get("return_signature") or ""),
            "X-SarahMemory-Return-Nonce": str(data.get("return_nonce") or ""),
        },
        "json": {
            "agent_id": agent_id, "passport_id": passport_id, "task_id": task_id,
            "requested_lane": str(data.get("requested_lane") or "research"),
            "requested_capabilities": data.get("requested_capabilities") if isinstance(data.get("requested_capabilities"), list) else ["return_data"],
            "requested_resources": data.get("requested_resources") if isinstance(data.get("requested_resources"), list) else [],
            "risk_tier": str(data.get("risk_tier") or "low"), "payload_hash": payload_hash,
            "result_summary": str(data.get("result_summary") or "")[:4000],
        },
    }
    verdict = firewall.inspect_payload(packet, source="appnet2.agent_return", remote_addr=str(request.remote_addr or "agent-return"))
    containment = str(verdict.get("containment_state") or "QUARANTINED")
    status = "captured_review" if str(verdict.get("verdict")) == "REQUIRE_REVIEW" else "blocked" if containment == "BLOCKED" else "quarantined"
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        row = con.execute("SELECT agent_id,passport_id,status FROM net2_agent_tasks WHERE task_id=? LIMIT 1", (task_id,)).fetchone()
        if not row or str(row[0]) != agent_id or str(row[1]) != passport_id:
            status = "quarantined"
            verdict = {**verdict, "verdict": "DENY", "reason": "broker_task_identity_mismatch", "containment_state": "QUARANTINED"}
        con.execute("UPDATE net2_agent_tasks SET status=?,returned_ts=?,result_summary=?,result_payload_hash=?,capture_report_path=?,review_verdict=? WHERE task_id=?", (status, _now(), str(data.get("result_summary") or "")[:4000], payload_hash[:128], str(verdict.get("capture_report_path") or "")[:1000], str(verdict.get("verdict") or "")[:64], task_id)); con.commit()
    except Exception as exc:
        return _err("Return capture persistence failed", 500, detail=str(exc))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass
    _agent_event(task_id, passport_id, agent_id, "AGENT_RETURN_CAPTURED" if status == "captured_review" else "AGENT_RETURN_DENIED", str(verdict.get("verdict") or "DENY"), str(verdict.get("reason") or ""), payload_hash)
    return _ok(task_id=task_id, passport_id=passport_id, status=status, firewall_verdict={k: verdict.get(k) for k in ("verdict", "reason", "risk_tier", "containment_state", "capture_report_path", "passport_verified")}, execution_authority=False, note="Returned data was not executed or passed to Neuron. User-governed review is required.")


@bp2.get("/api/net2/agent/status")
def net2_agent_status():
    if not _verify_auth(b""):
        return _err("Authentication required", 401)
    task_id = str(request.args.get("task_id") or "").strip(); passport_id = str(request.args.get("passport_id") or "").strip()
    if not task_id and not passport_id:
        return _err("task_id or passport_id required")
    _ensure_tables(); con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        con.row_factory = __import__("sqlite3").Row
        row = con.execute("SELECT * FROM net2_agent_tasks WHERE task_id=? LIMIT 1", (task_id,)).fetchone() if task_id else con.execute("SELECT * FROM net2_agent_tasks WHERE passport_id=? LIMIT 1", (passport_id,)).fetchone()
        if not row:
            return _ok(found=False)
        rec = dict(row); rec.pop("transport_json", None); rec["execution_authority"] = False
        return _ok(found=True, task=rec)
    except Exception as exc:
        return _err("Status error", 500, detail=str(exc))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@bp2.post("/api/net2/agent/review")
def net2_agent_review():
    body_bytes = _body_bytes()
    if not _verify_auth(body_bytes):
        return _err("Authentication required", 401)
    data = _j()
    if not _confirmed(data):
        return _err("Explicit user approval required", 403)
    task_id = str(data.get("task_id") or "").strip(); decision = str(data.get("decision") or "reject").strip().lower()
    if decision not in ("accept", "reject"):
        return _err("decision must be accept or reject")
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        row = con.execute("SELECT passport_id,agent_id,status FROM net2_agent_tasks WHERE task_id=? LIMIT 1", (task_id,)).fetchone()
        if not row:
            return _err("Agent task not found", 404)
        passport_id, agent_id, current = row
        if current != "captured_review":
            return _err("Task is not awaiting review", 409, status=current)
        _, registry = _agent_modules()
        if registry is None:
            return _err("TrustRegistry unavailable", 503)
        if decision == "accept":
            closed = registry.consume_agent_passport(passport_id, user_approved=True, reason="agent_return_review_accepted")
            status = "reviewed_accepted"
        else:
            closed = registry.revoke_agent_passport(passport_id, reason=str(data.get("reason") or "agent_return_review_rejected"), user_approved=True)
            status = "reviewed_rejected"
        if not closed.get("ok"):
            return _err("Passport close failed", 409, detail=str(closed.get("error")))
        con.execute("UPDATE net2_agent_tasks SET status=?,reviewed_ts=?,review_verdict=? WHERE task_id=?", (status, _now(), decision.upper(), task_id)); con.commit()
        _agent_event(task_id, passport_id, agent_id, "AGENT_RETURN_REVIEWED", decision.upper(), str(data.get("reason") or "user_review"))
        return _ok(task_id=task_id, passport_id=passport_id, status=status, decision=decision, execution_authority=False, note="Acceptance records chain of custody only. It does not automatically execute or route the returned data.")
    except Exception as exc:
        return _err("Review error", 500, detail=str(exc))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

@bp2.get("/api/net2/governance")
def net2_governance():
    """Read-only SarahNet identity/trust governance probe.

    This reports the identity/trust contract without implying VPN, kernel,
    tunnel-driver, or OS-level network execution.
    """
    return _ok(
        api_domain="net2",
        route_base="/api/net2",
        governance={
            "identity_model": "node_identity_and_trust_probe",
            "os_vpn_execution": False,
            "kernel_tunnel_execution": False,
            "requires_auth_for_mutations": True,
            "supports": [
                "node_register",
                "challenge",
                "attestation",
                "dns_directory",
                "tunnel_session_metadata",
            ],
            "safety_notes": [
                "net2 is a control-plane API, not an OS VPN driver.",
                "Tunnel issue/status routes create governed metadata only unless a separate approved transport implements runtime data movement.",
            ],
        },
    )



@bp2.get("/api/net2/energetics/status")
def net2_energetics_status():
    # Status endpoint is read-only. It returns constitution lockout state if the
    # Energetics organ is missing or blocked so network trust surfaces cannot imply authority.
    ctx = {"source": "appnet2.status", "domain": "network_identity_trust"}
    try:
        import SarahMemoryEnergetics as _Energetics  # type: ignore
        fn = getattr(_Energetics, "get_energetics_status", None)
        if callable(fn):
            return jsonify(fn(ctx)), 200
    except Exception as exc:
        try:
            import SarahMemoryGlobals as _SMG  # type: ignore
            status_fn = getattr(_SMG, "sm_hazardous_energy_status", None)
            if callable(status_fn):
                return jsonify({"ok": True, "energetics_import_error": str(exc), "constitution": status_fn(ctx)}), 200
        except Exception:
            pass
        return _err("Energetics status error", 500, detail=str(exc))
    try:
        import SarahMemoryGlobals as _SMG  # type: ignore
        status_fn = getattr(_SMG, "sm_hazardous_energy_status", None)
        if callable(status_fn):
            return jsonify({"ok": True, "energetics_unavailable": True, "constitution": status_fn(ctx)}), 200
    except Exception:
        pass
    return _err("Energetics unavailable", 503)

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

# ====================================================================
# END OF appnet2.py v9.0.0
# ====================================================================
