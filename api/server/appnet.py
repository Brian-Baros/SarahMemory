# --==The SarahMemory Project==--
# File: /app/server/appnet.py
# Purpose: SarahNet / MCP "one-way broker" endpoints (store-and-forward)
# Part of the SarahMemory Companion AI-bot Platform
# Author: © 2025 Brian Lee Baros. All Rights Reserved.
# www.linkedin.com/in/brian-baros-29962a176
# https://www.facebook.com/bbaros
# brian.baros@sarahmemory.com
# 'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
# https://www.sarahmemory.com
# https://api.sarahmemory.com
# https://ai.sarahmemory.com
# Design goals:
# - NO duplicate endpoints with app.py (avoids Flask AssertionError collisions)
# - Everything is namespaced under /api/net/*
# - Uses Blueprint registration with "register once" guards
# - Broker queues messages/commands; does NOT execute them

from __future__ import annotations

import json
import os
import time
import uuid
import hashlib
from typing import Any, Callable, Optional, Dict, Tuple

from flask import Blueprint, request, jsonify

# ---------------------------------------------------------------------
# Blueprint (kept isolated so app.py can stay stable)
# ---------------------------------------------------------------------
bp = Blueprint("appnet_v800", __name__)

# ---------------------------------------------------------------------
# Helpers (injected by app.py at init_app time)
# ---------------------------------------------------------------------
_CONNECT_SQLITE: Optional[Callable[..., Any]] = None
_META_DB: Optional[str] = None
_API_KEY_AUTH_OK: Optional[Callable[[], bool]] = None
_SIGN_OK: Optional[Callable[[bytes, str], bool]] = None

def _now() -> float:
    return time.time()

def _j() -> Dict[str, Any]:
    return request.get_json(silent=True) or {}

def _ok(**kw) -> Tuple[Any, int]:
    payload = {"ok": True, **kw}
    return jsonify(payload), 200

def _err(msg: str, code: int = 400, **kw) -> Tuple[Any, int]:
    payload = {"ok": False, "error": msg, **kw}
    return jsonify(payload), code

def _require_injected() -> bool:
    return bool(_CONNECT_SQLITE and _META_DB)

def _ensure_tables() -> None:
    """
    Creates minimal broker tables in META_DB.
    Uses the existing SQLite connection helper from app.py.
    """
    if not _require_injected():
        return
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()

        # Presence / rendezvous
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net_presence (
                node_id TEXT PRIMARY KEY,
                last_ts REAL,
                meta_json TEXT
            )
        """)

        # Store-and-forward messages
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net_messages (
                id TEXT PRIMARY KEY,
                to_node TEXT,
                from_node TEXT,
                ts REAL,
                kind TEXT,
                body_json TEXT,
                delivered INTEGER DEFAULT 0,
                delivered_ts REAL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_net_messages_to_delivered ON net_messages(to_node, delivered, ts)")

        # Store-and-forward command envelopes (NOT executed here)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net_commands (
                id TEXT PRIMARY KEY,
                to_node TEXT,
                from_node TEXT,
                ts REAL,
                command_type TEXT,
                envelope_json TEXT,
                delivered INTEGER DEFAULT 0,
                delivered_ts REAL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_net_commands_to_delivered ON net_commands(to_node, delivered, ts)")

        # Receipts / acks
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net_receipts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL,
                node_id TEXT,
                ref_id TEXT,
                ref_kind TEXT,
                status TEXT,
                note TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_net_receipts_node_ts ON net_receipts(node_id, ts)")

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

def _verify_broker_auth(body_bytes: bytes) -> bool:
    """
    Accept either:
    - API key auth (admin-ish)
    - Signed requests (hub signature) for node-to-broker traffic
    If neither is configured, app.py may allow open access; you can tighten later.
    """
    # Prefer signature if provided
    sig = (request.headers.get("X-Sarah-Signature") or "").strip()
    if sig and _SIGN_OK:
        try:
            return bool(_SIGN_OK(body_bytes, sig))
        except Exception:
            return False

    # Fall back to API key guard if available
    if _API_KEY_AUTH_OK:
        try:
            return bool(_API_KEY_AUTH_OK())
        except Exception:
            return False

    # If nothing injected, allow (dev). In production you SHOULD inject auth.
    return True

def _body_bytes() -> bytes:
    try:
        return request.get_data(cache=False) or b""
    except Exception:
        return b""

def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"

# ---------------------------------------------------------------------
# /api/net/* endpoints (NO /api/health, NO duplicates)
# ---------------------------------------------------------------------

@bp.get("/api/net/ping")
def net_ping():
    return _ok(pong=True, ts=_now(), broker="api.sarahmemory.com", version=os.environ.get("PROJECT_VERSION", "v8"))

@bp.post("/api/net/rendezvous/announce")
def net_rendezvous_announce():
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)

    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    data = _j()
    node_id = (data.get("node_id") or "").strip()
    if not node_id:
        return _err("Missing node_id")

    meta = data.get("meta") or {}
    try:
        meta_json = json.dumps(meta, ensure_ascii=False)
    except Exception:
        meta_json = "{}"

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute(
            "INSERT INTO net_presence(node_id,last_ts,meta_json) VALUES(?,?,?) "
            "ON CONFLICT(node_id) DO UPDATE SET last_ts=excluded.last_ts, meta_json=excluded.meta_json",
            (node_id, _now(), meta_json),
        )
        con.commit()
        return _ok(node_id=node_id, saved=True)
    except Exception as e:
        return _err("DB error announcing presence", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

@bp.get("/api/net/rendezvous/lookup")
def net_rendezvous_lookup():
    """
    Lookup presence data for a node (or list recent nodes if node_id omitted).
    """
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    node_id = (request.args.get("node_id") or "").strip()
    limit = request.args.get("limit") or "50"
    try:
        lim = max(1, min(200, int(limit)))
    except Exception:
        lim = 50

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()

        if node_id:
            cur.execute("SELECT node_id,last_ts,meta_json FROM net_presence WHERE node_id=?", (node_id,))
            row = cur.fetchone()
            if not row:
                return _ok(found=False, node_id=node_id)
            return _ok(found=True, node_id=row[0], last_ts=row[1], meta=json.loads(row[2] or "{}"))
        else:
            cur.execute("SELECT node_id,last_ts,meta_json FROM net_presence ORDER BY last_ts DESC LIMIT ?", (lim,))
            rows = cur.fetchall() or []
            out = []
            for r in rows:
                try:
                    out.append({"node_id": r[0], "last_ts": r[1], "meta": json.loads(r[2] or "{}")})
                except Exception:
                    out.append({"node_id": r[0], "last_ts": r[1], "meta": {}})
            return _ok(nodes=out, count=len(out))
    except Exception as e:
        return _err("DB error", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

@bp.post("/api/net/message/send")
def net_message_send():
    """
    Store-and-forward message:
    - from_node -> to_node
    - broker stores it
    - target polls /fetch
    """
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)

    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    data = _j()
    to_node = (data.get("to_node") or "").strip()
    from_node = (data.get("from_node") or "").strip()
    kind = (data.get("kind") or "message").strip()
    body = data.get("body") or {}

    if not to_node or not from_node:
        return _err("Missing to_node/from_node")

    try:
        body_json = json.dumps(body, ensure_ascii=False)
    except Exception:
        body_json = "{}"

    _ensure_tables()
    msg_id = _new_id("msg")
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute(
            "INSERT INTO net_messages(id,to_node,from_node,ts,kind,body_json,delivered) VALUES(?,?,?,?,?,?,0)",
            (msg_id, to_node, from_node, _now(), kind, body_json),
        )
        con.commit()
        return _ok(id=msg_id, queued=True)
    except Exception as e:
        return _err("DB error queueing message", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

@bp.get("/api/net/message/poll")
def net_message_poll():
    """
    Target node polls for undelivered messages.
    """
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    to_node = (request.args.get("to_node") or "").strip()
    if not to_node:
        return _err("Missing to_node")

    limit = request.args.get("limit") or "25"
    try:
        lim = max(1, min(200, int(limit)))
    except Exception:
        lim = 25

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute(
            "SELECT id,from_node,ts,kind,body_json FROM net_messages "
            "WHERE to_node=? AND delivered=0 ORDER BY ts ASC LIMIT ?",
            (to_node, lim),
        )
        rows = cur.fetchall() or []
        out = []
        for r in rows:
            try:
                out.append({"id": r[0], "from_node": r[1], "ts": r[2], "kind": r[3], "body": json.loads(r[4] or "{}")})
            except Exception:
                out.append({"id": r[0], "from_node": r[1], "ts": r[2], "kind": r[3], "body": {}})
        return _ok(messages=out, count=len(out))
    except Exception as e:
        return _err("DB error polling messages", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

@bp.post("/api/net/message/ack")
def net_message_ack():
    """
    Target acknowledges delivery so broker marks message delivered.
    """
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)

    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    data = _j()
    to_node = (data.get("to_node") or "").strip()
    msg_id = (data.get("id") or "").strip()
    if not to_node or not msg_id:
        return _err("Missing to_node/id")

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute(
            "UPDATE net_messages SET delivered=1, delivered_ts=? WHERE id=? AND to_node=?",
            (_now(), msg_id, to_node),
        )
        con.commit()
        return _ok(acked=True, id=msg_id)
    except Exception as e:
        return _err("DB error acking message", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

@bp.post("/api/net/command/submit")
def net_command_submit():
    """
    Store-and-forward "command envelope".
    Broker does NOT execute.
    Envelope should include:
      - command_type (string)
      - payload (dict)
      - signatures / nonce / ts (node-side enforced)
    """
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)

    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    data = _j()
    to_node = (data.get("to_node") or "").strip()
    from_node = (data.get("from_node") or "").strip()
    command_type = (data.get("command_type") or "").strip() or "command"
    envelope = data.get("envelope") or {}

    if not to_node or not from_node:
        return _err("Missing to_node/from_node")

    try:
        env_json = json.dumps(envelope, ensure_ascii=False)
    except Exception:
        env_json = "{}"

    _ensure_tables()
    cmd_id = _new_id("cmd")
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute(
            "INSERT INTO net_commands(id,to_node,from_node,ts,command_type,envelope_json,delivered) VALUES(?,?,?,?,?,?,0)",
            (cmd_id, to_node, from_node, _now(), command_type, env_json),
        )
        con.commit()
        return _ok(id=cmd_id, queued=True)
    except Exception as e:
        return _err("DB error queueing command", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

@bp.get("/api/net/command/poll")
def net_command_poll():
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    to_node = (request.args.get("to_node") or "").strip()
    if not to_node:
        return _err("Missing to_node")

    limit = request.args.get("limit") or "25"
    try:
        lim = max(1, min(200, int(limit)))
    except Exception:
        lim = 25

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute(
            "SELECT id,from_node,ts,command_type,envelope_json FROM net_commands "
            "WHERE to_node=? AND delivered=0 ORDER BY ts ASC LIMIT ?",
            (to_node, lim),
        )
        rows = cur.fetchall() or []
        out = []
        for r in rows:
            try:
                out.append({"id": r[0], "from_node": r[1], "ts": r[2], "command_type": r[3], "envelope": json.loads(r[4] or "{}")})
            except Exception:
                out.append({"id": r[0], "from_node": r[1], "ts": r[2], "command_type": r[3], "envelope": {}})
        return _ok(commands=out, count=len(out))
    except Exception as e:
        return _err("DB error polling commands", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

@bp.post("/api/net/command/ack")
def net_command_ack():
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)

    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    data = _j()
    to_node = (data.get("to_node") or "").strip()
    cmd_id = (data.get("id") or "").strip()
    status = (data.get("status") or "delivered").strip()
    note = (data.get("note") or "").strip()

    if not to_node or not cmd_id:
        return _err("Missing to_node/id")

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute(
            "UPDATE net_commands SET delivered=1, delivered_ts=? WHERE id=? AND to_node=?",
            (_now(), cmd_id, to_node),
        )
        cur.execute(
            "INSERT INTO net_receipts(ts,node_id,ref_id,ref_kind,status,note) VALUES(?,?,?,?,?,?)",
            (_now(), to_node, cmd_id, "command", status, note),
        )
        con.commit()
        return _ok(acked=True, id=cmd_id)
    except Exception as e:
        return _err("DB error acking command", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass

# ---------------------------------------------------------------------
# init_app: called by app.py ONCE to inject helpers and register blueprint
# ---------------------------------------------------------------------
def init_app(app, connect_sqlite, meta_db_path: str, api_key_auth_ok=None, sign_ok=None) -> None:
    global _CONNECT_SQLITE, _META_DB, _API_KEY_AUTH_OK, _SIGN_OK
    _CONNECT_SQLITE = connect_sqlite
    _META_DB = meta_db_path
    _API_KEY_AUTH_OK = api_key_auth_ok
    _SIGN_OK = sign_ok

    # Register once (prevents accidental double-import / reloader collisions)
    if "appnet_v800" in getattr(app, "blueprints", {}):
        return

    # Ensure tables exist early (safe)
    _ensure_tables()

    app.register_blueprint(bp)
# ====================================================================
# END OF appnet.py v8.0.0
# ====================================================================