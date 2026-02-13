# --==The SarahMemory Project==--
# File: /app/server/appnet.py
# Purpose: SarahNet / MCP "one-way broker" endpoints (store-and-forward + signaling)
# Part of the SarahMemory Companion AI-bot Platform
# Author: © 2025 Brian Lee Baros. All Rights Reserved.
# www.linkedin.com/in/brian-baros-29962a176
# https://www.facebook.com/bbaros
# brian.baros@sarahmemory.com
# 'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
# https://www.sarahmemory.com
# https://api.sarahmemory.com
# https://ai.sarahmemory.com

# Purpose: SarahNet / MCP "one-way broker" endpoints (store-and-forward + signaling)
# Design goals:
# - NO duplicate endpoints with app.py (avoids Flask AssertionError collisions)
# - Everything is namespaced under /api/net/*
# - Broker stores presence/messages/commands/groups/signals/files
# - Broker does NOT execute commands; nodes verify/auth/execute locally
#
# IMPORTANT:
# - This file FIXES your current bug: you had DUPLICATE Flask routes:
#     /api/net/file/send
#     /api/net/file/poll
#     /api/net/file/ack
#   defined TWICE (one “small file” version and one “chunked” version).
#   That WILL cause endpoint collisions / undefined behavior.
#
# WHAT THIS VERSION DOES:
# - Keeps your working “small broker file” API (Test3 compatible):
#     POST /api/net/file/send
#     GET  /api/net/file/poll
#     POST /api/net/file/ack
# - Adds CHUNKED transfers without collisions using NEW endpoints:
#     POST /api/net/file/start
#     POST /api/net/file/chunk
#     GET  /api/net/file/chunk/poll
#     POST /api/net/file/chunk/ack
#     POST /api/net/file/finish
# - Adds CRC32 + SHA256 per chunk, optional zlib compression, resume-friendly polling.

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import uuid
import zlib
from typing import Any, Callable, Dict, Optional, Tuple

from flask import Blueprint, jsonify, request

# Use a stable blueprint name so app.register_blueprint() won't collide.
bp = Blueprint("appnet_v800", __name__)

# Injected by app.py at init_app time
_CONNECT_SQLITE: Optional[Callable[..., Any]] = None
_META_DB: Optional[str] = None
_API_KEY_AUTH_OK: Optional[Callable[[], bool]] = None
_SIGN_OK: Optional[Callable[[bytes, str], bool]] = None

# Safety limits
MAX_BROKER_FILE_BYTES = int(os.environ.get("SARAHNET_MAX_BROKER_FILE_BYTES", "2000000"))  # 2 MB default (single-shot)
MAX_GROUP_MEMBERS = int(os.environ.get("SARAHNET_MAX_GROUP_MEMBERS", "200"))

# File-offer security controls
REQUIRE_FILE_ACCEPT = os.environ.get("SARAHNET_REQUIRE_FILE_ACCEPT", "true").strip().lower() in ("1","true","yes","on")
# Optional server-side "disk dump" for audit/debug (disabled unless enabled AND key matches)
SERVER_DUMP_ENABLED = os.environ.get("SARAHNET_SERVER_DUMP_ENABLED", "false").strip().lower() in ("1","true","yes","on")
SERVER_DUMP_KEY = os.environ.get("SARAHNET_SERVER_DUMP_KEY", "").strip()  # shared secret for debug dumping
SERVER_DUMP_SUBDIR = os.environ.get("SARAHNET_SERVER_DUMP_SUBDIR", "broker_downloads").strip()

# Optional broker-side AV scan (best-effort; typically off on PA unless you install tools)
BROKER_AV_SCAN = os.environ.get("SARAHNET_BROKER_AV_SCAN", "false").strip().lower() in ("1","true","yes","on")
BROKER_AV_TOOL = os.environ.get("SARAHNET_BROKER_AV_TOOL", "clamscan").strip()

# Client hint (nodes can ignore / override); local SarahMemory uses SarahMemoryGlobals.DOWNLOADS_DIR
SUGGESTED_CLIENT_DOWNLOAD_DIR = os.environ.get("SARAHNET_SUGGESTED_CLIENT_DOWNLOAD_DIR", "C:\\SarahMemory\\downloads")


# Chunked transfer limits
MAX_CHUNK_BYTES = int(os.environ.get("SARAHNET_MAX_CHUNK_BYTES", str(512 * 1024)))  # 512KB default
MAX_TRANSFER_BYTES = int(os.environ.get("SARAHNET_MAX_TRANSFER_BYTES", str(25 * 1024 * 1024)))  # 25MB default
MAX_POLL_CHUNKS = int(os.environ.get("SARAHNET_MAX_POLL_CHUNKS", "10"))  # per poll


# ----------------------------- helpers ---------------------------------

def _now() -> float:
    return time.time()


def _sanitize_filename(name: str) -> str:
    # Prevent path traversal / weird chars
    name = (name or "").replace("\\", "/").split("/")[-1]
    name = name.strip().replace("..", "_")
    if not name:
        return "file.bin"
    return name[:255]


def _broker_dump_root() -> Optional[str]:
    """Best-effort: where broker writes debug/audit copies of inbound files."""
    try:
        if not _META_DB:
            return None
        base = os.path.dirname(_META_DB)
        # Prefer a 'network' subdir next to meta.db if present, else use base directly.
        candidate = os.path.join(base, "network", SERVER_DUMP_SUBDIR)
        if os.path.isdir(os.path.join(base, "network")) or "network" in candidate:
            return candidate
        return os.path.join(base, SERVER_DUMP_SUBDIR)
    except Exception:
        return None


def _maybe_dump_file_bytes(to_node: str, file_id: str, filename: str, blob: bytes) -> Tuple[Optional[str], Optional[str]]:
    """Write a copy to disk ONLY when SERVER_DUMP_ENABLED + correct key header is present."""
    if not SERVER_DUMP_ENABLED:
        return (None, "disabled")
    try:
        key = (request.headers.get("X-SarahNet-Dump-Key") or "").strip()
        if SERVER_DUMP_KEY and key != SERVER_DUMP_KEY:
            return (None, "bad_key")
        root = _broker_dump_root()
        if not root:
            return (None, "no_root")
        safe_node = (to_node or "unknown").replace("/", "_")[:128]
        os.makedirs(os.path.join(root, safe_node), exist_ok=True)
        safe_name = _sanitize_filename(filename)
        out_path = os.path.join(root, safe_node, f"{file_id}__{safe_name}")
        with open(out_path, "wb") as f:
            f.write(blob)
        return (out_path, None)
    except Exception as e:
        return (None, f"error:{e}")


def _scan_blob_best_effort(blob: bytes, filename: str) -> Tuple[str, str]:
    """Broker-side AV scan hook. Defaults to unscanned unless BROKER_AV_SCAN is enabled."""
    if not BROKER_AV_SCAN:
        return ("unscanned", "broker_av_disabled")
    # Best-effort: try clamscan if available. If not, return error.
    try:
        import subprocess
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix="__" + _sanitize_filename(filename)) as tf:
            tf.write(blob)
            tmp_path = tf.name
        try:
            res = subprocess.run([BROKER_AV_TOOL, "--no-summary", tmp_path], capture_output=True, text=True, timeout=30)
            out = (res.stdout or "") + (res.stderr or "")
            # clamscan returns 0 = clean, 1 = infected
            if res.returncode == 0:
                return ("clean", out.strip()[:500])
            if res.returncode == 1:
                return ("infected", out.strip()[:500])
            return ("error", out.strip()[:500] or f"returncode={res.returncode}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    except Exception as e:
        return ("error", str(e)[:500])


def _j() -> Dict[str, Any]:
    return request.get_json(silent=True) or {}


def _ok(**kw) -> Tuple[Any, int]:
    return jsonify({"ok": True, **kw}), 200


def _err(msg: str, code: int = 400, **kw) -> Tuple[Any, int]:
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


def _as_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"), validate=False)


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _crc32_int(b: bytes) -> int:
    return int(zlib.crc32(b) & 0xFFFFFFFF)


def _verify_broker_auth(body_bytes: bytes) -> bool:
    """
    Accept either:
      - X-Sarah-Signature verified by injected _SIGN_OK(body, sig)
      - API key verified by injected _API_KEY_AUTH_OK()
    If nothing injected, allow (dev mode). In production inject auth.
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

    # Dev fallback
    return True


def _table_has_column(cur, table: str, col: str) -> bool:
    try:
        cur.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in (cur.fetchall() or [])]
        return col in cols
    except Exception:
        return False


def _ensure_tables() -> None:
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

        # Direct messages (1:1)
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

        # Command envelopes (NOT executed here)
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

        # Groups (team chat / meetings)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net_groups (
                group_id TEXT PRIMARY KEY,
                owner_node TEXT,
                created_ts REAL,
                meta_json TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net_group_members (
                group_id TEXT,
                node_id TEXT,
                role TEXT,
                joined_ts REAL,
                PRIMARY KEY (group_id, node_id)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_net_group_members_node ON net_group_members(node_id)")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS net_group_messages (
                id TEXT PRIMARY KEY,
                group_id TEXT,
                from_node TEXT,
                ts REAL,
                kind TEXT,
                body_json TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_net_group_messages_gid_ts ON net_group_messages(group_id, ts)")

        # WebRTC signaling (broker only stores offer/answer/ice)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net_signals (
                id TEXT PRIMARY KEY,
                to_node TEXT,
                from_node TEXT,
                ts REAL,
                signal_type TEXT,
                payload_json TEXT,
                delivered INTEGER DEFAULT 0,
                delivered_ts REAL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_net_signals_to_delivered ON net_signals(to_node, delivered, ts)")
        # -----------------------------------------------------------------
        # Privacy / safety controls (spam protection, DND/away/invisible, blocks)
        # -----------------------------------------------------------------
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net_privacy (
                node_id TEXT PRIMARY KEY,
                allow_messages INTEGER DEFAULT 1,
                allow_calls INTEGER DEFAULT 1,
                allow_files INTEGER DEFAULT 1,
                require_file_accept INTEGER DEFAULT 1,
                auto_accept_calls INTEGER DEFAULT 0,
                invisible INTEGER DEFAULT 0,
                status TEXT DEFAULT 'online',
                status_msg TEXT DEFAULT '',
                updated_ts REAL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS net_blocks (
                node_id TEXT,
                blocked_node TEXT,
                kind TEXT DEFAULT 'all',
                created_ts REAL,
                PRIMARY KEY (node_id, blocked_node, kind)
            )
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_net_blocks_node ON net_blocks(node_id)")


        # SMALL brokered attachments (single-shot)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net_files (
                id TEXT PRIMARY KEY,
                to_node TEXT,
                from_node TEXT,
                ts REAL,
                filename TEXT,
                mime TEXT,
                size_bytes INTEGER,
                sha256 TEXT,
                data_b64 TEXT,

                -- security / user-consent gate
                status TEXT DEFAULT 'offered',      -- offered|accepted|rejected|delivered
                accepted INTEGER DEFAULT 0,
                rejected INTEGER DEFAULT 0,
                decision_ts REAL,
                save_to TEXT,                        -- user-chosen save directory/path (client hint)

                -- optional broker-side scan
                scan_status TEXT,                    -- unscanned|clean|infected|error
                scan_detail TEXT,

                -- optional broker disk dump (debug / audit)
                server_saved_path TEXT,
                server_saved_ts REAL,

                delivered INTEGER DEFAULT 0,
                delivered_ts REAL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_net_files_to_delivered ON net_files(to_node, delivered, ts)")

        # CHUNKED transfers (headers)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net_file_transfers (
                transfer_id TEXT PRIMARY KEY,
                from_node TEXT,
                to_node TEXT,
                filename TEXT,
                mime TEXT,
                total_chunks INTEGER,
                chunk_bytes INTEGER,
                total_bytes INTEGER,
                created_ts REAL,
                file_sha256 TEXT,
                status TEXT
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_net_file_transfers_to ON net_file_transfers(to_node, created_ts)")

        # CHUNKED transfers (chunks)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS net_file_chunks (
                transfer_id TEXT,
                chunk_index INTEGER,
                chunk_b64 TEXT,
                compression TEXT,
                crc32 INTEGER,
                chunk_sha256 TEXT,
                size_bytes INTEGER,
                delivered INTEGER DEFAULT 0,
                created_ts REAL,
                PRIMARY KEY (transfer_id, chunk_index)
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_net_file_chunks_delivered ON net_file_chunks(transfer_id, delivered, chunk_index)")

        # Lightweight migrations if older table exists without new columns
        if not _table_has_column(cur, "net_file_chunks", "compression"):
            try:
                cur.execute("ALTER TABLE net_file_chunks ADD COLUMN compression TEXT")
            except Exception:
                pass
        if not _table_has_column(cur, "net_file_transfers", "chunk_bytes"):
            try:
                cur.execute("ALTER TABLE net_file_transfers ADD COLUMN chunk_bytes INTEGER")
            except Exception:
                pass
        if not _table_has_column(cur, "net_file_transfers", "total_bytes"):
            try:
                cur.execute("ALTER TABLE net_file_transfers ADD COLUMN total_bytes INTEGER")
            except Exception:
                pass


        # net_files: security + accept/reject workflow columns (safe file offers)
        if not _table_has_column(cur, "net_files", "status"):
            try:
                cur.execute("ALTER TABLE net_files ADD COLUMN status TEXT")
            except Exception:
                pass
        if not _table_has_column(cur, "net_files", "accepted"):
            try:
                cur.execute("ALTER TABLE net_files ADD COLUMN accepted INTEGER DEFAULT 0")
            except Exception:
                pass
        if not _table_has_column(cur, "net_files", "rejected"):
            try:
                cur.execute("ALTER TABLE net_files ADD COLUMN rejected INTEGER DEFAULT 0")
            except Exception:
                pass
        if not _table_has_column(cur, "net_files", "decision_ts"):
            try:
                cur.execute("ALTER TABLE net_files ADD COLUMN decision_ts REAL")
            except Exception:
                pass
        if not _table_has_column(cur, "net_files", "save_to"):
            try:
                cur.execute("ALTER TABLE net_files ADD COLUMN save_to TEXT")
            except Exception:
                pass
        if not _table_has_column(cur, "net_files", "scan_status"):
            try:
                cur.execute("ALTER TABLE net_files ADD COLUMN scan_status TEXT")
            except Exception:
                pass
        if not _table_has_column(cur, "net_files", "scan_detail"):
            try:
                cur.execute("ALTER TABLE net_files ADD COLUMN scan_detail TEXT")
            except Exception:
                pass
        if not _table_has_column(cur, "net_files", "server_saved_path"):
            try:
                cur.execute("ALTER TABLE net_files ADD COLUMN server_saved_path TEXT")
            except Exception:
                pass
        if not _table_has_column(cur, "net_files", "server_saved_ts"):
            try:
                cur.execute("ALTER TABLE net_files ADD COLUMN server_saved_ts REAL")
            except Exception:
                pass

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


# ----------------------------- rate limiting + privacy -----------------------------
# Rate limiting state (in-memory, process-local)
from collections import defaultdict, deque
_RATE_LIMITS = defaultdict(lambda: deque(maxlen=200))


def _check_rate_limit(key: str, max_per_min: int = 30) -> bool:
    """Simple rolling-window limiter."""
    now = time.time()
    q = _RATE_LIMITS[key]
    while q and q[0] < now - 60:
        q.popleft()
    if len(q) >= max_per_min:
        return False
    q.append(now)
    return True


def _privacy_allows(cur, to_node: str, from_node: str, kind: str) -> tuple[bool, str]:
    """Enforce net_privacy + net_blocks. Returns (ok, reason_code)."""
    try:
        # Blocks (either specific kind or all)
        cur.execute(
            "SELECT 1 FROM net_blocks WHERE node_id=? AND blocked_node=? AND (kind='all' OR kind=?) LIMIT 1",
            (to_node, from_node, kind),
        )
        if cur.fetchone():
            return False, "blocked"
    except Exception:
        pass

    try:
        cur.execute(
            "SELECT allow_messages, allow_calls, allow_files, require_file_accept, invisible FROM net_privacy WHERE node_id=?",
            (to_node,),
        )
        row = cur.fetchone()
        if not row:
            # default allow
            return True, "ok"
        allow_messages = int(row[0] or 0)
        allow_calls = int(row[1] or 0)
        allow_files = int(row[2] or 0)

        if kind in ("message", "chat", "text"):
            return (allow_messages == 1), ("messages_off" if allow_messages != 1 else "ok")
        if kind in ("call", "signal", "webrtc"):
            return (allow_calls == 1), ("calls_off" if allow_calls != 1 else "ok")
        if kind in ("file", "attachment"):
            return (allow_files == 1), ("files_off" if allow_files != 1 else "ok")
        return True, "ok"
    except Exception:
        return True, "ok"


# ---------------------------------------------------------------------
# Basic broker ping
# ---------------------------------------------------------------------
@bp.get("/api/net/ping")
def net_ping():
    return _ok(
        pong=True,
        ts=_now(),
        broker="api.sarahmemory.com",
        version=os.environ.get("PROJECT_VERSION", "8.0.0"),
    )


# ---------------------------------------------------------------------
# Rendezvous (presence)
# ---------------------------------------------------------------------
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
            try:
                meta = json.loads(row[2] or "{}")
            except Exception:
                meta = {}
            return _ok(found=True, node_id=row[0], last_ts=row[1], meta=meta)

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


# ---------------------------------------------------------------------
# 1:1 Messages (store-and-forward)
# ---------------------------------------------------------------------
@bp.post("/api/net/message/send")
def net_message_send():
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

    # Rate limit per sender
    if not _check_rate_limit(f"msg:{from_node}", max_per_min=30):
        return _err("Rate limit exceeded", 429, error_code="rate_limited")


    # Privacy / block enforcement (best-effort)
    try:
        con2 = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur2 = con2.cursor()
        ok_priv, reason = _privacy_allows(cur2, to_node, from_node, "message")
        con2.close()
        if not ok_priv:
            return _err("Recipient blocked or unavailable", 403, error_code=reason)
    except Exception:
        pass


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


# ---------------------------------------------------------------------
# Commands (envelopes only; broker never executes)
# ---------------------------------------------------------------------
@bp.post("/api/net/command/submit")
def net_command_submit():
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
# Group / Team Chat (store-and-forward)
# ---------------------------------------------------------------------
@bp.post("/api/net/group/create")
def net_group_create():
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    data = _j()
    owner = (data.get("owner_node") or "").strip()
    meta = data.get("meta") or {}
    if not owner:
        return _err("Missing owner_node")

    try:
        meta_json = json.dumps(meta, ensure_ascii=False)
    except Exception:
        meta_json = "{}"

    gid = _new_id("grp")
    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute("INSERT INTO net_groups(group_id,owner_node,created_ts,meta_json) VALUES(?,?,?,?)", (gid, owner, _now(), meta_json))
        cur.execute("INSERT INTO net_group_members(group_id,node_id,role,joined_ts) VALUES(?,?,?,?)", (gid, owner, "owner", _now()))
        con.commit()
        return _ok(group_id=gid, created=True)
    except Exception as e:
        return _err("DB error creating group", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@bp.post("/api/net/group/join")
def net_group_join():
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    data = _j()
    gid = (data.get("group_id") or "").strip()
    node = (data.get("node_id") or "").strip()
    role = (data.get("role") or "member").strip()[:32]
    if not gid or not node:
        return _err("Missing group_id/node_id")

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()

        cur.execute("SELECT COUNT(*) FROM net_group_members WHERE group_id=?", (gid,))
        cnt = int((cur.fetchone() or [0])[0] or 0)
        if cnt >= MAX_GROUP_MEMBERS:
            return _err("Group is full", 400)

        cur.execute(
            "INSERT INTO net_group_members(group_id,node_id,role,joined_ts) VALUES(?,?,?,?) "
            "ON CONFLICT(group_id,node_id) DO UPDATE SET role=excluded.role",
            (gid, node, role, _now()),
        )
        con.commit()
        return _ok(group_id=gid, joined=True)
    except Exception as e:
        return _err("DB error joining group", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@bp.post("/api/net/group/leave")
def net_group_leave():
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    data = _j()
    gid = (data.get("group_id") or "").strip()
    node = (data.get("node_id") or "").strip()
    if not gid or not node:
        return _err("Missing group_id/node_id")

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute("DELETE FROM net_group_members WHERE group_id=? AND node_id=?", (gid, node))
        con.commit()
        return _ok(group_id=gid, left=True)
    except Exception as e:
        return _err("DB error leaving group", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@bp.post("/api/net/group/message/send")
def net_group_message_send():
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    data = _j()
    gid = (data.get("group_id") or "").strip()
    from_node = (data.get("from_node") or "").strip()
    kind = (data.get("kind") or "message").strip()
    body = data.get("body") or {}
    if not gid or not from_node:
        return _err("Missing group_id/from_node")

    try:
        body_json = json.dumps(body, ensure_ascii=False)
    except Exception:
        body_json = "{}"

    mid = _new_id("gmsg")
    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()

        cur.execute("SELECT 1 FROM net_group_members WHERE group_id=? AND node_id=?", (gid, from_node))
        if not cur.fetchone():
            return _err("Not a member of this group", 403)

        cur.execute(
            "INSERT INTO net_group_messages(id,group_id,from_node,ts,kind,body_json) VALUES(?,?,?,?,?,?)",
            (mid, gid, from_node, _now(), kind, body_json),
        )
        con.commit()
        return _ok(id=mid, queued=True)
    except Exception as e:
        return _err("DB error queueing group message", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@bp.get("/api/net/group/message/poll")
def net_group_message_poll():
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    gid = (request.args.get("group_id") or "").strip()
    node = (request.args.get("node_id") or "").strip()
    since = request.args.get("since") or "0"
    limit = request.args.get("limit") or "50"
    if not gid or not node:
        return _err("Missing group_id/node_id")

    try:
        since_ts = float(since)
    except Exception:
        since_ts = 0.0
    try:
        lim = max(1, min(500, int(limit)))
    except Exception:
        lim = 50

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()

        cur.execute("SELECT 1 FROM net_group_members WHERE group_id=? AND node_id=?", (gid, node))
        if not cur.fetchone():
            return _err("Not a member of this group", 403)

        cur.execute(
            "SELECT id,from_node,ts,kind,body_json FROM net_group_messages "
            "WHERE group_id=? AND ts>? ORDER BY ts ASC LIMIT ?",
            (gid, since_ts, lim),
        )
        rows = cur.fetchall() or []
        out = []
        for r in rows:
            try:
                out.append({"id": r[0], "from_node": r[1], "ts": r[2], "kind": r[3], "body": json.loads(r[4] or "{}")})
            except Exception:
                out.append({"id": r[0], "from_node": r[1], "ts": r[2], "kind": r[3], "body": {}})
        last_ts = out[-1]["ts"] if out else since_ts
        return _ok(messages=out, count=len(out), last_ts=last_ts)
    except Exception as e:
        return _err("DB error polling group messages", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# WebRTC Signaling (for calls + large file P2P data channel)
# ---------------------------------------------------------------------
@bp.post("/api/net/signal/send")
def net_signal_send():
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    data = _j()
    to_node = (data.get("to_node") or "").strip()
    from_node = (data.get("from_node") or "").strip()
    signal_type = (data.get("signal_type") or "").strip() or "signal"
    payload = data.get("payload") or {}

    if not to_node or not from_node:
        return _err("Missing to_node/from_node")

    try:
        payload_json = json.dumps(payload, ensure_ascii=False)
    except Exception:
        payload_json = "{}"

    sid = _new_id("sig")
    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute(
            "INSERT INTO net_signals(id,to_node,from_node,ts,signal_type,payload_json,delivered) VALUES(?,?,?,?,?,?,0)",
            (sid, to_node, from_node, _now(), signal_type, payload_json),
        )
        con.commit()
        return _ok(id=sid, queued=True)
    except Exception as e:
        return _err("DB error queueing signal", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@bp.get("/api/net/signal/poll")
def net_signal_poll():
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    to_node = (request.args.get("to_node") or "").strip()
    if not to_node:
        return _err("Missing to_node")

    limit = request.args.get("limit") or "50"
    try:
        lim = max(1, min(500, int(limit)))
    except Exception:
        lim = 50

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute(
            "SELECT id,from_node,ts,signal_type,payload_json FROM net_signals "
            "WHERE to_node=? AND delivered=0 ORDER BY ts ASC LIMIT ?",
            (to_node, lim),
        )
        rows = cur.fetchall() or []
        out = []
        for r in rows:
            try:
                out.append({"id": r[0], "from_node": r[1], "ts": r[2], "signal_type": r[3], "payload": json.loads(r[4] or "{}")})
            except Exception:
                out.append({"id": r[0], "from_node": r[1], "ts": r[2], "signal_type": r[3], "payload": {}})
        return _ok(signals=out, count=len(out))
    except Exception as e:
        return _err("DB error polling signals", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@bp.post("/api/net/signal/ack")
def net_signal_ack():
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    data = _j()
    to_node = (data.get("to_node") or "").strip()
    sig_id = (data.get("id") or "").strip()
    if not to_node or not sig_id:
        return _err("Missing to_node/id")

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute("UPDATE net_signals SET delivered=1, delivered_ts=? WHERE id=? AND to_node=?", (_now(), sig_id, to_node))
        con.commit()
        return _ok(acked=True, id=sig_id)
    except Exception as e:
        return _err("DB error acking signal", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# SMALL brokered file attachments (Test3 compatible)
# ---------------------------------------------------------------------
@bp.post("/api/net/file/send")
def net_file_send_small():
    """
    Broker stores SMALL attachments only (<= MAX_BROKER_FILE_BYTES).
    Payload format:
      {
        "to_node": "...",
        "from_node": "...",
        "filename": "x.png",
        "mime": "image/png",
        "data_b64": "...",
        "sha256": "optional"
      }
    Nodes should encrypt before sending if you want true privacy.
    """
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    data = _j()
    to_node = (data.get("to_node") or "").strip()
    from_node = (data.get("from_node") or "").strip()
    filename = (data.get("filename") or "").strip()[:255]
    mime = (data.get("mime") or "application/octet-stream").strip()[:128]
    data_b64 = (data.get("data_b64") or "").strip()
    sha256_in = (data.get("sha256") or "").strip()

    if not to_node or not from_node or not filename or not data_b64:
        return _err("Missing to_node/from_node/filename/data_b64")

    # Rate limit per sender
    if not _check_rate_limit(f"file:{from_node}", max_per_min=10):
        return _err("Rate limit exceeded", 429, error_code="rate_limited")

    # Privacy / block enforcement (best-effort)
    try:
        con2 = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur2 = con2.cursor()
        ok_priv, reason = _privacy_allows(cur2, to_node, from_node, "file")
        con2.close()
        if not ok_priv:
            return _err("Recipient blocked or unavailable", 403, error_code=reason)
    except Exception:
        pass


    try:
        blob = base64.b64decode(data_b64.encode("utf-8"), validate=True)
    except Exception:
        return _err("Invalid base64")

    size_bytes = len(blob)
    if size_bytes <= 0 or size_bytes > MAX_BROKER_FILE_BYTES:
        return _err(f"File too large for brokered transfer (max {MAX_BROKER_FILE_BYTES} bytes)", 413)

    sha256 = hashlib.sha256(blob).hexdigest()
    if sha256_in and sha256_in.lower() != sha256.lower():
        return _err("sha256 mismatch", 400)

    # Security: broker can optionally scan inbound bytes; client still must scan on download.
    scan_status, scan_detail = _scan_blob_best_effort(blob, filename)

    fid = _new_id("file")
    # Optional: broker disk dump for audit/debug (requires SERVER_DUMP_ENABLED and correct X-SarahNet-Dump-Key)
    server_saved_path = None
    server_saved_ts = None
    if SERVER_DUMP_ENABLED:
        pth, note = _maybe_dump_file_bytes(to_node, fid, filename, blob)
        if pth:
            server_saved_path = pth
            server_saved_ts = _now()

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute(
            "INSERT INTO net_files(id,to_node,from_node,ts,filename,mime,size_bytes,sha256,data_b64,"
            "status,accepted,rejected,decision_ts,save_to,scan_status,scan_detail,server_saved_path,server_saved_ts,delivered) "
            "VALUES(?,?,?,?,?,?,?,?,?, ?,0,0,NULL,NULL,?,?,?, ?,0)",
            (
                fid, to_node, from_node, _now(),
                filename, mime, size_bytes, sha256, data_b64,
                "offered",
                scan_status, scan_detail,
                server_saved_path, server_saved_ts,
            ),
        )

        con.commit()
        return _ok(id=fid, queued=True, size_bytes=size_bytes, sha256=sha256, requires_accept=bool(REQUIRE_FILE_ACCEPT), scan_status=scan_status)
    except Exception as e:
        return _err("DB error queueing file", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@bp.get("/api/net/file/poll")
def net_file_poll_small():
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    to_node = (request.args.get("to_node") or "").strip()
    if not to_node:
        return _err("Missing to_node")

    limit = request.args.get("limit") or "10"
    try:
        lim = max(1, min(50, int(limit)))
    except Exception:
        lim = 10

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute(
            "SELECT id,from_node,ts,filename,mime,size_bytes,sha256,data_b64,"
            "status,accepted,rejected,decision_ts,save_to,scan_status,scan_detail "
            "FROM net_files "
            "WHERE to_node=? AND delivered=0 AND rejected=0 ORDER BY ts ASC LIMIT ?",
            (to_node, lim),
        )
        rows = cur.fetchall() or []

        # If REQUIRE_FILE_ACCEPT is enabled, we ONLY include file bytes after the receiver has accepted it.
        include_data = (request.args.get("include_data") or "").strip().lower() in ("1","true","yes","on")
        if not REQUIRE_FILE_ACCEPT:
            include_data = True

        out = []
        for r in rows:
            file_id = r[0]
            accepted = int(r[9] or 0)
            payload = {
                "id": file_id,
                "from_node": r[1],
                "ts": r[2],
                "filename": r[3],
                "mime": r[4],
                "size_bytes": r[5],
                "sha256": r[6],
                "status": r[8] or "offered",
                "accepted": accepted,
                "rejected": int(r[10] or 0),
                "decision_ts": r[11],
                "save_to": r[12],
                "scan_status": r[13],
                "scan_detail": r[14],
                "requires_accept": bool(REQUIRE_FILE_ACCEPT),
                "suggested_download_dir": SUGGESTED_CLIENT_DOWNLOAD_DIR,
            }
            if include_data and (not REQUIRE_FILE_ACCEPT or accepted == 1):
                payload["data_b64"] = r[7]
            else:
                payload["data_b64"] = None
            out.append(payload)
        return _ok(files=out, count=len(out))
    except Exception as e:
        return _err("DB error polling files", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@bp.post("/api/net/file/decision")
def net_file_decision():
    """
    Receiver-side consent gate:
      - decision=accept => broker will allow bytes to be returned via /api/net/file/poll?include_data=1
      - decision=reject => broker will never return bytes; sender still gets an ack of offer delivery
    Body JSON:
      { "to_node": "...", "id": "file_...", "decision": "accept|reject", "save_to": "optional path hint" }
    """
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)

    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except Exception:
        payload = {}

    to_node = (payload.get("to_node") or "").strip()
    file_id = (payload.get("id") or "").strip()
    decision = (payload.get("decision") or "").strip().lower()
    save_to = (payload.get("save_to") or "").strip() or None

    if not to_node or not file_id:
        return _err("Missing to_node or id", 400)
    if decision not in ("accept", "reject"):
        return _err("Invalid decision (accept|reject)", 400)

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()

        # Ensure the file exists and belongs to the receiver
        cur.execute("SELECT id, accepted, rejected FROM net_files WHERE id=? AND to_node=? LIMIT 1", (file_id, to_node))
        row = cur.fetchone()
        if not row:
            return _err("File not found for this to_node", 404)

        now_ts = _now()
        if decision == "accept":
            cur.execute(
                "UPDATE net_files SET accepted=1, rejected=0, status='accepted', decision_ts=?, save_to=? "
                "WHERE id=? AND to_node=?",
                (now_ts, save_to, file_id, to_node),
            )
        else:
            cur.execute(
                "UPDATE net_files SET rejected=1, accepted=0, status='rejected', decision_ts=?, save_to=? "
                "WHERE id=? AND to_node=?",
                (now_ts, save_to, file_id, to_node),
            )

        con.commit()
        return _ok(id=file_id, decision=decision, to_node=to_node, save_to=save_to)
    except Exception as e:
        return _err("DB error saving decision", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass



@bp.post("/api/net/file/ack")
def net_file_ack_small():
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    data = _j()
    to_node = (data.get("to_node") or "").strip()
    fid = (data.get("id") or "").strip()
    if not to_node or not fid:
        return _err("Missing to_node/id")

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute("UPDATE net_files SET delivered=1, delivered_ts=?, status='delivered' WHERE id=? AND to_node=?", (_now(), fid, to_node))
        con.commit()
        return _ok(acked=True, id=fid)
    except Exception as e:
        return _err("DB error acking file", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# CHUNKED file transfers (fast/resumable + CRC32)
# ---------------------------------------------------------------------
@bp.post("/api/net/file/start")
def net_file_start():
    """
    Start a chunked transfer.
    Payload:
      {
        "from_node": "...",
        "to_node": "...",
        "filename": "...",
        "mime": "...",
        "total_chunks": 123,
        "chunk_bytes": 262144,
        "total_bytes": 999999,
        "file_sha256": "optional"
      }
    """
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    data = _j()
    from_node = (data.get("from_node") or "").strip()
    to_node = (data.get("to_node") or "").strip()
    filename = (data.get("filename") or "").strip()[:255] or "file.bin"
    mime = (data.get("mime") or "application/octet-stream").strip()[:128]
    total_chunks = _as_int(data.get("total_chunks"), 0)
    chunk_bytes = _as_int(data.get("chunk_bytes"), 0)
    total_bytes = _as_int(data.get("total_bytes"), 0)
    file_sha256 = (data.get("file_sha256") or "").strip() or None

    if not from_node or not to_node:
        return _err("Missing from_node/to_node")
    if total_chunks <= 0:
        return _err("Missing/invalid total_chunks")
    if chunk_bytes <= 0 or chunk_bytes > MAX_CHUNK_BYTES:
        return _err(f"Invalid chunk_bytes (max {MAX_CHUNK_BYTES})")
    if total_bytes <= 0 or total_bytes > MAX_TRANSFER_BYTES:
        return _err(f"Invalid total_bytes (max {MAX_TRANSFER_BYTES})")

    transfer_id = _new_id("tx")

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute("""
            INSERT INTO net_file_transfers
              (transfer_id, from_node, to_node, filename, mime, total_chunks, chunk_bytes, total_bytes, created_ts, file_sha256, status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            transfer_id, from_node, to_node, filename, mime,
            total_chunks, chunk_bytes, total_bytes, _now(),
            file_sha256, "active"
        ))
        con.commit()
        return _ok(transfer_id=transfer_id, started=True, chunk_bytes=chunk_bytes)
    except Exception as e:
        return _err("DB error starting transfer", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@bp.post("/api/net/file/chunk")
def net_file_chunk_put():
    """
    Upload a single chunk.
    Payload:
      {
        "transfer_id": "tx_...",
        "from_node": "...",
        "to_node": "...",
        "chunk_index": 0,
        "chunk_b64": "...",
        "compression": "" | "zlib"   (meaning chunk_b64 bytes are compressed)
      }
    Response includes crc32 + chunk_sha256 of *decompressed* bytes.
    """
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    data = _j()
    transfer_id = (data.get("transfer_id") or "").strip()
    from_node = (data.get("from_node") or "").strip()
    to_node = (data.get("to_node") or "").strip()
    chunk_index = _as_int(data.get("chunk_index"), -1)
    chunk_b64 = (data.get("chunk_b64") or "").strip()
    compression = (data.get("compression") or "").strip().lower()

    if not transfer_id:
        return _err("Missing transfer_id")
    if not from_node or not to_node:
        return _err("Missing from_node/to_node")
    if chunk_index < 0:
        return _err("Missing/invalid chunk_index")
    if not chunk_b64:
        return _err("Missing chunk_b64")
    if compression not in ("", "zlib"):
        return _err("Invalid compression (use '' or 'zlib')")

    try:
        wire_bytes = _b64d(chunk_b64)
        plain = zlib.decompress(wire_bytes) if compression == "zlib" else wire_bytes
    except Exception as e:
        return _err("Bad chunk encoding", 400, detail=str(e))

    if len(plain) <= 0 or len(plain) > MAX_CHUNK_BYTES:
        return _err(f"Chunk too large (max {MAX_CHUNK_BYTES} bytes)", 400)

    crc32 = _crc32_int(plain)
    chunk_sha256 = _sha256_hex(plain)

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()

        # Validate transfer exists + node pairing matches (prevents cross-talk)
        cur.execute("""
            SELECT from_node, to_node, total_chunks, status
            FROM net_file_transfers
            WHERE transfer_id=?
            LIMIT 1
        """, (transfer_id,))
        row = cur.fetchone()
        if not row:
            return _err("Unknown transfer_id", 404)
        if (row[0] or "") != from_node or (row[1] or "") != to_node:
            return _err("transfer_id node mismatch", 403)
        if (row[3] or "") not in ("active", "sealed"):
            return _err("transfer not active", 409)

        total_chunks = _as_int(row[2], 0)
        if total_chunks > 0 and chunk_index >= total_chunks:
            return _err("chunk_index out of range", 400)

        # Store exactly what we received (chunk_b64 + compression)
        cur.execute("""
            INSERT INTO net_file_chunks
              (transfer_id, chunk_index, chunk_b64, compression, crc32, chunk_sha256, size_bytes, delivered, created_ts)
            VALUES (?,?,?,?,?,?,?,0,?)
            ON CONFLICT(transfer_id, chunk_index) DO UPDATE SET
              chunk_b64=excluded.chunk_b64,
              compression=excluded.compression,
              crc32=excluded.crc32,
              chunk_sha256=excluded.chunk_sha256,
              size_bytes=excluded.size_bytes
        """, (transfer_id, chunk_index, chunk_b64, compression, crc32, chunk_sha256, len(plain), _now()))

        con.commit()
        return _ok(
            transfer_id=transfer_id,
            chunk_index=chunk_index,
            saved=True,
            crc32=crc32,
            chunk_sha256=chunk_sha256,
            size_bytes=len(plain),
        )
    except Exception as e:
        return _err("DB error saving file chunk", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@bp.get("/api/net/file/chunk/poll")
def net_file_chunk_poll():
    """
    Receiver polls for chunks.
    Query:
      ?to_node=BeachLaptopNode02&transfer_id=tx_...&max_chunks=1
    If transfer_id omitted, broker picks the oldest transfer with undelivered chunks.
    """
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    to_node = (request.args.get("to_node") or "").strip()
    transfer_id = (request.args.get("transfer_id") or "").strip()
    max_chunks = _as_int(request.args.get("max_chunks"), 1)
    if not to_node:
        return _err("Missing to_node")
    if max_chunks < 1:
        max_chunks = 1
    if max_chunks > MAX_POLL_CHUNKS:
        max_chunks = MAX_POLL_CHUNKS

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()

        if transfer_id:
            cur.execute("""
                SELECT transfer_id, from_node, to_node, filename, mime, total_chunks, chunk_bytes, total_bytes, file_sha256, status
                FROM net_file_transfers
                WHERE transfer_id=? AND to_node=?
                LIMIT 1
            """, (transfer_id, to_node))
        else:
            # Find the oldest transfer for this to_node that still has undelivered chunks
            cur.execute("""
                SELECT t.transfer_id, t.from_node, t.to_node, t.filename, t.mime, t.total_chunks, t.chunk_bytes, t.total_bytes, t.file_sha256, t.status
                FROM net_file_transfers t
                WHERE t.to_node=?
                ORDER BY t.created_ts ASC
                LIMIT 1
            """, (to_node,))

        header = cur.fetchone()
        if not header:
            return _ok(found=False, chunks=[])

        tx_id, from_node, to_node2, filename, mime, total_chunks, chunk_bytes, total_bytes, file_sha256, status = header

        cur.execute("""
            SELECT chunk_index, chunk_b64, compression, crc32, chunk_sha256, size_bytes
            FROM net_file_chunks
            WHERE transfer_id=? AND delivered=0
            ORDER BY chunk_index ASC
            LIMIT ?
        """, (tx_id, max_chunks))

        chunks = []
        for (idx, b64, comp, crc, sh, sz) in (cur.fetchall() or []):
            chunks.append({
                "chunk_index": idx,
                "chunk_b64": b64,
                "compression": (comp or ""),
                "crc32": crc,
                "chunk_sha256": sh,
                "size_bytes": sz,
            })

        return _ok(
            found=True,
            transfer_id=tx_id,
            from_node=from_node,
            to_node=to_node2,
            filename=filename,
            mime=mime,
            total_chunks=total_chunks,
            chunk_bytes=chunk_bytes,
            total_bytes=total_bytes,
            file_sha256=file_sha256,
            status=status,
            chunks=chunks,
        )
    except Exception as e:
        return _err("DB error polling file chunks", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@bp.post("/api/net/file/chunk/ack")
def net_file_chunk_ack():
    """
    Receiver acks a chunk after verifying CRC/SHA locally.
    Payload:
      { "transfer_id":"tx_...", "chunk_index": 0 }
    """
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    data = _j()
    transfer_id = (data.get("transfer_id") or "").strip()
    chunk_index = _as_int(data.get("chunk_index"), -1)
    if not transfer_id:
        return _err("Missing transfer_id")
    if chunk_index < 0:
        return _err("Missing/invalid chunk_index")

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()

        cur.execute("""
            UPDATE net_file_chunks
            SET delivered=1
            WHERE transfer_id=? AND chunk_index=?
        """, (transfer_id, chunk_index))

        # completion check
        cur.execute("SELECT total_chunks FROM net_file_transfers WHERE transfer_id=? LIMIT 1", (transfer_id,))
        r = cur.fetchone()
        total = int(r[0]) if r and r[0] else 0

        cur.execute("SELECT COUNT(*) FROM net_file_chunks WHERE transfer_id=? AND delivered=1", (transfer_id,))
        delivered = int((cur.fetchone() or [0])[0])

        if total > 0 and delivered >= total:
            cur.execute("UPDATE net_file_transfers SET status='complete' WHERE transfer_id=?", (transfer_id,))

        con.commit()
        return _ok(acked=True, transfer_id=transfer_id, chunk_index=chunk_index, delivered=delivered, total=total)
    except Exception as e:
        return _err("DB error acking file chunk", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


@bp.post("/api/net/file/finish")
def net_file_finish():
    """
    Sender seals a transfer to indicate “all chunks uploaded”.
    Receiver still verifies final SHA256 locally (recommended).
    Payload: { "transfer_id":"tx_..." }
    """
    raw = _body_bytes()
    if not _verify_broker_auth(raw):
        return _err("Unauthorized", 401)
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500)

    data = _j()
    transfer_id = (data.get("transfer_id") or "").strip()
    if not transfer_id:
        return _err("Missing transfer_id")

    _ensure_tables()
    con = None
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute("UPDATE net_file_transfers SET status='sealed' WHERE transfer_id=? AND status='active'", (transfer_id,))
        con.commit()
        return _ok(transfer_id=transfer_id, sealed=True)
    except Exception as e:
        return _err("DB error sealing transfer", 500, detail=str(e))
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


# ---------------------------------------------------------------------
# init_app (called by app.py ONCE)
# ---------------------------------------------------------------------


@bp.get("/api/net/diagnostics")
def net_diagnostics():
    """Unified broker diagnostics for WebUI."""
    if not _require_injected():
        return _err("Broker storage not configured (META_DB).", 500, error_code="no_storage")

    results = {"ok": True, "ts": _now(), "tests": []}

    # Test 1: DB connectivity
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM net_presence")
        count = int(cur.fetchone()[0])
        con.close()
        results["tests"].append({"name": "db_connectivity", "status": "pass", "detail": f"{count} nodes in presence"})
    except Exception as e:
        results["ok"] = False
        results["tests"].append({"name": "db_connectivity", "status": "fail", "detail": str(e)})

    # Test 2: Required tables
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in (cur.fetchall() or [])]
        con.close()
        required = ["net_presence", "net_messages", "net_files", "net_file_transfers", "net_privacy", "net_blocks"]
        missing = [t for t in required if t not in tables]
        if missing:
            results["tests"].append({"name": "table_schema", "status": "warn", "detail": f"Missing tables: {missing}"})
        else:
            results["tests"].append({"name": "table_schema", "status": "pass", "detail": f"{len(tables)} tables present"})
    except Exception as e:
        results["ok"] = False
        results["tests"].append({"name": "table_schema", "status": "fail", "detail": str(e)})

    # Test 3: Active transfers
    try:
        con = _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM net_file_transfers WHERE status='active'")
        active = int(cur.fetchone()[0])
        con.close()
        results["tests"].append({"name": "file_transfers", "status": "pass", "detail": f"{active} active"})
    except Exception as e:
        results["ok"] = False
        results["tests"].append({"name": "file_transfers", "status": "fail", "detail": str(e)})

    return jsonify(results), 200
def init_app(app, connect_sqlite, meta_db_path: str, api_key_auth_ok=None, sign_ok=None) -> None:
    """
    app.py should call:
        import appnet
        appnet.init_app(app, CONNECT_SQLITE, META_DB, api_key_auth_ok=..., sign_ok=...)
    """
    global _CONNECT_SQLITE, _META_DB, _API_KEY_AUTH_OK, _SIGN_OK
    _CONNECT_SQLITE = connect_sqlite
    _META_DB = meta_db_path
    _API_KEY_AUTH_OK = api_key_auth_ok
    _SIGN_OK = sign_ok

    # Prevent double-register
    if "appnet_v800" in getattr(app, "blueprints", {}):
        return

    _ensure_tables()
    app.register_blueprint(bp)
