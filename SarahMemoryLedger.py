"""--==The SarahMemory Project==--
File: SarahMemoryLedger.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2025-12-05
Time: 10:11:54
Author: © 2025 Brian Lee Baros. All Rights Reserved.
www.linkedin.com/in/brian-baros-29962a176
https://www.facebook.com/bbaros
brian.baros@sarahmemory.com
'The SarahMemory Companion AI-Bot Platform, are property of SOFTDEV0 LLC., & Brian Lee Baros'
https://www.sarahmemory.com
https://api.sarahmemory.com
https://ai.sarahmemory.com
===============================================================================
"""

from __future__ import annotations

import os
import json
import hmac
import hashlib
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, getcontext
from typing import Any, Dict, Optional, Tuple, List

from flask import Flask, request, jsonify

# ========================= Precision / Constants =========================
getcontext().prec = 50  # plenty of headroom
TOKEN_DECIMALS = 7
TOKEN_UNIT = Decimal("0.0000001")
MAX_SUPPLY = Decimal("1000000")  # 1,000,000
GENESIS_REWARD = Decimal("50")
BLOCK_MAX_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_BLOCKS = 400

# ========================= Directories =========================
try:
    import SarahMemoryGlobals as config
    BASE_DIR = getattr(config, "BASE_DIR", os.getcwd())
except Exception:
    BASE_DIR = os.getcwd()

API_ROOT = os.path.join(BASE_DIR, "public_html", "api")
if not os.path.isdir(API_ROOT):
    API_ROOT = os.path.join(BASE_DIR, "api")

WALLETS_DIR = os.path.join(API_ROOT, "wallets")
BLOCKS_DIR = os.path.join(API_ROOT, "block")
META_DIR = os.path.join(API_ROOT, "meta")
LOGS_DIR = os.path.join(BASE_DIR, "data", "logs")

for _d in (WALLETS_DIR, BLOCKS_DIR, META_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

MASTER_DB = os.path.join(META_DIR, "ledger_master.db")

# ========================= Flask =========================
app = Flask(__name__)
_lock = threading.RLock()

# ========================= SQLite Helpers =========================
def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

def _append_only_guards(conn: sqlite3.Connection) -> None:
    # Prevent UPDATE/DELETE on critical tables.
    guards = [
        ("tx", "UPDATE OF json_record ON tx BEGIN SELECT RAISE(ABORT,'append-only: tx update forbidden'); END;"),
        ("tx", "DELETE ON tx BEGIN SELECT RAISE(ABORT,'append-only: tx delete forbidden'); END;"),
    ]
    for tbl, body in guards:
        try:
            conn.execute(f"CREATE TRIGGER IF NOT EXISTS guard_{tbl}_upd {body}")
        except Exception:
            pass

def _init_master() -> None:
    with _connect(MASTER_DB) as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS supply (
            id INTEGER PRIMARY KEY CHECK (id=1),
            total_supply TEXT NOT NULL,
            issued TEXT NOT NULL,
            last_block INTEGER NOT NULL
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS knowledge_requests (
            id TEXT PRIMARY KEY,
            requester TEXT NOT NULL,
            provider TEXT NOT NULL,
            amount TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,          -- pending, fulfilled, expired, canceled
            response_proof TEXT,           -- arbitrary provider response hash/id
            fulfilled_at TEXT
        );
        """)
        cur = c.execute("SELECT COUNT(*) FROM supply WHERE id=1")
        if cur.fetchone()[0] == 0:
            c.execute(
                "INSERT INTO supply (id, total_supply, issued, last_block) VALUES (1, ?, ?, ?)",
                (str(MAX_SUPPLY), "0", 1),
            )
        c.commit()

_init_master()

# ========================= Block Files =========================
def _block_path(block_id: int) -> str:
    return os.path.join(BLOCKS_DIR, f"SarahCryptCoin{block_id:03d}.db")

def _ensure_block(block_id: int) -> None:
    path = _block_path(block_id)
    with _connect(path) as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS tx (
            id TEXT PRIMARY KEY,           -- tx_id
            json_record TEXT NOT NULL
        );
        """)
        _append_only_guards(c)
        c.commit()

def _current_block_id() -> int:
    with _connect(MASTER_DB) as c:
        row = c.execute("SELECT last_block FROM supply WHERE id=1").fetchone()
        return int(row[0])

def _rotate_block_if_needed() -> int:
    with _lock:
        block_id = _current_block_id()
        path = _block_path(block_id)
        _ensure_block(block_id)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        if size >= BLOCK_MAX_BYTES:
            if block_id >= MAX_BLOCKS:
                raise RuntimeError("Block limit reached; cannot rotate further.")
            block_id += 1
            with _connect(MASTER_DB) as c:
                c.execute("UPDATE supply SET last_block=? WHERE id=1", (block_id,))
                c.commit()
            _ensure_block(block_id)
        return block_id

# ========================= Wallets =========================
def _wallet_path(node_name: str) -> str:
    safe = "".join(ch for ch in node_name if ch.isalnum() or ch in ("-", "_"))
    return os.path.join(WALLETS_DIR, f"{safe}.wallet.db")

def _init_wallet(node_id: str, node_name: str) -> str:
    path = _wallet_path(node_name)
    with _connect(path) as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS wallet (
            id INTEGER PRIMARY KEY CHECK (id=1),
            node_id TEXT UNIQUE NOT NULL,
            node_name TEXT UNIQUE NOT NULL,
            balance TEXT NOT NULL,
            reputation INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS tx_index (
            tx_id TEXT PRIMARY KEY,
            block_id INTEGER NOT NULL
        );
        """)
        _append_only_guards(c)
        cur = c.execute("SELECT COUNT(*) FROM wallet WHERE id=1").fetchone()
        if cur[0] == 0:
            c.execute(
                "INSERT INTO wallet (id, node_id, node_name, balance, reputation, created_at) VALUES (1, ?, ?, ?, 0, ?)",
                (node_id, node_name, "0", datetime.now(timezone.utc).isoformat()),
            )
        c.commit()
    return path

def _get_wallet(node_name: str) -> Tuple[str, Dict[str, Any]]:
    path = _wallet_path(node_name)
    if not os.path.exists(path):
        raise FileNotFoundError("wallet not found")
    with _connect(path) as c:
        row = c.execute("SELECT node_id, node_name, balance, reputation, created_at FROM wallet WHERE id=1").fetchone()
        data = {
            "node_id": row[0],
            "node_name": row[1],
            "balance": str(Decimal(row[2])),
            "reputation": int(row[3]),
            "created_at": row[4],
        }
    return path, data

def _update_balance(path: str, new_balance: Decimal) -> None:
    with _connect(path) as c:
        c.execute("UPDATE wallet SET balance=? WHERE id=1", (str(new_balance),))
        c.commit()

def _delta_reputation(path: str, delta: int) -> None:
    with _connect(path) as c:
        c.execute("UPDATE wallet SET reputation=reputation+? WHERE id=1", (delta,))
        c.commit()

def _index_wallet_tx(path: str, tx_id: str, block_id: int) -> None:
    with _connect(path) as c:
        c.execute("INSERT OR IGNORE INTO tx_index (tx_id, block_id) VALUES (?, ?)", (tx_id, block_id))
        c.commit()

# ========================= Hashing / Proof =========================
def _hmac_key() -> Optional[bytes]:
    try:
        secret = getattr(config, "MESH_SHARED_SECRET", None)
        if secret:
            if isinstance(secret, str):
                secret = secret.encode("utf-8")
            return hashlib.sha256(secret).digest()
    except Exception:
        pass
    # deterministic fallback (local only): AUTHOR + VERSION
    try:
        author = getattr(config, "AUTHOR", "Brian Lee Baros")
        version = getattr(config, "PROJECT_VERSION", "7.7.1")
        seed = (author + version).encode("utf-8")
        return hashlib.sha256(seed).digest()
    except Exception:
        return None

def _tx_proof(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    key = _hmac_key()
    if key:
        return hmac.new(key, canonical, hashlib.sha256).hexdigest()
    return hashlib.sha256(canonical).hexdigest()

# ========================= Supply Accounting =========================
def _issue_tokens(to_node: str, amount: Decimal, reason: str) -> Dict[str, Any]:
    """Mint from treasury (used at registration). Enforces cap."""
    with _lock, _connect(MASTER_DB) as m:
        total_supply, issued, last_block = m.execute("SELECT total_supply, issued, last_block FROM supply WHERE id=1").fetchone()
        total_supply = Decimal(total_supply)
        issued = Decimal(issued)

        if issued + amount > total_supply:
            raise RuntimeError("Issuance exceeds total supply")

        # build tx
        tx = {
            "from": "Treasury",
            "to": to_node,
            "amount": float(amount.quantize(TOKEN_UNIT, rounding=ROUND_DOWN)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tx_id": uuid.uuid4().hex,
            "reason": reason,
        }
        tx["proof"] = _tx_proof(tx)

        # choose block
        block_id = _rotate_block_if_needed()
        _ensure_block(block_id)
        with _connect(_block_path(block_id)) as b:
            b.execute("INSERT INTO tx (id, json_record) VALUES (?, ?)", (tx["tx_id"], json.dumps(tx)))
            b.commit()

        # credit wallet
        wpath, wdata = _get_wallet(to_node)
        new_bal = Decimal(wdata["balance"]) + amount
        _update_balance(wpath, new_bal)
        _index_wallet_tx(wpath, tx["tx_id"], block_id)

        # update issued
        m.execute("UPDATE supply SET issued=? WHERE id=1", (str(issued + amount),))
        m.commit()
        return {"block_id": block_id, "tx": tx}

# ========================= Transaction Processing =========================
def _transfer(from_node: str, to_node: str, amount: Decimal, meta: Dict[str, Any]) -> Dict[str, Any]:
    if amount <= 0:
        raise ValueError("amount must be positive")
    amount = amount.quantize(TOKEN_UNIT, rounding=ROUND_DOWN)

    with _lock:
        # Load wallets
        fpath, fdata = _get_wallet(from_node)
        tpath, tdata = _get_wallet(to_node)

        fbal = Decimal(fdata["balance"])
        if fbal < amount:
            raise RuntimeError("insufficient funds")

        # Build tx
        payload = {
            "from": from_node,
            "to": to_node,
            "amount": float(amount),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tx_id": uuid.uuid4().hex,
            "meta": meta or {},
        }
        payload["proof"] = _tx_proof(payload)

        # Append to block
        block_id = _rotate_block_if_needed()
        with _connect(_block_path(block_id)) as b:
            b.execute("INSERT INTO tx (id, json_record) VALUES (?, ?)", (payload["tx_id"], json.dumps(payload)))
            b.commit()

        # Update balances
        _update_balance(fpath, fbal - amount)
        _update_balance(tpath, Decimal(tdata["balance"]) + amount)

        # Index in both wallets
        _index_wallet_tx(fpath, payload["tx_id"], block_id)
        _index_wallet_tx(tpath, payload["tx_id"], block_id)

        return {"block_id": block_id, "tx": payload}

# ========================= Reputation =========================
def _reward_reputation(node_name: str, delta: int) -> None:
    path, _ = _get_wallet(node_name)
    _delta_reputation(path, delta)

def _bonus_micro_tokens(node_name: str, base: Decimal = Decimal("0.00001")) -> Optional[Dict[str, Any]]:
    try:
        return _issue_tokens(node_name, base.quantize(TOKEN_UNIT, rounding=ROUND_DOWN), reason="reputation-bonus")
    except Exception:
        return None

# ========================= Public API =========================

app = Flask(__name__)

@app.post("/api/register-node")
def register_node():
    data = request.get_json(silent=True) or {}
    node_name = (data.get("node_name") or "").strip()
    version = (data.get("version") or "").strip()
    capabilities = data.get("capabilities") or []

    if not node_name:
        return jsonify({"error": "node_name required"}), 400

    node_id = uuid.uuid4().hex
    _init_wallet(node_id, node_name)

    # initial issuance
    try:
        issuance = _issue_tokens(node_name, GENESIS_REWARD, reason="node-registration")
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # genesis log stored in block file already; set reputation=0
    wpath, wdata = _get_wallet(node_name)
    _update_balance(wpath, Decimal(wdata["balance"]))  # noop write; ensures wallet present
    # store minimal node registry alongside wallet meta
    registry_db = os.path.join(META_DIR, "registry.db")
    with _connect(registry_db) as c:
        c.execute("""
        CREATE TABLE IF NOT EXISTS nodes (
            node_id TEXT PRIMARY KEY,
            node_name TEXT UNIQUE NOT NULL,
            version TEXT,
            capabilities TEXT,
            registered_at TEXT
        );
        """)
        c.execute(
            "INSERT OR REPLACE INTO nodes (node_id,node_name,version,capabilities,registered_at) VALUES (?,?,?,?,?)",
            (node_id, node_name, version, json.dumps(capabilities), datetime.now(timezone.utc).isoformat()),
        )
        c.commit()

    return jsonify({
        "node_id": node_id,
        "wallet": {
            "path": _wallet_path(node_name),
            "balance": str(Decimal(GENESIS_REWARD)),
            "reputation_score": 0
        },
        "issuance": issuance
    }), 201

@app.post("/api/send-token")
def send_token():
    data = request.get_json(silent=True) or {}
    from_node = (data.get("from") or "").strip()
    to_node = (data.get("to") or "").strip()
    amount = data.get("amount")
    meta = data.get("meta") or {}

    if not from_node or not to_node:
        return jsonify({"error": "from and to required"}), 400
    try:
        amount = Decimal(str(amount))
    except Exception:
        return jsonify({"error": "invalid amount"}), 400

    # Optional guard: give-to-receive enforcement hook
    # If a contribution reference is required, check it here.
    contrib_ref = meta.get("contribution_ref")
    if contrib_ref:
        with _connect(MASTER_DB) as c:
            row = c.execute("SELECT status FROM knowledge_requests WHERE id=?", (contrib_ref,)).fetchone()
            if not row or row[0] != "fulfilled":
                return jsonify({"error": "contribution_ref not fulfilled"}), 400

    try:
        res = _transfer(from_node, to_node, amount, meta)
        # light reputation tweak: sender negative if amount tiny and frequent? (omitted here)
        _reward_reputation(to_node, +1)
        # micro bonus occasionally
        if amount >= Decimal("1"):
            _bonus_micro_tokens(to_node, Decimal("0.0001"))
        return jsonify(res), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.get("/api/wallet/<node_name>")
def get_wallet(node_name: str):
    try:
        path, data = _get_wallet(node_name)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

    # last 25 tx from all indexed blocks
    with _connect(path) as c:
        tx_ids = [r[0] for r in c.execute("SELECT tx_id FROM tx_index ORDER BY rowid DESC LIMIT 25")]
        block_ids = {tx_id: r[0] for tx_id, r in ((tid, c.execute("SELECT block_id FROM tx_index WHERE tx_id=?", (tid,)).fetchone()) for tid in tx_ids)}

    txs: List[Dict[str, Any]] = []
    for tid in tx_ids:
        bid = block_ids.get(tid)
        if not bid: 
            continue
        bpath = _block_path(int(bid))
        if not os.path.exists(bpath):
            continue
        with _connect(bpath) as bc:
            row = bc.execute("SELECT json_record FROM tx WHERE id=?", (tid,)).fetchone()
            if row:
                try:
                    txs.append(json.loads(row[0]))
                except Exception:
                    pass

    return jsonify({
        "wallet": data,
        "transactions": txs
    }), 200

@app.get("/api/block/<int:block_id>")
def get_block(block_id: int):
    path = _block_path(block_id)
    if not os.path.exists(path):
        return jsonify({"error": "block not found"}), 404
    txs: List[Dict[str, Any]] = []
    with _connect(path) as c:
        for (record,) in c.execute("SELECT json_record FROM tx ORDER BY rowid DESC LIMIT 100"):
            try:
                txs.append(json.loads(record))
            except Exception:
                pass
    size = os.path.getsize(path)
    return jsonify({
        "block_id": block_id,
        "size_bytes": size,
        "tx_count": len(txs),
        "transactions": txs
    }), 200

@app.get("/api/top-nodes")
def top_nodes():
    # rank by reputation then recent activity (tx count in last 7 days)
    import glob
    rank: List[Tuple[str,int,int]] = []
    for wfile in glob.glob(os.path.join(WALLETS_DIR, "*.wallet.db")):
        try:
            with _connect(wfile) as c:
                node_name, rep = c.execute("SELECT node_name, reputation FROM wallet WHERE id=1").fetchone()
                # quick activity proxy: how many tx indexed
                cnt = c.execute("SELECT COUNT(*) FROM tx_index").fetchone()[0]
                rank.append((node_name, int(rep), int(cnt)))
        except Exception:
            continue
    rank.sort(key=lambda x: (-x[1], -x[2], x[0].lower()))
    return jsonify([{"node": n, "reputation": r, "activity": a} for n, r, a in rank[:50]]), 200

@app.post("/api/request-knowledge")
def request_knowledge():
    """
    Create or fulfill a knowledge exchange.
    POST body:
        {
          "requester": "Azurion",
          "provider": "Meta",
          "amount": 0.01,
          "response_proof": "optional-hash-when-fulfilling",
          "notes": "short description"
        }
    - When response_proof is missing: create pending request.
    - When response_proof is present: atomically transfer tokens requester -> provider and set fulfilled.
    """
    data = request.get_json(silent=True) or {}
    requester = (data.get("requester") or "").strip()
    provider = (data.get("provider") or "").strip()
    amount = data.get("amount")
    response_proof = data.get("response_proof")
    notes = (data.get("notes") or "").strip()

    if not requester or not provider or requester == provider:
        return jsonify({"error": "valid requester and provider required"}), 400
    try:
        amount = Decimal(str(amount))
    except Exception:
        return jsonify({"error": "invalid amount"}), 400
    if amount <= 0:
        return jsonify({"error": "amount must be positive"}), 400

    with _lock, _connect(MASTER_DB) as m:
        if response_proof:
            # fulfill an existing request (by pair and amount, most recent pending)
            row = m.execute("""
            SELECT id FROM knowledge_requests 
            WHERE requester=? AND provider=? AND amount=? AND status='pending'
            ORDER BY created_at DESC LIMIT 1
            """, (requester, provider, str(amount))).fetchone()
            if not row:
                return jsonify({"error": "no pending request to fulfill"}), 404
            req_id = row[0]
            try:
                res = _transfer(requester, provider, amount, {"contribution_ref": req_id, "notes": notes})
            except Exception as e:
                return jsonify({"error": str(e)}), 400
            m.execute("UPDATE knowledge_requests SET status='fulfilled', response_proof=?, fulfilled_at=? WHERE id=?",
                      (str(response_proof), datetime.now(timezone.utc).isoformat(), req_id))
            m.commit()
            # Reputation updates
            _reward_reputation(provider, +3)
            _reward_reputation(requester, +1)
            _bonus_micro_tokens(provider, Decimal("0.00005"))
            return jsonify({"request_id": req_id, "transfer": res}), 200
        else:
            # create pending
            req_id = uuid.uuid4().hex
            m.execute("""
            INSERT INTO knowledge_requests (id, requester, provider, amount, created_at, status, response_proof, fulfilled_at)
            VALUES (?,?,?,?,?,?,?,?)
            """, (
                req_id, requester, provider, str(amount),
                datetime.now(timezone.utc).isoformat(), "pending", None, None
            ))
            m.commit()
            return jsonify({"request_id": req_id, "status": "pending"}), 201

# ========================= Sample Flow (for quick test) =========================
@app.get("/api/_sample-flow")
def sample_flow():
    """
    Quick sanity route that demonstrates:
    - register two nodes if missing
    - create a knowledge request
    - fulfill it
    """
    a, b = "Azurion", "Meta"
    # ensure wallets
    for n in (a, b):
        if not os.path.exists(_wallet_path(n)):
            _init_wallet(uuid.uuid4().hex, n)
            _issue_tokens(n, GENESIS_REWARD, reason="sample-bootstrap")
    # request
    with _connect(MASTER_DB) as m:
        rid = uuid.uuid4().hex
        m.execute("""INSERT INTO knowledge_requests (id, requester, provider, amount, created_at, status)
                     VALUES (?,?,?,?,?,?)""",
                  (rid, a, b, str(Decimal("1.5")), datetime.now(timezone.utc).isoformat(), "pending"))
        m.commit()
    # fulfill
    res = _transfer(a, b, Decimal("1.5"), {"contribution_ref": rid, "notes": "sample exchange"})
    with _connect(MASTER_DB) as m:
        m.execute("UPDATE knowledge_requests SET status='fulfilled', response_proof=?, fulfilled_at=? WHERE id=?",
                  (uuid.uuid4().hex, datetime.now(timezone.utc).isoformat(), res["tx"]["meta"]["contribution_ref"]))
        m.commit()
    return jsonify({"ok": True, "rid": rid, "tx": res}), 200

# ========================= WSGI entry =========================
def create_app():
    return app


# --- injected: on-demand ensure table for `response` ---
def _ensure_response_table(db_path=None):
    try:
        import sqlite3, os, logging
        try:
            import SarahMemoryGlobals as config
        except Exception:
            class config: pass
        if db_path is None:
            base = getattr(config, "BASE_DIR", os.getcwd())
            db_path = os.path.join(config.DATASETS_DIR, "system_logs.db")
        con = sqlite3.connect(db_path); cur = con.cursor()
        cur.execute('CREATE TABLE IF NOT EXISTS response (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, user TEXT, content TEXT, source TEXT, intent TEXT)'); con.commit(); con.close()
        logging.debug("[DB] ensured table `response` in %s", db_path)
    except Exception as e:
        try:
            import logging; logging.warning("[DB] ensure `response` failed: %s", e)
        except Exception:
            pass
try:
    _ensure_response_table()
except Exception:
    pass
