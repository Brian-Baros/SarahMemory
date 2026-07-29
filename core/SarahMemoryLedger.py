"""--==The SarahMemory Project==--
File: SarahMemoryLedger.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-07-11
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

Governed immutable receipt, SarahNet wallet, and accountability ledger.

Authority model:
- Governance decides.
- Ledger records.
- AgentFirewall/RoachMotel contain.
- TrustRegistry maintains current passport state.
- User remains final authority.

The Ledger stores compact receipts and hashes, not unrestricted raw prompts,
secrets, credentials, or full payload dumps. Large evidence remains in its owning
organ and is referenced by a bounded payload_ref/hash.
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "ledger_service"
# CATEGORY = "ledger_wallet_governance_receipts"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "core_api_server"
# API_DOMAIN = "ledger"
# HARDWARE_DOMAIN = "filesystem"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "ledger"
# FAMILY = "governance_accountability"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-07-11"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Unified append-only accountability ledger for SarahNet, cognitive traces, chat receipts, tickets, terminal actions, AI-agent passports, firewall events, and bounded retention projections. Ledger never authorizes execution."
# --- SARAHMETA END ---

import hashlib
import hmac
import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN, getcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Flask is optional for local/offline core imports. The ledger receipt API is
# mounted only when Flask is available; core receipt functions remain usable
# without importing or starting a web server.
try:
    from flask import Blueprint, Flask, jsonify, request  # type: ignore
    _FLASK_AVAILABLE = True
except Exception:
    _FLASK_AVAILABLE = False

    class _LedgerRequestFallback:
        remote_addr = "127.0.0.1"
        args: Dict[str, Any] = {}
        headers: Dict[str, Any] = {}

        def get_json(self, silent: bool = True):
            return None

    request = _LedgerRequestFallback()  # type: ignore

    def jsonify(*args: Any, **kwargs: Any):  # type: ignore
        if args and isinstance(args[0], dict) and not kwargs:
            return args[0]
        return kwargs if kwargs else {"ok": True}

    class Blueprint:  # type: ignore
        def __init__(self, name: str, import_name: str, *args: Any, **kwargs: Any):
            self.name = name

        @staticmethod
        def _decorator(*args: Any, **kwargs: Any):
            def wrap(func):
                return func
            return wrap

        get = _decorator
        post = _decorator
        put = _decorator
        delete = _decorator
        route = _decorator

    class Flask:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any):
            self.blueprints: Dict[str, Any] = {}

        @staticmethod
        def _decorator(*args: Any, **kwargs: Any):
            def wrap(func):
                return func
            return wrap

        get = _decorator
        post = _decorator
        put = _decorator
        delete = _decorator
        route = _decorator

        def register_blueprint(self, blueprint: Any, *args: Any, **kwargs: Any):
            self.blueprints[getattr(blueprint, "name", str(blueprint))] = blueprint

try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None  # type: ignore

try:
    from SarahMemoryAudit import audit_event  # type: ignore
except Exception:
    audit_event = None  # type: ignore

getcontext().prec = 50
TOKEN_DECIMALS = 7
TOKEN_UNIT = Decimal("0.0000001")
MAX_SUPPLY = Decimal("1000000")
GENESIS_REWARD = Decimal("50")
BLOCK_MAX_BYTES = 100 * 1024 * 1024
MAX_BLOCKS = 400
LEDGER_SCHEMA = "SARAHMEMORY_GOVERNED_LEDGER_V2"

_LOCK = threading.RLock()
_INITIALIZED = False
_RECEIPT_BP = Blueprint("sarahmemory_ledger_receipts_v900", __name__)
app = Flask(__name__)


def _base_dir() -> str:
    try:
        return str(Path(getattr(config, "BASE_DIR", os.getcwd())).expanduser().resolve())
    except Exception:
        return str(Path(os.getcwd()).resolve())


def _data_dir() -> str:
    try:
        return str(Path(getattr(config, "DATA_DIR", os.path.join(_base_dir(), "data"))).expanduser().resolve())
    except Exception:
        return os.path.join(_base_dir(), "data")


def _datasets_dir() -> str:
    try:
        return str(Path(getattr(config, "DATASETS_DIR", os.path.join(_data_dir(), "memory", "datasets"))).expanduser().resolve())
    except Exception:
        return os.path.join(_data_dir(), "memory", "datasets")


def _ledger_root() -> str:
    return os.path.join(_datasets_dir(), "ledger")


BASE_DIR = _base_dir()
DATASETS_DIR = _datasets_dir()
LEDGER_ROOT = _ledger_root()
WALLETS_DIR = os.path.join(LEDGER_ROOT, "wallets")
BLOCKS_DIR = os.path.join(LEDGER_ROOT, "blocks")
LOGS_DIR = os.path.join(_data_dir(), "logs")
MASTER_DB = str(getattr(config, "LEDGER_DB_PATH", os.path.join(DATASETS_DIR, "SarahMemoryLedger.db"))) if config is not None else os.path.join(DATASETS_DIR, "SarahMemoryLedger.db")


def _ensure_storage_dirs() -> None:
    for directory in (DATASETS_DIR, LEDGER_ROOT, WALLETS_DIR, BLOCKS_DIR, LOGS_DIR):
        os.makedirs(directory, exist_ok=True)


def _connect(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=15.0, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=15000;")
    return conn


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8", "ignore")).hexdigest()


_SECRET_KEYS = re.compile(r"(password|passwd|secret|token|api[_-]?key|private[_-]?key|credential|authorization|cookie)", re.I)


def _bounded_metadata(value: Any, max_chars: Optional[int] = None) -> Dict[str, Any]:
    limit = int(max_chars or getattr(config, "SARAH_LEDGER_RECEIPT_MAX_METADATA_CHARS", 16000) or 16000)

    def cleanse(obj: Any, depth: int = 0) -> Any:
        if depth > 6:
            return "<depth_limited>"
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            for key, val in list(obj.items())[:128]:
                key_s = str(key)[:128]
                out[key_s] = "<redacted>" if _SECRET_KEYS.search(key_s) else cleanse(val, depth + 1)
            return out
        if isinstance(obj, (list, tuple, set)):
            return [cleanse(item, depth + 1) for item in list(obj)[:128]]
        if isinstance(obj, bytes):
            return {"bytes_sha256": hashlib.sha256(obj).hexdigest(), "size_bytes": len(obj)}
        if obj is None or isinstance(obj, (bool, int, float)):
            return obj
        return str(obj)[:1000]

    clean = cleanse(value if isinstance(value, dict) else {"value": value})
    raw = _canonical_json(clean)
    if len(raw) <= limit:
        return clean if isinstance(clean, dict) else {"value": clean}
    return {
        "metadata_truncated": True,
        "metadata_sha256": hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest(),
        "preview": raw[: max(256, limit - 256)],
    }


def _append_only_guards(conn: sqlite3.Connection) -> None:
    statements = (
        "CREATE TRIGGER IF NOT EXISTS guard_tx_update BEFORE UPDATE ON tx BEGIN SELECT RAISE(ABORT,'append-only: tx update forbidden'); END;",
        "CREATE TRIGGER IF NOT EXISTS guard_tx_delete BEFORE DELETE ON tx BEGIN SELECT RAISE(ABORT,'append-only: tx delete forbidden'); END;",
    )
    for statement in statements:
        try:
            conn.execute(statement)
        except Exception:
            pass


def _receipt_guards(conn: sqlite3.Connection) -> None:
    for statement in (
        "CREATE TRIGGER IF NOT EXISTS guard_receipt_update BEFORE UPDATE ON governance_receipts BEGIN SELECT RAISE(ABORT,'append-only: governance receipt update forbidden'); END;",
        "CREATE TRIGGER IF NOT EXISTS guard_receipt_delete BEFORE DELETE ON governance_receipts BEGIN SELECT RAISE(ABORT,'append-only: governance receipt delete forbidden'); END;",
    ):
        try:
            conn.execute(statement)
        except Exception:
            pass


def _init_master() -> None:
    global _INITIALIZED
    with _LOCK:
        _ensure_storage_dirs()
        with _connect(MASTER_DB) as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS supply (
                    id INTEGER PRIMARY KEY CHECK (id=1),
                    total_supply TEXT NOT NULL,
                    issued TEXT NOT NULL,
                    last_block INTEGER NOT NULL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_requests (
                    id TEXT PRIMARY KEY,
                    requester TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    amount TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    response_proof TEXT,
                    fulfilled_at TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    node_name TEXT UNIQUE NOT NULL,
                    version TEXT,
                    capabilities TEXT,
                    registered_at TEXT NOT NULL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS governance_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    ts REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    schema_name TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    subject_id TEXT,
                    task_id TEXT,
                    conversation_id TEXT,
                    lane TEXT,
                    verdict TEXT,
                    risk TEXT,
                    retention_class TEXT,
                    payload_hash TEXT,
                    payload_ref TEXT,
                    summary TEXT,
                    metadata_json TEXT NOT NULL,
                    previous_hash TEXT,
                    receipt_hash TEXT NOT NULL UNIQUE
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_receipts_domain_ts ON governance_receipts(domain, ts)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_receipts_task ON governance_receipts(task_id, ts)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_receipts_subject ON governance_receipts(subject_id, ts)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_receipts_event ON governance_receipts(event_type, ts)")
            c.execute("""
                CREATE TABLE IF NOT EXISTS receipt_retention_projection (
                    receipt_id TEXT PRIMARY KEY,
                    evaluated_ts REAL NOT NULL,
                    age_seconds REAL NOT NULL,
                    half_life_seconds REAL NOT NULL,
                    decay_score REAL NOT NULL,
                    retention_state TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    reason TEXT,
                    FOREIGN KEY(receipt_id) REFERENCES governance_receipts(receipt_id)
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS ledger_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    created_ts REAL NOT NULL,
                    receipt_count INTEGER NOT NULL,
                    first_receipt_id TEXT,
                    last_receipt_id TEXT,
                    root_hash TEXT NOT NULL,
                    metadata_json TEXT
                )
            """)
            cur = c.execute("SELECT COUNT(*) FROM supply WHERE id=1")
            if int(cur.fetchone()[0]) == 0:
                c.execute(
                    "INSERT INTO supply (id,total_supply,issued,last_block) VALUES (1,?,?,1)",
                    (str(MAX_SUPPLY), "0"),
                )
            _receipt_guards(c)
            c.commit()
        _INITIALIZED = True


def initialize_ledger() -> Dict[str, Any]:
    try:
        _init_master()
        return {"ok": True, "db_path": MASTER_DB, "schema": LEDGER_SCHEMA}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "db_path": MASTER_DB, "schema": LEDGER_SCHEMA}


def _ensure_initialized() -> None:
    if not _INITIALIZED or not os.path.exists(MASTER_DB):
        _init_master()


def _hmac_key() -> Optional[bytes]:
    for name in ("SARAH_LEDGER_SECRET", "SARAH_AGENT_PASSPORT_SECRET", "MESH_SHARED_SECRET"):
        try:
            value = os.getenv(name) or (getattr(config, name, None) if config is not None else None)
            if value:
                return hashlib.sha256(str(value).encode("utf-8", "ignore")).digest()
        except Exception:
            pass
    # Local development integrity fallback. This is not represented as production
    # authentication and never grants authority.
    seed = f"{_base_dir()}|SarahMemoryLedger|{getattr(config, 'PROJECT_VERSION', '9.0.0') if config is not None else '9.0.0'}"
    return hashlib.sha256(seed.encode("utf-8", "ignore")).digest()


def _tx_proof(payload: Dict[str, Any]) -> str:
    canonical = _canonical_json(payload).encode("utf-8", "ignore")
    key = _hmac_key()
    return hmac.new(key, canonical, hashlib.sha256).hexdigest() if key else hashlib.sha256(canonical).hexdigest()


def record_governance_receipt(
    domain: str,
    event_type: str,
    *,
    subject_id: str = "",
    task_id: str = "",
    conversation_id: str = "",
    lane: str = "",
    verdict: str = "OBSERVED",
    risk: str = "low",
    retention_class: str = "standard",
    payload: Any = None,
    payload_hash: str = "",
    payload_ref: str = "",
    summary: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Append a compact immutable receipt. This function never authorizes action."""
    if config is not None and not bool(getattr(config, "SARAH_LEDGER_RECEIPTS_ENABLED", True)):
        return {"ok": False, "disabled": True, "reason": "ledger_receipts_disabled"}
    _ensure_initialized()
    now = time.time()
    receipt_id = f"rcpt_{uuid.uuid4().hex}"
    domain_s = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(domain or "general"))[:96]
    event_s = re.sub(r"[^a-zA-Z0-9_.:-]+", "_", str(event_type or "event"))[:128]
    metadata_clean = _bounded_metadata(metadata or {})
    effective_payload_hash = str(payload_hash or "").strip()
    if not effective_payload_hash and payload is not None:
        effective_payload_hash = _hash_payload(payload)

    with _LOCK, _connect(MASTER_DB) as con:
        row = con.execute(
            "SELECT receipt_hash FROM governance_receipts WHERE domain=? ORDER BY ts DESC,rowid DESC LIMIT 1",
            (domain_s,),
        ).fetchone()
        previous_hash = str(row[0]) if row else "GENESIS"
        body = {
            "receipt_id": receipt_id,
            "ts": now,
            "timestamp": datetime.fromtimestamp(now, timezone.utc).isoformat(),
            "schema_name": LEDGER_SCHEMA,
            "domain": domain_s,
            "event_type": event_s,
            "subject_id": str(subject_id or "")[:180],
            "task_id": str(task_id or "")[:180],
            "conversation_id": str(conversation_id or "")[:180],
            "lane": str(lane or "")[:96],
            "verdict": str(verdict or "OBSERVED")[:64],
            "risk": str(risk or "low")[:64],
            "retention_class": str(retention_class or "standard")[:64],
            "payload_hash": effective_payload_hash[:128],
            "payload_ref": str(payload_ref or "")[:500],
            "summary": str(summary or "")[:1000],
            "metadata": metadata_clean,
            "previous_hash": previous_hash,
        }
        receipt_hash = _tx_proof(body)
        con.execute(
            """INSERT INTO governance_receipts(
                receipt_id,ts,timestamp,schema_name,domain,event_type,subject_id,task_id,
                conversation_id,lane,verdict,risk,retention_class,payload_hash,payload_ref,
                summary,metadata_json,previous_hash,receipt_hash
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                body["receipt_id"], body["ts"], body["timestamp"], body["schema_name"],
                body["domain"], body["event_type"], body["subject_id"], body["task_id"],
                body["conversation_id"], body["lane"], body["verdict"], body["risk"],
                body["retention_class"], body["payload_hash"], body["payload_ref"],
                body["summary"], _canonical_json(metadata_clean), previous_hash, receipt_hash,
            ),
        )
        con.commit()

    result = {**body, "metadata": metadata_clean, "receipt_hash": receipt_hash, "ok": True, "execution_authority": False}
    try:
        if callable(audit_event):
            audit_event(
                "ledger",
                event_s,
                str(verdict or "OBSERVED"),
                {"receipt_id": receipt_id, "domain": domain_s, "task_id": task_id, "receipt_hash": receipt_hash},
                actor=str(subject_id or "SarahMemory"),
                risk=str(risk or "low"),
                source="SarahMemoryLedger",
                retention=str(retention_class or "standard"),
            )
    except Exception:
        pass
    return result


def get_governance_receipts(
    *, domain: str = "", event_type: str = "", subject_id: str = "", task_id: str = "", limit: int = 100
) -> List[Dict[str, Any]]:
    _ensure_initialized()
    limit = max(1, min(1000, int(limit or 100)))
    clauses: List[str] = []
    params: List[Any] = []
    for field, value in (("domain", domain), ("event_type", event_type), ("subject_id", subject_id), ("task_id", task_id)):
        if str(value or "").strip():
            clauses.append(f"{field}=?")
            params.append(str(value).strip())
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    sql = (
        "SELECT receipt_id,ts,timestamp,schema_name,domain,event_type,subject_id,task_id,conversation_id,lane,verdict,risk,retention_class,payload_hash,payload_ref,summary,metadata_json,previous_hash,receipt_hash "
        f"FROM governance_receipts{where} ORDER BY ts DESC,rowid DESC LIMIT ?"
    )
    params.append(limit)
    with _connect(MASTER_DB) as con:
        rows = con.execute(sql, tuple(params)).fetchall()
    out: List[Dict[str, Any]] = []
    keys = ("receipt_id", "ts", "timestamp", "schema_name", "domain", "event_type", "subject_id", "task_id", "conversation_id", "lane", "verdict", "risk", "retention_class", "payload_hash", "payload_ref", "summary", "metadata_json", "previous_hash", "receipt_hash")
    for row in rows:
        item = dict(zip(keys, row))
        try:
            item["metadata"] = json.loads(item.pop("metadata_json") or "{}")
        except Exception:
            item["metadata"] = {}
        out.append(item)
    return out


def verify_governance_chain(domain: str = "", limit: int = 5000) -> Dict[str, Any]:
    _ensure_initialized()
    limit = max(1, min(25000, int(limit or 5000)))
    params: Tuple[Any, ...]
    if domain:
        sql = "SELECT * FROM governance_receipts WHERE domain=? ORDER BY ts ASC,rowid ASC LIMIT ?"
        params = (domain, limit)
    else:
        sql = "SELECT * FROM governance_receipts ORDER BY domain ASC,ts ASC,rowid ASC LIMIT ?"
        params = (limit,)
    with _connect(MASTER_DB) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params).fetchall()
    previous_by_domain: Dict[str, str] = {}
    failures: List[Dict[str, Any]] = []
    checked = 0
    for row in rows:
        item = dict(row)
        d = str(item.get("domain") or "general")
        expected_previous = previous_by_domain.get(d, "GENESIS")
        metadata = {}
        try:
            metadata = json.loads(item.get("metadata_json") or "{}")
        except Exception:
            metadata = {}
        body = {
            "receipt_id": item.get("receipt_id"), "ts": item.get("ts"), "timestamp": item.get("timestamp"),
            "schema_name": item.get("schema_name"), "domain": d, "event_type": item.get("event_type"),
            "subject_id": item.get("subject_id"), "task_id": item.get("task_id"),
            "conversation_id": item.get("conversation_id"), "lane": item.get("lane"),
            "verdict": item.get("verdict"), "risk": item.get("risk"),
            "retention_class": item.get("retention_class"), "payload_hash": item.get("payload_hash"),
            "payload_ref": item.get("payload_ref"), "summary": item.get("summary"),
            "metadata": metadata, "previous_hash": item.get("previous_hash"),
        }
        computed = _tx_proof(body)
        if str(item.get("previous_hash")) != expected_previous or computed != str(item.get("receipt_hash")):
            failures.append({
                "receipt_id": item.get("receipt_id"), "domain": d,
                "previous_expected": expected_previous, "previous_actual": item.get("previous_hash"),
                "hash_valid": computed == str(item.get("receipt_hash")),
            })
        previous_by_domain[d] = str(item.get("receipt_hash"))
        checked += 1
    return {"ok": not failures, "checked": checked, "failures": failures[:100], "truncated": checked >= limit, "domain": domain or "*"}


_RETENTION_HALF_LIFE_SECONDS = {
    "volatile": 6 * 3600.0,
    "runtime": 24 * 3600.0,
    "standard": 30 * 24 * 3600.0,
    "chat": 90 * 24 * 3600.0,
    "ticket": 180 * 24 * 3600.0,
    "security": 365 * 24 * 3600.0,
    "passport": 365 * 24 * 3600.0,
    "identity": 10 * 365 * 24 * 3600.0,
    "permanent": 100 * 365 * 24 * 3600.0,
}


def _decay_score(age_seconds: float, half_life_seconds: float, initial_value: float = 1.0) -> float:
    if half_life_seconds <= 0:
        return 0.0
    return max(0.0, min(float(initial_value), float(initial_value) * math.pow(2.0, -max(0.0, age_seconds) / half_life_seconds)))


def _retention_state(score: float, pinned: bool = False) -> str:
    if pinned:
        return "PINNED"
    if score >= 0.70:
        return "HOT"
    if score >= 0.35:
        return "WARM"
    if score >= 0.12:
        return "COLD"
    if score >= 0.03:
        return "FOSSIL"
    return "HASH_ONLY_ELIGIBLE"


def apply_retention_decay(limit: Optional[int] = None, now_ts: Optional[float] = None) -> Dict[str, Any]:
    """Update the mutable retention projection; immutable receipts are untouched."""
    _ensure_initialized()
    batch = int(limit or getattr(config, "SARAH_LEDGER_RETENTION_BATCH", 250) or 250)
    batch = max(1, min(2000, batch))
    now = float(now_ts or time.time())
    with _LOCK, _connect(MASTER_DB) as con:
        rows = con.execute(
            "SELECT receipt_id,ts,retention_class,risk,event_type,metadata_json FROM governance_receipts ORDER BY ts ASC,rowid ASC LIMIT ?",
            (batch,),
        ).fetchall()
        changed = 0
        states: Dict[str, int] = {}
        for receipt_id, ts, retention_class, risk, event_type, metadata_json in rows:
            metadata: Dict[str, Any] = {}
            try:
                metadata = json.loads(metadata_json or "{}")
            except Exception:
                pass
            pinned = bool(metadata.get("user_pinned") or metadata.get("identity_critical") or str(retention_class).lower() == "permanent")
            cls = str(retention_class or "standard").lower()
            half_life = float(_RETENTION_HALF_LIFE_SECONDS.get(cls, _RETENTION_HALF_LIFE_SECONDS["standard"]))
            if str(risk or "").lower() in ("high", "critical") or "BLOCK" in str(event_type or "").upper():
                half_life = max(half_life, _RETENTION_HALF_LIFE_SECONDS["security"])
            age = max(0.0, now - float(ts or now))
            score = 1.0 if pinned else _decay_score(age, half_life)
            state = _retention_state(score, pinned)
            con.execute(
                """INSERT INTO receipt_retention_projection(receipt_id,evaluated_ts,age_seconds,half_life_seconds,decay_score,retention_state,pinned,reason)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(receipt_id) DO UPDATE SET evaluated_ts=excluded.evaluated_ts,age_seconds=excluded.age_seconds,
                   half_life_seconds=excluded.half_life_seconds,decay_score=excluded.decay_score,
                   retention_state=excluded.retention_state,pinned=excluded.pinned,reason=excluded.reason""",
                (receipt_id, now, age, half_life, score, state, 1 if pinned else 0, "nuclear_half_life_projection"),
            )
            states[state] = states.get(state, 0) + 1
            changed += 1
        con.commit()
    return {"ok": True, "evaluated": changed, "states": states, "immutable_receipts_modified": False}


def get_retention_projection(limit: int = 100) -> List[Dict[str, Any]]:
    _ensure_initialized()
    with _connect(MASTER_DB) as con:
        rows = con.execute(
            "SELECT receipt_id,evaluated_ts,age_seconds,half_life_seconds,decay_score,retention_state,pinned,reason FROM receipt_retention_projection ORDER BY evaluated_ts DESC LIMIT ?",
            (max(1, min(1000, int(limit or 100))),),
        ).fetchall()
    keys = ("receipt_id", "evaluated_ts", "age_seconds", "half_life_seconds", "decay_score", "retention_state", "pinned", "reason")
    return [dict(zip(keys, row)) for row in rows]


def create_ledger_checkpoint(domain: str, limit: int = 5000) -> Dict[str, Any]:
    receipts = list(reversed(get_governance_receipts(domain=domain, limit=limit)))
    if not receipts:
        return {"ok": False, "reason": "no_receipts", "domain": domain}
    leaves = [str(r.get("receipt_hash") or "") for r in receipts]
    level = [hashlib.sha256(x.encode("ascii", "ignore")).hexdigest() for x in leaves]
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [hashlib.sha256((level[i] + level[i + 1]).encode("ascii")).hexdigest() for i in range(0, len(level), 2)]
    root_hash = level[0]
    checkpoint_id = f"chk_{uuid.uuid4().hex}"
    with _connect(MASTER_DB) as con:
        con.execute(
            "INSERT INTO ledger_checkpoints(checkpoint_id,domain,created_ts,receipt_count,first_receipt_id,last_receipt_id,root_hash,metadata_json) VALUES (?,?,?,?,?,?,?,?)",
            (checkpoint_id, domain, time.time(), len(receipts), receipts[0]["receipt_id"], receipts[-1]["receipt_id"], root_hash, "{}"),
        )
        con.commit()
    return {"ok": True, "checkpoint_id": checkpoint_id, "domain": domain, "receipt_count": len(receipts), "root_hash": root_hash}


# ---------------------------------------------------------------------------
# SarahNet wallet/token compatibility layer
# ---------------------------------------------------------------------------
def _block_path(block_id: int) -> str:
    return os.path.join(BLOCKS_DIR, f"SarahCryptCoin{int(block_id):03d}.db")


def _ensure_block(block_id: int) -> None:
    _ensure_storage_dirs()
    with _connect(_block_path(block_id)) as con:
        con.execute("CREATE TABLE IF NOT EXISTS tx (id TEXT PRIMARY KEY,json_record TEXT NOT NULL)")
        _append_only_guards(con)
        con.commit()


def _current_block_id() -> int:
    _ensure_initialized()
    with _connect(MASTER_DB) as con:
        row = con.execute("SELECT last_block FROM supply WHERE id=1").fetchone()
        return int(row[0])


def _rotate_block_if_needed() -> int:
    with _LOCK:
        block_id = _current_block_id()
        _ensure_block(block_id)
        path = _block_path(block_id)
        if os.path.getsize(path) >= BLOCK_MAX_BYTES:
            if block_id >= MAX_BLOCKS:
                raise RuntimeError("Block limit reached; cannot rotate further.")
            block_id += 1
            with _connect(MASTER_DB) as con:
                con.execute("UPDATE supply SET last_block=? WHERE id=1", (block_id,))
                con.commit()
            _ensure_block(block_id)
        return block_id


def _wallet_path(node_name: str) -> str:
    safe = "".join(ch for ch in str(node_name or "") if ch.isalnum() or ch in ("-", "_"))[:128]
    if not safe:
        raise ValueError("invalid node name")
    return os.path.join(WALLETS_DIR, f"{safe}.wallet.db")


def _init_wallet(node_id: str, node_name: str) -> str:
    _ensure_initialized()
    path = _wallet_path(node_name)
    with _connect(path) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS wallet(
            id INTEGER PRIMARY KEY CHECK(id=1),node_id TEXT UNIQUE NOT NULL,node_name TEXT UNIQUE NOT NULL,
            balance TEXT NOT NULL,reputation INTEGER NOT NULL DEFAULT 0,created_at TEXT NOT NULL)""")
        con.execute("CREATE TABLE IF NOT EXISTS tx_index(tx_id TEXT PRIMARY KEY,block_id INTEGER NOT NULL)")
        row = con.execute("SELECT COUNT(*) FROM wallet WHERE id=1").fetchone()
        if int(row[0]) == 0:
            con.execute(
                "INSERT INTO wallet(id,node_id,node_name,balance,reputation,created_at) VALUES(1,?,?,?,0,?)",
                (str(node_id), str(node_name), "0", datetime.now(timezone.utc).isoformat()),
            )
        con.commit()
    return path


def _get_wallet(node_name: str) -> Tuple[str, Dict[str, Any]]:
    path = _wallet_path(node_name)
    if not os.path.exists(path):
        raise FileNotFoundError("wallet not found")
    with _connect(path) as con:
        row = con.execute("SELECT node_id,node_name,balance,reputation,created_at FROM wallet WHERE id=1").fetchone()
    if not row:
        raise RuntimeError("wallet row missing")
    return path, {"node_id": row[0], "node_name": row[1], "balance": str(Decimal(row[2])), "reputation": int(row[3]), "created_at": row[4]}


def _update_balance(path: str, new_balance: Decimal) -> None:
    with _connect(path) as con:
        con.execute("UPDATE wallet SET balance=? WHERE id=1", (str(new_balance),))
        con.commit()


def _delta_reputation(path: str, delta: int) -> None:
    with _connect(path) as con:
        con.execute("UPDATE wallet SET reputation=reputation+? WHERE id=1", (int(delta),))
        con.commit()


def _index_wallet_tx(path: str, tx_id: str, block_id: int) -> None:
    with _connect(path) as con:
        con.execute("INSERT OR IGNORE INTO tx_index(tx_id,block_id) VALUES(?,?)", (tx_id, int(block_id)))
        con.commit()


def _quantize(amount: Decimal) -> Decimal:
    return Decimal(amount).quantize(TOKEN_UNIT, rounding=ROUND_DOWN)


def _append_wallet_tx(record: Dict[str, Any], wallet_paths: Iterable[str]) -> Dict[str, Any]:
    block_id = _rotate_block_if_needed()
    tx_id = str(record.get("tx_id") or uuid.uuid4().hex)
    record["tx_id"] = tx_id
    record["block_id"] = block_id
    record["proof"] = _tx_proof({k: v for k, v in record.items() if k != "proof"})
    with _connect(_block_path(block_id)) as con:
        con.execute("INSERT INTO tx(id,json_record) VALUES(?,?)", (tx_id, _canonical_json(record)))
        con.commit()
    for path in wallet_paths:
        _index_wallet_tx(path, tx_id, block_id)
    return record


def _issue_tokens(to_node: str, amount: Decimal, reason: str) -> Dict[str, Any]:
    _ensure_initialized()
    amount = _quantize(amount)
    if amount <= 0:
        raise ValueError("amount must be positive")
    path, wallet = _get_wallet(to_node)
    with _LOCK, _connect(MASTER_DB) as master:
        issued = Decimal(master.execute("SELECT issued FROM supply WHERE id=1").fetchone()[0])
        if issued + amount > MAX_SUPPLY:
            raise RuntimeError("maximum token supply exceeded")
        master.execute("UPDATE supply SET issued=? WHERE id=1", (str(issued + amount),))
        master.commit()
        new_balance = Decimal(wallet["balance"]) + amount
        _update_balance(path, new_balance)
        record = _append_wallet_tx({
            "type": "issue", "to": to_node, "amount": str(amount), "reason": str(reason or ""),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, [path])
    return {"ok": True, "tx": record, "balance": str(new_balance)}


def _transfer(from_node: str, to_node: str, amount: Decimal, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    amount = _quantize(amount)
    if amount <= 0:
        raise ValueError("amount must be positive")
    if from_node == to_node:
        raise ValueError("from and to must differ")
    from_path, from_wallet = _get_wallet(from_node)
    to_path, to_wallet = _get_wallet(to_node)
    with _LOCK:
        from_balance = Decimal(from_wallet["balance"])
        if from_balance < amount:
            raise RuntimeError("insufficient balance")
        to_balance = Decimal(to_wallet["balance"])
        _update_balance(from_path, from_balance - amount)
        try:
            _update_balance(to_path, to_balance + amount)
            record = _append_wallet_tx({
                "type": "transfer", "from": from_node, "to": to_node, "amount": str(amount),
                "meta": _bounded_metadata(meta or {}, 8000), "timestamp": datetime.now(timezone.utc).isoformat(),
            }, [from_path, to_path])
        except Exception:
            _update_balance(from_path, from_balance)
            _update_balance(to_path, to_balance)
            raise
    return {"ok": True, "tx": record, "from_balance": str(from_balance - amount), "to_balance": str(to_balance + amount)}


def _reward_reputation(node_name: str, delta: int) -> None:
    try:
        path, _ = _get_wallet(node_name)
        _delta_reputation(path, int(delta))
    except Exception:
        pass


def _bonus_micro_tokens(node_name: str, amount: Decimal) -> Optional[Dict[str, Any]]:
    try:
        return _issue_tokens(node_name, amount, reason="reputation_micro_bonus")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Standalone legacy-compatible Flask routes
# ---------------------------------------------------------------------------
@app.post("/api/register-node")
def register_node():
    data = request.get_json(silent=True) or {}
    node_name = str(data.get("node_name") or data.get("name") or "").strip()
    if not node_name:
        return jsonify({"error": "node_name required"}), 400
    node_id = str(data.get("node_id") or uuid.uuid4().hex)
    version = str(data.get("version") or "")
    capabilities = data.get("capabilities") or []
    try:
        _init_wallet(node_id, node_name)
        _, wallet = _get_wallet(node_name)
        issuance = None
        if Decimal(wallet["balance"]) == 0:
            issuance = _issue_tokens(node_name, GENESIS_REWARD, "genesis_registration")
        with _connect(MASTER_DB) as con:
            con.execute(
                "INSERT OR REPLACE INTO nodes(node_id,node_name,version,capabilities,registered_at) VALUES(?,?,?,?,?)",
                (node_id, node_name, version, _canonical_json(capabilities), datetime.now(timezone.utc).isoformat()),
            )
            con.commit()
        return jsonify({"node_id": node_id, "wallet": _get_wallet(node_name)[1], "issuance": issuance}), 201
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/api/send-token")
def send_token():
    data = request.get_json(silent=True) or {}
    try:
        result = _transfer(str(data.get("from") or "").strip(), str(data.get("to") or "").strip(), Decimal(str(data.get("amount"))), data.get("meta") or {})
        _reward_reputation(str(data.get("to") or ""), 1)
        return jsonify(result), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/wallet/<node_name>")
def get_wallet(node_name: str):
    try:
        path, wallet = _get_wallet(node_name)
        with _connect(path) as con:
            rows = con.execute("SELECT tx_id,block_id FROM tx_index ORDER BY rowid DESC LIMIT 25").fetchall()
        txs: List[Dict[str, Any]] = []
        for tx_id, block_id in rows:
            bpath = _block_path(int(block_id))
            if not os.path.exists(bpath):
                continue
            with _connect(bpath) as con:
                row = con.execute("SELECT json_record FROM tx WHERE id=?", (tx_id,)).fetchone()
                if row:
                    try:
                        txs.append(json.loads(row[0]))
                    except Exception:
                        pass
        return jsonify({"wallet": wallet, "transactions": txs}), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 404


@app.get("/api/block/<int:block_id>")
def get_block(block_id: int):
    path = _block_path(block_id)
    if not os.path.exists(path):
        return jsonify({"error": "block not found"}), 404
    with _connect(path) as con:
        rows = con.execute("SELECT json_record FROM tx ORDER BY rowid DESC LIMIT 100").fetchall()
    txs = []
    for (raw,) in rows:
        try:
            txs.append(json.loads(raw))
        except Exception:
            pass
    return jsonify({"block_id": block_id, "size_bytes": os.path.getsize(path), "tx_count": len(txs), "transactions": txs}), 200


@app.get("/api/top-nodes")
def top_nodes():
    _ensure_storage_dirs()
    rank: List[Tuple[str, int, int]] = []
    for path in Path(WALLETS_DIR).glob("*.wallet.db"):
        try:
            with _connect(str(path)) as con:
                node_name, reputation = con.execute("SELECT node_name,reputation FROM wallet WHERE id=1").fetchone()
                activity = int(con.execute("SELECT COUNT(*) FROM tx_index").fetchone()[0])
            rank.append((str(node_name), int(reputation), activity))
        except Exception:
            continue
    rank.sort(key=lambda item: (-item[1], -item[2], item[0].lower()))
    return jsonify([{"node": n, "reputation": r, "activity": a} for n, r, a in rank[:50]]), 200


@app.post("/api/request-knowledge")
def request_knowledge():
    data = request.get_json(silent=True) or {}
    requester = str(data.get("requester") or "").strip()
    provider = str(data.get("provider") or "").strip()
    try:
        amount = _quantize(Decimal(str(data.get("amount"))))
    except Exception:
        return jsonify({"error": "invalid amount"}), 400
    if not requester or not provider or requester == provider or amount <= 0:
        return jsonify({"error": "valid requester/provider/positive amount required"}), 400
    _ensure_initialized()
    proof = str(data.get("response_proof") or "").strip()
    with _LOCK, _connect(MASTER_DB) as con:
        if proof:
            row = con.execute(
                "SELECT id FROM knowledge_requests WHERE requester=? AND provider=? AND amount=? AND status='pending' ORDER BY created_at DESC LIMIT 1",
                (requester, provider, str(amount)),
            ).fetchone()
            if not row:
                return jsonify({"error": "no pending request to fulfill"}), 404
            request_id = row[0]
            try:
                transfer = _transfer(requester, provider, amount, {"contribution_ref": request_id, "notes": str(data.get("notes") or "")})
            except Exception as exc:
                return jsonify({"error": str(exc)}), 400
            con.execute("UPDATE knowledge_requests SET status='fulfilled',response_proof=?,fulfilled_at=? WHERE id=?", (proof, datetime.now(timezone.utc).isoformat(), request_id))
            con.commit()
            _reward_reputation(provider, 3)
            _reward_reputation(requester, 1)
            return jsonify({"request_id": request_id, "transfer": transfer}), 200
        request_id = uuid.uuid4().hex
        con.execute(
            "INSERT INTO knowledge_requests(id,requester,provider,amount,created_at,status,response_proof,fulfilled_at) VALUES(?,?,?,?,?,'pending',NULL,NULL)",
            (request_id, requester, provider, str(amount), datetime.now(timezone.utc).isoformat()),
        )
        con.commit()
    return jsonify({"request_id": request_id, "status": "pending"}), 201


@app.post("/api/_sample-flow")
def sample_flow():
    if not bool(getattr(config, "DEVELOPERSMODE", False)):
        return jsonify({"ok": False, "error": "developer_mode_required"}), 403
    if str(request.headers.get("X-Sarah-Confirm") or "").lower() not in ("1", "true", "yes", "confirmed"):
        return jsonify({"ok": False, "error": "explicit_confirmation_required"}), 409
    return jsonify({"ok": True, "message": "Sample mutation route is armed but intentionally no-op in governed v9 runtime."}), 200


# ---------------------------------------------------------------------------
# Unified API receipt endpoints
# ---------------------------------------------------------------------------
def _local_request() -> bool:
    return str(getattr(request, "remote_addr", "") or "").lower() in ("", "127.0.0.1", "::1", "localhost")


@_RECEIPT_BP.get("/api/ledger/status")
def ledger_status_api():
    status = initialize_ledger()
    if not status.get("ok"):
        return jsonify(status), 500
    with _connect(MASTER_DB) as con:
        receipt_count = int(con.execute("SELECT COUNT(*) FROM governance_receipts").fetchone()[0])
        projection_count = int(con.execute("SELECT COUNT(*) FROM receipt_retention_projection").fetchone()[0])
    return jsonify({**status, "receipt_count": receipt_count, "retention_projection_count": projection_count, "execution_authority": False}), 200


@_RECEIPT_BP.get("/api/ledger/receipts")
def ledger_receipts_api():
    try:
        receipts = get_governance_receipts(
            domain=str(request.args.get("domain") or ""), event_type=str(request.args.get("event_type") or ""),
            subject_id=str(request.args.get("subject_id") or ""), task_id=str(request.args.get("task_id") or ""),
            limit=int(request.args.get("limit") or 100),
        )
        return jsonify({"ok": True, "receipts": receipts, "count": len(receipts)}), 200
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@_RECEIPT_BP.get("/api/ledger/verify")
def ledger_verify_api():
    result = verify_governance_chain(domain=str(request.args.get("domain") or ""), limit=int(request.args.get("limit") or 5000))
    return jsonify(result), 200 if result.get("ok") else 409


@_RECEIPT_BP.post("/api/ledger/retention/apply")
def ledger_retention_apply_api():
    if not _local_request():
        return jsonify({"ok": False, "error": "local_request_required"}), 403
    body = request.get_json(silent=True) or {}
    if not bool(body.get("confirmed") or body.get("user_confirmed")):
        return jsonify({"ok": False, "error": "explicit_confirmation_required"}), 409
    result = apply_retention_decay(limit=body.get("limit"))
    return jsonify(result), 200


def init_app(flask_app: Flask, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Explicitly initialize and mount receipt endpoints on the unified API app."""
    status = initialize_ledger()
    if not _FLASK_AVAILABLE:
        return {**status, "api_mounted": False, "api_reason": "flask_unavailable", "core_receipts_available": bool(status.get("ok"))}
    if "sarahmemory_ledger_receipts_v900" not in getattr(flask_app, "blueprints", {}):
        flask_app.register_blueprint(_RECEIPT_BP)
    return {**status, "api_mounted": True}


def create_app() -> Flask:
    initialize_ledger()
    if "sarahmemory_ledger_receipts_v900" not in app.blueprints:
        app.register_blueprint(_RECEIPT_BP)
    return app


def _ensure_response_table(db_path: Optional[str] = None) -> None:
    """Backward-compatible on-demand helper; never invoked at import time."""
    path = db_path or os.path.join(_datasets_dir(), "system_logs.db")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE IF NOT EXISTS response(id INTEGER PRIMARY KEY AUTOINCREMENT,ts TEXT,user TEXT,content TEXT,source TEXT,intent TEXT)")
        con.commit()


# ====================================================================
# END OF SarahMemoryLedger.py v9.0.0
# ====================================================================
# END OF LINE
