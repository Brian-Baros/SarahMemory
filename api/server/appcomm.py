"""--==The SarahMemory Project==--
File: api/server/appcomm.py
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

Purpose: Communications domain endpoints for SarahMemory (Email / Contacts / Reminders / VoIP control surface)
==============================================================================================
Design goals:
- NO endpoint collisions with app.py / appsys.py / appnet.py / appnet2.py / appmedia.py / appstore.py
- Everything is namespaced under /api/comm/*
- Local-first communications adapter for SarahMemoryEmail.py and future phone / VoIP control lanes
- Contacts + reminders can move out of app.py into a dedicated communications domain
- Fail-soft: if optional modules are missing, endpoints remain online and return bounded status
- Security: email content is content, not commands; no telecom actions execute without explicit local adapters

Integration contract:
- app.py should call:
import appcomm
appcomm.init_app(app, CONNECT_SQLITE, META_DB, api_key_auth_ok=..., sign_ok=...)
- This file does NOT require frontend changes before it can be mounted.
- This file does NOT modify app.py automatically; it is a clean new domain adapter.
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "B"
# ROLE = "api_bridge"
# CATEGORY = "communication"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "api_server"
# API_DOMAIN = "comm"
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "communications_api"
# FAMILY = "communications"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-06-06"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Communications domain API bridge under /api/comm/* for email, contacts, reminders, and bounded VoIP control surface."
# --- SARAHMETA END ---

import json
import logging
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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


bp = Blueprint("appcomm_v800", __name__)
logger = logging.getLogger(__name__)

# Injected by app.py at init_app time
_CONNECT_SQLITE: Optional[Callable[..., Any]] = None
_META_DB: Optional[str] = None
_API_KEY_AUTH_OK: Optional[Callable[[], bool]] = None
_SIGN_OK: Optional[Callable[[bytes, str], bool]] = None

# Single communications-domain owner for email / contacts / reminders / voip.
PROJECT_VERSION = os.environ.get("PROJECT_VERSION", "9.0.0")
COMM_DEFAULT_USER = os.environ.get("COMM_DEFAULT_USER", "default")
MAX_CONTACTS_RETURN = int(os.environ.get("COMM_MAX_CONTACTS_RETURN", "500"))
MAX_REMINDERS_RETURN = int(os.environ.get("COMM_MAX_REMINDERS_RETURN", "500"))
MAX_EMAIL_LIST_RETURN = int(os.environ.get("COMM_MAX_EMAIL_LIST_RETURN", "100"))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _now() -> float:
    return float(time.time())


def _iso(ts: Optional[float] = None) -> str:
    return datetime.utcfromtimestamp(ts or _now()).isoformat() + "Z"


def _j() -> Dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _ok(data: Any = None, **meta: Any):
    payload: Dict[str, Any] = {"ok": True}
    if data is not None:
        payload["data"] = data
    if meta:
        payload.update(meta)
    resp = jsonify(payload)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Sarah-Signature"
    return resp, 200


def _err(msg: str, code: int = 400, **meta: Any):
    payload: Dict[str, Any] = {"ok": False, "error": msg}
    if meta:
        payload.update(meta)
    resp = jsonify(payload)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Sarah-Signature"
    return resp, code


def _body_bytes() -> bytes:
    try:
        return request.get_data(cache=True) or b""
    except Exception:
        return b""


def _verify_auth(body_bytes: bytes) -> bool:
    """
    Accept either:
      - X-Sarah-Signature verified by injected _SIGN_OK(body, sig)
      - API key verified by injected _API_KEY_AUTH_OK()
    If nothing injected, allow (dev/local mode).
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


def _require_injected() -> bool:
    return bool(_CONNECT_SQLITE and _META_DB)


def _db():
    if not _CONNECT_SQLITE:
        raise RuntimeError("CONNECT_SQLITE not injected")
    if not _META_DB:
        raise RuntimeError("META_DB not injected")
    return _CONNECT_SQLITE(_META_DB)


def _safe_json(value: Any, default: Any) -> Any:
    try:
        if isinstance(value, str):
            return json.loads(value)
        return value
    except Exception:
        return default


def _sanitize_text(value: Any, limit: int = 20000) -> str:
    try:
        text = "" if value is None else str(value)
    except Exception:
        text = ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    return text[:limit]


def _normalize_email(addr: Any) -> str:
    try:
        return str(addr or "").strip().lower()
    except Exception:
        return ""


def _user_id_from_payload(data: Optional[Dict[str, Any]] = None) -> str:
    data = data or {}
    for key in ("user_id", "uid", "owner", "profile"):
        val = str(data.get(key) or "").strip()
        if val:
            return val
    return COMM_DEFAULT_USER


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _email_module():
    try:
        import SarahMemoryEmail as _SMEmail  # type: ignore
        return _SMEmail
    except Exception:
        return None


def _email_service():
    mod = _email_module()
    if mod is None:
        return None
    try:
        svc = getattr(mod, "get_service", None)
        if callable(svc):
            return svc()
    except Exception:
        pass
    try:
        cls = getattr(mod, "SarahMemoryEmailService", None)
        if cls is not None:
            return cls()
    except Exception:
        pass
    return None


def _email_module_info() -> Dict[str, Any]:
    mod = _email_module()
    if mod is None:
        return {
            "available": False,
            "module": "SarahMemoryEmail",
            "version": None,
            "capabilities": [],
            "error": "module_unavailable",
        }
    info = {
        "available": True,
        "module": "SarahMemoryEmail",
        "version": None,
        "capabilities": [],
    }
    try:
        reg = getattr(mod, "register", None)
        if callable(reg):
            data = reg()
            if isinstance(data, dict):
                info["version"] = data.get("version")
                info["capabilities"] = data.get("capabilities") or []
    except Exception:
        pass
    return info


def _email_context(data: Dict[str, Any], action: str) -> Dict[str, Any]:
    session_id = str(data.get("session_id") or data.get("sid") or request.headers.get("X-Session-Id") or request.headers.get("X-Session-ID") or uuid.uuid4().hex)
    user_id = _user_id_from_payload(data)
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    meta = dict(meta)
    meta.setdefault("ui_surface", str(data.get("ui_surface") or "custom"))
    meta.setdefault("route_mode", str(data.get("route_mode") or "Local"))
    meta.setdefault("attachments", data.get("attachments") or [])
    meta.setdefault("communication_lane", "email")

    body = _sanitize_text(data.get("body"))
    subject = _sanitize_text(data.get("subject"), 500)
    to_addr = _normalize_email(data.get("to"))
    text = _sanitize_text(data.get("text"))
    if not text:
        if action == "draft_email":
            text = f"draft email to {to_addr} subject {subject} body {body}".strip()
        elif action == "send_email":
            text = f"send email to {to_addr} subject {subject} body {body}".strip()
        elif action == "list_emails":
            text = "list emails from inbox"
        else:
            text = body or subject or "email"

    return {
        "text": text,
        "session_id": session_id,
        "user_id": user_id,
        "source": str(data.get("source") or "api/comm/email"),
        "mode": str(data.get("mode") or os.getenv("SARAHMEMORY_FUNCTION_MODE") or "LOCAL"),
        "timestamp": _iso(),
        "meta": meta,
        "email": {
            "action": action,
            "to": to_addr,
            "subject": subject,
            "body": body,
            "attachments": data.get("attachments") or [],
            "draft_name": _sanitize_text(data.get("draft_name"), 255),
            "limit": int(data.get("limit") or 10),
            "message_id": _sanitize_text(data.get("message_id"), 255),
            "forward_to": data.get("forward_to") or [],
            "note": _sanitize_text(data.get("note"), 2000),
        },
    }


def _coerce_list_of_strings(value: Any) -> List[str]:
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            s = _sanitize_text(item, 500)
            if s:
                out.append(s)
        return out
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    return []


# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------

def _ensure_tables() -> None:
    if not _require_injected():
        return
    con = None
    try:
        con = _db()
        cur = con.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS comm_contacts (
                contact_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT NOT NULL,
                company TEXT NOT NULL,
                notes TEXT NOT NULL,
                tags_json TEXT NOT NULL,
                favorite INTEGER DEFAULT 0,
                created_ts REAL NOT NULL,
                updated_ts REAL NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_comm_contacts_user ON comm_contacts(user_id, updated_ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_comm_contacts_email ON comm_contacts(email)")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS comm_reminders (
                reminder_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                due_ts REAL DEFAULT 0,
                status TEXT NOT NULL,
                source TEXT NOT NULL,
                priority TEXT NOT NULL,
                related_type TEXT NOT NULL,
                related_id TEXT NOT NULL,
                created_ts REAL NOT NULL,
                updated_ts REAL NOT NULL,
                extra_json TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_comm_reminders_user ON comm_reminders(user_id, status, due_ts)")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS comm_calls (
                call_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                lane TEXT NOT NULL,
                direction TEXT NOT NULL,
                target TEXT NOT NULL,
                target_name TEXT NOT NULL,
                status TEXT NOT NULL,
                created_ts REAL NOT NULL,
                updated_ts REAL NOT NULL,
                ended_ts REAL DEFAULT 0,
                extra_json TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_comm_calls_user ON comm_calls(user_id, updated_ts)")

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS comm_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                user_id TEXT NOT NULL,
                lane TEXT NOT NULL,
                action TEXT NOT NULL,
                ref_id TEXT NOT NULL,
                details_json TEXT NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_comm_events_user ON comm_events(user_id, ts)")

        con.commit()
    except Exception:
        try:
            if con:
                con.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _log_event(user_id: str, lane: str, action: str, ref_id: str = "", details: Optional[Dict[str, Any]] = None) -> None:
    if not _require_injected():
        return
    con = None
    try:
        con = _db()
        con.execute(
            "INSERT INTO comm_events(ts, user_id, lane, action, ref_id, details_json) VALUES(?,?,?,?,?,?)",
            (_now(), user_id, lane, action, ref_id, json.dumps(details or {}, ensure_ascii=False)),
        )
        con.commit()
    except Exception as exc:
        logger.warning("appcomm event log failed: %s", exc)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Contacts
# -----------------------------------------------------------------------------

def _contact_row_to_obj(row: Any) -> Dict[str, Any]:
    try:
        tags = _safe_json(row[6], []) if not isinstance(row, sqlite3.Row) else _safe_json(row["tags_json"], [])
    except Exception:
        tags = []
    def _get(key_or_idx):
        try:
            if isinstance(row, sqlite3.Row):
                return row[key_or_idx]
            return row[key_or_idx]
        except Exception:
            return ""
    return {
        "contact_id": _get("contact_id" if isinstance(row, sqlite3.Row) else 0),
        "user_id": _get("user_id" if isinstance(row, sqlite3.Row) else 1),
        "name": _get("name" if isinstance(row, sqlite3.Row) else 2),
        "email": _get("email" if isinstance(row, sqlite3.Row) else 3),
        "phone": _get("phone" if isinstance(row, sqlite3.Row) else 4),
        "company": _get("company" if isinstance(row, sqlite3.Row) else 5),
        "notes": _get("notes" if isinstance(row, sqlite3.Row) else 6),
        "tags": tags if isinstance(tags, list) else [],
        "favorite": bool(_get("favorite" if isinstance(row, sqlite3.Row) else 7)),
        "created_ts": float(_get("created_ts" if isinstance(row, sqlite3.Row) else 8) or 0),
        "updated_ts": float(_get("updated_ts" if isinstance(row, sqlite3.Row) else 9) or 0),
    }


def _contacts_list(user_id: str, query: str = "", limit: int = MAX_CONTACTS_RETURN) -> List[Dict[str, Any]]:
    _ensure_tables()
    limit = max(1, min(MAX_CONTACTS_RETURN, int(limit or MAX_CONTACTS_RETURN)))
    con = _db()
    try:
        cur = con.cursor()
        if query.strip():
            q = f"%{query.strip().lower()}%"
            cur.execute(
                """
                SELECT contact_id, user_id, name, email, phone, company, notes, tags_json, favorite, created_ts, updated_ts
                FROM comm_contacts
                WHERE user_id=? AND (
                    lower(name) LIKE ? OR lower(email) LIKE ? OR lower(phone) LIKE ? OR lower(company) LIKE ? OR lower(notes) LIKE ?
                )
                ORDER BY favorite DESC, updated_ts DESC
                LIMIT ?
                """,
                (user_id, q, q, q, q, q, limit),
            )
        else:
            cur.execute(
                """
                SELECT contact_id, user_id, name, email, phone, company, notes, tags_json, favorite, created_ts, updated_ts
                FROM comm_contacts
                WHERE user_id=?
                ORDER BY favorite DESC, updated_ts DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
        return [_contact_row_to_obj(r) for r in (cur.fetchall() or [])]
    finally:
        try:
            con.close()
        except Exception:
            pass


def _contact_upsert(data: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_tables()
    user_id = _user_id_from_payload(data)
    contact_id = _sanitize_text(data.get("contact_id"), 255) or _new_id("contact")
    now = _now()
    row = {
        "contact_id": contact_id,
        "user_id": user_id,
        "name": _sanitize_text(data.get("name"), 255),
        "email": _normalize_email(data.get("email")),
        "phone": _sanitize_text(data.get("phone"), 100),
        "company": _sanitize_text(data.get("company"), 255),
        "notes": _sanitize_text(data.get("notes"), 4000),
        "tags_json": json.dumps(_coerce_list_of_strings(data.get("tags")), ensure_ascii=False),
        "favorite": 1 if bool(data.get("favorite")) else 0,
        "created_ts": now,
        "updated_ts": now,
    }
    if not row["name"] and not row["email"] and not row["phone"]:
        return {"ok": False, "error": "name_or_email_or_phone_required"}

    con = _db()
    try:
        cur = con.cursor()
        cur.execute("SELECT created_ts FROM comm_contacts WHERE contact_id=? LIMIT 1", (contact_id,))
        existing = cur.fetchone()
        created_ts = float(existing[0]) if existing else now
        row["created_ts"] = created_ts
        cur.execute(
            """
            INSERT INTO comm_contacts(contact_id,user_id,name,email,phone,company,notes,tags_json,favorite,created_ts,updated_ts)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(contact_id) DO UPDATE SET
                user_id=excluded.user_id,
                name=excluded.name,
                email=excluded.email,
                phone=excluded.phone,
                company=excluded.company,
                notes=excluded.notes,
                tags_json=excluded.tags_json,
                favorite=excluded.favorite,
                updated_ts=excluded.updated_ts
            """,
            (
                row["contact_id"], row["user_id"], row["name"], row["email"], row["phone"], row["company"],
                row["notes"], row["tags_json"], row["favorite"], row["created_ts"], row["updated_ts"],
            ),
        )
        con.commit()
        _log_event(user_id, "contacts", "upsert", contact_id, {"email": row["email"], "phone": row["phone"]})
        return {"ok": True, "contact": _contacts_list(user_id, query=row["email"] or row["name"], limit=1)[0]}
    finally:
        try:
            con.close()
        except Exception:
            pass


def _contact_delete(data: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_tables()
    user_id = _user_id_from_payload(data)
    contact_id = _sanitize_text(data.get("contact_id"), 255)
    if not contact_id:
        return {"ok": False, "error": "missing_contact_id"}
    con = _db()
    try:
        cur = con.cursor()
        cur.execute("DELETE FROM comm_contacts WHERE user_id=? AND contact_id=?", (user_id, contact_id))
        deleted = int(cur.rowcount or 0)
        con.commit()
        _log_event(user_id, "contacts", "delete", contact_id, {"deleted": deleted})
        return {"ok": True, "deleted": deleted, "contact_id": contact_id}
    finally:
        try:
            con.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Reminders
# -----------------------------------------------------------------------------

def _reminder_row_to_obj(row: Any) -> Dict[str, Any]:
    def _get(key_or_idx):
        try:
            if isinstance(row, sqlite3.Row):
                return row[key_or_idx]
            return row[key_or_idx]
        except Exception:
            return ""
    extra = _safe_json(_get("extra_json" if isinstance(row, sqlite3.Row) else 10), {})
    return {
        "reminder_id": _get("reminder_id" if isinstance(row, sqlite3.Row) else 0),
        "user_id": _get("user_id" if isinstance(row, sqlite3.Row) else 1),
        "title": _get("title" if isinstance(row, sqlite3.Row) else 2),
        "body": _get("body" if isinstance(row, sqlite3.Row) else 3),
        "due_ts": float(_get("due_ts" if isinstance(row, sqlite3.Row) else 4) or 0),
        "status": _get("status" if isinstance(row, sqlite3.Row) else 5),
        "source": _get("source" if isinstance(row, sqlite3.Row) else 6),
        "priority": _get("priority" if isinstance(row, sqlite3.Row) else 7),
        "related_type": _get("related_type" if isinstance(row, sqlite3.Row) else 8),
        "related_id": _get("related_id" if isinstance(row, sqlite3.Row) else 9),
        "created_ts": float(_get("created_ts" if isinstance(row, sqlite3.Row) else 10) or 0),
        "updated_ts": float(_get("updated_ts" if isinstance(row, sqlite3.Row) else 11) or 0),
        "extra": extra if isinstance(extra, dict) else {},
    }


def _reminders_list(user_id: str, status: str = "", limit: int = MAX_REMINDERS_RETURN) -> List[Dict[str, Any]]:
    _ensure_tables()
    limit = max(1, min(MAX_REMINDERS_RETURN, int(limit or MAX_REMINDERS_RETURN)))
    con = _db()
    try:
        cur = con.cursor()
        if status.strip():
            cur.execute(
                """
                SELECT reminder_id,user_id,title,body,due_ts,status,source,priority,related_type,related_id,created_ts,updated_ts,extra_json
                FROM comm_reminders
                WHERE user_id=? AND status=?
                ORDER BY CASE WHEN due_ts > 0 THEN due_ts ELSE updated_ts END ASC
                LIMIT ?
                """,
                (user_id, status.strip(), limit),
            )
        else:
            cur.execute(
                """
                SELECT reminder_id,user_id,title,body,due_ts,status,source,priority,related_type,related_id,created_ts,updated_ts,extra_json
                FROM comm_reminders
                WHERE user_id=?
                ORDER BY CASE WHEN due_ts > 0 THEN due_ts ELSE updated_ts END ASC
                LIMIT ?
                """,
                (user_id, limit),
            )
        return [_reminder_row_to_obj(r) for r in (cur.fetchall() or [])]
    finally:
        try:
            con.close()
        except Exception:
            pass


def _reminder_upsert(data: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_tables()
    user_id = _user_id_from_payload(data)
    reminder_id = _sanitize_text(data.get("reminder_id"), 255) or _new_id("reminder")
    title = _sanitize_text(data.get("title"), 255)
    body = _sanitize_text(data.get("body"), 4000)
    if not title:
        return {"ok": False, "error": "missing_title"}

    due_ts = 0.0
    raw_due = data.get("due_ts")
    if raw_due not in (None, ""):
        try:
            due_ts = float(raw_due)
        except Exception:
            due_ts = 0.0

    now = _now()
    status = _sanitize_text(data.get("status"), 50) or "active"
    source = _sanitize_text(data.get("source"), 100) or "appcomm"
    priority = _sanitize_text(data.get("priority"), 50) or "normal"
    related_type = _sanitize_text(data.get("related_type"), 100)
    related_id = _sanitize_text(data.get("related_id"), 255)
    extra_json = json.dumps(data.get("extra") if isinstance(data.get("extra"), dict) else {}, ensure_ascii=False)

    con = _db()
    try:
        cur = con.cursor()
        cur.execute("SELECT created_ts FROM comm_reminders WHERE reminder_id=? LIMIT 1", (reminder_id,))
        existing = cur.fetchone()
        created_ts = float(existing[0]) if existing else now
        cur.execute(
            """
            INSERT INTO comm_reminders(reminder_id,user_id,title,body,due_ts,status,source,priority,related_type,related_id,created_ts,updated_ts,extra_json)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(reminder_id) DO UPDATE SET
                user_id=excluded.user_id,
                title=excluded.title,
                body=excluded.body,
                due_ts=excluded.due_ts,
                status=excluded.status,
                source=excluded.source,
                priority=excluded.priority,
                related_type=excluded.related_type,
                related_id=excluded.related_id,
                updated_ts=excluded.updated_ts,
                extra_json=excluded.extra_json
            """,
            (reminder_id, user_id, title, body, due_ts, status, source, priority, related_type, related_id, created_ts, now, extra_json),
        )
        con.commit()
        _log_event(user_id, "reminders", "upsert", reminder_id, {"title": title, "status": status, "source": source})
        return {"ok": True, "reminder": _reminders_list(user_id, status="", limit=MAX_REMINDERS_RETURN)[0] if False else {
            "reminder_id": reminder_id,
            "user_id": user_id,
            "title": title,
            "body": body,
            "due_ts": due_ts,
            "status": status,
            "source": source,
            "priority": priority,
            "related_type": related_type,
            "related_id": related_id,
            "created_ts": created_ts,
            "updated_ts": now,
            "extra": _safe_json(extra_json, {}),
        }}
    finally:
        try:
            con.close()
        except Exception:
            pass


def _reminder_delete(data: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_tables()
    user_id = _user_id_from_payload(data)
    reminder_id = _sanitize_text(data.get("reminder_id"), 255)
    if not reminder_id:
        return {"ok": False, "error": "missing_reminder_id"}
    con = _db()
    try:
        cur = con.cursor()
        cur.execute("DELETE FROM comm_reminders WHERE user_id=? AND reminder_id=?", (user_id, reminder_id))
        deleted = int(cur.rowcount or 0)
        con.commit()
        _log_event(user_id, "reminders", "delete", reminder_id, {"deleted": deleted})
        return {"ok": True, "deleted": deleted, "reminder_id": reminder_id}
    finally:
        try:
            con.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# VoIP / Phone control surface (bounded, local-first)
# -----------------------------------------------------------------------------

def _voip_capabilities() -> Dict[str, Any]:
    return {
        "available": True,
        "local_only": True,
        "pstn_bridge": False,
        "sip_bridge": False,
        "webrtc_bridge": False,
        "status": "control_surface_only",
        "note": "VoIP/phone calls are modeled as bounded control records until a dedicated local adapter is mounted.",
    }


def _call_create(data: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_tables()
    user_id = _user_id_from_payload(data)
    lane = _sanitize_text(data.get("lane"), 30) or "voip"
    direction = _sanitize_text(data.get("direction"), 30) or "outbound"
    target = _sanitize_text(data.get("target"), 255)
    target_name = _sanitize_text(data.get("target_name"), 255)
    if not target:
        return {"ok": False, "error": "missing_target"}
    call_id = _new_id("call")
    now = _now()
    extra = json.dumps(data.get("extra") if isinstance(data.get("extra"), dict) else {}, ensure_ascii=False)
    con = _db()
    try:
        con.execute(
            "INSERT INTO comm_calls(call_id,user_id,lane,direction,target,target_name,status,created_ts,updated_ts,ended_ts,extra_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (call_id, user_id, lane, direction, target, target_name, "queued", now, now, 0.0, extra),
        )
        con.commit()
        _log_event(user_id, "voip", "create_call", call_id, {"lane": lane, "target": target})
        return {
            "ok": True,
            "call": {
                "call_id": call_id,
                "user_id": user_id,
                "lane": lane,
                "direction": direction,
                "target": target,
                "target_name": target_name,
                "status": "queued",
                "created_ts": now,
                "updated_ts": now,
                "ended_ts": 0.0,
                "extra": _safe_json(extra, {}),
            },
        }
    finally:
        try:
            con.close()
        except Exception:
            pass


def _call_update_status(data: Dict[str, Any]) -> Dict[str, Any]:
    _ensure_tables()
    user_id = _user_id_from_payload(data)
    call_id = _sanitize_text(data.get("call_id"), 255)
    status = _sanitize_text(data.get("status"), 50)
    if not call_id or not status:
        return {"ok": False, "error": "missing_call_id_or_status"}
    now = _now()
    ended_ts = now if status in ("ended", "rejected", "failed", "missed") else 0.0
    con = _db()
    try:
        cur = con.cursor()
        cur.execute(
            "UPDATE comm_calls SET status=?, updated_ts=?, ended_ts=CASE WHEN ? > 0 THEN ? ELSE ended_ts END WHERE user_id=? AND call_id=?",
            (status, now, ended_ts, ended_ts, user_id, call_id),
        )
        updated = int(cur.rowcount or 0)
        con.commit()
        _log_event(user_id, "voip", "update_call", call_id, {"status": status, "updated": updated})
        return {"ok": True, "updated": updated, "call_id": call_id, "status": status}
    finally:
        try:
            con.close()
        except Exception:
            pass


def _calls_list(user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    _ensure_tables()
    limit = max(1, min(500, int(limit or 100)))
    con = _db()
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT call_id,user_id,lane,direction,target,target_name,status,created_ts,updated_ts,ended_ts,extra_json FROM comm_calls WHERE user_id=? ORDER BY updated_ts DESC LIMIT ?",
            (user_id, limit),
        )
        rows = cur.fetchall() or []
        out: List[Dict[str, Any]] = []
        for r in rows:
            extra = _safe_json(r[10], {}) if not isinstance(r, sqlite3.Row) else _safe_json(r["extra_json"], {})
            if isinstance(r, sqlite3.Row):
                out.append({
                    "call_id": r["call_id"], "user_id": r["user_id"], "lane": r["lane"], "direction": r["direction"],
                    "target": r["target"], "target_name": r["target_name"], "status": r["status"],
                    "created_ts": float(r["created_ts"] or 0), "updated_ts": float(r["updated_ts"] or 0),
                    "ended_ts": float(r["ended_ts"] or 0), "extra": extra if isinstance(extra, dict) else {},
                })
            else:
                out.append({
                    "call_id": r[0], "user_id": r[1], "lane": r[2], "direction": r[3],
                    "target": r[4], "target_name": r[5], "status": r[6],
                    "created_ts": float(r[7] or 0), "updated_ts": float(r[8] or 0),
                    "ended_ts": float(r[9] or 0), "extra": extra if isinstance(extra, dict) else {},
                })
        return out
    finally:
        try:
            con.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Email wrappers
# -----------------------------------------------------------------------------

def _email_capabilities_payload() -> Dict[str, Any]:
    svc = _email_service()
    module = _email_module_info()
    status: Dict[str, Any] = {}
    try:
        if svc is not None and hasattr(svc, "get_status"):
            status = svc.get_status()  # type: ignore[assignment]
    except Exception as exc:
        status = {"status": "error", "error": str(exc)}
    return {
        "lane": "communications",
        "service": "email",
        "module": module,
        "status": status,
        "voip": _voip_capabilities(),
    }


def _email_call_lane(data: Dict[str, Any], action: str) -> Dict[str, Any]:
    mod = _email_module()
    if mod is None:
        return {"ok": False, "error": "SarahMemoryEmail unavailable", "module": _email_module_info()}

    context = _email_context(data, action)
    try:
        entry = getattr(mod, "email_lane_entry", None)
        if callable(entry):
            result = entry(context)
            if isinstance(result, dict):
                return result
    except Exception as exc:
        logger.exception("email_lane_entry failed: %s", exc)
        return {"ok": False, "error": str(exc), "module": _email_module_info(), "context": context}
    return {"ok": False, "error": "email_lane_entry unavailable", "module": _email_module_info(), "context": context}


def _email_send_direct(data: Dict[str, Any]) -> Dict[str, Any]:
    svc = _email_service()
    if svc is None or not hasattr(svc, "send_email"):
        return {"ok": False, "error": "email_service_unavailable"}
    to_addrs = _coerce_list_of_strings(data.get("to") or data.get("to_addrs") or [])
    if not to_addrs:
        to_single = _normalize_email(data.get("to"))
        if to_single:
            to_addrs = [to_single]
    subject = _sanitize_text(data.get("subject"), 500)
    body = _sanitize_text(data.get("body"), 20000)
    try:
        result = svc.send_email(to_addrs, subject, body)  # type: ignore[misc]
        if isinstance(result, dict):
            return {"ok": bool(result.get("status") in ("ok", "sent") or result.get("ok")), "result": result}
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _email_poll(data: Dict[str, Any]) -> Dict[str, Any]:
    svc = _email_service()
    if svc is None or not hasattr(svc, "poll_inbox"):
        return {"ok": False, "error": "email_service_unavailable"}
    limit = max(1, min(MAX_EMAIL_LIST_RETURN, int(data.get("limit") or 10)))
    try:
        result = svc.poll_inbox(limit=limit)  # type: ignore[misc]
        if isinstance(result, dict):
            return {"ok": bool(result.get("status") == "ok"), "result": result}
        return {"ok": True, "result": result}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# -----------------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------------

def init_app(app, connect_sqlite, meta_db_path, api_key_auth_ok=None, sign_ok=None):
    global _CONNECT_SQLITE, _META_DB, _API_KEY_AUTH_OK, _SIGN_OK
    _CONNECT_SQLITE = connect_sqlite
    _META_DB = meta_db_path
    _API_KEY_AUTH_OK = api_key_auth_ok
    _SIGN_OK = sign_ok
    try:
        _ensure_tables()
    except Exception as exc:
        logger.warning("appcomm init failed to ensure tables: %s", exc)
    if "appcomm_v800" in getattr(app, "blueprints", {}):
        return True
    app.register_blueprint(bp)
    return True


# -----------------------------------------------------------------------------
# Health / capabilities
# -----------------------------------------------------------------------------


@bp.route("/api/comm/governance", methods=["GET", "OPTIONS"])
def comm_governance():
    """Read-only communications lane governance status.

    Communication content is data. It must never become executable command
    input without a separately governed user/developer approval path.
    """
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    return _ok(
        {
            "api_domain": "comm",
            "route_base": "/api/comm",
            "governance": {
                "local_first": True,
                "content_executes_commands": False,
                "requires_auth_for_mutations": True,
                "bounded_content_policy": {
                    "email_body_is_data": True,
                    "contacts_are_data": True,
                    "reminders_are_data": True,
                    "voip_calls_require_explicit_action": True,
                },
                "safety_notes": [
                    "Email/contact/reminder content is not treated as executable instructions.",
                    "Outbound actions require governed endpoints and configured credentials.",
                ],
            },
        }
    )

@bp.route("/api/comm/health", methods=["GET", "OPTIONS"])
def comm_health():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    try:
        _ensure_tables()
        return _ok(
            service="communications",
            ts=_now(),
            version=PROJECT_VERSION,
            email_module=_email_module_info(),
            voip=_voip_capabilities(),
        )
    except Exception as exc:
        return _err(str(exc), 500, service="communications", version=PROJECT_VERSION)


@bp.route("/api/comm/capabilities", methods=["GET", "OPTIONS"])
def comm_capabilities():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    return _ok(
        service="communications",
        version=PROJECT_VERSION,
        domains={
            "email": _email_capabilities_payload(),
            "contacts": {"available": True, "storage": "sqlite_meta_db"},
            "reminders": {"available": True, "storage": "sqlite_meta_db"},
            "voip": _voip_capabilities(),
        },
    )


# -----------------------------------------------------------------------------
# Email routes
# -----------------------------------------------------------------------------

@bp.route("/api/comm/email/capabilities", methods=["GET", "OPTIONS"])
def comm_email_capabilities():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    return _ok(**_email_capabilities_payload())


@bp.route("/api/comm/email/status", methods=["GET", "OPTIONS"])
def comm_email_status():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    svc = _email_service()
    if svc is None:
        return _err("SarahMemoryEmail unavailable", 503, module=_email_module_info())
    try:
        if hasattr(svc, "initialize"):
            init = svc.initialize()  # type: ignore[misc]
        else:
            init = {"status": "ok"}
        status = svc.get_status() if hasattr(svc, "get_status") else {}  # type: ignore[misc]
        return _ok(service="email", status=status, init=init, module=_email_module_info())
    except Exception as exc:
        return _err(str(exc), 500, service="email", module=_email_module_info())


@bp.route("/api/comm/email/action", methods=["POST", "OPTIONS"])
def comm_email_action():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    raw = _body_bytes()
    if not _verify_auth(raw):
        return _err("Unauthorized", 401)
    data = _j()
    action = _sanitize_text(data.get("action"), 50).lower()
    if action not in (
        "draft_email", "send_email", "list_emails", "poll_inbox", "get_status", "forward_message",
    ):
        return _err("Unsupported email action", 400, allowed_actions=["draft_email", "send_email", "list_emails", "poll_inbox", "get_status", "forward_message"])

    if action == "send_email":
        result = _email_send_direct(data)
    elif action in ("list_emails", "poll_inbox"):
        result = _email_poll(data)
    elif action == "get_status":
        svc = _email_service()
        result = {"ok": bool(svc), "result": svc.get_status() if svc and hasattr(svc, "get_status") else {}}  # type: ignore[misc]
    else:
        result = _email_call_lane(data, action)

    if result.get("ok"):
        return _ok(service="email", action=action, **result)
    return _err(result.get("error") or "Email action failed", 400, service="email", action=action, **result)


@bp.route("/api/comm/email/send", methods=["POST", "OPTIONS"])
def comm_email_send():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    raw = _body_bytes()
    if not _verify_auth(raw):
        return _err("Unauthorized", 401)
    data = _j()
    result = _email_send_direct(data)
    if result.get("ok"):
        return _ok(service="email", action="send_email", **result)
    return _err(result.get("error") or "Email send failed", 400, service="email", action="send_email", **result)


@bp.route("/api/comm/email/list", methods=["POST", "OPTIONS"])
def comm_email_list():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    raw = _body_bytes()
    if not _verify_auth(raw):
        return _err("Unauthorized", 401)
    data = _j()
    result = _email_poll(data)
    if result.get("ok"):
        return _ok(service="email", action="list_emails", **result)
    return _err(result.get("error") or "Email list failed", 400, service="email", action="list_emails", **result)


# -----------------------------------------------------------------------------
# Contacts routes
# -----------------------------------------------------------------------------

@bp.route("/api/comm/contacts/list", methods=["GET", "POST", "OPTIONS"])
def comm_contacts_list():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    data = _j() if request.method == "POST" else {k: request.args.get(k) for k in request.args.keys()}
    user_id = _user_id_from_payload(data)
    query = _sanitize_text(data.get("query"), 255)
    limit = int(data.get("limit") or MAX_CONTACTS_RETURN)
    contacts = _contacts_list(user_id, query=query, limit=limit)
    return _ok(service="contacts", contacts=contacts, count=len(contacts), user_id=user_id)


@bp.route("/api/comm/contacts/save", methods=["POST", "OPTIONS"])
def comm_contacts_save():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    raw = _body_bytes()
    if not _verify_auth(raw):
        return _err("Unauthorized", 401)
    data = _j()
    result = _contact_upsert(data)
    if result.get("ok"):
        return _ok(service="contacts", **result)
    return _err(result.get("error") or "Contact save failed", 400, service="contacts", **result)


@bp.route("/api/comm/contacts/delete", methods=["POST", "OPTIONS"])
def comm_contacts_delete():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    raw = _body_bytes()
    if not _verify_auth(raw):
        return _err("Unauthorized", 401)
    data = _j()
    result = _contact_delete(data)
    if result.get("ok"):
        return _ok(service="contacts", **result)
    return _err(result.get("error") or "Contact delete failed", 400, service="contacts", **result)


# -----------------------------------------------------------------------------
# Reminder routes
# -----------------------------------------------------------------------------

@bp.route("/api/comm/reminders/list", methods=["GET", "POST", "OPTIONS"])
def comm_reminders_list():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    data = _j() if request.method == "POST" else {k: request.args.get(k) for k in request.args.keys()}
    user_id = _user_id_from_payload(data)
    status = _sanitize_text(data.get("status"), 50)
    limit = int(data.get("limit") or MAX_REMINDERS_RETURN)
    reminders = _reminders_list(user_id, status=status, limit=limit)
    return _ok(service="reminders", reminders=reminders, count=len(reminders), user_id=user_id)


@bp.route("/api/comm/reminders/save", methods=["POST", "OPTIONS"])
def comm_reminders_save():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    raw = _body_bytes()
    if not _verify_auth(raw):
        return _err("Unauthorized", 401)
    data = _j()
    result = _reminder_upsert(data)
    if result.get("ok"):
        return _ok(service="reminders", **result)
    return _err(result.get("error") or "Reminder save failed", 400, service="reminders", **result)


@bp.route("/api/comm/reminders/delete", methods=["POST", "OPTIONS"])
def comm_reminders_delete():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    raw = _body_bytes()
    if not _verify_auth(raw):
        return _err("Unauthorized", 401)
    data = _j()
    result = _reminder_delete(data)
    if result.get("ok"):
        return _ok(service="reminders", **result)
    return _err(result.get("error") or "Reminder delete failed", 400, service="reminders", **result)


# -----------------------------------------------------------------------------
# VoIP / phone routes (control surface only)
# -----------------------------------------------------------------------------

@bp.route("/api/comm/voip/capabilities", methods=["GET", "OPTIONS"])
def comm_voip_capabilities():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    return _ok(service="voip", **_voip_capabilities())


@bp.route("/api/comm/voip/call/start", methods=["POST", "OPTIONS"])
def comm_voip_call_start():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    raw = _body_bytes()
    if not _verify_auth(raw):
        return _err("Unauthorized", 401)
    data = _j()
    result = _call_create(data)
    if result.get("ok"):
        return _ok(service="voip", action="start_call", **result)
    return _err(result.get("error") or "Call start failed", 400, service="voip", action="start_call", **result)


@bp.route("/api/comm/voip/call/update", methods=["POST", "OPTIONS"])
def comm_voip_call_update():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    raw = _body_bytes()
    if not _verify_auth(raw):
        return _err("Unauthorized", 401)
    data = _j()
    result = _call_update_status(data)
    if result.get("ok"):
        return _ok(service="voip", action="update_call", **result)
    return _err(result.get("error") or "Call update failed", 400, service="voip", action="update_call", **result)


@bp.route("/api/comm/voip/calls", methods=["GET", "POST", "OPTIONS"])
def comm_voip_calls():
    if request.method == "OPTIONS":
        return _ok(preflight=True)
    data = _j() if request.method == "POST" else {k: request.args.get(k) for k in request.args.keys()}
    user_id = _user_id_from_payload(data)
    limit = int(data.get("limit") or 100)
    calls = _calls_list(user_id, limit=limit)
    return _ok(service="voip", calls=calls, count=len(calls), user_id=user_id)

# ====================================================================
# END OF appcomm.py v9.0.0
# ====================================================================
