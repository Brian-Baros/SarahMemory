"""
--==The SarahMemory Project==--
File: SarahMemoryEmail.py
Part of the SarahMemory Companion AI-bot Platform
Version: v8.0.0
Date: 2026-03-11
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.

PURPOSE:
--------
SarahMemoryEmail is a local-first, sealed communications lane for mailbox intake,
classification, storage, forwarding, and safe auto-reply.

SECURITY MODEL:
---------------
- Email is CONTENT, not COMMANDS.
- No email text or attachment may directly execute system actions.
- Allowed bounded actions are limited to read/reply/store/forward/classify,
  plus reminder/calendar extraction when explicitly enabled.
- Attachments are staged and scanned before any downstream use.
- SMTP/IMAP credentials live in .env, not in SarahMemoryGlobals.py.

SUPPORTED PATH:
---------------
- IMAP-first inbound mailbox processing
- SMTP outbound replies/forwards
- Optional POP3 fallback (disabled by default)
- Reminder/calendar extraction via SarahMemoryReminder.py
- Attachment staging/scanning via SarahMemoryFilesystem.py
"""

from __future__ import annotations
# --- SARAHMETA START ---
# GRADE = "C"
# ROLE = "capability"
# CATEGORY = "communication"
# USER_FACING = True
# UI_EXPOSURE = "candidate"
# DEPLOYMENT_TARGET = "addon"
# API_DOMAIN = "comm"
# HARDWARE_DOMAIN = ""
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "email"
# FAMILY = "communications"
# GOVERNANCE_LEVEL = "bounded"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = True
# ADDON_CANDIDATE = True
# DRIVER_CANDIDATE = False
# NOTES = "Local-first sealed email lane for mailbox intake, classification, storage, forwarding, and safe auto-reply. Email is content, not commands."
# --- SARAHMETA END ---
import os
import re
import sys
import ssl
import json
import time
import html
import email
import imaplib
import poplib
import shutil
import sqlite3
import logging
import hashlib
import smtplib
import mimetypes
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from email import policy
from email.header import decode_header, make_header
from email.message import EmailMessage, Message
from email.utils import getaddresses, parseaddr, formataddr, make_msgid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None

try:
    import SarahMemoryReminder as reminder_mod  # type: ignore
except Exception:
    reminder_mod = None

try:
    import SarahMemoryFilesystem as filesystem_mod  # type: ignore
except Exception:
    filesystem_mod = None

try:
    import SarahMemoryReply as reply_mod  # type: ignore
except Exception:
    reply_mod = None

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("SarahMemoryEmail")
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - [%(name)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG if bool(getattr(config, "DEBUG_MODE", True)) else logging.INFO)
logger.propagate = False


# -----------------------------------------------------------------------------
# Helpers / config
# -----------------------------------------------------------------------------
def _env(name: str, default: str = "") -> str:
    try:
        val = os.getenv(name, default)
        return "" if val is None else str(val)
    except Exception:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    try:
        raw = _env(name, "true" if default else "false").strip().lower()
        return raw in ("1", "true", "yes", "on")
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(_env(name, str(default))).strip())
    except Exception:
        return default


def _base_dir() -> str:
    try:
        return str(getattr(config, "BASE_DIR", os.getcwd()))
    except Exception:
        return os.getcwd()


def _data_dir() -> str:
    try:
        return str(getattr(config, "DATA_DIR", os.path.join(_base_dir(), "data")))
    except Exception:
        return os.path.join(_base_dir(), "data")


def _datasets_dir() -> str:
    try:
        return str(getattr(config, "DATASETS_DIR", os.path.join(_data_dir(), "memory", "datasets")))
    except Exception:
        return os.path.join(_data_dir(), "memory", "datasets")


def _email_root() -> str:
    return os.path.join(_data_dir(), "email")


def _email_db_path() -> str:
    return os.path.join(_datasets_dir(), "email_system.db")


def _safe_bool_config(name: str, env_name: str, default: bool = False) -> bool:
    try:
        if config is not None and hasattr(config, name):
            return bool(getattr(config, name))
    except Exception:
        pass
    return _env_flag(env_name, default)


EMAIL_ENABLED = _safe_bool_config("EMAIL_ENABLED", "EMAIL_ENABLED", False)
EMAIL_AUTOREPLY_ENABLED = _safe_bool_config("EMAIL_AUTOREPLY_ENABLED", "EMAIL_AUTOREPLY_ENABLED", False)
EMAIL_FORWARDING_ENABLED = _safe_bool_config("EMAIL_FORWARDING_ENABLED", "EMAIL_FORWARDING_ENABLED", True)
EMAIL_REMINDER_PARSE_ENABLED = _safe_bool_config("EMAIL_REMINDER_PARSE_ENABLED", "EMAIL_REMINDER_PARSE_ENABLED", True)
EMAIL_CALENDAR_PARSE_ENABLED = _safe_bool_config("EMAIL_CALENDAR_PARSE_ENABLED", "EMAIL_CALENDAR_PARSE_ENABLED", True)
EMAIL_KNOWLEDGE_EXTRACT_ENABLED = _safe_bool_config("EMAIL_KNOWLEDGE_EXTRACT_ENABLED", "EMAIL_KNOWLEDGE_EXTRACT_ENABLED", True)
EMAIL_UNSUBSCRIBE_ENABLED = _safe_bool_config("EMAIL_UNSUBSCRIBE_ENABLED", "EMAIL_UNSUBSCRIBE_ENABLED", False)
EMAIL_ALLOW_ATTACHMENTS = _safe_bool_config("EMAIL_ALLOW_ATTACHMENTS", "EMAIL_ALLOW_ATTACHMENTS", True)
EMAIL_ALLOW_HTML = _safe_bool_config("EMAIL_ALLOW_HTML", "EMAIL_ALLOW_HTML", False)
EMAIL_IMAP_ENABLED = _safe_bool_config("EMAIL_IMAP_ENABLED", "EMAIL_IMAP_ENABLED", True)
EMAIL_SMTP_ENABLED = _safe_bool_config("EMAIL_SMTP_ENABLED", "EMAIL_SMTP_ENABLED", True)
EMAIL_ALLOW_POP3_FALLBACK = _safe_bool_config("EMAIL_ALLOW_POP3_FALLBACK", "EMAIL_ALLOW_POP3_FALLBACK", False)
EMAIL_COMMAND_EXECUTION_ENABLED = False  # hard lock by design

EMAIL_PROTOCOL = _env("EMAIL_PROTOCOL", "imap").strip().lower() or "imap"
EMAIL_POLL_SECONDS = max(30, _env_int("EMAIL_POLL_SECONDS", 120))
EMAIL_JUNK_RETENTION_DAYS = max(1, _env_int("EMAIL_JUNK_RETENTION_DAYS", 7))
EMAIL_MAX_BODY_CHARS = max(1000, _env_int("EMAIL_MAX_BODY_CHARS", 10000))
EMAIL_MAX_MESSAGES_PER_POLL = max(1, _env_int("EMAIL_MAX_MESSAGES_PER_POLL", 10))
EMAIL_AUTOREPLY_SUBJECT_PREFIX = _env("EMAIL_AUTOREPLY_SUBJECT_PREFIX", "Re:") or "Re:"
EMAIL_USER_ID = _env("EMAIL_USER_ID", "default") or "default"

SMTP_HOST = _env("SMTP_HOST", "")
SMTP_PORT = _env_int("SMTP_PORT", 587)
SMTP_USER = _env("SMTP_USER", "")
SMTP_PASSWORD = _env("SMTP_PASSWORD", "")
SMTP_STARTTLS = _env_flag("SMTP_STARTTLS", True)
FROM_EMAIL = _env("FROM_EMAIL", _env("AI_EMAIL_ADDRESS", ""))

IMAP_HOST = _env("IMAP_HOST", "")
IMAP_PORT = _env_int("IMAP_PORT", 993)
IMAP_USER = _env("IMAP_USER", SMTP_USER)
IMAP_PASSWORD = _env("IMAP_PASSWORD", SMTP_PASSWORD)
IMAP_SSL = _env_flag("IMAP_SSL", True)

POP3_HOST = _env("POP3_HOST", "")
POP3_PORT = _env_int("POP3_PORT", 995)
POP3_USER = _env("POP3_USER", SMTP_USER)
POP3_PASSWORD = _env("POP3_PASSWORD", SMTP_PASSWORD)
POP3_SSL = _env_flag("POP3_SSL", True)

INBOX_FOLDER = _env("EMAIL_INBOX_FOLDER", "INBOX")
JUNK_FOLDER = _env("EMAIL_JUNK_FOLDER", "Junk")
ARCHIVE_FOLDER = _env("EMAIL_ARCHIVE_FOLDER", "Archive")
AUTOPROCESSED_FOLDER = _env("EMAIL_AUTOPROCESSED_FOLDER", "SarahMemory/AutoProcessed")
HOLD_FOLDER = _env("EMAIL_HOLD_FOLDER", "SarahMemory/Hold")

KNOWN_NO_REPLY_MARKERS = (
    "no-reply", "noreply", "do-not-reply", "donotreply", "mailer-daemon", "postmaster"
)
SPAM_KEYWORDS = (
    "unsubscribe", "buy now", "limited time", "free money", "winner", "act now",
    "claim now", "risk free", "lowest price", "viagra", "lottery", "promo"
)
SYSTEM_COMMAND_KEYWORDS = (
    "shutdown", "restart", "reboot", "delete", "remove file", "copy file", "run", "execute",
    "install", "powershell", "cmd.exe", "terminal", "shell", "format disk", "wipe", "drop table"
)
REMINDER_HINT_WORDS = (
    "remind", "appointment", "meeting", "birthday", "anniversary", "schedule", "doctor",
    "clinic", "today", "tomorrow", "weekend", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "next week", "later", "date", "time", "cancel", "add"
)


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass
class EmailAttachmentRecord:
    filename: str
    content_type: str
    size_bytes: int
    saved_path: str
    sha256: str
    scan_status: str
    scan_details: str


@dataclass
class ParsedEmail:
    message_id: str
    subject: str
    sender_name: str
    sender_email: str
    to_addrs: List[str]
    cc_addrs: List[str]
    reply_to: List[str]
    sent_at: str
    body_text: str
    body_html: str
    body_sanitized: str
    is_spam_candidate: bool
    is_no_reply: bool
    contains_system_command_language: bool
    reminder_candidate: bool
    knowledge_candidate: bool
    attachments: List[EmailAttachmentRecord]
    raw_headers: Dict[str, str]


# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------
_DB: Optional[sqlite3.Connection] = None


def _ensure_dirs() -> None:
    for path in (
        _datasets_dir(),
        _email_root(),
        os.path.join(_email_root(), "attachments", "incoming"),
        os.path.join(_email_root(), "attachments", "safe"),
        os.path.join(_email_root(), "attachments", "quarantine"),
        os.path.join(_email_root(), "exports"),
        os.path.join(_email_root(), "logs"),
    ):
        os.makedirs(path, exist_ok=True)


def _get_db() -> sqlite3.Connection:
    global _DB
    if _DB is not None:
        return _DB
    _ensure_dirs()
    _DB = sqlite3.connect(_email_db_path(), check_same_thread=False, timeout=5.0)
    _DB.row_factory = sqlite3.Row
    return _DB


def init_db() -> None:
    db = _get_db()
    cur = db.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS email_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT UNIQUE,
            direction TEXT,
            sender_email TEXT,
            sender_name TEXT,
            recipients_json TEXT,
            subject TEXT,
            sent_at TEXT,
            received_at TEXT,
            body_text TEXT,
            body_sanitized TEXT,
            status TEXT,
            folder_name TEXT,
            is_spam_candidate INTEGER DEFAULT 0,
            is_no_reply INTEGER DEFAULT 0,
            contains_system_command_language INTEGER DEFAULT 0,
            reminder_candidate INTEGER DEFAULT 0,
            knowledge_candidate INTEGER DEFAULT 0,
            auto_replied INTEGER DEFAULT 0,
            forwarded INTEGER DEFAULT 0,
            extra_json TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS email_attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT,
            filename TEXT,
            content_type TEXT,
            size_bytes INTEGER,
            saved_path TEXT,
            sha256 TEXT,
            scan_status TEXT,
            scan_details TEXT,
            created_at TEXT,
            FOREIGN KEY(message_id) REFERENCES email_messages(message_id)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS email_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            event_type TEXT,
            message_id TEXT,
            details TEXT
        )
        """
    )
    db.commit()


def log_email_event(event_type: str, message_id: str = "", details: Any = "") -> None:
    try:
        db = _get_db()
        db.execute(
            "INSERT INTO email_events (ts, event_type, message_id, details) VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), event_type, message_id, json.dumps(details, ensure_ascii=False) if not isinstance(details, str) else details),
        )
        db.commit()
    except Exception as exc:
        logger.error(f"Could not write email event log: {exc}")


# -----------------------------------------------------------------------------
# Safe parsing / sanitization
# -----------------------------------------------------------------------------
def _decode_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        for enc in ("utf-8", "latin-1", "ascii"):
            try:
                return value.decode(enc, errors="replace")
            except Exception:
                continue
        return value.decode(errors="replace")
    return str(value)


def _decode_mime_header(value: Optional[str]) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value))).strip()
    except Exception:
        return _decode_text(value).strip()


def _sanitize_text(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", "[link]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > EMAIL_MAX_BODY_CHARS:
        text = text[:EMAIL_MAX_BODY_CHARS] + " ..."
    return text


def _extract_message_bodies(msg: Message) -> Tuple[str, str]:
    body_text = ""
    body_html = ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = (part.get_content_type() or "").lower()
            disposition = (part.get_content_disposition() or "").lower()
            if disposition == "attachment":
                continue
            try:
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                decoded = payload.decode(charset, errors="replace") if payload else ""
            except Exception:
                decoded = _decode_text(part.get_payload(decode=True) or part.get_payload())
            if content_type == "text/plain" and not body_text:
                body_text = decoded
            elif content_type == "text/html" and not body_html:
                body_html = decoded
    else:
        try:
            payload = msg.get_payload(decode=True)
            charset = msg.get_content_charset() or "utf-8"
            decoded = payload.decode(charset, errors="replace") if payload else ""
        except Exception:
            decoded = _decode_text(msg.get_payload(decode=True) or msg.get_payload())
        if (msg.get_content_type() or "").lower() == "text/html":
            body_html = decoded
        else:
            body_text = decoded
    return body_text or _sanitize_text(body_html), body_html


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_spam_candidate(subject: str, body: str, sender_email: str) -> bool:
    hay = f"{subject} {body} {sender_email}".lower()
    spam_hits = sum(1 for kw in SPAM_KEYWORDS if kw in hay)
    return spam_hits >= 2 or sender_email.lower().endswith((".ru", ".xyz"))


def _is_no_reply_address(sender_email: str) -> bool:
    val = (sender_email or "").lower()
    return any(marker in val for marker in KNOWN_NO_REPLY_MARKERS)


def _contains_system_command_language(text: str) -> bool:
    val = (text or "").lower()
    return any(token in val for token in SYSTEM_COMMAND_KEYWORDS)


def _is_reminder_candidate(text: str) -> bool:
    val = (text or "").lower()
    return any(token in val for token in REMINDER_HINT_WORDS)


def _extract_useful_knowledge(subject: str, body: str) -> Dict[str, Any]:
    clean = _sanitize_text(f"{subject} {body}")
    lines = [chunk.strip() for chunk in re.split(r"[.!?]\s+", clean) if chunk.strip()]
    salient = [ln for ln in lines if len(ln) > 20][:8]
    return {
        "summary": " | ".join(salient[:3]),
        "facts": salient,
    }


def _reply_targets(msg: ParsedEmail) -> List[str]:
    if msg.reply_to:
        return [addr for addr in msg.reply_to if addr]
    return [msg.sender_email] if msg.sender_email else []


# -----------------------------------------------------------------------------
# Attachment staging / scanning
# -----------------------------------------------------------------------------
def _scanner() -> Optional[Any]:
    try:
        if filesystem_mod is None:
            return None
        return filesystem_mod.FileScanner()
    except Exception:
        return None


def _stage_attachment(part: Message, message_id: str) -> Optional[EmailAttachmentRecord]:
    if not EMAIL_ALLOW_ATTACHMENTS:
        return None
    filename = _decode_mime_header(part.get_filename() or "attachment.bin") or "attachment.bin"
    payload = part.get_payload(decode=True) or b""
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", filename)
    msg_bucket = re.sub(r"[^A-Za-z0-9._-]", "_", message_id or str(int(time.time())))
    incoming_dir = os.path.join(_email_root(), "attachments", "incoming", msg_bucket)
    os.makedirs(incoming_dir, exist_ok=True)
    file_path = os.path.join(incoming_dir, safe_name)
    with open(file_path, "wb") as handle:
        handle.write(payload)

    sha = _sha256_bytes(payload)
    scan_status = "not_scanned"
    scan_details = ""

    scanner = _scanner()
    if scanner is not None:
        try:
            result = scanner.scan_file(file_path, quarantine_on_threat=True)
            threat_level = str(result.get("threat_level", "clean"))
            threats = "; ".join(result.get("threats", []) or [])
            if threat_level in ("critical", "high"):
                scan_status = "quarantined"
                scan_details = threats or threat_level
            else:
                scan_status = threat_level or "clean"
                scan_details = threats
        except Exception as exc:
            scan_status = "scan_error"
            scan_details = str(exc)
    else:
        scan_status = "scanner_unavailable"
        scan_details = "SarahMemoryFilesystem unavailable"

    return EmailAttachmentRecord(
        filename=safe_name,
        content_type=part.get_content_type() or mimetypes.guess_type(safe_name)[0] or "application/octet-stream",
        size_bytes=len(payload),
        saved_path=file_path,
        sha256=sha,
        scan_status=scan_status,
        scan_details=scan_details,
    )


# -----------------------------------------------------------------------------
# Reminder/calendar extraction
# -----------------------------------------------------------------------------
def _process_reminder_calendar_text(text: str) -> Dict[str, Any]:
    if reminder_mod is None:
        return {"status": "skipped", "reason": "SarahMemoryReminder unavailable"}
    if not text.strip():
        return {"status": "skipped", "reason": "empty_text"}
    try:
        return reminder_mod.process_calendar_command(text, user_id=EMAIL_USER_ID)
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


# -----------------------------------------------------------------------------
# Reply generation
# -----------------------------------------------------------------------------
def _default_reply_body(parsed: ParsedEmail, extracted: Dict[str, Any]) -> str:
    pieces = [
        f"Hello {parsed.sender_name or 'there'},",
        "",
        "Your email was received by SarahMemory.",
    ]
    if extracted.get("status") == "ok":
        pieces.append("I also identified a reminder or calendar item and stored it for review.")
    elif extracted.get("status") == "unknown":
        pieces.append("I reviewed the message, but I did not identify a clear calendar or reminder action.")
    if parsed.is_spam_candidate:
        pieces.append("This message appears promotional or automated, so no autonomous reply actions beyond acknowledgment were taken.")
    pieces.extend([
        "",
        "Regards,",
        "SarahMemory AiOS",
    ])
    return "\n".join(pieces).strip()


def _generate_reply_body(parsed: ParsedEmail, extracted: Dict[str, Any]) -> str:
    if reply_mod is None:
        return _default_reply_body(parsed, extracted)

    # Best-effort bridge without assuming a specific SarahMemoryReply API surface.
    prompt = (
        "Email reply request. Keep it safe, concise, and local-first. "
        "Do not expose internals, secrets, paths, stack traces, or system capability details.\n\n"
        f"Sender: {parsed.sender_name} <{parsed.sender_email}>\n"
        f"Subject: {parsed.subject}\n"
        f"Body: {parsed.body_sanitized}\n"
        f"Reminder extraction: {json.dumps(extracted, ensure_ascii=False)}"
    )
    try:
        if hasattr(reply_mod, "generate_reply"):
            out = reply_mod.generate_reply(prompt)
            if isinstance(out, str) and out.strip():
                return out.strip()
        if hasattr(reply_mod, "reply_to_user"):
            out = reply_mod.reply_to_user(prompt)
            if isinstance(out, str) and out.strip():
                return out.strip()
    except Exception as exc:
        logger.warning(f"SarahMemoryReply bridge failed, using default reply: {exc}")
    return _default_reply_body(parsed, extracted)


# -----------------------------------------------------------------------------
# Core mailbox service
# -----------------------------------------------------------------------------
class SarahMemoryEmailService:
    def __init__(self) -> None:
        _ensure_dirs()
        init_db()
        self.last_poll_at: Optional[str] = None

    def enabled(self) -> bool:
        return bool(EMAIL_ENABLED and EMAIL_SMTP_ENABLED and (EMAIL_IMAP_ENABLED or EMAIL_ALLOW_POP3_FALLBACK))

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled(),
            "email_enabled": EMAIL_ENABLED,
            "smtp_enabled": EMAIL_SMTP_ENABLED,
            "imap_enabled": EMAIL_IMAP_ENABLED,
            "protocol": EMAIL_PROTOCOL,
            "poll_seconds": EMAIL_POLL_SECONDS,
            "from_email": FROM_EMAIL,
            "last_poll_at": self.last_poll_at,
            "command_execution_enabled": EMAIL_COMMAND_EXECUTION_ENABLED,
        }

    def initialize(self) -> Dict[str, Any]:
        try:
            _ensure_dirs()
            init_db()
            result = self.get_status()
            result["status"] = "ok" if self.enabled() else "disabled"
            log_email_event("initialize", details=result)
            return result
        except Exception as exc:
            logger.error(f"Email initialization failed: {exc}")
            log_email_event("initialize_error", details=str(exc))
            return {"status": "error", "error": str(exc)}

    def _connect_imap(self):
        if not EMAIL_IMAP_ENABLED:
            raise RuntimeError("IMAP disabled")
        if not IMAP_HOST or not IMAP_USER:
            raise RuntimeError("IMAP configuration incomplete")
        if IMAP_SSL:
            client = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
        else:
            client = imaplib.IMAP4(IMAP_HOST, IMAP_PORT)
            try:
                client.starttls(ssl.create_default_context())
            except Exception:
                pass
        client.login(IMAP_USER, IMAP_PASSWORD)
        return client

    def _connect_pop3(self):
        if not EMAIL_ALLOW_POP3_FALLBACK:
            raise RuntimeError("POP3 fallback disabled")
        if not POP3_HOST or not POP3_USER:
            raise RuntimeError("POP3 configuration incomplete")
        if POP3_SSL:
            client = poplib.POP3_SSL(POP3_HOST, POP3_PORT, timeout=20)
        else:
            client = poplib.POP3(POP3_HOST, POP3_PORT, timeout=20)
        client.user(POP3_USER)
        client.pass_(POP3_PASSWORD)
        return client

    def _connect_smtp(self):
        if not EMAIL_SMTP_ENABLED:
            raise RuntimeError("SMTP disabled")
        if not SMTP_HOST or not SMTP_USER:
            raise RuntimeError("SMTP configuration incomplete")
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30)
        server.ehlo()
        if SMTP_STARTTLS:
            server.starttls(context=ssl.create_default_context())
            server.ehlo()
        if SMTP_USER:
            server.login(SMTP_USER, SMTP_PASSWORD)
        return server

    def poll_inbox(self, limit: int = EMAIL_MAX_MESSAGES_PER_POLL) -> Dict[str, Any]:
        if not EMAIL_ENABLED:
            return {"status": "disabled", "reason": "EMAIL_ENABLED=false"}
        processed = 0
        errors: List[str] = []
        items: List[Dict[str, Any]] = []
        try:
            if EMAIL_PROTOCOL == "imap" and EMAIL_IMAP_ENABLED:
                items = self._poll_imap(limit)
            elif EMAIL_ALLOW_POP3_FALLBACK:
                items = self._poll_pop3(limit)
            else:
                return {"status": "error", "error": "No inbound email protocol available"}
            processed = len(items)
            self.last_poll_at = datetime.utcnow().isoformat()
            log_email_event("poll_inbox", details={"processed": processed})
            return {"status": "ok", "processed": processed, "items": items, "errors": errors}
        except Exception as exc:
            logger.error(f"poll_inbox failed: {exc}")
            log_email_event("poll_error", details=str(exc))
            return {"status": "error", "error": str(exc), "processed": processed, "errors": errors}

    def _poll_imap(self, limit: int) -> List[Dict[str, Any]]:
        client = self._connect_imap()
        items: List[Dict[str, Any]] = []
        try:
            client.select(INBOX_FOLDER)
            status, data = client.search(None, "UNSEEN")
            if status != "OK":
                return items
            ids = [x for x in (data[0] or b"").split() if x][-limit:]
            for msg_id in ids:
                f_status, msg_data = client.fetch(msg_id, "(RFC822)")
                if f_status != "OK":
                    continue
                raw = b""
                for part in msg_data:
                    if isinstance(part, tuple) and len(part) >= 2:
                        raw = part[1]
                        break
                if not raw:
                    continue
                parsed = self.process_message_bytes(raw, source_folder=INBOX_FOLDER)
                items.append(parsed)
                try:
                    client.store(msg_id, "+FLAGS", "\\Seen")
                except Exception:
                    pass
            return items
        finally:
            try:
                client.close()
            except Exception:
                pass
            try:
                client.logout()
            except Exception:
                pass

    def _poll_pop3(self, limit: int) -> List[Dict[str, Any]]:
        client = self._connect_pop3()
        items: List[Dict[str, Any]] = []
        try:
            count = len(client.list()[1])
            start = max(1, count - limit + 1)
            for idx in range(start, count + 1):
                raw_lines = client.retr(idx)[1]
                raw = b"\n".join(raw_lines)
                parsed = self.process_message_bytes(raw, source_folder="POP3")
                items.append(parsed)
            return items
        finally:
            try:
                client.quit()
            except Exception:
                pass

    def process_message_bytes(self, raw_bytes: bytes, source_folder: str = INBOX_FOLDER) -> Dict[str, Any]:
        msg = email.message_from_bytes(raw_bytes, policy=policy.default)
        parsed = self._parse_message(msg)
        reminder_result = {"status": "skipped", "reason": "not_eligible"}
        if parsed.reminder_candidate and not parsed.is_spam_candidate and not parsed.contains_system_command_language:
            if EMAIL_REMINDER_PARSE_ENABLED or EMAIL_CALENDAR_PARSE_ENABLED:
                reminder_result = _process_reminder_calendar_text(parsed.body_sanitized)

        saved = self._store_message(parsed, source_folder, reminder_result)
        auto_reply = {"status": "skipped"}
        if self._should_auto_reply(parsed):
            auto_reply = self.reply_to_message(parsed, reminder_result)
            if auto_reply.get("status") == "sent":
                self._mark_auto_replied(parsed.message_id)

        result = {
            "status": "ok",
            "message_id": parsed.message_id,
            "subject": parsed.subject,
            "sender_email": parsed.sender_email,
            "spam_candidate": parsed.is_spam_candidate,
            "contains_system_command_language": parsed.contains_system_command_language,
            "reminder_result": reminder_result,
            "auto_reply": auto_reply,
            "stored": saved,
            "attachments": [asdict(a) for a in parsed.attachments],
        }
        log_email_event("message_processed", parsed.message_id, result)
        return result

    def _parse_message(self, msg: Message) -> ParsedEmail:
        subject = _decode_mime_header(msg.get("Subject", ""))
        sender_name, sender_email = parseaddr(_decode_mime_header(msg.get("From", "")))
        to_addrs = [addr for _, addr in getaddresses(msg.get_all("To", [])) if addr]
        cc_addrs = [addr for _, addr in getaddresses(msg.get_all("Cc", [])) if addr]
        reply_to = [addr for _, addr in getaddresses(msg.get_all("Reply-To", [])) if addr]
        message_id = _decode_mime_header(msg.get("Message-ID", "")).strip() or make_msgid(domain=(FROM_EMAIL.split("@")[-1] if "@" in FROM_EMAIL else None))
        sent_at = _decode_mime_header(msg.get("Date", "")) or datetime.utcnow().isoformat()
        body_text, body_html = _extract_message_bodies(msg)
        body_sanitized = _sanitize_text(body_text or body_html)

        attachments: List[EmailAttachmentRecord] = []
        for part in msg.walk():
            if (part.get_content_disposition() or "").lower() == "attachment":
                try:
                    record = _stage_attachment(part, message_id)
                    if record is not None:
                        attachments.append(record)
                except Exception as exc:
                    logger.warning(f"Attachment staging failed: {exc}")

        combined_text = f"{subject}\n{body_sanitized}".strip()
        parsed = ParsedEmail(
            message_id=message_id,
            subject=subject,
            sender_name=sender_name,
            sender_email=sender_email,
            to_addrs=to_addrs,
            cc_addrs=cc_addrs,
            reply_to=reply_to,
            sent_at=sent_at,
            body_text=body_text,
            body_html=body_html,
            body_sanitized=body_sanitized,
            is_spam_candidate=_is_spam_candidate(subject, body_sanitized, sender_email),
            is_no_reply=_is_no_reply_address(sender_email),
            contains_system_command_language=_contains_system_command_language(combined_text),
            reminder_candidate=_is_reminder_candidate(combined_text),
            knowledge_candidate=EMAIL_KNOWLEDGE_EXTRACT_ENABLED and len(body_sanitized) >= 20,
            attachments=attachments,
            raw_headers={k: _decode_mime_header(v) for k, v in msg.items()},
        )
        return parsed

    def _store_message(self, parsed: ParsedEmail, folder_name: str, reminder_result: Dict[str, Any]) -> bool:
        try:
            db = _get_db()
            recipients = list(dict.fromkeys(parsed.to_addrs + parsed.cc_addrs))
            extra = {
                "reply_to": parsed.reply_to,
                "raw_headers": parsed.raw_headers,
                "knowledge": _extract_useful_knowledge(parsed.subject, parsed.body_sanitized) if parsed.knowledge_candidate else {},
                "reminder_result": reminder_result,
            }
            db.execute(
                """
                INSERT OR REPLACE INTO email_messages (
                    message_id, direction, sender_email, sender_name, recipients_json,
                    subject, sent_at, received_at, body_text, body_sanitized, status,
                    folder_name, is_spam_candidate, is_no_reply,
                    contains_system_command_language, reminder_candidate,
                    knowledge_candidate, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    parsed.message_id,
                    "inbound",
                    parsed.sender_email,
                    parsed.sender_name,
                    json.dumps(recipients, ensure_ascii=False),
                    parsed.subject,
                    parsed.sent_at,
                    datetime.utcnow().isoformat(),
                    parsed.body_text,
                    parsed.body_sanitized,
                    "stored",
                    folder_name,
                    1 if parsed.is_spam_candidate else 0,
                    1 if parsed.is_no_reply else 0,
                    1 if parsed.contains_system_command_language else 0,
                    1 if parsed.reminder_candidate else 0,
                    1 if parsed.knowledge_candidate else 0,
                    json.dumps(extra, ensure_ascii=False),
                ),
            )
            db.execute("DELETE FROM email_attachments WHERE message_id = ?", (parsed.message_id,))
            for att in parsed.attachments:
                db.execute(
                    """
                    INSERT INTO email_attachments (
                        message_id, filename, content_type, size_bytes, saved_path,
                        sha256, scan_status, scan_details, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        parsed.message_id,
                        att.filename,
                        att.content_type,
                        att.size_bytes,
                        att.saved_path,
                        att.sha256,
                        att.scan_status,
                        att.scan_details,
                        datetime.utcnow().isoformat(),
                    ),
                )
            db.commit()
            return True
        except Exception as exc:
            logger.error(f"Failed to store email {parsed.message_id}: {exc}")
            return False

    def _should_auto_reply(self, parsed: ParsedEmail) -> bool:
        if not EMAIL_AUTOREPLY_ENABLED:
            return False
        if parsed.is_spam_candidate or parsed.is_no_reply:
            return False
        if parsed.contains_system_command_language:
            return False
        if parsed.sender_email.lower() == (FROM_EMAIL or "").lower():
            return False
        if not parsed.sender_email:
            return False
        return True

    def _mark_auto_replied(self, message_id: str) -> None:
        try:
            db = _get_db()
            db.execute("UPDATE email_messages SET auto_replied = 1 WHERE message_id = ?", (message_id,))
            db.commit()
        except Exception:
            pass

    def reply_to_message(self, parsed: ParsedEmail, reminder_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        targets = _reply_targets(parsed)
        if not targets:
            return {"status": "skipped", "reason": "no_reply_target"}
        body = _generate_reply_body(parsed, reminder_result or {})
        subject = parsed.subject.strip()
        if not subject.lower().startswith("re:"):
            subject = f"{EMAIL_AUTOREPLY_SUBJECT_PREFIX} {subject}".strip()
        return self.send_email(
            to_addrs=targets,
            subject=subject,
            body_text=body,
            in_reply_to=parsed.message_id,
            references=parsed.message_id,
        )

    def send_email(
        self,
        to_addrs: List[str],
        subject: str,
        body_text: str,
        cc_addrs: Optional[List[str]] = None,
        bcc_addrs: Optional[List[str]] = None,
        attachments: Optional[List[str]] = None,
        in_reply_to: str = "",
        references: str = "",
    ) -> Dict[str, Any]:
        if not EMAIL_SMTP_ENABLED:
            return {"status": "disabled", "reason": "SMTP disabled"}
        cc_addrs = cc_addrs or []
        bcc_addrs = bcc_addrs or []
        attachments = attachments or []
        msg = EmailMessage()
        msg["From"] = FROM_EMAIL or SMTP_USER
        msg["To"] = ", ".join([x for x in to_addrs if x])
        if cc_addrs:
            msg["Cc"] = ", ".join([x for x in cc_addrs if x])
        msg["Subject"] = subject.strip() or "SarahMemory Message"
        msg["Date"] = email.utils.format_datetime(datetime.now())
        msg["Message-ID"] = make_msgid(domain=(FROM_EMAIL.split("@")[-1] if "@" in FROM_EMAIL else None))
        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
        if references:
            msg["References"] = references
        msg.set_content(body_text or "")

        for attachment_path in attachments:
            try:
                path = Path(attachment_path)
                if not path.exists() or not path.is_file():
                    continue
                ctype, _ = mimetypes.guess_type(str(path))
                maintype, subtype = (ctype.split("/", 1) if ctype else ("application", "octet-stream"))
                with open(path, "rb") as handle:
                    msg.add_attachment(handle.read(), maintype=maintype, subtype=subtype, filename=path.name)
            except Exception as exc:
                logger.warning(f"Skipping outbound attachment {attachment_path}: {exc}")

        try:
            with self._connect_smtp() as server:
                server.send_message(msg, from_addr=FROM_EMAIL or SMTP_USER, to_addrs=to_addrs + cc_addrs + bcc_addrs)
            self._store_outbound_message(msg, body_text, to_addrs, cc_addrs, bcc_addrs)
            log_email_event("smtp_send", msg["Message-ID"], {"to": to_addrs, "subject": subject})
            return {"status": "sent", "message_id": msg["Message-ID"], "to": to_addrs, "subject": subject}
        except Exception as exc:
            logger.error(f"SMTP send failed: {exc}")
            log_email_event("smtp_send_error", details=str(exc))
            return {"status": "error", "error": str(exc)}

    def _store_outbound_message(
        self,
        msg: EmailMessage,
        body_text: str,
        to_addrs: List[str],
        cc_addrs: List[str],
        bcc_addrs: List[str],
    ) -> None:
        try:
            db = _get_db()
            db.execute(
                """
                INSERT OR REPLACE INTO email_messages (
                    message_id, direction, sender_email, sender_name, recipients_json,
                    subject, sent_at, received_at, body_text, body_sanitized, status,
                    folder_name, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg["Message-ID"],
                    "outbound",
                    FROM_EMAIL or SMTP_USER,
                    "SarahMemory AiOS",
                    json.dumps({"to": to_addrs, "cc": cc_addrs, "bcc": bcc_addrs}, ensure_ascii=False),
                    str(msg.get("Subject", "")),
                    str(msg.get("Date", datetime.utcnow().isoformat())),
                    datetime.utcnow().isoformat(),
                    body_text,
                    _sanitize_text(body_text),
                    "sent",
                    "Sent",
                    json.dumps({}, ensure_ascii=False),
                ),
            )
            db.commit()
        except Exception as exc:
            logger.warning(f"Could not persist outbound email: {exc}")

    def forward_message(self, stored_message_id: str, forward_to: List[str], note: str = "") -> Dict[str, Any]:
        if not EMAIL_FORWARDING_ENABLED:
            return {"status": "disabled", "reason": "forwarding disabled"}
        db = _get_db()
        row = db.execute("SELECT * FROM email_messages WHERE message_id = ?", (stored_message_id,)).fetchone()
        if row is None:
            return {"status": "error", "error": "message not found"}
        body = (note.strip() + "\n\n" if note.strip() else "") + (
            f"Forwarded message\nFrom: {row['sender_name']} <{row['sender_email']}>\n"
            f"Subject: {row['subject']}\n\n{row['body_sanitized']}"
        )
        subject = row["subject"] or ""
        if not subject.lower().startswith("fwd:"):
            subject = f"Fwd: {subject}".strip()
        result = self.send_email(forward_to, subject, body)
        if result.get("status") == "sent":
            db.execute("UPDATE email_messages SET forwarded = 1 WHERE message_id = ?", (stored_message_id,))
            db.commit()
        return result

    def purge_expired_junk_records(self, retention_days: int = EMAIL_JUNK_RETENTION_DAYS) -> Dict[str, Any]:
        cutoff = datetime.utcnow() - timedelta(days=retention_days)
        db = _get_db()
        rows = db.execute(
            "SELECT message_id FROM email_messages WHERE is_spam_candidate = 1 AND received_at < ?",
            (cutoff.isoformat(),),
        ).fetchall()
        count = 0
        for row in rows:
            db.execute("DELETE FROM email_attachments WHERE message_id = ?", (row["message_id"],))
            db.execute("DELETE FROM email_messages WHERE message_id = ?", (row["message_id"],))
            count += 1
        db.commit()
        log_email_event("purge_junk", details={"purged": count, "retention_days": retention_days})
        return {"status": "ok", "purged": count, "retention_days": retention_days}


# -----------------------------------------------------------------------------
# Module-level convenience API
# -----------------------------------------------------------------------------
_SERVICE: Optional[SarahMemoryEmailService] = None


def get_email_service() -> SarahMemoryEmailService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = SarahMemoryEmailService()
    return _SERVICE


def initialize() -> Dict[str, Any]:
    return get_email_service().initialize()


def poll_inbox(limit: int = EMAIL_MAX_MESSAGES_PER_POLL) -> Dict[str, Any]:
    return get_email_service().poll_inbox(limit=limit)


def send_email(
    to_addrs: List[str],
    subject: str,
    body_text: str,
    cc_addrs: Optional[List[str]] = None,
    bcc_addrs: Optional[List[str]] = None,
    attachments: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return get_email_service().send_email(to_addrs, subject, body_text, cc_addrs, bcc_addrs, attachments)


def forward_message(stored_message_id: str, forward_to: List[str], note: str = "") -> Dict[str, Any]:
    return get_email_service().forward_message(stored_message_id, forward_to, note)


def purge_expired_junk_records(retention_days: int = EMAIL_JUNK_RETENTION_DAYS) -> Dict[str, Any]:
    return get_email_service().purge_expired_junk_records(retention_days)


def get_status() -> Dict[str, Any]:
    return get_email_service().get_status()


if __name__ == "__main__":
    print(json.dumps(initialize(), indent=2))
