"""--==The SarahMemory Project==--
File: SarahMemoryAudit.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.1.0a (Enterprise)
Date: 2026-11-01
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

Central bounded audit utility for SarahMemory AiOS.

This file is intentionally small and deterministic. It gives organs a shared way to
record allow/deny/defer/stage decisions without pulling in cloud APIs, GUI stacks,
large dependencies, or heavy database workflows.
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "audit_ledger_utility"
# CATEGORY = "governance_audit"
# USER_FACING = False
# UI_EXPOSURE = "internal_only"
# DEPLOYMENT_TARGET = "core"
# API_DOMAIN = "audit"
# HARDWARE_DOMAIN = "filesystem"
# INTERNAL_ONLY = True
# CAPABILITY_NAME = "audit"
# FAMILY = "core_governance"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = True
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# RELEASE_PHASE = "ALPHA"
# RELEASE_TRACK = "developer"
# VALIDATION_DATE = "2026-11-01"
# VALIDATION_TIME = "10:11:54"
# PROJECT_SECTION = "SarahMemory AiOS Governed Cognitive Runtime"
# STRUCTURAL_MARKER = "from __future__ import annotations"
# NOTES = "Central bounded JSONL audit writer for governance decisions, sync events, security denials, and quantum-safe deterministic calculations."
# --- SARAHMETA END ---

import json
import os
import time
import hashlib
import threading
import math
import re
import stat
import uuid
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:  # pragma: no cover
    config = None  # type: ignore

_AUDIT_LOCK = threading.RLock()
_LAST_AUDIT_STATUS: Dict[str, Any] = {"attempted": False, "ok": None, "last_error": None, "write_latency_ms": None, "attempts": 0, "failures": 0, "last_failure": None}
_SCHEMA_VERSION = "9.1.0a"
_REDACTED = "[REDACTED]"
_SECRET_KEY_RE = re.compile(r"(?:passw(?:or)?d|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|token|authorization|auth[_-]?header|cookie|set[_-]?cookie|client[_-]?secret|private[_-]?key|credential|bearer)", re.I)
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{6,}")
_INLINE_SECRET_RE = re.compile(r"(?i)\b(api[_-]?key|password|secret|token|authorization|cookie)\s*[:=]\s*['\"]?[^\s'\",;]{3,}")
_HASH_EXCLUDE = frozenset({"event_hash", "ok", "audit_path", "audit_error", "persisted", "durability", "write_latency_ms"})


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


def _audit_root() -> str:
    """Retain the established local-first data/audit path."""
    return os.path.join(_data_dir(), "audit")


def _int_config(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return max(minimum, min(maximum, int(os.getenv(name, str(default)))))
    except (ValueError, TypeError, OverflowError):
        return default


def _float_config(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return max(minimum, min(maximum, value)) if math.isfinite(value) else default
    except (ValueError, TypeError, OverflowError):
        return default


def _redact_text(value: str, limit: int) -> str:
    # Do not allow secrets in ordinary descriptive strings either.
    value = _BEARER_RE.sub("Bearer " + _REDACTED, value[:limit * 2])
    value = _INLINE_SECRET_RE.sub(lambda m: m.group(1) + "=" + _REDACTED, value)
    return value[:limit] + ("…" if len(value) > limit else "")


def _safe_label(value: Any, default: str, limit: int) -> str:
    """Avoid invoking arbitrary object __str__ inside a security audit path."""
    if value is None or (isinstance(value, str) and not value):
        return default
    if isinstance(value, (str, int, float, bool)):
        try:
            return _redact_text(str(value)[:limit * 2], limit)
        except (ValueError, TypeError, OverflowError):
            pass
    return "[UNSUPPORTED: " + type(value).__name__[:40] + "]"


def _clean(value: Any, *, depth: int = 0, budget: Optional[list] = None) -> Any:
    """Strictly bounded, privacy-filtered JSON conversion (no arbitrary object repr)."""
    if budget is None:
        budget = [256]
    budget[0] -= 1
    if budget[0] < 0:
        return "[TRUNCATED: item budget]"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _redact_text(value, 2048)
    if isinstance(value, int):
        return value if abs(value).bit_length() <= 128 else "[TRUNCATED: large integer]"
    if isinstance(value, float):
        return value if math.isfinite(value) else "[INVALID: non-finite number]"
    if depth >= 5:
        return "[TRUNCATED: nesting limit]"
    if isinstance(value, dict):
        out = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= 32 or budget[0] <= 0:
                out["_truncated"] = True
                break
            key = _redact_text(str(k)[:128], 128) if isinstance(k, (str, int, float, bool)) else "[unsupported_key]"
            out[key] = _REDACTED if _SECRET_KEY_RE.search(key) else _clean(v, depth=depth + 1, budget=budget)
        return out
    if isinstance(value, (list, tuple)):
        out = [_clean(v, depth=depth + 1, budget=budget) for v in value[:32]]
        if len(value) > 32:
            out.append("[TRUNCATED: list limit]")
        return out
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "[REDACTED: binary data]"
    return "[UNSUPPORTED: " + type(value).__name__[:60] + "]"


def _safe_json(value: Any) -> Any:
    try:
        return _clean(value)
    except Exception:
        return "[INVALID: non-serializable detail]"


def _canonical_bytes(payload: Dict[str, Any]) -> bytes:
    record = {k: v for k, v in payload.items() if k not in _HASH_EXCLUDE}
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _fingerprint(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def verify_audit_record(record: Dict[str, Any]) -> bool:
    """Verify record consistency, NOT authenticity or a trusted ledger signature."""
    try:
        return isinstance(record, dict) and len(str(record.get("event_hash", ""))) == 64 and _fingerprint(record) == record["event_hash"]
    except (ValueError, TypeError, OverflowError):
        return False


def _checked_regular(path: str) -> None:
    """Reject file symlinks, directories and special files (including lock files)."""
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISREG(mode):
        raise PermissionError("audit_destination_not_regular_file")


def _ensure_audit_root(root: str) -> None:
    if os.path.islink(root):
        raise PermissionError("audit_directory_symlink_disallowed")
    os.makedirs(root, mode=0o700, exist_ok=True)
    if os.path.islink(root) or not os.path.isdir(root):
        raise PermissionError("audit_directory_invalid")


@contextmanager
def _process_lock(root: str):
    """Bounded OS-level lock coordinates all cooperating writer processes."""
    lock_path = os.path.join(root, ".sarahmemory_audit.lock")
    _checked_regular(lock_path)
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(lock_path, flags, 0o600)
    acquired = False
    win = os.name == "nt"
    try:
        if win:
            import msvcrt
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"0")
        else:
            import fcntl
        deadline = time.monotonic() + _float_config("SARAH_AUDIT_LOCK_TIMEOUT", 2.0, 0.1, 30.0)
        while not acquired:
            try:
                if win:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except (OSError, IOError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("audit_lock_timeout")
                time.sleep(0.02)
        yield
    finally:
        try:
            if acquired:
                if win:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _storage_budget(root: str, additional: int) -> None:
    """Bound archive growth without silently deleting forensic history."""
    max_total = _int_config("SARAH_AUDIT_TOTAL_MAX_BYTES", 268435456, 1048576, 17179869184)
    max_backups = _int_config("SARAH_AUDIT_MAX_BACKUPS", 256, 1, 100000)
    total, backups, scanned = 0, 0, 0
    with os.scandir(root) as entries:
        for entry in entries:
            if not (entry.name.endswith("_sarahmemory_audit.jsonl") or ("_sarahmemory_audit.jsonl." in entry.name and entry.name.endswith(".bak"))):
                continue
            scanned += 1
            if scanned > 4096:
                raise OSError("audit_storage_listing_limit_reached")
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise PermissionError("audit_storage_nonregular_entry")
            total += entry.stat(follow_symlinks=False).st_size
            backups += bool(entry.name.endswith(".bak"))
            if total + additional > max_total:
                raise OSError("audit_storage_capacity_exhausted")
    if backups > max_backups:
        raise OSError("audit_backup_count_exhausted")


def _rotate(path: str, root: str) -> None:
    _checked_regular(path)
    # Hard-link + unlink never overwrites an existing backup. If unlink fails,
    # both names retain evidence and a failure is reported to the caller.
    backup = path + "." + str(time.time_ns()) + "." + uuid.uuid4().hex[:12] + ".bak"
    _checked_regular(backup)
    try:
        os.link(path, backup)
    except OSError:
        # Non-overwriting rename fallback for platforms without hard links.
        # The OS lock serializes all compliant audit writers.
        if os.path.lexists(backup):
            raise FileExistsError("audit_rotation_backup_collision")
        os.rename(path, backup)
        return
    os.unlink(path)


def _append_record(root: str, path: str, payload: bytes) -> str:
    max_bytes = _int_config("SARAH_AUDIT_MAX_BYTES", 8388608, 4096, 1073741824)
    if len(payload) > max_bytes:
        raise ValueError("audit_record_exceeds_daily_file_cap")
    with _process_lock(root):
        _checked_regular(path)
        exists = os.path.exists(path)
        old_size = os.stat(path).st_size if exists else 0
        _storage_budget(root, len(payload))
        if exists and old_size + len(payload) > max_bytes:
            # Do not rotate if the operator-defined archive budget is exhausted.
            max_backups = _int_config("SARAH_AUDIT_MAX_BACKUPS", 256, 1, 100000)
            with os.scandir(root) as entries:
                count = sum(1 for entry in entries if entry.name.endswith(".bak") and "_sarahmemory_audit.jsonl." in entry.name)
            if count >= max_backups:
                raise OSError("audit_backup_count_exhausted")
            _rotate(path, root)
            old_size = 0
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise PermissionError("audit_target_not_regular_file")
            before = os.fstat(fd).st_size
            try:
                written = 0
                while written < len(payload):
                    n = os.write(fd, payload[written:])
                    if n <= 0:
                        raise OSError("audit_short_write")
                    written += n
                sync = os.getenv("SARAH_AUDIT_FSYNC", "0").strip().lower() in {"1", "true", "yes", "on"}
                if sync:
                    os.fsync(fd)
            except Exception:
                # Roll back incomplete record under the exclusive writer lock.
                # A failure during fsync can still mean disk persistence is uncertain.
                try:
                    os.ftruncate(fd, before)
                except Exception:
                    pass
                raise
        finally:
            os.close(fd)
    return "fsynced" if sync else "buffer_flushed"


def _record_status(ok: bool, error: Optional[str], latency_ms: float) -> None:
    with _AUDIT_LOCK:
        _LAST_AUDIT_STATUS.update(attempted=True, ok=bool(ok), last_error=error, write_latency_ms=round(latency_ms, 3))
        _LAST_AUDIT_STATUS["attempts"] += 1
        if not ok:
            _LAST_AUDIT_STATUS["failures"] += 1
            _LAST_AUDIT_STATUS["last_failure"] = error


def audit_event(
    family: str,
    action: str,
    verdict: str,
    details: Optional[Dict[str, Any]] = None,
    *,
    actor: str = "SarahMemory",
    risk: str = "low",
    source: str = "internal",
    retention: str = "standard",
) -> Dict[str, Any]:
    """Record a bounded local audit event; NEVER issue action authority.

    'event_hash' covers the immutable evidence fields, excluding transport
    outcomes (ok, persisted, path, latency, durability and error). An unkeyed
    digest detects accidental changes but is NOT an authenticity signature.
    """
    t0 = time.monotonic()
    now = datetime.now(timezone.utc)
    event: Dict[str, Any] = {
        "ok": True,
        "ts": now.timestamp(),
        "timestamp": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "family": _safe_label(family, "general", 96),
        "action": _safe_label(action, "unknown", 128),
        "verdict": _safe_label(verdict, "UNKNOWN", 64),
        "actor": _safe_label(actor, "SarahMemory", 128),
        "risk": _safe_label(risk, "low", 64),
        "source": _safe_label(source, "internal", 128),
        "retention": _safe_label(retention, "standard", 64),
        "details": _safe_json(details if details is not None else {}),
        "schema_version": _SCHEMA_VERSION,
        "event_id": uuid.uuid4().hex,
    }
    try:
        event_limit = _int_config("SARAH_AUDIT_MAX_EVENT_BYTES", 16384, 1024, 262144)
        for _ in range(3):
            event["event_hash"] = _fingerprint(event)
            line = (json.dumps(event, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
            if len(line) <= event_limit:
                break
            event["details"] = {"_truncated": True, "reason": "event_byte_budget"}
        else:
            raise ValueError("audit_event_byte_budget_too_small")
        if len(line) > event_limit:
            raise ValueError("audit_event_exceeds_byte_budget")
        root = _audit_root()
        _ensure_audit_root(root)
        path = os.path.join(root, now.strftime("%Y%m%d") + "_sarahmemory_audit.jsonl")
        with _AUDIT_LOCK:
            durability = _append_record(root, path, line)
        event["audit_path"] = path
        event["persisted"] = True
        event["durability"] = durability
        _record_status(True, None, (time.monotonic() - t0) * 1000)
    except Exception as exc:
        event["ok"] = False
        event["persisted"] = False
        event["audit_error"] = str(exc)[:240]
        event["durability"] = "unknown"
        _record_status(False, event["audit_error"], (time.monotonic() - t0) * 1000)
    event["write_latency_ms"] = round((time.monotonic() - t0) * 1000, 3)
    return event

def audit_allow(action: str, details: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    return audit_event("governance", action, "ALLOW", details, **kwargs)


def audit_deny(action: str, details: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    return audit_event("governance", action, "DENY", details, **kwargs)


def audit_defer(action: str, details: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    return audit_event("governance", action, "DEFER", details, **kwargs)

# ====================================================================
# END OF SarahMemoryAudit.py v9.1.0a
# ====================================================================
# END OF LINE

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryAudit',
    "version": "v9.1.0a-alpha-sml-0.2",
    "category": 'Governance',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": ['governance'],
    "supported_missions": ['Conversation', 'Execution', 'Governance', 'Security'],
    "supported_omega": ['Ω001', 'Ω050', 'Ω060'],
    "required_authority": ['Read', 'Research'],
    "priority": 90,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryAudit.py'},
}


def sml_get_metadata():
    """Return an isolated metadata snapshot; callers cannot mutate module state."""
    return copy.deepcopy(SML_ORGAN_METADATA)


def sml_health():
    """Evidence-based status; no disk operations or invented perfect scores."""
    with _AUDIT_LOCK:
        state = dict(_LAST_AUDIT_STATUS)
    attempted, ok = state["attempted"], state["ok"]
    status = "Unknown" if not attempted else ("Operational" if ok else "Degraded")
    return {
        "status": status,
        "availability": None if not attempted else (1.0 if ok else 0.0),
        "integrity": None,  # Requires trusted Ledger or signed baseline evidence.
        "performance": None,
        "reliability": (None if not attempted else round((state["attempts"] - state["failures"]) / max(1, state["attempts"]), 4)),
        "confidence": 0.0 if not attempted else 0.50,
        "latency_ms": state["write_latency_ms"],
        "stability": None,
        "compatibility": None,
        "notes": ["Last-write evidence only; unmeasured dimensions are unknown."],
        "last_error": state["last_error"],
        "last_write_ok": ok,
        "write_attempts": state["attempts"],
        "write_failures": state["failures"],
        "last_failure": state["last_failure"],
    }


def sml_diagnostics():
    """Expose operational evidence without executing an audit write."""
    health = sml_health()
    return {
        "status": health["status"],
        "component": "SarahMemoryAudit",
        "sml_adapter": True,
        "metadata": sml_get_metadata(),
        "health": health,
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryAudit', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---

