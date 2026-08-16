"""--==The SarahMemory Project==--
File: SarahMemoryAudit.py
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
# VALIDATION_DATE = "2026-07-11"
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
from datetime import datetime
from typing import Any, Dict, Optional

try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:  # pragma: no cover
    config = None  # type: ignore

_AUDIT_LOCK = threading.RLock()


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
    # SARAHMEMORY_PATCH_NOTE: Audit is placed under data/audit so it is portable,
    # local-first, and independent of the current working directory. This prevents
    # random launch folders from receiving governance records.
    return os.path.join(_data_dir(), "audit")


def _safe_json(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _fingerprint(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8", "ignore")
    return hashlib.sha256(raw).hexdigest()


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
    """Write a bounded JSONL audit event and return the event packet.

    # SARAHMEMORY_PATCH_NOTE: This function never authorizes an action. It only
    # records what happened or what was denied. Authority still belongs to the
    # governing organ and ultimately the user.
    """
    event = {
        "ok": True,
        "ts": time.time(),
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "family": str(family or "general")[:96],
        "action": str(action or "unknown")[:128],
        "verdict": str(verdict or "UNKNOWN")[:64],
        "actor": str(actor or "SarahMemory")[:128],
        "risk": str(risk or "low")[:64],
        "source": str(source or "internal")[:128],
        "retention": str(retention or "standard")[:64],
        "details": _safe_json(details or {}),
    }
    event["event_hash"] = _fingerprint(event)

    try:
        root = _audit_root()
        os.makedirs(root, exist_ok=True)
        day = datetime.utcnow().strftime("%Y%m%d")
        path = os.path.join(root, f"{day}_sarahmemory_audit.jsonl")
        with _AUDIT_LOCK:
            # SARAHMEMORY_PATCH_NOTE: Bounded rotate to prevent HDD/NVMe thrash.
            # If a daily audit grows beyond the configured cap, rename once and
            # continue a fresh file. This keeps auditability without unbounded spam.
            max_bytes = int(os.getenv("SARAH_AUDIT_MAX_BYTES", "8388608"))
            if os.path.exists(path) and os.path.getsize(path) > max(262144, max_bytes):
                rotated = path + f".{int(time.time())}.bak"
                try:
                    os.replace(path, rotated)
                except Exception:
                    pass
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, sort_keys=True, ensure_ascii=False, default=str) + "\n")
        event["audit_path"] = path
    except Exception as exc:
        event["ok"] = False
        event["audit_error"] = str(exc)
    return event


def audit_allow(action: str, details: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    return audit_event("governance", action, "ALLOW", details, **kwargs)


def audit_deny(action: str, details: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    return audit_event("governance", action, "DENY", details, **kwargs)


def audit_defer(action: str, details: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
    return audit_event("governance", action, "DEFER", details, **kwargs)

# ====================================================================
# END OF SarahMemoryAudit.py v9.0.0
# ====================================================================
# END OF LINE

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryAudit',
    "version": "v9.0.0-alpha-sml-0.2",
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
    """Return this organ's SML registration metadata."""
    return dict(SML_ORGAN_METADATA)


def sml_health():
    """Return a local SML health vector without side effects."""
    return {
        "status": "Healthy",
        "availability": 1.0,
        "integrity": 1.0,
        "performance": 1.0,
        "reliability": 1.0,
        "confidence": 0.75,
        "latency_ms": 0.0,
        "stability": 1.0,
        "compatibility": 1.0,
        "notes": ["SML adapter present"],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryAudit',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
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

