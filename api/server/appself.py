# --==The SarahMemory Project==--
# File: /api/server/appself.py
# Purpose: SarahMemory SelfAware / CognitiveSelf fact-ticket API bridge
# Part of the SarahMemory Companion AI-bot Platform
# Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.
# www.linkedin.com/in/brian-baros-29962a176
# https://www.facebook.com/bbaros
# brian.baros@sarahmemory.com
# 'The SarahMemory Companion AI-Bot Platform, SarahMemory AiOS, and all Parts of the SarahMemory Project are property of SOFTDEV0 LLC., & Brian Lee Baros'
# https://www.sarahmemory.com
# https://api.sarahmemory.com
# https://ai.sarahmemory.com
# https://store.sarahmemory.com
# ==============================================================================================
"""
SarahMemory appself.py v8.0.0 EvidenceCourt V8 topology-aware body capability cleanup

SelfAware / CognitiveSelf API bridge.

Design goals:
- Keep app.py from bloating.
- Namespaced routes only: /api/self/*.
- Read-only fact verification by default.
- No patching, no file mutation, no DevBridge execution, no hidden activation.
- Provide a backend evidence court for:
    1) live factual self/system questions
    2) REM MEDIUM candidate interrogation before CognitiveServices/DevBridge
- Enforce a jurisdiction-aware 3-artifact Evidence Court model:
    0/3 -> DENIED_NO_EVIDENCE
    1/3 -> DENIED_WEAK_EVIDENCE
    2/3 -> ESCALATE_HIGH_REVIEW
    3/3 -> APPROVED_FACT
- Reject double-jeopardy evidence stacking: one file/artifact family gets one vote.
- Treat OS/vendor adapters as support/hearsay only; never as constitutional quorum.
- Present network adapter facts without exposing IP/MAC details by default.
- Read Class A boot-driver and Class B runtime-driver catalogs as support-only body capability evidence.
- Build safe hardware topology briefs that separate device class, connection bus, protocol, physical location, mount surface, and ownership scope.

Integration contract:
    import appself
    appself.init_app(app, CONNECT_SQLITE, META_DB, api_key_auth_ok=..., sign_ok=...)

This module is an API bridge. It does not replace:
- SarahMemoryCognitiveSelf.py      canonical self truth authority
- SarahMemorySelfAware.py          introspection / investigation engine
- SarahMemoryCognitiveServices.py  logical governor / judge
- SarahMemoryAssuranceGate.py      assurance hard gate
- SarahMemorySecurityGovernor.py   trust boundary gate
- SarahMemoryOperatorCore.py       action contract executor choke-point
"""

from __future__ import annotations

# --- SARAHMETA START ---
# GRADE = "A"
# ROLE = "api_bridge"
# CATEGORY = "self_awareness_fact_ticketing"
# USER_FACING = False
# UI_EXPOSURE = "backend_only"
# DEPLOYMENT_TARGET = "api_server"
# API_DOMAIN = "self"
# HARDWARE_DOMAIN = "system_sensors_filesystem_runtime"
# INTERNAL_ONLY = False
# CAPABILITY_NAME = "selfaware_api"
# FAMILY = "core_cognition"
# GOVERNANCE_LEVEL = "critical"
# AUTONOMOUS_SAFE = False
# FRONTEND_CANDIDATE = False
# ADDON_CANDIDATE = False
# DRIVER_CANDIDATE = False
# NOTES = "SelfAware/CognitiveSelf API bridge under /api/self/* for fact tickets, 3-probe evidence quorum, body-map exposure, and REM MEDIUM interrogation. No execution or patching."
# --- SARAHMETA END ---

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import sqlite3
import subprocess
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from flask import Blueprint, Response, jsonify, request

bp = Blueprint("appself_v800", __name__)
logger = logging.getLogger(__name__)

_CONNECT_SQLITE: Optional[Callable[..., Any]] = None
_META_DB: Optional[str] = None
_API_KEY_AUTH_OK: Optional[Callable[[], bool]] = None
_SIGN_OK: Optional[Callable[[bytes, str], bool]] = None

PROJECT_VERSION = os.environ.get("PROJECT_VERSION", "8.0.0")
TICKET_TABLE = "selfaware_fact_tickets"
MAX_TICKETS_RETURN = int(os.environ.get("SELFWARE_MAX_TICKETS_RETURN", "200") or 200)
MAX_CLAIM_CHARS = int(os.environ.get("SELFWARE_MAX_CLAIM_CHARS", "12000") or 12000)
MAX_EVIDENCE_JSON_CHARS = int(os.environ.get("SELFWARE_MAX_EVIDENCE_JSON_CHARS", "200000") or 200000)

# V10/V9C: volatile live-body artifact contract used by /api/chat.
VERIFIED_ARTIFACT_PACKAGE_TYPE = "SELF_AWARE_LIVE_HARDWARE_FACT"
VERIFIED_ARTIFACT_PACKAGE_VERSION = "V10_V9C"


# -----------------------------------------------------------------------------
# Optional core imports
# -----------------------------------------------------------------------------
try:
    import SarahMemoryGlobals as config  # type: ignore
except Exception:
    config = None  # type: ignore

try:
    import SarahMemoryCognitiveSelf as _CogSelf  # type: ignore
except Exception:
    _CogSelf = None

try:
    import SarahMemorySelfAware as _SelfAware  # type: ignore
except Exception:
    _SelfAware = None

try:
    import SarahMemoryCognitiveCompass as _CogCompass  # type: ignore
except Exception:
    _CogCompass = None

try:
    import SarahMemoryHi as _Hi  # type: ignore
except Exception:
    _Hi = None

try:
    import SarahMemoryDiagnostics as _Diagnostics  # type: ignore
except Exception:
    _Diagnostics = None

try:
    import SarahMemoryOptimization as _Optimization  # type: ignore
except Exception:
    _Optimization = None

try:
    import SarahMemoryCognitiveServices as _CogServices  # type: ignore
except Exception:
    _CogServices = None

try:
    import SarahMemorySafetyPolicies as _SafetyPolicies  # type: ignore
except Exception:
    _SafetyPolicies = None

try:
    import SarahMemorySecurityGovernor as _SecurityGovernor  # type: ignore
except Exception:
    _SecurityGovernor = None

try:
    import SarahMemoryAssuranceGate as _AssuranceGate  # type: ignore
except Exception:
    _AssuranceGate = None

try:
    import SarahMemoryTrustRegistry as _TrustRegistry  # type: ignore
except Exception:
    _TrustRegistry = None

try:
    import SarahMemorySi as _Si  # type: ignore
except Exception:
    _Si = None

try:
    import SarahMemorySynapes as _Synapes  # type: ignore
except Exception:
    _Synapes = None

try:
    import SarahMemoryInitialization as _Initialization  # type: ignore
except Exception:
    _Initialization = None

try:
    import SarahMemoryIntegration as _Integration  # type: ignore
except Exception:
    _Integration = None

try:
    import SarahMemoryNetwork as _Network  # type: ignore
except Exception:
    _Network = None

try:
    import SarahMemorySync as _Sync  # type: ignore
except Exception:
    _Sync = None

try:
    import SarahMemorySMAPI as _SMAPI  # type: ignore
except Exception:
    _SMAPI = None

try:
    import SarahMemoryEvolution as _Evolution  # type: ignore
except Exception:
    _Evolution = None

try:
    import SarahMemoryFacialRecognition as _FacialRecognition  # type: ignore
except Exception:
    _FacialRecognition = None

try:
    import SarahMemorySOBJE as _SOBJE  # type: ignore
except Exception:
    _SOBJE = None

try:
    import SarahMemoryAiFunctions as _AiFunctions  # type: ignore
except Exception:
    _AiFunctions = None

try:
    import SarahMemoryDL as _DL  # type: ignore
except Exception:
    _DL = None

try:
    import SarahMemoryAdaptive as _Adaptive  # type: ignore
except Exception:
    _Adaptive = None

# -----------------------------------------------------------------------------
# Response/auth helpers
# -----------------------------------------------------------------------------
def _now() -> float:
    return float(time.time())


def _iso(ts: Optional[float] = None) -> str:
    return datetime.utcfromtimestamp(ts or _now()).isoformat(timespec="seconds") + "Z"


def _j() -> Dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _json_safe(value: Any, *, _depth: int = 0) -> Any:
    """Return a Flask/json.dumps-safe copy of any evidence payload.

    SelfAware evidence can contain values returned by many witnesses. Some
    Python/runtime objects are not directly JSON serializable. A single unsafe
    witness value must not crash /api/self/fact-check or the chat bridge.
    """
    if _depth > 24:
        try:
            return str(value)
        except Exception:
            return "<unserializable>"
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        try:
            # Flask/jsonify may reject NaN/Infinity. Evidence must never crash Reply.
            if value != value or value in (float("inf"), float("-inf")):
                return str(value)
        except Exception:
            return str(value)
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return repr(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, sqlite3.Row):
        try:
            return {str(k): _json_safe(value[k], _depth=_depth + 1) for k in value.keys()}
        except Exception:
            return str(tuple(value))
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            try:
                key = str(k)
            except Exception:
                key = "<key>"
            if callable(v):
                out[key] = f"<callable:{getattr(v, '__name__', type(v).__name__)}>"
            else:
                out[key] = _json_safe(v, _depth=_depth + 1)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v, _depth=_depth + 1) for v in list(value)]
    if hasattr(value, "to_dict") and callable(getattr(value, "to_dict")):
        try:
            return _json_safe(value.to_dict(), _depth=_depth + 1)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe(vars(value), _depth=_depth + 1)
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except Exception:
        try:
            return str(value)
        except Exception:
            return f"<unserializable:{type(value).__name__}>"



def _response_json(payload: Dict[str, Any], status_code: int = 200):
    """Hard-fail-safe JSON response builder for SelfAware routes.

    This bypasses Flask jsonify edge cases so unsupported runtime/evidence
    objects, NaN/Infinity, sqlite rows, Path values, or odd witness payloads
    cannot turn a SelfAware fact-check into HTTP 500.
    """
    try:
        safe_payload = _json_safe(payload)
        body = json.dumps(safe_payload, ensure_ascii=False, sort_keys=False, allow_nan=False, default=str)
    except Exception as exc:
        try:
            body = json.dumps({
                "ok": False,
                "error": "response_serialization_failed",
                "error_type": type(exc).__name__,
                "detail": str(exc),
            }, ensure_ascii=False, allow_nan=False)
        except Exception:
            body = '{"ok":false,"error":"response_serialization_failed"}'
    resp = Response(body, status=int(status_code or 200), mimetype="application/json")
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Sarah-Signature"
    return resp, int(status_code or 200)


def _ok(data: Any = None, **meta: Any):
    payload: Dict[str, Any] = {"ok": True}
    if data is not None:
        payload["data"] = data
    if meta:
        payload.update(meta)
    return _response_json(payload, 200)


def _err(msg: str, code: int = 400, **meta: Any):
    payload: Dict[str, Any] = {"ok": False, "error": _safe_str(msg, 1000) if "_safe_str" in globals() else str(msg)}
    if meta:
        payload.update(meta)
    return _response_json(payload, int(code or 400))


def _body_bytes() -> bytes:
    try:
        return request.get_data(cache=True) or b""
    except Exception:
        return b""


def _verify_auth(body_bytes: bytes) -> bool:
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

    # Local/dev mode: if app.py did not inject auth, remain usable.
    return True


def _require_injected() -> bool:
    return bool(_CONNECT_SQLITE and _META_DB)


def _db() -> sqlite3.Connection:
    if _CONNECT_SQLITE and _META_DB:
        return _CONNECT_SQLITE(_META_DB)  # type: ignore[misc]

    # Safe fallback for standalone local testing.
    db_path = _fallback_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path), timeout=5.0, check_same_thread=False)
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA busy_timeout=5000;")
    except Exception:
        pass
    return con


def _fallback_db_path() -> Path:
    try:
        data_dir = Path(str(getattr(config, "DATA_DIR"))).expanduser().resolve()  # type: ignore[arg-type]
    except Exception:
        try:
            data_dir = Path.cwd().joinpath("data").resolve()
        except Exception:
            data_dir = Path("data").resolve()
    return data_dir / "memory" / "datasets" / "selfaware_tickets.db"


def _ensure_tables() -> None:
    con = None
    try:
        con = _db()
        cur = con.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TICKET_TABLE} (
                ticket_id TEXT PRIMARY KEY,
                ts REAL,
                ts_iso TEXT,
                source TEXT,
                ticket_kind TEXT,
                claim TEXT,
                normalized_claim TEXT,
                requested_fact TEXT,
                quorum TEXT,
                pass_count INTEGER,
                decision TEXT,
                confidence TEXT,
                risk_tier TEXT,
                rem_ticket_id TEXT,
                rem_cycle_id TEXT,
                evidence_json TEXT,
                review_json TEXT
            )
            """
        )
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TICKET_TABLE}_ts ON {TICKET_TABLE}(ts)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TICKET_TABLE}_decision ON {TICKET_TABLE}(decision)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{TICKET_TABLE}_rem ON {TICKET_TABLE}(rem_ticket_id, rem_cycle_id)")
        con.commit()
    except Exception as exc:
        logger.warning("appself table ensure failed: %s", exc)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Core/path utilities
# -----------------------------------------------------------------------------
def _module_available(mod: Any) -> bool:
    return mod is not None


def _safe_str(value: Any, limit: int = 1000) -> str:
    try:
        text = "" if value is None else str(value)
    except Exception:
        text = ""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if limit and len(text) > limit:
        text = text[:limit].rstrip()
    return text


def _safe_json_obj(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default
    return default


def _json_dumps_limited(value: Any, limit: int = MAX_EVIDENCE_JSON_CHARS) -> str:
    try:
        text = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    except Exception as exc:
        text = json.dumps({"error": "json_encode_failed", "detail": str(exc)}, ensure_ascii=False)
    if len(text) > limit:
        text = text[:limit] + "...<truncated>"
    return text


def _new_ticket_id(prefix: str = "self") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _base_dir() -> Path:
    try:
        if config is not None and hasattr(config, "BASE_DIR"):
            return Path(str(getattr(config, "BASE_DIR"))).expanduser().resolve()
    except Exception:
        pass
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        return Path.cwd().resolve()


def _data_dir() -> Path:
    try:
        if config is not None and hasattr(config, "DATA_DIR"):
            return Path(str(getattr(config, "DATA_DIR"))).expanduser().resolve()
    except Exception:
        pass
    return (_base_dir() / "data").resolve()


def _datasets_dir() -> Path:
    try:
        if config is not None and hasattr(config, "DATASETS_DIR"):
            return Path(str(getattr(config, "DATASETS_DIR"))).expanduser().resolve()
    except Exception:
        pass
    return (_data_dir() / "memory" / "datasets").resolve()


def _read_json_file(path: Path) -> Dict[str, Any]:
    try:
        if path.exists() and path.is_file():
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    return {}


def _write_selfaware_report(name: str, payload: Dict[str, Any]) -> str:
    try:
        d = _data_dir() / "reports" / "v800" / "selfaware_api"
        d.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", name or "report")[:120] or "report"
        path = d / f"{safe}.json"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        os.replace(str(tmp), str(path))
        return str(path)
    except Exception:
        return ""


def _list_selfaware_reports(limit: int = 10) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        d = _data_dir() / "reports" / "v800" / "selfaware_api"
        if not d.exists():
            return []
        files = sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
        for p in files[: max(1, int(limit or 10))]:
            out.append({
                "name": p.name,
                "path": str(p),
                "mtime": p.stat().st_mtime,
                "mtime_iso": _iso(p.stat().st_mtime),
                "size_bytes": p.stat().st_size,
            })
    except Exception:
        pass
    return out


# -----------------------------------------------------------------------------
# Evidence Court helpers
# -----------------------------------------------------------------------------
# Core doctrine implemented here:
# - One file/person gets one witness vote.
# - One artifact family gets one artifact vote.
# - OS-specific adapters are support/hearsay only, never constitutional quorum.
# - SarahMemoryGlobals.py is the Constitution and is read-only here.
# - SarahMemoryDatabase.py is an evidence locker and is read-only here; multiple DB rows
#   never count as multiple independent witnesses.

_HARDWARE_KINDS = {"cpu", "gpu", "memory", "disk_space", "drive", "drives", "drive_label", "usb_label", "temperature", "fan_speed", "network", "network_card", "wifi_card", "bluetooth_card", "ethernet_card", "lan", "motherboard", "usb", "usb_ports", "usb_devices", "usb_storage", "usb_host_controllers", "sata", "sata_ports", "storage_devices", "storage_topology", "nvme", "pcie", "pci", "ports", "port", "operating_system", "platform", "general_system_fact"}
_SOFTWARE_KINDS = {"software", "application", "process", "window", "desktop_surface"}
_BOOT_RUNTIME_KINDS = {"boot", "startup", "shutdown", "reboot", "runtime", "lifecycle", "server_state"}
_NETWORK_KINDS = {"network", "sarahnet", "sync", "mesh", "rendezvous"}
_REM_CODE_KINDS = {"rem", "rem_medium", "code", "patch", "self_evolution", "sandbox", "module_generation"}
_VISION_KINDS = {"vision", "camera", "face", "object", "scene", "visual"}
_ADAPTIVE_KINDS = {"adaptive", "emotion", "personality", "preference", "behavior"}
_DRIVER_KINDS = {"driver", "drivers", "driver_stack", "device_driver", "body_capability", "body_capabilities", "hardware_capability"}


def _fact_jurisdiction(kind: str, claim: str = "") -> str:
    k = _safe_str(kind, 80).lower()
    c = _safe_str(claim, MAX_CLAIM_CHARS).lower()
    if k in _DRIVER_KINDS or any(x in c for x in ("driver stack", "class a driver", "class b driver", "boot driver", "runtime driver", "device driver", "driver registry", "body capability", "body capabilities")):
        return "driver_capability"
    if k in _SOFTWARE_KINDS or any(x in c for x in ("installed", "running", "focused", "window", "application", "program", "software")):
        return "software_desktop"
    if k in _BOOT_RUNTIME_KINDS or any(x in c for x in ("boot", "startup", "shutdown", "reboot", "server state", "runtime state")):
        return "boot_runtime"
    if k in _REM_CODE_KINDS or any(x in c for x in ("rem", "dream", "patch", "sandbox", "monkey patch", "self evolve", "code proposal")):
        return "rem_code_patch"
    if k in _VISION_KINDS or any(x in c for x in ("camera", "face", "object", "scene", "webcam", "picture", "image")):
        return "vision_environment"
    if k in _ADAPTIVE_KINDS or any(x in c for x in ("emotion", "personality", "preference", "behavior", "adaptive")):
        return "adaptive_behavior"
    if k in _NETWORK_KINDS and k != "network":
        return "network_sarahnet"
    if k in _HARDWARE_KINDS:
        return "hardware_body"
    return "general_system"



def _normalize_value(value: Any, kind: str = "") -> str:
    """Stable, low-noise normalization for Evidence Court comparison.

    This is intentionally local and deterministic. It does not call external
    services, does not mutate state, and does not treat OS adapters as truth.
    """
    if value is None:
        return ""
    k = _safe_str(kind, 80).lower()
    try:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, int):
            return str(value)
        if isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                return ""
            return (f"{value:.6f}".rstrip("0").rstrip("."))
        if isinstance(value, Path):
            return str(value).strip().lower()
        if isinstance(value, dict):
            # Prefer identity labels before full JSON so equivalent witnesses
            # can line up without being derailed by telemetry noise.
            if k == "cpu":
                for key in ("name", "Name", "processor", "Processor", "cpu", "CPU"):
                    if value.get(key) not in (None, "", [], {}):
                        return _normalize_value(value.get(key), k)
            if k == "gpu":
                for key in ("name", "Name", "gpu_name", "Info", "Type", "adapter", "device"):
                    if value.get(key) not in (None, "", [], {}):
                        return _normalize_value(value.get(key), k)
            if k == "motherboard":
                for key in ("motherboard", "baseboard", "mainboard", "name", "Name", "Product", "Manufacturer", "model", "Model"):
                    if value.get(key) not in (None, "", [], {}):
                        # Prefer a combined manufacturer/product string when both exist.
                        manufacturer = value.get("Manufacturer") or value.get("manufacturer")
                        product = value.get("Product") or value.get("product") or value.get("Name") or value.get("name")
                        if manufacturer and product and str(manufacturer).strip().lower() not in str(product).strip().lower():
                            return _normalize_value(f"{manufacturer} {product}", k)
                        return _normalize_value(value.get(key), k)
            if k == "memory":
                for key in ("total_gb", "Total Memory (GB)", "ram_total_gb", "ram_total_mb", "total"):
                    if value.get(key) not in (None, "", [], {}):
                        return _normalize_value(value.get(key), k)
            if k in {"disk_space", "usb_label"}:
                mount = value.get("mountpoint") or value.get("device") or value.get("DeviceID") or value.get("path")
                label = value.get("label") or value.get("VolumeName") or value.get("name")
                total = value.get("total_gb") or value.get("Size") or value.get("total")
                free = value.get("free_gb") or value.get("FreeSpace") or value.get("free")
                return _normalize_value({"mountpoint": mount, "label": label, "total": total, "free": free}, "")
            try:
                return json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")).lower()
            except Exception:
                return _safe_str(value, 20000).lower()
        if isinstance(value, (list, tuple, set)):
            items = [_normalize_value(v, k) for v in list(value)]
            items = [x for x in items if x]
            return "|".join(sorted(set(items)))
        text = _safe_str(value, 20000)
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text
    except Exception:
        return _safe_str(value, 20000).lower()


def _infer_fact_kind(claim: str, explicit_kind: str = "") -> str:
    """Classify the requested SelfAware fact kind without touching Globals.

    V8 keeps body questions separated by topology axis. A storage device may be
    SATA, NVMe-over-PCIe, NVMe-behind-USB, or remote NAS. A drive letter is not
    a bus. A bus is not a device class. This classifier only chooses the court
    question; it does not prove the answer.
    """
    kind = _safe_str(explicit_kind, 80).lower().replace("-", "_").replace(" ", "_")
    c = _safe_str(claim, MAX_CLAIM_CHARS).lower()

    generic_explicit_kinds = {"", "general", "general_system", "general_system_fact", "hardware", "system", "self", "body"}
    if kind and kind not in generic_explicit_kinds:
        aliases = {
            "processor": "cpu",
            "processor_info": "cpu",
            "graphics": "gpu",
            "video_card": "gpu",
            "ram": "memory",
            "mem": "memory",
            "drive": "disk_space",
            "drives": "storage_topology",
            "storage": "storage_topology",
            "storage_devices": "storage_devices",
            "storage_topology": "storage_topology",
            "disk": "disk_space",
            "disc": "disk_space",
            "volume_label": "drive_label",
            "drive_label": "drive_label",
            "usb_label": "usb_label",
            "thermal": "temperature",
            "temp": "temperature",
            "fan": "fan_speed",
            "rpm": "fan_speed",
            "mainboard": "motherboard",
            "baseboard": "motherboard",
            "system_board": "motherboard",
            "board": "motherboard",
            "app": "software",
            "application": "software",
            "program": "software",
            "network_card": "network_card",
            "network_adapter": "network_card",
            "nic": "network_card",
            "wifi": "wifi_card",
            "wi_fi": "wifi_card",
            "wireless": "wifi_card",
            "ethernet": "ethernet_card",
            "lan": "lan",
            "bluetooth": "bluetooth_card",
            "bluetooth_network": "bluetooth_card",
            "device_driver": "drivers",
            "driver_stack": "drivers",
            "body_capability": "drivers",
            "body_capabilities": "drivers",
            "hardware_capability": "drivers",
            "ports": "ports",
            "port": "ports",
            "usbhost": "usb_host_controllers",
            "usb_host": "usb_host_controllers",
            "usb_ports": "usb_ports",
            "sata": "sata_ports",
            "sata_ports": "sata_ports",
            "nvme": "nvme",
            "pcie": "pcie",
            "pci": "pci",
            "os": "operating_system",
            "operating_system": "operating_system",
            "platform": "platform",
        }
        return aliases.get(kind, kind)

    if any(k in c for k in ("driver stack", "class a driver", "class b driver", "boot driver", "runtime driver", "device driver", "driver registry", "body capability", "body capabilities")):
        return "drivers"
    if any(k in c for k in ("motherboard", "mainboard", "baseboard", "system board")):
        return "motherboard"
    if any(k in c for k in ("cpu temp", "processor temp", "temperature", "thermal", "degrees c", "degrees fahrenheit")):
        return "temperature"
    if any(k in c for k in ("fan", "rpm")):
        return "fan_speed"
    if any(k in c for k in ("gpu", "graphics", "video card", "vram")):
        return "gpu"
    if any(k in c for k in ("cpu", "processor")):
        return "cpu"

    if any(k in c for k in ("network card", "network adapter", "nic", "network interface", "ethernet card", "ethernet adapter", "lan adapter")):
        return "network_card"
    if any(k in c for k in ("wifi card", "wi-fi card", "wireless card", "wifi adapter", "wi-fi adapter", "wireless adapter")):
        return "wifi_card"
    if any(k in c for k in ("bluetooth card", "bluetooth adapter", "bluetooth network")):
        return "bluetooth_card"
    if any(k in c for k in ("ip address", "mac address", "lan address")):
        return "network_card"
    if any(k in c for k in ("network", "wifi", "wi-fi", "wireless", "ethernet")):
        return "network_card"

    if any(k in c for k in ("usb port", "usb ports", "how many usb", "usb host", "usb controller", "usb root hub")):
        return "usb_ports"
    if any(k in c for k in ("usb device", "usb devices", "connected usb", "what usb")):
        return "usb_devices"
    if any(k in c for k in ("usb storage", "usb drive", "external usb", "usb hdd", "usb hard drive", "usb ssd")):
        return "usb_storage"
    if any(k in c for k in ("sata port", "sata ports", "how many sata")):
        return "sata_ports"
    if any(k in c for k in ("nvme", "m.2", "m2 drive", "m.2 drive")):
        return "nvme"
    if any(k in c for k in ("pcie", "pci-e", "pci express")):
        return "pcie"
    if re.search(r"\bpci\b", c):
        return "pci"
    if any(k in c for k in ("what drives", "which drives", "storage topology", "storage devices", "drive topology", "hardware topology")):
        return "storage_topology"
    if any(k in c for k in ("operating system", "what os", "which os", "os are you", "os on top of", "platform are you")):
        return "operating_system"
    if any(k in c for k in ("port", "ports", "serial port", "com port", "hdmi port", "displayport")):
        return "ports"

    if any(k in c for k in ("usb", "drive label", "volume label", "label on")):
        return "usb_label"
    if any(k in c for k in ("disk", "disc", "drive", "space", "storage", "free gb", "used gb")):
        return "disk_space"
    if re.search(r"\b(ram|memory)\b", c) and "sarahmemory" not in c.replace(" ", ""):
        return "memory"
    if any(k in c for k in ("sarahnet", "mesh", "rendezvous", "sync")):
        return "sarahnet"
    if any(k in c for k in ("boot", "startup", "shutdown", "reboot", "server state", "runtime state")):
        return "runtime"
    if any(k in c for k in ("core file", "module", ".py", "importable", "approved")):
        return "core_file"
    if any(k in c for k in ("software", "application", "program", "process", "running", "window", "focused")):
        return "software"
    if any(k in c for k in ("rem", "dream", "patch", "sandbox", "self evolve", "code proposal")):
        return "rem_medium"
    if any(k in c for k in ("camera", "face", "object", "scene", "webcam", "image", "picture")):
        return "vision"
    if any(k in c for k in ("emotion", "personality", "preference", "behavior", "adaptive")):
        return "adaptive"
    return "general_system_fact"

def _extract_target_from_claim(claim: str, kind: str) -> str:
    """Best-effort target extraction; safe and non-authoritative."""
    text = _safe_str(claim, MAX_CLAIM_CHARS)
    low = text.lower()
    k = _safe_str(kind, 80).lower()

    if k in {"usb_label", "disk_space"}:
        m = re.search(r"\b([a-zA-Z]):\\?\b", text)
        if m:
            return m.group(1).upper() + ":"
        m = re.search(r"\bdrive\s+([a-zA-Z])\b", low)
        if m:
            return m.group(1).upper() + ":"

    if k == "core_file":
        m = re.search(r"\b((?:SarahMemory|app)[A-Za-z0-9_]*(?:\.py)?)\b", text)
        if m:
            name = m.group(1)
            return name if name.endswith(".py") else name + ".py"

    if k in {"software", "application", "process", "window", "desktop_surface"}:
        quoted = re.search(r"['\"]([^'\"]{2,120})['\"]", text)
        if quoted:
            return quoted.group(1).strip()
        for marker in ("is ", "open ", "launch ", "focus ", "running ", "program ", "application "):
            idx = low.find(marker)
            if idx >= 0:
                tail = text[idx + len(marker):].strip(" ?.!")
                if tail:
                    return tail[:120]

    return ""


def _evidence_source_registry() -> Dict[str, Any]:
    """SelfAware Evidence Court registry. This is descriptive and read-only."""
    return {
        "version": PROJECT_VERSION,
        "rule": "one_file_or_artifact_family_gets_one_vote_no_double_jeopardy",
        "os_adapters": "support_hearsay_only_not_quorum",
        "constitutional_read_only": ["SarahMemoryGlobals.py"],
        "evidence_locker_read_only": ["SarahMemoryDatabase.py", "*.db", "*.json", "*.log"],
        "jurisdictions": {
            "hardware_body": {
                "quorum": [
                    {"family": "SarahMemoryHi", "role": "live_or_cached_telemetry_witness"},
                    {"family": "runtime_environment_snapshot", "role": "persisted_boot_runtime_artifact"},
                    {"family": "SarahMemoryCognitiveSelf", "role": "canonical_body_map_case_file"},
                ],
                "support": ["SarahMemoryDiagnostics", "SarahMemoryOptimization", "SarahMemoryDriverRegistry", "appdrivers.py", "psutil/platform adapters", "OS vendor adapters"],
                "fact_kinds": sorted(_HARDWARE_KINDS),
            },
            "driver_capability": {
                "quorum": [],
                "support": [
                    {"family": "SarahMemoryDriverRegistry", "role": "Class A boot-driver and Class B runtime-driver catalog witness"},
                    {"family": "appdrivers.py", "role": "governed driver API contract witness"},
                ],
                "fact_kinds": sorted(_DRIVER_KINDS),
                "rule": "Driver folders prove native capability only; physical hardware still requires normal SelfAware quorum.",
            },
            "software_desktop": {
                "quorum": [
                    {"family": "SarahMemorySi", "role": "software_process_window_witness"},
                    {"family": "software_artifact", "role": "software_db_or_process_cache_artifact"},
                    {"family": "SarahMemoryCognitiveSelf", "role": "capability_desktop_surface_case_file"},
                ],
                "support": ["SarahMemoryDiagnostics", "SarahMemoryOptimization"],
            },
            "boot_runtime": {
                "quorum": [
                    {"family": "SarahMemoryInitialization", "role": "boot_sequence_witness"},
                    {"family": "SarahMemoryIntegration", "role": "runtime_orchestration_witness"},
                    {"family": "CognitiveSelf_lifecycle", "role": "lifecycle_case_file"},
                ],
                "support": ["server_state.json", "PID files", "Diagnostics"],
            },
            "network_sarahnet": {
                "quorum": [
                    {"family": "SarahMemoryNetwork", "role": "protocol_transport_witness"},
                    {"family": "SarahMemorySync", "role": "device_link_sync_witness"},
                    {"family": "network_artifact", "role": "rendezvous_or_network_log_artifact"},
                ],
                "support": ["SarahNetMCP_Diagnostics", "SarahMemoryHi network adapters"],
            },
            "rem_code_patch": {
                "quorum": [
                    {"family": "SarahMemorySelfAware", "role": "investigator_witness"},
                    {"family": "SarahMemorySynapes", "role": "sandbox_lab_provenance_witness"},
                    {"family": "SarahMemoryEvolution", "role": "restricted_monkey_patch_proposal_witness"},
                ],
                "support": ["AssuranceGate", "SecurityGovernor", "CognitiveServices"],
            },
            "vision_environment": {
                "quorum": [
                    {"family": "SarahMemoryFacialRecognition", "role": "face_vector_vision_witness"},
                    {"family": "SarahMemorySOBJE", "role": "object_scene_observation_witness"},
                    {"family": "vision_artifact", "role": "vision_log_or_observation_artifact"},
                ],
                "support": ["CognitiveSelf capabilities", "Diagnostics"],
            },
            "adaptive_behavior": {
                "quorum": [
                    {"family": "SarahMemoryAdaptive", "role": "behavior_emotion_witness"},
                    {"family": "adaptive_artifact", "role": "personality_or_interaction_artifact"},
                    {"family": "SarahMemoryAiFunctions", "role": "agent_context_or_interaction_witness"},
                ],
                "support": ["DL", "Reply"],
            },
        },
    }


def _runtime_environment_snapshot_path() -> Path:
    return _datasets_dir() / "runtime_environment_snapshot.json"


def _server_state_path() -> Path:
    return _data_dir() / "server_state.json"


def _software_db_path() -> Path:
    return _datasets_dir() / "software.db"


def _system_logs_db_path() -> Path:
    return _datasets_dir() / "system_logs.db"


# -----------------------------------------------------------------------------
# Driver Registry / Body Capability Evidence helpers
# -----------------------------------------------------------------------------
# These helpers make SelfAware driver-aware without becoming a driver manager.
# They only read manifests, defaults/config metadata, boot_driver registry data,
# and folder presence. They never import driver.py, never start sessions, never
# open ports, never scan networks, never pair devices, and never mutate configs.

def _boot_dir() -> Path:
    return (_data_dir() / "boot").resolve()


def _boot_drivers_root() -> Path:
    return (_boot_dir() / "drivers").resolve()


def _runtime_drivers_root() -> Path:
    return (_data_dir() / "drivers").resolve()


def _driver_registry_path() -> Path:
    return (_data_dir() / "registry" / "drivers.json").resolve()


def _boot_registry_candidates() -> List[Path]:
    # Historical layouts are both supported:
    #   data/boot/boot_drivers.json        (appdrivers.py current contract)
    #   data/boot/drivers/boot_drivers.json (observed development folder layout)
    return [
        (_boot_dir() / "boot_drivers.json").resolve(),
        (_boot_drivers_root() / "boot_drivers.json").resolve(),
    ]


def _boot_registry_path() -> Path:
    for p in _boot_registry_candidates():
        try:
            if p.exists() and p.is_file():
                return p
        except Exception:
            continue
    return _boot_registry_candidates()[0]


def _sha256_file(path: Path, limit_bytes: int = 8 * 1024 * 1024) -> str:
    try:
        if not path.exists() or not path.is_file():
            return ""
        h = hashlib.sha256()
        total = 0
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit_bytes:
                    return ""
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _read_boot_driver_registry() -> Dict[str, Any]:
    path = _boot_registry_path()
    blob = _read_json_file(path)
    entries = blob.get("drivers") if isinstance(blob, dict) else []
    if not isinstance(entries, list):
        entries = []
    indexed: Dict[str, Dict[str, Any]] = {}
    for item in entries:
        if not isinstance(item, dict):
            continue
        driver_id = _safe_str(item.get("id"), 220)
        if not driver_id:
            continue
        dpath_raw = _safe_str(item.get("path"), 500)
        if dpath_raw:
            dpath = (Path(dpath_raw) if Path(dpath_raw).is_absolute() else (_base_dir() / dpath_raw)).resolve()
        else:
            dpath = (_boot_drivers_root() / driver_id).resolve()
        manifest_path = dpath / "manifest.json"
        manifest = _read_json_file(manifest_path)
        indexed[driver_id] = {
            "id": driver_id,
            "class": "A",
            "driver_class": "boot_foundation",
            "path": str(dpath),
            "manifest_path": str(manifest_path),
            "folder_exists": dpath.exists() and dpath.is_dir(),
            "manifest_exists": manifest_path.exists() and manifest_path.is_file(),
            "manifest": manifest if isinstance(manifest, dict) else {},
            "manufacturer": item.get("manufacturer"),
            "trust_level": item.get("trust_level"),
            "signature_type": item.get("signature_type"),
            "driver_signature": item.get("driver_signature"),
            "declared_hash": item.get("hash"),
            "manifest_sha256": _sha256_file(manifest_path),
            "level": item.get("level"),
            "load_priority": item.get("load_priority"),
            "dependencies": item.get("dependencies") if isinstance(item.get("dependencies"), list) else [],
            "enabled": bool(item.get("enabled", True)),
        }
    return {
        "ok": bool(indexed),
        "class": "A",
        "registry_path": str(path),
        "registry_exists": path.exists() and path.is_file(),
        "count": len(indexed),
        "drivers": indexed,
        "action_taken": False,
        "read_only": True,
    }


def _boot_driver_order(entries: Dict[str, Dict[str, Any]]) -> List[str]:
    if not isinstance(entries, dict) or not entries:
        return []
    graph: Dict[str, set] = {}
    indegree: Dict[str, int] = {}
    levels: Dict[str, int] = {}
    for driver_id, meta in entries.items():
        deps = meta.get("dependencies") if isinstance(meta.get("dependencies"), list) else []
        graph[driver_id] = set(str(x) for x in deps if str(x) in entries and str(x) != driver_id)
        indegree.setdefault(driver_id, 0)
        try:
            levels[driver_id] = int(meta.get("level", meta.get("load_priority", 999)) or 999)
        except Exception:
            levels[driver_id] = 999
    for driver_id, deps in graph.items():
        for dep in deps:
            indegree[driver_id] = indegree.get(driver_id, 0) + 1
            indegree.setdefault(dep, 0)
    ready = sorted([x for x, deg in indegree.items() if deg == 0], key=lambda x: (levels.get(x, 999), x))
    ordered: List[str] = []
    while ready:
        driver_id = ready.pop(0)
        ordered.append(driver_id)
        for other, deps in graph.items():
            if driver_id in deps:
                deps.remove(driver_id)
                indegree[other] -= 1
                if indegree[other] == 0:
                    ready.append(other)
                    ready.sort(key=lambda x: (levels.get(x, 999), x))
    for driver_id in sorted(entries.keys(), key=lambda x: (levels.get(x, 999), x)):
        if driver_id not in ordered:
            ordered.append(driver_id)
    return ordered


def _flatten_driver_manifest_text(manifest: Optional[Dict[str, Any]]) -> str:
    try:
        return json.dumps(_json_safe(manifest or {}), ensure_ascii=False, sort_keys=True).lower()
    except Exception:
        return _safe_str(manifest, 4000).lower()


def _classify_driver_families(driver_id: str, manifest: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return multi-axis driver families for capability matching.

    A Class B driver can be both a device class and a transport. Example:
    a network printer is printer + network; a Bluetooth gamepad is input +
    bluetooth; an NVMe device may be storage + pcie. This prevents V7-style
    over-broad storage matches.
    """
    did = _safe_str(driver_id, 240).lower().replace("_", ".").replace("-", ".")
    text = did + " " + _flatten_driver_manifest_text(manifest)
    families: List[str] = []

    def add(name: str) -> None:
        if name and name not in families:
            families.append(name)

    if ".boot." in did:
        add("boot_foundation")
        return families

    # Specific device classes first.
    if any(x in text for x in ("motherboard", "baseboard", "mainboard", "bios", "uefi", "acpi")):
        add("motherboard")
    if any(x in text for x in ("camera", "uvc", "webcam", "vision")):
        add("camera_vision")
    if any(x in text for x in ("plc", "modbus", "opcua")):
        add("industrial_plc")
    if any(x in text for x in ("cnc", "grbl")):
        add("cnc_serial")
    if any(x in text for x in ("arduino", "microcontroller")):
        add("microcontroller")
    if any(x in text for x in ("printer", "esc/pos", "escpos", "receipt")):
        add("printer")
    if any(x in text for x in ("rfid", "zigbee", "z-wave", "zwave", "mqtt", "smartdevice", "smart-device", "sensor", "lirc", "infrared", "ir ")):
        add("iot_sensor")
    if any(x in text for x in ("vr", "xr", "headset")):
        add("vr")
    if any(x in text for x in ("midi", "audio", "speaker", "soundbar", "microphone", "ac97")):
        add("audio")
    if any(x in text for x in ("keyboard", "mouse", "gamepad", "controller", "joystick", "barcode", "hid", "input")):
        add("input")
    if any(x in text for x in ("display", "vga", "hdmi", "displayport", "monitor", "framebuffer", "gpu", "video")):
        add("display")
    if any(x in text for x in ("storage", "filesystem", "disk", "drive", "nvme", "smart", "vault_mount", "block-device", "partition")):
        add("storage")

    # Transport/protocol axes.
    if any(x in text for x in ("network", "ethernet", "wifi", "wi-fi", "wireless", "tcp", "http", "mqtt", "sip", "utpsip", "opc.tcp", "network.outbound")):
        add("network")
    if "bluetooth" in text or "ble" in text:
        add("bluetooth")
    if any(x in text for x in ("usb", "usb_serial", "usb/serial", "uasp")):
        add("usb")
    if any(x in text for x in ("serial", "com port", "rs232", "uart")):
        add("serial")
    if any(x in text for x in ("pci", "pcie", "pci express", "pci-e")):
        add("pcie")
    if "nvme" in text:
        add("nvme")
    if "sata" in text:
        add("sata")

    if not families:
        add("generic_device")
    return families


def _classify_driver_family(driver_id: str, manifest: Optional[Dict[str, Any]] = None) -> str:
    return _classify_driver_families(driver_id, manifest)[0]

def _read_runtime_driver_catalog() -> Dict[str, Any]:
    root = _runtime_drivers_root()
    reg = _read_json_file(_driver_registry_path())
    reg_entries = reg if isinstance(reg, dict) else {}
    drivers: Dict[str, Dict[str, Any]] = {}
    try:
        if root.exists() and root.is_dir():
            for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
                if not p.is_dir():
                    continue
                driver_id = p.name
                manifest_path = p / "manifest.json"
                defaults_path = p / "defaults.json"
                config_path = p / "config.json"
                ui_path = p / "ui.json"
                driver_py_path = p / "driver.py"
                manifest = _read_json_file(manifest_path)
                families = _classify_driver_families(driver_id, manifest)
                drivers[driver_id] = {
                    "id": driver_id,
                    "class": "B",
                    "driver_class": "runtime_device_capability",
                    "family": families[0] if families else "generic_device",
                    "families": families,
                    "path": str(p),
                    "folder_exists": True,
                    "manifest_exists": manifest_path.exists() and manifest_path.is_file(),
                    "ui_schema_exists": ui_path.exists() and ui_path.is_file(),
                    "defaults_exists": defaults_path.exists() and defaults_path.is_file(),
                    "config_exists": config_path.exists() and config_path.is_file(),
                    "driver_py_exists": driver_py_path.exists() and driver_py_path.is_file(),
                    "manifest": manifest if isinstance(manifest, dict) else {},
                    "manifest_sha256": _sha256_file(manifest_path),
                    "enabled": bool((reg_entries.get(driver_id) if isinstance(reg_entries.get(driver_id), dict) else {}).get("enabled", (manifest or {}).get("enabled", True))),
                    "autoload_requested": bool((manifest or {}).get("autoload", False)),
                    "action_taken": False,
                }
    except Exception as exc:
        return {
            "ok": False,
            "class": "B",
            "drivers_root": str(root),
            "drivers_root_exists": root.exists(),
            "error": f"{type(exc).__name__}:{exc}",
            "drivers": drivers,
            "count": len(drivers),
            "action_taken": False,
            "read_only": True,
        }
    return {
        "ok": True,
        "class": "B",
        "drivers_root": str(root),
        "drivers_root_exists": root.exists() and root.is_dir(),
        "registry_path": str(_driver_registry_path()),
        "registry_exists": _driver_registry_path().exists(),
        "drivers": drivers,
        "count": len(drivers),
        "action_taken": False,
        "read_only": True,
    }


def _driver_kind_profile(kind: str) -> Dict[str, Any]:
    k = _safe_str(kind, 80).lower()
    profiles: Dict[str, Dict[str, Any]] = {
        "cpu": {"class_a": ["com.softdev0.boot.firmware", "com.softdev0.boot.cpuarch"], "families": ["boot_foundation"]},
        "gpu": {"class_a": ["com.softdev0.boot.pci", "com.softdev0.boot.displaycore"], "families": ["display", "pcie"]},
        "motherboard": {"class_a": ["com.softdev0.boot.firmware", "com.softdev0.boot.pci", "com.softdev0.boot.usbhost"], "families": ["motherboard"]},
        "memory": {"class_a": ["com.softdev0.boot.firmware", "com.softdev0.boot.cpuarch"], "families": ["boot_foundation"]},
        "disk_space": {"class_a": ["com.softdev0.boot.storagecore", "com.softdev0.boot.filesystems"], "families": ["storage"]},
        "drive": {"class_a": ["com.softdev0.boot.storagecore", "com.softdev0.boot.filesystems"], "families": ["storage"]},
        "drives": {"class_a": ["com.softdev0.boot.storagecore", "com.softdev0.boot.filesystems"], "families": ["storage"]},
        "storage_devices": {"class_a": ["com.softdev0.boot.storagecore", "com.softdev0.boot.filesystems"], "families": ["storage"]},
        "storage_topology": {"class_a": ["com.softdev0.boot.storagecore", "com.softdev0.boot.filesystems", "com.softdev0.boot.pci", "com.softdev0.boot.usbhost"], "families": ["storage", "usb", "pcie", "sata", "nvme", "network"]},
        "usb_label": {"class_a": ["com.softdev0.boot.usbhost", "com.softdev0.boot.storagecore", "com.softdev0.boot.filesystems"], "families": ["usb", "storage"]},
        "drive_label": {"class_a": ["com.softdev0.boot.storagecore", "com.softdev0.boot.filesystems"], "families": ["storage", "usb"]},
        "usb": {"class_a": ["com.softdev0.boot.usbhost"], "families": ["usb", "input", "printer", "iot_sensor", "vr", "camera_vision", "microcontroller"]},
        "usb_ports": {"class_a": ["com.softdev0.boot.usbhost"], "families": ["usb"]},
        "usb_devices": {"class_a": ["com.softdev0.boot.usbhost"], "families": ["usb", "input", "printer", "iot_sensor", "vr", "camera_vision", "microcontroller", "storage"]},
        "usb_storage": {"class_a": ["com.softdev0.boot.usbhost", "com.softdev0.boot.storagecore", "com.softdev0.boot.filesystems"], "families": ["usb", "storage"]},
        "usb_host_controllers": {"class_a": ["com.softdev0.boot.usbhost"], "families": ["usb"]},
        "sata": {"class_a": ["com.softdev0.boot.storagecore", "com.softdev0.boot.filesystems"], "families": ["sata", "storage"]},
        "sata_ports": {"class_a": ["com.softdev0.boot.storagecore", "com.softdev0.boot.filesystems"], "families": ["sata", "storage"]},
        "nvme": {"class_a": ["com.softdev0.boot.pci", "com.softdev0.boot.storagecore", "com.softdev0.boot.filesystems"], "families": ["nvme", "pcie", "storage", "usb"]},
        "pcie": {"class_a": ["com.softdev0.boot.pci"], "families": ["pcie", "display", "storage", "network"]},
        "pci": {"class_a": ["com.softdev0.boot.pci"], "families": ["pcie", "display", "storage", "network"]},
        "ports": {"class_a": ["com.softdev0.boot.pci", "com.softdev0.boot.usbhost", "com.softdev0.boot.networkcore", "com.softdev0.boot.displaycore", "com.softdev0.boot.inputcore"], "families": ["network", "usb", "display", "input", "industrial_plc", "iot_sensor", "printer", "serial"]},
        "network": {"class_a": ["com.softdev0.boot.pci", "com.softdev0.boot.usbhost", "com.softdev0.boot.networkcore"], "families": ["network", "bluetooth", "usb", "pcie"]},
        "network_card": {"class_a": ["com.softdev0.boot.pci", "com.softdev0.boot.usbhost", "com.softdev0.boot.networkcore"], "families": ["network", "bluetooth", "usb", "pcie"]},
        "wifi_card": {"class_a": ["com.softdev0.boot.pci", "com.softdev0.boot.usbhost", "com.softdev0.boot.networkcore"], "families": ["network", "wifi", "usb", "pcie"]},
        "bluetooth_card": {"class_a": ["com.softdev0.boot.usbhost", "com.softdev0.boot.networkcore"], "families": ["bluetooth", "usb", "network"]},
        "ethernet_card": {"class_a": ["com.softdev0.boot.pci", "com.softdev0.boot.networkcore"], "families": ["network", "pcie"]},
        "lan": {"class_a": ["com.softdev0.boot.pci", "com.softdev0.boot.networkcore"], "families": ["network", "pcie"]},
        "operating_system": {"class_a": ["com.softdev0.boot.firmware", "com.softdev0.boot.cpuarch"], "families": ["boot_foundation"]},
        "platform": {"class_a": ["com.softdev0.boot.firmware", "com.softdev0.boot.cpuarch"], "families": ["boot_foundation"]},
        "temperature": {"class_a": ["com.softdev0.boot.firmware", "com.softdev0.boot.pci"], "families": ["motherboard", "display"]},
        "fan_speed": {"class_a": ["com.softdev0.boot.firmware", "com.softdev0.boot.pci"], "families": ["motherboard", "display"]},
        "drivers": {"class_a": [], "families": []},
        "general_system_fact": {"class_a": [], "families": []},
    }
    return profiles.get(k, {"class_a": [], "families": []})

def _build_driver_capability_snapshot(kind: str = "") -> Dict[str, Any]:
    kind = _infer_fact_kind(kind or "drivers", kind) if kind else "drivers"
    boot = _read_boot_driver_registry()
    runtime = _read_runtime_driver_catalog()
    boot_entries = boot.get("drivers") if isinstance(boot.get("drivers"), dict) else {}
    runtime_entries = runtime.get("drivers") if isinstance(runtime.get("drivers"), dict) else {}
    profile = _driver_kind_profile(kind)
    ordered_boot = _boot_driver_order(boot_entries if isinstance(boot_entries, dict) else {})

    class_a_ids = profile.get("class_a") if isinstance(profile.get("class_a"), list) else []
    families = profile.get("families") if isinstance(profile.get("families"), list) else []
    if not class_a_ids and kind == "drivers":
        class_a_ids = ordered_boot

    class_a_stack: List[Dict[str, Any]] = []
    for did in class_a_ids:
        meta = boot_entries.get(did) if isinstance(boot_entries, dict) else None
        if isinstance(meta, dict):
            class_a_stack.append({
                "id": did,
                "present": bool(meta.get("folder_exists") or meta.get("manifest_exists")),
                "manifest_exists": bool(meta.get("manifest_exists")),
                "trust_level": meta.get("trust_level"),
                "signature_type": meta.get("signature_type"),
                "dependencies": meta.get("dependencies") if isinstance(meta.get("dependencies"), list) else [],
                "level": meta.get("level"),
                "load_priority": meta.get("load_priority"),
            })
        else:
            class_a_stack.append({"id": did, "present": False, "missing": True})

    class_b_matches: List[Dict[str, Any]] = []
    for did, meta in (runtime_entries.items() if isinstance(runtime_entries, dict) else []):
        if not isinstance(meta, dict):
            continue
        fam = _safe_str(meta.get("family"), 80)
        fams = meta.get("families") if isinstance(meta.get("families"), list) else ([fam] if fam else [])
        fams = [_safe_str(x, 80) for x in fams if _safe_str(x, 80)]
        if kind == "drivers" or not families or any(f in families for f in fams):
            class_b_matches.append({
                "id": did,
                "family": fam,
                "families": fams,
                "manifest_exists": bool(meta.get("manifest_exists")),
                "ui_schema_exists": bool(meta.get("ui_schema_exists")),
                "defaults_exists": bool(meta.get("defaults_exists")),
                "config_exists": bool(meta.get("config_exists")),
                "driver_py_exists": bool(meta.get("driver_py_exists")),
                "enabled": bool(meta.get("enabled", True)),
                "autoload_requested": bool(meta.get("autoload_requested", False)),
            })

    missing_class_a = [x.get("id") for x in class_a_stack if not x.get("present")]
    missing_class_b = [] if (kind == "drivers" or class_b_matches or not families) else list(families)
    return {
        "kind": kind,
        "class_a": {
            "registry_path": boot.get("registry_path"),
            "registry_exists": bool(boot.get("registry_exists")),
            "drivers_root": str(_boot_drivers_root()),
            "ordered_boot_ids": ordered_boot,
            "required_stack": class_a_stack,
            "missing_required": missing_class_a,
            "count": boot.get("count", 0),
        },
        "class_b": {
            "drivers_root": runtime.get("drivers_root"),
            "drivers_root_exists": bool(runtime.get("drivers_root_exists")),
            "matched_drivers": class_b_matches,
            "matched_count": len(class_b_matches),
            "catalog_count": runtime.get("count", 0),
            "missing_family_matches": missing_class_b,
        },
        "dependency_status": "missing_required_driver" if missing_class_a or missing_class_b else "declared_or_available",
        "capability_present": bool((class_a_stack and not missing_class_a) or class_b_matches or kind == "drivers"),
        "read_only": True,
        "action_taken": False,
        "driver_code_imported": False,
        "driver_session_started": False,
        "security_boundary": "driver_registry_evidence_only_no_activation_no_ports_no_scans_no_pairing_no_config_mutation",
    }


def _driver_capability_support_fact(kind: str, target: str = "") -> Dict[str, Any]:
    snapshot = _build_driver_capability_snapshot(kind or target or "drivers")
    ok = bool(snapshot.get("capability_present"))
    return _probe_result(
        "SarahMemoryDriverRegistry.classA_classB_catalog",
        snapshot,
        ok=ok,
        error="no_matching_driver_capability" if not ok else "",
        kind=kind,
        source_family="SarahMemoryDriverRegistry",
        evidence_class="native_driver_capability_support",
        support_only=True,
        details={
            "class_a_root": str(_boot_drivers_root()),
            "class_b_root": str(_runtime_drivers_root()),
            "read_only": True,
            "driver_activation": False,
        },
    )


def _appdrivers_contract_fact(kind: str = "drivers", target: str = "") -> Dict[str, Any]:
    try:
        import appdrivers as _AppDrivers  # type: ignore
    except Exception as exc:
        return _probe_result(
            "appdrivers.module_contract",
            None,
            ok=False,
            error=f"module_unavailable:{exc}",
            kind=kind,
            source_family="appdrivers",
            evidence_class="api_domain_contract",
            support_only=True,
        )
    value = {
        "module_available": True,
        "has_init_app": callable(getattr(_AppDrivers, "init_app", None)),
        "has_driver_discovery": callable(getattr(_AppDrivers, "_discover_driver_ids", None)),
        "has_boot_registry_reader": callable(getattr(_AppDrivers, "_boot_registry_entries", None)),
        "has_boot_order": callable(getattr(_AppDrivers, "_boot_driver_order", None)),
        "safe_mode_function_present": callable(getattr(_AppDrivers, "_safe_mode", None)),
        "driver_activation_performed": False,
        "read_only_contract_probe": True,
    }
    return _probe_result(
        "appdrivers.module_contract",
        value,
        ok=True,
        kind=kind,
        source_family="appdrivers",
        evidence_class="api_domain_contract",
        support_only=True,
    )


def _body_capabilities_snapshot() -> Dict[str, Any]:
    kinds = ["cpu", "gpu", "motherboard", "memory", "disk_space", "storage_topology", "usb_ports", "usb_devices", "sata_ports", "nvme", "pcie", "ports", "network_card", "wifi_card", "bluetooth_card", "operating_system", "temperature", "fan_speed"]
    return {
        "ok": True,
        "module": "appself.py",
        "capability_model": "Class A boot foundation + Class B runtime driver catalog + SelfAware evidence court",
        "capabilities": {k: _build_driver_capability_snapshot(k) for k in kinds},
        "doctrine": {
            "driver_present_means_native_capability": True,
            "driver_present_does_not_prove_physical_device": True,
            "physical_device_still_requires_evidence_quorum": True,
            "driver_control_is_not_performed_here": True,
        },
        "read_only": True,
        "action_taken": False,
    }

# -----------------------------------------------------------------------------
# V8 Hardware Topology helpers
# -----------------------------------------------------------------------------
# These helpers keep facts multi-axis instead of flattening everything into
# "storage", "USB", or "network". A drive letter is not a hardware identity;
# a bus is not a device class; a protocol is not a physical location.

_TOPOLOGY_FACT_KINDS = {
    "usb", "usb_ports", "usb_devices", "usb_storage", "usb_host_controllers",
    "sata", "sata_ports", "storage_devices", "storage_topology", "nvme", "pcie", "pci",
    "ports", "operating_system", "platform", "drive", "drives", "drive_label",
}


def _listify(value: Any) -> List[Any]:
    if value in (None, "", {}, []):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _storage_row_mount(row: Dict[str, Any]) -> str:
    return _safe_str(row.get("mountpoint") or row.get("device") or row.get("DeviceID") or row.get("path") or "", 80).rstrip("\\")


def _infer_volume_connection(row: Dict[str, Any]) -> Dict[str, Any]:
    """Infer a conservative connection path from storage/mount metadata.

    This never claims a bus unless the current evidence clearly says it. Mounted
    drive letters and psutil partitions are mount surfaces, not physical proof.
    """
    opts = _safe_str(row.get("opts"), 300).lower()
    device = _safe_str(row.get("device") or row.get("DeviceID") or row.get("mountpoint"), 160).lower()
    fstype = _safe_str(row.get("fstype") or row.get("FileSystem"), 80)
    mount = _storage_row_mount(row)
    if "cdrom" in opts or "cdrom" in device:
        return {
            "device_class": "storage",
            "storage_subclass": "optical_or_cdrom",
            "connection_bus": "unknown_optical_path",
            "bus_confidence": "LOW",
            "physical_location": "local_or_virtual_optical_device",
            "mount_surface": mount,
            "filesystem": fstype,
            "ownership_scope": "local_or_virtual",
            "note": "CD/DVD mount detected; physical bus path is not proven by mount data alone.",
        }
    if "removable" in opts or "removable" in device:
        return {
            "device_class": "storage",
            "storage_subclass": "removable_or_external_volume",
            "connection_bus": "usb_or_removable_candidate",
            "bus_confidence": "MEDIUM_LOW",
            "physical_location": "external_or_removable_media",
            "mount_surface": mount,
            "filesystem": fstype,
            "ownership_scope": "local_external_candidate",
            "note": "Removable storage is often USB, but the exact bus still requires USB/device topology evidence.",
        }
    if "fixed" in opts or (mount and re.match(r"^[a-zA-Z]:$", mount)):
        return {
            "device_class": "storage",
            "storage_subclass": "fixed_volume",
            "connection_bus": "local_fixed_bus_unverified",
            "bus_confidence": "LOW",
            "physical_location": "local_internal_or_expansion_candidate",
            "mount_surface": mount,
            "filesystem": fstype,
            "ownership_scope": "local_fixed_candidate",
            "note": "A fixed drive letter does not prove SATA, NVMe, PCIe, or USB bridge by itself.",
        }
    return {
        "device_class": "storage",
        "storage_subclass": "unknown_volume",
        "connection_bus": "unknown",
        "bus_confidence": "LOW",
        "physical_location": "unknown",
        "mount_surface": mount,
        "filesystem": fstype,
        "ownership_scope": "unknown",
        "note": "Mount data alone cannot prove physical connection path.",
    }


def _storage_rows_from_snapshot(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    body = snapshot.get("body") if isinstance(snapshot.get("body"), dict) else {}
    rows = body.get("storage") if isinstance(body.get("storage"), list) else []
    out: List[Dict[str, Any]] = []
    for item in rows:
        if isinstance(item, dict):
            mount = _storage_row_mount(item)
            if not mount:
                continue
            safe = {
                "device": _safe_str(item.get("device"), 120),
                "mountpoint": _safe_str(item.get("mountpoint"), 120),
                "fstype": _safe_str(item.get("fstype"), 80),
                "opts": _safe_str(item.get("opts"), 160),
            }
            for key in ("total_gb", "used_gb", "free_gb", "usage_pct"):
                if item.get(key) not in (None, "", [], {}):
                    safe[key] = item.get(key)
            safe["topology"] = _infer_volume_connection(item)
            out.append(safe)
    return out


def _storage_topology_from_snapshot(snapshot: Dict[str, Any], target: str = "") -> Dict[str, Any]:
    rows = _storage_rows_from_snapshot(snapshot)
    if target:
        t = _safe_str(target, 40).upper().rstrip("\\")
        rows = [r for r in rows if _safe_str(r.get("mountpoint") or r.get("device"), 80).upper().rstrip("\\").startswith(t)]
    removable = [r for r in rows if "removable" in _safe_str(r.get("opts"), 160).lower()]
    fixed = [r for r in rows if "fixed" in _safe_str(r.get("opts"), 160).lower()]
    cdrom = [r for r in rows if "cdrom" in _safe_str(r.get("opts"), 160).lower()]
    return {
        "topology_kind": "storage_topology",
        "device_class": "storage",
        "volumes": rows,
        "counts": {
            "volume_count": len(rows),
            "fixed_volume_count": len(fixed),
            "removable_or_external_candidate_count": len(removable),
            "cdrom_or_optical_count": len(cdrom),
        },
        "classification_rule": {
            "drive_letter_is_mount_surface_not_hardware_identity": True,
            "fixed_volume_does_not_prove_sata_or_nvme": True,
            "removable_volume_does_not_prove_usb_without_device_topology": True,
            "nas_or_mapped_network_drive_requires_network_protocol_evidence": True,
        },
        "action_taken": False,
    }


def _usb_topology_from_snapshot(snapshot: Dict[str, Any], target: str = "") -> Dict[str, Any]:
    storage = _storage_topology_from_snapshot(snapshot, target)
    rows = storage.get("volumes") if isinstance(storage.get("volumes"), list) else []
    removable = [r for r in rows if "removable" in _safe_str(r.get("opts"), 160).lower()]
    driver_stack = _build_driver_capability_snapshot("usb_ports")
    return {
        "topology_kind": "usb_topology",
        "device_class": "bus_or_external_device_path",
        "connection_bus": "USB",
        "usb_host_capability_present": bool(driver_stack.get("capability_present")),
        "physical_usb_port_count": None,
        "physical_usb_port_count_verified": False,
        "physical_count_confidence": "NONE",
        "connected_usb_storage_candidates": removable,
        "connected_usb_storage_candidate_count": len(removable),
        "class_a_driver_stack": (driver_stack.get("class_a") or {}).get("required_stack") if isinstance(driver_stack.get("class_a"), dict) else [],
        "class_b_driver_matches": (driver_stack.get("class_b") or {}).get("matched_drivers") if isinstance(driver_stack.get("class_b"), dict) else [],
        "note": "USB host capability can be proven from SarahMemory drivers, but physical USB port count requires USB controller/root-hub topology evidence that is not derived from drive letters.",
        "action_taken": False,
    }


def _sata_topology_from_snapshot(snapshot: Dict[str, Any], target: str = "") -> Dict[str, Any]:
    storage = _storage_topology_from_snapshot(snapshot, target)
    rows = storage.get("volumes") if isinstance(storage.get("volumes"), list) else []
    fixed = [r for r in rows if "fixed" in _safe_str(r.get("opts"), 160).lower()]
    return {
        "topology_kind": "sata_topology",
        "device_class": "storage_bus_question",
        "connection_bus": "SATA",
        "sata_port_count": None,
        "sata_port_count_verified": False,
        "sata_confidence": "NONE",
        "local_fixed_storage_candidates": fixed,
        "local_fixed_storage_candidate_count": len(fixed),
        "note": "Fixed local volumes may be SATA, NVMe/PCIe, hardware RAID, virtual, or another local path. SATA port count requires controller/topology evidence, not drive letters.",
        "action_taken": False,
    }


def _nvme_topology_from_snapshot(snapshot: Dict[str, Any], target: str = "") -> Dict[str, Any]:
    storage = _storage_topology_from_snapshot(snapshot, target)
    return {
        "topology_kind": "nvme_topology",
        "device_class": "storage",
        "storage_protocol": "NVMe_candidate",
        "connection_bus": "PCIe_or_USB_bridge_unverified",
        "physical_location": "motherboard_m2_or_pcie_adapter_or_external_bridge_unverified",
        "mount_surface": "see volumes",
        "volumes": storage.get("volumes", []),
        "nvme_device_count": None,
        "nvme_device_count_verified": False,
        "note": "NVMe is a storage protocol usually exposed through PCIe, but can appear through adapter cards, motherboard M.2 slots, USB enclosures, NAS, or virtualization. Current mount data alone cannot prove the path.",
        "action_taken": False,
    }


def _pci_topology_from_snapshot(snapshot: Dict[str, Any], target: str = "") -> Dict[str, Any]:
    driver_stack = _build_driver_capability_snapshot("pcie")
    body = snapshot.get("body") if isinstance(snapshot.get("body"), dict) else {}
    return {
        "topology_kind": "pci_pcie_topology",
        "device_class": "system_bus",
        "connection_bus": "PCI/PCIe",
        "pci_pcie_capability_present": bool(driver_stack.get("capability_present")),
        "visible_pcie_related_devices": {
            "gpu": body.get("gpu") if isinstance(body.get("gpu"), dict) else None,
            "motherboard": body.get("motherboard"),
        },
        "class_a_driver_stack": (driver_stack.get("class_a") or {}).get("required_stack") if isinstance(driver_stack.get("class_a"), dict) else [],
        "note": "PCI/PCIe bus capability is available through the Class A driver stack, but exact slot/device topology requires deeper bus enumeration.",
        "action_taken": False,
    }


def _ports_topology_from_snapshot(snapshot: Dict[str, Any], target: str = "") -> Dict[str, Any]:
    return {
        "topology_kind": "ports_topology",
        "usb": _usb_topology_from_snapshot(snapshot, target),
        "sata": _sata_topology_from_snapshot(snapshot, target),
        "pci_pcie": _pci_topology_from_snapshot(snapshot, target),
        "network": _network_adapter_safe_summary(_select_snapshot_fact(snapshot, "network", target)),
        "rule": "Ports are connection surfaces. Mounted drives, network adapters, and display devices are not port counts by themselves.",
        "action_taken": False,
    }


def _operating_system_safe_summary(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    plat = snapshot.get("platform") if isinstance(snapshot.get("platform"), dict) else {}
    return {
        "topology_kind": "operating_system_platform",
        "system": plat.get("system") or platform.system(),
        "release": plat.get("release") or platform.release(),
        "version": plat.get("version") or platform.version(),
        "machine": plat.get("machine") or platform.machine(),
        "python": plat.get("python") or platform.python_version(),
        "os_boot_time": snapshot.get("os_boot_time"),
        "schema_version": snapshot.get("schema_version"),
        "source": snapshot.get("source"),
        "privacy_note": "Network addressing, adapter IPs, MACs, and full body-map details are excluded from OS presentation.",
        "action_taken": False,
    }


def _topology_fact_from_snapshot(snapshot: Dict[str, Any], kind: str, target: str = "") -> Any:
    k = _safe_str(kind, 80).lower()
    if k in {"storage_topology", "storage_devices", "drive", "drives"}:
        return _storage_topology_from_snapshot(snapshot, target)
    if k in {"usb", "usb_ports", "usb_devices", "usb_storage", "usb_host_controllers"}:
        return _usb_topology_from_snapshot(snapshot, target)
    if k in {"sata", "sata_ports"}:
        return _sata_topology_from_snapshot(snapshot, target)
    if k == "nvme":
        return _nvme_topology_from_snapshot(snapshot, target)
    if k in {"pci", "pcie"}:
        return _pci_topology_from_snapshot(snapshot, target)
    if k in {"ports", "port"}:
        return _ports_topology_from_snapshot(snapshot, target)
    if k in {"operating_system", "platform"}:
        return _operating_system_safe_summary(snapshot)
    return None

def _select_from_storage(storage: Any, target: str = "") -> Any:
    if not isinstance(storage, list):
        return storage
    if not target:
        return storage
    t = _safe_str(target, 40).upper().rstrip("\\")
    for row in storage:
        if not isinstance(row, dict):
            continue
        mount = _safe_str(row.get("mountpoint") or row.get("device") or "", 80).upper().rstrip("\\")
        if mount.startswith(t):
            return row
    return None


def _select_snapshot_fact(snapshot: Dict[str, Any], kind: str, target: str = "") -> Any:
    if not isinstance(snapshot, dict):
        return None
    body = snapshot.get("body") if isinstance(snapshot.get("body"), dict) else {}
    chat_facts = snapshot.get("chat_facts") if isinstance(snapshot.get("chat_facts"), dict) else {}
    model_metrics = snapshot.get("model_metrics") if isinstance(snapshot.get("model_metrics"), dict) else {}

    topology_fact = _topology_fact_from_snapshot(snapshot, kind, target)
    if topology_fact not in (None, {}, []):
        return topology_fact

    if kind == "cpu":
        return body.get("cpu") or chat_facts.get("cpu") or model_metrics.get("cpu_count")
    if kind == "gpu":
        return body.get("gpu") or chat_facts.get("gpu") or model_metrics.get("gpu_name")
    if kind == "motherboard":
        mb = body.get("motherboard") or body.get("baseboard") or body.get("mainboard")
        if mb not in (None, "", [], {}):
            return mb
        items = body.get("motherboard_items")
        if isinstance(items, list) and items:
            return items[0]
        return chat_facts.get("motherboard") or model_metrics.get("motherboard")
    if kind == "memory":
        return body.get("ram") or chat_facts.get("ram") or {"ram_total_mb": model_metrics.get("ram_total_mb"), "ram_avail_mb": model_metrics.get("ram_avail_mb")}
    if kind in {"disk_space", "usb_label"}:
        return _select_from_storage(body.get("storage"), target)
    if kind in {"network", "network_card", "wifi_card", "bluetooth_card", "ethernet_card", "lan"}:
        adapters = body.get("network_adapters") or chat_facts.get("network_adapters")
        if kind == "wifi_card" and isinstance(adapters, list):
            return [a for a in adapters if "wi" in _safe_str(a.get("name") if isinstance(a, dict) else "", 160).lower() or "wireless" in _safe_str(a.get("name") if isinstance(a, dict) else "", 160).lower()]
        if kind == "bluetooth_card" and isinstance(adapters, list):
            return [a for a in adapters if "bluetooth" in _safe_str(a.get("name") if isinstance(a, dict) else "", 160).lower()]
        if kind in {"ethernet_card", "lan"} and isinstance(adapters, list):
            return [a for a in adapters if "ethernet" in _safe_str(a.get("name") if isinstance(a, dict) else "", 160).lower() or "lan" in _safe_str(a.get("name") if isinstance(a, dict) else "", 160).lower()]
        return adapters
    if kind == "temperature":
        gpu = body.get("gpu") if isinstance(body.get("gpu"), dict) else {}
        return {
            "gpu_temp_c": gpu.get("temperature_c") or model_metrics.get("gpu_temp_c"),
            "cpu_temp_c": model_metrics.get("cpu_temp_c"),
        }
    if kind == "fan_speed":
        return body.get("fans") or snapshot.get("fans")
    return None


def _canonical_fact_value(value: Any, kind: str = "") -> str:
    """Normalize values for quorum comparison without letting source noise break a true match."""
    if value is None:
        return ""
    k = _safe_str(kind, 80).lower()
    try:
        if isinstance(value, dict):
            if k in _TOPOLOGY_FACT_KINDS or value.get("topology_kind"):
                counts = value.get("counts") if isinstance(value.get("counts"), dict) else {}
                stable = {
                    "topology_kind": value.get("topology_kind"),
                    "device_class": value.get("device_class"),
                    "connection_bus": value.get("connection_bus"),
                    "physical_usb_port_count_verified": value.get("physical_usb_port_count_verified"),
                    "sata_port_count_verified": value.get("sata_port_count_verified"),
                    "nvme_device_count_verified": value.get("nvme_device_count_verified"),
                    "volume_count": counts.get("volume_count"),
                    "fixed_volume_count": counts.get("fixed_volume_count"),
                    "removable_or_external_candidate_count": counts.get("removable_or_external_candidate_count"),
                    "connected_usb_storage_candidate_count": value.get("connected_usb_storage_candidate_count"),
                    "local_fixed_storage_candidate_count": value.get("local_fixed_storage_candidate_count"),
                    "system": value.get("system"),
                    "release": value.get("release"),
                    "machine": value.get("machine"),
                }
                return _normalize_value({kk: vv for kk, vv in stable.items() if vv not in (None, "", [], {})}, k)
            if k == "cpu":
                for key in ("name", "Name", "processor", "Processor"):
                    if value.get(key):
                        return _normalize_value(value.get(key), k)
            if k == "gpu":
                for key in ("name", "Name", "gpu_name", "Info", "Type"):
                    if value.get(key):
                        return _normalize_value(value.get(key), k)
            if k == "motherboard":
                manufacturer = value.get("Manufacturer") or value.get("manufacturer")
                product = value.get("Product") or value.get("product") or value.get("Name") or value.get("name") or value.get("motherboard")
                if manufacturer and product and str(manufacturer).strip().lower() not in str(product).strip().lower():
                    return _normalize_value(f"{manufacturer} {product}", k)
                if product:
                    return _normalize_value(product, k)
            if k == "memory":
                for key in ("total_gb", "Total Memory (GB)", "ram_total_gb", "ram_total_mb"):
                    if value.get(key) not in (None, "", [], {}):
                        return _normalize_value(value.get(key), k)
            if k in {"disk_space", "usb_label"}:
                mount = value.get("mountpoint") or value.get("device") or value.get("DeviceID") or value.get("path")
                total = value.get("total_gb") or value.get("Size") or value.get("total")
                free = value.get("free_gb") or value.get("FreeSpace") or value.get("free")
                label = value.get("label") or value.get("VolumeName") or value.get("name")
                return _normalize_value({"mountpoint": mount, "label": label, "total": total, "free": free}, k)
            if k in {"network", "network_card", "wifi_card", "bluetooth_card", "ethernet_card", "lan"}:
                name = value.get("name") or value.get("adapter") or value.get("interface")
                if name:
                    return _normalize_value(name, k)
            if k in _SOFTWARE_KINDS or k == "software":
                app = value.get("canonical_name") or value.get("app_name") or value.get("name") or value.get("module")
                found = value.get("found")
                running = value.get("running") or value.get("is_running")
                return _normalize_value({"app": app, "found": found, "running": running}, k)
            if "value" in value:
                return _canonical_fact_value(value.get("value"), k)
        if isinstance(value, list):
            if k in {"network", "network_card", "wifi_card", "bluetooth_card", "ethernet_card", "lan"}:
                names = []
                for item in value:
                    if isinstance(item, dict):
                        nm = item.get("name") or item.get("adapter") or item.get("interface")
                        if nm:
                            names.append(_normalize_value(nm, k))
                if names:
                    return ",".join(sorted(set(names)))
            if k in {"disk_space", "usb_label"}:
                rows = []
                for item in value:
                    if isinstance(item, dict):
                        rows.append(_canonical_fact_value(item, k))
                if rows:
                    return "|".join(sorted(set(rows)))
        return _normalize_value(value, k)
    except Exception:
        return _normalize_value(value, k)


def _probe_result(
    source: str,
    value: Any,
    *,
    ok: bool = True,
    details: Optional[Dict[str, Any]] = None,
    error: str = "",
    kind: str = "",
    source_family: str = "",
    evidence_class: str = "witness",
    support_only: bool = False,
    hearsay: bool = False,
) -> Dict[str, Any]:
    verified = bool(ok and value not in (None, "", [], {}))
    normalized = _canonical_fact_value(value, kind) if verified else ""
    return {
        "source": _safe_str(source, 300),
        "source_family": _safe_str(source_family or source.split(".")[0], 180),
        "evidence_class": _safe_str(evidence_class, 120),
        "support_only": bool(support_only),
        "hearsay": bool(hearsay),
        "ok": bool(ok),
        "verified": verified,
        "value": _json_safe(value),
        "normalized_value": _safe_str(normalized, 20000),
        "details": _json_safe(details or {}),
        "error": _safe_str(error, 2000),
        "ts": _iso(),
    }


def _run_command_json(cmd: List[str], timeout: float = 5.0) -> Any:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if res.returncode == 0 and (res.stdout or "").strip():
            return json.loads(res.stdout)
    except Exception:
        pass
    return None


def _powershell_json(script: str, timeout: float = 8.0) -> Any:
    try:
        if platform.system().lower() != "windows":
            return None
        ps = shutil.which("powershell") or shutil.which("pwsh")
        if not ps:
            return None
        return _run_command_json([ps, "-NoProfile", "-Command", script], timeout=timeout)
    except Exception:
        return None


def _direct_psutil_fact(kind: str, target: str = "") -> Dict[str, Any]:
    """Cross-platform live adapter. Support/hearsay only for constitutional SelfAware quorum."""
    try:
        import psutil  # type: ignore
    except Exception as exc:
        return _probe_result("direct_psutil", None, ok=False, error=f"psutil_unavailable:{exc}", kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)

    try:
        if kind == "cpu":
            return _probe_result("direct_psutil.cpu", {
                "architecture": platform.machine(),
                "processor": platform.processor() or platform.uname().processor,
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_threads": psutil.cpu_count(logical=True),
                "usage_pct": round(float(psutil.cpu_percent(interval=0.1)), 2),
            }, kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)
        if kind == "memory":
            vm = psutil.virtual_memory()
            return _probe_result("direct_psutil.memory", {
                "total_gb": round(float(vm.total) / (1024 ** 3), 2),
                "available_gb": round(float(vm.available) / (1024 ** 3), 2),
                "used_gb": round(float(vm.used) / (1024 ** 3), 2),
                "usage_pct": round(float(vm.percent), 2),
            }, kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)
        if kind == "disk_space":
            mount = target or ("C:" if platform.system().lower() == "windows" else "/")
            if platform.system().lower() == "windows" and re.match(r"^[A-Za-z]:$", mount):
                mount = mount + "\\"
            usage = psutil.disk_usage(mount)
            return _probe_result("direct_psutil.disk_usage", {
                "mountpoint": mount,
                "total_gb": round(float(usage.total) / (1024 ** 3), 2),
                "used_gb": round(float(usage.used) / (1024 ** 3), 2),
                "free_gb": round(float(usage.free) / (1024 ** 3), 2),
                "usage_pct": round(float(usage.percent), 2),
            }, kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)
        if kind == "temperature":
            temps = {}
            try:
                temps = psutil.sensors_temperatures(fahrenheit=False) or {}
            except Exception:
                temps = {}
            return _probe_result("direct_psutil.sensors_temperatures", temps if temps else None, ok=bool(temps), error="temperature_sensors_unavailable" if not temps else "", kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)
        if kind == "fan_speed":
            fans = {}
            try:
                fans = psutil.sensors_fans() or {}
            except Exception:
                fans = {}
            return _probe_result("direct_psutil.sensors_fans", fans if fans else None, ok=bool(fans), error="fan_sensors_unavailable" if not fans else "", kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)
        if kind in {"network", "network_card", "wifi_card", "bluetooth_card", "ethernet_card", "lan"}:
            addrs = psutil.net_if_addrs()
            stats = psutil.net_if_stats()
            out: Dict[str, Any] = {}
            for name, rows in addrs.items():
                out[name] = {
                    "isup": bool(getattr(stats.get(name), "isup", False)) if name in stats else None,
                    "addresses": [str(getattr(r, "address", "")) for r in rows if getattr(r, "address", "")],
                }
            return _probe_result("direct_psutil.net_if_addrs", out, kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)
        if kind == "usb_label":
            parts = []
            for part in psutil.disk_partitions(all=False) or []:
                mount = str(getattr(part, "mountpoint", "") or "")
                if target and not mount.upper().startswith(target.upper()):
                    continue
                parts.append({
                    "device": str(getattr(part, "device", "") or ""),
                    "mountpoint": mount,
                    "fstype": str(getattr(part, "fstype", "") or ""),
                    "opts": str(getattr(part, "opts", "") or ""),
                })
            return _probe_result("direct_psutil.disk_partitions", parts if parts else None, ok=bool(parts), error="drive_not_found" if not parts else "", kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)
    except Exception as exc:
        return _probe_result("direct_psutil", None, ok=False, error=f"{type(exc).__name__}:{exc}", kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)

    return _probe_result("direct_psutil", None, ok=False, error="unsupported_fact_kind", kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)


def _hi_fact(kind: str, target: str = "") -> Dict[str, Any]:
    if _Hi is None:
        return _probe_result("SarahMemoryHi", None, ok=False, error="module_unavailable", kind=kind, source_family="SarahMemoryHi", evidence_class="witness")
    try:
        if kind in _HARDWARE_KINDS:
            snap = {}
            fn = getattr(_Hi, "get_boot_environment_snapshot", None)
            if callable(fn):
                snap = fn(force_refresh=False, refresh_reason="appself_fact_ticket", persist=True) or {}
            val = _select_snapshot_fact(snap if isinstance(snap, dict) else {}, kind, target)
            return _probe_result(f"SarahMemoryHi.boot_environment.{kind}", val, kind=kind, source_family="SarahMemoryHi", evidence_class="witness", details={"snapshot_available": bool(snap)})

        if kind == "temperature":
            fn = getattr(_Hi, "get_thermal_info", None)
            val = fn() if callable(fn) else None
            return _probe_result("SarahMemoryHi.get_thermal_info", val, ok=val not in (None, {}, []), error="thermal_unavailable" if not val else "", kind=kind, source_family="SarahMemoryHi", evidence_class="witness")
    except Exception as exc:
        return _probe_result("SarahMemoryHi", None, ok=False, error=f"{type(exc).__name__}:{exc}", kind=kind, source_family="SarahMemoryHi", evidence_class="witness")
    return _probe_result("SarahMemoryHi", None, ok=False, error="unsupported_fact_kind", kind=kind, source_family="SarahMemoryHi", evidence_class="witness")


def _runtime_snapshot_artifact_fact(kind: str, target: str = "") -> Dict[str, Any]:
    path = _runtime_environment_snapshot_path()
    snap = _read_json_file(path)

    # If the artifact is missing, ask SarahMemoryHi to materialize the persisted
    # boot/runtime document. This still counts as the artifact family, not as a
    # second SarahMemoryHi witness vote.
    if not snap and _Hi is not None:
        try:
            fn = getattr(_Hi, "get_boot_environment_snapshot", None)
            if callable(fn):
                fn(force_refresh=False, refresh_reason="appself_runtime_artifact_materialize", persist=True)
                snap = _read_json_file(path)
        except Exception:
            snap = {}

    val = _select_snapshot_fact(snap, kind, target) if snap else None
    return _probe_result(
        "runtime_environment_snapshot.json",
        val,
        ok=val not in (None, {}, []),
        error="runtime_environment_snapshot_missing_or_no_fact" if val in (None, {}, []) else "",
        kind=kind,
        source_family="runtime_environment_snapshot",
        evidence_class="physical_artifact",
        details={"path": str(path), "exists": path.exists(), "generated_by": "SarahMemoryHi", "materialized_if_missing": True},
    )


def _cognitive_self_fact(kind: str, target: str = "") -> Dict[str, Any]:
    if _CogSelf is None:
        return _probe_result("SarahMemoryCognitiveSelf", None, ok=False, error="module_unavailable", kind=kind, source_family="SarahMemoryCognitiveSelf", evidence_class="case_file")
    try:
        model = {}
        fn = getattr(_CogSelf, "get_latest_cognitive_self_model", None)
        if callable(fn):
            model = fn(refresh_if_stale=True, context={"source": "appself_fact_ticket"}, max_age_sec=60) or {}
        elif hasattr(_CogSelf, "build_cognitive_self_model"):
            model = _CogSelf.build_cognitive_self_model(context={"source": "appself_fact_ticket"}, force_refresh=False)  # type: ignore[attr-defined]

        if kind == "core_file":
            core = model.get("core_files") or model.get("core_file_inventory") or model.get("body_map", {}).get("files") or model.get("body_map", {}).get("discovery")
            if target and isinstance(core, (dict, list)):
                present = target in json.dumps(core)
                return _probe_result("SarahMemoryCognitiveSelf.core_files", {"file": target, "present_in_self_model": bool(present), "core": core}, ok=present, error="not_present_in_self_model" if not present else "", kind=kind, source_family="SarahMemoryCognitiveSelf", evidence_class="case_file")
            return _probe_result("SarahMemoryCognitiveSelf.core_files", core, ok=core not in (None, {}, []), error="core_file_model_unavailable" if not core else "", kind=kind, source_family="SarahMemoryCognitiveSelf", evidence_class="case_file")

        if kind in _HARDWARE_KINDS:
            body_map = model.get("body_map") if isinstance(model.get("body_map"), dict) else {}
            env = body_map.get("runtime_environment") if isinstance(body_map.get("runtime_environment"), dict) else {}
            val = _select_snapshot_fact(env, kind, target)

            # If the cached model is stale or thin, force one CognitiveSelf rebuild.
            # CognitiveSelf owns the case file; this does not create a duplicate
            # SarahMemoryHi vote because the source family remains CognitiveSelf.
            if val in (None, {}, []) and hasattr(_CogSelf, "build_cognitive_self_model"):
                try:
                    rebuilt = _CogSelf.build_cognitive_self_model(context={"source": "appself_fact_ticket_rebuild"}, force_refresh=True)  # type: ignore[attr-defined]
                    if isinstance(rebuilt, dict):
                        body_map = rebuilt.get("body_map") if isinstance(rebuilt.get("body_map"), dict) else {}
                        env = body_map.get("runtime_environment") if isinstance(body_map.get("runtime_environment"), dict) else {}
                        val = _select_snapshot_fact(env, kind, target)
                except Exception:
                    pass

            return _probe_result("SarahMemoryCognitiveSelf.body_map.runtime_environment", val, ok=val not in (None, {}, []), error="cognitive_self_body_fact_unavailable" if val in (None, {}, []) else "", kind=kind, source_family="SarahMemoryCognitiveSelf", evidence_class="case_file")

        if kind in _SOFTWARE_KINDS or kind == "software":
            capability = model.get("capability_map") if isinstance(model.get("capability_map"), dict) else {}
            desktop = capability.get("desktop_surface") if isinstance(capability.get("desktop_surface"), dict) else {}
            return _probe_result("SarahMemoryCognitiveSelf.capability_map.desktop_surface", desktop, ok=desktop not in (None, {}, []), error="desktop_surface_unavailable" if not desktop else "", kind="software", source_family="SarahMemoryCognitiveSelf", evidence_class="case_file")

        if kind in _BOOT_RUNTIME_KINDS:
            lifecycle = model.get("lifecycle") or model.get("status") or model.get("runtime")
            return _probe_result("SarahMemoryCognitiveSelf.lifecycle", lifecycle, ok=lifecycle not in (None, {}, []), error="lifecycle_unavailable" if not lifecycle else "", kind=kind, source_family="CognitiveSelf_lifecycle", evidence_class="case_file")
    except Exception as exc:
        return _probe_result("SarahMemoryCognitiveSelf", None, ok=False, error=f"{type(exc).__name__}:{exc}", kind=kind, source_family="SarahMemoryCognitiveSelf", evidence_class="case_file")
    return _probe_result("SarahMemoryCognitiveSelf", None, ok=False, error="unsupported_fact_kind", kind=kind, source_family="SarahMemoryCognitiveSelf", evidence_class="case_file")


def _diagnostics_fact(kind: str, target: str = "") -> Dict[str, Any]:
    if _Diagnostics is None:
        return _probe_result("SarahMemoryDiagnostics", None, ok=False, error="module_unavailable", kind=kind, source_family="SarahMemoryDiagnostics", evidence_class="support_witness", support_only=True)
    try:
        if kind in {"disk_space", "usb_label"}:
            fn = getattr(_Diagnostics, "get_storage_health_summary", None)
            val = fn() if callable(fn) else None
            return _probe_result("SarahMemoryDiagnostics.get_storage_health_summary", val, ok=val not in (None, {}, []), error="storage_health_unavailable" if not val else "", kind=kind, source_family="SarahMemoryDiagnostics", evidence_class="support_witness", support_only=True)
        if kind == "core_file":
            base = _base_dir()
            file_name = target or ""
            if file_name:
                path = base / file_name
                exists = path.exists()
                return _probe_result("SarahMemoryDiagnostics.core_file_exists", {"file": file_name, "path": str(path), "exists": bool(exists)}, ok=exists, error="core_file_missing" if not exists else "", kind=kind, source_family="SarahMemoryDiagnostics", evidence_class="support_witness", support_only=True)
            req = getattr(_Diagnostics, "REQUIRED_FILES", None)
            if isinstance(req, list):
                return _probe_result("SarahMemoryDiagnostics.REQUIRED_FILES", {"required_count": len(req), "required_files": [str(x) for x in req[:120]]}, kind=kind, source_family="SarahMemoryDiagnostics", evidence_class="support_witness", support_only=True)
        fn = getattr(_Diagnostics, "run_hardware_diagnostics", None)
        if callable(fn) and kind in _HARDWARE_KINDS:
            val = fn()
            return _probe_result("SarahMemoryDiagnostics.run_hardware_diagnostics", val, ok=val not in (None, {}, []), error="hardware_diagnostics_empty" if not val else "", kind=kind, source_family="SarahMemoryDiagnostics", evidence_class="support_witness", support_only=True)
    except Exception as exc:
        return _probe_result("SarahMemoryDiagnostics", None, ok=False, error=f"{type(exc).__name__}:{exc}", kind=kind, source_family="SarahMemoryDiagnostics", evidence_class="support_witness", support_only=True)
    return _probe_result("SarahMemoryDiagnostics", None, ok=False, error="unsupported_fact_kind", kind=kind, source_family="SarahMemoryDiagnostics", evidence_class="support_witness", support_only=True)


def _optimization_fact(kind: str, target: str = "") -> Dict[str, Any]:
    if _Optimization is None:
        return _probe_result("SarahMemoryOptimization", None, ok=False, error="module_unavailable", kind=kind, source_family="SarahMemoryOptimization", evidence_class="support_witness", support_only=True)
    try:
        for fn_name in ("monitor_system_resources", "get_optimization_metrics", "get_runtime_phase", "get_startup_profile"):
            fn = getattr(_Optimization, fn_name, None)
            if callable(fn):
                try:
                    val = fn()
                except TypeError:
                    val = fn(False)
                if val not in (None, {}, []):
                    return _probe_result(f"SarahMemoryOptimization.{fn_name}", val, kind=kind, source_family="SarahMemoryOptimization", evidence_class="support_witness", support_only=True)
    except Exception as exc:
        return _probe_result("SarahMemoryOptimization", None, ok=False, error=f"{type(exc).__name__}:{exc}", kind=kind, source_family="SarahMemoryOptimization", evidence_class="support_witness", support_only=True)
    return _probe_result("SarahMemoryOptimization", None, ok=False, error="no_runtime_metrics", kind=kind, source_family="SarahMemoryOptimization", evidence_class="support_witness", support_only=True)


def _si_fact(kind: str, target: str = "") -> Dict[str, Any]:
    if _Si is None:
        return _probe_result("SarahMemorySi", None, ok=False, error="module_unavailable", kind="software", source_family="SarahMemorySi", evidence_class="witness")
    try:
        app_name = target or ""
        if not app_name:
            return _probe_result("SarahMemorySi", {"available": True, "module": "SarahMemorySi"}, kind="software", source_family="SarahMemorySi", evidence_class="witness")
        fn = getattr(_Si, "get_application_info", None)
        if callable(fn):
            val = fn(app_name)
            return _probe_result("SarahMemorySi.get_application_info", val, ok=val not in (None, {}, []), error="application_info_empty" if not val else "", kind="software", source_family="SarahMemorySi", evidence_class="witness")
        fn = getattr(_Si, "smart_app_search", None)
        if callable(fn):
            val = fn(app_name)
            return _probe_result("SarahMemorySi.smart_app_search", val, ok=val not in (None, {}, []), error="application_search_empty" if not val else "", kind="software", source_family="SarahMemorySi", evidence_class="witness")
    except Exception as exc:
        return _probe_result("SarahMemorySi", None, ok=False, error=f"{type(exc).__name__}:{exc}", kind="software", source_family="SarahMemorySi", evidence_class="witness")
    return _probe_result("SarahMemorySi", None, ok=False, error="unsupported_software_fact", kind="software", source_family="SarahMemorySi", evidence_class="witness")


def _software_artifact_fact(kind: str, target: str = "") -> Dict[str, Any]:
    path = _software_db_path()
    if not path.exists():
        return _probe_result("software.db", None, ok=False, error="software_db_missing", kind="software", source_family="software_artifact", evidence_class="physical_artifact", details={"path": str(path), "exists": False})
    # Read-only artifact probe. Do not create tables; do not count rows as multiple witnesses.
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=3.0)
        cur = con.cursor()
        tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        val: Dict[str, Any] = {"path": str(path), "tables": tables[:50], "target": target}
        if target:
            hits: List[Dict[str, Any]] = []
            for table in tables[:20]:
                try:
                    cols = [r[1] for r in cur.execute(f"PRAGMA table_info(`{table}`)").fetchall()]
                    text_cols = [c for c in cols if c.lower() in {"name", "app_name", "canonical_name", "path", "exe", "command"}]
                    for col in text_cols[:3]:
                        rows = cur.execute(f"SELECT * FROM `{table}` WHERE lower(`{col}`) LIKE ? LIMIT 3", (f"%{target.lower()}%",)).fetchall()
                        if rows:
                            hits.append({"table": table, "column": col, "rows_found": len(rows)})
                except Exception:
                    continue
            val["hits"] = hits
            val["found"] = bool(hits)
        con.close()
        return _probe_result("software.db.read_only", val, ok=bool(val), kind="software", source_family="software_artifact", evidence_class="physical_artifact")
    except Exception as exc:
        return _probe_result("software.db.read_only", None, ok=False, error=f"{type(exc).__name__}:{exc}", kind="software", source_family="software_artifact", evidence_class="physical_artifact", details={"path": str(path), "exists": True})


def _initialization_fact(kind: str, target: str = "") -> Dict[str, Any]:
    if _Initialization is None:
        return _probe_result("SarahMemoryInitialization", None, ok=False, error="module_unavailable", kind=kind, source_family="SarahMemoryInitialization", evidence_class="witness")
    value = {
        "module_available": True,
        "has_run_initial_checks": callable(getattr(_Initialization, "run_initial_checks", None)),
        "background_dataset_embedding_alive": callable(getattr(_Initialization, "_background_thread_alive", None)),
    }
    return _probe_result("SarahMemoryInitialization.module_contract", value, kind=kind, source_family="SarahMemoryInitialization", evidence_class="witness")


def _integration_fact(kind: str, target: str = "") -> Dict[str, Any]:
    if _Integration is None:
        return _probe_result("SarahMemoryIntegration", None, ok=False, error="module_unavailable", kind=kind, source_family="SarahMemoryIntegration", evidence_class="witness")
    value = {
        "module_available": True,
        "has_execute_action_ticket": callable(getattr(_Integration, "execute_action_ticket", None)),
        "shutdown_event_present": hasattr(_Integration, "terminate_flag"),
    }
    return _probe_result("SarahMemoryIntegration.module_contract", value, kind=kind, source_family="SarahMemoryIntegration", evidence_class="witness")


def _server_state_artifact_fact(kind: str, target: str = "") -> Dict[str, Any]:
    path = _server_state_path()
    value = _read_json_file(path)
    return _probe_result("server_state.json", value if value else None, ok=bool(value), error="server_state_missing_or_empty" if not value else "", kind=kind, source_family="server_state_artifact", evidence_class="physical_artifact", support_only=True, details={"path": str(path), "exists": path.exists()})


def _network_fact(kind: str, target: str = "") -> Dict[str, Any]:
    if _Network is None:
        return _probe_result("SarahMemoryNetwork", None, ok=False, error="module_unavailable", kind=kind, source_family="SarahMemoryNetwork", evidence_class="witness")
    value = {
        "module_available": True,
        "has_protocol": hasattr(_Network, "NetworkProtocol"),
        "has_node": hasattr(_Network, "NetworkNode"),
        "has_ids": hasattr(_Network, "NetworkIDS"),
    }
    return _probe_result("SarahMemoryNetwork.module_contract", value, kind=kind, source_family="SarahMemoryNetwork", evidence_class="witness")


def _sync_fact(kind: str, target: str = "") -> Dict[str, Any]:
    if _Sync is None:
        return _probe_result("SarahMemorySync", None, ok=False, error="module_unavailable", kind=kind, source_family="SarahMemorySync", evidence_class="witness")
    value = {
        "module_available": True,
        "phase_c_available": bool(getattr(_Sync, "PHASE_C_ENABLED", False)),
        "has_sync_data": callable(getattr(_Sync, "sync_data", None)),
        "has_log_sync_event": callable(getattr(_Sync, "log_sync_event", None)),
    }
    return _probe_result("SarahMemorySync.module_contract", value, kind=kind, source_family="SarahMemorySync", evidence_class="witness")


def _network_artifact_fact(kind: str, target: str = "") -> Dict[str, Any]:
    path = _system_logs_db_path()
    value = {"path": str(path), "exists": path.exists(), "artifact_family": "system_logs_network"}
    return _probe_result("network_artifact.system_logs", value if path.exists() else None, ok=path.exists(), error="network_artifact_missing" if not path.exists() else "", kind=kind, source_family="network_artifact", evidence_class="physical_artifact")


def _selfaware_fact(kind: str, target: str = "") -> Dict[str, Any]:
    if _SelfAware is None:
        return _probe_result("SarahMemorySelfAware", None, ok=False, error="module_unavailable", kind=kind, source_family="SarahMemorySelfAware", evidence_class="witness")
    value = {
        "module_available": True,
        "armed_function_present": callable(getattr(_SelfAware, "_armed", None)),
        "scan_codebase_present": callable(getattr(_SelfAware, "scan_codebase", None)),
        "log_event_present": callable(getattr(_SelfAware, "log_event", None)),
    }
    return _probe_result("SarahMemorySelfAware.module_contract", value, kind=kind, source_family="SarahMemorySelfAware", evidence_class="witness")


def _synapes_fact(kind: str, target: str = "") -> Dict[str, Any]:
    if _Synapes is None:
        return _probe_result("SarahMemorySynapes", None, ok=False, error="module_unavailable", kind=kind, source_family="SarahMemorySynapes", evidence_class="lab_witness")
    value = {
        "module_available": True,
        "sandbox_dir": str(getattr(_Synapes, "SANDBOX_DIR", "")),
        "testing_dir": str(getattr(_Synapes, "SANDBOX_TESTING_DIR", "")),
        "has_ensure_model_dirs": callable(getattr(_Synapes, "ensure_sarahmemory_model_dirs", None)),
        "has_register_model": callable(getattr(_Synapes, "register_model", None)),
    }
    return _probe_result("SarahMemorySynapes.module_contract", value, kind=kind, source_family="SarahMemorySynapes", evidence_class="lab_witness")


def _evolution_fact(kind: str, target: str = "") -> Dict[str, Any]:
    if _Evolution is None:
        return _probe_result("SarahMemoryEvolution", None, ok=False, error="module_unavailable", kind=kind, source_family="SarahMemoryEvolution", evidence_class="restricted_witness")
    value = {
        "module_available": True,
        "mods_dir": str(getattr(_Evolution, "MODS_DIR", "")),
        "reports_dir": str(getattr(_Evolution, "REPORTS_DIR", "")),
        "monkey_patch_only": True,
    }
    return _probe_result("SarahMemoryEvolution.module_contract", value, kind=kind, source_family="SarahMemoryEvolution", evidence_class="restricted_witness")


def _vision_fact(kind: str, target: str = "") -> Dict[str, Any]:
    if _FacialRecognition is None:
        return _probe_result("SarahMemoryFacialRecognition", None, ok=False, error="module_unavailable", kind=kind, source_family="SarahMemoryFacialRecognition", evidence_class="witness")
    value = {
        "module_available": True,
        "faiss_available": bool(getattr(_FacialRecognition, "faiss_available", False)),
        "has_find_similar_vectors": callable(getattr(_FacialRecognition, "find_similar_vectors", None)),
    }
    return _probe_result("SarahMemoryFacialRecognition.module_contract", value, kind=kind, source_family="SarahMemoryFacialRecognition", evidence_class="witness")


def _sobje_fact(kind: str, target: str = "") -> Dict[str, Any]:
    if _SOBJE is None:
        return _probe_result("SarahMemorySOBJE", None, ok=False, error="module_unavailable", kind=kind, source_family="SarahMemorySOBJE", evidence_class="witness")
    value = {
        "module_available": True,
        "object_detection_enabled": bool(getattr(_SOBJE, "OBJECT_DETECTION_ENABLED", False)),
        "has_visual_intent_parser": callable(getattr(_SOBJE, "sm_parse_visual_intent", None)),
    }
    return _probe_result("SarahMemorySOBJE.module_contract", value, kind=kind, source_family="SarahMemorySOBJE", evidence_class="witness")


def _vision_artifact_fact(kind: str, target: str = "") -> Dict[str, Any]:
    path = _system_logs_db_path()
    value = {"path": str(path), "exists": path.exists(), "artifact_family": "vision_logs"}
    return _probe_result("vision_artifact.system_logs", value if path.exists() else None, ok=path.exists(), error="vision_artifact_missing" if not path.exists() else "", kind=kind, source_family="vision_artifact", evidence_class="physical_artifact")


def _adaptive_fact(kind: str, target: str = "") -> Dict[str, Any]:
    if _Adaptive is None:
        return _probe_result("SarahMemoryAdaptive", None, ok=False, error="module_unavailable", kind=kind, source_family="SarahMemoryAdaptive", evidence_class="witness")
    value = {
        "module_available": True,
        "has_emotional_state": hasattr(_Adaptive, "STATE"),
        "has_update_personality": callable(getattr(_Adaptive, "update_personality", None)),
    }
    return _probe_result("SarahMemoryAdaptive.module_contract", value, kind=kind, source_family="SarahMemoryAdaptive", evidence_class="witness")


def _adaptive_artifact_fact(kind: str, target: str = "") -> Dict[str, Any]:
    path = _datasets_dir() / "personality1.db"
    value = {"path": str(path), "exists": path.exists(), "artifact_family": "personality_state"}
    return _probe_result("adaptive_artifact.personality1.db", value if path.exists() else None, ok=path.exists(), error="adaptive_artifact_missing" if not path.exists() else "", kind=kind, source_family="adaptive_artifact", evidence_class="physical_artifact")


def _ai_functions_fact(kind: str, target: str = "") -> Dict[str, Any]:
    if _AiFunctions is None:
        return _probe_result("SarahMemoryAiFunctions", None, ok=False, error="module_unavailable", kind=kind, source_family="SarahMemoryAiFunctions", evidence_class="witness")
    value = {
        "module_available": True,
        "context_buffer_present": hasattr(_AiFunctions, "context_buffer"),
        "advanced_agent_state_present": hasattr(_AiFunctions, "AdvancedAgentState"),
    }
    return _probe_result("SarahMemoryAiFunctions.module_contract", value, kind=kind, source_family="SarahMemoryAiFunctions", evidence_class="witness")


def _windows_fact(kind: str, target: str = "") -> Dict[str, Any]:
    """Windows adapter retained as hearsay/support only. Never counts as quorum."""
    if platform.system().lower() != "windows":
        return _probe_result("windows_probe", None, ok=False, error="not_windows", kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)

    try:
        if kind == "cpu":
            data = _powershell_json(
                "Get-CimInstance Win32_Processor | Select-Object Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | ConvertTo-Json -Depth 4",
                timeout=8,
            )
            return _probe_result("windows.Win32_Processor", data, ok=data not in (None, {}, []), error="wmi_cpu_unavailable" if not data else "", kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)
        if kind == "gpu":
            data = _powershell_json(
                "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,AdapterRAM,Status | ConvertTo-Json -Depth 4",
                timeout=8,
            )
            return _probe_result("windows.Win32_VideoController", data, ok=data not in (None, {}, []), error="wmi_gpu_unavailable" if not data else "", kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)
        if kind == "motherboard":
            data = _powershell_json(
                "Get-CimInstance Win32_BaseBoard | Select-Object Manufacturer,Product,SerialNumber,Version | ConvertTo-Json -Depth 4",
                timeout=8,
            )
            return _probe_result("windows.Win32_BaseBoard", data, ok=data not in (None, {}, []), error="wmi_baseboard_unavailable" if not data else "", kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)
        if kind in {"usb_label", "disk_space"}:
            drive_filter = ""
            if target and re.match(r"^[A-Z]:$", target.upper()):
                drive_filter = f" | Where-Object {{$_.DeviceID -eq '{target.upper()}'}}"
            data = _powershell_json(
                "Get-CimInstance Win32_LogicalDisk" + drive_filter + " | Select-Object DeviceID,VolumeName,DriveType,FileSystem,Size,FreeSpace | ConvertTo-Json -Depth 4",
                timeout=8,
            )
            return _probe_result("windows.Win32_LogicalDisk", data, ok=data not in (None, {}, []), error="logical_disk_unavailable" if not data else "", kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)
        if kind in {"temperature", "fan_speed"}:
            data = _powershell_json(
                "Get-CimInstance -Namespace root/wmi MSAcpi_ThermalZoneTemperature | Select-Object CurrentTemperature,InstanceName | ConvertTo-Json -Depth 4",
                timeout=8,
            )
            return _probe_result("windows.MSAcpi_ThermalZoneTemperature", data, ok=data not in (None, {}, []), error="thermal_zone_unavailable" if not data else "", kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)
    except Exception as exc:
        return _probe_result("windows_probe", None, ok=False, error=f"{type(exc).__name__}:{exc}", kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)

    return _probe_result("windows_probe", None, ok=False, error="unsupported_fact_kind", kind=kind, source_family="os_adapter", evidence_class="adapter_hearsay", support_only=True, hearsay=True)


def _collect_fact_probes(kind: str, claim: str, target: str = "") -> List[Dict[str, Any]]:
    """Collect jurisdiction-aware evidence without double jeopardy.

    This replaces the old fixed Hi + OS/WMI + CognitiveSelf model.
    OS calls are still allowed as support/hearsay, but are never one of the
    three constitutional votes.
    """
    jurisdiction = _fact_jurisdiction(kind, claim)
    probes: List[Dict[str, Any]] = []

    if jurisdiction in {"hardware_body", "general_system"}:
        probes.extend([
            _hi_fact(kind, target),
            _runtime_snapshot_artifact_fact(kind, target),
            _cognitive_self_fact(kind, target),
        ])
        probes.extend([
            _diagnostics_fact(kind, target),
            _optimization_fact(kind, target),
            _driver_capability_support_fact(kind, target),
            _appdrivers_contract_fact(kind, target),
        ])
        if kind in {"network", "network_card", "wifi_card", "bluetooth_card", "ethernet_card", "lan"}:
            probes.append(_network_operation_support_fact(kind, target))
        probes.append(_direct_psutil_fact(kind, target))
        if platform.system().lower() == "windows":
            probes.append(_windows_fact(kind, target))
        return probes


    if jurisdiction == "driver_capability":
        probes.extend([
            _driver_capability_support_fact(kind, target),
            _appdrivers_contract_fact(kind, target),
        ])
        # Driver-capability facts are support/capability records, not physical-hardware
        # proof. They should not become a fake 3/3 hardware quorum.
        return probes

    if jurisdiction == "software_desktop":
        app_target = target or _extract_target_from_claim(claim, "software")
        probes.extend([
            _si_fact("software", app_target),
            _software_artifact_fact("software", app_target),
            _cognitive_self_fact("software", app_target),
        ])
        probes.extend([_diagnostics_fact("core_file", app_target), _optimization_fact("software", app_target)])
        return probes

    if jurisdiction == "boot_runtime":
        probes.extend([
            _initialization_fact(kind, target),
            _integration_fact(kind, target),
            _cognitive_self_fact(kind, target),
        ])
        probes.append(_server_state_artifact_fact(kind, target))
        return probes

    if jurisdiction == "network_sarahnet":
        probes.extend([
            _network_fact(kind, target),
            _sync_fact(kind, target),
            _network_artifact_fact(kind, target),
        ])
        probes.extend([_hi_fact("network", target), _diagnostics_fact("network", target)])
        return probes

    if jurisdiction == "rem_code_patch":
        probes.extend([
            _selfaware_fact(kind, target),
            _synapes_fact(kind, target),
            _evolution_fact(kind, target),
        ])
        probes.extend([
            _probe_result("SarahMemoryAssuranceGate", {"available": _AssuranceGate is not None}, ok=_AssuranceGate is not None, kind=kind, source_family="SarahMemoryAssuranceGate", evidence_class="admissibility_gate", support_only=True),
            _probe_result("SarahMemorySecurityGovernor", {"available": _SecurityGovernor is not None}, ok=_SecurityGovernor is not None, kind=kind, source_family="SarahMemorySecurityGovernor", evidence_class="admissibility_gate", support_only=True),
        ])
        return probes

    if jurisdiction == "vision_environment":
        probes.extend([
            _vision_fact(kind, target),
            _sobje_fact(kind, target),
            _vision_artifact_fact(kind, target),
        ])
        probes.append(_cognitive_self_fact(kind, target))
        return probes

    if jurisdiction == "adaptive_behavior":
        probes.extend([
            _adaptive_fact(kind, target),
            _adaptive_artifact_fact(kind, target),
            _ai_functions_fact(kind, target),
        ])
        return probes

    probes.extend([
        _hi_fact(kind, target),
        _runtime_snapshot_artifact_fact(kind, target),
        _cognitive_self_fact(kind, target),
        _diagnostics_fact(kind, target),
    ])
    return probes


def _extract_majority_value(probes: List[Dict[str, Any]], kind: str = "") -> Tuple[int, str, Any, Dict[str, Any]]:
    seen_families: set = set()
    quorum_candidates: List[Dict[str, Any]] = []
    duplicate_rejections: List[Dict[str, Any]] = []

    for p in probes:
        if bool(p.get("support_only")) or bool(p.get("hearsay")):
            continue
        fam = _safe_str(p.get("source_family") or p.get("source") or "unknown", 160)
        if fam in seen_families:
            dup = dict(p)
            dup["ignored_reason"] = "duplicate_source_family_no_double_jeopardy"
            duplicate_rejections.append(dup)
            continue
        seen_families.add(fam)
        quorum_candidates.append(p)
        if len(quorum_candidates) >= 3:
            break

    normalized: Dict[str, List[Dict[str, Any]]] = {}
    for p in quorum_candidates:
        if not p.get("verified"):
            continue
        n = _safe_str(p.get("normalized_value") or _canonical_fact_value(p.get("value"), kind), 20000)
        if not n:
            continue
        normalized.setdefault(n, []).append(p)

    if not normalized:
        return 0, "", None, {
            "quorum_candidates": quorum_candidates,
            "normalized_groups": {},
            "duplicate_rejections": duplicate_rejections,
            "support_sources": [p for p in probes if bool(p.get("support_only"))],
            "rule": "one_source_family_one_vote_os_adapters_support_only",
        }

    best_key = ""
    best_items: List[Dict[str, Any]] = []
    for key, items in normalized.items():
        if len(items) > len(best_items):
            best_key = key
            best_items = items

    pass_count = len(best_items)
    value = best_items[0].get("value") if best_items else None
    return pass_count, best_key, value, {
        "quorum_candidates": quorum_candidates,
        "normalized_groups": {k: len(v) for k, v in normalized.items()},
        "winning_source_families": [p.get("source_family") for p in best_items],
        "duplicate_rejections": duplicate_rejections,
        "support_sources": [p for p in probes if bool(p.get("support_only"))],
        "rule": "0/3 denied, 1/3 denied, 2/3 high review, 3/3 approved fact; no duplicate source-family votes",
    }


def _decision_from_pass_count(pass_count: int) -> Tuple[str, str, str]:
    if pass_count >= 3:
        return "APPROVED_FACT", "HIGH", "low"
    if pass_count == 2:
        return "ESCALATE_HIGH_REVIEW", "MEDIUM_HIGH", "high"
    if pass_count == 1:
        return "DENIED_WEAK_EVIDENCE", "LOW", "medium"
    return "DENIED_NO_EVIDENCE", "NONE", "medium"


def _store_ticket(ticket: Dict[str, Any]) -> None:
    con = None
    try:
        _ensure_tables()
        con = _db()
        cur = con.cursor()
        evidence_json = _json_dumps_limited(ticket.get("evidence") or {})
        review_json = _json_dumps_limited(ticket)
        cur.execute(
            f"""
            INSERT OR REPLACE INTO {TICKET_TABLE}
            (ticket_id, ts, ts_iso, source, ticket_kind, claim, normalized_claim, requested_fact,
             quorum, pass_count, decision, confidence, risk_tier, rem_ticket_id, rem_cycle_id,
             evidence_json, review_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket.get("ticket_id"),
                float(ticket.get("ts") or _now()),
                ticket.get("ts_iso"),
                ticket.get("source"),
                ticket.get("ticket_kind"),
                ticket.get("claim"),
                ticket.get("normalized_claim"),
                ticket.get("requested_fact"),
                ticket.get("quorum"),
                int(ticket.get("pass_count") or 0),
                ticket.get("decision"),
                ticket.get("confidence"),
                ticket.get("risk_tier"),
                ticket.get("rem_ticket_id"),
                ticket.get("rem_cycle_id"),
                evidence_json,
                review_json,
            ),
        )
        con.commit()
    except Exception as exc:
        logger.warning("appself ticket store failed: %s", exc)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _read_ticket(ticket_id: str) -> Optional[Dict[str, Any]]:
    con = None
    try:
        _ensure_tables()
        con = _db()
        cur = con.cursor()
        cur.execute(f"SELECT review_json FROM {TICKET_TABLE} WHERE ticket_id = ?", (ticket_id,))
        row = cur.fetchone()
        if not row:
            return None
        raw = row[0] if isinstance(row, (list, tuple)) else row["review_json"]
        obj = json.loads(raw or "{}")
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass


def _list_tickets(limit: int = 50, decision: str = "", ticket_kind: str = "") -> List[Dict[str, Any]]:
    con = None
    out: List[Dict[str, Any]] = []
    try:
        _ensure_tables()
        con = _db()
        cur = con.cursor()
        limit = max(1, min(MAX_TICKETS_RETURN, int(limit or 50)))
        where: List[str] = []
        vals: List[Any] = []
        if decision:
            where.append("decision = ?")
            vals.append(decision)
        if ticket_kind:
            where.append("ticket_kind = ?")
            vals.append(ticket_kind)
        sql = f"SELECT ticket_id, ts_iso, source, ticket_kind, claim, requested_fact, quorum, pass_count, decision, confidence, risk_tier, rem_ticket_id, rem_cycle_id FROM {TICKET_TABLE}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts DESC LIMIT ?"
        vals.append(limit)
        cur.execute(sql, tuple(vals))
        cols = [d[0] for d in cur.description]
        for row in cur.fetchall() or []:
            if isinstance(row, sqlite3.Row):
                out.append({k: row[k] for k in cols})
            else:
                out.append({cols[i]: row[i] for i in range(len(cols))})
    except Exception as exc:
        logger.warning("appself ticket list failed: %s", exc)
    finally:
        try:
            if con:
                con.close()
        except Exception:
            pass
    return out




# Network adapter presentation is intentionally conservative. The evidence court
# may know IP/MAC data internally, but chat/UI presentation should answer the
# hardware/body question without exposing sensitive addressing data by default.
def _network_adapter_category(name: Any) -> str:
    n = _safe_str(name, 160).lower().replace("_", " ").replace("-", " ")
    if not n:
        return "unknown"
    if "loopback" in n or n in {"lo", "localhost"}:
        return "loopback"
    if "radmin" in n or "vpn" in n or "tap" in n or "tun" in n or "wireguard" in n or "openvpn" in n:
        return "virtual_vpn"
    if "bluetooth" in n or "pan" in n:
        return "bluetooth_network"
    if "wi fi" in n or "wifi" in n or "wireless" in n or "wlan" in n or "802.11" in n:
        return "wifi"
    if "ethernet" in n or "lan" in n or "gbe" in n or "2.5g" in n or "10g" in n:
        return "ethernet"
    if "local area connection" in n:
        return "virtual_or_auxiliary"
    return "network_adapter"


def _network_adapter_status(row: Dict[str, Any]) -> str:
    up = row.get("is_up")
    if up is None:
        up = row.get("isup")
    if up is True:
        return "up"
    if up is False:
        return "down"
    return "unknown"


def _coerce_network_adapter_rows(value: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        if isinstance(value, dict):
            # psutil-style support shape: {adapter_name: {isup, addresses}}
            for name, data in value.items():
                if isinstance(data, dict):
                    rows.append({
                        "name": name,
                        "is_up": data.get("is_up") if "is_up" in data else data.get("isup"),
                        "speed_mbps": data.get("speed_mbps") or data.get("speed"),
                        "mtu": data.get("mtu"),
                        "ip_addresses": data.get("ip_addresses") or data.get("addresses") or [],
                        "mac_addresses": data.get("mac_addresses") or [],
                    })
            if rows:
                return rows
        if isinstance(value, (list, tuple, set)):
            for item in list(value):
                if isinstance(item, dict):
                    name = item.get("name") or item.get("adapter") or item.get("interface") or item.get("device")
                    if not name:
                        continue
                    rows.append({
                        "name": _safe_str(name, 160),
                        "is_up": item.get("is_up") if "is_up" in item else item.get("isup"),
                        "speed_mbps": item.get("speed_mbps") or item.get("speed") or item.get("link_speed_mbps"),
                        "mtu": item.get("mtu"),
                        "ip_addresses": item.get("ip_addresses") or item.get("addresses") or [],
                        "mac_addresses": item.get("mac_addresses") or [],
                    })
    except Exception:
        return []
    return rows


def _network_adapter_safe_summary(value: Any) -> Dict[str, Any]:
    rows = _coerce_network_adapter_rows(value)
    adapters: List[Dict[str, Any]] = []
    for row in rows:
        name = _safe_str(row.get("name"), 160)
        if not name:
            continue
        category = _network_adapter_category(name)
        # Loopback is useful evidence internally but is not a network card answer.
        if category == "loopback":
            continue
        status = _network_adapter_status(row)
        speed = row.get("speed_mbps")
        mtu = row.get("mtu")
        safe_row: Dict[str, Any] = {
            "name": name,
            "category": category,
            "status": status,
        }
        if speed not in (None, "", [], {}):
            safe_row["speed_mbps"] = speed
        if mtu not in (None, "", [], {}):
            safe_row["mtu"] = mtu
        # Do not expose IP/MAC in presentation. Keep only presence flags/counts.
        ip_list = row.get("ip_addresses") or []
        mac_list = row.get("mac_addresses") or []
        if isinstance(ip_list, (list, tuple, set)):
            safe_row["ip_address_count"] = len([x for x in ip_list if _safe_str(x, 120)])
        if isinstance(mac_list, (list, tuple, set)):
            safe_row["mac_address_count"] = len([x for x in mac_list if _safe_str(x, 120)])
        adapters.append(safe_row)

    # Deterministic order: active physical first, then active virtual, then inactive.
    priority = {
        "ethernet": 0,
        "wifi": 1,
        "bluetooth_network": 2,
        "network_adapter": 3,
        "virtual_or_auxiliary": 4,
        "virtual_vpn": 5,
        "unknown": 6,
    }
    adapters.sort(key=lambda r: (0 if r.get("status") == "up" else 1, priority.get(str(r.get("category")), 9), str(r.get("name", "")).lower()))

    active = [a for a in adapters if a.get("status") == "up"]
    inactive = [a for a in adapters if a.get("status") != "up"]
    physical = [a for a in adapters if a.get("category") in {"ethernet", "wifi", "bluetooth_network", "network_adapter"}]
    virtual = [a for a in adapters if a.get("category") in {"virtual_vpn", "virtual_or_auxiliary"}]
    primary = None
    for cat in ("ethernet", "wifi", "network_adapter", "bluetooth_network", "virtual_vpn"):
        for row in active:
            if row.get("category") == cat:
                primary = row
                break
        if primary:
            break

    return {
        "adapters": adapters,
        "active_adapters": active,
        "inactive_adapters": inactive,
        "physical_or_local_interfaces": physical,
        "virtual_or_auxiliary_interfaces": virtual,
        "primary_adapter": primary,
        "sensitive_fields_redacted": ["ip_addresses", "mac_addresses"],
        "network_action_taken": False,
        "security_note": "Fact-check only; no sockets opened, ports exposed, pairing attempted, or network changes executed.",
    }


def _format_network_adapter_item(row: Dict[str, Any]) -> str:
    name = _safe_str(row.get("name"), 160)
    category = _safe_str(row.get("category"), 80).replace("_", " ")
    status = _safe_str(row.get("status"), 40)
    speed = row.get("speed_mbps")
    detail = f"{name} ({category}, {status}"
    if speed not in (None, "", [], {}):
        detail += f", {speed} Mbps"
    detail += ")"
    return detail


def _network_operation_support_fact(kind: str, target: str = "") -> Dict[str, Any]:
    """Support-only witness: SarahMemoryNetwork capability contract.

    This must never open sockets, bind ports, scan Wi-Fi/Bluetooth, pair devices,
    or start SarahNet. It only proves the network operations module exists and
    what guarded concepts are available for future action tickets.
    """
    if _Network is None:
        return _probe_result(
            "SarahMemoryNetwork.module_contract",
            None,
            ok=False,
            error="module_unavailable",
            kind=kind,
            source_family="SarahMemoryNetwork",
            evidence_class="support_witness",
            support_only=True,
        )
    value = {
        "module_available": True,
        "protocol_framing_present": hasattr(_Network, "NetworkProtocol"),
        "network_node_present": hasattr(_Network, "NetworkNode"),
        "ids_rate_limiter_present": hasattr(_Network, "NetworkIDS"),
        "server_connection_present": hasattr(_Network, "ServerConnection"),
        "mobile_io_present": hasattr(_Network, "MobileIO"),
        "safety_policy_present": hasattr(_Network, "SafetyPolicy"),
        "wifi_scan_declared": hasattr(getattr(_Network, "ServerConnection", object), "wifi_scan"),
        "bluetooth_scan_declared": hasattr(getattr(_Network, "ServerConnection", object), "list_bluetooth_devices"),
        "usb_tether_status_declared": hasattr(getattr(_Network, "ServerConnection", object), "usb_tethering_status"),
        "fact_check_only": True,
        "network_action_taken": False,
        "security_boundary": "support_only_no_socket_bind_no_scan_no_connect_no_pairing",
    }
    return _probe_result(
        "SarahMemoryNetwork.module_contract",
        value,
        kind=kind,
        source_family="SarahMemoryNetwork",
        evidence_class="support_witness",
        support_only=True,
    )

def _presentation_value(kind: str, value: Any) -> Any:
    """Return a compact fact value for UI/chat presentation.

    The full court record stays in evidence.probes. This field is intentionally
    small so Chat/Reply does not dump a whole boot snapshot or sensitive network
    addressing data when the approved fact is a body-map attribute.
    """
    k = _safe_str(kind, 80).lower()
    try:
        if value in (None, "", [], {}):
            return value
        if k in {"network", "network_card", "wifi_card", "bluetooth_card", "ethernet_card", "lan"}:
            safe_network = _network_adapter_safe_summary(value)
            safe_network["driver_capability"] = _build_driver_capability_snapshot("network_card")
            return safe_network
        if k in {"drivers", "driver", "driver_stack", "body_capability", "body_capabilities", "hardware_capability"}:
            return _build_driver_capability_snapshot("drivers")
        if k in _TOPOLOGY_FACT_KINDS and isinstance(value, dict):
            return value
        if k == "motherboard":
            if isinstance(value, str):
                return _safe_str(value, 300)
            if isinstance(value, dict):
                manufacturer = value.get("Manufacturer") or value.get("manufacturer")
                product = value.get("Product") or value.get("product") or value.get("Name") or value.get("name") or value.get("motherboard")
                if manufacturer and product and str(manufacturer).strip().lower() not in str(product).strip().lower():
                    return _safe_str(f"{manufacturer} {product}", 300)
                if product:
                    return _safe_str(product, 300)
        if k == "cpu" and isinstance(value, dict):
            name = value.get("name") or value.get("Name") or value.get("processor") or value.get("Processor")
            if name:
                return {
                    "name": name,
                    "physical_cores": value.get("physical_cores") or value.get("NumberOfCores"),
                    "logical_threads": value.get("logical_threads") or value.get("NumberOfLogicalProcessors"),
                    "current_clock_mhz": value.get("current_clock_mhz") or value.get("MaxClockSpeed"),
                    "max_clock_mhz": value.get("max_clock_mhz") or value.get("MaxClockSpeed"),
                }
        if k == "gpu" and isinstance(value, dict):
            name = value.get("name") or value.get("Name") or value.get("gpu_name")
            if name:
                return {
                    "name": name,
                    "temperature_c": value.get("temperature_c"),
                    "utilization_pct": value.get("utilization_pct"),
                    "vram_total_mb": value.get("vram_total_mb"),
                    "memory": value.get("memory"),
                    "driver": value.get("driver"),
                }
        return value
    except Exception:
        return value


def _presentation_text(kind: str, value: Any) -> str:
    """Short single-line fact for clients that prefer backend-ready wording."""
    pv = _presentation_value(kind, value)
    k = _safe_str(kind, 80).lower()
    try:
        if k in {"network", "network_card", "wifi_card", "bluetooth_card", "ethernet_card", "lan"} and isinstance(pv, dict):
            active = pv.get("active_adapters") if isinstance(pv.get("active_adapters"), list) else []
            inactive = pv.get("inactive_adapters") if isinstance(pv.get("inactive_adapters"), list) else []
            physical = pv.get("physical_or_local_interfaces") if isinstance(pv.get("physical_or_local_interfaces"), list) else []
            virtual = pv.get("virtual_or_auxiliary_interfaces") if isinstance(pv.get("virtual_or_auxiliary_interfaces"), list) else []
            primary = pv.get("primary_adapter") if isinstance(pv.get("primary_adapter"), dict) else None

            pieces = []
            if primary:
                pieces.append("Primary active adapter: " + _format_network_adapter_item(primary))
            if active:
                pieces.append("Active: " + "; ".join(_format_network_adapter_item(a) for a in active[:4]))
            else:
                pieces.append("Active: none detected")
            if inactive:
                pieces.append("Installed/inactive: " + "; ".join(_format_network_adapter_item(a) for a in inactive[:6]))
            if physical:
                cats = []
                for row in physical:
                    cat = _safe_str(row.get("category"), 80).replace("_", " ")
                    if cat and cat not in cats:
                        cats.append(cat)
                if cats:
                    pieces.append("Interface types: " + ", ".join(cats))
            if virtual:
                pieces.append("Virtual/auxiliary present: " + "; ".join(_safe_str(v.get("name"), 160) for v in virtual[:4]))
            drv = pv.get("driver_capability") if isinstance(pv.get("driver_capability"), dict) else {}
            class_a = (((drv.get("class_a") or {}).get("required_stack")) if isinstance(drv.get("class_a"), dict) else [])
            class_b = (((drv.get("class_b") or {}).get("matched_drivers")) if isinstance(drv.get("class_b"), dict) else [])
            if class_a or class_b:
                a_ids = [_safe_str(x.get("id"), 120) for x in class_a if isinstance(x, dict) and x.get("id")]
                b_ids = [_safe_str(x.get("id"), 120) for x in class_b if isinstance(x, dict) and x.get("id")]
                pieces.append("SarahMemory driver capability: Class A " + (", ".join(a_ids[:4]) if a_ids else "none") + "; Class B " + (", ".join(b_ids[:4]) if b_ids else "none"))
            pieces.append("Sensitive IP/MAC details are redacted in chat presentation.")
            pieces.append("Fact-check only; no network action was executed.")
            return "Network adapters = " + " | ".join(pieces)
        if k in {"usb", "usb_ports", "usb_devices", "usb_storage", "usb_host_controllers"} and isinstance(pv, dict):
            count = pv.get("physical_usb_port_count")
            verified = bool(pv.get("physical_usb_port_count_verified"))
            candidates = pv.get("connected_usb_storage_candidate_count")
            cap = "present" if pv.get("usb_host_capability_present") else "not proven"
            if verified and count is not None:
                return f"USB topology = USB host capability {cap}; physical USB port count verified: {count}; connected USB storage candidates: {candidates}; no USB action executed"
            return f"USB topology = USB host capability {cap}; physical USB port count is not yet proven 3/3; connected USB storage candidates: {candidates}; drive letters are not treated as USB port proof; no USB action executed"
        if k in {"sata", "sata_ports"} and isinstance(pv, dict):
            fixed = pv.get("local_fixed_storage_candidate_count")
            if pv.get("sata_port_count_verified") and pv.get("sata_port_count") is not None:
                return f"SATA topology = SATA port count verified: {pv.get('sata_port_count')}; local fixed storage candidates: {fixed}; no storage action executed"
            return f"SATA topology = SATA port count is not yet proven 3/3; local fixed storage candidates: {fixed}; fixed drive letters may be SATA, NVMe/PCIe, RAID, virtual, or another local path; no storage action executed"
        if k == "nvme" and isinstance(pv, dict):
            return "NVMe topology = NVMe is treated as a storage protocol, not a mount letter. Current evidence does not yet prove whether an NVMe device is on motherboard M.2, PCIe adapter card, USB bridge, NAS, or virtual path; no storage action executed"
        if k in {"storage_topology", "storage_devices", "drive", "drives"} and isinstance(pv, dict):
            counts = pv.get("counts") if isinstance(pv.get("counts"), dict) else {}
            return "Storage topology = volumes: " + str(counts.get("volume_count", 0)) + "; fixed candidates: " + str(counts.get("fixed_volume_count", 0)) + "; removable/external candidates: " + str(counts.get("removable_or_external_candidate_count", 0)) + "; drive letters are mount surfaces, not bus proof"
        if k in {"pci", "pcie"} and isinstance(pv, dict):
            return "PCI/PCIe topology = Class A PCI/PCIe capability present; exact slot/device map requires deeper bus enumeration; no PCI action executed"
        if k in {"ports", "port"} and isinstance(pv, dict):
            return "Ports topology = USB/SATA/PCIe/network ports are separate connection surfaces. Current evidence can describe capability and candidates, but mounted drives are not treated as physical port counts; no port or device action executed"
        if k in {"operating_system", "platform"} and isinstance(pv, dict):
            return "Operating platform = " + _safe_str(pv.get("system"), 80) + " " + _safe_str(pv.get("release"), 80) + " (" + _safe_str(pv.get("version"), 160) + "), machine " + _safe_str(pv.get("machine"), 80) + ", Python " + _safe_str(pv.get("python"), 80) + "; network/IP/MAC details redacted"
        if k in {"drivers", "driver", "driver_stack", "body_capability", "body_capabilities", "hardware_capability"} and isinstance(pv, dict):
            ca = pv.get("class_a") if isinstance(pv.get("class_a"), dict) else {}
            cb = pv.get("class_b") if isinstance(pv.get("class_b"), dict) else {}
            return "Driver capability = Class A boot drivers: " + str(ca.get("count", 0)) + "; Class B runtime drivers: " + str(cb.get("catalog_count", 0)) + "; read-only registry scan only; no driver activation executed"
        if k == "motherboard":
            return f"Motherboard = {_safe_str(pv, 300)}" if pv not in (None, "", [], {}) else "Motherboard = unavailable"
        if k == "cpu" and isinstance(pv, dict):
            parts = [_safe_str(pv.get("name"), 240)]
            cores = pv.get("physical_cores")
            threads = pv.get("logical_threads")
            clock = pv.get("current_clock_mhz") or pv.get("max_clock_mhz")
            extras = []
            if cores not in (None, ""):
                extras.append(f"{cores} physical cores")
            if threads not in (None, ""):
                extras.append(f"{threads} logical threads")
            if clock not in (None, ""):
                extras.append(f"clock about {clock} MHz")
            return "CPU = " + parts[0] + (" (" + ", ".join(extras) + ")" if extras else "")
        if k == "gpu" and isinstance(pv, dict):
            name = _safe_str(pv.get("name"), 240)
            extras = []
            if pv.get("temperature_c") not in (None, ""):
                extras.append(f"{pv.get('temperature_c')}°C")
            if pv.get("utilization_pct") not in (None, ""):
                extras.append(f"{pv.get('utilization_pct')}% utilization")
            if pv.get("vram_total_mb") not in (None, ""):
                extras.append(f"VRAM {pv.get('vram_total_mb')} MB")
            return "GPU = " + name + (" (" + ", ".join(extras) + ")" if extras else "")
        if isinstance(pv, (dict, list)):
            return _safe_str(json.dumps(_json_safe(pv), ensure_ascii=False, sort_keys=True), 1000)
        return _safe_str(pv, 1000)
    except Exception:
        return _safe_str(value, 1000)


def _run_fact_ticket(
    *,
    claim: str,
    kind: str = "",
    target: str = "",
    source: str = "chat",
    ticket_kind: str = "SELF_FACT_TICKET",
    rem_ticket: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    claim = _safe_str(claim, MAX_CLAIM_CHARS)
    kind = _infer_fact_kind(claim, kind)
    target = _safe_str(target or _extract_target_from_claim(claim, kind), 255)
    jurisdiction = _fact_jurisdiction(kind, claim)
    probes = _collect_fact_probes(kind, claim, target)
    pass_count, majority_key, majority_value, quorum_debug = _extract_majority_value(probes, kind)
    decision, confidence, risk_tier = _decision_from_pass_count(pass_count)
    driver_support = _build_driver_capability_snapshot(kind) if kind in _HARDWARE_KINDS or kind in _DRIVER_KINDS else {}
    public_majority_value = _presentation_value(kind, majority_value) if kind in ({"network", "network_card", "wifi_card", "bluetooth_card", "ethernet_card", "lan"} | _TOPOLOGY_FACT_KINDS) else majority_value

    ticket_id = _new_ticket_id("self_fact")
    ts = _now()
    ticket: Dict[str, Any] = {
        "ok": True,
        "ticket_id": ticket_id,
        "ts": ts,
        "ts_iso": _iso(ts),
        "source": _safe_str(source, 120) or "chat",
        "ticket_kind": ticket_kind,
        "claim": claim,
        "normalized_claim": _normalize_value(claim),
        "requested_fact": kind,
        "jurisdiction": jurisdiction,
        "target": target,
        "quorum": f"{min(max(pass_count, 0), 3)}/3",
        "pass_count": int(pass_count),
        "decision": decision,
        "confidence": confidence,
        "risk_tier": risk_tier,
        "approved_fact": bool(decision == "APPROVED_FACT"),
        "escalate_high_review": bool(decision == "ESCALATE_HIGH_REVIEW"),
        "majority_value": public_majority_value,
        "majority_key": majority_key,
        "driver_capability": driver_support,
        "presentation_value": _presentation_value(kind, majority_value),
        "presentation_text": _presentation_text(kind, majority_value),
        "evidence": {
            "evidence_court": "SelfAware Evidence Court",
            "jurisdiction": jurisdiction,
            "probes": probes,
            "quorum_debug": quorum_debug,
            "registry": _evidence_source_registry().get("jurisdictions", {}).get(jurisdiction, {}),
            "driver_capability_support": driver_support,
            "rule": "0/3 denied, 1/3 denied, 2/3 high review, 3/3 approved fact; no double jeopardy; OS adapters support only; driver registry support only",
        },
        "limits": {
            "executes_actions": False,
            "modifies_files": False,
            "patches_core": False,
            "network_required": False,
            "globals_read_only": True,
            "database_read_only_for_evidence": True,
            "network_fact_checks_do_not_open_ports": True,
            "network_presentation_redacts_ip_mac": True,
            "driver_registry_read_only": True,
            "driver_code_imported": False,
            "driver_session_started": False,
            "driver_configs_mutated": False,
        },
        "rem_ticket_id": "",
        "rem_cycle_id": "",
        "meta": meta or {},
    }

    if isinstance(rem_ticket, dict):
        ticket["rem_ticket_id"] = _safe_str(rem_ticket.get("ticket_id") or rem_ticket.get("dream_id") or rem_ticket.get("id"), 180)
        ticket["rem_cycle_id"] = _safe_str(rem_ticket.get("cycle_id") or rem_ticket.get("rem_cycle_id"), 180)
        ticket["rem_candidate"] = rem_ticket

    if decision == "APPROVED_FACT":
        if ticket_kind == "SELF_REM_MEDIUM_REVIEW_TICKET":
            ticket["promoted_ticket_type"] = "ACTION_CONCEPTUAL_TICKET"
            ticket["recommended_next"] = "Send fact-supported conceptual action to SarahMemoryCognitiveServices for judgment. User remains jury for material action."
        else:
            ticket["recommended_next"] = "Approved verified fact. Route to SarahMemoryReply/presentation boundary for safe reporting. No action execution required."
    elif decision == "ESCALATE_HIGH_REVIEW":
        ticket["recommended_next"] = "Escalate to HIGH review. Two independent evidence families agree, but SelfAware does not approve final fact."
    else:
        ticket["recommended_next"] = "Reject/archive as unsupported or weak evidence."

    ticket = _json_safe(ticket)
    _store_ticket(ticket)
    _write_selfaware_report(f"{ticket_id}", ticket)
    return ticket



def _factcheck_error_ticket(*, claim: str, kind: str = "", target: str = "", source: str = "appself", exc: Exception, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a safe failure ticket instead of allowing HTTP 500 or chat crashes."""
    ts = _now()
    error_type = type(exc).__name__
    detail = _safe_str(str(exc), 2000)
    tb = _safe_str(traceback.format_exc(), 8000)
    requested_fact = _infer_fact_kind(claim, kind)
    ticket: Dict[str, Any] = {
        "ok": True,
        "ticket_id": _new_ticket_id("self_fact_error"),
        "ts": ts,
        "ts_iso": _iso(ts),
        "source": _safe_str(source, 120) or "appself",
        "ticket_kind": "SELF_FACT_TICKET_ERROR",
        "claim": _safe_str(claim, MAX_CLAIM_CHARS),
        "normalized_claim": _normalize_value(claim),
        "requested_fact": requested_fact,
        "jurisdiction": _fact_jurisdiction(requested_fact, claim),
        "target": _safe_str(target, 255),
        "quorum": "0/3",
        "pass_count": 0,
        "decision": "FACT_CHECK_INTERNAL_ERROR",
        "confidence": "NONE",
        "risk_tier": "medium",
        "approved_fact": False,
        "escalate_high_review": False,
        "majority_value": None,
        "majority_key": "",
        "presentation_value": None,
        "presentation_text": "",
        "evidence": {
            "evidence_court": "SelfAware Evidence Court",
            "error_type": error_type,
            "error": detail,
            "traceback": tb,
            "rule": "Internal fact-check failure becomes an audit ticket; no guessing and no HTTP 500 required.",
        },
        "limits": {
            "executes_actions": False,
            "modifies_files": False,
            "patches_core": False,
            "network_required": False,
            "globals_read_only": True,
            "database_read_only_for_evidence": True,
        },
        "recommended_next": "Inspect this SelfAware error ticket and patch the failing witness/evidence adapter. No fact approved.",
        "rem_ticket_id": "",
        "rem_cycle_id": "",
        "meta": meta or {},
    }
    ticket = _json_safe(ticket)
    _store_ticket(ticket)
    _write_selfaware_report(str(ticket.get("ticket_id") or "self_fact_error"), ticket)
    return ticket


def run_selfaware_fact_check(
    *,
    claim: str,
    kind: str = "",
    target: str = "",
    source: str = "api_chat",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Public in-process API for app.py chat routing.

    This wraps _run_fact_ticket so the ChatPanel path and the HTTP
    /api/self/fact-check path use the same SelfAware evidence court. It is
    intentionally read-only and never executes repairs, patches, file writes,
    dependency installs, or DevBridge actions.
    """
    try:
        return _run_fact_ticket(
            claim=claim,
            kind=kind,
            target=target,
            source=source,
            ticket_kind="SELF_FACT_TICKET",
            meta=meta or {"source": source, "route": "run_selfaware_fact_check"},
        )
    except Exception as exc:
        return _factcheck_error_ticket(
            claim=claim,
            kind=kind,
            target=target,
            source=source,
            exc=exc,
            meta=meta or {"source": source, "route": "run_selfaware_fact_check", "caught": True},

        )


# -----------------------------------------------------------------------------
# V10/V9C verified artifact contract
# -----------------------------------------------------------------------------
def _artifact_clean_text(kind: str, text: Any) -> str:
    value = _safe_str(text, 2000)
    if not value:
        return ""
    try:
        value = re.sub(r"^\s*Verified\s+SelfAware\s+fact\s*\([^)]*\)\s*:\s*", "", value, flags=re.I).strip()
        labels = {
            "cpu": "CPU",
            "gpu": "GPU",
            "motherboard": "Motherboard",
            "memory": "Memory",
            "network": "Network adapters",
            "network_card": "Network adapters",
            "operating_system": "Operating platform",
            "platform": "Operating platform",
        }
        candidates = [labels.get(str(kind or "").lower(), ""), "CPU", "GPU", "Motherboard", "Memory", "Network adapters", "Operating platform"]
        for label in candidates:
            if label and value.lower().startswith((label + " =").lower()):
                value = value[len(label) + 2:].strip()
                break
    except Exception:
        pass
    return value.strip()


def _hardware_epoch_from_ticket(ticket: Dict[str, Any]) -> str:
    try:
        ev = ticket.get("evidence") if isinstance(ticket.get("evidence"), dict) else {}
        raw = json.dumps(_json_safe({
            "requested_fact": ticket.get("requested_fact"),
            "quorum": ticket.get("quorum"),
            "confidence": ticket.get("confidence"),
            "ts_iso": ticket.get("ts_iso"),
            "majority_key": ticket.get("majority_key"),
            "presentation_text": ticket.get("presentation_text"),
            "jurisdiction": ticket.get("jurisdiction"),
            "probe_count": len((ev.get("probes") or []) if isinstance(ev.get("probes"), list) else []),
        }), ensure_ascii=False, sort_keys=True)
        return "bootenv_" + hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:24]
    except Exception:
        return "bootenv_" + uuid.uuid4().hex[:24]


def build_verified_artifact_package(ticket: Dict[str, Any], *, claim: str = "", cycle_id: str = "") -> Optional[Dict[str, Any]]:
    """Build a one-cycle volatile package for live SelfAware hardware/body facts.

    This package is allowed to travel through /api/chat for presentation and
    Compare integrity/security review, but it must not be persisted as learned
    memory or sent back into Evidence Court during the same chat cycle.
    """
    if not isinstance(ticket, dict) or not bool(ticket.get("approved_fact")):
        return None
    kind = _safe_str(ticket.get("requested_fact") or "hardware_fact", 120) or "hardware_fact"
    artifact_value = ticket.get("presentation_value") if ticket.get("presentation_value") not in (None, "", [], {}) else ticket.get("majority_value")
    artifact_text = _artifact_clean_text(kind, ticket.get("presentation_text") or "")
    if not artifact_text:
        try:
            artifact_text = _artifact_clean_text(kind, _presentation_text(kind, artifact_value))
        except Exception:
            artifact_text = _artifact_clean_text(kind, artifact_value)
    artifact_id = "self_artifact_" + uuid.uuid4().hex
    artifact_hash = hashlib.sha256(str(artifact_text or "").encode("utf-8", errors="replace")).hexdigest()
    cycle = _safe_str(cycle_id or (ticket.get("meta") or {}).get("cycle_id") or "", 120)
    if not cycle:
        cycle = "chat_cycle_" + uuid.uuid4().hex[:16]
    pkg: Dict[str, Any] = {
        "ok": True,
        "package_type": VERIFIED_ARTIFACT_PACKAGE_TYPE,
        "package_version": VERIFIED_ARTIFACT_PACKAGE_VERSION,
        "artifact_id": artifact_id,
        "cycle_id": cycle,
        "hardware_epoch": _hardware_epoch_from_ticket(ticket),
        "source_authority": "appself.evidence_court",
        "origin_route": "/api/self/fact-check",
        "artifact_kind": kind,
        "artifact_value": _json_safe(artifact_value),
        "artifact_text": artifact_text,
        "artifact_hash": artifact_hash,
        "claim": _safe_str(claim or ticket.get("claim") or "", MAX_CLAIM_CHARS),
        "source_ticket_id": ticket.get("ticket_id"),
        "approved_fact": True,
        "evidence_court": {
            "decision": ticket.get("decision"),
            "quorum": ticket.get("quorum"),
            "confidence": ticket.get("confidence"),
            "pass_count": ticket.get("pass_count"),
        },
        "single_use": True,
        "volatile": True,
        "live_only": True,
        "immutable_artifact": True,
        "consumed": False,
        "compare_mode": "integrity_security_only",
        "do_not_reverify": True,
        "do_not_persist": True,
        "do_not_learn": True,
        "do_not_write_sql": True,
        "do_not_promote_to_memory": True,
        "volatile_runtime_fact": True,
        "redacted_audit_allowed": True,
        "raw_evidence_exposure_allowed": False,
        "allowed_use": ["chat.presentation", "reply.formatting", "compare.integrity_check"],
        "blocked_use": ["memory_learning", "sql_fact_storage", "response_history_fact_promotion", "selfaware_reverification_loop", "hardware_identity_cache_promotion"],
        "privacy_contract": {
            "raw_evidence_exposed": False,
            "ip_addresses_exposed": False,
            "mac_addresses_exposed": False,
            "full_environment_dump_exposed": False,
            "network_details_are_counted_or_redacted": True,
        },
        "read_only": True,
        "action_taken": False,
    }
    return _json_safe(pkg)


def run_selfaware_verified_artifact(
    *,
    claim: str,
    kind: str = "",
    target: str = "",
    source: str = "api_chat",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta = dict(meta or {})
    cycle_id = _safe_str(meta.get("cycle_id") or ("chat_cycle_" + uuid.uuid4().hex[:16]), 120)
    meta["cycle_id"] = cycle_id
    meta.setdefault("route", "run_selfaware_verified_artifact")
    ticket = run_selfaware_fact_check(claim=claim, kind=kind, target=target, source=source, meta=meta)
    package = build_verified_artifact_package(ticket, claim=claim, cycle_id=cycle_id)
    return _json_safe({
        "ok": True,
        "contract": "V10_V9C_UNIVERSAL_RUNTIME_BODY_MEMORY_AUTHORITY",
        "ticket": ticket,
        "verified_artifact_package": package,
        "approved_fact": bool(isinstance(package, dict) and package.get("approved_fact")),
        "presentation_text": package.get("artifact_text") if isinstance(package, dict) else "",
        "do_not_write_sql": True,
        "do_not_persist": True,
        "do_not_learn": True,
    })


# -----------------------------------------------------------------------------
# Self/governance snapshots
# -----------------------------------------------------------------------------
def _cognitive_self_status() -> Dict[str, Any]:
    if _CogSelf is None:
        return {"available": False, "error": "SarahMemoryCognitiveSelf unavailable"}
    try:
        fn = getattr(_CogSelf, "get_self_summary", None)
        if callable(fn):
            return {"available": True, "summary": fn(context={"source": "appself.status"})}
        fn = getattr(_CogSelf, "get_latest_cognitive_self_model", None)
        if callable(fn):
            return {"available": True, "model": fn(refresh_if_stale=True, context={"source": "appself.status"}, max_age_sec=60)}
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    return {"available": False, "error": "No compatible CognitiveSelf status function found"}


def _body_snapshot() -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False, "sources": {}}
    try:
        if _Hi is not None:
            fn = getattr(_Hi, "get_boot_environment_snapshot", None)
            if callable(fn):
                out["sources"]["SarahMemoryHi.get_boot_environment_snapshot"] = fn(force_refresh=False, refresh_reason="appself_body", persist=True)
            fn2 = getattr(_Hi, "get_extended_system_info", None)
            if callable(fn2):
                out["sources"]["SarahMemoryHi.get_extended_system_info"] = fn2()
            out["available"] = True
    except Exception as exc:
        out.setdefault("errors", []).append(f"SarahMemoryHi:{exc}")
    try:
        if _CogSelf is not None:
            fn = getattr(_CogSelf, "get_latest_cognitive_self_model", None)
            if callable(fn):
                model = fn(refresh_if_stale=True, context={"source": "appself_body"}, max_age_sec=60)
                out["sources"]["SarahMemoryCognitiveSelf.body_map"] = (model or {}).get("body_map") or model
                out["available"] = True
    except Exception as exc:
        out.setdefault("errors", []).append(f"CognitiveSelf:{exc}")
    return out


def _governance_contract() -> Dict[str, Any]:
    return {
        "ok": True,
        "module": "appself.py",
        "api_domain": "self",
        "namespace": "/api/self/*",
        "doctrine": {
            "thinker": "REM/CognitiveThinker may imagine.",
            "selfaware": "SelfAware/CognitiveSelf must prove with evidence.",
            "services": "CognitiveServices judges allowed action.",
            "user": "User remains jury for material action.",
        },
        "quorum_rule": {
            "0/3": "DENIED_NO_EVIDENCE",
            "1/3": "DENIED_WEAK_EVIDENCE",
            "2/3": "ESCALATE_HIGH_REVIEW",
            "3/3": "APPROVED_FACT",
        },
        "limits": {
            "executes_repairs": False,
            "patches_files": False,
            "modifies_core": False,
            "opens_attachments": False,
            "installs_dependencies": False,
            "replaces_cognitive_services": False,
            "frontend_ticket_queue": False,
            "driver_activation": False,
            "driver_config_mutation": False,
            "network_scanning": False,
        },
        "evidence_doctrine": {
            "one_file_family_one_vote": True,
            "one_artifact_family_one_vote": True,
            "os_adapters_are_support_only": True,
            "globals_read_only": True,
            "database_rows_do_not_count_as_multiple_witnesses": True,
            "double_jeopardy_rejected": True,
            "driver_folders_are_capability_not_hardware_proof": True,
            "class_a_class_b_driver_registry_read_only": True,
        },
        "evidence_registry": _evidence_source_registry(),
        "modules": _module_matrix(),
    }


def _module_matrix() -> Dict[str, Dict[str, Any]]:
    return {
        "SarahMemoryCognitiveSelf": {"available": _CogSelf is not None, "role": "canonical identity/capability/body-map authority"},
        "SarahMemorySelfAware": {"available": _SelfAware is not None, "role": "introspection/investigation engine"},
        "SarahMemoryCognitiveCompass": {"available": _CogCompass is not None, "role": "mission/case-completion supervisor"},
        "SarahMemoryHi": {"available": _Hi is not None, "role": "hardware/system telemetry witness"},
        "runtime_environment_snapshot.json": {"available": _runtime_environment_snapshot_path().exists(), "role": "persisted boot/runtime physical artifact"},
        "SarahMemoryDiagnostics": {"available": _Diagnostics is not None, "role": "diagnostics/health/core-file support witness"},
        "SarahMemoryOptimization": {"available": _Optimization is not None, "role": "runtime/resource support witness"},
        "SarahMemorySi": {"available": _Si is not None, "role": "software/process/window witness"},
        "SarahMemorySynapes": {"available": _Synapes is not None, "role": "sandbox lab/provenance witness"},
        "SarahMemoryInitialization": {"available": _Initialization is not None, "role": "boot-sequence witness"},
        "SarahMemoryIntegration": {"available": _Integration is not None, "role": "runtime/action orchestration witness"},
        "SarahMemoryNetwork": {"available": _Network is not None, "role": "SarahNet protocol/transport witness"},
        "SarahMemorySync": {"available": _Sync is not None, "role": "cross-device sync witness"},
        "SarahMemorySMAPI": {"available": _SMAPI is not None, "role": "safe internal runtime introspection witness"},
        "SarahMemoryEvolution": {"available": _Evolution is not None, "role": "restricted monkey-patch proposal witness"},
        "SarahMemoryFacialRecognition": {"available": _FacialRecognition is not None, "role": "vision/face-vector witness"},
        "SarahMemorySOBJE": {"available": _SOBJE is not None, "role": "object/scene observation witness"},
        "SarahMemoryAiFunctions": {"available": _AiFunctions is not None, "role": "agent/context witness"},
        "SarahMemoryDL": {"available": _DL is not None, "role": "learning/model support witness"},
        "SarahMemoryAdaptive": {"available": _Adaptive is not None, "role": "adaptive behavior/emotion witness"},
        "SarahMemoryCognitiveServices": {"available": _CogServices is not None, "role": "judge/governor"},
        "SarahMemorySafetyPolicies": {"available": _SafetyPolicies is not None, "role": "constitutional policy"},
        "SarahMemorySecurityGovernor": {"available": _SecurityGovernor is not None, "role": "trust boundary"},
        "SarahMemoryAssuranceGate": {"available": _AssuranceGate is not None, "role": "confidence/readiness gate"},
        "SarahMemoryTrustRegistry": {"available": _TrustRegistry is not None, "role": "witness trust registry"},
        "SarahMemoryDriverRegistry": {"available": bool(_read_boot_driver_registry().get("ok") or _read_runtime_driver_catalog().get("drivers_root_exists")), "role": "read-only Class A/Class B driver capability witness"},
        "appdrivers.py": {"available": _appdrivers_contract_fact("drivers").get("ok", False), "role": "governed driver API domain witness; not activated by SelfAware"},
    }

def _policy_snapshot() -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    try:
        if _SafetyPolicies is not None:
            for name in ("get_safety_policy_snapshot", "runtime_policy_snapshot", "get_runtime_policy_snapshot"):
                fn = getattr(_SafetyPolicies, name, None)
                if callable(fn):
                    val = fn()
                    out.update({"available": True, "source": f"SarahMemorySafetyPolicies.{name}", "snapshot": val})
                    return out
    except Exception as exc:
        out["error"] = str(exc)
    return out


def _govern_action_concept(ticket: Dict[str, Any], rem_candidate: Dict[str, Any]) -> Dict[str, Any]:
    if _CogServices is None:
        return {"available": False, "decision": "DEFER", "reason": "SarahMemoryCognitiveServices unavailable"}
    try:
        title = _safe_str(rem_candidate.get("title") or rem_candidate.get("summary") or rem_candidate.get("claim") or ticket.get("claim"), 600)
        proposed = {
            "type": "SELF_AWARE_APPROVED_ACTION_CONCEPT",
            "source_ticket_id": ticket.get("ticket_id"),
            "rem_ticket_id": ticket.get("rem_ticket_id"),
            "risk_tier": "medium",
            "fact_decision": ticket.get("decision"),
            "claim": ticket.get("claim"),
            "approved_fact": ticket.get("majority_value"),
            "writes_core_files": False,
            "executes_code": False,
            "requires_user_for_material_action": True,
        }
        fn = getattr(_CogServices, "govern_request", None)
        if callable(fn):
            decision = fn(
                f"Review SelfAware-approved REM MEDIUM action concept: {title}",
                caller="appself.SelfAware",
                caller_context={
                    "source": "appself.rem_medium_review",
                    "selfaware_ticket_id": ticket.get("ticket_id"),
                    "fact_quorum": ticket.get("quorum"),
                    "sandbox_verified": False,
                },
                user_present=False,
                user_consented=False,
                proposed_action=proposed,
            )
            return {"available": True, "decision": decision}
    except Exception as exc:
        return {"available": True, "decision": "ERROR", "error": str(exc)}
    return {"available": True, "decision": "DEFER", "reason": "No compatible govern_request function"}


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@bp.route("/api/self/health", methods=["GET", "OPTIONS"])
def self_health():
    if request.method == "OPTIONS":
        return _ok()
    return _ok({
        "module": "appself.py",
        "version": PROJECT_VERSION,
        "namespace": "/api/self/*",
        "mounted": True,
        "db_injected": _require_injected(),
        "fallback_db": str(_fallback_db_path()),
        "modules": _module_matrix(),
        "ts": _iso(),
    })


@bp.route("/api/self/governance", methods=["GET", "OPTIONS"])
def self_governance():
    if request.method == "OPTIONS":
        return _ok()
    contract = _governance_contract()
    contract["policy"] = _policy_snapshot()
    return _ok(contract)


@bp.route("/api/self/status", methods=["GET", "OPTIONS"])
def self_status():
    if request.method == "OPTIONS":
        return _ok()
    return _ok({
        "self": _cognitive_self_status(),
        "tickets_recent": _list_tickets(limit=10),
        "ts": _iso(),
    })


@bp.route("/api/self/body", methods=["GET", "OPTIONS"])
def self_body():
    if request.method == "OPTIONS":
        return _ok()
    force = str(request.args.get("force", "false")).strip().lower() in {"1", "true", "yes", "on"}
    if force and _Hi is not None:
        try:
            fn = getattr(_Hi, "get_boot_environment_snapshot", None)
            if callable(fn):
                fn(force_refresh=True, refresh_reason="appself_body_force", persist=True)
        except Exception:
            pass
    return _ok(_body_snapshot())


@bp.route("/api/self/verified-artifact", methods=["POST", "OPTIONS"])
def self_verified_artifact():
    """Build a volatile verified hardware/body artifact for /api/chat.

    This route is read-only. It does not execute hardware actions, patch files,
    or promote the fact into learned memory.
    """
    if request.method == "OPTIONS":
        return _ok({"options": True})
    body = _body_bytes()
    if not _verify_auth(body):
        return _err("unauthorized", 401)
    payload = _j()
    claim = _safe_str(payload.get("claim") or payload.get("text") or payload.get("query") or "", MAX_CLAIM_CHARS)
    kind = _safe_str(payload.get("kind") or payload.get("requested_fact") or "", 120)
    target = _safe_str(payload.get("target") or "", 255)
    source = _safe_str(payload.get("source") or "api_self_verified_artifact", 120)
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    try:
        result = run_selfaware_verified_artifact(claim=claim, kind=kind, target=target, source=source, meta=meta)
        return _ok(result)
    except Exception as exc:
        ticket = _factcheck_error_ticket(claim=claim, kind=kind, target=target, source=source, exc=exc, meta=meta)
        return _ok({"ok": False, "ticket": ticket, "verified_artifact_package": None, "error": str(exc)})


@bp.route("/api/self/fact-check", methods=["POST", "OPTIONS"])
def self_fact_check():
    """HTTP SelfAware fact-check route with outer crash containment.

    The ChatPanel and PowerShell tests must never receive HTTP 500 for a
    fact-check. If a witness crashes, the route returns a 200 response with a
    FACT_CHECK_INTERNAL_ERROR / FACT_CHECK_OUTER_ERROR ticket and no guessed fact.
    """
    claim = ""
    fact_kind = ""
    target = ""
    source = "chat"
    data: Dict[str, Any] = {}
    try:
        if request.method == "OPTIONS":
            return _ok()
        body = _body_bytes()
        if not _verify_auth(body):
            return _err("unauthorized", 401)
        data = _j()
        claim = _safe_str(data.get("claim") or data.get("question") or data.get("text"), MAX_CLAIM_CHARS)
        if not claim:
            return _err("claim required", 400)
        fact_kind = _safe_str(data.get("kind") or data.get("fact_kind"), 80)
        target = _safe_str(data.get("target") or data.get("drive") or data.get("module") or data.get("file"), 255)
        source = _safe_str(data.get("source") or "chat", 120)
        try:
            ticket = _run_fact_ticket(
                claim=claim,
                kind=fact_kind,
                target=target,
                source=source,
                ticket_kind="SELF_FACT_TICKET",
                meta={"request_meta": data.get("meta") if isinstance(data.get("meta"), dict) else {}},
            )
        except Exception as exc:
            ticket = _factcheck_error_ticket(
                claim=claim,
                kind=fact_kind,
                target=target,
                source=source,
                exc=exc,
                meta={"request_meta": data.get("meta") if isinstance(data.get("meta"), dict) else {}, "caught": True},
            )
        return _ok(ticket)
    except Exception as fatal:
        # Last-resort guard: never allow a SelfAware evidence failure to become HTTP 500.
        try:
            ticket = _factcheck_error_ticket(
                claim=claim or "unknown_fact_check_claim",
                kind=fact_kind,
                target=target,
                source=source or "chat",
                exc=fatal,
                meta={"outer_guard": True, "request_meta": data.get("meta") if isinstance(data, dict) and isinstance(data.get("meta"), dict) else {}},
            )
            ticket["decision"] = "FACT_CHECK_OUTER_ERROR"
            ticket["recommended_next"] = "Inspect appself fact-check outer guard traceback. No fact approved."
            return _response_json({"ok": True, "data": ticket}, 200)
        except Exception as final_exc:
            return _response_json({
                "ok": True,
                "data": {
                    "ok": True,
                    "ticket_kind": "SELF_FACT_TICKET_ERROR",
                    "decision": "FACT_CHECK_FATAL_RESPONSE_ERROR",
                    "approved_fact": False,
                    "quorum": "0/3",
                    "pass_count": 0,
                    "confidence": "NONE",
                    "claim": claim or "unknown_fact_check_claim",
                    "requested_fact": fact_kind or "unknown",
                    "error_type": type(final_exc).__name__,
                    "error": str(final_exc),
                    "outer_error_type": type(fatal).__name__,
                    "outer_error": str(fatal),
                    "recommended_next": "Patch appself response boundary. No fact approved.",
                }
            }, 200)


@bp.route("/api/self/fact-ticket/<ticket_id>", methods=["GET", "OPTIONS"])
def self_fact_ticket(ticket_id: str):
    if request.method == "OPTIONS":
        return _ok()
    tid = _safe_str(ticket_id, 180)
    ticket = _read_ticket(tid)
    if not ticket:
        return _err("ticket not found", 404, ticket_id=tid)
    return _ok(ticket)


@bp.route("/api/self/fact-tickets", methods=["GET", "OPTIONS"])
def self_fact_tickets():
    if request.method == "OPTIONS":
        return _ok()
    limit = max(1, min(MAX_TICKETS_RETURN, int(request.args.get("limit") or 50)))
    decision = _safe_str(request.args.get("decision") or "", 80)
    ticket_kind = _safe_str(request.args.get("ticket_kind") or "", 80)
    return _ok({"tickets": _list_tickets(limit=limit, decision=decision, ticket_kind=ticket_kind), "limit": limit})


@bp.route("/api/self/rem-medium/review", methods=["POST", "OPTIONS"])
def self_rem_medium_review():
    if request.method == "OPTIONS":
        return _ok()
    body = _body_bytes()
    if not _verify_auth(body):
        return _err("unauthorized", 401)
    data = _j()
    candidate = data.get("candidate") if isinstance(data.get("candidate"), dict) else data
    if not isinstance(candidate, dict):
        return _err("candidate object required", 400)

    risk_tier = _safe_str(candidate.get("risk_tier") or candidate.get("tier") or (candidate.get("proposed_action") or {}).get("risk_tier"), 80).lower()
    if risk_tier in {"high", "critical", "tier_3", "tier_4", "tier_3_privileged_system", "tier_4_network_remote_or_destructive"}:
        ticket = {
            "ok": True,
            "ticket_id": _new_ticket_id("self_rem_reject"),
            "ts": _now(),
            "ts_iso": _iso(),
            "ticket_kind": "SELF_REM_HIGH_REJECTED",
            "decision": "DENY_HIGH_REM_NOT_ACCEPTED",
            "risk_tier": risk_tier or "high",
            "reason": "SelfAware only reviews REM MEDIUM candidates. REM HIGH is denied/quarantined by doctrine.",
            "candidate": candidate,
        }
        _store_ticket({
            **ticket,
            "source": "rem",
            "claim": _safe_str(candidate.get("title") or candidate.get("summary") or "REM high candidate", MAX_CLAIM_CHARS),
            "normalized_claim": _normalize_value(candidate.get("title") or candidate.get("summary") or "REM high candidate"),
            "requested_fact": "rem_high_rejected",
            "quorum": "0/3",
            "pass_count": 0,
            "confidence": "NONE",
            "rem_ticket_id": _safe_str(candidate.get("ticket_id") or candidate.get("dream_id") or candidate.get("id"), 180),
            "rem_cycle_id": _safe_str(candidate.get("cycle_id") or candidate.get("rem_cycle_id"), 180),
            "evidence": {"rule": "REM HIGH does not enter SelfAware interrogation"},
        })
        return _ok(ticket)

    # LOW is not rejected as unsafe, but SelfAware does not consume it. LOW remains REM metadata-only.
    if risk_tier in {"low", "tier_0_info", "tier_1_harmless_local_ui"}:
        return _ok({
            "decision": "LOW_NOT_ROUTED_TO_SELFAWARE",
            "reason": "REM LOW remains metadata-only sandbox/audit. SelfAware reviews REM MEDIUM only.",
            "candidate": candidate,
        })

    # Treat empty/unknown risk as medium for review, because REM medium is the intended lane.
    claim = _safe_str(
        data.get("claim")
        or candidate.get("claim")
        or candidate.get("title")
        or candidate.get("summary")
        or candidate.get("description")
        or "REM MEDIUM candidate requires SelfAware fact interrogation",
        MAX_CLAIM_CHARS,
    )
    kind = _safe_str(data.get("kind") or data.get("fact_kind") or candidate.get("fact_kind") or "general_system_fact", 80)
    target = _safe_str(data.get("target") or candidate.get("target") or candidate.get("target_file") or "", 255)

    ticket = _run_fact_ticket(
        claim=claim,
        kind=kind,
        target=target,
        source="rem_medium",
        ticket_kind="SELF_REM_MEDIUM_REVIEW_TICKET",
        rem_ticket=candidate,
        meta={"request_meta": data.get("meta") if isinstance(data.get("meta"), dict) else {}},
    )

    if ticket.get("decision") == "APPROVED_FACT":
        ticket["cognitive_services_review"] = _govern_action_concept(ticket, candidate)
        _store_ticket(ticket)
        _write_selfaware_report(f"{ticket.get('ticket_id')}_governed", ticket)

    return _ok(ticket)


@bp.route("/api/self/evidence-registry", methods=["GET", "OPTIONS"])
def self_evidence_registry():
    if request.method == "OPTIONS":
        return _ok()
    return _ok(_evidence_source_registry())


@bp.route("/api/self/driver-registry", methods=["GET", "OPTIONS"])
def self_driver_registry():
    if request.method == "OPTIONS":
        return _ok()
    return _ok({
        "module": "appself.py",
        "mode": "read_only_driver_capability_registry",
        "class_a": _read_boot_driver_registry(),
        "class_b": _read_runtime_driver_catalog(),
        "appdrivers_contract": _appdrivers_contract_fact("drivers").get("value"),
        "limits": {
            "driver_code_imported": False,
            "driver_session_started": False,
            "driver_config_mutated": False,
            "network_action_taken": False,
        },
    })


@bp.route("/api/self/driver-stack", methods=["GET", "OPTIONS"])
def self_driver_stack():
    if request.method == "OPTIONS":
        return _ok()
    kind = _safe_str(request.args.get("kind") or request.args.get("fact") or "drivers", 80)
    return _ok(_build_driver_capability_snapshot(kind))


@bp.route("/api/self/body-capabilities", methods=["GET", "OPTIONS"])
def self_body_capabilities():
    if request.method == "OPTIONS":
        return _ok()
    return _ok(_body_capabilities_snapshot())


@bp.route("/api/self/driver-health", methods=["GET", "OPTIONS"])
def self_driver_health():
    if request.method == "OPTIONS":
        return _ok()
    body = _body_capabilities_snapshot()
    caps = body.get("capabilities") if isinstance(body.get("capabilities"), dict) else {}
    unhealthy = []
    for kind, snap in caps.items():
        if not isinstance(snap, dict):
            continue
        if snap.get("dependency_status") != "declared_or_available":
            unhealthy.append({"kind": kind, "dependency_status": snap.get("dependency_status"), "snapshot": snap})
    return _ok({
        "ok": True,
        "driver_health": "attention_required" if unhealthy else "declared_or_available",
        "unhealthy": unhealthy,
        "class_a_count": (_read_boot_driver_registry().get("count") or 0),
        "class_b_count": (_read_runtime_driver_catalog().get("count") or 0),
        "read_only": True,
        "action_taken": False,
    })


@bp.route("/api/self/hardware-topology", methods=["GET", "OPTIONS"])
def self_hardware_topology():
    if request.method == "OPTIONS":
        return _ok()
    kind = _safe_str(request.args.get("kind") or request.args.get("fact") or "storage_topology", 80)
    kind = _infer_fact_kind(kind, kind)
    target = _safe_str(request.args.get("target") or request.args.get("drive") or "", 255)
    snap: Dict[str, Any] = {}
    if _Hi is not None:
        try:
            fn = getattr(_Hi, "get_boot_environment_snapshot", None)
            if callable(fn):
                snap = fn(force_refresh=False, refresh_reason="appself_hardware_topology", persist=True) or {}
        except Exception:
            snap = {}
    if not snap:
        snap = _read_json_file(_runtime_environment_snapshot_path())
    topo = _topology_fact_from_snapshot(snap if isinstance(snap, dict) else {}, kind, target)
    if topo in (None, {}, []):
        topo = {
            "topology_kind": kind,
            "available": False,
            "reason": "No topology builder available for requested kind.",
            "action_taken": False,
        }
    return _ok({
        "kind": kind,
        "target": target,
        "topology": topo,
        "presentation_text": _presentation_text(kind, topo),
        "read_only": True,
        "action_taken": False,
        "rule": "Topology separates device class, connection bus, protocol, physical location, mount surface, and ownership scope.",
    })


@bp.route("/api/self/reports/latest", methods=["GET", "OPTIONS"])
def self_reports_latest():
    if request.method == "OPTIONS":
        return _ok()
    limit = max(1, min(50, int(request.args.get("limit") or 10)))
    return _ok({"reports": _list_selfaware_reports(limit=limit)})


# -----------------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------------
def init_app(app, connect_sqlite=None, meta_db_path=None, api_key_auth_ok=None, sign_ok=None):
    global _CONNECT_SQLITE, _META_DB, _API_KEY_AUTH_OK, _SIGN_OK
    _CONNECT_SQLITE = connect_sqlite
    _META_DB = meta_db_path
    _API_KEY_AUTH_OK = api_key_auth_ok
    _SIGN_OK = sign_ok
    try:
        _ensure_tables()
    except Exception as exc:
        logger.warning("appself init failed to ensure tables: %s", exc)
    if "appself_v800" in getattr(app, "blueprints", {}):
        return True
    app.register_blueprint(bp)
    return True


# ============================================================================
# END OF appself.py v8.0.0 EvidenceCourt V8 Topology-Aware
# ============================================================================
