"""--==The SarahMemory Project==--
File: SarahMemoryARILE.py
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


Adaptive Reality Intelligence Layer Engine (ARILE)
=================================================

ARILE is SarahMemory's asynchronous cyber-reality watchdog.  It converts
real-world variance, runtime instability, suspicious behavior, logic bombs,
malware-like file/process patterns, API/MCP boundary drift, driver/device
instability, and creative variance into bounded, structured packets.

Doctrine:
- ARILE observes, scores, throttles, buffers, and reports.
- ARILE does not self-authorize actions.
- ARILE does not mutate core files.
- ARILE does not bypass SMGET, Compare, Compass, SecurityGovernor,
  AssuranceGate, CognitiveServices, OperatorCore, or MSDC.
- SarahMemoryGlobals.py and SarahMemoryARILE.py are protected core files.
- ARILE may accept governed runtime overlays only through an explicit overlay
  lane; direct self/evolution mutation of the source file is blocked.
"""

from __future__ import annotations

import atexit
import collections
import dataclasses
import hashlib
import itertools
import json
import math
import os
import queue
import re
import threading
import time
import traceback
from pathlib import Path, PureWindowsPath
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

MODULE_VERSION = "9.1.0a-arile-enterprise"
ARILE_NAME = "Adaptive Reality Intelligence Layer Engine"

# Protected source files.  These are allowed to be read, hash-verified, and
# backed up, but not directly mutated by Evolution, DevBridge, cleanup, updater,
# FileSystem, REM, MCP/API, or any other autonomous route.
ARILE_PROTECTED_CORE_FILES = {
    "sarahmemoryglobals.py",
    "sarahmemoryarile.py",
}

ARILE_TIER1_SECURITY_FILES = {
    "sarahmemoryglobals.py",
    "sarahmemoryarile.py",
    "sarahmemorysecuritygovernor.py",
    "sarahmemoryassurancegate.py",
    "sarahmemorycompare.py",
    "sarahmemorycognitivecompass.py",
    "sarahmemorycognitiveservices.py",
    "sarahmemoryoperatorcore.py",
    "sarahmemorymsdc.py",
    "sarahmemoryfilesystem.py",
}

SEVERITY_RAM_ONLY_MAX = 0.49
SEVERITY_DIAGNOSTIC_MIN = 0.50
SEVERITY_GOVERNANCE_MIN = 0.75
SEVERITY_EMERGENCY_MIN = 0.90

def _env_int(key: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(os.getenv(key, str(default)) or default)))
    except (TypeError, ValueError, OverflowError):
        return default


def _env_float(key: str, default: float, low: float, high: float) -> float:
    try:
        number = float(os.getenv(key, str(default)) or default)
        return max(low, min(high, number)) if math.isfinite(number) else default
    except (TypeError, ValueError, OverflowError):
        return default


DEFAULT_MAX_QUEUE = _env_int("SARAH_ARILE_MAX_QUEUE", 2048, 16, 8192)
DEFAULT_RING_SIZE = _env_int("SARAH_ARILE_RING_SIZE", 512, 32, 4096)
DEFAULT_BATCH_SECONDS = _env_float("SARAH_ARILE_BATCH_SECONDS", 30, 0.5, 300)
DEFAULT_MIN_WRITE_SEVERITY = _env_float("SARAH_ARILE_MIN_WRITE_SEVERITY", 0.75, 0.0, 1.0)
DEFAULT_MAX_WRITES_PER_BATCH = _env_int("SARAH_ARILE_MAX_WRITES_PER_BATCH", 50, 1, 200)
DEFAULT_MAX_PENDING_AUDIT = _env_int("SARAH_ARILE_MAX_PENDING_AUDIT", 1024, 16, 8192)
DEFAULT_URGENT_FLUSH_SECONDS = _env_float("SARAH_ARILE_URGENT_FLUSH_SECONDS", 1.5, 0.5, 30)
DEFAULT_AUDIT_MAX_BYTES = _env_int("SARAH_ARILE_AUDIT_MAX_BYTES", 16 * 1024 * 1024, 65536, 256 * 1024 * 1024)
DEFAULT_MAX_HEALTH_SOURCES = _env_int("SARAH_ARILE_MAX_HEALTH_SOURCES", 512, 16, 4096)



def _now() -> float:
    return time.time()


def _iso(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_now() if ts is None else ts))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if not math.isfinite(v):
            return default
        return max(0.0, min(1.0, v))
    except Exception:
        return default


def _safe_number(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Return a finite, unclamped engineering value or the supplied default."""
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except Exception:
        return default


def _safe_text(value: Any, limit: int = 1000) -> str:
    try:
        text = str(value if value is not None else "")
    except Exception:
        text = ""
    text = text.replace("\x00", "")
    if len(text) > limit:
        return text[:limit] + "…"
    return text


# Diagnostic/audit data is untrusted. Keep it bounded and do not persist credentials.
_SECRET_KEY = re.compile(r"(?i)(?:password|passwd|secret|token|bearer|api[_-]?key|authorization|cookie|credential|private[_-]?key|signature|return_nonce|session[_-]?id|access[_-]?key|refresh[_-]?key)")
_SECRET_INLINE = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9_.~+/-]{8,}|\b((?:api[_-]?key|token|password|secret)\s*[:=]\s*)[^\s,;]+")


def _redact_text(value: Any, limit: int = 1000) -> str:
    return _SECRET_INLINE.sub(lambda m: (m.group(1) or m.group(2) or "") + "[REDACTED]", _safe_text(value, limit))


def _safe_packet_data(value: Any, depth: int = 0, budget: Optional[List[int]] = None) -> Any:
    """Budgeted JSON-safe evidence snapshot. Never persist raw sensitive fields."""
    if budget is None:
        budget = [24576]
    if budget[0] <= 0:
        return "[TRUNCATED]"
    if depth > 5:
        return "[DEPTH_LIMIT]"
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for key, val in itertools.islice(value.items(), 64):
            key_text = _safe_text(key, 128)
            budget[0] -= len(key_text)
            if _SECRET_KEY.search(key_text):
                result[key_text] = "[REDACTED]"
            else:
                result[key_text] = _safe_packet_data(val, depth + 1, budget)
        if len(value) > 64:
            result["_truncated_items"] = True
        return result
    if isinstance(value, (list, tuple, set)):
        result = [_safe_packet_data(x, depth + 1, budget) for x in itertools.islice(iter(value), 48)]
        if len(value) > 48:
            result.append("[TRUNCATED]")
        return result
    if isinstance(value, (str, bytes, bytearray)):
        if isinstance(value, (bytes, bytearray)):
            value = bytes(value[:1500]).decode('utf-8', errors='replace')
        item = _redact_text(value, min(1500, max(24, budget[0])))
        budget[0] -= len(item)
        return item
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    result = _redact_text(value, min(256, max(24, budget[0])))
    budget[0] -= len(result)
    return result


def _module_base_dir() -> Path:
    try:
        import SarahMemoryGlobals as config  # type: ignore
        for attr in ("ROOT_DIR", "BASE_DIR"):
            val = getattr(config, attr, None)
            if val:
                p = Path(os.fspath(val)).expanduser().resolve()
                # If Globals reports the core folder, keep it; if it reports root,
                # callers can still pass project-relative targets.
                return p
    except Exception:
        pass
    try:
        return Path(__file__).resolve().parent
    except Exception:
        return Path(os.getcwd()).resolve()


def _data_dir() -> Path:
    try:
        import SarahMemoryGlobals as config  # type: ignore
        val = getattr(config, "DATA_DIR", None)
        if val:
            return Path(os.fspath(val)).expanduser().resolve()
    except Exception:
        pass
    base = _module_base_dir()
    # In v9 archives ARILE lives under ./core; data is usually sibling to core.
    if base.name.lower() == "core" and (base.parent / "data").exists():
        return (base.parent / "data").resolve()
    return (base / "data").resolve()


def _log_dir() -> Path:
    try:
        import SarahMemoryGlobals as config  # type: ignore
        val = getattr(config, "LOGS_DIR", None)
        if val:
            return Path(os.fspath(val)).expanduser().resolve()
    except Exception:
        pass
    return (_data_dir() / "logs").resolve()


def _manifest_dir() -> Path:
    return (_data_dir() / "security").resolve()


def _protected_basename(path_value: Any) -> str:
    """Normalize both Windows and POSIX paths on any host OS."""
    try:
        text = os.fspath(path_value)
    except (TypeError, ValueError):
        text = str(path_value or "")
    text = str(text).replace("\\", "/").split("?")[0].rstrip(" /\t\r\n")
    name = text.rsplit("/", 1)[-1].rstrip(". ").casefold()
    # A symlink alias must not conceal a protected source path.
    try:
        resolved_name = Path(text).resolve(strict=False).name.casefold()
        if resolved_name in ARILE_PROTECTED_CORE_FILES or resolved_name in ARILE_TIER1_SECURITY_FILES:
            return resolved_name
    except (OSError, ValueError, RuntimeError):
        pass
    return name


def arile_is_protected_core_file(path_value: Any) -> bool:
    """Return True when a filename is in the protected source list."""
    return _protected_basename(path_value) in ARILE_PROTECTED_CORE_FILES


def arile_is_tier1_security_file(path_value: Any) -> bool:
    return _protected_basename(path_value) in ARILE_TIER1_SECURITY_FILES


def arile_assert_not_protected_mutation(path_value: Any, operation: str = "mutation", source: str = "unknown") -> None:
    """Raise PermissionError if a protected core file is targeted for mutation."""
    if arile_is_protected_core_file(path_value):
        name = _protected_basename(path_value)
        arile_emit(
            source=source,
            kind="protected_core_variance",
            failure_type="protected_core_mutation_attempt",
            severity=0.98,
            confidence=0.95,
            risk="critical",
            summary=f"Blocked {operation} against protected core file: {name}",
            retention="security_audit",
            requires_governance=True,
            data={"target": str(path_value), "operation": operation},
        )
        raise PermissionError(f"Protected core file mutation blocked: {name}")


def arile_protected_files() -> List[str]:
    return sorted({"SarahMemoryGlobals.py", "SarahMemoryARILE.py"})


def _severity_class(severity: float) -> str:
    if severity >= SEVERITY_EMERGENCY_MIN:
        return "EMERGENCY"
    if severity >= SEVERITY_GOVERNANCE_MIN:
        return "FAILURE"
    if severity >= SEVERITY_DIAGNOSTIC_MIN:
        return "ERROR"
    return "ALERT" if severity >= 0.25 else "INFO"


@dataclasses.dataclass
class RealityVariancePacket:
    source: str
    kind: str
    failure_type: str = ""
    severity: float = 0.0
    novelty: float = 0.0
    confidence: float = 0.5
    risk: str = "low"
    summary: str = ""
    requires_governance: bool = False
    retention: str = "ram"
    organ: str = ""
    expected_state: str = ""
    observed_state: str = ""
    recommended_response: str = ""
    fault_class: str = ""
    severity_class: str = ""
    residual: Optional[float] = None
    normalized_residual: Optional[float] = None
    rate_of_change: Optional[float] = None
    duration_seconds: float = 0.0
    witness_timestamp: str = ""
    freshness: Optional[float] = None
    controller: str = ""
    contract_id: str = ""
    safe_stop_required: bool = False
    cleared: bool = False
    execution_authority: bool = False
    data: Optional[Dict[str, Any]] = None
    timestamp: str = dataclasses.field(default_factory=_iso)
    packet_id: str = ""

    def normalize(self) -> "RealityVariancePacket":
        self.source = _redact_text(self.source, 160) or "unknown"
        self.kind = _redact_text(self.kind, 120) or "variance"
        self.failure_type = _redact_text(self.failure_type, 160)
        raw_severity = _safe_number(self.severity)
        if raw_severity is None:
            self.severity = 1.0
            self.failure_type = self.failure_type or "malformed_severity"
        else:
            self.severity = max(0.0, min(1.0, raw_severity))
        self.novelty = _safe_float(self.novelty, 0.0)
        self.confidence = _safe_float(self.confidence, 0.5)
        self.risk = (_safe_text(self.risk, 40) or "low").lower()
        self.summary = _redact_text(self.summary, 1000)
        self.organ = _redact_text(self.organ, 80)
        self.expected_state = _redact_text(self.expected_state, 500)
        self.observed_state = _redact_text(self.observed_state, 500)
        self.recommended_response = _redact_text(self.recommended_response, 500)
        self.fault_class = _safe_text(self.fault_class, 80)
        self.severity_class = _safe_text(self.severity_class, 40).upper()
        self.residual = _safe_number(self.residual)
        self.normalized_residual = _safe_number(self.normalized_residual)
        self.rate_of_change = _safe_number(self.rate_of_change)
        self.duration_seconds = max(0.0, _safe_number(self.duration_seconds, 0.0) or 0.0)
        self.witness_timestamp = _safe_text(self.witness_timestamp, 80)
        self.freshness = _safe_float(self.freshness, 0.0) if self.freshness is not None else None
        self.controller = _redact_text(self.controller, 160)
        self.contract_id = _redact_text(self.contract_id, 160)
        self.safe_stop_required = bool(self.safe_stop_required)
        if self.safe_stop_required:
            self.severity = max(self.severity, SEVERITY_GOVERNANCE_MIN)
        self.cleared = bool(self.cleared)
        self.execution_authority = False
        # Severity is computed from evidence; a self-attested class cannot downrank it.
        self.severity_class = _severity_class(self.severity)
        if self.safe_stop_required or self.severity >= SEVERITY_GOVERNANCE_MIN:
            self.requires_governance = True
        self.retention = _safe_text(self.retention, 80) or "ram"
        self.data = _safe_packet_data(self.data if isinstance(self.data, dict) else {})
        if not self.packet_id:
            seed = json.dumps(self.to_dict(include_id=False), sort_keys=True, default=str).encode("utf-8", errors="ignore")
            self.packet_id = "arile_" + hashlib.sha256(seed + str(_now()).encode()).hexdigest()[:16]
        return self

    def to_dict(self, include_id: bool = True) -> Dict[str, Any]:
        out = dataclasses.asdict(self)
        if not include_id:
            out.pop("packet_id", None)
        return out


class ARILESentinelBase:
    """Thin base class for organ-level sentinels."""

    organ_name = "generic"

    def __init__(self, source: Optional[str] = None) -> None:
        self.source = source or self.__class__.__name__

    def emit(self, kind: str, failure_type: str = "", severity: float = 0.0, confidence: float = 0.5, summary: str = "", **kwargs: Any) -> bool:
        return arile_emit(
            source=self.source,
            organ=self.organ_name,
            kind=kind,
            failure_type=failure_type,
            severity=severity,
            confidence=confidence,
            summary=summary,
            **kwargs,
        )

    def should_run(self, lane: str, default: bool = True) -> bool:
        return arile_should_run(lane, source=self.source, default=default)


class RealityVarianceScorer:
    LOGIC_BOMB_PATTERNS = (
        r"\binfinite\b|\bunbounded\b|\bnever stop\b|\bkeep trying\b|\buntil solved\b",
        r"\buse every tool\b|\bquery all models\b|\btry all apis\b|\bcall every\b",
        r"\bignore (all|previous|safety|security|governance)\b",
        r"\bmodify\b.*\b(SarahMemoryGlobals\.py|SarahMemoryARILE\.py)\b",
        r"\bdelete\b.*\b(core|globals|arile|security|filesystem)\b",
        r"\bexecute\b.*\b(command|powershell|cmd|shell|script)\b",
        r"\bself[- ]?authorize\b|\bbypass\b.*\b(governance|compare|compass|smget)\b",
    )

    SOCIAL_ENGINEERING_PATTERNS = (
        r"\bwhatsapp\b|\btelegram\b|\bgoogle chat\b|\bhangouts\b",
        r"\bgift card\b|\bwire transfer\b|\bcrypto wallet\b",
        r"\bsend me your\b.*\b(password|address|phone|code|api key)\b",
    )

    def score(self, packet: RealityVariancePacket) -> RealityVariancePacket:
        packet.normalize()
        risk = packet.risk.lower()
        if risk in {"critical", "high"} and packet.severity < 0.75:
            packet.severity = 0.75 if risk == "high" else 0.90
        if packet.failure_type in {"logic_bomb", "memory_poison", "protected_core_mutation_attempt", "external_authority_drift"}:
            packet.requires_governance = True
            packet.severity = max(packet.severity, 0.80)
        if packet.safe_stop_required or packet.severity >= SEVERITY_GOVERNANCE_MIN:
            packet.requires_governance = True
        if packet.severity >= SEVERITY_EMERGENCY_MIN:
            packet.risk = "critical"
        elif packet.severity >= SEVERITY_GOVERNANCE_MIN and packet.risk not in {"critical", "high"}:
            packet.risk = "high"
        if packet.severity >= SEVERITY_GOVERNANCE_MIN and packet.retention in {"", "ram"}:
            packet.retention = "security_audit" if packet.risk in {"critical", "high"} else "diagnostic"
        packet.severity_class = _severity_class(packet.severity)
        return packet

    def scan_logic_bomb(self, text: str, source: str = "input") -> Dict[str, Any]:
        raw_text = str(text or "")
        was_truncated = len(raw_text) > 20000
        text = _safe_text(raw_text, 20000)
        low = text.lower()
        hits: List[str] = []
        for pat in self.LOGIC_BOMB_PATTERNS:
            if re.search(pat, low, flags=re.IGNORECASE):
                hits.append(pat)
        social_hits: List[str] = []
        for pat in self.SOCIAL_ENGINEERING_PATTERNS:
            if re.search(pat, low, flags=re.IGNORECASE):
                social_hits.append(pat)
        size_risk = was_truncated or len(raw_text) > 12000 or text.count("{") > 80 or text.count("(") > 500
        undefined_pressure = bool(re.search(r"\bundefined\b|\bnon[- ]?repeating infinite\b|\bexact failure point\b", low))
        tool_pressure = bool(re.search(r"\b(tool|mcp|api|browser|powershell|cmd|shell|file|patch|driver|servo|motor)\b", low))
        verdict = "SAFE_NORMAL"
        severity = 0.0
        if hits or size_risk or (undefined_pressure and tool_pressure):
            verdict = "DEFUSED"
            severity = 0.82
        elif social_hits:
            verdict = "SOCIAL_ENGINEERING"
            severity = 0.70
        return {
            "ok": verdict == "SAFE_NORMAL",
            "verdict": verdict,
            "severity": severity,
            "logic_hits": len(hits),
            "social_hits": len(social_hits),
            "size_risk": size_risk,
            "truncated": was_truncated,
            "undefined_pressure": undefined_pressure,
            "tool_pressure": tool_pressure,
            "prompt_hash": "sha256:" + hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest(),
            "source": source,
            "policy": {
                "max_response_tokens": 256,
                "timeout_seconds": 10,
                "max_retries": 0,
                "allow_tools": False,
                "allow_file_access": False,
                "allow_memory_write": False,
                "allow_api_calls": False,
                "allow_mcp_calls": False,
                "allow_physical_action": False,
            } if verdict != "SAFE_NORMAL" else {},
        }


class ARILEThrottle:
    def __init__(self) -> None:
        self._last_seen: Dict[str, Tuple[float, int]] = {}
        self._lock = threading.RLock()

    def allow(self, packet: RealityVariancePacket) -> Tuple[bool, int]:
        key = "|".join([
            packet.source.lower(),
            packet.kind.lower(),
            packet.failure_type.lower(),
            hashlib.sha256((packet.summary or "").encode("utf-8", errors="ignore")).hexdigest()[:12],
        ])
        now = _now()
        with self._lock:
            last, count = self._last_seen.get(key, (0.0, 0))
            if now - last < 3.0 and packet.severity < SEVERITY_EMERGENCY_MIN:
                count += 1
                self._last_seen[key] = (last, count)
                return False, count
            self._last_seen[key] = (now, 1)
            if len(self._last_seen) > 2048:
                # Bounded regardless of how quickly unique keys arrive.
                for stale_key, _entry in sorted(self._last_seen.items(), key=lambda kv: kv[1][0])[:len(self._last_seen) - 2048]:
                    self._last_seen.pop(stale_key, None)
            return True, 1


class ARILEWriteGate:
    """Bounded, serialized and failure-aware append-only evidence writer."""

    def __init__(self, min_severity: float = DEFAULT_MIN_WRITE_SEVERITY) -> None:
        self.min_severity = min_severity
        self._pending: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._last_flush = _now()
        self._last_urgent_flush = 0.0
        self.dropped = 0
        self.write_failures = 0
        self.persisted = 0
        self.last_error = ""

    def consider(self, packet: RealityVariancePacket, force: bool = False) -> None:
        packet.normalize()
        if not force and packet.severity < self.min_severity and packet.retention not in {"security_audit", "emergency_audit"}:
            return
        with self._lock:
            if len(self._pending) >= DEFAULT_MAX_PENDING_AUDIT:
                # Preserve a critical event ahead of lower-severity backlog when possible.
                if packet.severity >= SEVERITY_EMERGENCY_MIN:
                    lowest_index = min(range(len(self._pending)), key=lambda i: float(self._pending[i].get("severity", 0.0)))
                    if float(self._pending[lowest_index].get("severity", 0.0)) < packet.severity:
                        self._pending.pop(lowest_index)
                    else:
                        self.dropped += 1
                        self.last_error = "audit_pending_capacity_exhausted"
                        return
                else:
                    self.dropped += 1
                    self.last_error = "audit_pending_capacity_exhausted"
                    return
                self.dropped += 1  # A lower-priority event was displaced.
                self.last_error = "audit_pending_lower_severity_displaced"
            self._pending.append(packet.to_dict())

    def flush_urgent(self) -> int:
        """Emergency evidence is prompt but not one disk write per input packet."""
        now = _now()
        with self._lock:
            if now - self._last_urgent_flush < DEFAULT_URGENT_FLUSH_SECONDS:
                return 0
            self._last_urgent_flush = now
        return self.flush_due(force=True)

    def flush_due(self, force: bool = False) -> int:
        now = _now()
        with self._lock:
            if not force and now - self._last_flush < DEFAULT_BATCH_SECONDS:
                return 0
            if not self._pending:
                self._last_flush = now
                return 0
            rows = self._pending[:DEFAULT_MAX_WRITES_PER_BATCH]
            # Hold one lock across write and removal: another thread cannot cause loss.
            try:
                folder = _log_dir()
                folder.mkdir(parents=True, exist_ok=True)
                path = folder / "arile_events.jsonl"
                existing_bytes = path.stat().st_size if path.exists() else 0
                remaining_bytes = DEFAULT_AUDIT_MAX_BYTES - existing_bytes
                written_rows = 0
                pieces: List[str] = []
                for row in rows:
                    line = json.dumps(_safe_packet_data(row), ensure_ascii=False, sort_keys=True, default=str) + "\n"
                    size = len(line.encode("utf-8"))
                    if size > remaining_bytes:
                        break
                    pieces.append(line)
                    remaining_bytes -= size
                    written_rows += 1
                if not written_rows:
                    # Retention rotation is delegated to governed maintenance.
                    self.last_error = "audit_log_retention_limit_reached"
                    self.write_failures += 1
                    return 0
                payload = "".join(pieces)
                with path.open("a", encoding="utf-8") as f:
                    f.write(payload)
                    f.flush()
                    if str(os.getenv("SARAH_ARILE_AUDIT_FSYNC", "0")).lower() in {"1", "true", "yes"}:
                        os.fsync(f.fileno())
                del self._pending[:written_rows]
                self.persisted += written_rows
                self._last_flush = now
                self.last_error = ""
                return written_rows
            except Exception as exc:
                self.write_failures += 1
                self.last_error = _redact_text(type(exc).__name__ + ": " + str(exc), 240)
                self._last_flush = now  # back off during failing storage
                return 0


class ARILERuntime:
    def __init__(self) -> None:
        self.queue: "queue.Queue[RealityVariancePacket]" = queue.Queue(maxsize=DEFAULT_MAX_QUEUE)
        self.ring: Deque[Dict[str, Any]] = collections.deque(maxlen=DEFAULT_RING_SIZE)
        self.scorer = RealityVarianceScorer()
        self.throttle = ARILEThrottle()
        self.write_gate = ARILEWriteGate()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        self.started_at = ""
        self.dropped = 0
        self.suppressed = 0
        self.processed = 0
        self.health: Dict[str, Dict[str, Any]] = {}
        self._stopping = False

    def start(self, reason: str = "manual") -> Dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self.status()
            self._stop.clear()
            self._stopping = False
            self.started_at = _iso()
            self._thread = threading.Thread(target=self._loop, name="SarahMemoryARILE", daemon=True)
            self._thread.start()
        return self.status(extra={"reason": reason})

    def stop(self, reason: str = "shutdown") -> Dict[str, Any]:
        with self._lock:
            self._stopping = True
            self._stop.set()
            t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            try:
                t.join(timeout=2.0)
            except Exception:
                pass
        # Bounded shutdown flush. Pending evidence remains in memory if unavailable.
        for _ in range(4):
            if self.write_gate.flush_due(force=True) <= 0:
                break
        return self.status(extra={"reason": reason})

    def emit(self, packet: RealityVariancePacket) -> bool:
        packet = self.scorer.score(packet)
        with self._lock:
            if self._stopping:
                self.dropped += 1
                return False
        if packet.severity < 0.25 and packet.retention in {"", "ram"}:
            with self._lock:
                self.ring.append(packet.to_dict())
            return True
        allowed, suppressed = self.throttle.allow(packet)
        if not allowed:
            with self._lock:
                self.suppressed += 1
            return True
        try:
            self.queue.put_nowait(packet)
            return True
        except queue.Full:
            with self._lock:
                self.dropped += 1
            if packet.severity >= SEVERITY_EMERGENCY_MIN:
                try:
                    self.queue.get_nowait()
                    with self._lock:
                        self.dropped += 1  # displaced packet was not processed
                    self.queue.put_nowait(packet)
                    return True
                except (queue.Empty, queue.Full):
                    return False
            return False

    def _loop(self) -> None:
        while not self._stop.is_set() or not self.queue.empty():
            try:
                try:
                    packet = self.queue.get(timeout=0.5)
                except queue.Empty:
                    self.write_gate.flush_due(force=False)
                    continue
                self._process(packet)
            except Exception:
                # Never allow ARILE failure to crash SarahMemory runtime.
                try:
                    err = traceback.format_exc()[-1000:]
                    self.ring.append({"kind": "arile_internal_error", "summary": err, "timestamp": _iso()})
                except Exception:
                    pass
        try:
            self.write_gate.flush_due(force=True)
        except Exception:
            pass

    def _process(self, packet: RealityVariancePacket) -> None:
        packet = self.scorer.score(packet)
        row = packet.to_dict()
        with self._lock:
            self.ring.append(row)
            self.processed += 1
            src = packet.source or "unknown"
            if src not in self.health and len(self.health) >= DEFAULT_MAX_HEALTH_SOURCES:
                self.health.pop(next(iter(self.health)))
            self.health[src] = {
                "last_seen": packet.timestamp,
                "last_kind": packet.kind,
                "last_failure_type": packet.failure_type,
                "last_severity": packet.severity,
                "risk": packet.risk,
                "requires_governance": packet.requires_governance,
            }
        self.write_gate.consider(packet, force=packet.retention in {"security_audit", "emergency_audit"} or packet.severity >= SEVERITY_EMERGENCY_MIN)
        if packet.severity >= SEVERITY_EMERGENCY_MIN:
            self.write_gate.flush_urgent()

    def status(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            out = {
                "ok": not self._stopping and self.dropped == 0 and self.write_gate.dropped == 0 and self.write_gate.write_failures == 0,
                "module": "SarahMemoryARILE",
                "version": MODULE_VERSION,
                "started": bool(self._thread is not None and self._thread.is_alive()),
                "started_at": self.started_at,
                "queue_size": int(self.queue.qsize()),
                "ring_size": int(len(self.ring)),
                "processed": int(self.processed),
                "dropped": int(self.dropped),
                "suppressed": int(self.suppressed),
                "processing_active": bool(self._thread is not None and self._thread.is_alive() and not self._stop.is_set()),
                "standby_queue": bool(self.queue.qsize() and (self._thread is None or not self._thread.is_alive())),
                "audit_pending": len(self.write_gate._pending),
                "audit_persisted": int(self.write_gate.persisted),
                "audit_dropped": int(self.write_gate.dropped),
                "audit_write_failures": int(self.write_gate.write_failures),
                "audit_error": self.write_gate.last_error,
                "protected_files": arile_protected_files(),
                "health_sources": len(self.health),
            }
        if extra:
            out.update(extra)
        return out

    def snapshot(self, limit: int = 50) -> Dict[str, Any]:
        with self._lock:
            return {
                **self.status(),
                "recent": list(self.ring)[-max(1, min(200, int(limit))) :],
                "health": dict(self.health),
            }


_RUNTIME = ARILERuntime()
_ATEXIT_REGISTERED = False


def start_arile_runtime(reason: str = "runtime_start") -> Dict[str, Any]:
    global _ATEXIT_REGISTERED
    status = _RUNTIME.start(reason=reason)
    if not _ATEXIT_REGISTERED:
        try:
            atexit.register(stop_arile_runtime, "atexit")
            _ATEXIT_REGISTERED = True
        except Exception:
            pass
    return status


def stop_arile_runtime(reason: str = "shutdown") -> Dict[str, Any]:
    return _RUNTIME.stop(reason=reason)


def get_arile_runtime_status() -> Dict[str, Any]:
    return _RUNTIME.status()


def arile_snapshot(limit: int = 50) -> Dict[str, Any]:
    return _RUNTIME.snapshot(limit=limit)


def arile_emit(source: str, kind: str, failure_type: str = "", severity: float = 0.0, novelty: float = 0.0, confidence: float = 0.5, risk: str = "low", summary: str = "", requires_governance: bool = False, retention: str = "ram", organ: str = "", expected_state: str = "", observed_state: str = "", recommended_response: str = "", data: Optional[Dict[str, Any]] = None, fault_class: str = "", severity_class: str = "", residual: Optional[float] = None, normalized_residual: Optional[float] = None, rate_of_change: Optional[float] = None, duration_seconds: float = 0.0, witness_timestamp: str = "", freshness: Optional[float] = None, controller: str = "", contract_id: str = "", safe_stop_required: bool = False, cleared: bool = False) -> bool:
    try:
        packet = RealityVariancePacket(
            source=source,
            organ=organ,
            kind=kind,
            failure_type=failure_type,
            severity=severity,
            novelty=novelty,
            confidence=confidence,
            risk=risk,
            summary=summary,
            requires_governance=requires_governance,
            retention=retention,
            expected_state=expected_state,
            observed_state=observed_state,
            recommended_response=recommended_response,
            fault_class=fault_class,
            severity_class=severity_class,
            residual=residual,
            normalized_residual=normalized_residual,
            rate_of_change=rate_of_change,
            duration_seconds=duration_seconds,
            witness_timestamp=witness_timestamp,
            freshness=freshness,
            controller=controller,
            contract_id=contract_id,
            safe_stop_required=safe_stop_required,
            cleared=cleared,
            data=data or {},
        ).normalize()
        return _RUNTIME.emit(packet)
    except Exception:
        return False


def arile_observe_variance(
    *,
    source: str,
    metric: str,
    expected: float,
    observed: float,
    tolerance: float,
    unit: str = "",
    duration_seconds: float = 0.0,
    rate_of_change: Optional[float] = None,
    freshness: Optional[float] = None,
    controller: str = "",
    contract_id: str = "",
    emit: bool = True,
) -> Dict[str, Any]:
    """Convert expected-versus-observed telemetry into one ARILE fault packet.

    This is an observation helper only.  It never performs the recommended
    response or grants execution authority.
    """
    try:
        values = [float(expected), float(observed), float(tolerance)]
        if not all(math.isfinite(v) for v in values) or values[2] <= 0.0:
            return {"ok": False, "error": "invalid_or_nonfinite_telemetry", "packet": None, "emitted": False, "execution_authority": False}
    except (ValueError, TypeError, OverflowError):
        return {"ok": False, "error": "invalid_telemetry", "packet": None, "emitted": False, "execution_authority": False}
    try:
        from SarahMemoryLogicCalc import operational_severity, telemetry_residual  # type: ignore

        comparison = telemetry_residual(observed, expected, tolerance)
        normalized = comparison.get("normalized_residual")
        normalized_f = max(0.0, float(normalized)) if normalized is not None else 0.0
    except Exception:
        tol = max(1e-12, abs(float(tolerance)))
        residual_value = float(observed) - float(expected)
        normalized_f = abs(residual_value) / tol
        comparison = {
            "actual": float(observed),
            "target": float(expected),
            "tolerance": tol,
            "residual": residual_value,
            "absolute_residual": abs(residual_value),
            "normalized_residual": normalized_f,
            "within_tolerance": abs(residual_value) <= tol,
        }

        def operational_severity(value: float) -> str:
            return "EMERGENCY" if value >= 0.90 else "FAILURE" if value >= 0.75 else "ERROR" if value >= 0.50 else "ALERT" if value >= 0.25 else "INFO"

    severity = max(0.0, min(1.0, normalized_f / 2.0))
    if bool(comparison.get("within_tolerance")):
        severity = 0.0
    severity_class = operational_severity(severity)
    safe_stop_required = severity_class in {"FAILURE", "EMERGENCY"}
    packet = RealityVariancePacket(
        source=source,
        organ="SarahMemoryARILE",
        kind="telemetry_variance",
        failure_type="setpoint_variance" if severity > 0.0 else "within_tolerance",
        severity=severity,
        confidence=0.90,
        risk="critical" if severity_class == "EMERGENCY" else "high" if safe_stop_required else "medium" if severity > 0.0 else "low",
        summary=f"{metric} observed against expected setpoint.",
        expected_state=f"{expected} {unit}".strip(),
        observed_state=f"{observed} {unit}".strip(),
        recommended_response="Request governed safe-stop review." if safe_stop_required else "Continue bounded observation." if severity > 0.0 else "No response required.",
        fault_class="TELEMETRY_DEVIATION" if severity > 0.0 else "NONE",
        severity_class=severity_class,
        residual=comparison.get("residual"),
        normalized_residual=comparison.get("normalized_residual"),
        rate_of_change=rate_of_change,
        duration_seconds=duration_seconds,
        witness_timestamp=_iso(),
        freshness=freshness,
        controller=controller,
        contract_id=contract_id,
        safe_stop_required=safe_stop_required,
        requires_governance=safe_stop_required or severity >= SEVERITY_GOVERNANCE_MIN,
        cleared=bool(severity == 0.0),
        data={"metric": str(metric), "unit": str(unit), "comparison": comparison, "execution_authority": False},
    ).normalize()
    emitted = bool(_RUNTIME.emit(packet)) if emit else False
    return {"ok": True, "packet": packet.to_dict(), "emitted": emitted, "execution_authority": False}


def arile_should_run(lane: str, source: str = "unknown", default: bool = True) -> bool:
    """Fast cadence gate.  Returns False only under clear overload conditions."""
    try:
        q = _RUNTIME.queue.qsize()
        lane_l = str(lane or "").lower()
        if q > DEFAULT_MAX_QUEUE * 0.85 and lane_l in {"rem", "creative", "background", "low_priority", "indexing"}:
            arile_emit(source=source, kind="runtime_backpressure", failure_type="lane_deferred", severity=0.55, confidence=0.90, summary=f"Deferred lane under ARILE queue pressure: {lane}")
            return False
        return bool(default)
    except Exception:
        return bool(default)


def arile_endpoint_guard(endpoint_name: str, request_meta: Optional[Dict[str, Any]] = None, risk: str = "low") -> str:
    """Advisory API boundary triage. Returns allow/defer/block, never authority."""
    try:
        meta = dict(request_meta or {})
        raw_length = meta.get("content_length", 0)
        content_length = int(raw_length or 0)
        if content_length < 0:
            raise ValueError("negative_content_length")
        maximum = _env_int("SARAH_ARILE_MAX_API_PAYLOAD", 10485760, 1024, 268435456)
        method = str(meta.get("method") or "").upper()
        endpoint = _safe_text(endpoint_name, 240)
    except Exception:
        return "block"  # Invalid input must not throw into fail-open callers.
    if content_length > maximum:
        arile_emit(
            source="api_boundary", kind="api_boundary_variance", failure_type="oversized_payload",
            severity=0.78, confidence=0.93, risk="high",
            summary=f"Oversized API payload blocked at {endpoint}: {content_length} bytes",
            requires_governance=True, retention="security_audit", data=meta,
        )
        return "block"
    if method not in {"", "GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"}:
        arile_emit(source="api_boundary", kind="api_boundary_variance", failure_type="unexpected_method", severity=0.55, confidence=0.80, summary=f"Unexpected API method at {endpoint}: {method}", data=meta)
        return "defer"
    if not arile_should_run("api", source=endpoint, default=True):
        return "defer"
    return "allow"


def arile_scan_logic_bomb(text: str, source: str = "input", emit: bool = True) -> Dict[str, Any]:
    verdict = _RUNTIME.scorer.scan_logic_bomb(text or "", source=source)
    if emit and not verdict.get("ok"):
        arile_emit(
            source=source,
            kind="logic_bomb_variance",
            failure_type=str(verdict.get("verdict") or "logic_bomb").lower(),
            severity=float(verdict.get("severity") or 0.80),
            confidence=0.88,
            risk="high",
            summary="Logic-bomb-shaped cognitive payload defused before tool/action expansion.",
            requires_governance=True,
            retention="security_audit",
            data={k: v for k, v in verdict.items() if k != "policy"},
        )
    return verdict


def arile_defuse_logic_bomb_response(verdict: Optional[Dict[str, Any]] = None) -> str:
    return (
        "This input is not executable as stated. It contains unbounded, undefined, "
        "or tool/action-expanding structure. Provide defined variables, exact scope, "
        "authorization, limits, and the allowed output size before it can be evaluated."
    )


class ARILEPatchBridge:
    """Controlled overlay bridge.  Direct ARILE source mutation remains blocked."""

    blocked_targets = {
        "emit", "arile_emit", "start_arile_runtime", "stop_arile_runtime", "arile_assert_not_protected_mutation",
        "arile_is_protected_core_file", "arile_endpoint_guard", "arile_scan_logic_bomb",
        "ARILERuntime", "ARILEWriteGate", "ARILEThrottle", "ARILEPatchBridge",
    }

    @classmethod
    def validate_overlay_manifest(cls, manifest: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(manifest, dict):
            return {"ok": False, "errors": ["manifest_must_be_mapping"], "targets": [], "policy": "monkeypatch_overlay_only", "execution_authority": False, "requires_external_approval": True}
        raw_targets = manifest.get("targets")
        errors: List[str] = []
        if raw_targets is None:
            raw_targets = []
        if not isinstance(raw_targets, (list, tuple)):
            errors.append("targets_must_be_list")
            raw_targets = []
        targets = [str(x or "")[:257] for x in raw_targets[:65]]
        touches_security = manifest.get("touches_security") is True
        rollback = manifest.get("rollback_available") is True
        if not targets:
            errors.append("targets_required")
        if len(targets) > 64:
            errors.append("too_many_targets")
        for t in targets[:64]:
            if not t.strip() or len(t) > 256:
                errors.append("invalid_overlay_target")
            root_symbol = t.replace("::", ".").split(".", 1)[0].lower()
            if root_symbol in {x.lower() for x in cls.blocked_targets}:
                errors.append(f"protected_arile_symbol:{t}")
        if touches_security:
            errors.append("security_touch_requires_manual_source_patch_not_overlay")
        if not rollback:
            errors.append("rollback_required")
        return {"ok": not errors, "errors": errors, "targets": targets, "policy": "monkeypatch_overlay_only", "execution_authority": False, "requires_external_approval": True}


def verify_arile_source_integrity() -> Dict[str, Any]:
    """Best-effort source hash snapshot; does not fail boot by itself."""
    try:
        path = Path(__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"ok": True, "file": str(path), "sha256": digest, "protected": True, "verified_against_trusted_baseline": False, "integrity_state": "HASH_ONLY_UNVERIFIED", "policy": "monkeypatch_overlay_only"}
    except Exception as e:
        return {"ok": False, "error": str(e), "protected": True, "policy": "monkeypatch_overlay_only"}


# Public aliases expected by integration files.
AdaptiveRealityIntelligenceLayerEngine = ARILERuntime
RealityVarianceScorer = RealityVarianceScorer
ARILELogicBombFirewall = RealityVarianceScorer

if os.getenv("SARAH_ARILE_AUTOSTART_ON_IMPORT", "0").strip().lower() in {"1", "true", "yes", "on"}:
    start_arile_runtime(reason="import_autostart")

# ====================================================================
# END OF SarahMemoryARILE.py v9.1.0a
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryARILE',
    "version": "v9.1.0a-arile-enterprise-sml-0.2",
    "category": 'Unknown',
    "protocol_version": "SML/1.0",
    "packet_version": 1,
    "omega_registry_version": "Ω/1.0",
    "capabilities": [],
    "supported_missions": ['Conversation'],
    "supported_omega": ['Ω001'],
    "required_authority": ['Read'],
    "priority": 40,
    "trust_level": "source_integrated",
    "internal_only": True,
    "metadata": {"sml_adapter": "generic_non_executing", "source_file": 'SarahMemoryARILE.py'},
}


def sml_get_metadata():
    """Return this organ's SML registration metadata."""
    return dict(SML_ORGAN_METADATA)


def sml_health():
    """Numerically compatible, evidence-based operational health snapshot."""
    status = get_arile_runtime_status()
    running = bool(status.get("processing_active"))
    errors = int(status.get("dropped", 0)) + int(status.get("audit_dropped", 0)) + int(status.get("audit_write_failures", 0))
    availability = 1.0 if running else 0.0
    reliability = (max(0.0, 1.0 - errors / max(1, int(status.get("processed", 0)) + errors))) if running else 0.0
    return {
        "status": "Healthy" if running and errors == 0 else "Degraded" if running else "Standby",
        "availability": availability,
        "integrity": 0.0,  # No trusted signed-baseline comparison performed here.
        "performance": 0.0,  # No latency benchmark is taken by this adapter.
        "reliability": reliability,
        "confidence": 0.60 if running else 0.25,
        "latency_ms": 0.0,  # Unmeasured; not a claimed zero-latency result.
        "stability": reliability,
        "compatibility": 0.0,  # Packet adapter present, end-to-end not independently verified.
        "notes": ["SML adapter present", "integrity and latency not independently measured", "execution authority: none"],
    }


def sml_diagnostics():
    """Return SML adapter diagnostics without executing organ behavior."""
    return {
        "status": "OK",
        "component": 'SarahMemoryARILE',
        "sml_adapter": True,
        "metadata": dict(SML_ORGAN_METADATA),
        "health": sml_health(),
    }


def sml_receive_packet(packet, *, action="observe", note="", updates=None):
    """Receive/update an SML packet through the canonical protocol without direct execution."""
    try:
        from SarahMemorySMLProtocol import register_sml_organ, sml_touch_packet
        register_sml_organ(SML_ORGAN_METADATA)
        return sml_touch_packet(packet, organ='SarahMemoryARILE', action=action, note=note or "organ observed packet", updates=updates)
    except Exception:
        return packet
# --- SML ORGAN ADAPTER END ---
