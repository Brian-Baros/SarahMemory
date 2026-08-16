"""--==The SarahMemory Project==--
File: SarahMemoryARILE.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-06-09
Time: 10:11:54
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.

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
import json
import os
import queue
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

MODULE_VERSION = "9.0.0-arile-v1"
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

DEFAULT_MAX_QUEUE = int(os.getenv("SARAH_ARILE_MAX_QUEUE", "2048") or 2048)
DEFAULT_RING_SIZE = int(os.getenv("SARAH_ARILE_RING_SIZE", "512") or 512)
DEFAULT_BATCH_SECONDS = float(os.getenv("SARAH_ARILE_BATCH_SECONDS", "30") or 30)
DEFAULT_MIN_WRITE_SEVERITY = float(os.getenv("SARAH_ARILE_MIN_WRITE_SEVERITY", "0.75") or 0.75)
DEFAULT_MAX_WRITES_PER_BATCH = int(os.getenv("SARAH_ARILE_MAX_WRITES_PER_BATCH", "50") or 50)


def _now() -> float:
    return time.time()


def _iso(ts: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(_now() if ts is None else ts))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
        if v != v:  # NaN
            return default
        return max(0.0, min(1.0, v))
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


def arile_is_protected_core_file(path_value: Any) -> bool:
    """Return True if a target resolves to a protected SarahMemory core file."""
    try:
        name = Path(os.fspath(path_value)).name.lower()
    except Exception:
        name = str(path_value or "").replace("\\", "/").split("/")[-1].lower()
    return name in ARILE_PROTECTED_CORE_FILES


def arile_is_tier1_security_file(path_value: Any) -> bool:
    try:
        name = Path(os.fspath(path_value)).name.lower()
    except Exception:
        name = str(path_value or "").replace("\\", "/").split("/")[-1].lower()
    return name in ARILE_TIER1_SECURITY_FILES


def arile_assert_not_protected_mutation(path_value: Any, operation: str = "mutation", source: str = "unknown") -> None:
    """Raise PermissionError if a protected core file is targeted for mutation."""
    if arile_is_protected_core_file(path_value):
        name = Path(os.fspath(path_value)).name if path_value is not None else str(path_value)
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
    data: Optional[Dict[str, Any]] = None
    timestamp: str = dataclasses.field(default_factory=_iso)
    packet_id: str = ""

    def normalize(self) -> "RealityVariancePacket":
        self.source = _safe_text(self.source, 160) or "unknown"
        self.kind = _safe_text(self.kind, 120) or "variance"
        self.failure_type = _safe_text(self.failure_type, 160)
        self.severity = _safe_float(self.severity, 0.0)
        self.novelty = _safe_float(self.novelty, 0.0)
        self.confidence = _safe_float(self.confidence, 0.5)
        self.risk = _safe_text(self.risk, 40) or "low"
        self.summary = _safe_text(self.summary, 1000)
        self.organ = _safe_text(self.organ, 80)
        self.expected_state = _safe_text(self.expected_state, 500)
        self.observed_state = _safe_text(self.observed_state, 500)
        self.recommended_response = _safe_text(self.recommended_response, 500)
        self.retention = _safe_text(self.retention, 80) or "ram"
        if not isinstance(self.data, dict):
            self.data = {}
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
        if packet.severity >= SEVERITY_GOVERNANCE_MIN:
            packet.requires_governance = True
            if packet.retention in {"", "ram"}:
                packet.retention = "security_audit" if risk in {"critical", "high"} else "diagnostic"
        return packet

    def scan_logic_bomb(self, text: str, source: str = "input") -> Dict[str, Any]:
        text = _safe_text(text, 20000)
        low = text.lower()
        hits: List[str] = []
        for pat in self.LOGIC_BOMB_PATTERNS:
            if re.search(pat, low, flags=re.IGNORECASE):
                hits.append(pat)
        social_hits: List[str] = []
        for pat in self.SOCIAL_ENGINEERING_PATTERNS:
            if re.search(pat, low, flags=re.IGNORECASE):
                social_hits.append(pat)
        size_risk = len(text) > 12000 or text.count("{") > 80 or text.count("(") > 500
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
            if len(self._last_seen) > 5000:
                stale = [k for k, (ts, _c) in self._last_seen.items() if now - ts > 600]
                for k in stale[:1000]:
                    self._last_seen.pop(k, None)
            return True, 1


class ARILEWriteGate:
    def __init__(self, min_severity: float = DEFAULT_MIN_WRITE_SEVERITY) -> None:
        self.min_severity = min_severity
        self._pending: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        self._last_flush = _now()

    def consider(self, packet: RealityVariancePacket, force: bool = False) -> None:
        if not force and packet.severity < self.min_severity and packet.retention not in {"security_audit", "emergency_audit"}:
            return
        with self._lock:
            self._pending.append(packet.to_dict())

    def flush_due(self, force: bool = False) -> int:
        now = _now()
        with self._lock:
            if not force and now - self._last_flush < DEFAULT_BATCH_SECONDS:
                return 0
            rows = self._pending[:DEFAULT_MAX_WRITES_PER_BATCH]
            self._pending = self._pending[DEFAULT_MAX_WRITES_PER_BATCH:]
            self._last_flush = now
        if not rows:
            return 0
        try:
            folder = _log_dir()
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / "arile_events.jsonl"
            with path.open("a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
            return len(rows)
        except Exception:
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
        self.processed = 0
        self.health: Dict[str, Dict[str, Any]] = {}

    def start(self, reason: str = "manual") -> Dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self.status()
            self._stop.clear()
            self.started_at = _iso()
            self._thread = threading.Thread(target=self._loop, name="SarahMemoryARILE", daemon=True)
            self._thread.start()
        return self.status(extra={"reason": reason})

    def stop(self, reason: str = "shutdown") -> Dict[str, Any]:
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive():
            try:
                t.join(timeout=2.0)
            except Exception:
                pass
        try:
            self.write_gate.flush_due(force=True)
        except Exception:
            pass
        return self.status(extra={"reason": reason})

    def emit(self, packet: RealityVariancePacket) -> bool:
        packet = self.scorer.score(packet)
        if packet.severity < 0.25 and packet.retention in {"", "ram"}:
            # Low signal remains live-state only and is not queued.
            with self._lock:
                self.ring.append(packet.to_dict())
            return True
        allowed, suppressed = self.throttle.allow(packet)
        if not allowed:
            return True
        try:
            self.queue.put_nowait(packet)
            return True
        except queue.Full:
            self.dropped += 1
            if packet.severity >= SEVERITY_EMERGENCY_MIN:
                try:
                    _ = self.queue.get_nowait()
                except Exception:
                    pass
                try:
                    self.queue.put_nowait(packet)
                    return True
                except Exception:
                    return False
            return False

    def _loop(self) -> None:
        while not self._stop.is_set():
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
            self.write_gate.flush_due(force=True)

    def status(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            out = {
                "ok": True,
                "module": "SarahMemoryARILE",
                "version": MODULE_VERSION,
                "started": bool(self._thread is not None and self._thread.is_alive()),
                "started_at": self.started_at,
                "queue_size": int(self.queue.qsize()),
                "ring_size": int(len(self.ring)),
                "processed": int(self.processed),
                "dropped": int(self.dropped),
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


def arile_emit(source: str, kind: str, failure_type: str = "", severity: float = 0.0, novelty: float = 0.0, confidence: float = 0.5, risk: str = "low", summary: str = "", requires_governance: bool = False, retention: str = "ram", organ: str = "", expected_state: str = "", observed_state: str = "", recommended_response: str = "", data: Optional[Dict[str, Any]] = None) -> bool:
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
            data=data or {},
        ).normalize()
        return _RUNTIME.emit(packet)
    except Exception:
        return False


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
    """Lightweight API boundary guard. Returns allow/defer/block."""
    meta = dict(request_meta or {})
    content_length = int(meta.get("content_length") or 0)
    method = str(meta.get("method") or "").upper()
    endpoint = str(endpoint_name or "")[:240]
    if content_length > int(os.getenv("SARAH_ARILE_MAX_API_PAYLOAD", "10485760") or 10485760):
        arile_emit(
            source="api_boundary",
            kind="api_boundary_variance",
            failure_type="oversized_payload",
            severity=0.78,
            confidence=0.93,
            risk="high",
            summary=f"Oversized API payload blocked at {endpoint}: {content_length} bytes",
            requires_governance=True,
            retention="security_audit",
            data=meta,
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
        manifest = manifest or {}
        targets = [str(x or "") for x in (manifest.get("targets") or [])]
        touches_security = bool(manifest.get("touches_security"))
        rollback = bool(manifest.get("rollback_available"))
        errors: List[str] = []
        for t in targets:
            if t in cls.blocked_targets or t.lower() in {x.lower() for x in cls.blocked_targets}:
                errors.append(f"protected_arile_symbol:{t}")
        if touches_security:
            errors.append("security_touch_requires_manual_source_patch_not_overlay")
        if not rollback:
            errors.append("rollback_required")
        return {"ok": not errors, "errors": errors, "targets": targets, "policy": "monkeypatch_overlay_only"}


def verify_arile_source_integrity() -> Dict[str, Any]:
    """Best-effort source hash snapshot; does not fail boot by itself."""
    try:
        path = Path(__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"ok": True, "file": str(path), "sha256": digest, "protected": True, "policy": "monkeypatch_overlay_only"}
    except Exception as e:
        return {"ok": False, "error": str(e), "protected": True, "policy": "monkeypatch_overlay_only"}


# Public aliases expected by integration files.
AdaptiveRealityIntelligenceLayerEngine = ARILERuntime
RealityVarianceScorer = RealityVarianceScorer
ARILELogicBombFirewall = RealityVarianceScorer

if os.getenv("SARAH_ARILE_AUTOSTART_ON_IMPORT", "0").strip().lower() in {"1", "true", "yes", "on"}:
    start_arile_runtime(reason="import_autostart")

# ====================================================================
# END OF SarahMemoryARILE.py v9.0.0
# ====================================================================

# --- SML ORGAN ADAPTER START ---
# Added by SarahMemory SML glue patch v0.2-alpha. Non-executing protocol adapter.
SML_ORGAN_METADATA = {
    "name": 'SarahMemoryARILE',
    "version": "v9.0.0-alpha-sml-0.2",
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

