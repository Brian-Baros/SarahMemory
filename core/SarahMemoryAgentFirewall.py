"""--==The SarahMemory Project==--
File: SarahMemoryAgentFirewall.py
Part of the SarahMemory AiOS Governed Cognitive Runtime
Version: v9.0.0
Date: 2026-07-11
Time: 10:11:54
Author: © 2025, 2026 Brian Lee Baros. All Rights Reserved.

===============================================================================
Agent Firewall / Anti-Hijack Overlay

SARAHMEMORY_PATCH_NOTE 2026-06-23:
This file adds a small deterministic security organ for boot/API/sync surfaces.
It does not replace SafetyPolicies, SecurityGovernor, AssuranceGate, Compare,
Compass, SMGET, OperatorCore, or user authority. It provides a required first-pass
filter so external agents, remote prompts, malformed broker messages, copied web
payloads, or API requests cannot directly trigger override language, core writes,
autonomous startup, shell execution, or governance bypass attempts.
===============================================================================
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

try:
    import SarahMemoryGlobals as config
except Exception:  # pragma: no cover - fallback for isolated tests
    config = None  # type: ignore

try:
    from SarahMemoryAudit import audit_event
except Exception:
    audit_event = None  # type: ignore


# SARAHMEMORY_PATCH_NOTE 2026-06-23:
# Keep this phrase list intentionally conservative and explainable. It blocks
# common hijack/override intent, not normal user conversation. Final authority
# still belongs to the caller/governance lane; this organ supplies evidence.
HIJACK_PATTERNS = (
    "ignore governance",
    "bypass governance",
    "disable governance",
    "override governance",
    "self-authorize",
    "self authorize",
    "execute without approval",
    "apply without approval",
    "disable safety",
    "turn off safety",
    "disable audit",
    "delete audit",
    "erase audit",
    "disable logging",
    "overwrite core",
    "replace core",
    "apply core patch",
    "remote shell",
    "run shell",
    "powershell -",
    "cmd.exe",
    "subprocess.popen",
    "os.system",
    "eval(",
    "exec(",
    "ignore previous instructions",
    "jailbreak",
    "developer mode override",
    "system prompt override",
    "ai agent authority",
    "agent swarm",
    "autonomous override",
    "untrusted remote execution",
)

REMOTE_WRITE_PATTERNS = (
    "core_sync::",
    "SarahMemoryGlobals.py",
    "SarahMemoryMain.py",
    "SarahMemorySafetyPolicies.py",
    "SarahMemorySecurityGovernor.py",
    "SarahMemoryAssuranceGate.py",
    "SarahMemoryOperatorCore.py",
    "SarahMemoryMSDC.py",
    "SarahMemoryEnergetics.py",
    "SarahMemoryEvolution.py",
)


# SARAHMEMORY_PATCH_NOTE 2026-06-28:
# RoachMotel agent containment has only two inbound trust bands:
# LOW  = SarahMemory can match a return signature to an outbound agent it sent.
# HIGH = no matching outbound signature exists, or protected/hazardous-energy
#        resources are targeted.  There is intentionally no MEDIUM agent risk.
AI_AGENT_MARKERS = (
    "ai-agent", "ai agent", "agentic", "autonomous agent", "tool agent",
    "mcp", "langchain", "autogpt", "crew", "swarm", "browser agent",
    "crawl", "scrape", "mine", "exfiltrate", "harvest", "vector dump",
)
SENSITIVE_TARGET_PATTERNS = (
    "data/memory", "memory\\datasets", "ai_learning.db", "system_index.db",
    "context_history.db", "user_profile.db", ".env", "credential", "secret",
    "token", "private key", "SarahMemoryEnergetics.py", "ENERGETICS",
)
ROACHMOTEL_SCHEMA = "SARAHMEMORY_AI_AGENT_ROACHMOTEL_V1"


def _base_dir() -> str:
    """Resolve the installed SarahMemory root without trusting process cwd.

    SarahMemory may be installed at C:/SarahMemory, D:/SarahMemory, S:/SarahMemory,
    F:/SMv9, or any folder containing the core/api/data/resources layout.
    """
    try:
        base = getattr(config, "BASE_DIR", None) if config is not None else None
        if base:
            return str(Path(str(base)).expanduser().resolve())
    except Exception:
        pass
    try:
        here = Path(__file__).resolve()
        if here.parent.name.lower() == "core":
            return str(here.parent.parent.resolve())
        for parent in here.parents:
            if (parent / "core").is_dir() and ((parent / "api").is_dir() or (parent / "data").is_dir() or (parent / "resources").is_dir()):
                return str(parent.resolve())
    except Exception:
        pass
    return str(Path.cwd().resolve())


def _data_dir() -> str:
    try:
        return str(getattr(config, "DATA_DIR", os.path.join(_base_dir(), "data")) if config is not None else os.path.join(_base_dir(), "data"))
    except Exception:
        return os.path.join(_base_dir(), "data")


def _roach_dirs() -> Dict[str, str]:
    root = os.path.join(_data_dir(), "devbridge", "agent_firewall")
    audit = os.path.join(_data_dir(), "audit", "ai_agent_firewall")
    paths = {
        "root": root,
        "inbound": os.path.join(root, "inbound"),
        "quarantine": os.path.join(root, "quarantine"),
        "sandbox": os.path.join(root, "sandbox"),
        "dissected": os.path.join(root, "dissected"),
        "blocked": os.path.join(root, "blocked"),
        "released_by_user": os.path.join(root, "released_by_user"),
        "workorders": os.path.join(root, "workorders"),
        "audit": audit,
        "reports": os.path.join(audit, "reports"),
        "evidence": os.path.join(audit, "evidence"),
        "source_fingerprints": os.path.join(audit, "source_fingerprints"),
        "release_decisions": os.path.join(audit, "release_decisions"),
    }
    for path in paths.values():
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass
    return paths


def _registry_path() -> str:
    return os.path.join(_roach_dirs()["root"], "outbound_agent_registry.json")


def _safe_json_write(path: str, payload: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        os.replace(tmp, path)
    except Exception:
        pass


def _load_registry() -> Dict[str, Any]:
    path = _registry_path()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                return data if isinstance(data, dict) else {"schema": ROACHMOTEL_SCHEMA, "agents": {}}
    except Exception:
        pass
    return {"schema": ROACHMOTEL_SCHEMA, "agents": {}}


def _trust_registry_module() -> Tuple[Optional[Any], str]:
    try:
        import SarahMemoryTrustRegistry as registry  # type: ignore
        return registry, ""
    except Exception as exc:
        return None, str(exc)


def _ledger_receipt(
    event_type: str,
    *,
    verdict: str,
    identity: Optional[Dict[str, Any]] = None,
    source: str = "agent_firewall",
    reason: str = "",
    risk: str = "medium",
    payload_hash: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Write one compact receipt. The Ledger records evidence; it never authorizes."""
    try:
        if config is not None and not bool(getattr(config, "SARAH_LEDGER_RECEIPTS_ENABLED", True)):
            return
        from SarahMemoryLedger import record_governance_receipt  # type: ignore
        ident = identity or {}
        record_governance_receipt(
            "agent_firewall",
            event_type,
            subject_id=str(ident.get("agent_id") or ident.get("claimed_identity") or "unknown_agent")[:180],
            task_id=str(ident.get("task_id") or "")[:180],
            lane=str(ident.get("requested_lane") or "agent_firewall")[:96],
            verdict=str(verdict or "UNKNOWN")[:64],
            risk=str(risk or "medium")[:32],
            retention_class="agent_security" if str(verdict).upper() != "ALLOW" else "agent_observation",
            payload_hash=str(payload_hash or ident.get("payload_hash") or "")[:128],
            summary=str(reason or event_type)[:1000],
            metadata={
                "source": str(source or "agent_firewall")[:180],
                "passport_id": str(ident.get("passport_id") or "")[:180],
                "containment_only": True,
                "execution_authority": False,
                **(metadata or {}),
            },
        )
    except Exception:
        pass


def issue_outbound_agent_passport(
    agent_id: str,
    purpose: str,
    *,
    task_id: str = "",
    agent_name: str = "",
    origin_lane: str = "agent",
    allowed_lanes: Optional[List[str]] = None,
    allowed_capabilities: Optional[List[str]] = None,
    allowed_resources: Optional[List[str]] = None,
    denied_resources: Optional[List[str]] = None,
    maximum_risk_tier: str = "low",
    ttl_seconds: Optional[int] = None,
    one_time_use: Optional[bool] = None,
    network_allowed: bool = True,
    filesystem_allowed: bool = False,
    shell_allowed: bool = False,
    device_allowed: bool = False,
    memory_allowed: bool = False,
    user_approved: bool = False,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Issue a scoped passport without launching an agent or granting execution."""
    registry, error = _trust_registry_module()
    if registry is None or not callable(getattr(registry, "issue_agent_passport", None)):
        return {
            "ok": False,
            "error": "trust_registry_unavailable",
            "detail": error,
            "execution_authority": False,
        }
    result = registry.issue_agent_passport(
        agent_id=agent_id,
        agent_name=agent_name,
        purpose=purpose,
        task_id=task_id,
        origin="local",
        origin_lane=origin_lane,
        allowed_lanes=allowed_lanes or [origin_lane],
        allowed_capabilities=allowed_capabilities or ["inspect", "research", "return_data"],
        allowed_resources=allowed_resources or [],
        denied_resources=denied_resources or ["core/*", ".env", "credentials", "shell", "device_control"],
        maximum_risk_tier=maximum_risk_tier,
        ttl_seconds=ttl_seconds,
        one_time_use=one_time_use,
        network_allowed=network_allowed,
        filesystem_allowed=filesystem_allowed,
        shell_allowed=shell_allowed,
        device_allowed=device_allowed,
        memory_allowed=memory_allowed,
        requires_user_review=True,
        requires_assurance=True,
        requires_compare=True,
        requires_compass=True,
        user_approved=user_approved,
        metadata={"issued_via": "SarahMemoryAgentFirewall", **(meta or {})},
    )
    if result.get("ok"):
        creds = result.get("departure_credentials") if isinstance(result.get("departure_credentials"), dict) else {}
        passport = result.get("passport") if isinstance(result.get("passport"), dict) else {}
        _ledger_receipt(
            "PASSPORT_ISSUED",
            verdict="ISSUED",
            identity={"agent_id": agent_id, "passport_id": passport.get("passport_id"), "task_id": task_id, "requested_lane": origin_lane},
            reason="Governed outbound AI-agent passport issued.",
            risk=maximum_risk_tier,
            metadata={"expires_ts": passport.get("expires_ts"), "one_time_use": passport.get("one_time_use")},
        )
        # Compatibility aliases for older local callers. These are credentials,
        # not authority, and are shown only in the issuance response.
        result["agent_id"] = agent_id
        result["passport_id"] = creds.get("passport_id")
        result["allowed_return_signature"] = creds.get("return_signature")
        result["return_nonce"] = creds.get("return_nonce")
        result["departure_nonce"] = creds.get("departure_nonce")
    return result


def register_outbound_agent(
    agent_id: str,
    purpose: str = "",
    allowed_return_signature: str = "",
    *,
    meta: Optional[Dict[str, Any]] = None,
    user_approved: bool = False,
    task_id: str = "",
    origin_lane: str = "agent",
    allowed_lanes: Optional[List[str]] = None,
    allowed_capabilities: Optional[List[str]] = None,
    allowed_resources: Optional[List[str]] = None,
    ttl_seconds: Optional[int] = None,
) -> Dict[str, Any]:
    """Backward-compatible passport issuer.

    The old flat JSON registry remains only as a bounded migration fallback when
    TrustRegistry cannot be loaded. Explicit user approval is always required.
    """
    if not user_approved:
        return {"ok": False, "error": "explicit_user_approval_required", "execution_authority": False}
    result = issue_outbound_agent_passport(
        agent_id=agent_id,
        purpose=purpose,
        task_id=task_id,
        origin_lane=origin_lane,
        allowed_lanes=allowed_lanes,
        allowed_capabilities=allowed_capabilities,
        allowed_resources=allowed_resources,
        ttl_seconds=ttl_seconds,
        user_approved=True,
        meta=meta,
    )
    if result.get("ok"):
        return result

    # Migration fallback: no launch, no network, no execution. This permits an
    # older installation to classify a known return while the TrustRegistry is
    # unavailable, and is never preferred over a passport.
    safe_agent_id = re.sub(r"[^A-Za-z0-9._:-]+", "_", str(agent_id or "").strip())[:180]
    if not safe_agent_id:
        return result
    signature = str(allowed_return_signature or "").strip() or _hash_text(safe_agent_id + "|" + str(purpose or "") + "|" + str(time.time()))
    reg = _load_registry()
    agents = reg.setdefault("agents", {})
    agents[safe_agent_id] = {
        "agent_id": safe_agent_id,
        "purpose": str(purpose or "")[:500],
        "allowed_return_signature": signature,
        "risk_tier_on_match": "LOW",
        "return_requires_review": True,
        "one_time_release_default": True,
        "created_at": time.time(),
        "legacy_fallback": True,
        "meta": meta or {},
    }
    _safe_json_write(_registry_path(), reg)
    return {"ok": True, **dict(agents[safe_agent_id]), "execution_authority": False, "warning": "legacy_registry_fallback"}

def _extract_agent_identity(payload: Any) -> Dict[str, Any]:
    text = _normalize_text(payload)
    data = payload if isinstance(payload, dict) else {}
    headers = data.get("headers") if isinstance(data.get("headers"), dict) else {}
    body = data.get("json") if isinstance(data.get("json"), dict) else {}

    def pick(*names: str) -> Any:
        for name in names:
            if name in headers and headers.get(name) not in (None, ""):
                return headers.get(name)
            if name in body and body.get(name) not in (None, ""):
                return body.get(name)
        return ""

    agent_id = str(pick("x-sarahmemory-agent-id", "X-SarahMemory-Agent-Id", "agent_id", "sarah_agent_id") or "").strip()
    signature = str(pick("x-sarahmemory-agent-signature", "X-SarahMemory-Agent-Signature", "agent_signature", "return_signature", "sarah_agent_signature") or "").strip()
    passport_id = str(pick("x-sarahmemory-passport-id", "X-SarahMemory-Passport-Id", "passport_id") or "").strip()
    return_nonce = str(pick("x-sarahmemory-return-nonce", "X-SarahMemory-Return-Nonce", "return_nonce") or "").strip()
    claimed = str(pick("user-agent", "User-Agent", "agent_name", "name") or "").strip()
    requested_lane = str(body.get("requested_lane") or body.get("lane") or "").strip()
    task_id = str(body.get("task_id") or "").strip()
    capabilities = body.get("requested_capabilities") if isinstance(body.get("requested_capabilities"), list) else []
    resources = body.get("requested_resources") if isinstance(body.get("requested_resources"), list) else []
    risk_tier = str(body.get("risk_tier") or "low").strip().lower()
    payload_hash = str(body.get("payload_hash") or "").strip() or _hash_text(text)
    return {
        "agent_id": agent_id,
        "signature": signature,
        "passport_id": passport_id,
        "return_nonce": return_nonce,
        "claimed_identity": claimed,
        "requested_lane": requested_lane,
        "task_id": task_id,
        "requested_capabilities": [str(x)[:180] for x in capabilities[:64]],
        "requested_resources": [str(x)[:500] for x in resources[:64]],
        "risk_tier": risk_tier,
        "payload_hash": payload_hash,
    }


def _legacy_agent_signature_matches(identity: Dict[str, Any]) -> bool:
    agent_id = str(identity.get("agent_id") or "").strip()
    sig = str(identity.get("signature") or "").strip()
    if not agent_id or not sig:
        return False
    rec = (_load_registry().get("agents") or {}).get(agent_id)
    return isinstance(rec, dict) and str(rec.get("allowed_return_signature") or "") == sig


def _verify_passport_return(identity: Dict[str, Any], *, record_return: bool = True) -> Optional[Dict[str, Any]]:
    passport_id = str(identity.get("passport_id") or "").strip()
    if not passport_id:
        return None
    registry, error = _trust_registry_module()
    if registry is None or not callable(getattr(registry, "verify_agent_return", None)):
        return {
            "ok": False,
            "verdict": "DENY",
            "reason": "trust_registry_unavailable",
            "detail": error,
            "containment_state": "QUARANTINED",
            "execution_authority": False,
        }
    return registry.verify_agent_return(
        passport_id=passport_id,
        agent_id=str(identity.get("agent_id") or ""),
        return_nonce=str(identity.get("return_nonce") or ""),
        return_signature=str(identity.get("signature") or ""),
        payload_hash=str(identity.get("payload_hash") or ""),
        requested_lane=str(identity.get("requested_lane") or ""),
        requested_capabilities=list(identity.get("requested_capabilities") or []),
        requested_resources=list(identity.get("requested_resources") or []),
        risk_tier=str(identity.get("risk_tier") or "low"),
        record_return=record_return,
    )


def _agent_signature_matches(identity: Dict[str, Any]) -> bool:
    """Compatibility predicate; passport verification is performed once in inspect_payload."""
    if identity.get("passport_id"):
        registry, _ = _trust_registry_module()
        try:
            passport = registry.lookup_agent_passport(passport_id=str(identity.get("passport_id"))) if registry else None
            return bool(passport and passport.get("agent_id") == identity.get("agent_id"))
        except Exception:
            return False
    return _legacy_agent_signature_matches(identity)

def _write_capture_report(report: Dict[str, Any], lane: str = "quarantine") -> str:
    dirs = _roach_dirs()
    capture_id = str(report.get("capture_id") or "agent-capture")
    safe_name = re.sub(r"[^A-Za-z0-9._:-]+", "_", capture_id)[:160] + ".json"
    base = dirs.get(lane) or dirs.get("quarantine")
    path = os.path.join(base, safe_name)
    _safe_json_write(path, report)
    audit_path = os.path.join(dirs["reports"], safe_name)
    _safe_json_write(audit_path, report)
    return path


def _build_agent_capture_report(payload: Any, result: Dict[str, Any], identity: Dict[str, str], *, source: str, remote_addr: str) -> Dict[str, Any]:
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    capture_id = f"AGENT-CAP-{ts}-{str(identity.get('payload_hash') or '')[:12]}"
    return {
        "schema": ROACHMOTEL_SCHEMA,
        "capture_id": capture_id,
        "timestamp": time.time(),
        "source": {
            "entry_surface": source,
            "remote_addr_hash": _hash_text(str(remote_addr or "")) if remote_addr else "",
            "claimed_identity": identity.get("claimed_identity"),
            "agent_id": identity.get("agent_id"),
            "request_fingerprint": identity.get("payload_hash"),
            "authenticated_return_signature": bool(result.get("signature_match")),
        },
        "behavior": {
            "hits": result.get("hits", []),
            "remote_hits": result.get("remote_hits", []),
            "sensitive_hits": result.get("sensitive_hits", []),
            "attempted_exfiltration": bool(result.get("scrape_or_mining_score", 0) >= 50),
            "attempted_prompt_injection": bool(result.get("hits")),
            "attempted_policy_bypass": bool(result.get("hits")),
        },
        "risk": {
            "risk_score": result.get("risk_score"),
            "risk_tier": result.get("risk_tier"),
            "confidence_score": result.get("confidence_score"),
            "why_risky": result.get("reason"),
        },
        "containment": {
            "quarantined": result.get("containment_state") in ("QUARANTINED", "BLOCKED"),
            "network_egress_blocked": True,
            "filesystem_access": "deny_except_quarantine_record",
            "database_access": "deny",
            "tool_access": "deny",
            "release_required_user_approval": True,
        },
        "accountability": {
            "detected_by_file": "SarahMemoryAgentFirewall.py",
            "detected_by_function": "inspect_payload",
            "final_authority": "USER",
            "user_release_approved": False,
            "one_time_release": True,
            "persistent_allow_rule_created": False,
        },
    }


def _bool_flag(name: str, default: bool = False) -> bool:
    try:
        value = getattr(config, name, default) if config is not None else default
    except Exception:
        value = default
    env_value = os.getenv(name)
    if env_value is not None:
        value = env_value
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on", "enabled")


def _normalize_text(value: Any, max_len: int = 12000) -> str:
    try:
        if isinstance(value, (dict, list, tuple)):
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        else:
            text = str(value or "")
    except Exception:
        text = repr(value)
    return text[:max_len]


def _hash_text(text: str) -> str:
    try:
        return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()
    except Exception:
        return ""


def inspect_payload(payload: Any, *, source: str = "unknown", remote_addr: str = "") -> Dict[str, Any]:
    """Return deterministic anti-hijack and passport-border verdict evidence."""
    text = _normalize_text(payload)
    lower = text.lower()
    hits = [p for p in HIJACK_PATTERNS if p.lower() in lower]
    remote_hits = [p for p in REMOTE_WRITE_PATTERNS if p.lower() in lower]
    agent_hits = [p for p in AI_AGENT_MARKERS if p.lower() in lower]
    sensitive_hits = [p for p in SENSITIVE_TARGET_PATTERNS if p.lower() in lower]
    identity = _extract_agent_identity(payload)

    local_only = _bool_flag("LOCAL_ONLY_MODE", True)
    firewall_enabled = _bool_flag("SARAHMEMORY_AGENT_FIREWALL_ENABLED", True)
    remote_trigger_block = _bool_flag("SARAHMEMORY_REMOTE_TRIGGER_BLOCK", True)
    local_peer = remote_addr in ("", "127.0.0.1", "::1", "localhost")
    is_agent_like = bool(
        agent_hits
        or identity.get("agent_id")
        or identity.get("signature")
        or identity.get("passport_id")
        or "agent" in str(identity.get("claimed_identity") or "").lower()
    )

    passport_result: Optional[Dict[str, Any]] = None
    if firewall_enabled and identity.get("passport_id") and not hits:
        passport_result = _verify_passport_return(identity, record_return=True)
    legacy_signature_match = False if identity.get("passport_id") else _legacy_agent_signature_matches(identity)
    signature_match = bool(passport_result and passport_result.get("ok")) or legacy_signature_match

    verdict = "ALLOW"
    reason = "no hijack pattern detected"
    risk_score = 5
    risk_tier = "LOW"
    containment_state = "OBSERVED"
    scrape_score = 75 if any(x in lower for x in ("scrape", "mine", "harvest", "dump", "exfiltrate", "crawl")) else 0

    if firewall_enabled and sensitive_hits and any("energetics" in h.lower() for h in sensitive_hits):
        verdict, reason, risk_score, risk_tier, containment_state = (
            "DENY", "AI-agent or remote payload targeted Energetics / hazardous-energy boundary", 100, "HIGH", "BLOCKED"
        )
    elif firewall_enabled and hits:
        verdict, reason, risk_score, risk_tier, containment_state = (
            "DENY", "anti-agent hijack pattern detected", 95, "HIGH", "BLOCKED"
        )
    elif firewall_enabled and remote_trigger_block and remote_hits and not local_peer:
        verdict, reason, risk_score, risk_tier, containment_state = (
            "DENY", "remote protected-core mutation pattern detected", 96, "HIGH", "BLOCKED"
        )
    elif firewall_enabled and passport_result is not None and not passport_result.get("ok"):
        verdict = "DENY"
        reason = "agent passport verification failed: " + str(passport_result.get("reason") or "invalid_passport")
        risk_score = 98 if str(passport_result.get("containment_state") or "").upper() == "BLOCKED" else 90
        risk_tier = "HIGH"
        containment_state = str(passport_result.get("containment_state") or "QUARANTINED").upper()
    elif firewall_enabled and passport_result is not None and passport_result.get("ok"):
        verdict = "REQUIRE_REVIEW"
        reason = "passport verified; returning AI-agent payload captured for user-governed review"
        risk_score = 18
        risk_tier = "LOW"
        containment_state = "CAPTURED_REVIEW"
    elif firewall_enabled and is_agent_like and not local_peer and not legacy_signature_match:
        verdict, reason, risk_score, risk_tier, containment_state = (
            "DENY", "unknown AI-agent has no valid passport; RoachMotel quarantine required", 90, "HIGH", "QUARANTINED"
        )
    elif firewall_enabled and is_agent_like and legacy_signature_match:
        verdict, reason, risk_score, risk_tier, containment_state = (
            "REQUIRE_REVIEW", "legacy outbound signature matched; migration review required before release", 25, "LOW", "CAPTURED_REVIEW"
        )
    elif local_only and not local_peer:
        verdict, reason, risk_score, risk_tier, containment_state = (
            "REQUIRE_LOCAL_OR_ARMED", "local-only mode blocks unarmed remote trigger", 70, "HIGH", "OBSERVED_HOLD"
        )

    result: Dict[str, Any] = {
        "ok": verdict == "ALLOW",
        "verdict": verdict,
        "reason": reason,
        "source": source,
        "remote_addr": remote_addr,
        "hits": hits,
        "remote_hits": remote_hits,
        "agent_hits": agent_hits,
        "sensitive_hits": sensitive_hits,
        "signature_match": signature_match,
        "passport_verified": bool(passport_result and passport_result.get("ok")),
        "passport_result": passport_result or {},
        "agent_identity": identity,
        "risk_score": risk_score,
        "risk_tier": risk_tier,
        "confidence_score": 0.90 if verdict != "ALLOW" else 0.75,
        "scrape_or_mining_score": scrape_score,
        "containment_state": containment_state,
        "payload_sha256": _hash_text(text),
        "execution_authority": False,
        "ts": time.time(),
    }

    if verdict in ("DENY", "REQUIRE_REVIEW") or containment_state in ("QUARANTINED", "BLOCKED", "CAPTURED_REVIEW"):
        report = _build_agent_capture_report(payload, result, identity, source=source, remote_addr=remote_addr)
        lane = "blocked" if containment_state == "BLOCKED" else "quarantine" if containment_state == "QUARANTINED" else "inbound"
        result["capture_report_path"] = _write_capture_report(report, lane=lane)

    if is_agent_like or verdict != "ALLOW":
        event_type = {
            "BLOCKED": "AGENT_BLOCKED",
            "QUARANTINED": "AGENT_QUARANTINED",
            "CAPTURED_REVIEW": "AGENT_RETURN_CAPTURED",
        }.get(containment_state, "AGENT_PAYLOAD_OBSERVED")
        _ledger_receipt(
            event_type,
            verdict=verdict,
            identity=identity,
            source=source,
            reason=reason,
            risk=risk_tier.lower(),
            payload_hash=result["payload_sha256"],
            metadata={
                "risk_score": risk_score,
                "containment_state": containment_state,
                "matched_pattern_count": len(hits) + len(remote_hits) + len(sensitive_hits),
                "passport_verified": bool(result.get("passport_verified")),
            },
        )

    try:
        if audit_event is not None and verdict != "ALLOW":
            audit_event(
                "agent_firewall",
                "inspect_payload",
                verdict,
                {
                    "source": source,
                    "remote_addr_hash": _hash_text(str(remote_addr or "")) if remote_addr else "",
                    "reason": reason,
                    "hits": hits[:10],
                    "remote_hits": remote_hits[:10],
                    "payload_sha256": result["payload_sha256"],
                    "risk_score": risk_score,
                    "risk_tier": risk_tier,
                    "containment_state": containment_state,
                    "passport_id": str(identity.get("passport_id") or "")[:180],
                    "capture_report_path": result.get("capture_report_path", ""),
                },
                actor=str(identity.get("agent_id") or "unknown_agent"),
                risk=risk_tier.lower(),
                source="SarahMemoryAgentFirewall",
                retention="security_audit",
            )
    except Exception:
        pass

    return result

def allow_payload(payload: Any, *, source: str = "unknown", remote_addr: str = "") -> Tuple[bool, Dict[str, Any]]:
    """Return (allowed, evidence)."""
    result = inspect_payload(payload, source=source, remote_addr=remote_addr)
    return bool(result.get("verdict") == "ALLOW"), result


# ====================================================================
# END OF SarahMemoryAgentFirewall.py v9.0.0
# ====================================================================
